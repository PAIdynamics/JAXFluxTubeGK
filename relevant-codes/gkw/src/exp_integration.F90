module exp_integration

use mpiinterface
use mpidatatypes
use global,    only : lverbose
use general,   only : gkw_abort
use control,   only : vp_trap,naverage
use grid,      only : parallel_s,parallel_vpar,lsendrecv_mu,&
                    & proc_mu_prev, &
                    & proc_mu_next

use mpicomms,  only : COMM_VPAR_NE, COMM_S_EQ, COMM_S_NE, COMM_MU_NE
use dist, only : fdisi,ghost_size_mu,nsolc
use normalise, only : normalize                  

implicit none

private

public :: explicit_integration, init_explicit, exp_integration_deallocate

! parameter used to set the implicitness in some of the schemes 
real :: delta

! parameters used for the 3rd order (or is it second ??) scheme 
! that is stable for waves 
real, dimension(3) :: alf, bet 
real :: gam

! the distribution function and rhs of the intermediate steps  
complex, allocatable, dimension(:,:) :: fdisk
complex, allocatable, dimension(:,:) :: rhsk

! mpi status and request for non-blocking vpar and s
integer, dimension(4) :: p_request_vpar, p_request_s, p_request_mu
integer, dimension(8) :: p_request_vpar_mu
integer, dimension(MPI_STATUS_SIZE,4) :: p_status_vpar, p_status_s, p_status_mu
integer, dimension(MPI_STATUS_SIZE,8) :: p_status_vpar_mu
integer, parameter :: tag_persist_vpar_prev = 98, tag_persist_vpar_next = 99
integer, parameter :: tag_persist_mu_prev = 88, tag_persist_mu_next = 89
integer, parameter :: tag_persist_vpar_next_mu_prev = 66
integer, parameter :: tag_persist_vpar_next_mu_next = 67
integer, parameter :: tag_persist_vpar_prev_mu_prev = 68
integer, parameter :: tag_persist_vpar_prev_mu_next = 69
integer, parameter :: tag_persist_s_prev = 48, tag_persist_s_next = 49
real, allocatable, dimension(:,:) :: coefs

! help array for the mpi 
complex, allocatable :: bufphi(:)
complex, allocatable :: fdis_tmp(:)

contains

!
!***********************************************************************
!

subroutine init_explicit
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Initialization routine for the explicit integration 
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
use control, only : meth
use grid,    only : nmod, nx, ns, nmu, nvpar, nsp
use dist,    only : nsolc, nphi,msolc, indx

! size of the fdisk and rhsk 
integer :: isizef, isizer, ierr, idx, imod, ix, jv, kt, is, ipar

! Set the parameters for the 3rd order scheme 
! second order (for completeness is written out here)
alf(1) = 2.E0
alf(2) = -0.5E0
alf(3) = 0.E0
bet(1) = 2.E0
bet(2) = -1.E0
bet(3) = 0.E0
gam = 1.5E0

! Third order (is actually used) 
alf(1) = 3.E0
alf(2) = -1.5E0
alf(3) = 1.E0/3.E0
bet(1) = 3.E0
bet(2) = -3.E0
bet(3) = 1.E0
gam = 11.E0/6.E0

! set the implicitness parameter 
delta = 0.5E0 

! allocate the arrays of the distribution function and the rhs 
! note the size depends on the scheme used and therefore on meth
select case(meth) 
case(1) 
  isizef = 1
  isizer = 1
case(2) 
  isizef = 2
  isizer = 1 
case(3) 
  isizef = 4
  isizer = 3  
case(5,40) 
  isizef = 3
  isizer = 2
  
case default 
  call gkw_abort('exp_integration : Unknown explicit integration &
                 &scheme') 
end select 

ierr = 0 
allocate(fdisk(nsolc,isizef),stat=ierr)
if (ierr.ne.0) call gkw_abort('Could not allocate fdisk in explicit_integration')

! allocate the right hand side  
allocate(rhsk(nsolc,isizer),stat=ierr) 
if (ierr.ne.0) call gkw_abort('Could not allocate rhsk in explicit_integration')

! allocate tmp space
! (N.B. this is defined in dist in an attempt to fool the compiler
!  for now)
allocate(fdis_tmp(msolc),stat=ierr)
if (ierr.ne.0) then
  stop 'Could not allocate fdis_tmp in explicit_integration'
endif
  
  ! allocate the help array for the use of mpi
  if (nsolc-nphi+1 < nx*2) then
    allocate(bufphi(2*nx), stat = ierr)
  else
    allocate(bufphi(nsolc-nphi+1), stat = ierr)
  end if
  if (ierr.ne.0) then 
    stop 'Could not allocate bufphi in matdat'
  endif 

end subroutine init_explicit

!
!***********************************************************************
!

subroutine explicit_integration(itime)
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! SVN:$Id: exp_integration.F90 1020 2009-07-02 19:56:45Z  $  
! This routine does the explicit time integration 
! Several methods can be used. Their choice is 
! controlled through the 'meth' parameter
! meth = 1 Midpoint method 
! meth = 2 Fourth order Runga Kutta
! meth = 3 Third order scheme that is stable for 
!          waves 
!
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
use dist,       only : fdisi, nsolc, nphi
use control,    only : ntime, meth, dtim, time, silent, naverage,       & 
                     & dtim_input, dtim_est, dtim_est_save, non_linear, &
                     & ntotstep, nl_dtim_est, lcalc_freq, stop_me, dt_min
use grid, only  : non_blocking_vpar
use rotation,   only :  shear_real, shear_shift_ky, shear_ky_shift, shear_remap, wavevector_remap
use general,    only : gkw_warn
integer :: ierr, iloop, i, ix, j, itime, nend 
real :: norm_factor, gammav, dtim_dum


  ! not absolutely necessary, but on the first enter of the routine 
  ! the distribution is normalized 
  if (itime == 1) then 
    call normalize(2,fdisi(1),nsolc)
  endif 

  ! If necessary, set up persistent communication. The actual communication
  ! is performed in calculate_rhs.
#if defined(mpi)
  if ((parallel_vpar .and. non_blocking_vpar .and. (vp_trap .eq. 0)) .or. &
    & parallel_s .or. lsendrecv_mu) then
    call persistent_comm_init
  end if
#endif

  ! time step : the loop over the total number of large timesteps
  time_stepping : do iloop = 1, naverage

    !By doing the shearing here we do not need to shift the potential
    !Since the potential is recalculated in calculate_rhs
    if (shear_ky_shift) call shear_shift_ky(fdisi)
    if (shear_remap) call wavevector_remap(fdisi,time)

    ! select the integration method 
    method_of_integration : select case (meth)
      case(1) ; call rk2
      case(2) ; call rk4
      case(3) ; call rk3(itime,iloop)
      case default
        ! Meth does not match any of the implemented methods 
        stop 'Not a proper selection of the numerical method [METH]'
    end select method_of_integration

    ! reset the time step if necessary - will not work for meth=3?
    if (meth /= 3) then
      if (nl_dtim_est) then
        if (dtim_est < dtim_input) then
          !if (root_processor) write(*,*) 'Timestep estimator is off'
          dtim = dtim_est 
          if (lverbose.and.root_processor) write (*,*) 'New dtim', dtim
        else
          dtim = dtim_input 
        end if  
      end if !nl_dtim_est
    end if !meth

    ! check the time step is not too small
    if (dtim < dt_min) then
      call gkw_warn('dt < dt_min; the run will terminate shortly')
      stop_me = .true.
    end if


    if (lcalc_freq) then
      ! keep track of the phase
      call calculate_fields(fdisi(1))
      call normalize(-1,fdisi(1),nsolc)
    endif

    ! advance the time 
    time=time+dtim*1.
   
    ! one more timestep done 
    ntotstep = ntotstep + 1 

  end do time_stepping

  !If the nonlinear timestep estimator reduction is off, check across processors

  if(non_linear .and.(.not.nl_dtim_est)) then
#if defined(mpi)
    ierr = 0
    call MPI_ALLREDUCE(dtim_est_save,dtim_dum,1,MPIREAL_X,MPI_MIN, &
       & MPI_COMM_WORLD,ierr)
#else
    dtim_dum=dtim_est_save
#endif 
    if (dtim_dum .lt. dtim) then
       call gkw_warn ('Timestep too big (nl_est)!, aborting')
       !The ideal solution is to make the code go back one large timestep
       !Until this is implemented will abort.
       stop_me=.true.
    end if
  end if


  ! at the end of the timestep, make the electromagnetic field
  ! consistent with the distribution function 
  call calculate_fields(fdisi(1))
  ! Normalization 
  ! done after calculate_fields to have the new potential
  call normalize(2,fdisi(1),nsolc)

  if (meth == 3) then
    ! normalize properly 
    ! done after calculate_fields to have the new potential
    call normalize(3,fdisk(1,1),nsolc)
    call normalize(3,fdisk(1,2),nsolc) 
    call normalize(3,fdisk(1,3),nsolc)
    call normalize(3,rhsk(1,1),nsolc)
    call normalize(3,rhsk(1,2),nsolc)
    call normalize(3,rhsk(1,3),nsolc)
  end if

  ! If necessary, stop persistent communication.
#if defined(mpi)
  if (parallel_vpar .and. non_blocking_vpar .and. (vp_trap .eq. 0)) then
    call persistent_comm_end
  else if (lsendrecv_mu) then
    call persistent_comm_end 
  endif
#endif

end subroutine explicit_integration

!****************************************************************************
!> A Runge Kutta second order timestep
!----------------------------------------------------------------------------

subroutine rk2

  integer :: i

  ! first 'half' time step 
  ! calculate the rhs 
  do i = 1, nsolc 
    fdisk(i,1) = fdisi(i)
  end do 
  call calculate_rhs(fdisk(1,1),rhsk(1,1))

  ! advance a timestep delta*dtime, calculate f 
  ! and store in fdisk(:,2)
  do i = 1, nsolc 
    fdisk(i,1) = fdisi(i) + delta*rhsk(i,1) 
  end do 

  !  second part full time step calculated from fdisk(:,2) 
  ! calculate the rhs 
  call calculate_rhs(fdisk(1,1),rhsk(1,1))

  ! add the rhs to fdisi 
  do i = 1, nsolc
    fdisi(i) = fdisi(i) + rhsk(i,1) 
  end do 

end subroutine rk2

!****************************************************************************
!> A Runge Kutta fourth order timestep
!----------------------------------------------------------------------------

subroutine rk4

  integer :: i

  ! initialize to fdisk(:,1)
  do i = 1, nsolc 
    fdisk(i,1) = fdisi(i)
    fdisk(i,2) = fdisi(i)
  end do 

  ! advance a timestep delta*dtime, calculate delta f 
  call calculate_rhs(fdisk(1,2),rhsk(1,1))

  ! first step into solution 
  do i = 1, nsolc 
    fdisi(i) = fdisi(i) + rhsk(i,1)/6.E0
  end do 

  ! second step initialization  
  do i = 1, nsolc 
    fdisk(i,2) = fdisk(i,1) + rhsk(i,1)/2.E0
  end do 

  ! advance a timestep delta*dtime, calculate delta f 
  call calculate_rhs(fdisk(1,2),rhsk(1,1)) 

  ! second step into solution 
  do i = 1, nsolc 
    fdisi(i) = fdisi(i) + rhsk(i,1)/3.E0
  end do 

  ! third step initialization  
  do i = 1, nsolc 
    fdisk(i,2) = fdisk(i,1) + rhsk(i,1)/2.E0
  end do 

  ! advance a timestep delta*dtime, calculate delta f 
  call calculate_rhs(fdisk(1,2),rhsk(1,1))

  ! third step into solution 
  do i = 1, nsolc 
    fdisi(i) = fdisi(i) + rhsk(i,1)/3.E0
  end do 

  ! fourth step initialization  
  do i = 1, nsolc 
    fdisk(i,2) = fdisk(i,1) + rhsk(i,1)
  end do 

  ! advance a timestep delta*dtime, calculate delta f 
  call calculate_rhs(fdisk(1,2),rhsk(1,1))

  ! fourth step into solution 
  do i = 1, nsolc 
    fdisi(i) = fdisi(i) + rhsk(i,1)/6.E0
  end do 

end subroutine rk4
 
!****************************************************************************
!> Runge Kutta third order (midpoint method; stable for waves) timestep. The
!> timestep estimator is not programmed for this method
!----------------------------------------------------------------------------

subroutine rk3(itime,iloop)

  integer, intent(in) :: itime,iloop
  integer :: i,j

  if (itime == 1) then 

    ! To set up the scheme use fourth order Runge Kutta
    if (iloop <= 3) then

      ! initialize to fdisi 
      do i = 1, nsolc 
        fdisk(i,1) = fdisi(i)
        fdisk(i,4) = fdisi(i)
      end do 

      ! advance a timestep delta*dtime, calculate delta f 
      call calculate_rhs(fdisk(1,4),rhsk(1,1))

      ! Put in storage 
      do i = 1, nsolc 
        fdisk(i,4-iloop) = fdisi(i)
        rhsk(i,4-iloop)  = rhsk(i,1)
      end do 

      ! no point to calculate the last time step 
      if (iloop /= 3) then 
 
        ! first step into solution 
        do i = 1, nsolc 
          fdisi(i) = fdisi(i) + rhsk(i,1)/6.E0
        end do 
  
        ! second step initialization  
        do i = 1, nsolc 
          fdisk(i,4) = fdisk(i,1) + rhsk(i,1)/2.E0
        end do 
  
        ! advance a timestep delta*dtime, calculate delta f 
        call calculate_rhs(fdisk(1,4),rhsk(1,1)) 
  
        ! second step into solution 
        do i = 1, nsolc 
          fdisi(i) = fdisi(i) + rhsk(i,1)/3.E0
        end do 
  
        ! third step initialization  
        do i = 1, nsolc 
          fdisk(i,4) = fdisk(i,1) + rhsk(i,1)/2.E0
        end do 
  
        ! advance a timestep delta*dtime, calculate delta f 
        call calculate_rhs(fdisk(1,4),rhsk(1,1))
  
        ! third step into solution 
        do i = 1, nsolc 
          fdisi(i) = fdisi(i) + rhsk(i,1)/3.E0
        end do 
  
        ! fourth step initialization  
        do i = 1, nsolc  
          fdisk(i,4) = fdisk(i,1) + rhsk(i,1)
        end do 
  
        ! advance a timestep delta*dtime, calculate delta f 
        call calculate_rhs(fdisk(1,4),rhsk(1,1))
  
        ! fourth step into solution 
        do i = 1, nsolc 
          fdisi(i) = fdisi(i) + rhsk(i,1)/6.E0
        end do

      end if 

    end if

  end if 

  if (itime == 1 .and. iloop < 3) return 

  ! initialize 
  do i = 1, nsolc 
    fdisk(i,4) = (0.,0.)    
  end do 

  ! update the solution 
  do i = 1, nsolc 
    do j = 1, 3 
      fdisk(i,4) = fdisk(i,4) + alf(j)*fdisk(i,j) &
                & + bet(j)*rhsk(i,j)
    end do 
    fdisk(i,4) = fdisk(i,4) / gam
  end do 

  ! resuffle (could be made more efficient)
  do i = 1, nsolc 
    fdisk(i,3) = fdisk(i,2)
    fdisk(i,2) = fdisk(i,1)
    fdisk(i,1) = fdisk(i,4)
    fdisi(i) = fdisk(i,4)        
    rhsk(i,3) = rhsk(i,2)
    rhsk(i,2) = rhsk(i,1)
  end do     
         
  ! calculate a new right hand side 
  call calculate_rhs(fdisk(1,4),rhsk(1,1))

end subroutine rk3

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine calculate_rhs(fdis,rhs) ! Optimised 
!--------------------------------------------------------------------
!  Subroutine that calculates the right hand side of the equation 
!  input fdis <- The distribution function 
!  output rhs -> the right hand side. An explict step can be calcu-
!                lated through fdis = fdis + dt*rhs 
!  In this routine the matrix is assume to be in the form used in 
!  the explict scheme. Calling this routine from the implict solved
!  will not work. 
!--------------------------------------------------------------------
use control,    only : nlapar,non_linear, dtim,time, matrix_format
use grid,       only : proc_vpar_next, proc_vpar_prev,                       &
                     & non_blocking_vpar, proc_s_next, proc_s_prev
use mpicomms,   only : COMM_VPAR_NE
use matdat,     only : mat, matr, ii, jj, n1, n2, n3, iir, jjr, n2r, &
                     & source, iac, nmata, mata, jja
use rotation,   only : shear_real
use dist,       only : nsolc, nphi, msolc, nf, ighost_sbp, ighost_sbn, ighost_mubn, &
     & ighost_mubp,fdisi,ghost_size_vpar_mu
use non_linear_terms, only : add_non_linear_terms
use collisionop, only : mom_conservation
use control,    only : collisions 

complex, dimension(nsolc) :: fdis
complex, intent(out), dimension(nsolc) :: rhs
integer :: i, j, ierr

integer, parameter :: stag = 34
ierr=0

! First calculate the electro-static and electro-magnetic fields 
call calculate_fields(fdis)
!
! for electro-magnetic runs undo the A|| correction of g 
! copy the rest of fdis into the fdis_tmp
if (nlapar) then 
  do i = 1, nmata
    fdis_tmp(i) = fdis(i) + mata(i)*fdis(jja(i))
  end do
  do i= nmata+1, nsolc 
    fdis_tmp(i) = fdis(i)
  enddo
else
  do i= 1, nsolc
    fdis_tmp(i) = fdis(i)
  enddo
endif 

  !
  ! Send/Recv the distribution function to neighbours if parallelizing in vpar, s
  ! or mu (if derivatives in mu are needed).
  !
#if defined(mpi)

  ! vpar (not needed with vp_trap = 1)
  if (parallel_vpar .and. (vp_trap .eq. 0)) then
    if (non_blocking_vpar) then
      call MPI_STARTALL(4, p_request_vpar, ierr)
    else
      call gkw_abort('exp_inte: I do not work with blocking vpar')
    end if
  end if
  
  ! mu (if needed)
  if (lsendrecv_mu) then
    call MPI_STARTALL(4, p_request_mu, ierr)
  end if
  
  ! s-direction
  if (parallel_s) then
    call MPI_STARTALL(4, p_request_s, ierr)
  end if

  ! vpar-mu diagonal
  if (ghost_size_vpar_mu >0) then
    call MPI_STARTALL(8, p_request_vpar_mu, ierr)
  end if
#endif

! initialize to zero 
do i = 1, nsolc 
  rhs(i) = (0.,0.)
end do 

!Call the nonlinear terms routine if necessary 
!Can only be called after initialisiation (now done in nonlinear_init)
! The nonlinear terms are called first since they work on the distribution 
! g = f + Z v\\ A\\ etc.  rather than f. After the nonlinear terms have 
! been called the correction due to the vector potential is subtracted 
! from g. The routine therefore changes fdis - Is this still true?!!!! 
if (non_linear.or.shear_real) call add_non_linear_terms(fdis,rhs)

!Now add the linear terms
select case(matrix_format)

case('complex')

  if (.true.) then 

    ! calculate the rhs 
    do i = 1, nsolc 
      rhs(i) = rhs(i) + dtim*source(i)
    end do

!!$WAH: Work in progress 
!!$    if(collisions.and.mom_conservation)then
!!$       do i= 1,1
!!$          rhs(i) = rhs(i) + dtim
!!$       end do
!!$    end if 

    if (non_blocking_vpar.or.parallel_s .or. lsendrecv_mu) call wait_localcom     
 !   write (*,*) processor_number,fdis_tmp(ighost_mubp+1),fdis_tmp(ighost_mubn+1)
 !   call mpibarrier()
 !   call gkw_abort('die')
    
    do i = 1, n2
      rhs(ii(i)) = rhs(ii(i))+dtim*mat(i)*fdis_tmp(jj(i)) 
    end do 

 else !never

    do i = 1, nsolc 
      rhs(i) = rhs(i) + dtim*source(i)
    end do 

    if (non_blocking_vpar.or.parallel_s .or. lsendrecv_mu) call wait_localcom

    do i = 1, nphi - 1 
      do j = iac(i), iac(i+1) - 1 
        rhs(i) = rhs(i)+dtim*mat(j)*fdis_tmp(jj(j))
      end do 
    end do 
    !write(*,*)n2,iac(nphi)-1

  endif 

case('complex-real')

  do i = 1, nsolc 
    rhs(i) = rhs(i) + dtim*source(i) 
  end do 

  !This can never be used since this case is never parallel
  if (non_blocking_vpar.or.parallel_s .or. lsendrecv_mu) call wait_localcom

  do i = 1, n2 
    rhs(ii(i)) = rhs(ii(i)) + dtim*mat(i)*fdis_tmp(jj(i))
  end do 
  do i = 1, n2r
    rhs(iir(i)) = rhs(iir(i)) + dtim*matr(i)*fdis_tmp(jjr(i))
  end do 
 
case default 

  call gkw_abort('calculate_rhs:&
                & Severe internal error, unkonwn matrix format')

end select 
 

return
end subroutine calculate_rhs

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine calculate_fields(fdis)  ! OPTIMISED routine 
!---------------------------------------------------------------------
! This routine caclulates the integrated quantities of the perturbed
! distribution function.
! vector potential (and in the future possibly the momentum conserving
! part of the collision operator)
! 
! It is an optimized routine, meaning that explicit assumptions are 
! made on the the memory layout of the code. 
!
! calls MPI_ALL_REDUCE  to sum and distribute the integrals to all 
!                       processors. 

!---------------------------------------------------------------------
  use dist,    only : nsolc, nphi,n_mom_conserve,nelem_mom_conserve,&
                    & nregular_fields_end
  use matdat,  only : mat, ii, jj, n2, n3, n4, matr, &
                  & iir, jjr, n2r, n3r, n4r, nmatz, nmaty, &
                  & iiy, jjy, iiz, jjz, maty, matz, iac 
  use control, only : nlphi, nlapar, matrix_format, zonal_adiabatic
  use grid,    only : nx, ns, nmod,nsp
  use mpicomms, only : COMM_S_EQ,COMM_SP_EQ_S_EQ
  use components, only : adiabatic_electrons

  complex, intent(inout) :: fdis(nsolc)

  integer :: ix, i, is, imod, nelem, ierr, j

  ! warning #1 here an assumption of the postion of phi in 
  ! the solution is made. I.e. the index function is not 
  ! used. This is of course faster.
  zero : do i = nphi, nsolc 
    fdis(i) = (0.E0,0.E0)
  end do zero

!!!select case(matrix_format) 

!!!case('complex')

  ! no point in calculating the potential if it is to 
  ! be zero. (important also for the testcases that 
  ! do not use phi)
  if ((.not. nlphi) .and. (.not.nlapar)) return 

  ! first calculate the contribution of f 
  do i = n2+1, n3
    fdis(ii(i)) = fdis(ii(i)) + mat(i)*fdis(jj(i))
  end do 

#if defined(mpi)
  ! only if run on more than one processor 
  if (number_of_processors > 1) then 

    nelem = nregular_fields_end - nphi + 1
    ierr = 0
    call MPI_ALLREDUCE(fdis(nphi),bufphi,nelem,MPICOMPLEX_X, &
                      & MPI_SUM, COMM_S_EQ, ierr)
    ! momentum conserving part reduces over same species and s points
    if (nelem_mom_conserve > 0) then
      call MPI_ALLREDUCE(fdis(n_mom_conserve+1),bufphi(nelem+1),nelem_mom_conserve,&
          & MPICOMPLEX_X, MPI_SUM, COMM_SP_EQ_S_EQ, ierr)
    end if
      
    do i = nphi, nsolc
      fdis(i) = bufphi(i-nphi+1)
    end do

  end if
#endif 

  ! if the zonal flow correction is present 
  zonal_adiabatic_correction : if (adiabatic_electrons .and. zonal_adiabatic) then 

    bufphi(:) = 0.
    do i = 1, nmatz 
      bufphi(iiz(i)) = bufphi(iiz(i)) + matz(i)*fdis(jjz(i))
    end do 
#if defined(mpi)
    ! Finish off the flux surface averaging
    ! N.B. bufphi will always be at least nx*2 in size.
    if (parallel_s) then
      call MPI_ALLREDUCE(bufphi(1:nx),bufphi(nx+1:2*nx),nx,MPICOMPLEX_X,MPI_SUM,COMM_S_NE,ierr)
      bufphi(1:nx)=bufphi(nx+1:2*nx)
    end if
#endif

    do i = 1, nmaty
      bufphi(iiy(i)) = bufphi(iiy(i)) / maty(i)
    end do 

    do i = 1, nmatz
      fdis(jjz(i)) = fdis(jjz(i)) + bufphi(iiz(i))
    enddo 

  end if zonal_adiabatic_correction

  ! then normalize
  normalise : do i = n3+1, n4 
    fdis(ii(i)) = - fdis(ii(i))/mat(i)
  end do normalise

end subroutine calculate_fields

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!-----------------------------------------------------------------------
!> init persistent communication in the vpar direction
!-----------------------------------------------------------------------
subroutine persistent_comm_init
!-----------------------------------------------------------------------
  use grid, only : proc_vpar_next, proc_vpar_prev,           &
                    & proc_s_next, proc_s_prev,proc_vpar_prev_mu_prev,        &
                    & proc_vpar_prev_mu_next, proc_vpar_next_mu_next,         &
                    & proc_vpar_next_mu_prev
  use mpicomms, only : COMM_MU_NE, COMM_VPAR_NE, COMM_CART 
  use dist,    only : nsolc, nf, ghost_size_vpar, ghost_size_s, ighost_sbp,           &
                    & ighost_sbn,ighost_vparbp,ighost_vparbn,ighost_mubp,ighost_mubn, &
                    & ighost_vparbp_mubp, ighost_vparbn_mubp,                 &
                    & ighost_vparbp_mubn, ighost_vparbn_mubn,ghost_size_vpar_mu

  integer :: ierr

#if defined(mpi)
if (parallel_s) then

  ! s-direction
  call MPI_RECV_INIT(&
    &        fdis_tmp(ighost_sbp+1),                ghost_size_s,  MPICOMPLEX_X,  &
    &                   proc_s_prev,      tag_persist_s_next,      COMM_S_NE,  &
    &                p_request_s(1),                    ierr )
  call MPI_SEND_INIT(& 
    &                   fdis_tmp(1),                       1,     TYPE_NEXT_S,  &
    &                   proc_s_next,      tag_persist_s_next,      COMM_S_NE,  &
    &                p_request_s(4),                    ierr )
  call MPI_RECV_INIT(&
    &        fdis_tmp(ighost_sbn+1),                ghost_size_s,  MPICOMPLEX_X,  &
    &                   proc_s_next,      tag_persist_s_prev,      COMM_S_NE,  &
    &                p_request_s(2),                    ierr )
  call MPI_SEND_INIT(&
      &                     fdis_tmp(1),                       1,    TYPE_PREV_S,  &
      &                   proc_s_prev,      tag_persist_s_prev,      COMM_S_NE,  &
      &                p_request_s(3),                    ierr )

endif

if (parallel_vpar) then

  ! vpar-direction
  call MPI_RECV_INIT(&
    &             fdis_tmp(ighost_vparbp+1),             ghost_size_vpar,  MPICOMPLEX_X,  &
    &                proc_vpar_prev,   tag_persist_vpar_next,   COMM_VPAR_NE,  &
    &             p_request_vpar(1),                    ierr )
  call MPI_SEND_INIT(&
    &                fdis_tmp(1),                       1,  TYPE_NEXT_VPAR,&
    &                proc_vpar_next,   tag_persist_vpar_next,   COMM_VPAR_NE,  &
    &             p_request_vpar(4),                    ierr )
  call MPI_RECV_INIT(&
    & fdis_tmp(ighost_vparbn+1),             ghost_size_vpar,  MPICOMPLEX_X,  &
    &                proc_vpar_next,   tag_persist_vpar_prev,   COMM_VPAR_NE,  &
    &             p_request_vpar(2),                    ierr )
  call MPI_SEND_INIT(&
    &                   fdis_tmp(1),                      1,  TYPE_PREV_VPAR,  &
    &                proc_vpar_prev,   tag_persist_vpar_prev,   COMM_VPAR_NE,  &
    &             p_request_vpar(3),                    ierr )

endif

if (lsendrecv_mu) then

  ! mu-direction
  call MPI_RECV_INIT(&
    &             fdis_tmp(ighost_mubp+1),     ghost_size_mu,    MPICOMPLEX_X,  &
    &                proc_mu_prev,   tag_persist_mu_next,   COMM_MU_NE,  &
    &             p_request_mu(1),                    ierr )
  call MPI_SEND_INIT(&
    &               fdis_tmp(1),                       1,  TYPE_NEXT_MU,  &
    &                proc_mu_next,   tag_persist_mu_next,   COMM_MU_NE,  &
    &             p_request_mu(4),                    ierr )
  call MPI_RECV_INIT(&
    & fdis_tmp(ighost_mubn+1),             ghost_size_mu,  MPICOMPLEX_X,  &
    &                proc_mu_next,   tag_persist_mu_prev,   COMM_MU_NE,  &
    &             p_request_mu(2),                    ierr )
  call MPI_SEND_INIT(&
    &                   fdis_tmp(1),             1,       TYPE_PREV_MU,  &
    &                proc_mu_prev,   tag_persist_mu_prev,   COMM_MU_NE,  &
    &             p_request_mu(3),                    ierr )

endif

if (ghost_size_vpar_mu > 0) then
  ! recv from prev vpar prev mu, tagged with next;next
  call MPI_RECV_INIT(&
    &             fdis_tmp(ighost_vparbp_mubp+1),ghost_size_vpar_mu,MPICOMPLEX_X,&
    &                proc_vpar_prev_mu_prev,tag_persist_vpar_next_mu_next,   &
    &                COMM_CART, p_request_vpar_mu(1), ierr)
  ! send to next vpar next mu, tagged with next;next
  call MPI_SEND_INIT(&
    &               fdis_tmp(1),  1,  TYPE_NEXT_VPAR_NEXT_MU,                &
    &               proc_vpar_next_mu_next, tag_persist_vpar_next_mu_next,   &
    &               COMM_CART, p_request_vpar_mu(8), ierr)
  ! recv from prev vpar next mu, tagged with next;prev
  call MPI_RECV_INIT(&
    &             fdis_tmp(ighost_vparbp_mubn+1),ghost_size_vpar_mu,MPICOMPLEX_X,&
    &                proc_vpar_prev_mu_next,tag_persist_vpar_next_mu_prev,   &
    &                COMM_CART, p_request_vpar_mu(4), ierr)
  ! send to next vpar prev mu, tagged with next;prev
  call MPI_SEND_INIT(&
    &               fdis_tmp(1),  1,  TYPE_NEXT_VPAR_PREV_MU,                &
    &               proc_vpar_next_mu_prev, tag_persist_vpar_next_mu_prev,   &
    &               COMM_CART, p_request_vpar_mu(6), ierr)
  ! recv from next vpar next mu, tagged with prev;prev
  call MPI_RECV_INIT(&
    &             fdis_tmp(ighost_vparbn_mubn+1),ghost_size_vpar_mu,MPICOMPLEX_X,&
    &                proc_vpar_next_mu_next,tag_persist_vpar_prev_mu_prev,   &
    &                COMM_CART, p_request_vpar_mu(2), ierr)
  ! send to prev vpar prev mu, tagged with prev;prev
  call MPI_SEND_INIT(&
    &               fdis_tmp(1),  1,  TYPE_PREV_VPAR_PREV_MU,                &
    &               proc_vpar_prev_mu_prev, tag_persist_vpar_prev_mu_prev,   &
    &               COMM_CART, p_request_vpar_mu(3), ierr)
  ! recv from next vpar prev mu, tagged with prev;next
  call MPI_RECV_INIT(&
    &             fdis_tmp(ighost_vparbn_mubp+1),ghost_size_vpar_mu,MPICOMPLEX_X,&
    &                proc_vpar_next_mu_prev,tag_persist_vpar_prev_mu_next,   &
    &                COMM_CART, p_request_vpar_mu(5), ierr)
  ! send to prev vpar next mu, tagged with prev;next
  call MPI_SEND_INIT(&
    &               fdis_tmp(1),  1,  TYPE_PREV_VPAR_NEXT_MU,                &
    &               proc_vpar_prev_mu_next, tag_persist_vpar_prev_mu_next,   &
    &               COMM_CART, p_request_vpar_mu(7), ierr)
end if

#endif

end subroutine persistent_comm_init

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!-----------------------------------------------------------------------
!> wait for communication to finish in vpar
!-----------------------------------------------------------------------
subroutine wait_localcom
!-----------------------------------------------------------------------

#if defined(mpi)

  use dist, only : ghost_size_vpar_mu

  integer :: ierr
!
! wait for non-blocking communcation to finish before continuing
!  
  if (parallel_vpar .and. vp_trap .eq. 0) then
    call MPI_WAITALL(4, p_request_vpar, p_status_vpar, ierr)
  endif

  if (lsendrecv_mu) then
    call MPI_WAITALL(4, p_request_mu, p_status_mu, ierr)
  endif

  if (parallel_s) then
    call MPI_WAITALL(4, p_request_s, p_status_s, ierr)
  endif

  if (ghost_size_vpar_mu > 0) then
    call MPI_WAITALL(8, p_request_vpar_mu, p_status_vpar_mu, ierr)
  end if

#endif

end subroutine wait_localcom

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!-----------------------------------------------------------------------
!> end persistent communication in the vpar direction
!-----------------------------------------------------------------------
subroutine persistent_comm_end
!-----------------------------------------------------------------------

#if defined(mpi)

  use dist, only : ghost_size_vpar_mu

  integer :: ierr

  if (parallel_vpar .and. vp_trap .eq. 0) then

    call MPI_REQUEST_FREE(p_request_vpar(1),ierr)
    call MPI_REQUEST_FREE(p_request_vpar(2),ierr)
    call MPI_REQUEST_FREE(p_request_vpar(3),ierr)
    call MPI_REQUEST_FREE(p_request_vpar(4),ierr)

  endif

  if (lsendrecv_mu) then

    call MPI_REQUEST_FREE(p_request_mu(1),ierr)
    call MPI_REQUEST_FREE(p_request_mu(2),ierr)
    call MPI_REQUEST_FREE(p_request_mu(3),ierr)
    call MPI_REQUEST_FREE(p_request_mu(4),ierr)

  endif

  if (parallel_s) then

    call MPI_REQUEST_FREE(p_request_s(1),ierr)
    call MPI_REQUEST_FREE(p_request_s(2),ierr)
    call MPI_REQUEST_FREE(p_request_s(3),ierr)
    call MPI_REQUEST_FREE(p_request_s(4),ierr)

  endif

  if (ghost_size_vpar_mu > 0) then
    call MPI_REQUEST_FREE(p_request_vpar_mu(1),ierr)
    call MPI_REQUEST_FREE(p_request_vpar_mu(2),ierr)
    call MPI_REQUEST_FREE(p_request_vpar_mu(3),ierr)
    call MPI_REQUEST_FREE(p_request_vpar_mu(4),ierr)
    call MPI_REQUEST_FREE(p_request_vpar_mu(5),ierr)
    call MPI_REQUEST_FREE(p_request_vpar_mu(6),ierr)
    call MPI_REQUEST_FREE(p_request_vpar_mu(7),ierr)
    call MPI_REQUEST_FREE(p_request_vpar_mu(8),ierr)
  end if
  
#endif

end subroutine persistent_comm_end

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine exp_integration_deallocate

  if (allocated(fdisk)) deallocate(fdisk)
  if (allocated(rhsk)) deallocate(rhsk)
  if (allocated(fdis_tmp)) deallocate(fdis_tmp)
  if (allocated(bufphi)) deallocate(bufphi)

end subroutine exp_integration_deallocate

!****************************************************************************

end module exp_integration
