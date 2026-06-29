!  -*-f90-*-
module linear_terms
!
! SVN:$Id: linear_terms.F90 1022 2009-07-02 20:17:51Z  $
  use general,       only : gkw_abort, gkw_warn, time_est, svn_id
  use dist,          only : iapar, iphi, ibpar
  use global,        only : lverbose
  use functions,     only : besselj0_gkw, gamma_gkw
  use mpiinterface

  implicit none

  private

  public :: init_linear_terms, calc_linear_terms, vpar_grad_df_4d

  type :: matrix_element
    complex :: val       !< the value of the element
    integer :: iloc      !< the desired location along the s-direction
    integer :: kloc      !< the desired location along the vpar-direction
    integer :: itype     !< type of term; ifdis => f, iphi => phi, iapar => apar
    integer :: i,j,k,imod,ix,is
  end type matrix_element

  integer, parameter :: ierr_UNDEFINED = -34266234
  integer, parameter :: ierr_OK         = 0
  integer, parameter :: ierr_BAD_ALL    = 11
  integer, parameter :: ierr_BAD_S      = 21
  integer, parameter :: ierr_BAD_VPAR   = 22
  integer, parameter :: ifdis           = 5842153
!
! switches for hard disabling of calls
!
  logical, parameter :: lvpar_grad_df       = .true.
  logical, parameter :: lvdgradf            = .true.
  logical, parameter :: ltrapdf             = .true.
  logical, parameter :: lcollision_operator = .true.
  logical, parameter :: lve_grad_fm         = .true.
  logical, parameter :: lvd_grad_phi_fm     = .true.
  logical, parameter :: lvpgrphi            = .true.
  logical, parameter :: lpoisson_int        = .true.        
  logical, parameter :: lg2f_correction     = .true.      
  logical, parameter :: lampere_int         = .true.      
  logical, parameter :: lpoisson_dia        = .true.       
  logical, parameter :: lpoisson_zf         = .true.        
  logical, parameter :: lampere_dia         = .true.            
  logical, parameter :: lneoclassical       = .true. 

  
  !> dissipation switch; 1 (standard) or 2 (stable for electromagnetic)
  integer, parameter :: idisp = 2
 
  ! help for calling function HH
  integer :: jref_mu
  
  !beta prime as be selected from either geom or species calcuation
  real :: beta_prime

  interface vpar_grad_df_4d
    module procedure vpar_grad_df_4d_testnewbc
  endinterface

contains

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine init_linear_terms
!
! report revision
!
  call svn_id('$Id: linear_terms.F90 1022 2009-07-02 20:17:51Z  $')
  !read_linear_terms_namelist
!
end subroutine init_linear_terms

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine calc_linear_terms
!-----------------------------------------------------------------------------
! This routine calls in sequence all the subroutines
! that put the linear terms in the equation
!-----------------------------------------------------------------------------
  use matdat,      only : finish_matrix_section
  use control,     only : dtim, order_of_the_scheme, nlapar, collisions, vp_trap,    &
      &                   neoclassics, disp_par, disp_vp, disp_x, disp_y, ltrapping_arakawa, &
      &                   nlbpar
  use collisionop, only : collision_operator_setup, mom_conservation,   &
       coll_mom_change_int, coll_mom_change_diag
  use geom, only        : beta_prime_real
  use components, only  : beta_prime_components
  use general, only     : time_est

  integer :: willv, wills, ierr
  real :: landau, trapping, gupw, disp_fe, lin_dtim_est, local_dtim_est
  complex :: local_dt_est

! Beta_prime is used from geom if present, otherwise as calculated in components
  if (beta_prime_real+100.E0.lt.1e-6) then !no geom value
    !Note beta is currently always set to zero for an electrostatic run.
    !Therefore beta_prime_components is zero for an electrostatic run
    !This could be changed.
    beta_prime=beta_prime_components
    !But for the moment, beta_prime=0 anyway.
    beta_prime=0.
  else ! take the value read from the equilibirum file in geom.f90
    beta_prime = beta_prime_real
  endif

! Switches for Landau damping and trapping
!
  landau   = 1.
  trapping = 1.
  if (landau .ne. 1.)   call gkw_warn('calc_linear_terms: Landau /= 1')
  if (trapping .ne. 1.) call gkw_warn('calc_linear_terms: no trapping')
!
! upwinding parameter and disipation
!
  gupw = disp_par
!
! no disipation on the fields (unstable otherwise ??)
!
  disp_fe = 0.E0
!
! will[s|v] = 0 no longer do anything; the following should always be set to 1
!
  willv = 1
  wills = 1
!
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! First part the linear terms of the perturbed distribution
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!
! add the convection parallel to the field (Term I in the manual)
!
  if (ltrapping_arakawa) call igh(disp_par,disp_vp)
  vpargraddf : if (lvpar_grad_df .and. (.not.ltrapping_arakawa)) then
    select case(order_of_the_scheme)
  
      case('second_order')
!
! The old central differencing with upwind
! (Term I in the manual)
!
        if (wills .eq. 0) then
          call gkw_warn('calc_linear_terms: wills == 0 does nothing')
!       call vpar_grad_df_2d(gupw)
        else if (wills .eq. 1) then
          call vpar_grad_df_4d_testnewbc(disp_par)
        else
          call gkw_abort('calc_linear_terms: Error in choice for term 1')
        endif
  
      case('fourth_order')
!
! Call the fourth order with v_par dissipation with v_par dissipation
! (Term I in the manual)
!
        if (wills .eq. 0) then
          call gkw_warn('calc_linear_terms: wills == 0 does nothing')
!        call vpar_grad_df_4d_test(disp_par)
        else if (wills .eq. 1)  then
          call vpar_grad_df_4d_testnewbc(disp_par)
        else
          call gkw_abort('calc_linear_terms: Error in choice for term 1')
        endif
  
      case default
  
        call gkw_abort('calc_linear_terms: order not implemented for term 1')
  
    end select
  endif vpargraddf
!
! Include the mirror terms with v_perp dissipation
! (Terms IX and IV in the manual)
!
  trapdf : if (vp_trap .ne. 1 .and. ltrapdf .and. (.not. ltrapping_arakawa)) then
  ! These terms are only added if the velocity grid doesnt follow the trapping
  ! condition.

    select case(order_of_the_scheme)

      case('second_order')

        ! all central difference with upwind with v_perp dissipation
        ! Terms IX and IV in the manual
        call trapdf_2d(trapping,disp_vp)

      case('fourth_order')

        ! fourth order trapping terms with v_perp dissipation
        ! Terms IX and IV in the manual
        call trapdf_4d(trapping,disp_vp)

      case default

        call gkw_abort('calc_linear_terms: order not implemented for term 2')

    end select

  endif trapdf
!
! add the part due to the drift in the gradient of the eikonal
! (Term II in the manual)
!
  if (lvdgradf) call vdgradf

  if (disp_x.gt.0.E0.or.disp_y.gt.0.E0) call hyper_disp_perp(disp_x,disp_y)
!
! The collision opeator
!
  if (collisions .and. lcollision_operator) then
     if (lverbose) write(*,*) 'Collision operator called'
     call collision_operator_setup
  else
     if (lverbose) write(*,*)'No collisions chosen'
  end if

!
! APS: will this be called?
!call par_vel_disp
!

!
! store the value of nmat (all left hand side terms)
!
  call finish_matrix_section(1)
!
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! Second part the Maxwell background
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!
! the ExB in the Maxwell background
! (Term V in the manual)
!
  if (lve_grad_fm) call ve_grad_fm
!
! The drift in the gradient of phi times the velocity derivative of the
! Maxwell background.
! (Term VIII in the manual)
!
  if (lvd_grad_phi_fm) call vd_grad_phi_fm

  vpgrdphi : if (lvpgrphi) then
    select case(order_of_the_scheme)
  
      case('second_order')
  
        ! second order with upwind
        ! (Term VII in the manual)
        if (willv .eq. 0) then
!     call vpgrdphi_2(landau,gupw)
        else if (willv .eq. 1) then
          call vpgrphi_3_newbc(landau,disp_fe)
        else
          stop 'Error in choice for Term VII'
        endif
  
      case('fourth_order')
  
        ! fourth order for landau damping
        ! (Term VII in the manual)
        if (willv .eq. 0) then
!     call vpgrphi_3(landau,disp_fe)
        else if (willv .eq. 1) then
          call vpgrphi_3_newbc(landau,disp_fe)
        else
          stop 'Error in choice for Term VII'
        endif
  
      case default
  
        stop 'error in linear_terms: 3'
  
    end select
  endif vpgrdphi
!
! The number of matrix elements after the Maxwell terms
!

!Calculate minimum timestep estimate for ALL terms (excluding feilds), 
  call time_est(local_dt_est,99)
  !Take real part
  local_dtim_est=real(local_dt_est)
  !If no MPI 
  lin_dtim_est=1./local_dt_est
  
#if defined(mpi)
   call MPI_ALLREDUCE(1./local_dt_est,lin_dtim_est,1,MPIREAL_X,MPI_MIN, &
       & MPI_COMM_WORLD,ierr)
#endif
    
  if(root_processor) write(*,*)

  if (lin_dtim_est<dtim) then
   call gkw_warn('Linear timestep too large')
  end if

   if (root_processor) then
      write(*,'(A,1pe13.5)') ' Maximum linear timestep estimate:', lin_dtim_est
      write(*,'(A,1pe13.5)') ' Input timestep:                  ', dtim
      !write(*,*) 'Per term / species timestep reporting currently disabled'
      write(*,*) 'Fields timestep estimate  currently disabled'
      write(*,*)
   end if


  call finish_matrix_section(2)
!
!--------------------------------------------------------------------
! Third part the field equations
!--------------------------------------------------------------------
!
! The poisson equation
!
  if (lpoisson_int) call poisson_int
!
! Electro-magnetic corrections
!
  if (nlapar) then

    ! The correction to be added to fdisi to generate the
    ! distribution without A|| correction
    if (lg2f_correction) call g2f_correct

    ! The integral part of ampere's law
    if (lampere_int) call ampere_int

  endif
  
!
!Initialisation of the integrals required for momentum conservation
!
  if (collisions.and.mom_conservation) call coll_mom_change_int
  
!
! number of elements after the poisson equation
!
  call finish_matrix_section(3)
!
! diagonal part of the poisson equation
!
  if (lpoisson_dia) call poisson_dia

!
! diagonal part of the momentum conservation equation
!
  if (collisions.and.mom_conservation) call coll_mom_change_diag
!
! calculate the zonal flow matrices when necessary
!
  if (lpoisson_zf) call poisson_zf
!
! diagonal part of Ampere's law
!
  if (nlapar .and. lampere_dia) call ampere_dia
!
! number of elements
!

  call finish_matrix_section(4)

!
! Finally, if neoclassical effects are to be kept, call the source routine.
! (Term VI in the manual)
!
  if (neoclassics .and. lneoclassical) call neoclassical


end subroutine calc_linear_terms

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine trapdf_4d(trapping,disp)
!-----------------------------------------------------------------------------
! This routine puts the trapping (4th order scheme)
! (Term IV in the manual; term IX is now elsewhere)
!
! + v_R mu_N B_N G (d f / d v_parallelN ) 
!
! ( no longer includes - 2 (Z/m_R) mu_N B_N x <A_parallelN> G F_MN )
!
! in the matrix. Boundary conditions are that f is zero outside the parallel
! velocity grid.
!-----------------------------------------------------------------------------
  use grid,      only : nx, ns, nmu, nvpar, nsp, nmod
  use components,   only : vthrat
  use geom,         only : bn, gfun
  use velocitygrid, only : mugr, dvp, mugr_rms

! The multiplication factor for the trapping and the dissipation
! coefficient.
  real, intent(in) :: trapping, disp

! The integers for the loop over all grid points
  integer :: imod, ix, i, j, k, is

! element to add
  type (matrix_element) :: elem
  complex :: mat_elem

! Dummy variables
  real :: dum, dum2
  integer :: itrap, ierr

  if (trapping .eq. 0) return

  itrap = 1
  elem%itype = ifdis

  do is = 1,nsp

    ! Clear the time-step estimate
    call time_est(mat_elem,0)

    do imod=1,nmod ; do ix=1,nx ; do i=1,ns ; do j=1,nmu ; do k=1,nvpar

      call set_indx(elem,imod,ix,i,j,k,is)

      ! true for all terms below
      elem%iloc = i

      dum=trapping*vthrat(is)*mugr(j)*bn(i)*gfun(i)

      select case(idisp)
        case(1)
          dum2 = dum
        case(2) ! use the mugr rms value only
          dum2 = trapping*vthrat(is)*bn(i)*gfun(i)*mugr_rms
      end select

      select case(itrap)
        case(1)

          elem%kloc = k - 2
          elem%val  = (dum-disp*abs(dum2)) / (12E0*dvp)
          call add_element(elem,ierr)

          elem%kloc = k - 1
          elem%val  = (-8.E0*dum+4.E0*disp*abs(dum2)) / (12E0*dvp)
          call add_element(elem,ierr)

          elem%kloc = k
          elem%val  = (-6.E0*disp*abs(dum2)) / (12E0*dvp)
          call add_element(elem,ierr)

          elem%kloc = k + 1
          elem%val  = (8.E0*dum+4.E0*disp*abs(dum2)) / (12E0*dvp)
          call add_element(elem,ierr)

          elem%kloc = k + 2
          elem%val  = (-dum-disp*abs(dum2)) / (12E0*dvp)
          call add_element(elem,ierr)

        case(2)

          elem%kloc = k - 1
          elem%val  =  (-1.E0*dum + disp*abs(dum2))/(2.E0*dvp)
          call add_element(elem,ierr)

          elem%kloc = k
          elem%val  = -2.E0*disp*abs(dum2)/(2.E0*dvp)
          call add_element(elem,ierr)

          elem%kloc = k + 1
          elem%val  = (1.E0*dum+disp*abs(dum2))/(2.E0*dvp)
          call add_element(elem,ierr)

        end select

      end do ; end do ; end do ; end do ; end do

    ! retrieve maximum time step
    call time_est(mat_elem,2)
    ! write(*,*)'species trapping ',is,' max time ',1./real(mat_elem)

  end do

end subroutine trapdf_4d

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine trapdf_2d(trapping,gupw)
!
! As trapdf_4d (Terms IX and IV in the manual), but second-order
!
  use grid,      only : nx, ns, nmu, nvpar, nsp, nmod
  use components,   only : vthrat
  use geom,         only : bn, gfun
  use velocitygrid, only : mugr, dvp

  real, intent(in) :: trapping, gupw

  integer :: ix, i, j, k, is, ihelp, imod
  integer :: ierr

  real :: dum
  real :: dvpp

  type (matrix_element) :: elem

  if (trapping .eq. 0) return

  elem%itype = ifdis

  do imod=1,nmod; do ix=1,nx; do is=1,nsp; do i=1,ns; do j=1,nmu; do k=1,nvpar

    call set_indx(elem,imod,ix,i,j,k,is)
    elem%iloc = i

    dum=-trapping*vthrat(is)*mugr(j)*bn(i)*gfun(i)

    if (dum .ge. 0) then
      ihelp = k - 1
      dvpp  = -dvp
    else
      ihelp = k + 1
      dvpp  = dvp
    endif

    elem%kloc = k + 1
    elem%val  = -dum*(1.-gupw) / (2.*dvp)
    call add_element(elem,ierr)

    elem%kloc = k - 1
    elem%val  = dum*(1.-gupw) / (2.*dvp)
    call add_element(elem,ierr)

    elem%kloc = ihelp
    elem%val  = -dum*gupw / dvpp
    call add_element(elem,ierr)

    elem%kloc = k
    elem%val  = dum*gupw / dvpp
    call add_element(elem,ierr)

    ! if (nlapar) then
    !   elem%val = - 2.*signz(is)*vthrat(is)**2*mugr(j)                        &
    !            &     *bn(i)*gfun(i)*fmaxwl(i,j,k) / tmp(is)
    !   call add_element(elem,ierr,iapar)
    ! endif

  end do ; end do ; end do ; end do ; end do ; end do

end subroutine trapdf_2d

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine vdgradf
!-----------------------------------------------------------------------------
! adds vdrif grad delta f into the matrix
! (Term II in the manual)
!
! -(1/Z) ci1 [ T_R E_D D^alpha + T_R v_parallelN^2 beta^prime x
!    E^(psi alpha) + 2 m_R v_R v_parallelN H^alpha ) k_alpha
!
! Centrifugal forces are not implemented, also the parallel term proportional
! to the parallel derivative is neglected. 
!-----------------------------------------------------------------------------
  use grid,      only : nx, ns, nmu, nvpar, nsp, nmod
  use dist,         only : indx
  use components,   only : signz, tmp, vthrat
  use mode,         only : krho, kxrh
  use matdat,       only : put_element
  use geom,         only : dfun, efun, hfun, bn
  use velocitygrid, only : vpgr, mugr
  use rotation,     only : vcor
  use constants,    only : ci1
  
  ! integers for the loop over all grid points
  integer :: imod, ix, i, j, k, is

  ! reference integers and matrix element
  complex :: mat_elem
  integer :: iih, jjh

  ! Dummy variables
  complex :: dumc
  real :: ED, daka, epaka, haka
  do is = 1, nsp

    ! clear the timestep estimate
    call time_est(mat_elem,0)

    do imod = 1, nmod ; do ix = 1, nx ; do i = 1, ns

      daka  = dfun(i,1)*kxrh(ix)+ dfun(i,2)*krho(imod)
      epaka = efun(i,1,1)*kxrh(ix)+efun(i,1,2)*krho(imod)
      haka  = hfun(i,1)*kxrh(ix)+ hfun(i,2)*krho(imod)

      do j = 1, nmu ; do k = 1, nvpar

        ED = vpgr(i,j,k)**2 + bn(i)*mugr(j)

        ! the B\times \nabla B component of the drift
        dumc = tmp(is)*ED*daka

        ! The finite beta correction of the curvature
        dumc = dumc + tmp(is)*vpgr(i,j,k)**2*beta_prime*epaka/ bn(i)**2

        ! The coriolis drift correction
        dumc = dumc + 2.E0*tmp(is)*vpgr(i,j,k)*vcor*haka/vthrat(is)

        ! common factor (-I/Z)
        dumc = -ci1*dumc / signz(is)

        iih = indx(imod,ix,i,j,k,is)
        jjh = iih
        mat_elem = dumc
        call put_element(iih,jjh,mat_elem,1)

      end do ; end do ; end do ; end do ; end do

    ! retrieve the time step estimate
    call time_est(mat_elem,2)

    ! write(*,*)'Drift species ',is,' timeest ', 1/real(mat_elem)

  end do

end subroutine vdgradf

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine poisson_int
!-----------------------------------------------------------------------------
! This routine adds the integral part of the Poisson equation in the matrix
!-----------------------------------------------------------------------------
  use grid,      only : nx,ns,nmu,nvpar,nsp,nmod
  use dist,         only : indx
  use components,   only : de, signz
  use geom,         only : bn
  use matdat,       only : put_element
  use velocitygrid, only : intvp, intmu

  real :: dum
  integer :: ix, i, j, k, is, imod
  integer :: iih, jjh
  complex :: mat_elem

  do is = 1, nsp ; do imod = 1, nmod ; do ix = 1, nx ; do i = 1, ns

    iih = indx(imod,ix,i,iphi)

    do j = 1, nmu ; do k = 1, nvpar

      dum = besselj0_gkw(imod,ix,i,j,is)
      jjh = indx(imod,ix,i,j,k,is)
      mat_elem = signz(is)*de(is)*intmu(j)*intvp(i,j,k)*dum*bn(i)
      call put_element(iih,jjh,mat_elem)

    end do ; end do

  end do ; end do ; end do ; end do

end subroutine poisson_int

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine poisson_dia
!-----------------------------------------------------------------------------
! add the diagonal part of the poisson equation into the matrix
!-----------------------------------------------------------------------------
  use grid,      only : nx,ns,nsp,nmod,n_procs_s,number_of_species
  use mpicomms,     only : COMM_SP_NE
  use dist,         only : indx,iadia
  use mode,         only : krloc
  use components,   only : de, tmp, signz, adiabatic_electrons
  use matdat,       only : put_element

  integer :: ix, i, is, imod, ierr
  integer :: iih, jjh
  complex :: mat_elem, cdum


  do imod = 1, nmod ; do ix = 1, nx ; do i = 1, ns

    ! reference the element of the potential
    iih = indx(imod,ix,i,iphi)
    jjh = iih

    ! initialize the mat_element
    mat_elem = (0.,0.)
   
    ! detect the (0,0) mode
    if (abs(krloc(imod,ix,i)) < 1E-5) then
      ! the (0,0) mode does not contain any physics.
      jjh = iih
      mat_elem = mat_elem + 1.
    else
      ! all other modes
      
      ! sum local species contributions, then non-local
      do is = 1, nsp
        mat_elem = mat_elem +                                                &
                 & (signz(is)**2)*de(is)*(gamma_gkw(imod,ix,i,is)-1)/tmp(is)
      end do
#if defined(mpi)
      if (number_of_species > nsp) then
        ierr = 0
        call MPI_ALLREDUCE(mat_elem,cdum,1,MPICOMPLEX_X,MPI_SUM,COMM_SP_NE,ierr)
        mat_elem = cdum
      end if
#endif
      
      ! add adiabatic electrons contribution
      if (adiabatic_electrons) then
        mat_elem = mat_elem + signz(nsp+iadia)*de(nsp+iadia) / tmp(nsp+iadia)
      end if
      
    end if
    
    ! put the element
    call put_element(iih,jjh,mat_elem)

  end do ; end do ; end do

end subroutine poisson_dia

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine poisson_zf
!--------------------------------------------------------------------
! add the zonal flow corrections
!--------------------------------------------------------------------
  use specfun,      only : bessi0
  use control,      only : zonal_adiabatic
  use grid,         only : nx,ns,nsp,nmod, number_of_species,ispb, &
                         & parallel_s
  use mpicomms,     only : COMM_SP_NE, COMM_S_NE
  use dist,         only : indx,iadia
  use mode,         only : krho,krloc
  use components,   only : de, tmp, signz, adiabatic_electrons, rhorat
  use matdat,       only : put_elem_zonal
  use geom,         only : bn, ints

  real :: dum1, dum2
  integer :: ix, i, is, imod, ierr
  integer :: iih, jjh
  complex :: mat_elem, cdum
  complex :: dum_elem
  real :: krloc_small, krloc_tmp

  ! If zonal_adiabatic = F then no zonal flow correction is used
  if (.not. zonal_adiabatic) return

  ! The correction is of importance only for adiabatic electrons
  if (.not.adiabatic_electrons) return

  ! Find the zonal mode
  i = 0
  do imod = 1, nmod
    if (abs(krho(imod)) < 1E-5) then
      i = imod
    endif
  end do
  ! No krho = 0 mode, i.e. no zonal flow correction
  if (i .eq. 0) return

  imod = i
  x_grid : do ix = 1, nx
    ! initialize the dummy element
    dum_elem = (0.E0,0.E0)

    s_grid : do i = 1, ns

      ! reference the element of the potential
      iih = ix

      ! reference the position in the average
      ! array
      jjh = indx(imod,ix,i,iphi)

      ! initialize the matrix element to zero
      mat_elem = (0.E0,0.E0)

      do is = 1, nsp
        dum1 = 0.5*(rhorat(is)*krloc(imod,ix,i)/bn(i))**2
        dum2 = ((exp(-dum1)*bessi0(dum1)-1.)/tmp(is)) -(1/tmp(nsp+iadia))
        mat_elem = mat_elem + (signz(is)**2)*de(is)*dum2
      end do

      ! sum all the diagonal contributions
#if defined(mpi)
      if (nsp < number_of_species) then
        ierr = 0
        call MPI_ALLREDUCE(mat_elem,cdum,1,MPICOMPLEX_X,MPI_SUM, COMM_SP_NE, &
            &              ierr)
        mat_elem = cdum
      endif
#endif

      mat_elem = -ints(ix,i) / mat_elem

      ! put first element
      call put_elem_zonal(iih,jjh,mat_elem,1)

      dum_elem = dum_elem - mat_elem

    end do s_grid

      ! sum all the dum elements over the s-direction?
#if defined(mpi)
      if (parallel_s) then
        ierr = 0
        call MPI_ALLREDUCE(dum_elem,cdum,1,MPICOMPLEX_X,MPI_SUM, COMM_S_NE,   &
            &              ierr)
        dum_elem = cdum
      endif
#endif

    dum_elem = dum_elem + tmp(nsp+iadia) / de(nsp+iadia)

    if (ispb == 1) then
      krloc_tmp = krloc(imod,ix,1)
    else
      krloc_tmp = huge(1.)
    endif
    call mpiallreduce_min(krloc_tmp,krloc_small,1)
    if (abs(krloc_small) < 1e-5) then
        dum_elem = (1.E0,0.E0)
    endif

    ! put element in the matrix
    call put_elem_zonal(iih,iih,dum_elem,2)

  end do x_grid

end subroutine poisson_zf

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine g2f_correct
!--------------------------------------------------------------------
! This routine caculates the correction necessary to go from the
! distribution g (which includes the correction of the parallel
! vector potential) to the distribution f
!--------------------------------------------------------------------
  use index_function, only : index_invert_,index_reorder
  use dist,         only : nphi, fmaxwl, indx
  use components,   only : signz, vthrat, tmp
  use matdat,       only : put_element_correct_apar
  use velocitygrid, only : vpgr

  integer :: ii, imod, ix, i, j, k, is, jjh
  real :: b0
  complex :: mat_elem

  integer, dimension(6) :: starts,ends
  integer :: l,m,n
  integer :: i_s, i_sp, i_mu, i_vpar, i_x, i_mod
  
  call index_invert_(starts,ends)
  do n=starts(6),ends(6)
    do m=starts(5),ends(5)
      do l=starts(4),ends(4)
        do k=starts(3),ends(3)
          do j=starts(2),ends(2)
            do i=starts(1),ends(1)
              call index_reorder(i,j,k,l,m,n,i_mod,i_x,i_s,i_mu,i_vpar,i_sp)
              b0 = besselj0_gkw(i_mod,i_x,i_s,i_mu,i_sp)
              mat_elem = -2.E0*signz(i_sp)*vthrat(i_sp)*vpgr(i_s,i_mu,i_vpar)*b0*fmaxwl(i_s,i_mu,i_vpar)/tmp(i_sp)
              jjh = indx(i_mod,i_x,i_s,iapar)
              call put_element_correct_apar(jjh,mat_elem)
            end do
          end do
        end do
      end do
    end do
  end do
              
end subroutine g2f_correct

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine ampere_int
!--------------------------------------------------------------------
! Add the part of the Ampere's equation that is related with the
! integral over the distribution function
!--------------------------------------------------------------------
  use control,      only : nlapar
  use grid,         only : nx,ns,nmu,nvpar,nsp,nmod
  use dist,         only : indx
  use components,   only : de, signz, vthrat, beta
  use matdat,       only : put_element
  use geom,         only : bn
  use velocitygrid, only : vpgr, intvp, intmu

  real :: bes
  integer :: ix, i, j, k, is, imod
  integer :: iih, jjh
  complex :: mat_elem

  if (.not. nlapar) return

  do imod = 1, nmod ; do ix = 1, nx ; do i = 1, ns

    ! reference Apar
    iih = indx(imod,ix,i,iapar)

    do j = 1, nmu ; do k = 1, nvpar ; do is = 1, nsp

      ! Integral over the distribution
      jjh = indx(imod,ix,i,j,k,is)

      ! Bessel function for gyro-average
      bes = besselj0_gkw(imod,ix,i,j,is)

      mat_elem = signz(is)*de(is)*beta*intvp(i,j,k)*intmu(j)                   &
          &    *vthrat(is)*bn(i)*vpgr(i,j,k)*bes

      call put_element(iih,jjh,mat_elem)

    end do ; end do ; end do

  end do ; end do ; end do

end subroutine ampere_int

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine ampere_dia
!--------------------------------------------------------------------
! Add the diagonal part of the Ampere's equation
!--------------------------------------------------------------------
  use control,      only : nlapar
  use grid,         only : nx,ns,nsp,nmod,nmu,nvpar
  use mpicomms,     only : COMM_S_EQ
  use dist,         only : indx,fmaxwl
  use components,   only : de, signz, mas, beta
  use matdat,       only : put_element
  use geom,         only : bn
  use mode,         only : krloc
  use velocitygrid, only : intvp, intmu

  real :: gamma, gamma_num, b, dum, dums
  integer :: imod, ix, i, j, k, is, ierr
  integer :: iih, jjh
  complex :: mat_elem

  if (.not. nlapar) return

  do imod = 1, nmod ; do ix = 1, nx ; do i = 1, ns

    ! reference Apar
    iih = indx(imod,ix,i,iapar)
    jjh = indx(imod,ix,i,iapar)

    ! The nabla^2 term
    mat_elem =  - krloc(imod,ix,i)**2

    ! calculate the Maxwell correction
    dum = 0.
    do is = 1, nsp

      ! The gamma function of the species
      gamma = gamma_gkw(imod,ix,i,is)

      ! numerical calculation of the gamma function
      gamma_num = 0.
      do j = 1, nmu ; do k = 1, nvpar
        b = besselj0_gkw(imod,ix,i,j,is)
        gamma_num = gamma_num + bn(i)*intmu(j)*intvp(i,j,k)*b**2*fmaxwl(i,j,k)
      end do ; end do

      ! The 'Maxwell correction'
      dum = dum - beta*signz(is)**2*de(is)*gamma_num / mas(is)

    end do

#if defined(mpi)
    ! MPI sum of the Maxwell correction
    if (number_of_processors .gt. 1) then
      ierr = 0
      call MPI_ALLREDUCE(dum,dums,1,MPIREAL_X,MPI_SUM, COMM_S_EQ, ierr)
      dum = dums
    endif
#endif

    mat_elem = mat_elem + dum

    call put_element(iih,jjh,mat_elem)

  end do ; end do ; end do

end subroutine ampere_dia

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine ve_grad_fm
!--------------------------------------------------------------------
! This routine adds the v_chi \nabla Fm term
! Term V in the manual
! This routine would be better named vchi_grad_fm as
! it now also includes v_del_B_perp
!
! +  I E^(a psi) k_a chi [ rln + Et rlt + 2 v_parallelN u^prime / v_R
!
! in the matrix.
!--------------------------------------------------------------------
  use control,      only : nlapar, nlbpar
  use grid,         only : nx,ns,nmu,nvpar,nsp,nmod
  use dist,         only : fmaxwl,indx,falpha
  use components,   only : fp, tp, vp, vthrat, types, pbg
  use mode,         only : krho, kxrh
  use matdat,       only : put_element
  use geom,         only : bn, efun, bt_frac, bp_frac
  use rotation,     only : toroidal_shear
  use velocitygrid, only : vpgr, mugr
  use constants,    only : ci1
  
  integer :: ix, i, j, k, is, imod
  integer :: iih, jjh
  complex :: mat_elem
  real :: dum, b, ET, ekapka, vn, uparallel

  ! calculate the terms due to the Maxwell background
  do imod=1,nmod; do is=1,nsp; do ix=1,nx; do i=1,ns; do j=1,nmu; do k=1,nvpar

    !calculate uparallel
    if (toroidal_shear) then
      !vp is actually the (unscaled) shear rate in this case
      uparallel=vp(is)/bp_frac(i) 
    else
      !vp is uprim
      uparallel=bt_frac(i)*vp(is)
    end if

    ! term in front of the temperature gradient
    ET = vpgr(i,j,k)**2 + 2.E0*mugr(j)* bn(i) - 1.5E0

    ! E^(alpha psi) k_alpha
    ekapka = efun(i,1,1)*kxrh(ix) + efun(i,2,1)*krho(imod)

    dum = (fp(is)+ET*tp(is)+2.E0*uparallel*vpgr(i,j,k)/vthrat(is))*    &
        & fmaxwl(i,j,k)*ekapka

    ! for an alpha particle distribution change to
    if (types(is) .eq. 'alpha') then
      vn = sqrt(vpgr(i,j,k)**2 + 2.E0*mugr(j)*bn(i))
      ET =   3.E0 / 2.E0 * (1.E0 / (log(1.E0 + 27.E0*pbg(is)**(-1.5))*         &
          & (1.E0+pbg(is)**(1.5)/27E0)) - pbg(is)**1.5/ (pbg(is)**1.5 +        &
          & vn**3))
      dum = (fp(is) + ET*tp(is))*falpha(i,j,k)*ekapka
    endif

    b = besselj0_gkw(imod,ix,i,j,is)

    iih = indx(imod,ix,i,j,k,is)
    jjh = indx(imod,ix,i,iphi)
    mat_elem = ci1*dum*b
    call put_element(iih,jjh,mat_elem)

    ! Electromagnetic correction, v_del_B_perp
    if (nlapar) then
      iih = indx(imod,ix,i,j,k,is)
      jjh = indx(imod,ix,i,iapar)
      mat_elem = -2.*mat_elem*vthrat(is)*vpgr(i,j,k)
      call put_element(iih,jjh,mat_elem)
    endif

!!     ! Compressional correction, v_grad_B_par
!!     if (nlbpar) then
!!       iih = indx(imod,ix,i,j,k,is)
!!       jjh = indx(imod,ix,i,ibpar)
!!       mat_elem = mat_elem.*bn(i).*tmp(is)
!!       call put_element(iih,jjh,mat_elem)
!!     endif

  end do ; end do ; end do ; end do ; end do ; end do

end subroutine ve_grad_fm

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine neoclassical
!--------------------------------------------------------------------
! This routine adds the neoclassical terms
! Term VI in the manual
!
! (1/Z)(T_R E_D D^psi + 2 m_R v_R v_parallelN H^psi ) (rln + Et rlt
!   2 v_parallelN u^prime / v_R
!
! in the matrix.
!--------------------------------------------------------------------
  use grid,      only : nx, ns, nmu, nvpar, nsp, nmod
  use dist,         only : fmaxwl, indx
  use components,   only : fp, tp, vp, vthrat, tmp, signz
  use rotation,     only : vcor, toroidal_shear
  use mode,         only : krho, kxrh
  use matdat,       only : put_source
  use geom,         only : bn, dfun, hfun, bt_frac, bp_frac
  use velocitygrid, only : vpgr, mugr
  use constants,    only : ci1
  
! integers for the loop over all grid points
  integer :: imod, ix, i, j, k, is

  complex :: mat_elem

! dummy variables
  integer :: iih
  real :: dum, ET, ED, uparallel

! calculate the terms due to the Maxwell background
  do imod = 1, nmod ; do is=1,nsp ; do ix=1,nx ; do i=1,ns ; do j=1,nmu ; do k=1,nvpar

    if ((abs(krho(imod)) .lt. 1e-6).and.(abs(kxrh(ix)).lt.1e-6)) then

      !calculate uparallel
      if (toroidal_shear) then
        !vp is actually the (unscaled) shear rate in this case
        uparallel=vp(is)/bp_frac(i) 
      else
        !vp is uprim
        uparallel=bt_frac(i)*vp(is)
      end if

      ! term in from of the themperature gradient
      ET = vpgr(i,j,k)**2 + 2.E0*mugr(j)* bn(i) - 1.5E0

      ED = vpgr(i,j,k)**2 + bn(i)*mugr(j)

      ! the B\times \nabla B component of the drift
      dum = tmp(is)*ED*dfun(i,1)

      ! The coriolis drift correction
      dum = dum + 2.E0*tmp(is)*vpgr(i,j,k)*vcor*hfun(i,1)/vthrat(is)

      dum = (fp(is)+ET*tp(is)+2.E0*uparallel*vpgr(i,j,k)/vthrat(is)) &
          & *fmaxwl(i,j,k)*dum/signz(is)

      iih = indx(imod,ix,i,j,k,is)

      mat_elem = dum
      call put_source(iih,mat_elem)

    endif

  end do ; end do ; end do ; end do  ; end do ; end do


end subroutine neoclassical

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine vd_grad_phi_fm
!--------------------------------------------------------------------
! This routine puts the drift in the gradient of phi times the maxwel
! Term VIII in the manual
!
! - (E_D D^alpha + beta^prime v_parallelN^2 E^(psi alpha) + 2 (m_R
!   v_R / T_R) v_parallel N H^alpha )k_alpha chi_N F_MN
!--------------------------------------------------------------------
  use control,      only : nlapar
  use grid,         only : nx, ns, nmu, nvpar, nsp, nmod
  use dist,         only : fmaxwl, indx, falpha
  use mode,         only : krho, kxrh
  use components,   only : vthrat, tmp, types, pbg
  use rotation,     only : vcor
  use matdat,       only : put_element
  use geom,         only : bn, dfun, efun, hfun
  use velocitygrid, only : vpgr, mugr
  use constants,    only : ci1

  ! integers for the loop over all grid points
  integer :: imod, ix, i, j, k, is

  ! reference values and matrix element
  integer :: iih, jjh
  complex :: mat_elem

  ! Dummy variables
  real :: daka, epaka, haka, b, ED, dum, vn
  do imod = 1, nmod ; do is = 1, nsp ; do ix = 1, nx ; do i = 1, ns

    daka  = dfun(i,1)*kxrh(ix)+ dfun(i,2)*krho(imod)
    epaka = efun(i,1,1)*kxrh(ix)+efun(i,1,2)*krho(imod)
    haka  = hfun(i,1)*kxrh(ix)+ hfun(i,2)*krho(imod)

    do j = 1, nmu ; do k = 1, nvpar

      ED = vpgr(i,j,k)**2 + bn(i)*mugr(j)

      ! the B\times \nabla B component of the drift
      dum = ED*daka

      ! The finite beta correction of the curvature
      dum = dum + vpgr(i,j,k)**2*beta_prime*epaka/ bn(i)**2

      ! The coriolis drift correction
      dum = dum + 2.E0*vpgr(i,j,k)*vcor*haka/vthrat(is)

      ! The bessel function for gyro-averaging
      b = besselj0_gkw(imod,ix,i,j,is)

      iih = indx(imod,ix,i,j,k,is)
      jjh = indx(imod,ix,i,iphi)
      mat_elem = -ci1*dum*b*fmaxwl(i,j,k)

      if (types(is) .eq. 'alpha') then
        vn = sqrt(vpgr(i,j,k)**2+2.E0*mugr(j)*bn(i))
        mat_elem = -ci1*dum*b*falpha(i,j,k)*(3./2.)*vn / (pbg(is)**1.5 + vn**3)
      endif

      call put_element(iih,jjh,mat_elem)

      ! Electromagnetic correction
      !   if (nlapar) then
      !     jjh = indx(imod,ix,i,iapar)
      !     mat_elem = -2.*mat_elem*vthrat(is)*vpgr(i,j,k)
      !     call put_element(iih,jjh,mat_elem)
      !   endif
      !call matout(iih,jjh,mat_elem)

    end do ; end do ; end do

  end do ; end do ; end do

end subroutine vd_grad_phi_fm

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

! Newly modified version of the fourth order scheme with the new upwinded
!'flappy' boundary condition
subroutine vpar_grad_df_4d_testnewbc(disp)
!--------------------------------------------------------------------
! This routine puts the motion along the field line
! (Term I in the manual)
!
! -v_R v_parallelN F (d f / d s)
!
! in the matrix. The implementation is fourth order in space. Disp
! determines the parallel disipation. The parallel boundary condi-
! tions are implemented throug a call to connect_parallel, and an !
! estimate of the critial time step is made for every species
!--------------------------------------------------------------------
  use dist,         only : indx,iapar,iphi
  use geom,         only : sgr, ffun, sgr_dist
  use matdat,       only : put_element
  use components,   only : vthrat
  use mode,         only : ixplus, ixminus
  use control,      only : order_of_the_scheme
  use grid,         only : nx,ns,nmu,nvpar,nsp,nmod
  use velocitygrid, only : vpgr, vpgr_rms

! The dissipation coefficient
  real, intent(in) :: disp

! The integers for the loop over all grid points
  integer :: imod, ix, i, j, k ,is, kref

! the variables used in the parallel boundary conditions
  integer :: iatempt, ixref, iref, ist
  logical :: ingrid

! the element to attempt
  type (matrix_element) :: elem

! dummy matrix element
  complex :: mat_elem

! dummy variables
  real :: dum, dum2, gm

! for the error code from add_element
  integer :: ierr

! if only one parallel grid point (2D case) return
  if (ns .lt. 2) return

! Set the upwinding parameter for the end points of the field line
! Upwinding parameter? WAH - Not actually used ??
  gm  = 1.0

  do is = 1, nsp

    ! clear the time step estimate
    call time_est(mat_elem,0)

    do imod=1,nmod ; do ix=1,nx ; do j = 1,nmu ; do k=1,nvpar ; do i=1,ns

      call set_indx(elem,imod,ix,i,j,k,is)

      ! check if the point is a begining or end
      iatempt = i
      call connect_parallel(imod,ix,i,k,iatempt,ingrid,ixref,iref,kref,ist)

      ! WARNING switched of end point correction
      !ist = 1
      dum = -ffun(i)*vthrat(is)*vpgr(i,j,k)

      select case(idisp)
        case(1) ; dum2  = dum
        case(2) ; dum2  = ffun(i)*vthrat(is)*vpgr_rms
        case default ; call gkw_abort('vpar_grad_df_4d: unknown idisp')
      end select

      ! true for all elements
      elem%kloc = k

      ! The boundary conditions have four different cases because of the
      ! upwinding scheme. Used on the opposite of the flappy boundary ->
      ! this also depends greatly on the local advective velocity.

      scheme_order : select case(order_of_the_scheme)
        case('fourth_order')

        grid_location_4th : select case(ist)

          case(-2) ! Point on left hand boundary

            if (dum .gt. 0) then
              ! Second order backwinded difference scheme
              elem%iloc = i
              elem%val  = -3.E0*abs(dum)/(2.E0*sgr_dist)
              call add_element(elem,ierr,1)

              elem%iloc = i + 1
              elem%val  = (2.E0*dum) /(1.E0*sgr_dist)
              call add_element(elem,ierr)

              elem%iloc = i + 2
              elem%val  = (-1.E0*dum) / (2.E0*sgr_dist)
              call add_element(elem,ierr)

            else if (dum .lt. 0) then
              ! Centre differenced second order scheme with a zero ghost cell
              elem%iloc = i
              elem%val  = -2.E0*disp*abs(dum2)/(1.E0*sgr_dist)
              call add_element(elem,ierr)

              elem%iloc = i + 1
              elem%val  = (1.E0*dum + 2.E0*disp*abs(dum2)) / (2.E0*sgr_dist)
              call add_element(elem,ierr)

            ! else ? is the dum == 0 case needed ?
            endif

          case(-1) ! One point in from the left hand boundary

            if (dum .gt. 0) then
              ! Third order backwinded difference scheme
              elem%iloc = i
              elem%val  = -(1.E0*dum)/(2.E0*sgr_dist)
              call add_element(elem,ierr)

              elem%iloc = i + 1
              elem%val  = (1.E0*dum) /(1.E0*sgr_dist)
              call add_element(elem,ierr)

              elem%iloc = i + 2
              elem%val  = -(1.E0*dum) / (6.E0*sgr_dist)
              call add_element(elem,ierr)

              elem%iloc = i - 1
              elem%val  = -(1.E0*dum)  /(3.E0*sgr_dist)
              call add_element(elem,ierr)

            else if (dum .lt. 0) then
              ! Fourth order cental differnced with a ghost cell
              elem%iloc = i - 1
              elem%val  = (-8.E0*dum + 4.E0*disp*abs(dum2)) / (12.E0*sgr_dist)
              call add_element(elem,ierr)

              elem%iloc = i
              elem%val  = -6.E0*disp*abs(dum2)/(12.E0*sgr_dist)
              call add_element(elem,ierr)

              elem%iloc = i + 1
              elem%val  = (8.E0*dum + 4.E0*disp*abs(dum2)) /(12.E0*sgr_dist)
              call add_element(elem,ierr)

              elem%iloc = i + 2
              elem%val  = (-1.E0*dum - 1.E0*disp*abs(dum2))/(12.E0*sgr_dist)
              call add_element(elem,ierr)
            ! else ? Is the case dum == 0 needed?
            endif

          case(0) ! Point is somewhere in the middle ->
                  ! this is the only part with disp.
            elem%iloc = i - 2
            elem%val  = (1.E0*dum-1.E0*disp*abs(dum2))/(12.E0*sgr_dist)
            call add_element(elem,ierr)

            elem%iloc = i - 1
            elem%val  = (-8.E0*dum+4.E0*disp*abs(dum2))/(12.E0*sgr_dist)
            call add_element(elem,ierr)

            elem%iloc = i
            elem%val  = -6.E0*disp*abs(dum2)/(12.E0*sgr_dist)
            call add_element(elem,ierr)

            elem%iloc = i + 1
            elem%val  = (8.E0*dum+4.E0*disp*abs(dum2)) /(12.E0*sgr_dist)
            call add_element(elem,ierr)

            elem%iloc = i + 2
            elem%val  = (-1.E0*dum-1.E0*disp*abs(dum2))/(12.E0*sgr_dist)
            call add_element(elem,ierr)

          case(1) ! One in from the right hand boundary cell

            if (dum .gt. 0) then
              ! Fourth order central differenced with a ghost cell
              elem%iloc = i - 2
              elem%val  = (1.E0*dum - 1.E0*disp*abs(dum2))/(12.E0*sgr_dist)
              call add_element(elem,ierr)

              elem%iloc = i - 1
              elem%val  = (-8.E0*dum + 4.E0*disp*abs(dum2))/(12.E0*sgr_dist)
              call add_element(elem,ierr)

              elem%iloc = i
              elem%val  = -6.E0*disp*abs(dum2)/(12.E0*sgr_dist)
              call add_element(elem,ierr)

              elem%iloc = i + 1
              elem%val  = (8.E0*dum + 4.E0*disp*abs(dum2)) /(12.E0*sgr_dist)
              call add_element(elem,ierr)

            else if (dum .lt. 0) then
              ! Third order backwinded scheme
              elem%iloc = i
              elem%val  = +1.E0*dum/(2.E0*sgr_dist)
              call add_element(elem,ierr)

              elem%iloc = i + 1
              elem%val  = (1.E0*dum)/(3.E0*sgr_dist)
              call add_element(elem,ierr)

              elem%iloc = i - 1
              elem%val  = (-1.E0*dum)/(1.E0*sgr_dist)
              call add_element(elem,ierr)

              elem%iloc = i - 2
              elem%val  = (1.E0*dum)/(6.E0*sgr_dist)
              call add_element(elem,ierr)
            ! else ? is the dum == 0 case needed ?
            endif

          case(2) ! Right hand boundary cell

            if (dum .gt. 0) then
              ! Second order cental difference with a zero ghost cell
              elem%iloc = i - 1
              elem%val  = (-1.E0*dum + 2.E0*disp*abs(dum2))/(2.E0*sgr_dist)
              call add_element(elem,ierr)

              elem%iloc = i
              elem%val  = -2.E0*disp*abs(dum2)/(1.E0*sgr_dist)
              call add_element(elem,ierr)

            else if (dum .lt. 0) then
              ! Second order backwinded scheme
              elem%iloc = i
              elem%val  = (3.E0*dum)/(2.E0*sgr_dist)
              call add_element(elem,ierr)

              elem%iloc = i - 1
              elem%val  = (-2.E0*dum)/(1.E0*sgr_dist)
              call add_element(elem,ierr)

              elem%iloc = i - 2
              elem%val  = (1.E0*dum)/(2.E0*sgr_dist)
              call add_element(elem,ierr)
            ! else ? is the dum == 0 case needed ?
            endif

          case default

            call gkw_abort('Cases are -2:2 in Boundary conditions of v_par')

        end select grid_location_4th

      case('second_order')

        grid_location_2nd : select case(ist)

          case(-1,0,1) ! Within the grid

            elem%iloc = i - 1
            elem%val  = (-1.E0*dum+2.E0*disp*abs(dum2)) /(2.E0*sgr_dist)
            call add_element(elem,ierr)

            elem%iloc = i
            elem%val  = -2.E0*disp*abs(dum2)/(1.E0*sgr_dist)
            call add_element(elem,ierr)

            elem%iloc = i + 1
            elem%val  = (1.E0*dum+2.E0*disp*abs(dum2))/(2.E0*sgr_dist)
            call add_element(elem,ierr)

          case(-2)

            if (dum .gt. 0E0) then

              elem%iloc = i
              elem%val  = -1.E0*dum/(1.E0*sgr_dist)
              call add_element(elem,ierr)

              elem%iloc = i + 1
              elem%val  = (1.E0*dum)/(1.E0*sgr_dist)
              call add_element(elem,ierr)

            else if (dum .lt. 0.) then

              elem%iloc = i + 1
              elem%val  = (1.E0*dum)/(2.E0*sgr_dist)
              call add_element(elem,ierr)
            ! else ? is the dum == 0 case needed ?
            end if

          case(2)

            if (dum .gt. 0E0) then

              elem%iloc = i - 1
              elem%val  = (-1.E0*dum)/(2.E0*sgr_dist)
              call add_element(elem,ierr)

            else if (dum .lt. 0.) then

              elem%iloc = i
              elem%val  = 1.E0*dum/(1.E0*sgr_dist)
              call add_element(elem,ierr)

              elem%iloc = i - 1
              elem%val  = (-1.E0*dum) /(1.E0*sgr_dist)
              call add_element(elem,ierr)
            ! else ? dum ==0 ?
            end if

          case default

            call gkw_abort('Error in second order scheme of vpgr')

        end select grid_location_2nd

      end select scheme_order

    end do ; end do ; end do ; end do ; end do

    ! retrieve the estimate of the timestep
    call time_est(mat_elem,2)

    ! write(*,*) 'parallel species ',is,'  ', 1./real(mat_elem)

  end do

end subroutine vpar_grad_df_4d_testnewbc

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Add terms I and IV as one term : v_R ffun {H,g_N}
!! where
!!         H  = 1/2 v_{\parN}^2 + \mu_N B_N  = H(s,\mu,v_{\par}),
!! and
!!         {H,g_N} = (d H/d s)(d g_N/d v) - (d H/d v)(d g_N/d s).
!!
!! At the boundaries in the s-direction, we apply something similar to the
!! wills=1 scheme to do upwinding. Note that the routines that this calls to
!! deal with points near the boundaries do not presently work in second order.
!!
!! See jhg_interior for the main bit of these terms.
!!
!! ( Note that above we dropped the second index with respect to `j'(for \mu)
!<  in H; the second index is always j. Below we use `HH' for `H')
!----------------------------------------------------------------------------

subroutine igh(disp_par,disp_vp)

  use components,   only : vthrat
  use grid,         only : nx,ns,nmu,nvpar,nsp,nmod
  use general,      only : time_est
  use geom,         only : ffun, sgr_dist, gfun, bn
  use velocitygrid, only : dvp,vpgr,mugr,vpgr_rms,mugr_rms

  real, intent(in) :: disp_vp, disp_par
  real :: dum, dum2, disp_v_dum, disp_s_dum
  integer :: is, imod, ix, i ,j ,k, ierr, ist,iref,kref,ixref

  type (matrix_element) :: elem
  complex :: mat_elem
  logical :: ingrid
  
  disp_v_dum =  0.
  disp_s_dum  =  0.
  ! all elements here are referencing f
  elem%itype = ifdis
  
  do is = 1, nsp

    ! clear the time step estimate
    call time_est(mat_elem,0)

    do imod=1,nmod ; do ix=1,nx ; do j = 1,nmu ; do k=1,nvpar ; do i=1,ns
      
      ! copy the loop indices to the element
      call set_indx(elem,imod,ix,i,j,k,is)
      
      ! needed for the function HH
      jref_mu = j

      ! common factor of the mat elements
      dum = vthrat(is)*ffun(i)/(sgr_dist*dvp)

      ! dum2 is for the advection sign and (standard) diffusion in s
      dum2 = -ffun(i)*vthrat(is)*vpgr(i,j,k)

      ! disp_v_dum is for diffusion in vpar
      select case(idisp)
        case(1) ; disp_v_dum = vthrat(is)*mugr(j)*bn(i)*gfun(i)
        case(2) ; disp_v_dum = vthrat(is)*mugr_rms*bn(i)*gfun(i)
      end select

      ! disp_s_dum is for diffusion in s
      select case(idisp)
        case(1) ; disp_s_dum = dum2
        case(2) ; disp_s_dum = ffun(i)*vthrat(is)*vpgr_rms
      end select
 
      ! Check if the point is near the boundary in s; 0 denotes somewhere in
      ! the middle and +/- 1, +/- 2 correspond to the points nearest the
      ! boundary.
      !
      !      ist:  -2 -1     0    1  2                  
      ! location:   |  |  -  -  - |  | 
      !
      call connect_parallel(imod,ix,i,k,i,ingrid,ixref,iref,kref,ist)
     
      ! Add different term depending on where the point is on the grid *and*
      ! the sign of the advection velocity.
      select case(ist)
        case(-2,2)
          
          if (dum2*ist < 0.) then
            call igh_zero_two(elem,dum,ist,dum2)
          else if (dum2*ist > 0.) then
            call jhg_interior(elem,dum)
          else
            ! do nothing; vanishing term
          end if
          
        case(-1,1)
          
          if (dum2*ist < 0.) then
            call igh_two(elem,dum,ist,dum2)
          else if (dum2*ist > 0.) then
            call jhg_interior(elem,dum)
          else
            ! do nothing; vanishing term
          end if
          
        case(0)
          
          ! original Arakawa differencing of the right order.
          call jhg_interior(elem,dum)
          
          ! only apply any diffusion (hyperdiffusion) to the interior part of
          ! the grid
          if (disp_par > 0.) call diffus(elem,disp_par,disp_s_dum,'s')
          if (disp_vp  > 0.) call diffus(elem,disp_vp, disp_v_dum,'vpar')

        case default
          
          call gkw_abort('igh: bad case of ist')
          
      end select
 
    end do ; end do ; end do ; end do ; end do

    call time_est(mat_elem,2)

  end do

end subroutine igh

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> This routine is called from igh in order to deal with points at the end of
!> the s grid with upwinding.
!> We difference 
!>               {H,g_N} = (d H/d s)(d g_N/d v) - (d H/d v)(d g_N/d s)
!> as
!>
!>               d Z(i)/d s = 1/(2 \delta s) [3 Z(i) - 4 Z(i-1) + Z(i-2)]
!>               d Z(j)/d v = 1/(2 \delta v) [Z(i+1) - Z(i-1)]
!>
!> For dum = vpgr*vthrat/(d v d s) and case(ist), we have the following;
!>
!> case(-2) .and. dum > 0:
!>
!>     elems = 0.25*dum*[
!>          + {  -3.*H(i,k) + 4.*H(i+1,k) - H(i+2,k) }    g(i,k+1)
!>          - {  -3.*H(i,k) + 4.*H(i+1,k) - H(i+2,k) }    g(i,k-1)
!>          - {      H(i,k+1) - H(i,k-1)             } -3*g(i,k) 
!>          - {      H(i,k+1) - H(i,k-1)             }  4*g(i+1,k)
!>          - {      H(i,k+1) - H(i,k-1)             }   -g(i+2,k)
!>                       ]
!>
!> case(-2) .and. dum < 0 .or. case(2) .and. dum > 0:
!>
!>     (apply J_1 or J_2 with regular bcs; this routine is not called)
!>
!> case(2) .and. dum < 0:
!>
!>     elems = 0.25*dum*[
!>          + {   3.*H(i,k) - 4.*H(i-1,k) + H(i-2,k) }    g(i,k+1)
!>          - {   3.*H(i,k) - 4.*H(i-1,k) + H(i-2,k) }    g(i,k-1)
!>          - {      H(i,k+1) - H(i,k-1)             }  3*g(i,k) 
!>          - {      H(i,k+1) - H(i,k-1)             } -4*g(i-1,k)
!>          - {      H(i,k+1) - H(i,k-1)             }    g(i-2,k)
!>                       ]
!----------------------------------------------------------------------------

subroutine igh_zero_two(elem,dum,ist,dum2)

  type (matrix_element), intent(inout) :: elem
  integer, intent(in) :: ist
  integer :: i, k
  real, intent(in) :: dum,dum2
  real :: val
  real, parameter :: fac=0.25
  integer :: ierr

  i = elem%i
  k = elem%k

  if (ist == 2 .and. dum2 < 0.) then
    
    val = fac*dum*(3.*HH(i,k) - 4.*HH(i-1,k) + 1.*HH(i-2,k))
    
    elem%iloc = i
    elem%kloc = k-1
    elem%val  = -1.*val
    call add_element(elem,ierr)
    
    elem%iloc = i
    elem%kloc = k+1
    elem%val  =  1.*val
    call add_element(elem,ierr)

    val = fac*dum*( 1.*HH(i,k+1) - 1.*HH(i,k-1))
    
    elem%iloc = i
    elem%kloc = k
    elem%val  = -3.*val
    call add_element(elem,ierr)
  
    elem%iloc = i-1
    elem%kloc = k
    elem%val  =  4.*val
    call add_element(elem,ierr)

    elem%iloc = i-2
    elem%kloc = k
    elem%val  = -1.*val
    call add_element(elem,ierr)

  else if (ist == -2 .and. dum2 > 0) then
    
    val = fac*dum*(-3.*HH(i,k) + 4.*HH(i+1,k) - 1.*HH(i+2,k)) 
    
    elem%iloc = i
    elem%kloc = k-1
    elem%val  = -1.*val
    call add_element(elem,ierr)
    
    elem%iloc = i
    elem%kloc = k+1
    elem%val  =  1.*val
    call add_element(elem,ierr)

    val = fac*dum*(1.*HH(i,k+1) - 1.*HH(i,k-1))
    
    elem%iloc = i
    elem%kloc = k
    elem%val  = 3.*val
    call add_element(elem,ierr)
  
    elem%iloc = i+1
    elem%kloc = k
    elem%val  = -4.*val
    call add_element(elem,ierr)

    elem%iloc = i+2
    elem%kloc = k
    elem%val  = 1.*val
    call add_element(elem,ierr)

  else
    
    call gkw_abort('igh_zero_two: bad call to this routine')
    
  end if

end subroutine igh_zero_two

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> This is a similar routine to igh_zero_two; differences
!>   
!>      {H,g_N} = (d H/d s)(d g_N/d v) - (d H/d v)(d g_N/d s)
!>
!> at the *second* from end point of the s direction with upwinds
!> dum = vpgr*vthrat/(d v d s).
!>
!> d Z(i)/d s = 1/(6 \delta s)  [2 Z(i+1) + 3 Z(i)   - 6 Z(i-1) + Z(i-2)]
!>      *or*    1/(6 \delta s) [- Z(i+2) + 6 Z(i+1) - 3 Z(i)   - 2 z(i-1)]
!> d Z(j)/d v = 1/(12 \delta v) [Z(i-2) - 8 Z(i-1) + 8 Z(i+1) - Z(i+2) ]
!>
!> case(-1) .and. dum > 0:
!>
!>   elems = (1/72)*dum*[
!>     + { - 2.*H(i-1,k) - 3.*H(i,k)   + 6.*H(i+1,k) - H(i+2,k) }    g(i,k-2)
!>     - { - 2.*H(i-1,k) - 3.*H(i,k)   + 6.*H(i+1,k) - H(i+2,k) }  8*g(i,k-1)
!>     + { - 2.*H(i-1,k) - 3.*H(i,k)   + 6.*H(i+1,k) - H(i+2,k) }  8*g(i,k+1)
!>     - { - 2.*H(i-1,k) - 3.*H(i,k)   + 6.*H(i+1,k) - H(i+2,k) }    g(i,k+2)
!>   -   {      H(i,k-2) - 8.*H(i,k-1) + 8.*H(i,k+1) - H(i,k+2) } -2*g(i-1,k)
!>   -   {      H(i,k-2) - 8.*H(i,k-1) + 8.*H(i,k+1) - H(i,k+2) } -3*g(i,k)
!>   -   {      H(i,k-2) - 8.*H(i,k-1) + 8.*H(i,k+1) - H(i,k+2) } +6*g(i+1,k)
!>   -   {      H(i,k-2) - 8.*H(i,k-1) + 8.*H(i,k+1) - H(i,k+2) } -1*g(i+2,k)
!>                       ]
!>
!> case(-1) .and. dum < 0 .or. case(1) .and. dum > 0:
!>
!>   (apply J_1 or J_2 with regular bcs; this routine is not called)
!>
!> case(1) .and. dum < 0:
!>
!>   elems = (1/72)*dum*[
!>     + {  H(i-2,k) - 6.*H(i-1,k) + 3.*H(i,k) + 2.*H(i+1,k) }    g(i,k-2)
!>     - {  H(i-2,k) - 6.*H(i-1,k) + 3.*H(i,k) + 2.*H(i+1,k) }  8*g(i,k-1)
!>     + {  H(i-2,k) - 6.*H(i-1,k) + 3.*H(i,k) + 2.*H(i+1,k) }  8*g(i,k+1)
!>     - {  H(i-2,k) - 6.*H(i-1,k) + 3.*H(i,k) + 2.*H(i+1,k) }    g(i,k+2)
!>   -   {  H(i,k-2) - 8.*H(i,k-1) + 8.*H(i,k+1)  - H(i,k+2) } +1*g(i-2,k)
!>   -   {  H(i,k-2) - 8.*H(i,k-1) + 8.*H(i,k+1)  - H(i,k+2) } -6*g(i-1,k)
!>   -   {  H(i,k-2) - 8.*H(i,k-1) + 8.*H(i,k+1)  - H(i,k+2) } +3*g(i,k)
!>   -   {  H(i,k-2) - 8.*H(i,k-1) + 8.*H(i,k+1)  - H(i,k+2) } +2*g(i+1,k)
!>                      ]
!----------------------------------------------------------------------------

subroutine igh_two(elem,dum,ist,dum2)

  type (matrix_element), intent(inout) :: elem
  integer, intent(in) :: ist
  integer :: i, k
  real, intent(in) :: dum,dum2
  real :: val
  integer :: ierr

  real, parameter :: fac=1./72.

  i = elem%i
  k = elem%k

  if (ist == -1 .and. dum2 > 0) then
    
    val = fac*dum*(-2.*HH(i-1,k)-3.*HH(i,k)+6.*HH(i+1,k)-HH(i+2,k))
    
    elem%iloc = i
    elem%kloc = k-2
    elem%val  = +1.*val
    call add_element(elem,ierr)
    
    elem%iloc = i
    elem%kloc = k-1
    elem%val  = -8.*val
    call add_element(elem,ierr)

    elem%iloc = i
    elem%kloc = k+1
    elem%val  = +8.*val
    call add_element(elem,ierr)
  
    elem%iloc = i
    elem%kloc = k+2
    elem%val  = -1.*val
    call add_element(elem,ierr)

    val =  -1.*fac*dum*(1.*HH(i,k-2)-8.*HH(i,k-1)+8.*HH(i,k+1)-1.*HH(i,k+2))
    
    elem%iloc = i-1
    elem%kloc = k
    elem%val  = -2.*val
    call add_element(elem,ierr)

    elem%iloc = i
    elem%kloc = k
    elem%val  = -3.*val
    call add_element(elem,ierr)

    elem%iloc = i+1
    elem%kloc = k
    elem%val  = +6.*val
    call add_element(elem,ierr)

    elem%iloc = i+2
    elem%kloc = k
    elem%val  = -1.*val
    call add_element(elem,ierr)

  else if (ist == 1 .and. dum2 < 0) then

    val = fac*dum*(HH(i-2,k)-6.*HH(i-1,k)+3.*HH(i,k)+ 2.*HH(i+1,k))
    
    elem%iloc = i
    elem%kloc = k-2
    elem%val  = +1.*val
    call add_element(elem,ierr)
    
    elem%iloc = i
    elem%kloc = k-1
    elem%val  = -8.*val
    call add_element(elem,ierr)

    elem%iloc = i
    elem%kloc = k+1
    elem%val  = +8.*val
    call add_element(elem,ierr)
  
    elem%iloc = i
    elem%kloc = k+2
    elem%val  = -1.*val
    call add_element(elem,ierr)

    val =  -1.*fac*dum*(1.*HH(i,k-2)-8.*HH(i,k-1)+8.*HH(i,k+1)-1.*HH(i,k+2))

    elem%iloc = i-2
    elem%kloc = k
    elem%val  = +1.*val
    call add_element(elem,ierr)

    elem%iloc = i-1
    elem%kloc = k
    elem%val  = -6.*val
    call add_element(elem,ierr)

    elem%iloc = i
    elem%kloc = k
    elem%val  = +3.*val
    call add_element(elem,ierr)

    elem%iloc = i+1
    elem%kloc = k
    elem%val  = +2.*val
    call add_element(elem,ierr)
  
  else
    
    call gkw_abort('bad something wrong linear terms')
    
  end if

end subroutine igh_two

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Difference {H,g} in the interior, which includes the boundary in the vpar
!> direction.
!> For 2nd order we difference {H,g_N} = J as J = J_1 *OR* J = J_2, with
!> J_1 and J_2 as defined below.  For a 4th order scheme, we combine J_1 and
!> J_2; J = 2*J_1 - J_2 (Arakawa, JcompPhys,1,1,_119_ 1966).
!>
!> J_1 =
!>   1/3 (1 / (4 \delta s \delta v_{\par}) )
!>   *[
!>        {              H(i-1,k)   - H(i,k-1)                } g(i-1,k-1)
!>      + { H(i-1,k+1) - H(i-1,k-1) - H(i,k-1)   + H(i,k+1)   } g(i-1,k)
!>      + {              H(i,k+1)   - H(i-1,k)                } g(i-1,k+1)
!>      + { H(i-1,k-1) + H(i-1,k)   - H(i+1,k-1) - H(i+1,k)   } g(i,k-1)
!>      + { H(i+1,k)   + H(i+1,k+1) - H(i-1,k)   - H(i-1,k+1) } g(i,k+1)
!>      + {              H(i,k-1)   - H(i+1,k)                } g(i+1,k-1)
!>      + { H(i,k-1)   + H(i+1,k-1) - H(i,k+1)   - H(i+1,k+1) } g(i+1,k)
!>      + {              H(i+1,k)   - H(i,k+1)                } g(i+1,k+1)
!>    ]
!> J_2 =
!>   1/3 (1 / (8 \delta s \delta v_{\par}) )
!>   *[
!>        {            - H(i-1,k-1) + H(i-1,k+1)              } g(i-2,k)
!>      + { H(i-2,k)   + H(i-1,k+1) - H(i,k-2)   - H(i+1,k-1) } g(i-1,k-1)
!>      + { H(i+1,k+1) - H(i-1,k-1) - H(i-2,k)   + H(i,k+2)   } g(i-1,k+1)
!>      + {              H(i-1,k-1) - H(i+1,k-1)              } g(i,k-2)
!>      + {              H(i+1,k+1) - H(i-1,k+1)              } g(i,k+2)
!>      + { H(i-1,k-1) - H(i+1,k+1) - H(i+2,k)   + H(i,k-2)   } g(i+1,k-1)
!>      + { H(i+1,k-1) - H(i-1,k+1) + H(i+2,k)   - H(i,k+2)   } g(i+1,k+1)
!>      + {              H(i+1,k-1) - H(i+1,k+1)              } g(i+2,k)
!>    ]
!>
!> At the boundaries in the s-direction, we apply something similar to the
!> wills=1 scheme to do upwinding.
!>
!> (Above we drop the second index with respect to `j' (for \mu) in H; the
!>  second index is always j. Below we use `HH' for `H')
!----------------------------------------------------------------------------

subroutine jhg_interior(elem,dum,scheme)

  use control, only: order_of_the_scheme

  type (matrix_element), intent(inout) :: elem
  real, intent(in) :: dum

  real :: d1, d2
  integer :: i,k,ierr

  character (len=1), optional, intent(in) :: scheme
  character (len=1) :: second_order_scheme 

  if (present(scheme)) then
    second_order_scheme = scheme
  else
    second_order_scheme = "+" ! or "x"
  endif

  i=elem%i
  k=elem%k

  select case(order_of_the_scheme)

    case('second_order') 
      select case(second_order_scheme)
        case("+") ; d1 =  1. ; d2 =  0.
        case("x") ; d1 =  0. ; d2 =  1.
        case default ; call gkw_abort('unknown case of second_order_scheme')
      endselect

    case('fourth_order') 
      if (present(scheme)) then
        select case(second_order_scheme)
          case("+") ; d1 =  1. ; d2 =  0.
          case("x") ; d1 =  0. ; d2 =  1.
          case default ; call gkw_abort('unknown case of second_order_scheme')
        endselect
      else
        d1 =  2. ; d2 = -1.
      endif
      
    case default ; call gkw_abort('linear_terms: unknown case for J')

  end select

  d1 = d1*dum/12.
  d2 = d2*dum/24.

  ! g(i-2,k); J_2
  elem%iloc = i-2
  elem%kloc = k
  elem%val  =   d2*(               HH(i-1,k+1) - HH(i-1,k-1)               )
  call add_element(elem,ierr)

  ! g(i-1,k-1); J_1, J_2
  elem%iloc = i-1
  elem%kloc = k-1
  elem%val  =   d1*(               HH(i-1,k)   - HH(i,k-1)                 )   &
      &       + d2*( HH(i-2,k)   + HH(i-1,k+1) - HH(i,k-2)   - HH(i+1,k-1) )
  call add_element(elem,ierr)

  ! g(i-1,k); J_1
  elem%iloc = i-1
  elem%kloc = k
  elem%val  =   d1*( HH(i-1,k+1) - HH(i-1,k-1) - HH(i,k-1)   + HH(i,k+1)   )
  call add_element(elem,ierr)

  ! g(i-1,k+1); J_1, J_2
  elem%iloc = i-1
  elem%kloc = k+1
  elem%val  =   d1*(               HH(i,k+1)   - HH(i-1,k)                 )   &
      &       + d2*( HH(i+1,k+1) - HH(i-1,k-1) - HH(i-2,k)   + HH(i,k+2)   )
  call add_element(elem,ierr)

  ! g(i,k-2); J_2
  elem%iloc = i
  elem%kloc = k-2
  elem%val  =   d2*(               HH(i-1,k-1) - HH(i+1,k-1)               )
  call add_element(elem,ierr)

  ! g(i,k-1); J_1
  elem%iloc = i
  elem%kloc = k-1
  elem%val  =   d1*( HH(i-1,k-1) + HH(i-1,k)   - HH(i+1,k-1) - HH(i+1,k)   )  
  call add_element(elem,ierr)

  ! g(i,k+1); J_1
  elem%iloc = i
  elem%kloc = k+1
  elem%val  =   d1*( HH(i+1,k)   + HH(i+1,k+1) - HH(i-1,k)   - HH(i-1,k+1) ) 
  call add_element(elem,ierr)

  ! g(i,k+2); J_2
  elem%iloc = i
  elem%kloc = k+2
  elem%val  =   d2*(               HH(i+1,k+1) - HH(i-1,k+1)               )
  call add_element(elem,ierr)

  ! g(i+1,k-1); J_1, J_2
  elem%iloc = i+1
  elem%kloc = k-1
  elem%val  =   d1*(               HH(i,k-1)   - HH(i+1,k)                 )   &
      &       + d2*( HH(i-1,k-1) - HH(i+1,k+1) - HH(i+2,k)   + HH(i,k-2)   )
  call add_element(elem,ierr)

  ! g(i+1,k); J_1
  elem%iloc = i+1
  elem%kloc = k
  elem%val  =   d1*( HH(i,k-1)   + HH(i+1,k-1) - HH(i,k+1)   - HH(i+1,k+1) )
  call add_element(elem,ierr)

  ! g(i+1,k+1); J_1, J_2
  elem%iloc = i+1
  elem%kloc = k+1
  elem%val  =   d1*(               HH(i+1,k)   - HH(i,k+1)                 )   &
      &       + d2*( HH(i+1,k-1) - HH(i-1,k+1) + HH(i+2,k)   - HH(i,k+2)   )
  call add_element(elem,ierr)

  ! g(i+2,k); J_2
  elem%iloc = i+2
  elem%kloc = k
  elem%val  =   d2*(               HH(i+1,k-1) - HH(i+1,k+1)               )
  call add_element(elem,ierr)

end subroutine jhg_interior
  
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> add 4th or 2nd order diffusion in s or vpar (dx = dvp or sgr_dist)
!----------------------------------------------------------------------------

subroutine diffus(E_in,kdiff,dum,direction)

  use geom,         only : sgr_dist
  use velocitygrid, only : dvp
  use control,      only : order_of_the_scheme

  type(matrix_element), intent(in) :: E_in
  real, intent(in) :: kdiff,dum
  character (len=*), intent(in) :: direction
  
  type(matrix_element) :: E
  real :: d,dx
  real, dimension(5) :: st
  integer :: ierr,jj

  E=E_in
 
  select case(direction)
    case('vpar')
      dx = dvp
    case('s')
      dx =sgr_dist
    case default
      call gkw_abort('diffus: bad case of direction')
  end select
  
  select case (order_of_the_scheme)
    case ('fourth_order')
      st = (/ 1., -4., 6., -4., 1./)
      d=-kdiff*abs(dum)/(12.*dx)
    case ('second_order')
      st = (/ 0., 1., -2., 1., 0./)
      d=-kdiff*abs(dum)/(dx)
    case default
      call gkw_abort('diffus: bad case of scheme order')
  end select
       
  do jj=1, 5
    select case(direction)
      case('vpar')
        E%iloc = E%i
        E%kloc = E%k + (jj - 3)
      case('s')
        E%iloc = E%i + (jj - 3)
        E%kloc = E%k
    end select
 
    E%val = d*st(jj)
    call add_element(E,ierr)
  end do

end subroutine diffus

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> 1/2 v_parallel^2 + \mu b_n 
!----------------------------------------------------------------------------
function HH(i,k)

  use grid, only : ns
  use geom, only : bn
  use velocitygrid, only : vpgr, mugr

  real :: HH
  integer, intent(in) :: i,k
  integer :: iref
  
  iref = i
  ! this only will work if there is no dependence of vpar on s?
  if (i < 1 )  iref = 1
  if (i > ns)  iref = ns

  HH = 0.5*vpgr(iref,jref_mu,k)**2 + mugr(jref_mu)*bn(i)

end function HH
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine vpgrphi_3_newbc(landau,disp)
!
!------------------------------------------------------------------------------
! This routine puts the landau damping term (Term VII in the manual)
!
! - (Z/T_R) v_R v_{parallel N} ffun (d <\phi> / d s) F_MN
!
! in the matrix. The parallel derivative is based on a fourth order scheme.
! We operate on \phi, so in order to difference the gyro-averaged \phi
! ( = J_0 \phi), wherever \phi_{i+n} is referenced the corresponding value of
! the Bessel function is multiplied by the matrix element.
! The paralllel boundary conditions are implemented through calls to
! connect_parallel (which are made from add_element), and the time step is
! estimated through a call to time_est (via add_element and put_element).
!------------------------------------------------------------------------------

  use dist,         only : fmaxwl,indx, falpha, iphi, iapar
  use components,   only : tmp, signz, vthrat, types, pbg
  use geom,         only : sgr, ffun, bn
  use matdat,       only : put_element
  use control,      only : nlapar,order_of_the_scheme
  use grid,         only : nx,ns,nmu,nvpar,nsp,nmod
  use velocitygrid, only : vpgr, mugr

! landau damping multiplicator and dissipation coefficient
  real, intent(in) :: landau, disp

! integers for the loop over all grid points
  integer :: imod, ix, i, j, k, is

! variables for the parallel boundary conditions
  integer :: ist

! element to test/add to the matrix
  type (matrix_element) :: elem

 logical :: ingrid
 integer :: ixref,kref,iref
 integer :: ierr, iatempt
 ! real :: em_fac

! Dummy matrix element and reference value of the solution
  complex :: mat_elem

! Dummy variables
  real :: b, dum, sgr_dist, gm, vn

  if (ns .lt. 2) return

! set the upwinding at the end points of the field line
  gm = disp
  if (landau .eq. 0) return

! iphi is the default here
  elem%itype = iphi

  do is = 1,nsp

    call time_est(mat_elem,0)

    do imod = 1,nmod ; do ix = 1,nx

      ! delta s
      sgr_dist = sgr(ix,2) - sgr(ix,1)

      do j = 1,nmu ; do k = 1,nvpar ; do i = 1,ns


        call set_indx(elem,imod,ix,i,j,k,is)
        elem%kloc=k

        dum = -landau*signz(is)*ffun(i)*vthrat(is)*vpgr(i,j,k)*fmaxwl(i,j,k)   &
            & /tmp(is)

        ! for an alpha particle distribution replace
        if (types(is) .eq. 'alpha') then
          vn = sqrt(vpgr(i,j,k)**2 + 2.E0*mugr(j)*bn(i))
          dum = -landau*signz(is)*ffun(i)*vthrat(is)*vpgr(i,j,k)*falpha(i,j,k) &
              & *3.E0*vn/(2.E0*tmp(is)*(pbg(is)**1.5+vn**3))
        endif

        ! check if the point is a begining or end
        iatempt = i
        call connect_parallel(imod,ix,i,k,iatempt,ingrid,ixref,iref,kref,ist)

        scheme_order : select case(order_of_the_scheme)
          case('fourth_order')

            ! The boundary conditions have four different cases because of the
            ! upwinding scheme used on the opposite of the flappy boundary ->
            ! this also depends greatly on the local advective velocity.

            grid_location_4th : select case(ist)

              case(0) ! Within the grid

                elem%iloc = i - 2
                b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                elem%val  = (1.E0*dum*b - 1.E0*disp*abs(dum*b))/(12.E0*sgr_dist)
                call add_element(elem,ierr)
                ! Electromagnetic correction
                ! call em_correct(elem,ierr)

                elem%iloc = i - 1
                b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                elem%val  = (-8.E0*dum*b +4.E0*disp*abs(dum*b))/(12.E0*sgr_dist)
                call add_element(elem,ierr)
                ! call em_correct(elem,ierr)

                elem%iloc = i
                b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                elem%val  = -6.E0*disp*abs(dum*b)/(12.E0*sgr_dist)
                call add_element(elem,ierr)
                ! call em_correct(elem,ierr)

                elem%iloc = i + 1
                b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                elem%val  = (8.E0*dum*b + 4.E0*disp*abs(dum*b))/(12.E0*sgr_dist)
                call add_element(elem,ierr)
                ! call em_correct(elem,ierr)

                elem%iloc = i + 2
                b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                elem%val  = (-1.E0*dum*b -1.E0*disp*abs(dum*b))/(12.E0*sgr_dist)
                call add_element(elem,ierr)
                ! call em_correct(elem,ierr)

              case(-2)  ! Second order backwinded difference scheme

                if (dum .lt. 0E0) then

                  elem%iloc = i
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (-3.E0*dum*b)/(2.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i + 1
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (2.E0*dum*b)/(1.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i + 2
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (-1.E0*dum*b)/(2.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                else if (dum .gt. 0.) then

                  elem%iloc = i
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = -2.E0*disp*abs(dum*b)/(1.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i + 1
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (1.E0*dum*b + 2.*disp*abs(dum*b))/(2.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)
                !else
                endif

              case(-1)  ! Third order backwinded differece scheme

                if (dum .lt. 0.) then

                  elem%iloc = i
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (-1.E0*dum*b)/(2.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i + 1
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (1.E0*dum*b)/(1.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i + 2
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (-1.E0*dum*b)/(6.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i - 1
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = -(1.E0*dum*b)/(3.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                else if (dum .gt. 0.) then

                  elem%iloc = i - 1
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (-8.E0*dum*b + 4.*disp*abs(dum*b))/(12.*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = -6.E0*disp*abs(dum*b)/(12.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i + 1
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (8.E0*dum*b + 4.*disp*abs(dum*b))/(12.*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i + 2
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (-1.E0*dum*b - 1.E0*gm*abs(dum*b))/(12.*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)
                !else
                endif

              case(2) ! Second order cental difference with a zero ghost cell

                if (dum .lt. 0.) then

                  elem%iloc = i - 1
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (-1.E0*dum*b + disp*abs(dum*b))/(2.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = -2.E0*disp*abs(dum*b)/(2.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                else if (dum .gt. 0.) then

                  elem%iloc = i
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (3.E0*dum*b)/(2.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i - 1
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (-2.E0*dum*b)/(1.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i - 2
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (1.E0*dum*b)/(2.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)
                !else
                endif

              case(1) ! Fourth order central differenced with a zero ghost cell

                if (dum .lt. 0.) then

                  elem%iloc = i - 2
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (1.*dum*b+1.*disp*abs(dum*b))/(12.*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i - 1
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (-8.E0*dum*b + 4.*disp*abs(dum*b))/(12.*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = -6.E0*disp*abs(dum*b)/(12.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i + 1
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (8.E0*dum*b + 4.*disp*abs(dum*b))/(12.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                else if (dum .gt. 0.) then

                  elem%iloc = i
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (1.E0*dum*b)/(2.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i + 1
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (1.E0*dum*b)/(3.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i - 1
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (-1.E0*dum*b)/(1.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i - 2
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (1.E0*dum*b)/(6.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)
                !else
                endif

              case default

                call gkw_abort('vpgrphi_3_newbc: error in potential function')

            end select grid_location_4th

          case ('second_order')

            grid_location_2nd : select case(ist)

              case(-1,0,1) ! In the middle of the grid

                elem%iloc = i - 1
                b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                elem%val  = (-1.*dum*b + 2.*disp*abs(dum*b)) /(2.*sgr_dist)
                call add_element(elem,ierr)
                ! call em_correct(elem,ierr)

                elem%iloc = i
                b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                elem%val  = -2.*disp*abs(dum*b)/(1.*sgr_dist)
                call add_element(elem,ierr)
                ! call em_correct(elem,ierr)

                elem%iloc = i + 1
                b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                elem%val  = (1.*dum*b + 2.*disp*abs(dum*b)) /(2.*sgr_dist)
                call add_element(elem,ierr)
                ! call em_correct(elem,ierr)

              case(-2)

                if (dum .lt. 0E0) then

                  elem%iloc = i
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = 1.E0*dum*b/(1.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i + 1
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = -(1.E0*dum*b)/(1.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                else if (dum .gt. 0E0) then

                  elem%iloc = i
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = -2.E0*disp*abs(dum*b)/(1.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i + 1
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (1.E0*dum*b + 2.E0*disp*abs(dum*b))/(2.*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)
                ! else ?
                end if

              case(2)

                if (dum .lt. 0E0) then

                  elem%iloc = i - 1
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (-1.E0*dum*b + 2.*disp*abs(dum*b))/(2.*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = -2.E0*disp*abs(dum*b)/(1.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                else if (dum .gt. 0E0) then

                  elem%iloc = i
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = -1.E0*dum*b/(1.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)

                  elem%iloc = i - 1
                  b         = bessel_j0(imod,ix,i,j,k,is,elem%iloc)
                  elem%val  = (1.E0*dum*b)/(1.E0*sgr_dist)
                  call add_element(elem,ierr)
                  ! call em_correct(elem,ierr)
                ! else
                end if

              case default

                call gkw_abort('Error in second order Boundary conditions')

            end select grid_location_2nd

            case default
              call gkw_abort('Error in linear_terms.F90 -> vpgrphi_3_newbc -'//&
                  &          'Choices are second or fourth order')

          end select scheme_order

        enddo ; enddo ; enddo

      enddo ; enddo

    ! retrieve the maximum time step estimate
    call time_est(mat_elem,2)
    ! write(*,*)'species ',is,' max time ',1/real(mat_elem)

  enddo

end subroutine vpgrphi_3_newbc

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine hyper_disp_perp(disp_x,disp_y)
!-----------------------------------------------------------------------------
!This routine adds perpendicular hyperdissipation, coeffcients in control
!-----------------------------------------------------------------------------
  use grid,         only : nx, ns, nmu, nvpar, nsp, nmod
  use dist,         only : indx
  use mode,         only : krho, kxrh
  use matdat,       only : put_element

  real, intent(in) :: disp_x, disp_y

  ! integers for the loop over all grid points
  integer :: imod, ix, i, j, k, is

  ! reference integers and matrix element
  complex :: mat_elem
  integer :: iih, jjh

  ! Dummy variables
  complex :: dumc

  if (root_processor) then
     write(*,*) 'Hyperdisp: check linear timestep courant stability'
  end if

    ! clear the timestep estimate
    call time_est(mat_elem,0)

    do is = 1, nsp; do imod = 1, nmod ; do ix = 1, nx ; do i = 1, ns

      do j = 1, nmu ; do k = 1, nvpar

        iih = indx(imod,ix,i,j,k,is)
        jjh = iih
        mat_elem = -(disp_y*krho(imod)**4 + disp_x*kxrh(ix)**4)
        !The line below is for normal (not hyper) dissipiation 
        !mat_elem = -(disp_y*krho(imod)**2 + disp_x*kxrh(ix)**2)
        call put_element(iih,jjh,mat_elem,1)

      end do ; end do ; end do ; end do ; end do ; end do

    ! retrieve the time step estimate
    call time_est(mat_elem,2)

end subroutine hyper_disp_perp

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


subroutine connect_parallel(imod,ix,i,k,iatempt,ingrid,ixref,iref,ikref,ist)
!--------------------------------------------------------------------
! This soubroutine is used in connection with the parallel boundary
! conditions. It determines the point in the grid to connect with
!--------------------------------------------------------------------
  use control,      only : parallel_boundary_conditions, vp_trap
  use grid,         only : ns, nvpar, lproc_s_lowerb, lproc_s_upperb,        &
                         & parallel_s
  use mode,         only : mode_box, ixplus, ixminus, krho
  use velocitygrid, only : iblow, ibhig

  integer :: imod    ! <=  Toroidal mode number of the iih point
  integer :: ix      ! <=  kxrh mode number of the iih point
  integer :: i       ! <=  parallel grid location of the iih point
  integer :: k       ! <=  The value of the parallel velocity grid
  integer :: iatempt ! <=  The element that is referenced as jjh (for
                     !     instance i-2
  logical :: ingrid  ! =>  True if a point in the grid can be references
  integer :: ixref   ! =>  The proper reference of the jjh point in kxrh
  integer :: iref    ! =>  The proper reference of the parallel grid point
                     !     of jjh
  integer :: ikref   ! =>  The proper reference of the parallel velocity
                     !     grid point. Note this is aways k, unless
                     !     vp_trap = 1
  integer :: ist     ! =>  output that determines if the grid point is
                     !     the begining or end of a field line (this is
                     !     used in some of the schemes  -2 = begining
                     !     0 = somewhere in the middle 2 = end


! Does the grid follow the trapping condition ?
  if (vp_trap .eq. 1) then
    if (parallel_s) call gkw_abort('connect_parallel: I can not deal with '//  &
        &                          'parallel_s and vp_trap = 1 yet!')
    ! is the particle trapped ?
    if (iblow(k) .ne. 0) then

      ! never an end point
      ist = 0

      ! aways in the grid
      ingrid = .true.

      ! aways refer to the same field line
      ixref = ix

      if (iatempt .lt. iblow(k)) then
        iref = 2*iblow(k)-iatempt-1
        ikref    = nvpar - k + 1
        return
      endif
      if (iatempt .gt. ibhig(k)) then
        iref = 2*ibhig(k) - iatempt + 1
        ikref    = nvpar - k + 1
        return
      endif

      ! not bouncing
      iref = iatempt
      ikref = k
      return

    endif
  endif

! for all passing particles ikref is always k
  ikref = k

! Determine ist
  ist = 0
  if (i .eq. 1 .and. ixminus(imod,ix) .eq. 0 .and. lproc_s_lowerb) ist = -2
  if (i .eq. ns .and. ixplus(imod,ix) .eq. 0 .and. lproc_s_upperb) ist = 2
  ! the following are for 4th order bcs
  if (i .eq. 2 .and. ixminus(imod,ix) .eq. 0 .and. lproc_s_lowerb) ist = -1
  if (i .eq. (ns-1) .and. ixplus(imod,ix) .eq. 0 .and. lproc_s_upperb) ist = 1
! check if the point lies on the grid
  if ( (iatempt .ge. 1 .and. iatempt .le. ns) .or.                             &
      &(parallel_s .and. ( (iatempt .lt. 1 .and. (.not. lproc_s_lowerb)) .or.  &
      &                     (iatempt .gt. ns .and. (.not.lproc_s_upperb))  )  )&
      &  )  then

    ingrid = .true.
    ixref = ix
    iref = iatempt
    return
  endif

  if (mode_box) then

    ! A 2Dimensional array of modes is used (i.e. different ix must
    ! be connected
    if (abs(krho(imod)) .lt. 1e-10) then

      ! the ky = 0 mode is always periodic
      if (iatempt .le. 0) then
        ixref  = ix
        if (parallel_s) then
          iref = iatempt
        else
          iref = ns + iatempt
        endif
        ingrid = .true.
        return
      endif
      if (iatempt .gt. ns) then
        ixref = ix
        if (parallel_s) then
          iref = iatempt
        else
          iref = iatempt - ns
        endif
        ingrid = .true.
        return
      endif

    else

      if (iatempt .le. 0) then
        if (ixminus(imod,ix).ne.0) then
          ixref = ixminus(imod,ix)
          if (parallel_s) then
            iref = iatempt
          else
            iref = ns + iatempt
          endif
          ingrid = .true.
          return
        else
          select case(parallel_boundary_conditions)
            case('zero')
              iref   = 0
              ixref  = 0
              ingrid = .false.
              return
            case('zero_derivative')
              if (parallel_s) call gkw_abort('connect_parallel: please check'//&
                  &                          'here 1')
              ingrid = .true.
              iref   = 1 - iatempt
              ixref  = ix
              return
            case('periodic_noshift')
              call gkw_abort('No option periodic_noshift with mode_box true &
                              & but you can run with shat=0')
            case default
              call gkw_abort('Unkown switch for the parallel boundary cond.')
          end select
        endif
      endif

      if (iatempt .gt. ns) then
        if (ixplus(imod,ix) .ne. 0) then
          ixref = ixplus(imod,ix)
          if (parallel_s) then
            iref = iatempt
          else
            iref = iatempt - ns
          endif
          ingrid = .true.
          return
        else
          select case(parallel_boundary_conditions)
            case('zero')
              iref   = 0
              ixref  = 0
              ingrid = .false.
              return
            case('zero_derivative')
              if (parallel_s) call gkw_abort('connect_parallel: check here 2')
              ingrid = .true.
              iref   = 2*ns - iatempt + 1
              ixref  = ix
              return
            case('periodic_noshift')
              call gkw_abort('No option periodic_noshift with mode_box=T')
            case default
              call gkw_abort('Unknown switch for the parallel boundary conditions')
          end select
        endif
      endif
    endif

  else !not mode_box

    ! Single modes are used. No coupling with other ix through the
    ! parallel boundary conditions
    ixref = ix

    select case(parallel_boundary_conditions)

      case('zero')

        ! do not reference any element
        iref    = 0
        ingrid  = .false.
        return

      case('periodic_noshift')

        if (iatempt .le. 0) then
          ingrid = .true.
          iref   = ns + iatempt
          if (parallel_s) iref = iatempt
          ixref  = ix
          return
        endif
        if (iatempt .gt. ns) then
          ingrid = .true.
          iref   = iatempt - ns
          if (parallel_s) iref = iatempt
          ixref  = ix
          return
        endif

      case('zero_derivative')

        if (parallel_s) call gkw_abort('connect_parallel: please check here 4')

        if (iatempt .le. 0) then
          ingrid = .true.
          iref   = 1 - iatempt
          ixref  = ix
          return
        endif

        if (iatempt .gt. ns) then
          ingrid = .true.
          iref   = 2*ns - iatempt + 1
          ixref  = ix
          return
        endif

      case default

        call gkw_abort('No known option for the parallel boundary condtions')

    end select

  endif

! Something went wrong if you reach this point
  call gkw_abort('Internal error in connect_parallel')

end subroutine connect_parallel

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine add_element(E,ierr,i_type)
!
! Check if the point is in the grid, then put it into the right place in the
! matrix. Return an appropriate error code.
!
  use dist, only : indx
  use matdat, only : put_element
  use velocitygrid, only : connect_vpar

  type (matrix_element), intent(in) :: E
  integer, intent(out) :: ierr
  integer, optional, intent(in) :: i_type

  logical :: ingrid_s, ingrid_vpar
  integer :: ist, ixref, kref, iref, jjh, iih
  integer :: itype

  logical, parameter :: ltime_est = .true.

  ingrid_vpar = .false.
  ingrid_s    = .false.
  ist         = -999
  ierr        = ierr_UNDEFINED
!
! iih is the same in all cases?
!
  iih = indx(E%imod,E%ix,E%i,E%j,E%k,E%is)
!
  if (present(i_type)) then
    itype = i_type
  else
    itype = E%itype
  endif
!
! diagonal element
!
  if ((E%iloc .eq. E%i) .and. (E%kloc .eq. E%k)) then

    if (itype .eq. iapar .or. itype .eq. iphi) then
      jjh = indx(E%imod,E%ix,E%i,itype)
    else
      jjh = iih
    endif

    if (ltime_est) then
      call put_element(iih,jjh,E%val,1)
    else
      call put_element(iih,jjh,E%val)
    endif

    ierr = ierr_OK
    return ! jump out here

  endif
!
! otherwise, check if the element is in the grid
!
  if (E%kloc .ne. E%k) then
    call connect_vpar(E%kloc,ingrid_vpar)
  else
    ingrid_vpar = .true.
  endif

  if (E%iloc .ne. E%i) then
    call connect_parallel(E%imod,E%ix,E%i,E%kloc,E%iloc,ingrid_s,ixref,iref,  &
        & kref,ist)
  else
     ingrid_s = .true.
     iref  = E%iloc
     kref  = E%kloc
     ixref = E%ix
  endif
!
! add the element
!
  if (ingrid_vpar .and. ingrid_s) then

    if (itype .eq. iapar .or. itype .eq. iphi) then
      jjh = indx(E%imod,ixref,iref,itype)
    else
      jjh = indx(E%imod,ixref,iref,E%j,kref,E%is)
    endif

    call put_element(iih,jjh,E%val,1)
    ierr = ierr_OK

  else if(ingrid_vpar) then
    ierr = ierr_BAD_S
  else if(ingrid_s) then
    ierr = ierr_BAD_VPAR
  else
    ierr = ierr_BAD_ALL
  endif

  if (ierr .eq. ierr_UNDEFINED) call gkw_abort('add_element: undefined ierr')

end subroutine add_element

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine set_indx(E,imod,ix,i,j,k,is)

  type (matrix_element), intent(inout) :: E
  integer, intent(in) :: imod,ix,i,j,k,is

  E%imod=imod
  E%ix  =ix
  E%i   =i
  E%j   =j
  E%k   =k
  E%is  =is

end subroutine set_indx

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

function bessel_j0(imod,ix,i,j,k,is,iatempt)
!
! a wrapper for besselj0_gkw using connect parallel
!
  integer, intent(in) :: imod,ix,i,j,k,is,iatempt
  real :: bessel_j0

  integer :: ist,ixref,iref,kref
  logical :: ingrid

  call connect_parallel(imod,ix,i,k,iatempt,ingrid,ixref,iref,kref,ist)

  if (ingrid) then
    bessel_j0 = besselj0_gkw(imod,ixref,iref,j,is)
  else
    bessel_j0 = 0.
  endif

end function bessel_j0

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine em_correct(elem,ierr)
!
! Electromagnetic correction factor
!   mat_elem = -2.*mat_elem*vthrat(is)*vpgr(i,j,k)
!
  use components, only : vthrat
  use velocitygrid, only : vpgr

  type (matrix_element), intent(in) :: elem
  type (matrix_element) :: E
  integer, intent(out) :: ierr
  integer :: i_err

  E = elem
  E%val = -2.*elem%val*vthrat(elem%is)*vpgr(elem%i,elem%j,elem%k)
  call add_element(E,i_err,iapar)
  ierr = i_err

end subroutine em_correct

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

end module linear_terms
