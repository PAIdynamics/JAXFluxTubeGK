module non_linear_terms
! SVN:$Id: non_linear_terms.F90 1008 2009-07-02 16:44:23Z  $
use mpiinterface

implicit none

private

public :: add_non_linear_terms, jind, jinv, mphi, mphiw3, mrad
public :: a, ar, nonlinear_allocate, nonlinear_init, nl_initialised

!> Index array for the storage of the kx modes in the arrays for the FFT 
!> jind(nx)
integer,    allocatable :: jind(:)

!> Inverse index array for the storage of the kx modes in the arrays for the FFT.
!> jinv(mrad)
integer,    allocatable :: jinv(:)

! arrays for the FFT 

! Throughout the code:
! x is used to refer to the radial (psi) direction
! y is used to refer to the perpendicular (zeta) direction within the flux surfuce
! This direction can be called 'polodial' but is also referred to as 'toroidal'
! Gradient d/dx in k space is an ik_x multiplier

!> a(mphiw3,mrad)
!> 1st usage: a = grad_y_k <phi_k> = zeta gradient of the gyroaverage of potential in k space.
!> 2nd usage: a = grad_y_k <A||_k>, electromagnetic terms
!> 3rd usage: a = grad_x f_k = radial gradient of the distribution function in k space.
complex, allocatable :: a(:,:)

!> b(mphiw3,mrad)
!> 1st usage: b = grad_x_k <phi_k> = radial gradient of the gyroaverage of potential in k space.
!> 2nd usage: b = grad_x_k <A||_k>, electromagnetic terms
!> 3rd usage: b = grad_y f_k = zeta gradient of the distribution function in k space.
complex, allocatable :: b(:,:)

complex, allocatable :: c(:,:)  !< Another fft dummy array

!> ar = grad_y <phi> = zeta gradient of the gyroaverage of potential in real space.
!> ar(mphi,mrad)
real, allocatable :: ar(:,:)

!> br = grad_x <phi> = radial gradient of the gyroaverage of potential in real space.
!> br(mphi,mrad)
real, allocatable :: br(:,:)

!> cr = grad_x f = radial gradient of the gyroaverage of potential in real space.
!> later reused for the rhs
!> cr(mphi,mrad)
real, allocatable :: cr(:,:)

!> dr = grad_y f = zeta gradient of the gyroaverage of potential in real space.
!> dr(mphi,mrad)
real, allocatable :: dr(:,:)

!> er = grad_zeta ( 2. vthref  <A||> ) = poloidal gradient of the the parallel vector 
!> potential. er (mphi,mrad) 
real, allocatable :: er(:,:) 

!> fr = grad_psi ( 2. vthref  <A||> ) = radial gradient of the the parallel vector 
!> potential.  fr (mphi,mrad) 
real, allocatable :: fr(:,:) 

!> The number of 'poloidal' (elsewhere called toroidal) points in the FFT 
!> Notationally this should probably be called mpsi!
!> this is bigger than nmod because of the dealiasing
integer :: mphi, mphiw3

!> The number of radial points in the FFT 
!> This is bigger than nmod because of the dealiasing
integer :: mrad

!>Distribution function index translation array
!>To facilitate copying of the distribution function
integer, allocatable, save :: lindx(:)

! integer array to copy back the values of the arrays 
integer, allocatable, save :: lincopy(:)

!> Flag to set once nonlinear_init is called
logical :: nl_initialised = .false. 

contains 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


subroutine nonlinear_allocate 
!--------------------------------------------------------------------
!> This routine allocates the help arrays for the FFTs
!--------------------------------------------------------------------

use grid,    only : nmod, nx, ns, nvpar, nmu, nsp
use general, only : gkw_abort

integer ierr 
real dum 

! Caculate the size of the grid for the FFT 
if (nmod.eq.1) then 
  dum = 1.E0
else 
  dum  = 1.5*real(2*nmod - 2) 
endif 
mphi = log(dum)/log(2.E0) + 1.E0  
mphi = 2**mphi 

! Test if a factor 3 would be more useful 
! no great performance enhancement
!if (3.0*real(mphi)/4.0 .ge. dum ) then 
!  write(*,*)'using a factor 3 in the fft' 
!  mphi = 3*mphi / 4
!endif 

! Caculate the size of the grid from krho and kxrh 
dum  = 1.5*real(nx+1) 
mrad = log(dum)/log(2.E0) + 1.E0  
mrad = 2**mrad 

mphiw3=(mphi/2+1)

! set the error parameter 
ierr = 0 

! allocate the arrays 
allocate(a(mphiw3,mrad), stat = ierr) 
if (ierr.ne.0) then 
  call gkw_abort('Could not allocate a in non_linear_terms')
endif 

allocate(b(mphiw3,mrad), stat = ierr) 
if (ierr.ne.0) then 
  call gkw_abort('Could not allocate b in non_linear_terms')
endif 

allocate(ar(mphi,mrad), stat = ierr) 
if (ierr.ne.0) then 
  call gkw_abort('Could not allocate ar in non_linear_terms')
endif 

allocate(br(mphi,mrad), stat = ierr) 
if (ierr.ne.0) then 
  call gkw_abort('Could not allocate br in non_linear_terms')
endif 

allocate(cr(mphi,mrad), stat = ierr) 
if (ierr.ne.0) then 
  call gkw_abort('Could not allocate cr in non_linear_terms')
endif 

allocate(dr(mphi,mrad), stat = ierr) 
if (ierr.ne.0) then 
  call gkw_abort('Could not allocate dr in non_linear_terms')
endif 

allocate(er(mphi,mrad), stat = ierr)
if (ierr.ne.0) then 
  call gkw_abort('Could not allocate er in non_linear_terms')
endif 

allocate(fr(mphi,mrad), stat = ierr) 
if (ierr.ne.0) then 
  call gkw_abort('Could not allocate fr in non_linear_terms')
endif 

allocate(c(mphi,mrad), stat = ierr) 
if (ierr.ne.0) then 
  call gkw_abort('Could not allocate c in non_linear_terms')
endif 

allocate(jinv(mrad), stat = ierr) 
if (ierr.ne.0) then 
  call gkw_abort('Could not allocate jinv in non_linear_terms')
endif 

allocate(jind(nx), stat = ierr) 
if (ierr.ne.0) then 
  call gkw_abort('Could not allocate jind in non_linear_terms')
endif 

allocate(lindx(1:ns*nsp*nmu*nvpar*nmod*nx)) 
if (ierr.ne.0) then 
  call gkw_abort('Could not allocate lindx in non_linear_terms')
endif 

allocate(lincopy(1:ns*nsp*nmu*nvpar*nmod*nx)) 
if (ierr.ne.0) then 
  call gkw_abort('Could not allocate lincopy in non_linear_terms')
endif 

return 
end subroutine nonlinear_allocate 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


subroutine nonlinear_init
!--------------------------------------------------------------------
!> This routine performs initialisations of quantites
!> Note it may be called even if add_non_linear terms is not,
!> As the parameters it calculates are in used for mode_box, 
!> 2D diagnostics and some of the experiemental shear routines.
!--------------------------------------------------------------------
use mode,         only : krho, kxrh, ixzero
use dist,         only : indx
use grid,         only : nmod, nx, ns, nmu, nvpar, nsp
use general,      only : gkw_abort

!Just loop indices
integer :: imod, ix, i, j, jv, kt, is, ipar, idx

!This initialization routine should only be called once
if (nl_initialised) call gkw_abort('nonlinear_init called twice') 

!The array over which the radial wavevectors are stored differs
!in the code and in the fft.  The jind array transalates
! for fast evaluation the indices are calculated
jind = 0
do ix = ixzero, nx
  jind(ix) = ix - ixzero + 1  
end do  
do ix = ixzero-1, 1, -1 
  jind(ix) = mrad + ix - ixzero + 1
end do  

!The jinv array provides the inverse transalation
do i = 1, mrad
  jinv(i) = 0 
  do j = 1, nx 
    if (jind(j).eq.i) then 
      jinv(i) = j 
    endif 
  end do  
end do 


! construct the array for quick look up of the distribution 
idx=1
lindx=0
do ipar = 1, ns ; do is = 1, nsp ; do jv = 1, nmu ; do kt = 1, nvpar 

  do ix = 1, nx 
    do imod = 1, nmod  
       lindx(idx) = indx(imod,ix,ipar,jv,kt,is)
       idx=idx+1
    end do 
  end do 

end do ; end do ; end do ; end do 

! array for to copy back the nonlinear terms in the right hand side 
lincopy = 0 
idx = 1 
do ipar = 1, ns ; do is = 1, nsp ; do jv = 1, nmu ; do kt = 1, nvpar 

  do j = 1, nx-ixzero+1 
    do i = 1, nmod 
      lincopy(idx) = indx(i,jinv(j),ipar,jv,kt,is)
      idx = idx + 1
    end do 
  end do 
         
  do j = mrad +2 - ixzero, mrad 
    do i = 1, nmod 
      lincopy(idx) =  indx(i,jinv(j),ipar,jv,kt,is)
      idx = idx + 1 
    end do 
  end do 

end do ; end do ; end do ; end do ; 

! set initialised to true 
nl_initialised = .true. 

return

end subroutine nonlinear_init

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine add_non_linear_terms(fdis,rhs)
!--------------------------------------------------------------------
!> Term III in the manual
!> Note the electromagnetic corrections are now implemented
!--------------------------------------------------------------------
use dist,         only : nsolc, indx, get_phi, phi, get_apar, apar
use mode,         only : krho, kxrh, lxinv, lyinv, ixzero
use geom,         only : efun
use control,      only : dtim, dtim_est, dtim_est_save
use control,      only : non_linear, nlapar, nl_dtim_est
use grid,         only : nmod, nx, ns, nmu, nvpar, nsp, parallel_vpar, vpmax
use functions,    only : besselj0_gkw 
use velocitygrid, only : vpgr
use fft,          only : four2D_real
use general,      only : gkw_abort 
use constants,    only : ci1
use components,   only : vthrat
use rotation,     only : shear_real, grad_pot

complex, intent(in)    :: fdis(nsolc) 
complex, intent(inout) :: rhs(nsolc) 

integer :: ierr, idx, ic, idxcopy
integer :: imod, ix, i, j, jv, kt, is, ipar
real    :: dum, maxvalue, b0, bessj0, dtim_est_dum, maxvalapar, dtim_est_apar 

! Abort if called incorrectly
if (.not.(non_linear.or.shear_real)) then 
    call gkw_abort('Invalid call to add_non_linear_terms')
end if

if (.not.nl_initialised) then
    call gkw_abort('add_non_linear terms: cannot call before nonlinear_init')
end if

! set the value for the estimate of the timestep to zero (The restriction 
! due to the electromagnetic scheme are measured separately with maxvalapar)
maxvalue   = 0.E0
maxvalapar = 0.E0

! obtain phi in a separate array 
if (non_linear) call get_phi(fdis,phi) 

! initialize the indices of the help arrays 
idx=1
idxcopy = 1

loop_ns: do ipar = 1, ns           ! Loop along field line points
  loop_nsp: do is = 1, nsp         ! Loop over species
    loop_nmu: do jv = 1, nmu       ! Loop over magnetic moment 

      ! fill the array for the potential
      get_real_grad_phi: if(non_linear) then

        !Gyroaveraged potential in k space obtained with Bessel function J_0
        !<phi_k> = J_0(k_perp rho) phi_k
        !Does not depend on parallel velocity, calculated outside loop

        !a=grad_y_k <phi_k> = zeta gradient of the gyroaverage of potential
        !                      = i J_0() k_zeta phi_k 
        a = (0.,0.) 

        !b=grad_x_k <phi_k> = radial gradient of the gyroaverage of potential
        !                     = i J_0() k_psi phi_k

        b = (0.,0.) 

        loop_nx1: do ix = 1, nx                !Loop over radial modes
          loop_nmod1: do imod = 1, nmod        !Loop over poloidal modes
             b0 = besselj0_gkw(imod,ix,ipar,jv,is) 
             a(imod,jind(ix)) = ci1*b0*krho(imod)*phi(imod,ix,ipar) 
             b(imod,jind(ix)) = ci1*b0*kxrh(ix) * phi(imod,ix,ipar)
          end do loop_nmod1
        end do  loop_nx1

        !This initialisation could be ommitted for optimisation
        ar=0.
        br=0.

        !Inverse fourier transform of potential k space to real space
        !ar = grad_zeta <phi> = 
        !poloidal gradient of the gyroaverage of potential in real space
        !br = grad_psi <phi> = 
        !radial gradient of the gyroaverage of potential in real space
        call four2D_real(ar,a,1)
        call four2D_real(br,b,1)
 
        ! The electro-magnetic corrections if needed 
        if (nlapar) then 

          call get_apar(fdis,apar) 

          ! initialization
          a = (0.,0.) 
          b = (0.,0.) 

          do imod = 1, nmod        !Loop over poloidal modes
            do ix = 1, nx          !Loop over radial modes
              b0 = besselj0_gkw(imod,ix,ipar,jv,is)
              a(imod,jind(ix)) = &
      &         2.*vthrat(is)*ci1*b0*krho(imod)*apar(imod,ix,ipar)
              b(imod,jind(ix)) = &
      &         2.*vthrat(is)*ci1*b0*kxrh(ix)*apar(imod,ix,ipar)
            end do 
          end do 

          er = 0.
          fr = 0. 

          ! inverse fourier transform (to real space) 
          call four2D_real(er,a,1)
          call four2D_real(fr,b,1)

        endif 

        !Estimate maximum velcocity for timestep estimator
        maxvalue=MAX(MAXVAL(ABS(ar))*mrad * lxinv ,maxvalue)
        maxvalue=MAX(MAXVAL(ABS(br))*mphi * lyinv ,maxvalue)
        if (shear_real) then
          maxvalue=MAX(MAXVAL(ABS(grad_pot))*mphi * lyinv ,maxvalue)
        end if

        if (nlapar) then 
          maxvalapar=MAX(MAXVAL(ABS(er))*mrad*lxinv,maxvalapar)
          maxvalapar=MAX(MAXVAL(ABS(fr))*mphi*lyinv,maxvalapar)
        endif 

      end if get_real_grad_phi

      !Not until here is the parallel velocity looped over
      loop_nvpar: do kt = 1, nvpar

        ! initialize a and b 
        a = (0.,0.)
        b = (0.,0.)
 
        ! a = grad_x f_k
        ! b = grad_y f_k
        loop_nx2: do ix = 1, nx       !Loop over radial modes
          loop_nmod2: do imod = 1, nmod    !Loop over poloidal modes
            a(imod,jind(ix)) =  ci1 * kxrh(ix) * fdis(lindx(idx))
            b(imod,jind(ix)) =  ci1 * krho(imod) * fdis(lindx(idx)) 
            idx=idx+1
          end do loop_nmod2
        end do loop_nx2
        
        !Inverse fourier transform k space to real space 
        !For gradients of distribution function
        if(non_linear) then
          call four2D_real(cr,a,1)
        else !perp shear requires only the zeta component of the distribution
          cr=0
        end if
        call four2D_real(dr,b,1)

        !Now everything is in real space
        !cr = grad_x f = radial gradient of the distribution in real space
        !dr = grad_y f = zeta gradient of the distribution in real space

        ! v_E grad f = (b x grad<phi> . grad f) /B
        ! As written in term III in the manual
        ! The factor of rhorat^2 comes from the normalisation of the FFT
        ! We neglect the d/ds variations
        ! Using also that the diagonal elements of efun tensor are 0:
        ! v_E grad f = (d<phi>/d(zeta))(df/d(psi))*
        !              E^zeta-phi+(d<phi>/d(psi))(df/d(zeta))E^phi-zeta

        ! The zeta psi component = efun(ipar,2,1) 
        ! The psi zeta component = -efun(ipar,2,1) = efun(ipar,1,2)
        dum = - efun(ipar,2,1)
        if (non_linear) then
          if (nlapar) then 
            do j = 1, mrad ; do i = 1, mphi 
              ! cr = V_E . grad f = (b x grad <chi>).grad f
              ! The minus sign is due to the antisymmetry of efun
              ! efun is antisymmetric for all geometries, not just circular
              cr(i,j) = dum*((ar(i,j)-vpgr(ipar,jv,kt)*er(i,j))*cr(i,j) - & 
                      &      (br(i,j)-vpgr(ipar,jv,kt)*fr(i,j))*dr(i,j))
            end do ; end do 
          else 
            do j = 1, mrad ; do i = 1, mphi 
              ! cr = V_E . grad f = (b x grad <phi>).grad f
              ! The minus sign is due to the antisymmetry of efun
              ! efun is antisymmetric for all geometries, not just circular
              cr(i,j) = dum*(ar(i,j)*cr(i,j)-br(i,j)*dr(i,j))
            end do ; end do 
          end if
        end if 

        if (shear_real) then
          do j = 1, mrad ; do i = 1, mphi 
            cr(i,j) = cr(i,j) + dum*grad_pot(j,ipar)*dr(i,j)
          end do ; end do 
        end if

        !a=(0.,0.)
        !Forward Fourier transform real back to k space
        !Note that the positions of input and output arguments are reversed
        call four2D_real(cr,a,-1)
        !Normalise - for more effciency do this operation after the copy
        !so less elements are being normalised.. Also make division into a mult.
        a = a / (mphi*mrad)
       
        do j = 1, nx-ixzero+1 
          do i = 1, nmod 
             rhs(lincopy(idxcopy)) = rhs(lincopy(idxcopy)) + dtim*a(i,j) 
             idxcopy = idxcopy + 1 
          end do
        end do
        do j = mrad +2 - ixzero, mrad 
          do i = 1, nmod 
             rhs(lincopy(idxcopy)) = rhs(lincopy(idxcopy)) + dtim*a(i,j) 
             idxcopy = idxcopy + 1 
          end do
        end do
   

      end do loop_nvpar
    end do loop_nmu
  end do loop_nsp
end do loop_ns

dtim_est = 2./maxvalue 
!Save minimum timestep so far for local processor
dtim_est_save = min(dtim_est,dtim_est_save)

#if defined(mpi)
if (number_of_processors.gt.1 .and. nl_dtim_est) then 
  ierr = 0
  call MPI_ALLREDUCE(dtim_est,dtim_est_dum,1,MPIREAL_X,MPI_MIN, &
     & MPI_COMM_WORLD,ierr)
  dtim_est = dtim_est_dum 
  !?? dtim_est = min(dtim_est,dtim_est_dum)  
endif 
#endif 

if (nlapar) then 
! rather crude way to measure the time step limit for electromagnetic
! cases 
  maxvalapar = maxvalapar*vpmax
  if (maxvalapar .gt. tiny(1.E0)) dtim_est_apar = 2./maxvalapar
  
  !Save minimum timestep so far for local processor
  dtim_est_save = min(dtim_est_apar,dtim_est_save)

#if defined(mpi)
  if (number_of_processors.gt.1 .and. nl_dtim_est) then 
    ierr = 0
    call MPI_ALLREDUCE(dtim_est_apar,dtim_est_dum,1,MPIREAL_X,MPI_MIN, &
       & MPI_COMM_WORLD,ierr)
    dtim_est = min(dtim_est,dtim_est_dum)  
  endif 
#endif 
endif 

end subroutine add_non_linear_terms

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

end module non_linear_terms 
