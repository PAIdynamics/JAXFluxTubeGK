module rotation
!---------------------------------------------------------------------------
! SVN:$Id: rotation.F90 1014 2009-07-02 18:42:39Z  $
! This module contains variables related to toroidal rotation
! And routines and variables associated with perpendicular shearing.
!
! This module reads the parameters in the ROTATION namelist.
!
!---------------------------------------------------------------------------
  use global
  use general
  use mpiinterface
  use constants

  implicit none

  private

  public :: rotation_read_nml, rotation_write_nml, rotation_check_params
  public :: rotation_bcast_nml, shear_rate, vcor, need_fft
  public :: shear_init, rotation_allocate, grad_pot
  public :: perp_shear, toroidal_shear
  public :: shear_real, shear_remap, shear_ky_shift
  public :: shear_shift_ky, wavevector_remap

  interface rotation_write_nml
    module procedure rotation_read_nml
  end interface
  
  !> The rotation of the plasma vcor = Vtor / vthref
  !> Will be parallel to toroidal magnetic field if signB=1 
  real :: vcor 

  !> Normalised shearing rate for the perp shear added in non linear terms
  !> Only used if perp_shear = true
  real :: shear_rate

  !
  !> Selects the options for the perpendicular shear flow
  character (len = lenswitch) :: shear_profile

  logical :: perp_shear           !< true if perpendicular shear included
  logical :: shear_real           !< true if shear flow added in real space
  logical :: shear_remap          !< true for shearing by wavevector remap
  logical :: shear_ky_shift       !< true for shearing by wavevector remap
  logical :: need_fft             !< true if fft is used, make public?
  logical :: toroidal_shear       !< true if uprim inputs should be overidden

  !>Array that tracks number of kx mode shifts: kxshift(nmod)
  integer, allocatable, dimension(:) :: kxshift

  !>Indexing arrays for shearing wavevector remap:
  !> aindx(nmod*nmu*nvpar*ns*nsp*(nx-1))
  integer, allocatable, dimension(:) :: aindx, aindx_shift

  !> integers used in remap loops
  integer :: ixstart, ixend, ixedge, ixdir

  !> Array for shear flow function potential gradient. grad_pot(mrad,ns)
  !> used for shear_real
  real, allocatable :: grad_pot(:,:)

  !Arrays for the ffts, if required.
  complex, ALLOCATABLE :: arr(:,:)
  complex, ALLOCATABLE :: arr_pad(:,:)

  !Index arrays for the ffts - one needed for both forward and back
  integer,    allocatable :: jind_nx(:)

  !>Size of padded fft array
  integer ::mx

  !>Should be identical to the ones in non_linear_terms
  integer :: mrad

  !>Diagnostics file lun and intialisation check
  integer :: idiag=0

contains


!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Reads the plasma toroidal rotation and perpendicular shearing rate, and 
!> shearing method.
!----------------------------------------------------------------------------

subroutine rotation_read_nml(ilun,io_stat,lwrite)

  namelist / rotation / vcor, shear_rate, shear_profile, toroidal_shear
  
  integer, intent(in)  :: ilun
  integer, intent(out) :: io_stat

  logical, optional, intent(in) :: lwrite

  io_stat = 0

  if (present(lwrite)) then
    if (.not. lwrite) then
      
    ! the default is no rotation, no shearing
    vcor = 0
    shear_rate = 0.E0
    ! Set the type of perpendicular shearing
    shear_profile     =  'none' 
    toroidal_shear    =  .false.

    ! read namelist and return on error
    read(ilun,NML=rotation,IOSTAT=io_stat)
    else
      ! do nothing
    end if
  else
    write(ilun,NML=rotation)
  end if

end subroutine rotation_read_nml

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! broadcast the rotation parameters
!----------------------------------------------------------------------------

subroutine rotation_bcast_nml

  call mpibcast_real(vcor, 1)
  call mpibcast_real(shear_rate,         1) 
  call mpibcast_logical(toroidal_shear, 1)
  call mpibcast_character(shear_profile, lenswitch)

end subroutine rotation_bcast_nml

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Run some checks to test for appropriate rotation paremters.
!----------------------------------------------------------------------------

subroutine rotation_check_params

  use control,    only : method, meth, non_linear
  use grid,       only : nx
  use geom, only : q, eps, geom_type, bp_frac
  use mode, only : mode_box
  use fft, only : working_fft_library

  select case(shear_profile)

    case('none')
      perp_shear=.false.
      shear_real=.false.
      shear_remap=.false.
      shear_ky_shift=.false.
      need_fft=.false.
      if (toroidal_shear) then
        toroidal_shear = .false.
        call gkw_warn('No E x B shear: uprim not overridden')
      end if

      !Warn if shearing rate is not zero
      if(abs(shear_rate) > 1e-10) then
        call gkw_warn('No E x B shear active')
        shear_rate=0.
      end if

    case('wavevector_remap')
      perp_shear=.true.
      shear_real=.false.
      shear_remap=.true.
      shear_ky_shift=.false.
      need_fft=.false.
      if (root_processor) write(*,*) &
        & 'Shear method: wavevector_remap: Check for nx convergence'

    case('ky_shift')
      perp_shear=.true.
      shear_real=.false.
      shear_remap=.false.
      shear_ky_shift=.true.
      need_fft=.true.
      if(root_processor) then
      write(*,*) 'Shear method: ky_shift: Boundary discontinuity in shear profile'
      write(*,*) 'Shear profile: Linear sawtooth with boundary discontinuity'
      call gkw_warn('Check shear rate normalisation for ky_shift')
      end if
 
    case('symmetric')
      perp_shear=.true.
      shear_real=.true.
      shear_remap=.false.
      need_fft=.true.
      shear_ky_shift=.false.
      write(*,*) 'Shear method: Triangle saw symmetric '
      call gkw_warn('Check shear rate normalisation for shear_real')
 
    case('linear')
      perp_shear=.true.
      shear_real=.true.
      shear_remap=.false.
      need_fft=.true.
      shear_ky_shift=.false.
      if (root_processor) then
      write(*,*) 'Shear profile: Linear sawtooth with boundary discontinuity'
      call gkw_warn('Check shear rate normalisation for shear_real')
      end if 

    case default 
      call gkw_abort('Unknown shearing option')
 
  end select
 
  if (non_linear) need_fft=.true.
  if (mode_box) need_fft=.true.

  if(perp_shear.and.geom_type/='s-alpha'.and.toroidal_shear) then 
     call gkw_warn('uprim not (yet) consistent for general geom')
  end if

  !Run some checks to test for appropriate control paremters
  if (shear_remap) then
    select case (method)
       case('EXP')
          select case(meth)
             case(3) 
                call gkw_abort('Shear not implemented for 3rd order scheme')
          end select
       case default ; call gkw_abort('rotation: This shearing option only '//&
                          &          'works with explicit time integration')
    end select
  end if

  !Needs to be tested after reading mode namelist....
  if(perp_shear.and..not.mode_box) then
     call gkw_abort('Shearing only availiable with mode_box')
     !This restriction could be lifted only if mode_box=false 
     !is made to work with multiple nx modes.
  end if

  if (.not. working_fft_library.and.need_fft) then
      call gkw_abort('(shear?) option requires a working FFT library')
  end if

  ! abort if NX is even for nonlinear runs or other runs that need the FFT
  ! kgrid (ixplus calc) also requires even nx for mode_box=.true.
  ! Perhaps this check better belongs somewhere else -duplicated in control
  if (mod(nx,2) == 0 .and. need_fft) then
    !write(*,*) nx, mod(nx,2)
    call gkw_abort('grid_size: '//                                           &
                  &'The use of conjugates in the implementation of the '//   &
                  &'fft requires NX to be odd.')
  end if

end subroutine rotation_check_params 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine shear_shift_ky(inout)
!--------------------------------------------------------------------
!<This routine is significantly faster than adding using shear_real
!<to add the shear as a term in add_non_linear_terms
!--------------------------------------------------------------------
use fft,          only : fourcol
use control,      only : dtim
use grid,         only : nx, nmod, nsp, nmu, nvpar, ns
use mode,         only : krho, lx
use dist,         only : indx, nsolc
COMPLEX,DIMENSION(nsolc),INTENT(INOUT) :: inout
REAL :: dx, dum, dum2
INTEGER :: ix,iky, ipar, is, jv, kt, imod

if(.not.shear_ky_shift) call gkw_abort('Invalid call to shear_shift_ky')

!arr=0.E0

!Possibly could drop repeated nyquist mode and do nx-1 sized fft. 
!But check they are the same first.
dx=lx/(nx+1)
dum2=-shear_rate*dx*dtim

DO ipar = 1, ns                   !Loop along feild line points
  DO is = 1, nsp                  !Loop over species
    DO jv = 1, nmu                !Loop over perpendicular velocity
      DO kt = 1, nvpar          !Loop over parallel velocity
          DO ix=1,nx
              DO imod=1, nmod
                arr(jind_nx(ix),imod)=inout(indx(imod,ix,ipar,jv,kt,is))
              END DO
          END DO

          call fourcol(arr, 1)
          arr = arr / ((nx+1))

          DO iky=1,nmod
              !Add the shear by shifting in ky.
              DO ix=1,nx+1
                  arr(ix,iky)=exp(ci1*krho(iky)*(ix-nx/2)*dum2)*arr(ix,iky)
              END DO
          END DO

          !Forward FFT of 2D array in x only
          !Intent (INOUT, For dir -1)
          call fourcol(arr, -1)

          !Do the inverse rearrangment
          DO ix=1,nx
            DO imod=1, nmod
              inout(indx(imod,ix,ipar,jv,kt,is))=arr(jind_nx(ix),imod)
            END DO
          END DO
        END DO
    END DO
  END DO
END DO

return
end subroutine shear_shift_ky

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine shear_remap_init
!--------------------------------------------------------------------
!<This routine is intended to be used for setting up an index array
!<But currently is not needed.
!--------------------------------------------------------------------
use grid, only : nmod, nx, ns, nmu, nvpar, nsp
use dist,    only : nf, msolc, indx
integer ::  ierr, idx=1, imod, ix, jv, kt, is, ipar

!Try adding back an extra copy of fdisi so as to be confident
!Not overwriting for spar.

if (.not.shear_remap) call gkw_abort('Invalid call to shear_remap_init')

    kxshift=0
    aindx=0
    aindx_shift=0

    if (shear_rate.gt.0) then 
        ixdir=1
        ixstart=1
        ixend=nx-1
        ixedge=nx
    else if (shear_rate.lt.0) then
        ixdir=-1
        ixstart=nx
        ixend=2
        ixedge=1
    else          !This might happen if the shear rate is zero
        shear_remap=.false.
        if (root_processor) write(*,*) 'Shear_remap off'
        return    !The index arrays will be blank but should never be used
    end if

  idx=1
  !Setup the indexing arrays for remapping
  !Based on the assumption that the index function increases monotonically with ix
  !Direction is reversed for negative shearing rates
  !The order of the loops must exactly match that in wavevector_remap
  DO imod = 1, nmod               !Loop over poloidal modes
    DO ipar = 1, ns               !Loop along feild line points
      DO jv = 1, nmu              !Loop over perpendicular velocity
        DO kt = 1, nvpar          !Loop over parallel velocity
          DO is = 1, nsp          !Loop over species
            DO ix = ixstart, ixend, ixdir       !Loop over radial modes
                aindx(idx) = indx(imod,ix,ipar,jv,kt,is)
                aindx_shift(idx)= indx(imod,ix+ixdir,ipar,jv,kt,is)
                idx=idx+1         
            END DO
          END DO
        END DO
      END DO
    END DO
  END DO

  !Perform check
  !write(*,*) nf, nmod*ns*nvpar*nmu*nsp, idx
  if (idx.ne.nf-(nmod*ns*nmu*nvpar*nsp)+1) then
    call gkw_abort('Severe error in wavevector remap init: idx')
  end if

return
end subroutine shear_remap_init

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine wavevector_remap(fdis,t)
!--------------------------------------------------------------------
!<Perpendicular shearing by remapping of kx wavevector grid
!--------------------------------------------------------------------
   use dist, only : nsolc, nf, indx, iphi, iapar
   use grid, only : nmod, nx, ns, nmu, nvpar, nsp
   use control, only : nlapar, nlphi
   use mode, only : krho, kxspace

   complex, intent(inout) :: fdis(nsolc)
   real, INTENT(IN) :: t !Time
   REAL :: shift 
   INTEGER :: dkx, ix, idx=1, imod, jv, kt, is, ipar

  if(.not.shear_remap) then
     call gkw_abort('exp_integration: Invalid call to wavevector_remap')
  end if

  idx=1

  DO imod = 1, nmod       !Loop over "poloidal" modes

     shift=krho(imod)*shear_rate*t

     !If the wavevectors need to be remapped
     if (nint(shift/kxspace).ne.kxshift(imod)) THEN

        !Calculate the number of kx grid points to shift the wavevectors
        dkx=nint(shift/kxspace)-kxshift(imod) 

        if (abs(dkx).gt.1) then
           call gkw_abort('Time resolution insufficient for shear remap')
        end if

        if (ixdir.ne.dkx) then
            write(*,*) "kxspace", kxspace
            write(*,*) "t, imod, ixdir, dkx", t, imod, ixdir, dkx
            call gkw_abort("Severe error in rotation:wavevector_remap")
        end if

        DO ipar = 1, ns                          !Loop along feild line points
          DO jv = 1, nmu                         !Loop over perpendicular velocity
             DO kt = 1, nvpar                     !Loop over parallel velocity
                DO is = 1, nsp                             !Loop over species
                    !Shift the wavevectors
                    DO ix = ixstart, ixend, ixdir     !Loop over radial modes
                        fdis(aindx(idx))=fdis(aindx_shift(idx))
                        idx=idx+1
                    END DO !nx
                    !Boundary condition in kx space
                    !This call to index function could also be removed
                    fdis(indx(imod,ixedge,ipar,jv,kt,is))=0.E0
                END DO !nsp
             END DO  !vpar
          END DO !nmu
        END DO !ns

        kxshift(imod)=kxshift(imod)+dkx

      ELSE
         idx=idx+ns*nmu*nvpar*nsp*(nx-1)
      ENDIF

   END DO !nmod

 !Perform check
 !write(*,*) idx, nf-(nmod*ns*nvpar*nmu*nsp)
 if (idx.ne.nf-(nmod*ns*nvpar*nmu*nsp)+1) then
     call gkw_abort('Severe error in wavevector remap: idx')
 end if

return
end subroutine wavevector_remap

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine shear_init
!--------------------------------------------------------------------
!<This subroutine initialises anything required for shearing
!<Depending on the shearing method, different init routines are called
!<This routine also recaluates some init quantities from non_linear terms
!<This is done to avoid a circular dependency
!--------------------------------------------------------------------
  use grid,      only : nmod, nx, n_s_grid, nmu, nvpar, nsp
  use mode,      only : kxrh, lx, ixzero
  use geom,      only : efun

  integer imod, ix, i, j, k, is, ifile, isp
  real ED

    if(.not.perp_shear) return

    !Account for the different coordinate normalisations of kx and ky
    !*(tensor picture) do BEFORE parallelize geom
    !The psi zeta component = efun(ipar,1,2)
    !shifts must occur at same time for all points, use outboard point
    !efun(i,1,2) is a flux function
    !The factor 2 arises because v_s & v_\chi have different normalisations
    shear_rate=2.*efun(n_s_grid/2+1,1,2)*shear_rate

    !Equivalent to: 
    !*(lx/ly box picture) shear_rate=s_j*shear_rate*q/(2.*pi*eps)
    !*(normalisation picture) shear_rate=s_j*shear_rate*kthnorm
    !This is normalisation FORWARDS from rho ref units to angular units

    if(shear_real) call shear_real_init

    if(shear_remap) call shear_remap_init

    if(shear_ky_shift) then
    !With odd number of modes
    !The nyquist frequency is repeated - incorrectly?
    !Setup index arrays for ky_shift
        DO ix = ixzero, nx
          jind_nx(ix) = ix - ixzero + 1
        END DO 
 
        DO ix = ixzero-1, 1, -1 
          jind_nx(ix) = nx + ix - ixzero + 2
        END DO  
    endif

return
end subroutine shear_init

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine shear_real_init
!--------------------------------------------------------------------
!<This subroutine initialises the grad_pot array for the shearing
!<When added in add_non_linear terms.
!--------------------------------------------------------------------
use grid,         only : ns, nmod
use mode,         only : lx
!use geom,         only : bn

integer j, ipar, qmrad

if(.not.shear_real) call gkw_abort('Invalid call to shear_real_init')

!Beware integer division
qmrad = mrad/4

!!NEED TO CHECK NORMALISATION OF SHEAR RATE IF USING THIS METHOD

  if(4*qmrad.ne.mrad) then
      write(*,*) 'Warning: mrad not a multiple of 4' 
      write(*,*)  '-> small discontinuity in shear'
  endif
  !Could do this with qmrad=real(mrad/4).

  !if(lverbose) write(*,*) 'mrad', mrad, 'qmrad', qmrad, 'lx', lx 

  select case(shear_profile)

  case('linear') !Linear shear discontinuous in periodic radial boundary
            if(lverbose) write(*,*) 'Discontinuous linear shearing' 

            shear_rate=shear_rate*lx/mrad
 
            DO ipar = 1, ns  
                DO j = 1, mrad
                    !grad_pot(j, ipar)= (-(mrad/2)+j)*bn(ipar)*shear_rate
                    grad_pot(j, ipar) =(-(mrad/2)+j)*shear_rate
                END DO
            END DO

  case('symmetric') !Symmetric profile continuous at boundary
            !The factor of B_N in geom has been removed.
            !Factor of two to cancel out efun divisor of two???
            shear_rate=shear_rate*lx/mrad

            if(lverbose) write(*,*) 'Symmetric shearing profile' 
            DO ipar = 1, ns
                !mrad will always be a multiple of 2
                DO j = 1, mrad/2
                  !grad_pot(j, ipar) =(-qmrad+j)*bn(ipar)*shear_rate
                  grad_pot(j, ipar) =(-qmrad+j)*shear_rate
                END DO

                DO j = mrad/2+1, mrad
                  !grad_pot(j, ipar) =(3*qmrad-j)*bn(ipar)*shear_rate
                  grad_pot(j, ipar) =(3*qmrad-j)*shear_rate 
                END DO
            END DO
  
  case default
        call gkw_abort ('shear_real_init called incorrectly')
  
  end select
  
  return

end subroutine shear_real_init

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine rotation_allocate
!--------------------------------------------------------------------
!<This subroutine allocates the arrays needed for the rotation module
!--------------------------------------------------------------------
use grid, only : ns,nx,nmod,nsp,nmu,nvpar

integer ierr
real dum

!Note this must be identical to the one in non_linear_terms
!Must also be calculated here to avoid a circular dependecy
dum  = 1.5*real(nx+1) 
mrad = log(dum)/log(2.E0) + 1.E0  
mrad = 2**mrad 
mx=2*nx

if(.not.perp_shear) return

if(shear_real) then
  ierr = 0 
  allocate(grad_pot(mrad,ns), stat = ierr) 
  if (ierr.ne.0) then 
    write(*,*)'Could not allocate grad_pot in module rotation'
    stop
  endif
endif

if(shear_ky_shift) then
  ierr = 0 
  allocate(arr(nx+1,nmod), stat = ierr) 
  if (ierr.ne.0) then 
    write(*,*)'Could not allocate arr in module rotation'
    stop
  endif

  ierr = 0 
  allocate(jind_nx(nx), stat = ierr) 
  if (ierr.ne.0) then 
    write(*,*)'Could not allocate jind_nx in module rotation'
    stop
  endif
endif

if (shear_remap) then
    ierr=0
    allocate(aindx(nmod*nmu*nvpar*ns*nsp*(nx-1)),stat=ierr)
    if (ierr.ne.0) call gkw_abort('Could not allocate aindx in module rotation')

    ierr=0
    allocate(aindx_shift(nmod*nmu*nvpar*ns*nsp*(nx-1)),stat=ierr)
    if (ierr.ne.0) call gkw_abort('Could not allocate aindx_shift, module rotation')

    ierr=0
    allocate(kxshift(nmod),stat=ierr)
    if (ierr.ne.0) call gkw_abort('Could not allocate kxshift in module rotation')
    kxshift=0.E0
end if

return
end subroutine rotation_allocate

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

end module rotation
