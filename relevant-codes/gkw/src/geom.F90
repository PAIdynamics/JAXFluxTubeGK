module geom
! SVN:$Id: geom.F90 1005 2009-07-02 16:12:03Z  $
use global
use mpiinterface
use general, only : gkw_abort, gkw_warn

implicit none

private

  public :: geom_read_nml, geom_write_nml, geom_bcast_nml
  public :: geom_check_params, parallelize_geom 
  public :: geom_output

public :: bn, dfun, efun, eps, ffun, geom_allocate, bp_frac, bt_frac
public :: geom_init_grids, gfun, hfun, ints, kthnorm, metric, q, sgr
public :: sgr_dist, shat, bmin, bmax, metric_G
public :: beta_real, beta_prime_real, pol_angle, Rref, Bref
!Ideally the ones below should not need to be public.
public :: geom_type


  interface geom_write_nml
    module procedure geom_read_nml
  end interface

!> radial coordinate 
real eps

!> Safety factor 
real q

!> Magnetic shear 
real shat

!> Switch for the geometry (private)
character (len = lenswitch) :: geom_type  

integer, parameter :: lenfile = 180 
!> File to read the equilibrium related quantities (private)
character (len = lenfile) :: eqfile

!> Sign of B.grad_phi, the toroidal component of the magnetic field
integer signB

!> Sign of j.grad_phi, the toroidal component of the plasma current
integer signJ

!> Radial coordinate used to specify the chosen FS
!> 1=eps, 2=rho_psi
integer eps_type

!> The plasma beta 
real :: beta_real 

!> and beta_prime
real :: beta_prime_real 

!> The reference major radius 
real :: Rref 

!> The reference magnetic field
real :: Bref 

!> Grid distance along the field line 
real :: sgr_dist 

!> the normfactor for calculating the k_zeta values in the code 
!> initial value given to catch possible errors
real :: kthnorm=1.23e4

!> The minimum and maximum magnetic field on the flux surface 
real :: bmin, bmax 

!> the normalized magnetic field strength : bn_G(1:n_s_grid) (GLOBAL)
real, allocatable :: bn_G(:)
!> the normalized magnetic field strength : bn(1:n_s_grid) (LOCAL)
real, allocatable :: bn(:)

!> The coordinate along the field line : sgr(0:n_s_grid+1)
real, allocatable :: sgr(:,:)

!> and associated poloidal angle: pol_angle(0:n_s_grid+1)
real, allocatable :: pol_angle(:)

!> this array is used for the flux surface average.
real, allocatable :: ints(:,:)

!> the curvature operator  dfun(n_s_grid,3) 
real, allocatable :: dfun(:,:)

!> the ExB operator efun(n_s_grid,3,3) 
real, allocatable :: efun(:,:,:)

!> the factor in front of the parallel derivative ffun(n_s_grid,3)
real, allocatable :: ffun(:) 

!> The trapping operator gfun(n_s_grid)
real, allocatable :: gfun(:)

!> The tensor that determines the Coriolis drift 
real, allocatable :: hfun(:,:)

!> The array connected to the centrifugal drift 
real, allocatable :: ifun(:,:)

!> Bp / over B  
real, allocatable :: bp_frac(:)

!> |Bt| / over B
real, allocatable :: bt_frac(:)

!APS: > unused array at pressent 
!APS: real, allocatable :: xgr(:)

!> the metric (GLOBAL)
real, allocatable :: metric_G(:,:,:)
!> the metric (LOCAL)
real, allocatable :: metric(:,:,:)

contains

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> read (or write) geom namelist
!----------------------------------------------------------------------------

subroutine geom_read_nml(ifile,io_stat,lwrite)

  integer, intent(in)  :: ifile
  integer, intent(out) :: io_stat

  logical, optional, intent(in) :: lwrite

  namelist /geom/ shat, q, eps, eps_type, geom_type, eqfile, signB, signJ

  io_stat = 0
  
  if (present(lwrite)) then
    if (.not. lwrite) then
        
      ! Set the standard values  
      q    = 1.23e4
      shat = 1.23e4
      eps  = 1.23e4
      eps_type = 1
      geom_type = 's-alpha'
      eqfile = "not_needed_for_s-alpha_option"
      signB = 1
      signJ = 1
      
      ! read nml
      read(ifile,NML=geom,IOSTAT=io_stat)
    else
      ! do nothing
    end if
  else
    ! write nml
    write(ifile,NML=geom)
  end if
  
end subroutine geom_read_nml
               
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> broadcast geom input parameters
!----------------------------------------------------------------------------

subroutine geom_bcast_nml

  call mpibcast_real(q,    1)
  call mpibcast_real(shat, 1)
  call mpibcast_real(eps,  1)
  call mpibcast_integer(eps_type,  1)
  call mpibcast_character(geom_type,  lenswitch)
  call mpibcast_character(eqfile,  lenfile)
  call mpibcast_integer(signB,1)
  call mpibcast_integer(signJ,1)

end subroutine geom_bcast_nml
                             
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> check geom input parameters
!> in the case of chease this routine may be called twice
!----------------------------------------------------------------------------

subroutine geom_check_params(icall)
use control, only : nlapar

integer, intent(in) :: icall
logical :: eqfile_exists

  if (abs(eps-1.23e4) < 1e-4) then
    call gkw_abort('geom_check: You have not specified eps in the input')
  endif

  if (eps < 0.) then
    call gkw_abort('geom_check: I do not understand negative aspect ratio')
  end if

  if(q < 0.) then
    call gkw_abort('geom_check: To run negative q, please reverse signJ')
  end if

  if (geom_type == 's-alpha') then
    if (abs(q-1.23e4) < 1e-4) then
      call gkw_abort('geom_check: You have not specified q in the input')
    endif
    if (abs(shat-1.23e4) < 1e-4) then
      call gkw_abort('geom_check: You have not specified shat in the input')
    endif
    eqfile = 'not_needed_for_s-alpha_option'
  else if (geom_type == 'chease') then
    if (eqfile == 'not_needed_for_s-alpha_option') then
      call gkw_abort('geom_check: You have not specified eqfile in the input')
    endif
    inquire(file=eqfile,EXIST=eqfile_exists)
    if (.not. eqfile_exists) call gkw_abort('geom: not found: '//eqfile)
  !Namelist input values of q and shat are not used for chease case
    !The values are checked again in kgrid and krbal to make sure they have
    !been read from the chease input and set correctly in geom_init_grids
    if(icall==1) then
        if(q<1.22e4) then
          call gkw_warn('Namelist input q not used for geom type chease')
          q = 1.23e4
        endif
        if(shat<1.22e4) then
          call gkw_warn('Namelist input shat not used for geom type chease')
          shat = 1.23e4
        endif
    endif !call
  endif !geom type

  ! make sure that abs(signB)=1 and abs(signJ)=1
  if (signB >= 0) then
     signB = 1
  else
     signB = -1
  endif
  if (signJ >= 0) then
     signJ = 1
  else
     signJ = -1
  endif

end subroutine geom_check_params

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!--------------------------------------------------------------------
!> allocate the arrays of geom
!--------------------------------------------------------------------
subroutine geom_allocate
use grid, only : n_s_grid, nx, nmod, ns, parallel_s 

implicit none

! local parameters 
integer ierr

! intialize the error parameter
ierr=0

! allocate the magnetic field array
allocate(bn_G(1:n_s_grid),stat=ierr)
if (ierr.ne.0) then
  stop 'Could not allocate bn_G in geom'
endif
! allocate the magnetic field array
! 2 extra points, parallel or otherwise
allocate(bn(1-2:ns+2),stat=ierr)
if (ierr.ne.0) then
  stop 'Could not allocate bn in geom'
endif

! allocate the field line length array
allocate(sgr(nx,0:n_s_grid+1),stat=ierr)
if (ierr.ne.0) then
  stop 'Could not allocate sgr in geom'
endif

! allocate the poloidal angle array
allocate(pol_angle(1:n_s_grid),stat=ierr)
if (ierr.ne.0) then
  stop 'Could not allocate pol_angle in geom'
endif

! allocate the array for integration along the 
! field line 
allocate(ints(nx,n_s_grid),stat=ierr)
if (ierr.ne.0) then
  stop 'Could not allocate ints in geom'
endif

! allocate the array that contains the function 
! connected witht the curvature operator 
allocate(dfun(n_s_grid,3), stat = ierr)
if (ierr.ne.0) then 
  stop 'could not allocate dfun in geom' 
endif 

! allocate the array that contains the function 
! connected with the ExB velocity 
allocate(efun(n_s_grid,3,3), stat = ierr)
if (ierr.ne.0) then 
  stop 'could not allocate efun in geom' 
endif 

! allocate the array that contains the function in 
! front of the parallel derivative 
allocate(ffun(n_s_grid), stat = ierr)
if (ierr.ne.0) then 
  stop 'could not allocate ffun in geom' 
endif 

! allocate the array that contains the function 
! connected with the trapping terms
allocate(gfun(n_s_grid), stat = ierr)
if (ierr.ne.0) then 
  stop 'could not allocate gfun in geom' 
endif 

! allocate the array that contains the function 
! connected with the coriolis terms 
allocate(hfun(n_s_grid,3), stat = ierr)
if (ierr.ne.0) then 
  stop 'could not allocate hfun in geom' 
endif 

! allocate the array that contains the function 
! connected with the centrifugal terms 
allocate(ifun(n_s_grid,3), stat = ierr)
if (ierr.ne.0) then 
  stop 'could not allocate ifun in geom' 
endif 

! allocate the array that contains the function 
! connected with the centrifugal terms 
allocate(bt_frac(n_s_grid), stat = ierr)
if (ierr.ne.0) then 
  stop 'could not allocate bt_frac in geom' 
endif

! allocate the array that contains the function 
! connected with the centrifugal terms 
allocate(bp_frac(n_s_grid), stat = ierr)
if (ierr.ne.0) then 
  stop 'could not allocate bp_frac in geom' 
endif

! allocate the radial grid array
!APS: allocate(xgr(0:nx+1),stat=ierr)
!APS: if (ierr.ne.0) then
!APS:   stop 'Could not allocate xgr in geom'
!APS: endif

! allocate the metric_G (GLOBAL)
allocate(metric_G(n_s_grid,3,3),stat=ierr)
if (ierr.ne.0) then
  stop 'Could not allocate the metric_G in geom'
endif
! allocate the metric (LOCAL)
if (parallel_s) then
  allocate(metric(1-2:ns+2,3,3),stat=ierr)
else
  allocate(metric(ns,3,3),stat=ierr)
endif
if (ierr.ne.0) then
  stop 'Could not allocate the metric in geom'
endif

end subroutine geom_allocate

!--------------------------------------------------------------------
!> Initializes the grids of geom 
!--------------------------------------------------------------------
subroutine geom_init_grids

use grid, only : nmod, n_s_grid, nx, nperiod, parallel_s
use constants, only : pi

! for geom_type='chease'

real  sgrmax, dum
integer i,ix,j
!
! variables specific to geom_type='chease'
integer :: igeom, ierr, ns_c, npsi_c, npol, s_coeff
real    :: depsdpsi, p, dpdeps, jac, dqdpsi, dqdpsi_dum, F
integer :: psi_indx ! index for the psi-grid
integer, allocatable :: s_indx(:)  ! indexes for the s-grid
real, allocatable    :: dBdeps(:), dBds(:) ! 1D (s) arrays
real, allocatable    :: dZdpsi(:), dZds(:), Z_FS(:)
real, allocatable    :: dRdpsi(:), dRds(:), R_FS(:)
real, allocatable    :: dzetadpsi(:), dzetadchi(:)
real, allocatable :: psi_dum(:)  ! 1D (psi) arrays
real, allocatable :: psi_s_dum(:,:)  ! 2D (psi,s) arrays
logical :: op
character (len=20) tdum
!

! grid along the field line ('s' coordinate)
! currently the same for all radial modes 
sgrmax = real(nperiod) - 0.5E0

do ix = 1, nx
  do i  = 0, n_s_grid+1
    sgr(ix,i) = -sgrmax + 2E0*sgrmax*(i-0.5E0)/n_s_grid
  end do 
end do 

! ints is used for the flux surface average 
do ix = 1, nx
  dum = 0. 
  do i = 1, n_s_grid
    ints(ix,i) = 0.5*(sgr(ix,i+1)-sgr(ix,i-1)) 
    dum = dum + ints(ix,i) 
  end do 
  do i = 1, n_s_grid 
    ints(ix,i) = ints(ix,i) / dum 
  end do 
end do 

! Set the grid distance (currently the same for all radial 
! surface, i.e. flux tube limit) 
sgr_dist = sgr(1,2)-sgr(1,1)

! Initialises Bref, Rref, beta and beta_prime
Rref = -100.E0
Bref = -100.E0
beta_real = -100.E0
beta_prime_real = -100.E0
bp_frac(:) = -100.E0
bt_frac(:) = -100.E0

! metric and associated elements (magnetic field, beta,...)
select case(geom_type)

  case('s-alpha') 

    ! Magnetic field strength (same for all radial modes)
    ! and poloidal angle 
    do i = 1, n_s_grid
      dum = 1. + eps * cos(2.E0*pi*sgr(1,i))
      bn_G(i) = 1.E0 / dum 
      pol_angle(i)=2E0*pi*sgr(1,i)
    end do

    ! set the minimum and maximum magnetic field strength 
    bmin = 1.E0 / (1.E0 + eps) 
    bmax = 1.E0 / (1.E0 - eps) 

    !Set the toroidal and polodial fractions of the field
    !bt_frac should always be positive
    bt_frac(:)=1.
    bp_frac(:)=signJ*eps/q 

    ! the function connected with the trapping terms 
    do i = 1, n_s_grid 
      gfun(i) = eps * sin(2*pi*sgr(1,i))/ q * signJ
    end do 

    
    ! calculate the array for the parallel derivative 
    do i = 1, n_s_grid 
      ffun(i) = signJ / ( 2.E0* pi * q ) 
    end do 

    ! replace by numerical derivative 
    call logbderiv(gfun)
    
    ! calculate the curvature function 
    do i = 1, n_s_grid 

      ! the psi component
      dfun(i,1) = - sin(2.E0*pi*sgr(1,i)) / bn_G(i) * signB

      ! the zeta component 
      dfun(i,2) = -  q * (cos(2.E0*pi*sgr(1,i)) + 2.E0 * pi *  &
                &    shat* sgr(1,i) * sin(2.E0*pi*sgr(1,i)))/  & 
                &    (2.E0 * pi * eps * bn_G(i)) * signJ

      ! the s component 
      dfun(i,3) = - cos(2.E0*pi*sgr(1,i)) / (2.*pi*eps*bn_G(i)) &
                &   * signB

      !write(*,*)'Warning constant curvature' 
 
      ! the psi component
      dfun(i,1) = - sin(2.E0*pi*sgr(1,i)) * signB

      ! the zeta component 
      dfun(i,2) = -  q * (cos(2.E0*pi*sgr(1,i)) + 2.E0 * pi *  &
                &    shat* sgr(1,i) * sin(2.E0*pi*sgr(1,i)))/  & 
               &    (2.E0 * pi * eps ) * signJ
      ! the s component 
      dfun(i,3) = - cos(2.E0*pi*sgr(1,i)) / (2.*pi*eps) * signB

    end do 

    ! calculate the array connected with the ExB velocity 
    do i = 1, n_s_grid 

      ! the diagonal components are zero 
      efun(i,1,1) = 0.
      efun(i,2,2) = 0. 
      efun(i,3,3) = 0. 
      
      ! the psi zeta component 
      efun(i,1,2) =  q / (4.E0*pi*eps*bn_G(i)) * signJ
      
      ! the psi s component 
      efun(i,1,3) =  1.E0 / (4.E0*pi*eps*bn_G(i)) * signB
      
      ! the zeta s component 
      efun(i,2,3) =  q * shat * sgr(1,i) / &
                  &  (4.E0*pi*eps**2*bn_G(i)) * signJ
      
      !Write(*,*)'warning no bfield in the ExB velocity'
      
      ! the psi zeta component 
      efun(i,1,2) =  q / (4.E0*pi*eps) * signJ
      
      ! the psi s component 
      efun(i,1,3) =  1.E0 / (4.E0*pi*eps) * signB
      
      ! the zeta s component 
      efun(i,2,3) =  q * shat * sgr(1,i) / &
                  & (4.E0*pi*eps**2) * signJ
      
      ! the other components are anti-symmetric 
      efun(i,2,1) = - efun(i,1,2) 
      efun(i,3,1) = - efun(i,1,3) 
      efun(i,3,2) = - efun(i,2,3) 
    
    end do 
    
    
    ! in cirular geometry hfun can be approximated by dfun 
    do i = 1, n_s_grid
      do j = 1, 3  
        hfun(i,j) = dfun(i,j)
      end do 
    end do 
    
    ! Calculate the normfactor for k_zeta 
    kthnorm = q  / ( 2 * pi * eps) 
    
    ! metric elements
    ! Normalised: metric(i,2,2)=g_zeta_zeta*Rref**2
    do i = 1, n_s_grid 
    
      ! the psi psi element 
      metric_G(i,1,1) = 1.E0 
      
      ! the psi zeta element 
      metric_G(i,1,2) = q * shat * sgr(1,i) / eps *signB*signJ

      ! the psi s element 
      metric_G(i,1,3) = 0. 
      
      ! the zeta zeta element 
      metric_G(i,2,2) = ( q / (2 * pi * eps ))**2 * (1 +     &
                    & (2.E0*pi*sgr(1,i)*shat)**2)
      
      ! the zeta s element 
      metric_G(i,2,3) = q / (2 * pi * eps)**2 *signB*signJ

      ! the s s element 
      metric_G(i,3,3) = 1.E0 / (2 * pi * eps)**2
      
      ! for the other elements symmetry applies 
      metric_G(i,2,1) = metric_G(i,1,2) 
      metric_G(i,3,1) = metric_G(i,1,3) 
      metric_G(i,3,2) = metric_G(i,2,3) 
    
    end do 

    ! centrifugal terms set to zero for the moment
    do i = 1, n_s_grid
      do j = 1, 3  
        ifun(i,j) = 0
      end do 
    end do 


  case('chease') 
 

    ! 0) some checks
    if (mod(n_s_grid,2).eq.0) then
      call gkw_abort('N_s_grid must be odd for geom_type=chease')
    endif
    if (mod(n_s_grid,2*nperiod-1).ne.0) then
      call gkw_abort( &
  &        'N_s_grid/(2*NPERIOD-1) has to be an integer for geom_type=chease')
      stop
    else
      npol = n_s_grid/(2*nperiod-1) !nb of points per poloidal turn for the GKW s-grid
    endif

    !if (root_processor) then !Read on all processors to have clean aborts.
      ! 1) OPEN
      ! Open ASCII file
      igeom = 20
      do
        inquire(unit=igeom,opened=op)
        if (.NOT.op) exit
        igeom = igeom+1
      end do
      open(igeom, status='old', action='read', file=eqfile)
      rewind igeom

      ! 2) READ dimensions 
      ! number of points for psi-grid and s-grid
      read(igeom,*) tdum, npsi_c, tdum, ns_c
      ! reference R and B (used for normalisation in GKW)
      ! taken to be R0EXP and B0EXO used for normalisation in CHEASE 
      ! R0EXP and B0EXP close to the magnetic axis values (but not exactly) 
      read(igeom,*) tdum, Rref, tdum, Bref, tdum, tdum

      ! 3) more checks
      ! check that CHEASE and GKW s-grid are compatible and that
      ! CHEASE grid is at least two times denser than GKW grid
      if (mod(ns_c,2*npol).ne.0) then
        call gkw_abort( &
   &         'NCHI in chease has to be a multiple of 2*N_s_grid/(2*NPERIOD-1)')
      else
        s_coeff=ns_c/npol ! how much the s-grid is denser in chease
      endif

      ! 4) ALLOCATE the arrays
      allocate(s_indx(1:n_s_grid),stat=ierr)
      if (ierr.ne.0) then
        stop 'Could not allocate s_indx in geom'
      endif
      allocate(dBdeps(1:n_s_grid),stat=ierr)
      if (ierr.ne.0) then
        stop 'Could not allocate dBdeps in geom'
      endif
      allocate(dBds(1:n_s_grid),stat=ierr)
      if (ierr.ne.0) then
        stop 'Could not allocate dBds in geom'
      endif
      allocate(R_FS(1:n_s_grid),stat=ierr)
      if (ierr.ne.0) then
        stop 'Could not allocate R_FS in geom'
      endif
      allocate(Z_FS(1:n_s_grid),stat=ierr)
      if (ierr.ne.0) then
        stop 'Could not allocate R_FS in geom'
      endif
      allocate(dRdpsi(1:n_s_grid),stat=ierr)
      if (ierr.ne.0) then
        stop 'Could not allocate dRdpsi in geom'
      endif
      allocate(dRds(1:n_s_grid),stat=ierr)
      if (ierr.ne.0) then
        stop 'Could not allocate dRds in geom'
      endif
      allocate(dZdpsi(1:n_s_grid),stat=ierr)
      if (ierr.ne.0) then
        stop 'Could not allocate dZdpsi in geom'
      endif
      allocate(dZds(1:n_s_grid),stat=ierr)
      if (ierr.ne.0) then
        stop 'Could not allocate dZds in geom'
      endif
      allocate(dzetadpsi(1:n_s_grid),stat=ierr)
      if (ierr.ne.0) then
        stop 'Could not allocate dzetadpsi in geom'
      endif
      allocate(dzetadchi(1:n_s_grid),stat=ierr)
      if (ierr.ne.0) then
        stop 'Could not allocate dzetadchi in geom'
      endif
      allocate(psi_dum(1:npsi_c),stat=ierr)
      if (ierr.ne.0) then
        stop 'Could not allocate psi_dum in geom'
      endif
      allocate(psi_s_dum(1:npsi_c,1:ns_c),stat=ierr)
      if (ierr.ne.0) then
        stop 'Could not allocate psi_s_dum in geom'
      endif

      ! 5) READ 1D arrays

      ! find the index for the radial grid
      select case (eps_type)
        case (1) ! use eps to select FS
          ! psi-grid, s-grid, Rgeom (not used)
          read(igeom,*) tdum, (dum,i=1,npsi_c) !psi
          read(igeom,*) tdum, (dum,i=1,ns_c) !s 
          read(igeom,*) tdum, (dum,i=1,npsi_c) !Rgeom

          ! amin=(Rmax-Rmin)/2 
          read(igeom,*) tdum, (psi_dum(i),i=1,npsi_c) !amin
          ! find the index for the psi grid
          psi_indx=1
          do while (abs(psi_dum(psi_indx)/Rref-eps)>0.008*psi_dum(npsi_c)/Rref .and. psi_indx.LT.npsi_c) 
            psi_indx = psi_indx+1
          end do

        case (2)! use rho_psi to select FS
          ! psi-grid
          read(igeom,*) tdum, (psi_dum(i),i=1,npsi_c) !psi
          ! find the index for the psi grid
          psi_indx=1
          do while (abs(sqrt(psi_dum(psi_indx)/psi_dum(npsi_c))-eps)>0.008 .and. psi_indx.LT.npsi_c) 
            psi_indx = psi_indx+1
          end do

          ! s-grid, Rgeom (not used)
          read(igeom,*) tdum, (dum,i=1,ns_c) !s 
          read(igeom,*) tdum, (dum,i=1,npsi_c) !Rgeom

          ! amin=(Rmax-Rmin)/2 
          read(igeom,*) tdum, (psi_dum(i),i=1,npsi_c) !amin

        case default
          call gkw_abort('geom: value not allowed for eps_type')
      end select

      if (psi_indx .GE. npsi_c) then
        call gkw_abort('Radial grid too coarse in CHEASE: increase NPSI')
      endif

      ! value of eps=amin/R (used in mode.F90 for kxspace)
      eps = psi_dum(psi_indx)/Rref

      ! depsdpsi 
      ! -> to go from chease (psi,zeta,s) to GKW (eps,zeta,s)
      read(igeom,*) tdum, (psi_dum(i),i=1,npsi_c) ! damindpsi
      depsdpsi = psi_dum(psi_indx)/Rref
      
      ! Bmax, Bmin, q, shat
      read(igeom,*) tdum, (psi_dum(i),i=1,npsi_c) ! Bmax
      bmax = psi_dum(psi_indx)/Bref
      read(igeom,*) tdum, (psi_dum(i),i=1,npsi_c) ! Bmin
      bmin = psi_dum(psi_indx)/Bref
      read(igeom,*) tdum, (psi_dum(i),i=1,npsi_c) ! q
      q = psi_dum(psi_indx)
      read(igeom,*) tdum, (psi_dum(i),i=1,npsi_c) ! dqdpsi
      dqdpsi = psi_dum(psi_indx)
      shat = eps * dqdpsi / q / depsdpsi
      read(igeom,*) tdum, (psi_dum(i),i=1,npsi_c) ! dqdpsi_chk
      dqdpsi_dum = psi_dum(psi_indx)

      ! p and dpdeps (not normalised) 
      ! -> to be used for beta and beta_prime
      read(igeom,*) tdum, (psi_dum(i),i=1,npsi_c) ! p
      p = psi_dum(psi_indx)
      read(igeom,*) tdum, (psi_dum(i),i=1,npsi_c) ! dpdpsi
      dpdeps = psi_dum(psi_indx)/depsdpsi
      
      ! jacobian J_psi_zeta_s
      read(igeom,*) tdum, (psi_dum(i),i=1,npsi_c) ! jacobian
      jac=psi_dum(psi_indx)

      ! not used (don't change the order!)
      read(igeom,*) tdum, (dum,i=1,npsi_c) ! djacdpsi

      !chease F (not the same as GKW F tensor which is ffun)
      read(igeom,*) tdum, (psi_dum(i),i=1,npsi_c)
      F=psi_dum(psi_indx)

      !not used
      read(igeom,*) tdum, (dum,i=1,npsi_c) ! dFdpsi
  
      ! 6) READ 2D arrays
      ! build the index correspondance for the s-grid
      ! CHEASE s-array goes from 0 to 1 (LFS midplane, counterclockwise)
      do i = 1, npol
        do j = 1, 2*nperiod - 1
          dum=s_coeff*(i-1)+s_coeff/2+1
          if (dum.gt.ns_c/2) then
            s_indx(i+(j-1)*npol)=dum-ns_c/2
          else
            s_indx(i+(j-1)*npol)=dum+ns_c/2
          endif
        end do
      end do
  
      ! metric_G elements, (psi,zeta,s) coordinates
      ! g_psi_psi
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((psi_s_dum(i,j),i=1, npsi_c),j=1, ns_c) !g11
      do i = 1, n_s_grid
        metric_G(i,1,1)=psi_s_dum(psi_indx,s_indx(i))
      end do
      ! g_psi_zeta
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((psi_s_dum(i,j),i=1, npsi_c),j=1, ns_c) !g12
      do i = 1, n_s_grid
        metric_G(i,1,2)=psi_s_dum(psi_indx,s_indx(i))
        metric_G(i,2,1)=psi_s_dum(psi_indx,s_indx(i))
      end do
      ! g_psi_s
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((psi_s_dum(i,j),i=1, npsi_c),j=1, ns_c) !g13
      do i = 1, n_s_grid
        metric_G(i,1,3)=psi_s_dum(psi_indx,s_indx(i))
        metric_G(i,3,1)=psi_s_dum(psi_indx,s_indx(i))
      end do
      ! g_zeta_zeta
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((psi_s_dum(i,j),i=1, npsi_c),j=1, ns_c) !g22
      do i = 1, n_s_grid
        metric_G(i,2,2)=psi_s_dum(psi_indx,s_indx(i))
      end do
      ! g_zeta_s
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((psi_s_dum(i,j),i=1, npsi_c),j=1, ns_c) !g23
      do i = 1, n_s_grid
        metric_G(i,2,3)=psi_s_dum(psi_indx,s_indx(i))
        metric_G(i,3,2)=psi_s_dum(psi_indx,s_indx(i))
      end do
      ! g_s_s
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((psi_s_dum(i,j),i=1, npsi_c),j=1, ns_c) !g33
      do i = 1, n_s_grid
        metric_G(i,3,3)=psi_s_dum(psi_indx,s_indx(i))
      end do
  
  
      ! magnetic field
      ! norm(B)
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((psi_s_dum(i,j),i=1, npsi_c),j=1, ns_c) !B
      do i = 1, n_s_grid
        bn_G(i)=psi_s_dum(psi_indx,s_indx(i))/Bref
      end do
      ! dBdeps (not normalized)
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((psi_s_dum(i,j),i=1, npsi_c),j=1, ns_c) ! dBdpsi
      do i = 1, n_s_grid
        dBdeps(i)=psi_s_dum(psi_indx,s_indx(i))/depsdpsi
      end do
      ! dBds (not normalized)
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((psi_s_dum(i,j),i=1, npsi_c),j=1, ns_c) ! dBds
      do i = 1, n_s_grid
        dBds(i)=psi_s_dum(psi_indx,s_indx(i))
      end do
  
      ! R and Z
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((psi_s_dum(i,j),i=1, npsi_c),j=1, ns_c) ! R
      do i = 1, n_s_grid
        R_FS(i)=psi_s_dum(psi_indx,s_indx(i))
      end do
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((psi_s_dum(i,j),i=1, npsi_c),j=1, ns_c) ! Z
      do i = 1, n_s_grid
        Z_FS(i)=psi_s_dum(psi_indx,s_indx(i))
      end do
  
      ! R and Z gradients
      ! dRdpsi
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((psi_s_dum(i,j),i=1, npsi_c),j=1, ns_c) ! dRdpsi
      do i = 1, n_s_grid
        dRdpsi(i)=psi_s_dum(psi_indx,s_indx(i))
      end do
      ! dRds
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((psi_s_dum(i,j),i=1, npsi_c),j=1, ns_c) ! dRds
      do i = 1, n_s_grid
        dRds(i)=psi_s_dum(psi_indx,s_indx(i))
      end do
      ! dZdpsi
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((psi_s_dum(i,j),i=1, npsi_c),j=1, ns_c) ! dZdpsi
      do i = 1, n_s_grid
        dZdpsi(i)=psi_s_dum(psi_indx,s_indx(i))
      end do
      ! dZds
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((psi_s_dum(i,j),i=1, npsi_c),j=1, ns_c) ! dZds
      do i = 1, n_s_grid
        dZds(i)=psi_s_dum(psi_indx,s_indx(i))
      end do
  
      ! poloidal angle
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((psi_s_dum(i,j),i=1, npsi_c),j=1, ns_c) ! theta
      do i = 1, n_s_grid
        dum=psi_s_dum(psi_indx,s_indx(i))
        if (dum.gt.pi) then
          dum = dum - 2E0*pi
        endif
        pol_angle(i) = dum + 2E0*pi*(floor(real(i-1)/real(npol))-real(nperiod)+1)
      end do

      ! not used
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((dum,i=1, npsi_c),j=1, ns_c) ! Ah
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((dum,i=1, npsi_c),j=1, ns_c) ! dAhdpsi

      ! dzetadpsi
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((psi_s_dum(i,j),i=1, npsi_c),j=1, ns_c) ! dzetadpsi
      do i = 1, n_s_grid
        dzetadpsi(i)=psi_s_dum(psi_indx,s_indx(i))
      end do

      ! dzetadchi
      read(igeom,'(A)') tdum
      read(igeom,'(1P5E20.10)') ((psi_s_dum(i,j),i=1, npsi_c),j=1, ns_c) ! dzetadchi
      do i = 1, n_s_grid
        dzetadchi(i)=psi_s_dum(psi_indx,s_indx(i))
      end do

      close(igeom)
  
      
      ! 7) Compute the missing elements
  
      ! Correction to the metric_G elements involving dzetadpsi
      ! because dzetadpsi is not periodic in s:
      ! dzetadpsi(1) = dqdpsi_dum 
      ! dzetadpsi(s) = dzetadpsi(s_0) + dum * dqdpsi
      !    with  s=s_0+ dum  and  0<=s_0<1 
      do i = 1, n_s_grid 
        dum = int(i/npol) - nperiod
        if (mod(i,npol).GT.real(npol/2)) then
          dum = dum + 1 
        endif
        metric_G(i,1,2) = metric_G(i,1,2) + &
                      & dum * dqdpsi_dum * metric_G(i,1,1)
        metric_G(i,2,1) = metric_G(i,2,1) + &
                      & dum * dqdpsi_dum * metric_G(i,1,1)
        metric_G(i,2,3) = metric_G(i,2,3) + &
                      & dum * dqdpsi_dum * metric_G(i,1,3)
        metric_G(i,3,2) = metric_G(i,3,2) + &
                      & dum * dqdpsi_dum * metric_G(i,1,3)
        metric_G(i,2,2) = metric_G(i,2,2) + (dum * dqdpsi_dum)**2 * metric_G(i,1,1) + &
                      & 2 * dum * dqdpsi_dum * dzetadpsi(i) * metric_G(i,1,1) + & 
                      & 4 * pi * dum * dqdpsi_dum * dzetadchi(i) * metric_G(i,1,3)
      end do

      ! beta and beta prime (does not have the bn_G dependence)
      beta_real = 2 * 1.25663706144E-6 * p / Bref**2
      beta_prime_real = 2 * 1.25663706144E-6 * dpdeps / Bref**2
  
      ! calculate the array for the parallel derivative (ffun)
      do i = 1, n_s_grid
        ffun(i) = 2*pi*signJ*Rref/bn_G(i)/Bref/jac
      end do
  
      ! the function connected with the trapping terms (gfun)
      do i = 1, n_s_grid 
        gfun(i) = ffun(i)*dBds(i)/bn_G(i)/Bref
      end do 
  
      ! calculate the array connected with the ExB velocity (efun)
      do i = 1, n_s_grid 
        
        ! the diagonal components are zero 
        efun(i,1,1) = 0.
        efun(i,2,2) = 0. 
        efun(i,3,3) = 0. 
        
        ! the psi zeta component 
        efun(i,1,2) =  signJ*pi*Rref**2/bn_G(i)**2/Bref*  &
                    & (metric_G(i,1,1)*metric_G(i,2,2) -  &
                    &  metric_G(i,1,2)**2)*depsdpsi
        
        ! the psi s component 
        efun(i,1,3) =  signJ*pi*Rref**2/bn_G(i)**2/Bref*  &
                    & (metric_G(i,1,1)*metric_G(i,2,3) -  &
                    &  metric_G(i,1,2)*metric_G(i,1,3))*depsdpsi*signB*signJ
        
        ! the zeta s component 
        efun(i,2,3) =  signJ* pi*Rref**2/bn_G(i)**2/Bref* &
                    & (metric_G(i,1,2)*metric_G(i,2,3) -  &
                    &  metric_G(i,2,2)*metric_G(i,1,3))
  
        ! the other components are anti-symmetric_G 
        efun(i,2,1) = - efun(i,1,2) 
        efun(i,3,1) = - efun(i,1,3) 
        efun(i,3,2) = - efun(i,2,3) 
        
      end do 
  
      ! calculate the curvature function  (dfun)
      do i = 1, n_s_grid 
        
        ! the psi component
        dfun(i,1) = -2/bn_G(i)/Bref*efun(i,1,3)*dBds(i)
        
        ! the zeta component 
        dfun(i,2) = -2/bn_G(i)/Bref*(efun(i,2,1)*dBdeps(i) + &
                  &  efun(i,2,3)*dBds(i))

        ! the s component 
        dfun(i,3) = -2/bn_G(i)/Bref*efun(i,3,1)*dBdeps(i)
        
      end do 
  
      ! calculate the array connected with the coriolis drift (hfun)
      ! Omega/vthref=Cte assumed (rigid body)
      ! As VCOR=vtor(R=Rref)/vthref, Omega/vthref=VCOR/Rref
      ! term VCOR=vtor(R=Rref)/vthref not included here (added in linear_terms)
      do i = 1, n_s_grid 
    
        ! the psi component
        hfun(i,1) = - signB*Rref**2/bn_G(i)*(dZdpsi(i)*metric_G(i,1,1) + &
                  & dZds(i)*metric_G(i,3,1))*depsdpsi / Rref
        
        ! the zeta component 
        hfun(i,2) = - signB*Rref**2/bn_G(i)* &
                  & (dZdpsi(i)*metric_G(i,1,2)*signB*signJ + &
                  & dZds(i)*metric_G(i,3,2)*signB*signJ) / Rref
        
        ! the s component 
        hfun(i,3) = - signB*Rref**2/bn_G(i)*(dZdpsi(i)*metric_G(i,1,3) + &
                  & dZds(i)*metric_G(i,3,3) - dZds(i)*ffun(i)**2/Rref**2) &
                  & / Rref
        
      end do 

      ! calculate the array connected with centrifugal drift (ifun)
      ! does not include the (Rref*Omega/vthref)**2 term
      do i = 1, n_s_grid 
  
        ! the psi component
        ifun(i,1) = 2*R_FS(i)*(efun(i,1,3)*dRds(i))/ Rref**2
        
        ! the zeta component 
        ifun(i,2) = 2*R_FS(i)*(efun(i,2,1)*dRdpsi(i)/depsdpsi+ &
                  & efun(i,2,3)*dRds(i))/ Rref**2
        ! the s component 
        ifun(i,3) = 2*R_FS(i)*(efun(i,3,1)*dRdpsi(i)/depsdpsi)/ Rref**2
        
      end do 
  
      ! Calculate the normfactor for k_zeta 
      kthnorm = Rref * sqrt(metric_G((n_s_grid+1)/2,2,2)) 
  
      ! metric_G elements for the (eps,zeta,s) coordinates
      ! all multiplied by Rref**2 for normalisation
      do i=  1, n_s_grid
        metric_G(i,1,1) = metric_G(i,1,1) * depsdpsi**2 * Rref**2
        metric_G(i,1,2) = signB*signJ*metric_G(i,1,2) * depsdpsi * Rref**2
        metric_G(i,2,1) = signB*signJ*metric_G(i,2,1) * depsdpsi * Rref**2
        metric_G(i,1,3) = metric_G(i,1,3) * depsdpsi * Rref**2
        metric_G(i,3,1) = metric_G(i,3,1) * depsdpsi * Rref**2
        metric_G(i,2,2) = metric_G(i,2,2) * Rref**2
        metric_G(i,2,3) = signB*signJ*metric_G(i,2,3) * Rref**2
        metric_G(i,3,2) = signB*signJ*metric_G(i,3,2) * Rref**2
        metric_G(i,3,3) = metric_G(i,3,3) * Rref**2
      end do

      !Compute the toroidal and poloidal fractions of the field
      !CHECK IF THIS IS CORRECT, did not have details of hamada.dat normalisations
      !Done by trial / error, seems right..
      do i=  1, n_s_grid
        !bt_frac should always be positive
        bt_frac(i)=signB*F/(R_FS(i)*Bref*bn_G(i))
        !bt_frac should be negative for negative signJ
        bp_frac(i)=signJ*sqrt(1-bt_frac(i)**2)

        !Some checks
        if(bt_frac(i)<0..or.bt_frac(i)>1) then
          !call gkw_abort('error in bt_frac')
        end if

        if(signJ*bp_frac(i)<0.or.bp_frac(i)>1) then 
           !call gkw_abort('error in bt_frac')
        end if
      end do

      !NOT YET checked, for now revert to s-alpha usage!!!!!
      bt_frac(:)=1.
      bp_frac(:)=signJ*eps/q

    !end if !root processor

    call mpibcast_real(Bref,            1)
    call mpibcast_real(Rref,            1)
    call mpibcast_real(q,               1)
    call mpibcast_real(shat,            1)
    call mpibcast_real(eps,             1)
    call mpibcast_real(bmin,            1)
    call mpibcast_real(bmax,            1)
    call mpibcast_real(kthnorm,         1)
    call mpibcast_real(beta_real,       1)
    call mpibcast_real(beta_prime_real, 1)
    call mpibcast_real(bn_G,            n_s_grid)
    call mpibcast_real(pol_angle,       n_s_grid)
    call mpibcast_real(metric_G,      9*n_s_grid)
    call mpibcast_real(dfun,          3*n_s_grid)
    call mpibcast_real(efun,          9*n_s_grid)
    call mpibcast_real(ffun,            n_s_grid)
    call mpibcast_real(gfun,            n_s_grid)
    call mpibcast_real(hfun,          3*n_s_grid)
    call mpibcast_real(ifun,          3*n_s_grid)
    call mpibcast_real(bt_frac,          n_s_grid)
    call mpibcast_real(bp_frac,          n_s_grid)

  case default   
    ! No known option specified 
    write(*,*)'You specified ',geom_type, & 
    & 'for geom_type in the namelist GEOM'
    write(*,*)'Only known options are: s-alpha, chease' 
    stop 

end select

!call parallelize_geom()
!Now moved to init.

return

  end subroutine geom_init_grids

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine geom_output 
!--------------------------------------------------------------------
! This subroutine writes the equilibrium dependent quantities 
! Must be called before parallelize_geom
!--------------------------------------------------------------------

use grid, only : n_s_grid, parallel_s

implicit none 

integer :: i, igeom
logical :: op

if (.not.root_processor) return 

! Open ASCII file
igeom = 20
do
   inquire(unit=igeom,opened=op)
   if (.NOT.op) exit
   igeom = igeom+1
end do
open(igeom, file='geom.dat')
rewind igeom

! scalar quantities
write(igeom,'(A)') 'NS'
write(igeom,'(I4)') n_s_grid
write(igeom,'(A)') 'Rref'
write(igeom,'(1PE20.10)') Rref
write(igeom,'(A)') 'Bref'
write(igeom,'(1PE20.10)') Bref
write(igeom,'(A)') 'eps'
write(igeom,'(1PE20.10)') eps
write(igeom,'(A)') 'q'
write(igeom,'(1PE20.10)') q
write(igeom,'(A)') 'shat'
write(igeom,'(1PE20.10)') shat
write(igeom,'(A)') 'bmin'
write(igeom,'(1PE20.10)') bmin
write(igeom,'(A)') 'bmax'
write(igeom,'(1PE20.10)') bmax
write(igeom,'(A)') 'kthnorm' ! = Rref*sqrt(metric((n_s_grid+1)/2,2,2))
write(igeom,'(1PE20.10)') kthnorm
write(igeom,'(A)') 'beta_real' ! = 2*mu0*p/Bref**2
write(igeom,'(1PE20.10)') beta_real
write(igeom,'(A)') 'beta_prime_real' ! = 2*mu0*dpdeps/Bref**2
write(igeom,'(1PE20.10)') beta_prime_real

! s-dependent quantities
write(igeom,'(A)') 's_grid'
write(igeom,'(1P5E20.10)') (sgr(1,i),i=1, n_s_grid)
write(igeom,'(A)') 'bn' ! = B/Bref
write(igeom,'(1P5E20.10)') (bn_G(i),i=1, n_s_grid)
write(igeom,'(A)') 'poloidal_angle'
write(igeom,'(1P5E20.10)') (pol_angle(i),i=1, n_s_grid)
write(igeom,'(A)') 'g_eps_eps'
write(igeom,'(1P5E20.10)') (metric_G(i,1,1),i=1, n_s_grid)
write(igeom,'(A)') 'g_eps_zeta'
write(igeom,'(1P5E20.10)') (metric_G(i,1,2),i=1, n_s_grid)
write(igeom,'(A)') 'g_eps_s'
write(igeom,'(1P5E20.10)') (metric_G(i,1,3),i=1, n_s_grid)
write(igeom,'(A)') 'g_zeta_zeta'
write(igeom,'(1P5E20.10)') (metric_G(i,2,2),i=1, n_s_grid)
write(igeom,'(A)') 'g_zeta_s'
write(igeom,'(1P5E20.10)') (metric_G(i,2,3),i=1, n_s_grid)
write(igeom,'(A)') 'g_s_s'
write(igeom,'(1P5E20.10)') (metric_G(i,3,3),i=1, n_s_grid)
write(igeom,'(A)') 'D_eps'
write(igeom,'(1P5E20.10)') (dfun(i,1),i=1, n_s_grid)
write(igeom,'(A)') 'D_zeta'
write(igeom,'(1P5E20.10)') (dfun(i,2),i=1, n_s_grid)
write(igeom,'(A)') 'D_s'
write(igeom,'(1P5E20.10)') (dfun(i,3),i=1, n_s_grid)
write(igeom,'(A)') 'E_eps_zeta'
write(igeom,'(1P5E20.10)') (efun(i,1,2),i=1, n_s_grid)
write(igeom,'(A)') 'E_eps_s'
write(igeom,'(1P5E20.10)') (efun(i,1,3),i=1, n_s_grid)
write(igeom,'(A)') 'E_zeta_s'
write(igeom,'(1P5E20.10)') (efun(i,2,3),i=1, n_s_grid)
write(igeom,'(A)') 'F'
write(igeom,'(1P5E20.10)') (ffun(i),i=1, n_s_grid)
write(igeom,'(A)') 'G'
write(igeom,'(1P5E20.10)') (gfun(i),i=1, n_s_grid)
write(igeom,'(A)') 'H_eps'
write(igeom,'(1P5E20.10)') (hfun(i,1),i=1, n_s_grid)
write(igeom,'(A)') 'H_zeta'
write(igeom,'(1P5E20.10)') (hfun(i,2),i=1, n_s_grid)
write(igeom,'(A)') 'H_s'
write(igeom,'(1P5E20.10)') (hfun(i,3),i=1, n_s_grid)
write(igeom,'(A)') 'I_eps'
write(igeom,'(1P5E20.10)') (ifun(i,1),i=1, n_s_grid)
write(igeom,'(A)') 'I_zeta'
write(igeom,'(1P5E20.10)') (ifun(i,2),i=1, n_s_grid)
write(igeom,'(A)') 'I_s'
write(igeom,'(1P5E20.10)') (ifun(i,3),i=1, n_s_grid)
write(igeom,'(A)') 'Bt_frac'
write(igeom,'(1P5E20.10)') (Bt_frac(i),i=1, n_s_grid)
write(igeom,'(A)') 'Bp_frac'
write(igeom,'(1P5E20.10)') (Bp_frac(i),i=1, n_s_grid)

close(igeom)

return
end subroutine geom_output

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


!--------------------------------------------------------------------
!> This subroutine calculated the parallel derivative of log(B) 
!> it is assumed that the field line grid is periodic in the 
!> poloidal direction 
!--------------------------------------------------------------------
subroutine logbderiv(deriv)

use grid,    only : n_s_grid 

implicit none 

real deriv(n_s_grid)

integer i 
real lbm2, lbm1, lbp1, lbp2 

if (n_s_grid.lt.2) then 

  deriv(1) = 0.

else 

  do i = 1, n_s_grid 
    if (i-2.lt.1) then 
      lbm2 = bn_G(n_s_grid+i-2)  
    else 
      lbm2 = bn_G(i-2)
    endif 
    if (i-1.lt.1) then 
      lbm1 = bn_G(n_s_grid+i-1)  
    else 
      lbm1 = bn_G(i-1)
    endif 
    if (i+1.gt.n_s_grid) then 
      lbp1 = bn_G(i+1-n_s_grid)  
    else 
      lbp1 = bn_G(i+1)
    endif 
    if (i+2.gt.n_s_grid) then 
      lbp2 = bn_G(i+2-n_s_grid)  
    else 
      lbp2 = bn_G(i+2)
    endif 
 
    deriv(i) = ffun(i)*(lbm2-8E0*lbm1+8E0*lbp1-lbp2)/ &
             & (12E0*sgr_dist) / bn_G(i)    

    ! write(*,*)i,deriv(i)
  end do 
endif 

return 
end subroutine logbderiv 

!--------------------------------------------------------------------------
!> In order to solve the local problem in s, copy the local elements to
!> the begining of each array used elsewhere in the code. The alternatives
!> are to either change this module to work with ns or to change all the
!> other modules using this to use nspb/e and friends.
!> geom_output and krbal must be called before this.
subroutine parallelize_geom
!--------------------------------------------------------------------------
use grid, only : ns, n_s_grid, ispb, parallel_s

integer :: i1, i2, ib, ie


! for parallel_s we need to deal with extra points first
if (.true.) then

  ib = 1-2 ; ie = ns+2

  do i1=ib, 1-1
    i2 = i1 + ispb - 1
    if (i2 .lt. 1) then
      if (parallel_s) then
         metric(i1,:,:)=metric_G(i2+n_s_grid,:,:)
      end if
      bn(i1)=bn_G(i2+n_s_grid)
    else
      if (parallel_s) then
         metric(i1,:,:)=metric_G(i2,:,:)
      end if
      bn(i1)=bn_G(i2)
    endif
  enddo

  do i1=ns+1, ie
    i2 = i1 + ispb - 1
    if (i2 .gt. n_s_grid) then
      if (parallel_s) then
         metric(i1,:,:)=metric_G(i2-n_s_grid,:,:)
      end if
      bn(i1)=bn_G(i2-n_s_grid)
    else
      if (parallel_s) then
         metric(i1,:,:)=metric_G(i2,:,:)
      end if
      bn(i1)=bn_G(i2)
    endif
  enddo

endif


! all cases can use the following
do i1=1, ns
  i2 = i1 + ispb - 1
  pol_angle(i1)=pol_angle(i2)
  ints(:,i1)=ints(:,i2)
  dfun(i1,:)=dfun(i2,:)
  efun(i1,:,:)=efun(i2,:,:)
  ffun(i1)=ffun(i2)
  gfun(i1)=gfun(i2)
  hfun(i1,:)=hfun(i2,:)
  ifun(i1,:)=ifun(i2,:)
  metric(i1,:,:)=metric_G(i2,:,:)
  bn(i1)=bn_G(i2)
  bt_frac(i1)=bt_frac(i2)
  bp_frac(i1)=bp_frac(i2)
enddo

do i1=0, ns+1
  i2 = i1 + ispb - 1
  sgr(:,i1)=sgr(:,i2)
enddo


end subroutine parallelize_geom

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

end module geom
