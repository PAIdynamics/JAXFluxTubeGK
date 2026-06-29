module mode 
!--------------------------------------------------------------------
! SVN:$Id: mode.F90 1005 2009-07-02 16:12:03Z  $
! Module contains most of the information on the modes kept in the 
! simulation 
!--------------------------------------------------------------------
use mpiinterface
use general
use global

implicit none 

private 

  public :: mode_read_nml, mode_write_nml, mode_bcast_nml
  public :: mode_check_params, krbal, mode_box_recon
    
public :: ixminus, ixplus, ixzero, kgrid, krho, krloc 
public :: kxrh, kxspace, ikxspace, kxmax, kymax, lx, ly, lxinv, lyinv 
public :: mode_allocate, mode_box, mode_init
! 
  interface mode_write_nml
    module procedure mode_read_nml
  end interface
!> The array of 'toroidal' (zeta) wave vectors. krho(nmod)
!> These wavevectors are perpendicular to the field line
!> in the plane of the flux surface.  confusion can arise because
!> they are called both 'toroidal' and 'poloidal' at various points!
!> This direction is also often referred to as the y direction
real, allocatable :: krho(:) 

!> The perpendicular wave vector (as a function along the field line)
real, allocatable :: krloc(:,:,:)

!> The array of radial (psi) wave vectors. kxrh(nx)
real, allocatable :: kxrh(:)

!> Integers that determine to which kx mode the mode is connected 
!! over the parallel boundary conditions. ixplus(nmod,nx)
integer, allocatable :: ixplus(:,:)
integer, allocatable :: ixminus(:,:)
!< Integers that determine to which kx mode the mode is connected 
!! over the parallel boundary conditions  ixminus(nmod,nx)

!> The poloidal shift of the ballooning transform. 
real chin 

!> logical that determines if there is a 2D grid of ky,kx 
!! Use with nperiod = 1 (necessary for non linear runs)
logical mode_box 

!> logical for treating the special case when shat=0
logical :: lshat_zero=.false.

!> for mode_box, the maximum ky used in the simulations 
real krhomax 

!> Spacing between kxmodes
real kxspace

!> for mode_box, the integer that determines the spacing between 
!! the different kx modes 
integer ikxspace

!>The location of the kx=0 mode.  Needed by the FFT in nonlinear terms and rotation
integer ixzero 

!>The size of the box in real space
real lx, ly, lxinv, lyinv

!>maximum values of the kgrid.
real kxmax, kymax

integer, parameter :: nmmx = 512
real, dimension(nmmx) :: kthrho

contains 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> read (or write) mode nml
!----------------------------------------------------------------------------

subroutine mode_read_nml(ilun,io_stat,lwrite)

  use grid, only : nmod

  integer, intent(in)  :: ilun
  integer, intent(out) :: io_stat

  logical, optional, intent(in) :: lwrite

  namelist /mode/ kthrho,   & ! the poloidal wave vector times rho
                & chin,     & ! poloidal angle shift 
                & mode_box, & ! true if a 2D grid of modes is used 
                & krhomax,  & ! maximum krho used in 2D box 
                & ikxspace    ! the spacing of the kx modes 

  io_stat = 0
  
  if (present(lwrite)) then
    if (.not. lwrite) then
      
      ! test of nx is not too large 
      if (nmod > nmmx) then
        call gkw_abort('nmod > nmmx in mode.f90. Reset nmmx and recompile')
      end if
      
      ! Set default values 
      chin = 0.
      kthrho = 0.
      mode_box = .false.
      krhomax = 0.
      ikxspace = 0
      read(ilun,NML=mode,IOSTAT=io_stat)
    else
      ! do nothing
    end if
  else
    write(ilun,NML=mode)
  end if

end subroutine mode_read_nml

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> broadcast the input parameters for mode to other processors
!----------------------------------------------------------------------------

subroutine mode_bcast_nml

  use mpiinterface, only : mpibcast_real, mpibcast_logical, mpibcast_integer
  use grid, only : nmod

  call mpibcast_real(kthrho, nmod)
  call mpibcast_real(chin,      1)
  call mpibcast_logical(mode_box,  1)
  call mpibcast_real(krhomax,   1)
  call mpibcast_integer(ikxspace,  1)

end subroutine mode_bcast_nml

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> check the parameters for mode
!> In the case of chease this routine may be called twice.
!----------------------------------------------------------------------------
  
subroutine mode_check_params(icall)
 
  use fft,     only : working_fft_library
  use grid,    only : nmod, nx, nperiod
  use control, only : non_linear
  use global,  only : r_tiny
  use geom,    only : shat

  integer, intent(in) :: icall                
  integer :: imod 
                
  
  ! do a few checks on the given input 
  if (mode_box) then
    if (.not. working_fft_library) then
      call gkw_abort('mode_box requires a working FFT library')
    end if
    if (ikxspace <= 0.and.abs(shat)>r_tiny) then 
      call gkw_abort('Unreasonable value of ikxspace')
    end if
    if (nperiod /= 1) call gkw_abort('Use nperiod = 1 with mode_box = true')
    if (root_processor.and.icall==1) then
      write(*,*)
      write(*,*) 'With mode_box input value(s) of kthrho are ignored'
      write(*,*)
    end if
    
    !Do some checks on value of shat with mode_box
    !These are repeated in case chease changed shat
    if(abs(shat) < r_tiny) then  !The zero shear case is treated differently
      lshat_zero=.true.
      !Write actual values to file input.out
      shat=0.
      if (ikxspace.ne.0) then 
           call gkw_warn('ikxspace not used with shat=0')
           ikxspace=0
      end if
      if(root_processor) write(*,*) 'zero shear case selected'
    else if (abs(shat) < 0.05) then !Near zero shear not implemented
      call gkw_abort('Magnetic shear must be exactly zero, & 
                      & case close to zero not treated')
    else if (abs(shat) < 0.1) then
      call gkw_warn('Small magnetic shear requires very large nx for mode_box')
    else
      !In the "normal" case, do nothing
    end if

    !kthrho not used, say so in input.out
    kthrho=0.

  else !not mode_box
    if (non_linear) call gkw_abort('mode_box must be true for nonlinear runs')

    !If mode_box=false, all kxrh=0, since kgrid returns
    if (nx.gt.1) call gkw_abort('nx=1 should be used for mode_box=.false.')

    if (ikxspace.ne.0) then
       call gkw_warn('ikxspace not used with mode_box=.false.')
       ikxspace=0
    end if

    if (root_processor.and.icall==1) then
      write(*,*)
      write(*,*) 'With mode_box off input value of krhomax is ignored'
      write(*,*) 'No 2D diagnostics will be written'
      write(*,*) 'Ensure you have ', nmod, ' value(s) in kthrho list'
      write(*,*)
    end if
    
    !krhomax not used, say so in input.out
    krhomax=0.

  end if !mode_box

end subroutine mode_check_params

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!--------------------------------------------------------------------
!> Allocation of the arrays for mode 
!--------------------------------------------------------------------
subroutine mode_allocate 

use grid, only : nmod, nx, ns, parallel_s 

integer ierr 

! initialize the error integer 
ierr = 0 

  ! allocate the krloc array 
  if (parallel_s) then
    allocate(krloc(nmod,nx,-1:ns+2), stat=ierr)
  else
    allocate(krloc(nmod,nx,ns), stat=ierr)
  end if
  if (ierr.ne.0) then
    stop 'Could not allocate krloc in mode'
  end if
             
! allocate the krho array 
allocate(krho(nmod),stat=ierr)
if (ierr.ne.0) then 
  stop 'Could not allocate krho in mode'
endif
! allocate the kxrh array 
allocate(kxrh(nx),stat=ierr)
if (ierr.ne.0) then 
  stop 'Could not allocate kxrh in mode'
endif
! allocate the array with the integers for 
! the parallel boundary conditions  
allocate(ixplus(nmod,nx), stat=ierr) 
if (ierr.ne.0) then
  stop 'Could not allocate ixplus in mode' 
endif
! allocate the array with the integers for 
! the parallel boundary conditions  
allocate(ixminus(nmod,nx), stat=ierr)
if (ierr.ne.0) then
  stop 'Could not allocate ixminus in mode'
endif

return 
end subroutine mode_allocate 


!-------------------------------------------------------------------
!> This subroutine initializes the parameters of mode 
!> this subroutine could be merged with another, perhaps kgrid.
!-------------------------------------------------------------------

subroutine mode_init 

  use grid, only : nmod
  
  integer :: imod

  do imod = 1, nmod 
    krho(imod) = kthrho(imod)
  end do 

end subroutine mode_init 

!-----------------------------------------------------------------------
!> This subroutine calculates the grids for the 2D case 
!> it also determines the integers necessary for the parallel 
!> boundary conditions.   Must be called after geom_init_grids
!> but before parallelize_geom!
!-----------------------------------------------------------------------
subroutine kgrid 

use grid, only : nmod, nx 
use geom, only : q, eps, shat, kthnorm
use constants, only : pi

integer imod, ix, itemp, i   
real kxmax, kx, ky, kxplus, kxminus, kxhalf, dum,    &
     & kxmin      

!Check that shat is correctly intialised
if(shat>1.22e4) then
  call gkw_abort('shat not correctly intialised.  &
       & geom_init_grids must be called before kgrid for geom type chease')
end if
!Check that q is correctly intialised
if(q>1.22e4) then
  call gkw_abort('shat not correctly intialised.  &
       & geom_init_grids must be called before kgrid for geom type chease')
end if
!Check that kthnorm is correctly intialised
if(kthnorm>1.22e4) then
  call gkw_abort('kthnorm not correctly intialised.  &
       & geom_init_grids must be called before kgrid')
end if


! initialize 
ixplus = 0
ixminus = 0
kxrh = 0.E0
ixzero=0

! For mode_box = .false. the only function of this routine is to 
! properly normalise the wave vectors given in input 
if (.not.mode_box) then 
  do imod = 1, nmod 
    krho(imod) = krho(imod) / kthnorm 
    kxrh(1)    = chin* abs(q*shat*krho(imod)) / (2*pi*eps)
    ixzero = 1 !used in tearing modes
  end do
  return  
endif 

! Without the line below can lead to Severe internal error: ixplus below
! if(mode_box.and.mod(nx,2).eq.0) then
!   call gkw_abort ('Mode box requires nx to be odd for connect parallel')
! end if

! Calculate the 'toroidal' wave numbers. Note only nmod modes are
! used in the time intgration
if (nmod.gt.1) then 
  do imod = 1, nmod   
    krho(imod) = krhomax*(imod-1)/real(nmod-1)/kthnorm 
  end do 
else 
  krho(1) = krhomax/kthnorm
endif 

if (nmod.gt.1) then 
  !In order not to reverse the mode ordering need abs
  !in theory shat<0 is the only part that could be negative
  kxspace = abs(q*shat*krho(2) / (eps*real(ikxspace)))
  if (lshat_zero) then !Do a square box kxspace=kyspace
     kxspace=krho(2)*kthnorm
  endif
else 
  !kxspace = q*shat*krho(1) / (eps*real(ikxspace)) 
  kxspace = abs(q*shat*krho(1) / (eps*real(ikxspace)))
  if (lshat_zero) call gkw_abort('shat=0 not implemented for nmod=1')
endif


kxhalf = kxspace / 2.E0
 
! Determine the kx modes 
! again not all these modes are used in time integration 
if (mod(nx,2).eq.0) then 
  !Not ever used for nonlinear runs
  kxrh(1) = - real(nx)*kxspace / 2.E0
  !Possibility that the ordering of the modes for FFT should use this
  !Check against the indexing array jind
  !only test cases affected: shear_lin, modebox_freq
  !kxrh(1) = - real(nx)*kxspace / 2.E0 + kxspace

else 
  kxrh(1) = - real(nx-1)*kxspace / 2.E0
endif 
do ix = 2, nx
  kxrh(ix) = kxrh(ix-1) + kxspace
  if (kxrh(ix).lt.1e-10) ixzero=ix
end do 
! kxmax is the maximum kx used in the time integration. 
kxmax = kxrh(nx)
kxmin = kxrh(1)

if (ixzero.eq.0) call gkw_warn('No kx = 0 mode')

! Calculate the box size in real space
! warning still need to look at nmod = 1 case 
lx = 1. 
ly = 1. 
if (nmod.gt.1) then 
  ly = 2.*pi/krho(2)
  lx = 2.*pi/kxrh(ixzero+1) 
else 
endif 

lxinv = 1./lx
lyinv = 1./ly

!This line renormalises ly to put it in the same rho_ref units as lx
ly = ly*2.*pi*eps/q
!This is normalisation BACKWARDS from code units to units of rho_ref
!Equivalent to ly=ly/kthnorm

if (lverbose) write(*,*)
if (lverbose) write(*,*) 'Box size / rho_ref: lx', lx, 'ly', ly
if (lverbose) write(*,*)

kxmax = 0 
do ix = 1, nx 
  kxmax = max(kxmax, abs(kxrh(ix)))
end do
kymax = 0  
do imod = 1, nmod 
  kymax = max(kymax, abs(krho(imod)))
end do 


! Make the integer connections for the parallel boundary 
! conditions 
do imod = 1, nmod 
  do ix = 1, nx 

    ky = krho(imod) 
    kx = kxrh(ix)
     
    ! ky = 0 mode is always treated differently
    if (abs(ky).gt.1e-10) then 
 
      ! kx value after one poloidal turn 
       kxplus = kx + abs(q*shat*ky/eps)
    
      if (kxplus.gt.kxmax + kxhalf) then 
     
        ixplus(imod,ix) = 0 
    
      else 
     
        ! inefficient programming, but very general ... 
        i = 1 
        do while (abs(kxplus-kxrh(i)).gt.0.5*kxhalf) 
          i = i + 1 
          if (i.gt.nx) then 
            write(*,*)'Severe internal error: ixplus'
            stop
          endif
        end do 
        ixplus(imod,ix) = i 

      endif 
    
      kxminus = kx - abs(q*shat*ky/eps)
     
      if (kxminus.lt.kxmin - kxhalf) then 
     
        ixminus(imod,ix) = 0 
    
      else 

       ! inefficient programming, but very general ... 
        i = 1 
        do while (abs(kxminus-kxrh(i)).gt.0.5*kxhalf) 
          i = i + 1 
          if (i.gt.nx) then 
            write(*,*)'Severe internal error: ixminus'
            stop
          endif
        end do 
        ixminus(imod,ix) = i 
         
      endif 

    !Swap the connections over for shat < 0.
    !In theory q and eps should always be > 0
    if (q*shat*ky/eps.lt.0) then
      dum=ixminus(imod,ix)
      ixminus(imod,ix)=ixplus(imod,ix)
      ixplus(imod,ix)=dum
    end if

    else 
    
     ! the ky = 0 mode is always periodic
     ixminus(imod,ix) = ix 
     ixplus(imod,ix) = ix 
      
    endif 
        
  end do 
end do 

!Write the kx connections to file
if (root_processor) then
    open(10,file = "kx_connect.dat")
    22 format('shat: ',f6.2,2x,'q: ',f6.2,2x,'eps: ',f6.2,2x,'ikxspace: ', i4)
    23 format(i6,f10.6,i6,i6,i6)
    write(10,22) shat, q, eps, ikxspace 
    write(10,*) 'imod , kxrh(ix), ix,  ixplus(imod,ix),   ixminus(imod,ix)'
    do imod = 1, nmod 
      do ix = 1, nx 
        write(10,23) imod, kxrh(ix),ix, ixplus(imod,ix),ixminus(imod,ix)
      end do 
      !write(10,*) '---------------------------------'
    end do
    close(10)
end if

return 
end subroutine kgrid 

!-----------------------------------------------------------------------
!> This subroutine calculates some quantities that are necessary for 
!> the ballooning transform. gfun is related to the convection  due 
!> to the drift velocity. krloc is the local perpendicular wave 
!> vector.  This routine must be called after geom_init_grids and kgrid
!-----------------------------------------------------------------------
subroutine krbal

  use grid, only : nx,ns,nmod,parallel_s
  use geom, only : metric

  integer :: ix,i,imod,i1,i2

  if (parallel_s) then
    i1= -1 ; i2 = ns+2
  else
    i1=  1 ; i2 = ns
  end if
  
  ! Calculate the peperpendicular wave vector
  do imod = 1, nmod 
    do ix = 1, nx 
      do i = i1, i2
        krloc(imod,ix,i) = krho(imod)**2*metric(i,2,2) + 2.E0*krho(imod)* &
                         & kxrh(ix)*metric(i,1,2) + kxrh(ix)**2*metric(i,1,1) 
        krloc(imod,ix,i) = sqrt(krloc(imod,ix,i))
      end do 
    end do 
  end do
  
end subroutine krbal

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine mode_box_recon 
!--------------------------------------------------------------------
! Small routine that allows one to plot quantities along the field 
! line when mode_box = true is used. At pressent puts on file  
! krloc_G (the perpendicular wave number), the curvature, and the 
! coriolis operator 
! Must be called before parallelize geom shifts the tensors in s
!--------------------------------------------------------------------
use grid,    only : nmod, nx, n_s_grid
use geom, only : metric_G, dfun, hfun, sgr
use constants, only : pi

implicit none 

integer imod, i, it, ixref, ix
real kxmin, curv, cori, krloc_G  

if (.not. mode_box) return
if (.not. root_processor) return

!Find the minimum kx
!Could belong in mode, as ixzero now does.
kxmin = kxrh(1)
it = 1 
do i = 2, nx 
  if (kxrh(i).lt.kxmin) then 
    kxmin = kxrh(i)
    it = i 
  end if 
end do 

open(13,file = 'par.dat') 
do imod = 1, nmod 
  write(13,1)imod 
  1 format('The toroidal mode ',I4) 
  write(13,2)
  2 format(' perpend. k-vec  curvature        coriliolis')   

  ! start at minimum kx 
  ixref = it
  10 continue 
    do i = 1, n_s_grid 
      curv = dfun(i,1)*kxrh(ixref) + dfun(i,2)*krho(imod)
      cori = hfun(i,1)*kxrh(ixref) + hfun(i,2)*krho(imod) 
      krloc_G = krho(imod)**2*metric_G(i,2,2) + 2.E0*krho(imod)* &
                       & kxrh(ixref)*metric_G(i,1,2) + kxrh(ixref)**2*metric_G(i,1,1)
      krloc_G = sqrt(krloc_G)
      write(13,20)krloc_G,curv,cori 
      20 format(30(1pe16.8,1X))
    end do 
  if (ixplus(imod,ixref) /= 0) then 
    if (ixref /= ixplus(imod,ixref)) then
      ixref = ixplus(imod,ixref) 
      goto 10
    end if  
  end if
end do  
close(13)

!Curvature and kperp function for all modes.
open(13,file = "parfun.dat")
  do imod = 1, nmod ; do ix = 1, nx ; do i = 1, n_s_grid 
      krloc_G = krho(imod)**2*metric_G(i,2,2) + 2.E0*krho(imod)* &
                       & kxrh(ix)*metric_G(i,1,2) + kxrh(ix)**2*metric_G(i,1,1)
      krloc_G = sqrt(krloc_G)
      write(13,fmt = '(11(1pe13.5,1X))')sgr(ix,i),   &
                    & krloc_G
  end do ; end do ; end do  
close(13)

! open(13,file = 'lxly.dat') 
!   write(13,*) 's_point, kxmin, kymin, lx_perp, ly_perp, lx, ly'
!   do i = 1, n_s_grid 
!     write(13,21) i, krloc_G(1,ixzero+1,i), krloc_G(2,ixzero,i), 2.*pi/krloc_G(1,ixzero+1,i), 2.*pi/krloc_G(2,ixzero,i), lx, ly
!     21 format(i,30(1pe12.4,1X))
! 
!   end do
! close(13)

return 
end subroutine mode_box_recon 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 
end module mode
