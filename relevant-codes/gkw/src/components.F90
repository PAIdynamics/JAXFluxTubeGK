module components
! SVN:$Id: components.F90 1009 2009-07-02 17:08:44Z  $ 
use mpiinterface
use general, only : gkw_warn, gkw_abort
use global

implicit none

private

  public :: components_read_nml_spcg, components_bcast_nml_spcg
  public :: components_check_params_spcg, components_write_nml_spcg
  public :: components_read_nml_spec, components_bcast_nml_spec
  public :: components_check_params_spec, components_write_nml_spec
  public :: components_input_species

public :: adiabatic_electrons, amp_init
public :: components_allocate, de, finit
public :: fp, mas, pbg,rhorat, signz
public :: tmp, tp, types, vp, vthrat, beta, beta_prime_components

  interface components_write_nml_spcg
    module procedure components_read_nml_spcg
  end interface

  interface components_write_nml_spec
    module procedure components_read_nml_spec
  end interface
  
!> the ratio of the mass to the reference value 
real, allocatable :: mas(:)

!> The charge number of the particles (Z)
real, allocatable :: signz(:)

!> The ratio of the temperature to the reference temperature 
real, allocatable :: tmp(:)

!> The density of the species 
real, allocatable :: de(:)

!> The temperature gradient R/LT 
real, allocatable :: tp(:)

!> The density gradient of the species R/Ln 
real, allocatable :: fp(:)

!> The gradient of the toroidal velocity R grad U / v_thref
!> Or in the case of toroidal_shear contains the unscaled shear_rate
real, allocatable :: vp(:)

!> the type of the species (only necessary for non-maxwellian 
character (len=7), allocatable :: types(:)

!> A parameter that can be used to define the background (only 
!> necessary for non-maxwelian backgrounds 
real, allocatable :: pbg(:)

!> The ratio of the thermal velocity to the reference value 
real, allocatable :: vthrat(:)

!> The ratio of the Larmor radius and the reference value 
real, allocatable :: rhorat(:)

!> The plasma beta 
real :: beta 

!> The beta_prime calculated from the per species gradients.
real :: beta_prime_components 

!> Logical that is true if the electrons are adiabatic 
logical :: adiabatic_electrons

!> The initial amplitude of the distribution function 
real :: amp_init 

!> This charater string determines how the distribution is inintialez
!> allowed are 'noise' and 'cosine' 
character (len = 6) :: finit 

  ! for reading species namelists
  real :: mass, z, temp, dens, rlt, rln, uprim, param
  character (len=7) :: background

contains

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> read (or write) the species general namelist
!----------------------------------------------------------------------------

subroutine components_read_nml_spcg(ilun,io_stat,lwrite)

  integer, intent(in)  :: ilun
  integer, intent(out) :: io_stat

  logical, optional, intent(in) :: lwrite

  namelist /spcgeneral/ beta, adiabatic_electrons, amp_init, finit

  io_stat = 0

  if (present(lwrite)) then
    if (.not. lwrite) then
    
      ! Set the default values
      adiabatic_electrons = .false.
  
      ! zero beta is the default. Note that beta can affect electro-
      ! static runs because of the finite beta contribution to the 
      ! drift 
      beta = 0.E0

      ! The initial amplitude 
      amp_init = 1e-3 

      ! The choise of initialization 
      finit = 'cosine'

      ! read the general parameters 
      read(ilun,NML=spcgeneral,IOSTAT=io_stat)

    else
      ! do nothing
    end if
  else
    write(ilun,NML=spcgeneral)
  end if

end subroutine components_read_nml_spcg

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> broadcast species general namelist
!----------------------------------------------------------------------------

subroutine components_bcast_nml_spcg

  call mpibcast_real(beta,                  1)
  call mpibcast_real(amp_init,              1)
  call mpibcast_character(finit,            6)
  call mpibcast_logical(adiabatic_electrons,1)

  ! Note that this beta is used in the Ampere's law
  ! and is defined as beta=2*mu0*n_ref*T_ref/B_ref^2
  ! It is different from beta_real=2*mu0*p/B_ref^2 that is
  ! calculated in geom.f90

end subroutine components_bcast_nml_spcg

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> check species general params
!----------------------------------------------------------------------------

subroutine components_check_params_spcg
use control, only : zonal_adiabatic, nlapar

  if(beta < 0.) then
    call gkw_abort('spcgeneral: I do not understand negative beta')
  end if

  !Zonal_adiabatic is meaningless for kinetic eletrons
  if (.not.adiabatic_electrons .and. zonal_adiabatic) then
    call gkw_warn('spcgeneral: zonal_adiabatic always false for kinetic electrons')
    zonal_adiabatic=.false.
  endif

  !Ignore tiny beta
  if(beta < r_tiny.and.nlapar) then
    call gkw_warn('spcgeneral: beta=0 input, this run is electrostatic')
    nlapar=.false.
  end if

  if(beta > r_tiny.and..not.nlapar) then
    call gkw_warn('spcgeneral_check: input value of beta ignored for electrostatic run')
    beta=0.
  end if

end subroutine components_check_params_spcg

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> read (or write) 1 species namelist.  Called by one processor
!> Only for checking purposes.  The data is read a second time 
!> in components_input_species
!----------------------------------------------------------------------------

subroutine components_read_nml_spec(ilun,io_stat,lwrite)

  integer, intent(in)  :: ilun
  integer, intent(out) :: io_stat

  logical, optional, intent(in) :: lwrite

  namelist /species/ mass, z, temp, dens, rlt, rln, uprim, background, param

  io_stat = 0
  
  if (present(lwrite)) then
    if (.not. lwrite) then
  
      ! initialize the type to be a Maxwellian 
      background = 'maxwell'

      ! a parameter that can be used to define the background 
      param = 0.

      read (ilun,NML=species,IOSTAT=io_stat)

    else
      ! do nothing
    end if
  else
    write (ilun,NML=species)
  end if

end subroutine components_read_nml_spec

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> broadcast 1 species namelist
!----------------------------------------------------------------------------

subroutine components_bcast_nml_spec

  call mpibcast_real(mass,1)
  call mpibcast_real(temp,1)
  call mpibcast_real(dens,1)
  call mpibcast_real(z,1)
  call mpibcast_real(rln,1)
  call mpibcast_real(rlt,1)
  call mpibcast_real(uprim,1)
  call mpibcast_character(background,7)
  call mpibcast_real(param,1)

end subroutine components_bcast_nml_spec

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> check species namelist; called until all species are read
!> called by all processors after all species data has been broadcast
!----------------------------------------------------------------------------

subroutine components_check_params_spec

  use general, only : gkw_abort
  use grid, only : number_of_species
  use rotation, only : toroidal_shear, shear_rate

  real, save :: ntot = 0., nqas = 0., ngrad = 0.
  integer, save :: call_count = 0, nsps = 0

  if (call_count == 0) then

    nsps = number_of_species
    if (adiabatic_electrons) nsps = nsps + 1

    ! set the parameters for the check on quasi-neutrality 
    ntot  = 0.
    nqas  = 0.
    ngrad = 0.
    
    ! initialize beta_prime as calculated from components
    beta_prime_components = 0.
    
  end if

  call_count = call_count + 1
  if (call_count > nsps) then
    call gkw_abort('components_check_species called to many times')
  end if


  ! To check quasi-neutrality 
  ntot  = ntot  +   dens
  nqas  = nqas  + z*dens
  ngrad = ngrad + z*dens*rln
  
  ! calculate beta_prime
  ! Note beta is currently always set to zero for an electrostatic run.
  beta_prime_components = beta_prime_components + beta*dens*temp*(rln+rlt)

  last_species : if (call_count == nsps) then
    ! Stop if quasi-neutrality is not satisfied
    if (abs(nqas) > 1e-6*abs(ntot)) then
      write(*,*)'sum z_s * n_s ',nqas,nsps
      call gkw_abort('You do not satisfy quasi-neutrality')
    end if
    if (abs(ngrad) > 1e-6*abs(ntot)) then
      call gkw_abort('You do not satisfy quasi-neutrality for the gradients')
    end if
  end if last_species

  if (toroidal_shear) then !override input uprim
       uprim=shear_rate
       !Note this is done before the rescaling of shear_rate in shear_init
       !Will also be rescaled in linear terms by bp_frac
       if(root_processor) then
          write(*,*)
          write(*,*) 'Toroidal_shear: uprim overridden' 
          write(*,*)
       end if
  end if

end subroutine components_check_params_spec




!-------------------------------------------------------------------
!> This subroutine allocates the arrays of the module components 
!
! mas(nsp+iadia)     the normalised mass 
! signz(nsp+iadia)   the charge number 
! tmp(nsp+iadia)     the normalised temperature 
! de(nsp+iadia)      the normalised density 
! tp(nsp+iadai)      the temperature gradient 
! fp(nsp+iadia)      the density gradient 
! vp(nsp+iadai)      the parallel velocity gradient
! vthrat(nsp)        the ratio of the thermal / reference velocity
! rhorat(nsp)        the ratio of the Larmor radius 
! 
!--------------------------------------------------------------------
subroutine components_allocate

use control, only : collisions
use grid,    only : nsp 
use dist,    only : iadia

implicit none

integer ierr

! initilize the error parameter
ierr = 0

! allocate the mass array
allocate(mas(nsp+iadia),stat=ierr)
if (ierr.ne.0) then
  stop 'Could not allocate mas in components'
endif

! allocate the charge array
allocate(signz(nsp+iadia),stat=ierr)
if (ierr.ne.0) then
  stop 'Could not allocate signz in components'
endif

! allocate the temperature array
allocate(tmp(nsp+iadia),stat=ierr)
if (ierr.ne.0) then
  stop 'Could not allocate tmp in components'
endif

! allocate the density array
allocate(de(nsp+iadia),stat=ierr)
if (ierr.ne.0) then
  stop 'Could not allocate de in components'
endif

! allocate the temperature gradient array
allocate(tp(nsp+iadia),stat=ierr)
if (ierr.ne.0) then
  stop 'Could not allocate tp in components'
endif

! allocate the density gradient array
allocate(fp(nsp+iadia),stat=ierr)
if (ierr.ne.0) then
  stop 'Could not allocate fp in components'
endif

! allocate the velocity gradient array
allocate(vp(nsp+iadia),stat=ierr)
if (ierr.ne.0) then
  stop 'Could not allocate vp in components'
endif

! allocate the type of the species 
allocate(types(nsp+iadia),stat=ierr)
if (ierr.ne.0) then
  stop 'Could not allocate vp in components'
endif

! allocate the parameter for the background array 
allocate(pbg(nsp+iadia),stat=ierr)
if (ierr.ne.0) then
  stop 'Could not allocate vp in components'
endif

! the ratio of thermal and reference velocity 
allocate(vthrat(nsp),stat = ierr)
if (ierr.ne.0) then 
  stop 'Could not allocate vthrat in components'
endif

! The ratio of Larmor radius and reference value 
allocate(rhorat(nsp),stat=ierr)
if (ierr.ne.0) then 
  stop 'Could not allocate rhorat in components'
endif

return
end subroutine components_allocate

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine components_input_species
!--------------------------------------------------------------------
! This subroutine reads the species data again (after the checks and
! now from input.out) and puts the data into the arrays.
! This routine must be called by all processors
!-------------------------------------------------------------------- 
  use io, only : get_free_lun
  use grid, only : number_of_species, nsp, isppb, isppe
  use dist,    only : iadia
  
  integer :: nsps, i, ilun, io_stat, it1

  io_stat = 0
  it1 = 0
  nsps = number_of_species
  if (adiabatic_electrons) nsps = nsps + 1
  
  if (root_processor) then
    call get_free_lun(ilun)
    open(ilun,file='input.out',FORM='formatted',STATUS='old')
  end if
  
  do i=1,nsps
    if (root_processor) then
      call components_read_nml_spec(ilun,io_stat,LWRITE=.false.)
    end if
    call components_bcast_nml_spec
    if ( z < 0 .and. adiabatic_electrons) then  
      mas(nsp+iadia)   = mass
      tmp(nsp+iadia)   = temp
      de(nsp+iadia)    = dens
      signz(nsp+iadia) = z
      fp(nsp+iadia)    = rln
      tp(nsp+iadia)    = rlt
      vp(nsp+iadia)    = uprim
      types(nsp+iadia) = background
      pbg(nsp+iadia)   = param 
    else 
      it1 = it1 + 1
      if ( it1 >= isppb .and.  it1 <= isppe) then 
        mas(it1 - isppb + 1 )  = mass
        signz(it1 - isppb + 1) = z
        tmp(it1 - isppb + 1)   = temp
        de(it1 - isppb + 1)    = dens
        tp(it1 - isppb + 1)    = rlt
        fp(it1 - isppb + 1)    = rln 
        vp(it1 - isppb + 1)    = uprim
        types(it1 - isppb + 1) = background
        pbg(it1 - isppb + 1)   = param 
      end if 
    end if
  end do

  if (root_processor) close(ilun)

  ! Calculate the normalizing coefficients 
  do i = 1, nsp
    vthrat(i) = sqrt(tmp(i)/mas(i))
    rhorat(i) = sqrt(mas(i)*tmp(i))/abs(signz(i))
  end do  
    
end subroutine components_input_species

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



!--------------------------------------------------------------------
!> This subroutine deallocates the arrays of components 
!--------------------------------------------------------------------
subroutine components_deallocate 

if (allocated(mas))    deallocate(mas)
if (allocated(signz))  deallocate(signz)
if (allocated(tmp))    deallocate(tmp)
if (allocated(de))     deallocate(de)
if (allocated(tp))     deallocate(tp)
if (allocated(fp))     deallocate(fp)
if (allocated(vp))     deallocate(vp)
if (allocated(vthrat)) deallocate(vthrat)
if (allocated(rhorat)) deallocate(rhorat)

end subroutine components_deallocate 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


end module components
