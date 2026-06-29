! $Id: diagnostic.F90 1009 2009-07-02 17:08:44Z  $
!----------------------------------------------------------------------------

  module diagnostic

  use mpiinterface
  use mpidatatypes
  use mpicomms
  use global
  use general,   only : gkw_abort, gkw_warn, genfilename ! MOVE ME

  implicit none

  private

  public :: diagnostic_read_nml, diagnostic_write_nml
  public :: diagnostic_bcast_nml, diagnostic_check_params
  public :: diagnostic_allocate, final_output, fluxes, readfile, noreadfile
  public :: output_init,output_finalize, write_output, binarypotentialoutput

  interface diagnostic_write_nml
    module procedure diagnostic_read_nml
  end interface

  interface fluxes_det
    module procedure fluxes_det_original ! REMOVE ME
  end interface

  !> the particle flux per mode : pflux(nmod,nx,number_of_species) due to the 
  !> ExB motion 
  real, allocatable, dimension(:,:,:) :: pflux_es

  !> the energy flux per mode : eflux(nmod,nx,number_of_species) due to the 
  !> ExB motion 
  real, allocatable, dimension(:,:,:) :: eflux_es

  !> the parallel velocity flux per mode : vflux(nmod,nx,number_of_species)
  !> due to the ExB motion 
  real, allocatable, dimension(:,:,:) :: vflux_es

  !> the particle flux per mode pflux(nmod,nx,number_of_species) due to the
  !> flutter of the field  
  real, allocatable, dimension(:,:,:) :: pflux_em

  !> the energy flux per mode : eflux(nmod,nx,number_of_species) due to the 
  !> flutter of the field 
  real, allocatable, dimension(:,:,:) :: eflux_em

  !> the parallel velocity flux per mode : vflux(nmod,nx,number_of_species)
  !> due to the flutter of the field  
  real, allocatable, dimension(:,:,:) :: vflux_em
  
  !> total particle flux per species : pflux_tot(number_of_species)
  !> electrostatic, electromagnetic
  real, allocatable, dimension(:) :: pflux_tot_es
  real, allocatable, dimension(:) :: pflux_tot_em
  
  !> the total energy flux per species : eflux_tot(number_of_species) 
  !> electrostatic and electromagnetic contributions 
  real, allocatable, dimension(:) :: eflux_tot_es
  real, allocatable, dimension(:) :: eflux_tot_em
  
  !> total parallel velocity flux per species : vflux_tot(number_of_species)
  !> electrostatic and electromagnetic contributions 
  real, allocatable, dimension(:) :: vflux_tot_es
  real, allocatable, dimension(:) :: vflux_tot_em
  
  !> spectral particle flux per species : pflux_spec(nmod,number_of_species)
  !> summing electrostatic and electro-magnetic contributions 
  real, allocatable, dimension(:,:) :: pflux_spec
  real, allocatable, dimension(:,:) :: pflux_xspec !<(nx,number_of_species)
  
  !> spectral energy flux per species : eflux_spec(nmod,number_of_species)
  !> summing electrostatic and electromagnetic contributions 
  real, allocatable, dimension(:,:) :: eflux_spec
  real, allocatable, dimension(:,:) :: eflux_xspec !<(nx,number_of_species)
  
  !> the spectral parallel velocity flux per species : 
  !> vflux_spec(nmod,number_of_species)
  !> summing electrostatic and electromagnetic contributions 
  real, allocatable, dimension(:,:) :: vflux_spec
  real, allocatable, dimension(:,:) :: vflux_xspec !<(nx,number_of_species)
  
  !> The neoclassical particle flux.  pflux_nc(number_of_species)
  real, allocatable, dimension(:) :: pflux_nc
  
  !> The neoclassical energy flux. eflux_nc(number_of_species)
  real, allocatable, dimension(:) :: eflux_nc
  
  !> The neoclassical momentum flux. vflux_nc(number_of_species)
  real, allocatable, dimension(:) :: vflux_nc
  
  !> The buffer for communication. fluxbuf(nmod,nx,number_of_species)
  real, allocatable, dimension(:,:,:) :: fluxbuf(:,:,:)
  
  !> The buffer for communication. fluxncbuf(number_of_species)
  real, allocatable, dimension(:) :: fluxncbuf
  
  !> the ky-spectrum. ky_spec(nmod), kx_spec(nx)
  real, allocatable, dimension(:) :: ky_spec, kx_spec
  
  !> buffer for MPI reductions, buffer(max(nx,nmod))
  real, allocatable, dimension(:) :: buffer

  !> the particle flux  pflux(nmod,nx,number_of_species,ns,nmu,nvpar) 
  !> due to the ExB motion 
  real, allocatable, dimension(:,:,:,:,:,:) :: pflux_det
  
  !> the energy flux  eflux(nmod,nx,number_of_species,ns,nmu,nvpar) 
  !> due to the ExB motion 
  real, allocatable, dimension(:,:,:,:,:,:) :: eflux_det
  
  !> the parallel velocity flux vflux(nmod,nx,number_of_species,ns,nmu,nvpar) 
  !> due to the ExB motion 
  real, allocatable, dimension(:,:,:,:,:,:) :: vflux_det
  
  !> array for writing lines of the fluxes file
  real, allocatable, dimension(:) :: flux_tot_es
  real, allocatable, dimension(:) :: flux_tot_em
  
  !> arrays for parallel_phi phi_par(n_s_grid)
  real, allocatable, dimension(:) :: phi_par
  real, allocatable, dimension(:) :: phi_par_buf
  
  !> buffer for calculating 3D outputs
  real, allocatable :: datbuffer(:,:,:)

  !> complex slice in xy for mpi reduction
  complex, allocatable, dimension(:,:) :: cslice_xy
  
  !> buffer for writing fdisi in the right order for restart
  complex, allocatable, dimension(:) :: fdisiiobuf
 
  !> switch for output the fluxes on the s, mu and vpar grids
  logical, public :: lfluxes_detail = .false.

  ! logicals to control the fluxes output; default to T
  logical, save :: lpflux = .true., leflux = .true. , lvflux = .true. 
  logical, save :: lfluxes_spectra = .true.
  
  !logical to control parallel_phi output; default to T
  logical, save :: lparallel_phi=.true.

  !> screen output
  logical, save :: lscreen_output = .false.
  
  !> file output
  logical, save :: lfile_output = .false.
  
  !> number of slots in fluxes write array
  integer :: nfluxes
  
  ! integer for luns of various output files
  integer, save :: i_time = -1, i_fluxes = -1, i_fluxes_em = -1, i_neoclass = -1
  integer, save :: i_efluxspec = -1, i_pfluxspec = -1, i_vfluxspec = -1
  integer, save :: i_efluxxspec = -1, i_pfluxxspec = -1, i_vfluxxspec = -1
  integer, save :: i_kyspec = -1, i_kxspec = -1
  integer, save :: i_parphi = -1, i_potframe = -1

  ! for error
  integer :: ierr 

  !> norm of fdisi, used in file output
  real :: fnorm
 
  !> flush the fluxes and time etc. every nflush_ts timesteps (default 0)
  integer, save :: nflush_ts = 0

  ! status
  integer, dimension(MPI_STATUS_SIZE) :: state

  ! data representation
  character (len=6), parameter :: data_representation = 'native'

  ! if MPI IO does not fully support derived datatypes, this should be T
  logical :: lmpi_broken_io = .true.
  
contains 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> read or write the diagnostic nml if ldiagnostic_namelist is set in control
!----------------------------------------------------------------------------

subroutine diagnostic_read_nml(ilun,io_stat,lwrite)

  use control, only : ldiagnostic_namelist

  integer, intent(in) :: ilun
  integer,intent(out) :: io_stat
  logical, optional, intent(in) :: lwrite

  namelist /diagnostic/ lpflux,leflux,lvflux,lfluxes_spectra,lparallel_phi,  &
                      & lfluxes_detail,nflush_ts,lmpi_broken_io
   
  io_stat=0
 
  if (present(lwrite)) then
    if ((.not. lwrite) .and. ldiagnostic_namelist) then
      ! set defaults
      lfluxes_detail=.false.
      ! read nml
      read(ilun,NML=diagnostic,IOSTAT=io_stat)
    else
      ! do nothing
    end if
  else
    write(ilun,NML=diagnostic)
  end if

end subroutine diagnostic_read_nml

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> broadcast the diagnostic namelist
!----------------------------------------------------------------------------

subroutine diagnostic_bcast_nml

  call mpibcast_integer(nflush_ts,1)
  call mpibcast_logical(lpflux,1)
  call mpibcast_logical(leflux,1)
  call mpibcast_logical(lvflux,1)
  call mpibcast_logical(lfluxes_spectra,1)
  call mpibcast_logical(lfluxes_detail,1)
  call mpibcast_logical(lmpi_broken_io,1)
 
end subroutine diagnostic_bcast_nml

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> check the diagnostic params; calculate things needed before main allocate
!----------------------------------------------------------------------------

subroutine diagnostic_check_params

  !
  ! Put any necessary checks here 
  !

  
  !
  ! Set anything needed before allocate here
  !
  
  nfluxes=0
  if (lpflux) nfluxes=nfluxes+1
  if (leflux) nfluxes=nfluxes+1
  if (lvflux) nfluxes=nfluxes+1

end subroutine diagnostic_check_params

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Allocates the arrays of the diagnostic module that are used at runtime;
!> other diagnostics at the end of the run can manage their own memory after
!> most of the runtime memory has been deallocated.
!----------------------------------------------------------------------------

subroutine diagnostic_allocate 
  
  use control, only : output3d
  use grid,    only : nmod,nx,number_of_species,n_s_grid,ns,nsp,nmu,nvpar
  use non_linear_terms, only : mphi,mrad,mphiw3

  ierr=0
  allocate(pflux_es(nmod,nx,number_of_species),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: pflux_es')
  allocate(eflux_es(nmod,nx,number_of_species),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: eflux_es')
  allocate(vflux_es(nmod,nx,number_of_species),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: vflux_es')
  allocate(pflux_em(nmod,nx,number_of_species),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: pflux_em')
  allocate(eflux_em(nmod,nx,number_of_species),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: eflux_em') 
  allocate(vflux_em(nmod,nx,number_of_species),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: vflux_em') 
  allocate(fluxbuf(nmod,nx,number_of_species),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: fluxbuf') 
  allocate(pflux_tot_es(number_of_species),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: pflux_tot_es')
  allocate(pflux_tot_em(number_of_species),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: pflux_tot_em')
  allocate(eflux_tot_es(number_of_species),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: eflux_tot_es')
  allocate(eflux_tot_em(number_of_species),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: eflux_tot_em')
  allocate(vflux_tot_es(number_of_species),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: vflux_tot_es')
  allocate(vflux_tot_em(number_of_species),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: vflux_tot_em')
  allocate(pflux_nc(number_of_species),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: pflux_nc')
  allocate(eflux_nc(number_of_species),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: eflux_nc')
  allocate(vflux_nc(number_of_species),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: vflux_nc')
  allocate(fluxncbuf(number_of_species),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: fluxncbuf')
  allocate(flux_tot_es(nfluxes*number_of_species),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: flux_tot_es')
  allocate(flux_tot_em(nfluxes*number_of_species),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: flux_tot_em')

  ! Arrays for spectral fluxes
  if (lfluxes_spectra) then
    allocate(pflux_spec(nmod,number_of_species),stat=ierr)
    if (ierr /= 0) call gkw_abort('diagnostic :: pflux_spec')
    allocate(eflux_spec(nmod,number_of_species),stat=ierr)
    if (ierr /= 0) call gkw_abort('diagnostic :: eflux_spec')
    allocate(vflux_spec(nmod,number_of_species),stat=ierr)
    if (ierr /= 0) call gkw_abort('diagnostic :: vflux_spec')
    allocate(pflux_xspec(nx,number_of_species),stat=ierr)
    if (ierr /= 0) call gkw_abort('diagnostic :: pflux_xspec')
    allocate(eflux_xspec(nx,number_of_species),stat=ierr)
    if (ierr /= 0) call gkw_abort('diagnostic :: eflux_xspec')
    allocate(vflux_xspec(nx,number_of_species),stat=ierr)
    if (ierr /= 0) call gkw_abort('diagnostic :: vflux_xspec')
  end if
  
  ! Arrays used when the whole three dimensional outputs selected.
  if (output3d) then
    allocate(datbuffer(ns,mrad,mphi),stat=ierr)
    if (ierr /= 0) call gkw_abort('diagnostic :: datbuffer')
    ! array for slice reduction
    allocate(cslice_xy(mphiw3,mrad),stat=ierr)
    if (ierr /= 0) call gkw_abort('diagnostic :: cslice_xy')
  end if
  
  ! arrays for parallel_phi
  if (lparallel_phi) then
    allocate(phi_par(n_s_grid),stat=ierr)
    if (ierr /= 0) call gkw_abort('diagnostic :: phi_parallel')
    allocate(phi_par_buf(ns),stat=ierr)
    if (ierr /= 0) call gkw_abort('diagnostic :: phi_parallel_buf')
  end if

end subroutine diagnostic_allocate 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> deal with final output after time integration has been completed 
!----------------------------------------------------------------------------

subroutine final_output 

  use dist, only : fdisi,nsolc

  call velocity_space_output
  call fluxes 
  if (lfluxes_detail) call fluxes_det
  call parallel_output
  call write_restart_file 

  ! The append files should now be closed...

end subroutine final_output 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> open the ascii `time series' files; close_output_files will close them all
!> as the io module keeps track of the luns. N.B. the unit to which the file
!> is attached is *returned* by the open_file routine, in contrast to open(). 
!----------------------------------------------------------------------------

subroutine open_ascii_output

  use io,       only : open_file
  use control,  only : non_linear, output3d, nlapar, neoclassics
  use mode,     only : mode_box
  use grid,     only : nx, nmod
  use rotation, only : perp_shear
  
  ! open (or close) the file to write the time evolution of the growth rate
  call open_file(i_time,FILE='time.dat',POSITION='append')

  ! open (or close) the files to write the fluxes 
  call open_file(i_fluxes,FILE='fluxes.dat',POSITION='append')

  if(nlapar) then
    call open_file(i_fluxes_em,FILE='fluxes_em.dat',POSITION='append')
  end if

  if (neoclassics) then 
    call open_file(i_neoclass,FILE='fluxes_nc.dat',POSITION = 'append')
  end if

  if (lfluxes_spectra.and. nmod.gt.1) then
    call open_file(i_efluxspec,FILE='eflux_spectra.dat',POSITION='append')
    call open_file(i_pfluxspec,FILE='pflux_spectra.dat',POSITION='append')
    call open_file(i_vfluxspec,FILE='vflux_spectra.dat',POSITION='append')
  end if
 
  if (lfluxes_spectra.and. nx.gt.1) then
    call open_file(i_efluxxspec,FILE='eflux_xspec.dat',POSITION='append')
    call open_file(i_pfluxxspec,FILE='pflux_xspec.dat',POSITION='append')
    call open_file(i_vfluxxspec,FILE='vflux_xspec.dat',POSITION='append')
  end if

  if (non_linear .or. mode_box .or. perp_shear) then 
    call open_file(i_kyspec,FILE='kyspec',POSITION='append')
    call open_file(i_kxspec,FILE='kxspec',POSITION='append')
  end if 

  if (lparallel_phi) then
    call open_file(i_parphi,FILE='parallel_phi.dat',POSITION='append')
  end if

  if (output3d) then
    call open_file(i_potframe,FILE='Frames.dat',POSITION='asis')
  end if
    
end subroutine open_ascii_output


!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> perform any initialisation that needs to be done before the main time loop
!----------------------------------------------------------------------------

subroutine output_init

  use control, only : screen_output, output3d
  use rotation, only : perp_shear
  
  logical, parameter :: file_output = .true.
  integer :: lun
  
  lscreen_output = root_processor .and. screen_output
  lfile_output   = root_processor .and. file_output
  
  ! open formatted files
  if (lfile_output) then
    call open_ascii_output
  end if

  ! write various run parameters
  call write_run_params

end subroutine output_init

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Write any diagnostic related files that can be output before the main time
!> loop begins. 
!---------------------------------------------------------------------------

subroutine write_run_params

  use io,               only : open_file, close_file,decide_who_writes,      &
                             & get_free_lun
  use rotation,         only : perp_shear 
  use control,          only : output3d
  use mode,            only : lx, ly
  use non_linear_terms, only : mrad, mphi
  use grid,             only : n_s_grid, ns
  use geom,             only : q, eps, sgr
  
  integer :: lun,record_length,i
  logical :: lwrite

  ! write 3D params
  if (output3d .or. perp_shear) then
    
    if (lfile_output) then
      call open_file(lun,FILE='3DOutputParam.dat')
      write(lun,*) 'BoxSize:', lx
      write(lun,*) 'BoxSize:', ly
      write(lun,*) 'mrad:', mrad
      write(lun,*) 'mphi:', mphi
      write(lun,*) 'ns:', n_s_grid
      write(lun,*) 'q:', q
      write(lun,*) 'eps:', eps
      if (lverbose) write(*,*) '* Geom parameters written to 3DOutputParam.dat'
      call close_file(lun)
    end if
    
    call decide_who_writes(MPI_COMM_SELF,lwrite)
    ! The values of s written to file (will not work for parallel_s)
    if (ns == n_s_grid) then
      if (lwrite) then
        inquire(iolength=record_length) sgr(1,1)
        call get_free_lun(lun)
        open(lun,FILE='SPoints.dat',FORM='unformatted',ACCESS='direct',          &
            & RECL=record_length)
        do i =1, ns
          write (lun,rec=i) sgr(1,i)
        end do
        close(lun)
      end if
    end if
    
  end if

end subroutine write_run_params

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> This subroutine controls the writes to file and screen after each (large)
!> timestep (consisting on naverage small time steps).
!----------------------------------------------------------------------------

subroutine write_output

  use dist,         only : fdisi, nsolc, nf
  use mode,         only : mode_box
  use control,      only : non_linear, normalized, nlapar
  use control,      only : output3d, lphi_diagnostics
  use rotation,     only : perp_shear

  !
  ! Various diagnostics that must be called by all processors.
  !
  
  ! parallel_phi
  if (lparallel_phi) call parallel_phi

  ! 2D outputs: CHECK THESE WORK WITH PARALLEL_S
  if ((non_linear .or. mode_box .or. perp_shear) .and. lphi_diagnostics) then
    call phi_xy_output(fdisi(1:nsolc)) !works with parallel_s
    if (nlapar) call apar_xy_output(fdisi(1:nsolc)) !works with parallel_s
    call phi_ky_spec(fdisi(1:nsolc))   !works with parallel_s
    !if (perp_shear) call xy_output(fdisi,2,nmu,nvpar,1) !not with parallel_s
  end if
  ! Outputs files for the whole flux tube. Files are named PhiTD
  ! WARNING, data volume can be huge.
  if (output3d) then
  
    if (.not. mode_box) then
      call gkw_abort('write_output: To get 3D potential output data, '//     &
          &          'mode_box must also be set to true')
    end if
  
    call binarypotentialoutputlegacy(fdisi(1:nsolc))
    !call binaryaparoutputlegacy(fdisi(1:nsolc))
    !call binarydensityoutput(fdisi(1:nsolc))
  end if !output3D
  
  !Collision operator testing routine.
  !!call velspace

  ! fnorm is used in file output for linear, non-normalised runs; norm
  ! must be called by all processors.
  if ((.not.normalized) .and. (.not. non_linear)) fnorm=norm(fdisi(1:nf))
 
  !
  ! only a single processor will call any routines beyond here
  !
  
  if (lscreen_output) call write_screen_output
  if (lfile_output)   call write_file_output

end subroutine write_output

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Write output to the various formatted files. This routine should be called
!> by only 1 processor after every naverage small timesteps.
!----------------------------------------------------------------------------

subroutine write_file_output

  use io, only : flush_file
  use control,   only : non_linear, lcalc_freq, normalized, time, nlapar, &
                      & neoclassics 
  use grid,      only : number_of_species, nmod, nx
  use mode,      only : mode_box
  use normalise, only : growth_rate, real_frequency

  integer :: is, ix, imod
  
  integer, save :: next_flush_ts = 0
  
  !
  ! time, growth rates etc.
  !
  
  if (non_linear) then 
    !For nonlinear runs
    ! write time only 
    write(i_time, fmt = '(3(1pe13.5))')time
  else
    calc_freq : if (.not. lcalc_freq) then
      ! write time and growth rate 
      if(.not. normalized) then
        write(i_time,fmt = '(3(1pe13.5))')time, growth_rate, fnorm
      else
        write(i_time,fmt = '(3(1pe13.5))')time, growth_rate
      end if
    else
      ! write time and growth rate and real frequency
      write(i_time,fmt='(3(1pe13.5))')time,growth_rate,real_frequency
    end if calc_freq
  end if

  !
  ! write the fluxes
  !
  
  ! fluxes.dat
  !Must have 128 > number_of_species * 3
  write(i_fluxes,fmt = '(128(1pe13.5,1X))') &
       & (flux_tot_es(is),is = 1, number_of_species*nfluxes)

  ! fluxes_em.dat
  if (nlapar) then
    write(i_fluxes_em,fmt = '(128(1pe13.5,1X))') &
        & (flux_tot_em(is),is = 1, number_of_species*nfluxes)
  end if

  if (neoclassics) then 
    write(i_neoclass,fmt = '(128(1pe13.5,1X))') &
        & (pflux_nc(is), eflux_nc(is), vflux_nc(is), is = 1, &
        &  number_of_species)
  end if 

  ! fluxes spectra
  fluxes_spectra : if(lfluxes_spectra) then
    !Write the fluxes spectra
    !Must have 1024 > nmod * number_of_species
    !Possibility of a fortran column limit?
    do is = 1, number_of_species
      if (nmod.gt.1) then
          write(i_efluxspec,fmt = '(1024(1pe13.5,1X))',advance='no')             & 
                        &  (eflux_spec(imod,is),imod = 1,nmod)
          write(i_pfluxspec,fmt = '(1024(1pe13.5,1X))',advance='no')             &
                        &  (pflux_spec(imod,is),imod = 1,nmod)
          write(i_vfluxspec,fmt = '(1024(1pe13.5,1X))',advance='no')             &
                        &  (vflux_spec(imod,is),imod = 1,nmod)
      end if
      if (nx.gt.1) then 
          write(i_efluxxspec,fmt = '(2048(1pe13.5,1X))',advance='no')            & 
                        &  (eflux_xspec(ix,is),ix = 1,nx)
          write(i_pfluxxspec,fmt = '(2048(1pe13.5,1X))',advance='no')            &
                        &  (pflux_xspec(ix,is),ix = 1,nx)
          write(i_vfluxxspec,fmt = '(2048(1pe13.5,1X))',advance='no')            &
                        &  (vflux_xspec(ix,is),ix = 1,nx)
      end if
    end do

    ! New lines
    if (nmod.gt.1) then
      write(i_efluxspec,*)
      write(i_pfluxspec,*)
      write(i_vfluxspec,*)
    end if
    if (nx.gt.1) then
      write(i_efluxxspec,*)
      write(i_pfluxxspec,*)
      write(i_vfluxxspec,*)
    end if 

  end if fluxes_spectra

  ! flush the main ts files if nflush_ts > 0
  if (nflush_ts > 0) then
    next_flush_ts = next_flush_ts + 1
    if (next_flush_ts == nflush_ts) then
      next_flush_ts = 0
      call flush_file(i_time)
      call flush_file(i_fluxes)
      call flush_file(i_fluxes_em)      
      call flush_file(i_neoclass)
    end if
  end if
  
end subroutine write_file_output

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Write output to the screen. This routine should be called by only 1
!> processor after every naverage small timesteps.
!----------------------------------------------------------------------------

subroutine write_screen_output

  use control,   only : non_linear, ntotstep, time, lcalc_freq
  use grid,      only : number_of_species, nmod, nx
  use rotation,  only : perp_shear
  use mode,      only : mode_box
  use normalise, only : real_frequency, growth_rate

  integer :: is, imod, ix
  
  if (non_linear .or. perp_shear) then
    
    write(*,200) ntotstep, time
    write(*,*)
    do is = 1, number_of_species
      write(*,*)'Species ',is
      write(*,*)'Particle flux : ',pflux_tot_es(is)
      write(*,*)'Energy flux   : ',eflux_tot_es(is)
      write(*,*)'Momentum flux : ',vflux_tot_es(is)
      write(*,*)
    end do  
  
  else

    modebox : if (mode_box) then 
  
      write(*,200)ntotstep,time
      write(*,50)growth_rate 
      50 format('Global growth rate : ',1pe13.5)
      if (lcalc_freq) then
        write(*,60)real_frequency
        60 format('Real frequency : ',1pe13.5)
      end if
   
      do is = 1, number_of_species
        write(*,10)is 
        write(*,20)pflux_tot_es(is)
        write(*,30)eflux_tot_es(is)
        write(*,40)vflux_tot_es(is)
      end do 
      10 format('Species  : ',I4)
      20 format('The total particle flux   :',1pe13.5)
      30 format('The total energy flux     :',1pe13.5)
      40 format('The total momentum flux   :',1pe13.5)
  
    else modebox 
  
      write(*,200) ntotstep, time 
      200 format('Time step : ',I5,' Normalised time : ',1pe13.5)
      mod : do imod = 1, nmod 
        x : do ix = 1, nx 
  
          write(*,201)imod,ix,growth_rate
          201 format('nMode ',I3,' xMode ',I3,' Growth rate ',1pe13.5)

          if (lcalc_freq) then
            write(*,202)imod,ix,real_frequency 
            202 format('nMode ',I3,' xMode ',I3,' Real frequency ',1pe13.5)
          end if
  
          species : do is = 1, number_of_species  
  
            !Note this output block generates zeros for nlapar = false
            !Since the electromagnetic fluxes are zero
            write(*,1)imod,ix,is
            1 format('Toroidal mode ',I3,' Radial mode ',I3,' Species ',I2) 
  
            write(*,2)pflux_es(imod,ix,is), pflux_em(imod,ix,is) 
            2 format('The particle flux  (ES/EM) : ',2(1pe13.5,1X)) 
            write(*,3)eflux_es(imod,ix,is), eflux_em(imod,ix,is) 
            3 format('The energy flux    (ES/EM) : ',2(1pe13.5,1X)) 
            write(*,4)vflux_es(imod,ix,is), vflux_em(imod,ix,is) 
            4 format('The momentum flux  (ES/EM) : ',2(1pe13.5,1X))
            write(*,*)
  
          end do species
        end do x
      end do mod
  
    end if modebox

  end if

end subroutine write_screen_output

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> close the formatted ouput files etc.
!----------------------------------------------------------------------------

subroutine output_finalize

  use io, only : close_output_files

  ! close the files
  if (lfile_output) call close_output_files

end subroutine output_finalize

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!   THIS SHOULD USE decide_who_writes() from IO
!   DOES PARFUN WORK?
!  
!>  This subroutine writes some diagnostics connected with the parallel 
!>  mode structure. For every mode the folowing quantities are written
!>  to file
!>
!>  sgr         the length along the field line 
!>  phi         the potential 
!>  apar        the parallel component of the vector potential 
!>  dens        the perturbed density 
!>  tpar        the perturbed parallel temperature 
!>  tperp       the perturbed perpendicular temperature 
!>  wflow       the perturbed parallel flow velocity
!>
!>  Note only sgr is real. The other quantities are complex and fill 2
!>  columns in the output file.
!>
!----------------------------------------------------------------------------

subroutine parallel_output

  use mpicomms,     only : COMM_S_NE
  use grid,         only : nmod, nx, nsp, ns, nmu, nvpar, parallel_s, isppb, &
                       & isppe, number_of_species,iproc_s
  use dist,         only : fdisi, phi, apar, indx, get_apar, fmaxwl, nsolc
  use geom,         only : sgr, bn
  use mode,         only : kxrh 
  use components,   only : signz, vthrat, tmp
  use functions,    only : besselj0_gkw
  use velocitygrid, only : intmu, intvp, mugr, vpgr
  use io,           only : open_file, close_file

  ! integers for the loop over all grid points 
  integer :: imod, ix, i, j, k, is

  integer :: it, ifile, i_s,i_i
  character (len=9) :: file_name
  real :: b0, kxmin   
  ! for reduction:
  complex, dimension(4) :: arr, rarr
  !> T if processor communicates with root via COMM_S_NE
  logical :: lam_with_root
  complex :: tpar, tperp, dens, wflow, fdis

  ! get the values of the vector potential 
  ! Note they are set to zero if nlapar = false.
  call get_apar(fdisi(1:nsolc),apar)

  ! find the maximum kx
  kxmin = kxrh(1)
  it = 1 
  do i = 2, nx 
    if (kxrh(i) < kxmin) then 
      kxmin = kxrh(i)
      it = i 
    end if 
  end do 

  ! Decide which processors will *write* parallel output. This is all the
  ! processors along a line in the s-direction containing the root processor.
  lam_with_root = .false.
  call mpiallreduce_or(root_processor,lam_with_root,1,COMM_S_NE)

  ! Subset of processors opens an output file and writes the dimensions.
  ! If not parallel in the s-direction, the parallel output file can be
  ! written directly by the root processor.
  write_setup : if (lam_with_root .and. parallel_s) then
    call genfilename('par',file_name,iproc_s)
    call open_file(ifile,FILE=file_name,POSITION='asis')
    call write_parallel_header(ifile)
  else if (root_processor) then
    lam_with_root = .true.
    call open_file(ifile,FILE='parallel.dat',POSITION='asis')
  end if write_setup

  do i_s = 1, number_of_species
    do imod = 1, nmod
      do ix = 1, nx
        do i = 1, ns
  
          tpar  = 0.
          tperp = 0.
          dens  = 0.
          wflow = 0.
          
          is = i_s - isppb + 1
          ! When using parallel species, only perform the velocity space sums
          ! when the outer loop reaches the local species.
          if (i_s >= isppb .and. i_s <= isppe) then
            do k = 1, nvpar
              do j = 1, nmu

                ! Bessel function for gyro-avereging 
                b0 =  besselj0_gkw(imod,ix,i,j,is) 
      
                ! The distribution function 
                fdis = fdisi(indx(imod,ix,i,j,k,is)) 
                fdis = fdis - 2.*signz(is)*vthrat(is)*vpgr(i,j,k)*b0*        &
                     & apar(imod,ix,i)*fmaxwl(i,j,k)/tmp(is) 
      
                dens = dens + bn(i)*intvp(i,j,k)*intmu(j)*fdis
                wflow= wflow + bn(i)*intvp(i,j,k)*intmu(j)*vpgr(i,j,k)*      &
                     & fdis
                tpar = tpar + bn(i)*intvp(i,j,k)*intmu(j)*vpgr(i,j,k)**2*    &
                     & fdis
                tperp= tperp + bn(i)**2*intvp(i,j,k)*intmu(j)*mugr(j)*       &
                     & fdis

              end do 
            end do
          end if
          ! reduce over points of equal s; any procs not responsible for the
          ! current species will use zero values.
          if (number_of_processors > 1) then
            arr(1) = dens ; arr(2) = tpar ; arr(3) = tperp ; arr(4) = wflow
            call mpiallreduce_sum(arr,rarr,4,COMM_S_EQ)
            dens = rarr(1) ; tpar = rarr(2); tperp = rarr(3) ; wflow = rarr(4)
          end if
         
          ! this barrier appears to be necessary
          call mpibarrier()
          ! Some processors write to file here
          if (lam_with_root) then
            write(ifile,fmt = '(13(1x,e13.5))') sgr(ix,i), phi(imod,ix,i),   &
                  & apar(imod,ix,i), dens, tpar, tperp, wflow
          end if
          
        end do
      end do
    end do
  end do

  ! Need to be careful not to do close(ifile) if ifile is not defined on
  ! the local processor; close_file should take care of that.
  if (parallel_s .or. lam_with_root) call close_file(ifile)

end subroutine parallel_output 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> This routine writes the total abs(phi)^2 of parallel potential s summed
!> over all the modes. It is intended to be called at repeated timesteps.
!> This can also be used as a template for time-parallel diagnostics.
!----------------------------------------------------------------------------

subroutine parallel_phi

  use mpicomms, only : COMM_S_NE
  use grid,     only : ns,nmod,nx,n_s_grid
  use dist,     only : phi,fdisi,get_phi

  integer :: is,imod,ix
  logical,save :: lam_with_root=.false., first_call = .true.

  ! on the first call, decide which processors are required
  if (first_call) then
    lam_with_root=root_processor
    ! PARALLEL_S: Choose one processor responsible for each part of the s-grid.
    call mpiallreduce_or(root_processor,lam_with_root,1,COMM_S_NE)
    first_call = .false.
  end if
  
  ! processors not involved just return
  if (.not. lam_with_root) return

  ! obtain local phi
  call get_phi(fdisi,phi)

  ! Sum over modes for the local s points
  phi_par_buf(:) =0.
  do is=1,ns
    do ix=1,nx
      do imod=1,nmod
        phi_par_buf(is) = phi_par_buf(is) + abs(phi(imod,ix,is))**2
      end do
    end do
  end do

  ! gather all the bits of phi to the root processor (the calling processor
  ! with rank=0 in the communicator COMM_S_NE, which in this case is the
  ! global root processor).
  call gather_array(phi_par,n_s_grid,phi_par_buf,ns,COMM_S_NE)

  ! Rescale to be irrespective of number of modes
  phi_par(:)=phi_par(:)/(nmod*nx)

  ! Write with the root_processor.
  if(root_processor) then
    write(i_parphi,fmt = '(512(1pe13.5,1x))') (phi_par(is), is = 1, n_s_grid)
  end if

end subroutine parallel_phi

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Write various slices related to the velocity space to files. At present,
!> the species dependent parts are for the last species on the first
!> processor found to have the maximum potential on it.
!----------------------------------------------------------------------------

subroutine velocity_space_output

  use io,           only : open_file,close_file,output_slice
  use dist,         only : fdisi,phi,indx
  use grid,         only : ns,nmu,nvpar,nsp, lproc_s_upperb,lproc_s_lowerb,  &
                         & n_vpar_grid,n_mu_grid
  use geom,         only : bn
  use velocitygrid, only : intmu,intvp,mugr,vpgr

  integer :: i,j,k,ihelp,i_distr,i1,i2,iproc_vpar,iproc_mu,iproc
  real, dimension(2) :: phimax,phi_max
  logical :: lwrite,in_slice,match_proc
  real, allocatable, dimension(:,:) :: global_vpar_mu,local_vpar_mu
  character (len=18) :: file_fmt 
  
  ! Find the maximum of the potential and find a processor on which that
  ! point exists. This may be several processors. match_proc is T if the
  ! local processor contains the point.
  ihelp=1
  phimax(1)=abs(phi(1,1,1))
  i1=1 ; i2=ns
  if (lproc_s_lowerb) i1=2
  if (lproc_s_upperb) i2=ns-1
  do i=i1,i2
    if (abs(phi(1,1,i)) > phimax(1)) then
      ihelp = i
      phimax(1) = abs(phi(1,1,i))
    end if
  end do
  ! use real(processor_number) for maxloc interface
  ! (could make a more Fortran friendly wrapper here)
  phimax(2)=real(1.*processor_number)
  call mpiallreduce_maxloc(phimax,phi_max,1,COMM_S_NE)
  ! convert back to integer for comparison
  match_proc = (int(phi_max(2)) == processor_number)
 
  ! Find the processors that contain the mu-vpar slice of the point
  ! corresponding to the maximum potential by propagating the logical
  ! `match_proc' in those directions. Only 1 global processor on which
  ! match_proc = T is considered. Also, the ranks of those processes within
  ! the provided communicators; these are used later for gathering the data
  ! and picking a processor to write to file. in_slice = T means the local
  ! processor is in the gathering process. iproc is the global proc in
  ! MPI_COMM_WORLD found to match_proc.
  call get_common_procs_2d(match_proc,iproc,in_slice,COMM_VPAR_NE,COMM_MU_NE)

  ! if not in_slice, may as well return
  if (.not. in_slice) return

  ! set lwrite on one processor
  call mpicomm_rank(COMM_MU_NE,iproc_mu)
  call mpicomm_rank(COMM_VPAR_NE,iproc_vpar)
  lwrite = (iproc_vpar == 0 .and. iproc_mu == 0)
  
  ! set the format string for the output files
  file_fmt='(257(1x,e13.5))'
  
  ! allocate arrays to contain the full slice and local slice
  allocate(local_vpar_mu(nvpar,nmu))
  allocate(global_vpar_mu(n_vpar_grid,n_mu_grid))
  
  ! local processor vpgr
  do j=1,nmu
    do i=1,nvpar
      local_vpar_mu(i,j) = vpgr(1,1,i)
    end do
  end do

  ! gather the data
  call gather_array(global_vpar_mu,n_vpar_grid,n_mu_grid,local_vpar_mu,      &
      &             nvpar,nmu,COMM_VPAR_NE,COMM_MU_NE)
  ! write with 1 processor
  if (lwrite) call output_slice(global_vpar_mu,FILE='distr1.dat',FMT=file_fmt)

  ! local processor mugr
  do j=1,nmu
    do i=1,nvpar
      local_vpar_mu(i,j)=sqrt(2.*bn(ihelp)*mugr(j))
    end do
  end do

  ! gather the data
  call gather_array(global_vpar_mu,n_vpar_grid,n_mu_grid,local_vpar_mu,      &
        &           nvpar,nmu,COMM_VPAR_NE,COMM_MU_NE)
  ! write with 1 processor
  if (lwrite) call output_slice(global_vpar_mu,FILE='distr2.dat',FMT=file_fmt)

  ! local processor imaginary part
  do j=1,nmu
    do i=1,nvpar
      local_vpar_mu(i,j)=aimag(fdisi(indx(1,1,ihelp,j,i,nsp))*intmu(j)      &
          &             *intvp(1,j,i)/phi(1,1,ihelp))
    end do
  end do

  ! gather the data
  call gather_array(global_vpar_mu,n_vpar_grid,n_mu_grid,local_vpar_mu,      &
        &           nvpar,nmu,COMM_VPAR_NE,COMM_MU_NE)
  ! write with 1 processor
  if (lwrite) call output_slice(global_vpar_mu,FILE='distr3.dat',FMT=file_fmt)
        
  ! local processor real part
  do j=1,nmu
    do i=1,nvpar
      local_vpar_mu(i,j)=real(fdisi(indx(1,1,ihelp,j,i,nsp))*intmu(j)        &
          &             *intvp(1,j,i)/phi(1,1,ihelp))
    end do
  end do

  ! gather the data
  call gather_array(global_vpar_mu,n_vpar_grid,n_mu_grid,local_vpar_mu,      &
        &           nvpar,nmu,COMM_VPAR_NE,COMM_MU_NE)
  ! write with 1 processor
  if (lwrite) call output_slice(global_vpar_mu,FILE='distr4.dat',FMT=file_fmt)
  
  ! deallocate the temporary arrays
  if (allocated(local_vpar_mu)) deallocate(local_vpar_mu)
  if (allocated(global_vpar_mu)) deallocate(global_vpar_mu)

end subroutine velocity_space_output 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> routine for testing purposes?
!----------------------------------------------------------------------------

subroutine velspace

  use io,           only : open_file, close_file
  use dist,         only : fdisi, indx, i_mom
  use grid,         only : nmu, nvpar
  use control,      only : ntotstep
  use geom,         only : bn 
  use velocitygrid, only : intmu, intvp, vpgr

  integer :: i,j,k,lun
  real :: dum, dumden
  character (len=14) :: filename,filename2

  if (.not. root_processor) return 

  call genfilename('ReF',filename,ntotstep)  
  call genfilename('ImF',filename2,ntotstep)
  
  dum = 0.
  dumden = 0.
  
  i=0
  do j=1,nmu
    do k=1,nvpar
      dum=dum+bn(i)*intvp(1,j,k)*intmu(j)*vpgr(1,j,k)*fdisi(indx(1,1,1,j,k,1))
      dumden=dumden+bn(i)*intvp(1,j,k)*intmu(j)*fdisi(indx(1,1,1,j,k,1))
    end do
  end do
  
  write(*,*)'Momentum', dum, dumden,fdisi(indx(1,1,1,i_mom,1))

  call open_file(lun,FILE = filename,POSITION='asis')
  do j = 1, nmu 
    write(lun,fmt='(257(1x,e13.5))')(real(fdisi(indx(1,1,1,j,k,1))),k=1,nvpar)
  end do
  call close_file(lun)
      
  call open_file(lun,file='Momentum.dat',POSITION='append')
  write(lun,*)dum,dumden,real(fdisi(indx(1,1,1,i_mom,1)))
  call close_file(lun)
   
!!$  open(10,file = filename2)
!!$  do i = 1, nmu 
!!$     write(10,fmt ='(257(1x,1pe13.5))') (real(fdisi(indx(1,1,ns/2,i,j,2))),  j = 1, nvpar)
!!$  end do
!!$  close(10)

end subroutine velspace

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Write file(s) for restarting. In the latest version this is to a single
!> file "FDS" via MPI IO. This allows restarts on a different number of
!> processors. In the original version, one file per processor is output as
!> a fortran binary file, so this must be run again on the same number of
!> processors. Some MPI implementations do not support writing of the derived
!> datatypes used in the code, so in that case the original restart file
!> version must be selected in the program control.
!----------------------------------------------------------------------------

subroutine write_restart_file

  use global
  use mpiinterface
  use mpidatatypes
  use grid,    only : nsp,ns,nvpar,nmu,nmod,nx
  use dist,    only : fdisi, nf
  use control, only : restart_file_version
  use io,      only : get_free_lun
  use index_function, only : indx_

  integer (kind=MPI_OFFSET_KIND) :: idisp
  character (len=3) :: root 
  character (len=9) :: filename 
  integer :: np,lun,ierr,new_type
  integer :: i,j,k,l,m,n,ind

  ! generate the file name 
  root = 'FDS'

  ! do something based on the restart file version
  restartfileversion : select case(restart_file_version)
    
    case(1)   ! restart_file_version 1: multiple fortran raw binary files
      
      i = processor_number
      call genfilename(root,filename,i)

      ! In this method a barrier is used to prevent all processors outputing
      ! the data in one go. This might not always be necessary, and there
      ! should be more elegant solutions.
      call mpibarrier()

      do np = 1, number_of_processors

        if (np-1 == processor_number) then
          call get_free_lun(lun)
          open(lun,FILE = filename, STATUS= 'unknown', FORM = 'unformatted')
          write(lun)(fdisi(i), i = 1, nf)
          close(lun)
        end if
        call mpibarrier()

      end do 

    case(2)   ! restart_file_version 2: single binary file "FDS"
 
#ifdef mpi
  ! Create a sub-array dataype so that the local part can be written to the
  ! right part of the global array. This should be consistent with the call
  ! in the routine reading the restart file.
  call create_subarray_datatype(MPICOMPLEX_X,new_type,                       &
      &                         id_vpar,id_mu,id_s,id_x,id_mod,id_sp)
  ! open the file
  call MPI_FILE_OPEN(MPI_COMM_WORLD,root,MPI_MODE_WRONLY+MPI_MODE_CREATE,    &
      &              MPI_INFO_NULL,lun,ierr)
  ! Set the file displacement and view for each processor
  idisp = 0
  call MPI_FILE_SET_VIEW(lun,idisp,MPICOMPLEX_X,new_type,                    &
      &                  data_representation,MPI_INFO_NULL,ierr)
  ! write local fdisi from all processors to the file
  if (lmpi_broken_io) then
    ! for broken MPIIO, copy fdisi into a correctly ordered temporary array
    allocate (fdisiiobuf(nf),stat=ierr)
    ind=0
    do n=1,nsp; do m=1,nmod; do l=1,nx; do k=1,ns; do j=1,nmu; do i=1,nvpar
      ind=ind+1
      fdisiiobuf(ind) = fdisi(indx_(m,l,k,j,i,n))
    end do ; end do ; end do ; end do ; end do ; end do
    call MPI_FILE_WRITE_ALL(lun,fdisiiobuf(1),nf,MPICOMPLEX_X,state,ierr)
    deallocate (fdisiiobuf)
  else
    call MPI_FILE_WRITE_ALL(lun,fdisi(1),1,TYPE_RW_FDISI,state,ierr)
  end if
  call mpibarrier()
  ! close the file
  call MPI_FILE_CLOSE(lun,ierr)
#endif

  end select restartfileversion

  if (lverbose) then
    write(*,*) '* wrote restart file, count=', nf
    write(*,*) '*   restart_file_version=', restart_file_version
  end if

end subroutine write_restart_file

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Read the restart files(s).
!----------------------------------------------------------------------------

subroutine readfile

  use global
  use dist,     only : nsolc, fdisi,nf,indx
  use control,  only : non_linear, read_file
  use mode,     only : mode_box
  use grid,     only : nx,ns,nmod,nvpar,nmu,nsp
  use rotation, only : perp_shear
  use mpiinterface

  integer (kind=MPI_OFFSET_KIND) :: idisp
  integer :: NEW_TYPE
logical docont, check
logical :: all_files_present, individual_restart_files
character*3 root 
character*9 name 
integer  np, ierr, lun

  integer :: i,j,k,l,m,n,ind
!Still a potential problem if incorrect input files are provided
!This is hard to make foolproof but should be solved by the rewrite 
!of the restarts allowing changed parallel configurations

if (.not.read_file) call gkw_abort('Invalid call to readfile')

! If no restart file exists return and warn
individual_restart_files = .true.
inquire(file='FDS000000',exist = docont)

! check for the single restart file
if (.not. docont) then
  inquire(file='FDS',exist = docont)
  if (docont) then
    individual_restart_files = .false.
    ! Setup a MPI datatype to read the write part of the data from the file
    ! to the present memory layout.
    !call create_local_complex_subarray()
    call create_subarray_datatype(MPICOMPLEX_X,NEW_TYPE,                     &
        &                         id_vpar,id_mu,id_s,id_x,id_mod,id_sp)
  end if
end if
  
if (.not.docont) then
   if(root_processor) write(*,*) 'No restart file found, intialising new run'
   call noreadfile
   return
end if 

! generate the file name
root = 'FDS'

  ! for version 1 restarts, check all the files exist
  if (individual_restart_files) then
    i = processor_number 
    call genfilename(root,name,i)

    ! find out if the file exists for this processor
    docont=.false.
    inquire(file=name,exist = docont)

    ! check that all processors have a restart file
#if defined(mpi)
    call MPI_ALLREDUCE(docont,all_files_present,1,MPI_LOGICAL,MPI_LAND,      &
        &              MPI_COMM_WORLD,ierr)
#else
    all_files_present = docont
#endif
  
    if (.not. all_files_present) then
      call gkw_abort('not enough input files for restart')
    end if

  else
    ! do nothing -- only 1 file; nothing to check
  end if

!Check for existance of files to append to
if(root_processor) then
    write(*,*) 'Read restart files.  Data will append to time.dat, fluxes.dat and (ky/kx)spec'
        inquire(file='time.dat',exist = check)
        if(.not.check) write(*,*) 'time.dat not found. Will create new'
        inquire(file='fluxes.dat',exist = check)
        if(.not.check) write(*,*) 'fluxes.dat not found. Will create new'
        if(lfluxes_spectra) then
            inquire(file='pflux_spectra.dat',exist = check)
            if(.not.check) write(*,*) 'pflux_spectra.dat not found. Will create new'
            inquire(file='vflux_spectra.dat',exist = check)
            if(.not.check) write(*,*) 'vflux_spectra.dat not found. Will create new'
            inquire(file='eflux_spectra.dat',exist = check)
            if(.not.check) write(*,*) 'eflux_spectra.dat not found. Will create new'
            inquire(file='pflux_xspec.dat',exist = check)
            if(.not.check) write(*,*) 'pflux_xspec.dat not found. Will create new'
            inquire(file='vflux_xspec.dat',exist = check)
            if(.not.check) write(*,*) 'vflux_xspec.dat not found. Will create new'
            inquire(file='eflux_xspec.dat',exist = check)
            if(.not.check) write(*,*) 'eflux_xspec.dat not found. Will create new'
        end if
        if (non_linear.or.mode_box.or.perp_shear) then
            inquire(file='kyspec',exist = check)
            if(.not.check) write(*,*) 'kyspec not found. Will create new'
            inquire(file='kxspec',exist = check)
            if(.not.check) write(*,*) 'kxspec not found. Will create new'
        end if
        if (lparallel_phi) then
            inquire(file='parallel_phi.dat',exist = check)
            if(.not.check) write(*,*) 'parallel_phi.dat not found. Will create new'
        end if
END IF

! At pressent a barrier is used to prevent all processors of reading 
! in all the data in one go. This might not always be necessary, 
! and there should be more elegant solutions 
call mpibarrier()

restart_file_method : if (individual_restart_files) then
  do np = 1 , number_of_processors 

    if (np-1.eq.processor_number) then 
      open(11,file = name, status = 'unknown', form = 'unformatted')
        read(11)(fdisi(i), i = 1, nf)
        write(*,*)nf
      close(11)
    end if 
    
    call mpibarrier()

  end do 

else restart_file_method
! single restart file
#if defined(mpi)
  call MPI_FILE_OPEN(MPI_COMM_WORLD,"FDS",MPI_MODE_RDONLY,   &
      &              MPI_INFO_NULL,lun,ierr)
  idisp = 0
  call MPI_FILE_SET_VIEW(lun,idisp,MPICOMPLEX_X,NEW_TYPE,           &
      &                  data_representation,MPI_INFO_NULL,ierr)
  if (lmpi_broken_io) then
    ! for broken MPIIO, copy the input into a temporary buffer, then re-order
    allocate (fdisiiobuf(nf),stat=ierr)
    call MPI_FILE_READ_ALL(lun,fdisiiobuf(1),nf,MPICOMPLEX_X,state,ierr)
    ind=0
    do n=1,nsp; do m=1,nmod; do l=1,nx; do k=1,ns; do j=1,nmu; do i=1,nvpar
      ind=ind+1
      fdisi(indx(m,l,k,j,i,n)) = fdisiiobuf(ind)
    end do ; end do ; end do ; end do ; end do ; end do
    deallocate (fdisiiobuf)
  else
    call MPI_FILE_READ_ALL(lun,fdisi(1),1,TYPE_RW_FDISI,state,ierr)
  end if
  call mpibarrier()
  call MPI_FILE_CLOSE(lun,ierr)
#endif

end if restart_file_method

end subroutine readfile 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine noreadfile
!This routine checks for the existance of earlier data files
!If they exist the run is terminated.
!This prevents confusion of appending / overwriting data of an older run.

logical, dimension(8) :: checks

checks(:) = .false.

        inquire(file='time.dat',exist = checks(1))
        inquire(file='fluxes.dat',exist = checks(2))
        inquire(file='kyspec',exist = checks(3))
        inquire(file='kxspec',exist = checks(4))
        inquire(file='eflux_spectra.dat',exist = checks(5))
        inquire(file='vflux_spectra.dat',exist = checks(6))
        inquire(file='pflux_spectra.dat',exist = checks(7))
        inquire(file='parallel_phi.dat',exist = checks(8))
        if (any(checks)) then
            call gkw_abort('Existing data files found. &
                & Please remove previous run data (script gkw_clean_run) &
                & or set readfile=.true. to restart') 
        end if

       call mpibarrier

return
end subroutine noreadfile

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


subroutine fluxes
!--------------------------------------------------------------------
!
! This routine calculates the fluxes of particles, energy and  
! parallel momentum 
! 
!
! Normalisation is such that the flux in the gradient of psi 
! (R_ref \nabla \psi, because psi is normalised) is 
! 
! Gamma (R_r nabla psi)    = n_s rho^*2 v_thref pflux 
! Q_s (R_r nabla psi)      = n_s rho^*2 v_thref T_s eflux 
! Gamma_phi(R_r nabla psi) = n_s rho^*2 v_thref m_s v_ths vflux 
!
! So to get the real flux one still needs to multiply with the 
! density and or temperature of the particular species 
! 
! The normalization, however, has the advantage that 
! 
! D       = rho^*2 Ln vthref pflux 
! chi     = rho^*2 Lt vthref eflux 
! chi_phi = rho^*2 Lu vthref v_R vflux 
!
! where the L refers to the gradient length. Note that the momentum 
! flux must be multiplied with the relative velocity 
!
! The routine calculates both the anomalous as well as the neo-
! classical fluxes. The anomalous flux is furthermore split in 
! the contributions due to the ExB velocity and the magnetic 
! flutter. 
!--------------------------------------------------------------------
use control,    only : nlphi, nlapar
use grid,       only : nx, ns, nmu, nvpar, nsp, &
                     & nmod, number_of_species, isppb  
use dist,       only : fdisi, phi, get_phi,&
                     & indx, apar, get_apar, fmaxwl, nsolc 
use geom,       only : ints, bn, dfun, efun, hfun  
use mode,       only : krho, kxrh
use components, only : tmp, vthrat, de, signz  
use rotation,   only : vcor
use functions, only : besselj0_gkw 
use velocitygrid, only : intmu, intvp, mugr, vpgr

implicit none


! integers for the loop over all grid points 
integer imod, ix, i, j, k, is 

! Dummy variables 
complex dum, dumes1, dumes2, dumem1, dumem2, fdis   
real phi2, apa2, b0, ED, dumnc  

! The actual species index is in isglb
integer isglb

! number of elements in the buffer 
integer nelem 

! error integer 
integer ierr

! index for flux output array
integer :: iflux

! Copy phi from the solution. (note for runs without potential
! phi will be set to zero inside get_phi. For the coding below 
! this routine must be called even if phi is not kept in the 
! equations 
call get_phi(fdisi(1:nsolc),phi)

! copy apar from the solution (note for electrostatic runs apar 
! is set to zero inside get_apar. For the coding below this 
! routine must be called even for electrostatic cases 
call get_apar(fdisi(1:nsolc),apar)

! Initialize the fluxes to zero 
pflux_nc = 0.
eflux_nc = 0. 
vflux_nc = 0. 
pflux_es = 0. 
eflux_es = 0.
vflux_es = 0.
pflux_em = 0. 
eflux_em = 0.
vflux_em = 0.

!  Calculate the particle flux
do imod = 1, nmod 
  do ix = 1, nx 
    do is = 1, nsp

      ! the actual (global) species index 
      isglb = is + isppb  - 1 

      ! check if this is the 0,0 mode for which the neoclassical
      ! fluxes are calculated 
      if ((abs(kxrh(ix)).lt.1e-6).and.(abs(krho(imod)).lt.1e-6)) then

        do i = 1, ns 
          do j = 1, nmu 
            do k = 1, nvpar 
              
              ED = vpgr(i,j,k)**2 + mugr(j)*bn(i) 
              
              ! The B x nabla B contribution 
              dumnc = tmp(is)*ED*dfun(i,1) 

              ! The Coriolis contribution 
              dumnc = dumnc + 2.E0*vpgr(i,j,k)*tmp(is)*vcor*hfun(i,1)/     &
                    & vthrat(is) 

              ! fdis is the distribution without A_par contribution
              ! Note the bessel function is 1. since k_perp = 0  
              fdis = fdisi(indx(imod,ix,i,j,k,is)) 
              if (nlapar) then 
                 fdis = fdis - 2.E0*signz(is)*vpgr(i,j,k)*vthrat(is)*      & 
                      & apar(imod,ix,i)*fmaxwl(i,j,k)/tmp(is) 
              end if 

              ! common factors 
              dumnc = dumnc * (bn(i)*intmu(j)*intvp(i,j,k)/signz(is))*     &
                    & fdis*ints(ix,i)

              ! the fluxes
              pflux_nc(isglb) = pflux_nc(isglb) + dumnc  
              eflux_nc(isglb) = eflux_nc(isglb) + dumnc*(vpgr(i,j,k)**2 +  &
                           & 2.E0*mugr(j)*bn(i) ) 
              vflux_nc(isglb) = vflux_nc(isglb) + dumnc * vpgr(i,j,k)
            end do 
          end do 
        end do 


      end if 
    
      ! Integral over the velocity space 
      do j = 1, nmu
        do k = 1, nvpar

          phi2 = 0.
          apa2 = 0. 

          ! Do the average over the flux surface 
          do i = 1, ns

            ! Bessel function for gyro-averaging 
            b0 =  besselj0_gkw(imod,ix,i,j,is) 

            ! fdis is the distribution without A_par contribution  
            fdis = fdisi(indx(imod,ix,i,j,k,is)) 
            if (nlapar) then 
               fdis = fdis - 2.E0*signz(is)*vpgr(i,j,k)*vthrat(is)*   & 
                    & b0*apar(imod,ix,i)*fmaxwl(i,j,k)/tmp(is)  
            end if 


            dum  = 2.E0*ints(ix,i)*(efun(i,1,1)*kxrh(ix) +            &
                 & efun(i,2,1)*krho(imod))*bn(i)*fdis

            dumes1 = dum*b0*conjg(phi(imod,ix,i))*intvp(i,j,k)
            dumes2 = dum*b0*conjg(phi(imod,ix,i))*bn(i)*intvp(i,j,k)
            dumem1 = -2.E0*vthrat(is)*vpgr(i,j,k)*dum*b0*             &
                   & conjg(apar(imod,ix,i))*intvp(i,j,k)
            dumem2 = -2.E0*vthrat(is)*vpgr(i,j,k)*dum*b0*             &
                   & conjg(apar(imod,ix,i))*bn(i)*intvp(i,j,k)

            phi2 = phi2 + ints(ix,i)*abs(phi(imod,ix,i))**2
            apa2 = apa2 + ints(ix,i)*abs(apar(imod,ix,i))**2

            pflux_es(imod,ix,isglb)=pflux_es(imod,ix,isglb) +         &
              & aimag(dumes1)*intmu(j) 
            eflux_es(imod,ix,isglb) = eflux_es(imod,ix,isglb) +       &
              & intmu(j)*(vpgr(i,j,k)**2*aimag(dumes1)+                &
              & 2.E0*mugr(j)*aimag(dumes2))
            vflux_es(imod,ix,isglb)=vflux_es(imod,ix,isglb) +         &
              & aimag(dumes1)*intmu(j)*vpgr(i,j,k) 
            pflux_em(imod,ix,isglb)=pflux_em(imod,ix,isglb)+          &
              & aimag(dumem1)*intmu(j) 
            eflux_em(imod,ix,isglb) = eflux_em(imod,ix,isglb) +       &
              & intmu(j)*(vpgr(i,j,k)**2*aimag(dumem1) +               &
              & 2.E0*mugr(j)*aimag(dumem2))
            vflux_em(imod,ix,isglb)=vflux_em(imod,ix,isglb)+          &
              & aimag(dumem1)*intmu(j)*vpgr(i,j,k) 

          end do
        end do
      end do 
    end do 
  end do 
end do 



#if defined(mpi)
  ! only when run on more than one processor 
  if (number_of_processors.gt.1) then 

    ierr = 0
    nelem = nmod*nx*number_of_species
    call MPI_ALLREDUCE(pflux_es,fluxbuf,nelem,MPIREAL_X, &
       & MPI_SUM, MPI_COMM_WORLD, ierr)
    do imod = 1, nmod 
      do ix = 1, nx 
        do is = 1, number_of_species 
          pflux_es(imod,ix,is) = fluxbuf(imod,ix,is)
        end do 
      end do 
    end do 
    nelem = nmod*nx*number_of_species
    call MPI_ALLREDUCE(pflux_em,fluxbuf,nelem,MPIREAL_X, &
       & MPI_SUM, MPI_COMM_WORLD, ierr)
    do imod = 1, nmod 
      do ix = 1, nx 
        do is = 1, number_of_species 
          pflux_em(imod,ix,is) = fluxbuf(imod,ix,is)
        end do 
      end do 
    end do 
    nelem = nmod*nx*number_of_species


    ! testing 
    !write(*,*)processor_number, eflux_es(1,1,1)

   
    call MPI_ALLREDUCE(eflux_es,fluxbuf,nelem,MPIREAL_X, &
       & MPI_SUM, MPI_COMM_WORLD, ierr)
    do imod = 1, nmod 
      do ix = 1, nx 
        do is = 1, number_of_species 
          eflux_es(imod,ix,is) = fluxbuf(imod,ix,is)
        end do 
      end do 
    end do 

    !call mpi_barrier(mpi_comm_world,ierr)
    !write(*,*)processor_number, eflux_es(1,1,1)
    !call mpi_barrier(mpi_comm_world,ierr)
    !stop 



    nelem = nmod*nx*number_of_species
    call MPI_ALLREDUCE(eflux_em,fluxbuf,nelem,MPIREAL_X, &
       & MPI_SUM, MPI_COMM_WORLD, ierr)
    do imod = 1, nmod 
      do ix = 1, nx 
        do is = 1, number_of_species 
          eflux_em(imod,ix,is) = fluxbuf(imod,ix,is)
        end do 
      end do 
    end do 
    nelem = nmod*nx*number_of_species
    call MPI_ALLREDUCE(vflux_es,fluxbuf,nelem,MPIREAL_X, &
       & MPI_SUM, MPI_COMM_WORLD, ierr)
    do imod = 1, nmod 
      do ix = 1, nx 
        do is = 1, number_of_species 
          vflux_es(imod,ix,is) = fluxbuf(imod,ix,is)
        end do 
      end do 
    end do 
    nelem = nmod*nx*number_of_species
    call MPI_ALLREDUCE(vflux_em,fluxbuf,nelem,MPIREAL_X, &
       & MPI_SUM, MPI_COMM_WORLD, ierr)
    do imod = 1, nmod 
      do ix = 1, nx 
        do is = 1, number_of_species 
          vflux_em(imod,ix,is) = fluxbuf(imod,ix,is)
        end do 
      end do 
    end do 

    nelem = number_of_species
    call MPI_ALLREDUCE(pflux_nc,fluxncbuf,nelem,MPIREAL_X, &
       & MPI_SUM, MPI_COMM_WORLD, ierr)
    do is = 1, number_of_species 
      pflux_nc(is) = fluxncbuf(is)
    end do 
    nelem = number_of_species
    call MPI_ALLREDUCE(eflux_nc,fluxncbuf,nelem,MPIREAL_X, &
       & MPI_SUM, MPI_COMM_WORLD, ierr)
    do is = 1, number_of_species 
      eflux_nc(is) = fluxncbuf(is)
    end do 
    nelem = number_of_species
    call MPI_ALLREDUCE(vflux_nc,fluxncbuf,nelem,MPIREAL_X, &
       & MPI_SUM, MPI_COMM_WORLD, ierr)
    do is = 1, number_of_species 
      vflux_nc(is) = fluxncbuf(is)
    end do 
   
  end if 
#endif 

! the total flux 
do is = 1, number_of_species 
  pflux_tot_es(is) = 0. 
  eflux_tot_es(is) = 0. 
  vflux_tot_es(is) = 0. 
  pflux_tot_em(is) = 0. 
  eflux_tot_em(is) = 0. 
  vflux_tot_em(is) = 0. 


  do imod = 1, nmod

   if(lfluxes_spectra) then
      pflux_spec(imod,is) = 0. 
      eflux_spec(imod,is) = 0. 
      vflux_spec(imod,is) = 0. 
   end if

    do ix = 1, nx

      if(lfluxes_spectra) then
        pflux_spec(imod,is) = pflux_spec(imod,is) + pflux_es(imod,ix,is)
        eflux_spec(imod,is) = eflux_spec(imod,is) + eflux_es(imod,ix,is)
        vflux_spec(imod,is) = vflux_spec(imod,is) + vflux_es(imod,ix,is)
      end if

      pflux_tot_es(is) = pflux_tot_es(is) + pflux_es(imod,ix,is)
      eflux_tot_es(is) = eflux_tot_es(is) + eflux_es(imod,ix,is)
      vflux_tot_es(is) = vflux_tot_es(is) + vflux_es(imod,ix,is)

      pflux_tot_em(is) = pflux_tot_em(is) + pflux_em(imod,ix,is)
      eflux_tot_em(is) = eflux_tot_em(is) + eflux_em(imod,ix,is) 
      vflux_tot_em(is) = vflux_tot_em(is) + vflux_em(imod,ix,is) 
    end do !nx
  end do !nmod

  !kx_spectra
  if (lfluxes_spectra) then
    do ix = 1, nx
      pflux_xspec(ix,is) = 0. 
      eflux_xspec(ix,is) = 0. 
      vflux_xspec(ix,is) = 0.
      do imod = 1, nmod 
        pflux_xspec(ix,is) = pflux_xspec(ix,is) + pflux_es(imod,ix,is)
        eflux_xspec(ix,is) = eflux_xspec(ix,is) + eflux_es(imod,ix,is)
        vflux_xspec(ix,is) = vflux_xspec(ix,is) + vflux_es(imod,ix,is)
      end do !nmod
    end do !nx
  end if !lfluxes_spectra

  ! order the required fluxes for output
  iflux = 1
  if (lpflux) then
    flux_tot_es(iflux + (is-1)*nfluxes) = pflux_tot_es(is)
    flux_tot_em(iflux + (is-1)*nfluxes) = pflux_tot_em(is)
    iflux = iflux + 1
  end if
  if (leflux) then
    flux_tot_es(iflux + (is-1)*nfluxes) = eflux_tot_es(is)
    flux_tot_em(iflux + (is-1)*nfluxes) = eflux_tot_em(is)
    iflux = iflux + 1
  end if
  if (lvflux) then
    flux_tot_es(iflux + (is-1)*nfluxes) = vflux_tot_es(is)
    flux_tot_em(iflux + (is-1)*nfluxes) = vflux_tot_em(is)
    iflux = iflux + 1
  end if
end do !number_of_species


end subroutine fluxes

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine fluxes_det_original
!--------------------------------------------------------------------
!
! This routine calculates and outputs the "decomposed" fluxes of 
! particles, energy and parallel momentum, i.e. the integrals over 
! s, mu and vpar are not performed
!
! If the integrals are performed, the total fluxes are obtained
!   pflux = sum pflux_det*ds*dmu*dvpar
!   eflux = sum eflux_det*ds*dmu*dvpar
!   vflux = sum vflux_det*ds*dmu*dvpar
!
! Note: the neoclassical part is not included (could be done...)
! The electrostatic and electromagnetic contributions are grouped
! together
!--------------------------------------------------------------------
use control,    only : nlphi, nlapar
use grid,       only : nx, ns, nmu, nvpar, nsp, &
                     & nmod, isppb,number_of_species 
use dist,       only : fdisi, phi, get_phi,&
                     & indx, apar, get_apar, fmaxwl, nsolc 
use geom,       only : sgr, ints, bn, efun
use mode,       only : krho, kxrh
use components, only : tmp, vthrat, de, signz  
use functions, only  : besselj0_gkw 
use velocitygrid, only : intmu, intvp, mugr, vpgr
use io,           only : get_free_lun
! integers for the loop over all grid points 
integer imod, ix, i, j, k, is 

! The actual species index is in isglb
integer isglb

! other integers 
integer ierr, ipot
integer, dimension(MPI_STATUS_SIZE) :: state

! Dummy variables 
complex dum, dum1, dum2, fdis
real b0, ED
ierr =0
       allocate(pflux_det(nmod,nx,number_of_species,ns,nmu,nvpar),stat=ierr)
       if (ierr /= 0) then
         stop 'Could not allocate the array pflux_det in fluxes'
       end if
       allocate(eflux_det(nmod,nx,number_of_species,ns,nmu,nvpar),stat=ierr)
       if (ierr /= 0) then
         stop 'Could not allocate the array eflux_det in fluxes'
       end if
       allocate(vflux_det(nmod,nx,number_of_species,ns,nmu,nvpar),stat=ierr)
       if (ierr /= 0) then
         stop 'Could not allocate the array vflux_det in fluxes'
       end if    
! Copy phi from the solution. (note for runs without potential
! phi will be set to zero inside get_phi. For the coding below 
! this routine must be called even if phi is not kept in the 
! equations 
call get_phi(fdisi(1:nsolc),phi)

! copy apar from the solution (note for electrostatic runs apar 
! is set to zero inside get_apar. For the coding below this 
! routine must be called even for electrostatic cases 
call get_apar(fdisi(1:nsolc),apar)

!  Calculate the fluxes
do imod = 1, nmod 
  do ix = 1, nx 
    do is = 1, nsp

      ! the actual (global) species index 
      isglb = is + isppb  - 1 

      ! velocity space 
      do j = 1, nmu
        do k = 1, nvpar

          ! parallel coordinate 
          do i = 1, ns

            ! Bessel function for gyro-averaging 
            b0 =  besselj0_gkw(imod,ix,i,j,is) 

            ! fdis is the distribution without A_par contribution  
            fdis = fdisi(indx(imod,ix,i,j,k,is)) 
            if (nlapar) then 
               fdis = fdis - 2.E0*signz(is)*vpgr(i,j,k)*vthrat(is)*   & 
                    & b0*apar(imod,ix,i)*fmaxwl(i,j,k)/tmp(is)  
            end if 

            dum  = 2.E0*(efun(i,1,1)*kxrh(ix) +                       &
                 & efun(i,2,1)*krho(imod))*bn(i)*fdis

            dum1 = dum*b0*conjg(phi(imod,ix,i))                       &
                   & -2.E0*vthrat(is)*vpgr(i,j,k)*dum*b0*             &
                   & conjg(apar(imod,ix,i))

            dum2 = dum*b0*conjg(phi(imod,ix,i))*bn(i)                 &
                   &-2.E0*vthrat(is)*vpgr(i,j,k)*dum*b0*              &
                   & conjg(apar(imod,ix,i))*bn(i)

            pflux_det(imod,ix,isglb,i,j,k) = aimag(dum1) 
            eflux_det(imod,ix,isglb,i,j,k) =                          &
                        & vpgr(i,j,k)**2*aimag(dum1) +                 &
                        & 2.E0*mugr(j)*aimag(dum2)
            vflux_det(imod,ix,isglb,i,j,k)= aimag(dum1)*vpgr(i,j,k) 

          end do
        end do
      end do 
    end do 
  end do 
end do 


!APS call write6_end('pflux_det',pflux_det,TYPE_REAL)

! outputs in file (to be parallelized)
! 1) store the size of the arrays and the grids
      call get_free_lun(ipot)
#  if defined(mpi)
      call MPI_FILE_OPEN(MPI_COMM_WORLD,"int_grids.dat",MPI_MODE_WRONLY+MPI_MODE_CREATE,MPI_INFO_NULL,ipot,ierr)
      call MPI_FILE_WRITE_ALL(ipot,nmod,1,MPI_INTEGER,state,ierr)
      call MPI_FILE_WRITE_ALL(ipot,nx,1,MPI_INTEGER,state,ierr)
      call MPI_FILE_WRITE_ALL(ipot,nsp,1,MPI_INTEGER,state,ierr)
      call MPI_FILE_WRITE_ALL(ipot,nmu,1,MPI_INTEGER,state,ierr)
      call MPI_FILE_WRITE_ALL(ipot,nvpar,1,MPI_INTEGER,state,ierr)
      call MPI_FILE_WRITE_ALL(ipot,ns,1,MPI_INTEGER,state,ierr)

      call MPI_FILE_WRITE_ALL(ipot,krho,nmod,MPIREAL_X,state,ierr)
      call MPI_FILE_WRITE_ALL(ipot,kxrh,nx,MPIREAL_X,state,ierr)

      call MPI_FILE_WRITE_ALL(ipot,mugr(1:nmu),nmu,MPIREAL_X,state,ierr)
      call MPI_FILE_WRITE_ALL(ipot,vpgr(1:ns,1:nmu,1:nvpar),ns*nmu*nvpar,MPIREAL_X,state,ierr)
      call MPI_FILE_WRITE_ALL(ipot,sgr(1:nx,1:ns),nx*ns,MPIREAL_X,state,ierr)

      call MPI_FILE_WRITE_ALL(ipot,intmu,nmu,MPIREAL_X,state,ierr)
      call MPI_FILE_WRITE_ALL(ipot,intvp,ns*nmu*nvpar,MPIREAL_X,state,ierr)
      call MPI_FILE_WRITE_ALL(ipot,ints,nx*ns,MPIREAL_X,state,ierr)
      call MPI_FILE_CLOSE(ipot,ierr)
#endif

! 2) store the fluxes arrays
      call get_free_lun(ipot)
#  if defined(mpi)
      call MPI_FILE_OPEN(MPI_COMM_WORLD,"fluxes_det.dat",MPI_MODE_WRONLY+MPI_MODE_CREATE,MPI_INFO_NULL,ipot,ierr)
      call MPI_FILE_WRITE_ALL(ipot,pflux_det,nmod*nx*nsp*ns*nmu*nvpar,MPIREAL_X,state,ierr)
      call MPI_FILE_WRITE_ALL(ipot,eflux_det,nmod*nx*nsp*ns*nmu*nvpar,MPIREAL_X,state,ierr)
      call MPI_FILE_WRITE_ALL(ipot,vflux_det,nmod*nx*nsp*ns*nmu*nvpar,MPIREAL_X,state,ierr)
      call MPI_FILE_CLOSE(ipot,ierr)
#endif

! deallocate temp array

end subroutine fluxes_det_original

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> This routine calculates and outputs the "decomposed" fluxes of particles,
!> energy and parallel momentum, i.e. the integrals over s, mu and vpar are
!> not performed.
!>
!> If the integrals are performed, the total fluxes obtained are
!>   pflux = sum pflux_det*ds*dmu*dvpar
!>   eflux = sum eflux_det*ds*dmu*dvpar
!>   vflux = sum vflux_det*ds*dmu*dvpar
!>
!> Note: the neoclassical part is not included (could be done...)
!> The electrostatic and electromagnetic contributions are grouped together.
!----------------------------------------------------------------------------

subroutine fluxes_det_parallel

  use grid,       only : n_vpar_grid, n_s_grid, n_mu_grid, number_of_species
  use control,    only : nlphi, nlapar
  use grid,       only : nx, ns, nmu, nvpar, nsp, &
                     & nmod, isppb, ispb, ivparpb, imupb
  use dist,       only : fdisi, phi, get_phi,&
                     & indx, apar, get_apar, fmaxwl, nsolc 
  use geom,       only : sgr, ints, bn, efun
  use mode,       only : krho, kxrh
  use components, only : tmp, vthrat, de, signz  
  use functions, only  : besselj0_gkw 
  use velocitygrid, only : intmu, intvp, mugr, vpgr

  ! lun
  integer :: lun

  ! data representation
  character (len=6), parameter :: datarep = 'native'

  ! integers for the loop over all grid points 
  integer :: imod, ix, i, j, k, is

  ! The actual species index is in isglb
  integer :: isglb

  ! other integers 
  integer :: ierr, ipot
#if defined(mpi)
  integer (KIND=MPI_OFFSET_KIND) :: idisp
  integer, dimension(MPI_STATUS_SIZE) :: state
#else
  integer :: idisp
#endif
  ! Dummy variables 
  complex :: dum, dum1, dum2, fdis
  real :: b0, ED

  ! new type
  integer :: my_subarray_type
  
  ! ndims for arrays
  integer, parameter :: ndims = 6
  
  ! arrays of grid sizes
  integer, dimension(ndims) :: g_array_sizes, l_array_sizes, l_array_start

  ierr=0
  allocate(pflux_det(nvpar,nmu,ns,nx,nmod,nsp),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: pflux_det')
  allocate(eflux_det(nvpar,nmu,ns,nx,nmod,nsp),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: eflux_det')
  allocate(vflux_det(nvpar,nmu,ns,nx,nmod,nsp),stat=ierr)
  if (ierr /= 0) call gkw_abort('diagnostic :: vflux_det')
 
  ! global and local array sizes
  g_array_sizes = (/ n_vpar_grid, n_mu_grid, n_s_grid, nx, nmod,             &
      &              number_of_species /)
  l_array_sizes = (/ nvpar, nmu, ns, nx, nmod, nsp /)
  
  ! offset of the local array in the global array; start from 0
  l_array_start = (/ ivparpb - 1, imupb - 1, ispb - 1, 0, 0, isppb - 1 /)
#if defined(mpi)
  call MPI_TYPE_CREATE_SUBARRAY(6,g_array_sizes,l_array_sizes,l_array_start, &
      &  MPI_ORDER_FORTRAN, MPIREAL_X, my_subarray_type, ierr)
  call MPI_TYPE_COMMIT(my_subarray_type, ierr)
#endif
  ! Copy phi from the solution. (note for runs without potential phi will be
  ! set to zero inside get_phi. For the coding below this routine must be
  ! called even if phi is not kept in the equations.
  call get_phi(fdisi(1:nsolc),phi)

  ! copy apar from the solution (note for electrostatic runs apar is set to
  ! zero inside get_apar. For the coding below this routine must be called
  ! even for electrostatic cases.
  call get_apar(fdisi(1:nsolc),apar)

  !  Calculate the fluxes
  do is = 1, nsp ; do imod = 1, nmod ; do ix = 1, nx ; do i = 1, ns
  
    ! velocity space
    do j = 1, nmu ; do k = 1, nvpar
    
      ! Bessel function for gyro-averaging 
      b0 =  besselj0_gkw(imod,ix,i,j,is) 

      ! fdis is the distribution without A_par contribution  
      fdis = fdisi(indx(imod,ix,i,j,k,is)) 
      if (nlapar) then 
        fdis = fdis - 2.*signz(is)*vpgr(i,j,k)*vthrat(is)*b0*apar(imod,ix,i)*&
             &           fmaxwl(i,j,k)/tmp(is)  
      end if

      dum  = 2.0*(efun(i,1,1)*kxrh(ix) + efun(i,2,1)*krho(imod))*bn(i)*fdis

      dum1 = dum*b0*conjg(phi(imod,ix,i)) - 2.*vthrat(is)*vpgr(i,j,k)*dum*b0*&
           &                                   conjg(apar(imod,ix,i))

      dum2 = dum*b0*conjg(phi(imod,ix,i))*bn(i) -2.*vthrat(is)*vpgr(i,j,k)*  &
           &                                        dum*b0*bn(i)*            &
           &                                        conjg(apar(imod,ix,i))

      pflux_det(k,j,i,ix,imod,is) = aimag(dum1)
      eflux_det(k,j,i,ix,imod,is) = vpgr(i,j,k)**2*aimag(dum1) + 2.*mugr(j)*  &
                                  &                                aimag(dum2)
      vflux_det(k,j,i,ix,imod,is) = aimag(dum1)*vpgr(i,j,k)
      
    end do ; end do
    
  end do ; end do ; end do ; end do 

! outputs in file (to be parallelized)
! 1) store the size of the arrays and the grids

#if defined(mpi)
  call MPI_FILE_OPEN(MPI_COMM_WORLD,                                         &
      &                             "int_grids.dat",                         &
      &                             MPI_MODE_WRONLY + MPI_MODE_CREATE,       &
      &                             MPI_INFO_NULL,                           &
      &                             lun,                                     &
      &                                ierr)

  ! N.B. idisp is not usually a 4-byte integer
  idisp = 0
  call MPI_FILE_SET_VIEW(lun,idisp,MPI_INTEGER,MPI_INTEGER,datarep,MPI_INFO_NULL,&
        & ierr)
        
  if (root_processor) then
    call MPI_FILE_WRITE(lun, n_vpar_grid,       1, MPI_INTEGER, state, ierr)
    call MPI_FILE_WRITE(lun, n_mu_grid,         1, MPI_INTEGER, state, ierr)
    call MPI_FILE_WRITE(lun, n_s_grid,          1, MPI_INTEGER, state, ierr)
    call MPI_FILE_WRITE(lun, nx,                1, MPI_INTEGER, state, ierr)
    call MPI_FILE_WRITE(lun, nmod,              1, MPI_INTEGER, state, ierr)
    call MPI_FILE_WRITE(lun, number_of_species, 1, MPI_INTEGER, state, ierr)
!    call MPI_FILE_GET_POSITION(lun, offset, ierr)
  end if
  call mpigetfiledisp(lun,0,idisp,MPI_COMM_WORLD)
  call mpibarrier()
!  call mpi_bcast(idisp, 1,MPI_INTEGER, 0, MPI_COMM_WORLD,ierr)
  
  call MPI_FILE_SET_VIEW(lun,idisp,MPIREAL_X,MPIREAL_X,datarep,MPI_INFO_NULL,&
        & ierr)
  
  if (root_processor) then
    call MPI_FILE_WRITE(lun, krho(1),           nmod, MPIREAL_X,  state, ierr)
    call MPI_FILE_WRITE(lun, kxrh(1),             nx, MPIREAL_X,  state, ierr)
  end if

  call mpigetfiledisp(lun,0,idisp)
  call mpibarrier()

 
  ! write reals
  call MPI_FILE_SET_VIEW(lun,idisp,MPIREAL_X,MPIREAL_X,datarep,MPI_INFO_NULL,&
        & ierr)
        
  ! For now, write the values on the root processor as we need to (find a way
  ! to) pick several sets of processors to write the various full grids.
  if (root_processor) then
    call MPI_FILE_WRITE(lun, mugr(1:nmu),nmu,MPIREAL_X,state,ierr)
    call MPI_FILE_WRITE(lun,vpgr(1:ns,1:nmu,1:nvpar),ns*nmu*nvpar,MPIREAL_X,state,ierr)
    call MPI_FILE_WRITE(lun,sgr(1:nx,1:ns),nx*ns,MPIREAL_X,state,ierr)
    call MPI_FILE_WRITE(lun,intmu,nmu,MPIREAL_X,state,ierr)
    call MPI_FILE_WRITE(lun,intvp,ns*nmu*nvpar,MPIREAL_X,state,ierr)
    call MPI_FILE_WRITE(lun,ints,nx*ns,MPIREAL_X,state,ierr)
  end if

  ! update the displacement value for all processors
  call mpigetfiledisp(lun,0,idisp)
  call mpibarrier()
  
  call MPI_FILE_CLOSE(lun,ierr)
#endif

  ! 2) store the fluxes arrays
#if defined(mpi)

  ! open file on all processors
  call MPI_FILE_OPEN(MPI_COMM_WORLD,                                         &
      &                             "fluxes_det.dat",                        &
      &                             MPI_MODE_WRONLY + MPI_MODE_CREATE,       &
      &                             MPI_INFO_NULL,                           &
      &                             lun,                                     &
      &                                ierr)

  ! set the file view
  idisp = 0
  call MPI_FILE_SET_VIEW(lun,idisp,MPIREAL_X,my_subarray_type,datarep,      &
      &                  MPI_INFO_NULL,ierr)
  ! write the fluxes arrays in parallel
  call MPI_FILE_WRITE_ALL(lun,pflux_det,nmod*nx*nsp*ns*nmu*nvpar,MPIREAL_X,state,ierr)
  call MPI_FILE_WRITE_ALL(lun,eflux_det,nmod*nx*nsp*ns*nmu*nvpar,MPIREAL_X,state,ierr)
  call MPI_FILE_WRITE_ALL(lun,vflux_det,nmod*nx*nsp*ns*nmu*nvpar,MPIREAL_X,state,ierr)

  call mpigetfiledisp(lun,0,idisp)
  call mpibarrier()
  call MPI_FILE_CLOSE(lun,ierr)
  
#endif

end subroutine fluxes_det_parallel

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine phi_xy_output(fdis)

use general, only : gkw_abort
use grid, only : nmod, nx, ns, nsp,nmu,nvpar, n_s_grid,ispb
use control, only : output3d
use mpicomms, only : COMM_S_EQ
!!use dist,    only : nsolc, indx, get_phi, phi
use dist,    only : nsolc, indx, get_phi, phi, get_bpar, fdisi
use geom,    only : kthnorm, q, eps
use mode,    only : krho, kxrh, lx, ly
use non_linear_terms, only : jind, jinv, mrad, mphi, a, ar, &
                           & nl_initialised
use rotation, only : perp_shear
USE fft,      ONLY : four2D_real
implicit none 

complex, intent(in)    :: fdis(nsolc) 
integer i,j,imod,ix,ipar,ic,ispc,ispa,iproc_s,iproc_write,ipar_global 
character*9 filename
logical, save :: initialised = .false.
logical, save :: lwrite = .false.

save :: ispc, ispa  

  ! global s index; could be a diagnostic namelist to input this
  !This selects which slice we wish to output
  ipar_global = n_s_grid/2 + 1

  !local s index
  !This selects where the chosen slice is relative to local processor
  !But may not be on the local processor
  ipar = ipar_global - ispb + 1
  
! Check if the initialization has been done 
! Note this intialisation is now used by many other routines
! It should be a seperate routine with a module flag intialised
if (.not.initialised) then 
  if (.not.nl_initialised) then 
     call gkw_abort ('phi_xy_output: First call nonlinear_init')
  end if 
  
  if (lfile_output) then
    open(13, file = 'xphi')
    open(15, file = 'yphi')
    do i = 1, mphi 
      write(15,11) (real(i-1)*ly/real(mphi), j = 1, mrad)
      write(13,11) (real(j-1)*lx/real(mrad), j = 1, mrad) 
    end do 
    close(13)
    close(15)
    open(13, file = 'kxrh')
    open(15, file = 'krho') 
    do imod = 1, nmod 
      write(13,11)(kxrh(ix),ix = 1, nx)
      write(15,11)(krho(imod)*kthnorm, ix = 1, nx) 
    end do 
    close(13)
    close(15)
  
  end if !last_processor

  ispc = 0 
  ispa = 0 
  
  !Is the chosen slice on this processor?
  if (ipar >= 1 .and. ipar <= ns) then
    ! Select a processor that will write the slice by taking the processor with
    ! the greatest rank from those responsible for the same points in s.
    ! This is done via the COMM_S_EQ communicator; again
    ! this can all be done before any slices are made. N.B. this can result in
    ! several processors being selected, depending on parallelization.
    call mpicomm_rank(COMM_S_EQ,iproc_s)
    !Choose maximum ranked processor at this s slice to do the write
    !Since all the processors have the same potential
    call mpiallreduce_max(iproc_s,iproc_write,1,COMM_S_EQ)
    lwrite = (iproc_s == iproc_write)
  else
    lwrite = .false.
  end if
  ! Initialisation done
  initialised = .true. 

end if !Intialisation


  ! Only a subset of the processors processors need take any further part here
  ! i.e. those responsible for the point ipar_global in the s grid (assuming
  ! we have set up the appropriate communicators, otherwise this is dangerous).
  !  if (ipar < 1 .or. ipar > ns) return
 
  if (.not. lwrite) return
  
  !obtain phi in a separate array 
  call get_phi(fdis,phi)
  !!to obtain b_par in a separate array:
  !!call get_bpar(fdisi,phi) 
  call genfilename('spc',filename,ispc)
  open(13,file = filename)
  do imod = 1, nmod 
    write(13,11)(abs(phi(imod,ix,ipar)),ix = 1, nx) 
  end do  
  close(13)

  a = (0.,0.)
  ar = 0.

  ! fill the array for the potential 
  do imod = 1, nmod 
    do ix = 1, nx
       !This array is used from non_linear_terms to save memory?
       !It should only be used here at different times!
       a(imod,jind(ix)) = phi(imod,ix,ipar)
    end do 
  end do   

  !Do the inverse FFT 
   CALL four2D_real(ar,a,1)

  ! temporary output of phi 
  call genfilename('phi',filename,ispa)  
  open(13,file=filename) 
  do i = 1, mphi 
    write(13,11)(ar(i,j), j = 1, mrad)
    11 format(512(1pe13.5,1x))
  end do 
  close(13) 

end subroutine phi_xy_output

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine apar_xy_output(fdis)

use general, only : gkw_abort
use grid, only : nmod, nx, ns, nsp,nmu,nvpar, n_s_grid,ispb
use control, only : output3d
use mpicomms, only : COMM_S_EQ
use dist,    only : nsolc, indx, get_phi, phi, get_apar, apar
use geom,    only : kthnorm, q, eps
use mode,    only : krho, kxrh, lx, ly
use non_linear_terms, only : jind, jinv, mrad, mphi, a, ar, &
                           & nl_initialised
use rotation, only : perp_shear
USE fft,      ONLY : four2D_real
implicit none 

complex, intent(in)    :: fdis(nsolc) 
integer i,j,imod,ix,ipar,ic,ispc,ispa,iproc_s,iproc_write,ipar_global 
character*9 filename
logical, save :: initialised = .false.
logical, save :: lwrite = .false.

save  :: ispc, ispa  

  ! global s index; could be a diagnostic namelist to input this
  !This selects which slice we wish to output
  ipar_global = n_s_grid/2 + 1

  !local s index
  !This selects where the chosen slice is relative to local processor
  !But may not be on the local processor
  ipar = ipar_global - ispb + 1
  
! Check if the initialization has been done 
! Note this intialisation is now used by many other routines
! It should be a seperate routine with a module flag intialised
if (.not.initialised) then 
  if (.not.nl_initialised) then 
     call gkw_abort ('apar_xy_output: First call nonlinear_init')
  end if 
  
!   This is already output by phi_xy_output
!   if (lfile_output) then
!     open(13, file = 'xphi')
!     open(15, file = 'yphi')
!     do i = 1, mphi 
!       write(15,11) (real(i-1)*ly/real(mphi), j = 1, mrad)
!       write(13,11) (real(j-1)*lx/real(mrad), j = 1, mrad) 
!     end do 
!     close(13)
!     close(15)
!     open(13, file = 'kxrh')
!     open(15, file = 'krho') 
!     do imod = 1, nmod 
!       write(13,11)(kxrh(ix),ix = 1, nx)
!       write(15,11)(krho(imod)*kthnorm, ix = 1, nx) 
!     end do 
!     close(13)
!     close(15)
!   
!   end if !last_processor

  ispc = 0 
  ispa = 0 
  
  !Is the chosen slice on this processor?
  if (ipar >= 1 .and. ipar <= ns) then
    ! Select a processor that will write the slice by taking the processor with
    ! the greatest rank from those responsible for the same points in s.
    ! This is done via the COMM_S_EQ communicator; again
    ! this can all be done before any slices are made. N.B. this can result in
    ! several processors being selected, depending on parallelization.
    call mpicomm_rank(COMM_S_EQ,iproc_s)
    !Choose maximum ranked processor at this s slice to do the write
    !Since all the processors have the same potential
    call mpiallreduce_max(iproc_s,iproc_write,1,COMM_S_EQ)
    lwrite = (iproc_s == iproc_write)
  else
    lwrite = .false.
  end if
  ! Initialisation done
  initialised = .true. 

end if !Intialisation


  ! Only a subset of the processors processors need take any further part here
  ! i.e. those responsible for the point ipar_global in the s grid (assuming
  ! we have set up the appropriate communicators, otherwise this is dangerous).
  !  if (ipar < 1 .or. ipar > ns) return
 
  if (.not. lwrite) return
  
  !obtain phi in a separate array 
  call get_apar(fdis,apar) 
  call genfilename('sac',filename,ispc)
  open(13,file = filename)
  do imod = 1, nmod 
    write(13,11)(abs(apar(imod,ix,ipar)),ix = 1, nx) 
  end do  
  close(13)

  a = (0.,0.)
  ar = 0.

  ! fill the array for the potential 
  do imod = 1, nmod 
    do ix = 1, nx
       !This array is used from non_linear_terms to save memory?
       !It should only be used here at different times!
       a(imod,jind(ix)) = apar(imod,ix,ipar)
    end do 
  end do   

  !Do the inverse FFT 
  CALL four2D_real(ar,a,1)

  ! temporary output of phi 
  call genfilename('apa',filename,ispa)  
  open(13,file=filename) 
  do i = 1, mphi 
    write(13,11)(ar(i,j), j = 1, mrad)
    11 format(512(1pe13.5,1x))
  end do 
  close(13) 

end subroutine apar_xy_output

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine xy_output(fdis,s_point,mu_point,vpar_point,species)

use general, only : gkw_abort
use grid, only : nmod, nx, ns, nsp,nmu,nvpar
use dist,    only : nsolc, indx, get_phi, phi
use geom,    only : kthnorm
use non_linear_terms, only : jind, jinv, mrad, mphi, a, ar, &
                           & nl_initialised
USE fft,      ONLY : four2D_real

implicit none 

complex, intent(in)    :: fdis(nsolc) 
integer, intent(in)    :: mu_point,vpar_point,species, s_point
integer i,j,imod,ix,ipar,ic,ispc,ispa 
character*9 filename
logical, save :: initialised = .false.

save :: ispc, ispa  

! Check if the initialization has been done 
if (.not.initialised) then 
  if (.not.nl_initialised) then 
     call gkw_abort ('xy_output: First call nonlinear_init')
  end if 
  
  ispc = 0 
  ispa = 0 
  
  ! Initialisation done   
  initialised = .true. 

end if 

do i=1,ns
  do imod = 1, nmod 
    do ix = 1, nx
        phi(imod,ix,i)=fdis(indx(imod,ix,i,mu_point,vpar_point,species))
    end do
  end do
end do

! Select the point on the field line for the plot  
do ipar = s_point, s_point 

  call genfilename('spc',filename,ispc)
  open(13,file = filename) 
  do imod = 1, nmod 
    write(13,11)(abs(phi(imod,ix,ipar)),ix = 1, nx) 
  end do  
  close(13)

  a = (0.,0.)
  ar = 0.

  do imod = 1, nmod 
    do ix = 1, nx
       !This array is used from non_linear_terms to save memory?
       !It should only be used here at different times!
       a(imod,jind(ix)) = phi(imod,ix,ipar)
    end do 
  end do   

  !Do the inverse FFT
   CALL four2D_real(ar,a,1)   

  ! temporary output of fdis 
  call genfilename('fdis',filename,ispa)  
  open(13,file=filename) 
  do i = 1, mphi 
    write(13,11)(a(i,j), j = 1, mrad)
    11 format(512(1pe13.5,1x))
  end do 
  close(13) 
end do 

!Just in case any other routines are expecting the potential to be
!in the array phi, put phi in its separate array 
call get_phi(fdis,phi) 

return 
end subroutine xy_output 

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine phi_ky_spec(fdis) 
!Now works with parallel_s
use grid, only : nmod, nx, n_s_grid, ispb, ispe
use dist,    only : get_phi, nsolc, phi 
use mpicomms, only : COMM_S_NE

implicit none 

! The distribution 
complex, intent(in) :: fdis(nsolc)

 
integer imod, ix, is, ierr 
logical, save ::  initialised  = .false.

if (.not.initialised) then 
  
  ierr = 0 
  allocate(ky_spec(nmod), stat = ierr)
  if (ierr /= 0) then 
    stop 'could not allocate ky_spec in phi_ky_spec'
  end if 
  
  ierr = 0 
  allocate(kx_spec(nx), stat = ierr)
  if (ierr /= 0) then 
    stop 'could not allocate kx_spec in phi_ky_spec'
  end if 

  ierr = 0 
  allocate(buffer(max(nx,nmod)), stat = ierr)
  if (ierr /= 0) then 
    stop 'could not allocate buffer in phi_ky_spec'
  end if 

  initialised = .true. 

end if 

call get_phi(fdis,phi) 

!kyspec
do imod = 1, nmod 
  buffer(imod) = 0.
  do is = 1, n_s_grid
    if(is >= ispb .and. is <= ispe) then !For parallel_s
      do ix = 1, nx 
        buffer(imod) = buffer(imod) + abs(phi(imod,ix,is-ispb+1))**2 
      end do 
    end if
  end do 
end do 

!PARALLEL_S
#if defined(mpi)
  call MPI_ALLREDUCE(buffer(1:nmod),ky_spec,nmod,MPIREAL_X,MPI_SUM,COMM_S_NE,ierr)
#else
  ky_spec(:)=buffer(1:nmod)
#endif

!Rescale to be irrespective of number of grid points
ky_spec(:)=ky_spec(:)/(nx*n_s_grid)

!kxspec, buffer is reused
do ix = 1, nx 
  buffer(ix) = 0.
  do is = 1, n_s_grid 
    if(is >= ispb .and. is <= ispe) then !For parallel_s
      do imod = 1, nmod 
        buffer(ix) = buffer(ix) + abs(phi(imod,ix,is-ispb+1))**2 
      end do
    end if 
  end do 
end do 

!PARALLEL_S
#if defined(mpi)
  call MPI_ALLREDUCE(buffer(1:nx),kx_spec,nx,MPIREAL_X,MPI_SUM,COMM_S_NE,ierr)
#else
  kx_spec(:)=buffer(1:nx)
#endif

!Rescale to be irrespective of number of grid points
kx_spec(:)=kx_spec(:)/(nmod*n_s_grid)

if (lfile_output) then
    write(i_kyspec,fmt = '(256(1pe13.5,1x))')(ky_spec(imod), imod = 1, nmod)
    write(i_kxspec,fmt = '(1024(1pe13.5,1x))')(kx_spec(ix), ix = 1, nx)
end if

return 
end subroutine phi_ky_spec

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine binarypotentialoutput(fdis)

  use mpiinterface
  use general,          only : gkw_abort
  use io,               only : get_free_lun
  use grid,          only : nmod,nx,ns,nsp,n_procs_s,     &
      &                        iproc_s,n_s_grid,parallel_s
  use control,          only : ntotstep
  use mpicomms,         only : COMM_S_NE
  use dist,             only : nsolc,indx,get_phi,phi,apar,get_apar   
  use geom,             only : kthnorm,sgr
  use mode,             only : krho,kxrh,lx,ly 
  use non_linear_terms, only : jind,jinv,mrad,mphi,a,ar,nl_initialised
  use FFT,              only : four2D_real

  complex, dimension(nsolc), intent(in) :: fdis
  
  integer, dimension(MPI_STATUS_SIZE) :: state
  integer :: i,j,k,imod,ix,ipar,ic,i_sign,ispc,ispa,ierr,ilun,ipot,istart,ntype
  integer :: record_length, icount
  character (len=9) :: filename
  logical :: initialised = .false.
  logical :: lam_with_root
  save :: initialised, lam_with_root,ntype

  !> Used when the whole three dimension potential is selected for output to a
  !> binary file -> This is to be extended to the density and other parameters.
  real, allocatable, dimension(:,:,:) :: datbuffer

  if (lverbose) write(*,*) '* Output has been called'
  
  if (.not. initialised) then
    if (lverbose) write (*,*) '* first call - initialising' 
    lam_with_root = .false.
#if defined(mpi)
    if (parallel_s) then
      call MPI_ALLREDUCE(root_processor,lam_with_root,1,MPI_LOGICAL,MPI_LOR,COMM_S_NE,ierr)
      if (lam_with_root) then
        istart = ns*iproc_s
        call MPI_TYPE_CREATE_SUBARRAY(1,n_s_grid,ns,istart,MPI_ORDER_FORTRAN,MPIREAL_X,NTYPE,ierr)
        call MPI_TYPE_COMMIT(NTYPE,ierr)
      end if
      call mpibarrier()
    end if
#endif

    ! Write the values of s to a file
#if defined(GNU)
    ! do what?
#else
    call get_free_lun(ilun)
    if (parallel_s) then
      if (lam_with_root) then
#if defined(mpi)
        call MPI_FILE_OPEN(COMM_S_NE,'SPoints.dat',MPI_MODE_WRONLY + MPI_MODE_CREATE,MPI_INFO_NULL,ilun,ierr)
        call MPI_FILE_SET_VIEW(ilun, 0, MPIREAL_X, NTYPE, "native", MPI_INFO_NULL, ierr)
        call MPI_FILE_WRITE_ALL(ilun,sgr(1,1:ns),ns,MPIREAL_X,state,ierr)
        call MPI_FILE_CLOSE(ilun,ierr)
#endif
      end if
    else
      if (root_processor) then
        inquire(iolength=record_length) sgr(1,1:ns)
        open (unit=ilun,file='SPoints.dat',access='direct',recl=record_length)
        write (ilun,rec=1) sgr(1,1:ns)
      end if
    end if
#endif

    if (root_processor) then
      open (unit=ilun,file='Frames.dat',status='replace')
      close(ilun)
    end if
    initialised = .true.
  else
    ! nothing
  end if

  allocate(datbuffer(ns,mphi,mrad),stat=ierr)
  if (ierr /= 0) then 
    call gkw_abort('binarypotentialoutput: Could not allocate datbuffer')
  end if  

  ! obtain phi in a separate array 
  call get_phi(fdis,phi) 
     
  ! Select the point on the field line for the plot  
  do ipar = 1,ns 

    ! fill the array for the potential 
    a = (0.,0.)
    ar = 0.

    do imod = 1, nmod
      do ix = 1, nx
         !This array is used from non_linear_terms to save memory?
         !It should only be used here at different times!
         a(imod,jind(ix)) = phi(imod,ix,ipar)
      end do 
    end do   

    !Do the inverse FFT   
    CALL four2D_real(ar,a,1)   

    do i=1,mphi
      do j=1,mrad
           datbuffer(ipar,i,j)=a(i,j)
        end do
    end do

  end do

  !if (lverbose) write(*,*) mphi, mrad

  if (parallel_s) then
    if (lam_with_root) then
      call get_free_lun(ipot)
      call genfilename('Poten',filename,ntotstep)

#if defined(mpi)
      call MPI_FILE_OPEN(COMM_S_NE,filename,MPI_MODE_WRONLY+MPI_MODE_CREATE,MPI_INFO_NULL,ipot,ierr)
      call MPI_FILE_SET_VIEW(ipot, 0, MPIREAL_X, NTYPE, "native", MPI_INFO_NULL, ierr)
      do k=1,mphi ; do j=1,mrad
        call MPI_FILE_WRITE_ALL(ipot,datbuffer(1:ns,k,j),ns,MPIREAL_X,state,ierr)
      enddo ; enddo
      call MPI_FILE_CLOSE(ipot,ierr)

#endif
    end if
  else
#if defined(GNU)
    ! do what?
#else
     icount = 0
     
     if (root_processor) then
        icount = 0
        call get_free_lun(ipot)
        inquire(iolength=record_length) datbuffer(1:ns,1,1)
        call genfilename('Poten',filename,ntotstep)
        open (unit=ipot,file=filename,form='unformatted',access='direct',recl=record_length)
        do k=1,mphi
           do j=1,mrad
              icount  = icount + 1
              write (ipot,rec=icount)  (datbuffer(i,k,j),i=1,ns)
           end do
        end do
        close(ipot)
     end if
#endif
  end if

  if (root_processor) then
    call get_free_lun(ilun)
    open (unit=ilun,file='Frames.dat',position='append')
    write(ilun,*) filename
    close(ilun)
  end if

  deallocate(datbuffer)

end subroutine binarypotentialoutput

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine binarypotentialoutputlegacy(fdis)

  use io,      only : get_free_lun
  use general, only : gkw_abort
  use grid,    only : nmod, nx, ns, nsp
  use control, only : ntotstep 
  use dist,    only : nsolc, indx, get_phi, phi   
  use geom,    only : kthnorm,sgr
  use mode,    only : krho, kxrh, lx, ly 
  use non_linear_terms, only : jind, jinv, mrad, mphi,nl_initialised,a,ar
  use fft,              only : four2D_real

  complex, dimension(nsolc), intent(in) :: fdis
  integer :: i,j,k,imod,ix,ipar,ic,isign,ispc,ispa,ierr,record_length
  character (len=13) :: filename
  logical, save :: initialised = .false.
  save :: ispc, ispa
  integer :: lun, icount

  if (.not. root_processor) return

  ! obtain phi in a separate array 
  call get_phi(fdis,phi) 


  ! Select the point on the field line for the plot  
  do ipar = 1,ns
   
    ! fill the array for the potential 
    a = (0.,0.)
    ar = 0.
    do imod = 1, nmod
      do ix = 1, nx
        a(imod,jind(ix)) = phi(imod,ix,ipar)
      end do
    end do
   
    !Do the inverse FFT   
    CALL four2D_real(ar,a,1)   
   
    ! copy slice into write buffer
    do i=1,mphi
      do j=1,mrad
        datbuffer(ipar,j,i)=ar(i,j)
      end do
    end do
   
  end do
  
  icount = 0

  ! write the file and frame
  if (root_processor) then
    call get_free_lun(lun)
    call genfilename('Poten',ntotstep,len(filename),filename)
    inquire(iolength=record_length) datbuffer(1:ns,1,1)
    open (unit=lun,file=filename,form='unformatted',access='direct',&
       & recl=record_length)
    write(i_potframe,*) filename
    do k=1,mphi
      do j=1,mrad
         icount = icount + 1
        write (lun,rec=icount) (datbuffer(i,j,k),i=1,ns)
      end do
    end do
    close(lun)
  end if

end subroutine binarypotentialoutputlegacy

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

!> Write a header so that parallel.dat can be produced from several output
!> files. N.B. the construction routines will need to be modified in
!> conjunction with any changes made here. 
subroutine write_parallel_header(ifile)

  use grid, only : nx, ns, nmod, number_of_species, iproc_s, n_procs_s

  integer, intent(in) :: ifile

  write (ifile,*) 'PROC_S = ',iproc_s
  write (ifile,*) 'NS = ',ns
  write (ifile,*) 'NTOT = ',nx*nmod*number_of_species

end subroutine write_parallel_header

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Calculates the preserved quantity abs value squared of an array.
!> Useful for stability dignostics on linear runs if not normalised.
!----------------------------------------------------------------------------

function norm(f)

  complex, dimension(:), intent(in) :: f
  real :: norm
  real :: abs_sq
  integer :: ierr

  abs_sq = sum(abs(f)**2)

#ifdef mpi
  ierr = 0
  call MPI_ALLREDUCE(abs_sq,norm,1,MPIREAL_X,MPI_SUM,MPI_COMM_WORLD,ierr)
#else
  norm = abs_sq
#endif

end function norm

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> This routine is called by all processors to (1) find a single processor
!> (global_proc) which has some logical (match_proc) = T on it, then (2)
!> propagate this match to the processors in a slice defined by the
!> communicators COMM1 and COMM2 (in_slice=T).
!----------------------------------------------------------------------------

subroutine get_common_procs_2d(match_proc,global_proc,in_slice,COMM1,COMM2)

  integer, intent(in) :: COMM1,COMM2
  integer, intent(out) :: global_proc
  logical, intent(in) :: match_proc
  logical, intent(out) :: in_slice
  
  integer :: i
  logical :: seed1,seed2,match_tmp
  
  ! find the first processor where match_proc = T
  global_proc = -1
  findproc : do i=0,number_of_processors-1
    match_tmp = (processor_number == i .and. match_proc)
    call mpibcast_logical(match_tmp,1,IPROC=i)
    if (match_tmp) then
      global_proc = i
      seed1 = (processor_number == global_proc)
      exit findproc
    end if
  end do findproc

  ! Could be that match_proc was F on all procs
  if (global_proc < 0) then
    call gkw_warn('get_common_procs_2d: no matched proc; returning')
    ! return proper values
    in_slice = .false.
    return
  end if
  
  ! find the other processors in the first direction
  call mpiallreduce_or(seed1,seed2,1,COMM1)
  ! find the other processors in the second direction
  call mpiallreduce_or(seed2,in_slice,1,COMM2)
  
end subroutine get_common_procs_2d


!****************************************************************************
!****************************************************************************
      
end module diagnostic
