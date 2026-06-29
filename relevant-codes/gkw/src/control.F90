!! SVN:  $Id: control.F90 1014 2009-07-02 18:42:39Z  $
!> Control contains all the switches of the code, as well as the basic
!> grid layout over the processors. Control is the top most main code
!> module (above only mpiinterface containing the basic mpi parameters, and
!> general, which contains general purpose routines).
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
module control

  use global
  
  implicit none

  private

  !
  ! publicly available procedures
  !
  
  public :: control_init, control_initt, control_read_nml, control_bcast_nml
  public :: control_check_params, control_write_nml

  !
  ! publicly available variables
  !
  
  !> True if collisions are to be used
  logical, public :: collisions
  !< True if the neoclassical effects are to be calculated 
  logical, public :: neoclassics
  !> True if the zonal flows are used in the equation for the adiabatic response
  logical, public :: zonal_adiabatic
  !< true for including the nonlinear terms
  logical, public :: non_linear
  !> True if the electrostatic potential is kept in the equations
  logical, public :: nlphi
  !> True if A|| is kept in the equations 
  logical, public :: nlapar
  !> True if B|| is kept in the equations 
  logical, public :: nlbpar
  !> True if the code decides it needs to stop 
  logical, public :: stop_me = .false.
  !> True if the nonlinear timestep estimator is to be used (nonlinear only) 
  logical, public :: nl_dtim_est
  !> Disipation / upwind parameter for the parallel (to the field) derivatives
  real, public :: disp_par 
  !> Dissipation / upwind parameter for the parallel velocity 
  real, public :: disp_vp
  !> Perpendicular dissipation in x
  real, public :: disp_x
  !> Perpendicular dissipation in y
  real, public :: disp_y

  !> Integer that determines how the parallel velocity grid is set up. 
  !> If = 0, the parallel velocity of a grid point is constant along the field
  !> line and the grid is uniform. 
  !> If = 1, the parallel velocity follows the trapping condition. 
  integer, public :: vp_trap
  !> Switch for the parallel boundary conditions. Allowed are 'zero'
  !> 'periodic_no_shift' and 'zero_derivative'. Not all options work
  !> with all schemes, but they should all work with 'fourth_order'
  character (len = lenswitch), public :: parallel_boundary_conditions
  !> Selects the order of the numerical scheme (accuracy). Allowed are
  !> 'second_order' and 'fourth_order'
  character (len = lenswitch), public :: order_of_the_scheme
  !> Selects the format in which the matrix is stored. Allowed are 
  !> 'complex' and 'complex-real'
  character (len = lenswitch), public :: matrix_format
  !> True if a normalization is applied for the distrubtion function. Nonlinear
  !> runs are never normalized. 
  logical, public :: normalized
  !> Use Arakawa type differencing for trapping terms
  logical, public :: ltrapping_arakawa

  !> produce final output
  logical, public :: lfinal_output
  !> write small output
  logical, public :: lwrite_output1
  !> calculate fluxes
  logical, public :: lcalc_fluxes
  !> calculate mode frequency
  logical, public :: lcalc_freq
  !> potential output
  logical, public :: lphi_diagnostics
  !> read diagnostic namelist if true
  logical, public :: ldiagnostic_namelist
  !> Per timestep data output (fluxes or growth rates) to screen
  logical, public :: screen_output
  !> No printing of the timestep information
  logical, public :: silent
  !> Output full 3d potential if true
  logical, public :: output3d

  integer, public :: ntime            !< number of large timesteps
  integer, public :: naverage         !< number of small timesteps
  real, public :: dtim                !< Normalized timestep 
  real, public :: dtim_est            !< estimated timestep for nonlinear terms stability
  real, public :: dtim_est_save       !< keeps local minimum timestep for nonlinear terms
  real, public :: dtim_input          !< the original input time step
  character (len=lenswitch), public :: method  !< method for solving
  integer, public :: meth             !< choice of algorithm for method
  real, public :: time                !< Total time
  logical, public :: read_file        !< true to resume a run 
  logical, public :: testing          !< true for one of the testruns 
  integer, public :: ntotstep         !< total number of timesteps taken 
  real,    public :: max_seconds      !< Maximum number of second for a run 
  integer, public :: max_sec          !< Maximum number of second for a run (integer) 
  real,    public :: gamatol          !< The tolerance in gamma that stops the code. 
  real,    public :: dt_min           !< mimimum value of dt, below which the code stops. 

  !> restart file version
  integer, public :: restart_file_version
  !> current restart file version
  integer, parameter, public :: restart_file_current_version = 2

  !> run number; does not do anything 
  integer :: irun

  !
  ! interfaces
  !
  
  ! use one routine for reading and writing
  interface control_write_nml
    module procedure control_read_nml
  end interface
  
contains

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine control_init

  !call svn_id('$Id: control.F90 1014 2009-07-02 18:42:39Z  $')
  
end subroutine control_init
  
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Reads from the input file the various control switches; writes the
!> namelist to file if the optional switch is *not* present.
!----------------------------------------------------------------------------

subroutine control_read_nml(ilun,io_stat,lwrite)

  use mpiinterface, only : number_of_processors

  integer, intent(in)  :: ilun
  integer, intent(out) :: io_stat

  logical, optional, intent(in) :: lwrite

  namelist /control/ collisions, disp_par, disp_vp, disp_x, disp_y, dtim,    &
                  &  irun, lverbose, matrix_format, max_sec, meth, method,   &
                  &  naverage, neoclassics, nlapar, nlphi, non_linear,       &
                  &  nl_dtim_est, normalized, ntime, order_of_the_scheme,    &
                  &  parallel_boundary_conditions, read_file, silent,        &
                  &  testing, vp_trap, zonal_adiabatic, output3d,            &
                  &  lwrite_output1, lcalc_fluxes, dt_min,                   &
                  &  lcalc_freq,  lfinal_output,max_seconds,                 &
                  &  lphi_diagnostics, screen_output, ltrapping_arakawa,     &
                  &  ldiagnostic_namelist, restart_file_version, nlbpar, gamatol
     
  io_stat = 0
  
  ! read the input; this needs the switch
  if (present(lwrite)) then
    
    if (.not. lwrite) then
      ! Set the default values for the control parameters; the default
      ! corresponds to a linear run without resume.
      max_seconds       = -1.
      lverbose          = .false.
      lfinal_output     = .true.
      lphi_diagnostics  = .true.
      lwrite_output1    = .true.
      lcalc_fluxes      = .true.
      lcalc_freq        = .true.
      screen_output     = .true.
      non_linear        = .false.
      nl_dtim_est       = .true.
      read_file         = .false.
      testing           = .false.
      silent            = .false. 
      zonal_adiabatic   = .false.
      collisions        = .false. 
      nlphi             = .true. 
      nlapar            = .false.
      nlbpar            = .false.
      neoclassics       = .false. 
      normalized        = .true. 
      vp_trap           = 0 
      output3d          = .false.
      ltrapping_arakawa = .false.
      ldiagnostic_namelist = .false.
      restart_file_version = restart_file_current_version
      gamatol           = 0.
      dt_min            = 0.
    
      ! Also, the parallel boundary conditions are by default zero. 
      parallel_boundary_conditions = 'zero'
      ! The differentiation is by default 4th order 
      order_of_the_scheme = 'fourth_order'
      ! Set the default format of the matrix to complex related to numerical
      ! solution
      matrix_format = 'complex'
    
      irun       = 0 ! run number 
      ntime      = 0 ! number of large timesteps
      naverage   = 0 ! number of small time steps. Total number is ntime*naverage 
      dtim       = 0.005 ! time step 
      disp_par   = 0.2E0 ! The dissipation coefficient for parallel derivatives 
      disp_vp    = 0.2E0 ! The dissipation coefficient for parallel velocity space 
      disp_x     = 0.0E0 ! 'Radial' perpendicular dissipation coeffcient
      disp_y     = 0.0E0 ! 'Poloidal' perpendicular disspation coeffcient
      method     = 'EXP' ! method of integration
      meth       = 2     ! switch between different schemes
      max_sec    = -1    ! maximum seconds for a run (<=0) is infinite 
      ! initialize also the total number of steps taken 
      ntotstep = 0
    
      ! read namelist
      read(ilun,NML=control,IOSTAT=io_stat)
    else
      ! do nothing
    end if
    
  else

    ! write to input.out if called without the switch; this is the default
   
    ! Write the revision information generated using svnrev at compile time.
    ! APS: put the other information in here too
    ! APS: also put it in the HDF5 file
    write(ilun,*) '!Output data generated by GKW version ', GKW_REV
    write(ilun,*) '!  executable name: ',GKW_EXE
    write(ilun,*) '!  compiled with $(FC) '
    write(ilun,'(A,A)') ' !',GKW_FC
    write(ilun,*) '!Run on ', number_of_processors, 'processor(s)'
    write(ilun,*) '!If the input file generated warnings:'
    write(ilun,*) '!This file is a clean version that should not generate warnings'
    write(ilun,*) '!The following are the input variables are as used by the code:'  

    ! write the namelist
    write(ilun,NML=control)
    
  end if
  
end subroutine control_read_nml

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Broadcast control details to other processors
!----------------------------------------------------------------------------

subroutine control_bcast_nml

  use mpiinterface

  call mpibcast_integer(irun,1)
  call mpibcast_real(dt_min,1)
  call mpibcast_integer(ntime,1)
  call mpibcast_integer(naverage,1)
  call mpibcast_integer(max_sec,1)
  call mpibcast_integer(vp_trap,1)
  call mpibcast_real(dtim,1)
  call mpibcast_real(max_seconds,1)
  call mpibcast_real(disp_par,1)
  call mpibcast_real(disp_vp,1)
  call mpibcast_real(disp_x,1)
  call mpibcast_real(disp_y,1)
  call mpibcast_character(method,lenswitch)
  call mpibcast_integer(meth,1)  
  call mpibcast_logical(nlphi,          1) 
  call mpibcast_logical(nlapar,         1) 
  call mpibcast_logical(nlbpar,         1) 
  call mpibcast_logical(normalized,     1) 
  call mpibcast_logical(read_file,      1) 
  call mpibcast_logical(neoclassics,    1) 
  call mpibcast_logical(output3d,       1) 
  call mpibcast_logical(lfinal_output,  1) 
  call mpibcast_logical(lwrite_output1, 1) 
  call mpibcast_logical(lcalc_fluxes,   1) 
  call mpibcast_logical(lcalc_freq,     1) 
  call mpibcast_logical(lphi_diagnostics, 1) 
  call mpibcast_logical(testing,        1) 
  call mpibcast_logical(collisions,     1) 
  call mpibcast_logical(non_linear,     1)
  call mpibcast_logical(nl_dtim_est,    1)
  call mpibcast_logical(silent,         1)
  call mpibcast_logical(zonal_adiabatic,1) 
  call mpibcast_logical(ltrapping_arakawa,1) 
  call mpibcast_character(parallel_boundary_conditions, lenswitch) 
  call mpibcast_character(order_of_the_scheme,          lenswitch) 
  call mpibcast_character(matrix_format,                lenswitch)
  call mpibcast_logical(ldiagnostic_namelist,1)
  call mpibcast_logical(lverbose,1)
  call mpibcast_logical(screen_output,1)
  call mpibcast_integer(restart_file_version,1)
  call mpibcast_real(gamatol,1)

end subroutine control_bcast_nml

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

subroutine control_initt

  use mpiinterface, only : root_processor

  logical :: lproc_write_stdout
  
  ! lproc_write_stdout
  lproc_write_stdout = root_processor
  
  ! (re) set lverbose, screen_output
  lverbose = lverbose .and. lproc_write_stdout
  screen_output = screen_output .and. lproc_write_stdout

  if (lverbose) write (*,*) '* verbose output set'

end subroutine control_initt

!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!> Check the input and initialise everything necessary for allocation.
!----------------------------------------------------------------------------

subroutine control_check_params

  use fft,          only : working_fft_library
  use general,      only : gkw_abort, gkw_warn
  use mpiinterface, only : root_processor

  !bad values for vp_trap
  !Why not use a logical?
  if (.not. ( vp_trap == 0 .or. vp_trap == 1) ) then
    call gkw_abort('control: vp_trap must be 1 or 0')
  endif
  
  ! Do we need a FFT library?
  if (non_linear .and. (.not. working_fft_library)) then
    call gkw_abort('non_linear option require a working FFT '//&
        &          'library!')
  end if

  !method checks
  select case(method)
    case('EXP')
      select case(meth)
         case(1)
            if (non_linear) call gkw_warn('Meth=1 Nonlinear timestep estimator off?')
         case(2) !RK4 This is recommended
                 !No warnings
         case(3) 
            if (non_linear) call gkw_warn('Meth=3 Nonlinear timestep estimator off?')
         case default
            call gkw_abort('Unknown value of meth for method EXP')
      end select !meth
      
    case('IMP')
      call gkw_warn('Implict scheme under development &
                 & many options not supported: (nonlinear, parallel, ...)')
      !Should add throw outs here after more development before release.

    case default 
      if (root_processor) then
        write(*,*)'control: You have given ',method,' as method'
        write(*,*)'allowed are: EXP'
      endif
      call gkw_abort('(see message above)')
  end select  !method
  
  ! boundary conditions
  select case(parallel_boundary_conditions)

    case('zero') ! This should be fine
    case('periodic_noshift') ! only implemented for the fourth order scheme
      if (order_of_the_scheme .ne. 'fourth_order') then 
        call gkw_abort('control: '//                                         &
            &          'The option periodic_noshift for '//                  &
            &          'parallel_boundary_conditions in the namelist '//     &
            &          'switches is only implemented for the fourth_order'// &
            &          'scheme.')
      endif
    case('zero_derivative') ! only implemented for the fourth order scheme
      if (order_of_the_scheme .ne. 'fourth_order') then 
        call gkw_abort('control: '//                                         &
            &          'The option zero_derivative for '//                   &
            &          'parallel_boundary_conditions in the namelist '//     &
            &          'switches is only implemented for the fourth_order'// &
            &          'scheme.')
      endif
    case default
      if (root_processor) then 
        write(*,*) 'control_read_input: You specified ',parallel_boundary_conditions
        write(*,*) 'for parallel_boundary_conditions in the namelist switches'
        write(*,*) 'Only known options are: zero, periodic_noshift'
      endif
      call gkw_abort('(see message above)')
  end select !boundary conditions

  ! order of the scheme
  select case(order_of_the_scheme)
  
    case('second_order')
      
      if (ltrapping_arakawa) then
        call gkw_abort('second_order not fully implemented for ltrapping_'// &
            &          'arakawa')
      end if
      
! Need to correct normalizations and add the trapping due to Apar F_M G
! term  **** APS: what is this about? ****
    case('fourth_order')   
    case default
      if (root_processor) then 
        write(*,*) 'control: you specified ',order_of_the_scheme,' as input'
        write(*,*) 'to order_of_the_scheme.'
        write(*,*) 'Only known options are: second_order and fourth_order'
      endif
      call gkw_abort('(see message above)')
  end select !order of the scheme
 
  ! check if the matrix_format is known (obsolete)
  select case(matrix_format) 
    case('complex')
    case('complex-real')
      call gkw_abort('control: complex-real no-longer supported.')
      !Warning if you re-enable comples real beware that it does not work 
      !with many things such as some paralellisations or zonal adiabatic and...???
    case default 
      call gkw_abort('control: Unknown matrix format specified in input')
  end select !matrix format

  ! store the value given in the input file 
  dtim_input = dtim
  !The estimate value must be initialised larger than the input value. 
  dtim_est_save=dtim+1.
  !Do not use the nonlinear timestep estimator for linear runs
  if (.not.non_linear) nl_dtim_est=.false. 

  ! intialize the time to zero 
  time = 0.

  ! restart file version
  if (restart_file_version < 1 .or.                                          &
      &  restart_file_version > restart_file_current_version) then
    call gkw_abort('bad restart_file_version; current is '//'(add here)')
  end if

  ! check for vp_trap = 0 and collisions, not implemented
  if (vp_trap /= 0 .and. collisions) then
    call gkw_abort('control: vp_trap does not work with collisions')
  endif

  ! vp_trap is currently numerically unstable in nonlinear case
  if (vp_trap /= 0 .and. non_linear .and. method .eq. 'EXP') then
    call gkw_abort('control: vp_trap unstable for nonlinear explicit runs')
  endif

  !nonlinear runs are not normalized
  if (normalized.and.non_linear) then
    call gkw_warn('Control: nonlinear runs are not normalized')
    normalized=.false.
  endif

  !Mode frequncy only calculated with normalized
  if (.not.normalized.and.lcalc_freq) then
    call gkw_warn('Control: mode frequency only calculated with normalized')
    lcalc_freq=.false.
  endif

  !Negative dissipation coeffcients are not allowed
  if (disp_x < 0. .or. disp_y < 0.) then
     call gkw_abort('control: Negative dissipation coeffcients not allowed')
  endif
  if (disp_vp < 0. .or. disp_par < 0.) then
     call gkw_abort('control: Negative dissipation coeffcients not allowed')
  endif

  if (output3D) then
     call gkw_warn('3D output data can be very large')
  end if

  ! set the maximum number of seconds
  if (max_seconds > 0. .and. max_sec > 0) then
    call gkw_warn('max_seconds will be used, not max_sec')
  end if
  if (max_seconds > 0.) then
    max_sec=int(max_seconds)
  else
    max_seconds=real(max_sec)
  end iF
  
end subroutine control_check_params

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

end module control
