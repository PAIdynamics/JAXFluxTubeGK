module params_time_loop_m
    !! Module handling the parameters for the time loop
    use genex_fortran_env_m, only : GP
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error

    implicit none
    private

    real(kind=GP) :: dt = 0.01_GP
    !! Timestep
    integer :: n_timesteps = 10
    !! Maximum number of timesteps
    integer :: n_stages_pirock = 3
    !! Number of stages of PIROCK scheme, used if "pirock" is selected for the
    !! time-stepping scheme
    integer :: wrap_up_time = 0
    !! Time window (s) reserved for final file writing after safe termination
    integer :: steps_diagnostics = 10
    !! Number of timesteps after which the diagnostics are written
    integer :: steps_checkpoint  = 10000
    !! Number of timesteps after which a simulation checkpoint is written
    character(len=16) :: time_stepping_scheme = "rk4"
    !! Name of the time stepping scheme chosen
    !! Current options are: "euler", "rk4", "strang_splitting", "pirock"
    logical :: start_from_checkpoint = .false.
    !! Switch whether to read the mesh and initial condition from checkpoint
    logical :: run_mms = .false.
    !! Specifies if MMS tests should be run instead of conventional time loop

    namelist / params_time_loop / &
        dt, n_timesteps, wrap_up_time, steps_diagnostics, &
        steps_checkpoint, time_stepping_scheme, n_stages_pirock, &
        start_from_checkpoint, run_mms

    public :: read_params_time_loop
    public :: write_params_time_loop

    public :: get_dt
    public :: get_n_timesteps
    public :: get_n_stages_pirock
    public :: get_wrap_up_time
    public :: get_steps_diagnostics
    public :: get_steps_checkpoint
    public :: get_time_stepping_scheme
    public :: get_start_from_checkpoint
    public :: get_run_mms

contains

    character(len=16) function get_time_stepping_scheme()
        !! Returns the time stepping scheme
        get_time_stepping_scheme = time_stepping_scheme
    end function

    pure real(kind=GP) function get_dt()
        !! Returns the timestep
        get_dt = dt
    end function

    pure integer function get_n_timesteps()
        !! Returns the number of timesteps
        get_n_timesteps = n_timesteps
    end function

    pure integer function get_n_stages_pirock()
        !! Returns the number of stages for PIROCK scheme
        get_n_stages_pirock = n_stages_pirock
    end function

    pure integer function get_wrap_up_time()
        !! Returns the time window(s) reserved for final file writing after
        !! safe termination
        get_wrap_up_time = wrap_up_time
    end function

    pure integer function get_steps_diagnostics()
        !! Returns the number of timesteps after which the diagnostics
        !! are written
        get_steps_diagnostics = steps_diagnostics
    end function

    pure integer function get_steps_checkpoint()
        !! Returns the number of timesteps after which a simulation checkpoint
        !! is written
        get_steps_checkpoint = steps_checkpoint
    end function

    logical function get_start_from_checkpoint()
        !! Returns if the inital condition is read from checkpoint or not
        get_start_from_checkpoint = start_from_checkpoint
    end function

    logical function get_run_mms()
        !! Returns if MMS test should be run instead of the conventional
        !! time loop or not
        get_run_mms = run_mms
    end function

    subroutine write_params_time_loop(filename)
        !! Writes the time loop namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_time_loop, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="time_loop")

        close(iunit)
    end subroutine

    subroutine read_params_time_loop(filename)
        !! Reads the time loop namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_time_loop, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="time_loop", &
                               iunit=iunit)

        close(iunit)
    end subroutine

end module
