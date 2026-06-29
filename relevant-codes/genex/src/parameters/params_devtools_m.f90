module params_devtools_m
    !! Module handling parameters for the developer tools
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error
    use interop_params_m, only: set_cxx_params_devtools

    implicit none
    private

    integer :: parallax_dbgout = 0
    !! Used to change debug output of all PARALLAX routines.
    !! 0: no output, 1: minimal, 2: debug, 3: debug all MPI processes
    logical :: debug_profregion = .false.
    !! Specifies if the profiler region will be written to a debug file
    logical :: isolate_tsync = .false.
    !! Isolate the synchronization time from the communication time in the case
    !! of MPI_Allreduce in the profiler
    logical :: trace_device_memory = .false.
    !! Trace device memory usage
    logical :: trace_runtime = .false.
    !! Trace runtime based on profiling regions

    namelist / params_devtools / &
        parallax_dbgout, debug_profregion, isolate_tsync, trace_device_memory, &
        trace_runtime

    public :: read_params_devtools
    public :: write_params_devtools

    public :: get_parallax_dbgout
    public :: get_debug_profregion
    public :: get_isolate_tsync
    public :: get_trace_device_memory
    public :: get_trace_runtime

contains

    integer function get_parallax_dbgout()
        !! Returns the debug output level of PARALLAX routines
        get_parallax_dbgout = parallax_dbgout
    end function

    pure logical function get_debug_profregion()
        !! Returns if the debug logger of the region will be active or not
        get_debug_profregion = debug_profregion
    end function

    pure logical function get_isolate_tsync()
        !! Returns true if synchronization time isolation in the profiler
        !! is prescribed
        get_isolate_tsync = isolate_tsync
    end function

    pure logical function get_trace_device_memory()
        !! Returns .true. if device memory usage is traced
        get_trace_device_memory = trace_device_memory
    end function

    pure logical function get_trace_runtime()
        !! Returns .true. if runtime is traced
        get_trace_runtime = trace_runtime
    end function

    subroutine write_params_devtools(filename)
        !! Writes the namelist related to developer tools to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_devtools, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="profiler")

        close(iunit)
    end subroutine

    subroutine read_params_devtools(filename)
        !! Reads the namelist related to developer tools from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_devtools, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="profiler", &
                               iunit=iunit)

        close(iunit)

        call set_cxx_params_devtools()
    end subroutine

end module
