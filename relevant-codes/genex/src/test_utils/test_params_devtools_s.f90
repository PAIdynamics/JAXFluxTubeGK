submodule (test_params_m) test_params_devtools_s
    ! Submodule that contains helpers for unit tests to initialize
    ! the developer tools with non-default parameters.
    use test_params_io_m

    implicit none

contains

    module subroutine setup_test_devtools(comm, rank, debug_profregion, &
                                          isolate_tsync, trace_device_memory, &
                                          trace_runtime)
        integer, intent(in) :: comm
        integer, intent(in) :: rank
        logical, optional, intent(in) :: debug_profregion
        logical, optional, intent(in) :: isolate_tsync
        logical, optional, intent(in) :: trace_device_memory
        logical, optional, intent(in) :: trace_runtime

        logical :: debug_profregion_local, isolate_tsync_local, &
                   trace_dmem_local, trace_rt_local
        integer :: iunit, io_error, ierr

        ! Set default and overwrite if optionals present
        ! NOTE: The parameters are initially set to their default values.
        !       If not specified, default values are used using the getters
        !       and write to the namelist.
        debug_profregion_local = get_debug_profregion()
        isolate_tsync_local    = get_isolate_tsync()
        trace_dmem_local       = get_trace_device_memory()
        trace_rt_local         = get_trace_runtime()

        if(present(debug_profregion)) &
            debug_profregion_local = debug_profregion
        if(present(isolate_tsync)) &
            isolate_tsync_local = isolate_tsync
        if(present(trace_device_memory)) trace_dmem_local = trace_device_memory
        if(present(trace_runtime)) trace_rt_local = trace_runtime

        ! Let master proc write the file
        if(rank == 0) then
            open(newunit=iunit, file=trim(get_test_params_file())//".txt", &
                 action='write', status="replace", iostat=io_error)
            if(io_error /= 0) call test_params_error()
            call open_nml(iunit, "params_devtools")
            call write_nml(iunit, "params_devtools", "debug_profregion", &
                           debug_profregion_local)
            call write_nml(iunit, "params_devtools", "isolate_tsync", &
                           isolate_tsync_local)
            call write_nml(iunit, "params_devtools", "trace_device_memory", &
                           trace_dmem_local)
            call write_nml(iunit, "params_devtools", "trace_runtime", &
                           trace_rt_local)
            call close_nml(iunit, "params_devtools")
            close(iunit)
        end if
        call mpi_barrier(comm, ierr)

        ! Read file with all procs
        call read_params_devtools(trim(get_test_params_file())//".txt")
        call mpi_barrier(comm, ierr)

    end subroutine

end submodule
