submodule(device_memory_tracer_m) device_memory_tracer_s
    !! Submodule containing routines and other resources to initialize and
    !! finalize the device memory tracing feature
    use mpi
    use, intrinsic :: iso_c_binding, only: C_INT32_T, C_CHAR, C_DOUBLE
    use system_calls_m, only: mkdir, rm_rf
    use type_converters_m, only: string_f2c
    use params_gpu_offload_m, only: get_use_gpu_offload
    use params_devtools_m, only: get_trace_device_memory
    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_SUCCESS, GENEX_ERR_SYSTEM_CALL

    implicit none

#ifdef ENABLE_GPU
    interface
        subroutine cbind_device_memory_tracer_init(comm, trace_dir, &
            init_tstamp) bind(C, name="cbind_device_memory_tracer_init")
            !! Fortran/C++ interoperable routine for initialization of
            !! device memory tracing feature on the C++ layer
            import :: C_INT32_T, C_CHAR, C_DOUBLE
            integer(kind=C_INT32_T), value :: comm
            character(len=1, kind=C_CHAR), dimension(*), intent(in) :: trace_dir
            real(kind=C_DOUBLE), value :: init_tstamp
        end subroutine

        subroutine cbind_device_memory_tracer_fin() &
            bind(C, name="cbind_device_memory_tracer_fin")
            !! Fortran/C++ interoperable routine for initialization of
            !! device memory tracing feature on the C++ layer
        end subroutine
    end interface
#endif

contains

    module subroutine initialize_device_memory_tracer(comm, out_dir, &
                                                      init_tstamp)
        integer, intent(in) :: comm
        character(len=*), intent(in) :: out_dir
        real(kind=DP), intent(in) :: init_tstamp

        character(len=:), allocatable :: trace_dir
        ! Directory to a specialized folder containing all traces files
        character(len=1, kind=C_CHAR), dimension(128) :: trace_dir_c
        ! Directory to a specialized folder containing all traces files
        ! in C character array format
        integer :: rank, ierr

#ifdef ENABLE_GPU
        if(get_use_gpu_offload() .and. get_trace_device_memory()) then
            ! Create traces_dmem folder in out_dir
            trace_dir = out_dir // "traces_dmem/"
            call mpi_comm_rank(comm, rank, ierr)

            call mpi_barrier(comm, ierr)
            if(rank == 0) then
                call mkdir(trace_dir, ierr, parents=.true.)
            endif
            if(ierr /= GENEX_SUCCESS) then
                call handle_error( &
                    "Failed to create a folder in """//trace_dir//"""!", &
                    GENEX_ERR_SYSTEM_CALL, __LINE__, __FILE__)
            endif
            call mpi_barrier(comm, ierr)

            ! Convert Fortran-formatted strings to C character array format
            call string_f2c(trace_dir, trace_dir_c)

            call cbind_device_memory_tracer_init(comm, trace_dir_c, init_tstamp)
        endif
#endif
    end subroutine

    module subroutine finalize_device_memory_tracer()
#ifdef ENABLE_GPU
        if(get_use_gpu_offload() .and. get_use_gpu_offload()) then
            call cbind_device_memory_tracer_fin()
        endif
#endif
    end subroutine

end submodule
