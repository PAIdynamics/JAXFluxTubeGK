submodule(runtime_tracer_m) runtime_tracer_s
    !! Submodule containing routines and other resources to initialize and
    !! finalize the runtime feature
    use mpi
    use, intrinsic :: iso_c_binding, only: C_INT32_T, C_CHAR, C_DOUBLE
    use system_calls_m, only: mkdir, rm_rf
    use type_converters_m, only: string_f2c
    use params_devtools_m, only: get_trace_runtime
    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_SUCCESS, GENEX_ERR_SYSTEM_CALL

    implicit none

#ifdef ENABLE_GPU
    interface
        subroutine cbind_runtime_tracer_init(comm, out_dir, trace_folder, &
            init_tstamp) bind(C, name="cbind_runtime_tracer_init")
            !! Fortran/C++ interoperable routine for initialization of
            !! runtime tracing feature on the C++ layer
            import :: C_INT32_T, C_CHAR, C_DOUBLE
            integer(kind=C_INT32_T), value :: comm
            character(len=1, kind=C_CHAR), dimension(*), intent(in) :: out_dir
            character(len=1, kind=C_CHAR), dimension(*), intent(in) :: &
                trace_folder
            !! Folder name containing the temporary trace files
            real(kind=C_DOUBLE), value :: init_tstamp
        end subroutine

        subroutine cbind_runtime_tracer_fin(comm) &
            bind(C, name="cbind_runtime_tracer_fin")
            !! Fortran/C++ interoperable routine for finalization of
            !! runtime tracing feature on the C++ layer
            import :: C_INT32_T
            integer(kind=C_INT32_T), value :: comm
        end subroutine

        subroutine cbind_runtime_tracer_trace(mode, region_path, region_name, &
            tstamp) bind(C, name="cbind_runtime_tracer_trace")
            !! Fortran/C++ interoperable routine for tracing the runtime of
            !! profiling regions on the C++ layer
            import :: C_INT32_T, C_CHAR, C_DOUBLE
            integer(kind=C_INT32_T), value :: mode
            character(len=1, kind=C_CHAR), dimension(*), intent(in) :: &
                region_path
            character(len=1, kind=C_CHAR), dimension(*), intent(in) :: &
                region_name
            real(kind=C_DOUBLE), value :: tstamp
        end subroutine
    end interface
#endif

contains

    module subroutine initialize_runtime_tracer(comm, out_dir, init_tstamp)
        integer, intent(in) :: comm
        character(len=*), intent(in) :: out_dir
        real(kind=DP), intent(in) :: init_tstamp

        character(len=:), allocatable :: trace_dir
        ! Directory to a specialized folder containing all traces files
        character(len=1, kind=C_CHAR), dimension(128) :: out_dir_c
        ! Directory to the general output in C character array format
        character(len=1, kind=C_CHAR), dimension(128) :: trace_folder_c
        ! Directory to a specialized folder containing all traces files
        ! in C character array format
        integer :: rank, ierr

#ifdef ENABLE_GPU
        if(get_trace_runtime()) then
            ! Create traces_rt folder in out_dir
            trace_dir = out_dir // "traces_rt/"
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
            call string_f2c(out_dir, out_dir_c)
            call string_f2c("traces_rt", trace_folder_c)

            call cbind_runtime_tracer_init(comm, out_dir_c, trace_folder_c, &
                                           init_tstamp)
        endif
#endif
    end subroutine

    module subroutine finalize_runtime_tracer(comm)
        integer, intent(in) :: comm

#ifdef ENABLE_GPU
        if(get_trace_runtime()) then
            call cbind_runtime_tracer_fin(comm)
        endif
#endif
    end subroutine

    module subroutine trace_profile(mode, region_path, region_name, tstamp)
        integer, intent(in) :: mode
        character(len=*), intent(in) :: region_path
        character(len=*), intent(in) :: region_name
        real(kind=DP), intent(in) :: tstamp

        character(len=1, kind=C_CHAR), dimension(128) :: region_path_c
        ! Path of a profiling region in C character array format
        character(len=1, kind=C_CHAR), dimension(128) :: region_name_c
        ! Name of a profiling region in C character array format

#ifdef ENABLE_GPU
        if(get_trace_runtime()) then
            ! Convert Fortran-formatted strings to C character array format
            call string_f2c(region_path, region_path_c)
            call string_f2c(region_name, region_name_c)

            call cbind_runtime_tracer_trace(mode, region_path_c, &
                                            region_name_c, tstamp)
        endif
#endif
    end subroutine

end submodule
