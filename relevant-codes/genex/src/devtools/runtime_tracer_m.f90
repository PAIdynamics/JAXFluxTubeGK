module runtime_tracer_m
    !! Module containing routines to initialize and finalize the runtime
    !! tracing feature
    use genex_fortran_env_m, only: DP

    implicit none
    private

    public :: initialize_runtime_tracer
    public :: finalize_runtime_tracer
    public :: trace_profile
    public :: RT_TRACER_START
    public :: RT_TRACER_END

    !! Enumerator for the mode of a profiling region tracing entry
    !! NOTE: Synchronize with src/cxx/devtools/runtime_tracer.hxx
    enum, bind(C)
        enumerator :: RT_TRACER_START = 1
        !! Indicate the start of a profiling region
        enumerator :: RT_TRACER_END   = 0
        !! Indicate the end of a profiling region
    end enum

    interface
        module subroutine initialize_runtime_tracer(comm, out_dir, init_tstamp)
            !! Initialize runtime tracing feature on the C++ layer if enabled.
            !! Otherwise do nothing.
            integer, intent(in) :: comm
            !! MPI communicator
            character(len=*), intent(in) :: out_dir
            !! Output directory for any files written by the tracer
            real(kind=DP), intent(in) :: init_tstamp
            !! Initial timestamp used as reference for the time recording
        end subroutine

        module subroutine finalize_runtime_tracer(comm)
            !! Finalize runtime tracing feature on the C++ layer if enabled.
            !! Otherwise do nothing.
            integer, intent(in) :: comm
            !! MPI communicator
        end subroutine

        module subroutine trace_profile(mode, region_path, region_name, tstamp)
            !! Trace the runtime by registering the timestamp of a profiling
            !! region
            integer, intent(in) :: mode
            !! mode of a profiling region tracing entry
            character(len=*), intent(in) :: region_path
            !! Path of a profiling region
            character(len=*), intent(in) :: region_name
            !! Level of a profiling region
            real(kind=DP), intent(in) :: tstamp
            !! Timestamp of the tracing entry in second
        end subroutine

#ifdef ENABLE_GPU

#endif
    end interface

end module
