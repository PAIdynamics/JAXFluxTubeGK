module device_memory_tracer_m
    !! Module containing routines to initialize and finalize the device
    !! memory tracing feature
    use genex_fortran_env_m, only: DP

    implicit none
    private

    public :: initialize_device_memory_tracer
    public :: finalize_device_memory_tracer

    interface
        module subroutine initialize_device_memory_tracer(comm, out_dir, &
                                                          init_tstamp)
            !! Initialize device memory tracing feature on C++ layer if enabled.
            !! Otherwise do nothing.
            integer, intent(in) :: comm
            !! MPI communicator
            character(len=*), intent(in) :: out_dir
            !! Output directory for any files written by the tracer
            real(kind=DP), intent(in) :: init_tstamp
            !! Initial timestamp used as reference for the time recording
        end subroutine

        module subroutine finalize_device_memory_tracer()
            !! Finalize device memory tracing feature on C++ layer if enabled.
            !! Otherwise do nothing.
        end subroutine
    end interface

end module
