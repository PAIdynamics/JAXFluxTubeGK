module devtools_m
    !! Module containing routines to initialize and finalize the developer tools
    use mpi
    use genex_fortran_env_m, only: DP
    use device_memory_tracer_m, only: initialize_device_memory_tracer, &
                                      finalize_device_memory_tracer
    use runtime_tracer_m, only: initialize_runtime_tracer, &
                                finalize_runtime_tracer

    implicit none
    private

    public :: initialize_devtools
    public :: finalize_devtools

    contains

        subroutine initialize_devtools(comm, out_dir)
            !! Initialize the developer tools if any are enabled.
            !! Otherwise do nothing.
            integer, intent(in) :: comm
            !! MPI communicator
            character(len=*), intent(in) :: out_dir
            !! Output directory for any files written by the tools

            real(kind=DP) :: init_tstamp
            ! Initial timestamp that should be shared to any developer tools
            ! that requires such timestamp as reference

            init_tstamp = MPI_Wtime()

            call initialize_device_memory_tracer(comm, out_dir, init_tstamp)
            call initialize_runtime_tracer(comm, out_dir, init_tstamp)
        end subroutine

        subroutine finalize_devtools(comm)
            !! Finalize the developer tools if any are enabled.
            !! Otherwise do nothing.
            integer, intent(in) :: comm
            !! MPI communicator

            call finalize_runtime_tracer(comm)
            call finalize_device_memory_tracer()
        end subroutine

end module
