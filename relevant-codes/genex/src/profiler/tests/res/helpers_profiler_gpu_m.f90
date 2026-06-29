module helpers_profiler_gpu_m
    !! Contains Fortran/C++ interface for unit testing the C++ and GPU features
    !! of the profiler

    use, intrinsic :: iso_c_binding, only: C_INT32_T, C_CHAR

    implicit none

    interface

        integer(kind=C_INT32_T) function cbind_profiler_injection(t_sleep, &
            region_name, n_calls) bind(C, name="cbind_profiler_injection")
            !! Fortran/C++ interoperable routine for testing the profiler
            !! region injection feature of the profiler
            import :: C_INT32_T, C_CHAR
            integer(kind=C_INT32_T), value :: t_sleep
            ! Duration of execution time suspension in [ms]
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(in) :: region_name
            !! C char array describing the region name
            integer(kind=C_INT32_T), value :: n_calls
            !! Number of calls to test multi call feature
        end function

        subroutine cbind_profiler_allreduce(comm) &
            bind(C, name="cbind_profiler_allreduce")
            !! Fortran/C++ interoperable routine for creating a dummy
            !! mpi_allreduce profiling region
            import :: C_INT32_T
            integer(kind=C_INT32_T), value :: comm
            ! MPI communicator for the implicit MPI_Barrier
        end subroutine

    end interface

end module
