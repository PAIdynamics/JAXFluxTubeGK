module profiler_gpu_m
    !! Module containing additional resources related to GPU and Fortran/C++
    !! interoperability for the built-in profiler feature
    use, intrinsic :: iso_c_binding, only: C_CHAR, C_INT32_T, C_DOUBLE
    implicit none

    interface
        subroutine cbind_annotation_start(region_name) &
            bind(C, name="cbind_annotation_start")
            import :: C_CHAR
            !! Fortran/C++ interoperable routine to start a nested annotation
            !! region
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(in) :: region_name
            !! C char array describing the region name
        end subroutine

        subroutine cbind_annotation_end(region_name) &
            bind(C, name="cbind_annotation_end")
            import :: C_CHAR
            !! Fortran/C++ interoperable routine to end a nested annotation
            !! region
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(in) :: region_name
        end subroutine

        integer(kind=C_INT32_T) function cbind_get_profiler_region_time( &
            region_name_c, times, n_calls) &
            bind(C, name="cbind_get_profiler_region_time")
            import :: C_INT32_T, C_DOUBLE, C_CHAR
            !! Fortran/C++ interoperable routine to get the time interval of
            !! the profiler region from the C++ profiler
            character(len=1, kind=C_CHAR), dimension(*), &
                intent(in) :: region_name_c
            !! C char array describing the region name
            real(kind=C_DOUBLE), dimension(*), intent(inout) :: times
            !! Array containing the time measurements
            integer(kind=C_INT32_T), dimension(*), intent(inout) :: n_calls
            !! Array containing the number of calls
        end function
    end interface

end module
