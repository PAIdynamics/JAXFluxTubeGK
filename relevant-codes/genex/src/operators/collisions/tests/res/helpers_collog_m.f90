module helpers_collog_m
    !! Contains Fortran/C++ interface for testing the C++ and GPU features
    !! of calc_collog
    use, intrinsic :: iso_c_binding, only: C_PTR, C_INT32_T

    implicit none

    interface
        integer(kind=C_INT32_T) function cbind_calc_collog( &
            da_moments_cxx_pptr, da_collog_cxx_pptr) &
            bind(C, name="cbind_calc_collog")
            !! Interoperable interface for calc_collog
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(in)    :: da_moments_cxx_pptr
            type(C_PTR), intent(inout) :: da_collog_cxx_pptr

        end function
    end interface

end module
