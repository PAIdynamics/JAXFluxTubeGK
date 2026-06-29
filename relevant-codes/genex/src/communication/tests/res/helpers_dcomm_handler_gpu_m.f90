module helpers_dcomm_handler_gpu_m
    !! Contains Fortran/C++ interface for unit testing the C++ and GPU features
    !! of dcomm_handler_t

    use, intrinsic :: iso_c_binding, only: C_INT32_T, C_PTR
    use dcomm_handler_gpu_m, only: dcomm_handler_data_t

    implicit none

    interface
        integer(kind=C_INT32_T) function cbind_dcomm_handler_copy( &
            dcomm_handler_src_cxx_pptr, dcomm_handler_data_tgt) &
            bind(C, name="cbind_dcomm_handler_copy")
            !! Interoperable routine to copy the class members from the source
            !! dcomm handler to target dcomm handler struct
            import :: C_INT32_T, C_PTR, dcomm_handler_data_t
            type(C_PTR), intent(in) :: dcomm_handler_src_cxx_pptr
            type(dcomm_handler_data_t), intent(inout) :: dcomm_handler_data_tgt
        end function

        integer(kind=C_INT32_T) function cbind_dcomm_handler_test_rank_config( &
            dcomm_handler_cxx_pptr) &
            bind(C, name="cbind_dcomm_handler_test_rank_config")
            !! Interoperable routine to dcomm_handler_t for a given
            !! rank configuration
            import :: C_INT32_T, C_PTR
            type(C_PTR), intent(in) :: dcomm_handler_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_dcomm_handler_domain_copy( &
            dcomm_handler_src_cxx_pptr, dcomm_handler_data_tgt) &
            bind(C, name="cbind_dcomm_handler_domain_copy")
            !! Interoperable routine to copy the class members related to
            !! the domain decomposition from the source
            !! dcomm handler to target dcomm handler struct
            import :: C_INT32_T, C_PTR, dcomm_handler_data_t
            type(C_PTR), intent(in) :: dcomm_handler_src_cxx_pptr
            type(dcomm_handler_data_t), intent(inout) :: dcomm_handler_data_tgt
        end function
    end interface

end module
