module helpers_mail_delivery_gpu_m
    !! Contains Fortran/C++ interface for unit testing the C++ and GPU features
    !! of mailbox_t and mail delivery system
    use, intrinsic :: iso_c_binding, only: C_INT32_T, C_PTR

    implicit none

    interface

        integer(kind=C_INT32_T) function cbind_test_dimension( &
            dcomm_handler_cxx_pptr, dim_test, number_of_neighbors) &
            bind(C, name="cbind_test_dimension")
            !! Interoperable routine to test the mail delivery over the selected
            !! dimension with the specified number of neighbors/mail partners
            !! Return 0 for success and 1 for error
            import :: C_INT32_T, C_PTR
            type(C_PTR), intent(in)        :: dcomm_handler_cxx_pptr
            integer(kind=C_INT32_T), value :: dim_test
            integer(kind=C_INT32_T), value :: number_of_neighbors
        end function

    end interface

end module
