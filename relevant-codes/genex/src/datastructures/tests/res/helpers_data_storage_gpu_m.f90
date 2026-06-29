module helpers_data_storage_gpu_m
    !! Contains Fortran/C++ interface for unit testing the C++ and GPU features
    !! of data_storage_t

    use, intrinsic :: iso_c_binding, only: C_INT32_T, C_PTR
    use data_storage_m, only: data_storage_data_t
    use data_array_gpu_m, only: data_array_data_t

    implicit none

    interface

        integer(kind=C_INT32_T) function cbind_data_storage_2d_copy( &
            ds_src_cxx_ptr, ds_data_tgt, da_data_tgt) &
            bind(C, name="cbind_data_storage_2d_copy")
            !! Interoperable routine to copy the class members from the source
            !! da to target da struct for 2D case
            import :: C_INT32_T, C_PTR, data_storage_data_t, data_array_data_t
            type(C_PTR), intent(in)                  :: ds_src_cxx_ptr
            type(data_storage_data_t), intent(inout) :: ds_data_tgt
            type(data_array_data_t), intent(inout)   :: da_data_tgt
        end function

        integer(kind=C_INT32_T) function cbind_data_storage_4d_copy( &
            ds_src_cxx_ptr, ds_data_tgt, da_data_tgt) &
            bind(C, name="cbind_data_storage_4d_copy")
            !! Interoperable routine to copy the class members from the source
            !! da to target da struct for 4D case
            import :: C_INT32_T, C_PTR, data_storage_data_t, data_array_data_t
            type(C_PTR), intent(in)                  :: ds_src_cxx_ptr
            type(data_storage_data_t), intent(inout) :: ds_data_tgt
            type(data_array_data_t), intent(inout)   :: da_data_tgt
        end function

        integer(kind=C_INT32_T) function cbind_data_storage_5d_copy( &
            ds_src_cxx_ptr, ds_data_tgt, da_data_tgt) &
            bind(C, name="cbind_data_storage_5d_copy")
            !! Interoperable routine to copy the class members from the source
            !! da to target da struct for 5D case
            import :: C_INT32_T, C_PTR, data_storage_data_t, data_array_data_t
            type(C_PTR), intent(in)                  :: ds_src_cxx_ptr
            type(data_storage_data_t), intent(inout) :: ds_data_tgt
            type(data_array_data_t), intent(inout)   :: da_data_tgt
        end function

    end interface

end module
