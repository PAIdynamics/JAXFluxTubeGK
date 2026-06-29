module helpers_data_array_gpu_m
    !! Contains Fortran/C++ interface for unit testing the C++ and GPU features
    !! of data_array_t

    use, intrinsic :: iso_c_binding, only: C_INT32_T, C_PTR, &
                                           C_INTPTR_T, C_INT64_T, c_loc
    use genex_fortran_env_m, only: GP, CP
    use data_array_m, only: data_array_t, data_array_2d_t, data_array_3d_t, &
                            data_array_4d_t, data_array_5d_t
    use data_array_gpu_m, only: data_array_data_t

    implicit none

    interface

        integer(kind=C_INT32_T) function cbind_data_array_2d_copy( &
            da_src_cxx_ptr, da_data_tgt) &
            bind(C, name="cbind_data_array_2d_copy")
            !! Interoperable routine to copy the class members from the source
            !! da to target da struct for 2D case
            import :: C_INT32_T, C_PTR, data_array_data_t
            type(C_PTR), intent(in)                :: da_src_cxx_ptr
            type(data_array_data_t), intent(inout) :: da_data_tgt
        end function

        integer(kind=C_INT32_T) function cbind_data_array_3d_copy( &
            da_src_cxx_ptr, da_data_tgt) &
            bind(C, name="cbind_data_array_3d_copy")
            !! Interoperable routine to copy the class members from the source
            !! da to target da struct for 3D case
            import :: C_INT32_T, C_PTR, data_array_data_t
            type(C_PTR), intent(in)                :: da_src_cxx_ptr
            type(data_array_data_t), intent(inout) :: da_data_tgt
        end function

        integer(kind=C_INT32_T) function cbind_data_array_4d_copy( &
            da_src_cxx_ptr, da_data_tgt) &
            bind(C, name="cbind_data_array_4d_copy")
            !! Interoperable routine to copy the class members from the source
            !! da to target da struct for 4D case
            import :: C_INT32_T, C_PTR, data_array_data_t
            type(C_PTR), intent(in)                :: da_src_cxx_ptr
            type(data_array_data_t), intent(inout) :: da_data_tgt
        end function

        integer(kind=C_INT32_T) function cbind_data_array_5d_copy( &
            da_src_cxx_ptr, da_data_tgt) &
            bind(C, name="cbind_data_array_5d_copy")
            !! Interoperable routine to copy the class members from the source
            !! da to target da struct for 5D case
            import :: C_INT32_T, C_PTR, data_array_data_t
            type(C_PTR), intent(in)                :: da_src_cxx_ptr
            type(data_array_data_t), intent(inout) :: da_data_tgt
        end function

        integer(kind=C_INT32_T) function cbind_data_array_2d_add(add_const, &
            da_src_cxx_ptr) bind(C, name="cbind_data_array_2d_add")
            !! Interoperable routine to add the 2d array in data array with
            !! a given constant
            import :: C_INT32_T, CP, C_PTR
            real(kind=CP), value           :: add_const
            type(C_PTR), intent(inout)     :: da_src_cxx_ptr
        end function

        integer(kind=C_INT32_T) function cbind_data_array_3d_add(add_const, &
            da_src_cxx_ptr) bind(C, name="cbind_data_array_3d_add")
            !! Interoperable routine to add the 3d array in data array with
            !! a given constant
            import :: C_INT32_T, CP, C_PTR
            real(kind=CP), value           :: add_const
            type(C_PTR), intent(inout)     :: da_src_cxx_ptr
        end function

        integer(kind=C_INT32_T) function cbind_data_array_4d_add(add_const, &
            da_src_cxx_ptr) &
            bind(C, name="cbind_data_array_4d_add")
            !! Interoperable routine to add the 4d array in data array with
            !! a given constant
            import :: C_INT32_T, CP, C_PTR
            real(kind=CP), value           :: add_const
            type(C_PTR), intent(inout)     :: da_src_cxx_ptr
        end function

        integer(kind=C_INT32_T) function cbind_data_array_5d_add(add_const, &
            da_src_cxx_ptr) bind(C, name="cbind_data_array_5d_add")
            !! Interoperable routine to add the 5d array in data array with
            !! a given constant
            import :: C_INT32_T, CP, C_PTR
            real(kind=CP), value           :: add_const
            type(C_PTR), intent(inout)     :: da_src_cxx_ptr
        end function

        integer(kind=C_INTPTR_T) function cbind_data_array_get_device_address( &
            array_ptr) bind(C, name="cbind_data_array_get_device_address")
            !! Interoperable routine to get memory address of the array
            !! contained in data_array_t object on GPU
            import :: C_INTPTR_T, CP
            real(kind=CP), dimension(*), intent(in) :: array_ptr
        end function

        integer(kind=C_INT32_T) function cbind_data_array_set_uniform( &
            set_value, arr_size, dev_ptr) &
            bind(C, name="cbind_data_array_set_uniform")
            !! Interoperable routine to set uniform value to a device pointer
            !! on GPU
            import :: C_INT32_T, C_INT64_T, CP, C_PTR
            real(kind=CP), value           :: set_value
            integer(kind=C_INT64_T), value :: arr_size
            real(kind=CP), dimension(*), intent(inout) :: dev_ptr
        end function

    end interface

contains

    function assert_device_pointer(da) result(diff)
        !! Returns the integer difference of the memory address of the array
        !! contained in data_array_t object on GPU compared to the reference
        class(data_array_t), intent(inout) :: da

        real(kind=GP), contiguous, pointer, dimension(:,:) :: arr_2d_hptr, &
                                                              arr_2d_dptr
        real(kind=GP), contiguous, pointer, dimension(:,:,:) :: arr_3d_hptr, &
                                                                arr_3d_dptr
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: arr_4d_hptr, &
                                                                  arr_4d_dptr
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: &
            arr_5d_hptr, arr_5d_dptr
        integer(kind=C_INTPTR_T) :: address_test, address_ref, diff

        select type(da)
            type is(data_array_2d_t)
                arr_2d_hptr => da%get_pointer()
                arr_2d_dptr => da%get_device_pointer()
                address_ref = cbind_data_array_get_device_address(arr_2d_hptr)
                address_test = transfer(c_loc(arr_2d_dptr), mold=0_C_INTPTR_T)
            type is(data_array_3d_t)
                arr_3d_hptr => da%get_pointer()
                arr_3d_dptr => da%get_device_pointer()
                address_ref = cbind_data_array_get_device_address(arr_3d_hptr)
                address_test = transfer(c_loc(arr_3d_dptr), mold=0_C_INTPTR_T)
            type is(data_array_4d_t)
                arr_4d_hptr => da%get_pointer()
                arr_4d_dptr => da%get_device_pointer()
                address_ref = cbind_data_array_get_device_address(arr_4d_hptr)
                address_test = transfer(c_loc(arr_4d_dptr), mold=0_C_INTPTR_T)
            type is(data_array_5d_t)
                arr_5d_hptr => da%get_pointer()
                arr_5d_dptr => da%get_device_pointer()
                address_ref = cbind_data_array_get_device_address(arr_5d_hptr)
                address_test = transfer(c_loc(arr_5d_dptr), mold=0_C_INTPTR_T)
        end select

        diff = address_test - address_ref
    end function

end module
