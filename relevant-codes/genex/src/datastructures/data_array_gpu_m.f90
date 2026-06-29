module data_array_gpu_m
    !! Module containing additional resources related to GPU and Fortran/C++
    !! interoperability for the data_array_t type
    use, intrinsic :: iso_c_binding, only: C_PTR, C_INT32_T, C_INT64_T
    use genex_fortran_env_m, only: CP

    implicit none
    private

    public :: cbind_data_array_initialize
    public :: cbind_data_array_finalize
    public :: cbind_data_array_update_host
    public :: cbind_data_array_update_device
    public :: cbind_data_array_get_device_pointer

    type, public, bind(C) :: data_array_data_t
        !! Fortran/C++ interoperable structure of the class members
        !! relevant to data_array_t
        real(kind=CP) :: init_value
        !! Initial uniform value of the array
        integer(kind=C_INT32_T) :: array_dim
        !! Number of dimension or rank of the array
        integer(kind=C_INT64_T) :: array_size
        !! Total number of elements of the array
        integer(kind=C_INT64_T) :: array_size_stripped
        !! Total number of elements of the array without ghost
        integer(kind=C_INT32_T) :: is_distributed_array
        !! Bookeeping variable to indicate if the array is distributed array
        type(C_PTR) :: array_shape_ptr
        !! C pointer to the array specifying the number of elements
        !! for each dimension
        type(C_PTR) :: array_shape_stripped_ptr
        !! C pointer to the array specifying the number of elements
        !! without ghost for each dimension
        type(C_PTR) :: array_lb_ptr
        !! C pointer to the array specifying the lower boundary index
        !! for each dimension
        type(C_PTR) :: array_ub_ptr
        !! C pointer to the array specifying the upper boundary index
        !! for each dimension
        type(C_PTR) :: array_lb_stripped_ptr
        !! C pointer to the array specifying the lower boundary index
        !! without ghost for each dimension
        type(C_PTR) :: array_ub_stripped_ptr
        !! C pointer to the array specifying the upper boundary index
        !! without ghost for each dimension
        type(C_PTR) :: array_ptr
        !! C pointer to the multidimensional array stored in the data array type
    end type

    interface
        integer(kind=C_INT32_T) function cbind_data_array_initialize( &
            da_data, da_cxx_pptr) bind(C, name="cbind_data_array_initialize")
            !! Fortran/C++ interoperable routine for the initialization of
            !! the data_array_t C++ class
            import :: C_INT32_T, C_PTR, data_array_data_t
            type(data_array_data_t), intent(inout) :: da_data
            !! Object of Fortran/C++ interoperable
            !! structure based on data_array_data_t
            type(C_PTR), intent(inout) :: da_cxx_pptr
            !! C pointer to the data_array_t C++ class instance pointer
        end function

        integer(kind=C_INT32_T) function cbind_data_array_finalize( &
            da_cxx_pptr) bind(C, name="cbind_data_array_finalize")
            !! Fortran/C++ interoperable routine for the finalization of
            !! the data_array_t C++ class
            import :: C_INT32_T, C_PTR
            type(C_PTR), intent(inout) :: da_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_data_array_update_host( &
            da_cxx_pptr) bind(C, name="cbind_data_array_update_host")
            !! Fortran/C++ interoperable routine for updating the array on CPU
            !! from GPU
            import ::  C_INT32_T, C_PTR
            type(C_PTR), intent(inout) :: da_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_data_array_update_device( &
            da_cxx_pptr) bind(C, name="cbind_data_array_update_device")
            !! Fortran/C++ interoperable routine for updating the array on GPU
            !! from CPU
            import ::  C_INT32_T, C_PTR
            type(C_PTR), intent(inout) :: da_cxx_pptr
        end function

        function cbind_data_array_get_device_pointer(da_cxx_pptr) &
            bind(C, name="cbind_data_array_get_device_pointer")
            !! Fortran/C++ interoperable routine to get the device pointer
            !! of the array
            import :: C_PTR
            type(C_PTR), intent(inout) :: da_cxx_pptr
            type(C_PTR) :: cbind_data_array_get_device_pointer
        end function
    end interface

end module data_array_gpu_m
