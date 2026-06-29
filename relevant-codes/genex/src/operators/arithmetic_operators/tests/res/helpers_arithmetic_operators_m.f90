module helpers_arithmetic_operators_m
    !! Contains Fortran/C++ interface for dedicated reference values for
    !! the unit tests of arithmetic operators

    use, intrinsic :: iso_fortran_env
    use genex_fortran_env_m, only: GP, CP
    use, intrinsic :: iso_c_binding, only: C_INT32_T, C_INT64_T, C_PTR
    use params_gpu_offload_m, only: GPU_OFFLOAD_CUDA
    use params_gpu_offload_m, only: get_use_gpu_offload, get_gpu_offload_backend
    use genex_error_handling_m, only: handle_error_gpu, GPU_ERR_COPY

    implicit none
    private

    public :: copy_array_device_real
    public :: copy_array_device_integer
    public :: copy_array_host_real
    public :: copy_array_host_integer
    public :: ref_value_op_axpy
    public :: ref_value_op_lin_comb

#ifdef ENABLE_GPU

    interface
        real(kind=CP) function cbind_ref_op_axpy_core(a, x, y) &
            bind(C, name="cbind_ref_op_axpy_core")
            import :: CP
            real(kind=CP), value :: a
            real(kind=CP), value :: x
            real(kind=CP), value :: y
        end function

        real(kind=CP) function cbind_ref_op_lin_comb_core(a1, a2, x1, x2) &
            bind(C, name="cbind_ref_op_lin_comb_core")
            import :: CP
            real(kind=CP), value :: a1
            real(kind=CP), value :: a2
            real(kind=CP), value :: x1
            real(kind=CP), value :: x2
        end function

        integer(C_INT32_T) function cbind_copy_device_real( &
            length, array) bind(C, name="cbind_copy_device_real")
            import :: C_INT32_T, CP, C_INT64_T
            integer(kind=C_INT64_T), value :: length
            real(kind=CP), dimension(*), intent(inout) :: array
        end function

        integer(C_INT32_T) function cbind_copy_device_integer( &
            length, array) bind(C, name="cbind_copy_device_integer")
            import :: C_INT32_T, C_INT64_T
            integer(kind=C_INT64_T), value :: length
            integer(kind=C_INT32_T), dimension(*), intent(inout) :: array
        end function

        integer(C_INT32_T) function cbind_copy_host_real( &
            length, array) bind(C, name="cbind_copy_host_real")
            import :: C_INT32_T, CP, C_INT64_T
            integer(kind=C_INT64_T), value :: length
            real(kind=CP), dimension(*), intent(inout) :: array
        end function

        integer(C_INT32_T) function cbind_copy_host_integer( &
            length, array) bind(C, name="cbind_copy_host_integer")
            import :: C_INT32_T, C_INT64_T
            integer(kind=C_INT64_T), value :: length
            integer(kind=C_INT32_T), dimension(*), intent(inout) :: array
        end function
    end interface

#endif

#ifdef ENABLE_CUDA

    interface
        real(kind=CP) function cbind_ref_op_axpy_cuda(a, x, y) &
            bind(C, name="cbind_ref_op_axpy_cuda")
            import :: CP
            real(kind=CP), value :: a
            real(kind=CP), value :: x
            real(kind=CP), value :: y
        end function

        real(kind=CP) function cbind_ref_op_lin_comb_cuda(a1, a2, x1, x2) &
            bind(C, name="cbind_ref_op_lin_comb_cuda")
            import :: CP
            real(kind=CP), value :: a1
            real(kind=CP), value :: a2
            real(kind=CP), value :: x1
            real(kind=CP), value :: x2
        end function
    end interface

#endif

contains

    real(kind=GP) function ref_value_op_axpy(a, x, y)
        real(kind=GP), intent(in) :: a, x, y

        if(get_use_gpu_offload()) then
#ifdef ENABLE_GPU
            if(get_gpu_offload_backend() == GPU_OFFLOAD_CUDA) then
#ifdef ENABLE_CUDA
                ref_value_op_axpy = cbind_ref_op_axpy_cuda(a, x, y)
#endif
            else
                ref_value_op_axpy = cbind_ref_op_axpy_core(a, x, y)
            endif
#endif
        else
            ref_value_op_axpy = y + a * x
        endif
    end function

    real(kind=GP) function ref_value_op_lin_comb(a1, a2, x1, x2)
        real(kind=GP), intent(in) :: a1, a2, x1, x2

        if(get_use_gpu_offload()) then
#ifdef ENABLE_GPU
            if(get_gpu_offload_backend() == GPU_OFFLOAD_CUDA) then
#ifdef ENABLE_CUDA
                ref_value_op_lin_comb = &
                    cbind_ref_op_lin_comb_cuda(a1, a2, x1, x2)
#endif
            else
                ref_value_op_lin_comb = &
                    cbind_ref_op_lin_comb_core(a1, a2, x1, x2)
            endif
#endif
        else
            ref_value_op_lin_comb = a1 * x1 + a2 * x2
        endif
    end function

    subroutine copy_array_device_real(length, array)
        integer(kind=INT64) :: length
        real(kind=GP), dimension(*), intent(inout) :: array

        integer :: ierr

        if(get_use_gpu_offload()) then
#ifdef ENABLE_GPU
            ierr = cbind_copy_device_real(length, array)
            if(ierr /= 0) then
                call handle_error_gpu(GPU_ERR_COPY, __LINE__, __FILE__)
            endif
#endif
        endif
    end subroutine

    subroutine copy_array_device_integer(length, array)
        integer(kind=INT64) :: length
        integer, dimension(*), intent(inout) :: array

        integer :: ierr

        if(get_use_gpu_offload()) then
#ifdef ENABLE_GPU
            ierr = cbind_copy_device_integer(length, array)
            if(ierr /= 0) then
                call handle_error_gpu(GPU_ERR_COPY, __LINE__, __FILE__)
            endif
#endif
        endif
    end subroutine

    subroutine copy_array_host_real(length, array)
        integer(kind=INT64) :: length
        real(kind=GP), dimension(*), intent(inout) :: array

        integer :: ierr

        if(get_use_gpu_offload()) then
#ifdef ENABLE_GPU
            ierr = cbind_copy_host_real(length, array)
            if(ierr /= 0) then
                call handle_error_gpu(GPU_ERR_COPY, __LINE__, __FILE__)
            endif
#endif
        endif
    end subroutine

    subroutine copy_array_host_integer(length, array)
        integer(kind=INT64) :: length
        integer, dimension(*), intent(inout) :: array

        integer :: ierr

        if(get_use_gpu_offload()) then
#ifdef ENABLE_GPU
            ierr = cbind_copy_host_integer(length, array)
            if(ierr /= 0) then
                call handle_error_gpu(GPU_ERR_COPY, __LINE__, __FILE__)
            endif
#endif
        endif
    end subroutine

end module
