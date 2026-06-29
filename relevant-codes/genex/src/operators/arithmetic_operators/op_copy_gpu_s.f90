submodule (op_copy_m) op_copy_gpu_m
    use, intrinsic :: iso_c_binding, only: C_INT32_T, C_INT64_T
    use genex_fortran_env_m, only: CP
    use genex_error_handling_m, only: handle_error_gpu, GPU_ERR_INITIALIZE, &
                                      GPU_ERR_FINALIZE
    !$  use omp_lib

    implicit none

    interface
        integer(kind=C_INT32_T) function cbind_op_copy_initialize( &
            op_cxx_pptr) bind(C, name="cbind_op_copy_initialize")
            !! Fortran/C++ interoperable routine for initialization of
            !! op_copy_gpu_t C++ class
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(inout) :: op_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_op_copy_finalize( &
            op_cxx_pptr) bind(C, name="cbind_op_copy_finalize")
            !! Fortran/C++ interoperable routine for finalization of
            !! op_copy_gpu_t C++ class
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(inout) :: op_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_op_copy_apply_real( &
            op_cxx_pptr, size_y, x, y) &
            bind(C, name="cbind_op_copy_apply_real")
            !! Fortran/C++ interoperable routine for the apply routine of
            !! op_copy_gpu_t C++ class for real type
            import :: C_PTR, C_INT32_T, C_INT64_T, CP
            type(C_PTR), intent(in) :: op_cxx_pptr
            integer(kind=C_INT64_T), value :: size_y
            real(kind=CP), dimension(*), intent(in) :: x
            real(kind=CP), dimension(*), intent(inout) :: y
        end function

        integer(kind=C_INT32_T) function cbind_op_copy_apply_integer( &
            op_cxx_pptr, size_y, x, y) &
            bind(C, name="cbind_op_copy_apply_integer")
            !! Fortran/C++ interoperable routine for the apply routine of
            !! op_copy_gpu_t C++ class for integer type
            import :: C_PTR, C_INT32_T, C_INT64_T
            type(C_PTR), intent(in) :: op_cxx_pptr
            integer(kind=C_INT64_T), value :: size_y
            integer(kind=C_INT32_T), dimension(*), intent(in) :: x
            integer(kind=C_INT32_T), dimension(*), intent(inout) :: y
        end function
    end interface

contains

    module subroutine initialize_copy_gpu(this)
        class(op_copy_gpu_t), intent(inout) :: this

        integer :: ierr

        this%n_ls = 2
        this%n_flops = 0

        ! Initialize the operator on C++
        ierr = cbind_op_copy_initialize(this%op_cxx_pptr)
        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_INITIALIZE, __LINE__, __FILE__)
        endif
    end subroutine

    module subroutine apply_copy_real_core_gpu(this, y, x, size_y)
        class(op_copy_gpu_t), intent(inout) :: this
        real(kind=GP), dimension(*), intent(inout) :: y
        real(kind=GP), dimension(*), intent(in) :: x
        integer(kind=INT64), intent(in) :: size_y

        integer(kind=INT64) :: i
        integer :: ierr

        this%n_iterations = size_y

        call this%perf_counter%start_measurement()

        ierr = cbind_op_copy_apply_real(this%op_cxx_pptr, size_y, x, y)

        call this%perf_counter%end_measurement()
    end subroutine

    module subroutine apply_copy_integer_core_gpu(this, y, x, size_y)
        class(op_copy_gpu_t), intent(inout) :: this
        integer, dimension(*), intent(inout) :: y
        integer, dimension(*), intent(in) :: x
        integer(kind=INT64), intent(in) :: size_y

        integer(kind=INT64) :: i
        integer :: ierr

        this%n_iterations = size_y

        call this%perf_counter%start_measurement()

        ierr = cbind_op_copy_apply_integer(this%op_cxx_pptr, size_y, x, y)

        call this%perf_counter%end_measurement()
    end subroutine

    module subroutine apply_copy_logical_core_gpu(this, y, x, size_y)
        class(op_copy_gpu_t), intent(inout) :: this
        logical, dimension(*), intent(inout) :: y
        logical, dimension(*), intent(in) :: x
        integer(kind=INT64), intent(in) :: size_y
        integer(kind=INT64) :: i

        this%n_iterations = size_y

        call this%perf_counter%start_measurement()
        !$omp parallel default(none) &
        !$omp shared(y, x, size_y) &
        !$omp private(i)
        !$omp do simd schedule(static)
        do i = 1, size_y
            y(i) = x(i)
        enddo
        !$omp end do simd nowait
        !$omp end parallel
        call this%perf_counter%end_measurement()
    end subroutine

    module subroutine finalize_copy_gpu(this)
        type(op_copy_gpu_t), intent(inout) :: this

        integer :: ierr

        ! Finalize operator class on C++
        ierr = cbind_op_copy_finalize(this%op_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_FINALIZE, __LINE__, __FILE__)
        endif
    end subroutine

end submodule
