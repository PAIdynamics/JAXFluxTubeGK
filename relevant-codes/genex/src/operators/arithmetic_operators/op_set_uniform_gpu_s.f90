submodule (op_set_uniform_m) op_set_uniform_gpu_s
    use, intrinsic :: iso_c_binding, only:  C_INT32_T, C_INT64_T
    use genex_fortran_env_m, only: CP
    use genex_error_handling_m, only: handle_error_gpu, GPU_ERR_INITIALIZE, &
                                      GPU_ERR_FINALIZE
    !$  use omp_lib

    implicit none

    interface
        integer(kind=C_INT32_T) function cbind_op_set_uniform_initialize( &
            op_cxx_pptr) bind(C, name="cbind_op_set_uniform_initialize")
            !! Fortran/C++ interoperable routine for initialization of
            !! op_set_uniform_gpu_t C++ class
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(inout) :: op_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_op_set_uniform_finalize( &
            op_cxx_pptr) bind(C, name="cbind_op_set_uniform_finalize")
            !! Fortran/C++ interoperable routine for finalization of
            !! op_set_uniform_gpu_t C++ class
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(inout) :: op_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_op_set_uniform_apply_real( &
            op_cxx_pptr, size_y, a, y) &
            bind(C, name="cbind_op_set_uniform_apply_real")
            !! Fortran/C++ interoperable routine for the apply routine of
            !! op_set_uniform_gpu_t C++ class for real type
            import :: C_PTR, C_INT32_T, C_INT64_T, CP
            type(C_PTR), intent(in) :: op_cxx_pptr
            integer(kind=C_INT64_T), value :: size_y
            real(kind=CP), value :: a
            real(kind=CP), dimension(*), intent(inout) :: y
        end function

        integer(kind=C_INT32_T) function cbind_op_set_uniform_apply_integer( &
            op_cxx_pptr, size_y, a, y) &
            bind(C, name="cbind_op_set_uniform_apply_integer")
            !! Fortran/C++ interoperable routine for the apply routine of
            !! op_set_uniform_gpu_t C++ class for integer type
            import :: C_PTR, C_INT32_T, C_INT64_T
            type(C_PTR), intent(in) :: op_cxx_pptr
            integer(kind=C_INT64_T), value :: size_y
            integer(kind=C_INT32_T), value :: a
            integer(kind=C_INT32_T), dimension(*), intent(inout) :: y
        end function
    end interface

contains

    module subroutine initialize_set_uniform_gpu(this)
        class(op_set_uniform_gpu_t), intent(inout) :: this

        integer :: ierr

        this%n_ls = 1
        this%n_flops = 0

        ! Initialize the operator on C++
        ierr = cbind_op_set_uniform_initialize(this%op_cxx_pptr)
        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_INITIALIZE, __LINE__, __FILE__)
        endif
    end subroutine

    module subroutine apply_set_uniform_real_core_gpu(this, y, a, size_y)
        class(op_set_uniform_gpu_t), intent(inout) :: this
        real(kind=GP), dimension(*), intent(inout) :: y
        real(kind=GP), intent(in) :: a
        integer(kind=INT64), intent(in) :: size_y

        integer(kind=INT64) :: i
        integer :: ierr

        this%n_iterations = size_y

        call this%perf_counter%start_measurement()

        ierr = cbind_op_set_uniform_apply_real(this%op_cxx_pptr, size_y, a, y)

        call this%perf_counter%end_measurement()
    end subroutine

    module subroutine apply_set_uniform_integer_core_gpu(this, y, a, size_y)
        class(op_set_uniform_gpu_t), intent(inout) :: this
        integer, dimension(*), intent(inout) :: y
        integer, intent(in) :: a
        integer(kind=INT64), intent(in) :: size_y

        integer(kind=INT64) :: i
        integer :: ierr

        this%n_iterations = size_y

        call this%perf_counter%start_measurement()

        ierr = cbind_op_set_uniform_apply_integer(this%op_cxx_pptr, &
                                                  size_y, a, y)

        call this%perf_counter%end_measurement()
    end subroutine

    module subroutine apply_set_uniform_logical_core_gpu(this, y, a, size_y)
        class(op_set_uniform_gpu_t), intent(inout) :: this
        logical, dimension(*), intent(inout) :: y
        logical, intent(in) :: a
        integer(kind=INT64), intent(in) :: size_y
        integer(kind=INT64) :: i

        this%n_iterations = size_y

        call this%perf_counter%start_measurement()
        !$omp parallel default(none) &
        !$omp shared(y, a, size_y) &
        !$omp private(i)
        !$omp do simd schedule(static)
        do i = 1, size_y
            y(i) =  a
        enddo
        !$omp end do simd nowait
        !$omp end parallel
        call this%perf_counter%end_measurement()
    end subroutine

    module subroutine finalize_set_uniform_gpu(this)
        type(op_set_uniform_gpu_t), intent(inout) :: this

        integer :: ierr

        ! Finalize operator class on C++
        ierr = cbind_op_set_uniform_finalize(this%op_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_FINALIZE, __LINE__, __FILE__)
        endif
    end subroutine

end submodule
