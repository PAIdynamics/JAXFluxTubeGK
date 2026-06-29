submodule (op_lin_comb_m) op_lin_comb_gpu_s
    use, intrinsic :: iso_c_binding, only: C_INT32_T, C_INT64_T
    use genex_fortran_env_m, only: CP
    use genex_error_handling_m, only: handle_error_gpu, GPU_ERR_INITIALIZE, &
                                      GPU_ERR_FINALIZE

    implicit none

    interface
        integer(kind=C_INT32_T) function cbind_op_lin_comb_initialize( &
            op_cxx_pptr) bind(C, name="cbind_op_lin_comb_initialize")
            !! Fortran/C++ interoperable routine for initialization of
            !! op_lin_comb_gpu_t C++ class
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(inout) :: op_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_op_lin_comb_finalize( &
            op_cxx_pptr) bind(C, name="cbind_op_lin_comb_finalize")
            !! Fortran/C++ interoperable routine for finalization of
            !! op_lin_comb_gpu_t C++ class
            import :: C_PTR, C_INT32_T
            type(C_PTR), intent(inout) :: op_cxx_pptr
        end function

        integer(kind=C_INT32_T) function cbind_op_lin_comb_apply(op_cxx_pptr, &
            size_y, a1, a2, x1, x2, y) bind(C, name="cbind_op_lin_comb_apply")
            !! Fortran/C++ interoperable routine for the apply routine of
            !! op_lin_comb_gpu_t C++ class
            import :: C_PTR, C_INT32_T, C_INT64_T, CP
            type(C_PTR), intent(in) :: op_cxx_pptr
            integer(kind=C_INT64_T), value :: size_y
            real(kind=CP), value :: a1
            real(kind=CP), value :: a2
            real(kind=CP), dimension(*), intent(in) :: x1
            real(kind=CP), dimension(*), intent(in) :: x2
            real(kind=CP), dimension(*), intent(inout) :: y
        end function
    end interface

contains

    module subroutine initialize_lin_comb_gpu(this)
        class(op_lin_comb_gpu_t), intent(inout) :: this

        integer :: ierr

        this%n_ls = 3
        this%n_flops = 3

        ! Initialize the operator on C++
        ierr = cbind_op_lin_comb_initialize(this%op_cxx_pptr)
        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_INITIALIZE, __LINE__, __FILE__)
        endif
    end subroutine

    module subroutine apply_lin_comb_core_gpu(this, y, a1, x1, a2, x2, size_y)
        class(op_lin_comb_gpu_t), intent(inout) :: this
        real(kind=GP), dimension(*), intent(inout) :: y
        real(kind=GP), intent(in) :: a1
        real(kind=GP), dimension(*), intent(in) :: x1
        real(kind=GP), intent(in) :: a2
        real(kind=GP), dimension(*), intent(in) :: x2
        integer(kind=INT64), intent(in) :: size_y

        integer(kind=INT64) :: i
        integer :: ierr

        this%n_iterations = size_y

        call this%perf_counter%start_measurement()

        ierr = cbind_op_lin_comb_apply(this%op_cxx_pptr, size_y, a1, a2, &
                                       x1, x2, y)

        call this%perf_counter%end_measurement()
    end subroutine

    module subroutine finalize_lin_comb_gpu(this)
        type(op_lin_comb_gpu_t), intent(inout) :: this

        integer :: ierr

        ! Finalize operator class on C++
        ierr = cbind_op_lin_comb_finalize(this%op_cxx_pptr)

        if(ierr /= 0) then
            call handle_error_gpu(GPU_ERR_FINALIZE, __LINE__, __FILE__)
        endif
    end subroutine

end submodule
