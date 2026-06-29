submodule (op_lin_comb_m) op_lin_comb_cpu_s
    use, intrinsic :: iso_fortran_env
    implicit none

contains

    module subroutine initialize_lin_comb_cpu(this)
        class(op_lin_comb_cpu_t), intent(inout) :: this
        this%n_ls = 3
        this%n_flops = 3
    end subroutine

    module subroutine apply_lin_comb_core_cpu(this, y, a1, x1, a2, x2, size_y)
        class(op_lin_comb_cpu_t), intent(inout) :: this
        real(kind=GP), dimension(*), intent(inout) :: y
        real(kind=GP), intent(in) :: a1
        real(kind=GP), dimension(*), intent(in) :: x1
        real(kind=GP), intent(in) :: a2
        real(kind=GP), dimension(*), intent(in) :: x2
        integer(kind=INT64), intent(in) :: size_y
        integer(kind=INT64) :: i

        this%n_iterations = size_y

        call this%perf_counter%start_measurement()
        !$omp parallel default(none) &
        !$omp shared(y, x1, x2, a1, a2, size_y) &
        !$omp private(i)
        !$omp do simd schedule(static)
        do i = 1, size_y
            y(i) = a1 * x1(i) + a2 * x2(i)
        enddo
        !$omp end do simd nowait
        !$omp end parallel
        call this%perf_counter%end_measurement()
    end subroutine

end submodule
