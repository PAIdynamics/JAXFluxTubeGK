submodule (op_axpy_m) op_axpy_cpu_m
    use, intrinsic :: iso_fortran_env
    implicit none

contains

    module subroutine initialize_axpy_cpu(this)
        class(op_axpy_cpu_t), intent(inout) :: this
        this%n_ls = 3
        this%n_flops = 2
    end subroutine

    module subroutine apply_axpy_core_cpu(this, y, a, x, size_y)
        class(op_axpy_cpu_t), intent(inout) :: this
        real(kind=GP), dimension(*), intent(inout) :: y
        real(kind=GP), intent(in) :: a
        real(kind=GP), dimension(*),  intent(in) :: x
        integer(kind=INT64), intent(in) :: size_y
        integer(kind=INT64) :: i

        this%n_iterations = size_y

        call this%perf_counter%start_measurement()

        !$omp parallel default(none) &
        !$omp shared(y, x, a, size_y) &
        !$omp private(i)
        !$omp do simd schedule(static)
        do i = 1, size_y
            y(i) = y(i) + a * x(i)
        enddo
        !$omp end do simd nowait
        !$omp end parallel

        call this%perf_counter%end_measurement()

    end subroutine

end submodule
