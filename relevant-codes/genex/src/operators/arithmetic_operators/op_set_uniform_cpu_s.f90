submodule (op_set_uniform_m) op_set_uniform_cpu_s
    use, intrinsic :: iso_fortran_env
!$  use omp_lib

    implicit none

contains

    module subroutine initialize_set_uniform_cpu(this)
        class(op_set_uniform_cpu_t), intent(inout) :: this
        this%n_ls = 1
        this%n_flops = 0
    end subroutine

    module subroutine apply_set_uniform_real_core_cpu(this, y, a, size_y)
        class(op_set_uniform_cpu_t), intent(inout) :: this
        real(kind=GP), dimension(*), intent(inout) :: y
        real(kind=GP), intent(in) :: a
        integer(kind=INT64), intent(in) :: size_y
        integer(kind=INT64) :: i

        this%n_iterations = size_y

        call this%perf_counter%start_measurement()
        !$omp parallel default(none) &
        !$omp shared(y, a, size_y) &
        !$omp private(i)
        !$omp do simd schedule(static)
        do i = 1, size_y
            y(i) = a
        enddo
        !$omp end do simd nowait
        !$omp end parallel
        call this%perf_counter%end_measurement()
    end subroutine

    module subroutine apply_set_uniform_integer_core_cpu(this, y, a, size_y)
        class(op_set_uniform_cpu_t), intent(inout) :: this
        integer, dimension(*), intent(inout) :: y
        integer, intent(in) :: a
        integer(kind=INT64), intent(in) :: size_y
        integer(kind=INT64) :: i

        this%n_iterations = size_y

        call this%perf_counter%start_measurement()
        !$omp parallel default(none) &
        !$omp shared(y, a, size_y) &
        !$omp private(i)
        !$omp do simd schedule(static)
        do i = 1, size_y
            y(i) = a
        enddo
        !$omp end do simd nowait
        !$omp end parallel
        call this%perf_counter%end_measurement()
    end subroutine

    module subroutine apply_set_uniform_logical_core_cpu(this, y, a, size_y)
        class(op_set_uniform_cpu_t), intent(inout) :: this
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
            y(i) = a
        enddo
        !$omp end do simd nowait
        !$omp end parallel
        call this%perf_counter%end_measurement()
    end subroutine

end submodule
