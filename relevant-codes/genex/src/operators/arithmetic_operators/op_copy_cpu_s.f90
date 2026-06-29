submodule (op_copy_m) op_copy_cpu_m
    use, intrinsic :: iso_fortran_env
!$  use omp_lib

    implicit none

contains

    module subroutine initialize_copy_cpu(this)
        class(op_copy_cpu_t), intent(inout) :: this
        this%n_ls = 2
        this%n_flops = 0
    end subroutine

    module subroutine apply_copy_real_core_cpu(this, y, x, size_y)
        class(op_copy_cpu_t), intent(inout) :: this
        real(kind=GP), dimension(*), intent(inout) :: y
        real(kind=GP), dimension(*), intent(in) :: x
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

    module subroutine apply_copy_integer_core_cpu(this, y, x, size_y)
        class(op_copy_cpu_t), intent(inout) :: this
        integer, dimension(*), intent(inout) :: y
        integer, dimension(*), intent(in) :: x
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

    module subroutine apply_copy_logical_core_cpu(this, y, x, size_y)
        class(op_copy_cpu_t), intent(inout) :: this
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

end submodule
