module op_copy_m
    use, intrinsic :: iso_fortran_env
    use, intrinsic :: iso_c_binding, only: C_PTR
    use genex_fortran_env_m, only: GP
    use op_base_m, only: op_base_t

    implicit none

    type, public, abstract, extends(op_base_t) :: op_copy_t
        !! Operator to set
        !! y[i] = x[i]
    contains
        procedure(initialize_copy), deferred :: initialize

        generic, public :: apply => apply_copy_real_1d,    &
                                    apply_copy_real_2d,    &
                                    apply_copy_real_3d,    &
                                    apply_copy_real_4d,    &
                                    apply_copy_real_5d,    &
                                    apply_copy_integer_1d, &
                                    apply_copy_integer_2d, &
                                    apply_copy_integer_3d, &
                                    apply_copy_integer_4d, &
                                    apply_copy_integer_5d, &
                                    apply_copy_logical_1d, &
                                    apply_copy_logical_2d, &
                                    apply_copy_logical_3d, &
                                    apply_copy_logical_4d, &
                                    apply_copy_logical_5d

        procedure, private :: apply_copy_real_1d
        procedure, private :: apply_copy_real_2d
        procedure, private :: apply_copy_real_3d
        procedure, private :: apply_copy_real_4d
        procedure, private :: apply_copy_real_5d

        procedure, private :: apply_copy_integer_1d
        procedure, private :: apply_copy_integer_2d
        procedure, private :: apply_copy_integer_3d
        procedure, private :: apply_copy_integer_4d
        procedure, private :: apply_copy_integer_5d

        procedure, private :: apply_copy_logical_1d
        procedure, private :: apply_copy_logical_2d
        procedure, private :: apply_copy_logical_3d
        procedure, private :: apply_copy_logical_4d
        procedure, private :: apply_copy_logical_5d

        procedure(apply_copy_real_core), &
            deferred, private :: apply_real_core
        procedure(apply_copy_integer_core), &
            deferred, private :: apply_integer_core
        procedure(apply_copy_logical_core), &
            deferred, private :: apply_logical_core

    end type

    interface

        subroutine initialize_copy(this)
            !! Initializes the type
            import op_copy_t
            class(op_copy_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        subroutine apply_copy_real_core(this, y, x, size_y)
            !! Apply the operator to the given input values
            !! NOTE: The arrays are assumed to have the same shape. No checks
            !!       are performed
            import op_copy_t, GP, INT64
            class(op_copy_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), dimension(*), intent(inout) :: y
            !! Vector to set
            real(kind=GP), dimension(*), intent(in) :: x
            !! Vector to set the y to
            integer(kind=INT64), intent(in) :: size_y
        end subroutine

        subroutine apply_copy_integer_core(this, y, x, size_y)
            !! Apply the operator to the given input values
            !! NOTE: The arrays are assumed to have the same shape. No checks
            !!       are performed
            import op_copy_t, GP, INT64
            class(op_copy_t), intent(inout) :: this
            !! Instance of the type
            integer, dimension(*), intent(inout) :: y
            !! Vector to set
            integer, dimension(*), intent(in) :: x
            !! Vector to set the y to
            integer(kind=INT64), intent(in) :: size_y
        end subroutine

        subroutine apply_copy_logical_core(this, y, x, size_y)
            !! Apply the operator to the given input values
            !! NOTE: The arrays are assumed to have the same shape. No checks
            !!       are performed
            import op_copy_t, GP, INT64
            class(op_copy_t), intent(inout) :: this
            !! Instance of the type
            logical, dimension(*), intent(inout) :: y
            !! Vector to set
            logical, dimension(*), intent(in) :: x
            !! Vector to set the y to
            integer(kind=INT64), intent(in) :: size_y
        end subroutine

    end interface

    type, public, extends(op_copy_t) :: op_copy_cpu_t
        !! Operator to set
        !! y[i] = x[i]
        !! on the cpu. The vector is correctly first touch initialized
        !! with OpenMP
    contains
        procedure :: initialize => initialize_copy_cpu
        procedure :: apply_real_core => apply_copy_real_core_cpu
        procedure :: apply_integer_core => apply_copy_integer_core_cpu
        procedure :: apply_logical_core => apply_copy_logical_core_cpu
    end type

    interface
        module subroutine initialize_copy_cpu(this)
            class(op_copy_cpu_t), intent(inout) :: this
        end subroutine

        module subroutine apply_copy_real_core_cpu(this, y, x, size_y)
            class(op_copy_cpu_t), intent(inout) :: this
            real(kind=GP), dimension(*), intent(inout) :: y
            real(kind=GP), dimension(*), intent(in) :: x
            integer(kind=INT64), intent(in) :: size_y
        end subroutine

        module subroutine apply_copy_integer_core_cpu(this, y, x, size_y)
            class(op_copy_cpu_t), intent(inout) :: this
            integer, dimension(*), intent(inout) :: y
            integer, dimension(*), intent(in) :: x
            integer(kind=INT64), intent(in) :: size_y
        end subroutine

        module subroutine apply_copy_logical_core_cpu(this, y, x, size_y)
            class(op_copy_cpu_t), intent(inout) :: this
            logical, dimension(*), intent(inout) :: y
            logical, dimension(*), intent(in) :: x
            integer(kind=INT64), intent(in) :: size_y
        end subroutine
    end interface

#ifdef ENABLE_GPU
    type, public, extends(op_copy_t) :: op_copy_gpu_t
        !! Operator to set
        !! y[i] = x[i]
        !! on the GPU
        !! NOTE: Operation with logical array is not supported on the GPU and
        !!       is done on the CPU instead
        type(C_PTR) :: op_cxx_pptr
        !! C pointer to the op_copy_gpu_t C++ class instance pointer
    contains
        procedure :: initialize => initialize_copy_gpu
        procedure :: apply_real_core => apply_copy_real_core_gpu
        procedure :: apply_integer_core => apply_copy_integer_core_gpu
        procedure :: apply_logical_core => apply_copy_logical_core_gpu
        final :: finalize_copy_gpu
    end type

    interface
        module subroutine initialize_copy_gpu(this)
            class(op_copy_gpu_t), intent(inout) :: this
        end subroutine

        module subroutine apply_copy_real_core_gpu(this, y, x, size_y)
            class(op_copy_gpu_t), intent(inout) :: this
            real(kind=GP), dimension(*), intent(inout) :: y
            real(kind=GP), dimension(*), intent(in) :: x
            integer(kind=INT64), intent(in) :: size_y
        end subroutine

        module subroutine apply_copy_integer_core_gpu(this, y, x, size_y)
            class(op_copy_gpu_t), intent(inout) :: this
            integer, dimension(*), intent(inout) :: y
            integer, dimension(*), intent(in) :: x
            integer(kind=INT64), intent(in) :: size_y
        end subroutine

        module subroutine apply_copy_logical_core_gpu(this, y, x, size_y)
            class(op_copy_gpu_t), intent(inout) :: this
            logical, dimension(*), intent(inout) :: y
            logical, dimension(*), intent(in) :: x
            integer(kind=INT64), intent(in) :: size_y
        end subroutine

        module subroutine finalize_copy_gpu(this)
            type(op_copy_gpu_t), intent(inout) :: this
        end subroutine
    end interface
#endif

contains

        !! Real subroutines
        subroutine apply_copy_real_1d(this, y, x)
            class(op_copy_t), intent(inout) :: this
            real(kind=GP), contiguous, dimension(:), intent(inout) :: y
            real(kind=GP), contiguous, dimension(:), intent(in) :: x

            call this%apply_real_core(y, x, size(y, kind=INT64))
        end subroutine

        subroutine apply_copy_real_2d(this, y, x)
            class(op_copy_t), intent(inout) :: this
            real(kind=GP), contiguous, dimension(:,:), intent(inout) :: y
            real(kind=GP), contiguous, dimension(:,:), intent(in) :: x

            call this%apply_real_core(y, x, size(y, kind=INT64))
        end subroutine

        subroutine apply_copy_real_3d(this, y, x)
            class(op_copy_t), intent(inout) :: this
            real(kind=GP), contiguous, dimension(:,:,:), intent(inout) :: y
            real(kind=GP), contiguous, dimension(:,:,:), intent(in) :: x

            call this%apply_real_core(y, x, size(y, kind=INT64))
        end subroutine

        subroutine apply_copy_real_4d(this, y, x)
            class(op_copy_t), intent(inout) :: this
            real(kind=GP), contiguous, dimension(:,:,:,:), intent(inout) :: y
            real(kind=GP), contiguous, dimension(:,:,:,:), intent(in) :: x

            call this%apply_real_core(y, x, size(y, kind=INT64))
        end subroutine

        subroutine apply_copy_real_5d(this, y, x)
            class(op_copy_t), intent(inout) :: this
            real(kind=GP), contiguous, dimension(:,:,:,:,:), &
                intent(inout) :: y
            real(kind=GP), contiguous, dimension(:,:,:,:,:), intent(in) :: x

            call this%apply_real_core(y, x, size(y, kind=INT64))
        end subroutine

        !! Integer subroutines
        subroutine apply_copy_integer_1d(this, y, x)
            class(op_copy_t), intent(inout) :: this
            integer, contiguous, dimension(:), intent(inout) :: y
            integer, contiguous, dimension(:), intent(in) :: x

            call this%apply_integer_core(y, x, size(y, kind=INT64))
        end subroutine

        subroutine apply_copy_integer_2d(this, y, x)
            class(op_copy_t), intent(inout) :: this
            integer, contiguous, dimension(:,:), intent(inout) :: y
            integer, contiguous, dimension(:,:), intent(in) :: x

            call this%apply_integer_core(y, x, size(y, kind=INT64))
        end subroutine

        subroutine apply_copy_integer_3d(this, y, x)
            class(op_copy_t), intent(inout) :: this
            integer, contiguous, dimension(:,:,:), intent(inout) :: y
            integer, contiguous, dimension(:,:,:), intent(in) :: x

            call this%apply_integer_core(y, x, size(y, kind=INT64))
        end subroutine

        subroutine apply_copy_integer_4d(this, y, x)
            class(op_copy_t), intent(inout) :: this
            integer, contiguous, dimension(:,:,:,:), intent(inout) :: y
            integer, contiguous, dimension(:,:,:,:), intent(in) :: x

            call this%apply_integer_core(y, x, size(y, kind=INT64))
        end subroutine

        subroutine apply_copy_integer_5d(this, y, x)
            class(op_copy_t), intent(inout) :: this
            integer, contiguous, dimension(:,:,:,:,:), &
                intent(inout) :: y
            integer, contiguous, dimension(:,:,:,:,:), intent(in) :: x

            call this%apply_integer_core(y, x, size(y, kind=INT64))
        end subroutine

        !! Logical subroutines
        subroutine apply_copy_logical_1d(this, y, x)
            class(op_copy_t), intent(inout) :: this
            logical, contiguous, dimension(:), intent(inout) :: y
            logical, contiguous, dimension(:), intent(in) :: x

            call this%apply_logical_core(y, x, size(y, kind=INT64))
        end subroutine

        subroutine apply_copy_logical_2d(this, y, x)
            class(op_copy_t), intent(inout) :: this
            logical, contiguous, dimension(:,:), intent(inout) :: y
            logical, contiguous, dimension(:,:), intent(in) :: x

            call this%apply_logical_core(y, x, size(y, kind=INT64))
        end subroutine

        subroutine apply_copy_logical_3d(this, y, x)
            class(op_copy_t), intent(inout) :: this
            logical, contiguous, dimension(:,:,:), intent(inout) :: y
            logical, contiguous, dimension(:,:,:), intent(in) :: x

            call this%apply_logical_core(y, x, size(y, kind=INT64))
        end subroutine

        subroutine apply_copy_logical_4d(this, y, x)
            class(op_copy_t), intent(inout) :: this
            logical, contiguous, dimension(:,:,:,:), intent(inout) :: y
            logical, contiguous, dimension(:,:,:,:), intent(in) :: x

            call this%apply_logical_core(y, x, size(y, kind=INT64))
        end subroutine

        subroutine apply_copy_logical_5d(this, y, x)
            class(op_copy_t), intent(inout) :: this
            logical, contiguous, dimension(:,:,:,:,:), &
                intent(inout) :: y
            logical, contiguous, dimension(:,:,:,:,:), intent(in) :: x

            call this%apply_logical_core(y, x, size(y, kind=INT64))
        end subroutine

end module op_copy_m
