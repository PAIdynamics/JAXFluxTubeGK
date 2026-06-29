module op_axpy_m
    use, intrinsic :: iso_fortran_env
    use, intrinsic :: iso_c_binding, only: C_PTR
    use genex_fortran_env_m, only: GP
    use op_base_m, only: op_base_t

    implicit none

    type, public, abstract, extends(op_base_t) :: op_axpy_t
        !! Operator to calculate
        !! y[i] = y[i] + a * x[i]
    contains
        procedure(initialize_axpy), deferred :: initialize

        generic, public :: apply => apply_axpy_1d, &
                                    apply_axpy_2d, &
                                    apply_axpy_3d, &
                                    apply_axpy_4d, &
                                    apply_axpy_5d

        procedure, private :: apply_axpy_1d
        procedure, private :: apply_axpy_2d
        procedure, private :: apply_axpy_3d
        procedure, private :: apply_axpy_4d
        procedure, private :: apply_axpy_5d

        procedure(apply_axpy_core), deferred :: apply_core
    end type

    interface
        subroutine initialize_axpy(this)
            !! Initializes the type
            import op_axpy_t
            class(op_axpy_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        subroutine apply_axpy_core(this, y, a, x, size_y)
            !! Apply the operator to the given input values
            !! NOTE: The arrays are assumed to have the same shape. No checks
            !!  are performed
            import op_axpy_t, GP, INT64
            class(op_axpy_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), dimension(*), intent(inout) :: y
            !! Vector to add to
            real(kind=GP), intent(in) :: a
            !! First scalar
            real(kind=GP), dimension(*),  intent(in) :: x
            !! Vector to add
            integer(kind=INT64), intent(in) :: size_y
            !! Size of the input vector
        end subroutine

    end interface

    type, public, extends(op_axpy_t) :: op_axpy_cpu_t
        !! Operator to calculate
        !! y[i] = y[i] + a * x[i]
        !! on the CPU
    contains
        procedure :: initialize => initialize_axpy_cpu
        procedure :: apply_core => apply_axpy_core_cpu
    end type

    interface
        module subroutine initialize_axpy_cpu(this)
            class(op_axpy_cpu_t), intent(inout) :: this
        end subroutine

        module subroutine apply_axpy_core_cpu(this, y, a, x, size_y)
            class(op_axpy_cpu_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), dimension(*), intent(inout) :: y
            !! Vector to add to
            real(kind=GP), intent(in) :: a
            !! First scalar
            real(kind=GP), dimension(*),  intent(in) :: x
            !! Vector to add
            integer(kind=INT64), intent(in) :: size_y
            !! Size of the input vector
        end subroutine
    end interface

#ifdef ENABLE_GPU
    type, public, extends(op_axpy_t) :: op_axpy_gpu_t
        !! Operator to calculate
        !! y[i] = y[i] + a * x[i]
        !! on the GPU
        type(C_PTR) :: op_cxx_pptr
        !! C pointer to the op_axpy_gpu_t C++ class instance pointer
    contains
        procedure :: initialize => initialize_axpy_gpu
        procedure :: apply_core => apply_axpy_core_gpu
        final :: finalize_axpy_gpu
    end type

    interface
        module subroutine initialize_axpy_gpu(this)
            class(op_axpy_gpu_t), intent(inout) :: this
        end subroutine

        module subroutine apply_axpy_core_gpu(this, y, a, x, size_y)
            class(op_axpy_gpu_t), intent(inout) :: this
            real(kind=GP), dimension(*), intent(inout) :: y
            real(kind=GP), intent(in) :: a
            real(kind=GP), dimension(*),  intent(in) :: x
            integer(kind=INT64), intent(in) :: size_y
        end subroutine

        module subroutine finalize_axpy_gpu(this)
            type(op_axpy_gpu_t), intent(inout) :: this
        end subroutine
    end interface
#endif

contains

    subroutine apply_axpy_1d(this, y, a, x)
        class(op_axpy_t), intent(inout) :: this
        real(kind=GP), contiguous, dimension(:), intent(inout) :: y
        real(kind=GP), intent(in) :: a
        real(kind=GP), contiguous, dimension(:), intent(in) :: x

        call this%apply_core(y, a, x, size(y, kind=INT64))
    end subroutine

    subroutine apply_axpy_2d(this, y, a, x)
        class(op_axpy_t), intent(inout) :: this
        real(kind=GP), contiguous, dimension(:,:), intent(inout) :: y
        real(kind=GP), intent(in) :: a
        real(kind=GP), contiguous, dimension(:,:), intent(in) :: x

        call this%apply_core(y, a, x, size(y, kind=INT64))
    end subroutine

    subroutine apply_axpy_3d(this, y, a, x)
        class(op_axpy_t), intent(inout) :: this
        real(kind=GP), contiguous, dimension(:,:,:), intent(inout) :: y
        real(kind=GP), intent(in) :: a
        real(kind=GP), contiguous, dimension(:,:,:), intent(in) :: x

        call this%apply_core(y, a, x, size(y, kind=INT64))
    end subroutine

    subroutine apply_axpy_4d(this, y, a, x)
        class(op_axpy_t), intent(inout) :: this
        real(kind=GP), contiguous, dimension(:,:,:,:), intent(inout) :: y
        real(kind=GP), intent(in) :: a
        real(kind=GP), contiguous, dimension(:,:,:,:), intent(in) :: x

        call this%apply_core(y, a, x, size(y, kind=INT64))
    end subroutine

    subroutine apply_axpy_5d(this, y, a, x)
        class(op_axpy_t), intent(inout) :: this
        real(kind=GP), contiguous, dimension(:,:,:,:,:), intent(inout) :: y
        real(kind=GP), intent(in) :: a
        real(kind=GP), contiguous, dimension(:,:,:,:,:), intent(in) :: x

        call this%apply_core(y, a, x, size(y, kind=INT64))
    end subroutine

end module op_axpy_m
