module op_lin_comb_m
    use, intrinsic :: iso_fortran_env
    use, intrinsic :: iso_c_binding, only: C_PTR
    use genex_fortran_env_m, only: GP
    use op_base_m, only: op_base_t

    implicit none

    type, public, abstract, extends(op_base_t) :: op_lin_comb_t
        !! Operator to calculate
        !! y[i] = a1 * x1[i] + a2 * x2[i]
    contains

        procedure(initialize_lin_comb), deferred :: initialize

        generic, public :: apply => apply_lin_comb_1d, &
                                    apply_lin_comb_2d, &
                                    apply_lin_comb_3d, &
                                    apply_lin_comb_4d, &
                                    apply_lin_comb_5d

        procedure, private :: apply_lin_comb_1d
        procedure, private :: apply_lin_comb_2d
        procedure, private :: apply_lin_comb_3d
        procedure, private :: apply_lin_comb_4d
        procedure, private :: apply_lin_comb_5d

        procedure(apply_lin_comb_core), deferred, private :: apply_core

    end type

    interface
        subroutine initialize_lin_comb(this)
            !! Initializes the type
            import op_lin_comb_t
            class(op_lin_comb_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        subroutine apply_lin_comb_core(this, y, a1, x1, a2, x2, size_y)
            !! Apply the operator to the given input values
            !! NOTE: The arrays are assumed to have the same shape. No checks
            !!       are performed
            import op_lin_comb_t, GP, INT64
            class(op_lin_comb_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), dimension(*), intent(inout) :: y
            !! Output vector
            real(kind=GP), intent(in) :: a1
            !! Input scalar
            real(kind=GP), dimension(*), intent(in) :: x1
            !! Input vector
            real(kind=GP), intent(in) :: a2
            !! Input scalar
            real(kind=GP), dimension(*), intent(in) :: x2
            !! Input vector
            integer(kind=INT64), intent(in) :: size_y
            !! Size of the input vectors
        end subroutine

    end interface

    type, public, extends(op_lin_comb_t) :: op_lin_comb_cpu_t
        !! Operator to calculate
        !! y[i] = a1 * x1[i] + a2 * x2[i]
        !! on the CPU
    contains
        procedure :: initialize => initialize_lin_comb_cpu
        procedure, private :: apply_core => apply_lin_comb_core_cpu
    end type

    interface
        module subroutine initialize_lin_comb_cpu(this)
            class(op_lin_comb_cpu_t), intent(inout) :: this
        end subroutine

        module subroutine apply_lin_comb_core_cpu(this, y, a1, x1, a2, x2, &
                                                  size_y)
            !! Apply the operator to the given input values
            !! NOTE: The arrays are assumed to have the same shape. No checks
            !!       are performed
            class(op_lin_comb_cpu_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), dimension(*), intent(inout) :: y
            !! Output vector
            real(kind=GP), intent(in) :: a1
            !! Input scalar
            real(kind=GP), dimension(*), intent(in) :: x1
            !! Input vector
            real(kind=GP), intent(in) :: a2
            !! Input scalar
            real(kind=GP), dimension(*), intent(in) :: x2
            !! Input vector
            integer(kind=INT64), intent(in) :: size_y
            !! Size of the input vectors
        end subroutine
    end interface

#ifdef ENABLE_GPU
    type, public, extends(op_lin_comb_t) :: op_lin_comb_gpu_t
        !! Operator to calculate
        !! y[i] = a1 * x1[i] + a2 * x2[i]
        !! on the GPU
        type(C_PTR) :: op_cxx_pptr
        !! C pointer to the op_lin_comb_gpu_t C++ class instance pointer
    contains
        procedure :: initialize => initialize_lin_comb_gpu
        procedure, private :: apply_core => apply_lin_comb_core_gpu
        final :: finalize_lin_comb_gpu
    end type

    interface
        module subroutine initialize_lin_comb_gpu(this)
            class(op_lin_comb_gpu_t), intent(inout) :: this
        end subroutine

        module subroutine apply_lin_comb_core_gpu(this, y, a1, x1, a2, x2, &
                                                  size_y)
            class(op_lin_comb_gpu_t), intent(inout) :: this
            real(kind=GP), dimension(*), intent(inout) :: y
            real(kind=GP), intent(in) :: a1
            real(kind=GP), dimension(*), intent(in) :: x1
            real(kind=GP), intent(in) :: a2
            real(kind=GP), dimension(*), intent(in) :: x2
            integer(kind=INT64), intent(in) :: size_y
        end subroutine

        module subroutine finalize_lin_comb_gpu(this)
            type(op_lin_comb_gpu_t), intent(inout) :: this
        end subroutine
    end interface
#endif

contains

    subroutine apply_lin_comb_1d(this, y, a1, x1, a2, x2)
        class(op_lin_comb_t), intent(inout) :: this
        real(kind=GP), contiguous, dimension(:), intent(inout) :: y
        real(kind=GP), intent(in) :: a1
        real(kind=GP), contiguous, dimension(:), intent(in) :: x1
        real(kind=GP), intent(in) :: a2
        real(kind=GP), contiguous, dimension(:), intent(in) :: x2

        call this%apply_core(y, a1, x1, a2, x2, size(y, kind=INT64))
    end subroutine

    subroutine apply_lin_comb_2d(this, y, a1, x1, a2, x2)
        class(op_lin_comb_t), intent(inout) :: this
        real(kind=GP), contiguous, dimension(:,:), intent(inout) :: y
        real(kind=GP), intent(in) :: a1
        real(kind=GP), contiguous, dimension(:,:), intent(in) :: x1
        real(kind=GP), intent(in) :: a2
        real(kind=GP), contiguous, dimension(:,:), intent(in) :: x2

        call this%apply_core(y, a1, x1, a2, x2, size(y, kind=INT64))
    end subroutine

    subroutine apply_lin_comb_3d(this, y, a1, x1, a2, x2)
        class(op_lin_comb_t), intent(inout) :: this
        real(kind=GP), contiguous, dimension(:,:,:), intent(inout) :: y
        real(kind=GP), intent(in) :: a1
        real(kind=GP), contiguous, dimension(:,:,:), intent(in) :: x1
        real(kind=GP), intent(in) :: a2
        real(kind=GP), contiguous, dimension(:,:,:), intent(in) :: x2

        call this%apply_core(y, a1, x1, a2, x2, size(y, kind=INT64))
    end subroutine

    subroutine apply_lin_comb_4d(this, y, a1, x1, a2, x2)
        class(op_lin_comb_t), intent(inout) :: this
        real(kind=GP), contiguous, dimension(:,:,:,:), intent(inout) :: y
        real(kind=GP), intent(in) :: a1
        real(kind=GP), contiguous, dimension(:,:,:,:), intent(in) :: x1
        real(kind=GP), intent(in) :: a2
        real(kind=GP), contiguous, dimension(:,:,:,:), intent(in) :: x2

        call this%apply_core(y, a1, x1, a2, x2, size(y, kind=INT64))
    end subroutine

    subroutine apply_lin_comb_5d(this, y, a1, x1, a2, x2)
        class(op_lin_comb_t), intent(inout) :: this
        real(kind=GP), contiguous, dimension(:,:,:,:,:), intent(inout) :: y
        real(kind=GP), intent(in) :: a1
        real(kind=GP), contiguous, dimension(:,:,:,:,:), intent(in) :: x1
        real(kind=GP), intent(in) :: a2
        real(kind=GP), contiguous, dimension(:,:,:,:,:), intent(in) :: x2

        call this%apply_core(y, a1, x1, a2, x2, size(y, kind=INT64))
    end subroutine

end module op_lin_comb_m
