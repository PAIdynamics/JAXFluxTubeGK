module op_set_uniform_m
    use, intrinsic :: iso_fortran_env
    use, intrinsic :: iso_c_binding, only: C_PTR
    use genex_fortran_env_m, only: GP
    use op_base_m, only: op_base_t

    implicit none

    type, public, abstract, extends(op_base_t) :: op_set_uniform_t
        !! Operator to calculate
        !! y[i] = a
    contains
        procedure(initialize_set_uniform), deferred :: initialize

        generic, public :: apply => apply_set_uniform_real_1d,    &
                                    apply_set_uniform_real_2d,    &
                                    apply_set_uniform_real_3d,    &
                                    apply_set_uniform_real_4d,    &
                                    apply_set_uniform_real_5d,    &
                                    apply_set_uniform_real_8d,    &
                                    apply_set_uniform_integer_1d, &
                                    apply_set_uniform_integer_2d, &
                                    apply_set_uniform_integer_3d, &
                                    apply_set_uniform_integer_4d, &
                                    apply_set_uniform_integer_5d, &
                                    apply_set_uniform_logical_1d, &
                                    apply_set_uniform_logical_2d, &
                                    apply_set_uniform_logical_3d, &
                                    apply_set_uniform_logical_4d, &
                                    apply_set_uniform_logical_5d

        procedure, private :: apply_set_uniform_real_1d
        procedure, private :: apply_set_uniform_real_2d
        procedure, private :: apply_set_uniform_real_3d
        procedure, private :: apply_set_uniform_real_4d
        procedure, private :: apply_set_uniform_real_5d
        procedure, private :: apply_set_uniform_real_8d

        procedure, private :: apply_set_uniform_integer_1d
        procedure, private :: apply_set_uniform_integer_2d
        procedure, private :: apply_set_uniform_integer_3d
        procedure, private :: apply_set_uniform_integer_4d
        procedure, private :: apply_set_uniform_integer_5d

        procedure, private :: apply_set_uniform_logical_1d
        procedure, private :: apply_set_uniform_logical_2d
        procedure, private :: apply_set_uniform_logical_3d
        procedure, private :: apply_set_uniform_logical_4d
        procedure, private :: apply_set_uniform_logical_5d

        procedure(apply_set_uniform_real_core), &
            deferred, private :: apply_real_core
        procedure(apply_set_uniform_integer_core), &
            deferred, private :: apply_integer_core
        procedure(apply_set_uniform_logical_core), &
            deferred, private :: apply_logical_core

    end type

    interface

        subroutine initialize_set_uniform(this)
            !! Initializes the type
            import op_set_uniform_t
            class(op_set_uniform_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        subroutine apply_set_uniform_real_core(this, y, a, size_y)
            !! Apply the operator to the given input values
            import op_set_uniform_t, GP, INT64
            class(op_set_uniform_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), dimension(*), intent(inout) :: y
            !! Vector to set
            real(kind=GP), intent(in) :: a
            !! Value to set the vector to
            integer(kind=INT64), intent(in) :: size_y
        end subroutine

        subroutine apply_set_uniform_integer_core(this, y, a, size_y)
            !! Apply the operator to the given input values
            import op_set_uniform_t, GP, INT64
            class(op_set_uniform_t), intent(inout) :: this
            !! Instance of the type
            integer, dimension(*), intent(inout) :: y
            !! Vector to set
            integer, intent(in) :: a
            !! Value to set the vector to
            integer(kind=INT64), intent(in) :: size_y
        end subroutine

        subroutine apply_set_uniform_logical_core(this, y, a, size_y)
            !! Apply the operator to the given input values
            import op_set_uniform_t, GP, INT64
            class(op_set_uniform_t), intent(inout) :: this
            !! Instance of the type
            logical, dimension(*), intent(inout) :: y
            !! Vector to set
            logical, intent(in) :: a
            !! Value to set the vector to
            integer(kind=INT64), intent(in) :: size_y
        end subroutine

    end interface

    type, public, extends(op_set_uniform_t) :: op_set_uniform_cpu_t
        !! Operator to set
        !! y[i] = a
        !! on the CPU. The vector is correctly first touch initialized
        !! with OpenMP
    contains
        procedure :: initialize => initialize_set_uniform_cpu
        procedure :: apply_real_core => apply_set_uniform_real_core_cpu
        procedure :: apply_integer_core => apply_set_uniform_integer_core_cpu
        procedure :: apply_logical_core => apply_set_uniform_logical_core_cpu
    end type

    interface
        module subroutine initialize_set_uniform_cpu(this)
            class(op_set_uniform_cpu_t), intent(inout) :: this
        end subroutine

        module subroutine apply_set_uniform_real_core_cpu(this, y, a, size_y)
            class(op_set_uniform_cpu_t), intent(inout) :: this
            real(kind=GP), dimension(*), intent(inout) :: y
            real(kind=GP), intent(in) :: a
            integer(kind=INT64), intent(in) :: size_y
        end subroutine

        module subroutine apply_set_uniform_integer_core_cpu(this, y, a, size_y)
            class(op_set_uniform_cpu_t), intent(inout) :: this
            integer, dimension(*), intent(inout) :: y
            integer, intent(in) :: a
            integer(kind=INT64), intent(in) :: size_y
        end subroutine

        module subroutine apply_set_uniform_logical_core_cpu(this, y, a, size_y)
            class(op_set_uniform_cpu_t), intent(inout) :: this
            logical, dimension(*), intent(inout) :: y
            logical, intent(in) :: a
            integer(kind=INT64), intent(in) :: size_y
        end subroutine
    end interface

#ifdef ENABLE_GPU
    type, public, extends(op_set_uniform_t) :: op_set_uniform_gpu_t
        !! Operator to set
        !! y[i] = a
        !! on the GPU
        !! NOTE: Operation with logical array is not supported on the GPU and
        !!       is done on the CPU instead
        type(C_PTR) :: op_cxx_pptr
        !! C pointer to the op_set_uniform_gpu_t C++ class instance pointer
    contains
        procedure :: initialize => initialize_set_uniform_gpu
        procedure :: apply_real_core => apply_set_uniform_real_core_gpu
        procedure :: apply_integer_core => apply_set_uniform_integer_core_gpu
        procedure :: apply_logical_core => apply_set_uniform_logical_core_gpu
        final :: finalize_set_uniform_gpu
    end type

    interface
        module subroutine initialize_set_uniform_gpu(this)
            class(op_set_uniform_gpu_t), intent(inout) :: this
        end subroutine

        module subroutine apply_set_uniform_real_core_gpu(this, y, a, size_y)
            class(op_set_uniform_gpu_t), intent(inout) :: this
            real(kind=GP), dimension(*), intent(inout) :: y
            real(kind=GP), intent(in) :: a
            integer(kind=INT64), intent(in) :: size_y
        end subroutine

        module subroutine apply_set_uniform_integer_core_gpu(this, y, a, size_y)
            class(op_set_uniform_gpu_t), intent(inout) :: this
            integer, dimension(*), intent(inout) :: y
            integer, intent(in) :: a
            integer(kind=INT64), intent(in) :: size_y
        end subroutine

        module subroutine apply_set_uniform_logical_core_gpu(this, y, a, size_y)
            class(op_set_uniform_gpu_t), intent(inout) :: this
            logical, dimension(*), intent(inout) :: y
            logical, intent(in) :: a
            integer(kind=INT64), intent(in) :: size_y
        end subroutine

        module subroutine finalize_set_uniform_gpu(this)
            type(op_set_uniform_gpu_t), intent(inout) :: this
        end subroutine
    end interface
#endif

contains

        ! Real subroutines
        subroutine apply_set_uniform_real_1d(this, y, a)
            class(op_set_uniform_t), intent(inout) :: this
            real(kind=GP), dimension(:), intent(inout) :: y
            real(kind=GP), intent(in) :: a

            call this%apply_real_core(y, a, size(y, kind=INT64))
        end subroutine

        subroutine apply_set_uniform_real_2d(this, y, a)
            class(op_set_uniform_t), intent(inout) :: this
            real(kind=GP), dimension(:,:), intent(inout) :: y
            real(kind=GP), intent(in) :: a

            call this%apply_real_core(y, a, size(y, kind=INT64))
        end subroutine

        subroutine apply_set_uniform_real_3d(this, y, a)
            class(op_set_uniform_t), intent(inout) :: this
            real(kind=GP), dimension(:,:,:), intent(inout) :: y
            real(kind=GP), intent(in) :: a

            call this%apply_real_core(y, a, size(y, kind=INT64))
        end subroutine

        subroutine apply_set_uniform_real_4d(this, y, a)
            class(op_set_uniform_t), intent(inout) :: this
            real(kind=GP), dimension(:,:,:,:), intent(inout) :: y
            real(kind=GP), intent(in) :: a

            call this%apply_real_core(y, a, size(y, kind=INT64))
        end subroutine

        subroutine apply_set_uniform_real_5d(this, y, a)
            class(op_set_uniform_t), intent(inout) :: this
            real(kind=GP), dimension(:,:,:,:,:), intent(inout) :: y
            real(kind=GP), intent(in) :: a

            call this%apply_real_core(y, a, size(y, kind=INT64))
        end subroutine

        subroutine apply_set_uniform_real_8d(this, y, a)
            class(op_set_uniform_t), intent(inout) :: this
            real(kind=GP), dimension(:,:,:,:,:,:,:,:), intent(inout) :: y
            real(kind=GP), intent(in) :: a

            call this%apply_real_core(y, a, size(y, kind=INT64))
        end subroutine

        ! Integer subroutines
        subroutine apply_set_uniform_integer_1d(this, y, a)
            class(op_set_uniform_t), intent(inout) :: this
            integer, dimension(:), intent(inout) :: y
            integer, intent(in) :: a

            call this%apply_integer_core(y, a, size(y, kind=INT64))
        end subroutine

        subroutine apply_set_uniform_integer_2d(this, y, a)
            class(op_set_uniform_t), intent(inout) :: this
            integer, dimension(:,:), intent(inout) :: y
            integer, intent(in) :: a

            call this%apply_integer_core(y, a, size(y, kind=INT64))
        end subroutine

        subroutine apply_set_uniform_integer_3d(this, y, a)
            class(op_set_uniform_t), intent(inout) :: this
            integer, dimension(:,:,:), intent(inout) :: y
            integer, intent(in) :: a

            call this%apply_integer_core(y, a, size(y, kind=INT64))
        end subroutine

        subroutine apply_set_uniform_integer_4d(this, y, a)
            class(op_set_uniform_t), intent(inout) :: this
            integer, dimension(:,:,:,:), intent(inout) :: y
            integer, intent(in) :: a

            call this%apply_integer_core(y, a, size(y, kind=INT64))
        end subroutine

        subroutine apply_set_uniform_integer_5d(this, y, a)
            class(op_set_uniform_t), intent(inout) :: this
            integer, dimension(:,:,:,:,:), intent(inout) :: y
            integer, intent(in) :: a

            call this%apply_integer_core(y, a, size(y, kind=INT64))
        end subroutine

        ! Logical subroutines
        subroutine apply_set_uniform_logical_1d(this, y, a)
            class(op_set_uniform_t), intent(inout) :: this
            logical, dimension(:), intent(inout) :: y
            logical, intent(in) :: a

            call this%apply_logical_core(y, a, size(y, kind=INT64))
        end subroutine

        subroutine apply_set_uniform_logical_2d(this, y, a)
            class(op_set_uniform_t), intent(inout) :: this
            logical, dimension(:,:), intent(inout) :: y
            logical, intent(in) :: a

            call this%apply_logical_core(y, a, size(y, kind=INT64))
        end subroutine

        subroutine apply_set_uniform_logical_3d(this, y, a)
            class(op_set_uniform_t), intent(inout) :: this
            logical, dimension(:,:,:), intent(inout) :: y
            logical, intent(in) :: a

            call this%apply_logical_core(y, a, size(y, kind=INT64))
        end subroutine

        subroutine apply_set_uniform_logical_4d(this, y, a)
            class(op_set_uniform_t), intent(inout) :: this
            logical, dimension(:,:,:,:), intent(inout) :: y
            logical, intent(in) :: a

            call this%apply_logical_core(y, a, size(y, kind=INT64))
        end subroutine

        subroutine apply_set_uniform_logical_5d(this, y, a)
            class(op_set_uniform_t), intent(inout) :: this
            logical, dimension(:,:,:,:,:), intent(inout) :: y
            logical, intent(in) :: a

            call this%apply_logical_core(y, a, size(y, kind=INT64))
        end subroutine

end module op_set_uniform_m
