module op_mom_ohms_law_m
    !! Module defines an operator to calculate moments of the distribution
    !! function that are required to solve Ohm's law
    use genex_fortran_env_m, only: GP
    use data_array_m, only: data_array_2d_t, data_array_5d_t
    use op_base_m, only: op_base_t
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t
    use op_set_uniform_m, only: op_set_uniform_cpu_t
    use bsg_operators_m, only: bsg_operators_t
    ! NOTE: The following use only statement requires a manual renaming
    !       to the same name to mitigate an abnormal compile-time error
    !       with Intel Fortran 21.X.X. Within that error a submodule cannot
    !       find the name that has been associated here in the module.
    ! TODO: Remove the renaming if the compiler bug has been fixed.
    use, intrinsic :: iso_c_binding, only : C_PTR => C_PTR

    implicit none

    private

    type, public, abstract, extends(op_base_t) :: op_mom_ohms_law_base_t
        !! Operator to calculate moments of the distribution function that
        !! are required to solve Ohm's law. Ohm's law can be formulated as
        !! a Helmholtz problem
        !!
        !! lambda * g - xi * div (co * grad(g)) = b
        !!
        !! For Ohm's law lambda and b are nontrivial and calculated in this
        !! operator. b can be interpreted as the time derivative of the
        !! parallel current. lambda is proportional to the gyrocenter density.
        class(mesh_5d_t), private, pointer :: mesh
        !! Pointer to the mesh
        type(dcomm_handler_t), private, pointer :: dcomm_handler
        !! MPI communications handler
        class(bsg_operators_t), private, pointer :: bsg_op
        !! Pointer to the BSG operators
    contains
        procedure(apply_interface), deferred, public :: apply
    end type

    abstract interface
        subroutine apply_interface(this, da_f_in, da_dfdt_in, &
                                   da_lambda_ohms_law, da_b_ohms_law)
            !! Applies the operator to the given input values
            import op_mom_ohms_law_base_t, data_array_2d_t, data_array_5d_t
            class(op_mom_ohms_law_base_t), intent(inout) :: this
            !! Instance of the type
            class(data_array_5d_t), intent(in) :: da_f_in
            !! Input distribution function
            class(data_array_5d_t), intent(in) :: da_dfdt_in
            !! Input time derivative of the distribution function
            class(data_array_2d_t), intent(inout) :: da_lambda_ohms_law
            !! lambda field of the Helmholtz problem
            class(data_array_2d_t), intent(inout) :: da_b_ohms_law
            !! b field of the Helmholtz problem
        end subroutine
    end interface

    type, public, abstract, extends(op_mom_ohms_law_base_t) :: op_mom_ohms_law_t
        !! Operator to calculate moments of the distribution function that
        !! are required to solve Ohm's law.
    contains
        procedure(initialize_interface), deferred, public :: initialize
    end type

    interface
        module subroutine initialize_interface(this, dcomm_handler, mesh, &
                                               bsg_op)
            !! Initializes the op_mom_ohms_law_t type
            class(op_mom_ohms_law_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Pointer to communications handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
            class(bsg_operators_t), target, intent(inout) :: bsg_op
            !! Pointer to BSG operators
        end subroutine
    end interface

    type, public, extends(op_mom_ohms_law_t) :: op_mom_ohms_law_cpu_t
        !! Operator to calculate moments of the distribution function that
        !! are required to solve Ohm's law on the cpu.
        real(kind=GP), private, allocatable, dimension(:) :: prefac_bps
        !! Prefactor for the bps term for every species
        real(kind=GP), private, allocatable, dimension(:) :: prefac_norm_lambda
        !! Prefactor containing the normalization for the lambda term
        real(kind=GP), private, allocatable, dimension(:) :: prefac_norm_b
        !! Prefactor containing the normalization for the b term
        type(op_set_uniform_cpu_t), private :: op_set_uniform
        !! Operator for initializing buffers
    contains
        procedure :: initialize => initialize_cpu
        procedure :: apply => apply_cpu
    end type

    interface
        module subroutine initialize_cpu(this, dcomm_handler, mesh, bsg_op)
            class(op_mom_ohms_law_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            class(bsg_operators_t), target, intent(inout) :: bsg_op
        end subroutine

        module subroutine apply_cpu(this, da_f_in, da_dfdt_in, &
                                    da_lambda_ohms_law, da_b_ohms_law)
            class(op_mom_ohms_law_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_5d_t), intent(in) :: da_dfdt_in
            class(data_array_2d_t), intent(inout) :: da_lambda_ohms_law
            class(data_array_2d_t), intent(inout) :: da_b_ohms_law
        end subroutine
    end interface

    type, public, abstract, extends(op_mom_ohms_law_base_t) :: &
                                                         op_mom_ohms_law_vspec_t
        !! Operator to calculate moments of the spectral
        !! distribution function that are required to solve Ohm's law.
    contains
        procedure(initialize_interface_vspec), deferred, public :: initialize
    end type

    interface
        module subroutine initialize_interface_vspec(this, dcomm_handler, mesh)
            !! Initializes the op_mom_ohms_law_t type
            class(op_mom_ohms_law_vspec_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Pointer to communications handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
        end subroutine
    end interface

    type, public, extends(op_mom_ohms_law_vspec_t) :: &
                                                    op_mom_ohms_law_vspec_cpu_t
        !! Operator to calculate moments of the spectral
        !! distribution function that are required to solve Ohm's law
        !! on the CPU.
        real(kind=GP), pointer, contiguous, dimension(:, :, :) :: &
                                                               mom_weights_vspec
        !! Pointer to moment spectral weights
        real(kind=GP), private, allocatable, dimension(:) :: prefac_norm_lambda
        !! Prefactor containing the normalization for the lambda term
        real(kind=GP), private, allocatable, dimension(:) :: prefac_norm_b
        !! Prefactor containing the normalization for the b term
        real(kind=GP), private, allocatable, dimension(:) :: temp_scalings
        !! Temperature scaling
        type(op_set_uniform_cpu_t), private :: op_set_uniform
        !! Operator for initializing buffers
    contains
        procedure :: initialize => initialize_vspec_cpu
        procedure :: apply => apply_vspec_cpu
    end type

    interface
        module subroutine initialize_vspec_cpu(this, dcomm_handler, mesh)
            class(op_mom_ohms_law_vspec_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_vspec_cpu(this, da_f_in, da_dfdt_in, &
                                          da_lambda_ohms_law, da_b_ohms_law)
            class(op_mom_ohms_law_vspec_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_5d_t), intent(in) :: da_dfdt_in
            class(data_array_2d_t), intent(inout) :: da_lambda_ohms_law
            class(data_array_2d_t), intent(inout) :: da_b_ohms_law
        end subroutine
    end interface

#ifdef ENABLE_GPU
    type, public, extends(op_mom_ohms_law_t) :: op_mom_ohms_law_gpu_t
        !! Operator to calculate moments of the distribution function that
        !! are required to solve Ohm's law on the gpu.
        type(C_PTR) :: op_cxx_pptr
        !! C pointer to the op_mom_ohms_law_gpu_t C++ class instance pointer
    contains
        procedure :: initialize => initialize_gpu
        procedure :: apply => apply_gpu
        final :: finalize_gpu
    end type

    interface
        module subroutine initialize_gpu(this, dcomm_handler, mesh, bsg_op)
            class(op_mom_ohms_law_gpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            class(bsg_operators_t), target, intent(inout) :: bsg_op
        end subroutine

        module subroutine apply_gpu(this, da_f_in, da_dfdt_in, &
                                    da_lambda_ohms_law, da_b_ohms_law)
            class(op_mom_ohms_law_gpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_5d_t), intent(in) :: da_dfdt_in
            class(data_array_2d_t), intent(inout) :: da_lambda_ohms_law
            class(data_array_2d_t), intent(inout) :: da_b_ohms_law
        end subroutine

        module subroutine finalize_gpu(this)
            type(op_mom_ohms_law_gpu_t), intent(inout) :: this
        end subroutine
    end interface
#endif

end module
