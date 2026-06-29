module op_mom_maxwells_eq_m
    !! Module defining an operator to calculate moments of the distribution
    !! function that are required to solve the quasi-neutrality equation
    !! and Ampere's law
    use, intrinsic :: iso_c_binding, only: C_PTR
    use genex_fortran_env_m, only: GP
    use data_array_m, only: data_array_2d_t, data_array_5d_t
    use op_base_m, only: op_base_t
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t
    use op_set_uniform_m, only: op_set_uniform_cpu_t
    use bsg_operators_m, only: bsg_operators_t

    implicit none

    private

    type, public, abstract, extends(op_base_t) :: op_mom_maxwells_eq_base_t
        !! Operator to calculate moments of the distribution function that
        !! are required to solve the quasi-neutrality equation and Ampere's
        !! law. Both equations can be formulated as a Helmholtz problem
        !!
        !! lambda * g - xi * div (co * grad(g)) = b
        !!
        !! For both equations equation b is nontrivial and calculated
        !! here. b can be interpreted as the gyrocenter charge density in the
        !! case of the quasi-neutrality equation and the gyrocenter current
        !! density in the case of Ampere's law. In addition, co is nontrival
        !! for the quasi-neutrality equation and calculated here. co is
        !! proportional to the mass density.

        class(mesh_5d_t), private, pointer :: mesh
        !! Pointer to the mesh
        type(dcomm_handler_t), private, pointer :: dcomm_handler
        !! MPI communications handler
        class(bsg_operators_t), private, pointer :: bsg_op
        !! BSG operators
    contains
        procedure(apply_interface), deferred, public :: apply
    end type

    abstract interface
        subroutine apply_interface(this, da_f_in, da_co_qn_eq, da_b_qn_eq, &
                                   da_b_amps_law, da_b_bpar_eq)
            !! Applies the operator to the given input values
            import op_mom_maxwells_eq_base_t, data_array_2d_t, data_array_5d_t
            class(op_mom_maxwells_eq_base_t), intent(inout) :: this
            !! Instance of the type
            class(data_array_5d_t), intent(in) :: da_f_in
            !! Input distribution function
            class(data_array_2d_t), intent(inout) :: da_co_qn_eq
            !! co field of the quasi-neutrality equation
            class(data_array_2d_t), intent(inout) :: da_b_qn_eq
            !! b field of the quasi-neutrality equation
            class(data_array_2d_t), intent(inout) :: da_b_amps_law
            !! b field of Ampere's law
            class(data_array_2d_t), intent(inout) :: da_b_bpar_eq
            !! b field of the bpar equation (parallel Ampere's law)
        end subroutine
    end interface

    type, public, abstract, extends(op_mom_maxwells_eq_base_t) :: &
                                                            op_mom_maxwells_eq_t
        !! Operator to calculate moments of the distribution function that
        !! are required to solve the quasi-neutrality equation and Ampere's
        !! law.
    contains
        procedure(initialize_interface), deferred, public :: initialize
    end type

    interface
        module subroutine initialize_interface(this, dcomm_handler, mesh, &
                                               bsg_op)
            !! Initializes the op_mom_maxwells_eq_t type
            class(op_mom_maxwells_eq_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Pointer to communications handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
            class(bsg_operators_t), target, intent(inout) :: bsg_op
            !! BSG operators
        end subroutine
    end interface

    type, public, extends(op_mom_maxwells_eq_t) :: op_mom_maxwells_eq_cpu_t
        !! Operator to calculate moments of the distribution function that
        !! are required to solve the quasi-neutrality equation on the CPU.
        real(kind=GP), private, allocatable, dimension(:) :: prefac_bps
        !! Prefactor for the bps term for every species
        real(kind=GP), private, allocatable, dimension(:) :: prefac_b_amp
        !! Array containing the normalization of the b term of Ampere's law
        !! for each species
        real(kind=GP), private, allocatable, dimension(:) :: prefac_b_qn
        !! Array containing the normalization of the b term of the
        !! quasi-neutrality equation for each species
        real(kind=GP), private, allocatable, dimension(:) :: prefac_co_qn
        !! Array containing the prefactor of the co term of the
        !! quasi-neutrality equation for each species
        real(kind=GP), private, allocatable, dimension(:) :: prefac_b_bpar
        !! Contains the normalization of the b term of the bpar equation
        type(op_set_uniform_cpu_t), private :: op_set_uniform
        !! Operator for initializing buffers
    contains
        procedure :: initialize => initialize_cpu
        procedure :: apply => apply_cpu
    end type

    interface
        module subroutine initialize_cpu(this, dcomm_handler, mesh, bsg_op)
            class(op_mom_maxwells_eq_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            class(bsg_operators_t), target, intent(inout) :: bsg_op
        end subroutine initialize_cpu

        module subroutine apply_cpu(this, da_f_in, da_co_qn_eq, da_b_qn_eq, &
                                    da_b_amps_law, da_b_bpar_eq)
            class(op_mom_maxwells_eq_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_2d_t), intent(inout) :: da_co_qn_eq
            class(data_array_2d_t), intent(inout) :: da_b_qn_eq
            class(data_array_2d_t), intent(inout) :: da_b_amps_law
            class(data_array_2d_t), intent(inout) :: da_b_bpar_eq
        end subroutine apply_cpu
    end interface

    type, public, abstract, extends(op_mom_maxwells_eq_base_t) :: &
                                                      op_mom_maxwells_eq_vspec_t
        !! Operator to calculate moments of the spectral distribution function
        !! that are required to solve the quasi-neutrality equation and Ampere's
        !! law.
    contains
        procedure(initialize_interface_vspec), deferred, public :: initialize
    end type

    interface
        module subroutine initialize_interface_vspec(this, dcomm_handler, mesh)
            !! Initializes the op_mom_maxwells_eq_vspec_t type
            class(op_mom_maxwells_eq_vspec_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Pointer to communications handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
         end subroutine
    end interface

    type, public, extends(op_mom_maxwells_eq_vspec_t) :: &
                                                  op_mom_maxwells_eq_vspec_cpu_t
        !! Operator to calculate moments of the spectral distribution function
        !! that are required to solve the quasi-neutrality equation and Ampere's
        !! law on the CPU.
        real(kind=GP), pointer, contiguous, dimension(:, :, :) :: &
                                                              mom_weights_vspec
        !! Pointer to moment spectral weights
        real(kind=GP), private, allocatable, dimension(:) :: prefac_b_amp
        !! Prefactor containing the normalization for the b term of
        !! Ampere's law
        real(kind=GP), private, allocatable, dimension(:) :: prefac_b_qn
        !! Prefactor containing the normalization for the b term of the
        !! quasi-neutrality equation
        real(kind=GP), private, allocatable, dimension(:) :: prefac_co_qn
        !! Array containing the prefactor of the co term of the
        !! quasi-neutrality equation for each species
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
            class(op_mom_maxwells_eq_vspec_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_vspec_cpu(this, da_f_in, da_co_qn_eq, &
                                          da_b_qn_eq, da_b_amps_law, &
                                          da_b_bpar_eq)
            class(op_mom_maxwells_eq_vspec_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_2d_t), intent(inout) :: da_co_qn_eq
            class(data_array_2d_t), intent(inout) :: da_b_qn_eq
            class(data_array_2d_t), intent(inout) :: da_b_amps_law
            class(data_array_2d_t), intent(inout) :: da_b_bpar_eq
        end subroutine
    end interface

#ifdef ENABLE_GPU
    type, public, extends(op_mom_maxwells_eq_t) :: op_mom_maxwells_eq_gpu_t
        !! Operator to calculate moments of the distribution function that
        !! are required to solve the quasi-neutrality equation on the GPU.
        type(C_PTR) :: op_cxx_pptr
        !! C pointer to the op_mom_maxwells_eq_gpu_t C++ class instance
        !! pointer
    contains
        procedure :: initialize => initialize_gpu
        procedure :: apply => apply_gpu
        final :: finalize_gpu
    end type op_mom_maxwells_eq_gpu_t

    interface
        module subroutine initialize_gpu(this, dcomm_handler, mesh, bsg_op)
            class(op_mom_maxwells_eq_gpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            class(bsg_operators_t), target, intent(inout) :: bsg_op
        end subroutine initialize_gpu

        module subroutine apply_gpu(this, da_f_in, da_co_qn_eq, da_b_qn_eq, &
                                    da_b_amps_law, da_b_bpar_eq)
            class(op_mom_maxwells_eq_gpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_2d_t), intent(inout) :: da_co_qn_eq
            class(data_array_2d_t), intent(inout) :: da_b_qn_eq
            class(data_array_2d_t), intent(inout) :: da_b_amps_law
            class(data_array_2d_t), intent(inout) :: da_b_bpar_eq
        end subroutine apply_gpu

        module subroutine finalize_gpu(this)
            type(op_mom_maxwells_eq_gpu_t), intent(inout) :: this
        end subroutine
    end interface
#endif

end module
