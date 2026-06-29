module op_neut_mom_m
    !! Contains operators that calculate the velocity space moments for the
    !! use in the implementation of plasma-neutrals interaction operators

    use genex_fortran_env_m, only: GP
    use data_array_m, only: data_array_4d_t, data_array_5d_t
    use op_base_m, only: op_base_t
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t
    use op_set_uniform_m, only: op_set_uniform_cpu_t

    implicit none

    private

    type, public, abstract, extends(op_base_t) :: op_neut_mom_t
        !! Operator to calculate the velocity space moments, base
        class(mesh_5d_t), private, pointer :: mesh
        !! Pointer to the mesh
        type(dcomm_handler_t), private, pointer :: dcomm_handler
        !! MPI communications handler
        integer, private :: n_sp
        !! Number of species
        type(op_set_uniform_cpu_t), private :: op_set_uniform
        !! Operator for initializing buffers
    contains
        procedure(initialize_interface), deferred, public :: initialize
        !! Method to initialize the operator
        procedure(apply_interface), deferred, public :: apply
        !! Method to apply the operator
    end type

    abstract interface !op_neut_mom_t
        subroutine initialize_interface(this, dcomm_handler, mesh)
            !! Initializes the op_neut_mom_t type
            import op_neut_mom_t, dcomm_handler_t, mesh_5d_t
            class(op_neut_mom_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Pointer to communications handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
        end subroutine
        subroutine apply_interface(this, da_f_in, da_moments)
            !! Calculates the 0th, 1st and 2nd moment for each real-space point
            !! and each species.
            import op_neut_mom_t, data_array_4d_t, data_array_5d_t
            class(op_neut_mom_t), intent(inout) :: this
            !! Instance of the type
            class(data_array_5d_t), intent(in) :: da_f_in
            !! Input distribution function
            class(data_array_4d_t), intent(inout) :: da_moments
            !! Output moments
            !! The 0th, 1st and 2nd moment are stored in the 3rd index
        end subroutine
    end interface

    type, public, extends(op_neut_mom_t) :: op_neut_mom_cpu_t
        !! Operator to calculate the velocity space moments, cpu implementation
        real(kind=GP), private, allocatable, dimension(:) :: prefac_bps
        !! Prefactor for B parallel star
    contains
        procedure :: initialize => initialize_cpu
        procedure :: apply => apply_cpu
    end type

    interface !op_neut_mom_cpu_t
        module subroutine initialize_cpu(this, dcomm_handler, mesh)
            class(op_neut_mom_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_cpu(this, da_f_in, da_moments)
            class(op_neut_mom_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_4d_t), intent(inout) :: da_moments
        end subroutine
    end interface

end module
