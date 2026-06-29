module op_neut_evolve_m
    !! Operator to evolve the (fluid) evolution equations of the
    !! neutrals species

    use genex_fortran_env_m, only: GP
    use data_array_m, only: data_array_4d_t
    use op_base_m, only: op_base_t
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t
    use params_neutrals_m, only: n_neut_supported

    implicit none

    type, public, abstract, extends(op_base_t) :: op_neut_evolve_t
        !! Base type for all kinds of models for the evolution of neutrals
        type(dcomm_handler_t), private, pointer :: dcomm_handler
        !! MPI communications handler
        class(mesh_5d_t), private, pointer :: mesh
        !! Pointer to the mesh

    contains
        procedure(initialize_evolve_interface), deferred :: initialize
        !! Method to initialize the evolution operator
        procedure(apply_evolve_interface), deferred :: apply
        !! Method to evolve the neutrals species states
    end type

    interface !op_neut_evolve_t
        subroutine initialize_evolve_interface(this, dcomm_handler, mesh)
            import op_neut_evolve_t, dcomm_handler_t, mesh_5d_t
            class(op_neut_evolve_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Pointer to communications handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Pointer to the mesh
        end subroutine

        subroutine apply_evolve_interface(this, da_n_sources, da_n_in, da_n_out)
            import op_neut_evolve_t, data_array_4d_t
            class(op_neut_evolve_t), intent(inout) :: this
            !! Instance of the type
            class(data_array_4d_t), intent(in) :: da_n_sources
            !! Sources due to neutrals-plasma interactions
            class(data_array_4d_t), intent(in) :: da_n_in
            !! Input state of the neutrals species
            class(data_array_4d_t), intent(inout) :: da_n_out
            !! Output state of the neutrals species
        end subroutine
    end interface

    type, public, extends(op_neut_evolve_t) :: op_neut_evolve_dummy_t
        !! Concrete type that implements the evolution of neutrals with a dummy
        !! model (of fluid type).

    contains
        procedure :: initialize => initialize_evolve_dummy
        procedure :: apply => apply_evolve_dummy
    end type

    interface !op_neut_evolve_dummy_t
        module subroutine initialize_evolve_dummy(this, dcomm_handler, mesh)
            class(op_neut_evolve_dummy_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_evolve_dummy(this, da_n_sources, &
                                               da_n_in, da_n_out)
            class(op_neut_evolve_dummy_t), intent(inout) :: this
            class(data_array_4d_t), intent(in) :: da_n_sources
            class(data_array_4d_t), intent(in) :: da_n_in
            class(data_array_4d_t), intent(inout) :: da_n_out
        end subroutine
    end interface

    type, public, abstract, extends(op_neut_evolve_t) :: op_neut_evolve_1mom_t
        !! Abstract type that for operators that implement the evolution of
        !! neutrals with a one moment (of fluid type) model:
        !! only the density is evolved using a diffusion equation.

        ! The following quantities are attributes of the abstract class,
        ! as all child classes are currently implemented using central finite
        ! difference (cfd) in cylindrical coordinates.
        real(kind=GP) :: prefac_cd_xz
        !! Prefactor for perpendicular central difference stencil
        real(kind=GP) :: prefac_cd_phi
        !! Prefactor for toroidal central difference stencil
        real(kind=GP), dimension(n_neut_supported) :: prefac_diff_coef
        !! Prefactor for the normalized diffusion coefficient

    contains
        procedure(initialize_evolve_1mom), deferred :: initialize
        procedure(apply_evolve_1mom), deferred :: apply
    end type

    interface !op_neut_evolve_1mom_t
        module subroutine initialize_evolve_1mom(this, dcomm_handler, mesh)
            class(op_neut_evolve_1mom_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_evolve_1mom(this, da_n_sources, &
                                            da_n_in, da_n_out)
            class(op_neut_evolve_1mom_t), intent(inout) :: this
            class(data_array_4d_t), intent(in) :: da_n_sources
            class(data_array_4d_t), intent(in) :: da_n_in
            class(data_array_4d_t), intent(inout) :: da_n_out
        end subroutine
    end interface

    type, public, extends(op_neut_evolve_1mom_t) :: op_neut_evolve_1mom_2cfd_t
        !! Concrete type using 2nd order CFD stencil on abstract staggered grid
    contains
        procedure :: initialize => initialize_evolve_1mom_2cfd
        procedure :: apply => apply_evolve_1mom_2cfd
    end type

    interface !op_neut_evolve_1mom_2cfd_t
        module subroutine initialize_evolve_1mom_2cfd(this, dcomm_handler, mesh)
            class(op_neut_evolve_1mom_2cfd_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_evolve_1mom_2cfd(this, da_n_sources, &
                                                 da_n_in, da_n_out)
            class(op_neut_evolve_1mom_2cfd_t), intent(inout) :: this
            class(data_array_4d_t), intent(in) :: da_n_sources
            class(data_array_4d_t), intent(in) :: da_n_in
            class(data_array_4d_t), intent(inout) :: da_n_out
        end subroutine
    end interface

    type, public, extends(op_neut_evolve_1mom_t) :: op_neut_evolve_1mom_4cfd_t
        !! Concrete type using 4th order CFD stencil on the fully developed
        !! equation
        real(kind=GP) :: ddx_jacobian
        !! Derivative of the jacobian along x
    contains
        procedure :: initialize => initialize_evolve_1mom_4cfd
        procedure :: apply => apply_evolve_1mom_4cfd
    end type

    interface !op_neut_evolve_1mom_4cfd_t
        module subroutine initialize_evolve_1mom_4cfd(this, dcomm_handler, mesh)
            class(op_neut_evolve_1mom_4cfd_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_evolve_1mom_4cfd(this, da_n_sources, &
                                                 da_n_in, da_n_out)
            class(op_neut_evolve_1mom_4cfd_t), intent(inout) :: this
            class(data_array_4d_t), intent(in) :: da_n_sources
            class(data_array_4d_t), intent(in) :: da_n_in
            class(data_array_4d_t), intent(inout) :: da_n_out
        end subroutine
    end interface

    type, public, extends(op_neut_evolve_1mom_t) :: op_neut_evolve_1mom_diff_t
        !! Concrete type using 2nd order CFD stencil on abstract staggered grid
        !! with non-constant diffucion coefficient
    contains
        procedure :: initialize => initialize_evolve_1mom_diff
        procedure :: apply => apply_evolve_1mom_diff
    end type

    interface !op_neut_evolve_1mom_diff_t
        module subroutine initialize_evolve_1mom_diff(this, dcomm_handler, mesh)
            class(op_neut_evolve_1mom_diff_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_evolve_1mom_diff(this, da_n_sources, &
                                                 da_n_in, da_n_out)
            class(op_neut_evolve_1mom_diff_t), intent(inout) :: this
            class(data_array_4d_t), intent(in) :: da_n_sources
            class(data_array_4d_t), intent(in) :: da_n_in
            class(data_array_4d_t), intent(inout) :: da_n_out
        end subroutine
    end interface
end module
