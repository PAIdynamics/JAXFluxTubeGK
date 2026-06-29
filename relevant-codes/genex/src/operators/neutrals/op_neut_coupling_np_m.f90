module op_neut_coupling_np_m
    !! This module contains operators to couple the effect of the neutrals on
    !! the plasma evolution (neutrals to plasma == np).
    !! This is achieved through collision operators and two neutrals-plasma
    !! reactions are considered: ionization (iz) and recombination (rc).
    !! The concrete, i.e. non abstract, derived types implement specific
    !! collision operators, eg. op_neut_coupling_np_krook_t use Krook-like
    !! operators.

    use genex_fortran_env_m, only: GP
    use data_array_m, only: data_array_4d_t, data_array_5d_t
    use op_base_m, only: op_base_t
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t
    use neut_coupling_setup_m, only: setup_neut_ions_maps, &
                                     setup_reaction_rates, &
                                     setup_cooling_rates

    ! From REAX
    use rate_coefficients_m, only: rate_coefficient_t

    implicit none

    type, public, abstract, extends(op_base_t) :: op_neut_coupling_np_t
        !! Abstract type for coupling operators that calculate the effect of
        !! the neutrals on the plasma.
        type(dcomm_handler_t), private, pointer :: dcomm_handler
        !! MPI communications handler
        class(mesh_5d_t), private, pointer :: mesh
        !! Pointer to the mesh
        integer, private, dimension(:), allocatable :: map_neut_to_ion
        !! Map the neutrals indices to their corresponding ion indices
        integer, private, dimension(:), allocatable :: map_ion_to_neut
        !! Map the ion indices to their corresponding neutrals indices

    contains
        procedure, private :: initialize_parent

        procedure(initialize_coupling_np), deferred :: initialize
        !! Method to initialize the operator
        procedure(apply_coupling_np), deferred :: apply
        !! Method to apply the operator: compute modifications of the plasma
        !! distribution function due to neutrals.
    end type

    abstract interface !op_neut_coupling_np_t
        subroutine initialize_coupling_np(this, dcomm_handler, mesh)
            import op_neut_coupling_np_t, dcomm_handler_t, mesh_5d_t
            class(op_neut_coupling_np_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Pointer to communications handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Pointer to the mesh
        end subroutine

        subroutine apply_coupling_np(this, da_moments, da_f_in, &
                                     da_n_in, da_f_out)
            import op_neut_coupling_np_t, data_array_4d_t, data_array_5d_t
            class(op_neut_coupling_np_t), intent(inout) :: this
            !! Instance of the type
            class(data_array_4d_t), intent(in) :: da_moments
            !! Moments of the  distribution function of the plasma species
            class(data_array_5d_t), intent(in) :: da_f_in
            !! Input distribution function of the plasma species
            class(data_array_4d_t), intent(in) :: da_n_in
            !! Input neutrals species states (moments)
            class(data_array_5d_t), intent(inout) :: da_f_out
            !! Output distribution function of the plasma species (changed
            !! by the plasma-neutral collisions)
        end subroutine
    end interface

    interface
        module subroutine initialize_parent(this, dcomm_handler, mesh)
            class(op_neut_coupling_np_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Pointer to communications handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Pointer to the mesh
        end subroutine
    end interface

    type, public, extends(op_neut_coupling_np_t) :: op_neut_coupling_np_none_t
        !! Concrete neutrals-plasma coupling operator that do nothing.
        ! NOTE: This type is implemented for code development purposes only and
        !       will be removed as soon as more relevant coupling operators
        !       have been implemented and tested.
    contains
        procedure :: initialize => initialize_coupling_np_none
        procedure :: apply => apply_coupling_np_none
    end type

    interface !op_neut_coupling_np_none_t
        module subroutine initialize_coupling_np_none(this, dcomm_handler, &
                                                       mesh)
            class(op_neut_coupling_np_none_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_coupling_np_none(this, da_moments, &
                                                  da_f_in, da_n_in, &
                                                  da_f_out)
            class(op_neut_coupling_np_none_t), intent(inout) :: this
            class(data_array_4d_t), intent(in) :: da_moments
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_4d_t), intent(in) :: da_n_in
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine
    end interface

    type, public, extends(op_neut_coupling_np_t) :: &
                                            op_neut_coupling_np_krook_t
        !! Concrete neutrals-plasma coupling operator implementing a
        !! conservative Krook operator for the neutrals density.
        type(rate_coefficient_t) :: k_iz
        !! Ionization rate coefficient
        type(rate_coefficient_t) :: k_rc
        !! Recombination rate coefficient
        type(rate_coefficient_t) :: w_iz
        !! Ionization cooling coefficient
        type(rate_coefficient_t) :: w_rc
        !! Recombination cooling coefficient
        integer :: ion
        !! Species index of the ion used in the reaction
        integer :: elec
        !! Species index of the electron used in the reaction
        integer :: neut
        !! Species index of the neutrals used in the reaction
        real(kind=GP) :: prefactor_bps_ion
        !! Prefactor for B parallel star for the ion
        real(kind=GP) :: prefactor_bps_elec
        !! Prefactor for B parallel star for the electron
        real(kind=GP) :: gamma_u
        !! Coefficient for neutrals velocity
        real(kind=GP) :: gamma_T
        !! Coefficient for neutrals temperature
        real(kind=GP) :: gamma_W
        !! Coefficient electron temperature cooling

    contains
        procedure :: initialize => initialize_coupling_np_krook
        procedure :: apply => apply_coupling_np_krook
    end type

    interface !op_neut_coupling_np_krook_t
        module subroutine initialize_coupling_np_krook(this, &
                                                             dcomm_handler, &
                                                             mesh)
            class(op_neut_coupling_np_krook_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_coupling_np_krook(this, da_moments, &
                                                        da_f_in, da_n_in, &
                                                        da_f_out)
            class(op_neut_coupling_np_krook_t), intent(inout) :: this
            class(data_array_4d_t), intent(in) :: da_moments
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_4d_t), intent(in) :: da_n_in
            class(data_array_5d_t), intent(inout) :: da_f_out
        end subroutine
    end interface

end module
