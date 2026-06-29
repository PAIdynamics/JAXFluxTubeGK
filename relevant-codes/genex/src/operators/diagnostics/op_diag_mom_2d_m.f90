module op_diag_mom_2d_m
    use genex_fortran_env_m, only: GP
    use data_array_m, only: data_array_4d_t, data_array_5d_t
    use op_base_m, only: op_base_t
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t
    use op_set_uniform_m, only: op_set_uniform_cpu_t
    use bsg_operators_m, only: bsg_operators_t
    use bsg_types_m, only: vspace_bsg_t

    implicit none

    type, public, abstract, extends(op_base_t) :: op_diag_mom_2d_base_t
        !! Base operator to calculate the 2d moments associated with
        !! the density of the particle number n, parallel flow u_par,
        !! parallel energy E_par, perpendicular energy
        !! E_perp, the parallel energy fluxes Q_par and Q_perp, and
        !! the parallel and perpendicular diamagnetic moments K_par, K_perp.
        class(mesh_5d_t), private, pointer :: mesh
        !! Pointer to the mesh
        type(dcomm_handler_t), private, pointer :: dcomm_handler
        !! Pointer to the MPI communication handler
    contains
        procedure(apply), deferred :: apply
    end type

    interface
        subroutine apply(this, da_f, da_moments, diagnose_tpc)
            !! Applies the operator to the given input values
            import op_diag_mom_2d_base_t, data_array_4d_t, data_array_5d_t
            class(op_diag_mom_2d_base_t), target, intent(inout) :: this
            !! Instance of the type
            class(data_array_5d_t), intent(in) :: da_f
            !! Distribution function
            class(data_array_4d_t), intent(inout) :: da_moments
            !! Moments of the distribution function in the order
            !! n, u_par, E_par, E_perp, Q_par, Q_perp, K_par, K_perp
            !! NOTE: The moments are returned in normalized units. Only u_par,
            !!       Q_par and Q_perp are corrected by sqrt(2 / m) to account
            !!       for the different thermal velocities of the species.
            !!       They are normalised to the thermal velocity of a proton.
            !! NOTE: The geometrical prefactors of the diamagnetic
            !!       heat fluxes are not taken into account in the
            !!       diamagnetic moments K_par and K_perp.
            logical, optional, intent(in) :: diagnose_tpc
            !! Switch to compute TPC
        end subroutine
    end interface

    type, public, abstract, extends(op_diag_mom_2d_base_t) :: op_diag_mom_2d_t
        !! Operator to calculate the 2d moments n, u_par, E_par, E_perp, Q_par,
        !! Q_perp, K_par, K_perp using the grid approach.
        real(kind=GP), pointer, contiguous, dimension(:,:) :: absB
        !! Magnetic field
        real(kind=GP), pointer, contiguous, dimension(:,:) :: loss_cone
        !! Loss cone for TPC
        real(kind=GP), pointer, contiguous, dimension(:,:) :: curl_normb_y
        !! Parallel component of the curl of the magnetic field unit vector
        type(vspace_bsg_t), pointer, contiguous, dimension(:) :: vp_bsg
        !! Pointer to the vp grid with BSG
        real(kind=GP), pointer, contiguous, dimension(:) :: mu
        !! Values of the grid in mu direction
        real(kind=GP), pointer, contiguous, dimension(:) :: muw
        !! Integration weights in mu direction
        type(vspace_bsg_t), pointer, contiguous, dimension(:) :: vpw_bsg
        !! Pointer to weights in vp direction
        real(kind=GP), private, allocatable, dimension(:) :: prefac_bps
        !! Normalization factor for b_par^star
        real(kind=GP), private, allocatable, dimension(:) :: prefac_vth
        !! Prefactor containing the species dependent part of the thermal
        !! velocity
        real(kind=GP), private, allocatable, dimension(:) :: prefac_energy
        !! Prefactor for the energy normalization
        real(kind=GP), private, allocatable, dimension(:) :: prefac_heat_flux
        !! Prefactor for heat flux normalization
        real(kind=GP), private, allocatable, dimension(:) :: prefac_diam
        !! Prefactor the diamagnetic moments
        integer, pointer, contiguous, dimension(:) :: bsg_flags
        !! Pointer to the BSG flags
    contains
        procedure(initialize), deferred :: initialize
    end type

    interface
        module subroutine initialize(this, dcomm_handler, mesh, bsg_op)
            !! Initializes the type
            class(op_diag_mom_2d_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Comm handler for MPI communication
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Pointer to the mesh
            class(bsg_operators_t), target, intent(inout) :: bsg_op
            !! Pointer to the BSG operator
        end subroutine
    end interface

    type, public, extends(op_diag_mom_2d_t) :: op_diag_mom_2d_cpu_t
        !! Operator to calculate the 2d moments n, u_par, E_par, E_perp, Q_par,
        !! Q_perp, K_par, K_perp using the grid approach on the cpu
        type(op_set_uniform_cpu_t), private :: op_set_uniform
    contains
        procedure :: initialize => initialize_cpu
        procedure :: apply => apply_cpu
    end type

    interface
        module subroutine initialize_cpu(this, dcomm_handler, mesh, bsg_op)
            class(op_diag_mom_2d_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            class(bsg_operators_t), target, intent(inout) :: bsg_op
        end subroutine

        module subroutine apply_cpu(this, da_f,  da_moments, diagnose_tpc)
            class(op_diag_mom_2d_cpu_t), target, intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f
            class(data_array_4d_t), intent(inout) :: da_moments
            logical, optional, intent(in) :: diagnose_tpc
        end subroutine
    end interface

    type, public, abstract, extends(op_diag_mom_2d_base_t) :: &
                                                          op_diag_mom_2d_vspec_t
        !! Operator to calculate the 2d moments n, u_par, E_par, E_perp, Q_par,
        !! Q_perp, K_par, K_perp using the spectral approach.
        real(kind=GP), pointer, contiguous, dimension(:, :, :) :: &
                                                               mom_weights_vspec
        !! Pointer to moment spectral weights
        real(kind=GP), private, allocatable, dimension(:) :: prefac_vth
        !! Prefactor containing the species dependent part of the thermal
        !! velocity prefac_vth = sqrt(2/m)
        real(kind=GP), private, allocatable, dimension(:) :: prefac_diam
        !! Prefactor the diamagnetic moments
        real(kind=GP), private, allocatable, dimension(:) :: temp_scalings
        !! Temperature scaling
    contains
        procedure(initialize_vspec), deferred :: initialize
        !! Method to initialize the spectral operator
    end type

    interface
        module subroutine initialize_vspec(this, dcomm_handler, mesh)
            !! Initializes the type
            class(op_diag_mom_2d_vspec_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Comm handler for MPI communication
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
        end subroutine
    end interface

    type, public, extends(op_diag_mom_2d_vspec_t) :: op_diag_mom_2d_vspec_cpu_t
        !! Operator to calculate the 2d moments n, u_par, E_par, E_perp, Q_par,
        !! Q_perp, K_par, K_perp using the spectral approach on the cpu.
        type(op_set_uniform_cpu_t), private :: op_set_uniform
    contains
         procedure :: initialize => initialize_vspec_cpu
         procedure :: apply => apply_vspec_cpu
    end type

    interface
        module subroutine initialize_vspec_cpu(this, dcomm_handler, mesh)
            class(op_diag_mom_2d_vspec_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine
        module subroutine apply_vspec_cpu(this, da_f,  da_moments, diagnose_tpc)
            class(op_diag_mom_2d_vspec_cpu_t), target, intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f
            class(data_array_4d_t), intent(inout) :: da_moments
            logical, optional, intent(in) :: diagnose_tpc
            !! This argument is not currently used as TPC diagnostics are not
            !! supported with vspec
        end subroutine
    end interface

end module op_diag_mom_2d_m
