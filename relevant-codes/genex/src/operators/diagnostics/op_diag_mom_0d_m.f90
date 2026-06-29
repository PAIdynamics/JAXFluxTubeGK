module op_diag_mom_0d_m
    use genex_fortran_env_m, only: GP
    use data_array_m, only: data_array_2d_t, data_array_5d_t
    use op_base_m, only: op_base_t
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t
    use op_set_uniform_m, only: op_set_uniform_cpu_t
    use bsg_operators_m, only: bsg_operators_t
    use bsg_types_m, only: vspace_bsg_t

    implicit none

    type, public, abstract, extends(op_base_t) :: op_diag_mom_0d_base_t
        !! Base operator to calculate the total particle number n, parallel
        !! flow u_par, parallel energy E_par, perpendicular energy E_perp,
        !! electrostatic energy E_es and canonical toroidal momentum p_phi.
        class(mesh_5d_t), private, pointer :: mesh
        !! Pointer to the mesh
        type(dcomm_handler_t), private, pointer :: dcomm_handler
        !! Pointer to the communication handler
        class(bsg_operators_t), private, pointer :: bsg_op
        !! Pointer to the BSG operator
    contains
        procedure(apply), deferred :: apply
    end type

    interface
        subroutine apply(this, da_f, da_es_pot, da_moments)
            !! Applies the operator to the given input values
            import op_diag_mom_0d_base_t, data_array_2d_t, data_array_5d_t
            class(op_diag_mom_0d_base_t), intent(inout) :: this
            !! Instance of the type
            class(data_array_5d_t), intent(in) :: da_f
            !! Distribution function
            class(data_array_2d_t), intent(in) :: da_es_pot
            !! Electrostatic potential
            class(data_array_2d_t), intent(inout) :: da_moments
            !! Moments of the distribution function in the order
            !! n, u_par, E_par, E_perp, E_es, p_phi
            !! NOTE: The moments are returned in normalized units. Only u_par
            !!       is corrected by sqrt(2 / m) to account for the different
            !!       thermal velocities of the species. They are normalised to
            !!       the thermal velocity of a proton.
        end subroutine
    end interface

    type, public, abstract, extends(op_diag_mom_0d_base_t) :: op_diag_mom_0d_t
        !! Operator to calculate the total particle number n, parallel
        !! flow u_par, parallel energy E_par, perpendicular energy E_perp,
        !! electrostatic energy E_es and canonical toroidal momentum p_phi.
        real(kind=GP), pointer, contiguous, dimension(:,:) :: absB
        !! Magnetic field
        real(kind=GP), pointer, contiguous, dimension(:,:) :: normb_tor
        !! Toroidal magnetic field
        real(kind=GP), pointer, contiguous, dimension(:,:) :: curl_normb_y
        !! Parallel component of the curl of the magnetic field unit vector
        real(kind=GP), pointer, contiguous, dimension(:,:) :: RZw
        !! Quadrature weights in RZ direction
        real(kind=GP), pointer, contiguous, dimension(:,:) :: jacobian
        !! Values of the jacobian of the mesh
        type(vspace_bsg_t), pointer, contiguous, dimension(:) :: vp_bsg
        !! Pointer to the vp grid with BSG
        real(kind=GP), pointer, contiguous, dimension(:) :: mu
        !! Values of the grid in mu direction
        real(kind=GP), pointer, contiguous, dimension(:) :: muw
        !! Quadrature weights in mu direction
        type(vspace_bsg_t), pointer, contiguous, dimension(:) :: vpw_bsg
        !! Pointer to weights in vp direction
        real(kind=GP), pointer, contiguous, dimension(:) :: phiw
        !! Quadrature weights in phi direction
        real(kind=GP), pointer, contiguous, dimension(:,:) :: psi
        !! Poloidal flux function
        real(kind=GP), private, allocatable, dimension(:) :: prefac_bps
        !! Normalization factor for b_par^star
        real(kind=GP), private, allocatable, dimension(:) :: prefac_vth
        !! Prefactor containing the species dependent part of the thermal
        !! velocity
        real(kind=GP), private, allocatable, dimension(:) :: prefac_energy
        !! Prefactor for energy normalization
        real(kind=GP), private, allocatable, dimension(:) :: charges
        !! Charges of the different species
        real(kind=GP), private, allocatable, dimension(:) :: prefac_btor
        !! Prefactor for Btor in the canonical momentum calculation
        real(kind=GP), private, allocatable, dimension(:) :: prefac_psi
        !! Prefactor for psi in the canonical momentum calculation
        integer, pointer, contiguous, dimension(:) :: bsg_flags
        !! Pointer to the BSG flags
    contains
        procedure(initialize), deferred :: initialize
    end type

    interface
        module subroutine initialize(this, dcomm_handler, mesh, bsg_op)
            !! Initializes the type
            class(op_diag_mom_0d_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Comm handler for MPI communication
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
            class(bsg_operators_t), target, intent(inout) :: bsg_op
            !! BSG operator
        end subroutine
    end interface

    type, public, extends(op_diag_mom_0d_t) :: op_diag_mom_0d_cpu_t
        !! Operator to calculate the total particle number n, parallel
        !! flow u_par, parallel energy E_par, perpendicular energy E_perp,
        !! electrostatic energy E_es and canonical toroidal momentum p_phi
        !! on the cpu.
    contains
        procedure :: initialize => initialize_cpu
        procedure :: apply => apply_cpu
    end type

    interface
        module subroutine initialize_cpu(this, dcomm_handler, mesh, bsg_op)
            class(op_diag_mom_0d_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            class(bsg_operators_t), target, intent(inout) :: bsg_op
        end subroutine

        module subroutine apply_cpu(this, da_f, da_es_pot, da_moments)
            class(op_diag_mom_0d_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f
            class(data_array_2d_t), intent(in) :: da_es_pot
            class(data_array_2d_t), intent(inout) :: da_moments
        end subroutine
    end interface

    type, public, abstract, extends(op_diag_mom_0d_base_t) :: &
                                                          op_diag_mom_0d_vspec_t
        !! Operator to calculate the total particle number n, parallel
        !! flow u_par, parallel energy E_par, perpendicular energy E_perp,
        !! electrostatic energy E_es and canonical toroidal momentum p_phi
        !! using spectral method
        real(kind=GP), pointer, contiguous, dimension(:, :, :) :: &
                                                               mom_weights_vspec
        !! Pointer to moment spectral weights
        real(kind=GP), private, allocatable, dimension(:) :: prefac_vth
        !! Prefactor containing the species dependent part of the thermal
        !! velocity prefac_vth = sqrt(2 / m)
        real(kind=GP), private, allocatable, dimension(:) :: charges
        !! Charges of the different species
        real(kind=GP), private :: prefac_common
        !! Prefactor for common factors in all moments
        real(kind=GP), private, allocatable, dimension(:) :: temp_scalings
        !! Temperature scaling
        real(kind=GP), pointer, contiguous, dimension(:) :: phiw
        !! Quadrature weights in phi direction
        real(kind=GP), pointer, contiguous, dimension(:,:) :: RZw
        !! Quadrature weights in RZ direction
        real(kind=GP), pointer, contiguous, dimension(:,:) :: jacobian
        !! Values of the jacobian of the mesh
    contains
        procedure(initialize_vspec), deferred :: initialize
        !! Method to initialize the spectral operator
    end type

    interface
        module subroutine initialize_vspec(this, dcomm_handler, mesh)
            !! Initializes the type
            class(op_diag_mom_0d_vspec_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Comm handler for MPI communication
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
        end subroutine
    end interface

    type, public, extends(op_diag_mom_0d_vspec_t) :: op_diag_mom_0d_vspec_cpu_t
        !! Operator to calculate the total particle number n, parallel
        !! flow u_par, parallel energy E_par, perpendicular energy E_perp,
        !! electrostatic energy E_es and canonical toroidal momentum p_phi
        !! using spectral method. CPU implementation
        type(op_set_uniform_cpu_t), private :: op_set_uniform
    contains
        procedure :: initialize => initialize_vspec_cpu
        procedure :: apply => apply_vspec_cpu
    end type

    interface
        module subroutine initialize_vspec_cpu(this, dcomm_handler, mesh)
            class(op_diag_mom_0d_vspec_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine
        module subroutine apply_vspec_cpu(this, da_f, da_es_pot, da_moments)
            class(op_diag_mom_0d_vspec_cpu_t), intent(inout) :: this
            class(data_array_5d_t), intent(in) :: da_f
            class(data_array_2d_t), intent(in) :: da_es_pot
            class(data_array_2d_t), intent(inout) :: da_moments
        end subroutine
    end interface

end module op_diag_mom_0d_m
