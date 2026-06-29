module mesh_5d_m
    !! Module containing the implementation of the mesh_5d_t type
    use genex_fortran_env_m, only: GP
    use, intrinsic :: iso_c_binding, only: C_PTR
    use dcomm_handler_m, only: dcomm_handler_t
    use structured_grids_m, only: mu_grid_t, vp_grid_t, phi_grid_t
    use op_set_uniform_m, only: op_set_uniform_cpu_t
    use params_mesh_m, only: get_n_points_sp
    use params_diagnostics_m, only: get_polar_n_theta, get_polar_n_rho
    use block_structured_grids_m, only: block_structured_grids_t

    ! From PARALLAX
    use mesh_cart_m, only: mesh_cart_t
    use multigrid_m, only: multigrid_t
    use polar_grid_m, only : polar_grid_t
    use helmholtz_solver_m, only: helmholtz_solver_t
    use equilibrium_m, only: equilibrium_t
    use equilibrium_factory_m, only: SLAB, CIRCULAR, SALPHA, NUMERICAL, &
                                     DOMMASCHK
    use slab_equilibrium_m, only: slab_t
    use circular_equilibrium_m, only: circular_t
    use salpha_equilibrium_m, only: salpha_t
    use cerfons_equilibrium_m, only: cerfons_t
    use numerical_equilibrium_m, only: numerical_t
    use dommaschk_equilibrium_m, only: dommaschk_t

    ! NOTE: The following use only statement requires a manual renaming
    !       to the same name to mitigate an abnormal compile-time error
    !       with Intel Fortran 21.X.X. Within that error a submodule cannot
    !       find the name that has been associated here in the module.
    ! TODO: Remove the renaming if the compiler bug has been fixed.
    use csrmat_m, only: csrmat_t => csrmat_t, &
                        csrmat_data_t => csrmat_data_t

    implicit none
    private

    class(equilibrium_t), target, allocatable :: equilibrium_instance
    !! Singular equilibrium instance

    logical, private :: is_equilibrium_initialized = .false.
    !! Bookeeping variable to check whether the equilibrium has been
    !! initialized

    ! NOTE: Filler points are used for non-axisymmetric grids that may have
    !       different shapes and sizes to adjust the arrays to be of same size
    real(kind=GP), public, parameter :: FILLER_VALUE_REAL = &
                                                         -2.0023193043625635_GP
    !! Value of filler points in all real buffers
    integer, public, parameter :: FILLER_VALUE_INT = -2002319304
    !! Value of filler points in all integer buffers

    integer, public, parameter :: size_neighbor = 2
    !! How many points are considered neighbors in the RZ direction, determined
    !! by the discretization scheme.
    integer, private, parameter :: size_ghost_layer = 3
    !! How many ghost points are needed outside the RZ computation domain. This
    !! is determined by the discretization scheme.

    integer, private, parameter :: polar_intorder = 3
    !! Interpolation order for mapping to or from the polar mesh

    public :: BUF_ZONE_BOUNDARY_IN, BUF_ZONE_BOUNDARY_OUT, &
              BUF_ZONE_AXIS, BUF_ZONE_PARCON

    enum, bind(C)
        !! Enumerator defining the different numerical buffer zone types.
        enumerator :: NOT_BUF_ZONE = 0
        enumerator :: BUF_ZONE_BOUNDARY_IN = 1
        enumerator :: BUF_ZONE_BOUNDARY_OUT = 2
        enumerator :: BUF_ZONE_AXIS = 3
        enumerator :: BUF_ZONE_PARCON = 4
    end enum

    enum, bind(C)
        !! Enumerator defining the different parallel connection types.
        enumerator :: NO_PARCON = 0
        enumerator :: PARCON_COMPUTE_TO_TARGET = 1
        enumerator :: PARCON_TARGET_TO_COMPUTE = 2
    end enum

    type, public :: mesh_5d_t
        !! Type containing the grid information for all dimensions. It provides
        !! a unified interface for all grid information needed. Furthermore
        !! it functions as a facade to PARALLAX

        ! NOTE: There are three different coordinates systems in GENEX that
        !       need to be distinguished. The first one is the cylindrical
        !       coordinate system denoted by R, phi, Z provided by PARALLAX.
        !       The second one is the cartesian coordinate system within a
        !       poloidal plane denote by xc, yc, zc. The coordinate system is
        !       right handed and defined by
        !
        !       xc = R * cos(phi)
        !       yc = R * sin(phi)
        !       zc = Z
        !
        !       The third is the locally field aligned coordinate system
        !       denoted by x, y, z. On a poloidal plane, i.e. phi = y = yc = 0,
        !       it holds that
        !
        !       R = x = xc
        !       Z = z = zc
        !
        !       Derivative along the magnetic field line are denoted by ...dy.
        !       Parallel componentes of quantities are denoted by ...y.
        type(dcomm_handler_t), private, pointer :: dcomm_handler
        !! Communication handler
        type(op_set_uniform_cpu_t), private :: op_set_uniform
        !! Operator to set uniform values to arrays
        type(vp_grid_t), private, allocatable, dimension(:) :: vp_grid
        !! Grid for the parallel velocity dimension
        type(phi_grid_t), private, allocatable :: phi_grid
        !! Grid for the toroidal angle dimension
        type(mu_grid_t), private, allocatable  :: mu_grid
        !! Grid for the mu dimension (perp energy)
        type(block_structured_grids_t), private, allocatable :: bsg_instance
        !! Block-structured grid instance

        real(kind=GP), private, allocatable, dimension(:,:,:) :: &
                                                              mom_weights_vspec
        !! Spectral weights to compute moments
        !! (e.g. density, parallel velocity )

        type(multigrid_t), private, allocatable, dimension(:) :: multigrid_3d
        !! Multigrids for Helmholtz problem per phi plane
        type(mesh_cart_t), private, allocatable, dimension(:) :: mesh_3d
        !! Cartesian meshes per phi plane, created by the multigrid
        type(polar_grid_t), private, allocatable, dimension(:) :: mesh_3d_pol
        !! Polar meshes per phi plane

        integer, private, contiguous, pointer, dimension(:,:,:,:) :: &
            neighbor_indices
        !! Storage for the unstructured RZ indices of neighbor points for every
        !! plane. If there is no neighbor in a particular location, the array
        !! will contain the index of the point itself there.
        integer, private, contiguous, pointer, dimension(:,:) :: RZ_indices
        !! Indices for the mesh grid points in the RZ dimension

        integer, private :: equilibrium_type_storage
        !! Storage variable to save the equilibrium type
        real(kind=GP), private :: psi_norm_fac
        !! Normalization factor for poloidal flux
        real(kind=GP), private :: psi_lo
        !! Lower value of psi for normalization
        real(kind=GP), private :: psi_up
        !! Upper value of psi for normalization
        real(kind=GP), private :: absB_max
        !! Maximum value of absolute magnetic field

        integer, private :: lb_phi
        !! Lower bound on the index of phi planes
        integer, private :: ub_phi
        !! Upper bound on the index of phi planes

        integer, private :: size_RZ_max
        !! Maximum number of unstructured grid points from all phi planes, used
        !! to determine shape of all buffers
        real(kind=GP), private, allocatable, dimension(:,:) :: not_ghost
        !! Mask indicating whether a buffer point is a ghost, i.e. outside
        !! computation domain. The value is 0 for PARALLAX ghost or boundary
        !! points, otherwise the value is 1
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: &
            not_filler
        !! Mask indicating whether a buffer point is a filler point, for meshes
        !! which have fewer points than the size of the array. The value is 0
        !! for filler points, otherwise the value is 1.
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: &
            is_compute
        !! Mask indicating whether a buffer point is a computation point, i.e.
        !! not a target, ghost, or filler point.
        real(kind=GP), private, allocatable, dimension(:,:) :: is_core
        !! Mask indicating whether a buffer, computation, filler or ghost point
        !! is in the core.

        real(kind=GP), private, allocatable, dimension(:,:) :: R_buffer
        !! Buffer for the values of the unstructured grid in R direction
        real(kind=GP), private, allocatable, dimension(:,:) :: Z_buffer
        !! Buffer for the values of the unstructured grid in Z direction
        real(kind=GP), private, allocatable, dimension(:,:) :: RZw_buffer
        !! Buffer for the RZ quadrature weights
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: &
            jacobian_buffer
        !! Jacobian of the coordinate transform from lab coordinates into the
        !! coordinate system of the mesh. The coordinate system is cartesian
        !! or cylindrical depending on the type of equilibrium.

        real(kind=GP), private, allocatable, dimension(:,:) :: &
                                                              polar_theta_buffer
        !! Buffer for the values of the polar mesh in theta direction
        real(kind=GP), private, allocatable, dimension(:,:) :: &
                                                              polar_rho_buffer
        !! Buffer for the values of the polar mesh in rho direction

        ! Information about magnetic field (magfield)

        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: &
            absB_buffer
        !! B field on the 3D mesh
        real(kind=GP), private, allocatable, dimension(:,:,:) :: &
                                                               polar_absB_buffer
        !! B field absolute in polar coordinates on the 3D mesh
        real(kind=GP), private, allocatable, dimension(:,:) :: normb_tor_buffer
        !! Torodial component of the normalized magnetic field
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: &
            normb_R_buffer
        !! R component of the normalized magnetic field
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: &
            normb_Z_buffer
        !! Z component of the normalized magnetic field
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: &
            curl_normb_y
        !! y component of the curl of the normalized magnetic field
        real(kind=GP), private, allocatable, dimension(:,:) :: psi_buffer
        !! Poloidal flux function in normalized units
        real(kind=GP), private, allocatable, dimension(:,:) :: rho_buffer
        !! Flux surface label
        real(kind=GP), private, allocatable, dimension(:,:) :: theta_buffer
        !! Poloidal angle
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: &
            dgyxdy_over_g
        !! Derivative of b_R along the magnetic field on the 3D mesh
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: &
            dgyzdy_over_g
        !! Derivative of b_Z along the magnetic field on the 3D mesh
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: &
            dgyxdz_over_g
        !! Derivative of b_R in Z direction on the 3D mesh
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: &
            dgyzdx_over_g
        !! Derivative of b_Z in R direction on the 3D mesh
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: &
            inv_g
        !! Inverse of sqrt(abs(det(g))) where g denotes the metric tensor of
        !! the x, y, z coordinate system
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: &
            dabsBdx
        !! Derivative of abs B in R direction on the 3D mesh
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: &
            dabsBdz
        !! Derivative of abs B in Z direction on the 3D mesh
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: &
            dabsBdy
        !! Derivative of abs B in parallel direction on the 3D mesh
        real(kind=GP), private, allocatable, dimension(:,:) :: loss_cone
        !! Loss cone for Trapped Particle Contributions (TPC)

        ! Information about parallel connection (parcon)

        type(csrmat_t), private, dimension(:), allocatable :: map_positive1
        !! Map matrices to the poloidal planes k + 1
        type(csrmat_t), private, dimension(:), allocatable :: map_positive2
        !! Map matrices to the poloidal planes k + 2
        type(csrmat_t), private, dimension(:), allocatable :: map_negative1
        !! Map matrices to the poloidal planes k - 1
        type(csrmat_t), private, dimension(:), allocatable :: map_negative2
        !! Map matrices to the poloidal planes k - 2
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: &
            fll_positive1
        !! Field line lengths traced from each mesh point to the k + 1 plane
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: &
            fll_positive2
        !! Field line lengths traced from each mesh point to the k + 2 plane
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: &
            fll_negative1
        !! Field line lengths traced from each mesh point to the k - 1 plane
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: &
            fll_negative2
        !! Field line lengths traced from each mesh point to the k - 2 plane
        integer, private, contiguous, pointer, dimension(:,:) :: &
            not_in_target
        !! Array specifying whether a given point is not in the target
        !! (divertor, limiter, or wall)
        integer, private, allocatable, dimension(:,:) :: parcon_positive1
        !! Array specifying whether a given point is connected by a parallel
        !! stencil to the target on the k + 1 plane
        integer, private, allocatable, dimension(:,:) :: parcon_positive2
        !! Array specifying whether a given point is connected by a parallel
        !! stencil to the target on the k + 2 plane
        integer, private, allocatable, dimension(:,:) :: parcon_negative1
        !! Array specifying whether a given point is connected by a parallel
        !! stencil to the target on the k - 1 plane
        integer, private, allocatable, dimension(:,:) :: parcon_negative2
        !! Array specifying whether a given point is connected by a parallel
        !! stencil to the target on the k - 2 plane
        real(kind=GP), private, allocatable, dimension(:,:) :: maps_on_mesh
        !! Array specifying whether a given point maps onto all of the
        !! neighboring meshes (4 point stencil)
        integer, private, contiguous, pointer, dimension(:,:) :: buf_zone
        !! Array specifying which buffer zone a given point is in, identified
        !! by the buffer zone type enumerator

#ifdef ENABLE_GPU
        type(C_PTR), private :: mesh_5d_cxx_pptr
        !! C pointer to the mesh_5d_t C++ class instance pointer
#endif

    contains
        procedure, public :: initialize

        procedure, private :: initialize_ghost_and_filler_masks
        procedure, private :: initialize_buf_zone
        procedure, private :: initialize_RZ
        procedure, private :: initialize_RZ_weights
        procedure, private :: initialize_polar
        procedure, private :: initialize_jacobian
        procedure, private :: initialize_mom_weights_vspec
        procedure, private :: initialize_magfield
        procedure, private :: initialize_parcon
        procedure, private :: initialize_compute_mask
        procedure, private :: initialize_core_mask
        procedure, private :: initialize_neighbor_indices
        procedure, private :: construct_vp_bsg

#ifdef ENABLE_GPU
        procedure, private :: initialize_gpu
        procedure, private :: finalize_gpu
#endif

        final :: finalize

        ! Getters for mesh information
        procedure, public  :: get_mesh_3d
        procedure, public  :: get_mesh_3d_pol
        procedure, public  :: get_R_pointer
        procedure, public  :: get_Z_pointer
        procedure, public  :: get_RZw_pointer
        procedure, public  :: get_RZ_indices_pointer
        procedure, private :: get_vp_pointer_sg
        procedure, private :: get_vp_pointer_bsg
        generic, public    :: get_vp_pointer => get_vp_pointer_sg, &
                                                get_vp_pointer_bsg
        procedure, public  :: get_mu_pointer
        procedure, public  :: get_sqrt_mu_pointer
        procedure, private :: get_vpw_pointer_sg
        procedure, private :: get_vpw_pointer_bsg
        generic, public    :: get_vpw_pointer => get_vpw_pointer_sg, &
                                                 get_vpw_pointer_bsg
        procedure, public :: get_muw_pointer
        procedure, public :: get_phiw_pointer
        procedure, public :: get_mom_weights_vspec_pointer
        procedure, public :: get_phi_pointer
        procedure, public :: get_jacobian_pointer
        procedure, public :: get_not_ghost_pointer
        procedure, public :: get_not_filler_pointer
        procedure, public :: get_is_compute_pointer
        procedure, public :: get_is_core_pointer
        procedure, public :: get_neighbor_indices_pointer
        procedure, public :: get_buf_zone_pointer
        procedure, public :: get_bsg_flags_pointer
        procedure, public :: get_polar_theta_pointer
        procedure, public :: get_polar_rho_pointer

        ! Getters for magfield information
        procedure, public :: get_absB_pointer
        procedure, public :: get_polar_absB_pointer
        procedure, public :: get_normb_tor_pointer
        procedure, public :: get_normb_R_pointer
        procedure, public :: get_normb_Z_pointer
        procedure, public :: get_curl_normb_y_pointer
        procedure, public :: get_psi_pointer
        procedure, public :: get_rho_pointer
        procedure, public :: get_theta_pointer
        procedure, public :: get_dgyxdy_over_g_pointer
        procedure, public :: get_dgyzdy_over_g_pointer
        procedure, public :: get_dgyxdz_over_g_pointer
        procedure, public :: get_dgyzdx_over_g_pointer
        procedure, public :: get_inv_g_pointer
        procedure, public :: get_dabsBdx_pointer
        procedure, public :: get_dabsBdz_pointer
        procedure, public :: get_dabsBdy_pointer
        procedure, public :: get_loss_cone_pointer

        ! Getters for parcon information
        procedure, public :: get_map_positive1_pointer
        procedure, public :: get_map_positive2_pointer
        procedure, public :: get_map_negative1_pointer
        procedure, public :: get_map_negative2_pointer
        procedure, public :: get_fll_positive1_pointer
        procedure, public :: get_fll_positive2_pointer
        procedure, public :: get_fll_negative1_pointer
        procedure, public :: get_fll_negative2_pointer
        procedure, public :: get_parcon_positive1_pointer
        procedure, public :: get_parcon_positive2_pointer
        procedure, public :: get_parcon_negative1_pointer
        procedure, public :: get_parcon_negative2_pointer
        procedure, public :: get_not_in_target_pointer
        procedure, public :: get_maps_on_mesh_pointer

#ifdef ENABLE_GPU
        ! Getter for C pointer to the mesh_5d_t C++ class instance pointer
        procedure, public :: get_cxx_pointer => get_mesh_5d_cxx_pointer
#endif

        ! Wrappers for PARALLAX routines
        procedure, public :: create_helmholtz_solver
        procedure, public :: trace => trace_wrapper

        ! Properties
        procedure, public  :: delta_RZ
        procedure, private :: delta_vp_sg
        procedure, private :: delta_vp_bsg
        generic, public    :: delta_vp => delta_vp_sg, delta_vp_bsg
        procedure, public  :: delta_mu
        procedure, public  :: delta_sqrt_mu
        procedure, public  :: delta_phi
        procedure, public  :: size_RZ
        procedure, public  :: size_RZ_plane
        procedure, public  :: size_RZ_inner
        procedure, public  :: size_polar_theta
        procedure, public  :: size_polar_rho
        procedure, public  :: size_vp
        procedure, public  :: size_mu
        procedure, public  :: size_phi
        procedure, public  :: size_sp
        procedure, public  :: district
        procedure, public  :: RZphi
        procedure, public  :: canonical_rho
        procedure, public  :: canonical_psi
        procedure, public  :: rho_min
        procedure, public  :: rho_max
        procedure, public  :: equi_type
        procedure, public  :: is_axisymmetric
        procedure, public  :: grid_type_vp
        procedure, public  :: quad_type_vp
        procedure, public  :: is_vspectral_vp
        procedure, public  :: grid_type_mu
        procedure, public  :: quad_type_mu
        procedure, public  :: is_vspectral_mu
        procedure, public  :: grid_type_phi
        procedure, public  :: quad_type_phi
        procedure, private :: read_mesh
        procedure, private :: write_mesh

    end type

    interface
        module subroutine initialize(this, dcomm_handler, &
                                     read_mesh_from_file, mesh_filename)
            !! Initializes the mesh_5d_t type
            class(mesh_5d_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Communication handler
            logical, optional, intent(in) :: read_mesh_from_file
            !! Switch whether to read mesh from file or initialize from scratch.
            !! Default false. If true, mesh_filename must be given.
            character(len=*), optional, intent(in) :: mesh_filename
            !! Mesh file name. If read_mesh_from_file is true, mesh data will be
            !! read from this file. If read_mesh_from_file is false, mesh data
            !! will be written to this file. If not provided, the mesh will not
            !! be written to file.
        end subroutine

        module subroutine initialize_ghost_and_filler_masks(this)
            !! Initializes the not_ghost and not_filler masks for the mesh
            class(mesh_5d_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_buf_zone(this)
            !! Initializes the buffer zone array for the mesh
            class(mesh_5d_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_RZ(this)
            !! Initializes the R and Z coordinate buffers from mesh_3d
            class(mesh_5d_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_RZ_weights(this)
            !! Initializes the RZ quadrature weights
            class(mesh_5d_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_polar(this)
            !! Initializes the theta and rho coordinate buffers from mesh_3d_pol
            class(mesh_5d_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_jacobian(this)
            !! Initializes the jacobian
            class(mesh_5d_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_mom_weights_vspec(this)
            !! Initializes the moment spectral weights
            class(mesh_5d_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_magfield(this)
            !! Initializes the magnetic field information
            class(mesh_5d_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_parcon(this)
            !! Initializes the parallel connection information
            class(mesh_5d_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_compute_mask(this)
            !! Initializes the compute point mask as a combination of the
            !! target, ghost, and filler masks
            class(mesh_5d_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_core_mask(this)
            !! Initializes the core point mask
            class(mesh_5d_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_neighbor_indices(this)
            !! Initializes the neighbor indices buffer from the 2D meshes on
            !! every plane
            class(mesh_5d_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine construct_vp_bsg(this)
            !! Constructs vp_grid for BSG
            class(mesh_5d_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

#ifdef ENABLE_GPU
        module subroutine initialize_gpu(this)
            !! Initialize the mesh_5d_t C++ class, including allocation on
            !! the device memory
            class(mesh_5d_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine finalize_gpu(this)
            !! Finalize the mesh_5d_t C++ class, including deallocation from
            !! the device memory
            class(mesh_5d_t), intent(inout) :: this
            !! Instance of the type
        end subroutine
#endif

        module subroutine finalize(this)
            !! Finalizes the mesh and all underlying PARALLAX components
            type(mesh_5d_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine create_helmholtz_solver(this, lb, ub, &
                                                  lb_stripped, ub_stripped, &
                                                  co, lambda, xi, solvers)
            !! Creates an array of Helmholtz solvers given the coefficients
            !! of the Helmholtz problem and the parameters specified.
            class(mesh_5d_t), intent(inout) :: this
            !! Instance of the type
            integer, dimension(2), intent(in) :: lb, ub
            !! Lower and upper real-space bounds, including ghost points
            integer, dimension(2), intent(in) :: lb_stripped, ub_stripped
            !! Lower and upper real-space bounds, excluding ghost points
            real(kind=GP), dimension(lb(1):ub(1), &
                                     lb_stripped(2):ub_stripped(2) &
                                     ), intent(in) :: co
            !! Co coefficients of the Helmholtz problem
            real(kind=GP), dimension(lb_stripped(1):ub_stripped(1), &
                                     lb_stripped(2):ub_stripped(2) &
                                     ), intent(in) :: lambda
            !! Lambda coefficients of the Helmholtz problem
            real(kind=GP), dimension(lb_stripped(1):ub_stripped(1), &
                                     lb_stripped(2):ub_stripped(2) &
                                     ), intent(in) :: xi
            !! Xi coefficients of the Helmholtz problem
            class(helmholtz_solver_t), &
                dimension(lb_stripped(2):ub_stripped(2)), intent(out) :: solvers
            !! Created Helmholtz solver array
        end subroutine

        module subroutine trace_wrapper(this, x, y, phi, delta_phi, xp, yp, &
                                        fll, vol)
            !! Traces the point x, y, phi along the field line around the angle
            !! delta_phi. The end points of the fieldline are saved in xp and
            !! yp.
            class(mesh_5d_t), target, intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: x, y, phi
            real(kind=GP), intent(out) :: xp, yp
            real(kind=GP), intent(in) :: delta_phi
            real(kind=GP), intent(out), optional :: fll, vol
        end subroutine

        real(kind=GP) module function canonical_rho(this, i, k, l, m, n)
            !! Returns flux surface label rho given the index of the
            !! unstructured grid index and the velocity space index
            class(mesh_5d_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(in) :: i
            !! Index of the unstructured grid
            integer, intent(in) :: k
            !! Phi index
            integer, intent(in) :: l
            !! Velocity space index
            integer, intent(in) :: m
            !! Magnetic moment index
            integer, intent(in) :: n
            !! Species index
        end function

        real(kind=GP) module function canonical_psi(this, i, k, l, m, n)
            !! Returns canonical poloidal flux given the index of the
            !! unstructured grid index and the velocity space index
            class(mesh_5d_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(in) :: i
            !! Index of the unstructured grid
            integer, intent(in) :: k
            !! Phi index
            integer, intent(in) :: l
            !! Velocity space index
            integer, intent(in) :: m
            !! Magnetic moment index
            integer, intent(in) :: n
            !! Species index
        end function

        module subroutine read_mesh(this, mesh_filename, spacing_RZ, &
                                    multigrid_levels)
            !! Reads mesh data from file
            class(mesh_5d_t), target, intent(inout) :: this
            !! Instance of the type
            character(len=*), intent(in) :: mesh_filename
            !! Name of the mesh file
            real(kind=GP), intent(in) :: spacing_RZ
            !! RZ grid spacing. Checked for consistency with values read from
            !! file
            integer, intent(in) :: multigrid_levels
            !! Number of multigrid levels. Checked with consistency with values
            !! read from file
        end subroutine

        module subroutine write_mesh(this, mesh_filename)
            !! Writes the mesh to file
            class(mesh_5d_t), target, intent(inout) :: this
            !! Instance of the type
            character(len=*), intent(in) :: mesh_filename
            !! Name of the mesh file
        end subroutine

    end interface

contains

    ! Short subroutines and functions for mesh_5d_t

    ! Getters for mesh information

    function get_mesh_3d(this) result(ptr)
        !! Returns a pointer to the Cartesian mesh
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        type(mesh_cart_t), contiguous, pointer, dimension(:) :: ptr

        ptr => this%mesh_3d
    end function

    function get_mesh_3d_pol(this) result(ptr)
        !! Returns a pointer to the polar mesh
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        type(polar_grid_t), contiguous, pointer, dimension(:) :: ptr

        ptr => this%mesh_3d_pol
    end function

    function get_R_pointer(this) result(ptr)
        !! Returns a pointer to the values of the R grid
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%R_buffer
    end function

    function get_Z_pointer(this) result(ptr)
        !! Returns a pointer to the values of the Z grid
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%Z_buffer
    end function

    function get_RZw_pointer(this) result(ptr)
        !! Returns a pointer to the values of the RZ quadrature weights
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%RZw_buffer
    end function

    function get_RZ_indices_pointer(this) result(ptr)
        !! Returns a pointer to mesh indices in the RZ dimension
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        integer, contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%RZ_indices
    end function

    function get_polar_theta_pointer(this) result(ptr)
        !! Returns a pointer to the values of the theta polar mesh
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%polar_theta_buffer
    end function

    function get_polar_rho_pointer(this) result(ptr)
        !! Returns a pointer to the values of the rho polar mesh
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%polar_rho_buffer
    end function

    function get_vp_pointer_sg(this) result(ptr)
        !! Returns a pointer to the values of the vp grid (SG)
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:) :: ptr

        ptr => this%vp_grid(1)%get_pointer()
    end function

    function get_vp_pointer_bsg(this, block_number) result(ptr)
        !! Returns a pointer to the values of the vp grid (BSG)
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        integer, intent(in) :: block_number
        !! Block number
        real(kind=GP), contiguous, pointer, dimension(:) :: ptr

        ptr => this%vp_grid(block_number)%get_pointer()
    end function

    function get_mu_pointer(this) result(ptr)
        !! Returns a pointer to the values of the mu grid
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:) :: ptr

        ptr => this%mu_grid%get_pointer()
    end function

    function get_sqrt_mu_pointer(this) result(ptr)
        !! Returns a pointer to the values of the sqrt_mu grid
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:) :: ptr

        ptr => this%mu_grid%get_sqrt_pointer()
    end function

    function get_vpw_pointer_sg(this) result(ptr)
        !! Returns a pointer to the integration weights of the vp grid
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:) :: ptr

        ptr => this%vp_grid(1)%get_weights_pointer()
    end function

    function get_vpw_pointer_bsg(this, block_number) result(ptr)
        !! Returns a pointer to the integration weights of the vp grid (BSG)
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        integer, intent(in) :: block_number
        !! Block number
        real(kind=GP), contiguous, pointer, dimension(:) :: ptr

        ptr => this%vp_grid(block_number)%get_weights_pointer()
    end function

    function get_muw_pointer(this) result(ptr)
        !! Returns a pointer to the integration weights of the mu_grid
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:) :: ptr

        ptr => this%mu_grid%get_weights_pointer()
    end function

    function get_mom_weights_vspec_pointer(this) result(ptr)
        !! Returns a pointer to the moment spectral weights
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:,:) :: ptr

        ptr => this%mom_weights_vspec
    end function

    function get_phi_pointer(this) result(ptr)
        !! Returns a pointer to the values of the phi grid
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:) :: ptr

        ptr => this%phi_grid%get_pointer()
    end function

    function get_phiw_pointer(this) result(ptr)
        !! Returns a pointer to the integration weights of the phi grid
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:) :: ptr

        ptr => this%phi_grid%get_weights_pointer()
    end function

    function get_jacobian_pointer(this) result(ptr)
        !! Returns a pointer to the values of the Jacobian
        !! of the cylindrical R, phi, Z coordinate system
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%jacobian_buffer
    end function

    function get_not_ghost_pointer(this) result(ptr)
        !! Returns a pointer to the not_ghost mask
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%not_ghost
    end function

    function get_not_filler_pointer(this) result(ptr)
        !! Returns a pointer to the not_filler mask
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%not_filler
    end function

    function get_is_compute_pointer(this) result(ptr)
        !! Returns a pointer to the is_compute mask
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%is_compute
    end function

    function get_is_core_pointer(this) result(ptr)
        !! Returns a pointer to the is_core mask
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%is_core
    end function

    function get_neighbor_indices_pointer(this) result(ptr)
        !! Returns a pointer to the array containing the neighbor indices of a
        !! given point of the unstructured grid
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        integer, contiguous, pointer, dimension(:,:,:,:) :: ptr

        ptr => this%neighbor_indices
    end function

    function get_buf_zone_pointer(this) result(ptr)
        !! Returns a pointer to the array containing the buffer zone
        !! specification
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        integer, contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%buf_zone
    end function

    function get_bsg_flags_pointer(this) result(ptr)
        !! Returns the pointer to the bsg_flags array
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        integer, contiguous, pointer, dimension(:) :: ptr

        ptr => this%bsg_instance%get_bsg_flags_pointer()
    end function

    ! Getters for magfield information

    function get_absB_pointer(this) result(ptr)
        !! Returns a pointer to the values of the B field
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%absB_buffer
    end function

    function get_polar_absB_pointer(this) result(ptr)
        !! Returns a pointer to the values of the B field absolute
        !! on the polar mesh
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:,:) :: ptr

        ptr => this%polar_absB_buffer
    end function

    function get_normb_tor_pointer(this) result(ptr)
        !! Returns a pointer of rank 2 pointing to the values of the toroidal
        !! component of the B field
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%normb_tor_buffer
    end function

    function get_normb_R_pointer(this) result(ptr)
        !! Returns a pointer to the values of the R
        !! component of the normalized magnetic field vector
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%normb_R_buffer
    end function

    function get_normb_Z_pointer(this) result(ptr)
        !! Returns a pointer to the values of the Z
        !! component of the normalized magnetic field vector
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%normb_Z_buffer
    end function

    function get_curl_normb_y_pointer(this) result(ptr)
        !! Returns a pointer to the values of the curl B field
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%curl_normb_y
    end function

    function get_psi_pointer(this) result(ptr)
        !! Returns a pointer to the values of the poloidal flux function
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%psi_buffer
    end function

    function get_rho_pointer(this) result(ptr)
        !! Returns a pointer to the flux surface label
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%rho_buffer
    end function

    function get_theta_pointer(this) result(ptr)
        !! Returns a pointer to the poloidal angle
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%theta_buffer
    end function

    function get_dgyxdy_over_g_pointer(this) result(ptr)
        !! Returns a pointer to the values of the derivative
        !! of the R component of the magnetic field along the magnetic field
        !! field
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%dgyxdy_over_g
    end function

    function get_dgyzdy_over_g_pointer(this) result(ptr)
        !! Returns a pointer to the values of the derivative
        !! of the Z component of the magnetic field along the magnetic field
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%dgyzdy_over_g
    end function

    function get_dgyxdz_over_g_pointer(this) result(ptr)
        !! Returns a pointer to the values of the derivative
        !! of the R component of the magnetic field in Z direction
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%dgyxdz_over_g
    end function

    function get_dgyzdx_over_g_pointer(this) result(ptr)
        !! Returns a pointer to the values of the derivative
        !! of the Z component of the magnetic field in R direction
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%dgyzdx_over_g
    end function

    function get_inv_g_pointer(this) result(ptr)
        !! Returns a pointer to the values of the inverse
        !! of sqrt(abs(det(g))), where g denotes the metric tensor of the
        !! x, y, z coordinate system
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%inv_g
    end function

    function get_dabsBdx_pointer(this) result(ptr)
        !! Returns a pointer to the values of the derivative
        !! of the absolute value of the magnetic field in R direction
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%dabsBdx
    end function

    function get_dabsBdz_pointer(this) result(ptr)
        !! Returns a pointer to the values of the derivative
        !! of the absolute value of the magnetic field in Z direction
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%dabsBdz
    end function

    function get_dabsBdy_pointer(this) result(ptr)
        !! Returns a pointer to the values of the derivative
        !! of the absolute value of the magnetic field along the magnetic field
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%dabsBdy
    end function

    function get_loss_cone_pointer(this) result(ptr)
        !! Returns a pointer to the loss cone for TPC
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:,:) :: ptr

        ptr => this%loss_cone
    end function

    ! Getters for parallel connection information

    function get_map_positive1_pointer(this) result(ptr)
        !! Returns a pointer to the parallel map matrix on the poloidal plane
        !! k + 1
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        type(csrmat_t), dimension(:), pointer :: ptr

        ptr => this%map_positive1
    end function

    function get_map_positive2_pointer(this) result(ptr)
        !! Returns a pointer to the parallel map matrix on the poloidal plane
        !! k + 2
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        type(csrmat_t), dimension(:), pointer :: ptr

        ptr => this%map_positive2
    end function

    function get_map_negative1_pointer(this) result(ptr)
        !! Returns a pointer to the parallel map matrix on the poloidal plane
        !! k - 1
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        type(csrmat_t), dimension(:), pointer :: ptr

        ptr => this%map_negative1
    end function

    function get_map_negative2_pointer(this) result(ptr)
        !! Returns a pointer to the parallel map matrix on the poloidal plane
        !! k - 2
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        type(csrmat_t), dimension(:), pointer :: ptr

        ptr => this%map_negative2
    end function

    function get_fll_positive1_pointer(this) result(ptr)
        !! Returns a pointer to the values of the in target array
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:, :) :: ptr

        ptr => this%fll_positive1
    end function

    function get_fll_positive2_pointer(this) result(ptr)
        !! Returns a pointer to the values of the in target array
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:, :) :: ptr

        ptr => this%fll_positive2
    end function

    function get_fll_negative1_pointer(this) result(ptr)
        !! Returns a pointer to the values of the in target array
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:, :) :: ptr

        ptr => this%fll_negative1
    end function

    function get_fll_negative2_pointer(this) result(ptr)
        !! Returns a pointer to the values of the in target array
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:, :) :: ptr

        ptr => this%fll_negative2
    end function

    function get_parcon_positive1_pointer(this) result(ptr)
        !! Returns a pointer to the values of the parallel
        !! connection on the poloidal plane k + 1
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        integer, contiguous, pointer, dimension(:, :) :: ptr

        ptr => this%parcon_positive1
    end function

    function get_parcon_positive2_pointer(this) result(ptr)
        !! Returns a pointer to the values of the parallel
        !! connection on the poloidal plane k + 2
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        integer, contiguous, pointer, dimension(:, :) :: ptr

        ptr => this%parcon_positive2
    end function

    function get_parcon_negative1_pointer(this) result(ptr)
        !! Returns a pointer to the values of the parallel
        !! connection on the poloidal plane k - 2
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        integer, contiguous, pointer, dimension(:, :) :: ptr

        ptr => this%parcon_negative1
    end function

    function get_parcon_negative2_pointer(this) result(ptr)
        !! Returns a pointer to the values of the parallel
        !! connection on the poloidal plane k - 2
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        integer, contiguous, pointer, dimension(:, :) :: ptr

        ptr => this%parcon_negative2
    end function

    function get_not_in_target_pointer(this) result(ptr)
        !! Returns a pointer to the values of the not in target array
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        integer, contiguous, pointer, dimension(:, :) :: ptr

        ptr => this%not_in_target
    end function

    function get_maps_on_mesh_pointer(this) result(ptr)
        !! Returns a pointer to the values of the maps_on_mesh array
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        real(kind=GP), contiguous, pointer, dimension(:, :) :: ptr

        ptr => this%maps_on_mesh
    end function

#ifdef ENABLE_GPU
    function get_mesh_5d_cxx_pointer(this) result(pptr)
        !! Returns the C pointer of the mesh_5d_t C++ class instance pointer
        class(mesh_5d_t), target, intent(inout) :: this
        !! Instance of the type
        type(C_PTR), pointer :: pptr

        pptr => this%mesh_5d_cxx_pptr
    end function
#endif

    ! Properties

    pure real(kind=GP) function delta_RZ(this)
        !! Returns the grid spacing of the RZ grid (same for every plane)
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        delta_RZ = this%mesh_3d(1)%get_spacing_f()
    end function delta_RZ

    pure real(kind=GP) function delta_vp_sg(this)
        !! Returns the grid spacing of the parallel velocity grid
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        delta_vp_sg = this%vp_grid(1)%get_spacing(1)
    end function delta_vp_sg

    pure real(kind=GP) function delta_vp_bsg(this, block_number)
        !! Returns the grid spacing of the parallel velocity grid (BSG)
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        integer, intent(in) :: block_number
        delta_vp_bsg = this%vp_grid(block_number)%get_spacing(1)
    end function delta_vp_bsg

    pure real(kind=GP) function delta_mu(this, m)
        !! Returns the grid spacing of the magnetic moment grid at a given
        !! grid index (mu grid can be non-uniform, thus the spacing may
        !! vary)
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        integer, intent(in) :: m
        !! Grid index to fetch spacing from
        delta_mu = this%mu_grid%get_spacing(m)
    end function delta_mu

    pure real(kind=GP) function delta_sqrt_mu(this)
        !! Returns the grid spacing of the magnetic moment grid at the first
        !! point of the grid. If the mu grid is quadratic, this spacing
        !! will be the same for all points.
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        delta_sqrt_mu = this%mu_grid%get_sqrt_spacing(1)
    end function delta_sqrt_mu

    pure real(kind=GP) function delta_phi(this)
        !! Returns the grid spacing of the toroidal angle grid
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        delta_phi = this%phi_grid%get_spacing(1)
    end function delta_phi

    pure integer function size_RZ(this)
        !! Returns the maximum number of points in RZ direction including ghost
        !! and filler points
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        size_RZ = this%size_RZ_max
    end function

    pure integer function size_polar_theta(this, k)
        !! Returns the number of points in theta direction
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        integer, intent(in) :: k
        !! Phi index
        size_polar_theta = this%mesh_3d_pol(k)%get_ntheta()
    end function

    pure integer function size_polar_rho(this,k)
        !! Returns the maximum number of points in rho direction
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        integer, intent(in) :: k
        !! Phi index
        size_polar_rho = this%mesh_3d_pol(k)%get_nrho()
    end function

    pure integer function size_RZ_plane(this, k)
        !! Returns the number of points in RZ direction, including ghosts but
        !! excluding filler points, for the specified plane
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        integer, intent(in) :: k
        !! Phi index
        size_RZ_plane = this%mesh_3d(k)%get_n_points()
    end function

    pure integer function size_RZ_inner(this, k)
        !! Returns the number of points in RZ direction, excluding ghost and
        !! filler points, for the specified plane
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        integer, intent(in) :: k
        !! Phi index
        size_RZ_inner = this%mesh_3d(k)%get_n_points_inner()
    end function

    pure integer function size_vp(this)
        !! Returns the number of points in vp direction
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        size_vp = this%vp_grid(1)%get_size()
    end function

    pure integer function size_mu(this)
        !! Returns the number of points in mu direction
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        size_mu = this%mu_grid%get_size()
    end function

    pure integer function size_phi(this)
        !! Returns the number of points in phi direction
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        size_phi = this%phi_grid%get_size()
    end function

    pure integer function size_sp(this)
        !! Returns the number of points in sp direction
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        size_sp = get_n_points_sp()
    end function

    integer function district(this, i, k)
        !! Returns the district descriptor of the given index of the
        !! unstructured grid
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        integer, intent(in) :: i
        !! Index of the unstructured grid
        integer, intent(in) :: k
        !! Phi index

        if(this%not_filler(i, k) == 1.0_GP) then
            district = this%mesh_3d(k)%get_district(i)
        else
            district = FILLER_VALUE_INT
        end if
    end function

    subroutine RZphi(this, i, k, x, z, phi)
        !! Returns the x, z, and phi coordinates of the given unstructured grid
        !! index i and phi index k. Uses the safe R and Z buffers, with no risk
        !! of going out of bounds when passing a filler index.
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        integer, intent(in) :: i, k
        real(kind=GP), intent(out) :: x, z, phi

        integer :: k_mod

        ! Modulo the phi index to correct for MPI ghosts
        k_mod = modulo(k - 1, this%size_phi()) + 1

        x   = this%R_buffer(i, k_mod)
        z   = this%Z_buffer(i, k_mod)
        phi = this%mesh_3d(k_mod)%get_phi()
    end subroutine

    real(kind=GP) function rho_min(this)
        !! Returns the minimal flux surface label contained in the mesh
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        !$omp critical
        rho_min = equilibrium_instance%rhomin
        !$omp end critical
    end function

    real(kind=GP) function rho_max(this)
        !! Returns the maximal flux surface label contained in the mesh
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        !$omp critical
        rho_max = equilibrium_instance%rhomax
        !$omp end critical
    end function

    integer function equi_type(this)
        !! Returns the type of the equilibrium
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        equi_type = this%equilibrium_type_storage
    end function

    logical function is_axisymmetric(this)
        !! Returns a logical that indicates if the mesh (and equilibrium)
        !! is axisymmetric or not
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type

        is_axisymmetric = equilibrium_instance%is_axisymmetric()
    end function

    pure character(len=24) function grid_type_vp(this)
        !! Returns the type of the vp grid
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        grid_type_vp = this%vp_grid(1)%get_grid_type()
    end function

    pure character(len=24) function quad_type_vp(this)
        !! Returns the type of the vp grid
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        quad_type_vp = this%vp_grid(1)%get_quad_type()
    end function

    logical function is_vspectral_vp(this)
        !! Returns if spectral is used in vp grid
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        is_vspectral_vp = this%vp_grid(1)%get_is_vspectral()
    end function

    pure character(len=24) function grid_type_mu(this)
        !! Returns the type of the mu grid
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        grid_type_mu = this%mu_grid%get_grid_type()
    end function

    pure character(len=24) function quad_type_mu(this)
        !! Returns the type of the mu grid
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        quad_type_mu = this%mu_grid%get_quad_type()
    end function

    pure logical function is_vspectral_mu(this)
        !! Returns if spectral is used in mu grid
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        is_vspectral_mu = this%mu_grid%get_is_vspectral()
    end function

    pure character(len=24) function grid_type_phi(this)
        !! Returns the type of the phi grid
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        grid_type_phi = this%phi_grid%get_grid_type()
    end function

    pure character(len=24) function quad_type_phi(this)
        !! Returns the type of the phi grid
        class(mesh_5d_t), intent(in) :: this
        !! Instance of the type
        quad_type_phi = this%phi_grid%get_quad_type()
    end function
end module
