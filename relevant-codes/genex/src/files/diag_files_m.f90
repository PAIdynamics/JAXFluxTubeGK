module diag_files_m
    use netcdf
    use genex_fortran_env_m, only: GP, NF90_GP
    use dcomm_handler_m, only: dcomm_handler_t
    use genex_error_handling_m, only: handle_error, handle_error_netcdf, &
                                      error_info_t
    use genex_status_codes_m, only: GENEX_ERR_DIAG_FILES

    implicit none

    type, public, abstract :: diag_file_base_t
        !! Base diagnostic file
        type(dcomm_handler_t), private, pointer :: dcomm_handler
        !! MPI communication handler
        character(len=256), private :: filename
        !! File name of the diag file
        integer, private :: file_id
        !! Netcdf file id
    contains
        procedure, private :: check_array
    end type

    type, public, extends(diag_file_base_t) :: mom_0d_file_t
        !! Type to represent the diagnostic file for the total moments
        integer, private :: lb_mom_0d(2)
        integer, private :: dim_time_id
        integer, private :: time_id
        integer, private, dimension(:), allocatable :: u_par_ids
        integer, private, dimension(:), allocatable :: species_group
        integer, private, dimension(:), allocatable :: n_ids
        integer, private, dimension(:), allocatable :: E_par_ids
        integer, private, dimension(:), allocatable :: E_perp_ids
        integer, private, dimension(:), allocatable :: E_es_ids
        integer, private, dimension(:), allocatable :: p_phi_ids
        real(kind = GP), dimension(:, :), allocatable :: send_buffer, &
                                                         recv_buffer
        logical, private :: is_writing
        logical, private :: is_communicating
        integer, private :: write_counter
    contains
        procedure, public :: initialize => initialize_mom_0d
        procedure, public :: write => write_mom_0d
        final :: finalize_mom_0d
    end type

    interface
        module subroutine initialize_mom_0d(this, dcomm_handler, filename)
            !! Initializes the mom_0d_file_t type and creates and initializes
            !! the corresponding NetCDF file on disk.
            class(mom_0d_file_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Communication handler for MPI
            character(len=*), intent(in) :: filename
            !! Filename to write to
        end subroutine

        module subroutine write_mom_0d(this, time, mom_0d)
            !! Writes the time and the given total moments to file. The moments
            !! need to be stored in an array with shape (n_moments, n_points_sp)
            !! and the moments are assumed to be in the order n,
            !! u_par, E_par, E_perp
            class(mom_0d_file_t), intent(inout) :: this
            real(kind = GP), intent(in) :: time
            real(kind = GP), intent(in) :: mom_0d(this%lb_mom_0d(1):, &
                                                  this%lb_mom_0d(2):)
        end subroutine

        module subroutine finalize_mom_0d(this)
            !! Finalizes the mom_0d_file_t type
            type(mom_0d_file_t), intent(inout) :: this
        end subroutine
    end interface

    type, public, extends(diag_file_base_t) :: mom_2d_file_t
        !! Type to represent the diagnostic file for the 2D moments
        integer, private :: n_points_RZ
        integer, private :: n_points_phi
        integer, private :: n_points_sp
        integer, private :: dim_time_id
        integer, private :: dim_RZ_id
        integer, private :: dim_phi_id
        integer, private :: time_id
        integer, private, dimension(:), allocatable :: species_group
        integer, private, dimension(:), allocatable :: n_ids
        integer, private, dimension(:), allocatable :: u_par_ids
        integer, private, dimension(:), allocatable :: E_par_ids
        integer, private, dimension(:), allocatable :: E_perp_ids
        integer, private, dimension(:), allocatable :: Q_par_ids
        integer, private, dimension(:), allocatable :: Q_perp_ids
        integer, private, dimension(:), allocatable :: K_par_ids
        integer, private, dimension(:), allocatable :: K_perp_ids
        logical, private :: is_writing
        integer, private :: write_counter
        real(kind=GP), dimension(:,:,:,:), allocatable :: send_buffer
        real(kind=GP), dimension(:,:,:,:), allocatable :: recv_buffer
    contains
        procedure, public :: initialize => initialize_mom_2d
        procedure, public :: write => write_mom_2d
        final :: finalize_mom_2d
    end type

    interface
        module subroutine initialize_mom_2d(this, dcomm_handler, &
                                            filename, n_points_RZ, &
                                            n_points_phi, n_points_sp)
            !! Initializes the mom_2d_file_t type and creates the
            !! corresponding NetCDF file on disk.
            class(mom_2d_file_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Communcation handler for MPI
            character(len=*), intent(in) :: filename
            !! Filename to write to
            integer, intent(in) :: n_points_RZ
            !! Number of points in RZ direction
            integer, intent(in) :: n_points_phi
            !! Number of points in phi direction
            integer, intent(in) :: n_points_sp
            !! Number of points in sp direction
        end subroutine

        module subroutine write_mom_2d(this, time, lb_stripped, ub_stripped, &
                                       mom_2d)
            !! Writes the time and the given total moments to file. The order
            !! of the moments is n, u_par, E_par, E_perp, Q_par, Q_perp,
            !! K_par, K_perp
            class(mom_2d_file_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: time
            !! Current simulation time
            integer, intent(in), dimension(5) :: lb_stripped
            !! Lower boundary without ghost cells
            integer, intent(in), dimension(5) :: ub_stripped
            !! Upper boundary without ghost cells
            real(kind=GP), intent(in) :: mom_2d(&
                lb_stripped(1) : ub_stripped(1), &
                lb_stripped(2) : ub_stripped(2), &
                1              : 8, &
                lb_stripped(5) : ub_stripped(5))
            !! Moments
        end subroutine

        module subroutine finalize_mom_2d(this)
            !! Finalizes the mom_2d_file_t type
            type(mom_2d_file_t), intent(inout) :: this
        end subroutine
    end interface

    type, public, extends(diag_file_base_t) :: neutrals_file_t
        !! Type to represent the diagnostic file for the neutrals
        integer, private :: n_points_RZ
        integer, private :: n_points_phi
        integer, private :: n_neut_sp
        integer, private :: dim_time_id
        integer, private :: dim_RZ_id
        integer, private :: dim_phi_id
        integer, private :: time_id
        integer, private, dimension(:), allocatable :: species_group
        integer, private, dimension(:), allocatable :: n_ids
        logical, private :: is_writing
        integer, private :: write_counter
        real(kind=GP), dimension(:,:,:,:), allocatable :: send_buffer
        real(kind=GP), dimension(:,:,:,:), allocatable :: recv_buffer
    contains
        procedure, public :: initialize => initialize_neut
        procedure, public :: write => write_neut
        final :: finalize_neut
    end type

    interface
        module subroutine initialize_neut(this, dcomm_handler, &
                                          filename, n_points_RZ, &
                                          n_points_phi, n_neut_sp)
            !! Initializes the neutrals_file_t type and creates the
            !! corresponding NetCDF file on disk.
            class(neutrals_file_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Communcation handler for MPI
            character(len=*), intent(in) :: filename
            !! Filename to write to
            integer, intent(in) :: n_points_RZ
            !! Number of points in RZ direction
            integer, intent(in) :: n_points_phi
            !! Number of points in phi direction
            integer, intent(in) :: n_neut_sp
            !! Number of points in sp direction
        end subroutine

        module subroutine write_neut(this, time, lb_neut, ub_neut, &
                                     neutrals_state)
            !! Writes the time and the given neutrals moments to file.
            !! Only density n currently.
            class(neutrals_file_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: time
            !! Current simulation time
            integer, intent(in), dimension(4) :: lb_neut
            !! Lower boundary without ghost cells for neutrals
            integer, intent(in), dimension(4) :: ub_neut
            !! Upper boundary without ghost cells for neutrals
            real(kind=GP), intent(in) :: neutrals_state(&
                lb_neut(1) : ub_neut(1), &
                lb_neut(2) : ub_neut(2), &
                1, &
                lb_neut(4) : ub_neut(4))
            !! Neutrals moments
        end subroutine

        module subroutine finalize_neut(this)
            !! Finalizes the neutrals_file_t type
            type(neutrals_file_t), intent(inout) :: this
        end subroutine
    end interface

    type, public, extends(diag_file_base_t) :: em_fields_file_t
        !! Type to represent the diagnostic file for the electromagnetic
        !! fields
        integer, private :: n_points_RZ
        integer, private :: n_points_phi
        integer, private :: dim_time_id
        integer, private :: dim_RZ_id
        integer, private :: dim_phi_id
        integer, private :: time_id
        integer, private :: es_pot_id
        integer, private :: A_par_id
        integer, private :: B_par_id
        logical, private :: is_writing
        integer, private :: write_counter
        real(kind=GP), dimension(:, :, :), allocatable :: send_buffer
        real(kind=GP), dimension(:, :, :), allocatable :: recv_buffer
    contains
        procedure, public :: initialize => initialize_em_fields
        procedure, public :: write => write_em_fields
        final :: finalize_em_fields
    end type

    interface
        module subroutine initialize_em_fields(this, dcomm_handler, &
                                               filename, n_points_RZ, &
                                               n_points_phi)
            !! Initializes the em_fields_file_t type and creates the
            !! corresponding NetCDF file on disk.
            class(em_fields_file_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Communcation handler for MPI
            character(len=*), intent(in) :: filename
            !! Filename to write to
            integer, intent(in) :: n_points_RZ
            !! Number of points in RZ direction
            integer, intent(in) :: n_points_phi
            !! Number of points in phi direction
        end subroutine

        module subroutine write_em_fields(this, time, lb_stripped, &
                                          ub_stripped, es_pot, A_par, B_par)
            !! Writes the time and the given electrostatic potential, the
            !! parallel electromagnetic vector potential and the parallel
            !! magnetic fluctuations to disk.
            class(em_fields_file_t), intent(inout) :: this
            !! Instance of the type.
            real(kind=GP), intent(in) :: time
            !! Current simulation time
            integer, dimension(2), intent(in) :: lb_stripped
            !! Lower boundary without ghost cells
            integer, dimension(2), intent(in) :: ub_stripped
            !! Upper boundary without ghost cells
            real(kind=GP), intent(in) :: es_pot( &
                lb_stripped(1) :, &
                lb_stripped(2) :)
            !! Electrostatic potential
            real(kind=GP), intent(in) :: A_par( &
                lb_stripped(1) :, &
                lb_stripped(2) :)
            !! Parallel electromagnetic vector potential
            real(kind=GP), intent(in) :: B_par( &
                lb_stripped(1) :, &
                lb_stripped(2) :)
            !! Parallel magnetic fluctuations
        end subroutine

        module subroutine finalize_em_fields(this)
            !! Finalizes the mom_0d_file_t type
            type(em_fields_file_t), intent(inout) :: this
        end subroutine
    end interface

    type, public, extends(diag_file_base_t) :: checkpoint_file_t
        !! Type to represent the checkpoint file
        integer, private :: dim_time_id
        !! NetCDF ID for time dimension
        integer, private :: dim_RZ_id
        !! NetCDF ID for RZ dimension
        integer, private :: dim_phi_id
        !! NetCDF ID for phi dimension
        integer, private :: dim_vp_id
        !! NetCDF ID for vp dimension
        integer, private :: dim_mu_id
        !! NetCDF ID for mu dimension
        integer, private :: dim_sp_id
        !! NetCDF ID for ip dimension
        integer, private :: dist_func_id
        !! NetCDF ID for distribution function
        integer, private :: time_id
        !! NetCDF ID for time
        integer, private :: dim_bound_id
        !! NetCDF ID for boundaries (lb_stripped, ub_stripped) dimension
        integer, private :: lb_stripped_id
        !! NetCDF ID for lb_stripped
        integer, private :: ub_stripped_id
        !! NetCDF ID for ub_stripped
        integer, private :: write_counter
        !! Timestep indicating where NetCDF starts to write the file
    contains
        procedure, private :: calculate_process_id
        procedure, public :: initialize => initialize_checkpoint
        procedure, public :: write => write_checkpoint
        procedure, public :: read => read_checkpoint
        final :: finalize_checkpoint
    end type

    interface
        module subroutine initialize_checkpoint(this, dcomm_handler, &
                                                filename, n_points_RZ, &
                                                n_points_phi, n_points_vp, &
                                                n_points_mu, n_points_sp, &
                                                from_existing)
            !! Initializes the checkpoint_file_t type and creates the
            !! corresponding files on disk. Every MPI process writes to its own
            !! checkpoint file on disk. The handling of the different output
            !! files is done automatically within the type.
            class(checkpoint_file_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Communcation handler for MPI
            character(len=*), intent(in) :: filename
            !! Filename to write to
            integer, intent(in) :: n_points_RZ
            !! Number of points in RZ direction
            integer, intent(in) :: n_points_phi
            !! Number of points in phi direction
            integer, intent(in) :: n_points_vp
            !! Number of points in vp direction
            integer, intent(in) :: n_points_mu
            !! Number of points in mu direction
            integer, intent(in) :: n_points_sp
            !! Number of points in sp direction
            logical, optional, intent(in) :: from_existing
            !! .true. if the file should be created from an existing
            !! checkpoint file .false. if a new checkpoint file should be
            !! created
        end subroutine

        module subroutine write_checkpoint(this, time, dist_func)
            !! Writes the time and the given distribution function to file
            class(checkpoint_file_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: time
            !! Current simulation time
            real(kind=GP), pointer, &
                             dimension(:,:,:,:,:), intent(in) :: dist_func
            !! Distribution function
        end subroutine

        module subroutine read_checkpoint(this, time, dist_func)
            !! Reads the time and the given distribution function from file
            class(checkpoint_file_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(out) :: time
            !! Read simulation time
            real(kind=GP), pointer, dimension(:,:,:,:,:), &
                                                    intent(inout) :: dist_func
            !! Read distribution function
        end subroutine

        integer module function calculate_process_id(this)
            !! Calculates a unique process id corresponding to the coordinates
            !! of the current MPI process within the MPI topology
            class(checkpoint_file_t), intent(in) :: this
            !! Instance of the type
        end function

        module subroutine finalize_checkpoint(this)
            !! Finalizes the checkpoint_file_t type
            type(checkpoint_file_t), intent(inout) :: this
        end subroutine
    end interface

    type, public, extends(diag_file_base_t) :: checkpoint_neut_file_t
        !! Type to represent the checkpoint file
        integer, private :: dim_time_id
        !! NetCDF ID for time dimension
        integer, private :: dim_RZ_id
        !! NetCDF ID for RZ dimension
        integer, private :: dim_phi_id
        !! NetCDF ID for phi dimension
        integer, private :: dim_mom_id
        !! NetCDF ID for moment dimension
        integer, private :: dim_sp_id
        !! NetCDF ID for sp dimension
        integer, private :: neutrals_state_id
        !! NetCDF ID for neutrals_state
        integer, private :: time_id
        !! NetCDF ID for time
        integer, private :: dim_bound_id
        !! NetCDF ID for boundaries (lb_stripped, ub_stripped) dimension
        integer, private :: lb_stripped_id
        !! NetCDF ID for lb_stripped
        integer, private :: ub_stripped_id
        !! NetCDF ID for ub_stripped
        integer, private :: write_counter
        !! Timestep indicating where NetCDF starts to write the file
    contains
        procedure, private :: calculate_process_id_neut
        procedure, public :: initialize => initialize_checkpoint_neut
        procedure, public :: write => write_checkpoint_neut
        procedure, public :: read => read_checkpoint_neut
        final :: finalize_checkpoint_neut
    end type

    interface
        module subroutine initialize_checkpoint_neut(this, dcomm_handler, &
            filename, n_points_RZ, n_points_phi, n_points_mom, &
            n_points_sp, from_existing)
            !! Initializes the checkpoint_neut_file_t type and creates the
            !! corresponding files on disk. Every MPI process writes to its own
            !! checkpoint file on disk. The handling of the different output
            !! files is done automatically within the type.
            class(checkpoint_neut_file_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Communcation handler for MPI
            character(len=*), intent(in) :: filename
            !! Filename to write to
            integer, intent(in) :: n_points_RZ
            !! Number of points in RZ direction
            integer, intent(in) :: n_points_phi
            !! Number of points in phi direction
            integer, intent(in) :: n_points_mom
            !! Number of points in mom direction
            integer, intent(in) :: n_points_sp
            !! Number of points in sp direction
            logical, optional, intent(in) :: from_existing
            !! Indicates if the file should be created from an existing
            !! checkpoint file or not
        end subroutine

        module subroutine write_checkpoint_neut(this, time, neutrals_state)
            !! Writes the time and the neutrals_state to file
            class(checkpoint_neut_file_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: time
            !! Current simulation time
            real(kind=GP), pointer, &
                               dimension(:,:,:,:), intent(in) :: neutrals_state
            !! Neutrals state
        end subroutine

        module subroutine read_checkpoint_neut(this, time, neutrals_state)
            !! Reads the time and the given ineutrals state from file
            class(checkpoint_neut_file_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(out) :: time
            !! Read simulation time
            real(kind=GP), pointer, &
                             dimension(:,:,:,:), intent(inout) :: neutrals_state
            !! Neutrals state
        end subroutine

        integer module function calculate_process_id_neut(this)
            !! Calculates a unique process id corresponding to the coordinates
            !! of the current MPI process within the MPI topology
            !! NOTE: Same as calculate_process_id
            class(checkpoint_neut_file_t), intent(in) :: this
            !! Instance of the type
        end function

        module subroutine finalize_checkpoint_neut(this)
            !! Finalizes the checkpoint_neut_file_t type
            type(checkpoint_neut_file_t), intent(inout) :: this
        end subroutine
    end interface

    type, public, extends(diag_file_base_t) :: mesh_file_t
        !! Type to represent the mesh file
        integer, private, dimension(2) :: id_dim_RZphi
        !! NetCDF IDs for the RZ and phi dimensions
        integer, private, dimension(3) :: id_dim_TRphi
        !! NetCDF IDs for the rho, theta and phi dimensions
        integer, private, dimension(:), allocatable :: id_multigrid
        !! NetCDF IDs for the multigrid groups per phi plane
        integer, private, dimension(:), allocatable :: id_map_positive1
        !! NetCDF IDs for the map_positive1 groups per phi plane
        integer, private, dimension(:), allocatable :: id_map_positive2
        !! NetCDF IDs for the map_positive2 groups per phi plane
        integer, private, dimension(:), allocatable :: id_map_negative1
        !! NetCDF IDs for the map_negative1 groups per phi plane
        integer, private, dimension(:), allocatable :: id_map_negative2
        !! NetCDF IDs for the map_negative2 groups per phi plane
    contains
        procedure, public :: initialize => initialize_mesh
        procedure, public :: get_id_multigrid
        procedure, public :: get_id_map_positive1
        procedure, public :: get_id_map_positive2
        procedure, public :: get_id_map_negative1
        procedure, public :: get_id_map_negative2
        procedure, public :: write_mesh
        procedure, public :: write_polar_mesh
        procedure, public :: write_magnetic_field
        procedure, public :: write_polar_magnetic_field
        procedure, public :: write_parcon
        procedure, public :: read_dimensions
        procedure, public :: read_polar_dimensions
        procedure, public :: read_mesh
        procedure, public :: read_polar_mesh
        procedure, public :: read_magnetic_field
        procedure, public :: read_polar_magnetic_field
        procedure, public :: read_parcon
    end type

    interface
        module subroutine initialize_mesh(this, dcomm_handler, filename, &
                                          n_points_phi, is_axisymmetric, &
                                          read_mesh)
            !! Initializes the mesh_file_t type and creates the mesh NetCDF
            !! file.
            class(mesh_file_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Communication handler for MPI
            character(len=*), intent(in) :: filename
            !! Filename to write to
            integer, intent(in) :: n_points_phi
            !! Number of points in phi direction
            logical, intent(in) :: is_axisymmetric
            !! Bool indicating whether the underlying equilibrium is
            !! axisymmetric
            logical, optional, intent(in) :: read_mesh
            !! Indicates whether the mesh will be read from file. If not
            !! provided, it is assumed that the mesh will be written to file.
        end subroutine

        module subroutine write_mesh(this, equi_type, size_RZ_max, not_ghost, &
                not_filler, buf_zone, R, Z, RZw, x_ind, z_ind, RZ_ind, &
                jacobian, phi, mu, vp)
            !! Writes information about the mesh to file.
            class(mesh_file_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(in) :: equi_type
            !! Magnetic equilibrium type
            integer, intent(in) :: size_RZ_max
            !! Maximum number of unstructured grid points from all phi planes
            real(kind=GP), dimension(:,:), intent(in) :: not_ghost
            !! Mask indicating whether a buffer point is a ghost
            real(kind=GP), dimension(:,:), intent(in) :: not_filler
            !! Mask indicating whether a buffer point is a filler point
            integer, dimension(:,:), intent(in) :: buf_zone
            !! Indicates whether a points is in the numerical buffer zone
            real(kind=GP), dimension(:,:), intent(in) :: R
            !! X values of RZ grid
            real(kind=GP), dimension(:,:), intent(in) :: Z
            !! Z values of RZ grid
            real(kind=GP), dimension(:,:), intent(in) :: RZw
            !! RZ quadrature weights
            integer, dimension(:,:), intent(in) :: x_ind
            !! Cartesian x indices of RZ grid
            integer, dimension(:,:), intent(in) :: z_ind
            !! Cartesian z indices of RZ grid
            integer, dimension(:,:), intent(in) :: RZ_ind
            !! Array indices of RZ grid
            real(kind=GP), dimension(:,:), intent(in) :: jacobian
            !! Jacobian of the coordinate transform from lab coordinates into
            !! the coordinate system of the mesh
            real(kind=GP), dimension(:), intent(in) :: phi
            !! Values of phi grid
            real(kind=GP), dimension(:), intent(in) :: mu
            !! Values of mu grid
            real(kind=GP), dimension(:), intent(in) :: vp
            !! Values of v parallel grid

        end subroutine

        module subroutine write_polar_mesh(this, polar_theta, polar_rho, phi)
            !! Writes information about the polar mesh to file
            class(mesh_file_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), dimension(:,:), intent(in) :: polar_theta
            !! Theta values of polar mesh
            real(kind=GP), dimension(:,:), intent(in) :: polar_rho
            !! Rho values of polar mesh
            real(kind=GP), dimension(:), intent(in) :: phi
            !! Values of phi grid
        end subroutine

        module subroutine write_magnetic_field(this, &
                psi_norm_fac, psi_lo, psi_up, absB_max, &
                absB, normb_R, normb_Z, normb_tor, curl_normb_y, &
                dgyxdy_over_g, dgyzdy_over_g, dgyxdz_over_g, dgyzdx_over_g, &
                inv_g, dabsBdx, dabsBdz, dabsBdy, rho, theta, psi)
            !! Writes information about the magnetic field to file.
            class(mesh_file_t), intent(in) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: psi_norm_fac
            !! Normalization factor for poloidal flux
            real(kind=GP), intent(in) :: psi_lo
            !! Lower value of psi for normalization
            real(kind=GP), intent(in) :: psi_up
            !! Upper value of psi for normalization
            real(kind=GP), intent(in) :: absB_max
            !! Maximum value of absolute magnetic field
            real(kind=GP), dimension(:,:), intent(in) :: absB
            !! B field on the 3D mesh
            real(kind=GP), dimension(:,:), intent(in) :: normb_R
            !! R component of the normalized magnetic field
            real(kind=GP), dimension(:,:), intent(in) :: normb_Z
            !! Z component of the normalized magnetic field
            real(kind=GP), dimension(:,:), intent(in) :: normb_tor
            !! Torodial component of the normalized magnetic field
            real(kind=GP), dimension(:,:), intent(in) :: curl_normb_y
            !! y component of the curl of the normalized magnetic field
            real(kind=GP), dimension(:,:), intent(in) :: dgyxdy_over_g
            !! Derivative of b_R along the magnetic field on the 3D mesh
            real(kind=GP), dimension(:,:), intent(in) :: dgyzdy_over_g
            !! Derivative of b_Z along the magnetic field on the 3D mesh
            real(kind=GP), dimension(:,:), intent(in) :: dgyxdz_over_g
            !! Derivative of b_R in Z direction on the 3D mesh
            real(kind=GP), dimension(:,:), intent(in) :: dgyzdx_over_g
            !! Derivative of b_Z in R direction on the 3D mesh
            real(kind=GP), dimension(:,:), intent(in) :: inv_g
            !! Inverse metric determinant
            real(kind=GP), dimension(:,:), intent(in) :: dabsBdx
            !! Derivative of abs B in R direction on the 3D mesh
            real(kind=GP), dimension(:,:), intent(in) :: dabsBdz
            !! Derivative of abs B in Z direction on the 3D mesh
            real(kind=GP), dimension(:,:), intent(in) :: dabsBdy
            !! Derivative of abs B in u direction on the 3D mesh
            real(kind=GP), dimension(:,:), intent(in) :: rho
            !! Flux surface label on the 3D mesh
            real(kind=GP), dimension(:,:), intent(in) :: theta
            !! Poloidal angle
            real(kind=GP), dimension(:,:), intent(in) :: psi
            !! Poloidal flux function in normalized units
        end subroutine

        module subroutine write_polar_magnetic_field(this, polar_absB, &
                                                     loss_cone)
            !! Writes information about the magnetic field on polar mesh to file
            class(mesh_file_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), dimension(:,:,:), intent(in) :: polar_absB
            !! B field absolute on the polar mesh
            real(kind=GP), dimension(:,:),   intent(in) :: loss_cone
            !! Loss cone for TPC diagnostics
        end subroutine

        module subroutine write_parcon(this, not_in_target, maps_on_mesh, &
                                       fll_positive1, fll_positive2, &
                                       fll_negative1, fll_negative2, &
                                       parcon_positive1, parcon_positive2, &
                                       parcon_negative1, parcon_negative2)
            !! Writes information about the parallel connection to file.
            class(mesh_file_t), intent(in) :: this
            !! Instance of the type
            integer, intent(in), dimension(:,:) :: not_in_target
            !! Array specifying whether a given point is not in the target
            real(kind=GP), intent(in), dimension(:,:) :: maps_on_mesh
            !! Array specifying whether a given point maps onto all of the
            !! neighboring meshes (4 point stencil)
            real(kind=GP), intent(in), dimension(:,:) :: fll_positive1
            !! Field line lengths traced from each mesh point to the k + 1 plane
            real(kind=GP), intent(in), dimension(:,:) :: fll_positive2
            !! Field line lengths traced from each mesh point to the k + 2 plane
            real(kind=GP), intent(in), dimension(:,:) :: fll_negative1
            !! Field line lengths traced from each mesh point to the k - 1 plane
            real(kind=GP), intent(in), dimension(:,:) :: fll_negative2
            !! Field line lengths traced from each mesh point to the k - 2 plane
            integer, intent(in), dimension(:,:) :: parcon_positive1
            !! Array specifying whether a given point is connected by a parallel
            !! stencil to the target on the k + 1 plane
            integer, intent(in), dimension(:,:) :: parcon_positive2
            !! Array specifying whether a given point is connected by a parallel
            !! stencil to the target on the k + 2 plane
            integer, intent(in), dimension(:,:) :: parcon_negative1
            !! Array specifying whether a given point is connected by a parallel
            !! stencil to the target on the k - 1 plane
            integer, intent(in), dimension(:,:) :: parcon_negative2
            !! Array specifying whether a given point is connected by a parallel
            !! stencil to the target on the k - 2 plane
        end subroutine

        module subroutine read_dimensions(this, n_points_RZ, n_points_phi)
            !! Reads mesh data dimensions from file
            class(mesh_file_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(out) :: n_points_RZ
            !! Dimension in RZ direction
            integer, intent(out) :: n_points_phi
            !! Dimension in phi direction
        end subroutine

        module subroutine read_polar_dimensions(this, &
                                                polar_n_theta, polar_n_rho)
            !! Reads polar mesh data dimensions from file
            class(mesh_file_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(out) :: polar_n_theta
            !! Dimension of theta coordinate of the polar mesh
            integer, intent(out) :: polar_n_rho
            !! Dimension of rho coordinate of the polar mesh
        end subroutine

        module subroutine read_mesh(this, equi_type, size_RZ_max, not_ghost, &
                                    not_filler, buf_zone, R, Z, RZw, RZ_ind, &
                                    jacobian, phi)
            !! Reads information about the mesh from file.
            class(mesh_file_t), intent(in) :: this
            !! Instance of the type
            integer, intent(out) :: equi_type
            !! Magnetic equilibrium type
            integer, intent(out) :: size_RZ_max
            !! Maximum number of unstructured grid points from all phi planes
            real(kind=GP), dimension(:,:), intent(out) :: not_ghost
            !! Mask indicating whether a buffer point is a ghost
            real(kind=GP), dimension(:,:), intent(out) :: not_filler
            !! Mask indicating whether a buffer point is a filler point
            integer, dimension(:,:), intent(out) :: buf_zone
            !! Array specifying which buffer zone a given point is in
            real(kind=GP), dimension(:,:), intent(out) :: R
            !! X values of RZ grid
            real(kind=GP), dimension(:,:), intent(out) :: Z
            !! Z values of RZ grid
            real(kind=GP), dimension(:,:), intent(out) :: RZw
            !! RZ quadrature weights
            integer, dimension(:,:), intent(out) :: RZ_ind
            !! Array indices of RZ grid
            real(kind=GP), dimension(:,:), intent(out) :: jacobian
            !! Jacobian of the coordinate transform from lab coordinates into
            !! the coordinate system of the mesh
            real(kind=GP), dimension(:), intent(out) :: phi
            !! Values of phi grid
        end subroutine

        module subroutine read_polar_mesh(this, polar_theta, polar_rho)
            !! Reads information about the polar mesh from file
            class(mesh_file_t), intent(in) :: this
            !! Instance of the type
            real(kind=GP), dimension(:,:), intent(out) :: polar_theta
            !! Theta values on the polar mesh
            real(kind=GP), dimension(:,:), intent(out) :: polar_rho
            !! Rho values on the polar mesh
        end subroutine

        module subroutine read_magnetic_field(this, psi_norm_fac, psi_lo, &
            psi_up, absB_max, absB, normb_R, normb_Z, normb_tor, curl_normb_y, &
            dgyxdy_over_g, dgyzdy_over_g, dgyxdz_over_g, dgyzdx_over_g, &
            inv_g, dabsBdx, dabsBdz, dabsBdy, rho, theta, psi)
            !! Reads information about the magnetic field from file.
            class(mesh_file_t), intent(in) :: this
            !! Instance of the type
            real(kind=GP), intent(out) :: psi_norm_fac
            !! Normalization factor for poloidal flux
            real(kind=GP), intent(out) :: psi_lo
            !! Lower value of psi for normalization
            real(kind=GP), intent(out) :: psi_up
            !! Upper value of psi for normalization
            real(kind=GP), intent(out) :: absB_max
            !! Maximum value of absolute magnetic field
            real(kind=GP), dimension(:,:), intent(out) :: absB
            !! B field on the 3D mesh
            real(kind=GP), dimension(:,:), intent(out) :: normb_R
            !! R component of the normalized magnetic field
            real(kind=GP), dimension(:,:), intent(out) :: normb_Z
            !! Z component of the normalized magnetic field
            real(kind=GP), dimension(:,:), intent(out) :: normb_tor
            !! Torodial component of the normalized magnetic field
            real(kind=GP), dimension(:,:), intent(out) :: curl_normb_y
            !! y component of the curl of the normalized magnetic field
            real(kind=GP), dimension(:,:), intent(out) :: dgyxdy_over_g
            !! Derivative of b_R along the magnetic field on the 3D mesh
            real(kind=GP), dimension(:,:), intent(out) :: dgyzdy_over_g
            !! Derivative of b_Z along the magnetic field on the 3D mesh
            real(kind=GP), dimension(:,:), intent(out) :: dgyxdz_over_g
            !! Derivative of b_R in Z direction on the 3D mesh
            real(kind=GP), dimension(:,:), intent(out) :: dgyzdx_over_g
            !! Derivative of b_Z in R direction on the 3D mesh
            real(kind=GP), dimension(:,:), intent(out) :: inv_g
            !! Inverse metric determinant
            real(kind=GP), dimension(:,:), intent(out) :: dabsBdx
            !! Derivative of abs B in R direction on the 3D mesh
            real(kind=GP), dimension(:,:), intent(out) :: dabsBdz
            !! Derivative of abs B in Z direction on the 3D mesh
            real(kind=GP), dimension(:,:), intent(out) :: dabsBdy
            !! Derivative of abs B in u direction on the 3D mesh
            real(kind=GP), dimension(:,:), intent(out) :: rho
            !! Flux surface label on the 3D mesh
            real(kind=GP), dimension(:,:), intent(out) :: theta
            !! Poloidal angle
            real(kind=GP), dimension(:,:), intent(out) :: psi
            !! Poloidal flux function in normalized units
        end subroutine

        module subroutine read_polar_magnetic_field(this, polar_absB, loss_cone)
            !! Reads information about the magnetic field on polar mesh
            !! from file
            class(mesh_file_t), intent(in) :: this
            !! Instance of the type
            real(kind=GP), dimension(:,:,:), intent(out) :: polar_absB
            !! Magnetic field on the polar mesh
            real(kind=GP), dimension(:,:),   intent(out) :: loss_cone
            !! Loss cone for TPC diagnostics
        end subroutine

        module subroutine read_parcon(this, not_in_target, maps_on_mesh, &
                                      fll_positive1, fll_positive2, &
                                      fll_negative1, fll_negative2, &
                                      parcon_positive1, parcon_positive2, &
                                      parcon_negative1, parcon_negative2)
            !! Reads information about the parallel connection from file.
            class(mesh_file_t), intent(in) :: this
            !! Instance of the type
            integer, intent(out), dimension(:,:) :: not_in_target
            !! Array specifying whether a given point is not in the target
            real(kind=GP), intent(out), dimension(:,:) :: maps_on_mesh
            !! Array specifying whether a given point maps onto all of the
            !! neighboring meshes (4 point stencil)
            real(kind=GP), intent(out), dimension(:,:) :: fll_positive1
            !! Field line lengths traced from each mesh point to the k + 1 plane
            real(kind=GP), intent(out), dimension(:,:) :: fll_positive2
            !! Field line lengths traced from each mesh point to the k + 2 plane
            real(kind=GP), intent(out), dimension(:,:) :: fll_negative1
            !! Field line lengths traced from each mesh point to the k - 1 plane
            real(kind=GP), intent(out), dimension(:,:) :: fll_negative2
            !! Field line lengths traced from each mesh point to the k - 2 plane
            integer, intent(out), dimension(:,:) :: parcon_positive1
            !! Array specifying whether a given point is connected by a parallel
            !! stencil to the target on the k + 1 plane
            integer, intent(out), dimension(:,:) :: parcon_positive2
            !! Array specifying whether a given point is connected by a parallel
            !! stencil to the target on the k + 2 plane
            integer, intent(out), dimension(:,:) :: parcon_negative1
            !! Array specifying whether a given point is connected by a parallel
            !! stencil to the target on the k - 1 plane
            integer, intent(out), dimension(:,:) :: parcon_negative2
            !! Array specifying whether a given point is connected by a parallel
            !! stencil to the target on the k - 2 plane
        end subroutine
    end interface

contains

    integer function get_id_multigrid(this, k)
        !! Returns the k-th element of the id_multigrid array
        class(mesh_file_t), target, intent(in) :: this
        integer, intent(in) :: k

        get_id_multigrid = this%id_multigrid(k)
    end function

    integer function get_id_map_positive1(this, k)
        !! Returns the k-th element of the id_map_positive1 array
        class(mesh_file_t), target, intent(in) :: this
        integer, intent(in) :: k

        get_id_map_positive1 = this%id_map_positive1(k)
    end function

    integer function get_id_map_positive2(this, k)
        !! Returns the k-th element of the id_map_positive2 array
        class(mesh_file_t), target, intent(in) :: this
        integer, intent(in) :: k

        get_id_map_positive2 = this%id_map_positive2(k)
    end function

    integer function get_id_map_negative1(this, k)
        !! Returns the k-th element of the id_map_negative1 array
        class(mesh_file_t), target, intent(in) :: this
        integer, intent(in) :: k

        get_id_map_negative1 = this%id_map_negative1(k)
    end function

    integer function get_id_map_negative2(this, k)
        !! Returns the k-th element of the id_map_negative2 array
        class(mesh_file_t), target, intent(in) :: this
        integer, intent(in) :: k

        get_id_map_negative2 = this%id_map_negative2(k)
    end function

    subroutine check_array(this, desired_shape, array_shape, line_number, &
                           file_name)
        !! Checks the shape of an array to write in this file
        class(diag_file_base_t), intent(in) :: this
        !! Instance of the class
        integer, dimension(:), intent(in) :: desired_shape
        !! Shape that the array needs to have
        integer, dimension(:), intent(in) :: array_shape
        !! Shape of the array to be checked
        integer, intent(in) :: line_number
        !! Line number where error or warning occured, i.e. __LINE__
        character(len=*), intent(in) :: file_name
        !! File name where error or warning occured, i.e. __FILE__

        integer :: i

        ! Go through all dimensions in shape and compare them to the
        ! desired one
        do i = 1,size(array_shape)
            if(array_shape(i) /= desired_shape(i)) then
                call handle_error("Shape of array does not match the &
                                  &expected shape!", &
                                  GENEX_ERR_DIAG_FILES, line_number, &
                                  file_name, &
                                  additional_info=error_info_t(&
                                    "Desired/actual shape: ", &
                                    [desired_shape(:), array_shape(:)]))
            end if
        end do
    end subroutine

end module diag_files_m
