module params_mesh_m
    !! Module handling the parameters for the mesh
    use genex_fortran_env_m, only : GP
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error
    use genex_status_codes_m, only: GENEX_ERR_PARAMETERS
    use genex_error_handling_m, only: handle_error
    ! From PARALLAX
    use equilibrium_factory_m, only: CIRCULAR

    implicit none
    private

    integer, parameter, public :: n_levels_supported = 16
    !! Maximum number of multigrid levels supported

    integer :: n_levels = 2
    !! Number of multigrid levels
    real(kind=GP) :: spacing_RZ = 0.05
    !! Grid spacing in the poloidal plane
    real(kind=GP) :: length_vp = 3
    !! Length in vp direction
    real(kind=GP) :: length_mu = 9
    !! Length in mu direction
    integer :: n_points_phi = 16
    !! Number of points in phi direction
    integer :: n_points_vp = 32
    !! Number of points in vp direction
    integer :: n_points_mu = 16
    !! Number of points in mu direction
    integer :: n_points_sp = 2
    !! Number of species
    integer :: equilibrium_type = CIRCULAR
    !! Type of the equilibrium
    logical :: only_first_field_period = .false.
    !! Simulate only in the first field period of the equilibrium. Option only
    !! valid for non-axisymmetric equilibria.
    logical :: extend_beyond_wall = .false.
    !! Extend mesh to include the points which are identified as DISTRICT_WALL,
    !! stopping instead at DISTRICT_OUT
    character(len=24) :: quad_type_vp = "simpson"
    !! Type of the quadrature used for the vp grid.
    !! Valid options are: midpoint, trapezoidal, simpson
    character(len=24) :: grid_type_mu = "gauss-laguerre"
    !! Type of the grid used for the mu grid. Quadrature will be selected based
    !! on the grid type specified. The mu grid will be generated as a staggered
    !! grid except in the case of gauss-laguerre.
    !! Valid options are:           uniform,  quadratic, gauss-laguerre
    !! Corresponding quadrature is: midpoint, midpoint,  gauss-laguerre
    logical :: use_vspectral = .false.
    !! Switch spectral method in vp and mu
    logical :: use_bsg = .false.
    !! Activates block-structured grid for velocity space
    integer, dimension(n_levels_supported) :: reorder_size = 0
    !! Reorder block size of each multigrid level.
    !! Zero means no reorder, points are ordered: inner - boundary - ghost

    namelist / params_mesh / &
        n_levels, spacing_RZ, n_points_phi, n_points_vp, n_points_mu, &
        n_points_sp, length_vp, length_mu, quad_type_vp, grid_type_mu, &
        use_vspectral, equilibrium_type, only_first_field_period, use_bsg, &
        reorder_size, extend_beyond_wall

    public :: read_params_mesh
    public :: write_params_mesh

    public :: get_n_levels
    public :: get_spacing_RZ
    public :: get_length_vp
    public :: get_length_mu
    public :: get_n_points_phi
    public :: get_n_points_vp
    public :: get_n_points_mu
    public :: get_n_points_sp
    public :: get_equilibrium_type
    public :: get_only_first_field_period
    public :: get_extend_beyond_wall
    public :: get_quad_type_vp
    public :: get_grid_type_mu
    public :: get_use_vspectral
    public :: get_use_bsg
    public :: get_reorder_size

contains

    integer function get_n_levels()
        !! Returns the number of multigrid levels
        call handle_bounds_error(n_levels, 1, n_levels_supported, &
                                 __LINE__, __FILE__)
        get_n_levels = n_levels
    end function

    pure integer function get_equilibrium_type()
        !! Returns the grid spacing in the poloidal plane
        get_equilibrium_type = equilibrium_type
    end function

    pure logical function get_only_first_field_period()
        !! Returns the switch to simulate in only the first field period
        get_only_first_field_period = only_first_field_period
    end function

    pure logical function get_extend_beyond_wall()
        !! Returns the switch to extend the mesh beyond the vessel wall
        get_extend_beyond_wall = extend_beyond_wall
    end function

    pure real(kind=GP) function get_spacing_RZ()
        !! Returns the grid spacing in the poloidal plane
        get_spacing_RZ = spacing_RZ
    end function

    pure real(kind=GP) function get_length_mu()
        !! Returns the grid length in mu direction
        get_length_mu = length_mu
    end function

    pure real(kind=GP) function get_length_vp()
        !! Returns the grid length in vp direction
        get_length_vp = length_vp
    end function

    pure integer function get_n_points_phi()
        !! Returns the number of points in phi direction
        get_n_points_phi = n_points_phi
    end function

    pure integer function get_n_points_vp()
        !! Returns the number of points in vp direction
        get_n_points_vp = n_points_vp
    end function

    pure integer function get_n_points_mu()
        !! Returns the number of points in mu direction
        get_n_points_mu = n_points_mu
    end function

    pure integer function get_n_points_sp()
        !! Returns the number of species
        get_n_points_sp = n_points_sp
    end function

    pure function get_quad_type_vp() result(res)
        !! Returns the type of the vp quadrature
        character(len=24) :: res
        res = quad_type_vp
    end function

    pure function get_grid_type_mu() result(res)
        !! Returns the type of the mu grid
        character(len=24) :: res
        res = grid_type_mu
    end function

    pure logical function get_use_vspectral()
        !! Returns the switch using spectral velocity space in vp and in mu
        get_use_vspectral = use_vspectral
    end function

    pure logical function get_use_bsg()
        !! Returns the switch using block-structured grid approach in genex
        get_use_bsg = use_bsg
    end function

    function get_reorder_size() result(res)
        !! Returns the reorder size for the selected number of multigrid levels
        integer, dimension(n_levels) :: res
        call handle_bounds_error(n_levels, 1, n_levels_supported, &
                                 __LINE__, __FILE__)
        res = reorder_size(1:n_levels)
    end function

    subroutine write_params_mesh(filename)
        !! Writes the mesh namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_mesh, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="mesh")

        close(iunit)
    end subroutine

    subroutine read_params_mesh(filename)
        !! Reads the mesh namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read',&
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_mesh, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="mesh", &
                               iunit=iunit)

        close(iunit)
    end subroutine

end module
