module params_numerical_scheme_m
    !! Module handling the parameters for different numerical schemes
    use genex_fortran_env_m, only : GP
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error
    use interop_params_m, only: set_cxx_params_numerical_scheme

    implicit none
    private

    integer :: int_order = 3
    !! Order of the two dimensional interpolation for the calcuation of field
    !! line mapped values of fields
    real(kind=GP) :: hyp_y = 0.0_GP
    !! Hyperdiffusion coefficient in parallel direction
    real(kind=GP) :: hyp_xz = 0.0_GP
    !! Hyperdiffusion coefficient in R and Z direction
    real(kind=GP) :: hyp_vp = 0.0_GP
    !! Hyperdiffusion coefficient vp direction
    integer :: buf_zone_size = 2
    !! Size of the boundary buffer zone in grid-points
    real(kind=GP) :: buf_zone_strength = 1.0_GP
    !! Diffusion coefficient applied in the buffer zone
    integer :: buf_zone_size_axis = 0
    !! Size of the axis buffer zone in grid-points
    real(kind=GP) :: buf_zone_strength_axis = 1.0_GP
    !! Diffusion coefficient applied in the buffer zone at the magnetic axis

    namelist / params_numerical_scheme / &
        int_order, hyp_y, hyp_xz, hyp_vp, buf_zone_size, buf_zone_strength, &
        buf_zone_size_axis, buf_zone_strength_axis

    public :: read_params_numerical_scheme
    public :: write_params_numerical_scheme

    public :: get_int_order
    public :: get_hyp_y
    public :: get_hyp_xz
    public :: get_hyp_vp
    public :: get_buf_zone_size
    public :: get_buf_zone_strength
    public :: get_buf_zone_size_axis
    public :: get_buf_zone_strength_axis

contains

    integer function get_int_order()
        !! Returns the interpolation order for the calculation of field line
        !! mapped values of fields
        get_int_order = int_order
    end function

    real(kind=GP) function get_hyp_y()
        !! Returns the hyperdiffusion coefficient in parallel direction
        get_hyp_y = hyp_y
    end function

    real(kind=GP) function get_hyp_xz()
        !! Returns the hyperdiffusion coefficient in RZ direction
        get_hyp_xz = hyp_xz
    end function

    real(kind=GP) function get_hyp_vp()
        !! Returns the hyperdiffusion coefficient in vp direction
        get_hyp_vp = hyp_vp
    end function

    integer function get_buf_zone_size()
        !! Returns the size of the buffer zone
        get_buf_zone_size = buf_zone_size
    end function

    real(kind=GP) function get_buf_zone_strength()
        !! Returns the diffusion coefficient in the buffer zone
        get_buf_zone_strength = buf_zone_strength
    end function

    integer function get_buf_zone_size_axis()
        !! Returns the size of the buffer zone on axis
        get_buf_zone_size_axis = buf_zone_size_axis
    end function

    real(kind=GP) function get_buf_zone_strength_axis()
        !! Returns the diffusion coefficient in the buffer zone on axis
        get_buf_zone_strength_axis = buf_zone_strength_axis
    end function

    subroutine write_params_numerical_scheme(filename)
        !! Writes the numerical_scheme namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_numerical_scheme, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="numerical_scheme")

        close(iunit)
    end subroutine

    subroutine read_params_numerical_scheme(filename)
        !! Reads the numerical_scheme namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read',&
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_numerical_scheme, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="numerical_scheme", iunit=iunit)

        close(iunit)

        call set_cxx_params_numerical_scheme()
    end subroutine

end module
