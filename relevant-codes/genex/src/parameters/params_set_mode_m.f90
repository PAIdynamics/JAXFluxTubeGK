module params_set_mode_m
    !! Module handling the parameters for the mode initialization
    use genex_fortran_env_m, only : GP
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error

    implicit none
    private

    real(kind=GP) :: strength = 1.0_GP
    !! Strength of the perturbation
    real(kind=GP) :: mean_rho = 0.0_GP
    !! Rho position of the perturbation
    real(kind=GP) :: sigma_rho = 0.0_GP
    !! Width of the perturbation in rho direction
    real(kind=GP) :: k_par = 4.17e-3_GP
    !! Parallel wave number
    integer, private :: toroidal_mode_number = 1
    !! Toroidal mode number
    integer, private :: poloidal_mode_number = 10
    !! Poloidal mode number

    namelist / params_set_mode / &
            strength, mean_rho, sigma_rho, k_par, &
            toroidal_mode_number, poloidal_mode_number

    public :: read_params_set_mode
    public :: write_params_set_mode

    public :: get_mean_rho
    public :: get_sigma_rho
    public :: get_strength
    public :: get_k_par
    public :: get_toroidal_mode_number
    public :: get_poloidal_mode_number

contains

    pure real(kind=GP) function get_mean_rho()
        !! Returns the rho position of the mode
        get_mean_rho = mean_rho
    end function

    pure real(kind=GP) function get_sigma_rho()
        !! Returns the width of the mode in rho direction
        get_sigma_rho = sigma_rho
    end function

    pure real(kind=GP) function get_strength()
        !! Returns the strength of the perturbation
        get_strength = strength
    end function

    pure real(kind=GP) function get_k_par()
        !! Returns the parallel wave number
        get_k_par = k_par
    end function

    pure integer function get_toroidal_mode_number()
        !! Returns the toroidal mode number
        get_toroidal_mode_number = toroidal_mode_number
    end function

    pure integer function get_poloidal_mode_number()
        !! Returns the poloidal mode number
        get_poloidal_mode_number = poloidal_mode_number
    end function

    subroutine write_params_set_mode(filename)
        !! Writes the set mode namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_set_mode, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="set_mode")

        close(iunit)
    end subroutine

    subroutine read_params_set_mode(filename)
        !! Reads the set mode namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_set_mode, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="set_mode", &
                               iunit=iunit)

        close(iunit)
    end subroutine

end module
