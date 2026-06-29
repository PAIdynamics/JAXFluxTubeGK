module params_diagnostics_m
    !! Module handling the parameters for the diagnostics
    use genex_fortran_env_m, only : GP
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error
    use genex_status_codes_m, only: GENEX_ERR_PARAMETERS
    use genex_error_handling_m, only: handle_error

    implicit none
    private

    integer :: polar_n_theta = 250
    !! Number of points in theta direction in polar diagnostic
    integer :: polar_n_rho = 15
    !! Number of points in rho direction in polar diagnostic
    logical :: diagnose_tpc = .false.
    !! Compute Trapped Particle Contributions (TPC) diagnostics

    namelist / params_diagnostics / &
        polar_n_theta, polar_n_rho, diagnose_tpc

    public :: read_params_diagnostics
    public :: write_params_diagnostics

    public :: get_polar_n_theta
    public :: get_polar_n_rho
    public :: get_diagnose_tpc

contains

    pure integer function get_polar_n_theta()
        !! Returns the number of points in theta direction
        get_polar_n_theta = polar_n_theta
    end function

    pure integer function get_polar_n_rho()
        !! Returns the number of points in rho direction
        get_polar_n_rho = polar_n_rho
    end function

    pure logical function get_diagnose_tpc()
        !! Returns the switch to compute TPC diagnostics
        get_diagnose_tpc = diagnose_tpc
    end function

    subroutine write_params_diagnostics(filename)
        !! Writes the diagnostics namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_diagnostics, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="diagnostics")

        close(iunit)
    end subroutine

    subroutine read_params_diagnostics(filename)
        !! Reads the diagnostics namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read',&
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_diagnostics, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="diagnostics", &
                               iunit=iunit)

        close(iunit)
    end subroutine

end module
