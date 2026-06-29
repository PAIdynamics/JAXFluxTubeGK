module params_source_lapd_m
    !! Module handling parameters for the LAPD source
    use genex_fortran_env_m, only : GP
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error

    implicit none
    private

    real(kind=GP) :: extent = 0.95_GP
    !! Radial extent of the source region
    real(kind=GP) :: width = 0.01_GP
    !! Radial fall of length
    real(kind=GP) :: strength = 1.0_GP
    !! Strength of the source
    real(kind=GP) :: c_Te = 1.0_GP / 2.5_GP
    !! c parameter of the electron source temperature profile
    real(kind=GP) :: prefac_Te = 6.8_GP
    !! Prefactor of the electrion source temperature profile
    real(kind=GP) :: prefac_Ti = 1.0_GP
    !! Prefactor of the ion source temperature profile

    public :: read_params_source_lapd
    public :: write_params_source_lapd

    public :: get_extent
    public :: get_width
    public :: get_strength
    public :: get_c_Te
    public :: get_prefac_Te
    public :: get_prefac_Ti

contains

    pure real(kind=GP) function get_extent()
        !! Returns the radial extent of the source
        get_extent = extent
    end function

    pure real(kind=GP) function get_width()
        !! Returns the radial fall of length of the source
        get_width = width
    end function

    pure real(kind=GP) function get_strength()
        !! Returns the strength of the source
        get_strength = strength
    end function

    pure real(kind=GP) function get_c_Te()
        !! Returns the c parameter of the electron source temperature profile
        get_c_Te = c_Te
    end function

    pure real(kind=GP) function get_prefac_Te()
        !! Returns the prefactor for the electron source temperature profile
        get_prefac_Te = prefac_Te
    end function

    pure real(kind=GP) function get_prefac_Ti()
        !! Returns the prefactor for the ion source temperature profile
        get_prefac_Ti = prefac_Ti
    end function

    subroutine write_params_source_lapd(filename)
        !! Writes the LAPD source namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        namelist / params_source_lapd / &
            extent, width, strength, c_Te, prefac_Te, prefac_Ti

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_source_lapd, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="source_lapd")

        close(iunit)
    end subroutine

    subroutine read_params_source_lapd(filename)
        !! Reads the LAPD source namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        namelist / params_source_lapd / &
            extent, width, strength, c_Te, prefac_Te, prefac_Ti

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_source_lapd, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="source_lapd", &
                               iunit=iunit)

        close(iunit)
    end subroutine

end module
