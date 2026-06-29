module params_profile_lapd_m
    !! Module handling the parameters for the LAPD profile
    use params_profile_m, only: params_profile_t
    use genex_fortran_env_m, only : GP
    use params_species_m, only: n_spec_supported
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error

    implicit none
    private

    type, public, extends(params_profile_t) :: params_profile_lapd_t
        !! Container type to store parameters for the lapd profile
        real(kind=GP), public :: cfac = 1.0_GP
        !! C-factor of the profile
        real(kind=GP), public :: prefac = 1.0_GP
        !! Prefactor of the profile
    end type

    type(params_profile_lapd_t), dimension(n_spec_supported) :: &
                                                params_profile_lapd_dens_array
    !! Params for density profile
    type(params_profile_lapd_t), dimension(n_spec_supported) :: &
                                                params_profile_lapd_temp_array
    !! Params for temperature profile

    public :: read_params_profile_lapd
    public :: write_params_profile_lapd

    public :: get_params_profile_lapd_dens
    public :: get_params_profile_lapd_temp

contains

    function get_params_profile_lapd_dens(n) result(par)
        integer, intent(in) :: n
        type(params_profile_lapd_t) :: par

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        par = params_profile_lapd_dens_array(n)
    end function

    function get_params_profile_lapd_temp(n) result(par)
        integer, intent(in) :: n
        type(params_profile_lapd_t) :: par

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        par = params_profile_lapd_temp_array(n)
    end function

    subroutine write_params_profile_lapd(filename, n)
        !! Writes the profile namelist for species n to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to
        integer, intent(in) :: n
        !! Species index

        integer :: iunit, io_error

        real(kind=GP) :: cfac, prefac

        namelist / params_profile_lapd_dens / cfac, prefac
        namelist / params_profile_lapd_temp / cfac, prefac

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        ! Write density namelist
        cfac   = params_profile_lapd_dens_array(n)%cfac
        prefac = params_profile_lapd_dens_array(n)%prefac
        write(iunit, nml=params_profile_lapd_dens, iostat=io_error)
        call handle_write_error(filename, io_error, &
                                nml_name="profile_lapd_dens")

        ! Write temperature namelist
        cfac   = params_profile_lapd_temp_array(n)%cfac
        prefac = params_profile_lapd_temp_array(n)%prefac
        write(iunit, nml=params_profile_lapd_temp, iostat=io_error)
        call handle_write_error(filename, io_error, &
                                nml_name="profile_lapd_temp")

        close(iunit)
    end subroutine

    subroutine read_params_profile_lapd(filename, n)
        !! Reads the profile namelist for species n from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from
        integer, intent(in) :: n
        !! Species index

        integer :: iunit, io_error

        real(kind=GP) :: cfac, prefac

        namelist / params_profile_lapd_dens / cfac, prefac
        namelist / params_profile_lapd_temp / cfac, prefac

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        ! Initialize temporaries with default values from params array
        cfac   = params_profile_lapd_dens_array(n)%cfac
        prefac = params_profile_lapd_dens_array(n)%prefac

        ! Read density namelist
        read(iunit, nml=params_profile_lapd_dens, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="profile_lapd_dens", iunit=iunit)
        params_profile_lapd_dens_array(n)%cfac   = cfac
        params_profile_lapd_dens_array(n)%prefac = prefac

        ! Initialize temporaries with default values from params array
        cfac   = params_profile_lapd_temp_array(n)%cfac
        prefac = params_profile_lapd_temp_array(n)%prefac

        ! Read temperature namelist
        read(iunit, nml=params_profile_lapd_temp, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="profile_lapd_temp", iunit=iunit)
        params_profile_lapd_temp_array(n)%cfac   = cfac
        params_profile_lapd_temp_array(n)%prefac = prefac

        close(iunit)
    end subroutine

end module
