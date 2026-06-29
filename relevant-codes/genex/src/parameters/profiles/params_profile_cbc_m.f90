module params_profile_cbc_m
    !! Module handling the parameters for the cbc profile
    use params_profile_m, only: params_profile_t
    use genex_fortran_env_m, only : GP
    use params_species_m, only: n_spec_supported
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error

    implicit none
    private

    type, public, extends(params_profile_t):: params_profile_cbc_t
        !! Container type to store parameters for the cbc profile
        real(kind=GP), public :: kappa  = 1.0_GP
        !! Steepness of the profile
        real(kind=GP), public :: prefac = 1.0_GP
        !! Prefactor of the profile
        real(kind=GP), public :: delta  = 0.04_GP
        !! Width of the profile
        real(kind=GP), public :: center = 0.3_GP
        !! Center of the profile
    end type

    type(params_profile_cbc_t), dimension(n_spec_supported) :: &
                                            params_profile_cbc_dens_array
    !! Params for density profile
    type(params_profile_cbc_t), dimension(n_spec_supported) :: &
                                            params_profile_cbc_temp_array
    !! Params for temperature profile
    type(params_profile_cbc_t), dimension(n_spec_supported) :: &
                                            params_profile_cbc_temp_par_array
    !! Params for parallel temperature profile
    type(params_profile_cbc_t), dimension(n_spec_supported) :: &
                                            params_profile_cbc_temp_perp_array
    !! Params for perpendicular temperature profile

    public :: read_params_profile_cbc
    public :: write_params_profile_cbc

    public :: get_params_profile_cbc_dens
    public :: get_params_profile_cbc_temp
    public :: get_params_profile_cbc_temp_par
    public :: get_params_profile_cbc_temp_perp

contains
    function get_params_profile_cbc_dens(n) result(par)
        integer, intent(in) :: n
        type(params_profile_cbc_t) :: par

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        par = params_profile_cbc_dens_array(n)
    end function

    function get_params_profile_cbc_temp(n) result(par)
        integer, intent(in) :: n
        type(params_profile_cbc_t) :: par

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        par = params_profile_cbc_temp_array(n)
    end function

    function get_params_profile_cbc_temp_par(n) result(par)
        integer, intent(in) :: n
        type(params_profile_cbc_t) :: par

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        par = params_profile_cbc_temp_par_array(n)
    end function

    function get_params_profile_cbc_temp_perp(n) result(par)
        integer, intent(in) :: n
        type(params_profile_cbc_t) :: par

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        par = params_profile_cbc_temp_perp_array(n)
    end function

    subroutine write_params_profile_cbc(filename, n)
        !! Writes the profile namelist for species n to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to
        integer, intent(in) :: n
        !! Species index

        integer :: iunit, io_error

        real(kind=GP) :: kappa, prefac, delta, center

        namelist / params_profile_cbc_dens      / kappa, prefac, delta, center
        namelist / params_profile_cbc_temp      / kappa, prefac, delta, center
        namelist / params_profile_cbc_temp_par  / kappa, prefac, delta, center
        namelist / params_profile_cbc_temp_perp / kappa, prefac, delta, center

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        ! Write density namelist
        kappa  = params_profile_cbc_dens_array(n)%kappa
        prefac = params_profile_cbc_dens_array(n)%prefac
        delta  = params_profile_cbc_dens_array(n)%delta
        center = params_profile_cbc_dens_array(n)%center
        write(iunit, nml=params_profile_cbc_dens, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="profile_cbc_dens")

        ! Write temperature namelist
        kappa  = params_profile_cbc_temp_array(n)%kappa
        prefac = params_profile_cbc_temp_array(n)%prefac
        delta  = params_profile_cbc_temp_array(n)%delta
        center = params_profile_cbc_temp_array(n)%center
        write(iunit, nml=params_profile_cbc_temp, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="profile_cbc_temp")

        ! Write parallel temperature namelist
        kappa  = params_profile_cbc_temp_par_array(n)%kappa
        prefac = params_profile_cbc_temp_par_array(n)%prefac
        delta  = params_profile_cbc_temp_par_array(n)%delta
        center = params_profile_cbc_temp_par_array(n)%center
        write(iunit, nml=params_profile_cbc_temp_par, iostat=io_error)
        call handle_write_error(filename, io_error, &
                                nml_name="profile_cbc_temp_par")

        ! Write perpendicular temperature namelist
        kappa  = params_profile_cbc_temp_perp_array(n)%kappa
        prefac = params_profile_cbc_temp_perp_array(n)%prefac
        delta  = params_profile_cbc_temp_perp_array(n)%delta
        center = params_profile_cbc_temp_perp_array(n)%center
        write(iunit, nml=params_profile_cbc_temp_perp, iostat=io_error)
        call handle_write_error(filename, io_error, &
                                nml_name="profile_cbc_temp_perp")

        close(iunit)
    end subroutine

    subroutine read_params_profile_cbc(filename, n)
        !! Reads the profile namelist for species n from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from
        integer, intent(in) :: n
        !! Species index

        integer :: iunit, io_error

        real(kind=GP) :: kappa, prefac, delta, center

        namelist / params_profile_cbc_dens      / kappa, prefac, delta, center
        namelist / params_profile_cbc_temp      / kappa, prefac, delta, center
        namelist / params_profile_cbc_temp_par  / kappa, prefac, delta, center
        namelist / params_profile_cbc_temp_perp / kappa, prefac, delta, center

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        ! Initialize temporaries with default values from params array
        kappa  = params_profile_cbc_dens_array(n)%kappa
        prefac = params_profile_cbc_dens_array(n)%prefac
        delta  = params_profile_cbc_dens_array(n)%delta
        center = params_profile_cbc_dens_array(n)%center

        ! Read density namelist
        read(iunit, nml=params_profile_cbc_dens, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="profile_cbc_dens", iunit=iunit)
        params_profile_cbc_dens_array(n)%kappa  = kappa
        params_profile_cbc_dens_array(n)%prefac = prefac
        params_profile_cbc_dens_array(n)%delta  = delta
        params_profile_cbc_dens_array(n)%center = center

        ! Initialize temporaries with default values from params array
        kappa  = params_profile_cbc_temp_array(n)%kappa
        prefac = params_profile_cbc_temp_array(n)%prefac
        delta  = params_profile_cbc_temp_array(n)%delta
        center = params_profile_cbc_temp_array(n)%center

        ! Read temperature namelist
        read(iunit, nml=params_profile_cbc_temp, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="profile_cbc_temp", iunit=iunit)
        params_profile_cbc_temp_array(n)%kappa  = kappa
        params_profile_cbc_temp_array(n)%prefac = prefac
        params_profile_cbc_temp_array(n)%delta  = delta
        params_profile_cbc_temp_array(n)%center = center

        ! Initialize temporaries with default values from params array
        kappa  = params_profile_cbc_temp_par_array(n)%kappa
        prefac = params_profile_cbc_temp_par_array(n)%prefac
        delta  = params_profile_cbc_temp_par_array(n)%delta
        center = params_profile_cbc_temp_par_array(n)%center

        ! Read parallel temperature namelist
        read(iunit, nml=params_profile_cbc_temp_par, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="profile_cbc_temp_par", iunit=iunit)
        params_profile_cbc_temp_par_array(n)%kappa  = kappa
        params_profile_cbc_temp_par_array(n)%prefac = prefac
        params_profile_cbc_temp_par_array(n)%delta  = delta
        params_profile_cbc_temp_par_array(n)%center = center

        ! Initialize temporaries with default values from params array
        kappa  = params_profile_cbc_temp_perp_array(n)%kappa
        prefac = params_profile_cbc_temp_perp_array(n)%prefac
        delta  = params_profile_cbc_temp_perp_array(n)%delta
        center = params_profile_cbc_temp_perp_array(n)%center

        ! Read perpendicular temperature namelist
        read(iunit, nml=params_profile_cbc_temp_perp, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="profile_cbc_temp_perp", iunit=iunit)
        params_profile_cbc_temp_perp_array(n)%kappa  = kappa
        params_profile_cbc_temp_perp_array(n)%prefac = prefac
        params_profile_cbc_temp_perp_array(n)%delta  = delta
        params_profile_cbc_temp_perp_array(n)%center = center

        close(iunit)
    end subroutine

end module
