module params_species_config_m
    !! Module handling the configurations for the different species.
    !! Configuration means the parameters that specify the initial distribution
    !! and the different profiles for density and temperature of a species.
    use genex_fortran_env_m, only: GP
    use params_species_m, only: n_spec_supported
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error

    implicit none
    private

    type, public :: params_species_config_t
        !! Container type to store parameters for a species initialization
        character(len=24), public :: dist_initial_type = "maxw"
        !! Type of the initial distribution.
        !! Current possibilities: maxw, double_maxw, ring,
        !! slowing_down, bi_maxw
        logical, public :: is_isotropic = .true.
        !! Indicates if distribution is isotropic or not
        character(len=24), public :: profile_type_dens = "uniform"
        !! Type of density profile
        !! Current possibilities: uniform, cbc, sine, lapd
        character(len=24), public :: profile_type_temp = "uniform"
        !! Type of temperature profile
        !! Current possibilities: uniform, cbc, sine, lapd
        character(len=24), public :: profile_type_temp_par = "uniform"
        !! Type of parallel temperature profile
        !! Current possibilities: uniform, cbc, sine, lapd
        character(len=24), public :: profile_type_temp_perp = "uniform"
        !! Type of perpendicular temperature profile
        !! Current possibilities: uniform, cbc, sine, lapd
    end type

    type(params_species_config_t), &
        dimension(n_spec_supported) :: params_species_config_array
    !! Configurations of the different species

    public :: read_params_species_config
    public :: write_params_species_config

    public :: get_params_species_config

contains

    function get_params_species_config(n) result(par)
        integer, intent(in) :: n
        type(params_species_config_t) :: par

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        par = params_species_config_array(n)
    end function

    subroutine write_params_species_config(filename, n)
        !! Writes the species namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to
        integer, intent(in) :: n
        !! Species index

        integer :: iunit, io_error

        logical :: is_isotropic
        character(len=24) :: profile_type_dens, profile_type_temp, &
                             profile_type_temp_par, profile_type_temp_perp, &
                             dist_initial_type

        namelist / params_species_config / &
            dist_initial_type, is_isotropic, profile_type_dens, &
            profile_type_temp, profile_type_temp_par, profile_type_temp_perp

        ! Get parameter to write
        associate(params => params_species_config_array(n))
            dist_initial_type      = params%dist_initial_type
            is_isotropic           = params%is_isotropic
            profile_type_dens      = params%profile_type_dens
            profile_type_temp      = params%profile_type_temp
            profile_type_temp_par  = params%profile_type_temp_par
            profile_type_temp_perp = params%profile_type_temp_perp
        end associate

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_species_config, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="species_config")

        close(iunit)
    end subroutine

    subroutine read_params_species_config(filename, n)
        !! Reads the species namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from
        integer, intent(in) :: n
        !! Species index

        integer :: iunit, io_error

        logical :: is_isotropic
        character(len=24) :: profile_type_dens, profile_type_temp, &
                               profile_type_temp_par, profile_type_temp_perp, &
                               dist_initial_type

        namelist / params_species_config / &
            dist_initial_type, is_isotropic, profile_type_dens, &
            profile_type_temp, profile_type_temp_par, profile_type_temp_perp

        ! Initialize temporaries with default values from params array
        ! This is needed because we use temporaries to fill the array, the
        ! namelist variables will be overwritten even if there is no content
        ! inside. This will not happen, however, if they are pre-initialized.
        ! In the case of no input, the defaults will simply be transferred
        ! back to the array after the read.
        associate(params => params_species_config_array(n))
            dist_initial_type      = params%dist_initial_type
            is_isotropic           = params%is_isotropic
            profile_type_dens      = params%profile_type_dens
            profile_type_temp      = params%profile_type_temp
            profile_type_temp_par  = params%profile_type_temp_par
            profile_type_temp_perp = params%profile_type_temp_perp
        end associate

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_species_config, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="params_species_config", iunit=iunit)

        close(iunit)

        associate(params => params_species_config_array(n))
            params%is_isotropic           = is_isotropic
            params%dist_initial_type      = trim(dist_initial_type)
            params%profile_type_dens      = trim(profile_type_dens)
            params%profile_type_temp      = trim(profile_type_temp)
            params%profile_type_temp_par  = trim(profile_type_temp_par)
            params%profile_type_temp_perp = trim(profile_type_temp_perp)
        end associate

    end subroutine

end module
