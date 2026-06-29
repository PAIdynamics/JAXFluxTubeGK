module params_profile_sine_m
    !! Module handling the parameters for the sine profile
    use params_profile_m, only: params_profile_t
    use genex_fortran_env_m, only : GP
    use params_species_m, only: n_spec_supported
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error

    implicit none
    private

    type, public, extends(params_profile_t):: params_profile_sine_t
        !! Container type to store parameters for the sine profile
        real(kind=GP), public :: rho_min = 0.0_GP
        !! Initial point of the sine profile. For rho < rho_min the profile has
        !! has a constant value.
        real(kind=GP), public :: rho_max = 1.0_GP
        !! Final point of the sine profile. For rho > rho_max the profile has
        !! has a constant value.
        real(kind=GP), public :: amp_min = 0.0_GP
        !! Amplitude for rho >= rho_max
        real(kind=GP), public :: amp_max = 1.0_GP
        !! Amplitude for rho <= rho_min
    end type

    type(params_profile_sine_t), dimension(n_spec_supported) :: &
                                            params_profile_sine_dens_array
    !! Params for density profile
    type(params_profile_sine_t), dimension(n_spec_supported) :: &
                                            params_profile_sine_temp_array
    !! Params for temperature profile
    type(params_profile_sine_t), dimension(n_spec_supported) :: &
                                            params_profile_sine_temp_par_array
    !! Params for parallel temperature profile
    type(params_profile_sine_t), dimension(n_spec_supported) :: &
                                            params_profile_sine_temp_perp_array
    !! Params for perpendicular temperature profile

    public :: read_params_profile_sine
    public :: write_params_profile_sine

    public :: get_params_profile_sine_dens
    public :: get_params_profile_sine_temp
    public :: get_params_profile_sine_temp_par
    public :: get_params_profile_sine_temp_perp

contains

    function get_params_profile_sine_dens(n) result(par)
        integer, intent(in) :: n
        type(params_profile_sine_t) :: par

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        par = params_profile_sine_dens_array(n)
    end function

    function get_params_profile_sine_temp(n) result(par)
        integer, intent(in) :: n
        type(params_profile_sine_t) :: par

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        par = params_profile_sine_temp_array(n)
    end function

    function get_params_profile_sine_temp_par(n) result(par)
        integer, intent(in) :: n
        type(params_profile_sine_t) :: par

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        par = params_profile_sine_temp_par_array(n)
    end function

    function get_params_profile_sine_temp_perp(n) result(par)
        integer, intent(in) :: n
        type(params_profile_sine_t) :: par

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        par = params_profile_sine_temp_perp_array(n)
    end function

    subroutine write_params_profile_sine(filename, n)
        !! Writes the profile namelist for species n to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to
        integer, intent(in) :: n
        !! Species index

        integer :: iunit, io_error

        real(kind=GP) :: rho_min, rho_max, amp_min, amp_max

        namelist / params_profile_sine_dens      / rho_min, rho_max, &
                                                   amp_min, amp_max
        namelist / params_profile_sine_temp      / rho_min, rho_max, &
                                                   amp_min, amp_max
        namelist / params_profile_sine_temp_par  / rho_min, rho_max, &
                                                   amp_min, amp_max
        namelist / params_profile_sine_temp_perp / rho_min, rho_max, &
                                                   amp_min, amp_max

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        ! Write density namelist
        rho_min = params_profile_sine_dens_array(n)%rho_min
        rho_max = params_profile_sine_dens_array(n)%rho_max
        amp_min = params_profile_sine_dens_array(n)%amp_min
        amp_max = params_profile_sine_dens_array(n)%amp_max
        write(iunit, nml=params_profile_sine_dens, iostat=io_error)
        call handle_write_error(filename, io_error, &
                                nml_name="profile_sine_dens")

        ! Write temperature namelist
        rho_min = params_profile_sine_temp_array(n)%rho_min
        rho_max = params_profile_sine_temp_array(n)%rho_max
        amp_min = params_profile_sine_temp_array(n)%amp_min
        amp_max = params_profile_sine_temp_array(n)%amp_max
        write(iunit, nml=params_profile_sine_temp, iostat=io_error)
        call handle_write_error(filename, io_error, &
                                nml_name="profile_sine_temp")

        ! Write parallel temperature namelist
        rho_min = params_profile_sine_temp_par_array(n)%rho_min
        rho_max = params_profile_sine_temp_par_array(n)%rho_max
        amp_min = params_profile_sine_temp_par_array(n)%amp_min
        amp_max = params_profile_sine_temp_par_array(n)%amp_max
        write(iunit, nml=params_profile_sine_temp_par, iostat=io_error)
        call handle_write_error(filename, io_error, &
                                nml_name="profile_sine_temp_par")

        ! Write perpendicular temperature namelist
        rho_min = params_profile_sine_temp_perp_array(n)%rho_min
        rho_max = params_profile_sine_temp_perp_array(n)%rho_max
        amp_min = params_profile_sine_temp_perp_array(n)%amp_min
        amp_max = params_profile_sine_temp_perp_array(n)%amp_max
        write(iunit, nml=params_profile_sine_temp_perp, iostat=io_error)
        call handle_write_error(filename, io_error, &
                                nml_name="profile_sine_temp_perp")

        close(iunit)
    end subroutine

    subroutine read_params_profile_sine(filename, n)
        !! Reads the profile namelist for species n from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from
        integer, intent(in) :: n
        !! Species index

        integer :: iunit, io_error

        real(kind=GP) :: rho_min, rho_max, amp_min, amp_max

        namelist / params_profile_sine_dens      / rho_min, rho_max, &
                                                   amp_min, amp_max
        namelist / params_profile_sine_temp      / rho_min, rho_max, &
                                                   amp_min, amp_max
        namelist / params_profile_sine_temp_par  / rho_min, rho_max, &
                                                   amp_min, amp_max
        namelist / params_profile_sine_temp_perp / rho_min, rho_max, &
                                                   amp_min, amp_max

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        ! Initialize temporaries with default values from params array
        rho_min = params_profile_sine_dens_array(n)%rho_min
        rho_max = params_profile_sine_dens_array(n)%rho_max
        amp_min = params_profile_sine_dens_array(n)%amp_min
        amp_max = params_profile_sine_dens_array(n)%amp_max

        ! Read density namelist
        read(iunit, nml=params_profile_sine_dens, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="profile_sine_dens", iunit=iunit)
        params_profile_sine_dens_array(n)%rho_min = rho_min
        params_profile_sine_dens_array(n)%rho_max = rho_max
        params_profile_sine_dens_array(n)%amp_min = amp_min
        params_profile_sine_dens_array(n)%amp_max = amp_max

        ! Initialize temporaries with default values from params array
        rho_min = params_profile_sine_temp_array(n)%rho_min
        rho_max = params_profile_sine_temp_array(n)%rho_max
        amp_min = params_profile_sine_temp_array(n)%amp_min
        amp_max = params_profile_sine_temp_array(n)%amp_max

        ! Read temperature namelist
        read(iunit, nml=params_profile_sine_temp, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="profile_sine_temp", iunit=iunit)
        params_profile_sine_temp_array(n)%rho_min = rho_min
        params_profile_sine_temp_array(n)%rho_max = rho_max
        params_profile_sine_temp_array(n)%amp_min = amp_min
        params_profile_sine_temp_array(n)%amp_max = amp_max

        ! Initialize temporaries with default values from params array
        rho_min = params_profile_sine_temp_par_array(n)%rho_min
        rho_max = params_profile_sine_temp_par_array(n)%rho_max
        amp_min = params_profile_sine_temp_par_array(n)%amp_min
        amp_max = params_profile_sine_temp_par_array(n)%amp_max

        ! Read parallel temperature namelist
        read(iunit, nml=params_profile_sine_temp_par, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="profile_sine_temp_par", iunit=iunit)
        params_profile_sine_temp_par_array(n)%rho_min = rho_min
        params_profile_sine_temp_par_array(n)%rho_max = rho_max
        params_profile_sine_temp_par_array(n)%amp_min = amp_min
        params_profile_sine_temp_par_array(n)%amp_max = amp_max

        ! Initialize temporaries with default values from params array
        rho_min = params_profile_sine_temp_perp_array(n)%rho_min
        rho_max = params_profile_sine_temp_perp_array(n)%rho_max
        amp_min = params_profile_sine_temp_perp_array(n)%amp_min
        amp_max = params_profile_sine_temp_perp_array(n)%amp_max

        ! Read perpendicular temperature namelist
        read(iunit, nml=params_profile_sine_temp_perp, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="profile_sine_temp_perp", iunit=iunit)
        params_profile_sine_temp_perp_array(n)%rho_min = rho_min
        params_profile_sine_temp_perp_array(n)%rho_max = rho_max
        params_profile_sine_temp_perp_array(n)%amp_min = amp_min
        params_profile_sine_temp_perp_array(n)%amp_max = amp_max

        close(iunit)
    end subroutine

end module
