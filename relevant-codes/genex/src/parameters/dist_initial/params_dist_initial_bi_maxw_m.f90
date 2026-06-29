module params_dist_initial_bi_maxw_m
    !! Module handling the parameters for dist initial bi-maxwellian
    use genex_fortran_env_m, only: GP
    use params_dist_initial_m, only: params_dist_initial_t
    use params_species_m, only: n_spec_supported
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error

    implicit none
    private

    type, public, extends(params_dist_initial_t) :: &
                                                   params_dist_initial_bi_maxw_t
        !! Container type to store parameters for the bi maxwellian dist init
        real(kind=GP), public :: drift_vpar = 0.0_GP
        !! Parallel drift velocity (shift of the bi-maxw center)
        real(kind=GP), public :: tau_par = 1.0_GP
        !! This factor is multiplied on the parallel temperature upon evaluation
        !! of the distribution.
        !! Can be used together with isotropic temperature profile to
        !! create par and perp temperatures from a single temperature profile
        !! which differ by a constant factor only.
        real(kind=GP), public :: tau_perp = 1.0_GP
        !! This factor is multiplied on the perpendicular temperature upon
        !! evaluation of the distribution.
        !! Can be used together with isotropic temperature profile to
        !! create par and perp temperatures from a single temperature profile
        !! which differ by a constant factor only.
    end type

    type(params_dist_initial_bi_maxw_t), dimension(n_spec_supported) :: &
                                        params_dist_initial_bi_maxw_array

    public :: read_params_dist_initial_bi_maxw
    public :: write_params_dist_initial_bi_maxw

    public :: get_params_dist_initial_bi_maxw

contains

    function get_params_dist_initial_bi_maxw(n) result(par)
        integer, intent(in) :: n
        type(params_dist_initial_bi_maxw_t) :: par

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        par = params_dist_initial_bi_maxw_array(n)
    end function

    subroutine write_params_dist_initial_bi_maxw(filename, n)
        !! Writes the bi-maxw namelist to the given filename
        character(len=*), intent(in):: filename
        !! Filename to write to
        integer, intent(in) :: n
        !! Species index

        integer :: iunit, io_error

        real(kind=GP) :: drift_vpar, tau_par, tau_perp

        namelist / params_dist_initial_bi_maxw / &
            drift_vpar, tau_par, tau_perp

        drift_vpar = params_dist_initial_bi_maxw_array(n)%drift_vpar
        tau_par    = params_dist_initial_bi_maxw_array(n)%tau_par
        tau_perp   = params_dist_initial_bi_maxw_array(n)%tau_perp

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_dist_initial_bi_maxw, iostat=io_error)
        call handle_write_error(filename, io_error, &
                                nml_name="dist_initial_bi_maxw")

        close(iunit)
    end subroutine

    subroutine read_params_dist_initial_bi_maxw(filename, n)
        !! Reads the bi-maxw namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from
        integer, intent(in) :: n
        !! Species index

        integer :: iunit, io_error

        real(kind=GP) :: drift_vpar, tau_par, tau_perp

        namelist / params_dist_initial_bi_maxw / &
            drift_vpar, tau_par, tau_perp

        ! Initialize temporaries with default values from params array
        drift_vpar = params_dist_initial_bi_maxw_array(n)%drift_vpar
        tau_par    = params_dist_initial_bi_maxw_array(n)%tau_par
        tau_perp   = params_dist_initial_bi_maxw_array(n)%tau_perp

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_dist_initial_bi_maxw, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="dist_initial_bi_maxw", &
                               iunit=iunit)

        params_dist_initial_bi_maxw_array(n)%drift_vpar = drift_vpar
        params_dist_initial_bi_maxw_array(n)%tau_par    = tau_par
        params_dist_initial_bi_maxw_array(n)%tau_perp   = tau_perp

        close(iunit)
    end subroutine

end module
