module params_dist_initial_maxw_vspec_m
    !! This module contains the parameters for the
    !! spectral distribution function
    use params_dist_initial_m, only: params_dist_initial_t
    use genex_fortran_env_m, only: GP
    use params_species_m, only: n_spec_supported
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error

    implicit none
    private

    type, public, extends(params_dist_initial_t) :: &
                                             params_dist_initial_maxw_vspec_t
        !! Type to store parameters for the spectral Maxwellian
        real(kind=GP), public :: drift_vpar = 0.0_GP
        !! Parallel drift velocity
    end type

    type(params_dist_initial_maxw_vspec_t), dimension(n_spec_supported) :: &
                                            params_dist_initial_maxw_vspec_array

    public :: read_params_dist_initial_maxw_vspec
    public :: write_params_dist_initial_maxw_vspec

    public :: get_params_dist_initial_maxw_vspec

contains

    function get_params_dist_initial_maxw_vspec(n) result(par)
        integer, intent(in) :: n
        type(params_dist_initial_maxw_vspec_t) :: par

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        par = params_dist_initial_maxw_vspec_array(n)
    end function

    subroutine write_params_dist_initial_maxw_vspec(filename, n)
        !! Write the Maxwellian spectral parameters to the given filename
        character(len=*), intent(in):: filename
        !! Filename to write to
        integer, intent(in) :: n
        !! Species index
        integer :: iunit, io_error
        real(kind=GP) :: drift_vpar

        namelist / params_dist_initial_maxw_vspec / drift_vpar

        drift_vpar = params_dist_initial_maxw_vspec_array(n)%drift_vpar

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_dist_initial_maxw_vspec, iostat=io_error)
        call handle_write_error(filename, io_error, &
                                nml_name="dist_initial_maxw_vspec")

        close(iunit)

    end subroutine

    subroutine read_params_dist_initial_maxw_vspec(filename, n)
        !! Reads the Maxwellian spectral parameters from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from
        integer, intent(in) :: n
        !! Species index
        integer :: iunit, io_error

        real(kind=GP) :: drift_vpar

        namelist / params_dist_initial_maxw_vspec / drift_vpar

        ! Initialize temporaries with default values from params array
        drift_vpar = params_dist_initial_maxw_vspec_array(n)%drift_vpar

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_dist_initial_maxw_vspec, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="dist_initial_maxw_vspec", &
                               iunit=iunit)

        params_dist_initial_maxw_vspec_array(n)%drift_vpar = drift_vpar

        close(iunit)

    end subroutine

end module
