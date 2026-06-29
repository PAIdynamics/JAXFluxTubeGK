module params_dist_initial_ring_m
    !! Module handling the parameters for dist initial ring
    use genex_fortran_env_m, only: GP
    use params_dist_initial_m, only: params_dist_initial_t
    use params_species_m, only: n_spec_supported
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error

    implicit none
    private

    type, public, extends(params_dist_initial_t) :: params_dist_initial_ring_t
        !! Container type to store params for the ring dist init
        real(kind=GP), public :: drift_par  = 0.0_GP
        !! Parallel drift (shift of the peak in vp)
        real(kind=GP), public :: drift_perp = 0.0_GP
        !! Perpendicular drift (shift of the peak in muB)
        real(kind=GP), public :: width_par  = 1.0_GP
        !! Parallel width
        real(kind=GP), public :: width_perp = 1.0_GP
        !! Perpendicular width
    end type

    type(params_dist_initial_ring_t), dimension(n_spec_supported) :: &
                                                params_dist_initial_ring_array

    public :: read_params_dist_initial_ring
    public :: write_params_dist_initial_ring

    public :: get_params_dist_initial_ring

contains

    function get_params_dist_initial_ring(n) result(par)
        integer, intent(in) :: n
        type(params_dist_initial_ring_t) :: par

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        par = params_dist_initial_ring_array(n)
    end function

    subroutine write_params_dist_initial_ring(filename, n)
        !! Writes the ring namelist to the given filename
        character(len=*), intent(in):: filename
        !! Filename to write to
        integer, intent(in) :: n
        !! Species index

        integer :: iunit, io_error

        real(kind=GP) :: drift_par, drift_perp, width_par, width_perp

        namelist / params_dist_initial_ring / &
            drift_par, drift_perp, width_par, width_perp

        drift_par  = params_dist_initial_ring_array(n)%drift_par
        drift_perp = params_dist_initial_ring_array(n)%drift_perp
        width_par  = params_dist_initial_ring_array(n)%width_par
        width_perp = params_dist_initial_ring_array(n)%width_perp

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_dist_initial_ring, iostat=io_error)
        call handle_write_error(filename, io_error, &
                                nml_name="dist_initial_ring")

        close(iunit)
    end subroutine

    subroutine read_params_dist_initial_ring(filename, n)
        !! Reads the ring namelist from the given filename
        character(len=*), intent(in):: filename
        !! Filename to write to
        integer, intent(in) :: n
        !! Species index

        integer :: iunit, io_error

        real(kind=GP) :: drift_par, drift_perp, width_par, width_perp

        namelist / params_dist_initial_ring / &
            drift_par, drift_perp, width_par, width_perp

        ! Initialize temporaries with default values from params array
        drift_par  = params_dist_initial_ring_array(n)%drift_par
        drift_perp = params_dist_initial_ring_array(n)%drift_perp
        width_par  = params_dist_initial_ring_array(n)%width_par
        width_perp = params_dist_initial_ring_array(n)%width_perp

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_dist_initial_ring, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="dist_initial_ring", &
                               iunit=iunit)

        params_dist_initial_ring_array(n)%drift_par  = drift_par
        params_dist_initial_ring_array(n)%drift_perp = drift_perp
        params_dist_initial_ring_array(n)%width_par  = width_par
        params_dist_initial_ring_array(n)%width_perp = width_perp

        close(iunit)
    end subroutine

end module
