module params_dist_initial_double_maxw_m
    !! Module handling the parameters for dist initial double-maxw
    use genex_fortran_env_m, only: GP
    use params_dist_initial_m, only: params_dist_initial_t
    use params_species_m, only: n_spec_supported
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error

    implicit none
    private

    type, public, extends(params_dist_initial_t) :: &
                                           params_dist_initial_double_maxw_t
        !! Container type to store params for the double maxwellian dist init
        real(kind=GP), public :: drift_par1 = 0.0_GP
        !! Parallel drift of 1st maxwellian (units of vp)
        real(kind=GP), public :: drift_par2 = 0.0_GP
        !! Parallel drift of 2nd maxwellian (units of vp)
        real(kind=GP), public :: amp1 = 1.0_GP
        !! Amplitude of 1st maxwellian
        real(kind=GP), public :: amp2 = 1.0_GP
        !! Amplitude of 2nd maxwellian
    end type

    type(params_dist_initial_double_maxw_t), &
        dimension(n_spec_supported) :: &
            params_dist_initial_double_maxw_array

    public :: read_params_dist_initial_double_maxw
    public :: write_params_dist_initial_double_maxw

    public :: get_params_dist_initial_double_maxw

contains

    function get_params_dist_initial_double_maxw(n) result(par)
        integer, intent(in) :: n
        type(params_dist_initial_double_maxw_t) :: par

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        par = params_dist_initial_double_maxw_array(n)
    end function

    subroutine write_params_dist_initial_double_maxw(filename, n)
        !! Writes the double-maxw namelist to the given filename
        character(len=*), intent(in):: filename
        !! Filename to write to
        integer, intent(in) :: n
        !! Species index

        integer :: iunit, io_error

        real(kind=GP) :: drift_par1, drift_par2, amp1, amp2

        namelist / params_dist_initial_double_maxw / &
            drift_par1, drift_par2, amp1, amp2

        drift_par1  = params_dist_initial_double_maxw_array(n)%drift_par1
        drift_par2  = params_dist_initial_double_maxw_array(n)%drift_par2
        amp1        = params_dist_initial_double_maxw_array(n)%amp1
        amp2        = params_dist_initial_double_maxw_array(n)%amp2

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_dist_initial_double_maxw, iostat=io_error)
        call handle_write_error(filename, io_error, &
                                nml_name="dist_initial_double_maxw")

        close(iunit)
    end subroutine

    subroutine read_params_dist_initial_double_maxw(filename, n)
        !! Reads the bi-maxw namelist from the given filename
        character(len=*), intent(in):: filename
        !! Filename to read from
        integer, intent(in) :: n
        !! Species index

        integer :: iunit, io_error

        real(kind=GP) :: drift_par1, drift_par2, amp1, amp2

        namelist / params_dist_initial_double_maxw / &
            drift_par1, drift_par2, amp1, amp2

        ! Initialize temporaries with default values from params array
        drift_par1  = params_dist_initial_double_maxw_array(n)%drift_par1
        drift_par2  = params_dist_initial_double_maxw_array(n)%drift_par2
        amp1        = params_dist_initial_double_maxw_array(n)%amp1
        amp2        = params_dist_initial_double_maxw_array(n)%amp2

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_dist_initial_double_maxw, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="dist_initial_double_maxw", &
                               iunit=iunit)

        params_dist_initial_double_maxw_array(n)%drift_par1 = drift_par1
        params_dist_initial_double_maxw_array(n)%drift_par2 = drift_par2
        params_dist_initial_double_maxw_array(n)%amp1       = amp1
        params_dist_initial_double_maxw_array(n)%amp2       = amp2

        close(iunit)
    end subroutine

end module
