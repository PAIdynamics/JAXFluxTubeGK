module params_source_heat_m
    !! Module handling parameters for the heat source
    use genex_fortran_env_m, only : GP
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error
    use params_source_m, only: params_source_loc_t, &
                               n_source_loc_supported
    use params_species_m, only: n_spec_supported

    implicit none
    private

    type(params_source_loc_t), dimension(n_spec_supported) :: &
                                                  params_source_heat_array
    !! Params for heat source

    real(kind=GP), dimension(n_source_loc_supported):: temp, rho_mid, &
                                                       width, amp
    !! Reference temperatures, flux surface labels of the center, widths, and
    !! amplitudes of the localized heat sources

    logical, dimension(n_source_loc_supported) :: is_pure
    !! Indicates if the localized heat source is a pure source
    !! NOTE: Not required for the heat sources. With the current
    !!       implementation the heat source is pure by default.

    namelist / params_source_heat / temp, rho_mid, width, amp, is_pure
    !! Namelist of the localized heat source parameters

    public :: read_params_source_heat
    public :: write_params_source_heat

    public :: get_params_source_heat
    public :: get_n_source_loc_heat

contains

    function get_params_source_heat(n) result(par)
        !! Returns the params of heat source of the species with index n
        integer, intent(in) :: n
        !! Species index
        type(params_source_loc_t) :: par

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        par = params_source_heat_array(n)
    end function

    function get_n_source_loc_heat(n) result(n_source_loc)
        !! Returns the number of nonzero (with amp > 0) heat source of
        !! species with index n
        integer, intent(in) :: n
        !! Species index
        type(params_source_loc_t) :: params
        integer :: s, n_source_loc

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        params = params_source_heat_array(n)

        n_source_loc = 0
        do s = 1, n_source_loc_supported
            if(params%amp(s) > 0.0_GP) &
                n_source_loc = n_source_loc + 1
        enddo
    end function

    subroutine write_params_source_heat(filename, n)
        !! Writes the heat source namelist for species n to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to
        integer, intent(in) :: n
        !! Species index

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        temp    = params_source_heat_array(n)%temp
        rho_mid = params_source_heat_array(n)%rho_mid
        width   = params_source_heat_array(n)%width
        amp     = params_source_heat_array(n)%amp
        is_pure = params_source_heat_array(n)%is_pure

        write(iunit, nml=params_source_heat, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="source_heat")

        close(iunit)
    end subroutine

    subroutine read_params_source_heat(filename, n)
        !! Reads the heat source namelist for species n
        !! from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from
        integer, intent(in) :: n
        !! Species index

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        ! Initialize temporaries with default values from params array
        temp    = params_source_heat_array(n)%temp
        rho_mid = params_source_heat_array(n)%rho_mid
        width   = params_source_heat_array(n)%width
        amp     = params_source_heat_array(n)%amp
        is_pure = params_source_heat_array(n)%is_pure

        ! Read source heat namelist
        read(iunit, nml=params_source_heat, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="source_heat", &
                               iunit=iunit)

        params_source_heat_array(n)%temp    = temp
        params_source_heat_array(n)%rho_mid = rho_mid
        params_source_heat_array(n)%width   = width
        params_source_heat_array(n)%amp     = amp
        params_source_heat_array(n)%is_pure = is_pure

        close(iunit)
    end subroutine

end module
