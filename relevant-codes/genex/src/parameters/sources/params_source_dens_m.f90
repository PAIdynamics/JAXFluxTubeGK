module params_source_dens_m
    !! Module handling parameters for the density source
    use genex_fortran_env_m, only : GP
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error
    use params_source_m, only: params_source_loc_t, &
                               n_source_loc_supported
    use params_species_m, only: n_spec_supported

    implicit none
    private

    type(params_source_loc_t), dimension(n_spec_supported) :: &
                                                  params_source_dens_array
    !! Params for density source

    real(kind=GP), dimension(n_source_loc_supported):: temp, rho_mid, &
                                                       width, amp
    !! Reference temperatures, flux surface labels of the center, widths, and
    !! amplitudes of the localized density sources

    logical, dimension(n_source_loc_supported) :: is_pure
    !! Indicates if the localized density sources is a pure source
    !! NOTE: If set to .false., the density source also injects
    !!       energy. If set to .true., the density source also
    !!       removes energy to satisfy the energy conservation.

    namelist / params_source_dens / temp, rho_mid, width, amp, is_pure
    !! Namelist of the localized density source parameters

    public :: read_params_source_dens
    public :: write_params_source_dens

    public :: get_params_source_dens
    public :: get_n_source_loc_dens

contains

    function get_params_source_dens(n) result(par)
        !! Returns the params of density source of the species with index n
        integer, intent(in) :: n
        !! Species index
        type(params_source_loc_t) :: par

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        par = params_source_dens_array(n)
    end function

    function get_n_source_loc_dens(n) result(n_source_loc)
        !! Returns the number of nonzero (with amp > 0) density source of
        !! species with index n
        integer, intent(in) :: n
        !! Species index
        type(params_source_loc_t) :: params
        integer :: s, n_source_loc

        call handle_bounds_error(n, 1, n_spec_supported, __LINE__, __FILE__)
        params = params_source_dens_array(n)

        n_source_loc = 0
        do s = 1, n_source_loc_supported
            if(params%amp(s) > 0.0_GP) &
                n_source_loc = n_source_loc + 1
        enddo
    end function

    subroutine write_params_source_dens(filename, n)
        !! Writes the dens source namelist for species n to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to
        integer, intent(in) :: n
        !! Species index

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        temp    = params_source_dens_array(n)%temp
        rho_mid = params_source_dens_array(n)%rho_mid
        width   = params_source_dens_array(n)%width
        amp     = params_source_dens_array(n)%amp
        is_pure = params_source_dens_array(n)%is_pure

        write(iunit, nml=params_source_dens, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="source_dens")

        close(iunit)
    end subroutine

    subroutine read_params_source_dens(filename, n)
        !! Reads the density source namelist for species n
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
        temp    = params_source_dens_array(n)%temp
        rho_mid = params_source_dens_array(n)%rho_mid
        width   = params_source_dens_array(n)%width
        amp     = params_source_dens_array(n)%amp
        is_pure = params_source_dens_array(n)%is_pure

        ! Read source dens namelist
        read(iunit, nml=params_source_dens, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="source_dens", &
                               iunit=iunit)

        params_source_dens_array(n)%temp    = temp
        params_source_dens_array(n)%rho_mid = rho_mid
        params_source_dens_array(n)%width   = width
        params_source_dens_array(n)%amp     = amp
        params_source_dens_array(n)%is_pure = is_pure

        close(iunit)
    end subroutine

end module
