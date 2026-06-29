module params_source_m
    !! Module handling parameters for the source
    use genex_fortran_env_m, only : GP
    use genex_status_codes_m, only: GENEX_ERR_PARAMETERS
    use genex_error_handling_m, only: handle_error, error_info_t
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error

    implicit none
    private

    integer, parameter, public :: n_source_type_supported = 4
    !! Maximum number of source type supported, whith type specified
    !! by source_type

    character(len=16), dimension(n_source_type_supported) :: source_type = &
                               [character(16) :: "none", "none", "none", "none"]
    !! Type of source operator to be used.
    !! Current options are: "lapd", "dens", "heat", "torque", "none"

    integer, parameter, public :: n_source_loc_supported = 4
    !! Maximum number of localized source supported per type

    type, public :: params_source_t
        !! Base type for parameter type of source
    end type

    type, public, extends(params_source_t) :: params_source_loc_t
        !! Parameter type of localized source
        real(kind=GP), dimension(n_source_loc_supported) :: temp = &
            [1.0_GP, 1.0_GP, 1.0_GP, 1.0_GP]
        !! Reference temperature of the density source setting the width of the
        !! source in velocity-space
        real(kind=GP), dimension(n_source_loc_supported):: rho_mid = &
            [0.95_GP, 0.95_GP, 0.95_GP, 0.95_GP]
        !! Flux surface label of the center of the localized source
        real(kind=GP), dimension(n_source_loc_supported) :: width = &
            [0.01_GP, 0.01_GP, 0.01_GP, 0.01_GP]
        !! Width of the localized source
        real(kind=GP), dimension(n_source_loc_supported) :: amp = &
            [1.0_GP, 0.0_GP, 0.0_GP, 0.0_GP]
        !! Amplitude of the localized source. Units should be given according
        !! to the source type:
        !!     dens:     n_ref / s
        !!     heat:     P_ref
        !!     torque:   n_ref keV/m
        logical, dimension(n_source_loc_supported) :: is_pure = &
            [.false., .false., .false., .false.]
        !! Flag to set the sources to pure, meaning that the considered source
        !! does not affect other conservation laws. For instance, if the density
        !! source is pure, it injects particles at constant energy.
        !! NOTE: Only affects density source for the moment.
    end type

    namelist / params_source / source_type

    public :: read_params_source
    public :: write_params_source

    public :: get_source_type
    public :: get_n_source

contains

    character(len=16) function get_source_type(s)
        !! Returns the source type of the source s.
        integer, intent(in) :: s
        call handle_bounds_error(s, 1, n_source_type_supported, &
                                 __LINE__, __FILE__)
        get_source_type = source_type(s)
    end function

    integer function get_n_source()
        !! Returns the number of sources which are not 'none'
        integer :: s
        get_n_source = 0
        do s = 1, n_source_type_supported
            if(get_source_type(s) == 'lapd' &
               .or. get_source_type(s) == 'dens' &
               .or. get_source_type(s) == 'torque' &
               .or. get_source_type(s) == 'heat') then
                get_n_source = get_n_source + 1
            elseif(.not. get_source_type(s) == 'none') then
                call handle_error(&
                          "Selected source type is not supported!", &
                          GENEX_ERR_PARAMETERS, __LINE__, __FILE__, &
                          additional_info=&
                          error_info_t("Source type was: "//get_source_type(s)))
            endif
        enddo
    end function

    subroutine write_params_source(filename)
        !! Writes the source namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_source, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="source")

        close(iunit)
    end subroutine

    subroutine read_params_source(filename)
        !! Reads the source namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_source, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="source", &
                               iunit=iunit)

        close(iunit)
    end subroutine

end module
