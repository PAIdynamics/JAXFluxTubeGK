module params_neutrals_m
    !! Module handling the parameters for the different neutrals species
    use genex_fortran_env_m, only: GP
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error

    implicit none
    private

    integer, parameter, public :: n_neut_supported = 2
    !! Maximum number of supported neutral species (can be increased if needed)
    integer :: n_neut = 1
    !! Number of neutral species
    real(kind=GP), dimension(n_neut_supported) :: masses = &
        [1.0_GP, 1.0_GP]
    !! Masses of the different species
    character(len=24), dimension(n_neut_supported) :: names = &
        [character(len=24) :: "hydrogen", ""]
    !! Names of the different species
    character(len=24), dimension(n_neut_supported) :: mapped_ions_names = &
        [character(len=24) :: "protons", ""]
    !! Names of the corresponding ions

    namelist / params_neutrals / masses, names, mapped_ions_names

    public :: read_params_neutrals
    public :: write_params_neutrals

    public :: get_neut_mass
    public :: get_neut_name
    public :: get_neut_mapped_ion_name
    public :: get_n_points_neut

contains

    real(kind=GP) function get_neut_mass(sp)
        !! Returns the mass of the neutral species
        integer, intent(in) :: sp
        !! Index of the species
        call handle_bounds_error(sp, 1, n_neut_supported, __LINE__, __FILE__)
        get_neut_mass = masses(sp)
    end function

    character(len=24) function get_neut_name(sp)
        !! Returns the name of the neutral species
        integer, intent(in) :: sp
        !! Index of the species
        call handle_bounds_error(sp, 1, n_neut_supported, __LINE__, __FILE__)
        get_neut_name = names(sp)
    end function

    character(len=24) function get_neut_mapped_ion_name(sp)
        !! Returns the name of the corresponding ion of the neutral species
        integer, intent(in) :: sp
        !! Index of the species
        call handle_bounds_error(sp, 1, n_neut_supported, __LINE__, __FILE__)
        get_neut_mapped_ion_name = mapped_ions_names(sp)
    end function

    integer function get_n_points_neut()
        !! Returns the numbers of neutral species
        get_n_points_neut = n_neut
    end function

    subroutine write_params_neutrals(filename)
        !! Writes the neutral species namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_neutrals, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="neutrals")

        close(iunit)
    end subroutine

    subroutine read_params_neutrals(filename)
        !! Reads the neutral species namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error
        integer :: n_sp, o

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)
        read(iunit, nml=params_neutrals, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="neutrals", &
                               iunit=iunit)
        close(iunit)

        ! Set n_neut accordingly to the names read
        n_sp = 0
        do o = 1, n_neut_supported
            if (names(o) /= "" ) n_sp = n_sp + 1
        end do
        n_neut = n_sp
    end subroutine

end module
