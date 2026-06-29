module params_species_m
    !! Module handling the parameters for the different species
    use genex_fortran_env_m, only: GP
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error
    use interop_params_m, only: set_cxx_params_species

    implicit none
    private

    integer, parameter, public :: n_spec_supported = 8
    !! Maximum number of supported species. We only support a limited number of
    !! species to allow to treat the data in params_species_m as default
    !! initializable parameters. The number of supported species can be
    !! increased if needed. We chose 8 because the masses and charges array,
    !! when stored as single precision, require 64Byte of memory filling one
    !! cache line on many systems.

    real(kind=GP), dimension(n_spec_supported) :: masses = &
        [1.0_GP, 0.5_GP, 1.0_GP, 1.0_GP, 1.0_GP, 1.0_GP, 1.0_GP, 1.0_GP]
    !! Masses of the different species
    real(kind=GP), dimension(n_spec_supported) :: charges = &
        [1.0_GP, -1.0_GP, 1.0_GP, 1.0_GP, 1.0_GP, 1.0_GP, 1.0_GP, 1.0_GP]
    !! Charges of the different species
    real(kind=GP), dimension(n_spec_supported) :: temp_scalings = &
        [1.0_GP, 1.0_GP, 1.0_GP, 1.0_GP, 1.0_GP, 1.0_GP, 1.0_GP, 1.0_GP]
    !! Scaling factor for temperature
    character(len=24), dimension(n_spec_supported) :: names = &
        [character(len=24) :: "protons", "electrons", "", "", "", "", "", ""]
    !! Names of the different species

    namelist / params_species / masses, charges, temp_scalings, names

    public :: read_params_species
    public :: write_params_species

    public :: get_mass
    public :: get_charge
    public :: get_temp_scaling
    public :: get_name
    public :: get_is_electrons
contains

    real(kind=GP) function get_mass(sp)
        !! Returns the mass of the species with index sp
        integer, intent(in) :: sp
        !! Index of the species
        call handle_bounds_error(sp, 1, n_spec_supported, __LINE__, __FILE__)
        get_mass = masses(sp)
    end function

    real(kind=GP) function get_charge(sp)
        !! Returns the charge of the species with index sp
        integer, intent(in) :: sp
        !! Index of the species
        call handle_bounds_error(sp, 1, n_spec_supported, __LINE__, __FILE__)
        get_charge = charges(sp)
    end function

    real(kind=GP) function get_temp_scaling(sp)
        !! Returns the temperature scaling of the species with index sp
        integer, intent(in) :: sp
        !! Index of the species
        call handle_bounds_error(sp, 1, n_spec_supported, __LINE__, __FILE__)
        get_temp_scaling = temp_scalings(sp)
    end function

    character(len=24) function get_name(sp)
        !! Returns the name of the species with index sp
        integer, intent(in) :: sp
        !! Index of the species
        call handle_bounds_error(sp, 1, n_spec_supported, __LINE__, __FILE__)
        get_name = names(sp)
    end function

    logical function get_is_electrons(sp)
        !! Returns true if the species with index sp is electrons
        integer, intent(in) :: sp
        !! Index of the species
        call handle_bounds_error(sp, 1, n_spec_supported, __LINE__, __FILE__)
        get_is_electrons = (charges(sp) == -1.0_GP)
    end function

    subroutine write_params_species(filename)
        !! Writes the species namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_species, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="species")

        close(iunit)
    end subroutine

    subroutine read_params_species(filename)
        !! Reads the species namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)
        read(iunit, nml=params_species, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="species", &
                               iunit=iunit)

        close(iunit)

        call set_cxx_params_species()
    end subroutine

end module
