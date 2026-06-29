module params_neutrals_init_m
    !! Module handling the initialization of neutrals species-specific
    !! parameters
    use genex_fortran_env_m, only : GP, GP_EPS
    use params_neutrals_m, only: n_neut_supported
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error

    implicit none
    private

    type, public :: params_neutrals_init_t
        !! Container type to store parameters for neutrals initialization
        character(len=24), public :: profile_type_dens = "uniform"
        !! Type of density profile
        !! Current possibilities: uniform, plateau_sine
        character(len=24), public :: initial_perturbation_dens = "none"
        !! Type of perturbation applied on top of the density profile
        !! Current possibilities: none, noise, blob
    end type

    type(params_neutrals_init_t), &
        dimension(n_neut_supported) :: params_neutrals_init_array
    !! Initial configurations of the different neutrals species

    character(len=24) :: profile_type_dens, initial_perturbation_dens
    namelist / params_neutrals_init / profile_type_dens, &
                                      initial_perturbation_dens

    public :: read_params_neutrals_init
    public :: write_params_neutrals_init

    public :: get_neut_params_init
    public :: get_neut_profile_type_dens
    public :: get_neut_initial_perturbation_dens

contains

    type(params_neutrals_init_t) function get_neut_params_init(sp)
        !! Returns the initial configuration of the given neutrals species
        integer, intent(in) :: sp
        !! Index of the species
        call handle_bounds_error(sp, 1, n_neut_supported, __LINE__, __FILE__)
        get_neut_params_init = params_neutrals_init_array(sp)
    end function

    character(len=24) function get_neut_profile_type_dens(sp)
        !! Returns the initial density profile of the given neutrals species
        integer, intent(in) :: sp
        !! Index of the species
        call handle_bounds_error(sp, 1, n_neut_supported, __LINE__, __FILE__)
        get_neut_profile_type_dens = &
                               params_neutrals_init_array(sp)%profile_type_dens
    end function

    character(len=24) function get_neut_initial_perturbation_dens(sp)
        !! Returns the initial density perturbation of the given neutrals
        !! species
        integer, intent(in) :: sp
        !! Index of the species
        call handle_bounds_error(sp, 1, n_neut_supported, __LINE__, __FILE__)
        get_neut_initial_perturbation_dens = &
                       params_neutrals_init_array(sp)%initial_perturbation_dens
    end function

    subroutine write_params_neutrals_init(filename, sp)
        !! Writes the neutrals config namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to
        integer, intent(in) :: sp
        !! Index of the species

        integer :: iunit, io_error
        character(len=24) :: profile_type_dens, &
                             initial_perturbation_dens

        namelist / params_neutrals_init / profile_type_dens, &
                                          initial_perturbation_dens

        ! Get parameter to write
        associate(params => params_neutrals_init_array(sp))
            profile_type_dens = params%profile_type_dens
            initial_perturbation_dens = params%initial_perturbation_dens
        end associate

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_neutrals_init, iostat=io_error)
        call handle_write_error(filename, io_error, &
                                nml_name="params_neutrals_init")

        close(iunit)
    end subroutine

    subroutine read_params_neutrals_init(filename, sp)
        !! Reads the neutrals config namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read to
        integer, intent(in) :: sp
        !! Index of the species

        integer :: iunit, io_error
        character(len=24) :: profile_type_dens, &
                             initial_perturbation_dens

        namelist / params_neutrals_init / profile_type_dens, &
                                          initial_perturbation_dens

        ! Initialize temporaries with default values from params array
        ! This is needed because we use temporaries to fill the array, the
        ! namelist variables will be overwritten even if there is no content
        ! inside. This will not happen, however, if they are pre-initialized.
        ! In the case of no input, the defaults will simply be transferred
        ! back to the array after the read.
        associate(params => params_neutrals_init_array(sp))
            profile_type_dens = params%profile_type_dens
            initial_perturbation_dens = params%initial_perturbation_dens
        end associate

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_neutrals_init, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="params_neutrals_init", iunit=iunit)
        close(iunit)

        associate(params => params_neutrals_init_array(sp))
            params%profile_type_dens = trim(profile_type_dens)
            params%initial_perturbation_dens = trim(initial_perturbation_dens)
        end associate
    end subroutine

end module
