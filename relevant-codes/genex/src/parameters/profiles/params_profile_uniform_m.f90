module params_profile_uniform_m
    !! Module handling the parameters for the uniform profile
    !! This profile type is currently use only by the neutrals, so the default
    !! container size is n_neut_supported.
    use params_profile_m, only: params_profile_t
    use genex_fortran_env_m, only : GP
    use params_neutrals_m, only: n_neut_supported
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error

    implicit none
    private

    type, public, extends(params_profile_t):: params_profile_uniform_t
        !! Container type to store parameters for the uniform profile
        real(kind=GP), public :: strength = 1.0_GP
        !! Value to be applied uniformly
    end type

    type(params_profile_uniform_t), dimension(n_neut_supported) :: &
                                              params_profile_uniform_dens_array
    !! Params for density profile

    public :: read_params_profile_uniform
    public :: write_params_profile_uniform

    public :: get_params_profile_uniform_dens

contains

    function get_params_profile_uniform_dens(sp) result(par)
        integer, intent(in) :: sp
        type(params_profile_uniform_t) :: par

        call handle_bounds_error(sp, 1, n_neut_supported, __LINE__, __FILE__)
        par = params_profile_uniform_dens_array(sp)
    end function

    subroutine write_params_profile_uniform(filename, sp)
        !! Writes the profile namelist for species sp to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to
        integer, intent(in) :: sp
        !! Species index

        integer :: iunit, io_error
        real(kind=GP) :: strength

        namelist / params_profile_uniform_dens / strength

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        ! Write density namelist
        strength = params_profile_uniform_dens_array(sp)%strength
        write(iunit, nml=params_profile_uniform_dens, iostat=io_error)
        call handle_write_error(filename, io_error, &
                                nml_name="profile_uniform_dens")

        close(iunit)
    end subroutine

    subroutine read_params_profile_uniform(filename, sp)
        !! Reads the profile namelist for species sp from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from
        integer, intent(in) :: sp
        !! Species index

        integer :: iunit, io_error
        real(kind=GP) :: strength

        namelist / params_profile_uniform_dens / strength

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        ! Initialize temporaries with default values from params array
        strength = params_profile_uniform_dens_array(sp)%strength

        ! Read density namelist
        read(iunit, nml=params_profile_uniform_dens, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="profile_uniform_dens", &
                               iunit=iunit)
        params_profile_uniform_dens_array(sp)%strength = strength

        close(iunit)
    end subroutine

end module
