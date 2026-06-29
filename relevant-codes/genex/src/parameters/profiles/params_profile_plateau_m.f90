module params_profile_plateau_m
    !! Module handling the parameters for the sine plateau profile
    !! This profile type is mostly relevant for the neutrals, thus the default
    !! container size is n_neut_supported.
    use params_profile_m, only: params_profile_t
    use genex_fortran_env_m, only : GP
    use params_neutrals_m, only: n_neut_supported
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error, handle_bounds_error

    implicit none
    private

    type, public, extends(params_profile_t):: params_profile_plateau_t
        !! Container type to store parameters for the plateau profile
        real(kind=GP), public :: rho_inner_foot = 0.80_GP
        !! Starting point of the inner step
        real(kind=GP), public :: rho_inner_top = 0.85_GP
        !! End point of the inner step and start of the plateau
        real(kind=GP), public :: rho_outer_top = 1.00_GP
        !! End point of the plateau and start of the outer step
        real(kind=GP), public :: rho_outer_foot = 1.10_GP
        !! End point of the outer step
        real(kind=GP), public :: amp_in = 0.0_GP
        !! Amplitude for rho < rho_inner_foot and thus the value at the inner
        !! boundary
        real(kind=GP), public :: amp_plateau = 1.0_GP
        !! Amplitude at the plateau value and thus give the forcing source term
        !! at the divertor target plates
        real(kind=GP), public :: amp_out = 0.0_GP
        !! Amplitude for rho >= rho_outer_foot and thus the value at the
        !! outer/wall boundary
    end type

    type(params_profile_plateau_t), dimension(n_neut_supported) :: &
                                         params_profile_plateau_dens_array
    !! Params for density profile

    public :: read_params_profile_plateau
    public :: write_params_profile_plateau

    public :: get_params_profile_plateau_dens

contains

    function get_params_profile_plateau_dens(sp) result(par)
        integer, intent(in) :: sp
        type(params_profile_plateau_t) :: par

        call handle_bounds_error(sp, 1, n_neut_supported, __LINE__, __FILE__)
        par = params_profile_plateau_dens_array(sp)
    end function

    subroutine write_params_profile_plateau(filename, sp)
        !! Writes the profile namelist for species sp to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to
        integer, intent(in) :: sp
        !! Species index

        integer :: iunit, io_error

        real(kind=GP) :: rho_inner_foot, rho_inner_top, &
                         rho_outer_top, rho_outer_foot, &
                         amp_in, amp_plateau, amp_out

        namelist / params_profile_plateau_dens / rho_inner_foot, &
                                                 rho_inner_top, &
                                                 rho_outer_top, &
                                                 rho_outer_foot, &
                                                 amp_in, amp_plateau, amp_out

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        ! Write density namelist
        rho_inner_foot = params_profile_plateau_dens_array(sp)%rho_inner_foot
        rho_inner_top = params_profile_plateau_dens_array(sp)%rho_inner_top
        rho_outer_top = params_profile_plateau_dens_array(sp)%rho_outer_top
        rho_outer_foot = params_profile_plateau_dens_array(sp)%rho_outer_foot
        amp_in = params_profile_plateau_dens_array(sp)%amp_in
        amp_plateau = params_profile_plateau_dens_array(sp)%amp_plateau
        amp_out = params_profile_plateau_dens_array(sp)%amp_out
        write(iunit, nml=params_profile_plateau_dens, iostat=io_error)
        call handle_write_error(filename, io_error, &
                                nml_name="profile_plateau_dens")

        close(iunit)
    end subroutine

    subroutine read_params_profile_plateau(filename, sp)
        !! Reads the profile namelist for species sp from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from
        integer, intent(in) :: sp
        !! Species index

        integer :: iunit, io_error

        real(kind=GP) :: rho_inner_foot, rho_inner_top, &
                         rho_outer_top, rho_outer_foot, &
                         amp_in, amp_plateau, amp_out

        namelist / params_profile_plateau_dens / rho_inner_foot, &
                                                 rho_inner_top, &
                                                 rho_outer_top, &
                                                 rho_outer_foot, &
                                                 amp_in, amp_plateau, amp_out

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        ! Initialize temporaries with default values from params array
        rho_inner_foot = params_profile_plateau_dens_array(sp)%rho_inner_foot
        rho_inner_top = params_profile_plateau_dens_array(sp)%rho_inner_top
        rho_outer_top = params_profile_plateau_dens_array(sp)%rho_outer_top
        rho_outer_foot = params_profile_plateau_dens_array(sp)%rho_outer_foot
        amp_in = params_profile_plateau_dens_array(sp)%amp_in
        amp_plateau = params_profile_plateau_dens_array(sp)%amp_plateau
        amp_out = params_profile_plateau_dens_array(sp)%amp_out

        ! Read density namelist
        read(iunit, nml=params_profile_plateau_dens, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="profile_plateau_dens", &
                               iunit=iunit)
        params_profile_plateau_dens_array(sp)%rho_inner_foot = rho_inner_foot
        params_profile_plateau_dens_array(sp)%rho_inner_top = rho_inner_top
        params_profile_plateau_dens_array(sp)%rho_outer_top = rho_outer_top
        params_profile_plateau_dens_array(sp)%rho_outer_foot = rho_outer_foot
        params_profile_plateau_dens_array(sp)%amp_in = amp_in
        params_profile_plateau_dens_array(sp)%amp_plateau = amp_plateau
        params_profile_plateau_dens_array(sp)%amp_out = amp_out

        close(iunit)
    end subroutine

end module
