module params_neutrals_set_blob_m
    !! Module handling the blob initialization parameters for neutrals species.
    use genex_fortran_env_m, only : GP, GP_EPS
    use params_neutrals_m, only: n_neut_supported
    use error_params_m, only : handle_open_error, handle_read_error, &
                               handle_write_error, handle_bounds_error

    implicit none
    private

    type, public :: params_neut_blob_t
        !! Container to store parameters for the initial neutrals blob
        real(kind=GP) :: mean_R = 1.0_GP
        !! R Position of the blob
        real(kind=GP) :: mean_Z = 0.0_GP
        !! Z Position of the blob
        real(kind=GP) :: mean_phi = 0.0_GP
        !! phi Position of the blob
        real(kind=GP) :: sigma_R = 0.2_GP
        !! Width of the blob in R direction
        real(kind=GP) :: sigma_Z = 0.2_GP
        !! Width of the blob in Z direction
        real(kind=GP) :: sigma_phi = 1.0_GP
        !! Width of the blob in phi direction
        real(kind=GP) :: strength = 1.0_GP
        !! Strength of the blob perturbation
    end type

    type(params_neut_blob_t), dimension(n_neut_supported) :: &
                                                    params_neut_blob_dens_array
    !! Params for neutrals density blob

    public :: read_params_neutrals_set_blob
    public :: write_params_neutrals_set_blob

    public :: get_neut_params_blob_dens

contains

    function get_neut_params_blob_dens(sp) result(par)
        integer, intent(in) :: sp
        type(params_neut_blob_t) :: par

        call handle_bounds_error(sp, 1, n_neut_supported, __LINE__, __FILE__)
        par = params_neut_blob_dens_array(sp)
    end function

    subroutine write_params_neutrals_set_blob(filename, sp)
        !! Writes the neutrals set blob namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to
        integer, intent(in) :: sp
        !! Species index

        integer :: iunit, io_error
        real(kind=GP) :: mean_R, mean_Z, mean_phi, &
                         sigma_R, sigma_Z, sigma_phi, &
                         strength

        namelist / params_neutrals_blob_dens / mean_R, mean_Z, mean_phi, &
                                               sigma_R, sigma_Z, sigma_phi, &
                                               strength

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        ! Write density namelist
        mean_R = params_neut_blob_dens_array(sp)%mean_R
        mean_Z = params_neut_blob_dens_array(sp)%mean_Z
        mean_phi = params_neut_blob_dens_array(sp)%mean_phi
        sigma_R = params_neut_blob_dens_array(sp)%sigma_R
        sigma_Z = params_neut_blob_dens_array(sp)%sigma_Z
        sigma_phi = params_neut_blob_dens_array(sp)%sigma_phi
        strength = params_neut_blob_dens_array(sp)%strength
        write(iunit, nml=params_neutrals_blob_dens, iostat=io_error)
        call handle_write_error(filename, io_error, &
                                nml_name="neutrals_blob_dens")

        close(iunit)
    end subroutine

    subroutine read_params_neutrals_set_blob(filename, sp)
        !! Reads the neutrals set list namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from
        integer, intent(in) :: sp
        !! Species index

        integer :: iunit, io_error
        real(kind=GP) :: mean_R, mean_Z, mean_phi, &
                         sigma_R, sigma_Z, sigma_phi, &
                         strength

        namelist / params_neutrals_blob_dens / mean_R, mean_Z, mean_phi, &
                                               sigma_R, sigma_Z, sigma_phi, &
                                               strength

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        mean_R = params_neut_blob_dens_array(sp)%mean_R
        mean_Z = params_neut_blob_dens_array(sp)%mean_Z
        mean_phi = params_neut_blob_dens_array(sp)%mean_phi
        sigma_R = params_neut_blob_dens_array(sp)%sigma_R
        sigma_Z = params_neut_blob_dens_array(sp)%sigma_Z
        sigma_phi = params_neut_blob_dens_array(sp)%sigma_phi
        strength = params_neut_blob_dens_array(sp)%strength

        read(iunit, nml=params_neutrals_blob_dens, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="neutrals_blob_dens", &
                               iunit=iunit)
        params_neut_blob_dens_array(sp)%mean_R = mean_R
        params_neut_blob_dens_array(sp)%mean_Z = mean_Z
        params_neut_blob_dens_array(sp)%mean_phi = mean_phi
        params_neut_blob_dens_array(sp)%sigma_R = sigma_R
        params_neut_blob_dens_array(sp)%sigma_Z = sigma_Z
        params_neut_blob_dens_array(sp)%sigma_phi = sigma_phi
        params_neut_blob_dens_array(sp)%strength = strength

        close(iunit)
    end subroutine

end module
