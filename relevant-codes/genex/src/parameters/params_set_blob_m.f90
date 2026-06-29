module params_set_blob_m
    !! Module handling the parameters for the blob initialization
    use genex_fortran_env_m, only : GP
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error

    implicit none
    private

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
    real(kind=GP) :: mean_vp = 0.3_GP
    !! Mean parallel velocity of the blob
    real(kind=GP) :: strength = 0.1_GP
    !! Strength of the blob perturbation

    namelist / params_set_blob / &
            mean_R, mean_Z, mean_phi, mean_vp, &
            sigma_R, sigma_Z, sigma_phi, strength

    public :: read_params_set_blob
    public :: write_params_set_blob

    public :: get_mean_R
    public :: get_mean_Z
    public :: get_mean_phi
    public :: get_mean_vp
    public :: get_sigma_R
    public :: get_sigma_Z
    public :: get_sigma_phi
    public :: get_strength

contains

    pure real(kind=GP) function get_mean_R()
        !! Returns the R position of the blob
        get_mean_R = mean_R
    end function

    pure real(kind=GP) function get_mean_vp()
        !! Returns the vp position of the blob
        get_mean_vp = mean_vp
    end function

    pure real(kind=GP) function get_mean_Z()
        !! Returns the Z position of the blob
        get_mean_Z = mean_Z
    end function

    pure real(kind=GP) function get_mean_phi()
        !! Returns the phi position of the blob
        get_mean_phi = mean_phi
    end function

    pure real(kind=GP) function get_sigma_R()
        !! Returns the width of the blob in R direction
        get_sigma_R = sigma_R
    end function

    pure real(kind=GP) function get_sigma_Z()
        !! Returns the width of the blob in Z direction
        get_sigma_Z = sigma_Z
    end function

    pure real(kind=GP) function get_sigma_phi()
        !! Returns the width of the blob in phi direction
        get_sigma_phi = sigma_phi
    end function

    pure real(kind=GP) function get_strength()
        !! Returns the strength of blob
        get_strength = strength
    end function

    subroutine write_params_set_blob(filename)
        !! Writes the set blob namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_set_blob, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="set_blob")

        close(iunit)
    end subroutine

    subroutine read_params_set_blob(filename)
        !! Reads the set blob namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_set_blob, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="set_blob", &
                               iunit=iunit)

        close(iunit)
    end subroutine

end module
