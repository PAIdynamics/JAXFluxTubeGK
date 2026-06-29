module params_mms_m
    !! Module handling the parameters for MMS
    use math_m, only: PI
    use genex_fortran_env_m, only : GP
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error

    implicit none
    private

    integer :: npol = 1
    !! Poloidal mode number in MMS test
    integer :: ntor = 1
    !! Toroidal mode number in MMS test
    integer :: nrad = 1
    !! Radial mode number in MMS test
    real(kind=GP) :: omega = 2.0_GP * PI
    !! Frequency in time dependence in MMS test
    real(kind=GP) :: q = 0.854_GP
    !! Safety factor parameter in MMS test
    real(kind=GP) :: shear = 2.184_GP
    !! Shear parameter in MMS test
    real(kind=GP) :: minor_r = 0.365_GP
    !! Minor radius in MMS test
    real(kind=GP) :: ell_ax1 = 1.397e-1_GP
    !! Major axis in rotated ellipse rho in Dommaschk MMS test
    real(kind=GP) :: ell_ax2 = 7.208e-2_GP
    !! Minor axis in rotated ellipse rho in Dommaschk MMS test
    real(kind=GP) :: rho_min = 0.1_GP
    !! Minimum value of rho for MMS tests
    real(kind=GP) :: rho_max = 0.5_GP
    !! Maximum value of rho for MMS tests

    namelist / params_mms / &
        npol, ntor, nrad, omega, q, shear, minor_r, ell_ax1, ell_ax2, rho_min, &
        rho_max

    public :: read_params_mms
    public :: write_params_mms

    public :: get_npol
    public :: get_ntor
    public :: get_nrad
    public :: get_omega
    public :: get_q
    public :: get_shear
    public :: get_minor_r
    public :: get_ell_ax1
    public :: get_ell_ax2
    public :: get_rho_min
    public :: get_rho_max

contains

    pure integer function get_npol()
        !! Returns the MMS poloidal mode number parameter
        get_npol = npol
    end function

    pure integer function get_ntor()
        !! Returns the MMS toroidal mode number parameter
        get_ntor = ntor
    end function

    pure integer function get_nrad()
        !! Returns the MMS radial mode number parameter
        get_nrad = nrad
    end function

    pure real(kind=GP) function get_omega()
        !! Returns the MMS omega parameter
        get_omega = omega
    end function

    pure real(kind=GP) function get_q()
        !! Returns the MMS q parameter
        get_q = q
    end function

    pure real(kind=GP) function get_shear()
        !! Returns the MMS shear parameter
        get_shear = shear
    end function

    pure real(kind=GP) function get_minor_r()
        !! Returns the MMS minor_r parameter
        get_minor_r = minor_r
    end function

    pure real(kind=GP) function get_ell_ax1()
        !! Returns the MMS ell_ax1 parameter
        get_ell_ax1 = ell_ax1
    end function

    pure real(kind=GP) function get_ell_ax2()
        !! Returns the MMS ell_ax2 parameter
        get_ell_ax2 = ell_ax2
    end function

    pure real(kind=GP) function get_rho_min()
        !! Returns the MMS rho_min parameter
        get_rho_min = rho_min
    end function

    pure real(kind=GP) function get_rho_max()
        !! Returns the MMS rho_max parameter
        get_rho_max = rho_max
    end function

    subroutine write_params_mms(filename)
        !! Writes the mms namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_mms, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="mms")

        close(iunit)
    end subroutine

    subroutine read_params_mms(filename)
        !! Reads the mms namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_mms, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="mms", iunit=iunit)

        close(iunit)
    end subroutine

end module params_mms_m
