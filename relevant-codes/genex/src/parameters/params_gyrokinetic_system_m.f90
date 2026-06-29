module params_gyrokinetic_system_m
    !! Module handling parameters for the gyrokinetic system
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error
    use interop_params_m, only: set_cxx_params_gyrokinetic_system

    implicit none
    private

    ! NOTE: These flags are only useful for specific application cases where
    !       the effect of certain terms need to be investigated. In a standard
    !       run, keep all to true or default.

    logical :: with_rhs = .true.
    !! Flag to activate the rhs of the gyrokinetic system
    logical :: with_em_flutter = .true.
    !! Flag to activate electromagnetic flutter in the gyrokinetic system
    logical :: with_nlin_polarization = .true.
    !! Flag to activate nonlinear polarization in the quasi-neutrality
    !! equation of the gyrokinetic system
    logical :: with_bpar = .false.
    !! Flag to activate the parallel magnetic field fluctuations in the
    !! gyrokinetic system
    character(len=10) :: flr_model = "lwa"
    !! Specifies the type of FLR effects
    !! Currently supported: "lwa"

    namelist / params_gyrokinetic_system / &
        with_rhs, with_em_flutter, with_nlin_polarization, with_bpar, flr_model

    public :: read_params_gyrokinetic_system
    public :: write_params_gyrokinetic_system

    public :: get_with_rhs
    public :: get_with_em_flutter
    public :: get_with_nlin_polarization
    public :: get_with_bpar
    public :: get_flr_model

contains

    pure logical function get_with_rhs()
        !! Returns .true. if the rhs is activated and .false. otherwise
        get_with_rhs = with_rhs
    end function

    pure logical function get_with_em_flutter()
        !! Returns .true. if electromagnetic flutter is activated in the
        !! gyrokinetic system and .false. otherwise
        get_with_em_flutter = with_em_flutter
    end function

    pure logical function get_with_nlin_polarization()
        !! Returns .true. if nonlinear polarization is activated in the
        !! gyrokinetic system and .false. otherwise
        get_with_nlin_polarization = with_nlin_polarization
    end function

    pure logical function get_with_bpar()
        !! Returns .true. if the parallel magnetic field fluctuations are
        !! activated in the gyrokinetic system and .false. otherwise
        get_with_bpar = with_bpar
    end function

    pure character(len=10) function get_flr_model()
        !! Returns the type of specified FLR effects
        get_flr_model = flr_model
    end function

    subroutine write_params_gyrokinetic_system(filename)
        !! Writes the gyrokinetic_system namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_gyrokinetic_system, iostat=io_error)
        call handle_write_error(filename, io_error, &
                                nml_name="gyrokinetic_system")

        close(iunit)
    end subroutine

    subroutine read_params_gyrokinetic_system(filename)
        !! Reads the gyrokinetic_system namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_gyrokinetic_system, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="gyrokinetic_system", iunit=iunit)

        close(iunit)

        call set_cxx_params_gyrokinetic_system()
    end subroutine

end module
