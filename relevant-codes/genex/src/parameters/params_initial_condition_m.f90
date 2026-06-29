module params_initial_condition_m
    !! Module handling the parameters for the initial condition
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error

    implicit none
    private

    character(len=16) :: initial_perturbation = "none"
    !! Name of the initial perturbation of the simulation. Current options are
    !! "noise", "blob", "mode", or "none"
    logical :: use_canonical_maxw = .false.
    !! .true. if a canonical maxwellian is initialized, .false. if a standard
    !! maxwellian is initialized

    namelist / params_initial_condition / &
        initial_perturbation, use_canonical_maxw

    public :: read_params_initial_condition
    public :: write_params_initial_condition

    public :: get_initial_perturbation
    public :: get_use_canonical_maxw

contains

    pure character(len=16) function get_initial_perturbation()
        !! Returns the name of the initial condition of the simulation
        get_initial_perturbation = initial_perturbation
    end function

    pure logical function get_use_canonical_maxw()
        !! Returns .true. if a canonical maxwellian should be initialized and
        !! .false. otherwise.
        get_use_canonical_maxw = use_canonical_maxw
    end function

    subroutine write_params_initial_condition(filename)
        !! Writes the initial condition namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_initial_condition, iostat=io_error)
        call handle_write_error(filename, io_error, &
                                nml_name="initial_condition")

        close(iunit)

    end subroutine

    subroutine read_params_initial_condition(filename)
        !! Reads the initial condition namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_initial_condition, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="initial_condition", iunit=iunit)

        close(iunit)
    end subroutine

end module params_initial_condition_m
