module params_time_split_m
    !! Module handling the parameters for the time substeps in a
    !! splitting algorithm
    use genex_fortran_env_m, only : GP
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error

    implicit none
    private

    character(len=16) :: time_scheme_collisions = "rk4"
    !! Name of the time stepping scheme chosen for collisions.
    !! Current options are: "rk4", "rkc"
    character(len=16) :: time_scheme_neutrals = "rk4"
    !! Name of the time stepping scheme chosen for neutrals.
    !! Current options are: "rk4", "rkc"
    integer :: n_stages_collisions = 2
    !! Number of stages for collisions time scheme. Used if selected scheme
    !! supports this parameter.
    integer :: n_stages_neutrals = 2
    !! Number of stages for neutrals time scheme. Used if selected scheme
    !! supports this parameter.
    real(kind=GP) :: eta_collisions = 2.0_GP / 13.0_GP
    !! Damping factor for collisions. Used if selected scheme supports
    !! this parameter.
    real(kind=GP) :: eta_neutrals = 2.0_GP / 13.0_GP
    !! Damping factor for neutrals. Used if selected scheme supports
    !! this parameter.

    namelist / params_time_split / &
        time_scheme_collisions, time_scheme_neutrals, &
        n_stages_collisions, n_stages_neutrals, &
        eta_collisions, eta_neutrals

    public :: read_params_time_split
    public :: write_params_time_split

    public :: get_time_scheme_collisions, &
              get_time_scheme_neutrals, &
              get_n_stages_collisions, &
              get_n_stages_neutrals, &
              get_eta_collisions, &
              get_eta_neutrals

contains

    character(len=16) function get_time_scheme_collisions()
        !! Returns the time stepping scheme for collisions
        get_time_scheme_collisions = time_scheme_collisions
    end function

    character(len=16) function get_time_scheme_neutrals()
        !! Returns the time stepping scheme for neutrals
        get_time_scheme_neutrals = time_scheme_neutrals
    end function

    integer function get_n_stages_collisions()
        !! Returns the number of stages used for collisions
        get_n_stages_collisions = n_stages_collisions
    end function

    integer function get_n_stages_neutrals()
        !! Returns the number of stages used for neutrals
        get_n_stages_neutrals = n_stages_neutrals
    end function

    real(kind=GP) function get_eta_collisions()
        !! Returns the damping factor for collisions
        get_eta_collisions = eta_collisions
    end function

    real(kind=GP) function get_eta_neutrals()
        !! Returns the damping factor for neutrals
        get_eta_neutrals = eta_neutrals
    end function

    subroutine write_params_time_split(filename)
        !! Writes the time split namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_time_split, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="time_split")

        close(iunit)
    end subroutine

    subroutine read_params_time_split(filename)
        !! Reads the time split namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_time_split, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="time_split", &
                               iunit=iunit)

        close(iunit)
    end subroutine

end module
