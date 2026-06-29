module params_collisions_m
    !! Module handling parameters for collisions
    use genex_fortran_env_m, only : GP, GP_EPS
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error
    use interop_params_m, only: set_cxx_params_collisions

    implicit none
    private

    character(len=16) :: coll_type = "none"
    !! Collision operator to be used. Current options are: "none", "bgk",
    !! "lbd", "lorentz", and "fpl"
    character(len=16) :: relx_type = "em"
    !! Relaxation type used for the collision frequency. Options are either
    !! "et" (entropic, temperature relaxation) or "em" (entropic, momentum
    !! relaxation)
    real(kind=GP) :: dens_floor = GP_EPS
    !! Parameter that sets a minimum limit on the density in the calculations
    !! of collisional moments. If the calculated density is below this value,
    !! it is replaced by it. This is needed because density is used as a
    !! normalization for the other collisional moments.
    real(kind=GP) :: temp_floor = GP_EPS
    !! Parameter that sets a minimum limit on the temperature
    !! in the calculations of collisional moments. This is needed to prevent
    !! possible negative temperatures that produce floating point exceptions
    !! in the sqrt.

    namelist / params_collisions / coll_type, relx_type, dens_floor, temp_floor

    public :: read_params_collisions
    public :: write_params_collisions

    public :: get_coll_type
    public :: get_relx_type
    public :: get_dens_floor
    public :: get_temp_floor

contains

    pure function get_coll_type() result(res)
        !! Returns collision type of the simulation
        character(len=16) :: res
        res = coll_type
    end function

    pure function get_relx_type() result(res)
        !! Returns relaxation type of the simulation
        character(len=16) :: res
        res = relx_type
    end function

    pure real(kind=GP) function get_dens_floor()
        !! Returns density floor parameter
        get_dens_floor = dens_floor
    end function

    pure real(kind=GP) function get_temp_floor()
        !! Returns temperature floor parameter
        get_temp_floor = temp_floor
    end function

    subroutine write_params_collisions(filename)
        !! Writes the collision namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_collisions, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="collisions")

        close(iunit)
    end subroutine

    subroutine read_params_collisions(filename)
        !! Reads the collision namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read', &
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_collisions, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="collisions", &
                               iunit=iunit)

        close(iunit)

        call set_cxx_params_collisions()
    end subroutine

end module
