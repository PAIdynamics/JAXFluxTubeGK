module params_bnd_cond_m
    !! Module handling paramters for the boundary conditions
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error

    implicit none
    private

    character(len=16) :: bnd_cond_type = "dirichlet"
    !! Type of boundary condition.
    !! Valid options are: dirichlet, neumann

    logical :: freeze_dens = .false.
    !! Flag to freeze the boundary conditions of
    !! the density. Only used if bnd_cond_type = 'neumann'
    !! and spectral approach.

    logical :: freeze_even_mom = .false.
    !! Flag to freeze the boundary conditions of
    !! the moment with vp even. Only used if bnd_cond_type = 'neumann'
    !! and spectral approach.

    namelist / params_bnd_cond / bnd_cond_type, freeze_dens, freeze_even_mom

    public :: read_params_bnd_cond
    public :: write_params_bnd_cond

    public :: get_bnd_cond_type
    public :: get_freeze_dens
    public :: get_freeze_even_mom

contains

    pure function get_bnd_cond_type() result(res)
        !! Returns the type of boundary condition
        character(len=16) :: res
        res = bnd_cond_type
    end function

    pure function get_freeze_dens() result(res)
        !! Returns freeze density boundary conditions
        logical :: res
        res = freeze_dens
    end function

    pure function get_freeze_even_mom() result(res)
        !! Returns freeze even moment boundary conditions
        logical :: res
        res = freeze_even_mom
    end function

    subroutine write_params_bnd_cond(filename)
        !! Writes the boundary condition namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_bnd_cond, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="bnd_cond")

        close(iunit)
    end subroutine

    subroutine read_params_bnd_cond(filename)
        !! Reads the boundary condition namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read',&
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_bnd_cond, iostat=io_error)
        call handle_read_error(filename, io_error, &
                               nml_name="bnd_cond", iunit=iunit)

        close(iunit)
    end subroutine

end module
