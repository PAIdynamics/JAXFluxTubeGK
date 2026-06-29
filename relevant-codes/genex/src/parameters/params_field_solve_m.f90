module params_field_solve_m
    !! Module handling the parameters for different field solvers
    use genex_fortran_env_m, only : GP
    use error_params_m, only: handle_open_error, handle_read_error, &
                              handle_write_error

    implicit none
    private

    !! Container type to store parameters for the field solvers
    integer :: maxiter = 15
    !! Maximum number of iterations of the GMRES solver
    character(len=16) :: smoother = "GSRB"
    !! Multigrid smoother, current options are JAC, GS and GSRB
    real(kind=GP) :: restol_zero = 1e-12_GP
    !! Threshold for the norm of the right hand side vector in computing
    !! the GMRES residuum. If the norm of the rhs vector is below this
    !! value, restol_zero is additionally added to the norm of the rhs
    !! vector in the denominator of the residuum calculation.

    namelist / params_field_solve / smoother, maxiter, restol_zero

    public :: read_params_field_solve
    public :: write_params_field_solve

    public :: get_maxiter
    public :: get_smoother
    public :: get_restol_zero

contains

    pure integer function get_maxiter()
        !! Returns the maxiter parameter
        get_maxiter = maxiter
    end function

    pure function get_smoother() result(res)
        !! Returns the smoother parameter
        character(len=16) :: res
        res = smoother
    end function

    pure real(kind=GP) function get_restol_zero()
        !! Returns the restol_zero parameter
        get_restol_zero = restol_zero
    end function

    subroutine write_params_field_solve(filename)
        !! Writes the field_solve namelist to the given filename
        character(len=*), intent(in) :: filename
        !! Filename to write to

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='unknown', &
             action='write', position='append', iostat=io_error)
        call handle_open_error(filename, io_error)

        write(iunit, nml=params_field_solve, iostat=io_error)
        call handle_write_error(filename, io_error, nml_name="field_solve")

        close(iunit)
    end subroutine

    subroutine read_params_field_solve(filename)
        !! Reads the field_solve namelist from the given filename
        character(len=*), intent(in) :: filename
        !! Filename to read from

        integer :: iunit, io_error

        open(newunit=iunit, file=filename, status='old', action='read',&
             iostat=io_error)
        call handle_open_error(filename, io_error)

        read(iunit, nml=params_field_solve, iostat=io_error)
        call handle_read_error(filename, io_error, nml_name="field_solve", &
                               iunit=iunit)

        close(iunit)
    end subroutine

end module
