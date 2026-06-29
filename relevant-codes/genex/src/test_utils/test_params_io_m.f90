module test_params_io_m

    use mpi
    use genex_fortran_env_m, only: GP
    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_ERR_UTESTS

    implicit none
    private

    interface write_nml
        procedure :: write_nml_char
        procedure :: write_nml_char_array_1d
        procedure :: write_nml_real
        procedure :: write_nml_real_array_1d
        procedure :: write_nml_real_array_2d
        procedure :: write_nml_real_array_3d
        procedure :: write_nml_integer
        procedure :: write_nml_logical
        procedure :: write_nml_logical_array_1d
    end interface

    character(len=*), parameter :: test_params_file = "test_params"
    ! Filename that should be used to write the parameters used in the test

    character(len=80) :: opened_nml = ""
    ! Keep track of the currently opened namelist when successively writing
    ! parameters.

    public :: write_nml, open_nml, close_nml
    public :: get_test_params_file
    public :: test_params_error

contains

    pure character(len=32) function get_test_params_file()
        !! Return the test parameter file
        get_test_params_file = test_params_file
    end function

    subroutine open_nml(iunit, nml_name)
        ! Write the namelist header
        integer, intent(in) :: iunit
        character(len=*), intent(in) :: nml_name
        integer :: io_error

        if(trim(opened_nml) /= "") call test_params_error()

        write(iunit, *, iostat=io_error) "&"//nml_name
        if(io_error /= 0) call test_params_error()
        opened_nml = nml_name
    end subroutine

    subroutine close_nml(iunit, nml_name)
        ! Write the namelist end symbol
        integer, intent(in) :: iunit
        character(len=*), intent(in) :: nml_name
        integer :: io_error

        if(trim(opened_nml) /= nml_name) call test_params_error()

        write(iunit, *, iostat=io_error) "/"
        if(io_error /= 0) call test_params_error()
        opened_nml = ""
    end subroutine

    subroutine write_nml_char(iunit, nml_name, param_name, param_value)
        ! Write a character parameter to the namelist
        integer, intent(in) :: iunit
        character(len=*), intent(in) :: nml_name, param_name, param_value
        integer :: io_error

        if(trim(opened_nml) /= nml_name) call test_params_error()

        write(iunit, *, iostat=io_error) param_name//"="""//param_value//""","
        if(io_error /= 0) call test_params_error()
    end subroutine

    subroutine write_nml_char_array_1d(iunit, nml_name, param_name, param_value)
        ! Write an array of character parameters to the namelist
        integer, intent(in) :: iunit
        character(len=*), intent(in) :: nml_name, param_name
        character(len=*), dimension(:), intent(in) :: param_value
        integer :: io_error, n
        character(len=128) :: values_str

        if(trim(opened_nml) /= nml_name) call test_params_error()

        values_str = ""
        do n = 1, size(param_value)
            values_str = trim(values_str)//' "'//trim(param_value(n))// '"'
        end do
        values_str = param_name//"="//trim(values_str)//","

        write(iunit, *, iostat=io_error) values_str
        if(io_error /= 0) call test_params_error()
    end subroutine

    subroutine write_nml_real(iunit, nml_name, param_name, param_value)
        ! Write a real parameter to the namelist
        integer, intent(in) :: iunit
        character(len=*), intent(in) :: nml_name, param_name
        real(kind=GP), intent(in) :: param_value
        integer :: io_error

        if(trim(opened_nml) /= nml_name) call test_params_error()

        write(iunit, *, iostat=io_error) param_name//"=", param_value, ","
        if(io_error /= 0) call test_params_error()
    end subroutine

    subroutine write_nml_real_array_1d(iunit, nml_name, param_name, param_value)
        ! Write a 1D array of real parameters to the namelist
        integer, intent(in) :: iunit
        character(len=*), intent(in) :: nml_name, param_name
        real(kind=GP), dimension(:), intent(in) :: param_value

        call write_nml_real_array(iunit, nml_name, param_name, param_value, &
                                  size(param_value))
    end subroutine

    subroutine write_nml_real_array_2d(iunit, nml_name, param_name, param_value)
        ! Write a 2D array of real parameters to the namelist
        integer, intent(in) :: iunit
        character(len=*), intent(in) :: nml_name, param_name
        real(kind=GP), dimension(:,:), intent(in) :: param_value

        call write_nml_real_array(iunit, nml_name, param_name, param_value, &
                                  size(param_value))
    end subroutine

    subroutine write_nml_real_array_3d(iunit, nml_name, param_name, param_value)
        ! Write a 3D array of real parameters to the namelist
        integer, intent(in) :: iunit
        character(len=*), intent(in) :: nml_name, param_name
        real(kind=GP), dimension(:,:,:), intent(in) :: param_value

        call write_nml_real_array(iunit, nml_name, param_name, param_value, &
                                  size(param_value))
    end subroutine

    subroutine write_nml_real_array(iunit, nml_name, param_name, param_value, &
                                    array_size)
        ! Write an arbitrary rank array of real parameters to the namelist
        integer, intent(in) :: iunit
        character(len=*), intent(in) :: nml_name, param_name
        real(kind=GP), dimension(*), intent(in) :: param_value
        integer, intent(in) :: array_size
        integer :: io_error

        if(trim(opened_nml) /= nml_name) call test_params_error()

        write(iunit, *, iostat=io_error) param_name//"=", &
                                         param_value(:array_size), ","
        if(io_error /= 0) call test_params_error()
    end subroutine

    subroutine write_nml_integer(iunit, nml_name, param_name, param_value)
        ! Write an integer parameter to the namelist
        integer, intent(in) :: iunit
        character(len=*), intent(in) :: nml_name, param_name
        integer, intent(in) :: param_value
        integer :: io_error

        if(trim(opened_nml) /= nml_name) call test_params_error()

        write(iunit, *, iostat=io_error) param_name//"=", param_value, ","
        if(io_error /= 0) call test_params_error()
    end subroutine

    subroutine write_nml_logical(iunit, nml_name, param_name, param_value)
        ! Write a logical parameter to the namelist
        integer, intent(in) :: iunit
        character(len=*), intent(in) :: nml_name, param_name
        logical, intent(in) :: param_value
        integer :: io_error

        if(trim(opened_nml) /= nml_name) call test_params_error()

        write(iunit, *, iostat=io_error) param_name//"=", param_value,","
        if(io_error /= 0) call test_params_error()
    end subroutine

    subroutine write_nml_logical_array_1d(iunit, nml_name, param_name, &
                                          param_value)
        ! Write a logical parameter to the namelist
        integer, intent(in) :: iunit
        character(len=*), intent(in) :: nml_name, param_name
        logical, dimension(:), intent(in) :: param_value
        integer :: io_error

        if(trim(opened_nml) /= nml_name) call test_params_error()

        write(iunit, *, iostat=io_error) param_name//"=", &
                                         param_value, ","
        if(io_error /= 0) call test_params_error()
    end subroutine

    subroutine test_params_error()
        ! Common subroutine to handle parameter file I/O errors
        call handle_error("Failed to create params for unit test!", &
                          GENEX_ERR_UTESTS, __LINE__, __FILE__)
    end subroutine

end module
