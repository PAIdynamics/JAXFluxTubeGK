module error_params_m
    !! Module containing error handling functionality for parameter I/O
    use, intrinsic :: iso_fortran_env, only : IOSTAT_END, IOSTAT_EOR, &
                                              IOSTAT_INQUIRE_INTERNAL_UNIT
    use genex_error_handling_m, only: handle_error, error_info_t
    use genex_status_codes_m, only: GENEX_ERR_PARAMETER_FILE, &
                                    GENEX_ERR_PARAMETERS
    use logger_m, only: logger_get_debug_channel
    use file_handling_m, only: file_exists

    implicit none
    private

    logical, private :: print_info = .false.
    !! Switch that indicates if info messages will be printed or not

    public :: handle_open_error
    public :: handle_read_error
    public :: handle_write_error
    public :: handle_bounds_error

    public :: get_print_info
    public :: set_print_info

contains

    pure logical function get_print_info()
        !! Returns the print_info flag
        get_print_info = print_info
    end function

    subroutine set_print_info(pinfo)
        !! Sets the print_info flag
        logical, intent(in) :: pinfo
        print_info = pinfo
    end subroutine

    subroutine handle_open_error(filename, io_error)
        !! Handles potential errors from opening a file by checking
        !! the error code
        character(len=*), intent(in) :: filename
        !! Name of the file that is opened
        integer, intent(in) :: io_error
        !! Error code returned by I/O routines via iostat variable

        if(io_error == 0) return

        if(.not. file_exists(filename)) then
            call handle_error("Parameter file "//trim(filename) &
                              //" does not exist!", &
                              GENEX_ERR_PARAMETER_FILE, __LINE__, __FILE__)
        else
            call handle_error("Failed to open "//trim(filename) &
                              //" parameter file!", &
                              GENEX_ERR_PARAMETER_FILE, __LINE__, __FILE__)
        endif
    end subroutine

    subroutine handle_read_error(filename, io_error, nml_name, iunit)
        !! Handles potential errors from reading a namelist in a file
        !! by checking the error code
        character(len=*), intent(in) :: filename
        !! Name of the file to read the namelist from
        integer, intent(in) :: io_error
        !! Error code returned by I/O routines via iostat variable
        character(len=*), intent(in) :: nml_name
        !! Name of the namelist to be read
        integer, intent(in) :: iunit
        !! I/O unit of the file to read from

        character(len=128) :: line
        integer :: ierr

        if(io_error == 0) return

        ! END and EOR are returned if namelist can not be found. We treat
        ! these cases as non fatal and write a notification.
        if(io_error == IOSTAT_END .or. io_error == IOSTAT_EOR) then
            if(print_info) then
                write(logger_get_debug_channel(), *) &
                    "Info: Namelist "//trim(nml_name)//" not present."
            endif
            return
        endif

        ! Only in the case of positive io_error which does not indicate
        ! the use of an internal unit, the error is suspected to be within
        ! the namelist itself. In that case print the erroneous line if
        ! possible.
        if(io_error > 0 .and. io_error /= IOSTAT_INQUIRE_INTERNAL_UNIT) then
            backspace(iunit, iostat=ierr)
            if(ierr == 0) read(iunit, fmt='(A)', iostat=ierr) line
            if(ierr == 0) then
                call handle_error("Invalid line in "//trim(nml_name) &
                                  //" namelist: "//trim(line), &
                                  GENEX_ERR_PARAMETER_FILE, __LINE__, __FILE__)
            endif
        else
            call handle_error("Failed to read "//trim(nml_name) &
                              //" namelist!", GENEX_ERR_PARAMETER_FILE, &
                              __LINE__, __FILE__)
        endif

        ! If no known error found, the file is considered corrupt
        call handle_error("Parameter file "//trim(filename)//" is corrupt!", &
                          GENEX_ERR_PARAMETER_FILE, __LINE__, __FILE__)

    end subroutine

    subroutine handle_write_error(filename, io_error, nml_name)
        !! Handles potential errors from writing a namelist in a file
        !! by checking the error code
        character(len=*), intent(in) :: filename
        !! Name of the file to write the namelist to
        integer, intent(in) :: io_error
        !! Error code returned by I/O routines via iostat variable
        character(len=*), intent(in) :: nml_name
        !! Name of the namelist to be written

        if(io_error == 0) return

        call handle_error("Failed to write "//trim(nml_name) &
                          //" namelist to "//trim(filename) &
                          //" parameter file!", &
                          GENEX_ERR_PARAMETER_FILE, __LINE__, __FILE__)
    end subroutine

    subroutine handle_bounds_error(i_was, i_min, i_max, line_number, file_name)
        !! Handles potential errors from out-of-bounds calls of species
        !! depended getters
        integer, intent(in) :: i_was
        !! Index used in call
        integer, intent(in) :: i_min
        !! Smallest index allowed
        integer, intent(in) :: i_max
        !! Largest index allowed
        integer, intent(in) :: line_number
        !! Line number where error or warning occured, i.e. __LINE__
        character(len=*), intent(in) :: file_name
        !! File name where error or warning occured, i.e. __FILE__

        if(i_was < i_min .or. i_was > i_max) then
            call handle_error("Selected index in is out of bounds!", &
                              GENEX_ERR_PARAMETERS, line_number, &
                              file_name, &
                              additional_info=error_info_t(&
                                                "Indices was/min/max: ", &
                                                [i_was, i_min, i_max]))
        end if

    end subroutine

end module
