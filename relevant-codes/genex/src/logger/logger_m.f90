module logger_m
    !! Module to handle standard input/output. It provides functionality to
    !! log info, debug and error information by returning the corresponding
    !! channel number that can be used in a write statement.

    ! NOTE: File output has not been implemented yet

    use, intrinsic :: iso_fortran_env, only: OUTPUT_UNIT, ERROR_UNIT
    implicit none

    private

    integer, public, parameter :: stdout = OUTPUT_UNIT
    !! Standard output channel
    integer, public, parameter :: stderr = ERROR_UNIT
    !! Standard error channel

    integer :: info_channel  = stdout
    !! Current info channel
    integer :: debug_channel = stdout
    !! Current debug channel
    integer :: error_channel = stderr
    !! Current error channel

    public :: logger_set_info_channel
    public :: logger_set_debug_channel
    public :: logger_set_error_channel

    public :: logger_get_info_channel
    public :: logger_get_debug_channel
    public :: logger_get_error_channel

    public :: logger_welcome

    public :: logger_log_region

contains

    subroutine logger_set_info_channel(channel)
        !! Sets the output channel for info output
        integer, intent(in) :: channel
        !! Channel number
        info_channel = channel
    end subroutine

    subroutine logger_set_debug_channel(channel)
        !! Sets the output channel for debug output
        integer, intent(in) :: channel
        !! Channel number
        debug_channel = channel
    end subroutine

    subroutine logger_set_error_channel(channel)
        !! Sets the output channel for error output
        integer, intent(in) :: channel
        !! Channel number
        error_channel = channel
    end subroutine

    integer function logger_get_info_channel()
        !! Returns the channel for info output
        logger_get_info_channel = info_channel
    end function

    integer function logger_get_debug_channel()
        !! Returns the channel for debug output
        logger_get_debug_channel = debug_channel
    end function

    integer function logger_get_error_channel()
        !! Returns the channel for error output
        logger_get_error_channel = error_channel
    end function

    subroutine logger_welcome()
        !! Prints a welcome message to info channel
        character(len=:), allocatable :: msg
        integer :: j

#include "./welcome.txt"

        ! NOTE: Numbers are given by properties of welcome text
        write (info_channel, "(A80)") (msg(1+j*80:(j+1)*80), j=0,16)

    end subroutine

    subroutine logger_log_region(region_name, ierr)
        !! Outputs the given region name to a file in overwrite mode.
        !! Can be used to trace problems in case of an unexpected termination.
        !! NOTE: As it is written, this subroutine can print any char
        !!       into a file in general.
        character(len=*), intent(in) :: region_name
        !! Name of the current region
        integer, intent(out) :: ierr
        !! Error code

        character(len=:), allocatable :: log_file
        !! Name of the log file
        integer :: iunit
        !! Assigned unit for debug

        log_file = "debug_region.txt"
        open(newunit=iunit, file=log_file, status="replace", &
                            action="write", iostat=ierr)
        write(iunit, '(A)') trim(region_name)
        close(iunit)
    end subroutine

end module logger_m
