module system_calls_m

    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_SUCCESS, GENEX_ERR_SYSTEM_CALL, &
                                    GENEX_WRN_SYSTEM_CALL
    use type_converters_m, only: string

    implicit none

    private
    public :: mkdir, rm_rf

    integer, parameter :: msg_len = 128
    !! Parameter that sets the maximum length of the error message that can
    !! be received by a system call.
contains

    subroutine mkdir(path_name, ierr, parents)
        !! Creates a path to a new directory given by part_name
        character(len=*), intent(in) :: path_name
        !! Name of the path to the directory
        integer, intent(out) :: ierr
        !! Error code
        logical, optional, intent(in) :: parents
        !! If true mkdir will return no error when the directory already exists
        !! and make parent directories as needed

        integer :: istat
        ! Execution status
        character(len=:), allocatable :: command
        ! Execution command
        character(len=msg_len) :: message = ""
        ! Error message

        command = "mkdir"
        if(present(parents)) then
            if(parents) then
                command = command//" --parents"
            endif
        endif

        call execute_command_line(trim(command)//" "//trim(path_name), &
                                  cmdmsg=message, cmdstat=ierr, exitstat=istat)
        call check_sys_error(ierr, istat, message)
    end subroutine

    subroutine rm_rf(path_name, ierr)
        !! Removes the given directory path with all its content
        character(len=*), intent(in) :: path_name
        !! Name of the path to the directory
        integer, intent(out) :: ierr
        !! Error code

        integer :: istat
        ! Execution status
        character(len=msg_len) :: message = ""
        ! Error message

        call execute_command_line("rm -rf "//trim(path_name), &
                                  cmdmsg=message, cmdstat=ierr, exitstat=istat)
        call check_sys_error(ierr, istat, message)
    end subroutine

    subroutine check_sys_error(ierr, istat, message)
        !! Subroutine that checks ierr and istat for errors and prints error
        !! message in that case. Afterwards sets the value of ierr to a
        !! standardized GENEX error code.
        integer, intent(inout) :: ierr
        !! Error code
        integer, intent(in) :: istat
        !! Execution status
        character(len=*), intent(in) :: message
        !! Error message

        if(ierr /= 0) then
            call handle_error("System call not executed. Returned code " &
                              //string(ierr)//". "//message, &
                              GENEX_WRN_SYSTEM_CALL, __LINE__, __FILE__)
        endif
        if(istat /= 0) then
            call handle_error("Failed system call. Returned code " &
                              //string(istat)//". "//message, &
                              GENEX_WRN_SYSTEM_CALL, __LINE__, __FILE__)
        endif
        ! We treat both, non-execution and non-silent output as an error
        if(ierr /= 0 .or. istat /= 0) then
            ierr = GENEX_ERR_SYSTEM_CALL
        else
            ierr = GENEX_SUCCESS
        endif
    end subroutine
end module
