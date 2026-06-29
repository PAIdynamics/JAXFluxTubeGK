module file_handling_m

    implicit none

    private
    public :: file_exists, delete_file_if_exists

contains

    subroutine delete_file_if_exists(filename)
        !! Delete the given file if it exists
        character(len=*), intent(in) :: filename
        !! File to delete

        integer :: iunit, ierr

        if(.not. file_exists(filename)) return

        open(newunit=iunit, iostat=ierr, file=filename, status='old')
        if(ierr == 0) close(iunit, status='delete')
    end subroutine

    logical function file_exists(filename)
        !! Checks if the given file exists
        character(len=*), intent(in) :: filename
        !! File to check

        inquire(file=filename, exist=file_exists)
    end function

end module
