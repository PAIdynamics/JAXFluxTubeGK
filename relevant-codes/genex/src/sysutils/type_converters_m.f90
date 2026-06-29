module type_converters_m
    use genex_fortran_env_m, only: GP
    use, intrinsic :: iso_c_binding, only : C_CHAR, C_NULL_CHAR
    implicit none

    private

    public :: string
    public :: string_f2c
    public :: string_c2f

    interface string
        procedure :: string_integer
        procedure :: string_real
    end interface

contains

    function string_integer(i) result(res)
        !! Returns the string representation of the given integer
        integer, intent(in) :: i
        !! Integer to convert
        character(len=:), allocatable :: res

        character(range(i)+2) :: tmp

        write(tmp,'(i0)') i
        res = trim(tmp)
    end function

    function string_real(r, ndec) result(res)
        !! Returns the string representation of the given real number. The
        !! string can have a total size of 16 characters, the amount of decimals
        !! can be specified. However if the number is too big to be printed
        !! with requested amount of decimals, this function returns asterisks.
        real(kind=GP), intent(in) :: r
        !! Real to convert
        integer, intent(in) :: ndec
        !! Number of decimals to write
        character(len=:), allocatable :: res

        character(len=16) :: tmp, fmt_tmp

        ! NOTE: This block catches the case that the number is too big to be
        !       written with the given amount of decimals specified. This may
        !       trigger an error in the write, depending on the compiler.
        !       Therefore we catch this case here and return asterisks manually.
        if(abs(r) > (10.0_GP**(14-ndec) - 1.0_GP))then
            res = repeat("*", 16)
            return
        endif

        fmt_tmp = '(f16.'//string(ndec)//')'
        write(tmp, fmt_tmp) r
        res = trim(adjustl(tmp))
    end function

    subroutine string_f2c(f_string, c_string)
        !! Convert Fortran string to C string formatted by char-array
        character(len=*), intent(in) :: f_string
        !! Fortran string
        character(len=1, kind=C_CHAR), dimension(:), intent(out) :: c_string
        !! C string

        integer :: i, string_len

        ! Truncation to accomodate the case where c_string array is shorter
        string_len = min(len(trim(f_string)), size(c_string) - 1)

        ! Copy each characters
        do i = 1, string_len
            c_string(i) = f_string(i:i)
        enddo

        ! Terminate the C-formatted string array
        c_string(string_len + 1) = C_NULL_CHAR

    end subroutine

    subroutine string_c2f(c_string, f_string)
        !! Convert C string formatted by char-array to Fortran string
        character(len=1, kind=C_CHAR), dimension(:), intent(in) :: c_string
        !! C string
        character(len=*), intent(out) :: f_string
        !! Fortran string

        integer :: i

        ! Initialize with empty string
        f_string = ""

        ! Copy each characters
        do i = 1, len(f_string)
            if(c_string(i) == C_NULL_CHAR) exit
            f_string(i:i) = c_string(i)
        enddo
    end subroutine

end module
