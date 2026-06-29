module timer_m
    !! Module implementing a timer to measure time differences for benchmarking
    use genex_fortran_env_m, only: GP
    implicit none

    real(kind = GP), private :: t1
    real(kind = GP), private :: t2

    public :: tick, tock, diff
contains
    function get_time() result(res)
        !! Returns the time since midnight in ms
        integer :: values(8)
        real(kind = GP) :: res

        call date_and_time(values=values)

        ! Calculate time since midnight
        res = real(values(5), kind = GP)*60.
        res = (res + real(values(6), kind = GP))*60.
        res = (res + real(values(7), kind = GP))*1e3
        res = res + real(values(8), kind = GP)
    end function

    subroutine tick()
        !! Saves the current time as a tick
        !$omp master
        t1 = get_time()
        !$omp end master
    end subroutine

    subroutine tock()
        !! Saves the current time as a tock
        !$omp master
        t2 = get_time()
        !$omp end master
    end subroutine

    real(kind = GP) function diff()
        !! Returns the time difference in ms between tick and tock
        !$omp master
        diff = t2 - t1
        !$omp end master
    end function
end module timer_m
