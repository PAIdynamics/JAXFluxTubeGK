module perf_counter_m
    use, intrinsic :: iso_fortran_env
    use timer_m, only: tick, tock, diff
    use genex_fortran_env_m, only: GP
    implicit none

    type, public :: perf_counter_t
        !! Type to measure flops rate and memory bandwidth of operators
        integer(kind = INT64), private :: n = 1
        !! Number of measurements
        real(kind = GP), private :: avg_time = 0.0_GP
        !! Average of the measure execution times
    contains
        procedure, public, nopass :: start_measurement
        procedure, public :: end_measurement
        procedure, public :: get_flop_rate
        procedure, public :: get_bandwidth
        procedure, public :: get_avg_time
        procedure, public :: reset
    end type

contains

    subroutine reset(this)
        !! Resets the counter
        class(perf_counter_t), intent(inout) :: this
        !! Instance of the type
        this%avg_time = 0.0_GP
        this%n = 1
    end subroutine

    function calc_rate(count, milliseconds) result(res)
        !! Calculates the rate in the unit Giga/s
        real(kind = GP), intent(in) :: count
        real(kind = GP), intent(in) :: milliseconds
        real :: res
        res = count /(milliseconds * 1e6)
    end function

    subroutine start_measurement()
        !! Starts the performance measurement
        call tick()
    end subroutine

    subroutine end_measurement(this)
        !! Stops the performance measurement
        class(perf_counter_t), intent(inout) :: this
        !! Instance of the type
        call tock()
        this%avg_time = this%avg_time + (diff() - this%avg_time) &
            / real(this%n, kind = GP)
        this%n = this%n + 1
    end subroutine

    real(kind = GP) function get_flop_rate(this, n_ops)
        !! Returns the flop rate in GFLOPS/s
        class(perf_counter_t), intent(in) :: this
        !! Instance of the type
        integer(kind = INT64), intent(in) :: n_ops
        !! Number of floating point operations
        get_flop_rate = calc_rate(real(n_ops, kind = GP), this%avg_time)
    end function

    real(kind = GP) function get_bandwidth(this, n_bytes)
        !! Returns the memory bandwidth GByte/s
        class(perf_counter_t), intent(in) :: this
        !! Instance of the type
        integer(kind = INT64), intent(in) :: n_bytes
        !! Number of loads and stores
        get_bandwidth = calc_rate(real(n_bytes, kind = GP), this%avg_time)
    end function

    real(kind = GP) function get_avg_time(this)
        !! Returns the average time of a single operator call in ms
        class(perf_counter_t), intent(in) :: this
        !! Instance of the type
        get_avg_time = this%avg_time
    end function

end module perf_counter_m
