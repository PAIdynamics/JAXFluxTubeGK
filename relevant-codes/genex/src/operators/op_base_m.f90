module op_base_m
    use, intrinsic :: iso_fortran_env
    use genex_fortran_env_m, only: GP
    use perf_counter_m, only: perf_counter_t
    use logger_m, only: logger_get_info_channel

    implicit none

    type, public, abstract :: op_base_t
        !! Base type for any operator. It provides the functionality to measure
        !! the memory bandwidth and flop rate of every operator that inherits
        !! from it.
        type(perf_counter_t), public :: perf_counter
        !! Type to measure flop rate and bandwidth of the operator
        integer, public :: n_flops = 0
        !! Number of floating point operations per iteration
        integer, public :: n_ls = 0
        !! Number of loads and stores per iteration
        integer(kind=INT64), public :: n_iterations = 0
        !! Number of iterations
    contains
        procedure :: reset_perf_counter
        procedure :: get_bandwidth
        procedure :: get_flop_rate
        procedure :: print_performance_summary
    end type

contains

    subroutine reset_perf_counter(this)
        class(op_base_t), intent(inout) :: this
        !! Instance of the type
        !! Deletes previous performance measurements
        call this%perf_counter%reset()
    end subroutine

    real(kind=GP) function get_flop_rate(this)
        !! Returns the flop rate in GFLOPS/s
        class(op_base_t), intent(in) :: this
        !! Instance of the type
        get_flop_rate = this%perf_counter%get_flop_rate( &
            this%n_iterations * this%n_flops)
    end function

    real(kind=GP) function get_bandwidth(this)
        !! Returns the memory bandwidth in GFLOPS/s
        class(op_base_t), intent(in) :: this
        !! Instance of the type
        integer(kind=INT64) :: n_bytes
        n_bytes = this%n_iterations &
                  * int(this%n_ls, kind=INT64) &
                  * int(GP, kind=INT64)
        get_bandwidth = this%perf_counter%get_bandwidth(n_bytes)
    end function

    subroutine print_performance_summary(this)
        !! Prints a performance summary on screen
        class(op_base_t), intent(in) :: this
        !! Instance of the type
        write(logger_get_info_channel(), *) &
            "    Measured flop rate [GFLOPS/s]: ", this%get_flop_rate()
        write(logger_get_info_channel(), *) &
            "    Measured bandwidth [GByte/s]:  ", this%get_bandwidth()
        write(logger_get_info_channel(), *) &
            "    Measured time per call [ms]:   ", &
            this%perf_counter%get_avg_time()
    end subroutine
end module op_base_m
