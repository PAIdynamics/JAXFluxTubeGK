submodule (profiles_m) profile_sine_cpu_s

    use math_m, only: PI

    implicit none

contains

    module subroutine initialize_sine_cpu(this, params)
        class(profile_sine_cpu_t), intent(inout) :: this
        class(params_profile_sine_t), intent(inout) :: params

        this%params = params
    end subroutine

    real(kind = GP) module function eval_sine_cpu(this, rho, theta, phi)
        class(profile_sine_cpu_t), intent(in) :: this
        real(kind = GP), intent(in) :: rho, theta, phi

        real(kind = GP) :: rho_min, rho_max, amp_min, amp_max
        real(kind = GP) :: c1, c2, c3, c4

        rho_min = this%params%rho_min
        rho_max = this%params%rho_max
        amp_min = this%params%amp_min
        amp_max = this%params%amp_max

        ! Const in rho_max < rho or rho < rho_min
        if(rho >= rho_max) then
            eval_sine_cpu = amp_min
        elseif(rho <= rho_min) then
            eval_sine_cpu = amp_max
        ! Sine function in rho_min < rho < rho_max
        else
            c4 = (amp_max + amp_min) / 2
            c1 = amp_max - c4
            c2 = PI / (rho_max - rho_min)
            c3 = PI / 2 - c2 * rho_min
            eval_sine_cpu = c1 * sin(c2 * rho + c3) + c4
        endif

    end function

end submodule
