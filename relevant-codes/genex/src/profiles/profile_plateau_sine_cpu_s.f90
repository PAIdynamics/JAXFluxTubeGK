submodule (profiles_m) profile_plateau_sine_cpu_s

    use math_m, only: PI

    implicit none

contains

    module subroutine initialize_plateau_sine_cpu(this, params)
        class(profile_plateau_sine_cpu_t), intent(inout) :: this
        class(params_profile_plateau_t), intent(inout) :: params

        this%params = params
    end subroutine

    real(kind=GP) module function eval_plateau_sine_cpu(this, rho, theta, phi)
        class(profile_plateau_sine_cpu_t), intent(in) :: this
        real(kind=GP), intent(in) :: rho, theta, phi

        real(kind=GP) :: rho_inner_foot, rho_inner_top, &
                         rho_outer_top, rho_outer_foot, &
                         amp_in, amp_plateau, amp_out
        real(kind=GP) :: c1, c2, c3, c4

        rho_inner_foot = this%params%rho_inner_foot
        rho_inner_top  = this%params%rho_inner_top
        rho_outer_top  = this%params%rho_outer_top
        rho_outer_foot = this%params%rho_outer_foot
        amp_in      = this%params%amp_in
        amp_plateau = this%params%amp_plateau
        amp_out     = this%params%amp_out

        ! Const in rho < rho_inner_foot
        if (rho < rho_inner_foot) then
            eval_plateau_sine_cpu = amp_in
        ! Sine step function in rho_inner_foot <= rho < rho_inner_top
        else if (rho < rho_inner_top) then
            c1 = (amp_in + amp_plateau) / 2.0_GP
            c2 = c1 - amp_plateau
            c3 = PI / (rho_inner_top - rho_inner_foot)
            c4 = PI / 2.0_GP - c3 * rho_inner_foot
            eval_plateau_sine_cpu = c1 + c2 * sin(c3 * rho + c4)
        ! Const in rho_inner_top <= rho < rho_outer_top
        else if (rho < rho_outer_top) then
            eval_plateau_sine_cpu = amp_plateau
        ! Sine step function in rho_outer_top <= rho < rho_outer_foot
        else if (rho < rho_outer_foot) then
            c1 = (amp_plateau + amp_out) / 2.0_GP
            c2 = amp_plateau - c1
            c3 = PI / (rho_outer_foot - rho_outer_top)
            c4 = PI / 2.0_GP - c3 * rho_outer_top
            eval_plateau_sine_cpu = c1 + c2 * sin(c3 * rho + c4)
        ! Const in rho >= rho_outer_foot
        else if (rho >= rho_outer_foot) then
            eval_plateau_sine_cpu = amp_out
        end if
    end function

end submodule
