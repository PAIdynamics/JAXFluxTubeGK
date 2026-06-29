submodule (profiles_m) profile_uniform_cpu_s

    implicit none

contains

    module subroutine initialize_uniform_cpu(this, params)
        class(profile_uniform_cpu_t), intent(inout) :: this
        class(params_profile_uniform_t), intent(inout) :: params

        this%params = params
    end subroutine

    real(kind = GP) module function eval_uniform_cpu(this, rho, theta, phi)
        class(profile_uniform_cpu_t), intent(in) :: this
        real(kind = GP), intent(in) :: rho, theta, phi

        eval_uniform_cpu = this%params%strength
    end function

end submodule
