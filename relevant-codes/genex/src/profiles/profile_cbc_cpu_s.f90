submodule (profiles_m) profile_cbc_cpu_s

    implicit none

contains

    module subroutine initialize_cbc_cpu(this, params)
        class(profile_cbc_cpu_t), intent(inout) :: this
        class(params_profile_cbc_t), intent(inout) :: params

        this%params = params
    end subroutine

    real(kind = GP) module function eval_cbc_cpu(this, rho, theta, phi)
        class(profile_cbc_cpu_t), intent(in) :: this
        real(kind = GP), intent(in) :: rho, theta, phi

        real(kind = GP) :: kappa, delta, center, prefac

        prefac = this%params%prefac
        kappa  = this%params%kappa
        delta  = this%params%delta
        center = this%params%center

        eval_cbc_cpu = prefac * exp(-kappa*delta * tanh((rho - center) / delta))
    end function
end submodule
