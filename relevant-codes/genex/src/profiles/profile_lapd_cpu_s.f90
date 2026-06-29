submodule (profiles_m) profile_lapd_cpu_s

    use params_species_m, only: get_charge

    implicit none

contains

    elemental function lapd_profile_function(rho, c, rho_max) result(res)
        real(kind = GP), intent(in) :: rho, c, rho_max
        real(kind = GP) :: res

        if(rho < rho_max) then
            res = c + (1.0_GP - c) * (1.0_GP - (rho / rho_max)**2)**3
        else
            res = c
        endif
    end function

    real(kind = GP) function lapd_dens(rho, c, prefac, rho_max, is_electrons)
        real(kind = GP), intent(in) :: rho, c, prefac, rho_max
        logical, intent(in) :: is_electrons

        ! Electrons and Ions have a non-uniform density profile
        lapd_dens = prefac * lapd_profile_function(rho, c, rho_max)
    end function

    real(kind = GP) function lapd_temp(rho, c, prefac, rho_max, is_electrons)
        real(kind = GP), intent(in) :: rho, c, prefac, rho_max
        logical, intent(in) :: is_electrons

        if(is_electrons) then
            ! Electrons have a non-uniform temperature profile
            lapd_temp = prefac * lapd_profile_function(rho, c, rho_max)
        else
            ! Ions have a uniform temperature profile
            lapd_temp = prefac
        endif
    end function

    module subroutine initialize_lapd_cpu(this, mesh, params, is_density, &
                                          is_electrons)
        class(profile_lapd_cpu_t), intent(inout) :: this
        class(mesh_5d_t), target, intent(inout) :: mesh
        class(params_profile_lapd_t), intent(inout) :: params
        logical, intent(in):: is_density
        logical, intent(in):: is_electrons

        this%mesh => mesh
        this%params = params
        this%is_density = is_density
        this%is_electrons = is_electrons
    end subroutine

    real(kind = GP) module function eval_lapd_cpu(this, rho, theta, phi)
        class(profile_lapd_cpu_t), intent(in) :: this
        real(kind = GP), intent(in) :: rho, theta, phi

        real(kind = GP) :: c, prefac, rho_max

        c       = this%params%cfac
        prefac  = this%params%prefac
        rho_max = this%mesh%rho_max()

        ! LAPD has different behaviour for density and temperature profiles
        if(this%is_density .eqv. .true.) then
            eval_lapd_cpu = lapd_dens(rho, c, prefac, rho_max,this%is_electrons)
        else
            eval_lapd_cpu = lapd_temp(rho, c, prefac, rho_max,this%is_electrons)
        endif
    end function

end submodule
