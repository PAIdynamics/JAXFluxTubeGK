submodule (dist_initial_m) dist_initial_slowing_down_cpu_s

    use math_m, only: PI, heaviside
    use params_mesh_m, only: get_n_points_sp
    use params_species_m, only: get_charge, get_mass, get_temp_scaling
    use params_dist_initial_slowing_down_m, only: &
                                            get_params_dist_initial_slowing_down

    implicit none

contains

    module subroutine initialize_slowing_down_cpu(this, profile_container, &
                                                  spec_ind)
        class(dist_initial_slowing_down_cpu_t), intent(inout) :: this
        type(profile_container_t), dimension(:), intent(in) :: &
                                                        profile_container
        integer, intent(in) :: spec_ind

        call this%initialize_isotropic(profile_container, spec_ind)

        this%params = get_params_dist_initial_slowing_down(spec_ind)

    end subroutine

    real(kind = GP) module function eval_slowing_down_cpu(this, rho, theta, &
                                        phi, vp, muB) result(f_out)
        class(dist_initial_slowing_down_cpu_t), intent(in) :: this
        real(kind = GP), intent(in) :: rho, theta, phi, vp, muB

        real(kind = GP) :: vcross, dens_f, temp_f, vth_f, &
                                          dens_e, temp_e, vth_e
        integer :: ne, ierr

        ! Nomenclature:
        ! f ... this species (for any)
        ! e ... electron only

        associate(pc => this%profile_container, nf => this%spec_ind)

            ! Evaluate density and temperature if this species
            call pc(nf)%eval_dens(rho, theta, phi, dens_f, ierr)
            call pc(nf)%eval_temp(rho, theta, phi, temp_f, ierr)
            temp_f = temp_f / get_temp_scaling(nf)
            vth_f = sqrt(2.0_GP * temp_f / get_mass(nf))

            ! Get index of electrons by exit if index is correct
            do ne = 1, get_n_points_sp()
                if(get_charge(ne) == -1.0_GP) exit
            end do

            ! Evaluate density and temperature of electrons
            call pc(ne)%eval_dens(rho, theta, phi, dens_e, ierr)
            call pc(ne)%eval_temp(rho, theta, phi, temp_e, ierr)
            temp_e = temp_e / get_temp_scaling(ne)
            vth_e = sqrt(2.0_GP * temp_e / get_mass(ne))

            block ! Calculate vcross
                integer :: ni
                real(kind = GP) :: dens_i, Z1, alpha

                Z1 = 0.0_GP
                ! Calculate effective charge
                do ni = 1, get_n_points_sp()
                    if(ni == ne) cycle
                    call pc(ni)%eval_dens(rho, theta, phi, dens_i, ierr)
                    Z1 = Z1 + dens_i * get_charge(ni)**2 / get_mass(ni)
                end do

                ! Calculate cross prefactor
                alpha = (3.0_GP * PI * get_mass(ne) * Z1 / &
                        (4.0_GP * dens_e))**(1.0_GP / 3.0_GP)

                vcross = alpha * vth_e
            end block
        end associate

        block ! Calculate distribution
            real(kind = GP) :: vabs_sq, log_term, heavi_arg, denom

            vabs_sq = vp**2 + muB

            log_term = log(1.0_GP + (this%params%vbirth * vth_f / vcross))
            heavi_arg = vth_f * (this%params%vbirth - sqrt(vabs_sq))
            denom = vabs_sq**(1.5_GP) + (vcross / vth_f)**3

            ! Return slowing down in 3D cartesian coord. system in 2D velspace
            f_out = 2.0_GP * dens_f / (4.0_GP * PI * log_term) * &
                    heaviside(heavi_arg) / denom
        end block

    end function

end submodule
