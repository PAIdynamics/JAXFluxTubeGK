submodule(dist_initial_m) dist_initial_maxw_vspec_cpu_s
    use genex_error_handling_m, only: handle_error_notimp

    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_ERR_DIST_INITIAL
    use params_mesh_m, only: get_use_vspectral
    use params_species_m, only: get_temp_scaling
    use params_dist_initial_maxw_vspec_m, only: &
                                              get_params_dist_initial_maxw_vspec

    implicit none

contains
    module subroutine initialize_maxw_vspec_cpu(this, &
                                                profile_container, &
                                                spec_ind)
        class(dist_initial_maxw_vspec_cpu_t), intent(inout) :: this
        type(profile_container_t), dimension(:), intent(in) :: &
                                                               profile_container
        integer, intent(in) :: spec_ind

        call handle_error_notimp('vspec', __LINE__, __FILE__)
    end subroutine

    real(kind=GP) module function eval_maxw_vspec_cpu(this, &
                                                      rho, theta, &
                                                      phi, vp, &
                                                      muB) result(f_out)
        class(dist_initial_maxw_vspec_cpu_t), intent(in) :: this
        real(kind=GP), intent(in) :: rho, theta, phi, vp, muB

        call handle_error_notimp('vspec', __LINE__, __FILE__)
    end function

end submodule
