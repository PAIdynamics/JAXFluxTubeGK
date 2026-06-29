submodule(op_neut_coupling_np_m) op_neut_coupling_np_krook_s
    use genex_error_handling_m, only: handle_error_notimp
    use genex_fortran_env_m, only: GP
    use math_m, only: PI
    use params_species_m, only: get_is_electrons, get_mass, get_charge
    use params_neutrals_m, only: get_neut_mass
    use params_neutrals_config_m, only: get_neut_gamma_u, get_neut_gamma_T, &
                                        get_neut_gamma_W
    use params_normalization_m, only: get_L_ref, get_rho_ref

    implicit none

contains
    module subroutine initialize_coupling_np_krook(this, dcomm_handler, &
                                                         mesh)
        class(op_neut_coupling_np_krook_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        call handle_error_notimp('neutrals', __LINE__, __FILE__)
    end subroutine

    module subroutine apply_coupling_np_krook(this, da_moments, &
                                                    da_f_in, da_n_in, &
                                                    da_f_out)
        class(op_neut_coupling_np_krook_t), intent(inout) :: this
        class(data_array_4d_t), intent(in) :: da_moments
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_4d_t), intent(in) :: da_n_in
        class(data_array_5d_t), intent(inout) :: da_f_out

        call handle_error_notimp('neutrals', __LINE__, __FILE__)
    end subroutine

end submodule
