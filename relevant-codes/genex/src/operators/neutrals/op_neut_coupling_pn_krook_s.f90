submodule(op_neut_coupling_pn_m) op_neut_coupling_pn_krook_s
    use genex_error_handling_m, only: handle_error_notimp
    use genex_fortran_env_m, only: GP
    use params_species_m, only: get_is_electrons

    implicit none

contains
    module subroutine initialize_coupling_pn_krook(this, dcomm_handler, &
                                                         mesh)
        class(op_neut_coupling_pn_krook_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        call handle_error_notimp('neutrals', __LINE__, __FILE__)
    end subroutine

    module subroutine apply_coupling_pn_krook(this, da_moments, da_f_in, &
                                              da_n_in, da_n_sources)
        class(op_neut_coupling_pn_krook_t), intent(inout) :: this
        class(data_array_4d_t), intent(in) :: da_moments
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_4d_t), intent(in) :: da_n_in
        class(data_array_4d_t), intent(inout) :: da_n_sources

        call handle_error_notimp('neutrals', __LINE__, __FILE__)
    end subroutine

end submodule
