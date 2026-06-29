submodule(op_neut_evolve_m) op_neutrals_1mom_2cfd_s
    use genex_error_handling_m, only: handle_error_notimp

    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_ERR_OPERATORS, &
                                    GENEX_ERR_PARAMETERS
    use mesh_5d_m, only: FILLER_VALUE_REAL
    use params_neutrals_m, only: get_n_points_neut, get_neut_mapped_ion_name, &
                                 get_neut_mass
    use params_neutrals_config_m, only: get_neut_dens_floor
    use params_normalization_m, only: get_xsec_ref

    implicit none

contains
   module subroutine initialize_evolve_1mom_2cfd(this, dcomm_handler, mesh)
        class(op_neut_evolve_1mom_2cfd_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        call handle_error_notimp('neutrals', __LINE__, __FILE__)
    end subroutine

    module subroutine apply_evolve_1mom_2cfd(this, da_n_sources, &
                                             da_n_in, da_n_out)
        class(op_neut_evolve_1mom_2cfd_t), intent(inout) :: this
        class(data_array_4d_t), intent(in) :: da_n_sources
        class(data_array_4d_t), intent(in) :: da_n_in
        class(data_array_4d_t), intent(inout) :: da_n_out

        call handle_error_notimp('neutrals', __LINE__, __FILE__)
    end subroutine

end submodule
