submodule (op_rhs_vlasov_eq_dynamic_m) op_rhs_vlasov_eq_dynamic_vspec_cpu_s
    use genex_error_handling_m, only: handle_error_notimp

    use, intrinsic :: iso_fortran_env
    use genex_error_handling_m, only: handle_error_vspec, &
                                      VSPEC_ERR_INITIALIZE
    use params_species_m, only: get_mass, get_charge, get_temp_scaling
    use params_mesh_m, only: get_use_vspectral
    use params_normalization_m, only: get_L_ref, get_rho_ref
    use op_set_uniform_m, only: op_set_uniform_cpu_t

    implicit none

contains
    module subroutine initialize_vspec_cpu(this, mesh)
        class(op_rhs_vlasov_eq_dynamic_vspec_cpu_t), intent(inout) :: this
        class(mesh_5d_t), target, intent(inout) :: mesh

        call handle_error_notimp('vspec', __LINE__, __FILE__)
    end subroutine

    module subroutine apply_vspec_cpu(this, da_f_in, da_E_par_in, da_f_out)
        class(op_rhs_vlasov_eq_dynamic_vspec_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_2d_t), intent(in) :: da_E_par_in
        class(data_array_5d_t), intent(inout) :: da_f_out

        call handle_error_notimp('vspec', __LINE__, __FILE__)
    end subroutine

end submodule
