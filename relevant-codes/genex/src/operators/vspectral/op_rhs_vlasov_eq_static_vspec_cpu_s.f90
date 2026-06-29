submodule (op_rhs_vlasov_eq_static_m) eq_rhs_vlasov_eq_static_vspec_s
    use genex_error_handling_m, only: handle_error_notimp

    use, intrinsic :: iso_fortran_env
    use genex_error_handling_m, only: handle_error_vspec, &
                                      VSPEC_ERR_INITIALIZE
    use params_mesh_m, only: get_use_vspectral
    use params_normalization_m, only: get_L_ref, get_rho_ref
    use params_gyrokinetic_system_m, only: get_with_nlin_polarization

    use csrmat_m, only: csrmat_t

    implicit none

contains
    module subroutine initialize_vspec_cpu(this, mesh)
        class(op_rhs_vlasov_eq_static_vspec_cpu_t), intent(inout) :: this
        class(mesh_5d_t), target, intent(inout) :: mesh

        call handle_error_notimp('vspec', __LINE__, __FILE__)
    end subroutine

    module subroutine apply_vspec_cpu(this, da_f_in, da_phi_in, &
                                      da_A_par_in, da_B_par_in, da_f_out)
        class(op_rhs_vlasov_eq_static_vspec_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_2d_t), intent(in) :: da_phi_in
        class(data_array_2d_t), intent(in) :: da_A_par_in
        class(data_array_2d_t), intent(in) :: da_B_par_in
        class(data_array_5d_t), intent(inout) :: da_f_out

        call handle_error_notimp('vspec', __LINE__, __FILE__)
    end subroutine

end submodule
