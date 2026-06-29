submodule (op_rhs_vlasov_eq_static_m) op_rhs_vlasov_eq_static_vspec_s
    use genex_error_handling_m, only: handle_error_notimp
    use, intrinsic :: iso_fortran_env
    use params_numerical_scheme_m, only: get_hyp_vp
    use params_mesh_m, only: get_n_points_sp

    implicit none

contains
    module subroutine initialize_parent_vspec(this, mesh)
        class(op_rhs_vlasov_eq_static_vspec_t), intent(inout) :: this
        class(mesh_5d_t), target, intent(inout) :: mesh

        call handle_error_notimp('vspec', __LINE__, __FILE__)
    end subroutine

end submodule
