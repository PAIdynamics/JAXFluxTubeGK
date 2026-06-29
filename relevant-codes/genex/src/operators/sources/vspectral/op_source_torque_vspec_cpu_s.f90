submodule (op_source_m) op_source_torque_vspec_cpu_s
    use genex_error_handling_m, only: handle_error_notimp
    use genex_error_handling_m, only: handle_error_notimp
    use, intrinsic :: iso_fortran_env
    use math_m, only: PI
    use genex_error_handling_m, only: handle_error_vspec, &
                                      VSPEC_ERR_INITIALIZE
    use params_normalization_m, only: get_torque_ref
    use params_source_torque_m, only: get_params_source_torque, &
                                      get_n_source_loc_torque
    use params_source_m, only: n_source_loc_supported
    use params_species_m, only: get_mass, get_temp_scaling
    use params_mesh_m, only: get_use_vspectral
    use op_set_uniform_m, only: op_set_uniform_cpu_t
    use polynomial_integrals_m, only: integral_hermite, &
                                      integral_laguerre

    implicit none

contains
    module subroutine initialize_torque_vspec_cpu(this, mesh)
        class(op_source_loc_torque_vspec_cpu_t), intent(inout) :: this
        class(mesh_5d_t), target, intent(inout) :: mesh

        call handle_error_notimp('sources', __LINE__, __FILE__)
    end subroutine

    module subroutine apply_torque_vspec_cpu(this, da_f_out)
        class(op_source_loc_torque_vspec_cpu_t), target, intent(inout) :: this
        class(data_array_5d_t), intent(inout) :: da_f_out

        call handle_error_notimp('sources', __LINE__, __FILE__)
    end subroutine

end submodule
