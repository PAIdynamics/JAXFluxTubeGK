submodule (op_source_m) op_source_heat_cpu_s
    use genex_error_handling_m, only: handle_error_notimp
    use, intrinsic :: iso_fortran_env
    use genex_error_handling_m, only: handle_error
    use math_m, only: PI
    use genex_status_codes_m, only: GENEX_ERR_OPERATORS
    use params_normalization_m, only: get_heat_ref, get_P_ref
    use params_species_m, only: get_temp_scaling
    use params_source_m, only: n_source_loc_supported
    use params_source_heat_m, only : get_params_source_heat, &
                                     get_n_source_loc_heat
    use op_set_uniform_m, only: op_set_uniform_cpu_t

    implicit none

contains
    module subroutine initialize_heat_cpu(this, mesh)
        class(op_source_loc_heat_cpu_t), intent(inout) :: this
        class(mesh_5d_t), target, intent(inout) :: mesh

        call handle_error_notimp('sources', __LINE__, __FILE__)
    end subroutine

    module subroutine apply_heat_cpu(this, da_f_out)
        class(op_source_loc_heat_cpu_t), target, intent(inout) :: this
        class(data_array_5d_t), intent(inout) :: da_f_out

        call handle_error_notimp('sources', __LINE__, __FILE__)
    end subroutine

end submodule
