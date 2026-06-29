submodule (op_source_m) op_source_dens_vspec_cpu_s
    use genex_error_handling_m, only: handle_error_notimp
    use genex_error_handling_m, only: handle_error_notimp
    use, intrinsic :: iso_fortran_env
    use math_m, only: PI
    use genex_error_handling_m, only: handle_error_vspec, &
                                      VSPEC_ERR_INITIALIZE
    use params_normalization_m, only: get_fuel_ref, get_n_ref
    use params_source_dens_m, only: get_params_source_dens, &
                                    get_n_source_loc_dens
    use params_source_m, only: n_source_loc_supported
    use params_mesh_m, only: get_use_vspectral
    use params_species_m, only: get_temp_scaling
    use op_set_uniform_m, only: op_set_uniform_cpu_t
    use polynomial_integrals_m, only: integral_hermite, &
                                      integral_laguerre

    implicit none

contains
    module subroutine initialize_dens_vspec_cpu(this, mesh)
        class(op_source_loc_dens_vspec_cpu_t), intent(inout) :: this
        class(mesh_5d_t), target, intent(inout) :: mesh

        call handle_error_notimp('sources', __LINE__, __FILE__)
    end subroutine

    module subroutine apply_dens_vspec_cpu(this, da_f_out)
        class(op_source_loc_dens_vspec_cpu_t), target, intent(inout) :: this
        class(data_array_5d_t), intent(inout) :: da_f_out

        call handle_error_notimp('sources', __LINE__, __FILE__)
    end subroutine

end submodule
