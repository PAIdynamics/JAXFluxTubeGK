submodule(op_coll_m) op_coll_lbd_vspec_cpu
    use genex_error_handling_m, only: handle_error_notimp

    use MPI
    use, intrinsic :: iso_fortran_env
    use genex_fortran_env_m, only: GP_EPS, MPI_GP
    use genex_error_handling_m, only: handle_error, &
                                      handle_error_vspec, &
                                      VSPEC_ERR_INITIALIZE
    use genex_status_codes_m, only: GENEX_ERR_COLL
    use profiler_m, only: profiler_start, profiler_stop
    use params_species_m, only: get_mass, get_charge, get_temp_scaling
    use params_normalization_m, only: get_L_ref, get_rho_ref, get_coll_ref
    use params_collisions_m, only: get_temp_floor, get_relx_type
    use params_mesh_m, only: get_use_vspectral
    use op_set_uniform_m, only: op_set_uniform_cpu_t

    implicit none

contains
    module subroutine initialize_coll_lbd_vspec_cpu(this, dcomm_handler, mesh)
        class(op_coll_lbd_vspec_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        call handle_error_notimp('vspec', __LINE__, __FILE__)
    end subroutine

    module subroutine apply_coll_lbd_vspec_cpu(this, da_f_in, da_moments, &
                                               da_f_out)
        class(op_coll_lbd_vspec_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_4d_t), intent(inout) :: da_moments
        class(data_array_5d_t), intent(inout) :: da_f_out

        call handle_error_notimp('vspec', __LINE__, __FILE__)
    end subroutine

end submodule
