submodule(op_coll_m) op_coll_lorentz_cpu_s
    use genex_error_handling_m, only: handle_error_notimp

    use MPI
    use, intrinsic :: iso_fortran_env
    use genex_fortran_env_m, only: GP_EPS, MPI_GP
    use math_m, only: PI
    use logger_m, only: logger_get_info_channel
    use profiler_m, only: profiler_start, profiler_stop
    use params_species_m, only: get_mass, get_charge, get_temp_scaling
    use params_normalization_m, only: get_L_ref, get_rho_ref, get_coll_ref
    use params_parallelization_m, only: get_n_procs_vp, get_n_procs_mu

    implicit none

contains
    module subroutine initialize_coll_lorentz_cpu(this, dcomm_handler, mesh)
        class(op_coll_lorentz_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        call handle_error_notimp('collision', __LINE__, __FILE__)
    end subroutine

    module subroutine apply_coll_lorentz_cpu(this, da_f_in, da_moments, &
                                             da_f_out)
        class(op_coll_lorentz_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_4d_t), intent(inout) :: da_moments
        class(data_array_5d_t), intent(inout) :: da_f_out

        call handle_error_notimp('collision', __LINE__, __FILE__)
    end subroutine

end submodule
