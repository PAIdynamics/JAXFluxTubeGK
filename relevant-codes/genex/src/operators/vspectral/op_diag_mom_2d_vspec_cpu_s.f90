submodule(op_diag_mom_2d_m) op_diag_mom_2d_vspec_cpu_s
    use genex_error_handling_m, only: handle_error_notimp
    use MPI
    use, intrinsic :: iso_fortran_env
    use genex_error_handling_m, only: handle_error, &
                                      handle_error_vspec, &
                                      VSPEC_ERR_INITIALIZE
    use genex_status_codes_m, only: GENEX_ERR_OPERATORS
    use params_species_m, only: get_mass, get_charge, get_temp_scaling
    use params_normalization_m, only: get_L_ref, get_rho_ref
    use params_mesh_m, only: get_n_points_sp, get_use_vspectral
    use math_m, only: PI
    use genex_fortran_env_m, only: MPI_GP
    use profiler_m, only: profiler_start, profiler_stop, &
                          profiler_start_allreduce, profiler_stop_allreduce
    implicit none

contains
    module subroutine initialize_vspec_cpu(this, dcomm_handler, mesh)
        class(op_diag_mom_2d_vspec_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        call handle_error_notimp('vspec', __LINE__, __FILE__)
    end subroutine

    module subroutine apply_vspec_cpu(this, da_f, da_moments, diagnose_tpc)
        class(op_diag_mom_2d_vspec_cpu_t), target, intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f
        class(data_array_4d_t), intent(inout) :: da_moments
        logical, optional, intent(in) :: diagnose_tpc

        call handle_error_notimp('vspec', __LINE__, __FILE__)
    end subroutine

end submodule
