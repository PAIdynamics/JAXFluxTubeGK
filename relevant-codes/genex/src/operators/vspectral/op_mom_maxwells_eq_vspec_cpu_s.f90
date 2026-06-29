submodule (op_mom_maxwells_eq_m) op_mom_maxwells_eq_vspec_cpu_s
    use genex_error_handling_m, only: handle_error_notimp

    use MPI
    use genex_fortran_env_m, only: MPI_GP
    use genex_error_handling_m, only: handle_error_vspec, &
                                      VSPEC_ERR_INITIALIZE
    use params_species_m, only: get_mass, get_charge, get_temp_scaling
    use params_mesh_m, only: get_use_vspectral
    use params_normalization_m, only: get_L_ref, get_rho_ref, get_beta_ref
    use profiler_m, only: profiler_start, profiler_stop, &
                          profiler_start_allreduce, profiler_stop_allreduce

    implicit none

contains
    module subroutine initialize_vspec_cpu(this, dcomm_handler, mesh)
        class(op_mom_maxwells_eq_vspec_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        call handle_error_notimp('vspec', __LINE__, __FILE__)
    end subroutine

    module subroutine apply_vspec_cpu(this, da_f_in, da_co_qn_eq, &
                                      da_b_qn_eq, da_b_amps_law, &
                                      da_b_bpar_eq)
        class(op_mom_maxwells_eq_vspec_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_2d_t), intent(inout) :: da_co_qn_eq
        class(data_array_2d_t), intent(inout) :: da_b_qn_eq
        class(data_array_2d_t), intent(inout) :: da_b_amps_law
        class(data_array_2d_t), intent(inout) :: da_b_bpar_eq

        call handle_error_notimp('vspec', __LINE__, __FILE__)
    end subroutine

end submodule
