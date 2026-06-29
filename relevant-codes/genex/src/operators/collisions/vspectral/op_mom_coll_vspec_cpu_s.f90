submodule(op_mom_coll_m) op_mom_coll_vspec_cpu_s
    use genex_error_handling_m, only: handle_error_notimp

    use MPI
    use, intrinsic :: iso_fortran_env
    use genex_fortran_env_m, only: MPI_GP
    use genex_error_handling_m, only: handle_error_vspec, &
                                      VSPEC_ERR_INITIALIZE
    use params_species_m, only: get_mass, get_charge
    use params_normalization_m, only: get_L_ref, get_rho_ref
    use params_collisions_m, only: get_dens_floor, get_temp_floor
    use params_mesh_m, only: get_use_vspectral
    use op_set_uniform_m, only: op_set_uniform_cpu_t
    use profiler_m, only: profiler_start, profiler_stop, &
                          profiler_start_allreduce, profiler_stop_allreduce

    implicit none

contains
    module subroutine initialize_vspec_cpu(this, dcomm_handler, mesh)
        class(op_mom_coll_vspec_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        call handle_error_notimp('vspec', __LINE__, __FILE__)
    end subroutine

    module subroutine apply_vspec_cpu(this, da_f_in, da_moments)
        class(op_mom_coll_vspec_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_4d_t), intent(inout) :: da_moments

        call handle_error_notimp('vspec', __LINE__, __FILE__)
    end subroutine

end submodule
