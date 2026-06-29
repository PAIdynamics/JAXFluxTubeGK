submodule (op_set_initial_condition_m) op_set_init_cond_vspec_s
    use genex_error_handling_m, only: handle_error_notimp
    use mpi
    use genex_fortran_env_m, only: MPI_GP, GP_EPS
    use, intrinsic :: iso_fortran_env
    use math_m, only: PI
    use genex_error_handling_m, only: handle_error, &
                                      handle_error_vspec, &
                                      VSPEC_ERR_INITIALIZE
    use genex_status_codes_m, only: GENEX_ERR_PARAMETERS
    use params_species_m, only: get_mass, get_charge
    use params_mesh_m, only: get_n_points_sp, get_use_vspectral
    use params_normalization_m, only: get_L_ref, get_rho_ref
    use op_set_uniform_m, only: op_set_uniform_cpu_t
    use params_initial_condition_m, only: get_initial_perturbation

    use equilibrium_factory_m, only: NUMERICAL
    use descriptors_m, only: DISTRICT_PRIVFLUX, DISTRICT_SOL, DISTRICT_WALL, &
                             DISTRICT_DOME, DISTRICT_OUT
    implicit none

contains
    module subroutine initialize_vspec_cpu(this, dcomm_handler, mesh, &
                                           dist_initial_container)
        class(op_set_initial_condition_vspec_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(in) :: mesh
        type(dist_initial_container_t), dimensiOn(:), allocatable, &
                                        intent(in) :: dist_initial_container

        call handle_error_notimp('vspec', __LINE__, __FILE__)
    end subroutine

    module subroutine apply_vspec_cpu(this, da_f_inout)
        class(op_set_initial_condition_vspec_cpu_t), &
                                                   target, intent(inout) :: this
        class(data_array_5d_t), intent(inout) :: da_f_inout

        call handle_error_notimp('vspec', __LINE__, __FILE__)
    end subroutine

end submodule
