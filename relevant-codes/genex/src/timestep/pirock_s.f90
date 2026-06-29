submodule(timestep_m) pirock_s
    use genex_error_handling_m, only: handle_error_notimp
    use pirock_coefs_m, only: compute_coefs
    use params_time_loop_m, only: get_n_stages_pirock

    implicit none
contains
    module subroutine initialize_pirock(this, dcomm_handler, mesh, dt, &
                                        diag_dir, start_from_checkpoint, &
                                        run_mms, run_test)
        class(pirock_t), target, intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        type(mesh_5d_t), pointer, intent(in) :: mesh
        real(kind=GP), intent(in) :: dt
        character(len=*), intent(in) :: diag_dir
        logical, optional, intent(inout) :: start_from_checkpoint
        logical, optional, intent(in) :: run_mms
        logical, optional, intent(in) :: run_test

        call handle_error_notimp('timestep', __LINE__, __FILE__)
    end subroutine

    module subroutine step_pirock(this, ierr)
        class(pirock_t), target, intent(inout) :: this
        integer, intent(out) :: ierr

        call handle_error_notimp('timestep', __LINE__, __FILE__)
    end subroutine

end submodule
