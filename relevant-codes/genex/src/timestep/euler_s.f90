submodule (timestep_m) euler_s
    implicit none
contains

    module subroutine initialize_euler(this, dcomm_handler, mesh, dt, &
                                       diag_dir, start_from_checkpoint, &
                                       run_mms, run_test)
        class(euler_t), target, intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        type(mesh_5d_t), pointer, intent(in) :: mesh
        real(kind=GP), intent(in) :: dt
        character(len=*), intent(in) :: diag_dir
        logical, optional, intent(inout) :: start_from_checkpoint
        logical, optional, intent(in) :: run_mms
        logical, optional, intent(in) :: run_test

        integer :: ierr

        this%diag_dir = diag_dir
        this%dt = dt

        ! NOTE: Defaults are set in the declaration (of parent)
        if(present(start_from_checkpoint)) &
            this%start_from_checkpoint = start_from_checkpoint
        if(present(run_mms)) this%run_mms = run_mms

        if(present(run_test)) call this%setup_test_mode(run_test)
        call this%initialize_parent(dcomm_handler, mesh)

        ! If no checkpoint found, start_from_checkpoint will be set to false in
        ! the initialize.
        if(present(start_from_checkpoint)) &
            start_from_checkpoint = this%start_from_checkpoint

        allocate(this%k)
        call this%k%initialize(dcomm_handler)

        ! Finish the initialized timestep
        call this%step_finish(this%state, this%k, ierr)
        if(ierr /= GENEX_SUCCESS) return
    end subroutine

    module subroutine step_euler(this, ierr)
        class(euler_t), target, intent(inout) :: this
        integer, intent(out) :: ierr

        ierr = GENEX_SUCCESS
        this%current_stage = 1

        call this%step_evolve(this%state, this%k, ierr)
        if(ierr /= GENEX_SUCCESS) return
        call this%state%add(this%dt, this%k)

        this%t = this%t + this%dt
        call this%step_finish(this%state, this%k, ierr)
        if(ierr /= GENEX_SUCCESS) return
    end subroutine

end submodule
