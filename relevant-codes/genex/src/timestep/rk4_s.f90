submodule (timestep_m) rk4_s
    implicit none
contains

    module subroutine initialize_rk4(this, dcomm_handler, mesh, dt, &
                                     diag_dir, start_from_checkpoint, &
                                     run_mms, run_test)
        class(rk4_t), target, intent(inout) :: this
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

        this%nodes = [0.0_GP, 0.5_GP, 0.5_GP, 1.0_GP]
        this%weights = [1.0_GP / 6.0_GP, &
                        1.0_GP / 3.0_GP, &
                        1.0_GP / 3.0_GP, &
                        1.0_GP / 6.0_GP]

        if(present(run_test)) call this%setup_test_mode(run_test)
        call this%initialize_parent(dcomm_handler, mesh)

        ! If no checkpoint found, start_from_checkpoint will be set to false in
        ! the initialize.
        if(present(start_from_checkpoint)) &
            start_from_checkpoint = this%start_from_checkpoint

        allocate(this%initial_state)
        allocate(this%increment)
        allocate(this%k)
        call this%initial_state%initialize(dcomm_handler)
        call this%increment%initialize(dcomm_handler)
        call this%k%initialize(dcomm_handler)

        ! Finish the initialized timestep
        call this%step_finish(this%state, this%k, ierr)
        if(ierr /= GENEX_SUCCESS) return
    end subroutine

    module subroutine step_rk4(this, ierr)
        class(rk4_t), target, intent(inout) :: this
        integer, intent(out) :: ierr

        real(kind=GP) :: initial_t
        integer :: j

        ierr = GENEX_SUCCESS
        this%current_stage = 1
        initial_t = this%t

        call this%initial_state%copy(this%state)
        call this%step_evolve(this%initial_state, this%k, ierr)
        if(ierr /= GENEX_SUCCESS) return
        call this%state%add(this%weights(1) * this%dt, this%k)

        do j = 2, 4
            this%current_stage = j
            this%t = initial_t + this%nodes(j) * this%dt
            call this%increment%lin_comb(1.0_GP, this%initial_state, &
                                         this%nodes(j) * this%dt, this%k)
            call this%step_evolve(this%increment, this%k, ierr)
            if(ierr /= GENEX_SUCCESS) return
            call this%state%add(this%weights(j) * this%dt, this%k)
        enddo

        this%t = initial_t + this%dt
        call this%step_finish(this%state, this%k, ierr)
        if(ierr /= GENEX_SUCCESS) return
    end subroutine

end submodule
