submodule (timestep_m) strang_splitting_s
    use genex_status_codes_m, only: GENEX_ERR_TIMESTEP, GENEX_WRN_TIMESTEP
    use params_time_split_m, only: get_time_scheme_collisions, &
                                   get_time_scheme_neutrals, &
                                   get_n_stages_collisions, &
                                   get_n_stages_neutrals, &
                                   get_eta_collisions, &
                                   get_eta_neutrals
    implicit none
contains

    module subroutine initialize_strang(this, dcomm_handler, mesh, dt, &
                                        diag_dir, start_from_checkpoint, &
                                        run_mms, run_test)
        class(strang_splitting_t), target, intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        type(mesh_5d_t), pointer, intent(in) :: mesh
        real(kind=GP), intent(in) :: dt
        character(len=*), intent(in) :: diag_dir
        logical, optional, intent(inout) :: start_from_checkpoint
        logical, optional, intent(in) :: run_mms
        logical, optional, intent(in) :: run_test

        integer :: num_modes
        character(len=:), allocatable :: scheme_coll, scheme_neut

        ! NOTE: Defaults are set in the declaration (of parent)
        if(present(start_from_checkpoint)) &
            this%start_from_checkpoint = start_from_checkpoint
        if(present(run_mms)) this%run_mms = run_mms
        if(present(run_test)) this%run_test = run_test

        call this%rk4_t%initialize(dcomm_handler, mesh, dt, diag_dir, &
                                   this%start_from_checkpoint, &
                                   this%run_mms, &
                                   this%run_test)

        ! Check the amount of substeps present in the splitting scheme
        num_modes = 0
        if(this%apply_plasma)     num_modes = num_modes + 1
        if(this%apply_collisions) num_modes = num_modes + 1
        if(this%apply_neutrals)   num_modes = num_modes + 1
        if(num_modes < 2) then
            call handle_error(&
                    "Strang splitting used with less than 2 operators!", &
                    GENEX_ERR_TIMESTEP, __LINE__, __FILE__)
        endif

        if(.not. this%apply_plasma) then
            call handle_error(&
                    "Strang splitting used with no plasma step. &
                    &For efficiency use rk4 instead!", &
                    GENEX_WRN_TIMESTEP, __LINE__, __FILE__)
        endif

        ! Check schemes used for collisions and neutrals
        scheme_coll = get_time_scheme_collisions()
        scheme_neut = get_time_scheme_neutrals()
        if(.not. (scheme_coll == "rk4" .or. scheme_coll == "rkc")) then
            call handle_error(&
                    "Strang splitting used with unsupported scheme &
                    &for collisions!", &
                    GENEX_ERR_TIMESTEP, __LINE__, __FILE__, &
                    additional_info=error_info_t( &
                        "(was: "//get_time_scheme_collisions()//")"))
        endif
        if(.not. (scheme_neut == "rk4" .or. scheme_neut == "rkc")) then
            call handle_error(&
                    "Strang splitting used with unsupported scheme &
                    &for neutrals!", &
                    GENEX_ERR_TIMESTEP, __LINE__, __FILE__, &
                    additional_info=error_info_t( &
                        "(was: "//get_time_scheme_neutrals()//")"))
        endif

        ! Check number of stages and initialize scheme coefficients
        ! NOTE: We throw warnings if the numbers of stages are beyond
        !       reasonable values. These may be adjusted if it turns out that
        !       a lot of stages need to be used routinely. We want to avoid
        !       having a hard cap on the number of stages to give the
        !       possibility for experimenting, thus we throw a warning only.
        if(get_n_stages_collisions() > 10) then
            call handle_error(&
                    "A large number of stages is used for the time integration &
                    &of collisions. This may heavily impact performance!", &
                    GENEX_WRN_TIMESTEP, __LINE__, __FILE__)
        endif
        if(scheme_coll == "rkc") then
            call this%initialize_rkc(get_n_stages_collisions(), &
                                     get_eta_collisions(), &
                                     this%rkc_coefs_collisions)
        endif
        if(get_n_stages_neutrals() > 20) then
            call handle_error(&
                    "A large number of stages is used for the time integration &
                    &of neutrals. This may heavily impact performance!", &
                    GENEX_WRN_TIMESTEP, __LINE__, __FILE__)
        endif
        if(scheme_neut == "rkc") then
            call this%initialize_rkc(get_n_stages_neutrals(), &
                                     get_eta_neutrals(), &
                                     this%rkc_coefs_neutrals)
        endif

        if(scheme_coll == "rkc" .or. scheme_neut == "rkc") then
            allocate(this%k2)
            allocate(this%increment2)
            call this%k2%initialize(dcomm_handler)
            call this%increment2%initialize(dcomm_handler)
        endif
    end subroutine

    module subroutine step_strang(this, ierr)
        class(strang_splitting_t), target, intent(inout) :: this
        integer, intent(out) :: ierr

        real(kind=GP) :: initial_time, initial_dt
        ! Initial time and timestep when entering the subroutine
        logical :: apply_plasma, apply_collisions, apply_neutrals
        ! Temporary variables to store the initial state of the apply variables
        ! in the timestep that have been set based on the parameters.
        character(len=:), allocatable :: scheme_coll, scheme_neut
        ! Scheme used for collision and neutral step

        ierr = GENEX_SUCCESS

        ! The splitting scheme works by setting the timestep variables for
        ! each step that has been split. This requires saving the initial
        ! state to set it back at the end.
        initial_time     = this%t
        initial_dt       = this%dt
        apply_plasma     = this%apply_plasma
        apply_collisions = this%apply_collisions
        apply_neutrals   = this%apply_neutrals

        scheme_coll = get_time_scheme_collisions()
        scheme_neut = get_time_scheme_neutrals()

        ! NOTE: In some splitting steps we have to reset the
        !       time to the correct one because the steps internally evolve
        !       the time within the rk4 stages and this time is needed for
        !       time dependent operators (MMS).

        ! Evolve neutrals operator with a quarter of time step at t_initial
        this%dt = initial_dt * 0.25_GP
        call step_neutrals(ierr)
        if(ierr /= GENEX_SUCCESS) return

        ! Evolve collisions operator with half time step at t_initial
        this%t  = initial_time
        this%dt = initial_dt * 0.5_GP
        call step_collisions(ierr)
        if(ierr /= GENEX_SUCCESS) return

        ! Evolve neutrals operator with a quarter of time step at
        ! t_initial + dt / 4
        this%t  = initial_time + initial_dt * 0.25_GP
        this%dt = initial_dt * 0.25_GP
        call step_neutrals(ierr)
        if(ierr /= GENEX_SUCCESS) return

        ! Evolve Vlasov operator with full time step at t_initial
        this%t  = initial_time
        this%dt = initial_dt
        call step_plasma(ierr)
        if(ierr /= GENEX_SUCCESS) return

        ! Evolve neutrals operator with a quarter of time step at
        ! t_initial + dt / 2
        this%t  = initial_time + initial_dt * 0.5_GP
        this%dt = initial_dt * 0.25_GP
        call step_neutrals(ierr)
        if(ierr /= GENEX_SUCCESS) return

        ! Evolve collisions operator with half time step at t_initial + dt / 2
        this%t  = initial_time + initial_dt * 0.5_GP
        this%dt = initial_dt * 0.5_GP
        call step_collisions(ierr)
        if(ierr /= GENEX_SUCCESS) return

        ! Evolve neutrals operator with a quarter of time step at
        ! t_initial + 3 dt / 4
        this%t  = initial_time + initial_dt * 0.75_GP
        this%dt = initial_dt * 0.25_GP
        call step_neutrals(ierr)
        if(ierr /= GENEX_SUCCESS) return

        this%dt               = initial_dt
        this%t                = initial_time + initial_dt
        this%apply_plasma     = apply_plasma
        this%apply_collisions = apply_collisions
        this%apply_neutrals   = apply_neutrals

    contains

        subroutine step_plasma(ierr)
            integer, intent(out) :: ierr

            if(apply_plasma) then
                this%apply_plasma     = .true.
                this%apply_collisions = .false.
                this%apply_neutrals   = .false.

                ! NOTE: Currently the plasma step is forced to be RK4
                !       even if splitting is used.
                call this%rk4_t%step(ierr)
            endif
        end subroutine

        subroutine step_collisions(ierr)
            integer, intent(out) :: ierr

            if(apply_collisions) then
                this%apply_plasma     = .false.
                this%apply_collisions = .true.
                this%apply_neutrals   = .false.

                if(scheme_coll == "rk4") then
                    call this%rk4_t%step(ierr)

                else if(scheme_coll == "rkc") then
                    call this%step_rkc(this%rkc_coefs_collisions, ierr)

                endif
            endif
        end subroutine

        subroutine step_neutrals(ierr)
            integer, intent(out) :: ierr

            if(apply_neutrals) then
                this%apply_plasma     = .false.
                this%apply_collisions = .false.
                this%apply_neutrals   = .true.

                if(scheme_neut == "rk4") then
                    call this%rk4_t%step(ierr)

                else if(scheme_neut == "rkc") then
                    call this%step_rkc(this%rkc_coefs_neutrals, ierr)

                endif
            endif
        end subroutine

    end subroutine

end submodule
