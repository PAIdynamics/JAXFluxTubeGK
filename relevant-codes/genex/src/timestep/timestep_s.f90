submodule (timestep_m) timestep_s
    use params_mesh_m, only: get_use_vspectral
    use params_collisions_m, only: get_coll_type
    use params_gyrokinetic_system_m, only: get_with_bpar
    use params_bnd_cond_m, only: get_bnd_cond_type
    use params_gpu_offload_m, only: get_parallax_gpu_offload_backend, &
                                    PARALLAX_BACKEND_CPU
    use params_diagnostics_m, only: get_diagnose_tpc
    use genex_status_codes_m, only: GENEX_ERR_FIELD_SOLVE, &
                                    GENEX_WRN_RESIDUUM_NAN, &
                                    GENEX_WRN_TIMESTEP
    implicit none

contains

    module subroutine save_checkpoint(this, ierr)
        class(timestep_t), target, intent(inout) :: this
        integer, intent(inout) :: ierr

        call profiler_start("save_checkpoint", ierr, set_path=.true.)

        ! NOTE: Remove the following update if save_checkpoint become
        !       GPU-supported
        call this%state%update_host_dist_func()
        call this%state%save_checkpoint(this%checkpoint_file, this%t)
        if (this%apply_neutrals) then
            call this%state%save_checkpoint_neut( &
                this%checkpoint_neut_file, &
                this%t)
        end if

        call profiler_stop( "save_checkpoint", ierr, set_path=.true.)
    end subroutine

    module subroutine diagnose_mom_0d(this)
        class(timestep_t), target, intent(inout) :: this

        ! NOTE: Remove the following update host if diagnose_mom_0d/2d/em_field
        !       become GPU-supported. E_par on CPU memory is already updated.
        call this%state%update_host_dist_func()
        call this%state%update_host_em_fields()
        ! TODO: Uncomment the following update
        !       if op_solve_ohms_law become GPU-supported
        ! call this%state%update_host_E_par()

        call this%state%apply(this%op_mom_0d, this%mom_0d_file, this%t)
        if(this%run_mms) then
            call this%state%apply(this%op_mms_error, this%t)
        endif
    end subroutine

    module subroutine diagnose_mom_2d(this, diagnose_tpc)
        class(timestep_t), target, intent(inout) :: this
        logical, intent(in) :: diagnose_tpc

        type(mom_2d_file_t), pointer :: file_to_use

        if (diagnose_tpc) then
            file_to_use => this%mom_2d_tpc_file
        else
            file_to_use => this%mom_2d_file
        endif

        call this%state%apply(this%op_mom_2d, file_to_use, this%t, diagnose_tpc)

    end subroutine

    module subroutine diagnose_neutrals(this)
        class(timestep_t), target, intent(inout) :: this
        if(this%apply_neutrals) then
            call this%state%save_neutrals(this%neutrals_file, this%t)
        end if
    end subroutine

    module subroutine diagnose_em_fields(this)
        class(timestep_t), target, intent(inout) :: this

        call this%state%save_em_fields(this%em_fields_file, this%t)
    end subroutine

    module subroutine diagnose_all(this, ierr)
        class(timestep_t), intent(inout) :: this
        integer, intent(inout) :: ierr

        logical, parameter :: set_path = .true.
        logical :: diagnose_tpc

        diagnose_tpc = get_diagnose_tpc()

        call profiler_start("diagnose_mom_0d",    ierr, set_path)
        call this%diagnose_mom_0d
        call profiler_stop( "diagnose_mom_0d",    ierr, set_path)

        call profiler_start("diagnose_mom_2d",    ierr, set_path)
        call this%diagnose_mom_2d(.false.)
        call profiler_stop( "diagnose_mom_2d",    ierr, set_path)

        call profiler_start("diagnose_em_fields", ierr, set_path)
        call this%diagnose_em_fields
        call profiler_stop( "diagnose_em_fields", ierr, set_path)

        call profiler_start("diagnose_neutrals",  ierr, set_path)
        call this%diagnose_neutrals
        call profiler_stop( "diagnose_neutrals",  ierr, set_path)

        if (diagnose_tpc) then
            call profiler_start("diagnose_mom_2d_tpc", ierr, set_path)
            call this%diagnose_mom_2d(diagnose_tpc)
            call profiler_stop( "diagnose_mom_2d_tpc", ierr, set_path)
        endif

    end subroutine

    module subroutine exchange(this, s_in)
        class(timestep_t), intent(inout) :: this
        type(state_vector_t), intent(inout) :: s_in

        integer :: ierr

        ierr = GENEX_SUCCESS

        call profiler_start("exchange", ierr, set_path=.true.)
        call s_in%exchange()
        call profiler_stop("exchange", ierr, set_path=.true.)

100     if (ierr /= GENEX_SUCCESS) then
            call handle_error(&
                    "Exchange in timestep failed!", ierr, __LINE__, __FILE__)
        endif
    end subroutine

    module subroutine exchange_neut(this, s_in)
        class(timestep_t), intent(inout) :: this
        type(state_vector_t), intent(inout) :: s_in

        integer :: ierr

        ierr = GENEX_SUCCESS

        call profiler_start("exchange", ierr, set_path=.true.)
        call s_in%exchange_neut()
        call profiler_stop("exchange", ierr, set_path=.true.)

150     if (ierr /= GENEX_SUCCESS) then
            call handle_error(&
                    "Exchange neutrals in timestep failed!", &
                    ierr, __LINE__, __FILE__)
        endif
    end subroutine

    module subroutine exchange_vp_mu(this, s_in)
        class(timestep_t), intent(inout) :: this
        type(state_vector_t), intent(inout) :: s_in

        integer :: ierr

        ierr = GENEX_SUCCESS

        call profiler_start("exchange", ierr, set_path=.true.)

        call s_in%start_exchange_vp()
        call s_in%start_exchange_mu()
        call s_in%finish_exchange_vp()
        call s_in%finish_exchange_mu()

        call profiler_stop("exchange", ierr, set_path=.true.)

200     if (ierr /= GENEX_SUCCESS) then
            call handle_error(&
                    "Exchange vp mu in timestep failed!", ierr, __LINE__, &
                    __FILE__)
        endif
    end subroutine

    module subroutine calc_rhs_static_core(this, s_in, s_out)
        class(timestep_t), target, intent(inout) :: this
        type(state_vector_t), target, intent(inout) :: s_in
        type(state_vector_t), target, intent(inout) :: s_out

        integer :: s, ierr

        ierr = GENEX_SUCCESS

        if(this%run_mms) then
            ! We set the solution of the Maxwell field solvers to the exact one
            ! to decouple the MMS tests of the Vlasov and field solver part.

            call profiler_start("buffer_mms", ierr, set_path=.true.)
            call profiler_start("buffer_mms_solution_maxwells_eq", ierr)
            call s_in%buffer_mms_solution_maxwells_eq(store_to_buffer=.true.)
            call profiler_stop("buffer_mms_solution_maxwells_eq", ierr)
            call profiler_stop("buffer_mms", ierr, set_path=.true.)

            call profiler_start("solve_maxwells_equation", ierr, &
                                set_path=.true.)
            call profiler_start("op_mms_solution_maxwells_eq", ierr)
            call s_in%apply(this%op_mms_solution_maxwells_eq, this%t)
            call profiler_stop("op_mms_solution_maxwells_eq", ierr)
            call profiler_stop("solve_maxwells_equation", ierr, set_path=.true.)

            call s_in%update_device_em_fields()
        endif

        call profiler_start("calc_rhs_static", ierr, set_path=.true.)

        ! Calculate the static part of the rhs of the Vlasov equation
        call profiler_start("op_rhs_vlasov_eq_static", ierr)
        call s_in%apply(this%op_rhs_vlasov_eq_static, s_out)
        call profiler_stop("op_rhs_vlasov_eq_static", ierr)

        if(this%apply_sources) then
            call profiler_start("op_source", ierr)
            do s = 1, this%n_source
                call s_out%apply(this%op_source_container(s)%op_source)
            enddo
            call profiler_stop("op_source", ierr)
        endif

        if(this%run_mms) then
            ! NOTE: Remove these following updates
            !       if op_mms_source_vlasov_eq becomes GPU-supported
            call s_out%update_host_dist_func()

            call profiler_start("op_mms_source", ierr)
            call s_out%apply(this%op_mms_source_vlasov_eq, this%t)
            call profiler_stop("op_mms_source", ierr)

            call s_out%update_device_dist_func()
        endif

        call profiler_stop("calc_rhs_static", ierr, set_path=.true.)

        if(this%run_mms) then
            call profiler_start("buffer_mms", ierr, set_path=.true.)
            call profiler_start("buffer_mms_solution_maxwells_eq", ierr)
            call s_in%buffer_mms_solution_maxwells_eq(store_to_buffer=.false.)
            call profiler_stop("buffer_mms_solution_maxwells_eq", ierr)
            call profiler_stop("buffer_mms", ierr, set_path=.true.)
        endif

300     if (ierr /= GENEX_SUCCESS) then
            call handle_error(&
                    "Calc rhs static in timestep failed!", ierr, __LINE__, &
                    __FILE__)
        endif
    end subroutine

    module subroutine apply_bnd_cond_core(this, s_inout)
        class(timestep_t), target, intent(inout) :: this
        type(state_vector_t), target, intent(inout) :: s_inout

        character(len=:), allocatable :: bnd_cond_type
        logical :: bnd_cond_support_gpu
        integer :: ierr

        ierr = GENEX_SUCCESS
        bnd_cond_support_gpu = .false.

        if(this%run_mms) return

        ! Check if op_bnd_cond is GPU-supported
        bnd_cond_type = get_bnd_cond_type()
        if(.not. get_use_vspectral()) then
            select case(bnd_cond_type)
                case('dirichlet')
                    bnd_cond_support_gpu = .true.
                case('neumann')
                    bnd_cond_support_gpu = .true.
            end select
        endif

        call profiler_start("apply_bnd_cond", ierr, set_path=.true.)

        ! NOTE: Remove the updates before and after the call if op_bnd_cond
        !       becomes GPU-supported
        if(.not. bnd_cond_support_gpu) then
            call s_inout%update_host_dist_func()
            call s_inout%update_host_maxwells_buffers()
            call s_inout%update_host_ohms_buffers()
        endif

        call profiler_start("op_bnd_cond", ierr)
        call s_inout%apply(this%op_bnd_cond, this%t)
        call profiler_stop("op_bnd_cond", ierr)

        if(.not. bnd_cond_support_gpu) then
            call s_inout%update_device_dist_func()
            call s_inout%update_device_maxwells_buffers()
            call s_inout%update_device_ohms_buffers()
        endif

        call profiler_stop("apply_bnd_cond", ierr, set_path=.true.)

310     if (ierr /= GENEX_SUCCESS) then
            call handle_error(&
                 "Apply boundary condition in timestep failed!", &
                 ierr, __LINE__, __FILE__)
        endif

    end subroutine

    module subroutine calc_rhs_dynamic_core(this, s_in, s_out)
        class(timestep_t), target, intent(inout) :: this
        type(state_vector_t), target, intent(inout) :: s_in
        type(state_vector_t), target, intent(inout) :: s_out

        integer :: ierr

        ierr = GENEX_SUCCESS

        if(this%run_mms) then
            ! We set the solution of the Ohm field solver to the exact one
            ! to decouple the MMS tests of the Vlasov and field solver part

            call profiler_start("buffer_mms", ierr, set_path=.true.)
            call profiler_start("buffer_mms_solution_ohms_law", ierr)
            call s_in%buffer_mms_solution_ohms_law(store_to_buffer=.true.)
            call profiler_stop("buffer_mms_solution_ohms_law", ierr)
            call profiler_stop("buffer_mms", ierr, set_path=.true.)

            call profiler_start("solve_ohms_law", ierr, set_path=.true.)
            call profiler_start("op_mms_solution_ohms_law", ierr)
            call s_in%apply(this%op_mms_solution_ohms_law, this%t)
            call profiler_stop("op_mms_solution_ohms_law", ierr)
            call profiler_stop("solve_ohms_law", ierr, set_path=.true.)

            ! NOTE: Remove the following update
            !       if op_mms_solution_ohms_law becomes GPU-supported
            call s_in%update_device_E_par()
        endif

        ! NOTE: The dynamic part of the Vlasov equation needs the evaluation of
        !       all terms that may generate currents that lead to induction.
        !       This includes sources and collisions.

        ! Calculate the dynamic part of the rhs of the Vlasov equation
        call profiler_start("calc_rhs_dynamic", ierr, set_path=.true.)
        call profiler_start("op_rhs_vlasov_eq_dynamic", ierr)
        call s_in%apply(this%op_rhs_vlasov_eq_dynamic, s_out)
        call profiler_stop("op_rhs_vlasov_eq_dynamic", ierr)
        call profiler_stop("calc_rhs_dynamic", ierr, set_path=.true.)

        if(this%run_mms) then
            call profiler_start("buffer_mms", ierr, set_path=.true.)
            call profiler_start("buffer_mms_solution_ohms_law", ierr)
            call s_in%buffer_mms_solution_ohms_law(store_to_buffer=.false.)
            call profiler_stop("buffer_mms_solution_ohms_law", ierr)
            call profiler_stop("buffer_mms", ierr, set_path=.true.)
        endif

400     if (ierr /= GENEX_SUCCESS) then
            call handle_error(&
                    "Calc rhs dynamic in timestep failed!", ierr, __LINE__, &
                    __FILE__)
        endif
    end subroutine

    module subroutine calc_collisions_core(this, s_in, s_out)
        class(timestep_t), target, intent(inout) :: this
        type(state_vector_t), target, intent(inout) :: s_in
        type(state_vector_t), target, intent(inout) :: s_out

        character(len=:), allocatable :: coll_type
        logical :: coll_support_gpu
        integer :: ierr

        ierr = GENEX_SUCCESS
        coll_support_gpu = .false.

        ! Check if op_coll is GPU-supported
        coll_type = get_coll_type()
        if(coll_type == "bgk") then
            coll_support_gpu = .true.
        endif

        call profiler_start("calc_collisions", ierr, set_path=.true.)

        ! NOTE: Remove these following updates
        !       if op_mom_coll and op_coll become GPU-supported
        if(.not. coll_support_gpu) then
            call s_in%update_host_dist_func()
            call s_out%update_host_dist_func()
        endif

        call profiler_start("op_mom_coll", ierr, set_path=.true.)
        call s_in%apply(this%op_mom_coll, s_out)
        call profiler_stop("op_mom_coll", ierr, set_path=.true.)

        call profiler_start("op_coll", ierr, set_path=.true.)
        call s_in%apply(this%op_coll, s_out)
        call profiler_stop("op_coll", ierr, set_path=.true.)

        ! Only update GPU memory if neutrals operators are not used
        if(.not. coll_support_gpu .and. .not. this%apply_neutrals) then
            call s_out%update_device_dist_func()
        endif

        call profiler_stop("calc_collisions", ierr, set_path=.true.)

500     if (ierr /= GENEX_SUCCESS) then
            call handle_error(&
                    "Calc collisions in timestep failed!", ierr, __LINE__, &
                    __FILE__)
        endif
    end subroutine

    module subroutine solve_maxwells_eq_core(this, s_inout, ierr)
        class(timestep_t), target, intent(inout) :: this
        type(state_vector_t), target, intent(inout) :: s_inout
        integer, intent(out) :: ierr
        real(kind=GP) :: residuum

        ierr = GENEX_SUCCESS

        if(this%run_mms) then
            ! Set the distribution function and the potentials to the exact
            ! solution on the boundary
            ! NOTE: This should be done before the buffering below, so that the
            !       boundary condition is also set correctly for the Vlasov
            !       part!
            call profiler_start("op_bnd_cond_mms", ierr)

            ! NOTE: Remove the updates before and after the call if op_bnd_cond
            !       becomes GPU-supported
            call s_inout%update_host_dist_func()
            call s_inout%update_host_maxwells_buffers()
            call s_inout%update_host_ohms_buffers()

            call s_inout%apply(this%op_bnd_cond, this%t)

            call s_inout%update_device_dist_func()
            call s_inout%update_device_maxwells_buffers()
            call s_inout%update_device_ohms_buffers()

            call profiler_stop("op_bnd_cond_mms", ierr)

            ! Decouple the MMS test of Maxwell's equation by setting the exact
            ! solution of the Vlasov part.

            call profiler_start("buffer_mms", ierr, set_path=.true.)
            call profiler_start("buffer_mms_solution_vlasov_eq", ierr)
            call s_inout%buffer_mms_solution_vlasov_eq(store_to_buffer=.true.)
            call profiler_stop("buffer_mms_solution_vlasov_eq", ierr)
            call profiler_stop("buffer_mms", ierr, set_path=.true.)

            call profiler_start("calc_rhs_static", ierr, set_path=.true.)
            call profiler_start("op_mms_solution_vlasov_eq", ierr)
            call s_inout%apply(this%op_mms_solution_vlasov_eq, this%t)
            call profiler_stop("op_mms_solution_vlasov_eq", ierr)
            call profiler_stop("calc_rhs_static", ierr, set_path=.true.)

            ! NOTE: Remove the following update if op_mms_solution_vlasov_eq
            !       becomes GPU-supported
            call s_inout%update_device_dist_func()
        endif

        call profiler_start("solve_maxwells_equation", ierr, set_path=.true.)

        ! Calculate the moments of the distribution function required for
        ! the solution of the quasi-neutrality equation and Ampere's law
        call profiler_start("op_mom_maxwells_eq", ierr, set_path=.true.)
        call s_inout%apply(this%op_mom_maxwells_eq)
        call profiler_stop("op_mom_maxwells_eq", ierr, set_path=.true.)

        if(this%run_mms) then
            call s_inout%update_host_maxwells_buffers()
            call profiler_start("op_mms_source", ierr)
            call s_inout%apply(this%op_mms_source_maxwells_eq, this%t)
            call profiler_stop("op_mms_source", ierr)
        endif

        if(get_parallax_gpu_offload_backend() == PARALLAX_BACKEND_CPU &
           .and. .not. this%run_mms) then
            call s_inout%update_host_maxwells_buffers()
        endif

        ! Solve the quasi-neutrality equation
        call profiler_start("op_solve_qn_eq", ierr, set_path=.true.)
        call s_inout%apply(this%op_solve_qn_eq, residuum)
        call profiler_stop("op_solve_qn_eq", ierr, set_path=.true.)

        if(isnan(residuum)) then
            ierr = GENEX_ERR_FIELD_SOLVE
            if(this%dcomm_handler%is_master()) then
                call handle_error(&
                        "Residuum of op_solve_qn_eq is NaN!", &
                        GENEX_WRN_RESIDUUM_NAN, __LINE__, __FILE__)
            endif
            goto 600
        endif

        ! Solve parallel Ampere's law
        call profiler_start("op_solve_amps_law", ierr, set_path=.true.)
        call s_inout%apply(this%op_solve_amps_law, residuum)
        call profiler_stop("op_solve_amps_law", ierr, set_path=.true.)

        if(get_parallax_gpu_offload_backend() == PARALLAX_BACKEND_CPU) then
            call s_inout%update_device_em_fields()
        endif

        if(isnan(residuum)) then
            ierr = GENEX_ERR_FIELD_SOLVE
            if(this%dcomm_handler%is_master()) then
                call handle_error(&
                        "Residuum of op_solve_amps_law is NaN!", &
                        GENEX_WRN_RESIDUUM_NAN, __LINE__, __FILE__)
            endif
            goto 600
        endif

        ! Solve B parallel equation
        if(get_with_bpar()) then
            call s_inout%update_device_maxwells_buffers_bpar_only()

            call profiler_start("op_solve_bpar_eq", ierr, set_path=.true.)
            call s_inout%apply(this%op_solve_bpar_eq)
            call profiler_stop("op_solve_bpar_eq", ierr, set_path=.true.)
        end if

        call profiler_stop("solve_maxwells_equation", ierr, set_path=.true.)

        if(this%run_mms) then
            call profiler_start("buffer_mms", ierr, set_path=.true.)
            call profiler_start("buffer_mms_solution_vlasov_eq", ierr)
            call s_inout%buffer_mms_solution_vlasov_eq(store_to_buffer=.false.)
            call profiler_stop("buffer_mms_solution_vlasov_eq", ierr)
            call profiler_stop("buffer_mms", ierr, set_path=.true.)
        endif

600     if (ierr /= GENEX_SUCCESS) then
            call handle_error(&
                    "Solve maxwells eq in timestep failed!", &
                    GENEX_WRN_TIMESTEP, __LINE__, __FILE__)
            return
        endif
    end subroutine

    module subroutine solve_ohms_law_core(this, dsdt_in, s_inout, ierr)
        class(timestep_t), target, intent(inout) :: this
        type(state_vector_t), target, intent(inout) :: dsdt_in
        type(state_vector_t), target, intent(inout) :: s_inout
        integer, intent(out) :: ierr
        real(kind=GP) :: residuum

        ierr = GENEX_SUCCESS

        if(this%run_mms) then
            ! NOTE: For MMS, the boundary condition of s_inout is already set in
            !       the Maxwell's equations solve. We do not need to set
            !       boundary conditions on dsdt_in because it is only needed to
            !       calculate the moments of Ohms's law.

            ! Decouple parts of the MMS test of Ohm's law by setting the exact
            ! solution of the Vlasov part on s_inout.

            ! NOTE: We can not fully decouple the MMS test of Ohm's law from
            !       the Vlasov part since we do not have exact solutions for
            !       (df/dt)^* in dsdt_in.

            call profiler_start("buffer_mms", ierr, set_path=.true.)
            call profiler_start("buffer_mms_solution_vlasov_eq", ierr)
            call s_inout%buffer_mms_solution_vlasov_eq(store_to_buffer=.true.)
            call profiler_stop("buffer_mms_solution_vlasov_eq", ierr)
            call profiler_stop("buffer_mms", ierr, set_path=.true.)

            call profiler_start("calc_rhs_static", ierr, set_path=.true.)
            call profiler_start("op_mms_solution_vlasov_eq", ierr)
            call s_inout%apply(this%op_mms_solution_vlasov_eq, this%t)
            call profiler_stop("op_mms_solution_vlasov_eq", ierr)
            call profiler_stop("calc_rhs_static", ierr, set_path=.true.)

            ! NOTE: Remove this update if op_mms_solution_vlasov_eq becomes
            !       GPU-supported
            call s_inout%update_device_dist_func()
        endif

        call profiler_start("solve_ohms_law", ierr, set_path=.true.)

        ! Calculate the moments of the distribution function required for
        ! the solution of Ohm's law
        call profiler_start("op_mom_ohms_law", ierr, set_path=.true.)
        call s_inout%apply(this%op_mom_ohms_law, dsdt_in)
        call profiler_stop("op_mom_ohms_law", ierr, set_path=.true.)

        if(get_parallax_gpu_offload_backend() == PARALLAX_BACKEND_CPU) then
            call s_inout%update_host_ohms_buffers()
        endif

        if(this%run_mms) then
            call profiler_start("op_mms_source", ierr)
            call s_inout%apply(this%op_mms_source_ohms_law, this%t)
            call profiler_stop("op_mms_source", ierr)
        endif

        ! Solve Ohm's law
        call profiler_start("op_solve_ohms_law", ierr, set_path=.true.)
        call s_inout%apply(this%op_solve_ohms_law, residuum)
        call profiler_stop("op_solve_ohms_law", ierr, set_path=.true.)

        if(get_parallax_gpu_offload_backend() == PARALLAX_BACKEND_CPU) then
            call s_inout%update_device_E_par()
        endif

        if(isnan(residuum)) then
            ierr = GENEX_ERR_FIELD_SOLVE
            if(this%dcomm_handler%is_master()) then
                call handle_error(&
                        "Residuum of op_solve_ohms_law is NaN!", &
                        GENEX_WRN_RESIDUUM_NAN, __LINE__, __FILE__)
            endif
            goto 700
        endif

        call profiler_stop("solve_ohms_law", ierr, set_path=.true.)

        if(this%run_mms) then
            call profiler_start("buffer_mms", ierr, set_path=.true.)
            call profiler_start("buffer_mms_solution_vlasov_eq", ierr)
            call s_inout%buffer_mms_solution_vlasov_eq(store_to_buffer=.false.)
            call profiler_stop("buffer_mms_solution_vlasov_eq", ierr)
            call profiler_stop("buffer_mms", ierr, set_path=.true.)
        endif

700     if (ierr /= GENEX_SUCCESS) then
            call handle_error(&
                    "Solve ohms law in timestep failed!", &
                    GENEX_WRN_TIMESTEP, __LINE__, __FILE__)
            return
        endif
    end subroutine

    module subroutine calc_neutrals_core(this, s_in, s_out)
        class(timestep_t), target, intent(inout) :: this
        type(state_vector_t), target, intent(inout) :: s_in
        type(state_vector_t), target, intent(inout) :: s_out

        integer :: ierr

        ierr = GENEX_SUCCESS

        call profiler_start("calc_neutrals", ierr, set_path=.true.)

        ! Only update CPU memory if collision operators are not used
        ! because in that case the update has been already performed
        ! NOTE: Remove these following updates
        !       if neutrals operators become GPU-supported
        if(.not. this%apply_collisions) then
            call s_in%update_host_dist_func()
            call s_out%update_host_dist_func()
        endif

        call profiler_start("op_neut_mom", ierr, set_path=.true.)
        call s_in%apply(this%op_neut_mom, s_out)
        call profiler_stop("op_neut_mom", ierr, set_path=.true.)
        call profiler_start("op_neut_coupling_pn", ierr, set_path=.true.)
        call s_in%apply(this%op_neut_coupling_pn, s_out)
        call profiler_stop("op_neut_coupling_pn", ierr, set_path=.true.)
        call profiler_start("op_neut_evolve", ierr, set_path=.true.)
        call s_in%apply(this%op_neut_evolve, s_out)
        call profiler_stop("op_neut_evolve", ierr, set_path=.true.)
        call profiler_start("op_neut_coupling_np", ierr, set_path=.true.)
        call s_in%apply(this%op_neut_coupling_np, s_out)
        call profiler_stop("op_neut_coupling_np", ierr, set_path=.true.)

        call profiler_stop("calc_neutrals", ierr, set_path=.true.)

        call s_out%update_device_dist_func()

800     if (ierr /= GENEX_SUCCESS) then
            call handle_error(&
                    "Calc neutrals in timestep failed!", ierr, __LINE__, &
                    __FILE__)
        endif
    end subroutine

    module subroutine step_evolve(this, s_in, s_out, ierr)
        class(timestep_t), target, intent(inout) :: this
        type(state_vector_t), target, intent(inout) :: s_in
        type(state_vector_t), target, intent(inout) :: s_out
        integer, intent(out) :: ierr

        ierr = GENEX_SUCCESS

        ! NOTE: To ensure that they are correctly initialized in all possible
        !       combinations of apply_plasma, apply_collisions,
        !       and apply_neutrals, we set both the output plasma distribution
        !       function and the neutrals state (if applicable) to zero.
        call s_out%set_uniform(0.0_GP)

        if(this%apply_plasma) then
            if(this%current_stage /= 1) then
                call this%apply_bnd_cond(s_in)
                call this%solve_maxwells_eq(s_in, ierr)
            endif
            if(ierr /= GENEX_SUCCESS) return
            call this%exchange(s_in)
            call this%calc_rhs_static(s_in, s_out)
        else
            ! NOTE: Collisions only require vp_mu exchange only in the case
            !       without the full exchange.
            if(this%apply_collisions) call this%exchange_vp_mu(s_in)
        end if

        if(this%apply_collisions) then
            call this%calc_collisions(s_in, s_out)
        end if

        if(this%apply_neutrals) then
            call this%exchange_neut(s_in)
            call this%calc_neutrals(s_in, s_out)
        end if

        if(this%apply_plasma) then
            call this%solve_ohms_law(s_out, s_in, ierr)
            if(ierr /= GENEX_SUCCESS) return
            call this%calc_rhs_dynamic(s_in, s_out)
        end if
    end subroutine

    module subroutine step_finish(this, s_inout, s_buffer, ierr)
        class(timestep_t), target, intent(inout) :: this
        type(state_vector_t), target, intent(inout) :: s_inout
        type(state_vector_t), target, intent(inout) :: s_buffer
        integer, intent(out) :: ierr

        ierr = GENEX_SUCCESS

        if(.not. this%apply_plasma) return

        call this%apply_bnd_cond(s_inout)
        call this%solve_maxwells_eq(s_inout, ierr)
        if(ierr /= GENEX_SUCCESS) return

        ! NOTE: For MMS test we need to run rhs_static and Ohm's law once
        !       after the simulation to get the correct E_par for the last
        !       time step.
        if(this%run_mms .and. this%final_time_reached()) then
            call this%exchange(s_inout)
            call this%calc_rhs_static(s_inout, s_buffer)
            call this%solve_ohms_law(s_buffer, s_inout, ierr)
            if(ierr /= GENEX_SUCCESS) return
        end if
    end subroutine

end submodule
