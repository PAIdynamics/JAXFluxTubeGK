submodule (state_vector_m) state_vector_s
    use MPI
    use genex_fortran_env_m, only: MPI_GP
    use dimensions_m, only: DIM_RZ, DIM_PHI, DIM_SP, DIM_MU, DIM_VP
    use params_collisions_m, only: get_coll_type
    use params_mesh_m, only: get_n_points_sp, get_use_vspectral
    use params_neutrals_m, only: get_n_points_neut
    use params_neutrals_config_m, only: get_use_neutrals
    use params_time_loop_m, only: get_run_mms
    use profiler_m, only: profiler_start, profiler_stop
    use logger_m, only: logger_get_info_channel
    use params_gpu_offload_m, only : get_use_gpu_offload
    use params_diagnostics_m, only: get_diagnose_tpc

    implicit none

contains

    module subroutine initialize(this, dcomm_handler)
        class(state_vector_t), target, intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler

        integer :: lb_2d(2), ub_2d(2), lb_stripped_2d(2), ub_stripped_2d(2)
        integer :: lb_mixed_2d(2), ub_mixed_2d(2)
        integer :: shp(5)

        this%dcomm_handler => dcomm_handler

        ! Allocate object with CPU or GPU class depends on the runtime mode
        if(get_use_gpu_offload()) then
#ifdef ENABLE_GPU
            allocate(op_axpy_gpu_t :: this%op_add)
            allocate(op_copy_gpu_t :: this%op_copy)
            allocate(op_lin_comb_gpu_t :: this%op_lin_comb)
            allocate(op_set_uniform_gpu_t :: this%op_set_uniform)
            allocate(data_storage_gpu_5d_t :: this%dist_func)
            allocate(data_storage_gpu_2d_t :: this%es_pot)
            allocate(data_storage_gpu_2d_t :: this%E_par)
            allocate(data_storage_gpu_2d_t :: this%A_par)
            allocate(data_storage_gpu_2d_t :: this%B_par)
            if(get_use_neutrals()) then
                allocate(data_storage_cpu_4d_t :: this%neutrals_state)
            end if
#endif
        else
            allocate(op_axpy_cpu_t :: this%op_add)
            allocate(op_copy_cpu_t :: this%op_copy)
            allocate(op_lin_comb_cpu_t :: this%op_lin_comb)
            allocate(op_set_uniform_cpu_t :: this%op_set_uniform)
            allocate(data_storage_cpu_5d_t :: this%dist_func)
            allocate(data_storage_cpu_2d_t :: this%es_pot)
            allocate(data_storage_cpu_2d_t :: this%E_par)
            allocate(data_storage_cpu_2d_t :: this%A_par)
            allocate(data_storage_cpu_2d_t :: this%B_par)
            if(get_use_neutrals()) then
                allocate(data_storage_cpu_4d_t :: this%neutrals_state)
            end if
        endif

        ! Initialize the arithmetic operator members
        call this%op_add%initialize()
        call this%op_copy%initialize()
        call this%op_lin_comb%initialize()
        call this%op_set_uniform%initialize()

        ! Initialize the data storage members
        call this%dist_func%initialize(dcomm_handler)
        call this%es_pot%initialize(dcomm_handler)
        call this%E_par%initialize(dcomm_handler)
        call this%A_par%initialize(dcomm_handler)
        call this%B_par%initialize(dcomm_handler)
        if(get_use_neutrals()) then
            ! TODO: Define parameter: n_neut_mom (=1 when "1mom")
            call this%neutrals_state%initialize(dcomm_handler, &
                                                n_mom=1, &
                                                n_spec=get_n_points_neut())
        end if

        ! Allocate buffers for diagnostic moments
        shp = this%dist_func%shape_stripped()
        allocate(this%mom_0d)
        allocate(this%mom_2d)
        call this%mom_0d%initialize([1,1], [6, shp(DIM_SP)])
        call this%mom_2d%initialize([1,1,1,1], &
                                    [shp(DIM_RZ), shp(DIM_PHI), 8, shp(DIM_SP)])
        if (get_diagnose_tpc()) then
            allocate(this%mom_2d_tpc)
            call this%mom_2d_tpc%initialize([1,1,1,1], &
                                    [shp(DIM_RZ), shp(DIM_PHI), 8, shp(DIM_SP)])
        endif

        ! Allocate buffers for fields moments
        lb_2d = this%es_pot%lbound()
        ub_2d = this%es_pot%ubound()
        lb_stripped_2d = this%es_pot%lbound_stripped()
        ub_stripped_2d = this%es_pot%ubound_stripped()
        lb_mixed_2d = [lb_2d(1), lb_stripped_2d(2)]
        ub_mixed_2d = [ub_2d(1), ub_stripped_2d(2)]
        allocate(this%co_qn_eq)
        allocate(this%b_qn_eq)
        allocate(this%b_ohms_law)
        allocate(this%b_amps_law)
        allocate(this%b_bpar_eq)
        allocate(this%lambda_ohms_law)
        call this%co_qn_eq%initialize(lb_mixed_2d, ub_mixed_2d)
        call this%b_qn_eq%initialize(mold=this%co_qn_eq)
        call this%b_ohms_law%initialize(mold=this%b_qn_eq)
        call this%b_amps_law%initialize(mold=this%b_qn_eq)
        call this%b_bpar_eq%initialize(mold=this%b_qn_eq)
        call this%lambda_ohms_law%initialize(lb_stripped_2d, ub_stripped_2d)

        ! We need 3 moments that are stored along the 3rd dimension
        if(get_coll_type() /= "none") then
            allocate(this%mom_coll)
            call this%mom_coll%initialize( &
                [lb_stripped_2d(1), lb_stripped_2d(2), 1, 1], &
                [ub_stripped_2d(1), ub_stripped_2d(2), 3, get_n_points_sp()])
        endif

        ! Allocate buffers for neutrals species
        if(get_use_neutrals()) then
            allocate(this%neutrals_mom)
            call this%neutrals_mom%initialize( &
                [lb_2d(1), lb_2d(2), 1, 1], &
                [ub_2d(1), ub_2d(2), 3, get_n_points_sp()])
            allocate(this%neutrals_sources)
            call this%neutrals_sources%initialize( &
                [lb_2d(1), lb_2d(2), 1, 1], &
                [ub_2d(1), ub_2d(2), 3, get_n_points_neut()])
        endif

        ! Allocate buffers if MMS tests are performed
        if(get_run_mms()) then
            allocate(this%mms_dist_func_buffer)
            call this%mms_dist_func_buffer%initialize(&
                                        mold=this%dist_func%get_data_pointer())
            allocate(this%mms_es_pot_buffer)
            call this%mms_es_pot_buffer%initialize(&
                                        mold=this%es_pot%get_data_pointer())
            allocate(this%mms_A_par_buffer)
            call this%mms_A_par_buffer%initialize(&
                                        mold=this%A_par%get_data_pointer())
            allocate(this%mms_B_par_buffer)
            call this%mms_B_par_buffer%initialize(&
                                        mold=this%B_par%get_data_pointer())
            allocate(this%mms_E_par_buffer)
            call this%mms_E_par_buffer%initialize(&
                                        mold=this%E_par%get_data_pointer())
        endif
    end subroutine initialize

    module subroutine add(this, alpha1, s1)
        class(state_vector_t), target, intent(inout) :: this
        real(kind=GP), intent(in) :: alpha1
        class(state_vector_t), target, intent(inout) :: s1

        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: &
            this_data_5d, s1_data_5d
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: &
            this_data_4d, s1_data_4d
        integer :: ierr

        call profiler_start("add", ierr)

        this_data_5d => this%dist_func%get_pointer()
        s1_data_5d => s1%dist_func%get_pointer()
        call this%op_add%apply(this_data_5d, alpha1, s1_data_5d)

        if(get_use_neutrals()) then
            this_data_4d => this%neutrals_state%get_pointer()
            s1_data_4d => s1%neutrals_state%get_pointer()
            call this%op_add%apply(this_data_4d, alpha1, s1_data_4d)
        endif

        call profiler_stop("add", ierr)
    end subroutine

    module subroutine lin_comb(this, alpha1, s1, alpha2, s2)
        class(state_vector_t), target, intent(inout) :: this
        real(kind=GP), intent(in) :: alpha1
        class(state_vector_t), target, intent(inout) :: s1
        real(kind=GP), intent(in) :: alpha2
        class(state_vector_t), target, intent(inout) :: s2

        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: &
            s1_data_5d, s2_data_5d, this_data_5d
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: &
            s1_data_4d, s2_data_4d, this_data_4d
        integer :: ierr

        call profiler_start("lin_comb", ierr)

        s1_data_5d => s1%dist_func%get_pointer()
        s2_data_5d => s2%dist_func%get_pointer()
        this_data_5d => this%dist_func%get_pointer()
        call this%op_lin_comb%apply(this_data_5d, alpha1, &
                                    s1_data_5d, alpha2, s2_data_5d)

        if(get_use_neutrals()) then
            s1_data_4d => s1%neutrals_state%get_pointer()
            s2_data_4d => s2%neutrals_state%get_pointer()
            this_data_4d => this%neutrals_state%get_pointer()
            call this%op_lin_comb%apply(this_data_4d, alpha1, &
                                        s1_data_4d, alpha2, s2_data_4d)
        endif

        call profiler_stop("lin_comb", ierr)
    end subroutine

    module subroutine set_uniform(this, input)
        class(state_vector_t), target, intent(inout) :: this
        real(kind=GP), intent(in) :: input
        real(kind=GP), dimension(:,:,:,:,:), pointer :: dist_func_ptr
        real(kind=GP), dimension(:,:,:,:), pointer :: neut_state_ptr

        dist_func_ptr => this%dist_func%get_pointer()
        call this%op_set_uniform%apply(dist_func_ptr, input)

        if(get_use_neutrals()) then
            neut_state_ptr => this%neutrals_state%get_pointer()
            call this%op_set_uniform%apply(neut_state_ptr, input)
        end if
    end subroutine

    module subroutine copy(this, s1)
        !! Copies the values of s1 to the current instance
        class(state_vector_t), target, intent(inout) :: this
        !! Instance of the type
        class(state_vector_t), target, intent(inout) :: s1
        !! State vector to copy
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: &
            s1_dist_func, this_dist_func
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: &
            s1_neut_state, this_neut_state
        real(kind=GP), contiguous, pointer, dimension(:,:) :: &
            s1_es_pot, this_es_pot, s1_A_par, this_A_par, s1_B_par, this_B_par
        integer :: ierr

        ! NOTE: Temporary adjustment
        ! TODO: Fully integrate data_array_t to the interface of the operator
        real(kind=GP), contiguous, pointer, dimension(:,:) :: &
            this_co_qn_eq, this_b_qn_eq, this_b_amps_law, this_b_bpar_eq, &
            s1_co_qn_eq, s1_b_qn_eq, s1_b_amps_law, s1_b_bpar_eq

        call profiler_start("copy", ierr)

        s1_dist_func => s1%dist_func%get_pointer()
        this_dist_func => this%dist_func%get_pointer()
        s1_es_pot => s1%es_pot%get_pointer()
        this_es_pot => this%es_pot%get_pointer()
        s1_A_par => s1%A_par%get_pointer()
        this_A_par => this%A_par%get_pointer()
        s1_B_par => s1%B_par%get_pointer()
        this_B_par => this%B_par%get_pointer()

        this_co_qn_eq => this%co_qn_eq%get_pointer()
        this_b_qn_eq => this%b_qn_eq%get_pointer()
        this_b_amps_law => this%b_amps_law%get_pointer()
        this_b_bpar_eq => this%b_bpar_eq%get_pointer()
        s1_co_qn_eq => s1%co_qn_eq%get_pointer()
        s1_b_qn_eq => s1%b_qn_eq%get_pointer()
        s1_b_amps_law => s1%b_amps_law%get_pointer()
        s1_b_bpar_eq => s1%b_bpar_eq%get_pointer()

        call this%op_copy%apply(this_dist_func, s1_dist_func)
        call this%op_copy%apply(this_es_pot, s1_es_pot)
        call this%op_copy%apply(this_A_par, s1_A_par)
        call this%op_copy%apply(this_B_par, s1_B_par)
        call this%op_copy%apply(this_b_qn_eq, s1_b_qn_eq)
        call this%op_copy%apply(this_b_amps_law, s1_b_amps_law)
        call this%op_copy%apply(this_b_bpar_eq, s1_b_bpar_eq)
        call this%op_copy%apply(this_co_qn_eq, s1_co_qn_eq)

        if(get_use_neutrals()) then
            this_neut_state => this%neutrals_state%get_pointer()
            s1_neut_state => s1%neutrals_state%get_pointer()
            call this%op_copy%apply(this_neut_state, s1_neut_state)
        endif

        call profiler_stop("copy", ierr)
    end subroutine copy

    module subroutine exchange(this)
        class(state_vector_t), intent(inout) :: this

        if(get_use_vspectral() .or. get_coll_type() == "lorentz") then
            ! NOTE: The order of the ghost exchange must be vp, mu, and phi
            !       for some operators that requires the edge ghost exchange.
            ! TODO: We need to test the performance in order to determine if
            !       it is possible to use the same exchange order for all cases.
            call this%dist_func%start_exchange(DIM_VP)
            call this%dist_func%finish_exchange(DIM_VP)
            call this%dist_func%start_exchange(DIM_MU)
            call this%dist_func%finish_exchange(DIM_MU)
            call this%dist_func%start_exchange(DIM_PHI)
            call this%dist_func%finish_exchange(DIM_PHI)

            ! Phi exchange is necessary for the calculation of parallel es_pot
            ! derivatives
            call this%es_pot%start_exchange()
            call this%A_par%start_exchange()
            call this%B_par%start_exchange()

            call this%B_par%finish_exchange()
            call this%A_par%finish_exchange()
            call this%es_pot%finish_exchange()
        else
            call this%dist_func%start_exchange(DIM_PHI)
            call this%dist_func%start_exchange(DIM_VP)
            call this%dist_func%start_exchange(DIM_MU)
            ! Phi exchange is necessary for the calculation of parallel es_pot
            ! derivatives
            call this%es_pot%start_exchange()
            call this%A_par%start_exchange()
            call this%B_par%start_exchange()

            ! NOTE: ask whether bpar should be in all these exchanges
            call this%B_par%finish_exchange()
            call this%A_par%finish_exchange()
            call this%es_pot%finish_exchange()
            call this%dist_func%finish_exchange(DIM_MU)
            call this%dist_func%finish_exchange(DIM_VP)
            call this%dist_func%finish_exchange(DIM_PHI)
        endif

    end subroutine exchange

    module subroutine exchange_neut(this)
        class(state_vector_t), intent(inout) :: this
        call this%neutrals_state%start_exchange()
        call this%neutrals_state%finish_exchange()
    end subroutine exchange_neut

    module subroutine start_exchange(this)
        class(state_vector_t), intent(inout) :: this
        call this%dist_func%start_exchange(DIM_PHI)
        call this%dist_func%start_exchange(DIM_VP)
        call this%dist_func%start_exchange(DIM_MU)
        call this%es_pot%start_exchange()
        call this%A_par%start_exchange()
        call this%B_par%start_exchange()
    end subroutine start_exchange

    module subroutine start_exchange_phi(this)
        class(state_vector_t), intent(inout) :: this
        call this%dist_func%start_exchange(DIM_PHI)
        call this%es_pot%start_exchange()
        call this%A_par%start_exchange()
        call this%B_par%start_exchange()
    end subroutine start_exchange_phi

    module subroutine start_exchange_vp(this)
        class(state_vector_t), intent(inout) :: this
        call this%dist_func%start_exchange(DIM_VP)
    end subroutine start_exchange_vp

    module subroutine start_exchange_mu(this)
        class(state_vector_t), intent(inout) :: this
        call this%dist_func%start_exchange(DIM_MU)
    end subroutine start_exchange_mu

    module subroutine finish_exchange_phi(this)
        class(state_vector_t), intent(inout) :: this
        call this%dist_func%finish_exchange(DIM_PHI)
        call this%es_pot%finish_exchange()
        call this%A_par%finish_exchange()
        call this%B_par%finish_exchange()
    end subroutine finish_exchange_phi

    module subroutine finish_exchange_vp(this)
        class(state_vector_t), intent(inout) :: this
        call this%dist_func%finish_exchange(DIM_VP)
    end subroutine finish_exchange_vp

    module subroutine finish_exchange_mu(this)
        class(state_vector_t), intent(inout) :: this
        call this%dist_func%finish_exchange(DIM_MU)
    end subroutine finish_exchange_mu

    module subroutine apply_diag_mom_0d(this, op, diag_file, t)
        class(state_vector_t), target, intent(inout) :: this
        class(op_diag_mom_0d_base_t), intent(inout) :: op
        type(mom_0d_file_t), intent(inout) :: diag_file
        real(kind=GP), intent(in) :: t

        class(data_array_5d_t), pointer :: this_dist_func
        class(data_array_2d_t), pointer :: this_es_pot

        ! TODO: Remove this once data_array_t is integrated to diag_file
        real(kind=GP), contiguous, pointer, dimension(:,:) :: this_mom_0d

        this_dist_func => this%dist_func%get_data_pointer()
        this_es_pot => this%es_pot%get_data_pointer()
        this_mom_0d => this%mom_0d%get_pointer()
        call op%apply(this_dist_func, this_es_pot, this%mom_0d)
        call diag_file%write(t, this_mom_0d)
    end subroutine

    module subroutine apply_diag_mom_2d(this, op, diag_file, t, diagnose_tpc)
        class(state_vector_t), target, intent(inout) :: this
        class(op_diag_mom_2d_base_t), intent(inout) :: op
        type(mom_2d_file_t), intent(inout) :: diag_file
        real(kind=GP), intent(in) :: t
        logical, optional, intent(in) :: diagnose_tpc

        class(data_array_5d_t), pointer :: this_dist_func
        type(data_array_4d_t), pointer :: this_mom_2d_darray
        logical :: diagnose_tpc_local

        ! TODO: Remove these once data_array_t is integrated to diag_file
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: this_mom_2d
        integer :: lb_stripped(5), ub_stripped(5)

        ! Optional TPC default false
        diagnose_tpc_local = .false.
        if (present(diagnose_tpc)) then
            diagnose_tpc_local = diagnose_tpc
        endif

        this_dist_func => this%dist_func%get_data_pointer()
        lb_stripped = this_dist_func%get_lbound_stripped()
        ub_stripped = this_dist_func%get_ubound_stripped()

        if (.not. diagnose_tpc_local) then
            this_mom_2d_darray => this%mom_2d
        else
            this_mom_2d_darray => this%mom_2d_tpc
        endif
        this_mom_2d => this_mom_2d_darray%get_pointer()

        call op%apply(this_dist_func, this_mom_2d_darray, diagnose_tpc_local)
        call diag_file%write(t, lb_stripped, ub_stripped, this_mom_2d)

    end subroutine

    module subroutine save_em_fields(this, diag_file, t)
        class(state_vector_t), target, intent(inout) :: this
        type(em_fields_file_t), intent(inout) :: diag_file
        real(kind=GP), intent(in) :: t

        real(kind=GP), pointer, dimension(:,:) :: this_es_pot, this_A_par, &
                                                  this_B_par
        integer :: lb_stripped(2), ub_stripped(2)

        lb_stripped = this%A_par%lbound_stripped()
        ub_stripped = this%A_par%ubound_stripped()
        this_es_pot => this%es_pot%get_pointer_stripped()
        this_A_par => this%A_par%get_pointer_stripped()
        this_B_par => this%B_par%get_pointer_stripped()
        call diag_file%write(t, lb_stripped, ub_stripped, this_es_pot, &
                             this_A_par, this_B_par)
    end subroutine

    module subroutine save_neutrals(this, diag_file, t)
        class(state_vector_t), target, intent(inout) :: this
        type(neutrals_file_t), intent(inout) :: diag_file
        real(kind=GP), intent(in) :: t

        real(kind=GP), pointer, dimension(:,:,:,:) :: this_neutrals_state
        integer :: lb_stripped(4), ub_stripped(4)

        lb_stripped = this%neutrals_state%lbound_stripped()
        ub_stripped = this%neutrals_state%ubound_stripped()
        this_neutrals_state => this%neutrals_state%get_pointer_stripped()
        call diag_file%write(t, lb_stripped, ub_stripped, this_neutrals_state)
    end subroutine

    module subroutine save_checkpoint(this, diag_file, t)
        class(state_vector_t), target, intent(inout) :: this
        type(checkpoint_file_t), intent(inout) :: diag_file
        real(kind=GP), intent(in) :: t

        real(kind=GP), pointer, dimension(:,:,:,:,:) :: this_data

        this_data => this%dist_func%get_pointer_stripped()
        call diag_file%write(t, this_data)
    end subroutine

    module subroutine save_checkpoint_neut(this, diag_file, t)
        class(state_vector_t), target, intent(inout) :: this
        type(checkpoint_neut_file_t), intent(inout) :: diag_file
        real(kind=GP), intent(in) :: t

        real(kind=GP), pointer, dimension(:,:,:,:) :: this_data

        this_data => this%neutrals_state%get_pointer_stripped()
        call diag_file%write(t, this_data)
    end subroutine

    module subroutine read_checkpoint(this, diag_file, t)
        class(state_vector_t), target, intent(inout) :: this
        type(checkpoint_file_t), intent(inout) :: diag_file
        real(kind=GP), intent(out) :: t

        real(kind=GP), pointer, dimension(:,:,:,:,:) :: this_data

        this_data => this%dist_func%get_pointer_stripped()
        call diag_file%read(t, this_data)
    end subroutine

    module subroutine read_checkpoint_neut(this, diag_file, t)
        class(state_vector_t), target, intent(inout) :: this
        type(checkpoint_neut_file_t), intent(inout) :: diag_file
        real(kind=GP), intent(out) :: t

        real(kind=GP), pointer, dimension(:,:,:,:) :: this_data

        this_data => this%neutrals_state%get_pointer_stripped()
        call diag_file%read(t, this_data)
    end subroutine

    module subroutine apply_set_initial_condition(this, op)
        class(state_vector_t), target, intent(inout) :: this
        class(op_set_initial_condition_base_t), intent(inout) :: op

        class(data_array_5d_t), pointer :: this_data

        this_data => this%dist_func%get_data_pointer()
        call op%apply(this_data)
    end subroutine

    module subroutine apply_rhs_vlasov_eq_static(this, op, s_out)
        class(state_vector_t), target, intent(inout) :: this
        class(op_rhs_vlasov_eq_static_base_t), intent(inout) :: op
        class(state_vector_t), target, intent(inout) :: s_out

        class(data_array_5d_t), pointer :: this_dist_func, s_out_dist_func
        class(data_array_2d_t), pointer :: this_es_pot, this_A_par, this_B_par

        this_dist_func  => this%dist_func%get_data_pointer()
        this_es_pot     => this%es_pot%get_data_pointer()
        this_A_par      => this%A_par%get_data_pointer()
        this_B_par      => this%B_par%get_data_pointer()
        s_out_dist_func => s_out%dist_func%get_data_pointer()

        call op%apply(this_dist_func, this_es_pot, this_A_par, this_B_par, &
                      s_out_dist_func)
    end subroutine

    module subroutine apply_rhs_vlasov_eq_dynamic(this, op, s_out)
        class(state_vector_t), target, intent(inout) :: this
        class(op_rhs_vlasov_eq_dynamic_base_t), intent(inout) :: op
        class(state_vector_t), target, intent(inout) :: s_out

        class(data_array_5d_t), pointer :: this_dist_func, s_out_dist_func
        class(data_array_2d_t), pointer :: this_E_par

        this_dist_func  => this%dist_func%get_data_pointer()
        this_E_par      => this%E_par%get_data_pointer()
        s_out_dist_func => s_out%dist_func%get_data_pointer()

        call op%apply(this_dist_func, this_E_par, s_out_dist_func)
    end subroutine

    module subroutine apply_solve_qn_eq(this, op, residuum)
        class(state_vector_t), target, intent(inout) :: this
        class(op_solve_qn_eq_t), intent(inout) :: op
        real(kind=GP), intent(out) :: residuum

        class(data_array_2d_t), pointer :: this_es_pot

        this_es_pot => this%es_pot%get_data_pointer()
        call op%apply(this%co_qn_eq, this%b_qn_eq, this_es_pot, residuum)
    end subroutine

    module subroutine apply_solve_amps_law(this, op, residuum)
        class(state_vector_t), target, intent(inout) :: this
        class(op_solve_amps_law_t), intent(inout) :: op
        real(kind=GP), intent(out) :: residuum

        class(data_array_2d_t), pointer :: this_A_par

        this_A_par => this%A_par%get_data_pointer()
        call op%apply(this%b_amps_law, this_A_par, residuum)
    end subroutine

    module subroutine apply_solve_bpar_eq(this, op)
        class(state_vector_t), target, intent(inout) :: this
        class(op_solve_bpar_eq_t), intent(inout) :: op

        class(data_array_2d_t), pointer :: this_B_par

        this_B_par => this%B_par%get_data_pointer()
        call op%apply(this%b_bpar_eq, this_B_par)
    end subroutine

    module subroutine apply_solve_ohms_law(this, op, residuum)
        class(state_vector_t), target, intent(inout) :: this
        class(op_solve_ohms_law_t), intent(inout) :: op
        real(kind=GP), intent(out) :: residuum

        class(data_array_2d_t), pointer :: this_E_par

        this_E_par => this%E_par%get_data_pointer()
        call op%apply(this%lambda_ohms_law, this%b_ohms_law, this_E_par, &
                      residuum)
    end subroutine

    module subroutine apply_mom_maxwells_eq(this, op)
        class(state_vector_t), target, intent(inout) :: this
        class(op_mom_maxwells_eq_base_t), intent(inout) :: op

        class(data_array_5d_t), pointer :: this_dist_func

        this_dist_func => this%dist_func%get_data_pointer()
        call op%apply(this_dist_func, this%co_qn_eq, this%b_qn_eq, &
                      this%b_amps_law, this%b_bpar_eq)
    end subroutine

    module subroutine apply_mom_ohms_law(this, op, ds_in)
        class(state_vector_t), target, intent(inout) :: this
        class(op_mom_ohms_law_base_t), intent(inout) :: op
        class(state_vector_t), target, intent(inout) :: ds_in

        class(data_array_5d_t), pointer :: this_dist_func, ds_in_dist_func

        this_dist_func => this%dist_func%get_data_pointer()
        ds_in_dist_func => ds_in%dist_func%get_data_pointer()
        call op%apply(this_dist_func, ds_in_dist_func, this%lambda_ohms_law, &
                      this%b_ohms_law)
    end subroutine

    module subroutine apply_mom_coll(this, op, s_out)
        class(state_vector_t), target, intent(inout) :: this
        class(op_mom_coll_base_t), intent(inout) :: op
        class(state_vector_t), target, intent(inout) :: s_out

        class(data_array_5d_t), pointer :: this_dist_func

        this_dist_func => this%dist_func%get_data_pointer()
        call op%apply(this_dist_func, this%mom_coll)
    end subroutine

    module subroutine apply_coll(this, op, s_out)
        class(state_vector_t), target, intent(inout) :: this
        class(op_coll_base_t), intent(inout) :: op
        class(state_vector_t), target, intent(inout) :: s_out

        class(data_array_5d_t), pointer :: this_dist_func, s_out_dist_func

        this_dist_func => this%dist_func%get_data_pointer()
        s_out_dist_func => s_out%dist_func%get_data_pointer()
        call op%apply(this_dist_func, this%mom_coll, s_out_dist_func)
    end subroutine

    module subroutine apply_bnd_cond(this, op, t)
        class(state_vector_t), target, intent(inout) :: this
        class(op_bnd_cond_base_t), intent(inout) :: op
        real(kind=GP), intent(in) :: t

        class(data_array_5d_t), pointer :: this_dist_func

        this_dist_func => this%dist_func%get_data_pointer()

        call op%apply(this_dist_func, this%co_qn_eq, this%b_qn_eq, &
                      this%b_amps_law, this%b_ohms_law, t)
    end subroutine

    module subroutine apply_source(this, op)
        class(state_vector_t), target, intent(inout) :: this
        class(op_source_t), intent(inout) :: op

        class(data_array_5d_t), pointer :: this_dist_func

        this_dist_func => this%dist_func%get_data_pointer()
        call op%apply(this_dist_func)
    end subroutine

    module subroutine apply_set_initial_neutrals(this, op)
        class(state_vector_t), target, intent(inout) :: this
        class(op_neut_set_initial_t), intent(inout) :: op

        class(data_array_4d_t), pointer :: this_neut_state

        this_neut_state => this%neutrals_state%get_data_pointer()
        call op%apply(this_neut_state)
    end subroutine

    module subroutine apply_neutrals_mom(this, op, s_out)
        class(state_vector_t), target, intent(inout) :: this
        class(op_neut_mom_t), intent(inout) :: op
        class(state_vector_t), target, intent(inout) :: s_out

        class(data_array_5d_t), pointer :: this_dist_func
        class(data_array_4d_t), pointer :: this_neut_mom

        this_dist_func => this%dist_func%get_data_pointer()
        this_neut_mom => this%neutrals_mom
        call op%apply(this_dist_func, this_neut_mom)
    end subroutine

    module subroutine apply_neutrals_coupling_pn(this, op, s_out)
        class(state_vector_t), target, intent(inout) :: this
        class(op_neut_coupling_pn_t), intent(inout) :: op
        class(state_vector_t), target, intent(inout) :: s_out

        class(data_array_5d_t), pointer :: this_dist_func
        class(data_array_4d_t), pointer :: this_neut_mom, this_neut_state, &
                                           this_neut_sources

        this_dist_func => this%dist_func%get_data_pointer()
        this_neut_state => this%neutrals_state%get_data_pointer()
        this_neut_mom => this%neutrals_mom
        this_neut_sources => this%neutrals_sources
        call op%apply(this_neut_mom, this_dist_func, this_neut_state, &
                      this_neut_sources)
    end subroutine

    module subroutine apply_neutrals_coupling_np(this, op, s_out)
        class(state_vector_t), target, intent(inout) :: this
        class(op_neut_coupling_np_t), intent(inout) :: op
        class(state_vector_t), target, intent(inout) :: s_out

        class(data_array_5d_t), pointer :: this_dist_func, s_out_dist_func
        class(data_array_4d_t), pointer :: this_neut_mom, this_neut_state

        this_dist_func => this%dist_func%get_data_pointer()
        s_out_dist_func => s_out%dist_func%get_data_pointer()
        this_neut_state => this%neutrals_state%get_data_pointer()
        this_neut_mom => this%neutrals_mom
        call op%apply(this_neut_mom, this_dist_func, this_neut_state, &
                      s_out_dist_func)
    end subroutine

    module subroutine apply_neutrals_evolve(this, op, s_out)
        class(state_vector_t), target, intent(inout) :: this
        class(op_neut_evolve_t), intent(inout) :: op
        class(state_vector_t), target, intent(inout) :: s_out

        class(data_array_4d_t), pointer :: this_neut_sources, this_neut_state, &
                                           s_out_neut_state

        this_neut_state => this%neutrals_state%get_data_pointer()
        s_out_neut_state => s_out%neutrals_state%get_data_pointer()
        this_neut_sources => this%neutrals_sources
        call op%apply(this_neut_sources, this_neut_state, s_out_neut_state)
    end subroutine

    module subroutine apply_mms_source_vlasov_eq(this, op, t)
        class(state_vector_t), target, intent(inout) :: this
        class(op_mms_source_vlasov_eq_t), intent(inout) :: op
        real(kind=GP), intent(in) :: t

        integer, dimension(5) :: lb, ub, lb_stripped, ub_stripped
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: &
            this_dist_func

        this_dist_func => this%dist_func%get_pointer()
        lb = this%dist_func%lbound()
        ub = this%dist_func%ubound()
        lb_stripped = this%dist_func%lbound_stripped()
        ub_stripped = this%dist_func%ubound_stripped()
        call op%apply(t, lb, ub, lb_stripped, ub_stripped, this_dist_func)
    end subroutine

    module subroutine apply_mms_source_maxwells_eq(this, op, t)
        class(state_vector_t), target, intent(inout) :: this
        class(op_mms_source_maxwells_eq_t), intent(inout) :: op
        real(kind=GP), intent(in) :: t

        integer, dimension(2) :: lb, ub, lb_stripped, ub_stripped

        ! NOTE: Temporary adjustment
        ! TODO: Fully integrate data_array_t to the interface of the operator
        real(kind=GP), contiguous, pointer, dimension(:,:) :: &
            this_b_qn_eq, this_b_amps_law, this_b_bpar_eq

        this_b_qn_eq => this%b_qn_eq%get_pointer()
        this_b_amps_law => this%b_amps_law%get_pointer()
        this_b_bpar_eq => this%b_bpar_eq%get_pointer()

        lb = this%es_pot%lbound()
        ub = this%es_pot%ubound()
        lb_stripped = this%es_pot%lbound_stripped()
        ub_stripped = this%es_pot%ubound_stripped()
        call op%apply(t, lb, ub, lb_stripped, ub_stripped, this_b_qn_eq, &
                      this_b_amps_law, this_b_bpar_eq)
    end subroutine

    module subroutine apply_mms_source_ohms_law(this, op, t)
        class(state_vector_t), target, intent(inout) :: this
        class(op_mms_source_ohms_law_t), intent(inout) :: op
        real(kind=GP), intent(in) :: t

        integer, dimension(2) :: lb, ub, lb_stripped, ub_stripped

        ! NOTE: Temporary adjustment
        ! TODO: Fully integrate data_array_t to the interface of the operator
        real(kind=GP), contiguous, pointer, dimension(:,:) :: this_b_ohms_law

        this_b_ohms_law => this%b_ohms_law%get_pointer()

        lb = this%E_par%lbound()
        ub = this%E_par%ubound()
        lb_stripped = this%E_par%lbound_stripped()
        ub_stripped = this%E_par%ubound_stripped()
        call op%apply(t, lb, ub, lb_stripped, ub_stripped, this_b_ohms_law)
    end subroutine

    module subroutine apply_mms_solution_vlasov_eq(this, op, t)
        class(state_vector_t), target, intent(inout) :: this
        class(op_mms_solution_vlasov_eq_t), intent(inout) :: op
        real(kind=GP), intent(in) :: t

        class(data_array_5d_t), pointer :: this_dist_func

        this_dist_func => this%dist_func%get_data_pointer()
        call op%apply(t, this_dist_func)
    end subroutine

    module subroutine apply_mms_solution_maxwells_eq(this, op, t)
        class(state_vector_t), target, intent(inout) :: this
        class(op_mms_solution_maxwells_eq_t), intent(inout) :: op
        real(kind=GP), intent(in) :: t

        class(data_array_2d_t), pointer :: this_es_pot, this_A_par, this_B_par

        this_es_pot => this%es_pot%get_data_pointer()
        this_A_par => this%A_par%get_data_pointer()
        this_B_par => this%B_par%get_data_pointer()
        call op%apply(t, this_es_pot, this_A_par, this_B_par)
    end subroutine

    module subroutine apply_mms_solution_ohms_law(this, op, t)
        class(state_vector_t), target, intent(inout) :: this
        class(op_mms_solution_ohms_law_t), intent(inout) :: op
        real(kind=GP), intent(in) :: t

        class(data_array_2d_t), pointer :: this_E_par

        this_E_par => this%E_par%get_data_pointer()
        call op%apply(t, this_E_par)
    end subroutine

    module subroutine buffer_mms_solution_vlasov_eq(this, store_to_buffer)
        class(state_vector_t), target, intent(inout) :: this
        logical, intent(in) :: store_to_buffer

        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: &
            this_dist_func, buffer_dist_func

        this_dist_func => this%dist_func%get_pointer()
        buffer_dist_func => this%mms_dist_func_buffer%get_pointer()

        if(store_to_buffer) then
            call this%op_copy%apply(buffer_dist_func, this_dist_func)
        else
            call this%op_copy%apply(this_dist_func, buffer_dist_func)
        endif
    end subroutine

    module subroutine buffer_mms_solution_maxwells_eq(this, store_to_buffer)
        class(state_vector_t), target, intent(inout) :: this
        logical, intent(in) :: store_to_buffer

        real(kind=GP), contiguous, pointer, dimension(:,:) :: &
            this_es_pot, this_A_par, this_B_par, buffer_es_pot, &
            buffer_A_par, buffer_B_par

        this_es_pot => this%es_pot%get_pointer()
        buffer_es_pot => this%mms_es_pot_buffer%get_pointer()
        this_A_par  => this%A_par%get_pointer()
        buffer_A_par => this%mms_A_par_buffer%get_pointer()
        this_B_par  => this%B_par%get_pointer()
        buffer_B_par => this%mms_B_par_buffer%get_pointer()

        if(store_to_buffer) then
            call this%op_copy%apply(buffer_es_pot, this_es_pot)
            call this%op_copy%apply(buffer_A_par, this_A_par)
            call this%op_copy%apply(buffer_B_par, this_B_par)
        else
            call this%op_copy%apply(this_es_pot, buffer_es_pot)
            call this%op_copy%apply(this_A_par, buffer_A_par)
            call this%op_copy%apply(this_B_par, buffer_B_par)
        endif
    end subroutine

    module subroutine buffer_mms_solution_ohms_law(this, store_to_buffer)
        class(state_vector_t), target, intent(inout) :: this
        logical, intent(in) :: store_to_buffer

        real(kind=GP), contiguous, pointer, dimension(:,:) :: this_E_par, &
                                                              buffer_E_par

        this_E_par => this%E_par%get_pointer()
        buffer_E_par => this%mms_E_par_buffer%get_pointer()

        if(store_to_buffer) then
            call this%op_copy%apply(buffer_E_par, this_E_par)
        else
            call this%op_copy%apply(this_E_par, buffer_E_par)
        endif
    end subroutine

    module subroutine apply_mms_error(this, op, t)
        class(state_vector_t), target, intent(inout) :: this
        class(op_mms_error_t), intent(inout) :: op
        real(kind=GP), intent(in) :: t

        class(data_array_5d_t), pointer :: this_dist_func
        class(data_array_2d_t), pointer :: this_es_pot, this_A_par, &
                                           this_B_par, this_E_par
        real(kind=GP), dimension(6) :: mms_l2_err, mms_linf_err
        integer, save :: counter = 0
        !! Variable used to determine if the header of the output should be
        !! written

        this_dist_func => this%dist_func%get_data_pointer()
        this_es_pot    => this%es_pot%get_data_pointer()
        this_A_par     => this%A_par%get_data_pointer()
        this_B_par     => this%B_par%get_data_pointer()
        this_E_par     => this%E_par%get_data_pointer()

        call op%apply(t, this_dist_func, this_es_pot, this_A_par, this_B_par, &
                      this_E_par, mms_l2_err, mms_linf_err)

        if(counter == 0) then
            ! Write mms result header
            if(this%dcomm_handler%is_master()) then
                write(logger_get_info_channel(), *) &
                    "Result of the MMS verification:"
                write(logger_get_info_channel(), &
                      "(4X, A10, A25, A25, A25, A25, &
                      &A25, A25, A25, A25, A25, A25, A25, A25)") &
                    "t", "L2 error ions", "L2 error electrons", &
                    "L2 error es_pot", "L2 error A_par", "L2 error B_par", &
                    "L2 error E_par", "Linf error ions", &
                    "Linf error electrons", "Linf error es_pot", &
                    "Linf error A_par", "Linf error B_par", "Linf error E_par"
            endif
        endif
        if(this%dcomm_handler%is_master()) then
            ! Write current mms result
            write(logger_get_info_channel(), &
                  "(4X, F10.4, E25.4, E25.4, E25.4, E25.4, E25.4,&
                  &E25.4, E25.4, E25.4, E25.4, E25.4, E25.4, E25.4)") &
                  t, mms_l2_err(1), mms_l2_err(2), &
                     mms_l2_err(3), mms_l2_err(4), mms_l2_err(5), &
                     mms_l2_err(6), mms_linf_err(1), mms_linf_err(2), &
                     mms_linf_err(3), mms_linf_err(4), mms_linf_err(5), &
                     mms_linf_err(6)
        endif
        counter = counter + 1
    end subroutine

    module subroutine update_host_dist_func(this)
        class(state_vector_t), target, intent(inout) :: this
        call this%dist_func%update_host()
    end subroutine

    module subroutine update_device_dist_func(this)
        class(state_vector_t), target, intent(inout) :: this
        call this%dist_func%update_device()
    end subroutine

    module subroutine update_host_em_fields(this)
        class(state_vector_t), target, intent(inout) :: this
        call this%es_pot%update_host()
        call this%A_par%update_host()
        call this%B_par%update_host()
    end subroutine

    module subroutine update_device_em_fields(this)
        class(state_vector_t), target, intent(inout) :: this
        call this%es_pot%update_device()
        call this%A_par%update_device()
        call this%B_par%update_device()
    end subroutine

    module subroutine update_device_E_par(this)
        class(state_vector_t), target, intent(inout) :: this
        call this%E_par%update_device()
    end subroutine

    module subroutine update_host_E_par(this)
        class(state_vector_t), target, intent(inout) :: this
        call this%E_par%update_host()
    end subroutine

    module subroutine update_host_maxwells_buffers(this)
        class(state_vector_t), target, intent(inout) :: this
        call this%co_qn_eq%update_host()
        call this%b_qn_eq%update_host()
        call this%b_amps_law%update_host()
        call this%b_bpar_eq%update_host()
    end subroutine

    module subroutine update_device_maxwells_buffers(this)
        class(state_vector_t), target, intent(inout) :: this
        call this%co_qn_eq%update_device()
        call this%b_qn_eq%update_device()
        call this%b_amps_law%update_device()
        call this%b_bpar_eq%update_device()
    end subroutine

    module subroutine update_device_maxwells_buffers_bpar_only(this)
        class(state_vector_t), target, intent(inout) :: this
        call this%b_bpar_eq%update_device()
    end subroutine

    module subroutine update_host_ohms_buffers(this)
        class(state_vector_t), target, intent(inout) :: this
        call this%b_ohms_law%update_host()
        call this%lambda_ohms_law%update_host()
    end subroutine

    module subroutine update_device_ohms_buffers(this)
        class(state_vector_t), target, intent(inout) :: this
        call this%b_ohms_law%update_device()
        call this%lambda_ohms_law%update_device()
    end subroutine

end submodule
