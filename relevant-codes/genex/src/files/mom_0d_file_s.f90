submodule (diag_files_m) mom_0d_file_s
    use mpi
    use genex_fortran_env_m, only: MPI_GP
    use params_mesh_m, only: get_n_points_sp
    use params_species_m, only: get_name
    use dimensions_m, only: DIM_VP, DIM_MU, DIM_PHI, DIM_SP
    implicit none

contains

    module subroutine initialize_mom_0d(this, dcomm_handler, filename)
        class(mom_0d_file_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        character(len=*), intent(in) :: filename
        integer :: ierr, n, n_spec, comm_sp, n_procs_sp
        integer :: coords(dcomm_handler%get_n_dims())

        this%dcomm_handler => dcomm_handler
        this%filename = filename

        ! calculate the correct start position to write in the file
        n_spec = get_n_points_sp()
        n_procs_sp = dcomm_handler%get_n_procs_sp()
        coords = dcomm_handler%get_cart_coords()
        this%lb_mom_0d(1) = 1
        this%lb_mom_0d(2) = n_spec / n_procs_sp * coords(DIM_SP) + 1

        this%is_writing = dcomm_handler%is_master()
        if(coords(DIM_PHI) == 0 .and. coords(DIM_VP) == 0 &
                                .and. coords(DIM_MU) == 0) then
            this%is_communicating = .true.
        else
            this%is_communicating = .false.
        endif

        if(.not. this%is_communicating) then
            return
        endif

        allocate(this%send_buffer(6, n_spec))
        allocate(this%recv_buffer(6, n_spec))

        ! return if process is not writing
        if(.not. this%is_writing) then
            return
        endif

        allocate(this%species_group(n_spec))
        allocate(this%n_ids(n_spec))
        allocate(this%E_par_ids(n_spec))
        allocate(this%E_perp_ids(n_spec))
        allocate(this%E_es_ids(n_spec))
        allocate(this%u_par_ids(n_spec))
        allocate(this%p_phi_ids(n_spec))

        comm_sp = this%dcomm_handler%get_comm_sp()

        ! create moments file
        ierr = nf90_create(this%filename, NF90_NETCDF4, this%file_id)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        ! define time dimension
        ierr = nf90_def_dim(this%file_id, 'dim_time',  NF90_UNLIMITED, &
                            this%dim_time_id)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        ! define time
        ierr = nf90_def_var(this%file_id, 'time',  NF90_GP, this%dim_time_id, &
                            this%time_id)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        do n = 1, n_spec
            ! create group
            ierr = nf90_def_grp(this%file_id, get_name(n), &
                                this%species_group(n))
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            ! particle number
            ierr = nf90_def_var(this%species_group(n), &
                                'n',  &
                                NF90_GP, this%dim_time_id, &
                                this%n_ids(n))
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            ! u_par
            ierr = nf90_def_var(this%species_group(n), 'u_par', &
                                NF90_GP, this%dim_time_id, &
                                this%u_par_ids(n))
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            ! E_par
            ierr = nf90_def_var(this%species_group(n), 'E_par', &
                                NF90_GP, this%dim_time_id, &
                                this%E_par_ids(n))
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            ! E_perp
            ierr = nf90_def_var(this%species_group(n), 'E_perp', &
                                NF90_GP, this%dim_time_id, &
                                this%E_perp_ids(n))
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            ! E_es
            ierr = nf90_def_var(this%species_group(n), 'E_es', &
                                NF90_GP, this%dim_time_id, &
                                this%E_es_ids(n))
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            ! p_phi
            ierr = nf90_def_var(this%species_group(n), &
                                'p_phi', &
                                NF90_GP, this%dim_time_id, &
                                this%p_phi_ids(n))
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
        end do
        ierr = nf90_enddef(this%file_id)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        this%write_counter = 1
    end subroutine

    module subroutine write_mom_0d(this, time, mom_0d)
        class(mom_0d_file_t), intent(inout) :: this
        real(kind = GP), intent(in) :: time
        real(kind = GP), intent(in) :: mom_0d(this%lb_mom_0d(1):, &
                                              this%lb_mom_0d(2):)
        integer :: n_spec, n, ierr
        integer :: lb(2), ub(2), start(1), comm_sp

        if(.not. this%is_communicating) then
            return
        endif

        lb = lbound(mom_0d)
        ub = ubound(mom_0d)

        comm_sp = this%dcomm_handler%get_comm_sp()
        ! TODO: use MPI_Gather here instead of reduce
        !call MPI_Gather(mom_0d, size(mom_0d), this%buffer, size(mom_0d), 0, &
        !                this%dcomm_handler%get_comm_cart(), ierr)
        this%send_buffer = 0.0_GP
        this%recv_buffer = 0.0_GP
        this%send_buffer(:, lb(2) : ub(2)) = mom_0d
        call MPI_Reduce(this%send_buffer, this%recv_buffer, &
                        size(this%send_buffer), &
                        MPI_GP, MPI_SUM, 0, comm_sp, ierr)

        if(.not. this%is_writing) then
            return
        endif

        n_spec = get_n_points_sp()

        start(1) = this%write_counter
        ierr = nf90_put_var(this%file_id, this%time_id, time, &
                            start = start)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)

        do n = 1, n_spec
            ! n
            ierr = nf90_put_var(this%species_group(n), &
                                this%n_ids(n), &
                                this%recv_buffer(1, n), &
                                start = start)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            ! u_par
            ierr = nf90_put_var(this%species_group(n), &
                                this%u_par_ids(n), &
                                this%recv_buffer(2, n), &
                                start = start)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            ! E_par
            ierr = nf90_put_var(this%species_group(n), &
                                this%E_par_ids(n), &
                                this%recv_buffer(3, n), &
                                start = start)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            ! E_perp
            ierr = nf90_put_var(this%species_group(n), &
                                this%E_perp_ids(n), &
                                this%recv_buffer(4, n), &
                                start = start)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            ! E_es
            ierr = nf90_put_var(this%species_group(n), &
                                this%E_es_ids(n), &
                                this%recv_buffer(5, n), &
                                start = start)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
            ! p_phi
            ierr = nf90_put_var(this%species_group(n), &
                                this%p_phi_ids(n), &
                                this%recv_buffer(6, n), &
                                start = start)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

        end do
        ierr = nf90_sync(this%file_id)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        this%write_counter = this%write_counter + 1
    end subroutine

    module subroutine finalize_mom_0d(this)
        type(mom_0d_file_t), intent(inout) :: this
        integer :: ierr

        if(.not. this%is_writing) then
            return
        endif

        ierr = nf90_close(this%file_id)
        call handle_error_netcdf(ierr, __LINE__, __FILE__)
        deallocate(this%u_par_ids)
        deallocate(this%E_perp_ids)
        deallocate(this%E_par_ids)
        deallocate(this%E_es_ids)
        deallocate(this%p_phi_ids)
        deallocate(this%n_ids)
        deallocate(this%species_group)
    end subroutine

end submodule
