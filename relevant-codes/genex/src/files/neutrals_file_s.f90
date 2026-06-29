submodule (diag_files_m) neutrals_file_s
    use mpi
    use genex_fortran_env_m, only: MPI_GP
    use params_neutrals_m, only: get_neut_name
    use op_set_uniform_m, only: op_set_uniform_cpu_t

    implicit none

contains

    module subroutine initialize_neut(this, dcomm_handler, filename, &
                                      n_points_RZ, n_points_phi, n_neut_sp)
        class(neutrals_file_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        character(len=*), intent(in) :: filename
        integer, intent(in) :: n_points_RZ, n_points_phi, n_neut_sp

        type(op_set_uniform_cpu_t) :: op_set_uniform
        integer :: dataset_dims(3)
        integer :: o, ierr

        this%dcomm_handler => dcomm_handler
        this%filename = filename
        this%n_points_RZ = n_points_RZ
        this%n_points_phi = n_points_phi

        this%n_neut_sp = n_neut_sp

        allocate(this%species_group(n_neut_sp))
        allocate(this%n_ids(n_neut_sp))

        allocate(this%send_buffer(n_points_RZ, n_points_phi, 1, n_neut_sp))
        allocate(this%recv_buffer(n_points_RZ, n_points_phi, 1, n_neut_sp))

        ! First touch initialize recv_buffer, send_buffer is initialised
        ! in the write subroutine.
        call op_set_uniform%initialize()
        call op_set_uniform%apply(this%recv_buffer, 0.0_GP)

        if(dcomm_handler%is_master()) then
            ! create parallel NetCDF moments file
            ierr = nf90_create(this%filename, NF90_NETCDF4, this%file_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! define time dimension
            ierr = nf90_def_dim(this%file_id, 'dim_time',  NF90_UNLIMITED, &
                                this%dim_time_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! define RZ dimension
            ierr = nf90_def_dim(this%file_id, 'dim_RZ',  n_points_RZ, &
                                this%dim_RZ_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! define phi dimension
            ierr = nf90_def_dim(this%file_id, 'dim_phi',  n_points_phi, &
                                this%dim_phi_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! define time
            ierr = nf90_def_var(this%file_id, 'time',  NF90_GP, &
                                this%dim_time_id, this%time_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            ! define the dimension for the dataset
            dataset_dims(1) = this%dim_RZ_id
            dataset_dims(2) = this%dim_phi_id
            dataset_dims(3) = this%dim_time_id
            do o = 1, n_neut_sp
                ! create group
                ierr = nf90_def_grp(this%file_id, get_neut_name(o), &
                                    this%species_group(o))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
                ! n, particle number
                ierr = nf90_def_var(this%species_group(o), 'n',  &
                                    NF90_GP, dataset_dims, &
                                    this%n_ids(o))
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
            end do
            ierr = nf90_enddef(this%file_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
        endif
        this%write_counter = 1
    end subroutine

    module subroutine write_neut(this, time, lb_neut, ub_neut, &
                                 neutrals_state)
        class(neutrals_file_t), intent(inout) :: this
        real(kind=GP), intent(in) :: time
        integer, intent(in), dimension(4) :: lb_neut, ub_neut
        real(kind=GP), intent(in) :: neutrals_state(&
                                              lb_neut(1) : ub_neut(1), &
                                              lb_neut(2) : ub_neut(2), &
                                              1, &
                                              lb_neut(4) : ub_neut(4))

        integer :: i, k, o, ierr, start(3), start_time(1)

        ! Collect the data on the master process
        !$omp parallel shared(this, neutrals_state) private(k, i, o) &
        !$omp firstprivate(lb_neut, ub_neut) default(none)
        do o = 1, this%n_neut_sp
        do k = 1, this%n_points_phi
        !$omp do schedule(static)
        do i = 1, this%n_points_RZ
            if(      i >= lb_neut(1) &
               .and. i <= ub_neut(1) &
               .and. k >= lb_neut(2) &
               .and. k <= ub_neut(2) &
               .and. o >= lb_neut(4) &
               .and. o <= ub_neut(4) &
              ) then
                this%send_buffer(i, k, 1, o) = neutrals_state(i, k, 1, o)
            else
                this%send_buffer(i, k, 1, o) = 0.0_GP
            endif
        enddo
        !$omp end do nowait
        enddo
        enddo
        !$omp end parallel

        call MPI_Reduce(this%send_buffer, this%recv_buffer, &
                        size(this%send_buffer), MPI_GP, MPI_SUM, &
                        0, this%dcomm_handler%get_comm_phi(), ierr)

        ! Write the data to file
        if(this%dcomm_handler%is_master()) then
            start_time(1) = this%write_counter
            ierr = nf90_put_var(this%file_id, this%time_id, time, &
                                start=start_time)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)

            start(1) = 1
            start(2) = 1
            start(3) = this%write_counter

            do o = 1, this%n_neut_sp
                ! n
                ierr = nf90_put_var(this%species_group(o), &
                                    this%n_ids(o), &
                                    this%recv_buffer(:, :, 1, o), &
                                    start=start)
                call handle_error_netcdf(ierr, __LINE__, __FILE__)
                ! TODO: Add higher moments (velocity, temperature, ...) as the
                !       neutrals evolution model is upgraded.
            end do
            ierr = nf90_sync(this%file_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
        endif

        this%write_counter = this%write_counter + 1
    end subroutine

    module subroutine finalize_neut(this)
        type(neutrals_file_t), intent(inout) :: this
        integer :: ierr

        if(this%dcomm_handler%is_master()) then
            ierr = nf90_close(this%file_id)
            call handle_error_netcdf(ierr, __LINE__, __FILE__)
        endif
    end subroutine

end submodule
