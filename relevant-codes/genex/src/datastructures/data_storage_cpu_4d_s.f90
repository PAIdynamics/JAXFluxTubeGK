submodule (data_storage_m) data_storage_cpu_4d_s
    use mpi, only: MPI_PROC_NULL
    use mail_delivery_m, only: deliver_outboxes, finish_delivery
    use profiler_m, only: profiler_start, profiler_stop

    implicit none

contains

    module subroutine initialize_cpu_4d(this, dcomm_handler, n_mom, n_spec, &
                                        init_value)
        class(data_storage_cpu_4d_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        integer, intent(in) :: n_mom
        integer, intent(in) :: n_spec
        real(kind=GP), optional, intent(in) :: init_value

        integer :: ind, send_width, ex_dim

        call this%initialize_parent(dcomm_handler, dimensions=4)
        ! Re-define the last 2 dimensions initially set from dcomm_handler
        ! i.e. (RZ, phi, vp, mu) -> (RZ, phi, mom, nsp)
        this%ub(3) = n_mom
        this%ub(4) = n_spec
        do ind = 3, 4
            this%lb(ind) = 1
            this%lb_stripped(ind) = this%lb(ind)
            this%ub_stripped(ind) = this%ub(ind)
            this%number_of_elements(ind) = this%ub(ind) - this%lb(ind) + 1
            this%number_of_ghost_cells(ind) = 0
            this%dim_permut(ind) = ind
            this%number_of_data_cells(ind) = this%number_of_elements(ind)
        enddo
        this%size = this%number_of_elements(1) * this%number_of_elements(2) &
                  * this%number_of_elements(3) * this%number_of_elements(4)

        allocate(this%data_array)
        call this%data_array%initialize(this%lb, this%ub, &
                                        this%lb_stripped, this%ub_stripped, &
                                        val=init_value)
        this%storage => this%data_array%get_pointer()

        ! NOTE: We only exchange in phi direction for the 4d datatype
        ex_dim = this%dim_permut(2)
        if(this%number_of_mail_partners(ex_dim) == 2) then
            send_width = this%number_of_ghost_cells(ex_dim)
        else
            send_width = 1
        endif

        this%mailbox%n_mailbox_cells = send_width &
            * this%number_of_data_cells(this%dim_permut(1)) &
            * this%number_of_data_cells(this%dim_permut(3)) &
            * this%number_of_data_cells(this%dim_permut(4))
        allocate(this%mailbox%inboxes( &
                 this%mailbox%n_mailbox_cells, &
                 this%number_of_mail_partners(ex_dim)))
        allocate(this%mailbox%outboxes( &
                 this%mailbox%n_mailbox_cells, &
                 this%number_of_mail_partners(ex_dim)))
        allocate(this%mailbox%requests_in( &
                 this%number_of_mail_partners(ex_dim)))
        allocate(this%mailbox%requests_out(&
                 this%number_of_mail_partners(ex_dim)))
        allocate(this%mailbox%partners( &
                 this%number_of_mail_partners(ex_dim)))
    end subroutine

    subroutine pack_4d(lb_arr_in, arr_in, arr_out, lb, ub, ind)
        !! Packs the elements stored in the part of arr_in specified by lb and
        !! ub into the part of arr_out specified by ind
        integer, dimension(4), intent(in) :: lb_arr_in
        real(kind=GP), dimension(lb_arr_in(1):, lb_arr_in(2):, &
                                 lb_arr_in(3):, lb_arr_in(4):), &
                                                            intent(in) :: arr_in
        real(kind=GP), dimension(:,:), intent(out) :: arr_out
        integer, dimension(4), intent(in) :: lb
        integer, dimension(4), intent(in) :: ub
        integer, intent(in) :: ind
        integer :: shp(4), i, k, m, o, ctr

        shp(1) = (ub(1) - lb(1) + 1)
        do i = 2, 4
            shp(i) = shp(i - 1) * (ub(i) - lb(i) + 1)
        enddo
        !$omp parallel firstprivate(lb, ub, shp, ind) &
        !$omp private(ctr, i, k, m, o) shared(arr_out, arr_in)
        do o = lb(4), ub(4)
        do m = lb(3), ub(3)
        do k = lb(2), ub(2)
        !$omp do schedule(static)
        do i = lb(1), ub(1)
            ctr = (o - lb(4)) * shp(3) &
                + (m - lb(3)) * shp(2) &
                + (k - lb(2)) * shp(1) &
                +  i - lb(1) + 1
            arr_out(ctr, ind) = arr_in(i, k, m, o)
        enddo
        !$omp enddo
        enddo
        enddo
        enddo
        !$omp end parallel
    end subroutine

    subroutine unpack_4d(arr_in, lb_arr_out, arr_out, lb, ub, ind)
        !! Unpacks the elements stored in the part of arr_in specified by ind
        !! into the part of arr_out specified by lb and ub
        real(kind=GP), dimension(:,:), intent(in) :: arr_in
        integer, dimension(4), intent(in) :: lb_arr_out
        real(kind=GP), dimension(lb_arr_out(1):, lb_arr_out(2):, &
                                 lb_arr_out(3):, lb_arr_out(4):), &
                                                          intent(out) :: arr_out
        integer, dimension(4), intent(in) :: lb
        integer, dimension(4), intent(in) :: ub
        integer, intent(in) :: ind
        integer :: shp(4), i, k, m, o, ctr

        shp(1) = (ub(1) - lb(1) + 1)
        do i = 2, 4
            shp(i) = shp(i - 1) * (ub(i) - lb(i) + 1)
        enddo
        !$omp parallel firstprivate(lb, ub, shp, ind) &
        !$omp private(ctr, i, k, m, o) shared(arr_out, arr_in)
        do o = lb(4), ub(4)
        do m = lb(3), ub(3)
        do k = lb(2), ub(2)
        !$omp do schedule(static)
        do i = lb(1), ub(1)
            ctr = (o - lb(4)) * shp(3) &
                + (m - lb(3)) * shp(2) &
                + (k - lb(2)) * shp(1) &
                +  i - lb(1) + 1
            arr_out(i, k, m, o) = arr_in(ctr, ind)
        enddo
        !$omp enddo
        enddo
        enddo
        enddo
        !$omp end parallel
    end subroutine

    module subroutine start_exchange_cpu_4d(this)
        class(data_storage_cpu_4d_t), intent(inout) :: this
        integer :: i, ex_dim
        integer :: lb_pack(4), ub_pack(4)
        integer :: ierr

        call profiler_start("pack_4d", ierr)

        ex_dim = this%dim_permut(2)
        do i = 1, 4
            lb_pack(i) = this%lb_stripped(i)
            ub_pack(i) = this%ub_stripped(i)
        end do

        if(this%number_of_mail_partners(ex_dim) == 4) then
            ! We have four mail partners but only 1 data cell ourselves
            lb_pack(ex_dim) = this%lb_stripped(ex_dim)
            ub_pack(ex_dim) = this%lb_stripped(ex_dim)
            do i = 1, 4
                call pack_4d(this%lb, this%storage, &
                             this%mailbox%outboxes, &
                             lb_pack, ub_pack, i)
            enddo
        else
            ! Left neighbor
            lb_pack(ex_dim) = this%lb_stripped(ex_dim)
            ub_pack(ex_dim) = this%lb_stripped(ex_dim) &
                + this%number_of_ghost_cells(ex_dim) - 1
            call pack_4d(this%lb, this%storage, &
                         this%mailbox%outboxes, &
                         lb_pack, ub_pack, 1)
            ! Right neighbor
            lb_pack(ex_dim) = this%ub_stripped(ex_dim) &
                - this%number_of_ghost_cells(ex_dim) + 1
            ub_pack(ex_dim) = this%ub_stripped(ex_dim)
            call pack_4d(this%lb, this%storage, &
                         this%mailbox%outboxes, &
                         lb_pack, ub_pack, 2)
        endif
        call profiler_stop("pack_4d", ierr)
        call profiler_start("send_4d", ierr)

        call deliver_outboxes(this%dcomm_handler, &
                              this%mailbox%n_mailbox_cells, &
                              this%number_of_mail_partners(ex_dim), &
                              2, &
                              this%mailbox%inboxes, &
                              this%mailbox%outboxes, &
                              this%mailbox%partners, &
                              this%mailbox%requests_in, &
                              this%mailbox%requests_out)
        call profiler_stop("send_4d", ierr)

    end subroutine

    module subroutine finish_exchange_cpu_4d(this)
        class(data_storage_cpu_4d_t), intent(inout) :: this
        integer :: i, ex_dim
        integer :: lb_pack(4), ub_pack(4)
        integer :: ghost_indices(4)
        integer :: ierr

        ex_dim = this%dim_permut(2)

        call profiler_start("receive_4d", ierr)

        call finish_delivery(this%number_of_mail_partners(ex_dim), &
                             this%mailbox%partners, &
                             this%mailbox%requests_in, &
                             this%mailbox%requests_out)
        call profiler_stop("receive_4d", ierr)
        call profiler_start("unpack_4d", ierr)

        do i = 1, 4
            lb_pack(i) = this%lb_stripped(i)
            ub_pack(i) = this%ub_stripped(i)
        enddo

        if(this%number_of_mail_partners(ex_dim) == 4) then
            ghost_indices(1) = this%lb(ex_dim)
            ghost_indices(2) = this%lb_stripped(ex_dim) - 1
            ghost_indices(3) = this%ub_stripped(ex_dim) + 1
            ghost_indices(4) = this%ub(ex_dim)

            do i = 1, 4
                lb_pack(ex_dim) = ghost_indices(i)
                ub_pack(ex_dim) = ghost_indices(i)
                ! Unpack if the mail partner exists
                if(this%mailbox%partners(i) /= MPI_PROC_NULL) then
                    call unpack_4d(this%mailbox%inboxes, &
                                   this%lb, this%storage, &
                                   lb_pack, ub_pack, i)
                endif
            end do
        else
            ! Unpack if the left mail partner exists
            if(this%mailbox%partners(1) /= MPI_PROC_NULL) then
                lb_pack(ex_dim) = this%lb(ex_dim)
                ub_pack(ex_dim) = this%lb_stripped(ex_dim) - 1
                call unpack_4d(this%mailbox%inboxes, &
                               this%lb, this%storage, &
                               lb_pack, ub_pack, 1)
            endif
            ! Unpack if the right mail partner exists
            if(this%mailbox%partners(2) /= MPI_PROC_NULL) then
                lb_pack(ex_dim) = this%ub_stripped(ex_dim) + 1
                ub_pack(ex_dim) = this%ub(ex_dim)
                call unpack_4d(this%mailbox%inboxes, &
                               this%lb, this%storage, &
                               lb_pack, ub_pack, 2)
            endif
        endif
        call profiler_stop("unpack_4d", ierr)

    end subroutine

end submodule
