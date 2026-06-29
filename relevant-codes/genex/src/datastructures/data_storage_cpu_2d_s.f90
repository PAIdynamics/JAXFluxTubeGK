submodule (data_storage_m) data_storage_cpu_2d_s
    use mpi, only: MPI_PROC_NULL
    use mail_delivery_m, only: deliver_outboxes, finish_delivery
    use profiler_m, only: profiler_start, profiler_stop

    implicit none

contains

    module subroutine initialize_cpu_2d(this, dcomm_handler, init_value)
        class(data_storage_cpu_2d_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        real(kind=GP), optional, intent(in) :: init_value

        integer :: send_width, ex_dim

        call this%initialize_parent(dcomm_handler, dimensions=2)

        allocate(this%data_array)
        call this%data_array%initialize(this%lb, this%ub, &
                                        this%lb_stripped, this%ub_stripped, &
                                        val=init_value)
        this%storage => this%data_array%get_pointer()

        ! NOTE: We can only exchange in phi direction for the 2d datatype
        ex_dim = this%dim_permut(2)
        if(this%number_of_mail_partners(ex_dim) == 2) then
            send_width = this%number_of_ghost_cells(ex_dim)
        else
            send_width = 1
        endif

        this%mailbox%n_mailbox_cells = send_width &
            * this%number_of_data_cells(this%dim_permut(1))

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

    subroutine pack_2d(lb_arr_in, arr_in, arr_out, lb, ub, ind)
        !! Packs the elements stored in the part of arr_in specified by lb and
        !! ub into the part of arr_out specified by ind
        integer, dimension(2), intent(in) :: lb_arr_in
        real(kind=GP), dimension(lb_arr_in(1):, &
                                 lb_arr_in(2):), intent(in) :: arr_in
        real(kind=GP), dimension(:,:), intent(out) :: arr_out
        integer, dimension(2), intent(in) :: lb
        integer, dimension(2), intent(in) :: ub
        integer, intent(in) :: ind
        integer :: shp(2), i, k, ctr

        shp = ub - lb + 1
        ! NOTE: The OpenMP code will only execute in full speed if
        !       dim_permut(1) = DIM_RZ. In the future, if different dim_permuts
        !       are actually used, one has to implement a branch cut with a
        !       specific OpenMP version for a specific dim_permut here and in
        !       unpack_2d
        !$omp parallel firstprivate(lb, ub, shp, ind) &
        !$omp private(ctr, i, k) shared(arr_out, arr_in)
        do k = lb(2), ub(2)
            !$omp do schedule(static)
            do i = lb(1), ub(1)
                ctr = (k - lb(2)) * shp(1) + (i - lb(1) + 1)
                arr_out(ctr, ind) = arr_in(i, k)
            enddo
            !$omp enddo
        enddo
        !$omp end parallel
    end subroutine

    subroutine unpack_2d(arr_in, lb_arr_out, arr_out, lb, ub, ind)
        !! Unpacks the elements stored in the part of arr_in specified by ind
        !! into the part of arr_out specified by lb and ub
        real(kind=GP), dimension(:,:), intent(in) :: arr_in
        integer, dimension(2), intent(in) :: lb_arr_out
        real(kind=GP), dimension(lb_arr_out(1):, &
                                 lb_arr_out(2):), intent(out) :: arr_out
        integer, dimension(2), intent(in) :: lb
        integer, dimension(2), intent(in) :: ub
        integer, intent(in) :: ind
        integer :: shp(2), i, k, ctr

        shp = ub - lb + 1

        !$omp parallel firstprivate(lb, ub, shp, ind) &
        !$omp private(ctr, i, k) shared(arr_out, arr_in)
        do k = lb(2), ub(2)
            !$omp do schedule(static)
            do i = lb(1), ub(1)
                ctr = (k - lb(2)) * shp(1) + i - lb(1) + 1
                arr_out(i, k) = arr_in(ctr, ind)
            enddo
            !$omp enddo
        enddo
        !$omp end parallel
    end subroutine

    module subroutine start_exchange_cpu_2d(this)
        class(data_storage_cpu_2d_t), intent(inout) :: this
        integer :: i, ex_dim
        integer :: lb_pack(5), ub_pack(5)
        integer :: ierr

        call profiler_start("pack_2d", ierr)

        ex_dim = this%dim_permut(2)
        do i = 1, 2
            lb_pack(i) = this%lb_stripped(i)
            ub_pack(i) = this%ub_stripped(i)
        end do

        if(this%number_of_mail_partners(ex_dim) == 4) then
            ! We have four mail partners but only 1 data cell ourselves
            lb_pack(ex_dim) = this%lb_stripped(ex_dim)
            ub_pack(ex_dim) = this%lb_stripped(ex_dim)
            ! Pack
            do i = 1, 4
                call pack_2d(this%lb, this%storage, &
                             this%mailbox%outboxes, &
                             lb_pack, ub_pack, i)
            enddo
        else
            ! Left neighbor
            lb_pack(ex_dim) = this%lb_stripped(ex_dim)
            ub_pack(ex_dim) = this%lb_stripped(ex_dim) &
                + this%number_of_ghost_cells(ex_dim) - 1
            call pack_2d(this%lb, this%storage, &
                         this%mailbox%outboxes, &
                         lb_pack, ub_pack, 1)
            ! Right neighbor
            lb_pack(ex_dim) = this%ub_stripped(ex_dim) &
                - this%number_of_ghost_cells(ex_dim) + 1
            ub_pack(ex_dim) = this%ub_stripped(ex_dim)
            call pack_2d(this%lb, this%storage, &
                         this%mailbox%outboxes, &
                         lb_pack, ub_pack, 2)
        endif
        call profiler_stop("pack_2d", ierr)
        call profiler_start("send_2d", ierr)

        call deliver_outboxes(this%dcomm_handler, &
                              this%mailbox%n_mailbox_cells, &
                              this%number_of_mail_partners(ex_dim), &
                              2, &
                              this%mailbox%inboxes, &
                              this%mailbox%outboxes, &
                              this%mailbox%partners, &
                              this%mailbox%requests_in, &
                              this%mailbox%requests_out)
        call profiler_stop("send_2d", ierr)

    end subroutine

    module subroutine finish_exchange_cpu_2d(this)
        class(data_storage_cpu_2d_t), intent(inout) :: this
        integer :: i, ex_dim
        integer :: lb_pack(5), ub_pack(5)
        integer :: ghost_indices(4)
        integer :: ierr

        ! We only exchange in phi direction
        ex_dim = this%dim_permut(2)

        call profiler_start("receive_2d", ierr)

        call finish_delivery(this%number_of_mail_partners(ex_dim), &
                             this%mailbox%partners, &
                             this%mailbox%requests_in, &
                             this%mailbox%requests_out)
        call profiler_stop("receive_2d", ierr)
        call profiler_start("unpack_2d", ierr)

        ! Set generic lower and upper bounds for the data to unpack
        do i = 1, 2
            lb_pack(i) = this%lb_stripped(i)
            ub_pack(i) = this%ub_stripped(i)
        enddo

        ! Unpack data
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
                    call unpack_2d(this%mailbox%inboxes, &
                                   this%lb, this%storage, &
                                   lb_pack, ub_pack, i)
                endif
            end do
        else
            ! Unpack if the left mail partner exists
            if(this%mailbox%partners(1) /= MPI_PROC_NULL) then
                lb_pack(ex_dim) = this%lb(ex_dim)
                ub_pack(ex_dim) = this%lb_stripped(ex_dim) - 1
                call unpack_2d(this%mailbox%inboxes, &
                               this%lb, this%storage, &
                               lb_pack, ub_pack, 1)
            endif
            ! Unpack if the right mail partner exists
            if(this%mailbox%partners(2) /= MPI_PROC_NULL) then
                lb_pack(ex_dim) = this%ub_stripped(ex_dim) + 1
                ub_pack(ex_dim) = this%ub(ex_dim)
                call unpack_2d(this%mailbox%inboxes, &
                               this%lb, this%storage, &
                               lb_pack, ub_pack, 2)
            endif
        endif
        call profiler_stop("unpack_2d", ierr)

    end subroutine

end submodule
