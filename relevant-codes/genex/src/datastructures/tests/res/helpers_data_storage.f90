module helpers_data_storage_m
!! Contains procedures needed for unit testing the mail_delivery_m module
    use pfunit
    use mpi
    use dimensions_m, only: DIM_PHI, DIM_VP, DIM_MU
    use genex_fortran_env_m, only : GP, GP_EPS
    use dcomm_handler_m, only : dcomm_handler_t
    use data_storage_m, only: data_storage_5d_t, data_storage_cpu_5d_t, &
                              data_storage_4d_t, data_storage_cpu_4d_t, &
                              data_storage_2d_t, data_storage_cpu_2d_t
    use math_m, only : almost_equal
    use params_gpu_offload_m, only: get_use_gpu_offload
    ! Unit test helpers
    use test_params_m
    ! From PARALLAX
    use equilibrium_factory_m, only: SLAB

#ifdef ENABLE_GPU
    use data_storage_m, only: data_storage_gpu_5d_t, data_storage_gpu_4d_t, &
                              data_storage_gpu_2d_t
#endif

    implicit none

contains
    function get_left_rank(comm) result(left_rank)
        integer, intent(in) :: comm
        integer :: ierr, left_rank, right_rank
        call mpi_cart_shift(comm, 0, 1, left_rank, right_rank, ierr)
    end function

    function get_sec_left_rank(comm) result(left_rank)
        integer, intent(in) :: comm
        integer :: ierr, left_rank, right_rank
        call mpi_cart_shift(comm, 0, 2, left_rank, right_rank, ierr)
    end function

    function get_right_rank(comm) result(right_rank)
        integer, intent(in) :: comm
        integer :: ierr, left_rank, right_rank
        call mpi_cart_shift(comm, 0, 1, left_rank, right_rank, ierr)
    end function

    function get_sec_right_rank(comm) result(right_rank)
        integer, intent(in) :: comm
        integer :: ierr, left_rank, right_rank
        call mpi_cart_shift(comm, 0, 2, left_rank, right_rank, ierr)
    end function

    function eval_5d_ptr_at(lb_stripped, ptr, ex_dim, index) result(res)
        integer, intent(in) :: lb_stripped(5)
        real(kind=GP), dimension(:,:,:,:,:), pointer, intent(in) :: ptr
        integer, intent(in) :: ex_dim
        integer, intent(in) :: index
        real(kind=GP) :: res

        select case(ex_dim)
        case(1)
            res = ptr(index, lb_stripped(2), lb_stripped(3), lb_stripped(4), &
                      lb_stripped(5))
        case(2)
            res = ptr(lb_stripped(1), index, lb_stripped(3), lb_stripped(4), &
                      lb_stripped(5))
        case(3)
            res = ptr(lb_stripped(1), lb_stripped(2), index, lb_stripped(4), &
                      lb_stripped(5))
        case(4)
            res = ptr(lb_stripped(1), lb_stripped(2), lb_stripped(3), index, &
                      lb_stripped(5))
        case(5)
            res = ptr(lb_stripped(1), lb_stripped(2), lb_stripped(3), &
                      lb_stripped(4), index)
        end select
    end function

    function eval_4d_ptr_at(lb_stripped, ptr, ex_dim, index) result(res)
        integer, intent(in) :: lb_stripped(4)
        real(kind=GP), dimension(:,:,:,:), pointer, intent(in) :: ptr
        integer, intent(in) :: ex_dim
        integer, intent(in) :: index
        real(kind=GP) :: res

        select case(ex_dim)
        case(1)
            res = ptr(index, lb_stripped(2), lb_stripped(3), lb_stripped(4))
        case(2)
            res = ptr(lb_stripped(1), index, lb_stripped(3), lb_stripped(4))
        case(3)
            res = ptr(lb_stripped(1), lb_stripped(2), index, lb_stripped(4))
        case(4)
            res = ptr(lb_stripped(1), lb_stripped(2), lb_stripped(3), index)
        end select
    end function

    function eval_2d_ptr_at(lb_stripped, ptr, ex_dim, index) result(res)
        integer, intent(in) :: lb_stripped(2)
        real(kind=GP), dimension(:,:), pointer, intent(in) :: ptr
        integer, intent(in) :: ex_dim
        integer, intent(in) :: index
        real(kind=GP) :: res

        select case(ex_dim)
        case(1)
            res = ptr(index, lb_stripped(2))
        case(2)
            res = ptr(lb_stripped(1), index)
        end select
    end function

    function check_coord_bounds(lbounds, ubounds, coords) result(res)
        integer, intent(in) :: lbounds(5), ubounds(5)
        integer, intent(in) :: coords(5)
        logical :: res
        res = .true.
        if(coords(1) < lbounds(1) &
           .or. coords(1) > ubounds(1)) res = .false.
        if(coords(2) < lbounds(2) &
           .or. coords(2) > ubounds(2)) res = .false.
        if(coords(3) < lbounds(3) &
           .or. coords(3) > ubounds(3)) res = .false.
        if(coords(4) < lbounds(4) &
           .or. coords(4) > ubounds(4)) res = .false.
        if(coords(5) < lbounds(5) &
           .or. coords(5) > ubounds(5)) res = .false.
    end function

    subroutine set_ranks_for_ex_dims(n_ranks_phi, n_ranks_vp, n_ranks_mu, &
                                     n_ranks, ex_dim)
        integer, intent(inout) :: n_ranks_phi, n_ranks_vp, n_ranks_mu
        integer, intent(in) :: n_ranks, ex_dim
        n_ranks_phi = 1
        n_ranks_mu = 1
        n_ranks_vp = 1
        select case(ex_dim)
        case(2)
            n_ranks_phi = n_ranks
        case(3)
            n_ranks_vp = n_ranks
        case(4)
            n_ranks_mu = n_ranks
        end select
    end subroutine

    function test_exchange_5d(test, &
                              n_ranks_phi, &
                              n_ranks_vp, &
                              n_ranks_mu, &
                              n_ranks_sp, &
                              n_points, &
                              ex_dim, &
                              dim_permut) result(res)
        class(MpiTestMethod), intent(inout) :: test
        !! Test instance
        integer, intent(in) :: n_ranks_phi
        !! number of ranks in phi direction
        integer, intent(in) :: n_ranks_vp
        !! number of ranks in vp direction
        integer, intent(in) :: n_ranks_mu
        !! number of ranks in mu direction
        integer, intent(in) :: n_ranks_sp
        !! number of ranks in sp direction
        integer, intent(in) :: n_points
        !! scalar number of points
        integer, intent(in) :: ex_dim
        !! exchange dimension to test
        integer, dimension(5), intent(in) :: dim_permut

        integer :: comm_world, rank, n_ghosts
        type(dcomm_handler_t), target :: dcomm_handler
        type(dcomm_handler_t), pointer :: dcomm_handler_ptr
        class(data_storage_5d_t), allocatable :: ds
        real(kind=GP), dimension(:,:,:,:,:), pointer :: data_ptr
        real(kind=GP) :: expected
        integer, parameter :: dimensions = 5
        integer :: number_of_points(dimensions)
        integer :: lb(dimensions), lb_stripped(dimensions)
        integer :: ub(dimensions), ub_stripped(dimensions)
        integer :: ex_comm, n_ranks_ex_dim
        integer :: left_rank, sec_left_rank, right_rank, sec_right_rank
        logical :: fine, res
        integer :: i

        rank = test%getProcessRank()
        comm_world = test%getMpiCommunicator()

        res = .true.

        number_of_points = [1, n_points, n_points, n_points, 2]
        call setup_test_mesh(comm_world, &
                             rank, &
                             equilibrium_type=SLAB, &
                             spacing_RZ=64.0_GP&
                                       /number_of_points(dim_permut(1)), &
                             n_levels=1, &
                             n_points_phi=number_of_points(dim_permut(2)), &
                             n_points_vp=number_of_points(dim_permut(3)), &
                             n_points_mu=number_of_points(dim_permut(4)))
        ! NOTE: We setup the coll type to LBD since this enables
        !       ghost exchange in mu
        call setup_test_coll(comm_world, rank, coll_type="lbd")

        call dcomm_handler%initialize(comm_world, n_ranks_phi, &
                                      n_ranks_vp, n_ranks_mu, n_ranks_sp, &
                                      dim_permut)
        dcomm_handler_ptr => dcomm_handler
        call dcomm_handler_ptr%initialize_RZ_domain(&
                                            number_of_points(dim_permut(1)))

        n_ghosts = 0
        if(ex_dim == dim_permut(2)) then
            ex_comm = dcomm_handler%get_comm_phi()
            n_ranks_ex_dim = n_ranks_phi
            n_ghosts = 2
        else if(ex_dim == dim_permut(3)) then
            ex_comm = dcomm_handler%get_comm_vp()
            n_ranks_ex_dim = n_ranks_vp
            n_ghosts = 2
        else if(ex_dim == dim_permut(4)) then
            ex_comm = dcomm_handler%get_comm_mu()
            n_ranks_ex_dim = n_ranks_mu
            n_ghosts = 1
        end if

        if(get_use_gpu_offload()) then
#ifdef ENABLE_GPU
            allocate(data_storage_gpu_5d_t :: ds)
#endif
        else
            allocate(data_storage_cpu_5d_t :: ds)
        endif

        call ds%initialize(dcomm_handler_ptr, init_value=1.0_GP * rank)

        data_ptr => ds%get_pointer()
        lb = ds%lbound()
        ub = ds%ubound()
        lb_stripped = ds%lbound_stripped()
        ub_stripped = ds%ubound_stripped()

        call ds%start_exchange(ex_dim)
        call ds%finish_exchange(ex_dim)
        call ds%update_host()

        if(n_points / n_ranks_ex_dim < n_ghosts) then
            ! n_ghosts == 2, n_poinst ==1
            ! left neighbors
            sec_left_rank = get_sec_left_rank(ex_comm)
            if(sec_left_rank /= MPI_PROC_NULL) then
                expected = real(sec_left_rank, kind=GP)
                fine =  almost_equal(eval_5d_ptr_at(lb_stripped, data_ptr, &
                                                    ex_dim, &
                                                    lb_stripped(ex_dim) -2), &
                                     expected, GP_EPS)
                if(.not. fine) res = .false.
            endif

            left_rank = get_left_rank(ex_comm)
            if(left_rank /= MPI_PROC_NULL) then
                expected = real(left_rank, kind=GP)
                fine =  almost_equal(eval_5d_ptr_at(lb_stripped, data_ptr, &
                                                    ex_dim, &
                                                    lb_stripped(ex_dim) -1), &
                                     expected, GP_EPS)
                if(.not. fine) res = .false.
            endif
            ! right neighbors
            right_rank = get_right_rank(ex_comm)
            if(right_rank /= MPI_PROC_NULL) then
                expected = real(right_rank, kind=GP)
                fine =  almost_equal(eval_5d_ptr_at(lb_stripped, data_ptr, &
                                                    ex_dim, &
                                                    ub_stripped(ex_dim) + 1), &
                                     expected, GP_EPS)
                if(.not. fine) res = .false.
            endif

            sec_right_rank = get_sec_right_rank(ex_comm)
            if(sec_right_rank /= MPI_PROC_NULL) then
                expected = real(sec_right_rank, kind=GP)
                fine =  almost_equal(eval_5d_ptr_at(lb_stripped, data_ptr, &
                                                    ex_dim, &
                                                    ub_stripped(ex_dim) + 2),&
                                     expected, GP_EPS)
                if(.not. fine) res = .false.
            endif
        else
            do i = lb(ex_dim), lb_stripped(ex_dim) - 1
                left_rank = get_left_rank(ex_comm)
                if(left_rank /= MPI_PROC_NULL) then
                    expected = real(left_rank, kind=GP)
                    fine =  almost_equal(eval_5d_ptr_at(lb_stripped, data_ptr,&
                                                        ex_dim, i), &
                                         expected, GP_EPS)
                    if(.not. fine) res = .false.
                endif
            enddo
            do i = ub_stripped(ex_dim) + 1, ub(ex_dim)
                right_rank = get_right_rank(ex_comm)
                if(right_rank /= MPI_PROC_NULL) then
                    expected = real(right_rank, kind=GP)
                    fine =  almost_equal(eval_5d_ptr_at(lb_stripped, data_ptr,&
                                                        ex_dim, i), &
                                         expected, GP_EPS)
                    if(.not. fine) res = .false.
                endif
            enddo
        endif

        deallocate(ds)
    end function test_exchange_5d

    function test_exchange_5d_vspec(test, &
                                    n_ranks_phi, &
                                    n_ranks_vp, &
                                    n_ranks_mu, &
                                    n_ranks_sp, &
                                    n_points, &
                                    ex_dim, &
                                    dim_permut) result(res)
        class(MpiTestMethod), intent(inout) :: test
        !! Test instance
        integer, intent(in) :: n_ranks_phi
        !! number of ranks in phi direction
        integer, intent(in) :: n_ranks_vp
        !! number of ranks in vp direction
        integer, intent(in) :: n_ranks_mu
        !! number of ranks in mu direction
        integer, intent(in) :: n_ranks_sp
        !! number of ranks in sp direction
        integer, intent(in) :: n_points
        !! scalar number of points
        integer, intent(in) :: ex_dim
        !! exchange dimension to test
        integer, dimension(5), intent(in) :: dim_permut

        integer :: rank, comm_world, &
                   n_ranks_ex_dim, ex_comm, &
                   left_rank, right_rank, i, j, m, l, &
                   n_ghosts_vp, n_ghosts_mu, n_ghosts_phi

        type(dcomm_handler_t), target :: dcomm_handler
        type(dcomm_handler_t), pointer :: dcomm_handler_ptr
        class(data_storage_5d_t), allocatable :: ds
        real(kind=GP), dimension(:,:,:,:,:), pointer :: data_ptr
        integer, parameter :: dimensions = 5

        integer :: number_of_points(dimensions)
        integer :: lb(dimensions), lb_stripped(dimensions)
        integer :: ub(dimensions), ub_stripped(dimensions)
        integer :: bounds(dimensions)
        integer :: ghost_bounds_mu(2), ghost_bounds_vp(4)
        real(kind=GP) :: expected
        logical :: fine, res

        res = .true.

        rank = test%getProcessRank()
        comm_world = test%getMpiCommunicator()

        number_of_points = [1, n_points, n_points, n_points, 2]

        ! Setup mesh with spectral method
        call setup_test_mesh(comm_world, &
                             rank, &
                             equilibrium_type=SLAB, &
                             spacing_RZ=64.0_GP&
                                     /number_of_points(dim_permut(1)), &
                             n_levels=1, &
                             n_points_phi=number_of_points(dim_permut(2)), &
                             n_points_vp=number_of_points(dim_permut(3)), &
                             n_points_mu=number_of_points(dim_permut(4)), &
                             use_vspectral = .true.)

        call dcomm_handler%initialize(comm_world, n_ranks_phi, &
                                      n_ranks_vp, n_ranks_mu, n_ranks_sp, &
                                      dim_permut)

        dcomm_handler_ptr => dcomm_handler
        call dcomm_handler_ptr%initialize_RZ_domain(&
                                            number_of_points(dim_permut(1)))

        if(ex_dim == dim_permut(DIM_PHI)) then
            ex_comm = dcomm_handler%get_comm_phi()
            n_ranks_ex_dim = n_ranks_phi
        else if(ex_dim == dim_permut(DIM_VP)) then
            ex_comm = dcomm_handler%get_comm_vp()
            n_ranks_ex_dim = n_ranks_vp
        else if(ex_dim == dim_permut(DIM_MU)) then
            ex_comm = dcomm_handler%get_comm_mu()
            n_ranks_ex_dim = n_ranks_mu
        end if

        n_ghosts_phi = 2
        n_ghosts_vp = 2
        n_ghosts_mu = 1

        ! Allocate data storage
        if(allocated(ds)) &
            deallocate(ds)
        allocate(data_storage_cpu_5d_t :: ds)

        ! Initialize ds with values equal to the rank where it is
        ! stored. All the points (ghosts and inner cells) are set to
        ! value = rank
        call ds%initialize(dcomm_handler_ptr, init_value=1.0_GP * rank)

        data_ptr => ds%get_pointer()
        lb = ds%lbound()
        ub = ds%ubound()
        lb_stripped = ds%lbound_stripped()
        ub_stripped = ds%ubound_stripped()

        ghost_bounds_vp = [lb(DIM_VP), lb(DIM_VP) + 1, &
                           ub(DIM_VP) - 1, ub(DIM_VP)]
        ghost_bounds_mu = [lb(DIM_MU), ub(DIM_MU)]

        ! Exchange the ghost along ex_dim
        call ds%start_exchange(ex_dim)
        call ds%finish_exchange(ex_dim)

        ! Test left exchange
        fine = .true.
        do i = lb(ex_dim), lb_stripped(ex_dim) - 1
            left_rank = get_left_rank(ex_comm)
            bounds = lb_stripped
            if(left_rank /= MPI_PROC_NULL) then
                ! Test inner points
                expected = real(left_rank, kind = GP)
                fine = almost_equal(expected, &
                                    eval_5d_ptr_at(bounds, data_ptr,&
                                                   ex_dim, i), &
                                    GP_EPS)
                if(.not. fine) res = .false.

                ! Test ghost points
                if(ex_dim == dim_permut(DIM_VP)) then
                    expected = real(rank, kind = GP)
                    do j = 1, n_ghosts_mu
                        bounds(DIM_MU) = ghost_bounds_mu(j)
                        fine = almost_equal(expected, &
                                            eval_5d_ptr_at(bounds, data_ptr,&
                                                           ex_dim, i), &
                                            GP_EPS)
                        if(.not. fine) res = .false.
                    enddo

                elseif(ex_dim == dim_permut(DIM_MU)) then
                    expected = real(left_rank, kind = GP)
                    do j = 1, n_ghosts_vp
                        bounds(DIM_VP) = ghost_bounds_vp(j)
                        fine = almost_equal(expected, &
                                        eval_5d_ptr_at(bounds, data_ptr,&
                                                       ex_dim, i), &
                                        GP_EPS)
                        if(.not. fine) res = .false.
                    enddo
                elseif(ex_dim == dim_permut(DIM_PHI)) then
                    expected = real(left_rank, kind=GP)
                    bounds = lb_stripped
                    do l = 1, n_ghosts_vp
                    do m = 1, n_ghosts_mu
                        bounds(DIM_VP) = ghost_bounds_vp(l)
                        bounds(DIM_MU) = ghost_bounds_mu(m)
                        fine = almost_equal(expected, &
                                            eval_5d_ptr_at(bounds, data_ptr,&
                                                           ex_dim, i), &
                                            GP_EPS)
                        if(.not. fine) res = .false.
                    enddo
                    enddo
                endif
             endif
        enddo

        ! Test right exchange
        do i = ub_stripped(ex_dim) + 1, ub(ex_dim)
            right_rank = get_right_rank(ex_comm)
            if(right_rank /= MPI_PROC_NULL) then
                ! This inner points
                expected = real(right_rank, kind = GP)
                bounds = lb_stripped
                fine = almost_equal(expected, &
                                    eval_5d_ptr_at(bounds, data_ptr,&
                                                   ex_dim, i), &
                                    GP_EPS)
                if(.not. fine) res = .false.

                if(ex_dim == dim_permut(DIM_VP)) then
                    expected = real(rank, kind = GP)
                    do j = 1, n_ghosts_mu
                        bounds(DIM_MU) = ghost_bounds_mu(j)
                        fine = almost_equal(expected, &
                                            eval_5d_ptr_at(bounds, data_ptr,&
                                                           ex_dim, i), &
                                            GP_EPS)
                        if(.not. fine) res = .false.
                    enddo
                elseif(ex_dim == dim_permut(DIM_MU)) then
                    expected = real(right_rank, kind = GP)
                    do j = 1, n_ghosts_vp
                        bounds(DIM_VP) = ghost_bounds_vp(j)
                        fine = almost_equal(expected, &
                                             eval_5d_ptr_at(bounds, data_ptr,&
                                                            ex_dim, i), &
                                             GP_EPS)
                        if(.not. fine) res = .false.
                   enddo
                elseif(ex_dim == dim_permut(DIM_PHI)) then
                    expected = real(right_rank, kind=GP)
                    bounds = lb_stripped
                    do l = 1, n_ghosts_vp
                    do m = 1, n_ghosts_mu
                        bounds(DIM_VP) = ghost_bounds_vp(l)
                        bounds(DIM_MU) = ghost_bounds_mu(m)
                        fine = almost_equal(expected, &
                                            eval_5d_ptr_at(bounds, data_ptr,&
                                                           ex_dim, i), &
                                            GP_EPS)
                        if(.not. fine) res = .false.
                    enddo
                    enddo
                endif
            endif
        enddo
    end function

    function test_exchange_5d_vpmu_vspec(test, &
                                         n_ranks_phi, &
                                         n_ranks_vp, &
                                         n_ranks_mu, &
                                         n_ranks_sp, &
                                         n_points, &
                                         dim_permut) result(res)
        class(MpiTestMethod), intent(inout) :: test
        !! Test instance
        integer, intent(in) :: n_ranks_phi
        !! number of ranks in phi direction
        integer, intent(in) :: n_ranks_vp
        !! number of ranks in vp direction
        integer, intent(in) :: n_ranks_mu
        !! number of ranks in mu direction
        integer, intent(in) :: n_ranks_sp
        !! number of ranks in sp direction
        integer, intent(in) :: n_points
        !! scalar number of points
        integer, dimension(5), intent(in) :: dim_permut

        integer :: rank, comm_world, cart_comm, ex_comm, &
                   left_rank, right_rank, up_rank, down_rank, &
                   ngb_rank, l, m, ierr, &
                   n_ghosts_vp, n_ghosts_mu

        integer, parameter :: dimensions = 5
        integer :: number_of_points(dimensions)
        integer :: lb(dimensions), lb_stripped(dimensions)
        integer :: ub(dimensions), ub_stripped(dimensions)
        integer :: bounds(dimensions)
        integer :: coord_ubounds(dimensions), coord_lbounds(dimensions)
        integer :: coords(dimensions), ngb_coords(dimensions)
        integer :: ghost_bounds_mu(2), ghost_bounds_vp(4)

        type(dcomm_handler_t), target :: dcomm_handler
        type(dcomm_handler_t), pointer :: dcomm_handler_ptr
        class(data_storage_5d_t), allocatable :: ds
        real(kind=GP), dimension(:,:,:,:,:), pointer :: data_ptr

        logical :: res, fine, coords_fine
        real(kind=GP) :: expected

        rank = test%getProcessRank()
        comm_world = test%getMpiCommunicator()

        res = .true.

        number_of_points = [1, n_points, n_points, n_points, 2]

        n_ghosts_vp = 2
        n_ghosts_mu = 1

        coord_lbounds = [0, 0, 0, 0, 0]
        coord_ubounds = [0, n_ranks_phi - 1, &
                         n_ranks_vp - 1, n_ranks_mu -1, 0]

        ! Setup mesh with spectral method
        call setup_test_mesh(comm_world, &
                             rank, &
                             equilibrium_type=SLAB, &
                             spacing_RZ=64.0_GP&
                                     /number_of_points(dim_permut(1)), &
                             n_levels=1, &
                             n_points_phi=number_of_points(dim_permut(2)), &
                             n_points_vp=number_of_points(dim_permut(3)), &
                             n_points_mu=number_of_points(dim_permut(4)), &
                             use_vspectral = .true.)

        call dcomm_handler%initialize(comm_world, n_ranks_phi, &
                                      n_ranks_vp, n_ranks_mu, n_ranks_sp, &
                                      dim_permut)
        dcomm_handler_ptr => dcomm_handler
        call dcomm_handler_ptr%initialize_RZ_domain(&
                                            number_of_points(dim_permut(1)))

        coords = dcomm_handler_ptr%get_cart_coords()
        cart_comm = dcomm_handler_ptr%get_comm_cart()

        ! Allocate data storage
        if(allocated(ds)) &
            deallocate(ds)
        allocate(data_storage_cpu_5d_t :: ds)

        ! Initialize ds with values equal to the rank where it is
        ! stored. All the points (ghosts and inner cells) are set to
        ! value = rank
        call ds%initialize(dcomm_handler_ptr, init_value=1.0_GP * rank)

        data_ptr => ds%get_pointer()

        lb = ds%lbound()
        ub = ds%ubound()
        lb_stripped = ds%lbound_stripped()
        ub_stripped = ds%ubound_stripped()

        ! Ghost bounds (from left to right)
        ghost_bounds_vp = [lb(DIM_VP), lb(DIM_VP) + 1, &
                           ub(DIM_VP) - 1, ub(DIM_VP)]
        ghost_bounds_mu = [lb(DIM_MU), ub(DIM_MU)]

        ! Exchange ghosts along vp (excluding mu ghosts)
        call ds%start_exchange(DIM_VP)
        call ds%finish_exchange(DIM_VP)

        ! Exchange ghost along mu (including vp ghosts)
        call ds%start_exchange(DIM_MU)
        call ds%finish_exchange(DIM_MU)

        ! Test edges ghost
        ex_comm = dcomm_handler_ptr%get_comm_vp_mu()
        call mpi_cart_shift(ex_comm, 0, 1, left_rank, right_rank, ierr)
        call mpi_cart_shift(ex_comm, 1, 1, down_rank, up_rank, ierr)

        ! Check inner down and up ghost points
        do l = lb_stripped(DIM_VP), ub_stripped(DIM_VP)
            bounds = lb_stripped

            ! Down
            if(down_rank /= MPI_PROC_NULL) then
                expected = real(down_rank, kind=GP)
                bounds(DIM_MU) = lb(DIM_MU)
                fine = almost_equal(eval_5d_ptr_at(bounds, data_ptr, &
                                                   DIM_VP, l), &
                                    expected, GP_EPS)
                if(.not. fine) res = .false.
            endif

            ! Up
            if(up_rank /= MPI_PROC_NULL) then
                expected = real(up_rank, kind=GP)
                bounds(DIM_MU) = ub(DIM_MU)
                fine = almost_equal(eval_5d_ptr_at(bounds, data_ptr, &
                                                   DIM_VP, l), &
                                    expected, GP_EPS)
                if(.not. fine) res = .false.
            endif
        enddo

        ! Check inner left and right ghost points
        do m = lb_stripped(DIM_MU), ub_stripped(DIM_MU)
            bounds = lb_stripped

            ! Left
            if(left_rank /= MPI_PROC_NULL) then
                expected = real(left_rank, kind=GP)
                do l = 1, 2
                    bounds(DIM_VP) = ghost_bounds_vp(l)
                    fine = almost_equal(eval_5d_ptr_at(bounds, data_ptr, &
                                                       DIM_MU, m), &
                                        expected, GP_EPS)
                    if(.not. fine) res = .false.
                enddo
            endif

            ! Right
            if(right_rank /= MPI_PROC_NULL) then
                expected = real(right_rank, kind=GP)
                do l = 3, 4
                bounds(DIM_VP) = ghost_bounds_vp(l)
                fine = almost_equal(eval_5d_ptr_at(bounds, data_ptr, &
                                                   DIM_MU, m), &
                                    expected, GP_EPS)
                if(.not. fine) res = .false.
                enddo
            endif
        enddo

        ! Check edges

        ! Top right
        ngb_coords = coords
        ngb_coords(DIM_VP) = ngb_coords(DIM_VP) + 1
        ngb_coords(DIM_MU) = ngb_coords(DIM_MU) + 1
        coords_fine = check_coord_bounds(coord_lbounds, coord_ubounds, &
                                         ngb_coords)

        if(coords_fine) then
            call mpi_cart_rank(cart_comm, ngb_coords, ngb_rank, ierr)
            if(ngb_rank /= MPI_PROC_NULL) then
                expected = real(ngb_rank, kind=GP)
                bounds = lb_stripped
                bounds(DIM_MU) = ub(DIM_MU)
                do l = 3, 4
                    fine = almost_equal(eval_5d_ptr_at(bounds, data_ptr, &
                                                       DIM_VP, &
                                                       ghost_bounds_vp(l)), &
                                        expected, GP_EPS)
                    if(.not. fine) res = .false.
                enddo
            endif
        endif

        ! Top left
        ngb_coords = coords
        ngb_coords(DIM_VP) = ngb_coords(DIM_VP) - 1
        ngb_coords(DIM_MU) = ngb_coords(DIM_MU) + 1
        coords_fine = check_coord_bounds(coord_lbounds, coord_ubounds, &
                                         ngb_coords)

        if(coords_fine) then
            call mpi_cart_rank(cart_comm, ngb_coords, ngb_rank, ierr)
            if(ngb_rank /= MPI_PROC_NULL) then
                expected = real(ngb_rank, kind=GP)
                bounds = lb_stripped
                bounds(DIM_MU) = ub(DIM_MU)
                do l = 1, 2
                    fine = almost_equal(eval_5d_ptr_at(bounds, data_ptr, &
                                                       DIM_VP, &
                                                       ghost_bounds_vp(l)), &
                                        expected, GP_EPS)
                    if(.not. fine) res = .false.
                enddo
            endif
        endif

        ! Bottom right
        ngb_coords = coords
        ngb_coords(DIM_VP) = ngb_coords(DIM_VP) + 1
        ngb_coords(DIM_MU) = ngb_coords(DIM_MU) - 1
        coords_fine = check_coord_bounds(coord_lbounds, coord_ubounds, &
                                         ngb_coords)

        if(coords_fine) then
            call mpi_cart_rank(cart_comm, ngb_coords, ngb_rank, ierr)
            if(ngb_rank /= MPI_PROC_NULL) then
                expected = real(ngb_rank, kind=GP)
                bounds = lb_stripped
                bounds(DIM_MU) = lb(DIM_MU)
                do l = 3, 4
                    fine = almost_equal(eval_5d_ptr_at(bounds, data_ptr, &
                                                       DIM_VP, &
                                                       ghost_bounds_vp(l)), &
                                        expected, GP_EPS)
                    if(.not. fine) res = .false.
                enddo
            endif
        endif

        ! Bottom left
        ngb_coords = coords
        ngb_coords(DIM_VP) = ngb_coords(DIM_VP) - 1
        ngb_coords(DIM_MU) = ngb_coords(DIM_MU) - 1
        coords_fine = check_coord_bounds(coord_lbounds, coord_ubounds, &
                                         ngb_coords)

        if(coords_fine) then
            call mpi_cart_rank(cart_comm, ngb_coords, ngb_rank, ierr)
            if(ngb_rank /= MPI_PROC_NULL) then
                expected = real(ngb_rank, kind=GP)
                bounds = lb_stripped
                bounds(DIM_MU) = lb(DIM_MU)
                do l = 1, 2
                    fine = almost_equal(eval_5d_ptr_at(bounds, data_ptr, &
                                                       DIM_VP, &
                                                       ghost_bounds_vp(l)), &
                                        expected, GP_EPS)
                    if(.not. fine) res = .false.
                enddo
            endif
        endif

        deallocate(ds)
    end function

    function test_exchange_4d(test, &
                              n_ranks_phi, &
                              n_ranks_vp, &
                              n_ranks_mu, &
                              n_ranks_sp, &
                              n_mom, &
                              n_spec, &
                              n_points, &
                              dim_permut) result(res)
        class (MpiTestMethod), intent(inout) :: test
        !! Test instance
        integer, intent(in) :: n_ranks_phi
        !! number of ranks in phi direction
        integer, intent(in) :: n_ranks_vp
        !! number of ranks in vp direction
        integer, intent(in) :: n_ranks_mu
        !! number of ranks in mu direction
        integer, intent(in) :: n_ranks_sp
        !! number of ranks in sp direction
        integer, intent(in) :: n_mom
        !! number of neutrals moments
        integer, intent(in) :: n_spec
        !! number of neutrals species
        integer, intent(in) :: n_points
        !! scalar number of points
        integer, dimension(4), intent(in) :: dim_permut

        integer :: comm_world, rank, n_ghosts
        type(dcomm_handler_t), target :: dcomm_handler
        type(dcomm_handler_t), pointer :: dcomm_handler_ptr
        class(data_storage_4d_t), allocatable :: ds
        real(kind=GP), dimension(:,:,:,:), pointer :: data_ptr
        real(kind=GP) :: eval, expected
        integer, parameter :: dimensions = 4
        integer :: number_of_points(dimensions)
        integer :: lb(dimensions), lb_stripped(dimensions)
        integer :: ub(dimensions), ub_stripped(dimensions)
        integer, dimension(5) :: dim_permut_local
        integer :: ex_comm, ex_dim, i
        logical :: fine, res

        rank = test%getProcessRank()
        comm_world = test%getMpiCommunicator()
        n_ghosts = 2

        res = .true.
        ! We only exchange over phi
        ex_dim = dim_permut(2)
        dim_permut_local(1:2) = dim_permut(1:2)
        dim_permut_local(3:5) = [3, 4, 5]

        number_of_points = [n_points, n_points, n_points, n_points]
        call setup_test_mesh(comm_world, &
                             rank, &
                             equilibrium_type=SLAB, &
                             spacing_RZ=64.0_GP&
                                       /number_of_points(dim_permut(1)), &
                             n_levels=1, &
                             n_points_phi=number_of_points(dim_permut(2)))

        call dcomm_handler%initialize(comm_world, n_ranks_phi, &
                                      n_ranks_vp, n_ranks_mu, n_ranks_sp, &
                                      dim_permut_local)
        dcomm_handler_ptr => dcomm_handler
        call dcomm_handler_ptr%initialize_RZ_domain(&
                                            number_of_points(dim_permut(1)))

        ex_comm = dcomm_handler%get_comm_phi()

        if(get_use_gpu_offload()) then
#ifdef ENABLE_GPU
            allocate(data_storage_gpu_4d_t :: ds)
#endif
        else
            allocate(data_storage_cpu_4d_t :: ds)
        endif

        call ds%initialize(dcomm_handler_ptr, n_mom, n_spec, &
                           init_value=1.0_GP * rank)

        data_ptr => ds%get_pointer()
        lb = ds%lbound()
        ub = ds%ubound()
        lb_stripped = ds%lbound_stripped()
        ub_stripped = ds%ubound_stripped()

        call ds%start_exchange()
        call ds%finish_exchange()
        call ds%update_host()

        if(n_points / n_ranks_phi < n_ghosts) then
            ! n_ghosts == 2, n_points / n_ranks_phi ==1
            ! left neighbors
            expected = real(get_sec_left_rank(ex_comm), kind=GP)
            eval = eval_4d_ptr_at(lb_stripped, data_ptr, ex_dim, &
                                  lb_stripped(ex_dim) - 2)
            fine =  almost_equal(eval, expected, GP_EPS)
            if(.not. fine) res = .false.
            expected = real(get_left_rank(ex_comm), kind=GP)
            eval = eval_4d_ptr_at(lb_stripped, data_ptr, ex_dim, &
                                  lb_stripped(ex_dim) - 1)
            fine =  almost_equal(eval, expected, GP_EPS)
            if(.not. fine) res = .false.
            ! right neighbors
            expected = real(get_right_rank(ex_comm), kind=GP)
            eval = eval_4d_ptr_at(lb_stripped, data_ptr, ex_dim, &
                                  ub_stripped(ex_dim) + 1)
            fine =  almost_equal(eval, expected, GP_EPS)
            if(.not. fine) res = .false.
            expected = real(get_sec_right_rank(ex_comm), kind=GP)
            eval = eval_4d_ptr_at(lb_stripped, data_ptr, ex_dim, &
                                  ub_stripped(ex_dim) + 2)
            fine =  almost_equal(eval, expected, GP_EPS)
            if(.not. fine) res = .false.
        else
            ! left neighbors
            do i = lb(ex_dim), lb_stripped(ex_dim) - 1
                expected = real(get_left_rank(ex_comm), kind=GP)
                eval = eval_4d_ptr_at(lb_stripped, data_ptr, ex_dim, i)
                fine =  almost_equal(eval, expected, GP_EPS)
                if(.not. fine) res = .false.
            enddo
            ! right neighbors
            do i = ub_stripped(ex_dim) + 1, ub(ex_dim)
                expected = real(get_right_rank(ex_comm), kind=GP)
                eval = eval_4d_ptr_at(lb_stripped, data_ptr, ex_dim, i)
                fine =  almost_equal(eval, expected, GP_EPS)
                if(.not. fine) res = .false.
            enddo
        endif

        deallocate(ds)
    end function test_exchange_4d

    function test_exchange_2d(test, &
                              n_ranks_phi, &
                              n_ranks_vp, &
                              n_ranks_mu, &
                              n_ranks_sp, &
                              n_points, &
                              dim_permut) result(res)
        class(MpiTestMethod), intent(inout) :: test
        !! Test instance
        integer, intent(in) :: n_ranks_phi
        !! number of ranks in phi direction
        integer, intent(in) :: n_ranks_vp
        !! number of ranks in vp direction
        integer, intent(in) :: n_ranks_mu
        !! number of ranks in mu direction
        integer, intent(in) :: n_ranks_sp
        !! number of ranks in sp direction
        integer, intent(in) :: n_points
        !! scalar number of points
        integer, dimension(2), intent(in) :: dim_permut

        integer :: comm_world, rank, n_ghosts
        type(dcomm_handler_t), target :: dcomm_handler
        type(dcomm_handler_t), pointer :: dcomm_handler_ptr
        class(data_storage_2d_t), allocatable :: ds
        real(kind=GP), dimension(:,:), pointer :: data_ptr
        real(kind=GP) :: eval, expected
        integer, parameter :: dimensions = 2
        integer :: number_of_points(dimensions)
        integer :: lb(dimensions), lb_stripped(dimensions)
        integer :: ub(dimensions), ub_stripped(dimensions)
        integer :: ex_comm, ex_dim
        logical :: fine, res
        integer :: i
        integer, dimension(5) :: dim_permut_local

        rank = test%getProcessRank()
        comm_world = test%getMpiCommunicator()
        n_ghosts = 2

        res = .true.
        ! We only exchange over phi
        ex_dim = dim_permut(2)
        dim_permut_local(1:2) = dim_permut(1:2)
        dim_permut_local(3:5) = [3, 4, 5]

        number_of_points = [n_points, n_points]
        call setup_test_mesh(comm_world, &
                             rank, &
                             equilibrium_type=SLAB, &
                             spacing_RZ=64.0_GP&
                                       /number_of_points(dim_permut(1)), &
                             n_levels=1, &
                             n_points_phi=number_of_points(dim_permut(2)))

        call dcomm_handler%initialize(comm_world, n_ranks_phi, &
                                      n_ranks_vp, n_ranks_mu, n_ranks_sp, &
                                      dim_permut_local)
        dcomm_handler_ptr => dcomm_handler
        call dcomm_handler_ptr%initialize_RZ_domain(&
                                            number_of_points(dim_permut(1)))

        ex_comm = dcomm_handler%get_comm_phi()

        if(get_use_gpu_offload()) then
#ifdef ENABLE_GPU
            allocate(data_storage_gpu_2d_t :: ds)
#endif
        else
            allocate(data_storage_cpu_2d_t :: ds)
        endif

        call ds%initialize(dcomm_handler_ptr, init_value=1.0_GP * rank)

        data_ptr => ds%get_pointer()
        lb = ds%lbound()
        ub = ds%ubound()
        lb_stripped = ds%lbound_stripped()
        ub_stripped = ds%ubound_stripped()

        call ds%start_exchange()
        call ds%finish_exchange()
        call ds%update_host()

        if(n_points / n_ranks_phi < n_ghosts) then
            ! n_ghosts == 2, n_points / n_ranks_phi ==1
            ! left neighbors
            expected = real(get_sec_left_rank(ex_comm), kind=GP)
            eval = eval_2d_ptr_at(lb_stripped, data_ptr, ex_dim, &
                                  lb_stripped(ex_dim) - 2)
            fine =  almost_equal(eval, expected, GP_EPS)
            if(.not. fine) res = .false.
            expected = real(get_left_rank(ex_comm), kind=GP)
            eval = eval_2d_ptr_at(lb_stripped, data_ptr, ex_dim, &
                                  lb_stripped(ex_dim) - 1)
            fine =  almost_equal(eval, expected, GP_EPS)
            if(.not. fine) res = .false.
            ! right neighbors
            expected = real(get_right_rank(ex_comm), kind=GP)
            eval = eval_2d_ptr_at(lb_stripped, data_ptr, ex_dim, &
                                  ub_stripped(ex_dim) + 1)
            fine =  almost_equal(eval, expected, GP_EPS)
            if(.not. fine) res = .false.
            expected = real(get_sec_right_rank(ex_comm), kind=GP)
            eval = eval_2d_ptr_at(lb_stripped, data_ptr, ex_dim, &
                                  ub_stripped(ex_dim) + 2)
            fine =  almost_equal(eval, expected, GP_EPS)
            if(.not. fine) res = .false.
        else
            ! left neighbors
            do i = lb(ex_dim), lb_stripped(ex_dim) - 1
                expected = real(get_left_rank(ex_comm), kind=GP)
                eval = eval_2d_ptr_at(lb_stripped, data_ptr, ex_dim, i)
                fine =  almost_equal(eval, expected, GP_EPS)
                if(.not. fine) res = .false.
            enddo
            ! right neighbors
            do i = ub_stripped(ex_dim) + 1, ub(ex_dim)
                expected = real(get_right_rank(ex_comm), kind=GP)
                eval = eval_2d_ptr_at(lb_stripped, data_ptr, ex_dim, i)
                fine =  almost_equal(eval, expected, GP_EPS)
                if(.not. fine) res = .false.
            enddo
        endif

        deallocate(ds)
    end function test_exchange_2d

end module helpers_data_storage_m
