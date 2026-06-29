submodule(op_bnd_cond_m) op_bnd_cond_neum_cpu_s
    use mesh_5d_m, only: FILLER_VALUE_INT
    implicit none

contains

    module subroutine initialize_neum_cpu(this, dcomm_handler, mesh)
        class(op_bnd_cond_neum_cpu_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        this%mesh    => mesh
        this%vp      => this%mesh%get_vp_pointer()
        this%mu      => this%mesh%get_mu_pointer()

        call this%op_set_uniform%initialize()

        call this%initialize_neum_parent(dcomm_handler, mesh)
    end subroutine

    module subroutine apply_neum_cpu(this, da_f_inout, da_co_qn_eq, &
                                     da_b_qn_eq, da_b_amps_law, &
                                     da_b_ohms_law, t)
        class(op_bnd_cond_neum_cpu_t), intent(inout) :: this
        class(data_array_5d_t), intent(inout) :: da_f_inout
        class(data_array_2d_t), intent(inout) :: da_co_qn_eq
        class(data_array_2d_t), intent(inout) :: da_b_qn_eq
        class(data_array_2d_t), intent(inout) :: da_b_amps_law
        class(data_array_2d_t), intent(inout) :: da_b_ohms_law
        real(kind=GP), intent(in) :: t

        integer :: i, j, k, l, m, n
        integer, dimension(5) :: lb, ub, lb_stripped, ub_stripped
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:,:) :: f_inout
        real(kind=GP), contiguous, pointer, dimension(:,:) :: b_qn_eq, &
                                                              b_amps_law, &
                                                              b_ohms_law
        real(kind=GP), pointer, dimension(:,:) :: is_compute
        real(kind=GP), allocatable, dimension(:,:,:,:) :: buf_bnd

        this%n_iterations = da_f_inout%get_size_stripped()
        call this%perf_counter%start_measurement()

        lb = da_f_inout%get_lbound()
        ub = da_f_inout%get_ubound()
        lb_stripped = da_f_inout%get_lbound_stripped()
        ub_stripped = da_f_inout%get_ubound_stripped()

        f_inout    => da_f_inout%get_pointer()
        b_qn_eq    => da_b_qn_eq%get_pointer()
        b_amps_law => da_b_amps_law%get_pointer()
        b_ohms_law => da_b_ohms_law%get_pointer()
        is_compute => this%mesh%get_is_compute_pointer()

        ! NOTE: To adapt the boundary points, we first compute the average
        !       value of the distribution functions within the inner
        !       boundary buffer region. This average is then imposed on the
        !       inner ghost points. The outer boundary is not updated, as it
        !       remains fixed by the initial conditions. Only the inner
        !       boundary is modified.

        ! Set boundary conditions on the distribution function.

        allocate(buf_bnd(lb_stripped(2):ub_stripped(2), &
                         lb_stripped(3):ub_stripped(3), &
                         lb_stripped(4):ub_stripped(4), &
                         lb_stripped(5):ub_stripped(5)))

        call this%op_set_uniform%apply(buf_bnd, 0.0_GP)

        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped) &
        !$omp shared(this, f_inout) &
        !$omp private(i, j, k, l, m, n) &
        !$omp reduction(+:buf_bnd)
        do n = lb_stripped(5), ub_stripped(5)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do j = 1, this%max_num_compute_buf_core
            i = this%ind_compute_buf_core(j, k)
            if(i == FILLER_VALUE_INT) cycle
            buf_bnd(k, l, m, n) = buf_bnd(k, l, m, n) + f_inout(i, k, l, m, n)
        end do
        !$omp end do
        end do
        end do
        end do
        end do
        !$omp end parallel

        !$omp parallel default(none) &
        !$omp firstprivate(lb, ub, lb_stripped, ub_stripped) &
        !$omp shared(this, f_inout, buf_bnd) &
        !$omp private(i, j, k, l, m, n)
        do n = lb_stripped(5), ub_stripped(5)
        do m = lb_stripped(4), ub_stripped(4)
        do l = lb_stripped(3), ub_stripped(3)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do j = 1, this%max_num_ghost_core
            i = this%ind_ghost_core(j, k)
            if(i == FILLER_VALUE_INT) cycle
            f_inout(i, k, l, m, n) = buf_bnd(k, l, m, n) &
                                   / this%number_bnd_points(k)
        end do
        !$omp end do
        end do
        end do
        end do
        end do
        !$omp end parallel

        ! Set boundary conditions on the electromagnetic fields

        ! The compute mask is used to set the boundary condition since its
        ! value is zero for ghosts, fillers, and points in the target, and one
        ! otherwise

        !$omp parallel default(none) &
        !$omp firstprivate(lb, ub, lb_stripped, ub_stripped) &
        !$omp shared(b_qn_eq, b_amps_law, b_ohms_law, is_compute) &
        !$omp private(i, k)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb(1), ub(1)
            b_qn_eq(i, k)    = b_qn_eq(i, k)    * is_compute(i, k)
            b_amps_law(i, k) = b_amps_law(i, k) * is_compute(i, k)
            b_ohms_law(i, k) = b_ohms_law(i, k) * is_compute(i, k)
        end do
        !$omp end do nowait
        end do
        !$omp end parallel

        call this%perf_counter%end_measurement()

        if(allocated(buf_bnd)) deallocate(buf_bnd)
    end subroutine

end submodule
