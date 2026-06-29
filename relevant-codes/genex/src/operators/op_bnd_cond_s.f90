submodule(op_bnd_cond_m) op_bnd_cond_s
    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_ERR_OPERATORS
    use mesh_5d_m, only: BUF_ZONE_BOUNDARY_IN, FILLER_VALUE_INT
    implicit none

contains

    module subroutine initialize_neum_parent(this, dcomm_handler, mesh)
        class(op_bnd_cond_base_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        integer :: i, k, j1, j2, size_RZ_ghost
        integer, dimension(5) :: lb, ub, lb_stripped, ub_stripped
        real(kind=GP), pointer, dimension(:,:) :: is_compute, is_core, rho
        integer, contiguous, pointer, dimension(:,:) :: buf_zone
        integer, allocatable, dimension(:,:) :: ind_cmcore_buf, ind_ghcore_buf
        integer, allocatable, dimension(:) :: num_cmcore_buf, num_ghcore_buf
        real(kind=GP) :: rho_center
        type(op_set_uniform_cpu_t) :: op_set_uniform

        select type(this)
            type is(op_bnd_cond_neum_cpu_t)
            type is(op_bnd_cond_neum_vspec_cpu_t)
            class default
                call handle_error("initialize_neum_parent() should only be &
                                  &called from a Neumman boundary condition &
                                  &operator!", &
                                  GENEX_ERR_OPERATORS,  __LINE__, __FILE__)
        end select

        lb          = dcomm_handler%get_lbound()
        ub          = dcomm_handler%get_ubound()
        lb_stripped = dcomm_handler%get_lbound_stripped()
        ub_stripped = dcomm_handler%get_ubound_stripped()

        is_compute => this%mesh%get_is_compute_pointer()
        is_core    => this%mesh%get_is_core_pointer()
        buf_zone   => this%mesh%get_buf_zone_pointer()
        rho        => this%mesh%get_rho_pointer()
        rho_center = (this%mesh%rho_max() + this%mesh%rho_min()) / 2.0_GP

        call op_set_uniform%initialize()

        size_RZ_ghost = 0
        do k = lb_stripped(2), ub_stripped(2)
            if((mesh%size_RZ() - mesh%size_RZ_inner(k)) > size_RZ_ghost) then
                size_RZ_ghost = mesh%size_RZ() - mesh%size_RZ_inner(k)
            end if
        end do

        allocate(this%number_bnd_points(lb_stripped(2):ub_stripped(2)))
        allocate(num_cmcore_buf(lb_stripped(2):ub_stripped(2)))
        allocate(num_ghcore_buf, mold=num_cmcore_buf)
        allocate(ind_cmcore_buf(size_RZ_ghost, lb_stripped(2):ub_stripped(2)))
        allocate(ind_ghcore_buf, mold=ind_cmcore_buf)
        call op_set_uniform%apply(num_cmcore_buf, 0)
        call op_set_uniform%apply(num_ghcore_buf, 0)

        do k = lb_stripped(2), ub_stripped(2)
            j1 = 1
            j2 = 1
            do i = lb(1), ub(1)
                ! Count the number of compute points within the inner boundary
                ! buffer zone per poloidal plane and store the RZ indices
                if(is_compute(i, k) == 1.0_GP &
                   .and. buf_zone(i, k) == BUF_ZONE_BOUNDARY_IN) then
                    num_cmcore_buf(k) = num_cmcore_buf(k) + 1
                    ind_cmcore_buf(j1, k) = i
                    j1 = j1 + 1
                end if

                ! Count the number of ghost points in the core per poloidal
                ! plane and store the RZ indices
                if(is_compute(i, k) == 0.0_GP .and. is_core(i, k) == 1.0_GP &
                   .and. rho(i, k) <= rho_center) then
                    num_ghcore_buf(k) = num_ghcore_buf(k) + 1
                    ind_ghcore_buf(j2, k) = i
                    j2 = j2 + 1
                end if
            end do
        end do

        this%max_num_compute_buf_core = maxval(num_cmcore_buf)
        this%max_num_ghost_core       = maxval(num_ghcore_buf)
        this%number_bnd_points        = real(num_cmcore_buf, kind=GP)

        allocate(this%ind_compute_buf_core(1:this%max_num_compute_buf_core, &
                                           lb_stripped(2):ub_stripped(2)))
        allocate(this%ind_ghost_core(1:this%max_num_ghost_core, &
                                     lb_stripped(2):ub_stripped(2)))

        ! Initialize RZ indices with filler points
        call op_set_uniform%apply(this%ind_compute_buf_core, FILLER_VALUE_INT)
        call op_set_uniform%apply(this%ind_ghost_core, FILLER_VALUE_INT)

        ! Copy the indices of the boundary and ghost points from buffer arrays
        do k = lb_stripped(2), ub_stripped(2)
            do i = 1, num_cmcore_buf(k)
                this%ind_compute_buf_core(i, k) = ind_cmcore_buf(i, k)
            end do

            do i = 1, num_ghcore_buf(k)
                this%ind_ghost_core(i, k) = ind_ghcore_buf(i, k)
            end do
        end do

        deallocate(ind_ghcore_buf, ind_cmcore_buf)
        deallocate(num_ghcore_buf, num_cmcore_buf)
    end subroutine

end submodule
