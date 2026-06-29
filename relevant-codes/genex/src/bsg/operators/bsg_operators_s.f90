submodule(bsg_operators_m) bsg_operators_s

    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_ERR_BSG
    use bsg_types_m, only: MAX_BLOCKS
    use bsg_interpolation_m, only: equidistant_interpolation_t
    use logger_m, only: logger_get_debug_channel

    implicit none

contains

    module subroutine initialize(this, mesh, num_blocks, interp_order)
        class(bsg_operators_t), intent(inout) :: this
        type(mesh_5d_t), target, intent(in) :: mesh
        integer, intent(in) :: num_blocks
        integer, optional, intent(in) :: interp_order

        integer :: nb, l, ngb
        real(kind=GP), dimension(:), allocatable :: vp_min, vp_max
        integer, dimension(2) :: nb_neighbor
        integer :: neighbor_block_idx, start_idx, end_idx
        real(kind=GP) :: value_at_l
        integer :: external_points

        this%mesh => mesh
        this%vp_size = this%mesh%size_vp()
        this%bsg_flags => this%mesh%get_bsg_flags_pointer()

        ! Set vp and vpw arrays
        allocate(this%vp_bsg_ptr(num_blocks))
        allocate(this%vpw_bsg_ptr(num_blocks))
        do nb = 1, num_blocks
            call this%initialize_vp_pointers(nb, &
                                       this%mesh%get_vp_pointer(nb), &
                                       this%vp_bsg_ptr)
            call this%initialize_vp_pointers(nb, &
                                       this%mesh%get_vpw_pointer(nb), &
                                       this%vpw_bsg_ptr)
        end do

        ! Evaluate delta and 1/delta
        allocate(this%vp_delta(num_blocks))
        allocate(this%vp_delta_inverse(num_blocks))
        do nb = 1, num_blocks
            this%vp_delta(nb) = this%mesh%delta_vp(nb)
            this%vp_delta_inverse(nb) = 1.0_GP / this%mesh%delta_vp(nb)
        end do

        this%initialized = .true.
        if (num_blocks == 1) then
            if (.not. present(interp_order)) return

            call handle_error("Interpolation order parameter cannot be &
                              &present if num_blocks=1", GENEX_ERR_BSG, &
                              __LINE__, __FILE__)
        elseif (.not. present(interp_order)) then
            call handle_error("Interpolation order parameter must be &
                              &present if num_blocks>1", GENEX_ERR_BSG, &
                              __LINE__, __FILE__)
        end if

        if (num_blocks < 2 .or. num_blocks > MAX_BLOCKS) then
            call handle_error("Invalid num_blocks specified", GENEX_ERR_BSG, &
                              __LINE__, __FILE__)
        end if

        if (interp_order <= 2 .or. interp_order > 5) then
            call handle_error("Invalid interp_order specified", GENEX_ERR_BSG, &
                              __LINE__, __FILE__)
        end if

        if (this%vp_size < interp_order + 1) then
            call handle_error("Size of vp grid too small for interpolation &
                              &order", GENEX_ERR_BSG, &
                              __LINE__, __FILE__)
        end if

        allocate(equidistant_interpolation_t :: this%interpolator)

        call this%interpolator%initialize(interp_order)

        ! Set vp min and max
        allocate(vp_min(num_blocks))
        allocate(vp_max(num_blocks))
        do nb = 1, num_blocks
            vp_min(nb) = minval(this%vp_bsg_ptr(nb)%array(:))
            vp_max(nb) = maxval(this%vp_bsg_ptr(nb)%array(:))
        end do

        ! Allocate helper arrays
        allocate(this%local_start_idx(this%vp_size, num_blocks, &
                                      num_blocks))
        allocate(this%local_end_idx(this%vp_size, num_blocks, &
                                    num_blocks))
        allocate(this%interpolation_coeff(this%vp_size, num_blocks, &
                                          num_blocks, interp_order+1))
        allocate(this%is_internal(this%vp_size, num_blocks, &
                                  num_blocks))

        this%is_internal = .false.
        this%local_start_idx = -huge(1)
        this%local_end_idx   = -huge(1)
        this%interpolation_coeff = 0.0_GP
        do nb = 1, num_blocks
            do l = 1, this%vp_size
                value_at_l = this%vp_bsg_ptr(nb)%array(l)

                nb_neighbor(1) = nb - 1
                nb_neighbor(2) = nb + 1
                ! Loop over neighbor blocks
                do ngb = 1, 2
                    neighbor_block_idx = nb_neighbor(ngb)
                    if (neighbor_block_idx >= 1 .and. &
                        neighbor_block_idx <= num_blocks) then

                        call this%interpolator%eval_local_index( &
                            value_at_l, &
                            this%vp_bsg_ptr(neighbor_block_idx)%array(:), &
                            start_idx, end_idx)

                        this%local_start_idx(l, nb, neighbor_block_idx) = &
                                                                       start_idx
                        this%local_end_idx(l, nb, neighbor_block_idx) = &
                                                                       end_idx

                        if (value_at_l <= vp_max(neighbor_block_idx) .and. &
                            value_at_l >= vp_min(neighbor_block_idx)) then
                                this%is_internal(l, nb, &
                                                 neighbor_block_idx) = .true.
                        end if

                        if (this%is_internal(l, nb, neighbor_block_idx)) then
                            call this%interpolator%eval_coeff( &
                                 this%vp_bsg_ptr(neighbor_block_idx)%&
                                                     array(start_idx:end_idx), &
                                 this%interpolation_coeff(l, nb, &
                                                          neighbor_block_idx, &
                                                          :))
                        end if
                    end if
                end do
            end do
            if (nb /= num_blocks) then
                external_points = count(.not. this%is_internal(:, nb, nb+1))
                write(logger_get_debug_channel(), *) "Block ", nb, " has ", &
                      external_points, " / ", this%vp_size, " points as &
                      &external points"
            end if
        end do
    end subroutine

    module subroutine initialize_vp_pointers(this, block_number, &
                                           ptr_from_mesh, ptr_accessor)
        class(bsg_operators_t), intent(inout) :: this
        integer, intent(in) :: block_number
        real(kind=GP), dimension(:), target, intent(in) :: ptr_from_mesh
        type(vspace_bsg_t), dimension(:), pointer, intent(inout) :: ptr_accessor

        ptr_accessor(block_number)%array(1:this%vp_size) => ptr_from_mesh
    end subroutine

end submodule
