submodule(block_structured_grids_m) block_structured_grids_s

    use genex_status_codes_m, only: GENEX_ERR_BSG
    use genex_error_handling_m, only: handle_error, error_info_t
    use bsg_types_m, only: MAX_BLOCKS, unpack_bflag, pack_flags

    implicit none

contains

    module subroutine initialize(this, num_blocks, radial_markers)
        class(block_structured_grids_t), intent(inout) :: this
        integer, intent(in) :: num_blocks
        real(kind=GP), dimension(:), optional, intent(in) :: radial_markers

        integer :: nb

        this%num_blocks = num_blocks
        this%initialized = .true.
        if (num_blocks == 1) return
        if (this%num_blocks > MAX_BLOCKS .or. num_blocks < 2) then
            call handle_error("Invalid number of blocks", GENEX_ERR_BSG, &
                              __LINE__, __FILE__, &
                              additional_info=error_info_t("Block number:", &
                              [num_blocks]))
        end if
        allocate(this%radial_markers(this%num_blocks-1))
        if (all(radial_markers == 0.0_GP)) then
            this%radial_markers = 0.0_GP
        else
            do nb = 1, this%num_blocks - 1
                this%radial_markers(nb) = radial_markers(nb)
            end do
        end if
    end subroutine

    module subroutine set_flags(this, map_p, map_pp, map_m, map_mm, &
                                neighbor_indices)
        class(block_structured_grids_t), intent(inout) :: this
        type(csrmat_t), intent(in) :: map_p, map_pp, map_m, map_mm
        integer, dimension(:,:,:), intent(in) :: neighbor_indices

        integer :: i, j, size_RZ
        integer :: ip, im, ngb
        integer :: flag_p, flag_pp, flag_m, flag_mm
        integer :: bflag_p
        ! bflag of the RZ point
        integer, allocatable, dimension(:) :: bflag
        integer :: indices_size
        logical :: blocks_are_invalid = .false.

        if (this%num_blocks == 1) return

        if (.not. this%initialized) then
            call handle_error("BSG type not initialized", GENEX_ERR_BSG, &
                              __LINE__, __FILE__)
        end if

        ! Unpack bflag
        size_RZ = size(this%bsg_flags)
        allocate(bflag(size_RZ))
        do i = 1, size_RZ
            bflag(i) = unpack_bflag(this%bsg_flags(i))
        end do

        ! Mark flags for map matrices
        do i = 1, size_RZ
            bflag_p = bflag(i)

            ! Check map_p
            flag_p = this%set_map_matrix_flag( &
                            bflag_p, &
                            bflag(map_p%j(map_p%i(i):map_p%i(i + 1) - 1)))

            ! Check map_pp
            flag_pp = this%set_map_matrix_flag( &
                            bflag_p, &
                            bflag(map_pp%j(map_pp%i(i):map_pp%i(i + 1) - 1)))

            ! Check map_m
            flag_m = this%set_map_matrix_flag( &
                            bflag_p, &
                            bflag(map_m%j(map_m%i(i):map_m%i(i + 1) - 1)))

            ! Check map_mm
            flag_mm = this%set_map_matrix_flag( &
                            bflag_p, &
                            bflag(map_mm%j(map_mm%i(i):map_mm%i(i + 1) - 1)))

            ! Check if at least one of the neighbors belongs to a
            ! different block
            outer: do ip = lbound(neighbor_indices, 1), &
                           ubound(neighbor_indices, 1)
                do im = lbound(neighbor_indices, 2), &
                        ubound(neighbor_indices, 2)
                    ngb = neighbor_indices(ip, im, i)
                    if (bflag(ngb) /= bflag_p) then
                        ! Check that neighboring point does not belong to a
                        ! non-neighboring block. In that case number of
                        ! blocks must be reduced.
                        if (bflag_p == 1) then
                            if (bflag(ngb) /= 2) then
                                blocks_are_invalid = .true.
                            end if
                        else if (bflag_p == this%num_blocks) then
                            if (bflag(ngb) /= this%num_blocks - 1) then
                                blocks_are_invalid = .true.
                            end if
                        else
                            if (bflag(ngb) /= bflag_p + 1 .and. &
                                bflag(ngb) /= bflag_p - 1) then
                                blocks_are_invalid = .true.
                            end if
                        end if
                        if (blocks_are_invalid) then
                            call handle_error("Neighboring block condition &
                                              &not met!", GENEX_ERR_BSG, &
                                              __LINE__, __FILE__)
                        end if
                        bflag_p = -bflag_p
                        exit outer
                    end if
                end do
            end do outer

            if (bflag_p < 0) then
                call pack_flags(abs(bflag_p), flag_p, &
                                flag_pp, flag_m, flag_mm, this%bsg_flags(i))
                this%bsg_flags(i) = -this%bsg_flags(i)
            else
                call pack_flags(abs(bflag_p), flag_p, &
                                flag_pp, flag_m, flag_mm, this%bsg_flags(i))
            end if
        end do
        deallocate(bflag)
    end subroutine

    integer module function set_map_matrix_flag(this, bflag_rz, bflag_array)
        class(block_structured_grids_t), intent(inout) :: this
        integer, intent(in) :: bflag_rz
        integer, dimension(:), intent(in) :: bflag_array

        set_map_matrix_flag = merge(1, 0, any(bflag_array /= bflag_rz))
    end function

    module subroutine init_vp_grid(this, vp_size, vp_length_total, &
                                   vplen_markers)
        class(block_structured_grids_t), target, intent(inout) :: this
        integer, intent(in) :: vp_size
        real(kind=GP), intent(in) :: vp_length_total
        real(kind=GP), dimension(:), intent(in) :: vplen_markers

        integer :: nb

        if (.not. this%initialized) then
            call handle_error("BSG type not initialized", GENEX_ERR_BSG, &
                              __LINE__, __FILE__)
        end if

        this%vp_size = vp_size
        this%vp_length_total = vp_length_total

        ! Allocate vp_length_blocks
        allocate(this%vp_length_blocks(this%num_blocks))

        ! Set vp_length_blocks
        if (all(vplen_markers == 0.0_GP)) then
            if (all(this%radial_markers == 0.0_GP)) then
                ! Set same vp_length value to all blocks (Testing only!)
                this%vp_length_blocks = vp_length_total
            else
                call handle_error("BSG vplen_markers must be non-zero!", &
                                  GENEX_ERR_BSG, &
                                  __LINE__, __FILE__)
            end if
        else
            ! Set vp_length from each block from the paramater file
            do nb = 1, this%num_blocks
                if (vplen_markers(nb) > vp_length_total .or. &
                    vplen_markers(nb) <= 0.0_GP) then
                    call handle_error("Value of lv_marker out of range", &
                                      GENEX_ERR_BSG, &
                                      __LINE__, __FILE__)
                end if
                this%vp_length_blocks(nb) = vplen_markers(nb)
            end do
        end if
    end subroutine

end submodule
