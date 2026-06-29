submodule(block_structured_grids_m) bsg_construct_s

    use logger_m, only: logger_get_debug_channel
    use genex_status_codes_m, only: GENEX_ERR_BSG
    use genex_error_handling_m, only: handle_error
    use bsg_types_m, only: pack_flags

    implicit none

contains

    module subroutine construct(this, mesh_r, mesh_z, is_core, &
                                equilibrium_instance)
        class(block_structured_grids_t), intent(inout) :: this
        real(kind=GP), dimension(:), target, intent(in) :: mesh_r, mesh_z
        real(kind=GP), dimension(:), optional, intent(in) :: is_core
        class(equilibrium_t), optional, intent(in) :: equilibrium_instance

        integer :: i, nb, rz_points
        real(kind=GP) :: rho_val
        integer, allocatable, dimension(:) ::bflag
        integer, allocatable, dimension(:) :: flag_count
        real(kind=GP) :: rho_diff, rho_min, rho_max

        if (.not. this%initialized) then
            call handle_error("BSG type not initialized", GENEX_ERR_BSG, &
                              __LINE__, __FILE__)
        end if

        rz_points = size(mesh_r)

        ! Allocate bsg_flags
        allocate(this%bsg_flags(size(mesh_r)))
        if (this%num_blocks == 1 .or. .not. present(equilibrium_instance) &
            .or. .not. present(is_core)) then
            do i = 1, rz_points
                call pack_flags(1, 0, 0, 0, 0, this%bsg_flags(i))
            end do
            return
        end if

        if (present(is_core) .neqv. present(equilibrium_instance)) then
            call handle_error("Both the optional args must be present", &
                              GENEX_ERR_BSG, &
                              __LINE__, __FILE__)
        end if

        ! Allocate flag count
        allocate(flag_count(this%num_blocks))
        allocate(bflag(size(mesh_r)))

        ! Generate test radial markers if not available from parameter file
        ! This approach is used for testing only!
        if (all(this%radial_markers == 0.0_GP)) then
            rho_min = equilibrium_instance%rhomin
            rho_max = equilibrium_instance%rhomax
            rho_diff = rho_max - rho_min
            do nb = 1, this%num_blocks - 1
                this%radial_markers(nb) = rho_min &
                                        + nb * (rho_diff &
                                        / real(this%num_blocks, kind=GP))
            end do
        end if

        ! Default bflag
        bflag = 0
        flag_count = 0
        do i = 1, rz_points
            rho_val = equilibrium_instance%rho(mesh_r(i), mesh_z(i), 0.0_GP)

            if (is_core(i) == 1.0_GP) then
                ! Determine the correct bflag based on rho_val value
                ! For first block, rho_val should be less than the radial
                ! marker in that block
                if (rho_val < this%radial_markers(1)) then
                    bflag(i) = 1
                    flag_count(1) = flag_count(1) + 1
                end if
                ! If last block, then rho_val should be greater than
                ! this%radial_markers(nb-1)
                if (this%radial_markers(this%num_blocks - 1) <= rho_val) then
                    bflag(i) = this%num_blocks
                    flag_count(this%num_blocks) = &
                                                 flag_count(this%num_blocks) + 1
                end if
                ! For all other cases except the last block
                inner_blocks: do nb = 2, (this%num_blocks - 1)
                    if (this%radial_markers(nb-1) <= rho_val .and. rho_val < &
                        this%radial_markers(nb)) then
                        bflag(i) = nb
                        flag_count(nb) = flag_count(nb) + 1
                        exit inner_blocks
                    end if
                end do inner_blocks
            else
                ! Outside of the core, we set the block number to the maximum
                bflag(i) = this%num_blocks
                flag_count(this%num_blocks) = flag_count(this%num_blocks) + 1
            end if

            if (bflag(i) <= 0) then
                call handle_error("rho_val value not in range of the blocks", &
                                  GENEX_ERR_BSG, &
                                  __LINE__, __FILE__)
            else if (bflag(i) > this%num_blocks) then
                call handle_error("bflag value greater than num_blocks", &
                                  GENEX_ERR_BSG, &
                                  __LINE__, __FILE__)
            end if
            ! Temporarily pack bflag
            call pack_flags(bflag(i), 0, 0, 0, 0, this%bsg_flags(i))
        end do

        write(logger_get_debug_channel(), *) "BSG Debug info:"

        do nb = 1, this%num_blocks
            if (nb == 1) then
                write(logger_get_debug_channel(), *) "Block ", nb, " has ", &
                      flag_count(nb), " / ", rz_points, " points", &
                      " with rho < ", this%radial_markers(nb), &
                      "and vp length: ", this%vp_length_blocks(nb)
            else if (nb == this%num_blocks) then
                write(logger_get_debug_channel(), *) "Block ", nb, " has ", &
                      flag_count(nb), " / ", rz_points, " points", &
                      " with rho >= ", this%radial_markers(nb-1), &
                      "and vp length: ", this%vp_length_blocks(nb)
            else
                write(logger_get_debug_channel(), *) "Block ", nb, " has ", &
                      flag_count(nb), " / ", rz_points, " points", &
                      " with ", this%radial_markers(nb-1), " <= rho < ", &
                      this%radial_markers(nb), "and vp length: ", &
                      this%vp_length_blocks(nb)
            end if
        end do

        deallocate(bflag)
    end subroutine

end submodule
