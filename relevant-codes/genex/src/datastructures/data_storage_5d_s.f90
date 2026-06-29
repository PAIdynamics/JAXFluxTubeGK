submodule (data_storage_m) data_storage_5d_s

    use params_mesh_m, only: get_use_vspectral
    use params_collisions_m, only: get_coll_type
    use dimensions_m, only: DIM_PHI, DIM_VP, DIM_MU
    implicit none

contains

    module function get_mailbox_index_5d(this, ex_dim) result(mb_ind)
        class(data_storage_5d_t), intent(in) :: this
        integer, intent(in) :: ex_dim
        integer :: mb_ind

        if(ex_dim == this%dim_permut(DIM_PHI)) then
            mb_ind = 1
        else if(ex_dim == this%dim_permut(DIM_VP)) then
            mb_ind = 2
        else if(ex_dim == this%dim_permut(DIM_MU)) then
            mb_ind = 3
        end if
    end function

    module function get_physical_dimension_5d(this, ex_dim) result(res)
        class(data_storage_5d_t), intent(in) :: this
        integer, intent(in) :: ex_dim
        integer :: res

        if(ex_dim == this%dim_permut(DIM_PHI)) then
            res = 2
        else if(ex_dim == this%dim_permut(DIM_VP)) then
            res = 3
        else if(ex_dim == this%dim_permut(DIM_MU)) then
            res = 4
        end if
    end function

    module function has_edge_ghosts_5d(this, ex_dim, comm_dim) result(res)
        class(data_storage_5d_t), intent(in) :: this
        integer, intent(in) :: ex_dim
        integer, intent(in) :: comm_dim
        integer :: res
        res = 0

        if(.not. get_use_vspectral() .and. get_coll_type() /= "lorentz") return

        if(ex_dim == this%dim_permut(DIM_MU)) then
            ! NOTE: Ghost cells in vp are included
            if(comm_dim == this%dim_permut(DIM_VP)) res = 1
        endif

        if(ex_dim == this%dim_permut(DIM_PHI)) then
            ! NOTE: Ghost cells in vp, mu are included
            if(comm_dim == this%dim_permut(DIM_VP)) res = 1
            if(comm_dim == this%dim_permut(DIM_MU)) res = 1
        endif
    end function

end submodule
