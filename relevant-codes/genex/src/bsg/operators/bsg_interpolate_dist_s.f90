submodule(bsg_operators_m) bsg_interpolate_dist_s

    implicit none

contains

    module subroutine interpolate_dist(this, nb_point, nb_neigh, i_neigh, &
                                       k, l, f_in, lb1, lb2, lb3, f_interp)
        class(bsg_operators_t), intent(inout) :: this
        integer, intent(in) :: nb_point
        integer, intent(in) :: nb_neigh
        integer, intent(in) :: i_neigh
        integer, intent(in) :: k
        integer, intent(in) :: l
        integer, intent(in) :: lb1
        integer, intent(in) :: lb2
        integer, intent(in) :: lb3
        real(kind=GP), dimension(lb1:,lb2:,lb3:), intent(in) :: f_in
        real(kind=GP), intent(inout) :: f_interp

        integer :: start_idx, end_idx
        ! start and end indices for interpolation

        if (this%is_internal(l, nb_point, nb_neigh)) then
            ! Internal case
            start_idx = this%local_start_idx(l, nb_point, nb_neigh)
            end_idx   = this%local_end_idx(l, nb_point, nb_neigh)
            call this%interpolator%interpolate(&
                        this%vp_bsg_ptr(nb_neigh)%array(start_idx:end_idx), &
                        f_in(i_neigh, k, start_idx:end_idx), &
                        this%interpolation_coeff(l, nb_point, nb_neigh, :), &
                        this%vp_bsg_ptr(nb_point)%array(l), &
                        f_interp)
        else
            ! External case
            f_interp = 0.0_GP
        end if

    end subroutine

end submodule