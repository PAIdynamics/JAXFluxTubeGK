submodule(bsg_operators_m) bsg_dot_product_s
    use bsg_types_m, only: unpack_bflag

    implicit none

contains

    module subroutine dot_product_bsg(this, coeff, points_neigh, lb1, lb2, &
                                      lb3, f_in, nb_point, k, l, f_dot)
        class(bsg_operators_t), intent(inout) :: this
        real(kind=GP), dimension(:), intent(in) :: coeff
        integer, dimension(:), intent(in) :: points_neigh
        integer, intent(in) :: lb1
        integer, intent(in) :: lb2
        integer, intent(in) :: lb3
        real(kind=GP), dimension(lb1:,lb2:,lb3:), intent(in) :: f_in
        integer, intent(in) :: nb_point
        integer, intent(in) :: k
        integer, intent(in) :: l
        real(kind=GP), intent(out) :: f_dot

        real(kind=GP), dimension(size(coeff)) :: f_local
        integer :: i, nb_neigh, i_neigh

        do i = 1, size(coeff)
            i_neigh = points_neigh(i)
            nb_neigh = unpack_bflag(this%bsg_flags(i_neigh))
            if (nb_point == nb_neigh) then
                f_local(i) = f_in(i_neigh, k, l)
            else
                call this%interpolate_dist(nb_point, nb_neigh, i_neigh, &
                                           k, l, f_in, lb1, lb2, lb3, &
                                           f_local(i))
            end if
        end do
        f_dot = dot_product(coeff, f_local)
    end subroutine

end submodule
