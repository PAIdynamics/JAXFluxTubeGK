submodule(bsg_interpolation_m) equidistant_interpolation_s

    use genex_error_handling_m, only: handle_error, error_info_t
    use genex_status_codes_m, only: GENEX_ERR_BSG

    implicit none

contains

    module subroutine eval_local_index_equidistant(this, x, x_values, &
                                                   start_idx, end_idx)
        class(equidistant_interpolation_t), intent(inout) :: this
        real(kind=GP), intent(in) :: x
        real(kind=GP), dimension(:), intent(in) :: x_values
        integer, intent(out) :: start_idx
        integer, intent(out) :: end_idx

        real(kind=GP) :: dx
        integer :: xdiff
        integer :: lbound_x_array, ubound_x_array

        lbound_x_array = lbound(x_values, 1)
        ubound_x_array = ubound(x_values, 1)

        ! Ensure x is between the bounds of x_values
        if (x < x_values(lbound_x_array) .or. &
            x > x_values(ubound_x_array)) then
            ! Return as it is an external point
            return
        end if

        ! Get grid spacing
        dx = x_values(lbound_x_array + 1) - x_values(lbound_x_array)

        start_idx = this%eval_lsp_equidistant(x, lbound_x_array, &
                                              x_values(lbound_x_array), dx)
        ! Ensure that the start index is within bounds
        start_idx = max(start_idx, lbound_x_array)

        end_idx = start_idx + this%interp_order

        ! Ensure that the end_index is below upper bounds and adjust the start
        ! index if not
        if (end_idx > ubound_x_array) then
            xdiff = end_idx - ubound_x_array
            end_idx = ubound_x_array
            start_idx = max(start_idx - xdiff, lbound_x_array)
        end if

        if (end_idx - start_idx /= this%interp_order) then
            call handle_error("Interpolation indices do not fit interpolation &
                            &order", GENEX_ERR_BSG, &
                            __LINE__, __FILE__, &
                            additional_info=error_info_t("Start, end, order:", &
                            [start_idx, end_idx, this%interp_order]))
        end if
    end subroutine

    module subroutine interpolate_equidistant(this, x_values, y_values, &
                                              coeff, x, y_interp)
        class(equidistant_interpolation_t), intent(inout) :: this
        real(kind=GP), dimension(:), intent(in) :: x_values
        real(kind=GP), dimension(:), intent(in) :: y_values
        real(kind=GP), dimension(:), intent(in) :: coeff
        real(kind=GP), intent(in) :: x
        real(kind=GP), intent(out) :: y_interp

        integer :: j1, j2
        real(kind=GP) :: term

        y_interp = 0.0_GP
        do j1 = 1, this%interp_order+1
            term = coeff(j1) * y_values(j1)
            do j2 = 1, this%interp_order+1
                if (j2 == j1) cycle
                term = term * (x - x_values(j2))
            end do
            y_interp = y_interp + term
        end do
      end subroutine

      module subroutine eval_coeff_equidistant(this, x_values, coeff)
          class(equidistant_interpolation_t), intent(inout) :: this
          real(kind=GP), dimension(:), intent(in) :: x_values
          real(kind=GP), dimension(:), intent(out) :: coeff

          integer :: j1, j2
          real(kind=GP) :: term

          do j1 = 1, this%interp_order + 1
              term = 1.0_GP
              do j2 = 1, this%interp_order + 1
                if (j2 == j1) cycle
                term = term / (x_values(j1) - x_values(j2))
              end do
              coeff(j1) = term
          end do
      end subroutine

end submodule
