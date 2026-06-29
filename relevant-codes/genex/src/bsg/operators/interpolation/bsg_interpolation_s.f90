submodule(bsg_interpolation_m) bsg_interpolation_s

    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_ERR_BSG

    implicit none

contains

    module subroutine initialize(this, interp_order)
        class(bsg_interpolation_t), intent(inout) :: this
        integer, intent(in) :: interp_order

        this%interp_order = interp_order
    end subroutine

    integer module function eval_lsp_equidistant(this, x, xlb, x1, dx)
        class(bsg_interpolation_t), intent(in) :: this
        real(kind=GP), intent(in) :: x
        integer, intent(in) :: xlb
        real(kind=GP), intent(in) :: x1
        real(kind=GP), intent(in) :: dx

        ! Assumes atleast 2 interpolation points
        if (this%interp_order < 2) then
            call handle_error("Interpolation order should be greater than 1", &
                              GENEX_ERR_BSG, &
                              __LINE__, __FILE__)
        end if
        eval_lsp_equidistant = floor((x - x1) / dx &
                             + xlb - ((this%interp_order+1) - 2.0) / 2.0)
    end function

end submodule
