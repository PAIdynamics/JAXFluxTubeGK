submodule (structured_grids_m) phi_grid_s
    use quadrature_m, only: create_midpoint_weights
    ! From PARALLAX
    use array_generation_m, only: linspace

    implicit none

contains

    module subroutine phi_initialize(this, length, npoints)
        class(phi_grid_t), intent(inout) :: this
        real(kind=GP), intent(in) :: length
        integer, intent(in) :: npoints

        this%number_of_points = npoints
        allocate(this%points(npoints))
        allocate(this%weights(npoints))

        ! Create uniform grid from 0 to length exluding the endpoint
        this%points = linspace(0.0_GP, length, npoints)
        this%grid_type = "uniform"

        ! Create midpoint weights for phi integration
        call create_midpoint_weights(npoints, this%points, this%weights)
        this%quad_type = "midpoint"

    end subroutine

end submodule
