submodule(structured_grids_m) vp_grid_s
    use quadrature_m, only: create_midpoint_weights, &
                            create_trapezoidal_weights, &
                            create_simpson_weights

    ! From PARALLAX
    use array_generation_m, only: linspace

    implicit none

contains

    module subroutine vp_initialize(this, npoints, &
                                    is_vspectral)
        class(vp_grid_t), intent(inout) :: this
        integer, intent(in) :: npoints
        logical, optional, intent(in) :: is_vspectral

        allocate(this%points(npoints))
        allocate(this%weights(npoints))

        this%number_of_points = npoints

        if(present(is_vspectral)) then
            this%is_vspectral = is_vspectral
        else
            this%is_vspectral = .false.
        end if

    end subroutine

    module subroutine vp_construct(this, length, quad_type )
        class(vp_grid_t), intent(inout) :: this
        real(kind=GP), intent(in) :: length
        character(len=*), intent(in) :: quad_type

        integer :: ierr
        ! Error code

        if(this%is_vspectral) then

            this%grid_type = 'spectral_hermite'

            ! Hermite spectral grid
            ! Create grid point values associated with moment order
            ! from [0, npoints - 1]
            this%points = linspace(0.0_GP, real(this%number_of_points, GP), &
                                   this%number_of_points, endpoint=.false.)
            this%weights = 0.0_GP
            this%weights(1) = 1.0_GP
            this%quad_type = "spectral_hermite"

        else

            this%grid_type = 'uniform'
            this%quad_type = quad_type

            ! Create uniform grid from -length to length including the endpoints
            this%points = linspace(-length, length, this%number_of_points, &
                                   endpoint=.true.)

            ! Create integration weights based on the option given
            select case(quad_type)
                case("midpoint")
                    ! Midpoint quadrature
                    call create_midpoint_weights(this%number_of_points, &
                                                 this%points, this%weights)

                case("trapezoidal")
                    ! Trapezoidal quadrature
                    call create_trapezoidal_weights(this%number_of_points, &
                                                    this%points, this%weights)

                case("simpson")
                    ! Simpson quadrature
                    call create_simpson_weights(this%number_of_points, &
                                                this%points, this%weights, ierr)
                    if(ierr /= 0) then
                        call handle_error("Failed to initialize Simpson &
                                          &quadrature for vp grid!", &
                                          GENEX_ERR_MESH, __LINE__, __FILE__,&
                                          additional_info=error_info_t(&
                                          "Number of points was: ", &
                                          [this%number_of_points]))
                    end if

                case default
                    ! Not supported
                    call handle_error("Selected quad type "//quad_type &
                                      //" is not supported for vp grid!", &
                                      GENEX_ERR_MESH, __LINE__, __FILE__)
            end select
        end if

    end subroutine

end submodule
