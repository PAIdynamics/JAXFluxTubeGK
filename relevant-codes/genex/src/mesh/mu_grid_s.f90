submodule(structured_grids_m) mu_grid_s
    use quadrature_m, only: create_midpoint_weights, create_gauss_laguerre

    ! From PARALLAX
    use array_generation_m, only: linspace

    implicit none

contains

    module subroutine mu_initialize(this, length, npoints, grid_type, &
                                    is_vspectral)
        class(mu_grid_t), intent(inout) :: this
        real(kind=GP), intent(in) :: length
        integer, intent(in) :: npoints
        character(len=*), intent(in) :: grid_type
        logical, optional, intent(in) :: is_vspectral

        integer :: ierr
        ! Error code

        allocate(this%points(npoints))
        allocate(this%sqrt_points(npoints))
        allocate(this%weights(npoints))

        this%number_of_points = npoints

        if(present(is_vspectral)) then
            this%is_vspectral = is_vspectral
        else
            this%is_vspectral = .false.
        end if

        if(this%is_vspectral) then

            this%grid_type = 'spectral_laguerre'

            ! Laguerre spectral grid
            ! Create grid point values associated with moment order
            ! from [0, npoints - 1]
            this%points = linspace(0.0_GP, real(npoints, GP), npoints, &
                                   endpoint=.false.)
            this%weights = 0.0_GP
            this%weights(1) = 1.0_GP
            this%quad_type = "spectral_laguerre"

        else

            this%grid_type = grid_type

            ! Check mu grid type
            select case(grid_type)
                case("uniform")
                    ! Create uniform grid from 0 to length excluding the
                    ! endpoint. Use staggering ifspecified
                    this%points = linspace(0.0_GP, length, &
                                           npoints, stagger=.true.)

                    ! Create midpoint weights for uniform grid
                    call create_midpoint_weights(npoints, &
                                                 this%points, this%weights)
                    this%quad_type = "midpoint"

                case("quadratic")
                    ! NOTE: The actual length of the grid generated is different
                    !       from the given length by a factor of (1 - 1/npoints)

                    ! Create a staggered uniform grid excluding the midpoint and
                    ! square it
                    this%points = linspace(0.0_GP, sqrt(length), npoints, &
                                           stagger=.true.)**2
                    this%sqrt_points = sqrt(this%points)

                    ! Create midpoint weights for quadratic grid
                    call create_midpoint_weights(npoints, this%points, &
                                                 this%weights, quadratic=.true.)
                    this%quad_type = "midpoint"

                case("gauss-laguerre")
                    ! Initialize Gauss-Laguerre points and weights at once
                    call create_gauss_laguerre(npoints, length, this%points, &
                                               this%weights, ierr)
                    this%quad_type = "gauss-laguerre"
                    if(ierr /= 0) then
                        call handle_error("Failed to initialize Gauss-Laguerre &
                                          &quadrature for mu grid!", &
                                          GENEX_ERR_MESH, __LINE__, __FILE__,&
                                          additional_info=error_info_t(&
                                            "Number of points was: ", &
                                            [npoints]))
                    end if

                case default
                    ! Not supported
                    call handle_error("Selected grid type "//grid_type &
                                      //" is not supported for mu grid!", &
                                      GENEX_ERR_MESH, __LINE__, __FILE__)
            end select
        end if
    end subroutine

    module function mu_get_sqrt_pointer(this) result(ptr)
        class(mu_grid_t), target, intent(inout) :: this
        real(kind=GP), pointer, contiguous, dimension(:) :: ptr

        ptr => this%sqrt_points
    end function

    pure module real(kind=GP) function mu_get_sqrt_spacing(this, i)
        class(mu_grid_t), target, intent(in) :: this
        integer, intent(in) :: i

        if (i >= 1 .and. i < this%number_of_points) then
            mu_get_sqrt_spacing = this%sqrt_points(i + 1) &
                                - this%sqrt_points(i)
        else if (this%number_of_points == 1) then
            mu_get_sqrt_spacing = this%sqrt_points(1)
        else
            mu_get_sqrt_spacing = 0.0_GP
        end if
    end function

end submodule
