submodule (dist_initial_m) dist_initial_anisotropic_s

    implicit none

contains

    module subroutine initialize_anisotropic(this, profile_container, spec_ind)
        class(dist_initial_anisotropic_t), intent(inout) :: this
        type(profile_container_t), dimension(:), intent(in) :: profile_container
        integer, intent(in) :: spec_ind

        this%spec_ind = spec_ind
        this%profile_container = profile_container
    end subroutine

end submodule
