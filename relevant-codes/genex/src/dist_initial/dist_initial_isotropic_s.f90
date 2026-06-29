submodule (dist_initial_m) dist_initial_isotropic_s

    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_ERR_DIST_INITIAL
    implicit none

contains

    module subroutine initialize_isotropic(this, profile_container, spec_ind)
        class(dist_initial_isotropic_t), intent(inout) :: this
        type(profile_container_t), dimension(:), intent(in) :: profile_container
        integer, intent(in) :: spec_ind

        this%spec_ind = spec_ind

        ! Check that the profile container contains isotropic temperature
        ! at this index, otherwise terminate
        if(profile_container(spec_ind)%get_is_isotropic()) then
            this%profile_container = profile_container
        else
            call handle_error("Tried to initialize isotropic initial &
                              &distribution with anisotropic profile &
                              &container!", &
                              GENEX_ERR_DIST_INITIAL, __LINE__, __FILE__)
        end if

    end subroutine

end submodule
