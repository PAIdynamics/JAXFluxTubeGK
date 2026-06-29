submodule(op_neut_coupling_pn_m) op_neut_coupling_pn_none_s
    !! Contain operator subroutines to calculate the effect of plasma
    !! on neutrals in the none case, i.e. nothing happen.

    implicit none

contains

    module subroutine initialize_coupling_pn_none(this, dcomm_handler, mesh)
        class(op_neut_coupling_pn_none_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        call this%initialize_parent(dcomm_handler, mesh)

    end subroutine

    module subroutine apply_coupling_pn_none(this, da_moments, da_f_in, &
                                             da_n_in, da_n_sources)
        class(op_neut_coupling_pn_none_t), intent(inout) :: this
        class(data_array_4d_t), intent(in) :: da_moments
        class(data_array_5d_t), intent(in) :: da_f_in
        class(data_array_4d_t), intent(in) :: da_n_in
        class(data_array_4d_t), intent(inout) :: da_n_sources

        call this%perf_counter%start_measurement()
        ! Nothing done here
        call this%perf_counter%end_measurement()

    end subroutine

end submodule
