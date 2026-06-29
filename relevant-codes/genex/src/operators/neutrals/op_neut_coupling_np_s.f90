submodule(op_neut_coupling_np_m) op_neut_coupling_np_s
    !! This module contains the general initialization routine

    implicit none

contains

    module subroutine initialize_parent(this, dcomm_handler, mesh)
        class(op_neut_coupling_np_t), intent(inout) :: this
        type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
        class(mesh_5d_t), target, intent(inout) :: mesh

        this%dcomm_handler => dcomm_handler
        this%mesh => mesh
        call setup_neut_ions_maps(this%map_neut_to_ion, this%map_ion_to_neut)
    end subroutine

end submodule
