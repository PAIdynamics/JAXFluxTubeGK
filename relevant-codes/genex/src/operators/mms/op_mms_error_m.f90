module op_mms_error_m
    use genex_fortran_env_m, only: GP
    use data_array_m, only: data_array_2d_t, data_array_5d_t
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t
    use op_base_m, only: op_base_t
    use bsg_operators_m, only: bsg_operators_t
    use bsg_types_m, only: vspace_bsg_t
    implicit none

    type, public, abstract, extends(op_base_t) :: op_mms_error_t
        !! Operator to calculate the normalized L2 and Linf error given by
        !! ||f - f_analytic||_p / ||f_analytic||_p
        !! with respect to the analytical MMS solution. For the L2 error p=2
        !! and for the Linf error p=inf.
        !! We calculate the error of the ion and electron distribution,
        !! the electrostatic potential, the parallel vector potential,
        !! the parallel magnetic field fluctuations, and the
        !! parallel electric field.
        class(mesh_5d_t), private, pointer :: mesh
        !! Pointer to the mesh
        class(bsg_operators_t), private, pointer :: bsg_op
        !! Pointer to the BSG operators
        real(kind=GP), pointer, contiguous, dimension(:,:) :: absB
        !! Pointer to the absolute magnetic field
        real(kind=GP), pointer, contiguous, dimension(:,:) :: curl_normb_y
        !! Pointer to the curl of the magnetic field in y direction
        real(kind=GP), pointer, contiguous, dimension(:,:) :: R
        !! Pointer to values of the unstructured grid in R direction
        real(kind=GP), pointer, contiguous, dimension(:,:) :: Z
        !! Pointer to values of the unstructured grid in Z direction
        real(kind=GP), pointer, contiguous, dimension(:) :: vp
        !! Pointer to values of the grid in vp direction
        type(vspace_bsg_t), pointer, contiguous, dimension(:) :: vp_bsg
        !! Pointer to the vp grid for BSG
        real(kind=GP), pointer, contiguous, dimension(:) :: phi
        !! Pointer to values of the grid in phi direction
        real(kind=GP), pointer, contiguous, dimension(:) :: mu
        !! Pointer to values of the grid in mu direction
        real(kind=GP), pointer, contiguous, dimension(:) :: muw
        !! Pointer to integration weights in mu direction
        real(kind=GP), pointer, contiguous, dimension(:) :: vpw
        !! Pointer to integration weights in vp direction
        type(vspace_bsg_t), pointer, contiguous, dimension(:) :: vpw_bsg
        !! Pointer to the vp grid for BSG
        real(kind=GP), pointer, contiguous, dimension(:) :: phiw
        !! Pointer to integration weights in phi direction
        real(kind=GP), pointer, contiguous, dimension(:,:) :: rzw
        !! Pointer to integration weights in RZ direction
        type(dcomm_handler_t), private, pointer :: dcomm_handler
        !! Pointer to the communication handler
        integer, pointer, contiguous, dimension(:) :: bsg_flags
        !! Pointer to the BSG flags
    contains
        procedure(initialize_interface), deferred :: initialize
        procedure(apply_interface), deferred :: apply
    end type

    interface
        subroutine initialize_interface(this, dcomm_handler, mesh, bsg_op)
            !! Initializes the operator
            import op_mms_error_t, mesh_5d_t, dcomm_handler_t, bsg_operators_t
            class(op_mms_error_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), target, intent(in) :: dcomm_handler
            !! Comm handler for MPI communication
            class(mesh_5d_t), target, intent(in) :: mesh
            !! Mesh
            class(bsg_operators_t), target, intent(in) :: bsg_op
            !! BSG operators
        end subroutine

        subroutine apply_interface(this, t, da_f_in, da_phi_in, da_A_par_in, &
                                   da_B_par_in, da_E_par_in, l2_err, linf_err)
            !! Applies the operator to the given input values defined on
            !! the stripped bounds.
            import op_mms_error_t, GP, data_array_2d_t, data_array_5d_t
            class(op_mms_error_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: t
            !! Simulation time
            class(data_array_5d_t), intent(in) :: da_f_in
            !! Distribution function (stripped)
            class(data_array_2d_t), intent(in) :: da_phi_in
            !! Electrostatic potential (stripped)
            class(data_array_2d_t), intent(in) :: da_A_par_in
            !! Parallel electromagnetic vector potential (stripped)
            class(data_array_2d_t), intent(in) :: da_B_par_in
            !! Parallel magnetic field fluctuations (stripped)
            class(data_array_2d_t), intent(in) :: da_E_par_in
            !! Parallel electric field (stripped)
            real(kind=GP), dimension(6), intent(out) :: l2_err
            !! L2 error in the order
            !! 1) Distribution function ions
            !! 2) Distribution function electrons
            !! 3) Electrostatic potential
            !! 4) Parallel electromagnetic vector potential
            !! 5) Parallel magnetic field fluctuations
            !! 6) Parallel electric field
            real(kind=GP), dimension(6), intent(out) :: linf_err
            !! Linf error in the order
            !! 1) Distribution function ions
            !! 2) Distribution function electrons
            !! 3) Electrostatic potential
            !! 4) Parallel electromagnetic vector potential
            !! 5) Parallel magnetic field fluctuations
            !! 6) Parallel electric field
        end subroutine
    end interface

    type, public, extends(op_mms_error_t) :: op_mms_error_cpu_t
        !! Operator to calculate the normalized L2 error and Linf error
        !! on the cpu
    contains
        procedure :: initialize => initialize_cpu
        procedure :: apply => apply_cpu
    end type

    interface
        module subroutine initialize_cpu(this, dcomm_handler, mesh, bsg_op)
            class(op_mms_error_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), target, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(in) :: mesh
            class(bsg_operators_t), target, intent(in) :: bsg_op
        end subroutine

        module subroutine apply_cpu(this, t, da_f_in, da_phi_in, &
                                    da_A_par_in, da_B_par_in, da_E_par_in, &
                                    l2_err, linf_err)
            class(op_mms_error_cpu_t), intent(inout) :: this
            real(kind=GP), intent(in) :: t
            class(data_array_5d_t), intent(in) :: da_f_in
            class(data_array_2d_t), intent(in) :: da_phi_in, da_A_par_in, &
                                                  da_B_par_in, da_E_par_in
            real(kind=GP), dimension(6), intent(out) :: l2_err
            real(kind=GP), dimension(6), intent(out) :: linf_err
        end subroutine
    end interface

end module op_mms_error_m
