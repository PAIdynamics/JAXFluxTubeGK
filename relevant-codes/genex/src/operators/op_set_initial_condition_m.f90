module op_set_initial_condition_m
    use genex_fortran_env_m, only: GP
    use data_array_m, only: data_array_5d_t
    use op_base_m, only: op_base_t
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t
    use dist_initial_container_m, only: dist_initial_container_t
    use bsg_operators_m, only: bsg_operators_t

    implicit none

    type, public, abstract, extends(op_base_t) :: &
                                                 op_set_initial_condition_base_t
        !! Operator to initialize the distribution function, base
        class(dcomm_handler_t), private, pointer :: dcomm_handler
        !! Pointer to the dcomm_handler
        class(mesh_5d_t), private, pointer :: mesh
        !! Pointer to the mesh
        type(dist_initial_container_t), private, dimension(:), allocatable :: &
                                                        dist_initial_container
        !! Container that contains all information about the initial dist func
        type(bsg_operators_t), private, pointer :: bsg_op
        !! Pointer to the BSG operators
    contains
        procedure(apply), deferred :: apply
        procedure :: set_noise_perturbation
    end type

    interface !op_set_initial_condition_t
        subroutine apply(this, da_f_inout)
            !! Applies the operator to the given input values
            import op_set_initial_condition_base_t, data_array_5d_t
            class(op_set_initial_condition_base_t), target, &
                                                           intent(inout) :: this
            !! Instance of the type
            class(data_array_5d_t), intent(inout) :: da_f_inout
            !! Distribution function
        end subroutine

        module subroutine set_noise_perturbation(this, da_f_inout)
            !! Adds a noise perturbation to the initial dist func
            class(op_set_initial_condition_base_t), target, &
                                                          intent(inout) :: this
            !! Instance of the type
            class(data_array_5d_t), intent(inout) :: da_f_inout
            !! Distribution function
        end subroutine
    end interface

    type, public, abstract, extends(op_set_initial_condition_base_t) :: &
                                                      op_set_initial_condition_t
        !! Operator to initialize the distribution function using grid approach
        real(kind=GP), private, allocatable, dimension(:) :: prefac_bps
        !! Normalization factor for b_par^star
    end type

    type, public, extends(op_set_initial_condition_t) :: &
                                                op_set_initial_condition_cpu_t
        !! Operator to initialize the distribution function, CPU implementation
    contains
        procedure :: initialize => initialize_cpu
        procedure :: apply => apply_cpu
        procedure :: set_blob_perturbation => set_blob_perturbation_cpu
        procedure :: set_mode_perturbation => set_mode_perturbation_cpu
    end type

    interface !op_set_initial_condition_cpu_t
        module subroutine initialize_cpu(this, dcomm_handler, mesh, &
                                         dist_initial_container, bsg_op)
            class(op_set_initial_condition_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(in) :: mesh
            type(dist_initial_container_t), &
                dimension(:), allocatable, intent(in) :: dist_initial_container
            type(bsg_operators_t), target, intent(inout) :: bsg_op
            !! BSG operators
        end subroutine

        module subroutine set_blob_perturbation_cpu(this, da_f_inout)
            !! Adds a blob perturbation to the initial dist func
            class(op_set_initial_condition_cpu_t), target, intent(inout) :: this
            !! Instance of the type
            class(data_array_5d_t), intent(inout) :: da_f_inout
            !! Distribution function
        end subroutine

        module subroutine set_mode_perturbation_cpu(this, da_f_inout)
            !! Adds a mode perturbation to the initial dist func
            class(op_set_initial_condition_cpu_t), target, intent(inout) :: this
            !! Instance of the type
            class(data_array_5d_t), intent(inout) :: da_f_inout
            !! Distribution function
        end subroutine

        module subroutine apply_cpu(this, da_f_inout)
            class(op_set_initial_condition_cpu_t), target, intent(inout) :: this
            class(data_array_5d_t), intent(inout) :: da_f_inout
        end subroutine
    end interface

    type, public, extends(op_set_initial_condition_t) :: &
                                              op_set_initial_condition_mms_cpu_t
        !! Operator to initialize the distribution function for MMS, CPU
        !! implementation
    contains
        procedure :: initialize => initialize_mms_cpu
        procedure :: apply => apply_mms_cpu
    end type

    interface !op_set_initial_condition_mms_cpu_t
        module subroutine initialize_mms_cpu(this, mesh)
            class(op_set_initial_condition_mms_cpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(in) :: mesh
        end subroutine

        module subroutine apply_mms_cpu(this, da_f_inout)
            class(op_set_initial_condition_mms_cpu_t), target, &
                                                           intent(inout) :: this
            class(data_array_5d_t), intent(inout) :: da_f_inout
        end subroutine
    end interface

    type, public, abstract, extends(op_set_initial_condition_base_t) :: &
                                                op_set_initial_condition_vspec_t
        !! Operator to initialize the distribution function using spectral
        !! approach
    end type

    type, public, extends(op_set_initial_condition_vspec_t) :: &
                                            op_set_initial_condition_vspec_cpu_t
        !! Operator to initialize the distribution function using
        !! spectral approach, CPU implementation
    contains
        procedure :: initialize => initialize_vspec_cpu
        procedure :: apply => apply_vspec_cpu
    end type

    interface !op_set_initial_condition_vspec_cpu_t

        module subroutine initialize_vspec_cpu(this, dcomm_handler, mesh, &
                                               dist_initial_container)
            !! Initializes the spectral distribution function and
            !! adds a perturbation to it
            class(op_set_initial_condition_vspec_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(in) :: mesh
            type(dist_initial_container_t), &
                dimension(:), allocatable, intent(in) :: dist_initial_container
        end subroutine
        module subroutine apply_vspec_cpu(this, da_f_inout)
            !! Applies the spectral operator to the given input values
            class(op_set_initial_condition_vspec_cpu_t), target, &
                                                           intent(inout) :: this
            class(data_array_5d_t), intent(inout) :: da_f_inout
        end subroutine apply_vspec_cpu
    end interface

end module
