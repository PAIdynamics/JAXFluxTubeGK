module op_neut_set_initial_m
    use genex_fortran_env_m, only: GP
    use data_array_m, only: data_array_4d_t
    use op_base_m, only: op_base_t
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t

    implicit none

    type, public, abstract, extends(op_base_t) :: op_neut_set_initial_t
        !! Operator to initialize the neutrals state
        class(dcomm_handler_t), private, pointer :: dcomm_handler
        !! Pointer to the dcomm_handler
        class(mesh_5d_t), private, pointer :: mesh
        !! Pointer to the mesh
        real(kind=GP), private, dimension(:), allocatable :: &
                                                        neutrals_state_initial
        !! Container for the initial values of the neutrals state (n, u, T)
    contains
        procedure(initialize_interface), deferred :: initialize
        procedure(apply_interface), deferred :: apply
    end type

    interface !op_neut_set_initial_t
        subroutine initialize_interface(this, dcomm_handler, mesh)
            !! Initializes the operator
            import op_neut_set_initial_t, dcomm_handler_t, mesh_5d_t

            class(op_neut_set_initial_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Pointer to communications handler
            class(mesh_5d_t), target, intent(in) :: mesh
            !! Mesh
        end subroutine

        subroutine apply_interface(this, neut_state)
            !! Set the initial neutrals states the given input values
            import op_neut_set_initial_t, data_array_4d_t

            class(op_neut_set_initial_t), target, intent(inout) :: this
            !! Instance of the type
            class(data_array_4d_t), intent(inout) :: neut_state
            !! Neutrals state
        end subroutine
    end interface

    type, public, extends(op_neut_set_initial_t) :: op_neut_set_initial_cpu_t
        !! Operator to initialize the neutrals state, cpu version
    contains
        procedure, private :: set_blob_perturbation_cpu
        procedure, private :: set_noise_perturbation_cpu
        procedure, private :: apply_uniform_cpu
        procedure, private :: apply_plateau_sine_cpu
        procedure :: initialize => initialize_cpu
        procedure :: apply => apply_cpu
    end type

    interface !op_neut_set_initial_cpu_t
        module subroutine set_blob_perturbation_cpu(this, neut_state, &
                                                    neut_mom, neut_spec)
            !! Add a blob perturbation on top of the initial state
            class(op_neut_set_initial_cpu_t), target, intent(inout) :: this
            !! Instance of the type
            class(data_array_4d_t), intent(inout) :: neut_state
            !! Neutrals state
            integer, intent(in) :: neut_mom
            !! Neutrals moments index
            integer, intent(in) :: neut_spec
            !! Neutrals species index
        end subroutine

        module subroutine set_noise_perturbation_cpu(this, neut_state, &
                                                     neut_mom, neut_spec)
            !! Add a noise perturbation on top of the initial state
            class(op_neut_set_initial_cpu_t), target, intent(inout) :: this
            !! Instance of the type
            class(data_array_4d_t), intent(inout) :: neut_state
            !! Neutrals state
            integer, intent(in) :: neut_mom
            !! Neutrals moments index
            integer, intent(in) :: neut_spec
            !! Neutrals species index
        end subroutine

        module subroutine apply_uniform_cpu(this, neut_state, &
                                            neut_mom, neut_spec)
            !! Apply uniform initial profile
            class(op_neut_set_initial_cpu_t), target, intent(inout) :: this
            !! Instance of the type
            class(data_array_4d_t), intent(inout) :: neut_state
            !! Neutrals state
            integer, intent(in) :: neut_mom
            !! Neutrals moments index
            integer, intent(in) :: neut_spec
            !! Neutrals species index
        end subroutine

        module subroutine apply_plateau_sine_cpu(this, neut_state, &
                                                 neut_mom, neut_spec)
            !! Apply initial profile as a plateau with sine slope
            class(op_neut_set_initial_cpu_t), target, intent(inout) :: this
            !! Instance of the type
            class(data_array_4d_t), intent(inout) :: neut_state
            !! Neutrals state
            integer, intent(in) :: neut_mom
            !! Neutrals moments index
            integer, intent(in) :: neut_spec
            !! Neutrals species index
        end subroutine

        module subroutine initialize_cpu(this, dcomm_handler, mesh)
            class(op_neut_set_initial_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(in) :: mesh
        end subroutine

        module subroutine apply_cpu(this, neut_state)
            class(op_neut_set_initial_cpu_t), target, intent(inout) :: this
            class(data_array_4d_t), intent(inout) :: neut_state
        end subroutine

    end interface
end module
