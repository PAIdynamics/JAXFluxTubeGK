module op_mms_solution_m
    !! Module containing the op_mms_solution operators.These operators set
    !! exact MMS solution on all points in the RZ plane and on all inner
    !! points in the other coordinates.
    use genex_fortran_env_m, only: GP
    use data_array_m, only: data_array_2d_t, data_array_5d_t
    use op_base_m, only: op_base_t
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t

    implicit none

    type, public, abstract, extends(op_base_t) :: &
                                               op_mms_solution_vlasov_eq_t
        !! Operator to set the distribution function to the exact MMS solution
        class(mesh_5d_t), private, pointer :: mesh
        !! Pointer to the mesh
    contains
        procedure(apply_vlasov_interface), deferred :: apply
        procedure(initialize_vlasov_interface), deferred :: initialize
    end type

    abstract interface
        subroutine apply_vlasov_interface(this, t, da_f_inout)
            !! Applies the operator to the given input values
            import op_mms_solution_vlasov_eq_t, GP, data_array_5d_t
            class(op_mms_solution_vlasov_eq_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: t
            !! Current time
            class(data_array_5d_t), intent(inout) :: da_f_inout
            !! Distribution function
        end subroutine

        subroutine initialize_vlasov_interface(this, dcomm_handler, mesh)
            !! Initializes the operator
            import op_mms_solution_vlasov_eq_t, dcomm_handler_t, mesh_5d_t
            class(op_mms_solution_vlasov_eq_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Domain and communication handler
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
        end subroutine

    end interface

    type, public, extends(op_mms_solution_vlasov_eq_t) :: &
                                                op_mms_solution_vlasov_eq_cpu_t
        !! Operator to set the distribution function to the exact
        !! MMS solution on the cpu
        real(kind=GP), private, allocatable, dimension(:,:) :: vp_total
        !! Array containing an extension of the vp grid to include ghosts
        !! generalized to BSG
    contains
        procedure, public :: initialize => initialize_mms_vlasov_cpu
        procedure, public :: apply => apply_mms_vlasov_cpu
    end type

    interface
        module subroutine initialize_mms_vlasov_cpu(this, dcomm_handler, mesh)
            class(op_mms_solution_vlasov_eq_cpu_t), intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_mms_vlasov_cpu(this, t, da_f_inout)
            class(op_mms_solution_vlasov_eq_cpu_t), intent(inout) :: this
            real(kind=GP), intent(in) :: t
            class(data_array_5d_t), intent(inout) :: da_f_inout
        end subroutine
    end interface

    type, public, abstract, extends(op_base_t) :: op_mms_solution_maxwells_eq_t
        !! Operator to set the electrostatic potential, the parallel
        !! electromagnetic vector potential and the parallel magnetic field
        !! to the exact MMS solution
        class(mesh_5d_t), private, pointer :: mesh
        !! Pointer to the mesh
    contains
        procedure(apply_interface), deferred :: apply
        procedure(initialize_interface), deferred :: initialize
    end type

    abstract interface
        subroutine apply_interface(this, t, da_es_pot_inout, da_A_par_inout, &
                                  da_B_par_inout)
            !! Applies the operator to the given input values
            import op_mms_solution_maxwells_eq_t, GP, data_array_2d_t
            class(op_mms_solution_maxwells_eq_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: t
            !! Current time
            class(data_array_2d_t), intent(inout) :: da_es_pot_inout
            !! Electrostatic potential
            class(data_array_2d_t), intent(inout) :: da_A_par_inout
            !! Parallel electromagnetic vector potential
            class(data_array_2d_t), intent(inout) :: da_B_par_inout
            !! Parallel magnetic field fluctuations
        end subroutine

        subroutine initialize_interface(this, mesh)
            !! Initializes the operator
            import op_mms_solution_maxwells_eq_t, mesh_5d_t
            class(op_mms_solution_maxwells_eq_t), intent(inout) :: this
            !! Instance of the type
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
        end subroutine

    end interface

    type, public, extends(op_mms_solution_maxwells_eq_t) :: &
        op_mms_solution_maxwells_eq_cpu_t
        !! Operator to set the electrostatic potential, the parallel
        !! electromagnetic vector potential and the parallel magnetic field
        !! to the exact MMS solution on the cpu
    contains
        procedure, public :: initialize => initialize_mms_cpu
        procedure, public :: apply => apply_mms_cpu
    end type

    interface
        module subroutine initialize_mms_cpu(this, mesh)
            class(op_mms_solution_maxwells_eq_cpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_mms_cpu(this, t, da_es_pot_inout, &
                                        da_A_par_inout, da_B_par_inout)
            class(op_mms_solution_maxwells_eq_cpu_t), intent(inout) :: this
            real(kind=GP), intent(in) :: t
            class(data_array_2d_t), intent(inout) :: da_es_pot_inout
            class(data_array_2d_t), intent(inout) :: da_A_par_inout
            class(data_array_2d_t), intent(inout) :: da_B_par_inout
        end subroutine
    end interface

    type, public, abstract, extends(op_base_t) :: op_mms_solution_ohms_law_t
        !! Operator to set the parallel electric field to the exact MMS solution
        class(mesh_5d_t), private, pointer :: mesh
    contains
        procedure(apply_interface_ohm), deferred :: apply
        procedure(initialize_interface_ohm), deferred :: initialize
    end type

    abstract interface
        subroutine apply_interface_ohm(this, t, da_E_par_inout)
            !! Applies the operator to the given input values
            import op_mms_solution_ohms_law_t, GP, data_array_2d_t
            class(op_mms_solution_ohms_law_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: t
            !! Current time
            class(data_array_2d_t), intent(inout) :: da_E_par_inout
            !! Parallel electric field
        end subroutine

        subroutine initialize_interface_ohm(this, mesh)
            !! Initializes the operator
            import op_mms_solution_ohms_law_t, mesh_5d_t
            class(op_mms_solution_ohms_law_t), intent(inout) :: this
            !! Instance of the type
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
        end subroutine

    end interface

    type, public, extends(op_mms_solution_ohms_law_t) :: &
        op_mms_solution_ohms_law_cpu_t
        !! Operator to set the parallel electric field to the exact
        !! MMS solution on the cpu
    contains
        procedure, public :: initialize => initialize_mms_ohm_cpu
        procedure, public :: apply => apply_mms_ohm_cpu
    end type

    interface
        module subroutine initialize_mms_ohm_cpu(this, mesh)
            class(op_mms_solution_ohms_law_cpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(inout) :: mesh
        end subroutine

        module subroutine apply_mms_ohm_cpu(this, t, da_E_par_inout)
            class(op_mms_solution_ohms_law_cpu_t), intent(inout) :: this
            real(kind=GP), intent(in) :: t
            class(data_array_2d_t), intent(inout) :: da_E_par_inout
        end subroutine
    end interface

end module
