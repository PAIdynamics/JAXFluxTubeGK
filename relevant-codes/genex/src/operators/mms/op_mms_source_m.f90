module op_mms_source_m
    !! Contains the MMS source operators used to apply the source terms to the
    !! rhs of the Vlasov-Maxwell system
    use genex_fortran_env_m, only: GP
    use mesh_5d_m, only: mesh_5d_t
    use op_base_m, only: op_base_t
    use bsg_operators_m, only: bsg_operators_t
    use bsg_types_m, only: vspace_bsg_t

    implicit none

    type, public, abstract, extends(op_base_t) :: op_mms_source_t
        !! Operator to provide the correct source for the MMS verification of
        !! the code
        class(mesh_5d_t), private, pointer :: mesh
        !! Pointer to the mesh
        class(bsg_operators_t), private, pointer :: bsg_op
        !! Pointer to the BSG operators
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: R
        !! Pointer to cylindrical R (in poloidal plane)
        real(kind=GP), private, contiguous, pointer, dimension(:) :: phi
        !! Pointer to toroidal angle
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: Z
        !! Pointer to cylindrical Z (in poloidal plane)
        real(kind=GP), private, contiguous, pointer, dimension(:) :: vp
        !! Pointer to parallel velocity
        type(vspace_bsg_t), private, contiguous, pointer, dimension(:) :: vp_bsg
        !! Pointer to the parallel velocity with BSG
        real(kind=GP), private, contiguous, pointer, dimension(:) :: mu
        !! Pointer to magnetic moment
        real(kind=GP), private, contiguous, pointer, dimension(:,:) :: B
        !! Pointer to absolute magnetic field
        integer, private, contiguous, pointer, dimension(:) :: bsg_flags
        !! Pointer to the BSG flags
    contains
        procedure(initialize_mms_interface), deferred :: initialize
    end type

    abstract interface
        subroutine initialize_mms_interface(this, mesh, bsg_op)
            !! Initializes the operator
            import op_mms_source_t, mesh_5d_t, GP, bsg_operators_t
            class(op_mms_source_t), intent(inout) :: this
            !! Instance of the type
            class(mesh_5d_t), target, intent(in) :: mesh
            !! Mesh
            class(bsg_operators_t), target, intent(in) :: bsg_op
            !! BSG operators
        end subroutine
    end interface

    type, public, abstract, extends(op_mms_source_t) :: &
                                                op_mms_source_vlasov_eq_t
        !! Base operator to calculate the MMS source of the Vlasov equation
    contains
        procedure(apply_interface_vlasov), deferred :: apply
    end type

    abstract interface
        subroutine apply_interface_vlasov(this, t, lb, ub, &
                                          lb_stripped, ub_stripped, f_out)
            !! Applies the operator by adding the source term to f_out
            import op_mms_source_vlasov_eq_t, GP
            class(op_mms_source_vlasov_eq_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: t
            !! Simulation time
            integer, intent(in) :: lb(5)
            !! Lower boundary with ghost cells
            integer, intent(in) :: ub(5)
            !! Upper boundary with ghost cells
            integer, intent(in) :: lb_stripped(5)
            !! Lower boundary without ghost cells
            integer, intent(in) :: ub_stripped(5)
            !! Upper boundary without ghost cells
            real(kind=GP), dimension(lb(1):ub(1), lb(2):ub(2), &
                                     lb(3):ub(3), lb(4):ub(4), &
                                     lb(5):ub(5)), intent(inout) :: f_out
            !! Distribution function
        end subroutine
    end interface

    type, public, extends(op_mms_source_vlasov_eq_t) :: &
                                                op_mms_source_vlasov_eq_cpu_t
        !! Operator to calculate the MMS source of the Vlasov equation for the
        !! grid approach on the CPU
    contains
        procedure :: initialize => initialize_vlasov_cpu
        procedure :: apply => apply_vlasov_cpu
    end type

    interface
        module subroutine initialize_vlasov_cpu(this, mesh, bsg_op)
            class(op_mms_source_vlasov_eq_cpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(in) :: mesh
            class(bsg_operators_t), target, intent(in) :: bsg_op
        end subroutine

        module subroutine apply_vlasov_cpu(this, t, lb, ub, &
                                           lb_stripped, ub_stripped, &
                                           f_out)
            class(op_mms_source_vlasov_eq_cpu_t), intent(inout) :: this
            real(kind=GP), intent(in) :: t
            integer, intent(in) :: lb(5), ub(5), lb_stripped(5), ub_stripped(5)
            real(kind=GP), dimension(lb(1):ub(1), lb(2):ub(2), &
                                     lb(3):ub(3), lb(4):ub(4), &
                                     lb(5):ub(5)), intent(inout) :: f_out
        end subroutine
    end interface

    type, public, abstract, extends(op_mms_source_t) :: &
                                            op_mms_source_maxwells_eq_t
        !! Base operator to calculate the MMS source of the quasi-neutrality
        !! equation and Ampere's law
    contains
        procedure(apply_interface_em_fields), deferred :: apply
    end type

    abstract interface
        subroutine apply_interface_em_fields(this, t, lb, ub, lb_stripped, &
                                             ub_stripped, &
                                             b_qn_eq, b_amps_law, b_bpar_eq)
            !! Applies the operator by adding the source terms to b_qn_eq,
            !! b_amps_law and b_bpar_eq
            import op_mms_source_maxwells_eq_t, GP
            class(op_mms_source_maxwells_eq_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: t
            !! Simulation time
            integer, intent(in) :: lb(2)
            !! Lower boundary with ghost cells
            integer, intent(in) :: ub(2)
            !! Upper boundary with ghost cells
            integer, intent(in) :: lb_stripped(2)
            !! Lower boundary without ghost cells
            integer, intent(in) :: ub_stripped(2)
            !! Upper boundary without ghost cells
            real(kind=GP), intent(inout) :: &
                b_qn_eq(lb(1):ub(1), lb_stripped(2):ub_stripped(2))
            !! Right hand side array of the quasi-neutrality equation
            real(kind=GP), intent(inout) :: &
                b_amps_law(lb(1):ub(1), lb_stripped(2):ub_stripped(2))
            !! Right hand side array of Ampere's law
            real(kind=GP), intent(inout) :: &
                b_bpar_eq(lb(1):ub(1), lb_stripped(2):ub_stripped(2))
            !! Right hand side array of the B parallel equation
        end subroutine
    end interface

    type, public, extends(op_mms_source_maxwells_eq_t) :: &
                                                op_mms_source_maxwells_eq_cpu_t
        !! Operator to calculate the MMS source of the quasi-neutrality equation
        !! and Ampere's law for the grid approach on the CPU
    contains
        procedure :: initialize => initialize_em_fields_cpu
        procedure :: apply => apply_em_fields_cpu
    end type

    interface
        module subroutine initialize_em_fields_cpu(this, mesh, bsg_op)
            class(op_mms_source_maxwells_eq_cpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(in) :: mesh
            class(bsg_operators_t), target, intent(in) :: bsg_op
        end subroutine

        module subroutine apply_em_fields_cpu(this, t, lb, ub, lb_stripped, &
                                              ub_stripped, &
                                              b_qn_eq, b_amps_law, b_bpar_eq)
            class(op_mms_source_maxwells_eq_cpu_t), intent(inout) :: this
            real(kind=GP), intent(in) :: t
            integer, intent(in) :: lb(2), ub(2), lb_stripped(2), ub_stripped(2)
            real(kind=GP), intent(inout) :: &
                b_qn_eq(lb(1):ub(1), lb_stripped(2):ub_stripped(2))
            real(kind=GP), intent(inout) :: &
                b_amps_law(lb(1):ub(1), lb_stripped(2):ub_stripped(2))
            real(kind=GP), intent(inout) :: &
                b_bpar_eq(lb(1):ub(1), lb_stripped(2):ub_stripped(2))
        end subroutine
    end interface

    type, abstract, public, extends(op_mms_source_t) :: &
                                                op_mms_source_ohms_law_t
        !! Base operator to calculate the MMS source of Ohm's law
    contains
        procedure(apply_interface_ohms_law), deferred :: apply
    end type

    abstract interface
        subroutine apply_interface_ohms_law(this, t, lb, ub, lb_stripped, &
                                            ub_stripped, b_ohms_law)
            !! Applies the operator by adding the source term to b_ohms_law
            import op_mms_source_ohms_law_t, GP
            class(op_mms_source_ohms_law_t), intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: t
            !! Simulation time
            integer, intent(in) :: lb(2)
            !! Lower boundary with ghost cells
            integer, intent(in) :: ub(2)
            !! Upper boundary with ghost cells
            integer, intent(in) :: lb_stripped(2)
            !! Lower boundary without ghost cells
            integer, intent(in) :: ub_stripped(2)
            !! Upper boundary without ghost cells
            real(kind=GP), intent(inout) :: &
                b_ohms_law(lb(1):ub(1), lb_stripped(2):ub_stripped(2))
            !! Right hand side array of Ohm's law
        end subroutine
    end interface

    type, public, extends(op_mms_source_ohms_law_t) :: &
                                                op_mms_source_ohms_law_cpu_t
        !! Operator to calculate the MMS source of Ohm's law
        !! for the grid approach on the CPU
    contains
        procedure :: initialize => initialize_ohms_law_cpu
        procedure :: apply => apply_ohms_law_cpu
    end type

    interface
        module subroutine initialize_ohms_law_cpu(this, mesh, bsg_op)
            class(op_mms_source_ohms_law_cpu_t), intent(inout) :: this
            class(mesh_5d_t), target, intent(in) :: mesh
            class(bsg_operators_t), target, intent(in) :: bsg_op
        end subroutine

        module subroutine apply_ohms_law_cpu(this, t, lb, ub, lb_stripped, &
                                             ub_stripped, b_ohms_law)
            class(op_mms_source_ohms_law_cpu_t), intent(inout) :: this
            real(kind=GP), intent(in) :: t
            integer, intent(in) :: lb(2), ub(2), lb_stripped(2), ub_stripped(2)
            real(kind=GP), intent(inout) :: &
                b_ohms_law(lb(1):ub(1), lb_stripped(2):ub_stripped(2))
        end subroutine
    end interface

end module
