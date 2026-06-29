module op_solve_bpar_eq_m
    !! Module defining the solve of the B parallel equation
    use genex_fortran_env_m, only: GP
    use data_array_m, only: data_array_2d_t
    use op_base_m, only: op_base_t
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t

    implicit none
    private

    type, public, extends(op_base_t) :: op_solve_bpar_eq_t
        !! Operator to solve the B parallel equation on the CPU and GPU
        !! NOTE: Since B_par has an explicit solution we do not need to use the
        !! Helmholtz solver nor have a residuum
        type(dcomm_handler_t), private, pointer :: dcomm_handler
        !! MPI communication handler
        class(mesh_5d_t), private, pointer :: mesh
        !! Pointer to the mesh
    contains
        procedure, public :: initialize
        procedure, public :: apply
    end type op_solve_bpar_eq_t

    interface
        module subroutine initialize(this, dcomm_handler, mesh)
            !! Initializes the type
            class(op_solve_bpar_eq_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), target, intent(inout) :: dcomm_handler
            !! Comm handler for MPI communication
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
        end subroutine

        module subroutine apply(this, da_b_bpar_eq, da_B_par)
            !! Solves the B parallel equation for the given
            !! right hand side b. Solution will be put into B_par.
            class(op_solve_bpar_eq_t), target, intent(inout) :: this
            !! Instance of the type
            class(data_array_2d_t), intent(in) :: da_b_bpar_eq
            !! Right hand side array of the problem
            class(data_array_2d_t), intent(inout) :: da_B_par
            !! Parallel magnetic field fluctuations
        end subroutine
    end interface

end module
