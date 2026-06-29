module op_solve_amps_law_m
    !! Module defining an operator to solve Amperes's law
    use genex_fortran_env_m, only: GP
    use data_array_m, only: data_array_2d_t
    use op_base_m, only: op_base_t
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t
    ! From PARALLAX
    use helmholtz_solver_m, only: helmholtz_solver_t

    implicit none
    private

    type, public, extends(op_base_t) :: op_solve_amps_law_t
        !! Operator to solve Ampere's law on the CPU and GPU
        type(dcomm_handler_t), private, pointer :: dcomm_handler
        !! MPI communication handler
        class(mesh_5d_t), private, pointer :: mesh
        !! Pointer to the mesh
        class(helmholtz_solver_t), private, allocatable, dimension(:) :: solvers
        !! Array of Helmholtz solver for each poloidal plane
        class(data_array_2d_t), private, allocatable :: da_lambda
        ! Lambda coefficient of the Helmholtz problem
        class(data_array_2d_t), private, allocatable :: da_xi
        ! Xi coefficient of the Helmholtz problem
        class(data_array_2d_t), private, allocatable :: da_co
        ! Co coefficient of the Helmholtz problem
    contains
        procedure, public :: initialize
        procedure, public :: apply
    end type op_solve_amps_law_t

    interface
        module subroutine initialize(this, dcomm_handler, mesh)
            !! Initializes the type
            class(op_solve_amps_law_t), intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), target, intent(inout) :: dcomm_handler
            !! Comm handler for MPI communication
            class(mesh_5d_t), target, intent(inout) :: mesh
            !! Mesh
        end subroutine

        module subroutine apply(this, da_b_amps_law, da_A_par, residuum)
            !! Solves Ampere's law for the given right hand side b and initial
            !! guess A_par. Solution will be put into A_par.
            class(op_solve_amps_law_t), target, intent(inout) :: this
            !! Instance of the type
            class(data_array_2d_t), intent(in) :: da_b_amps_law
            !! Right hand side array of the Helmholtz problem
            ! NOTE: b is needed to be intent(inout) due to the interface
            !       of the solver applied.
            class(data_array_2d_t), intent(inout) :: da_A_par
            !! Parallel vector potential
            real(kind=GP), intent(out) :: residuum
            !! Residuum of the solver
        end subroutine
    end interface

end module
