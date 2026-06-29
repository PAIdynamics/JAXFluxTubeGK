module timestep_m
    use genex_status_codes_m, only: GENEX_SUCCESS
    use genex_fortran_env_m, only: GP, GP_EPS
    use math_m, only: almost_equal
    use genex_error_handling_m, only: handle_error, error_info_t
    use profiler_m, only: profiler_start, profiler_stop, profiler_set_path
    use params_time_loop_m, only: get_dt, get_n_timesteps
    use dcomm_handler_m, only: dcomm_handler_t
    use mesh_5d_m, only: mesh_5d_t
    use state_vector_m, only: state_vector_t
    use diag_files_m, only: mom_0d_file_t, mom_2d_file_t, em_fields_file_t, &
                            neutrals_file_t, &
                            checkpoint_file_t, checkpoint_neut_file_t
    use profile_container_m, only: profile_container_t
    use dist_initial_container_m, only: dist_initial_container_t
    use op_set_initial_condition_m, only: op_set_initial_condition_base_t
    use op_neut_set_initial_m, only: op_neut_set_initial_t
    use op_diag_mom_0d_m, only: op_diag_mom_0d_base_t
    use op_diag_mom_2d_m, only: op_diag_mom_2d_base_t
    use op_rhs_vlasov_eq_static_m, only: op_rhs_vlasov_eq_static_base_t
    use op_rhs_vlasov_eq_dynamic_m, only: op_rhs_vlasov_eq_dynamic_base_t
    use op_mom_maxwells_eq_m, only: op_mom_maxwells_eq_base_t
    use op_mom_ohms_law_m, only: op_mom_ohms_law_base_t
    use op_solve_qn_eq_m, only: op_solve_qn_eq_t
    use op_solve_amps_law_m, only: op_solve_amps_law_t
    use op_solve_bpar_eq_m, only: op_solve_bpar_eq_t
    use op_solve_ohms_law_m, only: op_solve_ohms_law_t
    use op_mms_source_m, only: op_mms_source_vlasov_eq_t, &
                               op_mms_source_maxwells_eq_t, &
                               op_mms_source_ohms_law_t
    use op_source_container_m, only: op_source_container_t
    use op_mms_error_m, only: op_mms_error_t
    use op_mms_solution_m, only: op_mms_solution_vlasov_eq_t, &
                                 op_mms_solution_maxwells_eq_t, &
                                 op_mms_solution_ohms_law_t
    use op_bnd_cond_m, only: op_bnd_cond_base_t
    use op_coll_m, only: op_coll_base_t
    use op_mom_coll_m, only: op_mom_coll_base_t
    use op_neut_mom_m, only: op_neut_mom_t
    use op_neut_evolve_m, only: op_neut_evolve_t
    use op_neut_coupling_np_m, only: op_neut_coupling_np_t
    use op_neut_coupling_pn_m, only: op_neut_coupling_pn_t
    use bsg_operators_m, only: bsg_operators_t

    implicit none

    type, public, abstract :: timestep_t
        type(dcomm_handler_t), pointer, private :: dcomm_handler
        !! MPI communication handler
        type(mesh_5d_t), pointer, private :: mesh
        !! Pointer to the mesh
        class(bsg_operators_t), private, allocatable :: bsg_op
        !! BSG operator
        type(state_vector_t), private, allocatable :: state
        !! Current state of the system
        class(op_diag_mom_0d_base_t), private, allocatable :: op_mom_0d
        !! Diagnostic operator to calculate the 0d moments of the state
        !! vector
        class(op_diag_mom_2d_base_t), private, allocatable :: op_mom_2d
        !! Diagnostic operator to calculate the 2d moments of the state
        !! vector
        class(op_set_initial_condition_base_t), private, allocatable :: &
                                                        op_set_initial_condition
        !! Operator to set 5d datatypes to a given profile
        class(op_neut_set_initial_t), private, allocatable :: &
                                                        op_neut_set_initial
        !! Operator to set the initial state of neutrals
        class(op_rhs_vlasov_eq_static_base_t), private, allocatable :: &
                                                        op_rhs_vlasov_eq_static
        !! Operator to calculate parts of the RHS of the Vlasov equation
        !! excluding explicit time derivatives
        class(op_rhs_vlasov_eq_dynamic_base_t), private, allocatable :: &
                                                        op_rhs_vlasov_eq_dynamic
        !! Operator to calculate the part of the RHS of the Vlasov equation
        !! including explicit time derivatives
        type(op_solve_qn_eq_t), private, allocatable :: op_solve_qn_eq
        !! Operator to solve the quasi-neutrality equation
        type(op_solve_amps_law_t), private, allocatable :: op_solve_amps_law
        !! Operator to solve Ampere's law
        type(op_solve_bpar_eq_t), private, allocatable :: op_solve_bpar_eq
        !! Operator to solve the B parallel equation
        type(op_solve_ohms_law_t), private, allocatable :: op_solve_ohms_law
        !! Operator to solve Ohm's law
        class(op_mom_maxwells_eq_base_t), private, allocatable :: &
                                                             op_mom_maxwells_eq
        !! Operator to calculate the moments of the distribution function
        !! required to solve the quasi-neutrality equation and Ampere's law
        class(op_mom_ohms_law_base_t), private, allocatable :: op_mom_ohms_law
        !! Operator to calculate the moments of the distribution function
        !! required to solve Ohm's law
        type(op_source_container_t), private, allocatable, dimension(:) :: &
                                                             op_source_container
        !! Container of different source operators which apply a source
        !! to the distribution functions
        class(op_mms_source_vlasov_eq_t), private, allocatable :: &
            op_mms_source_vlasov_eq
        !! Operator to calculate the source term of the Vlasov equation for MMS
        class(op_mms_source_maxwells_eq_t), private, allocatable :: &
            op_mms_source_maxwells_eq
        !! Operator to calculate the source term of the Maxwell's equations for
        !! MMS
        class(op_mms_source_ohms_law_t), private, allocatable :: &
            op_mms_source_ohms_law
        !! Operator to calculate the source term of Ohm's law for MMS
        class(op_mms_solution_vlasov_eq_t), private, allocatable :: &
            op_mms_solution_vlasov_eq
        !! Operator to calculate the MMS solution of the Vlasov equation
        class(op_mms_solution_maxwells_eq_t), private, allocatable :: &
            op_mms_solution_maxwells_eq
        !! Operator to calculate the MMS solution of the Maxwell's equations
        class(op_mms_solution_ohms_law_t), private, allocatable :: &
            op_mms_solution_ohms_law
        !! Operator to calculate the MMS solution of Ohm's law
        class(op_mms_error_t), private, allocatable :: op_mms_error
        !! Operator to calculate the L2 and Linfinity error to the exact MMS
        !! solution
        class(op_bnd_cond_base_t), private, allocatable :: op_bnd_cond
        !! Operator to set boundary conditions via the ghost cell method
        class(op_coll_base_t), private, allocatable :: op_coll
        !! Operator to apply collisions to the distribution function
        class(op_mom_coll_base_t), private, allocatable :: op_mom_coll
        !! Operator to calculate moments for the collision operator
        class(op_neut_mom_t), private, allocatable :: op_neut_mom
        !! Operator to calculate moments for the neutrals coupling operator
        class(op_neut_coupling_np_t), private, allocatable :: &
                                                        op_neut_coupling_np
        !! Operator to couple neutrals to plasma
        class(op_neut_coupling_pn_t), private, allocatable :: &
                                                        op_neut_coupling_pn
        !! Operator to couple plasma to neutrals
        class(op_neut_evolve_t), private, allocatable :: op_neut_evolve
        !! Operator to evolve the neutrals
        type(profile_container_t), private, dimension(:), allocatable :: &
                                                        profile_container
        !! Collection of initial profiles
        type(dist_initial_container_t), private, dimension(:), allocatable :: &
             dist_initial_container
        !! Collection of initial distributions
        type(mom_0d_file_t), allocatable :: mom_0d_file
        !! Output file for the mom_0d diagnostic
        type(mom_2d_file_t), allocatable :: mom_2d_file
        !! Output file for the mom_2d diagnostic
        type(mom_2d_file_t), allocatable :: mom_2d_tpc_file
        !! Output file for the mom_2d diagnostic for TPC
        type(neutrals_file_t), allocatable :: neutrals_file
        !! Output file for the neutrals diagnostic
        type(em_fields_file_t), allocatable :: em_fields_file
        !! Output file for the electromagnetic fields
        type(checkpoint_file_t), allocatable :: checkpoint_file
        !! Output file for saving a checkpoint
        type(checkpoint_neut_file_t), allocatable :: checkpoint_neut_file
        !! Output file for saving a neutrals checkpoint
        real(kind=GP), allocatable, dimension(:) :: vp_full
        !! Array containing the velocity space grid including ghost cells with
        !! boundary values
        real(kind=GP), contiguous, pointer, dimension(:) :: vp_full_ptr
        !! Pointer to the velocity space grid including ghost cells
        real(kind=GP) :: dt
        !! Timestep
        real(kind=GP) :: t
        !! Current simulation time
        integer :: current_stage
        !! Keeps track of the current stage number of the multi-stage time
        !! integration. For schemes with a single stage this should be one
        !! always.
        character(len=256) :: diag_dir
        !! Diagnostic directory
        integer :: n_points(5)
        !! Number of grid points
        integer :: n_ghost_cells(5)
        !! Number of MPI ghost cells
        integer :: n_source
        !! Number of sources to apply
        logical :: apply_plasma = .true.
        !! Specifies if collisionless plasma equations should be applied or not
        logical :: apply_collisions = .false.
        !! Specifies if collisions should be applied or not
        logical :: apply_neutrals = .false.
        !! Specifies if neutrals should be applied or not
        logical :: apply_sources = .false.
        !! Specifies if (non-MMS) sources should be applied or not
        logical :: start_from_checkpoint = .false.
        !! Specifies if the calculation is restarted from a checkpoint or not
        logical :: run_mms = .false.
        !! Specifies if MMS tests are run or not
        logical :: run_test = .false.
        !! Specifies if unit tests are run or not
        real(kind=GP) :: test_result_maxw
        !! Variable used to store results in the test routines of this type
        real(kind=GP) :: test_result_ohm
        !! Variable used to store results in the test routines of this type

        procedure(calc_rhs_static_core), pointer :: &
                                    calc_rhs_static => calc_rhs_static_core
        !! Points to the calc_rhs_static subroutine that should be used in the
        !! time evolution
        procedure(calc_rhs_dynamic_core), pointer :: &
                                    calc_rhs_dynamic => calc_rhs_dynamic_core
        !! Points to the calc_rhs_dynamic subroutine that should be used in the
        !! time evolution
        procedure(calc_collisions_core), pointer :: &
                                    calc_collisions => calc_collisions_core
        !! Points to the calc_collisions subroutine that should be used in the
        !! time evolution
        procedure(calc_neutrals_core), pointer :: &
                                    calc_neutrals => calc_neutrals_core
        !! Points to the calc_neutrals subroutine that should be used in the
        !! time evolution
        procedure(apply_bnd_cond_core), pointer :: &
                                    apply_bnd_cond => apply_bnd_cond_core
        !! Points to the apply_bnd_cond subroutine that should be used in the
        !! time evolution
        procedure(solve_maxwells_eq_core), pointer :: &
                                    solve_maxwells_eq => solve_maxwells_eq_core
        !! Points to the solve_maxwells_eq subroutine that should be used in the
        !! time evolution
        procedure(solve_ohms_law_core), pointer :: &
                                    solve_ohms_law => solve_ohms_law_core
        !! Points to the solve_ohms_law subroutine that should be used in the
        !! time evolution
    contains
        procedure, private :: initialize_profiles
        procedure, private :: initialize_dist_initial
        procedure, private :: initialize_diagnostics
        procedure, private :: initialize_operators
        procedure, private :: initialize_operators_grid_cpu
        procedure, private :: initialize_operators_grid_gpu
        procedure, private :: initialize_operators_vspec_cpu
        procedure, private :: initialize_state_vector
        procedure, private :: initialize_parent
        procedure, private :: finalize_diagnostics
        procedure, private :: check_species_config
        procedure, private :: check_neutrals_config
        procedure, private :: exchange
        procedure, private :: exchange_neut
        procedure, private :: exchange_vp_mu
        procedure, private :: calc_rhs_static_core
        procedure, private :: calc_rhs_dynamic_core
        procedure, private :: calc_collisions_core
        procedure, private :: calc_neutrals_core
        procedure, private :: apply_bnd_cond_core
        procedure, private :: solve_maxwells_eq_core
        procedure, private :: solve_ohms_law_core
        procedure, private :: step_evolve
        procedure, private :: step_finish

        procedure(initialize_interface), public, deferred :: initialize
        procedure(step_interface), public, deferred :: step
        procedure, public :: diagnose_mom_0d
        procedure, public :: diagnose_mom_2d
        procedure, public :: diagnose_neutrals
        procedure, public :: diagnose_em_fields
        procedure, public :: diagnose_all
        procedure, public :: create_part => reinitialize_diagnostics
        procedure, public :: save_checkpoint
        procedure, public :: get_state_pointer
        procedure, public :: get_t
        procedure, public :: final_time_reached

        procedure, private :: setup_test_mode
    end type

    abstract interface
        subroutine step_interface(this, ierr)
            !! Calculate one time step
            import timestep_t
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
            integer, intent(out) :: ierr
            !! Error code
        end subroutine

        subroutine initialize_interface(this, dcomm_handler, mesh, dt, &
                                        diag_dir, start_from_checkpoint, &
                                        run_mms, run_test)
            !! Initializes the timestep
            import timestep_t, mesh_5d_t, dcomm_handler_t, GP
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! Communication handler for MPI communication
            type(mesh_5d_t), pointer, intent(in) :: mesh
            !! Mesh
            real(kind=GP), intent(in) :: dt
            !! Timestep
            character(len=*), intent(in) :: diag_dir
            !! Directory for the diagnostic output
            logical, optional, intent(inout) :: start_from_checkpoint
            !! Specifies if the simulation is restarted from a checkpoint
            !! (default=false). Returns the actual value if a restart was
            !! performed or not (if e.g. no file found).
            logical, optional, intent(in) :: run_mms
            !! Specifies if MMS tests should be run instead of conventional
            !! time loop (default=false)
            logical, optional, intent(in) :: run_test
            !! Specifies if tests should be run instead of conventional
            !! time loop (default=false)
        end subroutine
    end interface

    interface
        module subroutine initialize_profiles(this)
            !! Allocates and initializes the profile types
            class(timestep_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_dist_initial(this)
            !! Allocates and initializes the initial distribution types
            class(timestep_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_diagnostics(this)
            !! Allocates and initializes the diagnostic files
            class(timestep_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine finalize_diagnostics(this)
            !! Deallocates the diagnostic files
            class(timestep_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_operators(this)
            !! Allocates and initializes all operators
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_operators_grid_cpu(this)
            !! Allocates and initializes CPU-supported operators
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_operators_grid_gpu(this)
            !! Allocates and initializes GPU-supported operators
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_operators_vspec_cpu(this)
            !! Allocates and initializes CPU-spectral-method-supported operators
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_state_vector(this)
            !! Allocates and initializes the state vector
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine initialize_parent(this, dcomm_handler, mesh)
            !! Initializes the parent timestep_t type
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! MPI communication handler
            type(mesh_5d_t), pointer, intent(in) :: mesh
            !! Pointer to the mesh
        end subroutine

        module subroutine check_species_config(this)
            !! Checks the configuration of species for validity
            class(timestep_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine check_neutrals_config(this)
            !! Checks the configuration of neutrals for validity
            class(timestep_t), intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine exchange(this, s_in)
            !! Triggers data exchange between processes on all dimensions
            !! of the parallelization.
            class(timestep_t), intent(inout) :: this
            !! Instance of the type
            type(state_vector_t), intent(inout) :: s_in
            !! Input state_vector
        end subroutine

        module subroutine exchange_neut(this, s_in)
            !! Triggers data exchange between processes for the neutrals
            !! state.
            class(timestep_t), intent(inout) :: this
            !! Instance of the type
            type(state_vector_t), intent(inout) :: s_in
            !! Input state_vector
        end subroutine

        module subroutine exchange_vp_mu(this, s_in)
            !! Triggers data exchange between processes on the velocity
            !! space dimensions (vp and mu) of the parallelization.
            class(timestep_t), intent(inout) :: this
            !! Instance of the type
            type(state_vector_t), intent(inout) :: s_in
            !! Input state_vector
        end subroutine

        module subroutine calc_rhs_static_core(this, s_in, s_out)
            !! Calculates the parts of the right hand side of the gyrokinetic
            !! Vlasov equation excluding explicit time derivatives
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
            type(state_vector_t), target, intent(inout) :: s_in
            !! Input state_vector
            type(state_vector_t), target, intent(inout) :: s_out
            !! Output state_vector
        end subroutine

        module subroutine calc_rhs_dynamic_core(this, s_in, s_out)
            !! Calculates the parts of the right hand side of the gyrokinetic
            !! Vlasov equation including explicit time derivatives
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
            type(state_vector_t), target, intent(inout) :: s_in
            !! Input state_vector
            type(state_vector_t), target, intent(inout) :: s_out
            !! Output state_vector
        end subroutine

        module subroutine calc_collisions_core(this, s_in, s_out)
            !! Calculates collisional contribution to the gyrokinetic equation
            !! and adds it to the output state vector.
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
            type(state_vector_t), target, intent(inout) :: s_in
            !! Input state_vector
            type(state_vector_t), target, intent(inout) :: s_out
            !! Output state_vector
        end subroutine

        module subroutine calc_neutrals_core(this, s_in, s_out)
            !! Calculates neutrals contribution to the gyrokinetic equation
            !! and adds it to the output state vector.
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
            type(state_vector_t), target, intent(inout) :: s_in
            !! Input state_vector
            type(state_vector_t), target, intent(inout) :: s_out
            !! Output state_vector
        end subroutine

        module subroutine apply_bnd_cond_core(this, s_inout)
            !! Apply the boundary conditions to the distribution functions
            !! and electromagnetic fields.
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
            type(state_vector_t), target, intent(inout) :: s_inout
            !! Input and output state_vector
        end subroutine

        module subroutine solve_maxwells_eq_core(this, s_inout, ierr)
            !! Calculates the electrostatic potential from the gyrokinetic
            !! quasi-neutrality equation, the parallel
            !! electromagnetic vector potential from the parallel gyrokinetic
            !! Ampere's law and parallel magnetic field perturbation from the
            !! perpendicular gyrokinetic Ampere's law.
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
            type(state_vector_t), target, intent(inout) :: s_inout
            !! Input and output state_vector
            integer, intent(out) :: ierr
            !! Error code
        end subroutine

        module subroutine solve_ohms_law_core(this, dsdt_in, s_inout, ierr)
            !! Calculates the induced parallel electric field form the
            !! gyrokinetic Ohm's law
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
            type(state_vector_t), target, intent(inout) :: dsdt_in
            !! State vector containing the time derivative of the part of the
            !! gyrokinetic Vlasov equation excluding explicit time derivatives
            type(state_vector_t), target, intent(inout) :: s_inout
            !! Input and output state_vector
            integer, intent(out) :: ierr
            !! Error code
        end subroutine

        module subroutine step_evolve(this, s_in, s_out, ierr)
            !! Evolves the state with the underlying equations by a single step,
            !! calling the correct order of calc and solve operations.
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
            type(state_vector_t), target, intent(inout) :: s_in
            !! Input state vector
            type(state_vector_t), target, intent(inout) :: s_out
            !! Output state vector
            integer, intent(out) :: ierr
            !! Error code
        end subroutine

        module subroutine step_finish(this, s_inout, s_buffer, ierr)
            !! Finishes the evoution of a single step by calling wrap-up
            !! operations and preparing the execution of the next step.
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
            type(state_vector_t), target, intent(inout) :: s_inout
            !! In- and output state vector that will be updated
            type(state_vector_t), target, intent(inout) :: s_buffer
            !! Buffer state vector used for intermediate result storage
            integer, intent(out) :: ierr
            !! Error code
        end subroutine

        module subroutine reinitialize_diagnostics(this)
            !! Creates a new output part directory and reinitializes
            !! the diagnostics to write into the new part directory
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine diagnose_mom_0d(this)
            !! Executes the mom_0d diagnostics
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine diagnose_mom_2d(this, diagnose_tpc)
            !! Executes the mom_2d diagnostics
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
            logical, intent(in) :: diagnose_tpc
            !! Switch to compute TPC
        end subroutine

        module subroutine diagnose_neutrals(this)
            !! Executes the neutrals diagnostics
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine diagnose_em_fields(this)
            !! Executes the em_fields diagnostics
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine diagnose_all(this, ierr)
            !! Calls all the diagnostics
            class(timestep_t), intent(inout) :: this
            !! Instance of the type
            integer, intent(inout) :: ierr
            !! Error code
        end subroutine

        module subroutine save_checkpoint(this, ierr)
            !! Writes a checkpoint to file
            class(timestep_t), target, intent(inout) :: this
            !! Instance of the type
            integer, intent(inout) :: ierr
            !! Error code
        end subroutine

        module subroutine setup_test_mode(this, run_test)
            !! If run_test is true, redirects the core subroutines evolving the
            !! desired equations to specifically designed helper routines for
            !! testing.
            class(timestep_t), intent(inout) :: this
            !! Instance of the type
            logical, intent(in) :: run_test
            !! Specifies if tests should be run or not
        end subroutine
    end interface

    type, public, extends(timestep_t) :: rk4_t
        !! Type implementing the RK4 time stepping scheme
        type(state_vector_t), private, allocatable :: initial_state, k, &
                                                      increment
        !! Intermediate states for the RK4 algorithm
        real(kind=GP), private :: nodes(4)
        !! Nodes of the RK4 algorithm (c-values in Butcher table)
        real(kind=GP), private :: weights(4)
        !! Weights of the RK4 algorithm (b-values in Butcher table)
    contains
        procedure, public :: step => step_rk4
        procedure, public :: initialize => initialize_rk4
    end type

    interface
        module subroutine initialize_rk4(this, dcomm_handler, mesh, dt, &
                                         diag_dir, start_from_checkpoint, &
                                         run_mms, run_test)
            class(rk4_t), target, intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            type(mesh_5d_t), pointer, intent(in) :: mesh
            real(kind=GP), intent(in) :: dt
            character(len=*), intent(in) :: diag_dir
            logical, optional, intent(inout) :: start_from_checkpoint
            logical, optional, intent(in) :: run_mms
            logical, optional, intent(in) :: run_test
        end subroutine

        module subroutine step_rk4(this, ierr)
            class(rk4_t), target, intent(inout) :: this
            integer, intent(out) :: ierr
        end subroutine
    end interface

    type, public, extends(timestep_t) :: euler_t
        !! Type implementing the Euler time stepping scheme
        type(state_vector_t), private, allocatable :: k
    contains
        procedure, public :: step => step_euler
        procedure, public :: initialize => initialize_euler
    end type

    interface
        module subroutine initialize_euler(this, dcomm_handler, mesh, dt, &
                                           diag_dir, start_from_checkpoint, &
                                           run_mms, run_test)
            class(euler_t), target, intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            type(mesh_5d_t), pointer, intent(in) :: mesh
            real(kind=GP), intent(in) :: dt
            character(len=*), intent(in) :: diag_dir
            logical, optional, intent(inout) :: start_from_checkpoint
            logical, optional, intent(in) :: run_mms
            logical, optional, intent(in) :: run_test
        end subroutine

        module subroutine step_euler(this, ierr)
            class(euler_t), target, intent(inout) :: this
            integer, intent(out) :: ierr
        end subroutine
    end interface

    type, private :: rkc_coefs_t
        !! Container type collecting all required coefficients to perform
        !! an RKC timestep

        integer :: n_stages
        !! Number of RKC stages
        real(kind=GP), dimension(:), allocatable :: c_coef
        !! Coefficient c of the RKC scheme
        real(kind=GP), dimension(:), allocatable :: u_coef
        !! Coefficient u of the RKC scheme
        real(kind=GP), dimension(:), allocatable :: v_coef
        !! Coefficient v of the RKC scheme
        real(kind=GP), dimension(:), allocatable :: utilde_coef
        !! Coefficient u tilde of the RKC scheme
        real(kind=GP), dimension(:), allocatable :: gamma_coef
        !! Coefficient gamma (tilde) of the RKC scheme
    end type

    type, private :: pirock_coefs_t
        !! Container type collecting all required coefficients to perform
        !! a PIROCK timestep

        integer :: n_stages
        !! Number of PIROCK stages
        real(kind=GP), dimension(:), allocatable :: c_coef
        !! Coefficient c (nodes) of the PIROCK scheme
        real(kind=GP), dimension(:), allocatable :: mu_coef
        !! Coefficient mu of the PIROCK scheme
        real(kind=GP), dimension(:), allocatable :: nu_coef
        !! Coefficient nu of the PIROCK scheme
        real(kind=GP), dimension(:), allocatable :: sigma_coef
        !! Coefficient sigma tilde of the PIROCK scheme
        real(kind=GP) :: gamma_coef
        !! Constant gamma
    end type

    type, public, extends(rk4_t) :: strang_splitting_t
        !! Type implementing 2nd order Strang splitting

        type(rkc_coefs_t), private :: rkc_coefs_collisions
        !! RKC coefficients for collisions time step
        type(rkc_coefs_t), private :: rkc_coefs_neutrals
        !! RKC coefficients for neutrals time step
        type(state_vector_t), private, allocatable :: k2, increment2
        !! Additional (to the RK4) intermediate states for the RKC algorithm
    contains
        procedure, public :: step => step_strang
        procedure, public :: initialize => initialize_strang
        procedure, private :: initialize_rkc
        procedure, private :: step_rkc
    end type

    interface
        module subroutine initialize_strang(this, dcomm_handler, mesh, dt, &
                                diag_dir, start_from_checkpoint, run_mms, &
                                run_test)
            class(strang_splitting_t), target, intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            type(mesh_5d_t), pointer, intent(in) :: mesh
            real(kind=GP), intent(in) :: dt
            character(len=*), intent(in) :: diag_dir
            logical, optional, intent(inout) :: start_from_checkpoint
            logical, optional, intent(in) :: run_mms
            logical, optional, intent(in) :: run_test
        end subroutine

        module subroutine step_strang(this, ierr)
            class(strang_splitting_t), target, intent(inout) :: this
            integer, intent(out) :: ierr
        end subroutine

        module subroutine initialize_rkc(this, n_stages, eta, coefs)
            !! Initializes the second order RKC time integrator
            class(strang_splitting_t), target, intent(inout) :: this
            !! Instance of the type
            integer, intent(in) :: n_stages
            !! Number of stages of the RKC scheme
            real(kind=GP), intent(in) :: eta
            !! Damping factor of the RKC scheme
            type(rkc_coefs_t), intent(inout) :: coefs
            !! Coefficients to initialize
        end subroutine

        module subroutine step_rkc(this, coefs, ierr)
            !! Perform an RKC time step
            class(strang_splitting_t), target, intent(inout) :: this
            !! Instance of the type
            type(rkc_coefs_t), intent(in) :: coefs
            !! RKC coefficients
            integer, intent(out) :: ierr
            !! Error code
        end subroutine
    end interface

    type, public, extends(timestep_t) :: pirock_t
        !! Type implementing PIROCK
        type(pirock_coefs_t), private :: pirock_coefs
        !! PIROCK coefficients
        type(state_vector_t), private, allocatable :: initial_state, k, &
                                                      increment, k2, k3, k4
        !! Intermediate states for the PIROCK algorithm
    contains
        procedure, public :: step => step_pirock
        procedure, public :: initialize => initialize_pirock
    end type

    interface
        module subroutine initialize_pirock(this, dcomm_handler, mesh, dt, &
            diag_dir, start_from_checkpoint, run_mms, &
            run_test)
            class(pirock_t), target, intent(inout) :: this
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            type(mesh_5d_t), pointer, intent(in) :: mesh
            real(kind=GP), intent(in) :: dt
            character(len=*), intent(in) :: diag_dir
            logical, optional, intent(inout) :: start_from_checkpoint
            logical, optional, intent(in) :: run_mms
            logical, optional, intent(in) :: run_test
        end subroutine

        module subroutine step_pirock(this, ierr)
            !! Perform a PIROCK time step
            class(pirock_t), target, intent(inout) :: this
            !! Instance of the type
            integer, intent(out) :: ierr
            !! Error code
        end subroutine
    end interface

contains

    real(kind=GP) function get_t(this)
        !! Returns the current simulation time
        class(timestep_t), intent(in) :: this
        !! Instance of the type
        get_t = this%t
    end function

    function get_state_pointer(this) result(ptr)
        !! Returns a pointer to the current state vector.
        !! NOTE: This function breaks encapsulation and may only be used
        !!       for unit tests!
        class(timestep_t), target, intent(inout) :: this
        !! Instance of the type
        type(state_vector_t), pointer :: ptr
        !! Pointer pointing to the state vector of the timestep
        ptr => this%state
    end function

    logical function final_time_reached(this)
        !! Returns true if the final time given by nt * dt has been reached.
        class(timestep_t), intent(in) :: this
        !! Instance of the type
        real(kind=GP) :: dt
        integer :: nt

        dt = get_dt()
        nt = get_n_timesteps()

        ! NOTE: Tolerance has to be a value that is smaller than the smallest
        !       possible timestep in any stage of a time integrator.

        final_time_reached = almost_equal(this%t, nt * dt, 0.01_GP * dt)
    end function

end module
