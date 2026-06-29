module state_vector_m
    use genex_fortran_env_m, only: GP
    use dcomm_handler_m, only: dcomm_handler_t
    use data_storage_m, only: data_storage_5d_t, data_storage_4d_t, &
                              data_storage_2d_t, &
                              data_storage_cpu_5d_t, data_storage_cpu_4d_t, &
                              data_storage_cpu_2d_t
    use data_array_m, only: data_array_2d_t, data_array_4d_t, data_array_5d_t
    use op_axpy_m, only: op_axpy_t, op_axpy_cpu_t
    use op_set_uniform_m, only: op_set_uniform_t, op_set_uniform_cpu_t
    use op_lin_comb_m, only: op_lin_comb_t, op_lin_comb_cpu_t
    use op_copy_m, only: op_copy_t, op_copy_cpu_t
    use op_set_initial_condition_m, only: op_set_initial_condition_base_t
    use op_neut_set_initial_m, only: op_neut_set_initial_t
    use op_rhs_vlasov_eq_static_m, only: op_rhs_vlasov_eq_static_base_t
    use op_rhs_vlasov_eq_dynamic_m, only: op_rhs_vlasov_eq_dynamic_base_t
    use op_diag_mom_0d_m, only: op_diag_mom_0d_base_t
    use op_diag_mom_2d_m, only: op_diag_mom_2d_base_t
    use op_solve_qn_eq_m, only: op_solve_qn_eq_t
    use op_solve_amps_law_m, only: op_solve_amps_law_t
    use op_solve_bpar_eq_m, only: op_solve_bpar_eq_t
    use op_solve_ohms_law_m, only: op_solve_ohms_law_t
    use op_mom_maxwells_eq_m, only: op_mom_maxwells_eq_base_t
    use op_mom_ohms_law_m, only: op_mom_ohms_law_base_t
    use op_source_m, only: op_source_t
    use op_mms_source_m, only: op_mms_source_vlasov_eq_t, &
                               op_mms_source_maxwells_eq_t, &
                               op_mms_source_ohms_law_t
    use op_coll_m, only: op_coll_base_t
    use op_mom_coll_m, only: op_mom_coll_base_t
    use op_neut_mom_m, only: op_neut_mom_t
    use op_neut_evolve_m, only: op_neut_evolve_t
    use op_neut_coupling_np_m, only: op_neut_coupling_np_t
    use op_neut_coupling_pn_m, only: op_neut_coupling_pn_t
    use op_mms_error_m, only: op_mms_error_t
    use op_mms_solution_m, only: op_mms_solution_vlasov_eq_t, &
                                 op_mms_solution_maxwells_eq_t, &
                                 op_mms_solution_ohms_law_t
    use op_bnd_cond_m, only: op_bnd_cond_base_t
    use diag_files_m, only: mom_0d_file_t, mom_2d_file_t, em_fields_file_t, &
                            neutrals_file_t, &
                            checkpoint_file_t, checkpoint_neut_file_t
#ifdef ENABLE_GPU
    use op_axpy_m, only: op_axpy_gpu_t
    use op_copy_m, only: op_copy_gpu_t
    use op_lin_comb_m, only: op_lin_comb_gpu_t
    use op_set_uniform_m, only: op_set_uniform_gpu_t
    use data_storage_m, only: data_storage_gpu_5d_t, data_storage_gpu_4d_t, &
                              data_storage_gpu_2d_t
#endif
    implicit none

    type, public :: state_vector_t
        !! Type representing the current state of the partial differential
        !! equation
        type(dcomm_handler_t), pointer, private :: dcomm_handler
        !! Comm handler for MPI communication
        class(data_storage_5d_t), public, allocatable :: dist_func
        !! Distribution function
        !! TODO: make this private as soon as we have diagnostics
        class(data_storage_2d_t), private, allocatable :: es_pot
        !! Electrostatic potential
        class(data_storage_2d_t), private, allocatable :: E_par
        !! Induced parallel electric field
        class(data_storage_2d_t), private, allocatable :: A_par
        !! Parallal vector potential
        class(data_storage_2d_t), private, allocatable :: B_par
        !! Parallel magnetic field
        class(data_storage_4d_t), private, allocatable :: neutrals_state
        !! Neutrals fluid moments
        class(op_axpy_t), allocatable, private :: op_add
        !! Operator to perform the add operation
        class(op_lin_comb_t), allocatable, private :: op_lin_comb
        !! Operator to calculate a linear combination of state vectors
        class(op_copy_t), allocatable, private :: op_copy
        !! Operator to copy state vectors
        class(op_set_uniform_t), allocatable, private :: op_set_uniform
        !! Operator to set uniform value to state vector
        type(data_array_2d_t), private, allocatable :: co_qn_eq
        !! Buffer for the co array of the quasi-neutrality equation
        type(data_array_2d_t), private, allocatable :: b_qn_eq
        !! Buffer for the b array of the quasi-neutrality equation
        type(data_array_2d_t), private, allocatable :: b_amps_law
        !! Buffer for the b array of Ampere's law
        type(data_array_2d_t), private, allocatable :: b_bpar_eq
        !! Buffer for the b array of the B parallel equation
        type(data_array_2d_t), private, allocatable :: b_ohms_law
        !! Buffer for the b array of Ohm's law
        type(data_array_2d_t), private, allocatable :: lambda_ohms_law
        !! Buffer for the lambda field of Ohm's law
        type(data_array_2d_t), private, allocatable :: mom_0d
        !! Buffer for the mom_0d diagnostic
        type(data_array_4d_t), private, allocatable :: mom_2d
        !! Buffer for the mom_2d diagnostic
        type(data_array_4d_t), private, allocatable :: mom_2d_tpc
        !! Buffer for the mom_2d diagnostic (TPC)
        type(data_array_4d_t), private, allocatable :: mom_coll
        !! Buffer for the moments used in the collision operator
        type(data_array_4d_t), private, allocatable :: neutrals_mom
        !! Buffer for the moments of the plasma distribution, used by the
        !! neutrals operators
        type(data_array_4d_t), private, allocatable :: neutrals_sources
        !! Buffer for the sources representing the plasma to neutrals
        !! interactions

        ! NOTE: For the MMS tests we need to decouple parts of the equation
        !       by setting the exact MMS solution to the dist func and the
        !       em fields. This will overwrite the calculated solution. To
        !       still evaluate the MMS error we thus save the calculated
        !       solutions to buffers for later use.
        type(data_array_5d_t), private, allocatable :: mms_dist_func_buffer
        !! Buffer for the distribution function in MMS tests
        type(data_array_2d_t), private, allocatable :: mms_es_pot_buffer
        !! Buffer for the electrostatic potential in MMS tests
        type(data_array_2d_t), private, allocatable :: mms_E_par_buffer
        !! Buffer for the parallel electric field in MMS tests
        type(data_array_2d_t), private, allocatable :: mms_A_par_buffer
        !! Buffer for the parallal vector potential in MMS tests
        type(data_array_2d_t), private, allocatable :: mms_B_par_buffer
        !! Buffer for the parallel magnetic field in MMS tests

    contains
        procedure, public :: initialize
        procedure, public :: add
        procedure, public :: copy
        procedure, public :: lin_comb
        procedure, public :: set_uniform
        procedure, public :: exchange
        procedure, public :: exchange_neut
        procedure, public :: start_exchange
        procedure, public :: start_exchange_phi
        procedure, public :: start_exchange_vp
        procedure, public :: start_exchange_mu
        procedure, public :: finish_exchange_phi
        procedure, public :: finish_exchange_vp
        procedure, public :: finish_exchange_mu
        procedure, public :: save_em_fields
        procedure, public :: save_neutrals
        procedure, public :: save_checkpoint
        procedure, public :: save_checkpoint_neut
        procedure, public :: read_checkpoint
        procedure, public :: read_checkpoint_neut
        procedure, public :: buffer_mms_solution_vlasov_eq
        procedure, public :: buffer_mms_solution_maxwells_eq
        procedure, public :: buffer_mms_solution_ohms_law
        procedure, private :: apply_diag_mom_0d
        procedure, private :: apply_diag_mom_2d
        procedure, private :: apply_set_initial_condition
        procedure, private :: apply_set_initial_neutrals
        procedure, private :: apply_rhs_vlasov_eq_static
        procedure, private :: apply_rhs_vlasov_eq_dynamic
        procedure, private :: apply_solve_qn_eq
        procedure, private :: apply_solve_amps_law
        procedure, private :: apply_solve_bpar_eq
        procedure, private :: apply_solve_ohms_law
        procedure, private :: apply_mom_maxwells_eq
        procedure, private :: apply_mom_ohms_law
        procedure, private :: apply_source
        procedure, private :: apply_mms_source_vlasov_eq
        procedure, private :: apply_mms_source_maxwells_eq
        procedure, private :: apply_mms_source_ohms_law
        procedure, private :: apply_mms_solution_vlasov_eq
        procedure, private :: apply_mms_solution_maxwells_eq
        procedure, private :: apply_mms_solution_ohms_law
        procedure, private :: apply_mms_error
        procedure, private :: apply_bnd_cond
        procedure, private :: apply_coll
        procedure, private :: apply_mom_coll
        procedure, private :: apply_neutrals_mom
        procedure, private :: apply_neutrals_coupling_pn
        procedure, private :: apply_neutrals_coupling_np
        procedure, private :: apply_neutrals_evolve
        generic, public :: apply => apply_diag_mom_0d, &
                                    apply_diag_mom_2d, &
                                    apply_set_initial_condition, &
                                    apply_set_initial_neutrals, &
                                    apply_rhs_vlasov_eq_static, &
                                    apply_rhs_vlasov_eq_dynamic, &
                                    apply_solve_qn_eq, &
                                    apply_solve_amps_law, &
                                    apply_solve_bpar_eq, &
                                    apply_solve_ohms_law, &
                                    apply_mom_maxwells_eq, &
                                    apply_mom_ohms_law, &
                                    apply_source, &
                                    apply_mms_source_vlasov_eq, &
                                    apply_mms_source_maxwells_eq, &
                                    apply_mms_source_ohms_law, &
                                    apply_mms_solution_vlasov_eq, &
                                    apply_mms_solution_maxwells_eq, &
                                    apply_mms_solution_ohms_law, &
                                    apply_mms_error, &
                                    apply_bnd_cond, &
                                    apply_coll, &
                                    apply_mom_coll, &
                                    apply_neutrals_mom, &
                                    apply_neutrals_coupling_pn, &
                                    apply_neutrals_coupling_np, &
                                    apply_neutrals_evolve
        procedure, public :: update_device_dist_func
        procedure, public :: update_host_dist_func
        procedure, public :: update_device_em_fields
        procedure, public :: update_host_em_fields
        procedure, public :: update_device_E_par
        procedure, public :: update_host_E_par
        procedure, public :: update_device_maxwells_buffers
        procedure, public :: update_device_maxwells_buffers_bpar_only
        procedure, public :: update_host_maxwells_buffers
        procedure, public :: update_device_ohms_buffers
        procedure, public :: update_host_ohms_buffers
    end type

    interface

        module subroutine initialize(this, dcomm_handler)
            !! Initializes the state_vector_t type
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            type(dcomm_handler_t), pointer, intent(in) :: dcomm_handler
            !! MPI communication handler
        end subroutine initialize

        module subroutine add(this, alpha1, s1)
            !! Adds a scalar multiple of another state vector to the given
            !! instance
            !! this = this+alpha1*s1
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: alpha1
            !! Scalar
            class(state_vector_t), target, intent(inout) :: s1
            !! State vector to add to current instance
        end subroutine add

        module subroutine lin_comb(this, alpha1, s1, alpha2, s2)
            !! Calculates the linear combination
            !! res = alpha1 * s1 + alpha2 * s2
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: alpha1
            !! First scalar
            class(state_vector_t), target, intent(inout) :: s1
            !! First vector
            real(kind=GP), intent(in) :: alpha2
            !! Second scalar
            class(state_vector_t), target, intent(inout) :: s2
            !! Second vector
        end subroutine

        module subroutine set_uniform(this, input)
            !! Set uniform values to the plasma distribution function and the
            !! neutrals state (if present)
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            real(kind=GP), intent(in) :: input
            !! Input value to be set
        end subroutine

        module subroutine copy(this, s1)
            !! Copies the values of s1 to the current instance
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(state_vector_t), target, intent(inout) :: s1
            !! State vector to copy
        end subroutine copy

        module subroutine exchange(this)
            !! Performs the ghost boundary exchange over all dimensions and
            !! returns as soon as the exchange is finished.
            class(state_vector_t), intent(inout) :: this
            !! Instance of the type
        end subroutine exchange

        module subroutine exchange_neut(this)
            !! Performs the ghost boundary exchange for the neutrals state and
            !! returns as soon as the exchange is finished.
            class(state_vector_t), intent(inout) :: this
            !! Instance of the type
        end subroutine exchange_neut

        module subroutine start_exchange(this)
            !! Starts the exchange of ghost boundary cells over all dimensions
            !! and returns immediately.
            class(state_vector_t), intent(inout) :: this
            !! Instance of the type
        end subroutine start_exchange

        module subroutine start_exchange_phi(this)
            !! Starts the exchange of ghost boundary cells over the
            !! phi dimension and returns immediately.
            class(state_vector_t), intent(inout) :: this
            !! Instance of the type
        end subroutine start_exchange_phi

        module subroutine start_exchange_vp(this)
            !! Starts the exchange of ghost boundary cells over the
            !! vp dimension and returns immediately.
            class(state_vector_t), intent(inout) :: this
            !! Instance of the type
        end subroutine start_exchange_vp

        module subroutine start_exchange_mu(this)
            !! Starts the exchange of ghost boundary cells over the
            !! mu dimension and returns immediately.
            class(state_vector_t), intent(inout) :: this
            !! Instance of the type
        end subroutine start_exchange_mu

        module subroutine finish_exchange_phi(this)
            !! Finishes the ghost boundary exchange over the phi dimension and
            !! returns as soon as the exchange is finished.
            class(state_vector_t), intent(inout) :: this
            !! Instance of the type
        end subroutine finish_exchange_phi

        module subroutine finish_exchange_vp(this)
            !! Finishes the ghost boundary exchange over the vp dimension and
            !! returns as soon as the exchange is finished.
            class(state_vector_t), intent(inout) :: this
            !! Instance of the type
        end subroutine finish_exchange_vp

        module subroutine finish_exchange_mu(this)
            !! Finishes the ghost boundary exchange over the mu dimension and
            !! returns as soon as the exchange is finished.
            class(state_vector_t), intent(inout) :: this
            !! Instance of the type
        end subroutine finish_exchange_mu

        module subroutine apply_diag_mom_0d(this, op, diag_file, t)
            !! Applies the diag_mom_0d operator to the state_vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_diag_mom_0d_base_t), intent(inout) :: op
            !! Diagnostic request to apply
            type(mom_0d_file_t), intent(inout) :: diag_file
            !! Diag file to write to
            real(kind=GP), intent(in) :: t
            !! Current simulation time
        end subroutine

        module subroutine apply_diag_mom_2d(this, op, diag_file, t, &
                                            diagnose_tpc)
            !! Applies the diag_mom_2d operator to the state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_diag_mom_2d_base_t), intent(inout) :: op
            !! Diagnostic request to apply
            type(mom_2d_file_t), intent(inout) :: diag_file
            !! Diag file to write to
            real(kind=GP), intent(in) :: t
            !! Current simulation time
            logical, optional, intent(in) :: diagnose_tpc
            !! Switch to calculate TPC only
        end subroutine

        module subroutine save_neutrals(this, diag_file, t)
            !! Saves the neutrals state in the given diagnostic file
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            type(neutrals_file_t), intent(inout) :: diag_file
            !! Diag file to write to
            real(kind=GP), intent(in) :: t
            !! Current simulation time
        end subroutine

        module subroutine save_em_fields(this, diag_file, t)
            !! Saves the electromagnetic potential in the given diagnostic file
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            type(em_fields_file_t), intent(inout) :: diag_file
            !! Diag file to write to
            real(kind=GP), intent(in) :: t
            !! Current simulation time
        end subroutine

        module subroutine save_checkpoint(this, diag_file, t)
            !! Saves the distribution function in the given diagnostic file
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            type(checkpoint_file_t), intent(inout) :: diag_file
            !! Diag file to write to
            real(kind=GP), intent(in) :: t
            !! Current simulation time
        end subroutine

        module subroutine save_checkpoint_neut(this, diag_file, t)
            !! Saves the neutrals state in the given diagnostic file
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            type(checkpoint_neut_file_t), intent(inout) :: diag_file
            !! Diag file to write to
            real(kind=GP), intent(in) :: t
            !! Current simulation time
        end subroutine

        module subroutine read_checkpoint(this, diag_file, t)
            !! Reads the distribution function in the given diagnostic file
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            type(checkpoint_file_t), intent(inout) :: diag_file
            !! Diag file to read from
            real(kind=GP), intent(out) :: t
            !! Current simulation time
        end subroutine

        module subroutine read_checkpoint_neut(this, diag_file, t)
            !! Reads the neutrals state in the given diagnostic file
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            type(checkpoint_neut_file_t), intent(inout) :: diag_file
            !! Diag file to read from
            real(kind=GP), intent(out) :: t
            !! Current simulation time
        end subroutine

        module subroutine apply_set_initial_condition(this, op)
            !! Set initial condition to the distribution functions
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_set_initial_condition_base_t), intent(inout) :: op
            !! Set profile operator
        end subroutine

        module subroutine apply_rhs_vlasov_eq_static(this, op, s_out)
            !! Applies the static RHS operator to the state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_rhs_vlasov_eq_static_base_t), intent(inout) :: op
            !! Operator to apply
            class(state_vector_t), target, intent(inout) :: s_out
            !! State vector to which the result of the operation should be
            !! written to
        end subroutine

        module subroutine apply_rhs_vlasov_eq_dynamic(this, op, s_out)
            !! Applies the dynamic RHS operator to the state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_rhs_vlasov_eq_dynamic_base_t), intent(inout) :: op
            !! Operator to apply
            class(state_vector_t), target, intent(inout) :: s_out
            !! State vector to which the result of the operation should be
            !! written to
        end subroutine

        module subroutine apply_solve_qn_eq(this, op, residuum)
            !! Applies the solve quasi-neutrality equation operator to the
            !! state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_solve_qn_eq_t), intent(inout) :: op
            !! Operator to apply
            real(kind=GP), intent(out) :: residuum
            !! Residuum of the field solve operator
        end subroutine

        module subroutine apply_solve_amps_law(this, op, residuum)
            !! Applies the solve Ampere's law operator to the state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_solve_amps_law_t), intent(inout) :: op
            !! Operator to apply
            real(kind=GP), intent(out) :: residuum
            !! Residuum of the field solve operator
        end subroutine

        module subroutine apply_solve_bpar_eq(this, op)
            !! Applies the solve B parallel equation operator
            !! to the state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_solve_bpar_eq_t), intent(inout) :: op
            !! Operator to apply
        end subroutine

        module subroutine apply_solve_ohms_law(this, op, residuum)
            !! Applies the solve Ohm's law operator to the state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_solve_ohms_law_t), intent(inout) :: op
            !! Operator to apply
            real(kind=GP), intent(out) :: residuum
            !! Residuum of the field solve operator
        end subroutine

        module subroutine apply_mom_maxwells_eq(this, op)
            !! Applies the operator to calculate moments for the Maxwell's
            !! equation to the state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_mom_maxwells_eq_base_t), intent(inout) :: op
            !! Operator to apply
        end subroutine

        module subroutine apply_mom_ohms_law(this, op, ds_in)
            !! Applies the operator to calculate moments for Ohm's law to the
            !! state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_mom_ohms_law_base_t), intent(inout) :: op
            !! Operator to apply
            class(state_vector_t), target, intent(inout) :: ds_in
            !! State vector that contains the time derivative of the
            !! distribution function
        end subroutine

        module subroutine apply_mom_coll(this, op, s_out)
            !! Applies collisional moments operator to the state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_mom_coll_base_t), intent(inout) :: op
            !! Operator to apply
            class(state_vector_t), target, intent(inout) :: s_out
            !! State vector to which the result of the operation should be
            !! written to
        end subroutine

        module subroutine apply_coll(this, op, s_out)
            !! Applies collision operator to the state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_coll_base_t), intent(inout) :: op
            !! Operator to apply
            class(state_vector_t), target, intent(inout) :: s_out
            !! State vector to which the result of the operation should be
            !! written to
        end subroutine

        module subroutine apply_bnd_cond(this, op, t)
            !! Applies the boundary condition operator to the state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_bnd_cond_base_t), intent(inout) :: op
            !! Operator to apply
            real(kind=GP), intent(in) :: t
            !! Current simulation time
        end subroutine

        module subroutine apply_source(this, op)
            !! Applies the source operator to the distribution function in the
            !! state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_source_t), intent(inout) :: op
            !! Operator to apply
        end subroutine

        module subroutine apply_set_initial_neutrals(this, op)
            !! Set initial condition to the neutrals states
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_neut_set_initial_t), intent(inout) :: op
            !! Set initial condition of neutrals (operator)
        end subroutine

        module subroutine apply_neutrals_mom(this, op, s_out)
            !! Applies neutrals moments operator to the state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_neut_mom_t), intent(inout) :: op
            !! Operator to apply
            class(state_vector_t), target, intent(inout) :: s_out
            !! State vector to which the result of the operation should be
            !! written to
        end subroutine

        module subroutine apply_neutrals_coupling_pn(this, op, s_out)
            !! Applies neutrals coupling (plasma to neutrals) operator to
            !! the state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_neut_coupling_pn_t), intent(inout) :: op
            !! Operator to apply
            class(state_vector_t), target, intent(inout) :: s_out
            !! State vector to which the result of the operation should be
            !! written to
        end subroutine

        module subroutine apply_neutrals_coupling_np(this, op, s_out)
            !! Applies neutrals coupling (neutrals to plasma) operator to
            !! the state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_neut_coupling_np_t), intent(inout) :: op
            !! Operator to apply
            class(state_vector_t), target, intent(inout) :: s_out
            !! State vector to which the result of the operation should be
            !! written to
        end subroutine

        module subroutine apply_neutrals_evolve(this, op, s_out)
            !! Applies neutrals evolution operator to the state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_neut_evolve_t), intent(inout) :: op
            !! Operator to apply
            class(state_vector_t), target, intent(inout) :: s_out
            !! State vector to which the result of the operation should be
            !! written to
        end subroutine

        module subroutine apply_mms_source_vlasov_eq(this, op, t)
            !! Applies the MMS source operator for the Vlasov equation to the
            !! distribution function in the state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_mms_source_vlasov_eq_t), intent(inout) :: op
            !! Operator to apply
            real(kind=GP), intent(in) :: t
            !! Current simulation time
        end subroutine

        module subroutine apply_mms_source_maxwells_eq(this, op, t)
            !! Applies the MMS source operator for the quasi neutrality
            !! equation to the charge density in the state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_mms_source_maxwells_eq_t), intent(inout) :: op
            !! Operator to apply
            real(kind=GP), intent(in) :: t
            !! Current simulation time
        end subroutine

        module subroutine apply_mms_source_ohms_law(this, op, t)
            !! Applies the MMS source operator for Ohm's law to the
            !! right hand side of the Ohm's law in the state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_mms_source_ohms_law_t), intent(inout) :: op
            !! Operator to apply
            real(kind=GP), intent(in) :: t
            !! Current simulation time
        end subroutine

        module subroutine apply_mms_solution_vlasov_eq(this, op, t)
            !! Applies the MMS solution operator for the Vlasov equation
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_mms_solution_vlasov_eq_t), intent(inout) :: op
            !! Operator to apply
            real(kind = GP), intent(in) :: t
            !! Current simulation time
        end subroutine

        module subroutine apply_mms_solution_maxwells_eq(this, op, t)
            !! Applies the MMS solution operator for Maxwell's equation
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_mms_solution_maxwells_eq_t), intent(inout) :: op
            !! Operator to apply
            real(kind=GP), intent(in) :: t
            !! Current simulation time
        end subroutine

        module subroutine apply_mms_solution_ohms_law(this, op, t)
            !! Applies the MMS solution operator for Ohm's law
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_mms_solution_ohms_law_t), intent(inout) :: op
            !! Operator to apply
            real(kind=GP), intent(in) :: t
            !! Current simulation time
        end subroutine

        module subroutine apply_mms_error(this, op, t)
            !! Applies the MMS source operator to the state vector
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            class(op_mms_error_t), intent(inout) :: op
            !! Operator to apply
            real(kind=GP), intent(in) :: t
            !! Current simulation time
        end subroutine

        module subroutine buffer_mms_solution_vlasov_eq(this, store_to_buffer)
            !! Stores/loads the distribution function into/from a buffer array
            !! for later use
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            logical, intent(in) :: store_to_buffer
            !! If true, dist_func is stored. Otherwise it is loaded.
        end subroutine

        module subroutine buffer_mms_solution_maxwells_eq(this, store_to_buffer)
            !! Stores/loads es_pot, A_par and B_par into/from a buffer array
            !! for later use
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            logical, intent(in) :: store_to_buffer
            !! If true, dist_func is stored. Otherwise it is loaded.
        end subroutine

        module subroutine buffer_mms_solution_ohms_law(this, store_to_buffer)
            !! Stores/loads E_par into/from a buffer array for later use
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
            logical, intent(in) :: store_to_buffer
            !! If true, dist_func is stored. Otherwise it is loaded.
        end subroutine

        module subroutine update_host_dist_func(this)
            !! Updates the distribution function on CPU from GPU.
            !! This does nothing if no specific GPU backend is set.
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine update_device_dist_func(this)
            !! Updates the distribution function on GPU from CPU.
            !! This does nothing if no specific GPU backend is set.
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine update_host_em_fields(this)
            !! Updates the electromagnetic potential on CPU from GPU.
            !! This does nothing if no specific GPU backend is set.
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine update_device_em_fields(this)
            !! Updates the electromagnetic potential on GPU from CPU.
            !! This does nothing if no specific GPU backend is set.
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine update_host_maxwells_buffers(this)
            !! Updates the buffers related to Maxwell's eq on CPU from GPU.
            !! This does nothing if no specific GPU backend is set.
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine update_device_E_par(this)
            !! Updates the parallel electric field on GPU from CPU.
            !! This does nothing if no specific GPU backend is set.
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine update_host_E_par(this)
            !! Updates the parallel electric field on CPU from GPU.
            !! This does nothing if no specific GPU backend is set.
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine update_device_maxwells_buffers(this)
            !! Updates the buffers related to Maxwell's eq on GPU from CPU.
            !! This does nothing if no specific GPU backend is set.
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine update_device_maxwells_buffers_bpar_only(this)
            !! Updates the buffers related to Maxwell's eq on GPU from CPU,
            !! but only the buffer related to B_par.
            !! This does nothing if no specific GPU backend is set.
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine update_host_ohms_buffers(this)
            !! Updates the buffers related to Ohms's law on CPU from GPU.
            !! This does nothing if no specific GPU backend is set.
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

        module subroutine update_device_ohms_buffers(this)
            !! Updates the buffers related to Ohms's law on GPU from CPU.
            !! This does nothing if no specific GPU backend is set.
            class(state_vector_t), target, intent(inout) :: this
            !! Instance of the type
        end subroutine

    end interface

end module state_vector_m
