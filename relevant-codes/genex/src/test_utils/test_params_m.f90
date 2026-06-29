module test_params_m
    ! Module that contains helpers for unit tests to initialize non-default
    ! parameters. This module is intended to be used for unit tests, it should
    ! not be used in the main code.
    use mpi
    use genex_fortran_env_m, only: GP
    use genex_error_handling_m, only: handle_error
    use genex_status_codes_m, only: GENEX_ERR_UTESTS, GENEX_SUCCESS
    use file_handling_m, only: file_exists, delete_file_if_exists
    use test_params_io_m, only: get_test_params_file
    use params_normalization_m, only: read_params_normalization, &
                                      write_params_normalization
    use params_parallelization_m, only: read_params_parallelization, &
                                        write_params_parallelization, &
                                        get_n_procs_phi, get_n_procs_vp, &
                                        get_n_procs_mu, get_n_procs_sp
    use params_species_m, only: read_params_species, &
                                write_params_species, &
                                get_name, get_mass, get_charge, &
                                get_temp_scaling, &
                                n_spec_supported
    use params_species_config_m, only: read_params_species_config, &
                                       write_params_species_config
    use params_neutrals_m, only: read_params_neutrals, &
                                 write_params_neutrals, &
                                 get_neut_name, get_neut_mass, &
                                 get_neut_mapped_ion_name, &
                                 get_n_points_neut, n_neut_supported
    use params_neutrals_config_m, only: read_params_neutrals_config, &
                                        write_params_neutrals_config, &
                                        get_neut_evolve_type, &
                                        get_neut_coupling_type, &
                                        get_neut_temp_floor, &
                                        get_neut_dens_floor, &
                                        get_neut_gamma_u, &
                                        get_neut_gamma_T, &
                                        get_neut_gamma_W
    use params_neutrals_init_m, only: read_params_neutrals_init, &
                                      write_params_neutrals_init, &
                                      get_neut_params_init, &
                                      get_neut_profile_type_dens, &
                                      get_neut_initial_perturbation_dens, &
                                      params_neutrals_init_t
    use params_neutrals_set_blob_m, only: read_params_neutrals_set_blob, &
                                          write_params_neutrals_set_blob
    use params_profile_uniform_m, only: read_params_profile_uniform, &
                                        write_params_profile_uniform
    use params_profile_plateau_m, only: read_params_profile_plateau, &
                                        write_params_profile_plateau
    use params_mesh_m
    use params_bsg_m
    use params_time_loop_m, only: read_params_time_loop, &
                                  write_params_time_loop
    use params_time_split_m, only: read_params_time_split, &
                                   write_params_time_split
    use params_set_blob_m, only: read_params_set_blob, &
                                 write_params_set_blob
    use params_set_mode_m, only: read_params_set_mode, &
                                 write_params_set_mode
    use params_field_solve_m, only: read_params_field_solve, &
                                    write_params_field_solve
    use params_profile_cbc_m, only: read_params_profile_cbc, &
                                    write_params_profile_cbc
    use params_profile_sine_m, only: read_params_profile_sine, &
                                     write_params_profile_sine
    use params_profile_lapd_m, only: read_params_profile_lapd, &
                                     write_params_profile_lapd
    use params_numerical_scheme_m, only: read_params_numerical_scheme, &
                                         write_params_numerical_scheme
    use params_initial_condition_m
    use params_mms_m, only: read_params_mms, write_params_mms
    use params_collisions_m
    use params_gpu_offload_m
    use params_gyrokinetic_system_m, only: read_params_gyrokinetic_system, &
                                           write_params_gyrokinetic_system, &
                                           get_with_nlin_polarization, &
                                           get_with_bpar
    use params_source_m, only: read_params_source, write_params_source, &
                               params_source_loc_t
    use params_source_dens_m, only: read_params_source_dens, &
                                    write_params_source_dens
    use params_source_torque_m, only: read_params_source_torque, &
                                      write_params_source_torque
    use params_source_heat_m, only: read_params_source_heat, &
                                    write_params_source_heat
    use params_source_lapd_m, only: read_params_source_lapd, &
                                    write_params_source_lapd
    use params_diagnostics_m, only: read_params_diagnostics, &
                                    write_params_diagnostics, &
                                    get_polar_n_theta, get_polar_n_rho, &
                                    get_diagnose_tpc
    use params_devtools_m, only: read_params_devtools, write_params_devtools, &
                                 get_debug_profregion, get_isolate_tsync, &
                                 get_trace_device_memory, get_trace_runtime
    use params_bnd_cond_m, only: read_params_bnd_cond, &
                                 write_params_bnd_cond, &
                                 get_freeze_even_mom, &
                                 get_freeze_dens, &
                                 get_bnd_cond_type
    use params_dist_initial_m, only: params_dist_initial_t
    use params_dist_initial_bi_maxw_m
    use params_dist_initial_double_maxw_m
    use params_dist_initial_ring_m
    use params_dist_initial_slowing_down_m
    use params_dist_initial_maxw_vspec_m
    use bsg_types_m, only: MAX_BLOCKS

    ! From PARALLAX
    use params_equi_slab_m
    use params_equi_circular_m
    use params_equi_dommaschk_m

    implicit none
    private

    character(len=*), parameter :: default_params_file = "default_params"
    ! File to store default parameters

    public :: setup_test_params, teardown_test_params, write_default_params, &
              read_default_params
    public :: setup_test_coll
    public :: setup_test_dist_initial
    public :: setup_test_equi_slab
    public :: setup_test_equi_circular
    public :: setup_test_equi_dommaschk
    public :: setup_test_gpu_offload
    public :: setup_test_gyrokinetic_system
    public :: setup_test_initial_condition
    public :: setup_test_mesh
    public :: setup_test_bsg
    public :: setup_test_neutrals
    public :: setup_test_neutrals_config
    public :: setup_test_neutrals_init
    public :: setup_test_devtools
    public :: setup_test_species
    public :: setup_test_diagnostics
    public :: setup_test_time_split
    public :: setup_test_source_dens
    public :: setup_test_source_torque
    public :: setup_test_source_heat
    public :: setup_test_bnd_cond

    interface
        module subroutine setup_test_mesh(comm, rank, equilibrium_type, &
                                          spacing_RZ, n_points_phi, &
                                          n_points_vp, n_points_mu, &
                                          n_points_sp, length_vp, length_mu, &
                                          quad_type_vp, grid_type_mu, &
                                          n_levels, use_vspectral, &
                                          only_first_field_period, use_bsg)
            !! Initialize the simulation with the given test parameters for the
            !! mesh
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            integer, optional, intent(in) :: equilibrium_type
            real(kind=GP), optional, intent(in) :: spacing_RZ
            integer, optional, intent(in) :: n_points_phi
            integer, optional, intent(in) :: n_points_vp
            integer, optional, intent(in) :: n_points_mu
            integer, optional, intent(in) :: n_points_sp
            real(kind=GP), optional, intent(in) :: length_vp
            real(kind=GP), optional, intent(in) :: length_mu
            character(len=*), optional, intent(in) :: quad_type_vp
            character(len=*), optional, intent(in) :: grid_type_mu
            integer, optional, intent(in) :: n_levels
            logical, optional, intent(in) :: use_vspectral
            logical, optional, intent(in) :: only_first_field_period
            logical, optional, intent(in) :: use_bsg
        end subroutine

        module subroutine setup_test_bsg(comm, rank, &
                                         vplen_markers, radial_markers, &
                                         num_blocks, bsg_interp_order)
            !! Initialize the simulation with the given test parameters for the
            !! BSG
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            real(kind=GP), dimension(MAX_BLOCKS), intent(in) :: vplen_markers
            real(kind=GP), dimension(MAX_BLOCKS), intent(in) :: radial_markers
            integer, optional, intent(in) :: num_blocks
            integer, optional, intent(in) :: bsg_interp_order
        end subroutine

        module subroutine setup_test_coll(comm, rank, coll_type, relx_type)
            !! Initialize the simulation with the given test parameters for the
            !! collision operator
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            character(len=*), optional, intent(in) :: coll_type
            character(len=*), optional, intent(in) :: relx_type
        end subroutine

        module subroutine setup_test_species(comm, rank, &
                                             names, masses, charges, &
                                             temp_scalings)
            !! Initialize the simulation with the given test parameters for the
            !! species parameters
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            character(len=24), dimension(n_spec_supported), &
                optional, intent(in) :: names
            !! Species names
            real(kind=GP), dimension(n_spec_supported), &
                optional, intent(in) :: masses
            !! Species masses
            real(kind=GP), dimension(n_spec_supported), &
                optional, intent(in) :: charges
            !! Species charges
            real(kind=GP), dimension(n_spec_supported), &
                optional, intent(in) :: temp_scalings
            !! Species temperature scaling factors
        end subroutine

        module subroutine setup_test_diagnostics(comm, rank, &
                                                 polar_n_theta, polar_n_rho, &
                                                 diagnose_tpc)
            !! Initialize the simulation with the given test parameters for the
            !! diagnostics parameters
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            integer, optional, intent(in) :: polar_n_theta
            !! Number of points in theta direction (B in polar coordinates)
            integer, optional, intent(in) :: polar_n_rho
            !! Number of points in rho direction (B in polar coordinates)
            logical, optional, intent(in) :: diagnose_tpc
            !! Compute Trapped Particle Contributions (TPC) diagnostics
        end subroutine

        module subroutine setup_test_neutrals(comm, rank, &
                                              names, masses, mapped_ions_names)
            !! Initialize the simulation with the given test parameters for the
            !! neutrals parameters
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            character(len=24), dimension(n_neut_supported), &
                optional, intent(in) :: names
            !! Neutrals names
            real(kind=GP), dimension(n_neut_supported), &
                optional, intent(in) :: masses
            !! Neutrals masses
            character(len=24), dimension(n_neut_supported), &
                optional, intent(in) :: mapped_ions_names
            !! Names of the ions corresponding to neutrals species
        end subroutine

        module subroutine setup_test_neutrals_config(comm, rank, &
                                               neutrals_evolve_type, &
                                               neutrals_coupling_type, &
                                               neutrals_dens_floor, &
                                               neutrals_temp_floor, &
                                               neutrals_gamma_u, &
                                               neutrals_gamma_T, &
                                               neutrals_gamma_W)
            !! Initialize the simulation with the given test parameters for the
            !! neutrals parameters
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            character(len=*), optional, intent(in) :: neutrals_evolve_type
            !! Neutrals evolution operator to be used.
            character(len=*), optional, intent(in) :: neutrals_coupling_type
            !! Neutrals coupling operator to be used.
            real(kind=GP), optional, intent(in) :: neutrals_dens_floor
            !! Parameter that sets a minimum limit on the density in the
            !! calculations of plasma moments used for the neutral coupling.
            real(kind=GP), optional, intent(in) :: neutrals_temp_floor
            !! Parameter that sets a minimum limit on the temperature in the
            !! calculations of plasma to neutral coupling.
            real(kind=GP), optional, intent(in) :: neutrals_gamma_u
            !! Parameter that sets the neutrals velocity
            real(kind=GP), optional, intent(in) :: neutrals_gamma_T
            !! Parameter that sets the neutrals temperature
            real(kind=GP), optional, intent(in) :: neutrals_gamma_W
            !! Parameter that sets the electron temperature cooling due to
            !! neutrals
        end subroutine

        module subroutine setup_test_neutrals_init(comm, rank, o, &
                                                   params)
            !! Initialize the simulation with the given test parameters for the
            !! neutrals initial conditions parameters
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            integer, intent(in) :: o
            !! Neutrals species index
            type(params_neutrals_init_t), intent(in) :: params
            !! Parameter type
        end subroutine

        module subroutine setup_test_initial_condition(comm, rank, &
                                                       initial_perturbation)
            !! Initialize initial condition with the given test parameters
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            character(len=*), optional, intent(in) :: initial_perturbation
        end subroutine

        module subroutine setup_test_gyrokinetic_system(comm, rank, &
                                                    with_nlin_polarization, &
                                                    with_bpar)
            !! Initializes gyrokinetic system with the given test parameters
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            logical, optional, intent(in) :: with_nlin_polarization
            logical, optional, intent(in) :: with_bpar
        end subroutine

        module subroutine setup_test_devtools(comm, rank, debug_profregion, &
                                              isolate_tsync, &
                                              trace_device_memory, &
                                              trace_runtime)
            !! Initializes developer tools with the given test parameters
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            logical, optional, intent(in) :: debug_profregion
            !! Toggle to .true. to debug the profiler region
            logical, optional, intent(in) :: isolate_tsync
            !! Toggle to .true. to isolate synchronization time in the profiler
            logical, optional, intent(in) :: trace_device_memory
            !! Toggle to .true. to trace the memory usage on device memory
            logical, optional, intent(in) :: trace_runtime
            !! Toggle to .true. to trace the runtime based on profiling regions
        end subroutine

        module subroutine  setup_test_dist_initial_maxw_vspec(comm, rank, &
                                                              n, params)
            !! Initialize initial Maxwellian distribution function
            !! with the given test parameters using the spectral approach.
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            integer, intent(in) :: n
            !! Species index
            type(params_dist_initial_maxw_vspec_t), intent(in) :: params
            !! Parameter type
        end subroutine

        module subroutine setup_test_dist_initial_bi_maxw(comm, rank, n, params)
            !! Initialize initial bi Maxwellian distribution function
            !! with the given test parameters.
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            integer, intent(in) :: n
            !! Species index
            type(params_dist_initial_bi_maxw_t), intent(in) :: params
            !! Parameter type
        end subroutine

        module subroutine setup_test_dist_initial_double_maxw(comm, rank, &
                                                              n, params)
            !! Initialize initial double Maxwellian distribution function
            !! with the given test parameters.
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            integer, intent(in) :: n
            !! Species index
            type(params_dist_initial_double_maxw_t), intent(in) :: params
        end subroutine

        module subroutine setup_test_dist_initial_ring(comm, rank, n, params)
            !! Initialize initial ring distribution function
            !! with the given test parameters.
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            integer, intent(in) :: n
            !! Species index
            type(params_dist_initial_ring_t), intent(in) :: params
            !! Parameter of type
        end subroutine

        module subroutine setup_test_dist_initial_slowing_down(comm, rank, &
                                                               n, params)
            !! Initialize initial slowing down distribution function
            !! with the given test parameters.
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            integer, intent(in) :: n
            !! Species index
            type(params_dist_initial_slowing_down_t), intent(in) :: params
            !! Parameter of type
        end subroutine

        module subroutine setup_test_equi_slab(comm, rank, &
                                               boxsize, sol, yperiodic)
            !! Initialize the slab equilibrium with the given test
            !! parameters
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            real(kind=GP), optional, intent(in) :: boxsize
            logical, optional, intent(in) :: sol
            logical, optional, intent(in) :: yperiodic
        end subroutine

        module subroutine setup_test_equi_circular(comm, rank, &
                                                rhomin, rhomax, &
                                                qtype, rhoq_ref, q_ref, shear, &
                                                dtheta_limiter, rho_limiter, &
                                                theta_limiter)
            !! Initialize the circular equilibrium with the given test
            !! parameters
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            real(kind=GP), optional, intent(in) :: rhomin
            real(kind=GP), optional, intent(in) :: rhomax
            integer,       optional, intent(in) :: qtype
            real(kind=GP), optional, intent(in) :: rhoq_ref
            real(kind=GP), optional, intent(in) :: q_ref
            real(kind=GP), optional, intent(in) :: shear
            real(kind=GP), optional, intent(in) :: dtheta_limiter
            real(kind=GP), optional, intent(in) :: rho_limiter
            real(kind=GP), optional, intent(in) :: theta_limiter
        end subroutine

        module subroutine setup_test_equi_dommaschk(comm, rank, &
                                                    m_tor_consecutive, &
                                                    l_pol, &
                                                    num_field_periods, &
                                                    fitting_coef)
            !! Initialize the dommaschk equilibrium with the given test
            !! parameters
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            integer, optional, intent(in) :: m_tor_consecutive
            integer, optional, intent(in) :: l_pol
            integer, optional, intent(in) :: num_field_periods
            real(kind=GP), dimension(:,:,:), optional, intent(in) :: &
                                                                    fitting_coef
        end subroutine

        module subroutine setup_test_time_split(comm, rank, &
                                                time_scheme_collisions, &
                                                time_scheme_neutrals)
            !! Initializes time splitting scheme with the given test parameters
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            character(len=*), optional, intent(in) :: time_scheme_collisions
            !! Scheme to use for collisions
            character(len=*), optional, intent(in) :: time_scheme_neutrals
            !! Scheme to use for neutrals
        end subroutine

        module subroutine setup_test_gpu_offload(comm, rank, print_messages, &
                                                 debug, swap_mesh_members)
            !! Initializes the simulation with the given test parameters for the
            !! GPU offloading features
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            logical, optional, intent(in) :: print_messages
            !! Toggle to .true. to print messages
            logical, optional, intent(in) :: debug
            !! Toggle to .true. to check GPU availability and functionality
            logical, optional, intent(in) :: swap_mesh_members
            !! Toggle to .true. to swap mesh member pointers between Fortran/C++
        end subroutine

        module subroutine setup_test_source_dens(comm, rank, n, params)
            !! Initialize the simulation with the given test parameters for the
            !! density sources
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            integer, intent(in) :: n
            !! Species index
            type(params_source_loc_t), intent(in) :: params
            !! Parameter type
        end subroutine

        module subroutine setup_test_source_torque(comm, rank, n, params)
            !! Initialize the simulation with the given test parameters for the
            !! torque sources
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            integer, intent(in) :: n
            !! Species index
            type(params_source_loc_t), intent(in) :: params
            !! Parameter type
        end subroutine

        module subroutine setup_test_source_heat(comm, rank, n, params)
            !! Initialize the simulation with the given test parameters for the
            !! heat sources
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            integer, intent(in) :: n
            !! Species index
            type(params_source_loc_t), intent(in) :: params
            !! Parameter type
        end subroutine

        module subroutine setup_test_bnd_cond(comm, rank, &
                                              bnd_cond_type, &
                                              freeze_even_mom, &
                                              freeze_dens)
            !! Initialize the simulation with the given test parameters for the
            !! heat sources
            integer, intent(in) :: comm
            !! MPI communicator
            integer, intent(in) :: rank
            !! MPI rank
            character(len=*), optional, intent(in) :: bnd_cond_type
            !! Type of boundary condition
            logical, optional, intent(in) :: freeze_even_mom
            !! Flag freeze the boundary condition of even moments
            logical, optional, intent(in) :: freeze_dens
            !! Flag freeze the boundary condition of density
        end subroutine
    end interface

    interface setup_test_dist_initial
        module procedure setup_test_dist_initial_maxw_vspec
        module procedure setup_test_dist_initial_bi_maxw
        module procedure setup_test_dist_initial_double_maxw
        module procedure setup_test_dist_initial_ring
        module procedure setup_test_dist_initial_slowing_down
    end interface

contains

    subroutine setup_test_params(comm, rank)
        !! Setup test, flush out all the default parameters
        !! and active OpenMP testing
        integer, intent(in) :: comm
        !! MPI communicator
        integer, intent(in) :: rank
        !! MPI rank

        ! Activate OpenMP testing
        !$ call omp_set_dynamic(.true.)
        !$ call omp_set_num_threads(4)
        !$ call omp_set_dynamic(.false.)

        call write_default_params(comm, rank)

    end subroutine

    subroutine teardown_test_params(comm, rank)
        !! Read all the default parameters anad deactivates OpenMP testing
        integer, intent(in) :: comm
        !! MPI communicator
        integer, intent(in) :: rank
        !! MPI rank

        call read_default_params(comm, rank)

        ! Deactivate OpenMP testing
        !$ call omp_set_dynamic(.true.)
        !$ call omp_set_num_threads(1)
        !$ call omp_set_dynamic(.false.)

    end subroutine

    subroutine write_default_params(comm, rank)
        !! Write default parameters are saved in default_params_file
        !! before executing unit tests. If the file already exists, it will be
        !! deleted. This routine must be called before unit tests whenever
        !! parameters are modified.
        integer, intent(in) :: rank
        !! MPI rank
        integer, intent(in) :: comm
        !! MPI communicator

        integer :: ierr, n, o
        character(:), allocatable :: params_file, species_file, neutrals_file

        call mpi_barrier(comm, ierr)
        ! Write all default parameters
        if(rank == 0) then
            ! Delete old default parameter file if exists
            params_file = trim(default_params_file)//".txt"
            if(rank == 0) then
                call delete_file_if_exists(params_file)
            endif

            call write_params_parallelization(params_file)
            call write_params_normalization(params_file)
            call write_params_species(params_file)
            call write_params_diagnostics(params_file)
            call write_params_mesh(params_file)
            call write_params_bsg(params_file)
            call write_params_time_loop(params_file)
            call write_params_time_split(params_file)
            call write_params_set_blob(params_file)
            call write_params_set_mode(params_file)
            call write_params_numerical_scheme(params_file)
            call write_params_field_solve(params_file)
            call write_params_initial_condition(params_file)
            call write_params_mms(params_file)
            call write_params_collisions(params_file)
            call write_params_neutrals(params_file)
            call write_params_neutrals_config(params_file)
            call write_params_gyrokinetic_system(params_file)
            call write_params_source(params_file)
            call write_params_source_lapd(params_file)
            call write_params_devtools(params_file)
            call write_params_bnd_cond(params_file)

            call write_params_dommaschk(params_file)

            do n = 1, get_n_points_sp()
                species_file = trim(default_params_file) &
                            //"_"//trim(get_name(n))&
                            //".txt"
                call delete_file_if_exists(species_file)
                call write_params_species_config(species_file, n)
                call write_params_source_dens(species_file, n)
                call write_params_source_torque(species_file, n)
                call write_params_source_heat(species_file, n)
                call write_params_profile_cbc(species_file, n)
                call write_params_profile_sine(species_file, n)
                call write_params_profile_lapd(species_file, n)
                call write_params_dist_initial_bi_maxw(species_file, n)
                call write_params_dist_initial_double_maxw(species_file, n)
                call write_params_dist_initial_ring(species_file, n)
                call write_params_dist_initial_slowing_down(species_file, n)
                call write_params_dist_initial_maxw_vspec(species_file, n)
            end do
            do o = 1, get_n_points_neut()
                neutrals_file = trim(default_params_file) &
                              //"_"//trim(get_neut_name(o))&
                              //".txt"
                call delete_file_if_exists(neutrals_file)
                call write_params_neutrals_init(neutrals_file, o)
                call write_params_profile_uniform(neutrals_file, o)
                call write_params_profile_plateau(neutrals_file, o)
                call write_params_neutrals_set_blob(neutrals_file, o)
            end do
        endif

        call mpi_barrier(comm, ierr)

    end subroutine

    subroutine read_default_params(comm, rank)
        !! Read and reset all the default parameters from default_params_file
        !! after executing unit tests. This routine must be called at the end
        !! of unit tests whenever parameters are modified.
        integer, intent(in) :: rank
        !! Mpi rank
        integer, intent(in) :: comm
        !! Mpi communicator
        integer :: ierr, n, o

        character(:), allocatable :: params_file, species_file
        character(:), allocatable :: neutrals_file, default_neutrals_file, &
                                     test_neutrals_file

        ! Delete old default parameter file if exists
        params_file = trim(default_params_file)//".txt"

        if(.not. file_exists(params_file)) then
            call handle_error("Default parameter file "&
                              //trim(params_file)//" not found!", &
                              GENEX_ERR_UTESTS, __LINE__, __FILE__)
        endif

        call mpi_barrier(comm, ierr)

        ! Read main parameter file
        call read_params_parallelization(params_file)
        call read_params_normalization(params_file)
        call read_params_species(params_file)
        call read_params_diagnostics(params_file)
        call read_params_mesh(params_file)
        call read_params_bsg(params_file)
        call read_params_time_loop(params_file)
        call read_params_time_split(params_file)
        call read_params_set_blob(params_file)
        call read_params_set_mode(params_file)
        call read_params_numerical_scheme(params_file)
        call read_params_field_solve(params_file)
        call read_params_initial_condition(params_file)
        call read_params_mms(params_file)
        call read_params_collisions(params_file)
        call read_params_neutrals(params_file)
        call read_params_neutrals_config(params_file)
        call read_params_gyrokinetic_system(params_file)
        call read_params_source(params_file)
        call read_params_source_lapd(params_file)
        call read_params_devtools(params_file)
        call read_params_bnd_cond(params_file)

        ! Read PARALLAX params
        call read_params_dommaschk(params_file)

        ! Read species files defined by species names in main file
        do n = 1, get_n_points_sp()
            species_file = trim(default_params_file) &
                            //"_"//trim(get_name(n))&
                            //".txt"
            ! If file for species not found, skip reading and use default
            if(.not. file_exists(species_file)) then
                call handle_error("Default parameter file for species "&
                                  //trim(get_name(n))//" not found! (file "&
                                  //trim(species_file)//" not present)", &
                                  GENEX_ERR_UTESTS, __LINE__, __FILE__)
            endif
            call read_params_species_config(species_file, n)
            call read_params_source_dens(species_file, n)
            call read_params_source_torque(species_file, n)
            call read_params_source_heat(species_file, n)
            call read_params_profile_cbc(species_file, n)
            call read_params_profile_sine(species_file, n)
            call read_params_profile_lapd(species_file, n)
            call read_params_dist_initial_bi_maxw(species_file, n)
            call read_params_dist_initial_double_maxw(species_file, n)
            call read_params_dist_initial_ring(species_file, n)
            call read_params_dist_initial_slowing_down(species_file, n)
            call read_params_dist_initial_maxw_vspec(species_file, n)
        end do
       ! Read neutrals files defined by neutrals species names in main file
        do o = 1, get_n_points_neut()
            default_neutrals_file = trim(default_params_file) &
                                  //"_"//trim(get_neut_name(o))//".txt"
            test_neutrals_file    = trim(get_test_params_file())&
                                  //"_"//trim(get_neut_name(o))//".txt"
            if (file_exists(default_neutrals_file)) then
                neutrals_file = default_neutrals_file
            else if(file_exists(test_neutrals_file)) then
                neutrals_file = test_neutrals_file
            else
                call handle_error("Default parameter file for neutrals "&
                                 //trim(get_neut_name(o))//" not found! (file "&
                                 //trim(default_neutrals_file)&
                                 //" or automatically generated file "&
                                 //trim(test_neutrals_file)&
                                 //" not present)", &
                                  GENEX_ERR_UTESTS, __LINE__, __FILE__)
            end if
            call read_params_neutrals_init(neutrals_file, o)
            call read_params_profile_uniform(neutrals_file, o)
            call read_params_profile_plateau(neutrals_file, o)
            call read_params_neutrals_set_blob(neutrals_file, o)
        end do

        call mpi_barrier(comm, ierr)

    end subroutine

end module
