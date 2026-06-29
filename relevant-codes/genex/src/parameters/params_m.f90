module params_m
    !! Module containing general functionality related to the parameter file
    use file_handling_m, only: delete_file_if_exists, file_exists
    use logger_m, only: logger_get_debug_channel
    use params_normalization_m, only: read_params_normalization, &
                                      write_params_normalization
    use params_parallelization_m, only: read_params_parallelization, &
                                        write_params_parallelization, &
                                        get_n_procs_phi, get_n_procs_vp, &
                                        get_n_procs_mu, get_n_procs_sp
    use params_species_m, only: read_params_species, &
                                write_params_species, get_name
    use params_species_config_m, only: read_params_species_config, &
                                       write_params_species_config
    use params_neutrals_m, only: read_params_neutrals, &
                                 write_params_neutrals, &
                                 get_n_points_neut, get_neut_name
    use params_neutrals_config_m, only: read_params_neutrals_config, &
                                        write_params_neutrals_config
    use params_neutrals_init_m, only: read_params_neutrals_init, &
                                      write_params_neutrals_init
    use params_neutrals_set_blob_m, only: read_params_neutrals_set_blob, &
                                          write_params_neutrals_set_blob
    use params_profile_uniform_m, only: read_params_profile_uniform, &
                                        write_params_profile_uniform
    use params_profile_plateau_m, only: read_params_profile_plateau, &
                                        write_params_profile_plateau
    use params_mesh_m, only: read_params_mesh, &
                             write_params_mesh, get_n_points_sp, &
                             get_equilibrium_type
    use params_bsg_m, only: read_params_bsg, write_params_bsg
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
    use params_initial_condition_m, only: read_params_initial_condition, &
                                          write_params_initial_condition
    use params_mms_m, only: read_params_mms, write_params_mms
    use params_collisions_m, only: read_params_collisions, &
                                   write_params_collisions
    use params_gyrokinetic_system_m, only: read_params_gyrokinetic_system, &
                                           write_params_gyrokinetic_system
    use params_source_m, only: read_params_source, write_params_source
    use params_source_dens_m, only: read_params_source_dens, &
                                    write_params_source_dens
    use params_source_heat_m, only: read_params_source_heat, &
                                    write_params_source_heat
    use params_source_torque_m, only: read_params_source_torque, &
                                      write_params_source_torque
    use params_source_lapd_m, only: read_params_source_lapd, &
                                    write_params_source_lapd
    use params_bnd_cond_m, only: read_params_bnd_cond, &
                                 write_params_bnd_cond
    use params_dist_initial_bi_maxw_m, only: &
                                read_params_dist_initial_bi_maxw, &
                                write_params_dist_initial_bi_maxw
    use params_dist_initial_double_maxw_m, only: &
                                read_params_dist_initial_double_maxw,&
                                write_params_dist_initial_double_maxw
    use params_dist_initial_ring_m, only: &
                                read_params_dist_initial_ring, &
                                write_params_dist_initial_ring
    use params_dist_initial_slowing_down_m, only: &
                                read_params_dist_initial_slowing_down, &
                                write_params_dist_initial_slowing_down
    use params_dist_initial_maxw_vspec_m, only: &
                                read_params_dist_initial_maxw_vspec, &
                                write_params_dist_initial_maxw_vspec
    use params_gpu_offload_m, only: read_params_gpu_offload, &
                                    write_params_gpu_offload
    use params_diagnostics_m, only: read_params_diagnostics, &
                                    write_params_diagnostics
    use params_devtools_m, only: read_params_devtools, &
                                 write_params_devtools
    use error_params_m, only: get_print_info

    ! From PARALLAX
    use equilibrium_factory_m, only: SLAB, CIRCULAR, SALPHA, DOMMASCHK
    use params_equi_slab_m, only: read_params_slab, write_params_slab
    use params_equi_circular_m, only: read_params_circular, &
                                      write_params_circular
    use params_equi_salpha_m, only: read_params_salpha, write_params_salpha
    use params_equi_dommaschk_m, only: read_params_dommaschk, &
                                       write_params_dommaschk

    implicit none
    private

    character(len=64), private :: parameter_file = ""
    !! Name of the parameter file. We need to store the name of the parameter
    !! file because PARALLAX reads the parameter file itself. As the parameters
    !! are constant after first initialization they are saved in globally
    !! accessible modules. These modules serve as parameter singletons

    public :: get_parameter_file
    public :: set_parameter_file

    public :: read_parameter_file
    public :: write_parameter_file

contains

    pure character(len=64) function get_parameter_file()
        !! Returns the parameter file
        get_parameter_file = parameter_file
    end function

    subroutine set_parameter_file(filename)
        !! Sets the parameter file
        character(len=*), intent(in) :: filename
        parameter_file = filename
    end subroutine

    subroutine read_parameter_file()
        !! Reads the parameter file and all species files defined by
        !! the species names in the main parameter file

        integer :: n, o
        character(:), allocatable :: species_file, neutrals_file

        ! Read main parameter file
        call read_params_gpu_offload(parameter_file)
        call read_params_parallelization(parameter_file)
        call read_params_normalization(parameter_file)
        call read_params_species(parameter_file)
        call read_params_mesh(parameter_file)
        call read_params_bsg(parameter_file)
        call read_params_time_loop(parameter_file)
        call read_params_time_split(parameter_file)
        call read_params_set_blob(parameter_file)
        call read_params_set_mode(parameter_file)
        call read_params_numerical_scheme(parameter_file)
        call read_params_field_solve(parameter_file)
        call read_params_initial_condition(parameter_file)
        call read_params_mms(parameter_file)
        call read_params_collisions(parameter_file)
        call read_params_neutrals(parameter_file)
        call read_params_neutrals_config(parameter_file)
        call read_params_gyrokinetic_system(parameter_file)
        call read_params_source(parameter_file)
        call read_params_source_lapd(parameter_file)
        call read_params_diagnostics(parameter_file)
        call read_params_devtools(parameter_file)
        call read_params_bnd_cond(parameter_file)

        ! Read species files defined by species names in main file
        do n = 1, get_n_points_sp()
            species_file = "params_"//trim(get_name(n))//".txt"

            ! If file for species not found, skip reading and use default
            if(.not. file_exists(species_file)) then
                if(get_print_info()) then
                    write(logger_get_debug_channel(), *) &
                        "Info: Parameter file for species "&
                        //trim(get_name(n))//" not found! (file "&
                        //trim(species_file)//" not present)"
                endif
                cycle
            end if
            call read_params_species_config(species_file, n)
            call read_params_profile_cbc(species_file, n)
            call read_params_profile_sine(species_file, n)
            call read_params_profile_lapd(species_file, n)
            call read_params_dist_initial_bi_maxw(species_file, n)
            call read_params_dist_initial_double_maxw(species_file, n)
            call read_params_dist_initial_ring(species_file, n)
            call read_params_dist_initial_slowing_down(species_file, n)
            call read_params_dist_initial_maxw_vspec(species_file, n)
            call read_params_source_dens(species_file, n)
            call read_params_source_torque(species_file, n)
            call read_params_source_heat(species_file, n)
        end do

        ! Read neutrals files defined by neutrals species names in main file
        do o = 1, get_n_points_neut()
            neutrals_file = "params_"//trim(get_neut_name(o))//".txt"

            ! If file for neutrals not found, skip reading and use default
            if(.not. file_exists(neutrals_file)) then
                if(get_print_info()) then
                    write(logger_get_debug_channel(), *) &
                        "Info: Parameter file for neutrals species "&
                        //trim(get_neut_name(o))//" not found! (file "&
                        //trim(neutrals_file)//" not present)"
                endif
                cycle
            end if
            call read_params_neutrals_init(neutrals_file, o)
            call read_params_profile_uniform(neutrals_file, o)
            call read_params_profile_plateau(neutrals_file, o)
            call read_params_neutrals_set_blob(neutrals_file, o)
        end do

        ! Read equilibrium parameters. Note that this is redundant, since the
        ! parameters will be read again during the equilibrium initialization
        ! in the mesh. It is only included here as a check.
        select case(get_equilibrium_type())
        case(SLAB)
            call read_params_slab(parameter_file)
        case(CIRCULAR)
            call read_params_circular(parameter_file)
        case(SALPHA)
            call read_params_salpha(parameter_file)
        case(DOMMASCHK)
            call read_params_dommaschk(parameter_file)
        case default
            write(logger_get_debug_channel(), "(A,I2,A)") &
                "Info: Parameter check for equilibrium ", &
                get_equilibrium_type(), " is not yet implemented!"
        end select

    end subroutine

    subroutine write_parameter_file(out_dir)
        !! Writes the parameter file and all species files defined by
        !! the species names in the main parameter file
        character(len=*), intent(in) :: out_dir
        !! Output directory

        integer :: n, o
        character(:), allocatable :: params_out
        character(:), allocatable :: species_file, neutrals_file

        ! Delete old params_out.txt file if exists
        params_out = trim(out_dir)//"params_out.txt"
        call delete_file_if_exists(params_out)

        ! Write main parameter file
        call write_params_gpu_offload(params_out)
        call write_params_parallelization(params_out)
        call write_params_normalization(params_out)
        call write_params_species(params_out)
        call write_params_mesh(params_out)
        call write_params_bsg(params_out)
        call write_params_time_loop(params_out)
        call write_params_time_split(params_out)
        call write_params_set_blob(params_out)
        call write_params_set_mode(params_out)
        call write_params_numerical_scheme(params_out)
        call write_params_field_solve(params_out)
        call write_params_initial_condition(params_out)
        call write_params_mms(params_out)
        call write_params_collisions(params_out)
        call write_params_neutrals(params_out)
        call write_params_neutrals_config(params_out)
        call write_params_gyrokinetic_system(params_out)
        call write_params_source(params_out)
        call write_params_source_lapd(params_out)
        call write_params_diagnostics(params_out)
        call write_params_devtools(params_out)
        call write_params_bnd_cond(params_out)

        ! Write species files defined by species names in main file
        do n = 1, get_n_points_sp()
            species_file = trim(out_dir)//"params_"//trim(get_name(n))&
                           //"_out.txt"

            ! Delete species output file if it exists
            call delete_file_if_exists(species_file)

            call write_params_species_config(species_file, n)
            call write_params_profile_cbc(species_file, n)
            call write_params_profile_sine(species_file, n)
            call write_params_profile_lapd(species_file, n)
            call write_params_dist_initial_bi_maxw(species_file, n)
            call write_params_dist_initial_double_maxw(species_file, n)
            call write_params_dist_initial_ring(species_file, n)
            call write_params_dist_initial_slowing_down(species_file, n)
            call write_params_dist_initial_maxw_vspec(species_file, n)
            call write_params_source_dens(species_file, n)
            call write_params_source_torque(species_file, n)
            call write_params_source_heat(species_file, n)
        end do

        ! Write neutrals files defined by neutrals species names in main file
        do o = 1, get_n_points_neut()
            neutrals_file = trim(out_dir)//"params_"//trim(get_neut_name(o))&
                          //"_out.txt"

            ! Delete species output file if it exists
            call delete_file_if_exists(neutrals_file)

            call write_params_neutrals_init(neutrals_file, o)
            call write_params_profile_uniform(neutrals_file, o)
            call write_params_profile_plateau(neutrals_file, o)
            call write_params_neutrals_set_blob(neutrals_file, o)
        end do

        ! Write equilibrium parameters
        select case(get_equilibrium_type())
        case(SLAB)
            call write_params_slab(params_out)
        case(CIRCULAR)
            call write_params_circular(params_out)
        case(SALPHA)
            call write_params_salpha(params_out)
        case(DOMMASCHK)
            call write_params_dommaschk(params_out)
        case default
            write(logger_get_debug_channel(), "(A,I2,A)") &
                "Info: Parameter writing for equilibrium ", &
                get_equilibrium_type(), " is not yet implemented!"
        end select

    end subroutine
end module
