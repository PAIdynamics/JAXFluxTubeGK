module neut_coupling_setup_m
    !! This module contains subroutines for the coupling operators shared by
    !! the _pn and _np types.
    use params_mesh_m, only: get_n_points_sp
    use params_species_m, only: get_name
    use params_neutrals_m, only: get_n_points_neut, get_neut_mapped_ion_name
    use op_set_uniform_m, only: op_set_uniform_cpu_t

    ! From REAX
    use rate_coefficients_m, only: rate_coefficient_t, &
                                   REACTION_RATE, COOLING_RATE
    use reax_atomic_data_info_m, only: atomic_data_path => path

    implicit none

contains

    subroutine setup_neut_ions_maps(map_neut_to_ion, map_ion_to_neut)
        !! Setup the mapping between the neutrals and itheir corresponding ions
        integer, allocatable, intent(inout) :: map_neut_to_ion(:)
        !! Map from neutrals to ions
        integer, allocatable, intent(inout) :: map_ion_to_neut(:)
        !! Map from ions to neutrals

        integer :: n_spec, n_neut, n, o
        character(len=:), allocatable :: species_name, mapped_ion_name
        type(op_set_uniform_cpu_t) :: op_set_uniform

        n_spec = get_n_points_sp()
        n_neut = get_n_points_neut()
        allocate(map_neut_to_ion(n_neut))
        allocate(map_ion_to_neut(n_spec))
        call op_set_uniform%apply(map_neut_to_ion, 0)
        call op_set_uniform%apply(map_ion_to_neut, 0)

        do n = 1, n_spec
            species_name = get_name(n)
            do o = 1, n_neut
                mapped_ion_name = get_neut_mapped_ion_name(o)
                if(mapped_ion_name == species_name) then
                    map_neut_to_ion(o) = n
                    map_ion_to_neut(n) = o
                end if
            end do
        end do
    end subroutine

    subroutine setup_reaction_rates(k_iz, k_rc)
        type(rate_coefficient_t), intent(inout) :: k_iz
        !! Ionization rate coefficient
        type(rate_coefficient_t), intent(inout) :: k_rc
        !! Recombination rate coefficient

        call k_iz%create(atomic_data_path // '/amjuel/H_iz.dat', &
                              REACTION_RATE)
        call k_rc%create(atomic_data_path // '/amjuel/H_rc.dat', &
                              REACTION_RATE)

    end subroutine

    subroutine setup_cooling_rates(w_iz, w_rc)
        type(rate_coefficient_t), intent(inout) :: w_iz
        !! Ionization cooling coefficient
        type(rate_coefficient_t), intent(inout) :: w_rc
        !! Recombination cooling coefficient

        call w_iz%create(atomic_data_path // '/amjuel/H_cooling_iz.dat', &
                              COOLING_RATE)
        call w_rc%create(atomic_data_path // '/amjuel/H_cooling_rc.dat', &
                              COOLING_RATE)
    end subroutine

end module
