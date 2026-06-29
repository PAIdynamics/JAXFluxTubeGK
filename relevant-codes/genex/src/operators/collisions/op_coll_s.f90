submodule(op_coll_m) op_coll_s
    !! Contains common op_coll implementations.

    use params_normalization_m, only: get_T_ref, get_n_ref
    use params_species_m, only: get_charge, get_temp_scaling
    implicit none

contains

    module subroutine calc_collog(da_moments, da_collog)
        class(data_array_4d_t), intent(in) :: da_moments
        class(data_array_2d_t), intent(inout) :: da_collog

        ! Local
        integer :: n_sp, lb_stripped(2), ub_stripped(2), buff_shape(4)
        ! Total number of species, lower and upper bounds of the domain, and
        ! buffer for domain shape
        real(kind=GP), contiguous, pointer, dimension(:,:,:,:) :: moments
        ! Pointer to the output moments
        real(kind=GP), contiguous, pointer, dimension(:,:) :: collog
        ! Pointer to the Coulomb logarithm
        real(kind=GP) :: T_ref
        ! Reference temperature multiplied by scaling factor of electrons
        real(kind=GP) :: prefac_sqlambda_D, sqlambda_D
        ! Debye length squared
        real(kind=GP) :: prefac_sqlambda_dB, sqlambda_dB
        ! de Broglie wavelength (electrons) squared
        real(kind=GP) :: prefac_sqlambda_L, sqlambda_L
        ! Landau length (closest approach) squared
        real(kind=GP) :: dens, temp
        ! Density and temperature
        integer :: i, k, n
        ! Loop indices
        integer :: n_electrons
        ! Index of electron species

        buff_shape = da_moments%get_shape()
        n_sp = buff_shape(4)
        lb_stripped = da_collog%get_lbound_stripped()
        ub_stripped = da_collog%get_ubound_stripped()

        moments => da_moments%get_readonly_pointer()
        collog => da_collog%get_pointer()

        ! This subroutine calculates the Coulomb logarithm for a 2d
        ! density and temperature array using the expression with the
        ! lambdas instead of the plain NRL form. The lambda values are
        ! taken from NRL.
        ! We only use the electron density and temperature for the calculation.

        ! Get electron index
        do n = 1, n_sp
            if(get_charge(n) == -1.0_GP) then
                n_electrons = n
                exit
            end if
        end do

        ! Prefactor formulas from NRL squared, with reference values included
        ! (1e3 for T and 1e13 for n)
        T_ref = get_T_ref() * get_temp_scaling(n_electrons)
        prefac_sqlambda_D  = 5.52049e-5_GP * T_ref / get_n_ref()
        prefac_sqlambda_dB = 7.6176e-19_GP / T_ref
        prefac_sqlambda_L  = 2.0736e-20_GP / T_ref**2

        ! Calc collog for each real space point
        !$omp parallel default(none) &
        !$omp firstprivate(lb_stripped, ub_stripped) &
        !$omp private(i, k, dens, temp, sqlambda_D, sqlambda_dB, sqlambda_L) &
        !$omp shared(collog, moments, n_electrons, prefac_sqlambda_D, &
        !$omp        prefac_sqlambda_dB, prefac_sqlambda_L)
        do k = lb_stripped(2), ub_stripped(2)
        !$omp do schedule(static)
        do i = lb_stripped(1), ub_stripped(1)

            dens = moments(i, k, 1, n_electrons)
            temp = moments(i, k, 3, n_electrons)

            sqlambda_D  = prefac_sqlambda_D * temp / dens
            sqlambda_dB = prefac_sqlambda_dB / temp
            sqlambda_L  = prefac_sqlambda_L / (temp**2)

            collog(i, k) = 0.5_GP &
                         * log(1.0_GP + sqlambda_D / (sqlambda_dB + sqlambda_L))
        end do
        !$omp end do nowait
        end do
        !$omp end parallel

    end subroutine

end submodule
