submodule (mesh_5d_m) mesh_5d_vspec_s
    !! This module contains the implementation of the moment spectral weights
    !! to compute fluid moments.
    implicit none
contains

    module subroutine initialize_mom_weights_vspec(this)
        class(mesh_5d_t), target, intent(inout) :: this

        ! NOTE: We need to ensure that the dimensions of mom_weights_vspec
        !       are at least 5 in vp and 3 in mu to prevent
        !       segmentation faults when evaluating the moments
        !       with size_vp < 5 and size_mu < 3. This ensures
        !       that all moments can be computed without errors
        !       when using small size_vp and size_mu.

        allocate(this%mom_weights_vspec(max(5, this%size_vp()), &
                                        max(3, this%size_mu()), &
                                        9))
        call this%op_set_uniform%apply(this%mom_weights_vspec, 0.0_GP)

        ! Density spectral weights
        this%mom_weights_vspec(1, 1, 1) = 1.0_GP
        ! U_par spectral weights
        this%mom_weights_vspec(2, 1, 2) = 1.0_GP / sqrt(2.0_GP)
        ! E_par spectral weights
        this%mom_weights_vspec(3, 1, 3) = 1.0_GP / sqrt(2.0_GP)
        this%mom_weights_vspec(1, 1, 3) = 1.0_GP / 2.0_GP
        ! E_perp spectral weights
        this%mom_weights_vspec(1, 1, 4) = 1.0_GP
        this%mom_weights_vspec(1, 2, 4) = - 1.0_GP
        ! Q_par spectral weights
        this%mom_weights_vspec(2, 1, 5) = 3.0_GP / (2.0_GP * sqrt(2.0_GP))
        this%mom_weights_vspec(4, 1, 5) = sqrt(3.0_GP) / 2.0_GP
        ! Q_perp spectral weights
        this%mom_weights_vspec(2, 1, 6) = 1.0_GP / sqrt(2.0_GP)
        this%mom_weights_vspec(2, 2, 6) = - 1.0_GP / sqrt(2.0_GP)
        ! K_par spectral weights
        this%mom_weights_vspec(1, 1, 7) = 3.0_GP / 2.0_GP
        this%mom_weights_vspec(3, 1, 7) = 3.0_GP * sqrt(2.0_GP)
        this%mom_weights_vspec(5, 1, 7) = sqrt(6.0_GP)
        ! K_perp spectral weights
        this%mom_weights_vspec(1, 1, 8) = 0.5_GP
        this%mom_weights_vspec(1, 2, 8) = - 0.5_GP
        this%mom_weights_vspec(3, 1, 8) = 1.0_GP / sqrt(2.0_GP)
        this%mom_weights_vspec(3, 2, 8) = - 1.0_GP / sqrt(2.0_GP)
        ! K_perpperp spectral weights
        this%mom_weights_vspec(1, 1, 9) = 2.0_GP
        this%mom_weights_vspec(1, 2, 9) = - 4.0_GP
        this%mom_weights_vspec(1, 3, 9) = 2.0_GP

    end subroutine

end submodule
