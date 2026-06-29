submodule (timestep_m) strang_splitting_rkc_s
    use genex_status_codes_m, only: GENEX_ERR_TIMESTEP
    use chebyshev_polynomials_m, only: chebyshev_T
    ! From PARALLAX
    use polynomials_m, only: polyval, polyder

    implicit none
contains

    module subroutine initialize_rkc(this, n_stages, eta, coefs)
        class(strang_splitting_t), target, intent(inout) :: this
        integer, intent(in) :: n_stages
        real(kind=GP), intent(in) :: eta
        type(rkc_coefs_t), intent(inout) :: coefs

        integer :: j
        real(kind=GP) :: w0, w1
        real(kind=GP), dimension(0:n_stages) :: b_coef, c_coef
        real(kind=GP), dimension(1:n_stages) :: u_coef, v_coef, &
                                                utilde_coef, gamma_coef
        real(kind=GP), dimension(:), allocatable :: poly, dpoly, ddpoly

        ! See equations (3) and (7) in
        ! X. Tang and A. Xiao,
        ! Improved Runge–Kutta–Chebyshev methods
        ! Math Comput Simulat 174, 59–75 (2020)
        ! DOI: 10.1016/j.matcom.2020.02.021

        w0 = 1.0_GP + eta / n_stages**2
        allocate(poly(1:(n_stages+1)))
        allocate(dpoly(1:n_stages))
        allocate(ddpoly(1:(n_stages-1)))
        poly   = chebyshev_T(n_stages)
        dpoly  = polyder(poly)
        ddpoly = polyder(dpoly)
        w1 = polyval(dpoly, w0) / polyval(ddpoly, w0)
        deallocate(poly, dpoly, ddpoly)

        do j = 2, n_stages
            allocate(poly(1:(j+1)), dpoly(1:j), ddpoly(1:(j-1)))
            poly   = chebyshev_T(j)
            dpoly  = polyder(poly)
            ddpoly = polyder(dpoly)
            b_coef(j) = polyval(ddpoly, w0) / polyval(dpoly, w0)**2
            deallocate(poly, dpoly, ddpoly)
        enddo
        b_coef(1) = b_coef(2)
        b_coef(0) = b_coef(2)

        c_coef(0) = 0.0_GP
        do j = 1, n_stages
            u_coef(j)      = 2.0_GP * w0 * b_coef(j) / b_coef(j - 1)
            utilde_coef(j) = 2.0_GP * w1 * b_coef(j) / b_coef(j - 1)

            allocate(poly(1:j))
            poly = chebyshev_T(j-1)
            gamma_coef(j) = -utilde_coef(j) &
                          * (1.0_GP - b_coef(j - 1) * polyval(poly, w0))
            deallocate(poly)

            if(j == 1) then
                utilde_coef(1) = b_coef(1) * w1
                c_coef(1) = utilde_coef(1)
                v_coef(1) = 0.0_GP
            else
                v_coef(j) =-b_coef(j) / b_coef(j - 2)
                c_coef(j) = u_coef(j) * c_coef(j - 1) &
                          + v_coef(j) * c_coef(j - 2) &
                          + utilde_coef(j) &
                          + gamma_coef(j)
            endif
        enddo

        ! NOTE: We only need c(j) with j>0 for the algorithm
        coefs = rkc_coefs_t(n_stages, c_coef(1:), u_coef, v_coef, &
                            utilde_coef, gamma_coef)
    end subroutine

    module subroutine step_rkc(this, coefs, ierr)
        class(strang_splitting_t), target, intent(inout) :: this
        type(rkc_coefs_t), intent(in) :: coefs
        integer, intent(out) :: ierr

        type(state_vector_t), pointer :: Kj, K0, F0, Km1, Km2, Fm1

        real(kind=GP) :: initial_t
        real(kind=GP) :: c1, c2
        integer :: j, n_stages

        ! See equation (2) in
        ! X. Tang and A. Xiao,
        ! Improved Runge–Kutta–Chebyshev methods
        ! Math Comput Simulat 174, 59–75 (2020)
        ! DOI: 10.1016/j.matcom.2020.02.021

        ierr = GENEX_SUCCESS
        this%current_stage = 1
        initial_t = this%t
        n_stages = coefs%n_stages

        ! Use names from formula for simplicity. Notation Km1 means K_{j-1}
        ! (similar for other variables).
        Kj  => this%state
        K0  => this%initial_state
        F0  => this%k
        Fm1 => this%k2
        Km1 => this%increment
        Km2 => this%increment2

        ! K0 = y0
        ! F0 = f(K0, t)
        call K0%copy(Kj)
        call this%step_evolve(K0, F0, ierr)
        if(ierr /= GENEX_SUCCESS) return

        ! Kj  = y0 + ut(1) * h * F0
        ! Fm1 = f(Kj, t + c(1) * h)
        call Kj%add(coefs%utilde_coef(1) * this%dt, F0)
        this%t = initial_t + coefs%c_coef(1) * this%dt
        call this%step_evolve(Kj, Fm1, ierr)

        ! TODO: Investigate if performance gain can be achieved by removing
        !       the following copy operations and adding if clauses in the
        !       loop.
        call Km2%copy(K0)
        call Km1%copy(Kj)

        do j = 2, n_stages
            this%current_stage = j

            ! NOTE: To save an instance of state_vector we use Km2 to collect
            !       the individual terms and gradually add them to Kj. Km2 can
            !       be used because it is not required in the next stage
            !       anymore.

            ! A = v(j) * Km2 + (1 - u(j) - v(j)) * K0
            c1 = coefs%v_coef(j)
            c2 = 1.0_GP - coefs%u_coef(j) - coefs%v_coef(j)
            call Km2%lin_comb(c1, Km2, c2, K0)

            ! Kj = u(j) * Km1 + A
            c1 = coefs%u_coef(j)
            c2 = 1.0_GP
            call Kj%lin_comb(c1, Km1, c2, Km2)

            ! Kj = Kj + ut(j) * h * Fm1 + ga(j) * h * F0
            c1 = coefs%utilde_coef(j) * this%dt
            c2 = coefs%gamma_coef(j) * this%dt
            call Km2%lin_comb(c1, Fm1, c2, F0)
            call Kj%add(1.0_GP, Km2)

            this%t = initial_t + coefs%c_coef(j) * this%dt

            ! NOTE: These operations are not required for the last stage, we
            !       save time by skipping them.
            if(j < n_stages) then
                ! Km2 = Km1
                ! Km1 = Kj
                call Km2%copy(Km1)
                call Km1%copy(Kj)

                ! Fm1 = f(Km1, t + c(j) * h)
                call this%step_evolve(Km1, Fm1, ierr)
                if(ierr /= GENEX_SUCCESS) return
            endif
        enddo

        this%t = initial_t + this%dt
        call this%step_finish(Kj, Fm1, ierr)
        if(ierr /= GENEX_SUCCESS) return

    end subroutine

end submodule
