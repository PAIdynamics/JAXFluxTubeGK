module bessel_m
    use genex_fortran_env_m, only: GP
    use, intrinsic :: iso_fortran_env

    implicit none
    private

    real(kind=GP), parameter :: xsmall     = 5.55e-17_GP
    real(kind=GP), parameter :: xmax       = 713.986_GP
    real(kind=GP), parameter :: err_val_i0 = -1.0_GP
    real(kind=GP), parameter :: err_val_k0 = -1.0_GP

    public :: err_val_i0, err_val_k0
    public :: bessel_i0, bessel_k0

contains

    pure function bessel_i0(arg) result(bi0)
        !! Modified Bessel function of the 1st kind, I0(x)
        !! SOURCE: SPECFUN library by W. Cody and L. Stoltz
        !! NOTE: The function will produce worse results for larger arguments,
        !!       around arg > 25.
        !! NOTE: The function returns -1 if an invalid argument was input
        !!       (if arg > xmax).
        real(kind=GP), intent(in) :: arg
        real(kind=GP) :: bi0

        real(kind=GP) :: a, b, res, sump, sumq, x, xx
        real(kind=GP), parameter :: one5  = 15.0_GP, &
                                    exp40 = 2.353852668370199854e17_GP, &
                                    forty = 40.0_GP, &
                                    rec15 = 6.6666666666666666666e-2_GP, &
                                    two25 = 225.0_GP
        ! Mathematical constants
        real(kind=GP), parameter :: p(15) = [ &
                  -5.2487866627945699800e-18_GP,-1.5982226675653184646e-14_GP, &
                  -2.6843448573468483278e-11_GP,-3.0517226450451067446e-08_GP, &
                  -2.5172644670688975051e-05_GP,-1.5453977791786851041e-02_GP, &
                  -7.0935347449210549190e+00_GP,-2.4125195876041896775e+03_GP, &
                  -5.9545626019847898221e+05_GP,-1.0313066708737980747e+08_GP, &
                  -1.1912746104985237192e+10_GP,-8.4925101247114157499e+11_GP, &
                  -3.2940087627407749166e+13_GP,-5.5050369673018427753e+14_GP, &
                  -2.2335582639474375249e+15_GP ]
        ! Coefficients for xmall <= abs(arg) < 15.0
        real(kind=GP), parameter :: q(5) = [ &
                  -3.7277560179962773046e+03_GP, 6.5158506418655165707e+06_GP, &
                  -6.5626560740833869295e+09_GP, 3.7604188704092954661e+12_GP, &
                  -9.7087946179594019126e+14_GP ]
        ! Coefficients for xmall <= abs(arg) < 15.0
        real(kind=GP), parameter :: pp(8) = [ &
                  -3.9843750000000000000e-01_GP, 2.9205384596336793945e+00_GP, &
                  -2.4708469169133954315e+00_GP, 4.7914889422856814203e-01_GP, &
                  -3.7384991926068969150e-03_GP,-2.6801520353328635310e-03_GP, &
                   9.9168777670983678974e-05_GP,-2.1877128189032726730e-06_GP ]
        ! Coefficients for 15.0 <= abs(arg)
        real(kind=GP), parameter :: qq(7) = [ &
                  -3.1446690275135491500e+01_GP, 8.5539563258012929600e+01_GP, &
                  -6.0228002066743340583e+01_GP, 1.3982595353892851542e+01_GP, &
                  -1.1151759188741312645e+00_GP, 3.2547697594819615062e-02_GP, &
                  -5.5194330231005480228e-04_GP ]
        ! Coefficients for 15.0 <= abs(arg)
        integer :: i

        x = abs(arg)

        if(x < xsmall) then
            res = 1.0_GP

        else if(x < one5) then
            ! xmall <= x < 15

            xx   = x * x
            sump = p(1)
            do i = 2, 15
                sump = sump * xx + p(i)
            enddo
            xx = xx - two25
            sumq = ((((   xx + q(1) ) &
                        * xx + q(2) ) &
                        * xx + q(3) ) &
                        * xx + q(4) ) &
                        * xx + q(5)

            res = sump / sumq

        else if(one5 <= x) then

            if(xmax < x) then
                ! Error value return when arg is too large
                res = err_val_i0

            else
                ! 15.0 <= x

                xx = 1.0_GP / x - rec15
                sump = ((((((        pp(1) &
                              * xx + pp(2) ) &
                              * xx + pp(3) ) &
                              * xx + pp(4) ) &
                              * xx + pp(5) ) &
                              * xx + pp(6) ) &
                              * xx + pp(7) ) &
                              * xx + pp(8)
                sumq = ((((((   xx + qq(1) ) &
                              * xx + qq(2) ) &
                              * xx + qq(3) ) &
                              * xx + qq(4) ) &
                              * xx + qq(5) ) &
                              * xx + qq(6) ) &
                              * xx + qq(7)
                res = sump / sumq

                if(x <= (xmax - one5)) then
                    a = exp(x)
                    b = 1.0_GP
                else
                    a = exp(x - forty)
                    b = exp40
                endif

                res = ((res * a - pp(1) * a) / sqrt(x)) * b
            endif
        endif

        bi0 = res
    end function

    pure function bessel_k0(arg) result(bk0)
        !! Modified Bessel function of the 2nd kind, K0(x)
        !! SOURCE: SPECFUN library by W. Cody and L. Stoltz
        !! NOTE: The function returns -1 if an invalid argument was input
        !!       (if arg <= 0).
        real(kind=GP), intent(in) :: arg
        real(kind=GP) :: bk0

        real(kind=GP) :: res, temp, &
                         sumf, sumg, sump, sumq, &
                         x, xx
        real(kind=GP), parameter :: p(6) = [ &
                   5.8599221412826100000e-04_GP, 1.3166052564989571850e-01_GP, &
                   1.1999463724910714109e+01_GP, 4.6850901201934832188e+02_GP, &
                   5.9169059852270512312e+03_GP, 2.4708152720399552679e+03_GP ]
        ! Coefficients for xmall <= arg <= 1.0
        real(kind=GP), parameter :: q(2) = [ &
                   -2.4994418972832303646e+02_GP, 2.1312714303849120380e+04_GP ]
        ! Coefficients for xmall <= arg <= 1.0
        real(kind=GP), parameter :: f(4) = [ &
                 -1.6414452837299064100e+00_GP, -2.9601657892958843866e+02_GP, &
                 -1.7733784684952985886e+04_GP, -4.0320340761145482298e+05_GP ]
        ! Coefficients for xmall <= arg <= 1.0
        real(kind=GP), parameter :: g(3) = [ &
                  -2.5064972445877992730e+02_GP, 2.9865713163054025489e+04_GP, &
                  -1.6128136304458193998e+06_GP ]
        ! Coefficients for xmall <= arg <= 1.0
        real(kind=GP), parameter :: pp(10) = [ &
                   1.1394980557384778174e+02_GP, 3.6832589957340267940e+03_GP, &
                   3.1075408980684392399e+04_GP, 1.0577068948034021957e+05_GP, &
                   1.7398867902565686251e+05_GP, 1.5097646353289914539e+05_GP, &
                   7.1557062783764037541e+04_GP, 1.8321525870183537725e+04_GP, &
                   2.3444738764199315021e+03_GP, 1.1600249425076035558e+02_GP ]
        ! Coefficients for 1.0 < arg
        real(kind=GP), parameter :: qq(10) = [ &
                   2.0013443064949242491e+02_GP, 4.4329628889746408858e+03_GP, &
                   3.1474655750295278825e+04_GP, 9.7418829762268075784e+04_GP, &
                   1.5144644673520157801e+05_GP, 1.2689839587977598727e+05_GP, &
                   5.8824616785857027752e+04_GP, 1.4847228371802360957e+04_GP, &
                   1.8821890840982713696e+03_GP, 9.2556599177304839811e+01_GP ]
        ! Coefficients for 1.0 < arg
        integer :: i

        x = arg

        if(0.0_GP < x) then
            ! 0.0 < arg <= 1.0

            if(x <= 1.0_GP) then
                temp = log(x)

                if(x < xsmall) then
                    ! Small arg
                    res = p(6) / q(2) - temp

                else
                    xx = x * x
                    sump = ((((        p(1) &
                                * xx + p(2) ) &
                                * xx + p(3) ) &
                                * xx + p(4) ) &
                                * xx + p(5) ) &
                                * xx + p(6)
                    sumq = (xx + q(1)) * xx + q(2)
                    sumf = ((        f(1) &
                            * xx + f(2) ) &
                            * xx + f(3) ) &
                            * xx + f(4)
                    sumg = ((xx + g(1)) * xx + g(2)) * xx + g(3)

                    res = sump / sumq - xx * sumf * temp / sumg - temp
                endif

            else if(xmax < x) then
                ! Error return for xmax < arg
                res = 0.0_GP

            else
                ! 1.0 < arg

                xx   = 1.0_GP / x
                sump = pp(1)
                do i = 2, 10
                    sump = sump * xx + pp(i)
                enddo
                sumq = xx
                do i = 1, 9
                    sumq = (sumq + qq(i)) * xx
                enddo
                sumq = sumq + qq(10)

                res = sump / sumq / sqrt(x) * exp(-x)
            endif

        else
            ! Error value return for arg <= 0.0
            res = err_val_k0
        endif

        bk0 = res
    end function

end module
