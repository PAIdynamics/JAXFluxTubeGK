module math_m
    use genex_fortran_env_m, only: GP
    use, intrinsic :: iso_fortran_env

    implicit none
    private

    real(kind=GP), parameter, public :: PI = 3.1415926535897932_GP

    public :: heaviside, almost_equal, is_abs_le, is_abs_lt, &
              random_real, random_positive_real, random_int, krond, sqrthv

contains

    real(kind=GP) function heaviside(x)
        !! Heaviside step function. Returns 1 if x > 0 and 0 otherwise
        real(kind=GP), intent(in) :: x
        !! Argument
        heaviside = 0.0_GP
        if (x >= 0) heaviside = 1.0_GP
    end function

    pure function is_abs_le(a, b, tolerance)
        !! Checks whether a is smaller equal b within a floating point
        !! tolerance. Returns true if a == b within tolerance, otherwise true
        !! if a < b and false if a > b
        real(kind=GP), intent(in) :: a, b, tolerance
        real(kind=GP) :: absa, absb
        logical :: is_abs_le

        absa = abs(a)
        absb = abs(b)
        if(abs(absa - absb) > tolerance) then
            if(absa < absb) then
                is_abs_le = .true.
            else
                is_abs_le = .false.
            end if
        else
            is_abs_le = .true.
        end if
    end function

    pure function is_abs_lt(a, b, tolerance)
        !! Checks whether a is smaller than b within a floating point
        !! tolerance. Returns false if a == b within tolerance, otherwise true
        !! if a < b and false if a > b
        real(kind=GP), intent(in) :: a, b, tolerance
        real(kind=GP) :: absa, absb
        logical :: is_abs_lt

        absa = abs(a)
        absb = abs(b)
        if(abs(absa - absb) > tolerance) then
            if(absa < absb) then
                is_abs_lt = .true.
            else
                is_abs_lt = .false.
            end if
        else
            is_abs_lt = .false.
        end if
    end function

    logical elemental function almost_equal(a, b, tol) result(res)
        !! Checks whether a == b within tolerance
        !! Elemental functions operate on scalars or arrays of compatible shape
        !! If one of the arguments is an array, the output is an array of same
        !! shape
        real(kind=GP), intent(in) :: a
        real(kind=GP), intent(in) :: b
        real(kind=GP), intent(in) :: tol
        res = abs(a - b) < tol
    end function

    real(kind=GP) function random_real(lb, ub)
        !! Generates a random real between lb and ub
        real(kind=GP), intent(in), optional :: lb
        real(kind=GP), intent(in), optional :: ub

        real(kind=GP) :: r, s
        real(kind=GP) :: lb_loc = -1e20
        real(kind=GP) :: ub_loc = +1e20

        if (present(lb)) lb_loc = lb
        if (present(ub)) ub_loc = ub

        call random_number(r)

        random_real = r * (ub_loc - lb_loc) + lb_loc
    end function

    real(kind=GP) function random_positive_real(ub)
        !! Generates a random real between 0.0 and ub
        real(kind=GP), intent(in), optional :: ub

        if (present(ub)) then
            random_positive_real = random_real(0.0_GP, ub)
        else
            random_positive_real = random_real(0.0_GP)
        end if
    end function

    integer function random_int()
        !! Generates a random integer between -1*HUGE() and +1*HUGE()
        real(kind=GP) :: r, s

        call random_number(r)
        call random_number(s)

        if (s < 0.5_GP) then
            s = -1.0_GP
        else
            s =  1.0_GP
        end if
        random_int = floor(s * r * huge(0))
    end function

    pure function krond(p, j) result(res)
        !! Kronecker delta function.
        !! Returns 1 if p==j and 0 otherwise.
        integer, intent(in) :: p
        integer, intent(in) :: j
        real(kind=GP) :: res
        res = 0.0_GP
        if(p == j) &
            res = 1.0_GP
    end function

    pure function sqrthv(a) result(res)
        !! Square root of the product between a and heavside function of a.
        !! Returns sqrt(a) if a >=0 and 0 otherwise.
        real(kind=GP), intent(in) :: a
        real(kind=GP) :: res
        res = 0.0_GP
        if(a > 0.0_GP) &
            res = sqrt(a)
    end function

end module
