#ifndef LAGRANGE_INTERPOLATOR_HXX
#define LAGRANGE_INTERPOLATOR_HXX

#include "genex_cxx_env.hxx"
#include <tuple>
#include <cmath>

// Namespace for containing functions to calculate prefactors to calculate
// derivatives on an unstructured grid based on lagrange interpolation up to
// fourth order. The grid is defined by:
// y1-- --y2-- --y3-- --y4-- --y5
namespace lagrange_interpolator
{
    // Calculate the monomials of the first derivative of the
    // interpolating lagrange polynomial evaluated at y3 up to fourth order
    std::tuple<real_t, real_t, real_t, real_t, real_t>
    monomial_1st_derivative(real_t y1, real_t y2, real_t y3, real_t y4,
                            real_t y5)
    {
        // Values of the monomials
        real_t m1;
        real_t m2;
        real_t m3;
        real_t m4;
        real_t m5;

        m1 = -(((y2 - y3) * (y3 - y4) * (y3 - y5))
             / ((y1 - y2) * (y1 - y3) * (y1 - y4) * (y1 - y5)));
        m2 = ((y1 - y3) * (y3 - y4) * (y3 - y5))
             / ((y1 - y2) * (y2 - y3) * (y2 - y4) * (y2 - y5));
        m3 = ((y2 * (3.0 * y3 * y3 + y4 * y5 - 2.0 * y3 * (y4 + y5))
              + y3 * (-4.0 * y3 * y3 - 2.0 * y4 * y5 + 3.0 * y3 * (y4 + y5))
              + y1 * (3.0 * y3 * y3 + y4 * y5 - 2.0 * y3 * (y4 + y5)
                      + y2 * (-2.0 * y3 + y4 + y5))))
             / ((y1 - y3) * (-y2 + y3) * (y3 - y4) * (y3 - y5));
        m4 = ((y1 - y3) * (-y2 + y3) * (y3 - y5))
             / ((y1 - y4) * (-y2 + y4) * (-y3 + y4) * (y4 - y5));
        m5 = ((y1 - y3) * (-y2 + y3) * (y3 - y4))
             / ((y1 - y5) * (-y2 + y5) * (-y3 + y5) * (-y4 + y5));

        return {m1, m2, m3, m4, m5};
    }

    // Calculate the monomials of the second derivative of the
    // interpolating lagrange polynomial evaluated at y3 up to fourth order
    std::tuple<real_t, real_t, real_t, real_t, real_t>
    monomial_2nd_derivative(real_t y1, real_t y2, real_t y3, real_t y4,
                            real_t y5)
    {
        // Values of the monomials
        real_t m1;
        real_t m2;
        real_t m3;
        real_t m4;
        real_t m5;
        real_t poly = (y3 - y1) * (y3 - y2) * (y3 - y4) * (y3 - y5);

        m1 = (2.0 * poly * (3.0 * y3 * y3 + y4 * y5
              - 2.0 * y3 * (y4 + y5) + y2 *(-2.0 * y3 + y4 + y5)))
             / ((y1 - y2) * (y1 - y3) * (y1 - y4) * (y1 - y5));
        m2 = (-2.0 * poly * (3.0 * y3 * y3 + y4 * y5
              - 2.0 * y3 * (y4 + y5) + y1 * (-2.0 * y3 + y4 + y5)))
             / ((y1 - y2) * (y2 - y3) * (y2 - y4) * (y2 - y5));
        m3 = (-2.0 * poly * (6.0 * y3 * y3 - 3.0 * y3 * y4
              - 3.0 * y3 * y5 + y4 * y5 + y2 * (-3.0 * y3 + y4 + y5)
              + y1 * (y2 - 3.0 * y3 + y4 + y5)))
             / ((y1 - y3) * (-y2 + y3) * (y3 - y4) * (y3 - y5));
        m4 = (-2.0 * poly * (y3 * (3.0 * y3 - 2.0 * y5)
              + y2 * (-2.0 * y3 + y5) + y1 * (y2 - 2.0 * y3 + y5)))
             / ((y1 - y4) * (-y2 + y4) * (-y3 + y4) * (y4 - y5));
        m5 = (-2.0 * poly * (y3 * (3.0 * y3 - 2.0 * y4)
              + y2 * (-2.0 * y3 + y4) + y1 * (y2 - 2.0 * y3 + y4)))
             / ((y1 - y5) * (-y2 + y5) * (-y3 + y5) * (-y4 + y5));

        return {m1, m2, m3, m4, m5};
    }

    // Calculate the monomials of the fourth derivative of the
    // interpolating lagrange polynomial evaluated at y3 up to second order
    std::tuple<real_t, real_t, real_t, real_t, real_t>
    monomial_4th_derivative(real_t y1, real_t y2, real_t y3, real_t y4,
                            real_t y5)
    {
        // Values of the monomials
        real_t m1;
        real_t m2;
        real_t m3;
        real_t m4;
        real_t m5;

        m1 =  24.0 / ((y1 - y2) * (y1  - y3) * (y1  - y4) * (y1 - y5));
        m2 = -24.0 / ((y1 - y2) * (y2  - y3) * (y2  - y4) * (y2 - y5));
        m3 = -24.0 / ((y1 - y3) * (-y2 + y3) * (y3  - y4) * (y3 - y5));
        m4 = -24.0 / ((y1 - y4) * (-y2 + y4) * (-y3 + y4) * (y4 - y5));
        m5 =  24.0 / ((y1 - y5) * (y2  - y5) * (y3  - y5) * (y4 - y5));

        return {m1, m2, m3, m4, m5};
    }
}

#endif
