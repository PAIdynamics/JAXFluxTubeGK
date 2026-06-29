#ifndef OP_COLL_HXX
#define OP_COLL_HXX

#include "genex_cxx_env.hxx"
#include "params_species.hxx"
#include "params_normalization.hxx"
#include "data_array.hxx"
#include <cmath>
#include <omp.h>

#ifdef ENABLE_OPENACC
#include <openacc.h>
#endif

// Alias for parameter namespaces
namespace pspec = params_species;
namespace pnorm = params_normalization;

// Routine calculating the Coulomb logarithm on CPU via OpenMP
// Return 0 for success and 1 for error
inline int32_t calc_collog_omp(const data_array_t<const real_t, 4>& moments,
                               data_array_t<real_t, 2>& collog)
{
    const int32_t (&lb_stripped)[2] = collog.get_lbound_stripped();
    const int32_t (&ub_stripped)[2] = collog.get_ubound_stripped();

    // Get electron index
    int32_t electron_id = pspec::get_electron_id();

    // This subroutine calculates the Coulomb logarithm for a 2d density and
    // temperature array using the expression with the lambdas instead of
    // the plain NRL form. The lambda values are taken from NRL.
    // We only use the electron density and temperature for the calculation.
    real_t T_ref = pnorm::get_T_ref() * pspec::get_temp_scaling(electron_id);
    real_t prefac_sqlambda_D  = 5.52049e-5 * T_ref / pnorm::get_n_ref();
    real_t prefac_sqlambda_dB = 7.6176e-19 / T_ref;
    real_t prefac_sqlambda_L  = 2.0736e-20 / (T_ref * T_ref);

    // Calc collog for each real space point
    #pragma omp parallel default(none) \
        shared(lb_stripped, ub_stripped, collog, moments, prefac_sqlambda_D, \
               prefac_sqlambda_dB, prefac_sqlambda_L, electron_id)
    for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
    #pragma omp for simd schedule(static) nowait
    for (int32_t i = lb_stripped[0]; i <= ub_stripped[0]; i++)
    {
        real_t dens = moments(i, k, 1, electron_id);
        real_t temp = moments(i, k, 3, electron_id);

        real_t sqlambda_D  = prefac_sqlambda_D * temp / dens;
        real_t sqlambda_dB = prefac_sqlambda_dB / temp;
        real_t sqlambda_L  = prefac_sqlambda_L / (temp * temp);

        collog(i, k) = 0.5
                     * log(1.0 + sqlambda_D / (sqlambda_dB + sqlambda_L));
    }

    return 0;
}

#ifdef ENABLE_OPENACC

// Routine calculating the Coulomb logarithm on GPU via OpenACC
// Return 0 for success and 1 for error
inline int32_t calc_collog_acc(const data_array_t<const real_t, 4>& moments,
                               data_array_t<real_t, 2>& collog)
{
    const int32_t (&lb_stripped)[2] = collog.get_lbound_stripped();
    const int32_t (&ub_stripped)[2] = collog.get_ubound_stripped();

    // Get electron index
    int32_t electron_id = pspec::get_electron_id();

    real_t T_ref = pnorm::get_T_ref() * pspec::get_temp_scaling(electron_id);
    real_t prefac_sqlambda_D  = 5.52049e-5 * T_ref / pnorm::get_n_ref();
    real_t prefac_sqlambda_dB = 7.6176e-19 / T_ref;
    real_t prefac_sqlambda_L  = 2.0736e-20 / (T_ref * T_ref);

    // Calc collog for each real space point
    #pragma acc parallel default(none) \
        copyin(prefac_sqlambda_D, prefac_sqlambda_dB, \
               prefac_sqlambda_L, electron_id) \
        present(lb_stripped, ub_stripped, collog, moments)
    #pragma acc loop independent vector collapse(2)
    for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
    for (int32_t i = lb_stripped[0]; i <= ub_stripped[0]; i++)
    {
        real_t dens = moments(i, k, 1, electron_id);
        real_t temp = moments(i, k, 3, electron_id);

        real_t sqlambda_D  = prefac_sqlambda_D * temp / dens;
        real_t sqlambda_dB = prefac_sqlambda_dB / temp;
        real_t sqlambda_L  = prefac_sqlambda_L / (temp * temp);

        collog(i, k) = 0.5
                     * log(1.0 + sqlambda_D / (sqlambda_dB + sqlambda_L));
    }

    return 0;
}

#endif

#ifdef ENABLE_OPENMPX

// Routine calculating the Coulomb logarithm on GPU via OpenMP offload
// Return 0 for success and 1 for error
inline int32_t calc_collog_ompx(const data_array_t<const real_t, 4>& moments,
                                data_array_t<real_t, 2>& collog)
{
    const int32_t (&lb_stripped)[2] = collog.get_lbound_stripped();
    const int32_t (&ub_stripped)[2] = collog.get_ubound_stripped();

    // Get electron index
    int32_t electron_id = pspec::get_electron_id();

    real_t T_ref = pnorm::get_T_ref() * pspec::get_temp_scaling(electron_id);
    real_t prefac_sqlambda_D  = 5.52049e-5 * T_ref / pnorm::get_n_ref();
    real_t prefac_sqlambda_dB = 7.6176e-19 / T_ref;
    real_t prefac_sqlambda_L  = 2.0736e-20 / (T_ref * T_ref);

    // Calc collog for each real space point
    #pragma omp target teams distribute parallel for simd collapse(2) \
        default(none) defaultmap(none) \
        map(to: prefac_sqlambda_D, prefac_sqlambda_dB, \
                prefac_sqlambda_L, electron_id) \
        shared(lb_stripped, ub_stripped, collog, moments, electron_id, \
               prefac_sqlambda_D, prefac_sqlambda_dB, prefac_sqlambda_L)
    for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
    for (int32_t i = lb_stripped[0]; i <= ub_stripped[0]; i++)
    {
        real_t dens = moments(i, k, 1, electron_id);
        real_t temp = moments(i, k, 3, electron_id);

        real_t sqlambda_D  = prefac_sqlambda_D * temp / dens;
        real_t sqlambda_dB = prefac_sqlambda_dB / temp;
        real_t sqlambda_L  = prefac_sqlambda_L / (temp * temp);

        collog(i, k) = 0.5
                     * log(1.0 + sqlambda_D / (sqlambda_dB + sqlambda_L));
    }

    return 0;
}

#endif

#endif
