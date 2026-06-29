#ifndef PARAMS_NORMALIZATION_HXX
#define PARAMS_NORMALIZATION_HXX

#include "genex_cxx_env.hxx"

// Namespace for parameters related to the normalization
namespace params_normalization
{
    // Getter for the reference mass
    real_t get_m_ref();

    // Getter for the reference temperature
    real_t get_T_ref();

    // Getter for the reference density
    real_t get_n_ref();

    // Getter for the reference length
    real_t get_L_ref();

    // Getter for the reference magnetic field
    real_t get_B_ref();

    // Getter for the reference velocity
    real_t get_c_ref();

    // Getter for the reference gyroradius
    real_t get_rho_ref();

    // Getter for the reference collision frequency prefactor
    real_t get_coll_ref();

    // Getter for the reference beta
    real_t get_beta_ref();
}

#ifdef __cplusplus
extern "C" {
#endif

struct params_normalization_data_t
{
    real_t m_ref;
    real_t T_ref;
    real_t n_ref;
    real_t L_ref;
    real_t B_ref;
    real_t c_ref;
    real_t rho_ref;
    real_t coll_ref;
    real_t beta_ref;
};

void cbind_set_params_normalization(
    struct params_normalization_data_t* params_data);

#ifdef __cplusplus
}
#endif

#endif
