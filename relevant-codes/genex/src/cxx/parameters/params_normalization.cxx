#include "params_normalization.hxx"

// Namespace for parameters related to the normalization
namespace params_normalization
{
    // Private objects with file scope and external linkage

    // Reference mass (in proton mass)
    static real_t m_ref = 1.0;
    // Reference temperature (in keV)
    static real_t T_ref = 1.0;
    // Reference density (in 1e19 m^{-3})
    static real_t n_ref = 1.0;
    // Reference length (in m)
    static real_t L_ref = 1.0;
    // Reference magnetic field (in T)
    static real_t B_ref = 1.0;
    // Reference velocity (in m/s)
    static real_t c_ref = 3.09496901846338e5;
    // Reference gyroradius (in m)
    static real_t rho_ref = 3.23105013899193e-3;
    // Reference collision frequency prefactor (unitless)
    // (not nu_ref and not collisionality) static real_t coll_ref;
    static real_t coll_ref = 2.7719928e-4;
    // Reference beta (unitless)
    static real_t beta_ref = 403.0e-5;

    // Getters for the private objects

    real_t get_m_ref() { return m_ref; }
    real_t get_T_ref() { return T_ref; }
    real_t get_n_ref() { return n_ref; }
    real_t get_L_ref() { return L_ref; }
    real_t get_B_ref() { return B_ref; }
    real_t get_c_ref() { return c_ref; }
    real_t get_rho_ref()  { return rho_ref; }
    real_t get_coll_ref() { return coll_ref; }
    real_t get_beta_ref() { return beta_ref; }
}

void cbind_set_params_normalization(
    struct params_normalization_data_t* params_data)
{
    params_normalization::m_ref = params_data->m_ref;
    params_normalization::T_ref = params_data->T_ref;
    params_normalization::n_ref = params_data->n_ref;
    params_normalization::L_ref = params_data->L_ref;
    params_normalization::B_ref = params_data->B_ref;
    params_normalization::c_ref = params_data->c_ref;
    params_normalization::rho_ref  = params_data->rho_ref;
    params_normalization::coll_ref = params_data->coll_ref;
    params_normalization::beta_ref = params_data->beta_ref;
}
