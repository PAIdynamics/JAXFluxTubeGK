#ifndef HELPERS_PARAMS_NORMALIZATION_CXX
#define HELPERS_PARAMS_NORMALIZATION_CXX

#include "genex_cxx_env.hxx"
#include "params_normalization.hxx"

#ifdef __cplusplus
extern "C" {
#endif

void cbind_get_params_normalization(
    struct params_normalization_data_t* params_data)
{
    params_data->m_ref = params_normalization::get_m_ref();
    params_data->T_ref = params_normalization::get_T_ref();
    params_data->n_ref = params_normalization::get_n_ref();
    params_data->L_ref = params_normalization::get_L_ref();
    params_data->B_ref = params_normalization::get_B_ref();
    params_data->c_ref = params_normalization::get_c_ref();
    params_data->rho_ref  = params_normalization::get_rho_ref();
    params_data->coll_ref = params_normalization::get_coll_ref();
    params_data->beta_ref = params_normalization::get_beta_ref();
}

#ifdef __cplusplus
}
#endif

#endif
