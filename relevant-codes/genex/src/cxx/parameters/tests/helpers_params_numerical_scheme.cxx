#ifndef HELPERS_PARAMS_NUMERICAL_SCHEME_CXX
#define HELPERS_PARAMS_NUMERICAL_SCHEME_CXX

#include "genex_cxx_env.hxx"
#include "params_numerical_scheme.hxx"

#ifdef __cplusplus
extern "C" {
#endif

void cbind_get_params_numerical_scheme(
    struct params_numerical_scheme_data_t* params_data)
{
    params_data->int_order = params_numerical_scheme::get_int_order();
    params_data->hyp_y = params_numerical_scheme::get_hyp_y();
    params_data->hyp_xz = params_numerical_scheme::get_hyp_xz();
    params_data->hyp_vp = params_numerical_scheme::get_hyp_vp();
    params_data->buf_zone_size = params_numerical_scheme::get_buf_zone_size();
    params_data->buf_zone_strength =
        params_numerical_scheme::get_buf_zone_strength();
    params_data->buf_zone_size_axis =
        params_numerical_scheme::get_buf_zone_size_axis();
    params_data->buf_zone_strength_axis =
        params_numerical_scheme::get_buf_zone_strength_axis();
}

#ifdef __cplusplus
}
#endif

#endif
