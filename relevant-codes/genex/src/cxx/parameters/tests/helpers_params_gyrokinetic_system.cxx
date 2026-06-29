#ifndef HELPERS_PARAMS_GYROKINETIC_SYSTEM_CXX
#define HELPERS_PARAMS_GYROKINETIC_SYSTEM_CXX

#include "genex_cxx_env.hxx"
#include "params_gyrokinetic_system.hxx"

#ifdef __cplusplus
extern "C" {
#endif

void cbind_get_params_gyrokinetic_system(int32_t* scalar_params)
{
    scalar_params[0] = static_cast<int32_t>(
                       params_gyrokinetic_system::get_with_rhs());
    scalar_params[1] = static_cast<int32_t>(
                       params_gyrokinetic_system::get_with_em_flutter());
    scalar_params[2] = static_cast<int32_t>(
                       params_gyrokinetic_system::get_with_nlin_polarization());
    scalar_params[3] = static_cast<int32_t>(
                       params_gyrokinetic_system::get_with_bpar());
}

#ifdef __cplusplus
}
#endif

#endif
