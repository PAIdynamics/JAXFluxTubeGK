#ifndef HELPERS_PARAMS_COLLISIONS_CXX
#define HELPERS_PARAMS_COLLISIONS_CXX

#include "genex_cxx_env.hxx"
#include "params_collisions.hxx"
#include <string>

#ifdef __cplusplus
extern "C" {
#endif

void cbind_get_params_collisions(real_t* scalar_params, char* coll_type,
                                 char* relx_type)
{
    scalar_params[0] = params_collisions::get_dens_floor();
    scalar_params[1] = params_collisions::get_temp_floor();
    strcpy(coll_type, params_collisions::get_coll_type().c_str());
    strcpy(relx_type, params_collisions::get_relx_type().c_str());
}

#ifdef __cplusplus
}
#endif

#endif
