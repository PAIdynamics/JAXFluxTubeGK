#ifndef HELPERS_PARAMS_DEVTOOLS_CXX
#define HELPERS_PARAMS_DEVTOOLS_CXX

#include "params_devtools.hxx"

#ifdef __cplusplus
extern "C" {
#endif

void cbind_get_params_devtools(int32_t* scalar_params)
{
    scalar_params[0] =
        static_cast<int32_t>(params_devtools::get_isolate_tsync());
    scalar_params[1] =
        static_cast<int32_t>(params_devtools::get_trace_device_memory());
}

#ifdef __cplusplus
}
#endif

#endif
