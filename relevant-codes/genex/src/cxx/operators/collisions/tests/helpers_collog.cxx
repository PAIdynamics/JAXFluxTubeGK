#ifndef HELPERS_COLLOG_CXX
#define HELPERS_COLLOG_CXX

#include "genex_cxx_env.hxx"
#include "params_gpu_offload.hxx"
#include "data_array.hxx"
#include "op_coll.hxx"

#ifdef __cplusplus
extern "C" {
#endif

int32_t cbind_calc_collog(
    const data_array_t<const real_t, 4>** moments_cxx_pptr,
    data_array_t<real_t, 2>** collog_cxx_pptr)
{
    // Assign the C++ class instances
    const data_array_t<const real_t, 4>& moments = *(*moments_cxx_pptr);
    data_array_t<real_t, 2>& collog = *(*collog_cxx_pptr);

    // Return 0 for success and 1 for error
    switch (params_gpu_offload::get_backend())
    {
        case params_gpu_offload::backend_t::CPU:
            return calc_collog_omp(moments, collog);
#ifdef ENABLE_OPENACC
        case params_gpu_offload::backend_t::ACC:
            return calc_collog_acc(moments, collog);
#endif
#ifdef ENABLE_OPENMPX
        case params_gpu_offload::backend_t::OMPX:
            return calc_collog_ompx(moments, collog);
#endif
        default:
            return 1;
    }
}

#ifdef __cplusplus
}
#endif

#endif
