#ifndef HELPERS_DATA_ARRAY_5D_CXX
#define HELPERS_DATA_ARRAY_5D_CXX

#include "genex_cxx_env.hxx"
#include "params_gpu_offload.hxx"
#include "data_array.hxx"

#ifdef ENABLE_OPENACC
#include <openacc.h>
#endif

// Simple 5D addition operation with data_array to test the array getter and
// setter via OpenMP on CPU
int32_t data_array_5d_add_cxx(real_t add_const, data_array_t<real_t, 5>& da)
{
    #pragma omp parallel for simd default(none) shared(da, add_const) \
                         schedule(static) collapse(5)
    for (int32_t n = da.get_lbound(5); n <= da.get_ubound(5); n++)
    for (int32_t m = da.get_lbound(4); m <= da.get_ubound(4); m++)
    for (int32_t l = da.get_lbound(3); l <= da.get_ubound(3); l++)
    for (int32_t k = da.get_lbound(2); k <= da.get_ubound(2); k++)
    for (int32_t i = da.get_lbound(1); i <= da.get_ubound(1); i++)
    {
        da(i, k, l, m, n) += add_const;
    }
    return 0;
}

#ifdef ENABLE_OPENACC
// Simple 5D addition operation with data_array to test the array getter and
// setter via OpenACC on GPU
int32_t data_array_5d_add_acc(real_t add_const, data_array_t<real_t, 5>& da)
{
    #pragma acc parallel default(none) present(add_const, da) copyin(add_const)
    {
        #pragma acc loop independent collapse(5) firstprivate(add_const)
        for (int32_t n = da.get_lbound(5); n <= da.get_ubound(5); n++)
        for (int32_t m = da.get_lbound(4); m <= da.get_ubound(4); m++)
        for (int32_t l = da.get_lbound(3); l <= da.get_ubound(3); l++)
        for (int32_t k = da.get_lbound(2); k <= da.get_ubound(2); k++)
        for (int32_t i = da.get_lbound(1); i <= da.get_ubound(1); i++)
        {
            da(i, k, l, m, n) += add_const;
        }
    }
    return 0;
}
#endif

#ifdef ENABLE_OPENMPX
// Simple 5D addition operation with data_array to test the array getter and
// setter via OpenMP offload on GPU
int32_t data_array_5d_add_ompx(real_t add_const,
                               data_array_t<real_t, 5>& da)
{
    #pragma omp target teams default(none) defaultmap(none) \
        shared(da, add_const) map(to: add_const)
    {
        #pragma omp distribute simd collapse(5)
        for (int32_t n = da.get_lbound(5); n <= da.get_ubound(5); n++)
        for (int32_t m = da.get_lbound(4); m <= da.get_ubound(4); m++)
        for (int32_t l = da.get_lbound(3); l <= da.get_ubound(3); l++)
        for (int32_t k = da.get_lbound(2); k <= da.get_ubound(2); k++)
        for (int32_t i = da.get_lbound(1); i <= da.get_ubound(1); i++)
        {
            da(i, k, l, m, n) += add_const;
        }
    }
    return 0;
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Interoperable routine to add the 5d array in data array with a given constant
int32_t cbind_data_array_5d_add(real_t add_const,
                                data_array_t<real_t, 5>** da_cxx_pptr)
{
    // Assign the C++ class instances
    data_array_t<real_t, 5>& da = *(*da_cxx_pptr);

    // Return 0 for success and 1 for error
    switch (params_gpu_offload::get_backend())
    {
        case params_gpu_offload::backend_t::CPU:
            return data_array_5d_add_cxx(add_const, da);
#ifdef ENABLE_OPENACC
        case params_gpu_offload::backend_t::ACC:
            return data_array_5d_add_acc(add_const, da);
#endif
#ifdef ENABLE_OPENMPX
        case params_gpu_offload::backend_t::OMPX:
            return data_array_5d_add_ompx(add_const, da);
#endif
        default:
            return 1;
    }
}

#ifdef __cplusplus
}
#endif

#endif
