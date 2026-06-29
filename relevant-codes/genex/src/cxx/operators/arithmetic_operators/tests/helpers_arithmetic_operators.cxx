#ifndef HELPERS_ARITHMETIC_OPERATORS_CXX
#define HELPERS_ARITHMETIC_OPERATORS_CXX

#include "genex_cxx_env.hxx"
#include "device_memory_debugger.hxx"
#include "params_gpu_offload.hxx"
#include <omp.h>

#ifdef ENABLE_OPENACC
#include <openacc.h>
#endif

// Alias forexternal namespace
namespace dmd = device_memory_debugger;

// Templated function to copy an array from CPU to GPU
// Returns 0 if the copy is successful, otherwise 1 if erroneous
template<typename T>
int32_t copy_device(const int64_t size, T* array)
{
    switch (params_gpu_offload::get_backend())
    {
        case params_gpu_offload::backend_t::CPU:
        {
            return 0;
        }
#ifdef ENABLE_OPENACC
        case params_gpu_offload::backend_t::ACC:
        {
            #pragma acc enter data copyin(array[:size])
            dmd::start_region("copy_device", dmd::mode_t::ALLOC);
            bool err = dmd::is_invalid(array, size);
            dmd::end_region("copy_device");
            return (int32_t) err;
        }
#endif
#ifdef ENABLE_OPENMPX
        case params_gpu_offload::backend_t::OMPX:
        {
            #pragma omp target enter data map(to: array[:size])
            dmd::start_region("copy_device", dmd::mode_t::ALLOC);
            bool err = dmd::is_invalid(array, size);
            dmd::end_region("copy_device");
            return (int32_t) err;
        }
#endif
        case params_gpu_offload::backend_t::CUDA:
        {
            return 0;
        }
        default:
        {
            return 0;
        }
    }
}

// Templated function to copy an array from GPU to CPU
// Returns 0 if the copy is successful, otherwise 1 if erroneous
template<typename T>
int32_t copy_host(const int64_t size, T* array)
{
    switch (params_gpu_offload::get_backend())
    {
        case params_gpu_offload::backend_t::CPU:
        {
            return 0;
        }
#ifdef ENABLE_OPENACC
        case params_gpu_offload::backend_t::ACC:
        {
            #pragma acc exit data copyout(array[:size])
            dmd::start_region("copy_host", dmd::mode_t::DEALLOC);
            bool err = dmd::is_invalid(array, size);
            dmd::end_region("copy_host");
            return (int32_t) err;
        }
#endif
#ifdef ENABLE_OPENMPX
        case params_gpu_offload::backend_t::OMPX:
        {
            #pragma omp target exit data map(from: array[:size])
            dmd::start_region("copy_host", dmd::mode_t::DEALLOC);
            bool err = dmd::is_invalid(array, size);
            dmd::end_region("copy_host");
            return (int32_t) err;
        }
#endif
        case params_gpu_offload::backend_t::CUDA:
        {
            return 0;
        }
        default:
        {
            return 0;
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

real_t cbind_ref_op_axpy_core(const real_t a, const real_t x,
                              const real_t y)
{
    return y + a * x;
}

real_t cbind_ref_op_lin_comb_core(const real_t a1, const real_t a2,
                                  const real_t x1, const real_t x2)
{
    return a1 * x1 + a2 * x2;
}

int32_t cbind_copy_device_real(const int64_t size, real_t* array)
{
    return copy_device<real_t>(size, array);
}

int32_t cbind_copy_device_integer(const int64_t size, int32_t* array)
{
    return copy_device<int32_t>(size, array);
}

int32_t cbind_copy_host_real(const int64_t size, real_t* array)
{
    return copy_host<real_t>(size, array);
}

int32_t cbind_copy_host_integer(const int64_t size, int32_t* array)
{
    return copy_host<int32_t>(size, array);
}

#ifdef __cplusplus
}
#endif

#endif
