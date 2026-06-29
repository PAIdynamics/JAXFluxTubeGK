#ifndef HELPERS_DATA_ARRAY_DEV_PTR_CXX
#define HELPERS_DATA_ARRAY_DEV_PTR_CXX

#include "genex_cxx_env.hxx"
#include "params_gpu_offload.hxx"

#ifdef ENABLE_OPENACC
#include <openacc.h>
#endif

// Set uniform operation to a host pointer on CPU via OpenMP
int32_t data_array_set_uniform_cxx(const real_t set_value, const int64_t size,
                                   real_t* host_ptr)
{
    #pragma omp parallel for simd default(none) \
        shared(host_ptr, size, set_value) schedule(static)
    for (int32_t j = 0; j < size; j++)
    {
        host_ptr[j] = set_value;
    }
    return 0;
}

#ifdef ENABLE_OPENACC
// Set uniform operation directly to a device pointer on GPU via OpenACC
int32_t data_array_set_uniform_acc(const real_t set_value, const int64_t size,
                                   real_t* dev_ptr)
{
    #pragma acc parallel default(none) \
        deviceptr(dev_ptr) copyin(size, set_value)
    {
        #pragma acc loop independent firstprivate(set_value)
        for (int32_t j = 0; j < size; j++)
        {
            dev_ptr[j] = set_value;
        }
    }
    return 0;
}
#endif

#ifdef ENABLE_OPENMPX
// Set uniform operation directly to a device pointer on GPU
// via OpenMP offload
int32_t data_array_set_uniform_ompx(const real_t set_value, const int64_t size,
                                    real_t* dev_ptr)
{
    #pragma omp target teams distribute parallel for \
        default(none) defaultmap(none) \
        is_device_ptr(dev_ptr) map(to : size, set_value) \
        shared(dev_ptr, size, set_value)
    for (int32_t j = 0; j < size; j++)
    {
        dev_ptr[j] = set_value;
    }
    return 0;
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Interoperable routine to get memory address of the array
// contained in data_array_t object on GPU
intptr_t cbind_data_array_get_device_address(real_t* array_ptr)
{
    int32_t gpu_rank;
    void* dev_ptr;

    switch (params_gpu_offload::get_backend())
    {
        case params_gpu_offload::backend_t::CPU:
            return reinterpret_cast<uintptr_t>(array_ptr);
#ifdef ENABLE_OPENACC
        case params_gpu_offload::backend_t::ACC:
            dev_ptr = acc_deviceptr(array_ptr);
            return reinterpret_cast<uintptr_t>(dev_ptr);
#endif
#ifdef ENABLE_OPENMPX
        case params_gpu_offload::backend_t::OMPX:
            gpu_rank = omp_get_default_device();
            dev_ptr = omp_get_mapped_ptr(array_ptr, gpu_rank);
            return reinterpret_cast<uintptr_t>(dev_ptr);
#endif
        default:
            return 0;
    }
}

// Interoperable routine to set uniform value to a device pointer on GPU
int32_t cbind_data_array_set_uniform(const real_t set_value, const int64_t size,
                                     real_t* dev_ptr)
{
    switch (params_gpu_offload::get_backend())
    {
        case params_gpu_offload::backend_t::CPU:
            return data_array_set_uniform_cxx(set_value, size, dev_ptr);
#ifdef ENABLE_OPENACC
        case params_gpu_offload::backend_t::ACC:
            return data_array_set_uniform_acc(set_value, size, dev_ptr);
#endif
#ifdef ENABLE_OPENMPX
        case params_gpu_offload::backend_t::OMPX:
            return data_array_set_uniform_ompx(set_value, size, dev_ptr);
#endif
        default:
            return 1;
    }
}

#ifdef __cplusplus
}
#endif

#endif
