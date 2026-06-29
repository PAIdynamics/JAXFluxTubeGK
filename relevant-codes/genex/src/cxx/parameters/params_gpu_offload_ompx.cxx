#include "params_gpu_offload.hxx"
#include <omp.h>

int32_t get_num_devices_ompx()
{
    return omp_get_num_devices();
}

bool is_on_device_ompx()
{
    int32_t kernel_on_host = 0;

    #pragma omp target defaultmap(tofrom:scalar)
    {
        kernel_on_host = omp_is_initial_device();
    }

    return (kernel_on_host == 0);
}
