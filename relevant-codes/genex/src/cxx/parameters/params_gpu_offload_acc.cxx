#include "params_gpu_offload.hxx"
#include <openacc.h>

int32_t get_num_devices_acc()
{
    return acc_get_num_devices(acc_device_nvidia);
}

bool is_on_device_acc()
{
    int32_t acc_is_on_host = 0;

    #pragma acc data copyout(acc_is_on_host)
    #pragma acc kernels
    {
        acc_is_on_host = acc_on_device(acc_device_host);
    }

    return (acc_is_on_host == 0);
}
