#include "params_gpu_offload.hxx"

const int32_t arr_size = 3;

__global__ void kernel_cuda_on_device(int32_t* darr)
{
    int32_t i = blockIdx.x;
    if(i >= arr_size) return;
    darr[i]++;
}

int32_t get_num_devices_cuda()
{
    int32_t cuda_get_num_devices = 0;
    cudaError_t err;

    // Get number of GPUs via CUDA API
    err = cudaGetDeviceCount(&cuda_get_num_devices);
    switch (err) {
    case cudaSuccess:
      return cuda_get_num_devices;
      break;
    case cudaErrorNoDevice:
      std::cerr<<"No CUDA Device found! Aborting."<<std::endl;
      break;
    case cudaErrorInsufficientDriver:
      std::cerr<<"Insufficient CUDA Driver! Aborting."<<std::endl;
      break;
    default:
      std::cerr<<" Unspecified error in cudaGetDeviceCount. "<<std::endl;
      break;
    }
    return 0;
}

bool is_on_device_cuda() {
    int32_t cuda_is_on_device = 1;
    int32_t harr[arr_size];
    int32_t* darr;
    cudaError_t err;

    // Allocate array on device memory and check cudaMalloc
    err = cudaMalloc((void**) &darr, arr_size * sizeof(int32_t));
    if (err != cudaSuccess)
    {
        cuda_is_on_device = 0;
    }

    // Initialize host array
    for (int32_t i = 0; i < arr_size; i++)
    {
        harr[i] = i;
    }

    // Copy host array to device and check cudaMemcpy
    err = cudaMemcpy(darr, harr, arr_size * sizeof(int32_t),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cuda_is_on_device = 0;
    }

    // CUDA kernel launch and check the kernel launch
    kernel_cuda_on_device<<<arr_size, 1>>>(darr);
    if (cudaPeekAtLastError() != cudaSuccess)
    {
        cuda_is_on_device = 0;
    }

    // Copy device array to host and check cudaMemcpy
    err = cudaMemcpy(harr, darr, arr_size * sizeof(int32_t),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cuda_is_on_device = 0;
    }

    // Check the correctness of the results
    for (int32_t i = 0; i < arr_size; i++)
    {
        if (harr[i] != i + 1)
        {
            cuda_is_on_device = 0;
        }
    }

    // Deallocate array on device
    cudaFree(darr);

    return (cuda_is_on_device == 1);
}
