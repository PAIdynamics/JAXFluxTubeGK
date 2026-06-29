#include "op_set_uniform_cuda.cuh"

__global__ void kernel_set_uniform_real_core_cuda(const int64_t size_y,
                                                  const real_t a,
                                                  real_t* __restrict__ y_device)
{
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= size_y) return;
    y_device[i] = a;
}

__global__ void kernel_set_uniform_integer_core_cuda(const int64_t size_y,
                                                     const int a,
                                                     int* __restrict__ y_device)
{
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= size_y) return;
    y_device[i] = a;
}

int32_t op_set_uniform_cuda_t::apply(const int64_t size_y, const real_t a,
                                     real_t* __restrict__ y) const
{
    const int32_t block_size = 256;
    const int32_t num_blocks = (size_y + block_size - 1) / block_size;
    const int64_t size = size_y * sizeof(real_t);
    real_t* y_device;

    cudaMalloc(&y_device, size);

    kernel_set_uniform_real_core_cuda<<<num_blocks, block_size>>>
        (size_y, a, y_device);
    cudaMemcpy(y, y_device, size, cudaMemcpyDeviceToHost);

    cudaFree(y_device);

    return 0;
}

int32_t op_set_uniform_cuda_t::apply(const int64_t size_y, const int32_t a,
                                     int32_t* __restrict__ y) const
{
    const int32_t block_size = 256;
    const int32_t num_blocks = (size_y + block_size - 1) / block_size;
    const int64_t size = size_y * sizeof(int32_t);
    int32_t* y_device;

    cudaMalloc(&y_device, size);

    kernel_set_uniform_integer_core_cuda<<<num_blocks, block_size>>>
        (size_y, a, y_device);
    cudaMemcpy(y, y_device, size, cudaMemcpyDeviceToHost);

    cudaFree(y_device);

    return 0;
}
