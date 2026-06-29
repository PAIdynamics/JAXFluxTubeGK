#include "op_lin_comb_cuda.cuh"

__global__ void kernel_lin_comb_core_cuda(const int64_t size_y,
                                          const real_t a1, const real_t a2,
                                          const real_t* __restrict__ x1_device,
                                          const real_t* __restrict__ x2_device,
                                                real_t* __restrict__ y_device)
{
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= size_y) return;
    y_device[i] = a1 * x1_device[i] + a2 * x2_device[i];
}

int32_t op_lin_comb_cuda_t::apply(const int64_t size_y, const real_t a1,
                                  const real_t a2,
                                  const real_t* __restrict__ x1,
                                  const real_t* __restrict__ x2,
                                        real_t* __restrict__ y) const
{
    const int32_t block_size = 256;
    const int32_t num_blocks = (size_y + block_size - 1) / block_size;
    const int64_t size = size_y * sizeof(real_t);
    real_t* x1_device;
    real_t* x2_device;
    real_t* y_device;

    cudaMalloc(&x1_device, size);
    cudaMalloc(&x2_device, size);
    cudaMalloc(&y_device, size);

    cudaMemcpy(x1_device, x1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(x2_device, x2, size, cudaMemcpyHostToDevice);
    kernel_lin_comb_core_cuda<<<num_blocks, block_size>>>
        (size_y, a1, a2, x1_device, x2_device, y_device);
    cudaMemcpy(y, y_device, size, cudaMemcpyDeviceToHost);

    cudaFree(x1_device);
    cudaFree(x2_device);
    cudaFree(y_device);

    return 0;
}
