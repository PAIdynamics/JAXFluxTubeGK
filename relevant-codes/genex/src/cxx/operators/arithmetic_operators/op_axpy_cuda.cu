#include "op_axpy_cuda.cuh"

__global__ void kernel_axpy_core_cuda(const int64_t size_y, const real_t a,
                                      const real_t* __restrict__ x_device,
                                            real_t* __restrict__ y_device)
{
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= size_y) return;
    y_device[i] += a * x_device[i];
}

int32_t op_axpy_cuda_t::apply(const int64_t size_y, const real_t a,
                              const real_t* __restrict__ x,
                                    real_t* __restrict__ y) const
{
    const int32_t block_size = 256;
    const int32_t num_blocks = (size_y + block_size - 1) / block_size;
    const int64_t size = size_y * sizeof(real_t);
    real_t* x_device;
    real_t* y_device;

    cudaMalloc(&x_device, size);
    cudaMalloc(&y_device, size);

    cudaMemcpy(x_device, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y, size, cudaMemcpyHostToDevice);
    kernel_axpy_core_cuda<<<num_blocks, block_size>>>
        (size_y, a, x_device, y_device);
    cudaMemcpy(y, y_device, size, cudaMemcpyDeviceToHost);

    cudaFree(x_device);
    cudaFree(y_device);

    return 0;
}
