#ifndef OP_LIN_COMB_CUDA_CUH
#define OP_LIN_COMB_CUDA_CUH

#include "genex_cxx_env.hxx"
#include "op_lin_comb.hxx"

// C++ class which corresponds to the Fortran class op_lin_comb_gpu_t
// with CUDA on GPU
class op_lin_comb_cuda_t: public op_lin_comb_gpu_t
{
public:
    // Default constructor
    op_lin_comb_cuda_t() = default;

    // Default destructor
    ~op_lin_comb_cuda_t() override = default;

    // Apply the operator to the given input values with CUDA on GPU
    // Return 0 for success and 1 for error
    int32_t apply(const int64_t size_y, const real_t a1, const real_t a2,
                  const real_t* __restrict__ x1,
                  const real_t* __restrict__ x2,
                        real_t* __restrict__ y) const override;
};

#endif
