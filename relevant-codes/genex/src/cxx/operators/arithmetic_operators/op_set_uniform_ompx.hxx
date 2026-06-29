#ifndef OP_SET_UNIFORM_OMPX_HXX
#define OP_SET_UNIFORM_OMPX_HXX

#include "genex_cxx_env.hxx"
#include "op_set_uniform.hxx"
#include <omp.h>

// C++ class which corresponds to the Fortran class op_set_uniform_gpu_t
// with OpenMP offload on GPU
class op_set_uniform_ompx_t: public op_set_uniform_gpu_t
{
public:
    // Default constructor
    op_set_uniform_ompx_t() = default;

    // Default destructor
    virtual ~op_set_uniform_ompx_t() = default;

    // Apply the operator to the given input values for real type
    // with OpenMP offload on GPU. Return 0 for success and 1 for error.
    int32_t apply(const int64_t size_y, const real_t a,
                  real_t* __restrict__ y) const override
    {
        #pragma omp target teams distribute parallel for simd \
            default(none) defaultmap(to: scalar) \
            shared(size_y, a, y)
        for (int64_t i = 0; i < size_y; i++)
        {
            y[i] = a;
        }

        return 0;
    }

    // Apply the operator to the given input values for integer type
    // with OpenMP offload on GPU. Return 0 for success and 1 for error.
    int32_t apply(const int64_t size_y, const int32_t a,
                  int32_t* __restrict__ y) const override
    {
        #pragma omp target teams distribute parallel for simd \
            default(none) defaultmap(to: scalar) \
            shared(size_y, a, y)
        for (int64_t i = 0; i < size_y; i++)
        {
            y[i] = a;
        }

        return 0;
    }
};

#endif
