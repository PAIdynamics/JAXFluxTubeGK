#ifndef OP_AXPY_OMPX_HXX
#define OP_AXPY_OMPX_HXX

#include "genex_cxx_env.hxx"
#include "op_axpy.hxx"
#include <omp.h>

// C++ class which corresponds to the Fortran class op_axpy_gpu_t
// with OpenMP offload on GPU
class op_axpy_ompx_t: public op_axpy_gpu_t
{
public:
    // Default constructor
    op_axpy_ompx_t() = default;

    // Default destructor
    ~op_axpy_ompx_t() override = default;

    // Apply the operator to the given input values with OpenMP offload on GPU
    // Return 0 for success and 1 for error
    int32_t apply(const int64_t size_y, const real_t a,
                  const real_t* __restrict__ x,
                        real_t* __restrict__ y) const override
    {
        #pragma omp target teams distribute parallel for simd \
            default(none) defaultmap(to: scalar) \
            shared(size_y, a, x, y)
        for (int64_t i = 0; i < size_y; i++)
        {
            y[i] += a * x[i];
        }

        return 0;
    }
};

#endif
