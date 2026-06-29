#ifndef OP_COPY_OMP_HXX
#define OP_COPY_OMP_HXX

#include "genex_cxx_env.hxx"
#include "op_copy.hxx"

// C++ class which corresponds to the Fortran class op_copy_gpu_t
// with OpenMP on CPU
class op_copy_omp_t: public op_copy_gpu_t
{
public:
    // Default constructor
    op_copy_omp_t() = default;

    // Default destructor
    ~op_copy_omp_t() override = default;

    // Apply the operator to the given input values for real type
    // with OpenMP on CPU. Return 0 for success and 1 for error.
    int32_t apply(const int64_t size_y,
                  const real_t* __restrict__ x,
                        real_t* __restrict__ y) const override
    {
        #pragma omp parallel default(none) shared(size_y, x, y)
        #pragma omp for simd schedule(static)
        for (int64_t i = 0; i < size_y; i++)
        {
            y[i] = x[i];
        }

        return 0;
    }

    // Apply the operator to the given input values for integer type
    // with OpenMP on CPU. Return 0 for success and 1 for error.
    int32_t apply(const int64_t size_y,
                  const int32_t* __restrict__ x,
                        int32_t* __restrict__ y) const override
    {
        #pragma omp parallel default(none) shared(size_y, x, y)
        #pragma omp for simd schedule(static)
        for (int64_t i = 0; i < size_y; i++)
        {
            y[i] = x[i];
        }

        return 0;
    }
};

#endif
