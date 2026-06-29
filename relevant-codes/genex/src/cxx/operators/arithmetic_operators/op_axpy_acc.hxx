#ifndef OP_AXPY_ACC_HXX
#define OP_AXPY_ACC_HXX

#include "genex_cxx_env.hxx"
#include "op_axpy.hxx"
#include <openacc.h>

// C++ class which corresponds to the Fortran class op_axpy_gpu_t
// with OpenACC on GPU
class op_axpy_acc_t: public op_axpy_gpu_t
{
public:
    // Default constructor
    op_axpy_acc_t() = default;

    // Default destructor
    ~op_axpy_acc_t() override = default;

    // Apply the operator to the given input values with OpenACC on GPU
    // Return 0 for success and 1 for error
    int32_t apply(const int64_t size_y, const real_t a,
                  const real_t* __restrict__ x,
                        real_t* __restrict__ y) const override
    {
        #pragma acc parallel default(none) copyin(size_y, a) present(x, y)
        #pragma acc loop independent gang vector
        for (int64_t i = 0; i < size_y; i++)
        {
            y[i] += a * x[i];
        }

        return 0;
    }
};

#endif
