#ifndef OP_SET_UNIFORM_ACC_HXX
#define OP_SET_UNIFORM_ACC_HXX

#include "genex_cxx_env.hxx"
#include "op_set_uniform.hxx"
#include <openacc.h>

// C++ class which corresponds to the Fortran class op_set_uniform_gpu_t
// with OpenACC on GPU
class op_set_uniform_acc_t: public op_set_uniform_gpu_t
{
public:
    // Default constructor
    op_set_uniform_acc_t() = default;

    // Default destructor
    virtual ~op_set_uniform_acc_t() = default;

    // Apply the operator to the given input values for real type
    // with OpenACC on GPU. Return 0 for success and 1 for error.
    int32_t apply(const int64_t size_y, const real_t a,
                  real_t* __restrict__ y) const override
    {
        #pragma acc parallel default(none) copyin(size_y, a) present(y)
        #pragma acc loop independent gang vector
        for (int64_t i = 0; i < size_y; i++)
        {
            y[i] = a;
        }

        return 0;
    }

    // Apply the operator to the given input values for integer type
    // with OpenACC on GPU. Return 0 for success and 1 for error.
    int32_t apply(const int64_t size_y, const int32_t a,
                  int32_t* __restrict__ y) const override
    {
        #pragma acc parallel default(none) copyin(size_y, a) present(y)
        #pragma acc loop independent gang vector
        for (int64_t i = 0; i < size_y; i++)
        {
            y[i] = a;
        }

        return 0;
    }
};

#endif
