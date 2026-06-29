#ifndef OP_LIN_COMB_ACC_HXX
#define OP_LIN_COMB_ACC_HXX

#include "genex_cxx_env.hxx"
#include "op_lin_comb.hxx"
#include <openacc.h>

// C++ class which corresponds to the Fortran class op_lin_comb_gpu_t
// with OpenACC on GPU
class op_lin_comb_acc_t: public op_lin_comb_gpu_t
{
public:
    // Default constructor
    op_lin_comb_acc_t() = default;

    // Default destructor
    ~op_lin_comb_acc_t() override = default;

    // Apply the operator to the given input values with OpenACC on GPU
    // Return 0 for success and 1 for error
    int32_t apply(const int64_t size_y, const real_t a1, const real_t a2,
                  const real_t* __restrict__ x1,
                  const real_t* __restrict__ x2,
                        real_t* __restrict__ y) const override
    {
        #pragma acc parallel default(none) copyin(size_y, a1, a2) \
            present(x1, x2, y)
        #pragma acc loop independent gang vector
        for (int64_t i = 0; i < size_y; i++)
        {
            y[i] = a1 * x1[i] + a2 * x2[i];
        }

        return 0;
    }
};

#endif
