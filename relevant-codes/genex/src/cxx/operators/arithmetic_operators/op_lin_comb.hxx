#ifndef OP_LIN_COMB_HXX
#define OP_LIN_COMB_HXX

#include "genex_cxx_env.hxx"

// C++ class which corresponds to the Fortran class op_lin_comb_gpu_t
class op_lin_comb_gpu_t
{
public:
    // Default constructor
    op_lin_comb_gpu_t() = default;

    // Default destructor
    virtual ~op_lin_comb_gpu_t() = default;

    // Apply the operator to the given input values
    // Return 0 for success and 1 for error
    virtual int32_t apply(const int64_t size_y, const real_t a1,
                          const real_t a2,
                          const real_t* __restrict__ x1,
                          const real_t* __restrict__ x2,
                                real_t* __restrict__ y) const = 0;
};

#ifdef __cplusplus
extern "C" {
#endif

// Return 0 for success and 1 for error
int32_t cbind_op_lin_comb_initialize(op_lin_comb_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_lin_comb_finalize(op_lin_comb_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_lin_comb_apply(const op_lin_comb_gpu_t** op_cxx_pptr,
                                const int64_t size_y, const real_t a1,
                                const real_t a2, const real_t* x1,
                                const real_t* x2, real_t* y);

#ifdef __cplusplus
}
#endif

#endif
