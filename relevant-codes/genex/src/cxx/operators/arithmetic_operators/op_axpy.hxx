#ifndef OP_AXPY_HXX
#define OP_AXPY_HXX

#include "genex_cxx_env.hxx"

// C++ class which corresponds to the Fortran class op_axpy_gpu_t
class op_axpy_gpu_t
{
public:
    // Default constructor
    op_axpy_gpu_t() = default;

    // Default destructor
    virtual ~op_axpy_gpu_t() = default;

    // Apply the operator to the given input values
    // Return 0 for success and 1 for error
    virtual int32_t apply(const int64_t size_y, const real_t a,
                          const real_t* __restrict__ x,
                                real_t* __restrict__ y) const = 0;
};

#ifdef __cplusplus
extern "C" {
#endif

// Return 0 for success and 1 for error
int32_t cbind_op_axpy_initialize(op_axpy_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_axpy_finalize(op_axpy_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_axpy_apply(const op_axpy_gpu_t** op_cxx_pptr,
                            const int64_t size_y, const real_t a,
                            const real_t* x, real_t* y);

#ifdef __cplusplus
}
#endif

#endif
