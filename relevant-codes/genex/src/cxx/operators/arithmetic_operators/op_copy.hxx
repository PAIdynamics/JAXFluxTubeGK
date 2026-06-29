#ifndef OP_COPY_HXX
#define OP_COPY_HXX

#include "genex_cxx_env.hxx"

// C++ class which corresponds to the Fortran class op_copy_gpu_t
class op_copy_gpu_t
{
public:
    // Default constructor
    op_copy_gpu_t() = default;

    // Default destructor
    virtual ~op_copy_gpu_t() = default;

    // Apply the operator to the given input values for real type
    // Return 0 for success and 1 for error
    virtual int32_t apply(const int64_t size_y,
                          const real_t* __restrict__ x,
                                real_t* __restrict__ y) const = 0;

    // Apply the operator to the given input values for integer type
    // Return 0 for success and 1 for error
    virtual int32_t apply(const int64_t size_y,
                          const int32_t* __restrict__ x,
                                int32_t* __restrict__ y) const = 0;
};

#ifdef __cplusplus
extern "C" {
#endif

// Return 0 for success and 1 for error
int32_t cbind_op_copy_initialize(op_copy_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_copy_finalize(op_copy_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_copy_apply_real(const op_copy_gpu_t** op_cxx_pptr,
    const int64_t size_y, const real_t* x, real_t* y);

// Return 0 for success and 1 for error
int32_t cbind_op_copy_apply_integer(const op_copy_gpu_t** op_cxx_pptr,
    const int64_t size_y, const int32_t* x, int32_t* y);

#ifdef __cplusplus
}
#endif

#endif
