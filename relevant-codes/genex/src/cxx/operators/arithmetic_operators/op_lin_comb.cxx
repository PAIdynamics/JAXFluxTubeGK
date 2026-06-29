#include "op_lin_comb.hxx"
#include "op_lin_comb_factory.hxx"

int32_t cbind_op_lin_comb_initialize(op_lin_comb_gpu_t** op_cxx_pptr)
{
    bool err = false;

    // Allocate and construct the C++ class instance of the operator
    *op_cxx_pptr = op_lin_comb_gpu::create();
    if (!*op_cxx_pptr) err = true;

    return (int32_t) err;
}

int32_t cbind_op_lin_comb_finalize(op_lin_comb_gpu_t** op_cxx_pptr)
{
    // Assign the op_lin_comb_gpu_t C++ class
    op_lin_comb_gpu_t& op = *(*op_cxx_pptr);

    // Deallocate the host op_lin_comb_gpu_t C++ class instance
    delete &op;

    return 0;
}

int32_t cbind_op_lin_comb_apply(const op_lin_comb_gpu_t** op_cxx_pptr,
    const int64_t size_y, const real_t a1, const real_t a2,
    const real_t* x1, const real_t* x2, real_t* y)
{
    // Assign the op_lin_comb_gpu_t C++ class
    const op_lin_comb_gpu_t& op = *(*op_cxx_pptr);

    return op.apply(size_y, a1, a2, x1, x2, y);
}
