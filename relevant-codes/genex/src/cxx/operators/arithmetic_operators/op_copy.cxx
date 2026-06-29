#include "op_copy.hxx"
#include "op_copy_factory.hxx"

int32_t cbind_op_copy_initialize(op_copy_gpu_t** op_cxx_pptr)
{
    bool err = false;

    // Allocate and construct the C++ class instance of the operator
    *op_cxx_pptr = op_copy_gpu::create();
    if (!*op_cxx_pptr) err = true;

    return (int32_t) err;
}

int32_t cbind_op_copy_finalize(op_copy_gpu_t** op_cxx_pptr)
{
    // Assign the op_copy_gpu_t C++ class
    op_copy_gpu_t& op = *(*op_cxx_pptr);

    // Deallocate the host op_copy_gpu_t C++ class instance
    delete &op;

    return 0;
}

int32_t cbind_op_copy_apply_real(const op_copy_gpu_t** op_cxx_pptr,
    const int64_t size_y, const real_t* x, real_t* y)
{
    // Assign the op_copy_gpu_t C++ class
    const op_copy_gpu_t& op = *(*op_cxx_pptr);

    return op.apply(size_y, x, y);
}

int32_t cbind_op_copy_apply_integer(const op_copy_gpu_t** op_cxx_pptr,
    const int64_t size_y, const int32_t* x, int32_t* y)
{
    // Assign the op_copy_gpu_t C++ class
    const op_copy_gpu_t& op = *(*op_cxx_pptr);

    return op.apply(size_y, x, y);
}
