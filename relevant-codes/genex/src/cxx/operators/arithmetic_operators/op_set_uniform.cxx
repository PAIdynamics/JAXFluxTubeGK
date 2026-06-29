#include "op_set_uniform.hxx"
#include "op_set_uniform_factory.hxx"

int32_t cbind_op_set_uniform_initialize(op_set_uniform_gpu_t** op_cxx_pptr)
{
    bool err = false;

    // Allocate and construct the C++ class instance of the operator
    *op_cxx_pptr = op_set_uniform_gpu::create();
    if (!*op_cxx_pptr) err = true;

    return (int32_t) err;
}

int32_t cbind_op_set_uniform_finalize(op_set_uniform_gpu_t** op_cxx_pptr)
{
    // Assign the op_set_uniform_gpu_t C++ class
    op_set_uniform_gpu_t& op = *(*op_cxx_pptr);

    // Deallocate the host op_set_uniform_gpu_t C++ class instance
    delete &op;

    return 0;
}

int32_t cbind_op_set_uniform_apply_real(
    const op_set_uniform_gpu_t** op_cxx_pptr, const int64_t size_y,
    const real_t a, real_t* y)
{
    // Assign the op_set_uniform_gpu_t C++ class
    const op_set_uniform_gpu_t& op = *(*op_cxx_pptr);

    return op.apply(size_y, a, y);
}

int32_t cbind_op_set_uniform_apply_integer(
    const op_set_uniform_gpu_t** op_cxx_pptr, const int64_t size_y,
    const int32_t a, int32_t* y)
{
    // Assign the op_set_uniform_gpu_t C++ class
    const op_set_uniform_gpu_t& op = *(*op_cxx_pptr);

    return op.apply(size_y, a, y);
}
