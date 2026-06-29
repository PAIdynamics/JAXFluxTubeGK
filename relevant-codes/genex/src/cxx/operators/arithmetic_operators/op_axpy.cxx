#include "op_axpy.hxx"
#include "op_axpy_factory.hxx"

int32_t cbind_op_axpy_initialize(op_axpy_gpu_t** op_cxx_pptr)
{
    bool err = false;

    // Allocate and construct the C++ class instance of the operator
    *op_cxx_pptr = op_axpy_gpu::create();
    if (!*op_cxx_pptr) err = true;

    return (int32_t) err;
}

int32_t cbind_op_axpy_finalize(op_axpy_gpu_t** op_cxx_pptr)
{
    // Assign the op_axpy_gpu_t C++ class
    op_axpy_gpu_t& op = *(*op_cxx_pptr);

    // Deallocate the host op_axpy_gpu_t C++ class instance
    delete &op;

    return 0;
}

int32_t cbind_op_axpy_apply(const op_axpy_gpu_t** op_cxx_pptr,
                            const int64_t size_y, const real_t a,
                            const real_t* x, real_t* y)
{
    // Assign the op_axpy_gpu_t C++ class
    const op_axpy_gpu_t& op = *(*op_cxx_pptr);

    return op.apply(size_y, a, x, y);
}
