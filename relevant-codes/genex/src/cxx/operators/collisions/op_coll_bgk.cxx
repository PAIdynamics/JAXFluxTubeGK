#include "op_coll_bgk.hxx"
#include "op_coll_bgk_factory.hxx"

int32_t cbind_op_coll_bgk_initialize(
    const dcomm_handler_t** dcomm_handler_cxx_pptr,
    const mesh_5d_t** mesh_cxx_pptr,
    data_array_t<real_t, 2>** collog_cxx_pptr,
    op_coll_bgk_gpu_t** op_cxx_pptr)
{
    // Allocate and construct the C++ class instance of the operator
    const dcomm_handler_t& dcomm_handler = *(*dcomm_handler_cxx_pptr);
    const mesh_5d_t& mesh = *(*mesh_cxx_pptr);
    data_array_t<real_t, 2>& collog = *(*collog_cxx_pptr);
    *op_cxx_pptr = op_coll_bgk_gpu::create(dcomm_handler, mesh, collog);

    return (int32_t) op_coll_bgk_gpu::is_erroneous;
}

int32_t cbind_op_coll_bgk_finalize(
    op_coll_bgk_gpu_t** op_cxx_pptr)
{
    // Assign the BGK collision operator C++ class
    op_coll_bgk_gpu_t& op = *(*op_cxx_pptr);

    // Deallocate the host BGK collision operator C++ class instance
    delete &op;

    return (int32_t) op_coll_bgk_gpu::is_erroneous;
}

int32_t cbind_op_coll_bgk_apply(
    op_coll_bgk_gpu_t** op_cxx_pptr,
    const data_array_t<const real_t, 5>** f_in_cxx_pptr,
    const data_array_t<const real_t, 4>** moments_cxx_pptr,
    data_array_t<real_t, 5>** f_out_cxx_pptr)
{
    // Assign the C++ class instances
    op_coll_bgk_gpu_t& op = *(*op_cxx_pptr);
    const data_array_t<const real_t, 5>& f_in = *(*f_in_cxx_pptr);
    const data_array_t<const real_t, 4>& moments = *(*moments_cxx_pptr);
    data_array_t<real_t, 5>& f_out = *(*f_out_cxx_pptr);

    return op.apply(f_in, moments, f_out);
}
