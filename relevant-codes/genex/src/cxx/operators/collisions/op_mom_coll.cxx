#include "op_mom_coll.hxx"
#include "op_mom_coll_factory.hxx"

int32_t cbind_op_mom_coll_initialize(
    const dcomm_handler_t** dcomm_handler_cxx_pptr,
    const mesh_5d_t** mesh_cxx_pptr,
    op_mom_coll_gpu_t** op_cxx_pptr)
{
    // Allocate and construct the C++ class instance of the operator
    const dcomm_handler_t& dcomm_handler = *(*dcomm_handler_cxx_pptr);
    const mesh_5d_t& mesh = *(*mesh_cxx_pptr);
    *op_cxx_pptr = op_mom_coll_gpu::create(dcomm_handler, mesh);

    return (int32_t) op_mom_coll_gpu::is_erroneous;
}

int32_t cbind_op_mom_coll_finalize(
    op_mom_coll_gpu_t** op_cxx_pptr)
{
    // Assign the moment operator C++ class for the collision
    op_mom_coll_gpu_t& op = *(*op_cxx_pptr);

    // Deallocate the host moment operator C++ class instance for the collision
    delete &op;

    return (int32_t) op_mom_coll_gpu::is_erroneous;
}

int32_t cbind_op_mom_coll_apply(
    const dcomm_handler_t** dcomm_handler_cxx_pptr,
    const mesh_5d_t** mesh_cxx_pptr,
    const op_mom_coll_gpu_t** op_cxx_pptr,
    const data_array_t<const real_t, 5>** f_in_cxx_pptr,
    data_array_t<real_t, 4>** moments_cxx_pptr)
{
    // Assign the C++ class instances
    const dcomm_handler_t& dcomm_handler = *(*dcomm_handler_cxx_pptr);
    const mesh_5d_t& mesh = *(*mesh_cxx_pptr);
    const op_mom_coll_gpu_t& op = *(*op_cxx_pptr);
    const data_array_t<const real_t, 5>& f_in = *(*f_in_cxx_pptr);
    data_array_t<real_t, 4>& moments = *(*moments_cxx_pptr);

    return op.apply(dcomm_handler, mesh, f_in, moments);
}
