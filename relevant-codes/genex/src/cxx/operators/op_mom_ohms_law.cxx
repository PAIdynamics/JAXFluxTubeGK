#include "op_mom_ohms_law.hxx"
#include "op_mom_ohms_law_factory.hxx"

int32_t cbind_op_mom_ohms_law_initialize(const mesh_5d_t** mesh_cxx_pptr,
                                         op_mom_ohms_law_gpu_t** op_cxx_pptr)
{
    // Assign the mesh_5d_t class instance
    const mesh_5d_t& mesh = *(*mesh_cxx_pptr);

    // Allocate and construct the C++ class instance of the operator
    *op_cxx_pptr = op_mom_ohms_law_gpu::create(mesh);

    return (int32_t) op_mom_ohms_law_gpu::is_erroneous;
}

int32_t cbind_op_mom_ohms_law_finalize(op_mom_ohms_law_gpu_t** op_cxx_pptr)
{
    // Assign the op_mom_ohms_law_gpu_t C++ class
    op_mom_ohms_law_gpu_t& op = *(*op_cxx_pptr);

    // Deallocate the host op_mom_ohms_law_gpu_t C++ class instance
    delete &op;

    return (int32_t) op_mom_ohms_law_gpu::is_erroneous;
}

int32_t cbind_op_mom_ohms_law_apply(
    const mesh_5d_t** mesh_cxx_pptr,
    const dcomm_handler_t** dcomm_handler_cxx_pptr,
    const op_mom_ohms_law_gpu_t** op_cxx_pptr,
    const data_array_t<const real_t, 5>** f_in_cxx_pptr,
    const data_array_t<const real_t, 5>** dfdt_in_cxx_pptr,
    data_array_t<real_t, 2>** lambda_ohms_law_cxx_pptr,
    data_array_t<real_t, 2>** b_ohms_law_cxx_pptr)
{
    // Assign the C++ class instances
    const mesh_5d_t& mesh = *(*mesh_cxx_pptr);
    const dcomm_handler_t& dcomm_handler = *(*dcomm_handler_cxx_pptr);
    const op_mom_ohms_law_gpu_t& op = *(*op_cxx_pptr);
    const data_array_t<const real_t, 5>& f_in = *(*f_in_cxx_pptr);
    const data_array_t<const real_t, 5>& dfdt_in = *(*dfdt_in_cxx_pptr);
    data_array_t<real_t, 2>& lambda_ohms_law = *(*lambda_ohms_law_cxx_pptr);
    data_array_t<real_t, 2>& b_ohms_law = *(*b_ohms_law_cxx_pptr);

    return op.apply(mesh, dcomm_handler, f_in, dfdt_in, lambda_ohms_law,
                    b_ohms_law);
}
