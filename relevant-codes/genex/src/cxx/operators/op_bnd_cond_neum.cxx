#include "op_bnd_cond_neum.hxx"
#include "op_bnd_cond_neum_factory.hxx"

int32_t cbind_op_bnd_cond_neum_initialize(
    const dcomm_handler_t** dcomm_handler_cxx_pptr,
    const mesh_5d_t** mesh_cxx_pptr,
    real_t rho_center,
    real_t* is_core_ptr,
    real_t* rho_ptr,
    op_bnd_cond_neum_gpu_t** op_cxx_pptr)
{
    // Allocate and construct the C++ class instance of the operator
    const dcomm_handler_t& dcomm_handler = *(*dcomm_handler_cxx_pptr);
    const mesh_5d_t& mesh = *(*mesh_cxx_pptr);
    *op_cxx_pptr = op_bnd_cond_neum_gpu::create(dcomm_handler, mesh, rho_center,
                                                is_core_ptr, rho_ptr);

    return (int32_t) op_bnd_cond_neum_gpu::is_erroneous;
}

int32_t cbind_op_bnd_cond_neum_finalize(
    op_bnd_cond_neum_gpu_t** op_cxx_pptr)
{
    // Assign the Dirichlet boundary condition operator C++ class
    op_bnd_cond_neum_gpu_t& op = *(*op_cxx_pptr);

    // Deallocate the host Dirichlet boundary condition operator
    // C++ class instance
    delete &op;

    return (int32_t) op_bnd_cond_neum_gpu::is_erroneous;
}

int32_t cbind_op_bnd_cond_neum_apply(
    op_bnd_cond_neum_gpu_t** op_cxx_pptr,
    data_array_t<real_t, 5>** f_inout_cxx_pptr,
    data_array_t<real_t, 2>** b_qn_eq_cxx_pptr,
    data_array_t<real_t, 2>** b_amps_law_cxx_pptr,
    data_array_t<real_t, 2>** b_ohms_law_cxx_pptr)
{
    // Assign the C++ class instances
    op_bnd_cond_neum_gpu_t& op = *(*op_cxx_pptr);
    data_array_t<real_t, 5>& f_inout = *(*f_inout_cxx_pptr);
    data_array_t<real_t, 2>& b_qn_eq = *(*b_qn_eq_cxx_pptr);
    data_array_t<real_t, 2>& b_amps_law = *(*b_amps_law_cxx_pptr);
    data_array_t<real_t, 2>& b_ohms_law = *(*b_ohms_law_cxx_pptr);

    return op.apply(f_inout, b_qn_eq, b_amps_law, b_ohms_law);
}
