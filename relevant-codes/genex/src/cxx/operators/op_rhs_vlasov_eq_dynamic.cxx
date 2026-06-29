#include "op_rhs_vlasov_eq_dynamic.hxx"
#include "op_rhs_vlasov_eq_dynamic_factory.hxx"

int32_t cbind_op_vlasov_dynamic_initialize(
    const mesh_5d_t** mesh_cxx_ptr,
    op_rhs_vlasov_eq_dynamic_gpu_t** op_cxx_pptr)
{
    // Assign the mesh_5d_t class instance
    const mesh_5d_t& mesh = *(*mesh_cxx_ptr);

    // Allocate and construct the C++ class instance of the operator
    *op_cxx_pptr = op_rhs_vlasov_eq_dynamic_gpu::create(mesh);

    return (int32_t) op_rhs_vlasov_eq_dynamic_gpu::is_erroneous;
}

int32_t cbind_op_vlasov_dynamic_finalize(
    op_rhs_vlasov_eq_dynamic_gpu_t** op_cxx_pptr)
{
    // Assign the dynamic operator C++ class
    op_rhs_vlasov_eq_dynamic_gpu_t& op = *(*op_cxx_pptr);

    // Deallocate the host dynamic operator C++ class instance
    delete &op;

    return (int32_t) op_rhs_vlasov_eq_dynamic_gpu::is_erroneous;
}

int32_t cbind_op_vlasov_dynamic_apply(
    const op_rhs_vlasov_eq_dynamic_gpu_t** op_cxx_pptr,
    const mesh_5d_t** mesh_cxx_ptr,
    const data_array_t<const real_t, 5>** f_in_cxx_pptr,
    const data_array_t<const real_t, 2>** E_par_in_cxx_pptr,
    data_array_t<real_t, 5>** f_out_cxx_pptr)
{
    // Assign the C++ class instances
    const mesh_5d_t& mesh = *(*mesh_cxx_ptr);
    const op_rhs_vlasov_eq_dynamic_gpu_t& op = *(*op_cxx_pptr);
    const data_array_t<const real_t, 5>& f_in = *(*f_in_cxx_pptr);
    const data_array_t<const real_t, 2>& E_par_in = *(*E_par_in_cxx_pptr);
    data_array_t<real_t, 5>& f_out = *(*f_out_cxx_pptr);

    return op.apply(mesh, f_in, E_par_in, f_out);
}
