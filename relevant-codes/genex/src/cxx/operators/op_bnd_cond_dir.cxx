#include "op_bnd_cond_dir.hxx"
#include "op_bnd_cond_dir_factory.hxx"

int32_t cbind_op_bnd_cond_dir_initialize(
    const mesh_5d_t** mesh_cxx_pptr,
    op_bnd_cond_dir_gpu_t** op_cxx_pptr)
{
    // Allocate and construct the C++ class instance of the operator
    const mesh_5d_t& mesh = *(*mesh_cxx_pptr);
    *op_cxx_pptr = op_bnd_cond_dir_gpu::create(mesh);

    return (int32_t) op_bnd_cond_dir_gpu::is_erroneous;
}

int32_t cbind_op_bnd_cond_dir_finalize(
    op_bnd_cond_dir_gpu_t** op_cxx_pptr)
{
    // Assign the Dirichlet boundary condition operator C++ class
    op_bnd_cond_dir_gpu_t& op = *(*op_cxx_pptr);

    // Deallocate the host Dirichlet boundary condition operator
    // C++ class instance
    delete &op;

    return (int32_t) op_bnd_cond_dir_gpu::is_erroneous;
}

int32_t cbind_op_bnd_cond_dir_apply(
    op_bnd_cond_dir_gpu_t** op_cxx_pptr,
    data_array_t<real_t, 5>** f_inout_cxx_pptr,
    data_array_t<real_t, 2>** b_qn_eq_cxx_pptr,
    data_array_t<real_t, 2>** b_amps_law_cxx_pptr,
    data_array_t<real_t, 2>** b_ohms_law_cxx_pptr)
{
    // Assign the C++ class instances
    op_bnd_cond_dir_gpu_t& op = *(*op_cxx_pptr);
    data_array_t<real_t, 5>& f_inout = *(*f_inout_cxx_pptr);
    data_array_t<real_t, 2>& b_qn_eq = *(*b_qn_eq_cxx_pptr);
    data_array_t<real_t, 2>& b_amps_law = *(*b_amps_law_cxx_pptr);
    data_array_t<real_t, 2>& b_ohms_law = *(*b_ohms_law_cxx_pptr);

    return op.apply(f_inout, b_qn_eq, b_amps_law, b_ohms_law);
}
