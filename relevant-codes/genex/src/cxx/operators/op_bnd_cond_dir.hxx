#ifndef OP_BND_COND_DIR_HXX
#define OP_BND_COND_DIR_HXX

#include "genex_cxx_env.hxx"
#include "mesh_5d.hxx"
#include "data_array.hxx"

// Error flag for op_bnd_cond_dir_gpu_t
namespace op_bnd_cond_dir_gpu
{
    inline bool is_erroneous = false;
}

// C++ class which corresponds to Fortran class op_bnd_cond_dir_gpu_t
class op_bnd_cond_dir_gpu_t
{
protected:
    // Mesh 5D
    const mesh_5d_t& mesh;

    // The compute kernel of the apply() method
    #pragma acc routine seq
    inline void comp_kernel(const int32_t i, const int32_t k,
                            const mesh_5d_t& mesh,
                            data_array_t<real_t, 2>& b_qn_eq,
                            data_array_t<real_t, 2>& b_amps_law,
                            data_array_t<real_t, 2>& b_ohms_law) const
    {
        b_qn_eq(i, k)    *= mesh.is_compute(i, k);
        b_amps_law(i, k) *= mesh.is_compute(i, k);
        b_ohms_law(i, k) *= mesh.is_compute(i, k);
    }

    // Check if the device memory is set and used as intended
    inline void dmem_debug(const dmd::mode_t mode)
    {
        bool err = false;

        dmd::start_region("op_bnd_cond_dir_gpu_t", mode);
        err = err || dmd::is_invalid(this, 1);
        dmd::end_region("op_bnd_cond_dir_gpu_t");

        op_bnd_cond_dir_gpu::is_erroneous =
            op_bnd_cond_dir_gpu::is_erroneous || err;
    }

public:
    // Constructor
    op_bnd_cond_dir_gpu_t(const mesh_5d_t& _mesh) : mesh{_mesh} {};

    // Destructor
    virtual ~op_bnd_cond_dir_gpu_t() {};

    // Copy constructor is disabled
    op_bnd_cond_dir_gpu_t(const op_bnd_cond_dir_gpu_t&) = delete;

    // Copy-assignment operator is disabled
    op_bnd_cond_dir_gpu_t& operator=(const op_bnd_cond_dir_gpu_t&) = delete;

    // Applies the operator to the given input values
    // Return 0 for success and 1 for error
    virtual int32_t apply(data_array_t<real_t, 5>& f_inout,
                          data_array_t<real_t, 2>& b_qn_eq,
                          data_array_t<real_t, 2>& b_amps_law,
                          data_array_t<real_t, 2>& b_ohms_law) = 0;
};

#ifdef __cplusplus
extern "C" {
#endif

// Return 0 for success and 1 for error
int32_t cbind_op_bnd_cond_dir_initialize(
    const mesh_5d_t** mesh_cxx_pptr,
    op_bnd_cond_dir_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_bnd_cond_dir_finalize(op_bnd_cond_dir_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_bnd_cond_dir_apply(
    op_bnd_cond_dir_gpu_t** op_cxx_pptr,
    data_array_t<real_t, 5>** f_inout_cxx_pptr,
    data_array_t<real_t, 2>** b_qn_eq_cxx_pptr,
    data_array_t<real_t, 2>** b_amps_law_cxx_pptr,
    data_array_t<real_t, 2>** b_ohms_law_cxx_pptr);

#ifdef __cplusplus
}
#endif

#endif
