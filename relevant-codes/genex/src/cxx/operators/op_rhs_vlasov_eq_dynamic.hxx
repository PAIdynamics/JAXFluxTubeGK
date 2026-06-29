#ifndef OP_RHS_VLASOV_EQ_DYNAMIC_HXX
#define OP_RHS_VLASOV_EQ_DYNAMIC_HXX

#include "genex_cxx_env.hxx"
#include "device_memory_debugger.hxx"
#include "mesh_5d.hxx"
#include "data_array.hxx"
#include "params_species.hxx"
#include <cmath>

// Alias for external namespaces
namespace dmd   = device_memory_debugger;
namespace pspec = params_species;

// Error flag for op_rhs_vlasov_eq_dynamic_gpu_t
namespace op_rhs_vlasov_eq_dynamic_gpu
{
    inline bool is_erroneous = false;
}

// C++ class which corresponds to Fortran class op_rhs_vlasov_eq_dynamic_gpu_t
class op_rhs_vlasov_eq_dynamic_gpu_t
{
protected:
    // Number of points in sp direction
    int32_t size_sp;
    // Prefactor for the calculation of the vp centered difference stencil
    real_t  prefac_cd_vp;
    // Pointer to the normalization prefactor array of the induction term
    real_t* prefac_norm_ptr;

    // Check if the device memory is set and used as intended
    inline void dmem_debug(const dmd::mode_t mode)
    {
        bool err = false;

        dmd::start_region("op_rhs_vlasov_eq_dynamic_gpu_t", mode);
        err = err || dmd::is_invalid(this, 1);
        err = err || dmd::is_invalid(this->prefac_norm_ptr, this->size_sp);
        dmd::end_region("op_rhs_vlasov_eq_dynamic_gpu_t");

        op_rhs_vlasov_eq_dynamic_gpu::is_erroneous =
            op_rhs_vlasov_eq_dynamic_gpu::is_erroneous || err;
    }

    // The compute kernel of the apply() method
    #pragma acc routine seq
    inline void comp_kernel(const int32_t i, const int32_t k,
                            const int32_t l, const int32_t m,
                            const int32_t n, const mesh_5d_t& mesh,
                            const data_array_t<const real_t, 5>& f_in,
                            const data_array_t<const real_t, 2>& E_par_in,
                            data_array_t<real_t, 5>& f_out) const
    {
        // Calculate vp derivatives
        real_t dfdvp = this->prefac_cd_vp
                     * (- 1.0 / 12.0 * f_in(i, k, l + 2, m, n)
                        + 2.0 / 3.0  * f_in(i, k, l + 1, m, n)
                        - 2.0 / 3.0  * f_in(i, k, l - 1, m, n)
                        + 1.0 / 12.0 * f_in(i, k, l - 2, m, n));
        // Calculate the electromagnetic induction
        f_out(i, k, l, m, n) += mesh.is_compute(i, k) * this->prefac_norm(n)
                              * E_par_in(i, k) * dfdvp;
    }

public:
    // Constructor
    op_rhs_vlasov_eq_dynamic_gpu_t(const mesh_5d_t& mesh)
    {
        this->size_sp = mesh.get_size_sp();
        this->prefac_cd_vp = 1.0 / mesh.get_delta_vp();

        this->prefac_norm_ptr = new real_t[size_sp]{};
        for (int32_t n = 1; n <= this->size_sp; n++)
        {
            this->prefac_norm_ptr[n - 1] = pspec::get_charge(n)
                                         / sqrt(2.0 * pspec::get_mass(n)
                                                * pspec::get_temp_scaling(n));
        }
    }

    // Destructor
    virtual ~op_rhs_vlasov_eq_dynamic_gpu_t()
    {
        delete[] this->prefac_norm_ptr;
    }

    // Copy constructor is disabled
    op_rhs_vlasov_eq_dynamic_gpu_t(const op_rhs_vlasov_eq_dynamic_gpu_t&)
        = delete;

    // Copy-assignment operator is disabled
    op_rhs_vlasov_eq_dynamic_gpu_t&
        operator=(const op_rhs_vlasov_eq_dynamic_gpu_t&) = delete;

    // Applies the operator to the given input values
    // Return 0 for success and 1 for error
    virtual int32_t apply(const mesh_5d_t& mesh,
                          const data_array_t<const real_t, 5>& f_in,
                          const data_array_t<const real_t, 2>& E_par_in,
                          data_array_t<real_t, 5>& f_out) const = 0;

    // Getter for prefac_norm
    #pragma acc routine seq
    inline real_t prefac_norm(int j0) const
    {
        return this->prefac_norm_ptr[j0 - 1];
    }
};

#ifdef __cplusplus
extern "C" {
#endif

// Return 0 for success and 1 for error
int32_t cbind_op_vlasov_dynamic_initialize(
    const mesh_5d_t** mesh_cxx_ptr,
    op_rhs_vlasov_eq_dynamic_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_vlasov_dynamic_finalize(
    op_rhs_vlasov_eq_dynamic_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_vlasov_dynamic_apply(
    const op_rhs_vlasov_eq_dynamic_gpu_t** op_cxx_pptr,
    const mesh_5d_t** mesh_cxx_ptr,
    const data_array_t<const real_t, 5>** f_in_cxx_pptr,
    const data_array_t<const real_t, 2>** E_par_in_cxx_pptr,
    data_array_t<real_t, 5>** f_out_cxx_pptr);

#ifdef __cplusplus
}
#endif

#endif
