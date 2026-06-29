#ifndef OP_MOM_OHMS_LAW_HXX
#define OP_MOM_OHMS_LAW_HXX

#include "genex_cxx_env.hxx"
#include "device_memory_debugger.hxx"
#include "mesh_5d.hxx"
#include "dcomm_handler.hxx"
#include "data_array.hxx"
#include "params_species.hxx"
#include "params_normalization.hxx"
#include "math.hxx"
#include "profiler.hxx"
#include <cmath>

// Alias for external namespaces
namespace dmd   = device_memory_debugger;
namespace pspec = params_species;
namespace pnorm = params_normalization;

// Error flag for op_mom_ohms_law_gpu_t
namespace op_mom_ohms_law_gpu
{
    inline bool is_erroneous = false;
}

// C++ class which corresponds to Fortran class op_mom_ohms_law_gpu_t
class op_mom_ohms_law_gpu_t
{
protected:
    // Number of points in sp direction
    int32_t size_sp;
    // Value of PI
    real_t PI;

    // Prefactor for the calculation of the vp derivatives
    real_t prefac_cd_vp;
    // Pointer to the prefactor array for the bps term for every species
    real_t* prefac_bps_ptr;
    // Pointer to the prefactor array containing the normalization for
    // the lambda term
    real_t* prefac_norm_lambda_ptr;
    // Pointer to the prefactor array containing the normalization
    // for the n term
    real_t* prefac_norm_b_ptr;

    // Check if the device memory is set and used as intended
    inline void dmem_debug(const dmd::mode_t mode)
    {
        bool err = false;

        dmd::start_region("op_mom_ohms_law_gpu_t", mode);
        err = err || dmd::is_invalid(this, 1);
        err = err || dmd::is_invalid(this->prefac_bps_ptr, this->size_sp);
        err = err
            || dmd::is_invalid(this->prefac_norm_lambda_ptr, this->size_sp);
        err = err || dmd::is_invalid(this->prefac_norm_b_ptr, this->size_sp);
        dmd::end_region("op_mom_ohms_law_gpu_t");

        op_mom_ohms_law_gpu::is_erroneous =
            op_mom_ohms_law_gpu::is_erroneous || err;
    }

    // The first compute kernel of the apply() method
    #pragma acc routine seq
    inline void comp_kernel_prior(const int32_t i, const int32_t k,
                                  const int32_t l, const int32_t m,
                                  const int32_t n, const mesh_5d_t& mesh,
                                  const data_array_t<const real_t, 5>& f_in,
                                  const data_array_t<const real_t, 5>& dfdt_in,
                                  data_array_t<real_t, 3>& send_buffer) const
    {
        real_t bps = mesh.absB(i, k) + this->prefac_bps(n) * mesh.vp(l) *
                     mesh.curl_normb_y(i, k);
        real_t area = mesh.vpw(l) * mesh.muw(m) * bps * this->PI;

        real_t dfdvp = this->prefac_cd_vp
                     * (- 1.0 / 12.0 * f_in(i, k, l + 2, m, n)
                        + 2.0 / 3.0  * f_in(i, k, l + 1, m, n)
                        - 2.0 / 3.0  * f_in(i, k, l - 1, m, n)
                        + 1.0 / 12.0 * f_in(i, k, l - 2, m, n));
        // Calculate lambda
        send_buffer(i, k, 1) -= this->prefac_norm_lambda(n) * mesh.vp(l)
                              * dfdvp * area;
        // Calculate b
        send_buffer(i, k, 2) += this->prefac_norm_b(n) * mesh.vp(l)
                              * dfdt_in(i, k, l, m, n) * area;
    }

    // The second compute kernel of the apply() method
    #pragma acc routine seq
    inline void comp_kernel_post(
        const int32_t i, const int32_t k, const mesh_5d_t& mesh,
        data_array_t<real_t, 3>& receive_buffer,
        data_array_t<real_t, 2>& lambda_ohms_law,
        data_array_t<real_t, 2>& b_ohms_law) const
    {
        // NOTE: We leave the values unchanged on the ghost points, and in
        //       the target for b, since these values are already set by the
        //       boundary condition operators
        if (mesh.is_compute(i, k) == 1.0)
        {
            b_ohms_law(i, k)      = receive_buffer(i, k, 2);
        }

        int32_t ir = mesh.RZ_indices(i, k);
        if (mesh.is_compute(ir, k) == 1.0)
        {
            lambda_ohms_law(i, k) = receive_buffer(ir, k, 1);
        }
        else if (mesh.not_in_target(ir, k) == 0.0)
        {
            // NOTE: We set the values of lambda to one in the target, since
            //       they are not used by the solver.
            lambda_ohms_law(i, k) = 1.0;
        }
    }

public:
    // Constructor
    op_mom_ohms_law_gpu_t(const mesh_5d_t& mesh)
    {
        this->size_sp = mesh.get_size_sp();
        this->PI = math::PI;
        this->prefac_cd_vp = 1.0 / mesh.get_delta_vp();

        // Allocate class members
        this->prefac_bps_ptr         = new real_t[size_sp]{};
        this->prefac_norm_lambda_ptr = new real_t[size_sp]{};
        this->prefac_norm_b_ptr      = new real_t[size_sp]{};

        // Calculate the prefactors
        for (int32_t n = 1; n <= this->size_sp; n++)
        {
            this->prefac_bps_ptr[n - 1] = sqrt(2.0 * pspec::get_mass(n))
                * sqrt(pspec::get_temp_scaling(n))
                * pnorm::get_rho_ref()
                / (pspec::get_charge(n) * pnorm::get_L_ref());
            this->prefac_norm_lambda_ptr[n - 1] = pnorm::get_beta_ref()
                * pspec::get_charge(n) * pspec::get_charge(n)
                / (2.0 * pspec::get_mass(n));
            this->prefac_norm_b_ptr[n - 1] = pnorm::get_beta_ref()
                * pspec::get_charge(n)
                * sqrt(pspec::get_temp_scaling(n) / 2.0 / pspec::get_mass(n));
        }
    }

    // Destructor
    virtual ~op_mom_ohms_law_gpu_t()
    {
        delete[] this->prefac_norm_b_ptr;
        delete[] this->prefac_norm_lambda_ptr;
        delete[] this->prefac_bps_ptr;
    }

    // Copy constructor is disabled
    op_mom_ohms_law_gpu_t(const op_mom_ohms_law_gpu_t&) = delete;

    // Copy-assignment operator is disabled
    op_mom_ohms_law_gpu_t& operator=(const op_mom_ohms_law_gpu_t&) = delete;

    // Applies the operator to the given input values
    // Return 0 for success and 1 for error
    virtual int32_t apply(const mesh_5d_t& mesh,
                          const dcomm_handler_t& dcomm_handler,
                          const data_array_t<const real_t, 5>& f_in,
                          const data_array_t<const real_t, 5>& dfdt_in,
                          data_array_t<real_t, 2>& lambda_ohms_law,
                          data_array_t<real_t, 2>& b_ohms_law) const = 0;

    // Getter for prefac_bps
    #pragma acc routine seq
    inline real_t prefac_bps(int j0) const
    {
        return this->prefac_bps_ptr[j0 - 1];
    }
    // Getter for prefac_norm_lambda
    #pragma acc routine seq
    inline real_t prefac_norm_lambda(int j0) const
    {
        return this->prefac_norm_lambda_ptr[j0 - 1];
    }
    // Getter for prefac_norm_b
    #pragma acc routine seq
    inline real_t prefac_norm_b(int j0) const
    {
        return this->prefac_norm_b_ptr[j0 - 1];
    }
};

#ifdef __cplusplus
extern "C" {
#endif

// Return 0 for success and 1 for error
int32_t cbind_op_mom_ohms_law_initialize(const mesh_5d_t** mesh_cxx_pptr,
                                         op_mom_ohms_law_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_mom_ohms_law_finalize(op_mom_ohms_law_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_mom_ohms_law_apply(
    const mesh_5d_t** mesh_cxx_pptr,
    const dcomm_handler_t** dcomm_handler_cxx_pptr,
    const op_mom_ohms_law_gpu_t** op_cxx_pptr,
    const data_array_t<const real_t, 5>** f_in_cxx_pptr,
    const data_array_t<const real_t, 5>** dfdt_in_cxx_pptr,
    data_array_t<real_t, 2>** lambda_ohms_law_cxx_pptr,
    data_array_t<real_t, 2>** b_ohms_law_cxx_pptr);

#ifdef __cplusplus
}
#endif

#endif
