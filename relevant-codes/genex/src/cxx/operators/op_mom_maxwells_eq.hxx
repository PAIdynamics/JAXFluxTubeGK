#ifndef OP_MOM_MAXWELLS_EQ_HXX
#define OP_MOM_MAXWELLS_EQ_HXX

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

// Error flag for op_mom_maxwells_eq_gpu_t
namespace op_mom_maxwells_eq_gpu
{
    inline bool is_erroneous = false;
}

// C++ class which corresponds to Fortran class op_mom_maxwells_eq_gpu_t
class op_mom_maxwells_eq_gpu_t
{
protected:
    // Number of points in sp direction
    int32_t size_sp;
    // Value of PI
    real_t PI;

    // Pointer to the prefactor array for the bps term for every species
    real_t* prefac_bps_ptr;
    // Pointer to the prefactor array the normalization of the b term of
    // Ampere's law for each species
    real_t* prefac_b_amp_ptr;
    // Pointer to the prefactor array containing the normalization of the b term
    // of the quasi-neutrality equation for each species
    real_t* prefac_b_qn_ptr;
    // Pointer to the prefactor array containing the prefactor of the co term
    // of the quasi-neutrality equation for each species
    real_t* prefac_co_qn_ptr;
    // Prefactor of the b term of the parallel Ampere's law
    real_t* prefac_b_bpar_ptr;

    // Check if the device memory is set and used as intended
    inline void dmem_debug(const dmd::mode_t mode)
    {
        bool err = false;

        dmd::start_region("op_mom_maxwells_eq_gpu_t", mode);
        err = err || dmd::is_invalid(this, 1);
        err = err || dmd::is_invalid(this->prefac_bps_ptr, this->size_sp);
        err = err || dmd::is_invalid(this->prefac_b_amp_ptr, this->size_sp);
        err = err || dmd::is_invalid(this->prefac_b_qn_ptr, this->size_sp);
        err = err || dmd::is_invalid(this->prefac_co_qn_ptr, this->size_sp);
        err = err || dmd::is_invalid(this->prefac_b_bpar_ptr, this->size_sp);
        dmd::end_region("op_mom_maxwells_eq_gpu_t");

        op_mom_maxwells_eq_gpu::is_erroneous =
            op_mom_maxwells_eq_gpu::is_erroneous || err;
    }

    // The first compute kernel of the apply() method
    #pragma acc routine seq
    inline void comp_kernel_prior(const int32_t i, const int32_t k,
                                  const int32_t l, const int32_t m,
                                  const int32_t n, const mesh_5d_t& mesh,
                                  const data_array_t<const real_t, 5>& f_in,
                                  data_array_t<real_t, 3>& send_buffer) const
    {
        real_t bps = mesh.absB(i, k) + this->prefac_bps(n) * mesh.vp(l) *
                                       mesh.curl_normb_y(i, k);
        real_t area = mesh.vpw(l) * mesh.muw(m) * bps * this->PI;

        // Calculate co for the quasi-neutrality equation
        send_buffer(i, k, 1) += this->prefac_co_qn(n)
                              * f_in(i, k, l, m, n) * area * mesh.jacobian(i, k)
                              / (mesh.absB(i, k) * mesh.absB(i, k));
        // Calculate b for the quasi-neutrality equation
        send_buffer(i, k, 2) += this->prefac_b_qn(n)
                              * f_in(i, k, l, m, n) * area;
        // Calculate b for Ampere's law
        send_buffer(i, k, 3) += this->prefac_b_amp(n)
                              * mesh.vp(l) * f_in(i, k, l, m, n) * area;
        // Calculate b for bpar equation
        send_buffer(i, k, 4) += this->prefac_b_bpar(n)
                              * mesh.mu(m) * f_in(i, k, l, m, n) * area;
    }

    // The second compute kernel of the apply() method
    #pragma acc routine seq
    inline void comp_kernel_post(
        const int32_t i, const int32_t k, const mesh_5d_t& mesh,
        data_array_t<real_t, 3>& receive_buffer,
        data_array_t<real_t, 2>& co_qn_eq,
        data_array_t<real_t, 2>& b_qn_eq,
        data_array_t<real_t, 2>& b_amps_law,
        data_array_t<real_t, 2>& b_bpar_eq) const
    {
        co_qn_eq(i, k)  = receive_buffer(i, k, 1);
        b_bpar_eq(i, k) = receive_buffer(i, k, 4);
        // NOTE: We leave the values unchanged on the ghost points, and in
        //       the target for b, since these values are already set by the
        //       boundary condition operators
        if (mesh.is_compute(i, k) == 1.0)
        {
            b_qn_eq(i, k)    = receive_buffer(i, k, 2);
            b_amps_law(i, k) = receive_buffer(i, k, 3);
        }
    }

public:
    // Constructor
    op_mom_maxwells_eq_gpu_t(const mesh_5d_t& mesh)
    {
        this->size_sp = mesh.get_size_sp();
        this->PI = math::PI;

        // Allocate class members
        this->prefac_bps_ptr    = new real_t[size_sp]{};
        this->prefac_b_amp_ptr  = new real_t[size_sp]{};
        this->prefac_b_qn_ptr   = new real_t[size_sp]{};
        this->prefac_co_qn_ptr  = new real_t[size_sp]{};
        this->prefac_b_bpar_ptr = new real_t[size_sp]{};

        // Calculate the prefactors
        for (int32_t n = 1; n <= this->size_sp; n++)
        {
            this->prefac_bps_ptr[n - 1] = sqrt(2.0 * pspec::get_mass(n))
                * sqrt(pspec::get_temp_scaling(n))
                * pnorm::get_rho_ref()
                / (pspec::get_charge(n) * pnorm::get_L_ref());
            this->prefac_b_amp_ptr[n - 1] = pnorm::get_beta_ref()
                * pspec::get_charge(n)
                * sqrt(pspec::get_temp_scaling(n) / 2.0 / pspec::get_mass(n));
            this->prefac_b_qn_ptr[n - 1] = pspec::get_charge(n);
            this->prefac_co_qn_ptr[n - 1] = pspec::get_mass(n)
                * pow(pnorm::get_rho_ref() / pnorm::get_L_ref(), 2);
            this->prefac_b_bpar_ptr[n - 1] = -pnorm::get_beta_ref() / 2.0
                * pspec::get_temp_scaling(n);
        }
    }

    // Destructor
    virtual ~op_mom_maxwells_eq_gpu_t()
    {
        delete[] this->prefac_b_bpar_ptr;
        delete[] this->prefac_co_qn_ptr;
        delete[] this->prefac_b_qn_ptr;
        delete[] this->prefac_b_amp_ptr;
        delete[] this->prefac_bps_ptr;
    }

    // Copy constructor is disabled
    op_mom_maxwells_eq_gpu_t(const op_mom_maxwells_eq_gpu_t&) = delete;

    // Copy-assignment operator is disabled
    op_mom_maxwells_eq_gpu_t& operator=(const op_mom_maxwells_eq_gpu_t&)
    = delete;

    // Applies the operator to the given input values
    // Return 0 for success and 1 for error
    virtual int32_t apply(const mesh_5d_t& mesh,
                          const dcomm_handler_t& dcomm_handler,
                          const data_array_t<const real_t, 5>& f_in,
                          data_array_t<real_t, 2>& co_qn_eq,
                          data_array_t<real_t, 2>& b_qn_eq,
                          data_array_t<real_t, 2>& b_amps_law,
                          data_array_t<real_t, 2>& b_bpar_eq) const = 0;

    // Getter for prefac_bps
    #pragma acc routine seq
    inline real_t prefac_bps(int j0) const
    {
        return this->prefac_bps_ptr[j0 - 1];
    }
    // Getter for prefac_b_amp
    #pragma acc routine seq
    inline real_t prefac_b_amp(int j0) const
    {
        return this->prefac_b_amp_ptr[j0 - 1];
    }
    // Getter for prefac_b_qn
    #pragma acc routine seq
    inline real_t prefac_b_qn(int j0) const
    {
        return this->prefac_b_qn_ptr[j0 - 1];
    }
    // Getter for prefac_co_qn
    #pragma acc routine seq
    inline real_t prefac_co_qn(int j0) const
    {
        return this->prefac_co_qn_ptr[j0 - 1];
    }
    // Getter for prefac_b_bpar
    #pragma acc routine seq
    inline real_t prefac_b_bpar(int j0) const
    {
        return this->prefac_b_bpar_ptr[j0 - 1];
    }
};

#ifdef __cplusplus
extern "C" {
#endif

// Return 0 for success and 1 for error
int32_t cbind_op_mom_maxwells_eq_initialize(
    const mesh_5d_t** mesh_cxx_pptr,
    op_mom_maxwells_eq_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_mom_maxwells_eq_finalize(
    op_mom_maxwells_eq_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_mom_maxwells_eq_apply(
    const mesh_5d_t** mesh_cxx_pptr,
    const dcomm_handler_t** dcomm_handler_cxx_pptr,
    const op_mom_maxwells_eq_gpu_t** op_cxx_pptr,
    const data_array_t<const real_t, 5>** f_in_cxx_pptr,
    data_array_t<real_t, 2>** co_qn_eq_cxx_pptr,
    data_array_t<real_t, 2>** b_qn_eq_cxx_pptr,
    data_array_t<real_t, 2>** b_amps_law_cxx_pptr,
    data_array_t<real_t, 2>** b_bpar_eq_cxx_pptr);

#ifdef __cplusplus
}
#endif

#endif
