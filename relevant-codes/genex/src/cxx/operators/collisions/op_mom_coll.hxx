#ifndef OP_MOM_COLL_HXX
#define OP_MOM_COLL_HXX

#include "genex_cxx_env.hxx"
#include "device_memory_debugger.hxx"
#include "params_species.hxx"
#include "params_collisions.hxx"
#include "params_normalization.hxx"
#include "math.hxx"
#include "dcomm_handler.hxx"
#include "mesh_5d.hxx"
#include "data_array.hxx"
#include "profiler.hxx"
#include <cmath>

// Alias for external namespaces
namespace dmd   = device_memory_debugger;
namespace pspec = params_species;
namespace pcoll = params_collisions;
namespace pnorm = params_normalization;

// Error flag for op_mom_coll_gpu_t
namespace op_mom_coll_gpu
{
    inline bool is_erroneous = false;
}

// C++ class which corresponds to Fortran class op_mom_coll_gpu_t
class op_mom_coll_gpu_t
{
protected:
    // Number of points in mu direction
    int32_t size_mu;
    // Number of points in sp direction
    int32_t size_sp;
    // Value of PI
    real_t PI;
    // Minimum limit of the density
    real_t dens_floor;
    // Minimum limit of the temperature
    real_t temp_floor;
    // Pointer of prefactor for B parallel star
    real_t* prefac_bps_ptr;

    // Check if the device memory is set and used as intended
    inline void dmem_debug(const dmd::mode_t mode)
    {
        bool err = false;

        dmd::start_region("op_mom_coll_gpu_t", mode);
        err = err || dmd::is_invalid(this, 1);
        err = err || dmd::is_invalid(this->prefac_bps_ptr, this->size_sp);
        dmd::end_region("op_mom_coll_gpu_t");

        op_mom_coll_gpu::is_erroneous = op_mom_coll_gpu::is_erroneous || err;
    }

    // The first compute kernel of the apply() method
    #pragma acc routine seq
    inline void comp_kernel_prior(const int32_t i, const int32_t k,
                                  const int32_t l, const int32_t m,
                                  const int32_t n, const mesh_5d_t& mesh,
                                  const data_array_t<const real_t, 5>& f_in,
                                  data_array_t<real_t, 4>& send_buffer) const
    {
        // B parallel star
        real_t bps  = mesh.absB(i, k) + this->prefac_bps(n) * mesh.vp(l)
                    * mesh.curl_normb_y(i, k);
        // Vspace area element
        real_t area = mesh.vpw(l) * mesh.muw(m) * bps * this->PI;
        // Density
        send_buffer(i, k, 1, n) += area * f_in(i, k, l, m, n);
        // Parallel drift velocity
        send_buffer(i, k, 2, n) += area * mesh.vp(l) * f_in(i, k, l, m, n);
        // Temperature
        send_buffer(i, k, 3, n) += area * f_in(i, k, l, m, n)
                                 * ( mesh.vp(l) * mesh.vp(l)
                                   + mesh.absB(i, k) * mesh.mu(m));
    }

    // The second compute kernel of the apply() method
    #pragma acc routine seq
    inline void comp_kernel_post(
        const int32_t i, const int32_t k, const int32_t n,
        data_array_t<real_t, 4>& moments) const
    {
        // Density floor: take larger value (i.e. prevent value zero)
        moments(i, k, 1, n) = fmax(moments(i, k, 1, n), this->dens_floor);
        // V par normalized by density
        moments(i, k, 2, n) /= moments(i, k, 1, n);
        // 2nd moment normalized by density + correction by
        // V par squared
        moments(i, k, 3, n) = fmax(2.0 / 3.0 * ( moments(i, k, 3, n)
                                               / moments(i, k, 1, n)
                                               - ( moments(i, k, 2, n)
                                                 * moments(i, k, 2, n))),
                                   this->temp_floor);
    }

public:
    // Constructor
    op_mom_coll_gpu_t(const dcomm_handler_t& dcomm_handler,
                      const mesh_5d_t& mesh)
    {
        this->size_mu = mesh.get_size_mu();
        this->size_sp = mesh.get_size_sp();
        this->PI = math::PI;
        this->dens_floor = pcoll::get_dens_floor();
        this->temp_floor = pcoll::get_temp_floor();

        // Allocate class members
        this->prefac_bps_ptr   = new real_t[size_sp] {};

        // Calculate prefactor for moment calculation
        for (int32_t n = 1; n <= this->size_sp; n++)
        {
            this->prefac_bps_ptr[n - 1] = sqrt(2.0 * pspec::get_mass(n))
                * sqrt(pspec::get_temp_scaling(n))
                * pnorm::get_rho_ref()
                / ( pspec::get_charge(n) * pnorm::get_L_ref());
        }
    }

    // Destructor
    virtual ~op_mom_coll_gpu_t()
    {
        delete[] this->prefac_bps_ptr;
    }

    // Copy constructor is disabled
    op_mom_coll_gpu_t(const op_mom_coll_gpu_t&) = delete;

    // Copy-assignment operator is disabled
    op_mom_coll_gpu_t& operator=(const op_mom_coll_gpu_t&) = delete;

    // Applies the operator to the given input values
    // Return 0 for success and 1 for error
    virtual int32_t apply(const dcomm_handler_t& dcomm_handler,
                          const mesh_5d_t& mesh,
                          const data_array_t<const real_t, 5>& f_in,
                          data_array_t<real_t, 4>& moments) const = 0;

    // Getter for prefac_bps
    #pragma acc routine seq
    inline real_t prefac_bps(int32_t n) const
    {
        return this->prefac_bps_ptr[n - 1];
    }
};

#ifdef __cplusplus
extern "C" {
#endif

// Return 0 for success and 1 for error
int32_t cbind_op_mom_coll_initialize(
    const dcomm_handler_t** dcomm_handler_cxx_pptr,
    const mesh_5d_t** mesh_cxx_pptr,
    op_mom_coll_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_mom_coll_finalize(op_mom_coll_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_mom_coll_apply(
    const dcomm_handler_t** dcomm_handler_cxx_pptr,
    const mesh_5d_t** mesh_cxx_pptr,
    const op_mom_coll_gpu_t** op_cxx_pptr,
    const data_array_t<const real_t, 5>** f_in_cxx_pptr,
    data_array_t<real_t, 4>** moments_cxx_pptr);

#ifdef __cplusplus
}
#endif

#endif
