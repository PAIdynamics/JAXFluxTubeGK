#ifndef OP_COLL_BGK_HXX
#define OP_COLL_BGK_HXX

#include "genex_cxx_env.hxx"
#include "device_memory_debugger.hxx"
#include "params_species.hxx"
#include "params_collisions.hxx"
#include "params_normalization.hxx"
#include "math.hxx"
#include "dcomm_handler.hxx"
#include "mesh_5d.hxx"
#include "data_array.hxx"
#include <cmath>

// Alias for external namespaces
namespace dmm   = device_memory_debugger;
namespace pspec = params_species;
namespace pcoll = params_collisions;
namespace pnorm = params_normalization;

// Error flag for op_coll_bgk_gpu_t
namespace op_coll_bgk_gpu
{
    inline bool is_erroneous = false;
}

// C++ class which corresponds to Fortran class op_coll_bgk_gpu_t
class op_coll_bgk_gpu_t
{
protected:
    // Boolean to keep track of the allocation of Coulomb logarithm array
    bool is_initialized = false;
    // Number of points in sp direction
    int32_t size_sp;
    // Number of points in sp direction without ghost cells
    int32_t size_sp_stripped;
    // Lower bound of species axis without ghost cells
    int32_t lb_sp_stripped;
    // Value of PI
    real_t PI;
    // Mass array for each species
    real_t* masses_ptr;
    // Temperature scaling array for each species
    real_t* temp_scalings_ptr;
    // Inverse squared mass array for each species
    real_t* prefac_vth_ptr;
    // Pointer of prefactor for B parallel star
    real_t* prefac_bps_ptr;
    // Prefactor for the collision frequency with masses and charges
    real_t* prefac_full_nu_ptr;
    // Prefactor for the momentum relaxation of alpha
    real_t* prefac_relx_u_a_ptr;
    // Prefactor for the momentum relaxation of beta
    real_t* prefac_relx_u_b_ptr;
    // Prefactor for the temperature relaxation of alpha
    real_t* prefac_relx_T_a_ptr;
    // Prefactor for the temperature relaxation of beta
    real_t* prefac_relx_T_b_ptr;
    // Prefactor for mixing temperature with masses
    real_t* prefac_velpart_T_ptr;

    // Array of Coloumb logaritm
    data_array_t<real_t, 2>& collog;
    // Mesh 5D
    const mesh_5d_t& mesh;

    // Check if the device memory is set and used as intended
    inline void dmem_debug(const dmm::mode_t mode)
    {
        bool err = false;
        int32_t size_sp_2d = this->size_sp * this->size_sp_stripped;

        dmm::start_region("op_coll_bgk_gpu_t", mode);
        err = err || dmm::is_invalid(this, 1);
        err = err || dmm::is_invalid(this->masses_ptr, this->size_sp);
        err = err || dmm::is_invalid(this->temp_scalings_ptr, this->size_sp);
        err = err || dmm::is_invalid(this->prefac_vth_ptr, this->size_sp);
        err = err || dmm::is_invalid(this->prefac_bps_ptr, this->size_sp);
        err = err || dmm::is_invalid(this->prefac_full_nu_ptr, size_sp_2d);
        err = err || dmm::is_invalid(this->prefac_relx_u_a_ptr, size_sp_2d);
        err = err || dmm::is_invalid(this->prefac_relx_u_b_ptr, size_sp_2d);
        err = err || dmm::is_invalid(this->prefac_relx_T_a_ptr, size_sp_2d);
        err = err || dmm::is_invalid(this->prefac_relx_T_b_ptr, size_sp_2d);
        err = err || dmm::is_invalid(this->prefac_velpart_T_ptr, size_sp_2d);
        dmm::end_region("op_coll_bgk_gpu_t");

        op_coll_bgk_gpu::is_erroneous = op_coll_bgk_gpu::is_erroneous || err;
    }

    // The compute kernel of the apply() method
    #pragma acc routine seq
    inline void comp_kernel(const int32_t i, const int32_t k,
                            const int32_t l, const int32_t m,
                            const int32_t beta, const int32_t alpha,
                            const mesh_5d_t& mesh,
                            const data_array_t<const real_t, 5>& f_in,
                            const data_array_t<const real_t, 4>& moments,
                            const data_array_t<real_t, 2>& collog,
                            data_array_t<real_t, 5>& f_out) const
    {
        // Collision frequency
        real_t nu = this->prefac_full_nu(beta, alpha)
                  * collog(i, k) * moments(i, k, 1, beta)
                  * pow( this->masses(alpha) * moments(i, k, 3, beta)
                                             * this->temp_scalings(beta)
                       + this->masses(beta)  * moments(i, k, 3, alpha)
                                             * this->temp_scalings(alpha)
                       , -1.5);

        // Maxwellian parameter: mixing flow
        real_t u_ab = this->prefac_relx_u_a(beta, alpha)
                    * moments(i, k, 2, alpha)
                    + this->prefac_relx_u_b(beta, alpha)
                    * moments(i, k, 2, beta);

        // Maxwellian parameter: mixing temperature
        real_t T_ab = this->prefac_relx_T_a(beta, alpha)
                    * moments(i, k, 3, alpha)
                    + this->prefac_relx_T_b(beta, alpha)
                    * moments(i, k, 3, beta)
                    + this->prefac_velpart_T(beta, alpha)
                    * pow( moments(i, k, 2, beta) * this->prefac_vth(beta)
                         - moments(i, k, 2, alpha) * this->prefac_vth(alpha)
                    , 2);

        // Prefactor for Maxwellian
        real_t Mfac_ab = moments(i, k, 1, alpha) * pow(this->PI * T_ab, -1.5);

        // B parallel star
        real_t bps = mesh.absB(i, k) + this->prefac_bps(alpha)
                   * mesh.vp(l) * mesh.curl_normb_y(i, k);

        // Calculate collision operator:
        // nu * (Maxwellian - f_in) and
        // add to ouput distribution function
        real_t maxw = Mfac_ab * (mesh.absB(i, k) / bps)
                    * exp(-(pow(mesh.vp(l) - u_ab, 2)
                    + mesh.absB(i, k) * mesh.mu(m)) / T_ab);

        f_out(i, k, l, m, alpha) += mesh.is_compute(i, k) * nu
                                 * (maxw - f_in(i, k, l, m, alpha));
    }

public:
    // Constructor
    op_coll_bgk_gpu_t(const dcomm_handler_t& dcomm_handler,
                      const mesh_5d_t& _mesh,
                      data_array_t<real_t, 2>& _collog)
    : collog{_collog}, mesh{_mesh}
    {
        this->size_sp = this->mesh.get_size_sp();
        this->PI      = math::PI;
        this->lb_sp_stripped   = dcomm_handler.get_lbound_stripped(5);
        int32_t ub_sp_stripped = dcomm_handler.get_ubound_stripped(5);
        this->size_sp_stripped = (ub_sp_stripped - this->lb_sp_stripped + 1);

        // Allocate class members
        this->masses_ptr        = new real_t[size_sp] {};
        this->temp_scalings_ptr = new real_t[size_sp] {};
        this->prefac_vth_ptr    = new real_t[size_sp] {};
        this->prefac_bps_ptr    = new real_t[size_sp] {};

        this->prefac_full_nu_ptr   = new real_t[size_sp * size_sp_stripped] {};
        this->prefac_relx_u_a_ptr  = new real_t[size_sp * size_sp_stripped] {};
        this->prefac_relx_u_b_ptr  = new real_t[size_sp * size_sp_stripped] {};
        this->prefac_relx_T_a_ptr  = new real_t[size_sp * size_sp_stripped] {};
        this->prefac_relx_T_b_ptr  = new real_t[size_sp * size_sp_stripped] {};
        this->prefac_velpart_T_ptr = new real_t[size_sp * size_sp_stripped] {};

        // Calculate prefactor for bps, store masses and charges
        for (int32_t n = 1; n <= this->size_sp; n++)
        {
            this->masses_ptr[n - 1] = pspec::get_mass(n);
            this->temp_scalings_ptr[n - 1] = pspec::get_temp_scaling(n);
            this->prefac_vth_ptr[n - 1] = sqrt(this->temp_scalings_ptr[n - 1]
                                              / this->masses_ptr[n - 1]);
            this->prefac_bps_ptr[n - 1] = sqrt(2.0 * pspec::get_mass(n))
                * sqrt(pspec::get_temp_scaling(n))
                * pnorm::get_rho_ref()
                / (pspec::get_charge(n) * pnorm::get_L_ref());
        }

        // Loop over all species combinations
        for (int32_t alpha = this->lb_sp_stripped; alpha <= ub_sp_stripped;
             alpha++)
        // This has to go explicitly through all species because of
        // MPI processes may be parallelized on the species dimension
        for (int32_t beta = 1; beta <= this->size_sp; beta++)
        {
            int32_t idx = beta - 1
                        + (alpha - this->lb_sp_stripped) * this->size_sp;

            real_t sum_m       = this->masses(alpha) + this->masses(beta);
            real_t prod_m      = this->masses(alpha) * this->masses(beta);
            real_t sqrt_prod_m = sqrt(prod_m);

            // Prefac of mixing temperature with masses
            this->prefac_velpart_T_ptr[idx] = (1.0 / 3.0) * (prod_m / sum_m)
                                            / this->temp_scalings(alpha);

            // Prefac nu with masses and charges
            this->prefac_full_nu_ptr[idx] = pnorm::get_coll_ref() * sqrt_prod_m
                * pow(pspec::get_charge(alpha) * pspec::get_charge(beta), 2);

            // Depending on relaxation type we adjust the prefactors that
            // are multiplied on nu, u and T. We do this before the main
            // loop to avoid conditionals there
            if (pcoll::get_relx_type() == "et")
            {
                // Temperature relaxation
                // NOTE: prefac_full_nu does not change in temp relaxation
                this->prefac_relx_u_a_ptr[idx] = this->masses(alpha) / sum_m;
                this->prefac_relx_u_b_ptr[idx] = this->masses(beta) / sum_m;
                this->prefac_relx_T_a_ptr[idx] = 0.5;
                this->prefac_relx_T_b_ptr[idx] = 0.5;
            }
            else
            {
                // Momentum relaxation
                this->prefac_full_nu_ptr[idx] *= 0.5 * sum_m
                                               / this->masses(alpha);
                this->prefac_relx_u_a_ptr[idx] = 0.5;
                this->prefac_relx_u_b_ptr[idx] = 0.5;
                this->prefac_relx_T_a_ptr[idx] = this->masses(beta) / sum_m;
                this->prefac_relx_T_b_ptr[idx] = this->masses(alpha) / sum_m;
            }
            this->prefac_relx_u_b_ptr[idx] *= this->prefac_vth(beta)
                                            / this->prefac_vth(alpha);
            this->prefac_relx_T_b_ptr[idx] *= this->temp_scalings(beta)
                                            / this->temp_scalings(alpha);
        }
    }

    // Destructor
    virtual ~op_coll_bgk_gpu_t()
    {
        delete[] prefac_velpart_T_ptr;
        delete[] prefac_relx_T_b_ptr;
        delete[] prefac_relx_T_a_ptr;
        delete[] prefac_relx_u_b_ptr;
        delete[] prefac_relx_u_a_ptr;
        delete[] prefac_full_nu_ptr;
        delete[] prefac_bps_ptr;
        delete[] prefac_vth_ptr;
        delete[] temp_scalings_ptr;
        delete[] masses_ptr;
    }

    // Copy constructor is disabled
    op_coll_bgk_gpu_t(const op_coll_bgk_gpu_t&) = delete;

    // Copy-assignment operator is disabled
    op_coll_bgk_gpu_t& operator=(const op_coll_bgk_gpu_t&) = delete;

    // Applies the operator to the given input values
    // Return 0 for success and 1 for error
    virtual int32_t apply(const data_array_t<const real_t, 5>& f_in,
                          const data_array_t<const real_t, 4>& moments,
                          data_array_t<real_t, 5>& f_out) = 0;

    // Getter for masses
    #pragma acc routine seq
    inline real_t masses(int32_t n) const
    {
        return this->masses_ptr[n - 1];
    }
    // Getter for temp scalings
    #pragma acc routine seq
    inline real_t temp_scalings(int32_t n) const
    {
        return this->temp_scalings_ptr[n - 1];
    }
    // Getter for prefac_vth
    #pragma acc routine seq
    inline real_t prefac_vth(int32_t n) const
    {
        return this->prefac_vth_ptr[n - 1];
    }
    // Getter for prefac_bps
    #pragma acc routine seq
    inline real_t prefac_bps(int32_t n) const
    {
        return this->prefac_bps_ptr[n - 1];
    }
    // Getter for prefac_full_nu
    #pragma acc routine seq
    inline real_t prefac_full_nu(int32_t beta, int32_t alpha) const
    {
        return this->prefac_full_nu_ptr[beta - 1
            + (alpha - this->lb_sp_stripped) * this->size_sp];
    }
    // Getter for prefac_relx_u_a
    #pragma acc routine seq
    inline real_t prefac_relx_u_a(int32_t beta, int32_t alpha) const
    {
        return this->prefac_relx_u_a_ptr[beta - 1
            + (alpha - this->lb_sp_stripped) * this->size_sp];
    }
    // Getter for prefac_relx_u_b
    #pragma acc routine seq
    inline real_t prefac_relx_u_b(int32_t beta, int32_t alpha) const
    {
        return this->prefac_relx_u_b_ptr[beta - 1
            + (alpha - this->lb_sp_stripped) * this->size_sp];
    }
    // Getter for prefac_relx_T_a
    #pragma acc routine seq
    inline real_t prefac_relx_T_a(int32_t beta, int32_t alpha) const
    {
        return this->prefac_relx_T_a_ptr[beta - 1
            + (alpha - this->lb_sp_stripped) * this->size_sp];
    }
    // Getter for prefac_relx_T_b
    #pragma acc routine seq
    inline real_t prefac_relx_T_b(int32_t beta, int32_t alpha) const
    {
        return this->prefac_relx_T_b_ptr[beta - 1
            + (alpha - this->lb_sp_stripped) * this->size_sp];
    }
    // Getter for prefac_velpart_T
    #pragma acc routine seq
    inline real_t prefac_velpart_T(int32_t beta, int32_t alpha) const
    {
        return this->prefac_velpart_T_ptr[beta - 1
            + (alpha - this->lb_sp_stripped) * this->size_sp];
    }
};

#ifdef __cplusplus
extern "C" {
#endif

// Return 0 for success and 1 for error
int32_t cbind_op_coll_bgk_initialize(
    const dcomm_handler_t** dcomm_handler_cxx_pptr,
    const mesh_5d_t** mesh_cxx_pptr,
    data_array_t<real_t, 2>** collog_cxx_pptr,
    op_coll_bgk_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_coll_bgk_finalize(op_coll_bgk_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_coll_bgk_apply(
    op_coll_bgk_gpu_t** op_cxx_pptr,
    const data_array_t<const real_t, 5>** f_in_cxx_pptr,
    const data_array_t<const real_t, 4>** moments_cxx_pptr,
    data_array_t<real_t, 5>** f_out_cxx_pptr);

#ifdef __cplusplus
}
#endif

#endif
