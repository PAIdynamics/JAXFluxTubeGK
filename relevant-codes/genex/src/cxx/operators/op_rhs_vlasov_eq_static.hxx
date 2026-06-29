#ifndef OP_RHS_VLASOV_EQ_STATIC_HXX
#define OP_RHS_VLASOV_EQ_STATIC_HXX

#include "genex_cxx_env.hxx"
#include "device_memory_debugger.hxx"
#include "mesh_5d.hxx"
#include "data_array.hxx"
#include "params_gpu_offload.hxx"

// Alias for external namespaces
namespace dmd   = device_memory_debugger;
namespace pgpus = params_gpu_offload;

// Error flag for op_rhs_vlasov_eq_static_gpu_t
namespace op_rhs_vlasov_eq_static_gpu
{
    inline bool is_erroneous = false;
}

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __cplusplus
}
#endif

// C++ class which corresponds to Fortran class op_rhs_vlasov_eq_static_gpu_t
class op_rhs_vlasov_eq_static_gpu_t
{
protected:
    // Size informations about the mesh related to the operator
    int32_t size_RZ;
    int32_t size_phi;
    int32_t size_sp;

    // Scalar prefactors for the operators

    // Prefactor for Arakawa bracket
    real_t  prefac_arakawa;
    // Prefactor for the centered finite difference in orthogonal direction
    real_t  prefac_cd_xz;
    // Prefactor for the centered finite difference in parallel velocity space
    real_t  prefac_cd_vp;
    // Prefactor for the hyperdiffusion in orthogonal direction
    real_t  prefac_hyp_xz;
    // Prefactor for the hyperdiffusion in parallel velocity space
    real_t  prefac_hyp_vp;
    // Prefactor for the second order Hamiltonian
    real_t prefac_H2;

    // Pointer to the species-dependent prefactor arrays
    real_t* prefac_bps_ptr;
    real_t* prefac_orth_ptr;
    real_t* prefac_par_ptr;
    real_t* charges_ptr;
    real_t* masses_ptr;
    real_t* temp_scalings_ptr;
    real_t* prefac_flutter_xyz_ptr;
    real_t* prefac_flutter_vp_ptr;
    real_t* prefac_Bpar_ptr;

    // Pointer to the real-space-dependent prefactor arrays

    // Prefactor for the diffusion applied in the buffer zone
    real_t* prefac_buffer_zone_ptr;
    // Prefactor of the k - 2 contribution in the centered finite
    // difference stencil of the parallel derivative
    real_t* prefac_cd_y_mm_ptr;
    // Prefactor of the k - 1 contribution in the centered finite
    // difference stencil of the parallel derivative
    real_t* prefac_cd_y_m_ptr;
    // Prefactor of the k contribution in the centered finite
    // difference stencil of the parallel derivative
    real_t* prefac_cd_y_o_ptr;
    // Prefactor of the k + 1 contribution in the centered finite
    // difference stencil of the parallel derivative
    real_t* prefac_cd_y_p_ptr;
    // Prefactor of the k + 2 contribution in the centered finite
    // difference stencil of the parallel derivative
    real_t* prefac_cd_y_pp_ptr;
    // Prefactor of the k - 2 contribution in the parallel hyperdiffusion
    real_t* prefac_hyp_y_mm_ptr;
    // Prefactor of the k - 1 contribution in the parallel hyperdiffusion
    real_t* prefac_hyp_y_m_ptr;
    // Prefactor of the k contribution in the parallel hyperdiffusion
    real_t* prefac_hyp_y_o_ptr;
    // Prefactor of the k + 1 contribution in the parallel hyperdiffusion
    real_t* prefac_hyp_y_p_ptr;
    // Prefactor of the k + 2 contribution in the parallel hyperdiffusion
    real_t* prefac_hyp_y_pp_ptr;

    // Check if the device memory is set and used as intended
    inline void dmem_debug(const dmd::mode_t mode)
    {
        bool err = false;
        int32_t size_mesh = this->size_RZ * this->size_phi;

        dmd::start_region("op_rhs_vlasov_eq_static_gpu_t", mode);
        err = err || dmd::is_invalid(this, 1);
        err = err || dmd::is_invalid(this->prefac_bps_ptr, this->size_sp);
        err = err || dmd::is_invalid(this->prefac_orth_ptr, this->size_sp);
        err = err || dmd::is_invalid(this->prefac_par_ptr, this->size_sp);
        err = err || dmd::is_invalid(this->charges_ptr, this->size_sp);
        err = err || dmd::is_invalid(this->masses_ptr, this->size_sp);
        err = err || dmd::is_invalid(this->temp_scalings_ptr, this->size_sp);
        err = err
            || dmd::is_invalid(this->prefac_flutter_xyz_ptr, this->size_sp);
        err = err
            || dmd::is_invalid(this->prefac_flutter_vp_ptr, this->size_sp);
        err = err || dmd::is_invalid(this->prefac_buffer_zone_ptr, size_mesh);
        err = err || dmd::is_invalid(this->prefac_cd_y_mm_ptr, size_mesh);
        err = err || dmd::is_invalid(this->prefac_cd_y_m_ptr, size_mesh);
        err = err || dmd::is_invalid(this->prefac_cd_y_o_ptr, size_mesh);
        err = err || dmd::is_invalid(this->prefac_cd_y_p_ptr, size_mesh);
        err = err || dmd::is_invalid(this->prefac_cd_y_pp_ptr, size_mesh);
        err = err || dmd::is_invalid(this->prefac_hyp_y_mm_ptr, size_mesh);
        err = err || dmd::is_invalid(this->prefac_hyp_y_m_ptr, size_mesh);
        err = err || dmd::is_invalid(this->prefac_hyp_y_o_ptr, size_mesh);
        err = err || dmd::is_invalid(this->prefac_hyp_y_p_ptr, size_mesh);
        err = err || dmd::is_invalid(this->prefac_hyp_y_pp_ptr, size_mesh);
        dmd::end_region("op_rhs_vlasov_eq_static_gpu_t");

        op_rhs_vlasov_eq_static_gpu::is_erroneous =
            op_rhs_vlasov_eq_static_gpu::is_erroneous || err;
    }

    // Initialization kernel for the second order Hamiltonian
    #pragma acc routine seq
    inline void initialize_H2(const int32_t i, const int32_t k,
                              const mesh_5d_t& mesh,
                              const data_array_t<const real_t, 2>& phi_in,
                              data_array_t<real_t, 2>& H2) const
    {
        // Index k for poloidal planes but wrapped around periodically for
        // ghost points in the phi dimension
        int32_t k_periodic = ((k - 1 + size_phi) % size_phi) + 1;

        real_t ngb_m  = mesh.neighbors(-1,  0, i, k_periodic);
        real_t ngb_p  = mesh.neighbors( 1,  0, i, k_periodic);
        real_t ngb_mm = mesh.neighbors( 0, -1, i, k_periodic);
        real_t ngb_pp = mesh.neighbors( 0,  1, i, k_periodic);

        // NOTE: The outermost ghost layer does not have neighbors on
        //       the outside. The neighbors array will contain the index of
        //       the point itself then. Since values of H2 on the outermost
        //       ghost layer are not used, we can just fill it with these dummy
        //       values from the computation with that index.

        real_t dphidx = this->prefac_cd_xz * 0.5
                      * (phi_in(ngb_p,  k) - phi_in(ngb_m,  k));
        real_t dphidz = this->prefac_cd_xz * 0.5
                      * (phi_in(ngb_pp, k) - phi_in(ngb_mm, k));

        // NOTE: H2 misses a factor of mass in order to avoid a species
        //       dependence of the array and save memory and performance
        // NOTE: We need to use k_periodic here since absB does not
        //       contain values at ghost points in the phi dimension
        H2(i, k) = -this->prefac_H2 * (dphidx * dphidx + dphidz * dphidz)
                 / (2.0 * mesh.absB(i, k_periodic) * mesh.absB(i, k_periodic));
    }

    // The compute kernel of the apply() method
    #pragma acc routine seq
    inline void comp_kernel(const int32_t i, const int32_t k,
                            const int32_t l, const int32_t m,
                            const int32_t n, const mesh_5d_t& mesh,
                            const data_array_t<const real_t, 5>& f_in,
                            const data_array_t<const real_t, 2>& phi_in,
                            const data_array_t<const real_t, 2>& A_par_in,
                            const data_array_t<const real_t, 2>& B_par_in,
                            const data_array_t<real_t, 2>& H2,
                            data_array_t<real_t, 5>& f_out) const;

public:
    // Constructor
    op_rhs_vlasov_eq_static_gpu_t(const mesh_5d_t& mesh);

    // Destructor
    virtual ~op_rhs_vlasov_eq_static_gpu_t()
    {
        if(pgpus::get_use_array_alignment())
        {
            std::align_val_t align = pgpus::get_array_alignment();
            operator delete(this->prefac_hyp_y_pp_ptr, align);
            operator delete(this->prefac_hyp_y_p_ptr, align);
            operator delete(this->prefac_hyp_y_o_ptr, align);
            operator delete(this->prefac_hyp_y_m_ptr, align);
            operator delete(this->prefac_hyp_y_mm_ptr, align);
            operator delete(this->prefac_cd_y_pp_ptr, align);
            operator delete(this->prefac_cd_y_p_ptr, align);
            operator delete(this->prefac_cd_y_o_ptr, align);
            operator delete(this->prefac_cd_y_m_ptr, align);
            operator delete(this->prefac_cd_y_mm_ptr, align);
            operator delete(this->prefac_buffer_zone_ptr, align);
        }
        else
        {
            delete[] this->prefac_hyp_y_pp_ptr;
            delete[] this->prefac_hyp_y_p_ptr;
            delete[] this->prefac_hyp_y_o_ptr;
            delete[] this->prefac_hyp_y_m_ptr;
            delete[] this->prefac_hyp_y_mm_ptr;
            delete[] this->prefac_cd_y_pp_ptr;
            delete[] this->prefac_cd_y_p_ptr;
            delete[] this->prefac_cd_y_o_ptr;
            delete[] this->prefac_cd_y_m_ptr;
            delete[] this->prefac_cd_y_mm_ptr;
            delete[] this->prefac_buffer_zone_ptr;
        }

        delete[] this->prefac_Bpar_ptr;
        delete[] this->prefac_flutter_vp_ptr;
        delete[] this->prefac_flutter_xyz_ptr;
        delete[] this->temp_scalings_ptr;
        delete[] this->masses_ptr;
        delete[] this->charges_ptr;
        delete[] this->prefac_par_ptr;
        delete[] this->prefac_orth_ptr;
        delete[] this->prefac_bps_ptr;
    }

    // Copy constructor is disabled
    op_rhs_vlasov_eq_static_gpu_t(const op_rhs_vlasov_eq_static_gpu_t&)
        = delete;

    // Copy-assignment operator is disabled
    op_rhs_vlasov_eq_static_gpu_t&
        operator=(const op_rhs_vlasov_eq_static_gpu_t&) = delete;

    // Applies the operator to the given input values
    // Return 0 for success and 1 for error
    virtual int32_t apply(const mesh_5d_t& mesh,
                          const data_array_t<const real_t, 5>& f_in,
                          const data_array_t<const real_t, 2>& phi_in,
                          const data_array_t<const real_t, 2>& A_par_in,
                          const data_array_t<const real_t, 2>& B_par_in,
                          data_array_t<real_t, 5>& f_out) const = 0;

    // Getter for prefac_arakawa mainly for unit testing
    #pragma acc routine seq
    inline real_t get_prefac_arakawa() const
    {
        return this->prefac_arakawa;
    }

    // Getter for prefac_arakawa mainly for unit testing
    #pragma acc routine seq
    inline real_t get_prefac_cd_xz() const
    {
        return this->prefac_cd_xz;
    }

    // Getter for prefac_arakawa mainly for unit testing
    #pragma acc routine seq
    inline real_t get_prefac_cd_vp() const
    {
        return this->prefac_cd_vp;
    }

    // Getter for prefac_arakawa mainly for unit testing
    #pragma acc routine seq
    inline real_t get_prefac_hyp_vp() const
    {
        return this->prefac_hyp_vp;
    }

    // Getter for prefac_arakawa mainly for unit testing
    #pragma acc routine seq
    inline real_t get_prefac_hyp_xz() const
    {
        return this->prefac_hyp_xz;
    }

    // Getter for prefac_H2 mainly for unit testing
    #pragma acc routine seq
    inline real_t get_prefac_H2() const
    {
        return this->prefac_H2;
    }

    // NOTE: The following getters are for multidimensional array access of the
    //       array members with Fortran-style indexing, i.e., 1-based usage

    // Getter for prefac_bps
    #pragma acc routine seq
    inline real_t prefac_bps(int j0) const
    {
        return this->prefac_bps_ptr[j0 - 1];
    }

    // Getter for prefac_orth
    #pragma acc routine seq
    inline real_t prefac_orth(int j0) const
    {
        return this->prefac_orth_ptr[j0 - 1];
    }

    // Getter for prefac_par
    #pragma acc routine seq
    inline real_t prefac_par(int j0) const
    {
        return this->prefac_par_ptr[j0 - 1];
    }

    // Getter for charges
    #pragma acc routine seq
    inline real_t charges(int j0) const
    {
        return this->charges_ptr[j0 - 1];
    }

    // Getter for masses
    #pragma acc routine seq
    inline real_t masses(int j0) const
    {
        return this->masses_ptr[j0 - 1];
    }

    // Getter for temperature scalings
    #pragma acc routine seq
    inline real_t temp_scalings(int j0) const
    {
        return this->temp_scalings_ptr[j0 - 1];
    }

    // Getter for prefac_flutter_xyz
    #pragma acc routine seq
    inline real_t prefac_flutter_xyz(int j0) const
    {
        return this->prefac_flutter_xyz_ptr[j0 - 1];
    }

    // Getter for prefac_flutter_vp
    #pragma acc routine seq
    inline real_t prefac_flutter_vp(int j0) const
    {
        return this->prefac_flutter_vp_ptr[j0 - 1];
    }

    // Getter for prefac_Bpar
    #pragma acc routine seq
    inline real_t prefac_Bpar(int j0) const
    {
        return this->prefac_Bpar_ptr[j0 - 1];
    }

    // Getter for prefac_buffer_zone
    #pragma acc routine seq
    inline real_t prefac_buffer_zone(int j0, int j1) const
    {
        return this->prefac_buffer_zone_ptr[(j0 - 1) +
                                            (j1 - 1) * this->size_RZ];
    }

    // Getter for prefac_cd_y_mm
    #pragma acc routine seq
    inline real_t prefac_cd_y_mm(int j0, int j1) const
    {
        return this->prefac_cd_y_mm_ptr[(j0 - 1) +
                                        (j1 - 1) * this->size_RZ];
    }

    // Getter for prefac_cd_y_m
    #pragma acc routine seq
    inline real_t prefac_cd_y_m(int j0, int j1) const
    {
        return this->prefac_cd_y_m_ptr[(j0 - 1) +
                                       (j1 - 1) * this->size_RZ];
    }

    // Getter for prefac_cd_y_o
    #pragma acc routine seq
    inline real_t prefac_cd_y_o(int j0, int j1) const
    {
        return this->prefac_cd_y_o_ptr[(j0 - 1) +
                                       (j1 - 1) * this->size_RZ];
    }

    // Getter for prefac_cd_y_p
    #pragma acc routine seq
    inline real_t prefac_cd_y_p(int j0, int j1) const
    {
        return this->prefac_cd_y_p_ptr[(j0 - 1) +
                                       (j1 - 1) * this->size_RZ];
    }

    // Getter for prefac_cd_y_pp
    #pragma acc routine seq
    inline real_t prefac_cd_y_pp(int j0, int j1) const
    {
        return this->prefac_cd_y_pp_ptr[(j0 - 1) +
                                        (j1 - 1) * this->size_RZ];
    }

    // Getter for prefac_hyp_y_mm
    #pragma acc routine seq
    inline real_t prefac_hyp_y_mm(int j0, int j1) const
    {
        return this->prefac_hyp_y_mm_ptr[(j0 - 1) +
                                         (j1 - 1) * this->size_RZ];
    }

    // Getter for prefac_hyp_y_m
    #pragma acc routine seq
    inline real_t prefac_hyp_y_m(int j0, int j1) const
    {
        return this->prefac_hyp_y_m_ptr[(j0 - 1) +
                                        (j1 - 1) * this->size_RZ];
    }

    // Getter for prefac_hyp_y_o
    #pragma acc routine seq
    inline real_t prefac_hyp_y_o(int j0, int j1) const
    {
        return this->prefac_hyp_y_o_ptr[(j0 - 1) +
                                        (j1 - 1) * this->size_RZ];
    }

    // Getter for prefac_hyp_y_p
    #pragma acc routine seq
    inline real_t prefac_hyp_y_p(int j0, int j1) const
    {
        return this->prefac_hyp_y_p_ptr[(j0 - 1) +
                                        (j1 - 1) * this->size_RZ];
    }

    // Getter for prefac_hyp_y_pp
    #pragma acc routine seq
    inline real_t prefac_hyp_y_pp(int j0, int j1) const
    {
        return this->prefac_hyp_y_pp_ptr[(j0 - 1) +
                                         (j1 - 1) * this->size_RZ];
    }

};

#ifdef __cplusplus
extern "C" {
#endif

// Shallow copy Fortran class to C++ class and allocate class to GPU
// if intended. Return 0 for success and 1 for error.
int32_t cbind_op_vlasov_static_initialize(
    const mesh_5d_t** mesh_cxx_pptr,
    op_rhs_vlasov_eq_static_gpu_t** op_cxx_pptr);

// Deallocate class from GPU if intended and C++ class instance while keeping
// the member allocations on CPU until they are freed from the Fortran layer.
// Return 0 for success and 1 for error.
int32_t cbind_op_vlasov_static_finalize(
    op_rhs_vlasov_eq_static_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_vlasov_static_apply(
    const op_rhs_vlasov_eq_static_gpu_t** op_cxx_pptr,
    const mesh_5d_t** mesh_cxx_pptr,
    const data_array_t<const real_t, 5>** f_in_cxx_pptr,
    const data_array_t<const real_t, 2>** phi_in_cxx_pptr,
    const data_array_t<const real_t, 2>** A_par_in_cxx_pptr,
    const data_array_t<const real_t, 2>** B_par_in_cxx_pptr,
    data_array_t<real_t, 5>** f_out_cxx_pptr);

#ifdef __cplusplus
}
#endif

#endif
