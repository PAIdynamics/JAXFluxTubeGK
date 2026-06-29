#ifndef OP_BND_COND_NEUM_HXX
#define OP_BND_COND_NEUM_HXX

#include "genex_cxx_env.hxx"
#include "device_memory_debugger.hxx"
#include "dcomm_handler.hxx"
#include "mesh_5d.hxx"
#include "data_array.hxx"
#include "op_set_uniform_omp.hxx"
#include <vector>
#include <algorithm>

// Alias for external namespaces
namespace dmd = device_memory_debugger;

// Error flag for op_bnd_cond_neum_gpu_t
namespace op_bnd_cond_neum_gpu
{
    inline bool is_erroneous = false;
}

// C++ class which corresponds to Fortran class op_bnd_cond_neum_gpu_t
class op_bnd_cond_neum_gpu_t
{
protected:
    // Integer value for filler points of non-axisymmetry
    int32_t FILLER_VALUE_INT;
    // Size of the stripped domain on phi dimension
    int32_t size_phi_s;
    // Lower bound of the stripped domain on phi dimension
    int32_t lb_phi_s;
    // Maximum number of compute points in the inner boundary buffer region
    int32_t max_num_compute_buf_core;
    // Maximum number of ghost points in the core region
    int32_t max_num_ghost_core;
    // Array indicating RZ indices of compute points in the inner boundary
    // buffer region
    int32_t* ind_compute_buf_core_ptr;
    // Array indicating RZ indices of ghost points in the core region
    int32_t* ind_ghost_core_ptr;
    // Number of boundary points in the inner buffer region
    real_t* number_bnd_points_ptr;

    // Mesh 5D
    const mesh_5d_t& mesh;

    // The first compute kernel of the apply() method
    #pragma acc routine seq
    inline void comp_kernel_1(const int32_t j, const int32_t k, const int32_t l,
                              const int32_t m, const int32_t n,
                              data_array_t<real_t, 5>& f_inout,
                              data_array_t<real_t, 4>& buf_bnd) const
    {
        int32_t i = this->ind_compute_buf_core(j, k);
        if (i == this->FILLER_VALUE_INT) return;
        buf_bnd(k, l, m, n) += f_inout(i, k, l, m, n);
    }

    // The second compute kernel of the apply() method
    #pragma acc routine seq
    inline void comp_kernel_2(const int32_t j, const int32_t k, const int32_t l,
                              const int32_t m, const int32_t n,
                              data_array_t<real_t, 4>& buf_bnd,
                              data_array_t<real_t, 5>& f_inout) const
    {
        int32_t i = this->ind_ghost_core(j, k);
        if (i == this->FILLER_VALUE_INT) return;
        f_inout(i, k, l, m, n) = buf_bnd(k, l, m, n)
                               / this->number_bnd_points(k);
    }

    // The third compute kernel of the apply() method
    #pragma acc routine seq
    inline void comp_kernel_3(const int32_t i, const int32_t k,
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
        int32_t len1 = this->max_num_compute_buf_core * this->size_phi_s;
        int32_t len2 = this->max_num_ghost_core * this->size_phi_s;

        dmd::start_region("op_bnd_cond_neum_gpu_t", mode);
        err = err || dmd::is_invalid(this, 1);
        err = err || dmd::is_invalid(this->ind_compute_buf_core_ptr, len1);
        err = err || dmd::is_invalid(this->ind_ghost_core_ptr, len2);
        err = err || dmd::is_invalid(this->number_bnd_points_ptr,
                                     this->size_phi_s);
        dmd::end_region("op_bnd_cond_neum_gpu_t");

        op_bnd_cond_neum_gpu::is_erroneous =
            op_bnd_cond_neum_gpu::is_erroneous || err;
    }

public:
    // Constructor
    op_bnd_cond_neum_gpu_t(const dcomm_handler_t& dcomm_handler,
                           const mesh_5d_t& _mesh,
                           real_t rho_center,
                           real_t* is_core_ptr,
                           real_t* rho_ptr)
    : mesh{_mesh}
    {
        const int32_t* lb          = dcomm_handler.get_lbound();
        const int32_t* ub          = dcomm_handler.get_ubound();
        const int32_t* lb_stripped = dcomm_handler.get_lbound_stripped();
        const int32_t* ub_stripped = dcomm_handler.get_ubound_stripped();

        this->FILLER_VALUE_INT = mesh_5d::FILLER_VALUE_REAL;
        this->size_phi_s = ub_stripped[1] - lb_stripped[1] + 1;
        this->lb_phi_s   = lb_stripped[1];

        op_set_uniform_omp_t op;
        this->number_bnd_points_ptr = new real_t[this->size_phi_s]{};
        std::vector<int32_t> num_cmcore_buf(this->size_phi_s, 0);
        std::vector<int32_t> num_ghcore_buf(this->size_phi_s, 0);
        std::vector<int32_t> ind_cmcore_buf;
        std::vector<int32_t> ind_ghcore_buf;

        for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
        for (int32_t i = lb[0]; i <= ub[0]; i++)
        {
            int32_t idx = (i - 1) + (k - lb_stripped[1]) * mesh.get_size_RZ();

            // Count the number of compute points within the inner boundary
            // buffer zone per poloidal plane
            if (mesh.is_compute(i, k) == 1.0 &&
                mesh.buf_zone(i, k) == mesh_5d::buf_zone::BUF_ZONE_BOUNDARY_IN)
            {
                num_cmcore_buf[k - lb_stripped[1]] += 1;
                this->number_bnd_points_ptr[k - lb_stripped[1]] += 1.0;
                ind_cmcore_buf.push_back(i);
            }

            // Count the number of ghost points in the core per poloidal plane
            if (mesh.is_compute(i, k) == 0.0 &&
                is_core_ptr[idx] == 1.0 && rho_ptr[idx] <= rho_center)
            {
                num_ghcore_buf[k - lb_stripped[1]] += 1;
                ind_ghcore_buf.push_back(i);
            }
        }

        this->max_num_compute_buf_core =
            *std::max_element(num_cmcore_buf.begin(), num_cmcore_buf.end());
        this->max_num_ghost_core =
            *std::max_element(num_ghcore_buf.begin(), num_ghcore_buf.end());

        int32_t len1 = this->max_num_compute_buf_core * this->size_phi_s;
        int32_t len2 = this->max_num_ghost_core * this->size_phi_s;
        this->ind_compute_buf_core_ptr = new int32_t[len1];
        this->ind_ghost_core_ptr       = new int32_t[len2];
        op.apply(len1, this->FILLER_VALUE_INT, this->ind_compute_buf_core_ptr);
        op.apply(len2, this->FILLER_VALUE_INT, this->ind_ghost_core_ptr);

        for (int32_t k = 0; k < this->size_phi_s; k++)
        {
            int32_t j1 = k * this->max_num_compute_buf_core;
            int32_t j2 = k * num_cmcore_buf[k];
            for (int32_t i = 0; i < num_cmcore_buf[k]; i++)
            {
                this->ind_compute_buf_core_ptr[j1 + i] = ind_cmcore_buf[j2 + i];
            }

            j1 = k * this->max_num_ghost_core;
            j2 = k * num_ghcore_buf[k];
            for (int32_t i = 0; i < num_ghcore_buf[k]; i++)
            {
                this->ind_ghost_core_ptr[j1 + i] = ind_ghcore_buf[j2 + i];
            }
        }
    }

    // Destructor
    virtual ~op_bnd_cond_neum_gpu_t()
    {
        delete[] this->number_bnd_points_ptr;
        delete[] this->ind_ghost_core_ptr;
        delete[] this->ind_compute_buf_core_ptr;
    }

    // Copy constructor is disabled
    op_bnd_cond_neum_gpu_t(const op_bnd_cond_neum_gpu_t&) = delete;

    // Copy-assignment operator is disabled
    op_bnd_cond_neum_gpu_t& operator=(const op_bnd_cond_neum_gpu_t&) = delete;

    // Applies the operator to the given input values
    // Return 0 for success and 1 for error
    virtual int32_t apply(data_array_t<real_t, 5>& f_inout,
                          data_array_t<real_t, 2>& b_qn_eq,
                          data_array_t<real_t, 2>& b_amps_law,
                          data_array_t<real_t, 2>& b_ohms_law) = 0;

    // Getter for number_bnd_points
    #pragma acc routine seq
    inline real_t number_bnd_points(int j0) const
    {
        return this->number_bnd_points_ptr[j0 - this->lb_phi_s];
    }

    // Getter for ind_compute_buf_core
    #pragma acc routine seq
    inline real_t ind_compute_buf_core(int j0, int j1) const
    {
        return this->ind_compute_buf_core_ptr[
            j0 + (j1 - this->lb_phi_s) * this->max_num_compute_buf_core];
    }

    // Getter for ind_ghost_core
    #pragma acc routine seq
    inline real_t ind_ghost_core(int j0, int j1) const
    {
        return this->ind_ghost_core_ptr[
            j0 + (j1 - this->lb_phi_s) * this->max_num_ghost_core];
    }
};

#ifdef __cplusplus
extern "C" {
#endif

// Return 0 for success and 1 for error
int32_t cbind_op_bnd_cond_neum_initialize(
    const dcomm_handler_t** dcomm_handler_cxx_pptr,
    const mesh_5d_t** mesh_cxx_pptr,
    real_t rho_center,
    real_t* is_core_ptr,
    real_t* rho_ptr,
    op_bnd_cond_neum_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_bnd_cond_neum_finalize(op_bnd_cond_neum_gpu_t** op_cxx_pptr);

// Return 0 for success and 1 for error
int32_t cbind_op_bnd_cond_neum_apply(
    op_bnd_cond_neum_gpu_t** op_cxx_pptr,
    data_array_t<real_t, 5>** f_inout_cxx_pptr,
    data_array_t<real_t, 2>** b_qn_eq_cxx_pptr,
    data_array_t<real_t, 2>** b_amps_law_cxx_pptr,
    data_array_t<real_t, 2>** b_ohms_law_cxx_pptr);

#ifdef __cplusplus
}
#endif

#endif
