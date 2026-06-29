#ifndef OP_BND_COND_NEUM_ACC_HXX
#define OP_BND_COND_NEUM_ACC_HXX

#include "genex_cxx_env.hxx"
#include "op_bnd_cond_neum.hxx"
#include "data_array_acc.hxx"
#include <openacc.h>

// C++ class which corresponds to the Fortran class
// op_bnd_cond_neum_gpu_t with OpenACC on GPU
class op_bnd_cond_neum_acc_t: public op_bnd_cond_neum_gpu_t
{
public:
    // Constructor of the OpenACC child class
    op_bnd_cond_neum_acc_t(const dcomm_handler_t& dcomm_handler,
                           const mesh_5d_t& mesh,
                           real_t rho_center,
                           real_t* is_core_ptr,
                           real_t* rho_ptr)
    : op_bnd_cond_neum_gpu_t{dcomm_handler, mesh, rho_center, is_core_ptr,
                             rho_ptr}
    {
        int32_t len1 = this->max_num_compute_buf_core * this->size_phi_s;
        int32_t len2 = this->max_num_ghost_core * this->size_phi_s;

        // Allocate device pointer, map the host pointer to it and copy
        // the values
        #pragma acc enter data \
            copyin(this[:1], \
                   this->FILLER_VALUE_INT, \
                   this->size_phi_s, \
                   this->lb_phi_s, \
                   this->max_num_compute_buf_core, \
                   this->max_num_ghost_core, \
                   this->ind_compute_buf_core_ptr[:len1], \
                   this->ind_ghost_core_ptr[:len2], \
                   this->number_bnd_points_ptr[:this->size_phi_s])

        this->dmem_debug(dmd::mode_t::ALLOC);
    }

    // Destructor of the OpenACC child class
    ~op_bnd_cond_neum_acc_t() override
    {
        int32_t len1 = this->max_num_compute_buf_core * this->size_phi_s;
        int32_t len2 = this->max_num_ghost_core * this->size_phi_s;

        // Deallocate the device pointers of the class members from the device
        #pragma acc exit data \
            delete(this->number_bnd_points_ptr[:this->size_phi_s], \
                   this->ind_ghost_core_ptr[:len2], \
                   this->ind_compute_buf_core_ptr[:len1], \
                   this->max_num_ghost_core, \
                   this->max_num_compute_buf_core, \
                   this->lb_phi_s, \
                   this->size_phi_s, \
                   this->FILLER_VALUE_INT, \
                   this[:1])

        this->dmem_debug(dmd::mode_t::DEALLOC);
    }

    // Copy constructor is disabled
    op_bnd_cond_neum_acc_t(const op_bnd_cond_neum_acc_t&) = delete;

    // Copy-assignment operator is disabled
    op_bnd_cond_neum_acc_t& operator=(const op_bnd_cond_neum_acc_t&) = delete;

    int32_t apply(data_array_t<real_t, 5>& f_inout,
                  data_array_t<real_t, 2>& b_qn_eq,
                  data_array_t<real_t, 2>& b_amps_law,
                  data_array_t<real_t, 2>& b_ohms_law) override
    {
        const mesh_5d_t& mesh = this->mesh;
        const int32_t (&lb)[5]          = f_inout.get_lbound();
        const int32_t (&ub)[5]          = f_inout.get_ubound();
        const int32_t (&lb_stripped)[5] = f_inout.get_lbound_stripped();
        const int32_t (&ub_stripped)[5] = f_inout.get_ubound_stripped();

        // NOTE: To adapt the boundary points, we first compute the average
        //       value of the distribution functions within the inner
        //       boundary buffer region. This average is then imposed on the
        //       inner ghost points. The outer boundary is not updated, as it
        //       remains fixed by the initial conditions. Only the inner
        //       boundary is modified.

        // Set boundary conditions on the distribution function.

        data_array_acc_t<real_t, 4> buf_bnd(
            {lb_stripped[1], lb_stripped[2], lb_stripped[3], lb_stripped[4]},
            {ub_stripped[1], ub_stripped[2], ub_stripped[3], ub_stripped[4]},
            0.0);

        #pragma acc parallel default(none) \
            present(lb_stripped, ub_stripped, this, f_inout, buf_bnd)
        #pragma acc loop independent
        for (int32_t n = lb_stripped[4]; n <= ub_stripped[4]; n++)
        #pragma acc loop independent
        for (int32_t m = lb_stripped[3]; m <= ub_stripped[3]; m++)
        #pragma acc loop independent
        for (int32_t l = lb_stripped[2]; l <= ub_stripped[2]; l++)
        #pragma acc loop independent
        for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
        #pragma acc loop seq
        for (int32_t j = 0; j < this->max_num_compute_buf_core; j++)
        {
            this->comp_kernel_1(j, k, l, m, n, f_inout, buf_bnd);
        }

        #pragma acc parallel default(none) \
            present(lb, ub, lb_stripped, ub_stripped, this, buf_bnd, f_inout)
        #pragma acc loop independent gang worker vector collapse(5)
        for (int32_t n = lb_stripped[4]; n <= ub_stripped[4]; n++)
        for (int32_t m = lb_stripped[3]; m <= ub_stripped[3]; m++)
        for (int32_t l = lb_stripped[2]; l <= ub_stripped[2]; l++)
        for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
        for (int32_t j = 0; j < this->max_num_ghost_core; j++)
        {
            this->comp_kernel_2(j, k, l, m, n, buf_bnd, f_inout);
        }

        #pragma acc parallel default(none) \
        present(lb, ub, lb_stripped, ub_stripped, \
                this, mesh, b_qn_eq, b_amps_law, b_ohms_law)
        #pragma acc loop independent gang worker vector collapse(2)
        for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
        for (int32_t i = lb[0]; i <= ub[0]; i++)
        {
            this->comp_kernel_3(i, k, mesh, b_qn_eq, b_amps_law, b_ohms_law);
        }

        return 0;
    }
};

#endif
