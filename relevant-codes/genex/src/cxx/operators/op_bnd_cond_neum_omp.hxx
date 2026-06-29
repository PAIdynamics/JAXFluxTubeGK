#ifndef OP_BND_COND_NEUM_OMP_HXX
#define OP_BND_COND_NEUM_OMP_HXX

#include "genex_cxx_env.hxx"
#include "op_bnd_cond_neum.hxx"
#include "data_array_omp.hxx"
#include <omp.h>

// C++ class which corresponds to the Fortran class
// op_bnd_cond_neum_gpu_t with OpenMP on CPU
class op_bnd_cond_neum_omp_t: public op_bnd_cond_neum_gpu_t
{
public:
    // Constructor of the OpenMP child class
    op_bnd_cond_neum_omp_t(const dcomm_handler_t& dcomm_handler,
                           const mesh_5d_t& mesh,
                           real_t rho_center,
                           real_t* is_core_ptr,
                           real_t* rho_ptr)
    : op_bnd_cond_neum_gpu_t{dcomm_handler, mesh, rho_center, is_core_ptr,
                             rho_ptr} {};

    // Destructor of the OpenMP child class
    ~op_bnd_cond_neum_omp_t() override {};

    // Copy constructor is disabled
    op_bnd_cond_neum_omp_t(const op_bnd_cond_neum_omp_t&) = delete;

    // Copy-assignment operator is disabled
    op_bnd_cond_neum_omp_t& operator=(const op_bnd_cond_neum_omp_t&) = delete;

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

        data_array_omp_t<real_t, 4> buf_bnd(
            {lb_stripped[1], lb_stripped[2], lb_stripped[3], lb_stripped[4]},
            {ub_stripped[1], ub_stripped[2], ub_stripped[3], ub_stripped[4]},
            0.0);

        #pragma omp parallel default(none) \
            shared(lb_stripped, ub_stripped, f_inout, buf_bnd)
        #pragma omp for simd collapse(4) schedule(static) nowait
        for (int32_t n = lb_stripped[4]; n <= ub_stripped[4]; n++)
        for (int32_t m = lb_stripped[3]; m <= ub_stripped[3]; m++)
        for (int32_t l = lb_stripped[2]; l <= ub_stripped[2]; l++)
        for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
        for (int32_t j = 0; j < this->max_num_compute_buf_core; j++)
        {
            this->comp_kernel_1(j, k, l, m, n, f_inout, buf_bnd);
        }

        #pragma omp parallel default(none) \
            shared(lb, ub, lb_stripped, ub_stripped, buf_bnd, f_inout)
        for (int32_t n = lb_stripped[4]; n <= ub_stripped[4]; n++)
        for (int32_t m = lb_stripped[3]; m <= ub_stripped[3]; m++)
        for (int32_t l = lb_stripped[2]; l <= ub_stripped[2]; l++)
        for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
        #pragma omp for simd schedule(static) nowait
        for (int32_t j = 0; j < this->max_num_ghost_core; j++)
        {
            this->comp_kernel_2(j, k, l, m, n, buf_bnd, f_inout);
        }

        #pragma omp parallel default(none) \
            shared(lb, ub, lb_stripped, ub_stripped, mesh, \
                   b_qn_eq, b_amps_law, b_ohms_law)
        for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
        #pragma omp for simd schedule(static) nowait
        for (int32_t i = lb[0]; i <= ub[0]; i++)
        {
            this->comp_kernel_3(i, k, mesh, b_qn_eq, b_amps_law, b_ohms_law);
        }

        return 0;
    }
};

#endif
