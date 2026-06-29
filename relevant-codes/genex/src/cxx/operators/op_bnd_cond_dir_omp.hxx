#ifndef OP_BND_COND_DIR_OMP_HXX
#define OP_BND_COND_DIR_OMP_HXX

#include "genex_cxx_env.hxx"
#include "op_bnd_cond_dir.hxx"
#include <omp.h>

// C++ class which corresponds to the Fortran class
// op_bnd_cond_dir_gpu_t with OpenMP on CPU
class op_bnd_cond_dir_omp_t: public op_bnd_cond_dir_gpu_t
{
public:
    // Constructor of the OpenMP child class
    op_bnd_cond_dir_omp_t(const mesh_5d_t& mesh)
    : op_bnd_cond_dir_gpu_t{mesh} {};

    // Destructor of the OpenMP child class
    ~op_bnd_cond_dir_omp_t() override {};

    // Copy constructor is disabled
    op_bnd_cond_dir_omp_t(const op_bnd_cond_dir_omp_t&) = delete;

    // Copy-assignment operator is disabled
    op_bnd_cond_dir_omp_t& operator=(const op_bnd_cond_dir_omp_t&) = delete;

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

        #pragma omp parallel default(none) \
            shared(lb, ub, lb_stripped, ub_stripped, mesh, \
                   b_qn_eq, b_amps_law, b_ohms_law)
        for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
        #pragma omp for simd schedule(static) nowait
        for (int32_t i = lb[0]; i <= ub[0]; i++)
        {
            this->comp_kernel(i, k, mesh, b_qn_eq, b_amps_law, b_ohms_law);
        }

        return 0;
    }
};

#endif
