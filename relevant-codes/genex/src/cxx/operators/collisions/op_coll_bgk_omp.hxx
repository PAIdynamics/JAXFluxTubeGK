#ifndef OP_COLL_BGK_OMP_HXX
#define OP_COLL_BGK_OMP_HXX

#include "genex_cxx_env.hxx"
#include "op_coll.hxx"
#include "op_coll_bgk.hxx"
#include <omp.h>

// C++ class which corresponds to the Fortran class
// op_coll_bgk_gpu_t with OpenMP on CPU
class op_coll_bgk_omp_t: public op_coll_bgk_gpu_t
{
public:
    // Constructor of the OpenMP child class
    op_coll_bgk_omp_t(const dcomm_handler_t& dcomm_handler,
                      const mesh_5d_t& mesh,
                      data_array_t<real_t, 2>& collog)
    : op_coll_bgk_gpu_t{dcomm_handler, mesh, collog} {};

    // Destructor of the OpenMP child class
    ~op_coll_bgk_omp_t() override {};

    // Copy constructor is disabled
    op_coll_bgk_omp_t(const op_coll_bgk_omp_t&) = delete;

    // Copy-assignment operator is disabled
    op_coll_bgk_omp_t& operator=(const op_coll_bgk_omp_t&) = delete;

    int32_t apply(const data_array_t<const real_t, 5>& f_in,
                  const data_array_t<const real_t, 4>& moments,
                  data_array_t<real_t, 5>& f_out) override
    {
        const mesh_5d_t& mesh = this->mesh;
        const int32_t (&lb_stripped)[5] = f_in.get_lbound_stripped();
        const int32_t (&ub_stripped)[5] = f_in.get_ubound_stripped();

        // First time calculation of Coulomb logarithm
        int32_t ierr = 0;
        if (!this->is_initialized)
        {
            ierr = calc_collog_omp(moments, this->collog);
            this->is_initialized = true;
        }
        const data_array_t<real_t, 2>& collog = this->collog;

        #pragma omp parallel default(none) \
            shared(lb_stripped, ub_stripped, mesh, collog, moments, f_in, f_out)
        // Loop over all species combinations
        for (int32_t alpha = lb_stripped[4]; alpha <= ub_stripped[4]; alpha++)
        // This has to go explicitly through all species because of
        // MPI processes may be parallelized on the species dimension
        for (int32_t beta = 1; beta <= this->size_sp; beta++)
        for (int32_t m = lb_stripped[3]; m <= ub_stripped[3]; m++)
        for (int32_t l = lb_stripped[2]; l <= ub_stripped[2]; l++)
        for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
        #pragma omp for simd schedule(static) nowait
        for (int32_t i = lb_stripped[0]; i <= ub_stripped[0]; i++)
        {
            this->comp_kernel(i, k, l, m, beta, alpha,
                              mesh, f_in, moments, collog, f_out);
        }

        return ierr;
    }
};

#endif
