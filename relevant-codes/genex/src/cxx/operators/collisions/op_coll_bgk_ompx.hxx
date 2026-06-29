#ifndef OP_COLL_BGK_OMPX_HXX
#define OP_COLL_BGK_OMPX_HXX

#include "genex_cxx_env.hxx"
#include "op_coll.hxx"
#include "op_coll_bgk.hxx"
#include <omp.h>

// C++ class which corresponds to the Fortran class
// op_coll_bgk_gpu_t with OpenMP offload
class op_coll_bgk_ompx_t: public op_coll_bgk_gpu_t
{
public:
    // Constructor of the OpenMP offload child class
    op_coll_bgk_ompx_t(const dcomm_handler_t& dcomm_handler,
                       const mesh_5d_t& mesh,
                       data_array_t<real_t, 2>& collog)
    : op_coll_bgk_gpu_t{dcomm_handler, mesh, collog}
    {
        int32_t size_sp_2d = this->size_sp * this->size_sp_stripped;
        // Allocate device pointer, map the host pointer to it and copy
        // the values
        #pragma omp target enter data \
            map(to: this[:1], \
                    this->size_sp, \
                    this->size_sp_stripped, \
                    this->lb_sp_stripped, \
                    this->PI, \
                    this->masses_ptr[:this->size_sp], \
                    this->temp_scalings_ptr[:this->size_sp], \
                    this->prefac_vth_ptr[:this->size_sp], \
                    this->prefac_bps_ptr[:this->size_sp], \
                    this->prefac_full_nu_ptr[:size_sp_2d], \
                    this->prefac_relx_u_a_ptr[:size_sp_2d], \
                    this->prefac_relx_u_b_ptr[:size_sp_2d], \
                    this->prefac_relx_T_a_ptr[:size_sp_2d], \
                    this->prefac_relx_T_b_ptr[:size_sp_2d], \
                    this->prefac_velpart_T_ptr[:size_sp_2d])

        this->dmem_debug(dmd::mode_t::ALLOC);
    };

    // Destructor of the OpenMP offload child class
    ~op_coll_bgk_ompx_t() override
    {
        int32_t size_sp_2d = this->size_sp * this->size_sp_stripped;
        // Deallocate the device pointers of the class members from the device
        #pragma omp target exit data \
            map(delete: this->prefac_velpart_T_ptr[:size_sp_2d], \
                        this->prefac_relx_T_b_ptr[:size_sp_2d], \
                        this->prefac_relx_T_a_ptr[:size_sp_2d], \
                        this->prefac_relx_u_b_ptr[:size_sp_2d], \
                        this->prefac_relx_u_a_ptr[:size_sp_2d], \
                        this->prefac_full_nu_ptr[:size_sp_2d], \
                        this->prefac_bps_ptr[:this->size_sp], \
                        this->prefac_vth_ptr[:this->size_sp], \
                        this->temp_scalings_ptr[:this->size_sp], \
                        this->masses_ptr[:this->size_sp], \
                        this->PI, \
                        this->lb_sp_stripped, \
                        this->size_sp_stripped, \
                        this->size_sp, \
                        this[:1])

        this->dmem_debug(dmd::mode_t::DEALLOC);
    };

    // Copy constructor is disabled
    op_coll_bgk_ompx_t(const op_coll_bgk_ompx_t&) = delete;

    // Copy-assignment operator is disabled
    op_coll_bgk_ompx_t& operator=(const op_coll_bgk_ompx_t&) = delete;

    // Apply method of the OpenMP offload child class
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
            ierr = calc_collog_ompx(moments, this->collog);
            this->is_initialized = true;
        }
        const data_array_t<real_t, 2>& collog = this->collog;

        #pragma omp target teams default(none) defaultmap(none) \
            shared(lb_stripped, ub_stripped, mesh, collog, f_in, moments, f_out)
        // Loop over all species combinations
        for (int32_t alpha = lb_stripped[4]; alpha <= ub_stripped[4]; alpha++)
        // This has to go explicitly through all species because of
        // MPI processes may be parallelized on the species dimension
        for (int32_t beta = 1; beta <= this->size_sp; beta++)
        #pragma omp distribute collapse(2)
        for (int32_t m = lb_stripped[3]; m <= ub_stripped[3]; m++)
        for (int32_t l = lb_stripped[2]; l <= ub_stripped[2]; l++)
        #pragma omp parallel for simd collapse(2)
        for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
        for (int32_t i = lb_stripped[0]; i <= ub_stripped[0]; i++)
        {
            this->comp_kernel(i, k, l, m, beta, alpha,
                              mesh, f_in, moments, collog, f_out);
        }

        return (int32_t) (ierr != 0);
    }
};

#endif
