#ifndef OP_RHS_VLASOV_EQ_DYNAMIC_OMPX_HXX
#define OP_RHS_VLASOV_EQ_DYNAMIC_OMPX_HXX

#include "genex_cxx_env.hxx"
#include "op_rhs_vlasov_eq_dynamic.hxx"
#include <omp.h>

// C++ class which corresponds to the Fortran class
// op_rhs_vlasov_eq_dynamic_gpu_t with OpenMP offload
class op_rhs_vlasov_eq_dynamic_ompx_t: public op_rhs_vlasov_eq_dynamic_gpu_t
{
public:
    // Constructor of the OpenMP offload child class
    op_rhs_vlasov_eq_dynamic_ompx_t(const mesh_5d_t& mesh)
    : op_rhs_vlasov_eq_dynamic_gpu_t{mesh}
    {
        // Allocate device pointer, map the host pointer to it and copy
        // the values
        #pragma omp target enter data \
            map(to: this[:1], \
                    this->size_sp, \
                    this->prefac_cd_vp, \
                    this->prefac_norm_ptr[:this->size_sp])

        this->dmem_debug(dmd::mode_t::ALLOC);
    };

    // Destructor of the OpenMP offload child class
    ~op_rhs_vlasov_eq_dynamic_ompx_t() override
    {
        // Deallocate the device pointers of the class members from the device
        #pragma omp target exit data \
            map(delete: this->prefac_norm_ptr[:this->size_sp], \
                        this->prefac_cd_vp, \
                        this->size_sp, \
                        this[:1])

        this->dmem_debug(dmd::mode_t::DEALLOC);
    };

    // Copy constructor is disabled
    op_rhs_vlasov_eq_dynamic_ompx_t(const op_rhs_vlasov_eq_dynamic_ompx_t&)
        = delete;

    // Copy-assignment operator is disabled
    op_rhs_vlasov_eq_dynamic_ompx_t&
        operator=(const op_rhs_vlasov_eq_dynamic_ompx_t&) = delete;

    // Apply method of the OpenMP offload child class
    int32_t apply(const mesh_5d_t& mesh,
                  const data_array_t<const real_t, 5>& f_in,
                  const data_array_t<const real_t, 2>& E_par_in,
                  data_array_t<real_t, 5>& f_out) const override
    {
        const int32_t (&lb_stripped)[5] = f_in.get_lbound_stripped();
        const int32_t (&ub_stripped)[5] = f_in.get_ubound_stripped();

        #pragma omp target teams distribute parallel for simd collapse(5) \
            default(none) defaultmap(none) \
            shared(lb_stripped, ub_stripped, mesh, E_par_in, f_in, f_out)
        for (int32_t n = lb_stripped[4]; n <= ub_stripped[4]; n++)
        for (int32_t m = lb_stripped[3]; m <= ub_stripped[3]; m++)
        for (int32_t l = lb_stripped[2]; l <= ub_stripped[2]; l++)
        for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
        for (int32_t i = lb_stripped[0]; i <= ub_stripped[0]; i++)
        {
            this->comp_kernel(i, k, l, m, n, mesh, f_in, E_par_in, f_out);
        }

        return 0;
    }
};

#endif
