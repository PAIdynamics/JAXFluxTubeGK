#ifndef OP_RHS_VLASOV_EQ_STATIC_FACTORY_HXX
#define OP_RHS_VLASOV_EQ_STATIC_FACTORY_HXX

#include "params_gpu_offload.hxx"
#include "op_rhs_vlasov_eq_static.hxx"
#include "op_rhs_vlasov_eq_static_omp.hxx"

#ifdef ENABLE_OPENACC
#include "op_rhs_vlasov_eq_static_acc.hxx"
#endif

#ifdef ENABLE_OPENMPX
#include "op_rhs_vlasov_eq_static_ompx.hxx"
#endif

namespace op_rhs_vlasov_eq_static_gpu
{
    // Factory method to instantiate op_rhs_vlasov_eq_static_gpu_t derived class
    // based on the chosen GPU offload backend
    op_rhs_vlasov_eq_static_gpu_t* create(const mesh_5d_t& mesh)
    {
        switch (params_gpu_offload::get_backend())
        {
            case params_gpu_offload::backend_t::CPU:
                return new op_rhs_vlasov_eq_static_omp_t(mesh);
#ifdef ENABLE_OPENACC
            case params_gpu_offload::backend_t::ACC:
                return new op_rhs_vlasov_eq_static_acc_t(mesh);
#endif
#ifdef ENABLE_OPENMPX
            case params_gpu_offload::backend_t::OMPX:
                return new op_rhs_vlasov_eq_static_ompx_t(mesh);
#endif
            default:
                is_erroneous = true;
                return nullptr;
        }
    }
}

#endif
