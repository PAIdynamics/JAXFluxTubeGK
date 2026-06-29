#ifndef OP_BND_COND_NEUM_FACTORY_HXX
#define OP_BND_COND_NEUM_FACTORY_HXX

#include "params_gpu_offload.hxx"
#include "op_bnd_cond_neum.hxx"
#include "op_bnd_cond_neum_omp.hxx"

#ifdef ENABLE_OPENACC
#include "op_bnd_cond_neum_acc.hxx"
#endif

#ifdef ENABLE_OPENMPX
#include "op_bnd_cond_neum_ompx.hxx"
#endif

// Factory method to instantiate op_bnd_cond_neum_gpu_t derived class
// based on the chosen GPU offload backend
namespace op_bnd_cond_neum_gpu
{
    op_bnd_cond_neum_gpu_t* create(const dcomm_handler_t& dcomm_handler,
                                   const mesh_5d_t& mesh,
                                   real_t rho_center,
                                   real_t* is_core_ptr,
                                   real_t* rho_ptr)
    {
        switch (params_gpu_offload::get_backend())
        {
            case params_gpu_offload::backend_t::CPU:
                return new op_bnd_cond_neum_omp_t{dcomm_handler, mesh,
                                                  rho_center, is_core_ptr,
                                                  rho_ptr};
#ifdef ENABLE_OPENACC
            case params_gpu_offload::backend_t::ACC:
                return new op_bnd_cond_neum_acc_t{dcomm_handler, mesh,
                                                  rho_center, is_core_ptr,
                                                  rho_ptr};
#endif
#ifdef ENABLE_OPENMPX
            case params_gpu_offload::backend_t::OMPX:
                return new op_bnd_cond_neum_ompx_t{dcomm_handler, mesh,
                                                   rho_center, is_core_ptr,
                                                   rho_ptr};
#endif
    default:
        is_erroneous = true;
        return nullptr;
    }
    }
};

#endif
