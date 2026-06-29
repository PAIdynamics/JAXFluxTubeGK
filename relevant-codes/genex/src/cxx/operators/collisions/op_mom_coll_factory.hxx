#ifndef OP_MOM_COLL_FACTORY_HXX
#define OP_MOM_COLL_FACTORY_HXX

#include "params_gpu_offload.hxx"
#include "op_mom_coll.hxx"
#include "op_mom_coll_omp.hxx"

#ifdef ENABLE_OPENACC
#include "op_mom_coll_acc.hxx"
#endif

#ifdef ENABLE_OPENMPX
#include "op_mom_coll_ompx.hxx"
#endif

// Factory method to instantiate op_mom_coll_gpu_t derived class
// based on the chosen GPU offload backend
namespace op_mom_coll_gpu
{
    op_mom_coll_gpu_t* create(const dcomm_handler_t& dcomm_handler,
                              const mesh_5d_t& mesh)
    {
        switch (params_gpu_offload::get_backend())
        {
            case params_gpu_offload::backend_t::CPU:
                return new op_mom_coll_omp_t{dcomm_handler, mesh};
#ifdef ENABLE_OPENACC
            case params_gpu_offload::backend_t::ACC:
                return new op_mom_coll_acc_t{dcomm_handler, mesh};
#endif
#ifdef ENABLE_OPENMPX
            case params_gpu_offload::backend_t::OMPX:
                return new op_mom_coll_ompx_t{dcomm_handler, mesh};
#endif
            default:
                is_erroneous = true;
                return nullptr;
        }
    }
};

#endif
