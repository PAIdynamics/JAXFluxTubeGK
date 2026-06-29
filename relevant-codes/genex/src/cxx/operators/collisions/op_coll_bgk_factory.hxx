#ifndef OP_COLL_BGK_FACTORY_HXX
#define OP_COLL_BGK_FACTORY_HXX

#include "params_gpu_offload.hxx"
#include "op_coll_bgk.hxx"
#include "op_coll_bgk_omp.hxx"

#ifdef ENABLE_OPENACC
#include "op_coll_bgk_acc.hxx"
#endif

#ifdef ENABLE_OPENMPX
#include "op_coll_bgk_ompx.hxx"
#endif

// Factory method to instantiate op_coll_bgk_gpu_t derived class
// based on the chosen GPU offload backend
namespace op_coll_bgk_gpu
{
    op_coll_bgk_gpu_t* create(const dcomm_handler_t& dcomm_handler,
                              const mesh_5d_t& mesh,
                              data_array_t<real_t, 2>& collog)
    {
        switch (params_gpu_offload::get_backend())
        {
            case params_gpu_offload::backend_t::CPU:
                return new op_coll_bgk_omp_t{dcomm_handler, mesh, collog};
#ifdef ENABLE_OPENACC
            case params_gpu_offload::backend_t::ACC:
                return new op_coll_bgk_acc_t{dcomm_handler, mesh, collog};
#endif
#ifdef ENABLE_OPENMPX
            case params_gpu_offload::backend_t::OMPX:
                return new op_coll_bgk_ompx_t{dcomm_handler, mesh, collog};
#endif
            default:
                is_erroneous = true;
                return nullptr;
        }
    }
};

#endif
