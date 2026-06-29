#ifndef OP_MOM_OHMS_LAW_FACTORY_HXX
#define OP_MOM_OHMS_LAW_FACTORY_HXX

#include "params_gpu_offload.hxx"
#include "op_mom_ohms_law.hxx"
#include "op_mom_ohms_law_omp.hxx"

#ifdef ENABLE_OPENACC
#include "op_mom_ohms_law_acc.hxx"
#endif

#ifdef ENABLE_OPENMPX
#include "op_mom_ohms_law_ompx.hxx"
#endif

namespace op_mom_ohms_law_gpu
{
    // Factory method to instantiate op_mom_ohms_law_gpu_t derived class
    // based on the chosen GPU offload backend
    op_mom_ohms_law_gpu_t* create(const mesh_5d_t& mesh)
    {
        switch (params_gpu_offload::get_backend())
        {
            case params_gpu_offload::backend_t::CPU:
                return new op_mom_ohms_law_omp_t(mesh);
#ifdef ENABLE_OPENACC
            case params_gpu_offload::backend_t::ACC:
                return new op_mom_ohms_law_acc_t(mesh);
#endif
#ifdef ENABLE_OPENMPX
            case params_gpu_offload::backend_t::OMPX:
                return new op_mom_ohms_law_ompx_t(mesh);
#endif
            default:
                is_erroneous = true;
                return nullptr;
        }
    }
}

#endif
