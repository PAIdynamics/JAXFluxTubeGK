#ifndef OP_BND_COND_DIR_FACTORY_HXX
#define OP_BND_COND_DIR_FACTORY_HXX

#include "params_gpu_offload.hxx"
#include "op_bnd_cond_dir.hxx"
#include "op_bnd_cond_dir_omp.hxx"

#ifdef ENABLE_OPENACC
#include "op_bnd_cond_dir_acc.hxx"
#endif

#ifdef ENABLE_OPENMPX
#include "op_bnd_cond_dir_ompx.hxx"
#endif

// Factory method to instantiate op_bnd_cond_dir_gpu_t derived class
// based on the chosen GPU offload backend
namespace op_bnd_cond_dir_gpu
{
    op_bnd_cond_dir_gpu_t* create(const mesh_5d_t& mesh)
    {
        switch (params_gpu_offload::get_backend())
        {
            case params_gpu_offload::backend_t::CPU:
                return new op_bnd_cond_dir_omp_t{mesh};
#ifdef ENABLE_OPENACC
            case params_gpu_offload::backend_t::ACC:
                return new op_bnd_cond_dir_acc_t{mesh};
#endif
#ifdef ENABLE_OPENMPX
            case params_gpu_offload::backend_t::OMPX:
                return new op_bnd_cond_dir_ompx_t{mesh};
#endif
            default:
                is_erroneous = true;
                return nullptr;
        }
    }
};

#endif
