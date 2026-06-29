#ifndef MESH_5D_FACTORY_HXX
#define MESH_5D_FACTORY_HXX

#include "params_gpu_offload.hxx"
#include "mesh_5d.hxx"
#include "mesh_5d_omp.hxx"

#ifdef ENABLE_OPENACC
#include "mesh_5d_acc.hxx"
#endif

#ifdef ENABLE_OPENMPX
#include "mesh_5d_ompx.hxx"
#endif

namespace mesh_5d
{
    // Factory method to instantiate mesh_5D_t derived class
    // based on the chosen GPU offload backend
    mesh_5d_t* create(mesh_5d_data_t* mesh_data)
    {
        switch (params_gpu_offload::get_backend())
        {
            case params_gpu_offload::backend_t::CPU:
                return new mesh_5d_omp_t(mesh_data);
#ifdef ENABLE_OPENACC
            case params_gpu_offload::backend_t::ACC:
                return new mesh_5d_acc_t(mesh_data);
#endif
#ifdef ENABLE_OPENMPX
            case params_gpu_offload::backend_t::OMPX:
                return new mesh_5d_ompx_t(mesh_data);
#endif
            default:
                is_erroneous = true;
                return nullptr;
        }
    }
}

#endif
