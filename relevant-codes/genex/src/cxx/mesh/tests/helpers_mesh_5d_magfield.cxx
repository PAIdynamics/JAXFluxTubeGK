#ifndef HELPERS_MESH_5D_MAGFIELD_CXX
#define HELPERS_MESH_5D_MAGFIELD_CXX

#include "genex_cxx_env.hxx"
#include "params_gpu_offload.hxx"
#include "mesh_5d.hxx"
#include <omp.h>

#ifdef ENABLE_OPENACC
#include <openacc.h>
#endif

// Copy the class members related to the magnetic field (magfield) from
// the source mesh to target mesh struct via OpenMP on CPU
int32_t mesh_copy_magfield_cxx(const mesh_5d_t& mesh_src,
                               struct mesh_5d_data_t* mesh_data_tgt)
{
    // Assign the array member pointers of the target mesh
    real_t* absB_tgt          = mesh_data_tgt->absB_buffer_ptr;
    real_t* normb_R_tgt       = mesh_data_tgt->normb_R_buffer_ptr;
    real_t* normb_Z_tgt       = mesh_data_tgt->normb_Z_buffer_ptr;
    real_t* curl_normb_y_tgt  = mesh_data_tgt->curl_normb_y_ptr;
    real_t* dgyxdy_over_g_tgt = mesh_data_tgt->dgyxdy_over_g_ptr;
    real_t* dgyzdy_over_g_tgt = mesh_data_tgt->dgyzdy_over_g_ptr;
    real_t* dgyxdz_over_g_tgt = mesh_data_tgt->dgyxdz_over_g_ptr;
    real_t* dgyzdx_over_g_tgt = mesh_data_tgt->dgyzdx_over_g_ptr;
    real_t* inv_g_tgt         = mesh_data_tgt->inv_g_ptr;
    real_t* dabsBdx_tgt       = mesh_data_tgt->dabsBdx_ptr;
    real_t* dabsBdz_tgt       = mesh_data_tgt->dabsBdz_ptr;
    real_t* dabsBdy_tgt       = mesh_data_tgt->dabsBdy_ptr;

    // Assign the shape of the mesh grid
    int32_t size_RZ  = mesh_src.get_size_RZ();
    int32_t lb_phi   = mesh_src.get_lb_phi();
    int32_t ub_phi   = mesh_src.get_ub_phi();

    // Copy the array members from the source to the target mesh via OpenMP
    #pragma omp parallel default(none) \
                         firstprivate(size_RZ, lb_phi, ub_phi) \
                         shared(mesh_src, absB_tgt, normb_R_tgt) \
                         shared(normb_Z_tgt, curl_normb_y_tgt) \
                         shared(dgyxdy_over_g_tgt, dgyzdy_over_g_tgt) \
                         shared(dgyxdz_over_g_tgt, dgyzdx_over_g_tgt) \
                         shared(inv_g_tgt, dabsBdx_tgt) \
                         shared(dabsBdy_tgt, dabsBdz_tgt)
    {
    #pragma omp for simd schedule(static) nowait collapse(2)
    for (int32_t k = lb_phi; k <= ub_phi; k++)
    for (int32_t i = 1; i <= size_RZ; i++)
    {
        int idx = (i - 1) + (k - 1) * size_RZ;

        absB_tgt[idx]          = mesh_src.absB(i, k);
        normb_R_tgt[idx]       = mesh_src.normb_R(i, k);
        normb_Z_tgt[idx]       = mesh_src.normb_Z(i, k);
        curl_normb_y_tgt[idx]  = mesh_src.curl_normb_y(i, k);
        dgyxdy_over_g_tgt[idx] = mesh_src.dgyxdy_over_g(i, k);
        dgyzdy_over_g_tgt[idx] = mesh_src.dgyzdy_over_g(i, k);
        dgyxdz_over_g_tgt[idx] = mesh_src.dgyxdz_over_g(i, k);
        dgyzdx_over_g_tgt[idx] = mesh_src.dgyzdx_over_g(i, k);
        inv_g_tgt[idx]         = mesh_src.inv_g(i, k);
        dabsBdx_tgt[idx]       = mesh_src.dabsBdx(i, k);
        dabsBdz_tgt[idx]       = mesh_src.dabsBdz(i, k);
        dabsBdy_tgt[idx]       = mesh_src.dabsBdy(i, k);
    }
    }
    return 0;
}

#ifdef ENABLE_OPENACC

// Copy the array members related to the magnetic field (magfield) from
// the source mesh to target mesh struct via OpenACC on GPU
int32_t mesh_copy_magfield_acc(const mesh_5d_t& mesh_src,
                               struct mesh_5d_data_t* mesh_data_tgt)
{
    // Assign the array member pointers of the target mesh
    real_t* absB_tgt          = mesh_data_tgt->absB_buffer_ptr;
    real_t* normb_R_tgt       = mesh_data_tgt->normb_R_buffer_ptr;
    real_t* normb_Z_tgt       = mesh_data_tgt->normb_Z_buffer_ptr;
    real_t* curl_normb_y_tgt  = mesh_data_tgt->curl_normb_y_ptr;
    real_t* dgyxdy_over_g_tgt = mesh_data_tgt->dgyxdy_over_g_ptr;
    real_t* dgyzdy_over_g_tgt = mesh_data_tgt->dgyzdy_over_g_ptr;
    real_t* dgyxdz_over_g_tgt = mesh_data_tgt->dgyxdz_over_g_ptr;
    real_t* dgyzdx_over_g_tgt = mesh_data_tgt->dgyzdx_over_g_ptr;
    real_t* inv_g_tgt         = mesh_data_tgt->inv_g_ptr;
    real_t* dabsBdx_tgt       = mesh_data_tgt->dabsBdx_ptr;
    real_t* dabsBdz_tgt       = mesh_data_tgt->dabsBdz_ptr;
    real_t* dabsBdy_tgt       = mesh_data_tgt->dabsBdy_ptr;

    // Define dummy variables and mesh shape
    int32_t size_RZ       = mesh_src.get_size_RZ();
    int32_t size_phi      = mesh_src.get_size_phi();
    int32_t size_mesh_3d  = size_RZ * size_phi;

    // Allocate and copy data from the host to the device
    #pragma acc enter data copyin(size_RZ)
    #pragma acc enter data copyin(size_phi)
    #pragma acc enter data create(absB_tgt[:size_mesh_3d])
    #pragma acc enter data create(normb_R_tgt[:size_mesh_3d])
    #pragma acc enter data create(normb_Z_tgt[:size_mesh_3d])
    #pragma acc enter data create(curl_normb_y_tgt[:size_mesh_3d])
    #pragma acc enter data create(dgyxdy_over_g_tgt[:size_mesh_3d])
    #pragma acc enter data create(dgyzdy_over_g_tgt[:size_mesh_3d])
    #pragma acc enter data create(dgyxdz_over_g_tgt[:size_mesh_3d])
    #pragma acc enter data create(dgyzdx_over_g_tgt[:size_mesh_3d])
    #pragma acc enter data create(inv_g_tgt[:size_mesh_3d])
    #pragma acc enter data create(dabsBdx_tgt[:size_mesh_3d])
    #pragma acc enter data create(dabsBdz_tgt[:size_mesh_3d])
    #pragma acc enter data create(dabsBdy_tgt[:size_mesh_3d])

    // Copy the array members from the source to the target mesh via OpenACC
    #pragma acc parallel default(none) \
                         present(size_RZ, size_phi) \
                         present(mesh_src, absB_tgt, normb_R_tgt) \
                         present(normb_Z_tgt, curl_normb_y_tgt) \
                         present(dgyxdy_over_g_tgt, dgyzdy_over_g_tgt) \
                         present(dgyxdz_over_g_tgt, dgyzdx_over_g_tgt) \
                         present(inv_g_tgt, dabsBdx_tgt) \
                         present(dabsBdz_tgt, dabsBdy_tgt)
    {
    #pragma acc loop independent collapse(2)
    for (int32_t k = 1; k <= size_phi; k++)
    for (int32_t i = 1; i <= size_RZ; i++)
    {
        int idx = (i - 1) + (k - 1) * size_RZ;

        absB_tgt[idx]          = mesh_src.absB(i, k);
        normb_R_tgt[idx]       = mesh_src.normb_R(i, k);
        normb_Z_tgt[idx]       = mesh_src.normb_Z(i, k);
        curl_normb_y_tgt[idx]  = mesh_src.curl_normb_y(i, k);
        dgyxdy_over_g_tgt[idx] = mesh_src.dgyxdy_over_g(i, k);
        dgyzdy_over_g_tgt[idx] = mesh_src.dgyzdy_over_g(i, k);
        dgyxdz_over_g_tgt[idx] = mesh_src.dgyxdz_over_g(i, k);
        dgyzdx_over_g_tgt[idx] = mesh_src.dgyzdx_over_g(i, k);
        inv_g_tgt[idx]         = mesh_src.inv_g(i, k);
        dabsBdx_tgt[idx]       = mesh_src.dabsBdx(i, k);
        dabsBdz_tgt[idx]       = mesh_src.dabsBdz(i, k);
        dabsBdy_tgt[idx]       = mesh_src.dabsBdy(i, k);
    }
    }

    // Copy data from the device to the host and deallocate memory on the device
    #pragma acc exit data copyout(absB_tgt[:size_mesh_3d])
    #pragma acc exit data copyout(normb_R_tgt[:size_mesh_3d])
    #pragma acc exit data copyout(normb_Z_tgt[:size_mesh_3d])
    #pragma acc exit data copyout(curl_normb_y_tgt[:size_mesh_3d])
    #pragma acc exit data copyout(dgyxdy_over_g_tgt[:size_mesh_3d])
    #pragma acc exit data copyout(dgyzdy_over_g_tgt[:size_mesh_3d])
    #pragma acc exit data copyout(dgyxdz_over_g_tgt[:size_mesh_3d])
    #pragma acc exit data copyout(dgyzdx_over_g_tgt[:size_mesh_3d])
    #pragma acc exit data copyout(inv_g_tgt[:size_mesh_3d])
    #pragma acc exit data copyout(dabsBdx_tgt[:size_mesh_3d])
    #pragma acc exit data copyout(dabsBdz_tgt[:size_mesh_3d])
    #pragma acc exit data copyout(dabsBdy_tgt[:size_mesh_3d])

    // Deallocate data in the device
    #pragma acc exit data delete(size_RZ)
    #pragma acc exit data delete(size_phi)

    return 0;
}

#endif

#ifdef ENABLE_OPENMPX

// Copy the array members related to the magnetic field (magfield) from
// the source mesh to target mesh struct via OpenMP offload on GPU
int32_t mesh_copy_magfield_ompx(const mesh_5d_t& mesh_src,
                                struct mesh_5d_data_t* mesh_data_tgt)
{
    // Assign the array member pointers of the target mesh
    real_t* absB_tgt          = mesh_data_tgt->absB_buffer_ptr;
    real_t* normb_R_tgt       = mesh_data_tgt->normb_R_buffer_ptr;
    real_t* normb_Z_tgt       = mesh_data_tgt->normb_Z_buffer_ptr;
    real_t* curl_normb_y_tgt  = mesh_data_tgt->curl_normb_y_ptr;
    real_t* dgyxdy_over_g_tgt = mesh_data_tgt->dgyxdy_over_g_ptr;
    real_t* dgyzdy_over_g_tgt = mesh_data_tgt->dgyzdy_over_g_ptr;
    real_t* dgyxdz_over_g_tgt = mesh_data_tgt->dgyxdz_over_g_ptr;
    real_t* dgyzdx_over_g_tgt = mesh_data_tgt->dgyzdx_over_g_ptr;
    real_t* inv_g_tgt         = mesh_data_tgt->inv_g_ptr;
    real_t* dabsBdx_tgt       = mesh_data_tgt->dabsBdx_ptr;
    real_t* dabsBdz_tgt       = mesh_data_tgt->dabsBdz_ptr;
    real_t* dabsBdy_tgt       = mesh_data_tgt->dabsBdy_ptr;

    // Define dummy variables and mesh shape
    int32_t size_RZ      = mesh_src.get_size_RZ();
    int32_t size_phi     = mesh_src.get_size_phi();
    int32_t size_mesh_3d = size_RZ * size_phi;

    // Allocate and copy data from the host to the device
    #pragma omp target enter data map(alloc: absB_tgt[:size_mesh_3d])
    #pragma omp target enter data map(alloc: normb_R_tgt[:size_mesh_3d])
    #pragma omp target enter data map(alloc: normb_Z_tgt[:size_mesh_3d])
    #pragma omp target enter data map(alloc: curl_normb_y_tgt[:size_mesh_3d])
    #pragma omp target enter data map(alloc: dgyxdy_over_g_tgt[:size_mesh_3d])
    #pragma omp target enter data map(alloc: dgyzdy_over_g_tgt[:size_mesh_3d])
    #pragma omp target enter data map(alloc: dgyxdz_over_g_tgt[:size_mesh_3d])
    #pragma omp target enter data map(alloc: dgyzdx_over_g_tgt[:size_mesh_3d])
    #pragma omp target enter data map(alloc: inv_g_tgt[:size_mesh_3d])
    #pragma omp target enter data map(alloc: dabsBdx_tgt[:size_mesh_3d])
    #pragma omp target enter data map(alloc: dabsBdz_tgt[:size_mesh_3d])
    #pragma omp target enter data map(alloc: dabsBdy_tgt[:size_mesh_3d])

    // Copy the array members from the source to the target mesh
    // via OpenMP offload
    #pragma omp target teams default(none) defaultmap(none) \
        firstprivate(size_RZ, size_phi) \
        shared(mesh_src, absB_tgt, normb_R_tgt, normb_Z_tgt) \
        shared(curl_normb_y_tgt, dgyxdy_over_g_tgt, dgyzdy_over_g_tgt) \
        shared(dgyxdz_over_g_tgt, dgyzdx_over_g_tgt) \
        shared(inv_g_tgt, dabsBdx_tgt, dabsBdz_tgt, dabsBdy_tgt)
    {
    #pragma omp distribute parallel for simd collapse(2)
    for (int32_t k = 1; k <= mesh_src.get_size_phi(); k++)
    for (int32_t i = 1; i <= mesh_src.get_size_RZ(); i++)
    {
        int idx = (i - 1) + (k - 1) * mesh_src.get_size_RZ();

        absB_tgt[idx]          = mesh_src.absB(i, k);
        normb_R_tgt[idx]       = mesh_src.normb_R(i, k);
        normb_Z_tgt[idx]       = mesh_src.normb_Z(i, k);
        curl_normb_y_tgt[idx]  = mesh_src.curl_normb_y(i, k);
        dgyxdy_over_g_tgt[idx] = mesh_src.dgyxdy_over_g(i, k);
        dgyzdy_over_g_tgt[idx] = mesh_src.dgyzdy_over_g(i, k);
        dgyxdz_over_g_tgt[idx] = mesh_src.dgyxdz_over_g(i, k);
        dgyzdx_over_g_tgt[idx] = mesh_src.dgyzdx_over_g(i, k);
        inv_g_tgt[idx]         = mesh_src.inv_g(i, k);
        dabsBdx_tgt[idx]       = mesh_src.dabsBdx(i, k);
        dabsBdz_tgt[idx]       = mesh_src.dabsBdz(i, k);
        dabsBdy_tgt[idx]       = mesh_src.dabsBdy(i, k);
    }
    }

    // Copy data from the device to the host and deallocate memory on the device
    #pragma omp target exit data map(from: absB_tgt[:size_mesh_3d])
    #pragma omp target exit data map(from: normb_R_tgt[:size_mesh_3d])
    #pragma omp target exit data map(from: normb_Z_tgt[:size_mesh_3d])
    #pragma omp target exit data map(from: curl_normb_y_tgt[:size_mesh_3d])
    #pragma omp target exit data map(from: dgyxdy_over_g_tgt[:size_mesh_3d])
    #pragma omp target exit data map(from: dgyzdy_over_g_tgt[:size_mesh_3d])
    #pragma omp target exit data map(from: dgyxdz_over_g_tgt[:size_mesh_3d])
    #pragma omp target exit data map(from: dgyzdx_over_g_tgt[:size_mesh_3d])
    #pragma omp target exit data map(from: inv_g_tgt[:size_mesh_3d])
    #pragma omp target exit data map(from: dabsBdx_tgt[:size_mesh_3d])
    #pragma omp target exit data map(from: dabsBdz_tgt[:size_mesh_3d])
    #pragma omp target exit data map(from: dabsBdy_tgt[:size_mesh_3d])

    return 0;
}

#endif

#ifdef __cplusplus
extern "C" {
#endif

int32_t cbind_mesh_copy_magfield(const mesh_5d_t** mesh_src_cxx_pptr,
                                 struct mesh_5d_data_t* mesh_data_tgt)
{
    // Assign the source mesh_5d_t class instance
    const mesh_5d_t& mesh_src = *(*mesh_src_cxx_pptr);

    // Return 0 for success and 1 for error
    switch (params_gpu_offload::get_backend())
    {
        case params_gpu_offload::backend_t::CPU:
            return mesh_copy_magfield_cxx(mesh_src, mesh_data_tgt);
#ifdef ENABLE_OPENACC
        case params_gpu_offload::backend_t::ACC:
            return mesh_copy_magfield_acc(mesh_src, mesh_data_tgt);
#endif
#ifdef ENABLE_OPENMPX
        case params_gpu_offload::backend_t::OMPX:
            return mesh_copy_magfield_ompx(mesh_src, mesh_data_tgt);
#endif
        default:
            return 1;
    }
}

#ifdef __cplusplus
}
#endif

#endif
