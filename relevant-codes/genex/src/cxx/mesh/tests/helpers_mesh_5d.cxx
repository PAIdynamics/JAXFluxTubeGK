#ifndef HELPERS_MESH_5D_CXX
#define HELPERS_MESH_5D_CXX

#include "genex_cxx_env.hxx"
#include "params_gpu_offload.hxx"
#include "mesh_5d.hxx"
#include "mesh_5d_ompx.hxx"
#include <omp.h>

#ifdef ENABLE_OPENACC
#include <openacc.h>
#endif

// Copy the scalar members from the source mesh to target mesh struct
// via OpenMP on CPU
int32_t mesh_copy_scalar_cxx(const mesh_5d_t& mesh_src,
                             struct mesh_5d_data_t* mesh_data_tgt)
{
    mesh_data_tgt->size_RZ          = mesh_src.get_size_RZ();
    mesh_data_tgt->size_phi         = mesh_src.get_size_phi();
    mesh_data_tgt->size_vp          = mesh_src.get_size_vp();
    mesh_data_tgt->size_mu          = mesh_src.get_size_mu();
    mesh_data_tgt->size_sp          = mesh_src.get_size_sp();
    mesh_data_tgt->order_RZ_stencil = mesh_src.get_order_RZ_stencil();
    mesh_data_tgt->lb_phi           = mesh_src.get_lb_phi();
    mesh_data_tgt->ub_phi           = mesh_src.get_ub_phi();
    mesh_data_tgt->delta_RZ         = mesh_src.get_delta_RZ();
    mesh_data_tgt->delta_phi        = mesh_src.get_delta_phi();
    mesh_data_tgt->delta_vp         = mesh_src.get_delta_vp();
    mesh_data_tgt->delta_sqrt_mu    = mesh_src.get_delta_sqrt_mu();

    return 0;
}

// Copy the array members from the source mesh to target mesh struct
// via OpenMP on CPU
int32_t mesh_copy_array_cxx(const mesh_5d_t& mesh_src,
                            struct mesh_5d_data_t* mesh_data_tgt)
{
    // Assign the array member pointers of the target mesh
    int32_t* neighbors_tgt    = mesh_data_tgt->neighbors_ptr;
    int32_t* buf_zone_tgt     = mesh_data_tgt->buf_zone_ptr;
    int32_t* RZ_indices_tgt   = mesh_data_tgt->RZ_indices_ptr;
    real_t* not_filler_tgt    = mesh_data_tgt->not_filler_ptr;
    real_t* is_compute_tgt    = mesh_data_tgt->is_compute_ptr;
    real_t* vp_tgt            = mesh_data_tgt->vp_grid_ptr;
    real_t* mu_tgt            = mesh_data_tgt->mu_grid_ptr;
    real_t* sqrt_mu_tgt       = mesh_data_tgt->sqrt_mu_grid_ptr;
    real_t* vpw_tgt           = mesh_data_tgt->vp_weights_ptr;
    real_t* muw_tgt           = mesh_data_tgt->mu_weights_ptr;
    real_t* jacobian_tgt      = mesh_data_tgt->jacobian_buffer_ptr;

    // Assign the shape of the mesh grid
    int32_t size_phi         = mesh_src.get_size_phi();
    int32_t size_RZ          = mesh_src.get_size_RZ();
    int32_t size_vp          = mesh_src.get_size_vp();
    int32_t size_mu          = mesh_src.get_size_mu();
    int32_t order_RZ_stencil = mesh_src.get_order_RZ_stencil();
    int32_t size_RZ_stencil  = 2 * order_RZ_stencil + 1;
    int32_t lb_phi           = mesh_src.get_lb_phi();
    int32_t ub_phi           = mesh_src.get_ub_phi();

    // Copy the array members from the source to the target mesh via OpenMP
    #pragma omp parallel default(none) \
                         firstprivate(size_phi, size_RZ, size_vp, size_mu) \
                         firstprivate(order_RZ_stencil, size_RZ_stencil) \
                         firstprivate(lb_phi, ub_phi) \
                         shared(mesh_src, neighbors_tgt, buf_zone_tgt) \
                         shared(RZ_indices_tgt, not_filler_tgt) \
                         shared(is_compute_tgt, vp_tgt, mu_tgt, sqrt_mu_tgt) \
                         shared(vpw_tgt, muw_tgt, jacobian_tgt)
    {
    #pragma omp for simd schedule(static) nowait collapse(4)
    for (int32_t k = 1; k <= size_phi; k++)
    for (int32_t i = 1; i <= size_RZ; i++)
    for (int32_t oz = -order_RZ_stencil; oz <= order_RZ_stencil; oz++)
    for (int32_t ox = -order_RZ_stencil; ox <= order_RZ_stencil; ox++)
    {
        int32_t idx = (ox + order_RZ_stencil) +
                      (oz + order_RZ_stencil) * size_RZ_stencil +
                      (i - 1) * size_RZ_stencil * size_RZ_stencil +
                      (k - 1) * size_RZ_stencil * size_RZ_stencil * size_RZ;

        neighbors_tgt[idx] = mesh_src.neighbors(ox, oz, i, k);
    }

    #pragma omp for simd schedule(static) nowait
    for (int32_t l = 1; l <= size_vp; l++) {
        vp_tgt[l - 1]  = mesh_src.vp(l);
        vpw_tgt[l - 1] = mesh_src.vpw(l);
    }

    #pragma omp for simd schedule(static) nowait
    for (int32_t m = 1; m <= size_mu; m++) {
        mu_tgt[m - 1]       = mesh_src.mu(m);
        sqrt_mu_tgt[m - 1]  = mesh_src.sqrt_mu(m);
        muw_tgt[m - 1]      = mesh_src.muw(m);
    }

    #pragma omp for simd schedule(static) nowait collapse(2)
    for (int32_t k = lb_phi; k <= ub_phi; k++)
    for (int32_t i = 1; i <= size_RZ; i++)
    {
        int32_t idx = (i - 1) + (k - lb_phi) * size_RZ;

        buf_zone_tgt[idx]   = mesh_src.buf_zone(i, k);
        RZ_indices_tgt[idx] = mesh_src.RZ_indices(i, k);
        not_filler_tgt[idx] = mesh_src.not_filler(i, k);
        is_compute_tgt[idx] = mesh_src.is_compute(i, k);
        jacobian_tgt[idx]   = mesh_src.jacobian(i, k);
    }
    }

    return 0;
}

#ifdef ENABLE_OPENACC

// Copy the scalar members from the source mesh to target mesh struct
// via OpenACC on GPU
int32_t mesh_copy_scalar_acc(const mesh_5d_t& mesh_src,
                             struct mesh_5d_data_t* mesh_data_tgt)
{
    // Integer and real type array containers for scalar class members
    int32_t int_scalar_tgt[8] {};
    real_t real_scalar_tgt[4] {};

    // Allocate data on the device
    #pragma acc enter data create(int_scalar_tgt[:8], real_scalar_tgt[:4])

    // Copy the scalar members from the source to the target mesh via OpenACC
    #pragma acc parallel default(none)  \
        present(mesh_src, int_scalar_tgt, real_scalar_tgt)
    {
        // Copy the scalar members from the source mesh
        int_scalar_tgt[0]  = mesh_src.get_size_RZ();
        int_scalar_tgt[1]  = mesh_src.get_size_phi();
        int_scalar_tgt[2]  = mesh_src.get_size_vp();
        int_scalar_tgt[3]  = mesh_src.get_size_mu();
        int_scalar_tgt[4]  = mesh_src.get_size_sp();
        int_scalar_tgt[5]  = mesh_src.get_order_RZ_stencil();
        int_scalar_tgt[6]  = mesh_src.get_lb_phi();
        int_scalar_tgt[7]  = mesh_src.get_ub_phi();
        real_scalar_tgt[0] = mesh_src.get_delta_RZ();
        real_scalar_tgt[1] = mesh_src.get_delta_phi();
        real_scalar_tgt[2] = mesh_src.get_delta_vp();
        real_scalar_tgt[3] = mesh_src.get_delta_sqrt_mu();
    }

    // Copy data from the device to the host and deallocate memory on the device
    #pragma acc exit data copyout(int_scalar_tgt[:8], real_scalar_tgt[:4])

    // Copy the scalar members to the target mesh
    mesh_data_tgt->size_RZ          = int_scalar_tgt[0];
    mesh_data_tgt->size_phi         = int_scalar_tgt[1];
    mesh_data_tgt->size_vp          = int_scalar_tgt[2];
    mesh_data_tgt->size_mu          = int_scalar_tgt[3];
    mesh_data_tgt->size_sp          = int_scalar_tgt[4];
    mesh_data_tgt->order_RZ_stencil = int_scalar_tgt[5];
    mesh_data_tgt->lb_phi           = int_scalar_tgt[6];
    mesh_data_tgt->ub_phi           = int_scalar_tgt[7];
    mesh_data_tgt->delta_RZ         = real_scalar_tgt[0];
    mesh_data_tgt->delta_phi        = real_scalar_tgt[1];
    mesh_data_tgt->delta_vp         = real_scalar_tgt[2];
    mesh_data_tgt->delta_sqrt_mu    = real_scalar_tgt[3];

    return 0;
}

// Copy the array members from the source mesh to target mesh struct
// via OpenACC on GPU
int32_t mesh_copy_array_acc(const mesh_5d_t& mesh_src,
                            struct mesh_5d_data_t* mesh_data_tgt)
{
    // Assign the array member pointers of the target mesh
    int32_t* neighbors_tgt    = mesh_data_tgt->neighbors_ptr;
    int32_t* buf_zone_tgt     = mesh_data_tgt->buf_zone_ptr;
    int32_t* RZ_indices_tgt   = mesh_data_tgt->RZ_indices_ptr;
    real_t* not_filler_tgt    = mesh_data_tgt->not_filler_ptr;
    real_t* is_compute_tgt    = mesh_data_tgt->is_compute_ptr;
    real_t* vp_tgt            = mesh_data_tgt->vp_grid_ptr;
    real_t* mu_tgt            = mesh_data_tgt->mu_grid_ptr;
    real_t* sqrt_mu_tgt       = mesh_data_tgt->sqrt_mu_grid_ptr;
    real_t* vpw_tgt           = mesh_data_tgt->vp_weights_ptr;
    real_t* muw_tgt           = mesh_data_tgt->mu_weights_ptr;
    real_t* jacobian_tgt      = mesh_data_tgt->jacobian_buffer_ptr;

    // Define dummy variables and mesh shape
    int32_t size_RZ          = mesh_src.get_size_RZ();
    int32_t size_phi         = mesh_src.get_size_phi();
    int32_t size_vp          = mesh_src.get_size_vp();
    int32_t size_mu          = mesh_src.get_size_mu();
    int32_t order_RZ_stencil = mesh_src.get_order_RZ_stencil();
    int32_t lb_phi           = mesh_src.get_lb_phi();
    int32_t ub_phi           = mesh_src.get_ub_phi();
    int32_t size_RZ_stencil  = 2 * order_RZ_stencil + 1;
    int32_t size_neigh_arr   = size_RZ_stencil * size_RZ_stencil
                             * size_RZ * size_phi;
    int32_t size_3d          = size_RZ * (ub_phi - lb_phi + 1);

    // Allocate and copy data from the host to the device
    #pragma acc enter data copyin(size_RZ)
    #pragma acc enter data copyin(size_phi)
    #pragma acc enter data copyin(size_vp)
    #pragma acc enter data copyin(size_mu)
    #pragma acc enter data copyin(order_RZ_stencil)
    #pragma acc enter data copyin(size_RZ_stencil)
    #pragma acc enter data copyin(lb_phi)
    #pragma acc enter data copyin(ub_phi)
    #pragma acc enter data create(neighbors_tgt[:size_neigh_arr])
    #pragma acc enter data create(buf_zone_tgt[:size_3d])
    #pragma acc enter data create(RZ_indices_tgt[:size_3d])
    #pragma acc enter data create(not_filler_tgt[:size_3d])
    #pragma acc enter data create(is_compute_tgt[:size_3d])
    #pragma acc enter data create(vp_tgt[:size_vp])
    #pragma acc enter data create(mu_tgt[:size_mu])
    #pragma acc enter data create(sqrt_mu_tgt[:size_mu])
    #pragma acc enter data create(vpw_tgt[:size_vp])
    #pragma acc enter data create(muw_tgt[:size_mu])
    #pragma acc enter data create(jacobian_tgt[:size_3d])

    // Copy the array members from the source to the target mesh via OpenACC
    #pragma acc parallel default(none) \
                         present(size_phi, size_RZ, size_vp, size_mu) \
                         present(order_RZ_stencil, size_RZ_stencil) \
                         present(lb_phi, ub_phi) \
                         present(mesh_src, neighbors_tgt, buf_zone_tgt) \
                         present(RZ_indices_tgt, not_filler_tgt) \
                         present(is_compute_tgt, vp_tgt, mu_tgt, sqrt_mu_tgt) \
                         present(vpw_tgt, muw_tgt, jacobian_tgt)
    {
    #pragma acc loop independent collapse(4)
    for (int32_t k = 1; k <= size_phi; k++)
    for (int32_t i = 1; i <= size_RZ; i++)
    for (int32_t oz = -order_RZ_stencil; oz <= order_RZ_stencil; oz++)
    for (int32_t ox = -order_RZ_stencil; ox <= order_RZ_stencil; ox++)
    {
        int32_t idx = (ox + order_RZ_stencil) +
                      (oz + order_RZ_stencil) * size_RZ_stencil +
                      (i - 1) * size_RZ_stencil * size_RZ_stencil +
                      (k - 1) * size_RZ_stencil * size_RZ_stencil * size_RZ;

        neighbors_tgt[idx] = mesh_src.neighbors(ox, oz, i, k);
    }

    #pragma acc loop independent
    for (int32_t l = 1; l <= size_vp; l++) {
        vp_tgt[l - 1]  = mesh_src.vp(l);
        vpw_tgt[l - 1] = mesh_src.vpw(l);
    }

    #pragma acc loop independent
    for (int32_t m = 1; m <= size_mu; m++) {
        mu_tgt[m - 1]       = mesh_src.mu(m);
        sqrt_mu_tgt[m - 1]  = mesh_src.sqrt_mu(m);
        muw_tgt[m - 1]      = mesh_src.muw(m);
    }

    #pragma acc loop independent collapse(2)
    for (int32_t k = lb_phi; k <= ub_phi; k++)
    for (int32_t i = 1; i <= size_RZ; i++)
    {
        int32_t idx = (i - 1) + (k - lb_phi) * size_RZ;

        buf_zone_tgt[idx]   = mesh_src.buf_zone(i, k);
        RZ_indices_tgt[idx] = mesh_src.RZ_indices(i, k);
        not_filler_tgt[idx] = mesh_src.not_filler(i, k);
        is_compute_tgt[idx] = mesh_src.is_compute(i, k);
        jacobian_tgt[idx]   = mesh_src.jacobian(i, k);
    }
    }

    // Copy data from the device to the host and deallocate memory on the device
    #pragma acc exit data copyout(neighbors_tgt[:size_neigh_arr])
    #pragma acc exit data copyout(buf_zone_tgt[:size_3d])
    #pragma acc exit data copyout(RZ_indices_tgt[:size_3d])
    #pragma acc exit data copyout(not_filler_tgt[:size_3d])
    #pragma acc exit data copyout(is_compute_tgt[:size_3d])
    #pragma acc exit data copyout(vp_tgt[:size_vp])
    #pragma acc exit data copyout(mu_tgt[:size_mu])
    #pragma acc exit data copyout(sqrt_mu_tgt[:size_mu])
    #pragma acc exit data copyout(vpw_tgt[:size_vp])
    #pragma acc exit data copyout(muw_tgt[:size_mu])
    #pragma acc exit data copyout(jacobian_tgt[:size_3d])

    // Deallocate data in the device
    #pragma acc exit data delete(size_RZ)
    #pragma acc exit data delete(size_phi)
    #pragma acc exit data delete(size_vp)
    #pragma acc exit data delete(size_mu)
    #pragma acc exit data delete(order_RZ_stencil)
    #pragma acc exit data delete(size_RZ_stencil)
    #pragma acc exit data delete(lb_phi)
    #pragma acc exit data delete(ub_phi)

    return 0;
}

#endif

#ifdef ENABLE_OPENMPX

// Copy the scalar members from the source mesh to target mesh struct
// via OpenMP offload on GPU
int32_t mesh_copy_scalar_ompx(const mesh_5d_t& mesh_src,
                              struct mesh_5d_data_t* mesh_data_tgt)
{
    // Integer and real type array containers for scalar class members
    int32_t int_scalar_tgt[8] {};
    real_t real_scalar_tgt[4] {};

    // Allocate data on the device
    #pragma omp target enter data map(alloc: int_scalar_tgt[:8], \
                                             real_scalar_tgt[:4])

    // Copy the scalar members from the source to the target mesh
    // via OpenMP offload
    #pragma omp target teams default(none) defaultmap(default: pointer) \
        shared(mesh_src, int_scalar_tgt, real_scalar_tgt)
    {
        // Copy the scalar members from the source mesh
        int_scalar_tgt[0]  = mesh_src.get_size_RZ();
        int_scalar_tgt[1]  = mesh_src.get_size_phi();
        int_scalar_tgt[2]  = mesh_src.get_size_vp();
        int_scalar_tgt[3]  = mesh_src.get_size_mu();
        int_scalar_tgt[4]  = mesh_src.get_size_sp();
        int_scalar_tgt[5]  = mesh_src.get_order_RZ_stencil();
        int_scalar_tgt[6]  = mesh_src.get_lb_phi();
        int_scalar_tgt[7]  = mesh_src.get_ub_phi();
        real_scalar_tgt[0] = mesh_src.get_delta_RZ();
        real_scalar_tgt[1] = mesh_src.get_delta_phi();
        real_scalar_tgt[2] = mesh_src.get_delta_vp();
        real_scalar_tgt[3] = mesh_src.get_delta_sqrt_mu();
    }

    // Copy data from the device to the host and deallocate memory on the device
    #pragma omp target exit data map(from: int_scalar_tgt[:8], \
                                           real_scalar_tgt[:4])

    // Copy the scalar members to the target mesh
    mesh_data_tgt->size_RZ          = int_scalar_tgt[0];
    mesh_data_tgt->size_phi         = int_scalar_tgt[1];
    mesh_data_tgt->size_vp          = int_scalar_tgt[2];
    mesh_data_tgt->size_mu          = int_scalar_tgt[3];
    mesh_data_tgt->size_sp          = int_scalar_tgt[4];
    mesh_data_tgt->order_RZ_stencil = int_scalar_tgt[5];
    mesh_data_tgt->lb_phi           = int_scalar_tgt[6];
    mesh_data_tgt->ub_phi           = int_scalar_tgt[7];
    mesh_data_tgt->delta_RZ         = real_scalar_tgt[0];
    mesh_data_tgt->delta_phi        = real_scalar_tgt[1];
    mesh_data_tgt->delta_vp         = real_scalar_tgt[2];
    mesh_data_tgt->delta_sqrt_mu    = real_scalar_tgt[3];

    return 0;
}

// Copy the array members from the source mesh to target mesh struct
// via OpenMP offload on GPU
int32_t mesh_copy_array_ompx(const mesh_5d_t& mesh_src,
                             struct mesh_5d_data_t* mesh_data_tgt)
{
    // Assign the array member pointers of the target mesh
    int32_t* neighbors_tgt  = mesh_data_tgt->neighbors_ptr;
    int32_t* buf_zone_tgt   = mesh_data_tgt->buf_zone_ptr;
    int32_t* RZ_indices_tgt = mesh_data_tgt->RZ_indices_ptr;
    real_t* not_filler_tgt  = mesh_data_tgt->not_filler_ptr;
    real_t* is_compute_tgt  = mesh_data_tgt->is_compute_ptr;
    real_t* vp_tgt          = mesh_data_tgt->vp_grid_ptr;
    real_t* mu_tgt          = mesh_data_tgt->mu_grid_ptr;
    real_t* sqrt_mu_tgt     = mesh_data_tgt->sqrt_mu_grid_ptr;
    real_t* vpw_tgt         = mesh_data_tgt->vp_weights_ptr;
    real_t* muw_tgt         = mesh_data_tgt->mu_weights_ptr;
    real_t* jacobian_tgt    = mesh_data_tgt->jacobian_buffer_ptr;

    // Define dummy variable
    int32_t size_RZ          = mesh_src.get_size_RZ();
    int32_t size_phi         = mesh_src.get_size_phi();
    int32_t size_vp          = mesh_src.get_size_vp();
    int32_t size_mu          = mesh_src.get_size_mu();
    int32_t order_RZ_stencil = mesh_src.get_order_RZ_stencil();
    int32_t lb_phi           = mesh_src.get_lb_phi();
    int32_t ub_phi           = mesh_src.get_ub_phi();
    int32_t size_RZ_stencil  = 2 * order_RZ_stencil + 1;
    int32_t size_neigh_arr   = size_RZ_stencil * size_RZ_stencil
                             * size_RZ * size_phi;
    int32_t size_3d          = size_RZ * (ub_phi - lb_phi + 1);

    // Allocate and copy data from the host to the device
    #pragma omp target enter data map(alloc: neighbors_tgt[:size_neigh_arr])
    #pragma omp target enter data map(alloc: buf_zone_tgt[:size_3d])
    #pragma omp target enter data map(alloc: RZ_indices_tgt[:size_3d])
    #pragma omp target enter data map(alloc: not_filler_tgt[:size_3d])
    #pragma omp target enter data map(alloc: is_compute_tgt[:size_3d])
    #pragma omp target enter data map(alloc: vp_tgt[:size_vp])
    #pragma omp target enter data map(alloc: mu_tgt[:size_mu])
    #pragma omp target enter data map(alloc: sqrt_mu_tgt[:size_mu])
    #pragma omp target enter data map(alloc: vpw_tgt[:size_vp])
    #pragma omp target enter data map(alloc: muw_tgt[:size_mu])
    #pragma omp target enter data map(alloc: jacobian_tgt[:size_3d])

    // Copy the array members from the source to the target mesh
    // via OpenMP offload
    #pragma omp target teams distribute parallel for simd collapse(4) \
        default(none) defaultmap(default: pointer) \
        shared(mesh_src, neighbors_tgt)
    for (int32_t k = 1; k <= mesh_src.get_size_phi(); k++)
    for (int32_t i = 1; i <= mesh_src.get_size_RZ(); i++)
    for (int32_t oz = -mesh_src.get_order_RZ_stencil();
                 oz <= mesh_src.get_order_RZ_stencil(); oz++)
    for (int32_t ox = -mesh_src.get_order_RZ_stencil();
                 ox <= mesh_src.get_order_RZ_stencil(); ox++)
    {
        int32_t size_stencil = 2 * mesh_src.get_order_RZ_stencil() + 1;
        int32_t idx = (ox + mesh_src.get_order_RZ_stencil())
                    + (oz + mesh_src.get_order_RZ_stencil()) * size_stencil
                    + (i - 1) * size_stencil * size_stencil
                    + (k - 1) * size_stencil * size_stencil
                    * mesh_src.get_size_RZ();

        neighbors_tgt[idx] = mesh_src.neighbors(ox, oz, i, k);
    }

    #pragma omp target teams distribute parallel for simd \
        default(none) defaultmap(default: pointer) \
        shared(mesh_src, vp_tgt, vpw_tgt)
    for (int32_t l = 1; l <= mesh_src.get_size_vp(); l++)
    {
        vp_tgt[l - 1]  = mesh_src.vp(l);
        vpw_tgt[l - 1] = mesh_src.vpw(l);
    }

    #pragma omp target teams distribute parallel for simd \
        default(none) defaultmap(default: pointer) \
        shared(mesh_src, mu_tgt, sqrt_mu_tgt, muw_tgt)
    for (int32_t m = 1; m <= mesh_src.get_size_mu(); m++)
    {
        mu_tgt[m - 1]       = mesh_src.mu(m);
        sqrt_mu_tgt[m - 1]  = mesh_src.sqrt_mu(m);
        muw_tgt[m - 1]      = mesh_src.muw(m);
    }

    #pragma omp target teams distribute parallel for simd collapse(2) \
        default(none) defaultmap(default: pointer) \
        shared(mesh_src, buf_zone_tgt, RZ_indices_tgt, not_filler_tgt, \
               is_compute_tgt, jacobian_tgt)
    for (int32_t k = mesh_src.get_lb_phi(); k <= mesh_src.get_ub_phi(); k++)
    for (int32_t i = 1; i <= mesh_src.get_size_RZ(); i++)
    {
        int32_t idx = (i - 1)
                    + (k - mesh_src.get_lb_phi()) * mesh_src.get_size_RZ();

        buf_zone_tgt[idx]   = mesh_src.buf_zone(i, k);
        RZ_indices_tgt[idx] = mesh_src.RZ_indices(i, k);
        not_filler_tgt[idx] = mesh_src.not_filler(i, k);
        is_compute_tgt[idx] = mesh_src.is_compute(i, k);
        jacobian_tgt[idx]   = mesh_src.jacobian(i, k);
    }

    // Copy data from the device to the host and deallocate memory on the device
    #pragma omp target exit data map(from: neighbors_tgt[:size_neigh_arr])
    #pragma omp target exit data map(from: buf_zone_tgt[:size_3d])
    #pragma omp target exit data map(from: RZ_indices_tgt[:size_3d])
    #pragma omp target exit data map(from: not_filler_tgt[:size_3d])
    #pragma omp target exit data map(from: is_compute_tgt[:size_3d])
    #pragma omp target exit data map(from: vp_tgt[:size_vp])
    #pragma omp target exit data map(from: mu_tgt[:size_mu])
    #pragma omp target exit data map(from: sqrt_mu_tgt[:size_mu])
    #pragma omp target exit data map(from: vpw_tgt[:size_vp])
    #pragma omp target exit data map(from: muw_tgt[:size_mu])
    #pragma omp target exit data map(from: jacobian_tgt[:size_3d])

    return 0;
}

#endif

#ifdef __cplusplus
extern "C" {
#endif

int32_t cbind_mesh_copy_scalar(const mesh_5d_t** mesh_src_cxx_pptr,
                               struct mesh_5d_data_t* mesh_data_tgt,
                               char* grid_type_phi, char* grid_type_vp,
                               char* grid_type_mu, char* quad_type_phi,
                               char* quad_type_vp, char* quad_type_mu)
{
    // Assign the source mesh_5d_t class instance
    const mesh_5d_t& mesh_src = *(*mesh_src_cxx_pptr);

    // Copy grid and quadrature types
    strcpy(grid_type_phi, mesh_5d::get_grid_type_phi().c_str());
    strcpy(grid_type_vp, mesh_5d::get_grid_type_vp().c_str());
    strcpy(grid_type_mu, mesh_5d::get_grid_type_mu().c_str());
    strcpy(quad_type_phi, mesh_5d::get_quad_type_phi().c_str());
    strcpy(quad_type_vp, mesh_5d::get_quad_type_vp().c_str());
    strcpy(quad_type_mu, mesh_5d::get_quad_type_mu().c_str());

    // Return 0 for success and 1 for error
    switch (params_gpu_offload::get_backend())
    {
        case params_gpu_offload::backend_t::CPU:
            return mesh_copy_scalar_cxx(mesh_src, mesh_data_tgt);
#ifdef ENABLE_OPENACC
        case params_gpu_offload::backend_t::ACC:
            return mesh_copy_scalar_acc(mesh_src, mesh_data_tgt);
#endif
#ifdef ENABLE_OPENMPX
        case params_gpu_offload::backend_t::OMPX:
            return mesh_copy_scalar_ompx(mesh_src, mesh_data_tgt);
#endif
        default:
            return 1;
    }
}

int32_t cbind_mesh_copy_array(const mesh_5d_t** mesh_src_cxx_pptr,
                              struct mesh_5d_data_t* mesh_data_tgt)
{
    // Assign the source mesh_5d_t class instance
    const mesh_5d_t& mesh_src = *(*mesh_src_cxx_pptr);

    // Return 0 for success and 1 for error
    switch (params_gpu_offload::get_backend())
    {
        case params_gpu_offload::backend_t::CPU:
            return mesh_copy_array_cxx(mesh_src, mesh_data_tgt);
#ifdef ENABLE_OPENACC
        case params_gpu_offload::backend_t::ACC:
            return mesh_copy_array_acc(mesh_src, mesh_data_tgt);
#endif
#ifdef ENABLE_OPENMPX
        case params_gpu_offload::backend_t::OMPX:
            return mesh_copy_array_ompx(mesh_src, mesh_data_tgt);
#endif
        default:
            return 1;
    }
}

#ifdef __cplusplus
}
#endif

#endif
