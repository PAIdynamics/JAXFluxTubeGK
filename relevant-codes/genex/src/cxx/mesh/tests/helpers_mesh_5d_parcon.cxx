#ifndef HELPERS_MESH_5D_PARCON_CXX
#define HELPERS_MESH_5D_PARCON_CXX

#include "genex_cxx_env.hxx"
#include "params_gpu_offload.hxx"
#include "mesh_5d.hxx"
#include <omp.h>

#ifdef ENABLE_OPENACC
#include <openacc.h>
#endif

// Copy the class members related to the parallel connection (parcon) from
// the source mesh to target mesh struct via OpenMP on CPU
int32_t mesh_copy_parcon_cxx(const mesh_5d_t& mesh_src,
                             struct mesh_5d_data_t* mesh_data_tgt)
{
    // Assign the member pointers of the target par_con
    int32_t* not_in_target_tgt = mesh_data_tgt->not_in_target_ptr;
    real_t* fll_positive1_tgt = mesh_data_tgt->fll_positive1_ptr;
    real_t* fll_positive2_tgt = mesh_data_tgt->fll_positive2_ptr;
    real_t* fll_negative1_tgt = mesh_data_tgt->fll_negative1_ptr;
    real_t* fll_negative2_tgt = mesh_data_tgt->fll_negative2_ptr;
    csrmat_genex_data_t* map_pp_data_tgt =
        mesh_data_tgt->map_positive2_data_ptr;
    csrmat_genex_data_t* map_p_data_tgt =
        mesh_data_tgt->map_positive1_data_ptr;
    csrmat_genex_data_t* map_m_data_tgt =
        mesh_data_tgt->map_negative1_data_ptr;
    csrmat_genex_data_t* map_mm_data_tgt =
        mesh_data_tgt->map_negative2_data_ptr;

    // Assign the shape of mesh grid
    int32_t size_RZ  = mesh_src.get_size_RZ();
    int32_t size_phi = mesh_src.get_size_phi();

    // Copy the array members of par_con
    #pragma omp parallel default(none) \
                     firstprivate(size_RZ, size_phi) \
                     shared(mesh_src) \
                     shared(not_in_target_tgt) \
                     shared(fll_positive1_tgt, fll_positive2_tgt) \
                     shared(fll_negative1_tgt, fll_negative2_tgt) \
                     shared(map_pp_data_tgt, map_p_data_tgt) \
                     shared(map_mm_data_tgt, map_m_data_tgt)
    {
    #pragma omp for simd schedule(static) nowait collapse(2)
    for (int32_t k = 1; k <= size_phi; k++)
    for (int32_t i = 1; i <= size_RZ; i++)
    {
        int32_t idx = (i - 1) + (k - 1) * size_RZ;

        not_in_target_tgt[idx] = mesh_src.not_in_target(i, k);
        fll_positive1_tgt[idx] = mesh_src.fll_positive1(i, k);
        fll_positive2_tgt[idx] = mesh_src.fll_positive2(i, k);
        fll_negative1_tgt[idx] = mesh_src.fll_negative1(i, k);
        fll_negative2_tgt[idx] = mesh_src.fll_negative2(i, k);
    }

    // Copy the class members of the map matrices
    for (int32_t k = 1; k <= size_phi; k++)
    {
        // Copy class members of the map_positive2 matrices
        int32_t ndim = mesh_src.map_positive2(k).get_ndim();
        int32_t ncol = mesh_src.map_positive2(k).get_ncol();
        int32_t nnz  = mesh_src.map_positive2(k).get_nnz();

        map_pp_data_tgt[k-1].ndim = ndim;
        map_pp_data_tgt[k-1].ncol = ncol;
        map_pp_data_tgt[k-1].nnz  = nnz;

        #pragma omp for simd schedule(static) nowait
        for (int32_t i = 1; i <= ndim + 1; i++)
        {
            map_pp_data_tgt[k-1].i_ptr[i-1] = mesh_src.map_positive2(k).i(i);
        }
        #pragma omp for simd schedule(static) nowait
        for (int32_t i = 1; i <= nnz; i++)
        {
            map_pp_data_tgt[k-1].j_ptr[i-1]   =
                mesh_src.map_positive2(k).j(i);
            map_pp_data_tgt[k-1].val_ptr[i-1] =
                mesh_src.map_positive2(k).val(i);
        }

        // Copy class members of the map_positive1 matrices
        ndim = mesh_src.map_positive1(k).get_ndim();
        ncol = mesh_src.map_positive1(k).get_ncol();
        nnz  = mesh_src.map_positive1(k).get_nnz();

        map_p_data_tgt[k-1].ndim = ndim;
        map_p_data_tgt[k-1].ncol = ncol;
        map_p_data_tgt[k-1].nnz  = nnz;

        #pragma omp for simd schedule(static) nowait
        for (int32_t i = 1; i <= ndim + 1; i++)
        {
            map_p_data_tgt[k-1].i_ptr[i-1] = mesh_src.map_positive1(k).i(i);
        }
        #pragma omp for simd schedule(static) nowait
        for (int32_t i = 1; i <= nnz; i++)
        {
            map_p_data_tgt[k-1].j_ptr[i-1]   =
                mesh_src.map_positive1(k).j(i);
            map_p_data_tgt[k-1].val_ptr[i-1] =
                mesh_src.map_positive1(k).val(i);
        }

        // Copy class members of the map_negative1 matrices
        ndim = mesh_src.map_negative1(k).get_ndim();
        ncol = mesh_src.map_negative1(k).get_ncol();
        nnz  = mesh_src.map_negative1(k).get_nnz();

        map_m_data_tgt[k-1].ndim = ndim;
        map_m_data_tgt[k-1].ncol = ncol;
        map_m_data_tgt[k-1].nnz  = nnz;

        #pragma omp for simd schedule(static) nowait
        for (int32_t i = 1; i <= ndim + 1; i++)
        {
            map_m_data_tgt[k-1].i_ptr[i-1] = mesh_src.map_negative1(k).i(i);
        }
        #pragma omp for simd schedule(static) nowait
        for (int32_t i = 1; i <= nnz; i++)
        {
            map_m_data_tgt[k-1].j_ptr[i-1]   =
                mesh_src.map_negative1(k).j(i);
            map_m_data_tgt[k-1].val_ptr[i-1] =
                mesh_src.map_negative1(k).val(i);
        }

        // Copy class members of the map_negative2 matrices
        ndim = mesh_src.map_negative2(k).get_ndim();
        ncol = mesh_src.map_negative2(k).get_ncol();
        nnz  = mesh_src.map_negative2(k).get_nnz();

        map_mm_data_tgt[k-1].ndim = ndim;
        map_mm_data_tgt[k-1].ncol = ncol;
        map_mm_data_tgt[k-1].nnz  = nnz;

        #pragma omp for simd schedule(static) nowait
        for (int32_t i = 1; i <= ndim + 1; i++)
        {
            map_mm_data_tgt[k-1].i_ptr[i-1] = mesh_src.map_negative2(k).i(i);
        }
        #pragma omp for simd schedule(static) nowait
        for (int32_t i = 1; i <= nnz; i++)
        {
            map_mm_data_tgt[k-1].j_ptr[i-1]   =
                mesh_src.map_negative2(k).j(i);
            map_mm_data_tgt[k-1].val_ptr[i-1] =
                mesh_src.map_negative2(k).val(i);
        }
    }
    }

    return 0;
}

#ifdef ENABLE_OPENACC

// Copy the array members related to the parallel connection (parcon) from
// the source mesh to target mesh struct via OpenACC on GPU
int32_t mesh_copy_parcon_acc(const mesh_5d_t& mesh_src,
                             struct mesh_5d_data_t* mesh_data_tgt)
{
    // Assign the member pointers of the target par_con
    int32_t* not_in_target_tgt = mesh_data_tgt->not_in_target_ptr;
    real_t* fll_positive1_tgt = mesh_data_tgt->fll_positive1_ptr;
    real_t* fll_positive2_tgt = mesh_data_tgt->fll_positive2_ptr;
    real_t* fll_negative1_tgt = mesh_data_tgt->fll_negative1_ptr;
    real_t* fll_negative2_tgt = mesh_data_tgt->fll_negative2_ptr;
    csrmat_genex_data_t* map_pp_data_tgt =
        mesh_data_tgt->map_positive2_data_ptr;
    csrmat_genex_data_t* map_p_data_tgt =
        mesh_data_tgt->map_positive1_data_ptr;
    csrmat_genex_data_t* map_m_data_tgt =
        mesh_data_tgt->map_negative1_data_ptr;
    csrmat_genex_data_t* map_mm_data_tgt =
        mesh_data_tgt->map_negative2_data_ptr;

    // Assign the shape of mesh grid
    int32_t size_RZ      = mesh_src.get_size_RZ();
    int32_t size_phi     = mesh_src.get_size_phi();
    int32_t size_mesh_3d = size_RZ * size_phi;

    // Allocate and copy the local variables to the device
    #pragma acc enter data copyin(size_RZ)
    #pragma acc enter data copyin(size_phi)

    // Allocate the array member of par_con on the device
    #pragma acc enter data create(not_in_target_tgt[:size_mesh_3d])
    #pragma acc enter data create(fll_positive1_tgt[:size_mesh_3d])
    #pragma acc enter data create(fll_positive2_tgt[:size_mesh_3d])
    #pragma acc enter data create(fll_negative1_tgt[:size_mesh_3d])
    #pragma acc enter data create(fll_negative2_tgt[:size_mesh_3d])

    // Allocate the array of map matrices on the device
    #pragma acc enter data create(map_pp_data_tgt[:size_phi])
    #pragma acc enter data create(map_p_data_tgt[:size_phi])
    #pragma acc enter data create(map_m_data_tgt[:size_phi])
    #pragma acc enter data create(map_mm_data_tgt[:size_phi])

    for (int32_t k = 1; k <= size_phi; k++)
    {
        int32_t ndim = mesh_src.map_positive2(k).get_ndim();
        int32_t nnz  = mesh_src.map_positive2(k).get_nnz();

        // Allocate the class members of map_positive2 matrices
        #pragma acc enter data create(map_pp_data_tgt[k-1].ndim)
        #pragma acc enter data create(map_pp_data_tgt[k-1].ncol)
        #pragma acc enter data create(map_pp_data_tgt[k-1].nnz)
        #pragma acc enter data create(map_pp_data_tgt[k-1].i_ptr[:ndim+1])
        #pragma acc enter data create(map_pp_data_tgt[k-1].j_ptr[:nnz])
        #pragma acc enter data create(map_pp_data_tgt[k-1].val_ptr[:nnz])

        ndim = mesh_src.map_positive1(k).get_ndim();
        nnz  = mesh_src.map_positive1(k).get_nnz();

        // Allocate the class members of map_positive1 matrices
        #pragma acc enter data create(map_p_data_tgt[k-1].ndim)
        #pragma acc enter data create(map_p_data_tgt[k-1].ncol)
        #pragma acc enter data create(map_p_data_tgt[k-1].nnz)
        #pragma acc enter data create(map_p_data_tgt[k-1].i_ptr[:ndim+1])
        #pragma acc enter data create(map_p_data_tgt[k-1].j_ptr[:nnz])
        #pragma acc enter data create(map_p_data_tgt[k-1].val_ptr[:nnz])

        ndim = mesh_src.map_negative1(k).get_ndim();
        nnz  = mesh_src.map_negative1(k).get_nnz();

        // Allocate the class members of map_negative1 matrices
        #pragma acc enter data create(map_m_data_tgt[k-1].ndim)
        #pragma acc enter data create(map_m_data_tgt[k-1].ncol)
        #pragma acc enter data create(map_m_data_tgt[k-1].nnz)
        #pragma acc enter data create(map_m_data_tgt[k-1].i_ptr[:ndim+1])
        #pragma acc enter data create(map_m_data_tgt[k-1].j_ptr[:nnz])
        #pragma acc enter data create(map_m_data_tgt[k-1].val_ptr[:nnz])

        ndim = mesh_src.map_negative2(k).get_ndim();
        nnz  = mesh_src.map_negative2(k).get_nnz();

        // Allocate the class members of map_negative1 matrices
        #pragma acc enter data create(map_mm_data_tgt[k-1].ndim)
        #pragma acc enter data create(map_mm_data_tgt[k-1].ncol)
        #pragma acc enter data create(map_mm_data_tgt[k-1].nnz)
        #pragma acc enter data create(map_mm_data_tgt[k-1].i_ptr[:ndim+1])
        #pragma acc enter data create(map_mm_data_tgt[k-1].j_ptr[:nnz])
        #pragma acc enter data create(map_mm_data_tgt[k-1].val_ptr[:nnz])
    }

    #pragma acc parallel default(none) \
                         present(size_RZ, size_phi) \
                         present(mesh_src) \
                         present(not_in_target_tgt) \
                         present(fll_positive1_tgt, fll_positive2_tgt) \
                         present(fll_negative1_tgt, fll_negative2_tgt) \
                         present(map_pp_data_tgt, map_p_data_tgt) \
                         present(map_mm_data_tgt, map_m_data_tgt)
    {
    // Copy the array members of par_con
    #pragma acc loop vector collapse(2)
    for (int32_t k = 1; k <= size_phi; k++)
    for (int32_t i = 1; i <= size_RZ; i++)
    {
        int32_t idx = (i - 1) + (k - 1) * size_RZ;

        not_in_target_tgt[idx] = mesh_src.not_in_target(i, k);
        fll_positive1_tgt[idx] = mesh_src.fll_positive1(i, k);
        fll_positive2_tgt[idx] = mesh_src.fll_positive2(i, k);
        fll_negative1_tgt[idx] = mesh_src.fll_negative1(i, k);
        fll_negative2_tgt[idx] = mesh_src.fll_negative2(i, k);
    }

    // Copy the class members of the map matrices
    #pragma acc loop independent
    for (int32_t k = 1; k <= size_phi; k++)
    {
        // Copy class members of the map_positive2 matrices
        int32_t ndim = mesh_src.map_positive2(k).get_ndim();
        int32_t ncol = mesh_src.map_positive2(k).get_ncol();
        int32_t nnz  = mesh_src.map_positive2(k).get_nnz();

        map_pp_data_tgt[k-1].ndim = ndim;
        map_pp_data_tgt[k-1].ncol = ncol;
        map_pp_data_tgt[k-1].nnz  = nnz;

        #pragma acc loop independent vector
        for (int32_t i = 1; i <= ndim + 1; i++)
        {
            map_pp_data_tgt[k-1].i_ptr[i-1] = mesh_src.map_positive2(k).i(i);
        }
        #pragma acc loop independent vector
        for (int32_t i = 1; i <= nnz; i++)
        {
            map_pp_data_tgt[k-1].j_ptr[i-1]   =
                mesh_src.map_positive2(k).j(i);
            map_pp_data_tgt[k-1].val_ptr[i-1] =
                mesh_src.map_positive2(k).val(i);
        }

        // Copy class members of the map_positive1 matrices
        ndim = mesh_src.map_positive1(k).get_ndim();
        ncol = mesh_src.map_positive1(k).get_ncol();
        nnz  = mesh_src.map_positive1(k).get_nnz();

        map_p_data_tgt[k-1].ndim = ndim;
        map_p_data_tgt[k-1].ncol = ncol;
        map_p_data_tgt[k-1].nnz  = nnz;

        #pragma acc loop independent vector
        for (int32_t i = 1; i <= ndim + 1; i++)
        {
            map_p_data_tgt[k-1].i_ptr[i-1] = mesh_src.map_positive1(k).i(i);
        }
        #pragma acc loop independent vector
        for (int32_t i = 1; i <= nnz; i++)
        {
            map_p_data_tgt[k-1].j_ptr[i-1]   =
                mesh_src.map_positive1(k).j(i);
            map_p_data_tgt[k-1].val_ptr[i-1] =
                mesh_src.map_positive1(k).val(i);
        }

        // Copy class members of the map_negative1 matrices
        ndim = mesh_src.map_negative1(k).get_ndim();
        ncol = mesh_src.map_negative1(k).get_ncol();
        nnz  = mesh_src.map_negative1(k).get_nnz();

        map_m_data_tgt[k-1].ndim = ndim;
        map_m_data_tgt[k-1].ncol = ncol;
        map_m_data_tgt[k-1].nnz  = nnz;

        #pragma acc loop independent vector
        for (int32_t i = 1; i <= ndim + 1; i++)
        {
            map_m_data_tgt[k-1].i_ptr[i-1] = mesh_src.map_negative1(k).i(i);
        }
        #pragma acc loop independent vector
        for (int32_t i = 1; i <= nnz; i++)
        {
            map_m_data_tgt[k-1].j_ptr[i-1]   =
                mesh_src.map_negative1(k).j(i);
            map_m_data_tgt[k-1].val_ptr[i-1] =
                mesh_src.map_negative1(k).val(i);
        }

        // Copy class members of the map_negative2 matrices
        ndim = mesh_src.map_negative2(k).get_ndim();
        ncol = mesh_src.map_negative2(k).get_ncol();
        nnz  = mesh_src.map_negative2(k).get_nnz();

        map_mm_data_tgt[k-1].ndim = ndim;
        map_mm_data_tgt[k-1].ncol = ncol;
        map_mm_data_tgt[k-1].nnz  = nnz;

        #pragma acc loop independent vector
        for (int32_t i = 1; i <= ndim + 1; i++)
        {
            map_mm_data_tgt[k-1].i_ptr[i-1] = mesh_src.map_negative2(k).i(i);
        }
        #pragma acc loop independent vector
        for (int32_t i = 1; i <= nnz; i++)
        {
            map_mm_data_tgt[k-1].j_ptr[i-1]   =
                mesh_src.map_negative2(k).j(i);
            map_mm_data_tgt[k-1].val_ptr[i-1] =
                mesh_src.map_negative2(k).val(i);
        }
    }
    }

    // Copy data from the device to the host and deallocate memory on the device
    #pragma acc exit data delete(size_RZ)
    #pragma acc exit data delete(size_phi)

    #pragma acc exit data copyout(fll_negative2_tgt[:size_mesh_3d])
    #pragma acc exit data copyout(fll_negative1_tgt[:size_mesh_3d])
    #pragma acc exit data copyout(fll_positive2_tgt[:size_mesh_3d])
    #pragma acc exit data copyout(fll_positive1_tgt[:size_mesh_3d])
    #pragma acc exit data copyout(not_in_target_tgt[:size_mesh_3d])

    for (int32_t k = 1; k <= size_phi; k++)
    {
        int32_t ndim = mesh_src.map_positive2(k).get_ndim();
        int32_t nnz  = mesh_src.map_positive2(k).get_nnz();

        // Allocate the class members of map_positive2 matrices
        #pragma acc exit data copyout(map_pp_data_tgt[k-1].ndim)
        #pragma acc exit data copyout(map_pp_data_tgt[k-1].ncol)
        #pragma acc exit data copyout(map_pp_data_tgt[k-1].nnz)
        #pragma acc exit data copyout(map_pp_data_tgt[k-1].i_ptr[:ndim+1])
        #pragma acc exit data copyout(map_pp_data_tgt[k-1].j_ptr[:nnz])
        #pragma acc exit data copyout(map_pp_data_tgt[k-1].val_ptr[:nnz])

        ndim = mesh_src.map_positive1(k).get_ndim();
        nnz  = mesh_src.map_positive1(k).get_nnz();

        // Allocate the class members of map_positive1 matrices
        #pragma acc exit data copyout(map_p_data_tgt[k-1].ndim)
        #pragma acc exit data copyout(map_p_data_tgt[k-1].ncol)
        #pragma acc exit data copyout(map_p_data_tgt[k-1].nnz)
        #pragma acc exit data copyout(map_p_data_tgt[k-1].i_ptr[:ndim+1])
        #pragma acc exit data copyout(map_p_data_tgt[k-1].j_ptr[:nnz])
        #pragma acc exit data copyout(map_p_data_tgt[k-1].val_ptr[:nnz])

        ndim = mesh_src.map_negative1(k).get_ndim();
        nnz  = mesh_src.map_negative1(k).get_nnz();

        // Allocate the class members of map_negative1 matrices
        #pragma acc exit data copyout(map_m_data_tgt[k-1].ndim)
        #pragma acc exit data copyout(map_m_data_tgt[k-1].ncol)
        #pragma acc exit data copyout(map_m_data_tgt[k-1].nnz)
        #pragma acc exit data copyout(map_m_data_tgt[k-1].i_ptr[:ndim+1])
        #pragma acc exit data copyout(map_m_data_tgt[k-1].j_ptr[:nnz])
        #pragma acc exit data copyout(map_m_data_tgt[k-1].val_ptr[:nnz])

        ndim = mesh_src.map_negative2(k).get_ndim();
        nnz  = mesh_src.map_negative2(k).get_nnz();

        // Allocate the class members of map_negative1 matrices
        #pragma acc exit data copyout(map_mm_data_tgt[k-1].ndim)
        #pragma acc exit data copyout(map_mm_data_tgt[k-1].ncol)
        #pragma acc exit data copyout(map_mm_data_tgt[k-1].nnz)
        #pragma acc exit data copyout(map_mm_data_tgt[k-1].i_ptr[:ndim+1])
        #pragma acc exit data copyout(map_mm_data_tgt[k-1].j_ptr[:nnz])
        #pragma acc exit data copyout(map_mm_data_tgt[k-1].val_ptr[:nnz])
    }

    // Deallocate the array of map matrices on the device
    #pragma acc exit data copyout(map_pp_data_tgt[:size_phi])
    #pragma acc exit data copyout(map_p_data_tgt[:size_phi])
    #pragma acc exit data copyout(map_m_data_tgt[:size_phi])
    #pragma acc exit data copyout(map_mm_data_tgt[:size_phi])

    return 0;
}

#endif

#ifdef ENABLE_OPENMPX

// Copy the array members related to the parallel connection (parcon) from
// the source mesh to target mesh struct via OpenMP offload on GPU
int32_t mesh_copy_parcon_ompx(const mesh_5d_t& mesh_src,
                              struct mesh_5d_data_t* mesh_data_tgt)
{
    // Assign the member pointers of the target par_con
    int32_t* not_in_target_tgt = mesh_data_tgt->not_in_target_ptr;
    real_t* fll_positive1_tgt = mesh_data_tgt->fll_positive1_ptr;
    real_t* fll_positive2_tgt = mesh_data_tgt->fll_positive2_ptr;
    real_t* fll_negative1_tgt = mesh_data_tgt->fll_negative1_ptr;
    real_t* fll_negative2_tgt = mesh_data_tgt->fll_negative2_ptr;
    csrmat_genex_data_t* map_pp_data_tgt =
        mesh_data_tgt->map_positive2_data_ptr;
    csrmat_genex_data_t* map_p_data_tgt =
        mesh_data_tgt->map_positive1_data_ptr;
    csrmat_genex_data_t* map_m_data_tgt =
        mesh_data_tgt->map_negative1_data_ptr;
    csrmat_genex_data_t* map_mm_data_tgt =
        mesh_data_tgt->map_negative2_data_ptr;

    // Assign the shape of mesh grid
    int32_t size_RZ      = mesh_src.get_size_RZ();
    int32_t size_phi     = mesh_src.get_size_phi();
    int32_t size_mesh_3d = size_RZ * size_phi;

    // Allocate the array member of par_con on the device
    #pragma omp target enter data map(alloc: not_in_target_tgt[:size_mesh_3d])
    #pragma omp target enter data map(alloc: fll_positive1_tgt[:size_mesh_3d])
    #pragma omp target enter data map(alloc: fll_positive2_tgt[:size_mesh_3d])
    #pragma omp target enter data map(alloc: fll_negative1_tgt[:size_mesh_3d])
    #pragma omp target enter data map(alloc: fll_negative2_tgt[:size_mesh_3d])

    // Allocate the array of map matrices on the device
    #pragma omp target enter data map(alloc: map_pp_data_tgt[:size_phi])
    #pragma omp target enter data map(alloc: map_p_data_tgt[:size_phi])
    #pragma omp target enter data map(alloc: map_m_data_tgt[:size_phi])
    #pragma omp target enter data map(alloc: map_mm_data_tgt[:size_phi])

    for (int32_t k = 1; k <= size_phi; k++)
    {
        int32_t ndim = mesh_src.map_positive2(k).get_ndim();
        int32_t nnz  = mesh_src.map_positive2(k).get_nnz();

        // Allocate the class members of map_positive2 matrices
        #pragma omp target enter data map(alloc: map_pp_data_tgt[k-1].ndim)
        #pragma omp target enter data map(alloc: map_pp_data_tgt[k-1].ncol)
        #pragma omp target enter data map(alloc: map_pp_data_tgt[k-1].nnz)
        #pragma omp target enter data \
            map(alloc: map_pp_data_tgt[k-1].i_ptr[:ndim+1])
        #pragma omp target enter data \
            map(alloc: map_pp_data_tgt[k-1].j_ptr[:nnz])
        #pragma omp target enter data \
            map(alloc: map_pp_data_tgt[k-1].val_ptr[:nnz])

        ndim = mesh_src.map_positive1(k).get_ndim();
        nnz  = mesh_src.map_positive1(k).get_nnz();

        // Allocate the class members of map_positive1 matrices
        #pragma omp target enter data map(alloc: map_p_data_tgt[k-1].ndim)
        #pragma omp target enter data map(alloc: map_p_data_tgt[k-1].ncol)
        #pragma omp target enter data map(alloc: map_p_data_tgt[k-1].nnz)
        #pragma omp target enter data \
            map(alloc: map_p_data_tgt[k-1].i_ptr[:ndim+1])
        #pragma omp target enter data \
            map(alloc: map_p_data_tgt[k-1].j_ptr[:nnz])
        #pragma omp target enter data \
            map(alloc: map_p_data_tgt[k-1].val_ptr[:nnz])

        ndim = mesh_src.map_negative1(k).get_ndim();
        nnz  = mesh_src.map_negative1(k).get_nnz();

        // Allocate the class members of map_negative1 matrices
        #pragma omp target enter data map(alloc: map_m_data_tgt[k-1].ndim)
        #pragma omp target enter data map(alloc: map_m_data_tgt[k-1].ncol)
        #pragma omp target enter data map(alloc: map_m_data_tgt[k-1].nnz)
        #pragma omp target enter data \
            map(alloc: map_m_data_tgt[k-1].i_ptr[:ndim+1])
        #pragma omp target enter data \
            map(alloc: map_m_data_tgt[k-1].j_ptr[:nnz])
        #pragma omp target enter data \
            map(alloc: map_m_data_tgt[k-1].val_ptr[:nnz])

        ndim = mesh_src.map_negative2(k).get_ndim();
        nnz  = mesh_src.map_negative2(k).get_nnz();

        // Allocate the class members of map_negative1 matrices
        #pragma omp target enter data map(alloc: map_mm_data_tgt[k-1].ndim)
        #pragma omp target enter data map(alloc: map_mm_data_tgt[k-1].ncol)
        #pragma omp target enter data map(alloc: map_mm_data_tgt[k-1].nnz)
        #pragma omp target enter data \
            map(alloc: map_mm_data_tgt[k-1].i_ptr[:ndim+1])
        #pragma omp target enter data \
            map(alloc: map_mm_data_tgt[k-1].j_ptr[:nnz])
        #pragma omp target enter data \
            map(alloc: map_mm_data_tgt[k-1].val_ptr[:nnz])
    }

    // Copy the array members of par_con
    #pragma omp target teams distribute parallel for simd collapse(2) \
        default(none) defaultmap(none) \
        shared(mesh_src, not_in_target_tgt, fll_positive1_tgt, \
               fll_positive2_tgt, fll_negative1_tgt, fll_negative2_tgt)
    for (int32_t k = 1; k <= mesh_src.get_size_phi(); k++)
    for (int32_t i = 1; i <= mesh_src.get_size_RZ(); i++)
    {
        int32_t idx = (i - 1) + (k - 1) * mesh_src.get_size_RZ();

        not_in_target_tgt[idx] = mesh_src.not_in_target(i, k);
        fll_positive1_tgt[idx] = mesh_src.fll_positive1(i, k);
        fll_positive2_tgt[idx] = mesh_src.fll_positive2(i, k);
        fll_negative1_tgt[idx] = mesh_src.fll_negative1(i, k);
        fll_negative2_tgt[idx] = mesh_src.fll_negative2(i, k);
    }

    // Copy class members of the map_positive2 matrices
    #pragma omp target teams distribute parallel for simd \
        default(none) defaultmap(none) \
        shared(mesh_src, map_pp_data_tgt)
    for (int32_t k = 1; k <= mesh_src.get_size_phi(); k++)
    {
        int32_t ndim = mesh_src.map_positive2(k).get_ndim();
        int32_t ncol = mesh_src.map_positive2(k).get_ncol();
        int32_t nnz  = mesh_src.map_positive2(k).get_nnz();

        map_pp_data_tgt[k-1].ndim = ndim;
        map_pp_data_tgt[k-1].ncol = ncol;
        map_pp_data_tgt[k-1].nnz  = nnz;

        for (int32_t i = 1; i <= ndim + 1; i++)
        {
            map_pp_data_tgt[k-1].i_ptr[i-1] \
                = mesh_src.map_positive2(k).i(i);
        }

        for (int32_t i = 1; i <= nnz; i++)
        {
            map_pp_data_tgt[k-1].j_ptr[i-1]   =
                mesh_src.map_positive2(k).j(i);
            map_pp_data_tgt[k-1].val_ptr[i-1] =
                mesh_src.map_positive2(k).val(i);
        }
    }

    // Copy class members of the map_positive1 matrices
    #pragma omp target teams distribute parallel for simd \
        default(none) defaultmap(none) \
        shared(mesh_src, map_p_data_tgt)
    for (int32_t k = 1; k <= mesh_src.get_size_phi(); k++)
    {
        int32_t ndim = mesh_src.map_positive1(k).get_ndim();
        int32_t ncol = mesh_src.map_positive1(k).get_ncol();
        int32_t nnz  = mesh_src.map_positive1(k).get_nnz();

        map_p_data_tgt[k-1].ndim = ndim;
        map_p_data_tgt[k-1].ncol = ncol;
        map_p_data_tgt[k-1].nnz  = nnz;

        for (int32_t i = 1; i <= ndim + 1; i++)
        {
            map_p_data_tgt[k-1].i_ptr[i-1] \
                = mesh_src.map_positive1(k).i(i);
        }

        for (int32_t i = 1; i <= nnz; i++)
        {
            map_p_data_tgt[k-1].j_ptr[i-1]   =
                mesh_src.map_positive1(k).j(i);
            map_p_data_tgt[k-1].val_ptr[i-1] =
                mesh_src.map_positive1(k).val(i);
        }
    }

    // Copy class members of the map_negative1 matrices
    #pragma omp target teams distribute parallel for simd \
        default(none) defaultmap(none) \
        shared(mesh_src, map_m_data_tgt)
    for (int32_t k = 1; k <= mesh_src.get_size_phi(); k++)
    {
        int32_t ndim = mesh_src.map_negative1(k).get_ndim();
        int32_t ncol = mesh_src.map_negative1(k).get_ncol();
        int32_t nnz  = mesh_src.map_negative1(k).get_nnz();

        map_m_data_tgt[k-1].ndim = ndim;
        map_m_data_tgt[k-1].ncol = ncol;
        map_m_data_tgt[k-1].nnz  = nnz;

        for (int32_t i = 1; i <= ndim + 1; i++)
        {
            map_m_data_tgt[k-1].i_ptr[i-1] \
                = mesh_src.map_negative1(k).i(i);
        }

        for (int32_t i = 1; i <= nnz; i++)
        {
            map_m_data_tgt[k-1].j_ptr[i-1]   =
                mesh_src.map_negative1(k).j(i);
            map_m_data_tgt[k-1].val_ptr[i-1] =
                mesh_src.map_negative1(k).val(i);
        }
    }

    // Copy class members of the map_negative2 matrices
    #pragma omp target teams distribute parallel for simd \
        default(none) defaultmap(none) \
        shared(mesh_src, map_mm_data_tgt)
    for (int32_t k = 1; k <= mesh_src.get_size_phi(); k++)
    {
        int32_t ndim = mesh_src.map_negative2(k).get_ndim();
        int32_t ncol = mesh_src.map_negative2(k).get_ncol();
        int32_t nnz  = mesh_src.map_negative2(k).get_nnz();

        map_mm_data_tgt[k-1].ndim = ndim;
        map_mm_data_tgt[k-1].ncol = ncol;
        map_mm_data_tgt[k-1].nnz  = nnz;

        for (int32_t i = 1; i <= ndim + 1; i++)
        {
            map_mm_data_tgt[k-1].i_ptr[i-1] \
                = mesh_src.map_negative2(k).i(i);
        }

        for (int32_t i = 1; i <= nnz; i++)
        {
            map_mm_data_tgt[k-1].j_ptr[i-1]   =
                mesh_src.map_negative2(k).j(i);
            map_mm_data_tgt[k-1].val_ptr[i-1] =
                mesh_src.map_negative2(k).val(i);
        }
    }

    // Copy data from the device to the host and deallocate memory on the device
    #pragma omp target exit data map(from: fll_negative2_tgt[:size_mesh_3d])
    #pragma omp target exit data map(from: fll_negative1_tgt[:size_mesh_3d])
    #pragma omp target exit data map(from: fll_positive2_tgt[:size_mesh_3d])
    #pragma omp target exit data map(from: fll_positive1_tgt[:size_mesh_3d])
    #pragma omp target exit data map(from: not_in_target_tgt[:size_mesh_3d])

    for (int32_t k = 1; k <= size_phi; k++)
    {
        int32_t ndim = mesh_src.map_positive2(k).get_ndim();
        int32_t nnz  = mesh_src.map_positive2(k).get_nnz();

        // Allocate the class members of map_positive2 matrices
        #pragma omp target exit data map(from: map_pp_data_tgt[k-1].ndim)
        #pragma omp target exit data map(from: map_pp_data_tgt[k-1].ncol)
        #pragma omp target exit data map(from: map_pp_data_tgt[k-1].nnz)
        #pragma omp target exit data \
            map(from: map_pp_data_tgt[k-1].i_ptr[:ndim+1])
        #pragma omp target exit data \
            map(from: map_pp_data_tgt[k-1].j_ptr[:nnz])
        #pragma omp target exit data \
            map(from: map_pp_data_tgt[k-1].val_ptr[:nnz])

        ndim = mesh_src.map_positive1(k).get_ndim();
        nnz  = mesh_src.map_positive1(k).get_nnz();

        // Allocate the class members of map_positive1 matrices
        #pragma omp target exit data map(from: map_p_data_tgt[k-1].ndim)
        #pragma omp target exit data map(from: map_p_data_tgt[k-1].ncol)
        #pragma omp target exit data map(from: map_p_data_tgt[k-1].nnz)
        #pragma omp target exit data \
            map(from: map_p_data_tgt[k-1].i_ptr[:ndim+1])
        #pragma omp target exit data \
            map(from: map_p_data_tgt[k-1].j_ptr[:nnz])
        #pragma omp target exit data \
            map(from: map_p_data_tgt[k-1].val_ptr[:nnz])

        ndim = mesh_src.map_negative1(k).get_ndim();
        nnz  = mesh_src.map_negative1(k).get_nnz();

        // Allocate the class members of map_negative1 matrices
        #pragma omp target exit data map(from: map_m_data_tgt[k-1].ndim)
        #pragma omp target exit data map(from: map_m_data_tgt[k-1].ncol)
        #pragma omp target exit data map(from: map_m_data_tgt[k-1].nnz)
        #pragma omp target exit data \
            map(from: map_m_data_tgt[k-1].i_ptr[:ndim+1])
        #pragma omp target exit data \
            map(from: map_m_data_tgt[k-1].j_ptr[:nnz])
        #pragma omp target exit data \
            map(from: map_m_data_tgt[k-1].val_ptr[:nnz])

        ndim = mesh_src.map_negative2(k).get_ndim();
        nnz  = mesh_src.map_negative2(k).get_nnz();

        // Allocate the class members of map_negative1 matrices
        #pragma omp target exit data map(from: map_mm_data_tgt[k-1].ndim)
        #pragma omp target exit data map(from: map_mm_data_tgt[k-1].ncol)
        #pragma omp target exit data map(from: map_mm_data_tgt[k-1].nnz)
        #pragma omp target exit data \
            map(from: map_mm_data_tgt[k-1].i_ptr[:ndim+1])
        #pragma omp target exit data \
            map(from: map_mm_data_tgt[k-1].j_ptr[:nnz])
        #pragma omp target exit data \
            map(from: map_mm_data_tgt[k-1].val_ptr[:nnz])
    }

    // Deallocate the array of map matrices on the device
    #pragma omp target exit data map(from: map_pp_data_tgt[:size_phi])
    #pragma omp target exit data map(from: map_p_data_tgt[:size_phi])
    #pragma omp target exit data map(from: map_m_data_tgt[:size_phi])
    #pragma omp target exit data map(from: map_mm_data_tgt[:size_phi])

    return 0;
}

#endif

#ifdef __cplusplus
extern "C" {
#endif

int32_t cbind_mesh_copy_parcon(const mesh_5d_t** mesh_src_cxx_pptr,
                               struct mesh_5d_data_t* mesh_data_tgt)
{
    // Assign the source mesh_5d_t class instance
    const mesh_5d_t& mesh_src = *(*mesh_src_cxx_pptr);

    // Return 0 for success and 1 for error
    switch (params_gpu_offload::get_backend())
    {
        case params_gpu_offload::backend_t::CPU:
            return mesh_copy_parcon_cxx(mesh_src, mesh_data_tgt);
#ifdef ENABLE_OPENACC
        case params_gpu_offload::backend_t::ACC:
            return mesh_copy_parcon_acc(mesh_src, mesh_data_tgt);
#endif
#ifdef ENABLE_OPENMPX
        case params_gpu_offload::backend_t::OMPX:
            return mesh_copy_parcon_ompx(mesh_src, mesh_data_tgt);
#endif
        default:
            return 1;
    }
}

#ifdef __cplusplus
}
#endif

#endif
