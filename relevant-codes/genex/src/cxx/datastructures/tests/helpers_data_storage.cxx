#ifndef HELPERS_DATA_STORAGE_CXX
#define HELPERS_DATA_STORAGE_CXX

#include "genex_cxx_env.hxx"
#include "params_gpu_offload.hxx"
#include "data_storage.hxx"
#include "data_array.hxx"

// Copy the class members from the source ds class to the target ds struct
// via OpenMP on CPU
template<typename T, size_t DIM>
int32_t data_storage_copy_cxx(const data_storage_gpu_t<T, DIM>& ds_src,
                              struct data_storage_data_t* ds_data_tgt,
                              struct data_array_data_t* da_data_tgt)
{
    // Assign array member pointer to the target ds
    int32_t* num_elements_tgt      = ds_data_tgt->number_of_elements_ptr;
    int32_t* num_ghost_cells_tgt   = ds_data_tgt->number_of_ghost_cells_ptr;
    int32_t* num_data_cells_tgt    = ds_data_tgt->number_of_data_cells_ptr;
    int32_t* num_mail_partners_tgt = ds_data_tgt->number_of_mail_partners_ptr;
    int32_t* dim_permut_tgt        = ds_data_tgt->dim_permut_ptr;

    // Assign array member pointer to the target da
    int32_t* shape_tgt          = da_data_tgt->array_shape_ptr;
    int32_t* shape_stripped_tgt = da_data_tgt->array_shape_stripped_ptr;
    int32_t* lb_tgt             = da_data_tgt->array_lb_ptr;
    int32_t* ub_tgt             = da_data_tgt->array_ub_ptr;
    int32_t* lb_stripped_tgt    = da_data_tgt->array_lb_stripped_ptr;
    int32_t* ub_stripped_tgt    = da_data_tgt->array_ub_stripped_ptr;

    // Copy the class members from the source to the target da via OpenMP
    #pragma omp parallel default(none) \
                         shared(ds_src, ds_data_tgt, da_data_tgt) \
                         shared(num_elements_tgt) \
                         shared(num_ghost_cells_tgt, num_data_cells_tgt) \
                         shared(num_mail_partners_tgt, dim_permut_tgt) \
                         shared(shape_tgt, shape_stripped_tgt, lb_tgt, ub_tgt) \
                         shared(lb_stripped_tgt, ub_stripped_tgt)
    {
        ds_data_tgt->array_dim  = ds_src.get_dimension();
        ds_data_tgt->n_ex_dims  = ds_src.get_num_ex_dims();
        da_data_tgt->array_size = ds_src.get_size();

        #pragma omp for simd schedule(static) nowait
        for (int32_t i = 1; i <= ds_src.get_dimension(); i++)
        {
            num_elements_tgt[i-1]      = ds_src.get_num_elements(i);
            num_ghost_cells_tgt[i-1]   = ds_src.get_num_ghost_cells(i);
            num_data_cells_tgt[i-1]    = ds_src.get_num_data_cells(i);
            num_mail_partners_tgt[i-1] = ds_src.get_num_mail_partners(i);
            dim_permut_tgt[i-1]        = ds_src.get_dim_permut(i);

            shape_tgt[i-1]          = ds_src.get_shape(i);
            shape_stripped_tgt[i-1] = ds_src.get_shape_stripped(i);
            lb_tgt[i-1]             = ds_src.get_lbound(i);
            ub_tgt[i-1]             = ds_src.get_ubound(i);
            lb_stripped_tgt[i-1]    = ds_src.get_lbound_stripped(i);
            ub_stripped_tgt[i-1]    = ds_src.get_ubound_stripped(i);
        }
    }

    // Return 0 for success
    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif

// Interoperable routine to copy the class members from the source ds class
// to the target ds struct for 2D case
int32_t cbind_data_storage_2d_copy(
    const data_storage_gpu_t<real_t, 2>** ds_src_cxx_pptr,
    struct data_storage_data_t* ds_data_tgt,
    struct data_array_data_t* da_data_tgt)
{
    // Assign the C++ class instances
    const data_storage_gpu_t<real_t, 2>& ds_src = *(*ds_src_cxx_pptr);

    // Return 0 for success and 1 for error
    return data_storage_copy_cxx(ds_src, ds_data_tgt, da_data_tgt);
}

// Interoperable routine to copy the class members from the source ds class
// to the target ds struct for 4D case
int32_t cbind_data_storage_4d_copy(
    const data_storage_gpu_t<real_t, 4>** ds_src_cxx_pptr,
    struct data_storage_data_t* ds_data_tgt,
    struct data_array_data_t* da_data_tgt)
{
    // Assign the C++ class instances
    const data_storage_gpu_t<real_t, 4>& ds_src = *(*ds_src_cxx_pptr);

    // Return 0 for success and 1 for error
    return data_storage_copy_cxx(ds_src, ds_data_tgt, da_data_tgt);
}

// Interoperable routine to copy the class members from the source ds class
// to the target ds struct for 5D case
int32_t cbind_data_storage_5d_copy(
    const data_storage_gpu_t<real_t, 5>** ds_src_cxx_pptr,
    struct data_storage_data_t* ds_data_tgt,
    struct data_array_data_t* da_data_tgt)
{
    // Assign the C++ class instances
    const data_storage_gpu_t<real_t, 5>& ds_src = *(*ds_src_cxx_pptr);

    // Return 0 for success and 1 for error
    return data_storage_copy_cxx(ds_src, ds_data_tgt, da_data_tgt);
}

#ifdef __cplusplus
}
#endif

#endif
