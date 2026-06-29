#ifndef MESH_5D_OMPX_HXX
#define MESH_5D_OMPX_HXX

#include "genex_cxx_env.hxx"
#include "mesh_5d.hxx"
#include "csrmat_genex_ompx.hxx"
#include <omp.h>

// C++ class which corresponds to the Fortran class mesh_5d_t
// with OpenMP offload
class mesh_5d_ompx_t: public mesh_5d_t
{
public:
    // Constructor of the OpenMP offload child class
    mesh_5d_ompx_t(struct mesh_5d_data_t* mesh_data)
    : mesh_5d_t{mesh_data}
    {
        // Allocate arrays of the map matrix (csrmat) class instances
        this->map_positive1_ptr = new csrmat_genex_ompx_t[mesh_data->size_phi];
        this->map_negative1_ptr = new csrmat_genex_ompx_t[mesh_data->size_phi];
        this->map_positive2_ptr = new csrmat_genex_ompx_t[mesh_data->size_phi];
        this->map_negative2_ptr = new csrmat_genex_ompx_t[mesh_data->size_phi];

        // Construct the map matrix (csrmat) class instances
        for (int32_t k = 0; k < this->size_phi; k++)
        {
            this->map_positive1_ptr[k] = csrmat_genex_ompx_t(
                &(mesh_data->map_positive1_data_ptr[k]));
            this->map_negative1_ptr[k] = csrmat_genex_ompx_t(
                &(mesh_data->map_negative1_data_ptr[k]));
            this->map_positive2_ptr[k] = csrmat_genex_ompx_t(
                &(mesh_data->map_positive2_data_ptr[k]));
            this->map_negative2_ptr[k] = csrmat_genex_ompx_t(
                &(mesh_data->map_negative2_data_ptr[k]));
        }

        int32_t size_mesh      = this->size_RZ * this->size_phi;
        int32_t size_mesh_part = this->size_RZ
                               * (this->ub_phi - this->lb_phi + 1);
        int32_t size_neigh_arr = this->size_RZ_stencil * this->size_RZ_stencil *
                                 this->size_RZ * this->size_phi;
        // Allocate device pointer, map the host pointer to it and copy
        // the values
        #pragma omp target enter data \
            map(to: this[:1], \
                    this->size_RZ, \
                    this->size_phi, \
                    this->size_vp, \
                    this->size_mu, \
                    this->size_sp, \
                    this->order_RZ_stencil, \
                    this->size_RZ_stencil, \
                    this->lb_phi, \
                    this->ub_phi, \
                    this->delta_RZ, \
                    this->delta_phi, \
                    this->delta_vp, \
                    this->delta_sqrt_mu, \
                    this->neighbors_ptr[:size_neigh_arr], \
                    this->buf_zone_ptr[:size_mesh_part], \
                    this->RZ_indices_ptr[:size_mesh_part], \
                    this->not_filler_ptr[:size_mesh_part], \
                    this->is_compute_ptr[:size_mesh_part], \
                    this->vp_grid_ptr[:this->size_vp], \
                    this->mu_grid_ptr[:this->size_mu], \
                    this->sqrt_mu_grid_ptr[:this->size_mu], \
                    this->vp_weights_ptr[:this->size_vp], \
                    this->mu_weights_ptr[:this->size_mu], \
                    this->jacobian_buffer_ptr[:size_mesh_part], \
                    this->absB_buffer_ptr[:size_mesh], \
                    this->normb_R_buffer_ptr[:size_mesh], \
                    this->normb_Z_buffer_ptr[:size_mesh], \
                    this->curl_normb_y_ptr[:size_mesh], \
                    this->dgyxdy_over_g_ptr[:size_mesh], \
                    this->dgyzdy_over_g_ptr[:size_mesh], \
                    this->dgyxdz_over_g_ptr[:size_mesh], \
                    this->dgyzdx_over_g_ptr[:size_mesh], \
                    this->inv_g_ptr[:size_mesh], \
                    this->dabsBdx_ptr[:size_mesh], \
                    this->dabsBdz_ptr[:size_mesh], \
                    this->dabsBdy_ptr[:size_mesh], \
                    this->not_in_target_ptr[:size_mesh], \
                    this->fll_positive1_ptr[:size_mesh], \
                    this->fll_positive2_ptr[:size_mesh], \
                    this->fll_negative1_ptr[:size_mesh], \
                    this->fll_negative2_ptr[:size_mesh], \
                    this->map_positive1_ptr[:this->size_phi], \
                    this->map_negative1_ptr[:this->size_phi], \
                    this->map_positive2_ptr[:this->size_phi], \
                    this->map_negative2_ptr[:this->size_phi])

        // Allocate and deep copy the members of map matrix to the device
        bool err = false;
        for (int32_t k = 0; k < this->size_phi; k++)
        {
            err = err || this->map_positive1_ptr[k].initialize_device();
            err = err || this->map_negative1_ptr[k].initialize_device();
            err = err || this->map_positive2_ptr[k].initialize_device();
            err = err || this->map_negative2_ptr[k].initialize_device();
        }
        mesh_5d::is_erroneous = mesh_5d::is_erroneous || err;

        this->dmem_debug(dmd::mode_t::ALLOC);
    }

    // Destructor of the OpenMP offload child class
    ~mesh_5d_ompx_t() override
    {
        // Deallocate the members of map matrix from the device
        bool err = false;
        for (int32_t k = 0; k < this->size_phi; k++)
        {
            err = err || this->map_positive1_ptr[k].finalize_device();
            err = err || this->map_negative1_ptr[k].finalize_device();
            err = err || this->map_positive2_ptr[k].finalize_device();
            err = err || this->map_negative2_ptr[k].finalize_device();
        }
        mesh_5d::is_erroneous = mesh_5d::is_erroneous || err;

        int32_t size_mesh      = this->size_RZ * this->size_phi;
        int32_t size_mesh_part = this->size_RZ
                               * (this->ub_phi - this->lb_phi + 1);
        int32_t size_neigh_arr = this->size_RZ_stencil * this->size_RZ_stencil *
                                 this->size_RZ * this->size_phi;
        // Deallocate the device pointers of the class members from the device
        #pragma omp target exit data \
            map(delete: this->map_negative2_ptr[:this->size_phi], \
                        this->map_positive2_ptr[:this->size_phi], \
                        this->map_positive1_ptr[:this->size_phi], \
                        this->map_negative1_ptr[:this->size_phi], \
                        this->fll_negative2_ptr[:size_mesh], \
                        this->fll_negative1_ptr[:size_mesh], \
                        this->fll_positive2_ptr[:size_mesh], \
                        this->fll_positive1_ptr[:size_mesh], \
                        this->not_in_target_ptr[:size_mesh], \
                        this->dabsBdy_ptr[:size_mesh], \
                        this->dabsBdz_ptr[:size_mesh], \
                        this->dabsBdx_ptr[:size_mesh], \
                        this->inv_g_ptr[:size_mesh], \
                        this->dgyzdx_over_g_ptr[:size_mesh], \
                        this->dgyxdz_over_g_ptr[:size_mesh], \
                        this->dgyzdy_over_g_ptr[:size_mesh], \
                        this->dgyxdy_over_g_ptr[:size_mesh], \
                        this->curl_normb_y_ptr[:size_mesh], \
                        this->normb_Z_buffer_ptr[:size_mesh], \
                        this->normb_R_buffer_ptr[:size_mesh], \
                        this->absB_buffer_ptr[:size_mesh], \
                        this->jacobian_buffer_ptr[:size_mesh_part], \
                        this->mu_weights_ptr[:this->size_mu], \
                        this->vp_weights_ptr[:this->size_vp], \
                        this->sqrt_mu_grid_ptr[:this->size_mu], \
                        this->mu_grid_ptr[:this->size_mu], \
                        this->vp_grid_ptr[:this->size_vp], \
                        this->is_compute_ptr[:size_mesh_part], \
                        this->not_filler_ptr[:size_mesh_part], \
                        this->RZ_indices_ptr[:size_mesh_part], \
                        this->buf_zone_ptr[:size_mesh_part], \
                        this->neighbors_ptr[:size_neigh_arr], \
                        this->delta_sqrt_mu, \
                        this->delta_vp, \
                        this->delta_phi, \
                        this->delta_RZ, \
                        this->ub_phi, \
                        this->lb_phi, \
                        this->size_RZ_stencil, \
                        this->order_RZ_stencil, \
                        this->size_sp, \
                        this->size_mu, \
                        this->size_vp, \
                        this->size_phi, \
                        this->size_RZ, \
                        this[:1])

        this->dmem_debug(dmd::mode_t::DEALLOC);
    }

    // Copy constructor is disabled
    mesh_5d_ompx_t(const mesh_5d_ompx_t&) = delete;

    // Copy-assignment operator is disabled
    mesh_5d_ompx_t& operator=(const mesh_5d_ompx_t&) = delete;
};

#endif
