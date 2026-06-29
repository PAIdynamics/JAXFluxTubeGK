#ifndef MESH_5D_HXX
#define MESH_5D_HXX

#include "genex_cxx_env.hxx"
#include "device_memory_debugger.hxx"
#include "csrmat_genex.hxx"
#include "params_gpu_offload.hxx"

// Alias for external namespaces
namespace dmd   = device_memory_debugger;
namespace pgpus = params_gpu_offload;

namespace mesh_5d
{
    // Error flag for mesh_5d
    inline bool is_erroneous = false;

    // NOTE: Filler points are used for non-axisymmetric grids that may have
    //       different shapes and sizes to adjust the arrays to be of same size
    // NOTE: Synchronize with src/mesh/mesh_5d_m.f90
    inline constexpr real_t FILLER_VALUE_REAL = -2.0023193043625635;
    inline constexpr int32_t FILLER_VALUE_INT = -2002319304;

    // Getter for the type of phi grid
    const std::string& get_grid_type_phi();
    // Getter for the type of vp grid
    const std::string& get_grid_type_vp();
    // Getter for the type of mu grid
    const std::string& get_grid_type_mu();
    // Getter for the type of phi quadrature
    const std::string& get_quad_type_phi();
    // Getter for the type of vp quadrature
    const std::string& get_quad_type_vp();
    // Getter for the type of mu quadrature
    const std::string& get_quad_type_mu();

    // Enumerator defining the different numerical buffer zone types.
    enum buf_zone
    {
        NOT_BUF_ZONE = 0,
        BUF_ZONE_BOUNDARY_IN = 1,
        BUF_ZONE_BOUNDARY_OUT = 2,
        BUF_ZONE_AXIS = 3,
        BUF_ZONE_PARCON = 4
    };
}

#ifdef __cplusplus
extern "C" {
#endif

struct mesh_5d_data_t
{
    int32_t size_RZ;
    int32_t size_phi;
    int32_t size_vp;
    int32_t size_mu;
    int32_t size_sp;
    int32_t order_RZ_stencil;
    int32_t lb_phi;
    int32_t ub_phi;

    real_t delta_RZ;
    real_t delta_phi;
    real_t delta_vp;
    real_t delta_sqrt_mu;

    int32_t* neighbors_ptr;
    int32_t* buf_zone_ptr;
    int32_t* RZ_indices_ptr;
    real_t* not_filler_ptr;
    real_t* is_compute_ptr;
    real_t* vp_grid_ptr;
    real_t* mu_grid_ptr;
    real_t* sqrt_mu_grid_ptr;
    real_t* vp_weights_ptr;
    real_t* mu_weights_ptr;
    real_t* jacobian_buffer_ptr;

    real_t* absB_buffer_ptr;
    real_t* normb_R_buffer_ptr;
    real_t* normb_Z_buffer_ptr;
    real_t* curl_normb_y_ptr;
    real_t* dgyxdy_over_g_ptr;
    real_t* dgyzdy_over_g_ptr;
    real_t* dgyxdz_over_g_ptr;
    real_t* dgyzdx_over_g_ptr;
    real_t* inv_g_ptr;
    real_t* dabsBdx_ptr;
    real_t* dabsBdz_ptr;
    real_t* dabsBdy_ptr;

    struct csrmat_genex_data_t* map_positive1_data_ptr;
    struct csrmat_genex_data_t* map_negative1_data_ptr;
    struct csrmat_genex_data_t* map_positive2_data_ptr;
    struct csrmat_genex_data_t* map_negative2_data_ptr;
    real_t* fll_positive1_ptr;
    real_t* fll_positive2_ptr;
    real_t* fll_negative1_ptr;
    real_t* fll_negative2_ptr;
    int32_t* not_in_target_ptr;
};

#ifdef __cplusplus
}
#endif

// C++ class which corresponds to the Fortran class mesh_5d_t
class mesh_5d_t
{
protected:
    // Maximum number of points in RZ direction including ghost & filler points
    int32_t size_RZ;
    // Number of points in phi direction
    int32_t size_phi;
    // Number of points in vp direction
    int32_t size_vp;
    // Number of points in mu direction
    int32_t size_mu;
    // Number of points in sp direction
    int32_t size_sp;
    // Number of neighboring points in each direction (left, right, bottom,
    // top) or the order of RZ stencil
    int32_t order_RZ_stencil;
    // Total number of neighboring points on each axes or RZ stencil size/width
    int32_t size_RZ_stencil;
    // Lower bound on the index of phi planes
    int32_t lb_phi;
    // Upper bound on the index of phi planes
    int32_t ub_phi;

    // Grid spacing of the RZ grid (same for every plane)
    real_t delta_RZ;
    // Grid spacing of the toroidal angle grid
    real_t delta_phi;
    // Grid spacing of the parallel velocity grid
    real_t delta_vp;
    // Grid spacing of the magnetic moment grid at the first
    // point of the grid. If the mu grid is quadratic, this spacing
    // is the same for all points.
    real_t delta_sqrt_mu;

    // Pointer to the indices of the neighbor points
    int32_t* neighbors_ptr;
    // Pointer to an array specifying which buffer zone a given point is in,
    // identified by the buffer zone type enumerator
    int32_t* buf_zone_ptr;
    // Pointer to mask indicating whether a buffer point is a filler point, for
    // meshes which have fewer points than the size of the array. The value is 0
    // for filler points, otherwise the value is 1.
    int32_t* RZ_indices_ptr;
    // Pointer to the mesh grid indices in the RZ dimension
    real_t* not_filler_ptr;
    // Pointer to the array specifying whether a given point is in the
    // computation domain
    real_t* is_compute_ptr;
    // Pointer to parallel velocity grid
    real_t* vp_grid_ptr;
    // Pointer to parallel mu grid
    real_t* mu_grid_ptr;
    // Pointer to sqrt(mu) grid
    real_t* sqrt_mu_grid_ptr;
    // Pointer to the integration weights of the vp grid
    real_t* vp_weights_ptr;
    // Pointer to the integration weights of the mu grid
    real_t* mu_weights_ptr;
    // Pointer for the jacobian of the coordinate transform from lab coordinates
    // into the coordinate system of the mesh. The coordinate system is
    // cartesian or cylindrical depending on the type of equilibrium.
    real_t* jacobian_buffer_ptr;

    // Class members related to magnetic field (magfield)

    // Pointer to B field on the 3D mesh
    real_t* absB_buffer_ptr;
    // Pointer to R component of the normalized magnetic field
    real_t* normb_R_buffer_ptr;
    // Pointer to Z component of the normalized magnetic field
    real_t* normb_Z_buffer_ptr;
    // Pointer to y component of the curl of the normalized magnetic field
    real_t* curl_normb_y_ptr;
    // Pointer to the derivative of b_R along the magnetic field on the 3D mesh
    real_t* dgyxdy_over_g_ptr;
    // Pointer to the derivative of b_Z along the magnetic field on the 3D mesh
    real_t* dgyzdy_over_g_ptr;
    // Pointer the derivative of b_R in Z direction on the 3D mesh
    real_t* dgyxdz_over_g_ptr;
    // Pointer to the derivative of b_Z in R direction on the 3D mesh
    real_t* dgyzdx_over_g_ptr;
    // Pointer to the inverse of sqrt(abs(det(g)))
    real_t* inv_g_ptr;
    // Pointer to the derivative of abs B in R direction on the 3D mesh
    real_t* dabsBdx_ptr;
    // Pointer to the derivative of abs B in Z direction on the 3D mesh
    real_t* dabsBdz_ptr;
    // Pointer to the derivative of abs B in y direction on the 3D mesh
    real_t* dabsBdy_ptr;

    // Class members related to parallel connection (parcon)

    // Pointer to the array specifying whether a given point is not in
    // the divertor or wall
    int32_t* not_in_target_ptr;
    // Pointer to field line length array from each point to the k + 1 plane
    real_t* fll_positive1_ptr;
    // Pointer to field line length array from each point to the k + 2 plane
    real_t* fll_positive2_ptr;
    // Pointer to field line length array from each point to the k - 1 plane
    real_t* fll_negative1_ptr;
    // Pointer to field line length array from each point to the k - 2 plane
    real_t* fll_negative2_ptr;
    // Pointer to the map matrix instances for the poloidal planes k + 1
    csrmat_genex_t* map_positive1_ptr;
    // Pointer to the map matrix instances for the poloidal planes k - 1
    csrmat_genex_t* map_negative1_ptr;
    // Pointer to the map matrix instances for the poloidal planes k + 2
    csrmat_genex_t* map_positive2_ptr;
    // Pointer to the map matrix instances for the poloidal planes k - 2
    csrmat_genex_t* map_negative2_ptr;

    // Check if the device memory is set and used as intended
    inline void dmem_debug(const dmd::mode_t mode)
    {
        bool err = false;
        int32_t size_mesh = this->size_RZ * this->size_phi;
        int32_t size_mesh_part = this->size_RZ
                               * (this->ub_phi - this->lb_phi + 1);
        int32_t size_neigh_arr = this->size_RZ_stencil * this->size_RZ_stencil *
                                 this->size_RZ * this->size_phi;

        dmd::start_region("mesh_5d_t", mode);
        err = err || dmd::is_invalid(this, 1);
        err = err || dmd::is_invalid(this->neighbors_ptr, size_neigh_arr);
        err = err || dmd::is_invalid(this->buf_zone_ptr, size_mesh_part);
        err = err || dmd::is_invalid(this->RZ_indices_ptr, size_mesh_part);
        err = err || dmd::is_invalid(this->not_filler_ptr, size_mesh_part);
        err = err || dmd::is_invalid(this->is_compute_ptr, size_mesh_part);
        err = err || dmd::is_invalid(this->vp_grid_ptr, this->size_vp);
        err = err || dmd::is_invalid(this->mu_grid_ptr, this->size_mu);
        err = err || dmd::is_invalid(this->sqrt_mu_grid_ptr, this->size_mu);
        err = err || dmd::is_invalid(this->vp_weights_ptr, this->size_vp);
        err = err || dmd::is_invalid(this->mu_weights_ptr, this->size_mu);
        err = err || dmd::is_invalid(this->jacobian_buffer_ptr, size_mesh_part);

        err = err || dmd::is_invalid(this->absB_buffer_ptr, size_mesh);
        err = err || dmd::is_invalid(this->normb_R_buffer_ptr, size_mesh);
        err = err || dmd::is_invalid(this->normb_Z_buffer_ptr, size_mesh);
        err = err || dmd::is_invalid(this->curl_normb_y_ptr, size_mesh);
        err = err || dmd::is_invalid(this->dgyxdy_over_g_ptr, size_mesh);
        err = err || dmd::is_invalid(this->dgyzdy_over_g_ptr, size_mesh);
        err = err || dmd::is_invalid(this->dgyxdz_over_g_ptr, size_mesh);
        err = err || dmd::is_invalid(this->dgyzdx_over_g_ptr, size_mesh);
        err = err || dmd::is_invalid(this->inv_g_ptr, size_mesh);
        err = err || dmd::is_invalid(this->dabsBdx_ptr, size_mesh);
        err = err || dmd::is_invalid(this->dabsBdz_ptr, size_mesh);
        err = err || dmd::is_invalid(this->dabsBdy_ptr, size_mesh);

        err = err || dmd::is_invalid(this->not_in_target_ptr, size_mesh);
        err = err || dmd::is_invalid(this->fll_positive1_ptr, size_mesh);
        err = err || dmd::is_invalid(this->fll_positive2_ptr, size_mesh);
        err = err || dmd::is_invalid(this->fll_negative1_ptr, size_mesh);
        err = err || dmd::is_invalid(this->fll_negative2_ptr, size_mesh);
        err = err || dmd::is_invalid(this->map_positive1_ptr, this->size_phi);
        err = err || dmd::is_invalid(this->map_positive2_ptr, this->size_phi);
        err = err || dmd::is_invalid(this->map_negative1_ptr, this->size_phi);
        err = err || dmd::is_invalid(this->map_negative2_ptr, this->size_phi);
        dmd::end_region("mesh_5d_t");

        mesh_5d::is_erroneous = mesh_5d::is_erroneous || err;
    }

public:
    // Constructor
    mesh_5d_t(struct mesh_5d_data_t* mesh_data)
    {
        // Copy the value of the scalar members
        this->size_RZ          = mesh_data->size_RZ;
        this->size_phi         = mesh_data->size_phi;
        this->size_vp          = mesh_data->size_vp;
        this->size_mu          = mesh_data->size_mu;
        this->size_sp          = mesh_data->size_sp;
        this->order_RZ_stencil = mesh_data->order_RZ_stencil;
        this->size_RZ_stencil  = 2 * mesh_data->order_RZ_stencil + 1;
        this->lb_phi           = mesh_data->lb_phi;
        this->ub_phi           = mesh_data->ub_phi;
        this->delta_RZ         = mesh_data->delta_RZ;
        this->delta_phi        = mesh_data->delta_phi;
        this->delta_vp         = mesh_data->delta_vp;
        this->delta_sqrt_mu    = mesh_data->delta_sqrt_mu;

        // If mesh swap is false, soft copy mesh members. Otherwise, hard copy
        // the members, with or without custom alignment
        if (!pgpus::get_swap_mesh_members())
        {
            // Store the pointer of the array members related to mesh
            this->neighbors_ptr       = mesh_data->neighbors_ptr;
            this->buf_zone_ptr        = mesh_data->buf_zone_ptr;
            this->RZ_indices_ptr      = mesh_data->RZ_indices_ptr;
            this->not_filler_ptr      = mesh_data->not_filler_ptr;
            this->is_compute_ptr      = mesh_data->is_compute_ptr;
            this->vp_grid_ptr         = mesh_data->vp_grid_ptr;
            this->mu_grid_ptr         = mesh_data->mu_grid_ptr;
            this->sqrt_mu_grid_ptr    = mesh_data->sqrt_mu_grid_ptr;
            this->vp_weights_ptr      = mesh_data->vp_weights_ptr;
            this->mu_weights_ptr      = mesh_data->mu_weights_ptr;
            this->jacobian_buffer_ptr = mesh_data->jacobian_buffer_ptr;

            // Store the pointer of the array members related to magnetic field
            this->absB_buffer_ptr    = mesh_data->absB_buffer_ptr;
            this->normb_R_buffer_ptr = mesh_data->normb_R_buffer_ptr;
            this->normb_Z_buffer_ptr = mesh_data->normb_Z_buffer_ptr;
            this->curl_normb_y_ptr   = mesh_data->curl_normb_y_ptr;
            this->dgyxdy_over_g_ptr  = mesh_data->dgyxdy_over_g_ptr;
            this->dgyzdy_over_g_ptr  = mesh_data->dgyzdy_over_g_ptr;
            this->dgyxdz_over_g_ptr  = mesh_data->dgyxdz_over_g_ptr;
            this->dgyzdx_over_g_ptr  = mesh_data->dgyzdx_over_g_ptr;
            this->inv_g_ptr          = mesh_data->inv_g_ptr;
            this->dabsBdx_ptr        = mesh_data->dabsBdx_ptr;
            this->dabsBdz_ptr        = mesh_data->dabsBdz_ptr;
            this->dabsBdy_ptr        = mesh_data->dabsBdy_ptr;

            // Store the pointer of the array members related to
            // parallel connection
            this->not_in_target_ptr = mesh_data->not_in_target_ptr;
            this->fll_positive1_ptr = mesh_data->fll_positive1_ptr;
            this->fll_positive2_ptr = mesh_data->fll_positive2_ptr;
            this->fll_negative1_ptr = mesh_data->fll_negative1_ptr;
            this->fll_negative2_ptr = mesh_data->fll_negative2_ptr;
        }
        else
        {
            int32_t n_phi = this->ub_phi - this->lb_phi + 1;
            int32_t size_neigh_arr = this->size_RZ_stencil
                                   * this->size_RZ_stencil
                                   * this->size_RZ * n_phi;
            int32_t size_3d = this->size_RZ * n_phi;

            if(pgpus::get_use_array_alignment())
            {
                std::align_val_t align = pgpus::get_array_alignment();

                // Allocate the pointer of the array members related to mesh
                this->neighbors_ptr       = new (align) int32_t[size_neigh_arr];
                this->buf_zone_ptr        = new (align) int32_t[size_3d];
                this->RZ_indices_ptr      = new (align) int32_t[size_3d];
                this->not_filler_ptr      = new (align) real_t[size_3d];
                this->is_compute_ptr      = new (align) real_t[size_3d];
                this->vp_grid_ptr         = new (align) real_t[this->size_vp];
                this->mu_grid_ptr         = new (align) real_t[this->size_mu];
                this->sqrt_mu_grid_ptr    = new (align) real_t[this->size_mu];
                this->vp_weights_ptr      = new (align) real_t[this->size_vp];
                this->mu_weights_ptr      = new (align) real_t[this->size_mu];
                this->jacobian_buffer_ptr = new (align) real_t[size_3d];

                // Allocate the pointer of the array members related to
                // magnetic field
                this->absB_buffer_ptr    = new (align) real_t[size_3d];
                this->normb_R_buffer_ptr = new (align) real_t[size_3d];
                this->normb_Z_buffer_ptr = new (align) real_t[size_3d];
                this->curl_normb_y_ptr   = new (align) real_t[size_3d];
                this->dgyxdy_over_g_ptr  = new (align) real_t[size_3d];
                this->dgyzdy_over_g_ptr  = new (align) real_t[size_3d];
                this->dgyxdz_over_g_ptr  = new (align) real_t[size_3d];
                this->dgyzdx_over_g_ptr  = new (align) real_t[size_3d];
                this->inv_g_ptr          = new (align) real_t[size_3d];
                this->dabsBdx_ptr        = new (align) real_t[size_3d];
                this->dabsBdz_ptr        = new (align) real_t[size_3d];
                this->dabsBdy_ptr        = new (align) real_t[size_3d];

                // Allocate the pointer of the array members related to
                // parallel connection
                this->not_in_target_ptr = new (align) int32_t[size_3d];
                this->fll_positive1_ptr = new (align) real_t[size_3d];
                this->fll_positive2_ptr = new (align) real_t[size_3d];
                this->fll_negative1_ptr = new (align) real_t[size_3d];
                this->fll_negative2_ptr = new (align) real_t[size_3d];
            }
            else
            {
                this->neighbors_ptr       = new int32_t[size_neigh_arr];
                this->buf_zone_ptr        = new int32_t[size_3d];
                this->RZ_indices_ptr      = new int32_t[size_3d];
                this->not_filler_ptr      = new real_t[size_3d];
                this->is_compute_ptr      = new real_t[size_3d];
                this->vp_grid_ptr         = new real_t[this->size_vp];
                this->mu_grid_ptr         = new real_t[this->size_mu];
                this->sqrt_mu_grid_ptr    = new real_t[this->size_mu];
                this->vp_weights_ptr      = new real_t[this->size_vp];
                this->mu_weights_ptr      = new real_t[this->size_mu];
                this->jacobian_buffer_ptr = new real_t[size_3d];

                this->absB_buffer_ptr    = new real_t[size_3d];
                this->normb_R_buffer_ptr = new real_t[size_3d];
                this->normb_Z_buffer_ptr = new real_t[size_3d];
                this->curl_normb_y_ptr   = new real_t[size_3d];
                this->dgyxdy_over_g_ptr  = new real_t[size_3d];
                this->dgyzdy_over_g_ptr  = new real_t[size_3d];
                this->dgyxdz_over_g_ptr  = new real_t[size_3d];
                this->dgyzdx_over_g_ptr  = new real_t[size_3d];
                this->inv_g_ptr          = new real_t[size_3d];
                this->dabsBdx_ptr        = new real_t[size_3d];
                this->dabsBdz_ptr        = new real_t[size_3d];
                this->dabsBdy_ptr        = new real_t[size_3d];

                this->not_in_target_ptr = new int32_t[size_3d];
                this->fll_positive1_ptr = new real_t[size_3d];
                this->fll_positive2_ptr = new real_t[size_3d];
                this->fll_negative1_ptr = new real_t[size_3d];
                this->fll_negative2_ptr = new real_t[size_3d];
            }

            // Hard copy mesh members

            #pragma omp parallel for simd default(none) \
                shared(size_neigh_arr, mesh_data) schedule(static)
            for (int32_t j = 0; j < size_neigh_arr; j++)
            {
                this->neighbors_ptr[j] = mesh_data->neighbors_ptr[j];
            }

            #pragma omp parallel for simd default(none) \
                shared(size_vp, mesh_data) schedule(static)
            for (int32_t l = 0; l < this->size_vp; l++)
            {
                this->vp_grid_ptr[l]    = mesh_data->vp_grid_ptr[l];
                this->vp_weights_ptr[l] = mesh_data->vp_weights_ptr[l];
            }

            #pragma omp parallel for simd default(none) \
                shared(size_mu, mesh_data) schedule(static)
            for (int32_t m = 0; m < this->size_mu; m++)
            {
                this->mu_grid_ptr[m]         = mesh_data->mu_grid_ptr[m];
                this->sqrt_mu_grid_ptr[m]    = mesh_data->sqrt_mu_grid_ptr[m];
                this->mu_weights_ptr[m]      = mesh_data->mu_weights_ptr[m];
            }

            #pragma omp parallel for simd default(none) \
                shared(size_3d, mesh_data) schedule(static)
            for (int32_t j = 0; j < size_3d; j++)
            {
                this->buf_zone_ptr[j]        = mesh_data->buf_zone_ptr[j];
                this->RZ_indices_ptr[j]      = mesh_data->RZ_indices_ptr[j];
                this->not_filler_ptr[j]      = mesh_data->not_filler_ptr[j];
                this->is_compute_ptr[j]      = mesh_data->is_compute_ptr[j];
                this->jacobian_buffer_ptr[j] =
                    mesh_data->jacobian_buffer_ptr[j];

                this->absB_buffer_ptr[j]    = mesh_data->absB_buffer_ptr[j];
                this->normb_R_buffer_ptr[j] = mesh_data->normb_R_buffer_ptr[j];
                this->normb_Z_buffer_ptr[j] = mesh_data->normb_Z_buffer_ptr[j];
                this->curl_normb_y_ptr[j]   = mesh_data->curl_normb_y_ptr[j];
                this->dgyxdy_over_g_ptr[j]  = mesh_data->dgyxdy_over_g_ptr[j];
                this->dgyzdy_over_g_ptr[j]  = mesh_data->dgyzdy_over_g_ptr[j];
                this->dgyxdz_over_g_ptr[j]  = mesh_data->dgyxdz_over_g_ptr[j];
                this->dgyzdx_over_g_ptr[j]  = mesh_data->dgyzdx_over_g_ptr[j];
                this->inv_g_ptr[j]          = mesh_data->inv_g_ptr[j];
                this->dabsBdx_ptr[j]        = mesh_data->dabsBdx_ptr[j];
                this->dabsBdz_ptr[j]        = mesh_data->dabsBdz_ptr[j];
                this->dabsBdy_ptr[j]        = mesh_data->dabsBdy_ptr[j];

                this->not_in_target_ptr[j] = mesh_data->not_in_target_ptr[j];
                this->fll_positive1_ptr[j] = mesh_data->fll_positive1_ptr[j];
                this->fll_positive2_ptr[j] = mesh_data->fll_positive2_ptr[j];
                this->fll_negative1_ptr[j] = mesh_data->fll_negative1_ptr[j];
                this->fll_negative2_ptr[j] = mesh_data->fll_negative2_ptr[j];
            }

            // Swap the pointer of the array members related to mesh
            mesh_data->neighbors_ptr       = this->neighbors_ptr;
            mesh_data->buf_zone_ptr        = this->buf_zone_ptr;
            mesh_data->RZ_indices_ptr      = this->RZ_indices_ptr;
            mesh_data->not_filler_ptr      = this->not_filler_ptr;
            mesh_data->is_compute_ptr      = this->is_compute_ptr;
            mesh_data->vp_grid_ptr         = this->vp_grid_ptr;
            mesh_data->mu_grid_ptr         = this->mu_grid_ptr;
            mesh_data->sqrt_mu_grid_ptr    = this->sqrt_mu_grid_ptr;
            mesh_data->vp_weights_ptr      = this->vp_weights_ptr;
            mesh_data->mu_weights_ptr      = this->mu_weights_ptr;
            mesh_data->jacobian_buffer_ptr = this->jacobian_buffer_ptr;

            // Swap the pointer of the array members related to magnetic field
            mesh_data->absB_buffer_ptr    = this->absB_buffer_ptr;
            mesh_data->normb_R_buffer_ptr = this->normb_R_buffer_ptr;
            mesh_data->normb_Z_buffer_ptr = this->normb_Z_buffer_ptr;
            mesh_data->curl_normb_y_ptr   = this->curl_normb_y_ptr;
            mesh_data->dgyxdy_over_g_ptr  = this->dgyxdy_over_g_ptr;
            mesh_data->dgyzdy_over_g_ptr  = this->dgyzdy_over_g_ptr;
            mesh_data->dgyxdz_over_g_ptr  = this->dgyxdz_over_g_ptr;
            mesh_data->dgyzdx_over_g_ptr  = this->dgyzdx_over_g_ptr;
            mesh_data->inv_g_ptr          = this->inv_g_ptr;
            mesh_data->dabsBdx_ptr        = this->dabsBdx_ptr;
            mesh_data->dabsBdz_ptr        = this->dabsBdz_ptr;
            mesh_data->dabsBdy_ptr        = this->dabsBdy_ptr;

            // Swap the pointer of the array members related to
            // parallel connection
            mesh_data->not_in_target_ptr = this->not_in_target_ptr;
            mesh_data->fll_positive1_ptr = this->fll_positive1_ptr;
            mesh_data->fll_positive2_ptr = this->fll_positive2_ptr;
            mesh_data->fll_negative1_ptr = this->fll_negative1_ptr;
            mesh_data->fll_negative2_ptr = this->fll_negative2_ptr;
        }
    }

    // Destructor
    virtual ~mesh_5d_t()
    {
        // Deallocate the host arrays of map matrix (csrmat) C++ class
        // instances
        delete[] this->map_negative2_ptr;
        delete[] this->map_negative1_ptr;
        delete[] this->map_positive2_ptr;
        delete[] this->map_positive1_ptr;

        if (pgpus::get_swap_mesh_members())
        {
            if (pgpus::get_use_array_alignment())
            {
                std::align_val_t align = pgpus::get_array_alignment();

                // Deallocate the pointer of the array members related to
                // parallel connection
                operator delete(this->fll_negative2_ptr, align);
                operator delete(this->fll_negative1_ptr, align);
                operator delete(this->fll_positive2_ptr, align);
                operator delete(this->fll_positive1_ptr, align);
                operator delete(this->not_in_target_ptr, align);

                // Deallocate the pointer of the array members related to
                // magnetic field
                operator delete(this->dabsBdy_ptr, align);
                operator delete(this->dabsBdz_ptr, align);
                operator delete(this->dabsBdx_ptr, align);
                operator delete(this->inv_g_ptr, align);
                operator delete(this->dgyzdx_over_g_ptr, align);
                operator delete(this->dgyxdz_over_g_ptr, align);
                operator delete(this->dgyzdy_over_g_ptr, align);
                operator delete(this->dgyxdy_over_g_ptr, align);
                operator delete(this->curl_normb_y_ptr, align);
                operator delete(this->normb_Z_buffer_ptr, align);
                operator delete(this->normb_R_buffer_ptr, align);
                operator delete(this->absB_buffer_ptr, align);

                // Deallocate the pointer of the array members related to mesh
                operator delete(this->jacobian_buffer_ptr, align);
                operator delete(this->mu_weights_ptr, align);
                operator delete(this->vp_weights_ptr, align);
                operator delete(this->sqrt_mu_grid_ptr, align);
                operator delete(this->mu_grid_ptr, align);
                operator delete(this->vp_grid_ptr, align);
                operator delete(this->is_compute_ptr, align);
                operator delete(this->not_filler_ptr, align);
                operator delete(this->RZ_indices_ptr, align);
                operator delete(this->buf_zone_ptr, align);
                operator delete(this->neighbors_ptr, align);
            }
            else
            {
                delete[] this->fll_negative2_ptr;
                delete[] this->fll_negative1_ptr;
                delete[] this->fll_positive2_ptr;
                delete[] this->fll_positive1_ptr;
                delete[] this->not_in_target_ptr;

                delete[] this->dabsBdy_ptr;
                delete[] this->dabsBdz_ptr;
                delete[] this->dabsBdx_ptr;
                delete[] this->inv_g_ptr;
                delete[] this->dgyzdx_over_g_ptr;
                delete[] this->dgyxdz_over_g_ptr;
                delete[] this->dgyzdy_over_g_ptr;
                delete[] this->dgyxdy_over_g_ptr;
                delete[] this->curl_normb_y_ptr;
                delete[] this->normb_Z_buffer_ptr;
                delete[] this->normb_R_buffer_ptr;
                delete[] this->absB_buffer_ptr;

                delete[] this->jacobian_buffer_ptr;
                delete[] this->mu_weights_ptr;
                delete[] this->vp_weights_ptr;
                delete[] this->sqrt_mu_grid_ptr;
                delete[] this->mu_grid_ptr;
                delete[] this->vp_grid_ptr;
                delete[] this->is_compute_ptr;
                delete[] this->not_filler_ptr;
                delete[] this->RZ_indices_ptr;
                delete[] this->buf_zone_ptr;
                delete[] this->neighbors_ptr;
            }
        }
    }

    // Copy constructor is disabled
    mesh_5d_t(const mesh_5d_t&) = delete;

    // Copy-assignment operator is disabled
    mesh_5d_t& operator=(const mesh_5d_t&) = delete;

    // Getter for the maximum number of points in RZ direction including ghost
    // and filler points
    #pragma acc routine seq
    inline int32_t get_size_RZ() const
    {
        return this->size_RZ;
    }

    // Getter for the number of points in phi direction
    #pragma acc routine seq
    inline int32_t get_size_phi() const
    {
        return this->size_phi;
    }

    // Getter for the number of points in vp direction
    #pragma acc routine seq
    inline int32_t get_size_vp() const
    {
        return this->size_vp;
    }

    // Getter for the number of points in mu direction
    #pragma acc routine seq
    inline int32_t get_size_mu() const
    {
        return this->size_mu;
    }

    // Getter for the number of points in sp direction
    #pragma acc routine seq
    inline int32_t get_size_sp() const
    {
        return this->size_sp;
    }

    // Getter for the order of RZ stencil
    #pragma acc routine seq
    inline int32_t get_order_RZ_stencil() const
    {
        return this->order_RZ_stencil;
    }

    // Getter for the lower bound of phi direction
    #pragma acc routine seq
    inline int32_t get_lb_phi() const
    {
        return this->lb_phi;
    }

    // Getter for the upper bound of phi direction
    #pragma acc routine seq
    inline int32_t get_ub_phi() const
    {
        return this->ub_phi;
    }

    // Getter for the grid spacing of the RZ grid
    #pragma acc routine seq
    inline real_t get_delta_RZ() const
    {
        return this->delta_RZ;
    }

    // Getter for the grid spacing of the phi grid
    #pragma acc routine seq
    inline real_t get_delta_phi() const
    {
        return this->delta_phi;
    }

    // Getter for the grid spacing of the vp grid
    #pragma acc routine seq
    inline real_t get_delta_vp() const
    {
        return this->delta_vp;
    }

    // Getter for the grid spacing of the RZ grid
    #pragma acc routine seq
    inline real_t get_delta_sqrt_mu() const
    {
        return this->delta_sqrt_mu;
    }

    // NOTE: The following getters are for multidimensional array access of the
    //       array members with Fortran-style indexing

    // Getter for neighbors
    #pragma acc routine seq
    inline int32_t neighbors(int j0, int j1, int j2, int j3) const
    {
        return this->neighbors_ptr[(j0 + this->order_RZ_stencil) +
            (j1 + this->order_RZ_stencil) * this->size_RZ_stencil +
            (j2 - 1) * this->size_RZ_stencil * this->size_RZ_stencil +
            (j3 - 1) * this->size_RZ_stencil * this->size_RZ_stencil
                     * this->size_RZ];
    }

    // Getter for buf_zone
    #pragma acc routine seq
    inline int32_t buf_zone(int j0, int j1) const
    {
        return this->buf_zone_ptr[(j0 - 1) + (j1 - 1) * this->size_RZ];
    }

    // Getter for RZ_indices
    #pragma acc routine seq
    inline int32_t RZ_indices(int j0, int j1) const
    {
        return this->RZ_indices_ptr[(j0 - 1) + (j1 - 1) * this->size_RZ];
    }

    // Getter for not_filler
    #pragma acc routine seq
    inline real_t not_filler(int j0, int j1) const
    {
        return this->not_filler_ptr[(j0 - 1) + (j1 - 1) * this->size_RZ];
    }

    // Getter for is_compute
    #pragma acc routine seq
    inline real_t is_compute(int j0, int j1) const
    {
        return this->is_compute_ptr[(j0 - 1) + (j1 - 1) * this->size_RZ];
    }

    // Getter for vp_grid
    #pragma acc routine seq
    inline real_t vp(int j0) const
    {
        return this->vp_grid_ptr[j0 - 1];
    }

    // Getter for mu_grid
    #pragma acc routine seq
    inline real_t mu(int j0) const
    {
        return this->mu_grid_ptr[j0 - 1];
    }

    // Getter for sqrt(mu)_grid
    #pragma acc routine seq
    inline real_t sqrt_mu(int j0) const
    {
        return this->sqrt_mu_grid_ptr[j0 - 1];
    }

    // Getter for vp_weights
    #pragma acc routine seq
    inline real_t vpw(int j0) const
    {
        return this->vp_weights_ptr[j0 - 1];
    }

    // Getter for mu_weights
    #pragma acc routine seq
    inline real_t muw(int j0) const
    {
        return this->mu_weights_ptr[j0 - 1];
    }

    // Getter for jacobian buffer
    #pragma acc routine seq
    inline real_t jacobian(int j0, int j1) const
    {
        return this->jacobian_buffer_ptr[(j0 - 1) + (j1 - 1) * this->size_RZ];
    }

    // Getter for absB_buffer
    #pragma acc routine seq
    inline real_t absB(int j0, int j1) const
    {
        return this->absB_buffer_ptr[(j0 - 1) + (j1 - 1) * this->size_RZ];
    }

    // Getter for normb_R_buffer
    #pragma acc routine seq
    inline real_t normb_R(int j0, int j1) const
    {
        return this->normb_R_buffer_ptr[(j0 - 1) +
                                        (j1 - 1) * this->size_RZ];
    }

    // Getter for normb_Z_buffer
    #pragma acc routine seq
    inline real_t normb_Z(int j0, int j1) const
    {
        return this->normb_Z_buffer_ptr[(j0 - 1) +
                                        (j1 - 1) * this->size_RZ];
    }

    // Getter for curl_normb_y
    #pragma acc routine seq
    inline real_t curl_normb_y (int j0, int j1) const
    {
        return this->curl_normb_y_ptr[(j0 - 1) + (j1 - 1) * this->size_RZ];
    }

    // Getter for dgyxdy_over_g
    #pragma acc routine seq
    inline real_t dgyxdy_over_g(int j0, int j1) const
    {
        return this->dgyxdy_over_g_ptr[(j0 - 1) +
                                       (j1 - 1) * this->size_RZ];
    }

    // Getter for dgyzdy_over_g
    #pragma acc routine seq
    inline real_t dgyzdy_over_g(int j0, int j1) const
    {
        return this->dgyzdy_over_g_ptr[(j0 - 1) +
                                       (j1 - 1) * this->size_RZ];
    }

    // Getter for dgyxdz_over_g
    #pragma acc routine seq
    inline real_t dgyxdz_over_g(int j0, int j1) const
    {
        return this->dgyxdz_over_g_ptr[(j0 - 1) +
                                       (j1 - 1) * this->size_RZ];
    }

    // Getter for dgzxdx_over_g
    #pragma acc routine seq
    inline real_t dgyzdx_over_g(int j0, int j1) const
    {
        return this->dgyzdx_over_g_ptr[(j0 - 1) +
                                       (j1 - 1) * this->size_RZ];
    }

    // Getter for inv_g
    #pragma acc routine seq
    inline real_t inv_g(int j0, int j1) const
    {
        return this->inv_g_ptr[(j0 - 1) + (j1 - 1) * this->size_RZ];
    }

    // Getter for dabsBdx
    #pragma acc routine seq
    inline real_t dabsBdx(int j0, int j1) const
    {
        return this->dabsBdx_ptr[(j0 - 1) + (j1 - 1) * this->size_RZ];
    }

    // Getter for dabsBdz
    #pragma acc routine seq
    inline real_t dabsBdz(int j0, int j1) const
    {
        return this->dabsBdz_ptr[(j0 - 1) + (j1 - 1) * this->size_RZ];
    }

    // Getter for dabsBdy
    #pragma acc routine seq
    inline real_t dabsBdy(int j0, int j1) const
    {
        return this->dabsBdy_ptr[(j0 - 1) + (j1 - 1) * this->size_RZ];
    }

    // Getter for map_positive1
    #pragma acc routine seq
    inline csrmat_genex_t& map_positive1(int j0) const
    {
        return this->map_positive1_ptr[j0 - 1];
    }

    // Getter for map_negative1
    #pragma acc routine seq
    inline csrmat_genex_t& map_negative1(int j0) const
    {
        return this->map_negative1_ptr[j0 - 1];
    }

    // Getter for map_positive2
    #pragma acc routine seq
    inline csrmat_genex_t& map_positive2(int j0) const
    {
        return this->map_positive2_ptr[j0 - 1];
    }

    // Getter for map_negative2
    #pragma acc routine seq
    inline csrmat_genex_t& map_negative2(int j0) const
    {
        return this->map_negative2_ptr[j0 - 1];
    }

    // Getter for fll_positive1
    #pragma acc routine seq
    inline real_t fll_positive1(int j0, int j1) const
    {
        return this->fll_positive1_ptr[(j0 - 1) + (j1 - 1) * this->size_RZ];
    }

    // Getter for fll_positive2
    #pragma acc routine seq
    inline real_t fll_positive2(int j0, int j1) const
    {
        return this->fll_positive2_ptr[(j0 - 1) + (j1 - 1) * this->size_RZ];
    }

    // Getter for fll_negative1
    #pragma acc routine seq
    inline real_t fll_negative1(int j0, int j1) const
    {
        return this->fll_negative1_ptr[(j0 - 1) + (j1 - 1) * this->size_RZ];
    }

    // Getter for fll_negative2
    #pragma acc routine seq
    inline real_t fll_negative2(int j0, int j1) const
    {
        return this->fll_negative2_ptr[(j0 - 1) + (j1 - 1) * this->size_RZ];
    }

    // Getter for not_in_target
    #pragma acc routine seq
    inline int32_t not_in_target(int j0, int j1) const
    {
        return this->not_in_target_ptr[(j0 - 1) + (j1 - 1) * this->size_RZ];
    }
};

#ifdef __cplusplus
extern "C" {
#endif

// Shallow copy Fortran class to C++ class and allocate class to GPU
// Return 0 for success and 1 for error
int32_t cbind_mesh_5d_initialize(struct mesh_5d_data_t* mesh_data,
                                 const char* grid_type_phi,
                                 const char* grid_type_vp,
                                 const char* grid_type_mu,
                                 const char* quad_type_phi,
                                 const char* quad_type_vp,
                                 const char* quad_type_mu,
                                 mesh_5d_t** mesh_cxx_pptr);

// Deallocate class from GPU if intended and C++ class instance while keeping
// the member allocations on CPU until they are freed from the Fortran layer.
// Return 0 for success and 1 for error.
int32_t cbind_mesh_5d_finalize(mesh_5d_t** mesh_cxx_pptr);

#ifdef __cplusplus
}
#endif

#endif
