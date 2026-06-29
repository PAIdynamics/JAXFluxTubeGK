#include "mesh_5d.hxx"
#include "mesh_5d_factory.hxx"

namespace mesh_5d
{
    // Private objects with file scope and external linkage
    // NOTE: The reason why the following grid and quadrature types are not
    //       included to mesh_5d_t as class members are because:
    //       1) mesh_5d_t object is supposed to be accessible from inside
    //          GPU kernels, i.e. OpenACC and OpenMP offload;
    //       2) std::string is not supported by both frameworks.

    // Type of phi grid
    static std::string grid_type_phi = "uniform";
    // Type of vp grid
    static std::string grid_type_vp = "uniform";
    // Type of mu grid
    static std::string grid_type_mu = "gauss-laguerre";
    // Type of phi quadrature
    static std::string quad_type_phi = "midpoint";
    // Type of vp quadrature
    static std::string quad_type_vp = "simpson";
    // Type of mu quadrature
    static std::string quad_type_mu = "gauss-laguerre";

    // Getters for the private objects

    const std::string& get_grid_type_phi() { return grid_type_phi; }
    const std::string& get_grid_type_vp() { return grid_type_vp; }
    const std::string& get_grid_type_mu() { return grid_type_mu; }
    const std::string& get_quad_type_phi() { return quad_type_phi; }
    const std::string& get_quad_type_vp() { return quad_type_vp; }
    const std::string& get_quad_type_mu() { return quad_type_mu; }
}

int32_t cbind_mesh_5d_initialize(struct mesh_5d_data_t* mesh_data,
                                 const char* grid_type_phi,
                                 const char* grid_type_vp,
                                 const char* grid_type_mu,
                                 const char* quad_type_phi,
                                 const char* quad_type_vp,
                                 const char* quad_type_mu,
                                 mesh_5d_t** mesh_cxx_pptr)
{
    // Allocate and construct mesh_5d_t C++ class instance
    *mesh_cxx_pptr = mesh_5d::create(mesh_data);

    // Store the strings specifying the type of grids and quadratures used
    mesh_5d::grid_type_phi = std::string{grid_type_phi};
    mesh_5d::grid_type_vp  = std::string{grid_type_vp};
    mesh_5d::grid_type_mu  = std::string{grid_type_mu};
    mesh_5d::quad_type_phi = std::string{quad_type_phi};
    mesh_5d::quad_type_vp  = std::string{quad_type_vp};
    mesh_5d::quad_type_mu  = std::string{quad_type_mu};

    return (int32_t) mesh_5d::is_erroneous;
}

int32_t cbind_mesh_5d_finalize(mesh_5d_t** mesh_cxx_pptr)
{
    // Assign the mesh_5d_t class instance
    mesh_5d_t& mesh = *(*mesh_cxx_pptr);

    // Deallocate the host mesh_5d_t C++ class instance
    delete &mesh;

    // Clear the strings specifying the type of grids and quadratures used
    mesh_5d::grid_type_phi.clear();
    mesh_5d::grid_type_vp.clear();
    mesh_5d::grid_type_mu.clear();
    mesh_5d::quad_type_phi.clear();
    mesh_5d::quad_type_vp.clear();
    mesh_5d::quad_type_mu.clear();

    return (int32_t) mesh_5d::is_erroneous;
}
