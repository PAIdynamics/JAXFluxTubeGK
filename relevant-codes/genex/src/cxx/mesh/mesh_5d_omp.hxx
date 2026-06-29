#ifndef MESH_5D_OMP_HXX
#define MESH_5D_OMP_HXX

#include "genex_cxx_env.hxx"
#include "mesh_5d.hxx"
#include "csrmat_genex_omp.hxx"

// C++ class which corresponds to the Fortran class mesh_5d_t on CPU
class mesh_5d_omp_t: public mesh_5d_t
{
public:
    // Constructor of the OpenMP child class
    mesh_5d_omp_t(struct mesh_5d_data_t* mesh_data)
    : mesh_5d_t{mesh_data}
    {
        // Allocate arrays of the map matrix (csrmat) class instances
        this->map_positive1_ptr = new csrmat_genex_omp_t[mesh_data->size_phi];
        this->map_negative1_ptr = new csrmat_genex_omp_t[mesh_data->size_phi];
        this->map_positive2_ptr = new csrmat_genex_omp_t[mesh_data->size_phi];
        this->map_negative2_ptr = new csrmat_genex_omp_t[mesh_data->size_phi];

        // Construct the map matrix (csrmat) class instances
        for (int32_t k = 0; k < this->size_phi; k++)
        {
            this->map_positive1_ptr[k] = csrmat_genex_omp_t(
                &(mesh_data->map_positive1_data_ptr[k]));
            this->map_negative1_ptr[k] = csrmat_genex_omp_t(
                &(mesh_data->map_negative1_data_ptr[k]));
            this->map_positive2_ptr[k] = csrmat_genex_omp_t(
                &(mesh_data->map_positive2_data_ptr[k]));
            this->map_negative2_ptr[k] = csrmat_genex_omp_t(
                &(mesh_data->map_negative2_data_ptr[k]));
        }
        mesh_5d::is_erroneous = false;
    }

    // Destructor of the OpenMP child class
    ~mesh_5d_omp_t() override {};

    // Copy constructor is disabled
    mesh_5d_omp_t(const mesh_5d_omp_t&) = delete;

    // Copy-assignment operator is disabled
    mesh_5d_omp_t& operator=(const mesh_5d_omp_t&) = delete;
};

#endif
