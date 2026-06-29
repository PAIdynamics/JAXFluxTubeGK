#ifndef CSRMAT_GENEX_OMPX_HXX
#define CSRMAT_GENEX_OMPX_HXX

#include "genex_cxx_env.hxx"
#include "csrmat_genex.hxx"
#include <omp.h>

// GENE-X version of C++ class which corresponds to Fortran class csrmat_t
// with OpenMP offload
class csrmat_genex_ompx_t: public csrmat_genex_t
{
public:
    // Default constructor
    csrmat_genex_ompx_t() {};

    // Parameterized constructor of the OpenMP offload child class
    csrmat_genex_ompx_t(const struct csrmat_genex_data_t* csrmat_data)
    : csrmat_genex_t{csrmat_data} {};

    // Destructor of the OpenMP offload child class
    ~csrmat_genex_ompx_t() override {};

    // Copy constructor is disabled
    csrmat_genex_ompx_t(const csrmat_genex_ompx_t&) = delete;

    // Copy-assignment operator
    csrmat_genex_ompx_t& operator=(const csrmat_genex_ompx_t&) = default;

    // OpenMP offload routine for initialization of csrmat_t class
    // Return false if properly allocated on the device memory and true if not
    bool initialize_device() override
    {
        #pragma omp target enter data map(to: this->ndim, \
                                              this->ncol, \
                                              this->nnz, \
                                              this->i_ptr[:this->ndim + 1], \
                                              this->j_ptr[:this->nnz], \
                                              this->val_ptr[:this->nnz])

        return this->dmem_debug(dmd::mode_t::ALLOC);
    }

    // OpenMP offload routine for finalization of csrmat_t class
    // Return false if properly deallocated on the device memory and true if not
    bool finalize_device() override
    {
        #pragma omp target exit data map(delete: this->val_ptr[:this->nnz], \
                                                 this->j_ptr[:this->nnz], \
                                                 this->i_ptr[:this->ndim + 1], \
                                                 this->nnz, \
                                                 this->ncol, \
                                                 this->ndim)

        return this->dmem_debug(dmd::mode_t::DEALLOC);
    }
};

#endif
