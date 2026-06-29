#ifndef CSRMAT_GENEX_OMP_HXX
#define CSRMAT_GENEX_OMP_HXX

#include "genex_cxx_env.hxx"
#include "csrmat_genex.hxx"

// GENE-X version of C++ class which corresponds to Fortran class csrmat_t
// on CPU
class csrmat_genex_omp_t: public csrmat_genex_t
{
public:
    // Default constructor
    csrmat_genex_omp_t() = default;

    // Parameterized constructor of the OpenMP child class
    csrmat_genex_omp_t(const struct csrmat_genex_data_t* csrmat_data)
    : csrmat_genex_t{csrmat_data} {};

    // Destructor of the OpenMP child class
    ~csrmat_genex_omp_t() override {};

    // Copy constructor is disabled
    csrmat_genex_omp_t(const csrmat_genex_omp_t&) = delete;

    // Copy-assignment operator
    csrmat_genex_omp_t& operator=(const csrmat_genex_omp_t&) = default;

    // OpenMP routine for initialization of csrmat_t class. This does nothing.
    bool initialize_device() override
    {
        return false;
    }

    // OpenMP routine for finalization of csrmat_t class. This does nothing.
    bool finalize_device() override
    {
        return false;
    }
};

#endif
