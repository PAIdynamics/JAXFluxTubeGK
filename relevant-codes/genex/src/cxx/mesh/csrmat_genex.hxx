#ifndef CSRMAT_GENEX_HXX
#define CSRMAT_GENEX_HXX

#include "genex_cxx_env.hxx"
#include "device_memory_debugger.hxx"

// Alias for external namespace
namespace dmd = device_memory_debugger;

#ifdef __cplusplus
extern "C" {
#endif

// GENE-X version of Fortran/C++ interoperable struct containing
// csrmat_t class members
struct csrmat_genex_data_t {
    int32_t  ndim;
    int32_t  ncol;
    int32_t  nnz;

    int32_t* i_ptr;
    int32_t* j_ptr;
    real_t*  val_ptr;
};

#ifdef __cplusplus
}
#endif

// GENE-X version of C++ class which corresponds to Fortran class csrmat_t
class csrmat_genex_t
{
protected:
    // Number of rows (dimension)
    int32_t  ndim;
    // Number of columns
    int32_t  ncol;
    // Number of non-zero elements
    int32_t  nnz;

    // Pointer to the i-array in CSR format, of dimension ndim+1
    int32_t* i_ptr;
    // Pointer to the j-array in CSR format, of dimension nnz
    int32_t* j_ptr;
    // Pointer to the val-array in CSR format, of dimension nnz
    real_t*  val_ptr;

    // Check if the device memory is set and used as intended
    // Return false if it works as intended and true if not
    inline bool dmem_debug(const dmd::mode_t mode)
    {
        bool err = false;

        dmd::start_region("csrmat_genex_t", mode);
        err = err || dmd::is_invalid(this->i_ptr, this->ndim + 1);
        err = err || dmd::is_invalid(this->j_ptr, this->nnz);
        err = err || dmd::is_invalid(this->val_ptr, this->nnz);
        dmd::end_region("csrmat_genex_t");

        return err;
    }

public:
    // Default constructor
    csrmat_genex_t() = default;

    // Parameterized constructor
    csrmat_genex_t(const struct csrmat_genex_data_t* csrmat_data);

    // Virtual destructor
    virtual ~csrmat_genex_t() = default;

    // Copy constructor is disabled
    csrmat_genex_t(const csrmat_genex_t&) = delete;

    // Copy-assignment operator
    // NOTE: Copy assignment operator of csrmat is called during the constructor
    //       of mesh class due to the initialization of array of map matrices.
    //       The default operator is sufficient.
    csrmat_genex_t& operator=(const csrmat_genex_t&) = default;

    // Virtual routine for initialization of csrmat_t class on GPU
    // Return false if properly allocated on the device memory and true if not
    virtual bool initialize_device() = 0;

    // Virtual routine for finalization of csrmat_t class on GPU
    // Return false if properly deallocated on the device memory and true if not
    virtual bool finalize_device() = 0;

    // Getter for the matrix size
    #pragma acc routine seq
    inline int32_t get_ndim() const
    {
        return this->ndim;
    }

    // Getter for the number of columns
    #pragma acc routine seq
    inline int32_t get_ncol() const
    {
        return this->ncol;
    }

    // Getter for the number of the non-zero elements
    #pragma acc routine seq
    inline int32_t get_nnz() const
    {
        return this->nnz;
    }

    // Getter for the i value
    #pragma acc routine seq
    inline int32_t& i(int j0) const
    {
        return this->i_ptr[j0 - 1];
    }

    // Getter for the j value
    #pragma acc routine seq
    inline int32_t& j(int j0) const
    {
        return this->j_ptr[j0 - 1];
    }

    // Getter for the value of the non-zero elements
    #pragma acc routine seq
    inline real_t& val(int j0) const
    {
        return this->val_ptr[j0 - 1];
    }
};

#endif
