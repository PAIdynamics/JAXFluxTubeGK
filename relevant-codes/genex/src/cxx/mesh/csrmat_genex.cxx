#include "csrmat_genex.hxx"

csrmat_genex_t::csrmat_genex_t(const struct csrmat_genex_data_t* csrmat_data)
{
    this->ndim    = csrmat_data->ndim;
    this->ncol    = csrmat_data->ncol;
    this->nnz     = csrmat_data->nnz;
    this->i_ptr   = csrmat_data->i_ptr;
    this->j_ptr   = csrmat_data->j_ptr;
    this->val_ptr = csrmat_data->val_ptr;
}
