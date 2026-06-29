#ifndef HELPERS_DCOMM_HANDLER_CXX
#define HELPERS_DCOMM_HANDLER_CXX

#include "genex_cxx_env.hxx"
#include "params_gpu_offload.hxx"
#include "dcomm_handler.hxx"
#include <mpi.h>

int32_t is_communicator_erroneous(
    MPI_Comm comm, const int32_t initial_value, const int32_t expected_value)
{
    int32_t global_sum;

    int32_t ierr = MPI_Allreduce(&initial_value, &global_sum, 1, MPI_INT,
                                 MPI_SUM, comm);

    return (ierr != MPI_SUCCESS) || (global_sum != expected_value);
}

#ifdef __cplusplus
extern "C" {
#endif

int32_t cbind_dcomm_handler_copy(
    const dcomm_handler_t** dcomm_handler_src_cxx_pptr,
    dcomm_handler_data_t* dcomm_handler_data_tgt)
{
    // Assign the source dcomm_handler_t class instance
    const dcomm_handler_t& dcomm_handler_src = *(*dcomm_handler_src_cxx_pptr);

    MPI_Comm comm_base = dcomm_handler_src.get_comm_base();
    MPI_Fint f_comm;

    f_comm = MPI_Comm_c2f(dcomm_handler_src.get_comm_base());
    dcomm_handler_data_tgt->comm_base = f_comm;
    f_comm = MPI_Comm_c2f(dcomm_handler_src.get_comm_cart());
    dcomm_handler_data_tgt->comm_cart = f_comm;
    f_comm = MPI_Comm_c2f(dcomm_handler_src.get_comm_phi());
    dcomm_handler_data_tgt->comm_phi = f_comm;
    f_comm = MPI_Comm_c2f(dcomm_handler_src.get_comm_vp());
    dcomm_handler_data_tgt->comm_vp = f_comm;
    f_comm = MPI_Comm_c2f(dcomm_handler_src.get_comm_mu());
    dcomm_handler_data_tgt->comm_mu = f_comm;
    f_comm = MPI_Comm_c2f(dcomm_handler_src.get_comm_sp());
    dcomm_handler_data_tgt->comm_sp = f_comm;
    f_comm = MPI_Comm_c2f(dcomm_handler_src.get_comm_RZ_phi_sp());
    dcomm_handler_data_tgt->comm_phi_sp = f_comm;
    f_comm = MPI_Comm_c2f(dcomm_handler_src.get_comm_vp_mu());
    dcomm_handler_data_tgt->comm_vp_mu = f_comm;
    f_comm = MPI_Comm_c2f(dcomm_handler_src.get_comm_phi_vp_mu());
    dcomm_handler_data_tgt->comm_phi_vp_mu = f_comm;
    f_comm = MPI_Comm_c2f(dcomm_handler_src.get_comm_vp_mu_sp());
    dcomm_handler_data_tgt->comm_vp_mu_sp = f_comm;
    f_comm = MPI_Comm_c2f(dcomm_handler_src.get_comm_mu_sp());
    dcomm_handler_data_tgt->comm_mu_sp = f_comm;
    f_comm = MPI_Comm_c2f(dcomm_handler_src.get_comm_phi_mu_sp());
    dcomm_handler_data_tgt->comm_phi_mu_sp = f_comm;

    dcomm_handler_data_tgt->n_dims        = dcomm_handler_src.get_n_dims();
    dcomm_handler_data_tgt->n_procs_total =
        dcomm_handler_src.get_n_procs_total();
    dcomm_handler_data_tgt->n_procs_RZ    = dcomm_handler_src.get_n_procs_RZ();
    dcomm_handler_data_tgt->n_procs_phi   = dcomm_handler_src.get_n_procs_phi();
    dcomm_handler_data_tgt->n_procs_vp    = dcomm_handler_src.get_n_procs_vp();
    dcomm_handler_data_tgt->n_procs_mu    = dcomm_handler_src.get_n_procs_mu();
    dcomm_handler_data_tgt->n_procs_sp    = dcomm_handler_src.get_n_procs_sp();
    dcomm_handler_data_tgt->rank          = dcomm_handler_src.get_rank();

    // Local assertion to check if MPI also works on the C++ layer
    bool err = false;
    int32_t istat, rank, n_procs_found, n_devices;

    istat = MPI_Comm_rank(comm_base, &rank);
    err = err || (istat != MPI_SUCCESS);

    istat = MPI_Comm_size(comm_base, &n_procs_found);
    err = err || (istat != MPI_SUCCESS);

    err = err || (dcomm_handler_src.get_rank() != rank);
    err = err || (dcomm_handler_src.get_n_procs_total() != n_procs_found);

    // Local assertion for some getters
    err = err || (dcomm_handler_src.get_comm_RZ_phi() !=
                  dcomm_handler_src.get_comm_phi());
    err = err || (dcomm_handler_src.is_master() != (rank == 0));

    switch (params_gpu_offload::get_backend())
    {
        case params_gpu_offload::backend_t::CPU:
            break;
#if defined ENABLE_OPENACC || defined ENABLE_OPENMPX
        case params_gpu_offload::backend_t::ACC:
        case params_gpu_offload::backend_t::OMPX:
            n_devices = dcomm_handler_src.get_n_devices();
            err = err || (dcomm_handler_src.get_gpu_rank() != rank%n_devices);
            break;
#endif
        default:
            err = true;
    }

    // Return 0 for success or 1 for error
    return (int32_t) err;
}

int32_t cbind_dcomm_handler_test_rank_config(
    const dcomm_handler_t** dcomm_handler_cxx_pptr)
{
    // Assign the source dcomm_handler_t class instance
    const dcomm_handler_t& dcomm_handler = *(*dcomm_handler_cxx_pptr);

    MPI_Comm comm;
    int32_t initial_value, expected_value;
    bool err = false;

    int32_t rank        = dcomm_handler.get_rank();
    int32_t n_procs     = dcomm_handler.get_n_procs_total();
    int32_t n_procs_phi = dcomm_handler.get_n_procs_phi();
    int32_t n_procs_vp  = dcomm_handler.get_n_procs_vp();
    int32_t n_procs_mu  = dcomm_handler.get_n_procs_mu();
    int32_t n_procs_sp  = dcomm_handler.get_n_procs_sp();

    // Test complete topology
    comm = dcomm_handler.get_comm_cart();
    initial_value  = rank;
    expected_value = n_procs * (n_procs - 1) / 2;
    err = err || is_communicator_erroneous(comm, initial_value, expected_value);

    // Test phi communicator
    comm = dcomm_handler.get_comm_phi();
    initial_value  = 2;
    expected_value = 2 * n_procs_phi;
    err = err || is_communicator_erroneous(comm, initial_value, expected_value);

    // Test vp communicator
    comm = dcomm_handler.get_comm_vp();
    initial_value  = 2;
    expected_value = 2 * n_procs_vp;
    err = err || is_communicator_erroneous(comm, initial_value, expected_value);

    // Test mu communicator
    comm = dcomm_handler.get_comm_mu();
    initial_value  = 2;
    expected_value = 2 * n_procs_mu;
    err = err || is_communicator_erroneous(comm, initial_value, expected_value);

    // Test sp communicator
    comm = dcomm_handler.get_comm_sp();
    initial_value  = 2;
    expected_value = 2 * n_procs_sp;
    err = err || is_communicator_erroneous(comm, initial_value, expected_value);

    // Test vp_mu communicator
    comm = dcomm_handler.get_comm_vp_mu();
    initial_value  = 2;
    expected_value = 2 * n_procs_vp * n_procs_mu;
    err = err || is_communicator_erroneous(comm, initial_value, expected_value);

    // Test vp_mu_sp communicator
    comm = dcomm_handler.get_comm_vp_mu_sp();
    initial_value  = 2;
    expected_value = 2 * n_procs_vp * n_procs_mu * n_procs_sp;
    err = err || is_communicator_erroneous(comm, initial_value, expected_value);

    // Test phi_vp_mu communicator
    comm = dcomm_handler.get_comm_phi_vp_mu();
    initial_value  = 2;
    expected_value = 2 * n_procs_phi * n_procs_vp * n_procs_mu;
    err = err || is_communicator_erroneous(comm, initial_value, expected_value);

    // Test vp_mu communicator
    comm = dcomm_handler.get_comm_mu_sp();
    initial_value = 2;
    expected_value = 2 * n_procs_mu * n_procs_sp;
    err = err || is_communicator_erroneous(comm, initial_value, expected_value);

    // Test vp_mu_sp communicator
    comm = dcomm_handler.get_comm_phi_mu_sp();
    initial_value = 2;
    expected_value = 2 * n_procs_phi * n_procs_mu * n_procs_sp;
    err = err || is_communicator_erroneous(comm, initial_value, expected_value);

    return (int32_t) err;
}

#ifdef __cplusplus
}
#endif

#endif
