#ifndef HELPERS_MAIL_DELIVERY_CXX
#define HELPERS_MAIL_DELIVERY_CXX

#include "genex_cxx_env.hxx"
#include "params_gpu_offload.hxx"
#include "dcomm_handler.hxx"
#include "mailbox.hxx"
#include "mailbox_factory.hxx"
#include <mpi.h>
#include <math.h>

#ifdef ENABLE_OPENACC
#include <openacc.h>
#endif

// OpenMP routine to test the mail delivery over the selected dimension
// with the specified number of neighbors/mail partners
// Return 0 for success and 1 for error
int32_t test_dimension_omp(const dcomm_handler_t& dcomm_handler,
                           const int32_t dim_test,
                           const int32_t number_of_neighbors)
{
    int32_t ierr = 0;
    const int32_t n_cells = 4;
    const int32_t max_n_neighbors = 4;
    mailbox_t& mailbox = *(mailbox::create(n_cells, max_n_neighbors));
    bool err_mb = mailbox::is_erroneous;

    // Initialize outboxes
    for (int32_t j = 1; j <= mailbox.get_n_max_partners(); j++)
    {
        real_t* outbox = mailbox.outbox(j);

        #pragma omp parallel default(none) firstprivate(j, n_cells) \
            shared(outbox)
        #pragma omp for simd schedule(static)
        for (int32_t i = 1; i <= n_cells; i++)
        {
                outbox[i-1] = i * j;
        }
    }

    mailbox.deliver_outboxes(dcomm_handler, dim_test, number_of_neighbors);

    mailbox.finish_delivery(number_of_neighbors);

    // Calculate the difference between inboxes and outboxes and assert
    for (int32_t j = 1; j <= number_of_neighbors; j++)
    {
        const real_t* inbox = mailbox.inbox(j);
        const real_t* outbox = mailbox.outbox(number_of_neighbors - j + 1);

        for (int32_t i = 1; i <= n_cells; i++)
        {
            // NOTE: It is possible to compare real types without machine
            //       epsilon in this case because they are copy to each other
            //       and initially defined from integer type MPI rank.
            //       This applies to the GPU tests below as well.
            bool err = inbox[i-1] - outbox[i-1];
            if ((mailbox.get_partners(j) != MPI_PROC_NULL) && err) ierr = 1;
        }
    }
    delete &mailbox;
    err_mb = mailbox::is_erroneous;
    return (ierr != 0) || err_mb;
}

#ifdef ENABLE_OPENACC
// OpenACC routine to test the mail delivery over the selected dimension
// with the specified number of neighbors/mail partners
// Return 0 for success and 1 for error
int32_t test_dimension_acc(const dcomm_handler_t& dcomm_handler,
                           const int32_t dim_test,
                           const int32_t number_of_neighbors)
{
    int32_t ierr = 0;
    const int32_t n_cells = 4;
    const int32_t max_n_neighbors = 4;
    real_t diff[max_n_neighbors][n_cells] {};
    mailbox_t& mailbox = *(mailbox::create(n_cells, max_n_neighbors));
    bool err_dev = mailbox::is_erroneous;

    #pragma acc enter data copyin(n_cells) \
        create(diff[:max_n_neighbors][:n_cells])
    mailbox.allocate_device();

    // Initialize outboxes
    for (int32_t j = 1; j <= mailbox.get_n_max_partners(); j++)
    {
        real_t* outbox = mailbox.outbox(j);

        #pragma acc parallel default(none) firstprivate(j) \
            present(n_cells, outbox)
        {
            #pragma acc loop independent
            for (int32_t i = 1; i <= n_cells; i++)
            {
                outbox[i-1] = i * j;
            }
        }
    }

    mailbox.deliver_outboxes(dcomm_handler, dim_test, number_of_neighbors);

    mailbox.finish_delivery(number_of_neighbors);

    // Calculate the difference between inboxes and outboxes
    for (int32_t j = 1; j <= number_of_neighbors; j++)
    {
        const real_t* inbox = mailbox.inbox(j);
        const real_t* outbox = mailbox.outbox(number_of_neighbors - j + 1);

        #pragma acc parallel default(none) firstprivate(j) \
            present(n_cells, inbox, outbox, diff)
        {
            #pragma acc loop independent
            for (int32_t i = 1; i <= n_cells; i++)
            {
                diff[j-1][i-1] = inbox[i-1] - outbox[i-1];
            }
        }
    }

    mailbox.deallocate_device();
    #pragma acc exit data copyout(n_cells, diff[:max_n_neighbors][:n_cells])

    // Assertion
    for (int32_t j = 1; j <= number_of_neighbors; j++)
    for (int32_t i = 1; i <= n_cells; i++)
    {
        bool err = diff[j-1][i-1] != 0.0;
        if ((mailbox.get_partners(j) != MPI_PROC_NULL) && err) ierr = 1;
    }

    delete &mailbox;
    err_dev = err_dev || mailbox::is_erroneous;
    return (int32_t) (ierr != 0) || err_dev;
}
#endif

#ifdef ENABLE_OPENMPX
// OpenMP offload routine to test the mail delivery over the selected dimension
// with the specified number of neighbors/mail partners
// Return 0 for success and 1 for error
int32_t test_dimension_ompx(const dcomm_handler_t& dcomm_handler,
                            const int32_t dim_test,
                            const int32_t number_of_neighbors)
{
    int32_t ierr = 0;
    const int32_t n_cells = 4;
    const int32_t max_n_neighbors = 4;
    real_t diff[max_n_neighbors][n_cells] {};
    mailbox_t& mailbox = *(mailbox::create(n_cells, max_n_neighbors));
    bool err_dev = mailbox::is_erroneous;

    #pragma omp target enter data map(to: n_cells) \
        map(alloc: diff[:max_n_neighbors][:n_cells])
    mailbox.allocate_device();

    // Initialize outboxes
    for (int32_t j = 1; j <= mailbox.get_n_max_partners(); j++)
    {
        real_t* outbox = mailbox.outbox(j);

        #pragma omp target teams default(none) defaultmap(none) \
            firstprivate(j) shared(n_cells, outbox)
        {
            #pragma omp distribute parallel for simd
            for (int32_t i = 1; i <= n_cells; i++)
            {
                outbox[i-1] = i * j;
            }
        }
    }

    mailbox.deliver_outboxes(dcomm_handler, dim_test, number_of_neighbors);

    mailbox.finish_delivery(number_of_neighbors);

    // Calculate the difference between inboxes and outboxes
    for (int32_t j = 1; j <= number_of_neighbors; j++)
    {
        const real_t* inbox = mailbox.inbox(j);
        const real_t* outbox = mailbox.outbox(number_of_neighbors - j + 1);

        #pragma omp target teams default(none) defaultmap(none) \
            firstprivate(j) shared(n_cells, inbox, outbox, diff)
        {
            #pragma omp distribute parallel for simd
            for (int32_t i = 1; i <= n_cells; i++)
            {
                diff[j-1][i-1] = inbox[i-1] - outbox[i-1];
            }
        }
    }

    mailbox.deallocate_device();
    #pragma omp target exit data map(from: n_cells, \
                                           diff[:max_n_neighbors][:n_cells])

    for (int32_t j = 1; j <= number_of_neighbors; j++)
    for (int32_t i = 1; i <= mailbox.get_n_cells(); i++)
    {
        bool err = diff[j-1][i-1] != 0.0;
        if ((mailbox.get_partners(j) != MPI_PROC_NULL) && err) ierr = 1;
    }

    delete &mailbox;
    err_dev = mailbox::is_erroneous;
    return (int32_t) (ierr != 0) || err_dev;
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Interoperable routine to test the mail delivery over the selected dimension
// with the specified number of neighbors/mail partners
// Return 0 for success and 1 for error
int32_t cbind_test_dimension(const dcomm_handler_t** dcomm_handler_cxx_pptr,
                             const int32_t dim_test,
                             const int32_t number_of_neighbors)
{
    // Assign the C++ class instances
    const dcomm_handler_t& dcomm_handler = *(*dcomm_handler_cxx_pptr);

    // Return 0 for success and 1 for error
    switch (params_gpu_offload::get_backend())
    {
        case params_gpu_offload::backend_t::CPU:
            return test_dimension_omp(dcomm_handler, dim_test,
                                      number_of_neighbors);
#ifdef ENABLE_OPENACC
        case params_gpu_offload::backend_t::ACC:
            return test_dimension_acc(dcomm_handler, dim_test,
                                      number_of_neighbors);
#endif
#ifdef ENABLE_OPENMPX
        case params_gpu_offload::backend_t::OMPX:
            return test_dimension_ompx(dcomm_handler, dim_test,
                                       number_of_neighbors);
#endif
        default:
            return 1;
    }
}

#ifdef __cplusplus
}
#endif

#endif
