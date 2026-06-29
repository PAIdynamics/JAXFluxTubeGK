#ifndef MAILBOX_HXX
#define MAILBOX_HXX

#include "genex_cxx_env.hxx"
#include "device_memory_debugger.hxx"
#include "dcomm_handler.hxx"
#include "data_array.hxx"
#include <array>
#include <mpi.h>

// Alias for external namespace
namespace dmd = device_memory_debugger;

// Error flag for mailbox_t
namespace mailbox
{
    inline bool is_erroneous = false;
}

// Class containing the buffer data that is sent and received via
// MPI communication and information about the mail partners and
// the communication requests
class mailbox_t
{
protected:
    // Boolean flag indicating if all inboxes and outboxes are currently
    // allocated on the device memory
    bool is_allocated_on_device = false;
    // Number of cells of the buffer array in the mailbox
    int32_t n_mailbox_cells;
    // Maximum number of mail partners or neighbors to exchange data in mailbox
    int32_t max_n_mail_partners;
    // Pointer to buffer array for data received from MPI communication
    real_t* inboxes_ptr;
    // Pointer to buffer array for data sent to MPI communication
    real_t* outboxes_ptr;

    // Update the inbox of given partner/neighbor index from CPU to GPU
    virtual inline void update_device_inbox(int32_t partner_index) = 0;

    // Check if the device memory is set and used as intended
    inline void dmem_debug(const dmd::mode_t mode)
    {
        bool err = false;
        int32_t mb_size = this->max_n_mail_partners * this->n_mailbox_cells;

        if (mb_size > 0)
        {
            dmd::start_region("mailbox_t", mode);
            err = err || dmd::is_invalid(this->inboxes_ptr, mb_size);
            err = err || dmd::is_invalid(this->outboxes_ptr, mb_size);
            dmd::end_region("mailbox_t");

            mailbox::is_erroneous = mailbox::is_erroneous || err;
        }

        if (err == false)
        {
            if (mode == dmd::mode_t::ALLOC)
            {
                this->is_allocated_on_device = true;
            }
            else
            {
                this->is_allocated_on_device = false;
            }
        }
    }

private:
    // Pointer to the MPI rank array of mail partners/neighbors to exchange data
    int32_t* partners_ptr;
    // Pointer to communication requests array from MPI_Irecv for each partners
    MPI_Request* requests_in_ptr;
    // Pointer of communication requests array from MPI_Isend for each partners
    MPI_Request* requests_out_ptr;

public:
    // Parameterized constructor
    mailbox_t(const int32_t n_mailbox_cells, const int32_t max_n_mail_partners)
    {
        int32_t mb_size = max_n_mail_partners * n_mailbox_cells;
        this->n_mailbox_cells = n_mailbox_cells;
        this->max_n_mail_partners = max_n_mail_partners;
        this->inboxes_ptr = new real_t[mb_size] {};
        this->outboxes_ptr = new real_t[mb_size] {};
        this->partners_ptr = new int32_t[this->max_n_mail_partners] {};
        this->requests_in_ptr = new MPI_Request[this->max_n_mail_partners] {};
        this->requests_out_ptr = new MPI_Request[this->max_n_mail_partners] {};
    }

    // Destructor
    virtual ~mailbox_t()
    {
        delete[] this->requests_out_ptr;
        delete[] this->requests_in_ptr;
        delete[] this->partners_ptr;
        delete[] this->outboxes_ptr;
        delete[] this->inboxes_ptr;
    }

    // Copy constructor is disabled
    mailbox_t(const mailbox_t&) = delete;

    // Copy-assignment operator is disabled
    mailbox_t& operator=(const mailbox_t&) = delete;

    // Pure virtual routine to allocate all inboxes and outboxes
    //  on the device memory
    virtual void allocate_device() = 0;

    // Pure virtual routine to deallocate all inboxes and outboxes
    //  from the device memory
    virtual void deallocate_device() = 0;

    // Exchange the information stored in outboxes over the
    // selected dimension with the specified number of neighbors/mail partners
    void deliver_outboxes(const dcomm_handler_t& dcomm_handler,
                          const int32_t exchange_dimension,
                          const int32_t number_of_mail_partners)
    {
        MPI_Comm comm;
        int32_t ierr;

        // Select the communicator
        switch (exchange_dimension)
        {
        case 2:
            comm = dcomm_handler.get_comm_phi();
            break;
        case 3:
            comm = dcomm_handler.get_comm_vp();
            break;
        case 4:
            comm = dcomm_handler.get_comm_mu();
            break;
        default:
            std::cerr << "Error: Boundary exchange over the selected dimension "
                      << "index " << exchange_dimension << " is not supported!"
                      << std::endl;
            break;
        }

        // Find the mail partner / neighbor ranks and the number of neighbors
        if (number_of_mail_partners == 2)
        {
            ierr = MPI_Cart_shift(comm, 0, 1, &this->partners_ptr[0],
                                  &this->partners_ptr[1]);
        }
        else if (number_of_mail_partners == 4)
        {
            ierr = MPI_Cart_shift(comm, 0, 1, &this->partners_ptr[1],
                                  &this->partners_ptr[2]);

            ierr = MPI_Cart_shift(comm, 0, 2, &this->partners_ptr[0],
                                  &this->partners_ptr[3]);
        }
        else
        {
            std::cerr << "Error: number_of_mail_partners has "
                      << "to be equal to 2 or 4!"
                      << std::endl;
        }

        for (int32_t idx = 0; idx < number_of_mail_partners; idx++)
        {
            // Initialize request arrays
            this->requests_in_ptr[idx]  = MPI_REQUEST_NULL;
            this->requests_out_ptr[idx] = MPI_REQUEST_NULL;

            // Send to existing mail partners / neighbors
            if (this->partners_ptr[idx] != MPI_PROC_NULL)
            {
                ierr = dcomm_handler.Isend(this->outbox(idx + 1),
                                           this->n_mailbox_cells, MPI_REAL_T,
                                           this->partners_ptr[idx], idx, comm,
                                           &this->requests_out_ptr[idx]);
            }
        }

        for (int32_t idx = 0; idx < number_of_mail_partners; idx++)
        {
            int32_t recv_idx = number_of_mail_partners - idx - 1;
            // Receive from existing mail partners / neighbors
            if (this->partners_ptr[recv_idx] != MPI_PROC_NULL)
            {
                ierr = dcomm_handler.Irecv(this->inbox(recv_idx + 1),
                                           this->n_mailbox_cells, MPI_REAL_T,
                                           this->partners_ptr[recv_idx],
                                           idx, comm,
                                           &this->requests_in_ptr[recv_idx]);
            }
        }
    }

    // Finish the exchange of information stored in the outboxes
    void finish_delivery(const int32_t number_of_mail_partners)
    {
        int32_t ierr;

        for (int32_t idx = 0; idx < number_of_mail_partners; idx++)
        {
            int32_t recv_idx = number_of_mail_partners - idx - 1;
            // Check whether the neighbor we receive from exists
            if (this->partners_ptr[recv_idx] != MPI_PROC_NULL)
            {
                ierr = MPI_Wait(&this->requests_in_ptr[recv_idx],
                                MPI_STATUS_IGNORE);
                // Update inboxes in GPU after all MPI_Irecv is over
                this->update_device_inbox(recv_idx + 1);
            }
            // Check whether the neighbor we might have send to exists
            if (this->partners_ptr[idx] != MPI_PROC_NULL)
            {
                ierr = MPI_Wait(&this->requests_out_ptr[idx],
                                MPI_STATUS_IGNORE);
            }
        }
    }

    // Getter for the pointer of the inbox array at given mail partner index
    inline real_t* inbox(int32_t partner_index)
    {
        return this->inboxes_ptr + (partner_index - 1) * this->n_mailbox_cells;
    }

    // Getter for the pointer of the outbox array at given mail partner index
    inline real_t* outbox(int32_t partner_index)
    {
        return this->outboxes_ptr + (partner_index - 1) * this->n_mailbox_cells;
    }

    // Getter for the cell number of the mailbox
    inline int32_t get_n_cells() const
    {
        return this->n_mailbox_cells;
    }

    // Getter for the maximum number of mail partners
    inline int32_t get_n_max_partners() const
    {
        return this->max_n_mail_partners;
    }

    // Getter for the array specifying the partners
    inline int32_t get_partners(int32_t j0) const
    {
        return this->partners_ptr[j0 - 1];
    }

    // Pure virtual method to pack the elements in the part of 2D data_array_t
    virtual void pack(const std::array<int32_t, 2>& lb_pack,
                      const std::array<int32_t, 2>& ub_pack,
                      const data_array_t<real_t, 2>& data,
                      const int32_t partner_index) = 0;

    // Pure virtual method to pack the elements in the part of 4D data_array_t
    virtual void pack(const std::array<int32_t, 4>& lb_pack,
                      const std::array<int32_t, 4>& ub_pack,
                      const data_array_t<real_t, 4>& data,
                      const int32_t partner_index) = 0;

    // Pure virtual method to pack the elements in the part of 5D data_array_t
    virtual void pack(const std::array<int32_t, 5>& lb_pack,
                      const std::array<int32_t, 5>& ub_pack,
                      const data_array_t<real_t, 5>& data,
                      const int32_t partner_index) = 0;

    // Pure virtual method to unpack the elements in the part of 2D data_array_t
    virtual void unpack(const std::array<int32_t, 2>& lb_pack,
                        const std::array<int32_t, 2>& ub_pack,
                        data_array_t<real_t, 2>& data,
                        const int32_t partner_index) = 0;

    // Pure virtual method to unpack the elements in the part of 4D data_array_t
    virtual void unpack(const std::array<int32_t, 4>& lb_pack,
                        const std::array<int32_t, 4>& ub_pack,
                        data_array_t<real_t, 4>& data,
                        const int32_t partner_index) = 0;

    // Pure virtual method to unpack the elements in the part of 2D data_array_t
    virtual void unpack(const std::array<int32_t, 5>& lb_pack,
                        const std::array<int32_t, 5>& ub_pack,
                        data_array_t<real_t, 5>& data,
                        const int32_t partner_index) = 0;
};

#endif
