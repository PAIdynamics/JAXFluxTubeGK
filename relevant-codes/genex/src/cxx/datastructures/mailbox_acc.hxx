#ifndef MAILBOX_ACC_HXX
#define MAILBOX_ACC_HXX

#include "genex_cxx_env.hxx"
#include "mailbox.hxx"
#include <openacc.h>

// C++ class which corresponds to the Fortran class mailbox_t with OpenACC
// NOTE: Partial allocation of an object to GPU memory leads to unstable
//       behaviour under the influence of compiler optimization. To avoid this,
//       mailbox object is not allocated on GPU memory. Instead, only inboxes
//       and outboxes arrays are present on GPU memory.
class mailbox_acc_t: public mailbox_t
{
private:
    // Update the inbox of given partner/neighbor index from CPU to GPU
    // via OpenACC
    // NOTE: This routine is only needed if non GPU-aware MPI is used
    inline void update_device_inbox(int32_t partner_index) override
    {
#ifndef ENABLE_COMM_MPI_GPU
        // Starting index of inbox array slice associated by the mailbox partner
        int32_t start_idx = (partner_index - 1) * this->n_mailbox_cells;
        #pragma acc update \
            device(this->inboxes_ptr[start_idx:this->n_mailbox_cells])
#endif
    }

public:
    // Parameterized constructor of the OpenACC child class
    mailbox_acc_t(const int32_t n_mailbox_cells,
                  const int32_t max_n_mail_partners)
    : mailbox_t{n_mailbox_cells, max_n_mail_partners} {}

    // Destructor of the OpenACC child class
    ~mailbox_acc_t() override
    {
        if (this->is_allocated_on_device) this->deallocate_device();
    }

    // Copy constructor is disabled
    mailbox_acc_t(const mailbox_t&) = delete;

    // Copy-assignment operator is disabled
    mailbox_acc_t& operator=(const mailbox_t&) = delete;

    // Routine to allocate all inboxes and outboxes on the device memory
    void allocate_device() override
    {
        int32_t mb_size = this->max_n_mail_partners * this->n_mailbox_cells;

        #pragma acc enter data create(this->outboxes_ptr[:mb_size], \
                                      this->inboxes_ptr[:mb_size])
        this->dmem_debug(dmd::mode_t::ALLOC);
    }

    // Routine to deallocate all inboxes and outboxes from the device memory
    void deallocate_device() override
    {
        int32_t mb_size = this->max_n_mail_partners * this->n_mailbox_cells;

        #pragma acc exit data delete(this->inboxes_ptr[:mb_size], \
                                     this->outboxes_ptr[:mb_size])
        this->dmem_debug(dmd::mode_t::DEALLOC);
    }

    // Pack the elements in the part of 2D data_array_t via OpenACC on GPU
    virtual void pack(const std::array<int32_t, 2>& lb_pack,
                      const std::array<int32_t, 2>& ub_pack,
                      const data_array_t<real_t, 2>& data,
                      const int32_t partner_index) override
    {
        const int32_t* lb = lb_pack.data();
        const int32_t* ub = ub_pack.data();
        const int32_t n_cells = this->n_mailbox_cells;
        real_t* outbox = this->outbox(partner_index);

        #pragma acc parallel default(none) \
            copyin(lb[:2], ub[:2]) present(outbox[:n_cells], data)
        #pragma acc loop independent gang vector collapse(2)
        for (int32_t k = lb[1]; k <= ub[1]; k++)
        for (int32_t i = lb[0]; i <= ub[0]; i++)
        {
            int32_t ctr = i - lb[0] + (k - lb[1]) * (ub[0] - lb[0] + 1);
            outbox[ctr] = data(i, k);
        }
    }

    // Pack the elements in the part of 4D data_array_t via OpenACC on GPU
    virtual void pack(const std::array<int32_t, 4>& lb_pack,
                      const std::array<int32_t, 4>& ub_pack,
                      const data_array_t<real_t, 4>& data,
                      const int32_t partner_index) override
    {
        const int32_t* lb = lb_pack.data();
        const int32_t* ub = ub_pack.data();
        const int32_t n_cells = this->n_mailbox_cells;
        real_t* outbox = this->outbox(partner_index);

        // Array containing memory strides of the loop bound arrays
        // calculated by the cumulative product w.r.t. each axes/dimension
        int32_t  strides[4];
        strides[0] = ub[0] - lb[0] + 1;
        for (int32_t d = 1; d < 4; d++)
        {
            strides[d] = strides[d-1] * (ub[d] - lb[d] + 1);
        }

        #pragma acc parallel default(none) \
            copyin(lb[:4], ub[:4], strides[:4]) present(outbox[:n_cells], data)
        #pragma acc loop independent gang vector collapse(4)
        for (int32_t o = lb[3]; o <= ub[3]; o++)
        for (int32_t m = lb[2]; m <= ub[2]; m++)
        for (int32_t k = lb[1]; k <= ub[1]; k++)
        for (int32_t i = lb[0]; i <= ub[0]; i++)
        {
            int32_t ctr = i - lb[0]
                        + (k - lb[1]) * strides[0]
                        + (m - lb[2]) * strides[1]
                        + (o - lb[3]) * strides[2];
            outbox[ctr] = data(i, k, m, o);
        }
    }

    // Pack the elements in the part of 5D data_array_t via OpenACC on GPU
    virtual void pack(const std::array<int32_t, 5>& lb_pack,
                      const std::array<int32_t, 5>& ub_pack,
                      const data_array_t<real_t, 5>& data,
                      const int32_t partner_index) override
    {
        const int32_t* lb = lb_pack.data();
        const int32_t* ub = ub_pack.data();
        const int32_t n_cells = this->n_mailbox_cells;
        real_t* outbox = this->outbox(partner_index);

        // Array containing memory strides of the loop bound arrays
        // calculated by the cumulative product w.r.t. each axes/dimension
        int32_t  strides[5];
        strides[0] = ub[0] - lb[0] + 1;
        for (int32_t d = 1; d < 5; d++)
        {
            strides[d] = strides[d-1] * (ub[d] - lb[d] + 1);
        }

        #pragma acc parallel default(none) \
            copyin(lb[:5], ub[:5], strides[:5]) present(outbox[:n_cells], data)
        #pragma acc loop independent gang vector collapse(5)
        for (int32_t n = lb[4]; n <= ub[4]; n++)
        for (int32_t m = lb[3]; m <= ub[3]; m++)
        for (int32_t l = lb[2]; l <= ub[2]; l++)
        for (int32_t k = lb[1]; k <= ub[1]; k++)
        for (int32_t i = lb[0]; i <= ub[0]; i++)
        {
            int32_t ctr = i - lb[0]
                        + (k - lb[1]) * strides[0]
                        + (l - lb[2]) * strides[1]
                        + (m - lb[3]) * strides[2]
                        + (n - lb[4]) * strides[3];
            outbox[ctr] = data(i, k, l, m, n);
        }
    }

    // Unpack the elements in the part of 2D data_array_t via OpenACC on GPU
    virtual void unpack(const std::array<int32_t, 2>& lb_pack,
                        const std::array<int32_t, 2>& ub_pack,
                        data_array_t<real_t, 2>& data,
                        const int32_t partner_index) override
    {
        const int32_t* lb = lb_pack.data();
        const int32_t* ub = ub_pack.data();
        const int32_t n_cells = this->n_mailbox_cells;
        const real_t* inbox = this->inbox(partner_index);

        #pragma acc parallel default(none) \
            copyin(lb[:2], ub[:2]) present(inbox[:n_cells], data)
        #pragma acc loop independent gang vector collapse(2)
        for (int32_t k = lb[1]; k <= ub[1]; k++)
        for (int32_t i = lb[0]; i <= ub[0]; i++)
        {
            int32_t ctr = i - lb[0] + (k - lb[1]) * (ub[0] - lb[0] + 1);
            data(i, k) = inbox[ctr];
        }
    }

    // Unpack the elements in the part of 4D data_array_t via OpenACC on GPU
    virtual void unpack(const std::array<int32_t, 4>& lb_pack,
                        const std::array<int32_t, 4>& ub_pack,
                        data_array_t<real_t, 4>& data,
                        const int32_t partner_index) override
    {
        const int32_t* lb = lb_pack.data();
        const int32_t* ub = ub_pack.data();
        const int32_t n_cells = this->n_mailbox_cells;
        const real_t* inbox = this->inbox(partner_index);

        // Array containing memory strides of the loop bound arrays
        // calculated by the cumulative product w.r.t. each axes/dimension
        int32_t  strides[4];
        strides[0] = ub[0] - lb[0] + 1;
        for (int32_t d = 1; d < 4; d++)
        {
            strides[d] = strides[d-1] * (ub[d] - lb[d] + 1);
        }

        #pragma acc parallel default(none) \
            copyin(lb[:4], ub[:4], strides[:4]) present(inbox[:n_cells], data)
        #pragma acc loop independent gang vector collapse(4)
        for (int32_t o = lb[3]; o <= ub[3]; o++)
        for (int32_t m = lb[2]; m <= ub[2]; m++)
        for (int32_t k = lb[1]; k <= ub[1]; k++)
        for (int32_t i = lb[0]; i <= ub[0]; i++)
        {
            int32_t ctr = i - lb[0]
                        + (k - lb[1]) * strides[0]
                        + (m - lb[2]) * strides[1]
                        + (o - lb[3]) * strides[2];
            data(i, k, m, o) = inbox[ctr];
        }
    }

    // Unpack the elements in the part of 5D data_array_t via OpenACC on GPU
    virtual void unpack(const std::array<int32_t, 5>& lb_pack,
                        const std::array<int32_t, 5>& ub_pack,
                        data_array_t<real_t, 5>& data,
                        const int32_t partner_index) override
    {
        const int32_t* lb = lb_pack.data();
        const int32_t* ub = ub_pack.data();
        const int32_t n_cells = this->n_mailbox_cells;
        const real_t* inbox = this->inbox(partner_index);

        // Array containing memory strides of the loop bound arrays
        // calculated by the cumulative product w.r.t. each axes/dimension
        int32_t  strides[5];
        strides[0] = ub[0] - lb[0] + 1;
        for (int32_t d = 1; d < 5; d++)
        {
            strides[d] = strides[d-1] * (ub[d] - lb[d] + 1);
        }

        #pragma acc parallel default(none) \
            copyin(lb[:5], ub[:5], strides[:5]) present(inbox[:n_cells], data)
        #pragma acc loop independent gang vector collapse(5)
        for (int32_t n = lb[4]; n <= ub[4]; n++)
        for (int32_t m = lb[3]; m <= ub[3]; m++)
        for (int32_t l = lb[2]; l <= ub[2]; l++)
        for (int32_t k = lb[1]; k <= ub[1]; k++)
        for (int32_t i = lb[0]; i <= ub[0]; i++)
        {
            int32_t ctr = i - lb[0]
                        + (k - lb[1]) * strides[0]
                        + (l - lb[2]) * strides[1]
                        + (m - lb[3]) * strides[2]
                        + (n - lb[4]) * strides[3];
            data(i, k, l, m, n) = inbox[ctr];
        }
    }
};

#endif
