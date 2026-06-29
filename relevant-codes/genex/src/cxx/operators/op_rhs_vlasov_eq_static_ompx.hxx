#ifndef OP_RHS_VLASOV_EQ_STATIC_OMPX_HXX
#define OP_RHS_VLASOV_EQ_STATIC_OMPX_HXX

#include "genex_cxx_env.hxx"
#include "op_rhs_vlasov_eq_static.hxx"
#include "profiler.hxx"
#include <omp.h>

// C++ class which corresponds to the Fortran class
// op_rhs_vlasov_eq_static_gpu_t with OpenMP offload
class op_rhs_vlasov_eq_static_ompx_t: public op_rhs_vlasov_eq_static_gpu_t
{
public:
    // Constructor of the OpenMP offload child class
    op_rhs_vlasov_eq_static_ompx_t(const mesh_5d_t& mesh)
    : op_rhs_vlasov_eq_static_gpu_t{mesh}
    {
        int32_t size_mesh = this->size_RZ * this->size_phi;
        // Allocate device pointer, map the host pointer to it and copy
        // the values
        #pragma omp target enter data \
            map(to: this[:1], \
                    this->size_RZ, \
                    this->size_phi, \
                    this->size_sp, \
                    this->prefac_arakawa, \
                    this->prefac_cd_xz, \
                    this->prefac_cd_vp, \
                    this->prefac_hyp_xz, \
                    this->prefac_hyp_vp, \
                    this->prefac_bps_ptr[:this->size_sp], \
                    this->prefac_orth_ptr[:this->size_sp], \
                    this->prefac_par_ptr[:this->size_sp], \
                    this->charges_ptr[:this->size_sp], \
                    this->masses_ptr[:this->size_sp], \
                    this->temp_scalings_ptr[:this->size_sp], \
                    this->prefac_flutter_xyz_ptr[:this->size_sp], \
                    this->prefac_flutter_vp_ptr[:this->size_sp], \
                    this->prefac_Bpar_ptr[:this->size_sp], \
                    this->prefac_buffer_zone_ptr[:size_mesh], \
                    this->prefac_cd_y_mm_ptr[:size_mesh], \
                    this->prefac_cd_y_m_ptr[:size_mesh], \
                    this->prefac_cd_y_o_ptr[:size_mesh], \
                    this->prefac_cd_y_p_ptr[:size_mesh], \
                    this->prefac_cd_y_pp_ptr[:size_mesh], \
                    this->prefac_hyp_y_mm_ptr[:size_mesh], \
                    this->prefac_hyp_y_m_ptr[:size_mesh], \
                    this->prefac_hyp_y_o_ptr[:size_mesh], \
                    this->prefac_hyp_y_p_ptr[:size_mesh], \
                    this->prefac_hyp_y_pp_ptr[:size_mesh])

        this->dmem_debug(dmd::mode_t::ALLOC);
    };

    // Destructor of the OpenMP offload child class
    ~op_rhs_vlasov_eq_static_ompx_t() override
    {
        int32_t size_mesh = this->size_RZ * this->size_phi;
        // Deallocate the device pointers of the class members from the device
        #pragma omp target exit data \
            map(delete: this->prefac_hyp_y_pp_ptr[:size_mesh], \
                        this->prefac_hyp_y_p_ptr[:size_mesh], \
                        this->prefac_hyp_y_o_ptr[:size_mesh], \
                        this->prefac_hyp_y_m_ptr[:size_mesh], \
                        this->prefac_hyp_y_mm_ptr[:size_mesh], \
                        this->prefac_cd_y_pp_ptr[:size_mesh], \
                        this->prefac_cd_y_p_ptr[:size_mesh], \
                        this->prefac_cd_y_o_ptr[:size_mesh], \
                        this->prefac_cd_y_m_ptr[:size_mesh], \
                        this->prefac_cd_y_mm_ptr[:size_mesh], \
                        this->prefac_buffer_zone_ptr[:size_mesh], \
                        this->prefac_Bpar_ptr[:this->size_sp], \
                        this->prefac_flutter_vp_ptr[:this->size_sp], \
                        this->prefac_flutter_xyz_ptr[:this->size_sp], \
                        this->temp_scalings_ptr[:this->size_sp], \
                        this->masses_ptr[:this->size_sp], \
                        this->charges_ptr[:this->size_sp], \
                        this->prefac_par_ptr[:this->size_sp], \
                        this->prefac_orth_ptr[:this->size_sp], \
                        this->prefac_bps_ptr[:this->size_sp], \
                        this->prefac_hyp_vp, \
                        this->prefac_hyp_xz, \
                        this->prefac_cd_vp, \
                        this->prefac_cd_xz, \
                        this->prefac_arakawa, \
                        this->size_sp, \
                        this->size_phi, \
                        this->size_RZ, \
                        this[:1])

        this->dmem_debug(dmd::mode_t::DEALLOC);
    };

    // Copy constructor is disabled
    op_rhs_vlasov_eq_static_ompx_t(const op_rhs_vlasov_eq_static_ompx_t&)
        = delete;

    // Copy-assignment operator is disabled
    op_rhs_vlasov_eq_static_ompx_t&
        operator=(const op_rhs_vlasov_eq_static_ompx_t&) = delete;

    // Apply method of the OpenMP offload child class
    int32_t apply(const mesh_5d_t& mesh,
                  const data_array_t<const real_t, 5>& f_in,
                  const data_array_t<const real_t, 2>& phi_in,
                  const data_array_t<const real_t, 2>& A_par_in,
                  const data_array_t<const real_t, 2>& B_par_in,
                  data_array_t<real_t, 5>& f_out) const override
    {
        const int32_t (&lb)[2] = phi_in.get_lbound();
        const int32_t (&ub)[2] = phi_in.get_ubound();
        const int32_t (&lb_stripped)[5] = f_in.get_lbound_stripped();
        const int32_t (&ub_stripped)[5] = f_in.get_ubound_stripped();

        // Instantiate local 2D array objects for second order Hamiltonian
        const std::array<int32_t, 2> lb_H2 {lb[0], lb[1]};
        const std::array<int32_t, 2> ub_H2 {ub[0], ub[1]};
        data_array_t<real_t, 2>& H2 =
            *(data_array::create<real_t, 2>(lb_H2, ub_H2));
        bool da_err = data_array::is_erroneous;

        #pragma omp target teams distribute parallel for simd collapse(2) \
            default(none) defaultmap(none) \
            shared(lb, ub, mesh, phi_in, H2)
        for (int32_t k = lb[1]; k <= ub[1]; k++)
        for (int32_t i = lb[0]; i <= ub[0]; i++)
        {
            this->initialize_H2(i, k, mesh, phi_in, H2);
        }

        #pragma omp target teams distribute parallel for simd collapse(5) \
            default(none) defaultmap(none) \
            shared(lb_stripped, ub_stripped, mesh, f_in, phi_in, A_par_in, \
                   B_par_in, H2, f_out)
        for (int32_t n = lb_stripped[4]; n <= ub_stripped[4]; n++)
        for (int32_t m = lb_stripped[3]; m <= ub_stripped[3]; m++)
        for (int32_t l = lb_stripped[2]; l <= ub_stripped[2]; l++)
        for (int32_t k = lb_stripped[1]; k <= ub_stripped[1]; k++)
        for (int32_t i = lb_stripped[0]; i <= ub_stripped[0]; i++)
        {
            this->comp_kernel(i, k, l, m, n,
                              mesh, f_in, phi_in, A_par_in, B_par_in, \
                              H2, f_out);
        }

        delete &H2;
        da_err = da_err || data_array::is_erroneous;
        return (int32_t) da_err;
    }
};

#endif
