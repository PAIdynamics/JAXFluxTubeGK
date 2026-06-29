#include "op_rhs_vlasov_eq_static.hxx"
#include "op_rhs_vlasov_eq_static_factory.hxx"
#include "lagrange_interpolator.hxx"
#include "params_species.hxx"
#include "params_normalization.hxx"
#include "params_numerical_scheme.hxx"
#include "params_gyrokinetic_system.hxx"
#include <cmath>

// Alias for parameter namespaces
namespace pspec = params_species;
namespace pnorm = params_normalization;
namespace pnums = params_numerical_scheme;
namespace pgyro = params_gyrokinetic_system;

int32_t cbind_op_vlasov_static_initialize(
    const mesh_5d_t** mesh_cxx_pptr,
    op_rhs_vlasov_eq_static_gpu_t** op_cxx_pptr)
{
    // Assign the mesh_5d_t class instance
    const mesh_5d_t& mesh = *(*mesh_cxx_pptr);

    // Allocate and construct the C++ class instance of the operator
    *op_cxx_pptr = op_rhs_vlasov_eq_static_gpu::create(mesh);

    return (int32_t) op_rhs_vlasov_eq_static_gpu::is_erroneous;
}

int32_t cbind_op_vlasov_static_finalize(
    op_rhs_vlasov_eq_static_gpu_t** op_cxx_pptr)
{
    // Assign the static operator C++ class instance
    op_rhs_vlasov_eq_static_gpu_t& op = *(*op_cxx_pptr);

    // Deallocate the host static operator class instance
    delete &op;

    return (int32_t) op_rhs_vlasov_eq_static_gpu::is_erroneous;
}

int32_t cbind_op_vlasov_static_apply(
    const op_rhs_vlasov_eq_static_gpu_t** op_cxx_pptr,
    const mesh_5d_t** mesh_cxx_pptr,
    const data_array_t<const real_t, 5>** f_in_cxx_pptr,
    const data_array_t<const real_t, 2>** phi_in_cxx_pptr,
    const data_array_t<const real_t, 2>** A_par_in_cxx_pptr,
    const data_array_t<const real_t, 2>** B_par_in_cxx_pptr,
    data_array_t<real_t, 5>** f_out_cxx_pptr)
{
    // Assign the C++ class instances
    const mesh_5d_t& mesh = *(*mesh_cxx_pptr);
    const op_rhs_vlasov_eq_static_gpu_t& op = *(*op_cxx_pptr);
    const data_array_t<const real_t, 5>& f_in = *(*f_in_cxx_pptr);
    const data_array_t<const real_t, 2>& phi_in = *(*phi_in_cxx_pptr);
    const data_array_t<const real_t, 2>& A_par_in = *(*A_par_in_cxx_pptr);
    const data_array_t<const real_t, 2>& B_par_in = *(*B_par_in_cxx_pptr);
    data_array_t<real_t, 5>& f_out = *(*f_out_cxx_pptr);

    return op.apply(mesh, f_in, phi_in, A_par_in, B_par_in, f_out);
}

op_rhs_vlasov_eq_static_gpu_t::op_rhs_vlasov_eq_static_gpu_t(
    const mesh_5d_t& mesh)
{
    this->size_RZ  = mesh.get_size_RZ();
    this->size_phi = mesh.get_size_phi();
    this->size_sp  = mesh.get_size_sp();

    // Initialize prefactors for the x, z derivatives
    this->prefac_arakawa = 1.0
                         / (12.0 * mesh.get_delta_RZ() * mesh.get_delta_RZ());
    this->prefac_cd_xz   = 1.0 / mesh.get_delta_RZ();
    this->prefac_hyp_xz  = pnums::get_hyp_xz();

    // Initialize prefactors for the vp derivatives
    this->prefac_cd_vp  = 1.0 / mesh.get_delta_vp();
    this->prefac_hyp_vp = pnums::get_hyp_vp();

    // Initialize prefactors for the physical terms
    this->prefac_H2 = 0.0;
    if(pgyro::get_with_nlin_polarization())
    {
        this->prefac_H2 = pnorm::get_rho_ref() * pnorm::get_rho_ref()
                        / (pnorm::get_L_ref() * pnorm::get_L_ref());
    }

    // Allocate short arrays in class members
    this->prefac_bps_ptr         = new real_t[this->size_sp];
    this->prefac_orth_ptr        = new real_t[this->size_sp];
    this->prefac_par_ptr         = new real_t[this->size_sp];
    this->charges_ptr            = new real_t[this->size_sp];
    this->masses_ptr             = new real_t[this->size_sp];
    this->temp_scalings_ptr      = new real_t[this->size_sp];
    this->prefac_flutter_xyz_ptr = new real_t[this->size_sp];
    this->prefac_flutter_vp_ptr  = new real_t[this->size_sp];
    this->prefac_Bpar_ptr = new real_t[this->size_sp];

    // Allocate large arrays in class members
    int32_t size_mesh = this->size_RZ * this->size_phi;
    if(pgpus::get_use_array_alignment())
    {
        std::align_val_t align = pgpus::get_array_alignment();
        this->prefac_buffer_zone_ptr = new (align) real_t[size_mesh]{};
        this->prefac_cd_y_mm_ptr     = new (align) real_t[size_mesh]{};
        this->prefac_cd_y_m_ptr      = new (align) real_t[size_mesh]{};
        this->prefac_cd_y_o_ptr      = new (align) real_t[size_mesh]{};
        this->prefac_cd_y_p_ptr      = new (align) real_t[size_mesh]{};
        this->prefac_cd_y_pp_ptr     = new (align) real_t[size_mesh]{};
        this->prefac_hyp_y_mm_ptr    = new (align) real_t[size_mesh]{};
        this->prefac_hyp_y_m_ptr     = new (align) real_t[size_mesh]{};
        this->prefac_hyp_y_o_ptr     = new (align) real_t[size_mesh]{};
        this->prefac_hyp_y_p_ptr     = new (align) real_t[size_mesh]{};
        this->prefac_hyp_y_pp_ptr    = new (align) real_t[size_mesh]{};
    }
    else
    {
        this->prefac_buffer_zone_ptr = new real_t[size_mesh]{};
        this->prefac_cd_y_mm_ptr     = new real_t[size_mesh]{};
        this->prefac_cd_y_m_ptr      = new real_t[size_mesh]{};
        this->prefac_cd_y_o_ptr      = new real_t[size_mesh]{};
        this->prefac_cd_y_p_ptr      = new real_t[size_mesh]{};
        this->prefac_cd_y_pp_ptr     = new real_t[size_mesh]{};
        this->prefac_hyp_y_mm_ptr    = new real_t[size_mesh]{};
        this->prefac_hyp_y_m_ptr     = new real_t[size_mesh]{};
        this->prefac_hyp_y_o_ptr     = new real_t[size_mesh]{};
        this->prefac_hyp_y_p_ptr     = new real_t[size_mesh]{};
        this->prefac_hyp_y_pp_ptr    = new real_t[size_mesh]{};
    }

    for (int32_t n = 1; n <= this->size_sp; n++)
    {
        real_t prefac_vth = sqrt(2.0 * pspec::get_temp_scaling(n)
                                     / pspec::get_mass(n));
        real_t prefac_vth_mass = prefac_vth * pspec::get_mass(n);
        this->charges_ptr[n - 1] = pspec::get_charge(n);
        this->masses_ptr[n - 1]  = pspec::get_mass(n);
        this->temp_scalings_ptr[n - 1]  = pspec::get_temp_scaling(n);
        this->prefac_bps_ptr[n - 1] = prefac_vth_mass
            * pnorm::get_rho_ref()
            / (pspec::get_charge(n) * pnorm::get_L_ref());
        this->prefac_orth_ptr[n - 1] = pnorm::get_rho_ref()
            / (pspec::get_charge(n) * pnorm::get_L_ref());
        this->prefac_par_ptr[n - 1] = 1.0 / prefac_vth_mass;
        if(pgyro::get_with_em_flutter())
        {
            this->prefac_flutter_xyz_ptr[n - 1] = prefac_vth
                * pspec::get_charge(n);
            this->prefac_flutter_vp_ptr[n - 1] = pnorm::get_rho_ref()
                / pnorm::get_L_ref() / prefac_vth_mass;
        }
        else
        {
            this->prefac_flutter_xyz_ptr[n - 1] = 0.0;
            this->prefac_flutter_vp_ptr[n - 1]  = 0.0;
        }
        if(pgyro::get_with_bpar())
        {
            this->prefac_Bpar_ptr[n - 1] = pspec::get_temp_scaling(n);
        }
        else
        {
            this->prefac_Bpar_ptr[n - 1] = 0.0;
        }
    }

    #pragma omp parallel default(none) shared(mesh)
    {
        for (int32_t k = 1; k <= this->size_phi; k++)
        #pragma omp for schedule(static)
        for (int32_t i = 1; i <= this->size_RZ; i++)
        {
            int32_t idx = (i - 1) + (k - 1) * mesh.get_size_RZ();

            if(mesh.buf_zone(i, k) == mesh_5d::BUF_ZONE_BOUNDARY_IN ||
               mesh.buf_zone(i, k) == mesh_5d::BUF_ZONE_BOUNDARY_OUT ||
               mesh.buf_zone(i, k) == mesh_5d::BUF_ZONE_PARCON)
            {
                this->prefac_buffer_zone_ptr[idx] =
                    pnums::get_buf_zone_strength();
            }
            else if(mesh.buf_zone(i, k) == mesh_5d::BUF_ZONE_AXIS)
            {
                this->prefac_buffer_zone_ptr[idx] =
                    pnums::get_buf_zone_strength_axis();
            }
            else
            {
                this->prefac_buffer_zone_ptr[idx] = 0.0;
            }
        }
    }

    #pragma omp parallel default(none) shared(mesh)
    {
        for (int32_t k = 1; k <= this->size_phi; k++)
        #pragma omp for schedule(static)
        for (int32_t i = 1; i <= this->size_RZ; i++)
        {
            int32_t idx = (i - 1) + (k - 1) * mesh.get_size_RZ();

            if(mesh.is_compute(i, k) == 1.0)
            {
                auto [mon1_mm, mon1_m, mon1_o, mon1_p, mon1_pp] =
                lagrange_interpolator::monomial_1st_derivative(
                    -mesh.fll_negative2(i, k), -mesh.fll_negative1(i, k), 0.0,
                     mesh.fll_positive1(i, k),  mesh.fll_positive2(i, k));
                this->prefac_cd_y_mm_ptr[idx] = mon1_mm;
                this->prefac_cd_y_m_ptr[idx]  = mon1_m;
                this->prefac_cd_y_o_ptr[idx]  = mon1_o;
                this->prefac_cd_y_p_ptr[idx]  = mon1_p;
                this->prefac_cd_y_pp_ptr[idx] = mon1_pp;

                auto[mon4_mm, mon4_m, mon4_o, mon4_p, mon4_pp] =
                lagrange_interpolator::monomial_4th_derivative(
                    -mesh.fll_negative2(i, k), -mesh.fll_negative1(i, k), 0.0,
                     mesh.fll_positive1(i, k),  mesh.fll_positive2(i, k));
                this->prefac_hyp_y_mm_ptr[idx] = pnums::get_hyp_y() * mon4_mm;
                this->prefac_hyp_y_m_ptr[idx]  = pnums::get_hyp_y() * mon4_m;
                this->prefac_hyp_y_o_ptr[idx]  = pnums::get_hyp_y() * mon4_o;
                this->prefac_hyp_y_p_ptr[idx]  = pnums::get_hyp_y() * mon4_p;
                this->prefac_hyp_y_pp_ptr[idx] = pnums::get_hyp_y() * mon4_pp;
            }
            else
            {
                this->prefac_cd_y_mm_ptr[idx] = 0.0;
                this->prefac_cd_y_m_ptr[idx]  = 0.0;
                this->prefac_cd_y_o_ptr[idx]  = 0.0;
                this->prefac_cd_y_p_ptr[idx]  = 0.0;
                this->prefac_cd_y_pp_ptr[idx] = 0.0;

                this->prefac_hyp_y_mm_ptr[idx] = 0.0;
                this->prefac_hyp_y_m_ptr[idx]  = 0.0;
                this->prefac_hyp_y_o_ptr[idx]  = 0.0;
                this->prefac_hyp_y_p_ptr[idx]  = 0.0;
                this->prefac_hyp_y_pp_ptr[idx] = 0.0;
            }
        }
    }
}

// Compute Arakawa bracket [zeta,psi] for given 3x3 arrays zeta and psi
// Return the brackets in the order x,y; x,z; y,z
#pragma acc routine seq
inline real_t arakawa_stencil(const real_t (&zeta)[3][3],
                              const real_t (&psi)[3][3])
{
    real_t jplusplus, jpluscross, jcrossplus;

    jplusplus = (zeta[1][2] - zeta[1][0]) * (psi[2][1]  - psi[0][1])
              - (zeta[2][1] - zeta[0][1]) * (psi[1][2]  - psi[1][0]);

    jpluscross = zeta[1][2] * (psi[2][2] - psi[0][2])
               - zeta[1][0] * (psi[2][0] - psi[0][0])
               - zeta[2][1] * (psi[2][2] - psi[2][0])
               + zeta[0][1] * (psi[0][2] - psi[0][0]);

    jcrossplus = zeta[2][2] * (psi[2][1] - psi[1][2])
               - zeta[0][0] * (psi[1][0] - psi[0][1])
               - zeta[2][0] * (psi[2][1] - psi[1][0])
               + zeta[0][2] * (psi[1][2] - psi[0][1]);

    return jplusplus + jpluscross + jcrossplus;
}

#pragma acc routine seq
inline void op_rhs_vlasov_eq_static_gpu_t::comp_kernel(
    const int32_t i, const int32_t k, const int32_t l, const int32_t m,
    const int32_t n, const mesh_5d_t& mesh,
    const data_array_t<const real_t, 5>& f_in,
    const data_array_t<const real_t, 2>& phi_in,
    const data_array_t<const real_t, 2>& A_par_in,
    const data_array_t<const real_t, 2>& B_par_in,
    const data_array_t<real_t, 2>& H2,
    data_array_t<real_t, 5>& f_out) const
{
    // Neighbor indices
    int32_t ngb_m, ngb_p, ngb_mm, ngb_pp, ngb_mp, ngb_pm;

    // B parallel star
    real_t bps;
    // Poisson bracket
    real_t pb;
    // Derivatives of dist func
    real_t dfdx, dfdy, dfdz, dfdvp;
    // Derivatives of electrostatic potential
    real_t dphidx, dphidy, dphidz;
    // Derivatives of electromagnetic vector potential
    real_t dAdx, dAdy, dAdz;
    // Derivatives of the parallel magnetic fluctuations
    real_t dBpardx, dBpardy, dBpardz;
    // Derivatives of the Hamiltonian
    real_t grad_H_x, grad_H_y, grad_H_z;
    // Derivatives of the second order Hamiltonian
    real_t dH2dx, dH2dy, dH2dz;
    // Values of dist func used in derivative stencil
    real_t f_mm, f_m, f_p, f_pp;
    // Values of phi used in derivative stencil
    real_t phi_mm, phi_m, phi_p, phi_pp;
    // Values of Apar used in derivative stencil
    real_t A_par_mm, A_par_m, A_par_p, A_par_pp;
    // Values of Bpar used in derivative stencil
    real_t B_par_mm, B_par_m, B_par_p, B_par_pp;
    // Values of H2 used in derivative stencil
    real_t H2_mm, H2_m, H2_p, H2_pp;
    // Terms collecting hyperdiffusion and diffusion
    real_t hyp_xz, hyp_y, hyp_vp, dif_xz;
    // Right hand side terms
    real_t bs_adv, bx_adv, vp_adv;

    // Interpolate on planes k - 2, k - 1, k + 1 and k + 2
    // to be used for calculating y derivatives
    f_mm = f_m = f_p = f_pp = 0.0;
    phi_mm = phi_m = phi_p = phi_pp = 0.0;
    A_par_mm = A_par_m = A_par_p = A_par_pp = 0.0;
    B_par_mm = B_par_m = B_par_p = B_par_pp = 0.0;
    H2_mm = H2_m = H2_p = H2_pp = 0.0;

    // NOTE: The parallel derivatives of phi and H2 (propto dphi^2)
    //       could be combined to gain some performance
    // TODO: Investigate the potential performance gain

    for (int32_t q = mesh.map_negative2(k).i(i);
         q < mesh.map_negative2(k).i(i + 1); q++)
    {
        f_mm     += mesh.map_negative2(k).val(q)
                  * f_in(mesh.map_negative2(k).j(q), k - 2, l, m, n);
        phi_mm   += mesh.map_negative2(k).val(q)
                  * phi_in(mesh.map_negative2(k).j(q), k - 2);
        A_par_mm += mesh.map_negative2(k).val(q)
                  * A_par_in(mesh.map_negative2(k).j(q), k - 2);
        B_par_mm += mesh.map_negative2(k).val(q)
                  * B_par_in(mesh.map_negative2(k).j(q), k - 2);
        H2_mm    += mesh.map_negative2(k).val(q)
                  * H2(mesh.map_negative2(k).j(q), k - 2);
    }

    for (int32_t q = mesh.map_negative1(k).i(i);
         q < mesh.map_negative1(k).i(i + 1); q++)
    {
        f_m     += mesh.map_negative1(k).val(q)
                 * f_in(mesh.map_negative1(k).j(q), k - 1, l, m, n);
        phi_m   += mesh.map_negative1(k).val(q)
                 * phi_in(mesh.map_negative1(k).j(q), k - 1);
        A_par_m += mesh.map_negative1(k).val(q)
                 * A_par_in(mesh.map_negative1(k).j(q), k - 1);
        B_par_m += mesh.map_negative1(k).val(q)
                 * B_par_in(mesh.map_negative1(k).j(q), k - 1);
        H2_m    += mesh.map_negative1(k).val(q)
                 * H2(mesh.map_negative1(k).j(q), k - 1);
    }

    for (int32_t q = mesh.map_positive1(k).i(i);
         q < mesh.map_positive1(k).i(i + 1); q++)
    {
        f_p     += mesh.map_positive1(k).val(q)
                 * f_in(mesh.map_positive1(k).j(q), k + 1, l, m, n);
        phi_p   += mesh.map_positive1(k).val(q)
                 * phi_in(mesh.map_positive1(k).j(q), k + 1);
        A_par_p += mesh.map_positive1(k).val(q)
                 * A_par_in(mesh.map_positive1(k).j(q), k + 1);
        B_par_p += mesh.map_positive1(k).val(q)
                 * B_par_in(mesh.map_positive1(k).j(q), k + 1);
        H2_p    += mesh.map_positive1(k).val(q)
                 * H2(mesh.map_positive1(k).j(q), k + 1);
    }

    for (int32_t q = mesh.map_positive2(k).i(i);
         q < mesh.map_positive2(k).i(i + 1); q++)
    {
        f_pp     += mesh.map_positive2(k).val(q)
                  * f_in(mesh.map_positive2(k).j(q), k + 2, l, m, n);
        phi_pp   += mesh.map_positive2(k).val(q)
                  * phi_in(mesh.map_positive2(k).j(q), k + 2);
        A_par_pp += mesh.map_positive2(k).val(q)
                  * A_par_in(mesh.map_positive2(k).j(q), k + 2);
        B_par_pp += mesh.map_positive2(k).val(q)
                  * B_par_in(mesh.map_positive2(k).j(q), k + 2);
        H2_pp    += mesh.map_positive2(k).val(q)
                  * H2(mesh.map_positive2(k).j(q), k + 2);
    }

    // Calculate y derivatives

    dfdy = this->prefac_cd_y_mm(i, k) * f_mm
         + this->prefac_cd_y_m(i, k)  * f_m
         + this->prefac_cd_y_o(i, k)  * f_in(i, k, l, m, n)
         + this->prefac_cd_y_p(i, k)  * f_p
         + this->prefac_cd_y_pp(i, k) * f_pp;

    dphidy = this->prefac_cd_y_mm(i, k) * phi_mm
           + this->prefac_cd_y_m(i, k)  * phi_m
           + this->prefac_cd_y_o(i, k)  * phi_in(i, k)
           + this->prefac_cd_y_p(i, k)  * phi_p
           + this->prefac_cd_y_pp(i, k) * phi_pp;

    dAdy = this->prefac_cd_y_mm(i, k) * A_par_mm
         + this->prefac_cd_y_m(i, k)  * A_par_m
         + this->prefac_cd_y_o(i, k)  * A_par_in(i, k)
         + this->prefac_cd_y_p(i, k)  * A_par_p
         + this->prefac_cd_y_pp(i, k) * A_par_pp;

    dBpardy = this->prefac_cd_y_mm(i, k) * B_par_mm
            + this->prefac_cd_y_m(i, k)  * B_par_m
            + this->prefac_cd_y_o(i, k)  * B_par_in(i, k)
            + this->prefac_cd_y_p(i, k)  * B_par_p
            + this->prefac_cd_y_pp(i, k) * B_par_pp;

    dH2dy = this->prefac_cd_y_mm(i, k) * H2_mm
          + this->prefac_cd_y_m(i, k)  * H2_m
          + this->prefac_cd_y_o(i, k)  * H2(i, k)
          + this->prefac_cd_y_p(i, k)  * H2_p
          + this->prefac_cd_y_pp(i, k) * H2_pp;

    hyp_y = this->prefac_hyp_y_mm(i, k) * f_mm
          - this->prefac_hyp_y_m(i, k)  * f_m
          - this->prefac_hyp_y_o(i, k)  * f_in(i, k, l, m, n)
          - this->prefac_hyp_y_p(i, k)  * f_p
          - this->prefac_hyp_y_pp(i, k) * f_pp;

    // Load the index of neighbors in the x direction
    ngb_mm = mesh.neighbors(-2,  0, i, k);
    ngb_m  = mesh.neighbors(-1,  0, i, k);
    ngb_mp = mesh.neighbors(-1,  1, i, k);
    ngb_pm = mesh.neighbors( 1, -1, i, k);
    ngb_p  = mesh.neighbors( 1,  0, i, k);
    ngb_pp = mesh.neighbors( 2,  0, i, k);

    // Calculate the parts involving x derivatives for the biharmonic
    // hyperdiffusion operator
    hyp_xz = -this->prefac_hyp_xz * (  1.0 * f_in(ngb_mm, k, l, m, n)
                                    -  8.0 * f_in(ngb_m , k, l, m, n)
                                    +  2.0 * f_in(ngb_mp, k, l, m, n)
                                    + 20.0 * f_in(i     , k, l, m, n)
                                    +  2.0 * f_in(ngb_pm, k, l, m, n)
                                    -  8.0 * f_in(ngb_p , k, l, m, n)
                                    +  1.0 * f_in(ngb_pp, k, l, m, n));

    // Calculate the parts involving x derivatives for the harmonic
    // diffusion operator applied in the buffer zones
    dif_xz = this->prefac_buffer_zone(i, k) * ( 1.0 * f_in(ngb_m, k, l, m, n)
                                              - 4.0 * f_in(    i, k, l, m, n)
                                              + 1.0 * f_in(ngb_p, k, l, m, n));

    // Calculate x derivatives
    dfdx = this->prefac_cd_xz * (- 1.0 / 12.0 * f_in(ngb_pp, k, l, m, n)
                                 + 2.0 /  3.0 * f_in(ngb_p , k, l, m, n)
                                 - 2.0 /  3.0 * f_in(ngb_m , k, l, m, n)
                                 + 1.0 / 12.0 * f_in(ngb_mm, k, l, m, n));

    dphidx = this->prefac_cd_xz * (- 1.0 / 12.0 * phi_in(ngb_pp, k)
                                   + 2.0 /  3.0 * phi_in(ngb_p , k)
                                   - 2.0 /  3.0 * phi_in(ngb_m , k)
                                   + 1.0 / 12.0 * phi_in(ngb_mm, k));

    dAdx = this->prefac_cd_xz * (- 1.0 / 12.0 * A_par_in(ngb_pp, k)
                                 + 2.0 /  3.0 * A_par_in(ngb_p , k)
                                 - 2.0 /  3.0 * A_par_in(ngb_m , k)
                                 + 1.0 / 12.0 * A_par_in(ngb_mm, k));

    dBpardx = this->prefac_cd_xz * (- 1.0 / 12.0 * B_par_in(ngb_pp, k)
                                    + 2.0 /  3.0 * B_par_in(ngb_p , k)
                                    - 2.0 /  3.0 * B_par_in(ngb_m , k)
                                    + 1.0 / 12.0 * B_par_in(ngb_mm, k));

    dH2dx = this->prefac_cd_xz * (- 1.0 / 12.0 * H2(ngb_pp, k)
                                  + 2.0 /  3.0 * H2(ngb_p , k)
                                  - 2.0 /  3.0 * H2(ngb_m , k)
                                  + 1.0 / 12.0 * H2(ngb_mm, k));

    // Load the index of neighbors in the z direction
    ngb_mm = mesh.neighbors( 0, -2, i, k);
    ngb_m  = mesh.neighbors( 0, -1, i, k);
    ngb_mp = mesh.neighbors(-1, -1, i, k);
    ngb_pm = mesh.neighbors( 1,  1, i, k);
    ngb_p  = mesh.neighbors( 0,  1, i, k);
    ngb_pp = mesh.neighbors( 0,  2, i, k);

    // Calculate the parts involving z derivatives for the biharmonic
    // hyperdiffusion operator
    hyp_xz += -this->prefac_hyp_xz * (  1.0 * f_in(ngb_mm, k, l, m, n)
                                     -  8.0 * f_in(ngb_m , k, l, m, n)
                                     +  2.0 * f_in(ngb_mp, k, l, m, n)
                                     +  2.0 * f_in(ngb_pm, k, l, m, n)
                                     -  8.0 * f_in(ngb_p , k, l, m, n)
                                     +  1.0 * f_in(ngb_pp, k, l, m, n));

    // Calculate the parts involving z derivatives for the harmonic
    // diffusion operator applied in the buffer zones
    dif_xz += this->prefac_buffer_zone(i, k) * ( f_in(ngb_m, k, l, m, n)
                                               + f_in(ngb_p, k, l, m, n));

    // Calculate z derivatives
    dfdz = this->prefac_cd_xz * (- 1.0 / 12.0 * f_in(ngb_pp, k, l, m, n)
                                 + 2.0 /  3.0 * f_in(ngb_p , k, l, m, n)
                                 - 2.0 /  3.0 * f_in(ngb_m , k, l, m, n)
                                 + 1.0 / 12.0 * f_in(ngb_mm, k, l, m, n));

    dphidz = this->prefac_cd_xz * (- 1.0 / 12.0 * phi_in(ngb_pp, k)
                                   + 2.0 /  3.0 * phi_in(ngb_p , k)
                                   - 2.0 /  3.0 * phi_in(ngb_m , k)
                                   + 1.0 / 12.0 * phi_in(ngb_mm, k));

    dAdz = this->prefac_cd_xz * (- 1.0 / 12.0 * A_par_in(ngb_pp, k)
                                 + 2.0 /  3.0 * A_par_in(ngb_p , k)
                                 - 2.0 /  3.0 * A_par_in(ngb_m , k)
                                 + 1.0 / 12.0 * A_par_in(ngb_mm, k));

    dBpardz = this->prefac_cd_xz * (- 1.0 / 12.0 * B_par_in(ngb_pp, k)
                                    + 2.0 /  3.0 * B_par_in(ngb_p , k)
                                    - 2.0 /  3.0 * B_par_in(ngb_m , k)
                                    + 1.0 / 12.0 * B_par_in(ngb_mm, k));

    dH2dz = this->prefac_cd_xz * (- 1.0 / 12.0 * H2(ngb_pp, k)
                                  + 2.0 /  3.0 * H2(ngb_p , k)
                                  - 2.0 /  3.0 * H2(ngb_m , k)
                                  + 1.0 / 12.0 * H2(ngb_mm, k));

    // Calculate vp derivatives
    dfdvp = this->prefac_cd_vp * (- 1.0 / 12.0 * f_in(i, k, l + 2, m, n)
                                  + 2.0 /  3.0 * f_in(i, k, l + 1, m, n)
                                  - 2.0 /  3.0 * f_in(i, k, l - 1, m, n)
                                  + 1.0 / 12.0 * f_in(i, k, l - 2, m, n));

    hyp_vp = -this->prefac_hyp_vp * ( 1.0 * f_in(i, k, l - 2, m, n)
                                    - 4.0 * f_in(i, k, l - 1, m, n)
                                    + 6.0 * f_in(i, k, l    , m, n)
                                    - 4.0 * f_in(i, k, l + 1, m, n)
                                    + 1.0 * f_in(i, k, l + 2, m, n));

    // Calculate the right hand side

    bps = mesh.absB(i, k) + this->prefac_bps(n) * mesh.vp(l)
                                                * mesh.curl_normb_y(i, k);

    // Fill zeta and psi for the calculation of the Poisson brackets
    // {q*phi, f}_zx, {T*mu*B_par, f}_zx, {sqrt(2/m)*q*vp*A_par, f}_zx
    // with the Arakawa method.
    // NOTE: The function arakawa_stencil calculates the Poisson
    //       brackets in the order x;z. Therefore a minus is added to
    //       the arguments.
    // A.12 (a) 4th line (flutter terms)
    // A.13 (b) 4th line

    // Arrays used in Arakawa stencil
    real_t zeta[3][3];
    real_t psi[3][3];

    for (int32_t oz = -1; oz <= 1; oz++) {
        for (int32_t ox = -1; ox <= 1; ox++) {
            ngb_m = mesh.neighbors(ox, oz, i, k);
            zeta[oz+1][ox+1] = this->charges(n)            * phi_in(ngb_m, k)
                             + this->prefac_Bpar(n) * mesh.mu(m)
                                                           * B_par_in(ngb_m, k)
                             - this->prefac_flutter_xyz(n) * mesh.vp(l)
                                                           * A_par_in(ngb_m, k)
                             + this->masses(n)             * H2(ngb_m, k);
            psi[oz+1][ox+1] = f_in(ngb_m, k, l, m, n);
        }
    }

    grad_H_x = mesh.mu(m) * mesh.dabsBdx(i, k) * this->temp_scalings(n)
             + this->charges(n) * dphidx
             + this->masses(n)  * dH2dx
             + this->prefac_Bpar(n) * mesh.mu(m)
                                 * dBpardx;
    grad_H_y = mesh.mu(m) * mesh.dabsBdy(i, k) * this->temp_scalings(n)
             + this->charges(n) * dphidy
             + this->masses(n)  * dH2dy
             + this->prefac_Bpar(n) * mesh.mu(m)
                                 * dBpardy;
    grad_H_z = mesh.mu(m) * mesh.dabsBdz(i, k) * this->temp_scalings(n)
             + this->charges(n) * dphidz
             + this->masses(n)  * dH2dz
             + this->prefac_Bpar(n) * mesh.mu(m)
                                 * dBpardz;

    // 1) Calculate cross field advection

    // These terms can be expressed as Poisson brackets. Calculate the Poisson
    // brackets {q*phi, f}_zx, {T*mu*B_par, f}_zx, {sqrt(2/m)*q*vp*A_par, f}_zx
    // with the Arakawa method
    pb = this->prefac_arakawa * arakawa_stencil(zeta, psi);
    // Calculate the remaining Poisson brackets
    // {muB, f}_zx, bR{muB, f}_yz, bZ{muB, f}_xy and
    // {q*phi, f}_yz, {q*phi, f}_xy and
    // {T*mu*B_par, f}_yz, {T*mu*B_par, f}_xy and
    // {sqrt(2/m)*q*vp A_par, f}_yz, {sqrt(2/m)*q*vp*A_par, f}_xy
    // with centered finite differences
    // A.12 (a) 3rd line (flutter terms)
    // A.13 (b) 2nd line -> mu*dB
    pb -= dfdz * (-mesh.mu(m) * this->temp_scalings(n) * mesh.dabsBdx(i, k)
                 + mesh.normb_R(i, k)
                * (grad_H_y - this->prefac_flutter_xyz(n) * mesh.vp(l) * dAdy))
        + dfdx * ( mesh.mu(m) * this->temp_scalings(n) * mesh.dabsBdz(i, k)
                - mesh.normb_Z(i, k)
                * (grad_H_y - this->prefac_flutter_xyz(n) * mesh.vp(l) * dAdy))
        + dfdy
             * (- mesh.normb_R(i, k)
                * (grad_H_z - this->prefac_flutter_xyz(n) * mesh.vp(l) * dAdz)
                + mesh.normb_Z(i, k)
                * (grad_H_x - this->prefac_flutter_xyz(n) * mesh.vp(l) * dAdx));

    bx_adv = pb * mesh.inv_g(i, k) * this->prefac_orth(n);

    // 2) Calculate advection along corrected magnetic field B^*

    // Contribution of the parallel advection
    // A.12 (a) 1st line
    bs_adv = -2.0 * this->prefac_par(n) * mesh.vp(l) * mesh.absB(i, k) * dfdy;
    // Contribution of the curvature term
    // A.12 (a) 2nd line
    bs_adv += 2.0 * mesh.vp(l) * this->prefac_orth(n) * mesh.vp(l)
            * (                           - mesh.dgyzdy_over_g(i, k)  * dfdx
              + (mesh.dgyzdx_over_g(i, k) - mesh.dgyxdz_over_g(i, k)) * dfdy
              +  mesh.dgyxdy_over_g(i, k)                             * dfdz);
    bs_adv *= this->temp_scalings(n);

    // 3) Calculate advection in velocity space

    // Contribution of the parallel advection
    // A.14 (c) 1st+6th line very first terms
    vp_adv = this->prefac_par(n) * mesh.absB(i, k) * grad_H_y * dfdvp;
    // Contribution of the curvature term
    // A.14 (c) 2nd line + 6th line second part + 7th line
    vp_adv -= this->prefac_orth(n) * mesh.vp(l)
           * (                           - mesh.dgyzdy_over_g(i, k)  * grad_H_x
             + (mesh.dgyzdx_over_g(i, k) - mesh.dgyxdz_over_g(i, k)) * grad_H_y
             +  mesh.dgyxdy_over_g(i, k)                             * grad_H_z)
           * dfdvp;

    // Contribution of the electromagnetic flutter
    // A.14 (c) 3-5th + 8-10th line
    vp_adv -= this->prefac_flutter_vp(n) * mesh.inv_g(i, k)
            * ( grad_H_x
                  * (                     dAdz - mesh.normb_Z(i, k) * dAdy)
              + grad_H_z
                  * (mesh.normb_R(i, k) * dAdy -                      dAdx)
              + grad_H_y
                  * (mesh.normb_Z(i, k) * dAdx - mesh.normb_R(i, k) * dAdz))
            * dfdvp;

    // Collect all terms
    f_out(i, k, l, m, n) = mesh.is_compute(i, k)
                         * (bs_adv + bx_adv + vp_adv) / bps
                         + hyp_xz + hyp_vp + hyp_y + dif_xz;
}
