#ifndef PARAMS_NUMERICAL_SCHEME_HXX
#define PARAMS_NUMERICAL_SCHEME_HXX

#include "genex_cxx_env.hxx"

// Namespace for parameters related to the numerical scheme
namespace params_numerical_scheme
{
    // Getter for the order of the two dimensional interpolation
    int32_t get_int_order();

    // Getter for the hyperdiffusion coefficient in parallel direction
    real_t get_hyp_y();

    // Getter for the hyperdiffusion coefficient in R and Z direction
    real_t get_hyp_xz();

    // Getter for the hyperdiffusion coefficient in vp direction
    real_t get_hyp_vp();

    // Getter for the size of the boundary buffer zone in grid-points
    int32_t get_buf_zone_size();

    // Getter for the diffusion coefficient applied in the buffer zone
    real_t get_buf_zone_strength();

    // Getter for the size of the axis buffer zone in grid-points
    int32_t get_buf_zone_size_axis();

    // Getter for the diffusion coefficient applied in the buffer zone
    // at the magnetic axis
    real_t get_buf_zone_strength_axis();
}

#ifdef __cplusplus
extern "C" {
#endif

struct params_numerical_scheme_data_t
{
    int32_t int_order;
    real_t hyp_y;
    real_t hyp_xz;
    real_t hyp_vp;
    int32_t buf_zone_size;
    real_t buf_zone_strength;
    int32_t buf_zone_size_axis;
    real_t buf_zone_strength_axis;
};

void cbind_set_params_numerical_scheme(
    struct params_numerical_scheme_data_t* params_data);

#ifdef __cplusplus
}
#endif

#endif
