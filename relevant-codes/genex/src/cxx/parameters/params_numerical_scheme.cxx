#include "params_numerical_scheme.hxx"

namespace params_numerical_scheme
{
    // Private objects with file scope and external linkage

    // Order of the two dimensional interpolation for the calcuation of field
    // line mapped values of fields
    static int32_t int_order = 3;
    // Hyperdiffusion coefficient in parallel direction
    static real_t hyp_y = 0.0;
    // Hyperdiffusion coefficient in R and Z direction
    static real_t hyp_xz = 0.0;
    // Hyperdiffusion coefficient in vp direction
    static real_t hyp_vp = 0.0;
    // Size of the boundary buffer zone in grid-points
    static int32_t buf_zone_size = 2;
    // Diffusion coefficient applied in the buffer zone
    static real_t buf_zone_strength = 1.0;
    // Size of the axis buffer zone in grid-points
    static int32_t buf_zone_size_axis = 0;
    // Diffusion coefficient applied in the buffer zone at the magnetic axis
    static real_t buf_zone_strength_axis = 1.0;

    // Getters for the private objects

    int32_t get_int_order() { return int_order; }
    real_t get_hyp_y() { return hyp_y; }
    real_t get_hyp_xz() { return hyp_xz; }
    real_t get_hyp_vp() { return hyp_vp; }
    int32_t get_buf_zone_size() { return buf_zone_size; }
    real_t get_buf_zone_strength() { return buf_zone_strength; }
    int32_t get_buf_zone_size_axis() { return buf_zone_size_axis; }
    real_t get_buf_zone_strength_axis() { return buf_zone_strength_axis; }
}

void cbind_set_params_numerical_scheme(
    struct params_numerical_scheme_data_t* params_data)
{
    params_numerical_scheme::int_order = params_data->int_order;
    params_numerical_scheme::hyp_y = params_data->hyp_y;
    params_numerical_scheme::hyp_xz = params_data->hyp_xz;
    params_numerical_scheme::hyp_vp = params_data->hyp_vp;
    params_numerical_scheme::buf_zone_size = params_data->buf_zone_size;
    params_numerical_scheme::buf_zone_strength = params_data->buf_zone_strength;
    params_numerical_scheme::buf_zone_size_axis =
        params_data->buf_zone_size_axis;
    params_numerical_scheme::buf_zone_strength_axis =
        params_data->buf_zone_strength_axis;
}
