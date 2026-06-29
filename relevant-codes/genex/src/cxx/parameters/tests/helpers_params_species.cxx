#ifndef HELPERS_PARAMS_SPECIES_CXX
#define HELPERS_PARAMS_SPECIES_CXX

#include "genex_cxx_env.hxx"
#include "params_species.hxx"

#ifdef __cplusplus
extern "C" {
#endif

void cbind_get_params_species(int32_t n_spec, int32_t* electron_id,
    real_t* masses_ptr, real_t* charges_ptr, real_t* temp_scalings_ptr)
{
    for (int32_t n = 1; n <= n_spec; n++)
    {
        masses_ptr[n - 1] = params_species::get_mass(n);
        charges_ptr[n - 1] = params_species::get_charge(n);
        temp_scalings_ptr[n - 1] = params_species::get_temp_scaling(n);

        if (params_species::is_electron(n))
        {
            electron_id[0] = n;
        }
    }
}

#ifdef __cplusplus
}
#endif

#endif
