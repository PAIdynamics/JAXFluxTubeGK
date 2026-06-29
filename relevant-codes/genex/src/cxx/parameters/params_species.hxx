#ifndef PARAMS_SPECIES_HXX
#define PARAMS_SPECIES_HXX

#include "genex_cxx_env.hxx"

// Namespace for parameters related to the species
namespace params_species
{
    // Public parameter specifying the maximum number of species supported
    // NOTE: Synchronize with src/parameters/params_species_m.f90
    constexpr int32_t n_spec_supported = 8;

    // Getter for the mass of each species
    real_t get_mass(int32_t n);

    // Getter for the charge of each species
    real_t get_charge(int32_t n);

    // Getter for the temperature scaling of each species
    real_t get_temp_scaling(int32_t n);

    // Getter that returns true if the species index is pointing to electron
    bool is_electron(int32_t n);

    // Getter for electron index
    int32_t get_electron_id();

    // Getter for the pointer mass array
    const real_t* get_masses();

    // Getter for the pointer charge array
    const real_t* get_charges();

    // Getter for the pointer temp_scaling array
    const real_t* get_temp_scalings();
}

#ifdef __cplusplus
extern "C" {
#endif

void cbind_set_params_species(real_t* masses_ptr,
                              real_t* charges_ptr,
                              real_t* temp_scalings_ptr);

#ifdef __cplusplus
}
#endif

#endif
