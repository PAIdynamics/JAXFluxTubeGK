#include "params_species.hxx"

namespace params_species
{
    // Private objects with file scope and external linkage

    // Pointer of mass array each species
    // NOTE: Synchronize default values with src/parameters/params_species_m.f90
    static real_t masses[n_spec_supported]
        {1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    // Pointer of charge array of each species
    // NOTE: Synchronize default values with src/parameters/params_species_m.f90
    static real_t charges[n_spec_supported]
        {1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    // Pointer of temperature scaling array of each species
    // NOTE: Synchronize default values with src/parameters/params_species_m.f90
    static real_t temp_scalings[n_spec_supported]
        {1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    // Pointer of bool array indicating which one is electron
    static bool is_electron_mask[n_spec_supported] {};

    // Getters for the private objects

    real_t get_mass(int32_t n)
    {
        return masses[n - 1];
    }

    real_t get_charge(int32_t n)
    {
        return charges[n - 1];
    }

    real_t get_temp_scaling(int32_t n)
    {
        return temp_scalings[n - 1];
    }

    bool is_electron(int32_t n)
    {
        return is_electron_mask[n - 1];
    }

    int32_t get_electron_id()
    {
        for (int32_t n = 1; n <= n_spec_supported; n++)
        {
            if (is_electron(n)) return n;
        }
    }

    const real_t* get_masses()
    {
        return masses;
    }

    const real_t* get_charges()
    {
        return charges;
    }

    const real_t* get_temp_scalings()
    {
        return temp_scalings;
    }
}

void cbind_set_params_species(real_t* masses_ptr,
                              real_t* charges_ptr,
                              real_t* temp_scalings_ptr)
{

    for (int32_t n = 0; n < params_species::n_spec_supported; n++)
    {
        params_species::masses[n]  = masses_ptr[n];
        params_species::charges[n] = charges_ptr[n];
        params_species::temp_scalings[n] = temp_scalings_ptr[n];

        if (charges_ptr[n] == -1.0)
        {
            params_species::is_electron_mask[n] = true;
        }
    }
}
