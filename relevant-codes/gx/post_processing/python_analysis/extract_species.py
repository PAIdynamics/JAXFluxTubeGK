def extract_species(simulation):

    # Constructing list of species in simulation
    if len(simulation['Inputs']['Species']['species_type']) == 1:
        if simulation['Inputs']['Species']['Boltzmann']['Boltzmann_type_dum'] == "electrons":
            species_list = ["ions"]
        else:
            species_list = ["electrons"]
    else:
        species_list = ["ions", "electrons"]
        for i in simulation['Inputs']['Species']['species_type'][2:]:
            species_list.append("impurity{}".format(i-1))

    # Finding reference species
    for species_index, species in enumerate(species_list):
        if simulation['Inputs']['Species']['m'][species_index] == 1.0:
            ref_species = species

    return species_list, ref_species