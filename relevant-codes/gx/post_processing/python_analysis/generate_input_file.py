import numpy as np
import sys
import pathlib
from netCDF4 import Dataset
from load_files import load_files 
import matplotlib
from tqdm import tqdm 

# This python script will generate a commented .in file from an output file for simulations using GX.
# The .out file will be created (and written to) in the same directory as the specified output file(s)
# Only the minimum set of parameters required to uniquely specify the simulation will be written
# For runs using external geometry data, the appropriate .eik.out file will be created and used [TO BE IMPLEMENTED]

# Usage: python generate_input_file.py [list of output files or paths]


def write_dimensions(file, data):

    file.write("\n\n[Dimensions]")

    file.write("\n ntheta  = {}     # Number of points along field line (theta) per 2*pi segment".format(data['Dimensions']['ntheta']))
    file.write("\n nperiod = {}      # Number of 2*pi segments along field line is 2*nperiod-1".format(data['Dimensions']['nperiod']))
    file.write("\n ny      = {}    # Number of real-space grid-points in y".format(data['Dimensions']['ny']))
    file.write("\n nx      = {}    # Number of real-space grid-points in x".format(data['Dimensions']['nx']))

    file.write("\n\n nhermite  = {}   # Number of Hermite moments (v_parallel resolution)".format(data['Dimensions']['nhermite']))
    file.write("\n nlaguerre = {}    # Number of Laguerre moments (mu*B resolution)".format(data['Dimensions']['nlaguerre']))
    file.write("\n nspecies  = {}    # Number of evolved kinetic species (adiabatic electrons do not count towards nspecies)".format(data['Dimensions']['nspecies']))


def write_domain(file, data):

    file.write("\n\n[Domain]")

    boundary = data['Inputs']['Domain']['boundary_dum']

    if bool(boundary == "linked"):
        file.write("\n boundary = \"{}\"   # Use twist-shift boundary conditions along field line".format(boundary))

    else:
        file.write("\n boundary = \"{}\"   # Use periodic boundary conditions along field line".format(boundary))
        file.write("\n x0       = {:.1f}        # Controls box length in x, defined such that kx_min*rho_ref = 1/x0".format(data['Inputs']['Domain']['x0']))
    
    file.write("\n y0       = {}       # Controls box length in y, defined such that ky_min*rho_ref = 1/y0".format(data['Inputs']['Domain']['y0']))
    
    if bool(data['Geometry']['slab']): 
        file.write("\n z0       = {}         # Controls box length in z, defined such that kz_min*rho_ref = 1/z0".format(data['Inputs']['Domain']['z0']))

    if bool(data['Inputs']['Domain']['long_wavelength_GK']):
        file.write("\n long_wavelength_GK = true")


def write_resize(file, data):

    if bool(data['Inputs']['Resize']['domain_change']):

        file.write("\n\n[Resize]")

        file.write("\n domain_change = {}    # Specifies that there is a change in [Dimensions] and/or [Domain] variables".format(str(bool(data['Inputs']['Resize']['domain_change'])).lower()))

        if (bool(data['Inputs']['Resize']['x0_mult'] > 1) \
                or bool(data['Inputs']['Resize']['y0_mult'] > 1) \
                or bool(data['Inputs']['Resize']['z0_mult'] > 1)):
            file.write("\n\n x0_mult = {}    # Multiplier for both length in x".format(data['Inputs']['Resize']['x0_mult']))
            file.write("\n y0_mult = {}    # Multiplier for both length in y".format(data['Inputs']['Resize']['y0_mult']))
            file.write("\n z0_mult = {}    # Multiplier for both length in z".format(data['Inputs']['Resize']['z0_mult']))

        if (bool(data['Inputs']['Resize']['nx_mult'] > 1) \
                or bool(data['Inputs']['Resize']['ny_mult'] > 1) \
                or bool(data['Inputs']['Resize']['nz_mult'] > 1)):
            file.write("\n\n nx_mult = {}    # Multiplier for number of real-space grid-points in x".format(data['Inputs']['Resize']['nx_mult']))
            file.write("\n ny_mult = {}    # Multiplier for number of real-space grid-points in y".format(data['Inputs']['Resize']['ny_mult']))
            file.write("\n nz_mult = {}    # Multiplier for nx * ntheta".format(data['Inputs']['Resize']['nz_mult']))

        if (bool(data['Inputs']['Resize']['nm_add'] != 0) \
                or bool(data['Inputs']['Resize']['nl_add'] != 0)):
            file.write("\n\n nm_add = {}    # Adding or subtracting Hermite moments".format(data['Inputs']['Resize']['nm_add']))
            file.write("\n nl_add = {}    # Adding or subtracting Laguerre moments".format(data['Inputs']['Resize']['nl_add']))

        if bool(data['Inputs']['Resize']['ns_add'] > 0):
            file.write("\n\n ns_add = {}    # Adding ns_add species".format(data['Inputs']['Resize']['ns_add']))


def write_physics(file, data):

    file.write("\n\n[Physics]")

    file.write("\n beta           = {:}      # Reference normalized pressure, beta = n_ref * T_ref / ( B_ref**2 / (8*pi))".format(data['Geometry']['beta']))

    tqdm.write("[Physics] Parameter \"ei_colls\" not specified in output file. Setting to default value = false.") # Currently not saved to the output file.
    file.write("\n ei_colls       = false    # Whether to compute electron-ion collisions")

    file.write("\n nonlinear_mode = {}     # Whether to compute nonlinear terms".format(str(bool(data['Inputs']['Controls']['nonlinear_mode'])).lower()))

    if not (data['Inputs']['Controls']['fphi'] == 1.0):
        file.write("\n\n fphi  = {}    # Factor multiplying the electrostatic potential".format(data['Inputs']['Controls']['fphi']))
        file.write("\n fapar = {}    # Factor multiplying the parallel magnetic vector potential".format(data['Inputs']['Controls']['fapar']))
        file.write("\n fbpar = {}    # Factor multiplying the compressional magnetic fluctuations".format(data['Inputs']['Controls']['fbpar']))


def write_time(file, data):

    file.write("\n\n[Time]")

    file.write("\n dt    = {:.2f}   # Timestep size (in units of L_ref/vt_ref)".format(data['Inputs']['Time']['dt']))

    tqdm.write("[Time] Setting \"t_max\" from the final time reached by the simulation. Consider modifying manually.")
    file.write("\n t_max = {}   # Time to which the simiulation will run (in units of L_ref/vt_ref)".format(round(float(data['Dimensions']['time'][-1]))))

    if bool(data['Inputs']['Controls']['nonlinear_mode']):
        file.write("\n cfl   = {:.1f}    # For nonlinear runs, the maximum timestep allowed is proportional to cfl".format(data['Inputs']['Controls']['cfl']))

    if not bool(data['Inputs']['Controls']['scheme_dum'] == "sspx3"):
        file.write("\n\n scheme = \"{}\"   # Determines which timestepping algorithm is used".format(data['Inputs']['Controls']['scheme_dum']))

    if not (data['Inputs']['Controls']['stages'] == 10):
        file.write("\n stages = {}    # The number of Runge-Kutta stages to be used (if applicable)".format(data['Inputs']['Controls']['stages']))


def write_initialization(file, data):

    file.write("\n\n[Initialization]")

    file.write("\n init_field  = \"{}\"    # Moment to which the initial perturbation is applied".format(data['Inputs']['Controls']['init_field_dum']))
    file.write("\n init_amp    = {:.1e}      # Amplitude of initial perturbation".format(data['Inputs']['Controls']['init_amp']))

    if bool(data['Inputs']['Controls']['random_init']):
        file.write("\n random_init = {}        # Use completely random initial conditions (no structure in z)".format(str(bool(data['Inputs']['Controls']['random_init'])).lower()))
    else:
        file.write("\n ikpar_init  = {}            # Parallel wavenumber of initial perturbation".format(data['Inputs']['Controls']['ikpar_init']))

    if bool(data['Inputs']['Controls']['init_electrons_only']):
        file.write("\n\n init_electrons_only = {}  # Whether the initial perturbation is only applied to the electron species.".format(str(bool(data['Inputs']['Controls']['init_electrons_only'])).lower()))


def generate_geometry_file(data, filepath):

    try:
        geofile = open(filepath, 'a+')

        ntheta = data['Dimensions']['ntheta']

        gbdrift  = data['Geometry']['gbdrift']
        gradpar  = data['Geometry']['gradpar'] # Equispaced grid in theta means that gradpar is a number
        grho     = data['Geometry']['grho']
        tgrid    = data['Dimensions']['theta']
        cvdrift  = data['Geometry']['cvdrift'] 
        gds2     = data['Geometry']['gds2'] 
        bmag     = data['Geometry']['bmag'] 
        gds21    = data['Geometry']['gds21'] 
        gds22    = data['Geometry']['gds22'] 
        cvdrift0 = data['Geometry']['cvdrift0']  
        gbdrift0 = data['Geometry']['gbdrift0']  

        # Creating appropriate data layout
        geo_list_0 = []
        geo_list_1 = []
        geo_list_2 = []
        geo_list_3 = []
        geo_list_4 = []

        # Factors of two to maintain backwards compatability with GS2 .eik files
        for i in range(ntheta):
            geo_list_1.append("  {:.9e}    {:.9e}   {:.9e}   {:.9e}\n".format(2*gbdrift[i], gradpar, grho[i], tgrid[i]))
            geo_list_2.append("    {:.9e}    {:.9e}    {:.12e}    {:.9e}\n".format(2*cvdrift[i], gds2[i], bmag[i], tgrid[i]))
            geo_list_3.append("    {:.9e}    {:.9e}    {:.9e}\n".format(gds21[i], gds22[i], tgrid[i]))
            geo_list_4.append("    {:.9e}    {:.9e}    {:.9e}\n".format(2*cvdrift0[i], 2*gbdrift0[i], tgrid[i]))

        geo_list_0.append([geo_list_1, geo_list_2, geo_list_3, geo_list_4])
        geo_list_0 = geo_list_0[0]

        headings = ['ntgrid nperiod ntheta drhodpsi rmaj shat kxfac q\n',\
                    'gbdrift gradpar grho tgrid\n',\
                    'cvdrift gds2 bmag tgrid\n',\
                    'gds21 gds22 tgrid\n', \
                    'cvdrift0 gbdrift0 tgrid\n']
        
        geofile.write(headings[0])

        geofile.write("{}  {}  {}  {:.3f} {:.1f} {:.7e} {:.1f} {:.4f}\n".format(
            round((ntheta-1)/2),
            data['Dimensions']['nperiod'],
            ntheta,
            data['Geometry']['drhodpsi'],
            data['Geometry']['Rmaj'],  
            data['Geometry']['shat'],  
            data['Geometry']['kxfac'],
            data['Geometry']['q'],    
            )
            )
        
        # Manually appending the periodic point to geometric coefficients. Not currently required by .eik interpreter 
        if True:
            geo_list_1.append("    {:.9e}    {:.9e}    {:.9e}    {:.9e}\n".format(2*gbdrift[0], gradpar, grho[0], -tgrid[0]))   
            geo_list_2.append("    {:.9e}   {:.9e}    {:.12e}    {:.9e}\n".format(2*cvdrift[0], gds2[0], bmag[0], -tgrid[0]))
            geo_list_3.append("    {:.9e}    {:.9e}    {:.9e}\n".format(-gds21[0], gds22[0], -tgrid[0]))
            geo_list_4.append("    {:.9e}    {:.9e}    {:.9e}\n".format(- 2*cvdrift0[0], -2*gbdrift0[0], -tgrid[0]))
            ntheta += 1
        
        for i in np.arange(1, len(headings)):
            geofile.writelines(headings[i])
            for j in range(ntheta):                 # should be ntheta if the output geometric coefficients were stored the same as the ones generated in the .eik file
                geofile.writelines(geo_list_0[i-1][j])

        geofile.close()

    except:
        print("[Geometry] Error in creating geometry file. Aborting.")
        geofile.close()
        pathlib.Path.unlink(filepath)


def write_geometry(file, data, filepath, filename):

    file.write("\n\n[Geometry]")

    if bool(data['Geometry']['slab']):
        file.write("\n geo_option = \"slab\"   # Solving in a slab (straight magnetic field configuration)")

    elif bool(data['Geometry']['const_curv'] or data['Geometry']['const_curv']):
        file.write("\n geo_option = \"s-alpha\"    # Solving in an s-alpha geometry")
        file.write("\n eps = {:.5f}             # Flux surface label given by the inverse aspect ratio of the surface in question".format(data['Geometry']['eps']))
        file.write("\n Rmaj = {:.5f}            # Ratio of the major radius to the equilibrium-scale reference length. Rmaj = 1/eps gives L_ref = a".format(data['Geometry']['Rmaj']))
        file.write("\n qinp = {:.2f}               # Magnetic safety factor q".format(data['Geometry']['q']))
        file.write("\n shat = {:.2f}               # The global magnetic shear".format(data['Geometry']['shat']))
        file.write("\n shift = {:.2f}                # Shafranov shift".format(data['Geometry']['shift']))

    else:
        file.write("\n geo_option = \"eik\"")
        file.write("\n geo_file   = \"{}\"".format(filename))

        if pathlib.Path.exists(filepath):
            tqdm.write("[Geometry] External geometry file found. Using as input.")

        else:
            tqdm.write("[Geometry] Creating \".eik.out\" file from geometric coefficients found in the output file.")
            generate_geometry_file(data, filepath)


def write_species(file, data):

    file.write("\n\n[species]")

    nspecies = data['Dimensions']['nspecies']
    species_type = []

    if bool(nspecies == 1):
        if (data['Inputs']['Species']['Boltzmann']['Boltzmann_type_dum'] == "electrons"):
            species_type.append("ions")
        elif (data['Inputs']['Species']['Boltzmann']['Boltzmann_type_dum'] == "ions"):
            species_type.append("electrons")
    else:
        if not (data['Inputs']['Species']['Boltzmann']['Boltzmann_type_dum'] == "electrons"):
            species_type.append("ions")
        elif not (data['Inputs']['Species']['Boltzmann']['Boltzmann_type_dum'] == "ions"):
            species_type.append("electrons")
        if bool(nspecies >= 3):
            for i in range(nspecies-2):
                species_type.append("impurity_species_{:}".format(i))

    file.write("\n z     = [ {} ]              # Charge (normalized to Z_ref)".format(", ".join([str(item) for item in data['Inputs']['Species']['z']])))
    file.write("\n mass  = [ {} ]            # Mass (normalized to m_ref)".format(", ".join([str(item) for item in data['Inputs']['Species']['m']])))
    file.write("\n dens  = [ {} ]               # Density (normalized to dens_ref)".format(", ".join([str(item) for item in data['Inputs']['Species']['n0']])))
    file.write("\n temp  = [ {} ]               # Temperature (normalized to T_ref)".format(", ".join([str(item) for item in data['Inputs']['Species']['T0']])))
    file.write("\n tprim = [ {} ]               # Temperature gradient, L_ref/L_T".format(", ".join([str(item) for item in data['Inputs']['Species']['T0_prime']])))
    file.write("\n fprim = [ {} ]               # Density gradient, L_ref/L_n".format(", ".join([str(item) for item in data['Inputs']['Species']['n0_prime']])))
    file.write("\n type  = [ {} ]    # Species type".format(", ".join(['"{}"'.format(item) for item in species_type if item])))

    
def write_boltzmann(file, data):

    file.write("\n\n[Boltzmann]")

    file.write("\n add_Boltzmann_species = {}    # Whether to include a species with a Boltzmann response".format(str(bool(data['Inputs']['Species']['Boltzmann']['add_Boltzmann_species'])).lower()))

    if bool(data['Inputs']['Species']['Boltzmann']['add_Boltzmann_species']):
        file.write("\n Boltzmann_type = \"{}\"    # Species with the Boltzmann response".format(data['Inputs']['Species']['Boltzmann']['Boltzmann_type_dum']))
        file.write("\n tau_fac        = {:.2f}    # tau_fac = Z_s T_s / T_ref for the adiabatic species".format(data['Inputs']['Species']['Boltzmann']['tau_fac']))


def write_dissipation(file, data):

    file.write("\n\n[Dissipation]")

    par_count  = 0

    if bool(data['Inputs']['Controls']['Numerical_Diss']['hypercollisions']):
        file.write("\n hypercollisions = {}    # Use hypercollisions to bound velocity-space resolution".format(str(bool(data['Inputs']['Controls']['Numerical_Diss']['hypercollisions'])).lower()))
        file.write("\n nu_hyper_m      = {:.1f}     # Coefficient of the Hermite hypercollisions".format(data['Inputs']['Controls']['Numerical_Diss']['nu_hyper_m']))
        file.write("\n p_hyper_m       = {}       # Exponent of the Hermite hypercollisions".format(data['Inputs']['Controls']['Numerical_Diss']['p_hyper_m']))
        file.write("\n nu_hyper_l      = {:.1f}     # Coefficient of the Laguerre hypercollisions".format(data['Inputs']['Controls']['Numerical_Diss']['nu_hyper_l']))
        file.write("\n p_hyper_l       = {}       # Exponent of the Laguerre hypercollisions".format(data['Inputs']['Controls']['Numerical_Diss']['p_hyper_l']))
        par_count += 1
    else:
        print("[Dissipation] No hypercollisions were specified. Consider adding.")

    if bool(data['Inputs']['Controls']['Numerical_Diss']['hyper']):
        file.write("\n\n hyper   = {}    # Provide hyperdissipation at grid-scales in the perpendicular dimensions".format(str(bool(data['Inputs']['Controls']['Numerical_Diss']['hyper'])).lower()))
        file.write("\n D_hyper = {:.1f}     # Coefficient of hyperdissipation".format(data['Inputs']['Controls']['Numerical_Diss']['D_hyper']))
        file.write("\n p_hyper = {}       # Exponent of of hyperdissipation".format(data['Inputs']['Controls']['Numerical_Diss']['p_hyper']))
        par_count += 1
    elif bool(data['Inputs']['Controls']['Numerical_Diss']['HB_hyper']):
        file.write("\n\n HB_hyper = {}    # Use the Hammett-Belli hyperdiffusivity model".format(str(bool(data['Inputs']['Controls']['Numerical_Diss']['HB_hyper'])).lower()))
        file.write("\n D_HB     = {:.1f}      # Coefficient of HB hyperdiffusivity model".format(data['Inputs']['Controls']['Numerical_Diss']['D_HB']))
        file.write("\n p_HB     = {}        # Exponent of HB hyperdiffusivity model".format(data['Inputs']['Controls']['Numerical_Diss']['p_HB']))
        file.write("\n w_osc    = {}      # Frequency parameter in HB hyperdiffusivity model".format(data['Inputs']['Controls']['Numerical_Diss']['w_osc']))
    else:
        print("[Dissipation] No hyperdissipation was specified. Consider adding.")

    if not bool(data['Inputs']['Controls']['closure_model_dum'] == "none"):
        file.write("\n\n closure_model = \"{}\"    # Using a closure model for the velocity-space resolution".format(data['Inputs']['Controls']['closure_model_dum']))
        par_count += 1

    if bool(data['Inputs']['Controls']['closure_model_dum'] == "smith_par"):
        file.write("\n smith_par_q = {}   # A parameter model for closure_model = \"smith_par\"".format(data['Inputs']['Controls']['smith_par_q']))

    if bool(data['Inputs']['Controls']['closure_model_dum'] == "smith_perp"):
        file.write("\n smith_perp_q = {}   # A parameter model for closure_model = \"smith_perp\"".format(data['Inputs']['Controls']['smith_perp_q']))

    if bool(par_count == 0):

        file.write("\n# No dissipation was specified. Consider adding hypercollisions/hyperdissipation.")


def write_restart(file, data, filename):

    file.write("\n\n[Restart]")

    file.write("\n save_for_restart = {}    # Whether to write restart data".format(str(bool(data['Inputs']['Restart']['save_for_restart'])).lower()))

    if bool(data['Inputs']['Restart']['save_for_restart']):
        file.write("\n nsave            = {}    # Restart data will be written every nsave timesteps".format(data['Inputs']['Time']['nsave']))
        if not bool(data['Inputs']['Restart']['restart_to_file_dum'] == filename):
            file.write("\n restart_to_file  = \"{}\"   # Filename to write data needed for restarting the present run".format(data['Inputs']['Restart']['restart_to_file_dum']))

    if bool(data['Inputs']['Restart']['restart']):
        print("[Restart] This ouput file requires a restart file. Please check \"restart_from_file\" in input file.")

        file.write("\n\n restart           = {}    # Continue the simulation from a previous run".format(str(bool(data['Inputs']['Restart']['restart'])).lower()))
        file.write("\n scale             = {}      # Multiply all variables in the restart data by a factor of scale".format(data['Inputs']['Restart']['scale']))
        file.write("\n restart_from_file = \"{}\"    # Filename to read to restart the current run".format(data['Inputs']['Restart']['restart_from_file_dum']))


def write_diagnostics(file, data):

    file.write("\n\n[Diagnostics]")

    file.write("\n fixed_amplitude = {}    # Periodically rescale amplitude of phi to avoid overflow (only for linear calculations)".format(str(bool(data['Inputs']['Diagnostics']['fixed_amp'])).lower()))
    file.write("\n omega           = {}    # Write instantaneous estimates of the complex frequency for each Fourier component of the electrostatic potential".format(str(bool(data['Inputs']['Diagnostics']['omega'])).lower()))
    file.write("\n free_energy     = {}     # Write the total free energy (integrated over the phase-space domain and summed over species) as a function of time".format(str(bool(data['Inputs']['Diagnostics']['free_energy'])).lower()))
    file.write("\n fluxes          = {}     # Write the turbulent fluxes for each species.".format(str(bool(data['Inputs']['Diagnostics']['fluxes'])).lower()))
    # file.write("\n moms            = {}     ".format(str(bool(data.groups['Inputs'].groups['Diagnostics'].variables['fluxes'][:])).lower()))
    # file.write("\n fields          = {}     ".format(str(bool(data.groups['Inputs'].groups['Diagnostics'].variables['fields'][:])).lower()))
    # file.write("\n eigenfunctions  = {}     ".format(str(bool(data.groups['Inputs'].groups['Diagnostics'].variables['eigenfunctions'][:])).lower()))
    
    if bool(data['Inputs']['Diagnostics']['all_zonal_scalars']):
        file.write("\n all_zonal_scalars = {}    # Write quantities such as the RMS value of the zonal component of the ExB velocity as function of time".format(str(bool(data['Inputs']['Diagnostics']['all_zonal_scalars'])).lower()))
    else:
        if bool(data['Inputs']['Diagnostics']['avg_zvE']):
            file.write("\n avg_zvE     = {}    # Write the RMS value of the zonal component of ExB velocity as a function time".format(str(bool(data['Inputs']['Diagnostics']['avg_zvE'])).lower()))
        if bool(data['Inputs']['Diagnostics']['avg_zkxvEy']):
            file.write("\n avg_zkxvEy  = {}    # Write the RMS value of the zonal component of the shear of the ExB velocity as a function time".format(str(bool(data['Inputs']['Diagnostics']['avg_zkxvEy'])).lower()))    
        if bool(data['Inputs']['Diagnostics']['avg_zkden']):
            file.write("\n avg_zkden   = {}    # Write the RMS value of the zonal component of the guiding center radial density gradient as a function of time".format(str(bool(data['Inputs']['Diagnostics']['avg_zkden'])).lower()))    
        if bool(data['Inputs']['Diagnostics']['avg_zkUpar']):
            file.write("\n avg_zkUpar  = {}    # Write the RMS value of the zonal component of the guiding center parallel velocity as a function of time".format(str(bool(data['Inputs']['Diagnostics']['avg_zkUpar'])).lower()))    
        if bool(data['Inputs']['Diagnostics']['avg_zkTpar']):
            file.write("\n avg_zkTpar  = {}    # Write the RMS value of the zonal component of the guiding center parallel temperature as a function of time".format(str(bool(data['Inputs']['Diagnostics']['avg_zkTpar'])).lower()))   
        if bool(data['Inputs']['Diagnostics']['avg_zkTperp']):
            file.write("\n avg_zkTperp = {}    # Write the RMS value of the zonal component of the guiding center perpendicular temperature as a function of time".format(str(bool(data['Inputs']['Diagnostics']['avg_zkTperp'])).lower()))             
        if bool(data['Inputs']['Diagnostics']['avg_zkqpar']):
            file.write("\n avg_zkqpar  = {}    # Write the RMS value of the zonal component of the guiding center parallel-parallel heat flux as a function of time".format(str(bool(data['Inputs']['Diagnostics']['avg_zkqpar'])).lower()))             
        pass

    if bool(data['Inputs']['Diagnostics']['all_zonal']):
        file.write("\n all_zonal = {}    # Write quantities such as the zonal component of the ExB velocity as function of x and time".format(str(bool(data['Inputs']['Diagnostics']['all_zonal'])).lower()))
    else:
        if bool(data['Inputs']['Diagnostics']['vEy']):
            file.write("\n vEy    = {}    # Write the zonal component of ExB velocity as a function of x and time".format(str(bool(data['Inputs']['Diagnostics']['vEy'])).lower()))
        if bool(data['Inputs']['Diagnostics']['kxvEy']):
            file.write("\n kxvEy  = {}    # Write the zonal component of the shear of the ExB velocity as a function of x and time".format(str(bool(data['Inputs']['Diagnostics']['kxvEy'])).lower()))    
        if bool(data['Inputs']['Diagnostics']['kden']):
            file.write("\n kden   = {}    # Write the zonal component of the radial gradient of the guiding center density as a function of x and time".format(str(bool(data['Inputs']['Diagnostics']['kden'])).lower()))    
        if bool(data['Inputs']['Diagnostics']['kUpar']):
            file.write("\n kUpar  = {}    # Write the zonal component of the radial gradient of the guiding center parallel velocity as a function of x and time".format(str(bool(data['Inputs']['Diagnostics']['kUpar'])).lower()))    
        if bool(data['Inputs']['Diagnostics']['kTpar']):
            file.write("\n kTpar  = {}    # Write the zonal component of the radial gradient of the guiding center parallel temperature as a function of x and time".format(str(bool(data['Inputs']['Diagnostics']['kTpar'])).lower()))   
        if bool(data['Inputs']['Diagnostics']['kTperp']):
            file.write("\n kTperp = {}    # Write the zonal component of the radial gradient of the guiding center perpendicular temperature as a function of x and time".format(str(bool(data['Inputs']['Diagnostics']['kTperp'])).lower()))             
        if bool(data['Inputs']['Diagnostics']['kqpar']):
            file.write("\n kqpar  = {}    # Write the zonal component of the radial gradient of the guiding center parallel-parallel heat flux as a function of x and time".format(str(bool(data['Inputs']['Diagnostics']['kqpar'])).lower()))             
        pass

    if bool(data['Inputs']['Diagnostics']['all_non_zonal']):
        file.write("\n all_non_zonal = {}    # Write quantities such as the non-zonal y-component of the ExB velocity as function of x and time".format(str(bool(data['Inputs']['Diagnostics']['all_non_zonal'])).lower()))
    else:
        if bool(data['Inputs']['Diagnostics']['xyvEx']):
            file.write("\n xyvEx   = {}    # Write the x-component of the ExB velocity as a function of x, y, and time".format(str(bool(data['Inputs']['Diagnostics']['xyvEx'])).lower()))
        if bool(data['Inputs']['Diagnostics']['xyvEy']):
            file.write("\n xyvEy   = {}    # Write the non-zonal y-component of the ExB velocity as a function of x, y, and time".format(str(bool(data['Inputs']['Diagnostics']['xyvEy'])).lower()))  
        if bool(data['Inputs']['Diagnostics']['xykxvEy']):
            file.write("\n xykvE   = {}    # Write the non-zonal part of the shear of the y-component of the ExB velocity as a function of x and time".format(str(bool(data['Inputs']['Diagnostics']['xykxvEy'])).lower()))  
        if bool(data['Inputs']['Diagnostics']['xyden']):
            file.write("\n xyden   = {}    # Write the non-zonal component of the guiding center density as a function of x and time".format(str(bool(data['Inputs']['Diagnostics']['xyden'])).lower()))    
        if bool(data['Inputs']['Diagnostics']['xyUpar']):
            file.write("\n xyUpar  = {}    # Write the non-zonal component of the guiding center parallel velocity as a function of x and time".format(str(bool(data['Inputs']['Diagnostics']['xyUpar'])).lower()))    
        if bool(data['Inputs']['Diagnostics']['xyTpar']):
            file.write("\n xyTpar  = {}    # Write the non-zonal component of the guiding center parallel temperature as a function of x and time".format(str(bool(data['Inputs']['Diagnostics']['xyTpar'])).lower()))   
        if bool(data['Inputs']['Diagnostics']['xyTperp']):
            file.write("\n xyTperp = {}    # Write the non-zonal component of the guiding center perpendicular temperature as a function of x and time".format(str(bool(data['Inputs']['Diagnostics']['xyTperp'])).lower()))             
        if bool(data['Inputs']['Diagnostics']['xyqpar']):
            file.write("\n xyqpar  = {}    # Write the non-zonal component of the guiding center parallel-parallel heat flux as a function of x and time".format(str(bool(data['Inputs']['Diagnostics']['xyqpar'])).lower()))             
        pass


def write_wspectra(file, data):

    file.write("\n\n[Wspectra]                  # Spectra of W = |G_lm|**2")

    try:
        data['Spectra']['Wst']
        file.write("\n species          = true")
    except:
        file.write("\n species          = false")

    try:
        data['Spectra']['Wmst']
        file.write("\n hermite          = true")
    except:
        file.write("\n hermite          = false")

    try:
        data['Spectra']['Wlst']
        file.write("\n laguerre         = true")
    except:
        file.write("\n laguerre         = false")

    try:
        data['Spectra']['Wlmst']
        file.write("\n hermite_laguerre = true")
    except:
        file.write("\n hermite_laguerre = false")

    try:
        data['Spectra']['Wkxst']
        file.write("\n kx               = true")
    except:
        file.write("\n kx               = false")

    try:
        data['Spectra']['Wkyst']
        file.write("\n ky               = true")
    except:
        file.write("\n ky               = false")

    try:
        data['Spectra']['Wkxkyst']
        file.write("\n kxky             = true")
    except:
        file.write("\n kxky             = false")

    # try:
    #     data['Spectra']['Wkperpst']
    #     file.write("\n kperp            = true")
    # except:
    #     file.write("\n kperp            = false")

    try:
        data['Spectra']['Wkzst']
        file.write("\n kz               = true")
    except:
        file.write("\n kz               = false")

    try:
        data['Spectra']['Wzst']
        file.write("\n z                = true")
    except:
        file.write("\n z                = false")


def write_pspectra(file, data):

    file.write("\n\n[Pspectra]                  # Spectra of P = ( 1 - Gamma_0(b_s) ) |Phi|**2")

    try:
        data['Spectra']['Pst']
        file.write("\n species          = true")
    except:
        file.write("\n species          = false")

    try:
        data['Spectra']['Pkxst']
        file.write("\n kx               = true")
    except:
        file.write("\n kx               = false")

    try:
        data['Spectra']['Pkyst']
        file.write("\n ky               = true")
    except:
        file.write("\n ky               = false")

    try:
        data['Spectra']['Pkxkyst']
        file.write("\n kxky             = true")
    except:
        file.write("\n kxky             = false")

    # try:
    #     data['Spectra']['Pkperpst']
    #     file.write("\n kperp            = true")
    # except:
    #     file.write("\n kperp            = false")

    try:
        data['Spectra']['Pkzst']
        file.write("\n kz               = true")
    except:
        file.write("\n kz               = false")

    try:
        data['Spectra']['Pzst']
        file.write("\n z                = true")
    except:
        file.write("\n z                = false")


def write_qspectra(file, data):

    file.write("\n\n[Qspectra]                  # Spectra of heatflux")

    try:
        data['Spectra']['Qkxst']
        file.write("\n kx               = true")
    except:
        file.write("\n kx               = false")

    try:
        data['Spectra']['Qkyst']
        file.write("\n ky               = true")
    except:
        file.write("\n ky               = false")

    try:
        data['Spectra']['Qkxkyst']
        file.write("\n kxky             = true")
    except:
        file.write("\n kxky             = false")

    # try:
    #     data['Spectra']['Qkperpst']
    #     file.write("\n kperp            = true")
    # except:
    #     file.write("\n kperp            = false")

    try:
        data['Spectra']['Qkzst']
        file.write("\n kz               = true")
    except:
        file.write("\n kz               = false")

    try:
        data['Spectra']['Qzst']
        file.write("\n z                = true")
    except:
        file.write("\n z                = false")


def write_gamspectra(file, data):

    file.write("\n\n[Gamspectra]                # Spectra of particle flux")

    try:
        data['Spectra']['Gamkxst']
        file.write("\n kx               = true")
    except:
        file.write("\n kx               = false")

    try:
        data['Spectra']['Gamkyst']
        file.write("\n ky               = true")
    except:
        file.write("\n ky               = false")

    try:
        data['Spectra']['Gamkxkyst']
        file.write("\n kxky             = true")
    except:
        file.write("\n kxky             = false")

    # try:
    #     data['Spectra']['Gamkperpst']
    #     file.write("\n kperp            = true")
    # except:
    #     file.write("\n kperp            = false")

    try:
        data['Spectra']['Gamkzst']
        file.write("\n kz               = true")
    except:
        file.write("\n kz               = false")

    try:
        data['Spectra']['Gamzst']
        file.write("\n z                = true")
    except:
        file.write("\n z                = false")


def write_phi2spectra(file, data):

    file.write("\n\n[Phi2spectra]               # Spectra of |Phi|**2")

    try:
        data['Spectra']['Phi2t']
        file.write("\n time             = true")
    except:
        file.write("\n time             = false")

    try:
        data['Spectra']['Phi2kxt']
        file.write("\n kx               = true")
    except:
        file.write("\n kx               = false")

    try:
        data['Spectra']['Phi2kyt']
        file.write("\n ky               = true")
    except:
        file.write("\n ky               = false")

    try:
        data['Spectra']['Phi2kxkyt']
        file.write("\n kxky             = true")
    except:
        file.write("\n kxky             = false")

    # try:
    #     data['Spectra']['Phi2kperpst']
    #     file.write("\n kperp            = true")
    # except:
    #     file.write("\n kperp            = false")

    try:
        data['Spectra']['Phi2kzt']
        file.write("\n kz               = true")
    except:
        file.write("\n kz               = false")

    try:
        data['Spectra']['Phi2zt']
        file.write("\n z                = true")
    except:
        file.write("\n z                = false")


def write_forcing(file, data):

    file.write("\n\n[Forcing]")

    if not bool(data['Inputs']['Controls']['Forcing']['forcing_init']):

        file.write("\n forcing_init  = {}       # Whether to include external forcing".format(str(bool(data['Inputs']['Controls']['Forcing']['forcing_init'])).lower()))
        file.write("\n forcing_type  = \"{}\"        # Specifies the forcing options".format(data['Inputs']['Controls']['Forcing']['forcing_type_dum']))
        file.write("\n forcing_amp   = {}         # Amplitude of the forcing".format(data['Inputs']['Controls']['Forcing']['forcing_amp']))
        file.write("\n forcing_index = {}           # Index of the forcing".format(data['Inputs']['Controls']['Forcing']['forcing_index']))
        file.write("\n stir_field    = \"{}\"   # Which moment to force".format(data['Inputs']['Controls']['Forcing']['stir_field_dum']))

        if bool(data['Inputs']['Controls']['Forcing']['no_fields']):
            file.write("\n\n no_fields = \"{}\"    # Field terms are not included in the GK equation".format(str(bool(data['Inputs']['Controls']['Forcing']['no_fields'])).lower()))

    else:
        file.write("\n # No forcing was specified. See documentation for further details.")


def write_expert(file, data):

    file.write("\n\n[Expert]")

    par_count = 0

    if bool(data['Inputs']['Controls']['dealias_kz']):
        file.write("\n dealias_kz  = {}        # Dealias in kz. This is only available for boundary = \"periodic\"".format(str(bool(data['Inputs']['Controls']['dealias_kz'])).lower()))
        par_count += 1

    if bool(data['Dimensions']['local_limit']):
        file.write("\n local_limit = {}        # Run calculation in local limit where kz is a scalar parameter given by kz = 1.0/zp".format(str(bool(data['Dimensions']['local_limit'])).lower()))
        par_count += 1

    if not bool(data['Inputs']['Expert']['i_share'] == 8):
        file.write("\n i_share     = {}            # An integer related to the shared memory block used for the inner loop of the linear solver".format(data['Inputs']['Expert']['i_share']))
        par_count += 1

    if not bool(data['Inputs']['Expert']['nreal'] == 1):
        file.write("\n nreal       = {}            # Enforce the reality condition every nreal timesteps".format(data['Inputs']['Expert']['nreal']))
        par_count += 1

    if bool(data['Inputs']['Expert']['secondary']):
        file.write("\n secondary   = {}        # Perform a secondary calculation".format(str(bool(data['Inputs']['Expert']['secondary'])).lower()))
        par_count += 1

    if not bool(data['Inputs']['Expert']['phi_ext'] == 0.0):
        file.write("\n phi_ext     = {}        # Value of phi to use for a Rosenbluth-Hinton test".format(str(bool(data['Inputs']['Expert']['phi_ext'])).lower()))
        par_count += 1

    if not (str(data['Inputs']['Expert']['source_dum']) == "default" or str(data['Inputs']['Expert']['source_dum']) == "--"):
        file.write("\n source      = \"{}\"    # Used to specify various kinds of tests".format(data['Inputs']['Expert']['source_dum']))
        par_count += 1

    if bool(data['Inputs']['Expert']['init_single']):
        file.write("\n\n init_single = {}    # Only initialize a single Fourier mode".format(str(bool(data['Inputs']['Expert']['init_single'])).lower()))
        file.write("\n ikx_single  = {}        # Index of the kx mode to be initialized".format(data['Inputs']['Expert']['ikx_single']))
        file.write("\n iky_single  = {}        # Index of the ky mode to be initialized".format(data['Inputs']['Expert']['iky_single']))
        par_count += 1

    if bool(data['Inputs']['Expert']['eqfix']):
        file.write("\n\n eqfix     = {}    # Do not evolve some particular Fourier harmonic".format(str(bool(data['Inputs']['Expert']['eqfix'])).lower()))
        file.write("\n ikx_fixed = {}       # Index of the kx mode to be initialized".format(data['Inputs']['Expert']['ikx_fixed']))
        file.write("\n iky_fixed = {}       # Index of the ky mode to be initialized".format(data['Inputs']['Expert']['iky_fixed']))
        par_count += 1 

    if (data['Inputs']['Expert']['t0'] > 0.0):
        file.write("\n\n t0     = {}    # The time at which to start changing tprim".format(data['Inputs']['Expert']['t0']))
        file.write("\n tf     = {}    # The time at which to stop changing tprim".format(data['Inputs']['Expert']['tf']))
        file.write("\n tprim0 = {}    # The time at which to start changing tprim".format(data['Inputs']['Expert']['tprim0']))
        file.write("\n tprimf = {}    # The time at which to start changing tprim".format(data['Inputs']['Expert']['tprimf']))

    if bool(par_count == 0):
        file.write("\n # No expert parameters were specified. See documentation for further details.")


def generate_input_file(simulations):

    print("Generating input files(s).")
    print('─' * 100)
    simulation_iterable = tqdm(simulations.keys(), desc=None, leave=True, unit=" files")
    failed_count = 0
    
    for simulation_key in simulation_iterable:
    
        # Initialising
        simulation = simulations[simulation_key]
        simulation_iterable.set_postfix_str("Failed: {:}".format(failed_count))
        tqdm.write("{}".format(simulation_key.name))

        # Creating file paths and names
        input_filepath = pathlib.Path(simulation_key.parent, simulation_key.with_suffix('.in').name)
        geo_filepath = pathlib.Path(simulation_key.parent, simulation_key.with_suffix('.eik.out').name)
        geo_filename = pathlib.Path(simulation_key.with_suffix('.eik.out').name)
        restart_filename = pathlib.Path(simulation_key.with_suffix('.restart.nc').name)

        if pathlib.Path.exists(input_filepath):
            tqdm.write("Input file already exists. Aborting.")
            failed_count += 1
            simulation_iterable.set_postfix_str("Failed: {:}".format(failed_count))
            tqdm.write('─' * 100)
            continue   

        # Generate input file
        input_file = open(input_filepath, 'a+')
        input_file.write("# This is an input file generated from the output file: {:}\n\n".format(simulation_key))

        # debug is not associated with any input groups, and must be specified at the top of the input file
        input_file.write(" debug = {:}".format(str(bool(simulation['Dimensions']['debug'])).lower()))

        try:
            #Writing each of the input file groups
            write_dimensions(input_file, simulation)
            write_domain(input_file, simulation)
            write_resize(input_file, simulation)
            write_physics(input_file, simulation)
            write_time(input_file, simulation)
            write_initialization(input_file, simulation)
            write_geometry(input_file, simulation, geo_filepath, geo_filename)
            write_species(input_file, simulation)
            write_boltzmann(input_file, simulation)
            write_dissipation(input_file, simulation)
            write_restart(input_file, simulation, restart_filename)
            write_diagnostics(input_file, simulation)
            write_wspectra(input_file, simulation)
            write_pspectra(input_file, simulation)
            write_qspectra(input_file, simulation)
            write_gamspectra(input_file, simulation)
            write_phi2spectra(input_file, simulation)
            # #write_Apar2spectra
            # #write_dBpar2spectra
            write_forcing(input_file, simulation)
            write_expert(input_file, simulation)

        except:
            tqdm.write("Input file creation error. Aborting.")
            input_file.close()
            pathlib.Path.unlink(input_filepath)
            failed_count += 1
            simulation_iterable.set_postfix_str("Failed: {:}".format(failed_count))
            continue

        input_file.close()
        tqdm.write('─' * 100)
    
    simulation_iterable.set_postfix_str("Failed: {:}".format(failed_count))
    print("Completed.")
    print("")

    return


if __name__ == "__main__":

    # Interpreting command line input
    filenames = sys.argv[1:]

    if "latex" in filenames:
        filenames.remove("latex")

    simulations = load_files(filenames, groups = ['Inputs', 'Geometry', 'Special', 'Non_zonal', 'Zonal_x', 'Fluxes', 'Spectra'])

    generate_input_file(simulations)
