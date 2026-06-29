import matplotlib
import sys
from load_files import load_files
from extract_species import extract_species
from plot_spectra_1D import plot_spectra_1D


if __name__ == "__main__":

    # Interpreting command line input
    filenames = sys.argv[1:]

    if "latex" in filenames:
        filenames.remove("latex")

        font = {'family' : 'serif',
        'serif'  : ['Computer Modern Roman'],
        'weight' : 'bold',
        'size'   : 18} # Use 40 with latex for papers, 18 otherwise

        matplotlib.rc('font', **font)
        matplotlib.rc('text', usetex=True)
        matplotlib.rcParams['axes.labelpad']='20'

        print("")
        print("Using LaTeX")

    time_plots = False

    if "time_plots" in filenames:
        filenames.remove("time_plots")
        time_plots = True

    spectra       = ["Wkzst", "Pkzst", "Qkzst", "Gamkzst", "Phi2kzt"]
    spectra_label = [r"$W_s(k_z)$", r"$[1-\Gamma_0(b_s)]|\phi|^2(k_z)$", r"$Q_s(k_z)$", r"$\Gamma_s(k_z)$", r"$|\phi(k_z)|^2$"]

    spectra_plot       = []
    spectra_label_plot = []

    for spectrum_index, spectrum in enumerate(spectra):
        if spectrum in filenames:
            filenames.remove(spectrum)
            spectra_plot.append(spectrum)
            spectra_label_plot.append(spectra_label[spectrum_index])

    if len(spectra_plot) == 0:
        print("")
        print("Please specify one (or more) of the following as a command-line input:")
        print(spectra)
        print("")

        exit()

    else:
        simulations = load_files(filenames, groups=['Inputs', 'Spectra'], spectra=spectra_plot)

        for spectrum_index, spectrum in enumerate(spectra_plot):
            print("Plotting", spectrum, "...")
            plot_spectra_1D(simulations, spectrum=spectra_plot[spectrum_index], spectrum_label=spectra_label_plot[spectrum_index], \
                                spectrum_range='kz', spectrum_range_label=r"$k_z$", spectrum_xscale="log", spectrum_yscale="log", average_fraction=0.5, absolute_value=False, time_plots=time_plots)
