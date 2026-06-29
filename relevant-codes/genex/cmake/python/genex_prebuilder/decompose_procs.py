import sys
from math import log

def decompose_procs(n_procs, params_mesh, **kwargs):
    n_procs_left = n_procs
    n_species = kwargs.get('n_species', 2)
    n_points_mu = params_mesh["n_points_mu"]

    assert isinstance(n_procs, int), \
        f"Error: n_procs {n_procs} is not an integer!"
    assert isinstance(n_species, int), \
        f"Error: n_species {n_species} is not an integer!"

    dict_out = {}
    dict_out["params_parallelization"] = {}
    dict_out["params_parallelization"]["n_procs_phi"] = 1
    dict_out["params_parallelization"]["n_procs_vp"]  = 1
    dict_out["params_parallelization"]["n_procs_mu"]  = 1
    dict_out["params_parallelization"]["n_procs_sp"]  = 1

    if n_procs >= n_species:
        n_procs_vp = 1

        n_procs_sp = n_species
        n_procs_left /= n_procs_sp

        if n_points_mu > 2:
            exponent = int(log(n_procs_left, 2))
            y_exp = int(exponent / 2)
            x_exp = exponent - y_exp
            n_procs_phi = 2**x_exp
            n_procs_mu  = 2**y_exp
            n_procs_left /= (n_procs_phi * n_procs_mu)

        else:
            n_procs_mu = 1
            n_procs_phi = int(n_procs_left)
            n_procs_left /= n_procs_phi

        assert (n_procs_left % 1 == 0) and (int(n_procs_left) == 1), \
            "Error: decomposition of the processes is not complete! " + \
            f"Remainder number of processes: {n_procs_left}."

        assert isinstance(n_procs_phi, int), \
            f"Error: n_procs_phi {n_procs_phi} is not an integer!"
        assert isinstance(n_procs_vp, int), \
            f"Error: n_procs_vp {n_procs_vp} is not an integer!"
        assert isinstance(n_procs_mu, int), \
            f"Error: n_procs_mu {n_procs_mu} is not an integer!"
        assert isinstance(n_procs_sp, int), \
            f"Error: n_procs_sp {n_procs_sp} is not an integer!"

        dict_out["params_parallelization"]["n_procs_phi"] = n_procs_phi
        dict_out["params_parallelization"]["n_procs_vp"]  = n_procs_vp
        dict_out["params_parallelization"]["n_procs_mu"]  = n_procs_mu
        dict_out["params_parallelization"]["n_procs_sp"]  = n_procs_sp

    elif n_procs > 1 and n_procs < n_species:
        dict_out["params_parallelization"]["n_procs_phi"] = n_procs

    return dict_out
