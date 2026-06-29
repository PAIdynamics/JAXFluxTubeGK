from pathlib import Path
from shutil import copy2 as fcopy
from os.path import isfile

def create_mms_dir_tree(mms_config, source_dir, destination_dir,
                        geometries, n_resolutions, runtime_modes):

    subdir  = mms_config["subdir"]
    ref_dir = mms_config["reference_dir"]

    for geom in geometries:
        for res in range(1, n_resolutions + 1):
            for mode in runtime_modes:
                Path(destination_dir + geom + "/resolution_" \
                    + str(res) + "/" + mode).mkdir(parents=True, exist_ok=True)

            # Copy reference files to the build directory
            ref_src = source_dir + "/" + ref_dir + "/ref_" + subdir + "_" \
                    + geom + "_resolution" + str(res) + ".out"
            ref_dst = destination_dir + geom + "/resolution_" \
                    + str(res) + "/reference.out"
            if isfile(ref_src):
                fcopy(ref_src, ref_dst)
    return None
