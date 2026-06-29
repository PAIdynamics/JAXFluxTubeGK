"""
Contains unit test definitions for python tools
"""
import pytest
from pathlib import Path
import runpy
import sys
import os
import os.path
import contextlib
import matplotlib

def get_tools():
    """
    Returns a list of all tools to be tested
    """
    # We specify the tools to match the following filters
    filt = ["animate*", "diagnose*", "profiler*"]

    wdir = Path.cwd() / "../"

    files = []
    for f in filt:
        files = files + list(Path(wdir).glob(f+".py"))

    tools = dict()
    for file in files:
        name = file.stem
        tools[name] = file
    return tools

@contextlib.contextmanager
def working_directory(path: Path):
    """
    Changes working directory and returns to previous on exit
    """
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)

@pytest.mark.parametrize("filepath", get_tools().values(), \
                         ids=get_tools().keys())
def test_tool(filepath: Path):
    """
    Runs the python tool given by the filepath. The test checks for execution
    without error
    """
    assert filepath.exists()
    # Add path of execution of the tool to system to allow for using
    # modules and packages within the tools directories
    sys.path.append(os.path.dirname(filepath))

    # Disable opening of mpl windows that wait for user input
    matplotlib.use("Agg")
    with working_directory(filepath.parent):
        # Run test for grid approach
        run_test(filepath, "slab", "grid")
        run_test(filepath, "circular", "grid")
        run_test(filepath, "salpha", "grid")
        # Run test for spectral approach

def run_test(filepath: Path, equi: str, typebase: str):
    """
    Runs a single test for the given equilibrium
    """
    # Tools require path to directory except profiler which requires the file
    rpath = filepath.parent / f"../build-release-intel/mms-testing/{typebase}"\
                            / f"{equi}/resolution_1/cpu"
    arg = str(rpath.resolve()) + "/"
    if "profiler" in str(filepath.stem):
        arg = arg + "job.out"

    sys.argv = ["", arg]
    runpy.run_path(str(filepath))
