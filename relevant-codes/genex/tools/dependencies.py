from sys import exit
from pkg_resources import require, VersionConflict, get_distribution, \
                          parse_requirements

class bcolors:
    # Color settings for bash output
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_dependencies():
    return ["matplotlib >= 3.3.0", "xarray >= 0.19.0"]

def assert_package(dependency):
    # Check if a package meets the version requirements
    try:
        require(dependency)
    except VersionConflict as err_msg:
        dep = list(parse_requirements(dependency))[0]
        print(bcolors.FAIL + dependency + " is required!" + \
              " Current version: " + get_distribution(dep.name).version + \
              bcolors.ENDC)
        exit(0)

def check_dependency(dependency):
    # A wrapper to check version requirements of a package
    assert_package(dependency)
    print(dependency + ": " + bcolors.OKGREEN + "OK!" + bcolors.ENDC)

def check_dependencies():
    # A wrapper to check version requirements of all common packages
    for dependency in get_dependencies():
        check_dependency(dependency)
    print(bcolors.OKGREEN + "All common dependencies have been checked."+ bcolors.ENDC)
