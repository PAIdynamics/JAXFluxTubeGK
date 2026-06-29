# Welcome to the official repository of GENE-X!

This README gets you started with the code. For more info and to join the
community, explore our recommended resources listed below.

## Installation guide

**System Requirements and Compilation**

If you're already comfortable with GNU/Unix/Linux systems, you'll likely
find GENE-X easy to compile and install.

**Note for macOS Users**

While it's possible to compile GENE-X on macOS, our support for this
platform is limited. You may encounter issues or require additional
configuration.

### Getting the code

You can clone the GENE-X repository using **Git** and **ssh**
(or https using the corresponding link version of that)

```shell
git clone git@gitlab.mpcdf.mpg.de:phoenix-public/genex.git
cd genex
```

To initialize and update all submodules required by GENE-X, run

```shell
git submodule init
git submodule update
```

### Supported compilers

**Which compiler to choose**

To get started, you'll need to compile the code using a Fortran compiler
that supports features from the 2008 standard. We recommend using either the
Intel Fortran Compiler or the GNU Fortran Compiler. Please try to use the
most recent version of the compiler available.

The supported compiler versions are:

* **GNU (gfortran)**: >=10
* **Intel (ifx)**: >=2025.2
* **Intel (ifort)**: >=18 <br>
  (as of mid 2025 - our active support stopped)

Please note that these versions are subject to change, and we may update
the minimum required versions in the future.

We recommend checking our [recently run pipelines](
https://gitlab.mpcdf.mpg.de/phoenix-public/genex/pipelines)
to see which compiler versions we use routinely.

**Compilers not listed above**

If you want to use a compiler that's not on our list, you may encounter a
warning during the build process. While it may still be possible to build the
code, please note that we cannot guarantee optimal performance or
functionality with all features enabled.

**Note for GPU Users**

The GPU offloading features of GENE-X utilize interoperability between the
main code written in Fortran 2008 and the accelerated compute kernels written
in C++17. To use these features, both a Fortran and a C++ compiler are required.

The supported compiler versions for the GPU build are:

* **GNU (gfortran)**: >=11
* **NVHPC SDK (nvc, nvc++, nvcc)**: >=2022

### Libraries and tools

In order to clone the repository, you'll need to have both Git and Git LFS
installed. We use the CMake build system, which requires a minimum version
of CMake 3.18. Please note that the minimum CMake version may change over
time. Please try to use the most recent version available.

GENE-X depends on the following libraries that need to be installed on your
computer

- MPI
- MKL (for CPU, or optional for GPU)
- HDF5
- NetCDF

Additionally, if you want to use the included tools for quick-analysis of
simulation results, Python is required. The exact package dependencies
will be checked at runtime and is contained in the
[python requirements file](./tools/requirements.in).

### Quick installation

Before you start, please make sure that the necessary dependencies are
installed and/or modules are loaded on your machine.

To install GENE-X, create a build directory and enter that directory:

```shell
mkdir build
cd build
```

Next, run the following commands from the build directory to build GENE-X with
(for example) the GNU compiler

```shell
cmake .. -DCMAKE_Fortran_COMPILER=mpif90 <ADD_OPTIONS>
cmake --build . --parallel --target all
```

If you want to use a different Fortran compiler, you can replace `mpif90`
with the name of your preferred compiler. If you're building GENE-X from
a different directory, you can modify the cmake command to point to the
correct location. Simply replace `..` with the path to the GENE-X repository.

### Automatic build configuration with ConfiX

If you prefer to have more control over your build process, you can use the
above settings to configure your CMake build manually. However, in
most cases, this level of detail is not necessary.

We recommend using the automatic build configuration provided by the
[ConfiX library](https://gitlab.mpcdf.mpg.de/phoenix-public/confix),
which is used in GENE-X.

## Running tests

### Unit testing

**Run a single test**

Unit test executables are generated and placed in the `bin` directory within
the build. On a system with SLURM, you can run any one of them using:

```shell
salloc -n 8 -p <PARTITION> -t <TIME> --mem=32G <OTHER_OPTIONS> <TEST_EXEC_NAME>
```

The time required for each test can vary significantly, depending on the
test and build type (debug or release). Some tests may complete in a few
seconds, while others may take a handful of minutes. Be sure to allocate
sufficient time for each test to ensure it completes successfully. You can have
a look at our
[recently run pipelines](https://gitlab.mpcdf.mpg.de/phoenix-public/genex/pipelines)
to get an idea about typical test execution times.

The number of processes varies from test to test but typically not exceeds 8.
You also need to specify the partition (and, if required, other options
such as qos) and the time.

**Run multiple tests at once**

Multiple tests can be executed using CTest. For example, to test a full
release build (runs within 30min), you can use the following command
(from your build directory):

```shell
salloc -n 8 --mem=32G --time=00:30:00 cmake --test-dir ./src/ --target test
```

When using ConfiX to build the code, a directory `unit-testing` is created
in the build. This directory contains multiple batch scripts that can be
used to start multiple tests at once, making it easier to run and manage
unit tests.

### MMS testing

When you build the code, automated test resources for MMS tests are
generated, provided that Python can be found (e.g. module is loaded) and
the required dependencies are installed (such as Pandas). The test setups
are contained in the `mms-testing/` directory within the build folder.

We offer a range of testing options in MMS, each with its own unique setup.
This includes parameter files, SLURM batch scripts, and reference files.
You can also use our multitask SLURM batch scripts for GPU MMS, deep MMS,
and broad MMS testing, and dedicated shell scripts for result assertion.

### Running the benchmark program

A benchmark program for benchmarking individual compute kernels in the GENE-X
code is provided with an executable `benchmark-operators`. The benchmark
program takes a parameter file for configuration. To get a description of the
available command line arguments, use the option `-h`.

## Running the code

To run GENE-X, simply create a new folder in a directory of your choice.
Make sure you have enough storage space available, especially if you plan
to run large simulations. Next, create a symbolic link to the genex executable,
which can be found in the bin folder inside the build directory,

```shell
ln -s <PATH_TO_BUILD>/bin/genex
```

**Parameter file**

To run a simulation with GENE-X, you'll need to create a parameter file that
specifies the simulation settings. You can use the parameter files from the
MMS tests as a starting point, but be aware that most settings will need to
be adjusted.

For help with creating a parameter file, you can:

- Use the parameter files from the MMS tests as a starting point
  guide on setting up parameter files
- Use parameter files from our publications
- Contact the GENE-X team directly for assistance
  (see [below](#get-in-touch-with-the-development-team))

**Quick run**

Before running your simulation, you can check your parameter file with:

```shell
./genex -c params_in.txt
```

Once you've set up your parameter file and validated it, you can run the
code on a single node using:

```shell
./genex params_in.txt
```

To ensure exclusive access to a node, we recommend to first request
a node allocation with `salloc`. If you want to run with multiple MPI
processes, use:

```shell
srun genex params_in.txt
```

**Run with sbatch**

In general, we recommend running the code with the `sbatch` command utilizing
a submit script. For optimal performance, GENE-X requires machine-specific
parameters to be set when running on a cluster. These parameters are
typically found in the computing center's documentation, along with example
submit scripts. If you're unsure, don't hesitate to reach out to the GENE-X
team for guidance and support
(see [below](#get-in-touch-with-the-development-team)).

You run the code with:

```shell
sbatch submit.sh
```

## Where to get more information

### Code documentation

Code documentation is automatically generated from inline comments
in the source files using the FORD package (similar to Doxygen). This
documentation is hosted on
[Gitlab pages](https://phoenix-public.pages.mpcdf.de/genex/page/index.html).

### The Wiki

The [GENE-X Wiki](https://phoenix-public.pages.mpcdf.de/genex-wiki/) is a
collection of user and developer-specific information.

It features:

- Basic walk-throughs and beginners guides
- Tips and tricks for running the code
- Valuable details shared by users and developers

### Get in touch with the development team

If you need further assistance or have specific questions about the code,
don't hesitate to reach out to us. You can reach us via email through our
website contact form.

## Acknowledging the GENE-X team

### Citing GENE-X Papers

If you've found the GENE-X code helpful, we would be very happy if you
could cite our papers in any publication or presentation that you make!
This helps us to share our research with a wider audience and to continue
improving the code. For easy identification, please include the name `GENE-X`
in your publications.

For your convenience, we've compiled a list of our papers on our
website. You can also find a
[BibTeX file](./doc/genex.bib) in this repository, which you can easily
import into your LaTeX projects. If you're unsure what to cite, check out our
[citations guide](./doc/citations.md) for suggested references on specific
topics.

### Using the GENE-X logo

To acknowledge the GENE-X project in your work, you can use our
[code logo](./img/genex.png) in your presentations, posters, or publications.
This helps promote the project and recognize the contributions of our
developers.

<img src="./img/genex.png" alt="drawing" width="300"/>

## Community engagement

If you are interested in participating in any form of community engagement,
such as contributing to the code, attending workshops, or joining online
discussions, please contact us at the email address listed
[above](#get-in-touch-with-the-development-team).

### Stay up-to-date with GENE-X developments

Want to stay informed about the latest code developments and new releases?
We'd be happy to add you to our users mailing list, which broadcasts news and
updates about new and interesting activities in the GENE-X community.

### Becoming a proficient user

While we strive to make the GENE-X code accessible to a wide range of users,
mastering its full potential can be a challenging task. To help bridge this
gap, we offer extended user training opportunities, ranging from brief
introductions to more in-depth on-site training sessions.

If you're interested in participating in an extended user training activity,
please don't hesitate to reach out to us. We'll be happy to discuss your
needs and provide more information on our training options.

### Contributing to GENE-X

We're always eager to hear your suggestions on how to improve the code or
add new features. If you're interested in contributing to the GENE-X project,
please don't hesitate to get in touch with the development team.

**Ways to contribute**

- Create merge requests on the release version of the project
- Create a feature request in the Gitlab issues (using the "feature request"
  tag)
- Join our regular meetings and/or workshops (announced via mailing list)
- Join the development team and contribute to the project directly

## Versioning

### Versioning scheme

The project follows a year-based versioning scheme, where each version is
denoted by a year and a minor version number (e.g., 2026.1).

* The year (major version) indicates the release year and may include
significant changes, including potential breaking changes.

* The minor version number is incremented sequentially for each release and
typically includes small changes, such as bug fixes.

### Release policy

Major version updates (e.g., 2026 to 2027) may introduce breaking changes,
but this is not always the case. Release notes will be provided to clarify
whether a major release is breaking or not.

Minor version updates (e.g., 2026.1 to 2026.2) are intended to be backwards
compatible and will not introduce breaking changes, except in rare cases
where it is strictly necessary to fix a critical bug or ensure functional
integrity.

**New features and major code changes will only be introduced when the major
version number is updated.**

### Recommendations

* We recommend always using the latest minor version of the code.

* It is beneficial to use the latest major version of the code available.

* For developers, we advise frequently updating to new major versions.

* For users, we recommend adopting the newest version for **new** simulations.

## License

### License information

The project is licensed under the Mozilla Public License 2.0 (MPL 2.0). While
a license notice may not be present in every individual file, the MPL 2.0
applies to all files in this repository. If you plan to use this repository,
please familiarize yourself with the terms of this [license](LICENSE).

If you have any questions about this license, please consider the official
[MPL 2.0 FAQ](https://www.mozilla.org/en-US/MPL/2.0/FAQ/) or contact us at
the email address listed [above](#get-in-touch-with-the-development-team).

### Non-legally-binding and non-exhaustive summary

You only need to comply with the license requirements when you distribute
the software to others. MPL 2.0 is a "weak copyleft" license: any
modifications to existing files, as well as the original files provided here,
must remain under MPL 2.0. However, you can add new independent files under a
different license (with an explicit license notice for each file) or link
to external libraries that have a different license. The source code of all
MPL 2.0 files you distribute must be shared; files under other licenses may
not require this. You must keep all copyright, license, and attribution
notices in MPL 2.0-covered files. When you distribute the MPL 2.0 code, whether
unmodified or as part of your project, all MPL 2.0 source files must also be
provided.

### Recommendations

* If you have modified GENE-X as part of your research, we strongly recommend
making your modified version publicly available alongside your publication.

* Even if you have not modified GENE-X, always specify the exact version
used in your research.

Note that the MPL 2.0 only requires you to share modified source code if you
distribute the software itself, not your results or documents. However,
**sharing your modifications is good scientific practice and contributes
to the broader research community**.
