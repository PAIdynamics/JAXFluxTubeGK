cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(GENEX-Superbuild-Project
  LANGUAGES Fortran C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# To avoid that the utests are linked explicitly against libnvc etc.
# when using the Fortran linker, we have to unset the implicit libraries.
# Otherwise the acc calls are fulfilled by libnvc instead of the acc libraries
# leading to errors with GPU usage.
unset(CMAKE_C_IMPLICIT_LINK_LIBRARIES)

include(GNUInstallDirs)

# Extend CMake module path (contains Find-modules, patches, macros, etc.)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Load consistent ConfiX version
include(LoadConfiX)
load_confix(DEFAULT_PATH "${PROJECT_SOURCE_DIR}/confix/cmake")

# [ConfiX] Include all macros and functions from ConfiX
include(AllConfiXMacros)

# Display all variables including passed ones (blue)
option(VARS_VERBOSE "Prints all CMake variables" OFF)
if(VARS_VERBOSE)
  get_cmake_property(vars CACHE_VARIABLES)
  foreach(var IN LISTS vars)
    get_property(value CACHE "${var}" PROPERTY VALUE)
    if(NOT var MATCHES "^(CMAKE|_)")
      message(STATUS "  ${BoldBlue}${var} = ${value}${ColourReset}")
    else()
      message(STATUS "  ${var} = ${value}")
    endif()
  endforeach()
endif()

# [ConfiX] Set more readable application name
set_application_name("GENE-X")

# [ConfiX] Print a title banner for output readibility
print_title_banner()

# [ConfiX] Print the loaded modules (toolchain) to toolchain.sh
print_toolchain()

# [ConfiX] Automatically detect known machine
detect_machine()
set(MKL_INTERFACE lp64)
if(CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
    # Classic Intel Fortran (ifort)
    if (CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 18.0)
        message(FATAL_ERROR "ifort version must be at least 18.0!")
    endif()
    set(MKL_THREADING intel_thread)

elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "IntelLLVM")
    # Modern LLVM based Intel Fortran (ifx)
    if (CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 2025.0)
        message(FATAL_ERROR "ifx (IntelLLVM) version must be at least 2025.0!")
    endif()
    set(MKL_THREADING intel_thread)

elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 10.0)
        message(FATAL_ERROR "gfortran version must be at least 10.0!")
    endif()
    set(MKL_THREADING gnu_thread)

elseif(CMAKE_Fortran_COMPILER_ID STREQUAL "LLVMFlang")
    cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
    set(MKL_THREADING intel_thread)

else()
    message(WARNING "Fortran compiler chosen is not supported! "
                    "Compiler identification: ${CMAKE_Fortran_COMPILER_ID}")
endif()

# [ConfiX] Define configurations for CPU optimization based on the machine
configure_cpu()

if(ENABLE_GPU)
  if(CMAKE_Fortran_COMPILER_ID MATCHES GNU AND
     CMAKE_CXX_COMPILER_ID MATCHES Clang AND NOT DEFINED ENV{AMDLLVM_HOME})
    message(WARNING "FortranCInterface_VERIFY(CXX) is skipped due to known "
                    "CMake issue when using gfortran and bare clang++ "
                    "together. Build proceeds in user-risk mode!")
  else()
    include(FortranCInterface)
    FortranCInterface_VERIFY(CXX)
  endif()

  # [ConfiX] Define configurations for GPU offloading features
  # based on the machine
  configure_gpu()

elseif(ENABLE_PARALLAX_GPU)
  configure_gpu()

endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

#-----------------------------------------------------------------------------

# GENEX-Superbuild-Project specific options

#---------------------------------------------------------

option(PREBUILDER_REQUIRED "Requires successful installation of prebuilder" OFF)

# Switch for unit tests
option(ENABLE_UTESTS "Enable Unit-Tests with pFUnit" ON)

# Switch for compiling with the GPU features
option(ENABLE_GPU "Enable GPU offloading build" OFF)

# Switch for compiling with the PARALLAX GPU features
if(ENABLE_GPU)
  option(ENABLE_PARALLAX_GPU "Enable GPU offloading build" ON)
else()
  option(ENABLE_PARALLAX_GPU "Enable GPU offloading build" OFF)
endif()

# Non-overridable switch off for PARALLAX GPU build feature in TOK cluster
# as temporary solution
if(MACHINE_NAME MATCHES "TOK")
  message(WARNING "Disable PARALLAX GPU features (PAccX) build in IPP TOK!")
  set(ENABLE_PARALLAX_GPU OFF)
endif()

# Switch for GPU offloading backend
if(ENABLE_GPU)
  # Switch for GPU offloading backend in GENE-X
  if(NOT DEFINED OFFLOAD_BACKEND)
    set(OFFLOAD_BACKEND "multicore")
    message(WARNING "-DOFFLOAD_BACKEND is not defined. \"multicore\" is used as"
                    " default.")
  else()
    string(REPLACE "," ";" OFFLOAD_BACKEND "${OFFLOAD_BACKEND}")
    list(APPEND _available_offload_backend "all" "multicore" "openacc" "openmpx"
                                           "cuda")

    foreach(_backend ${OFFLOAD_BACKEND})
      if(NOT ${_backend} IN_LIST _available_offload_backend)
        message(FATAL_ERROR "Unknown GPU offloading backend: "
                            "\"${_backend}\"! Allowed backends are: "
                            "${_available_offload_backend}.")
      endif()
    endforeach()
  endif()

elseif(DEFINED OFFLOAD_BACKEND AND NOT OFFLOAD_BACKEND STREQUAL "multicore")
  message(FATAL_ERROR "Turn on -DENABLE_GPU to specify -DOFFLOAD_BACKEND!")

endif()

# Switch for GPU offloading backend in PARALLAX
if(ENABLE_PARALLAX_GPU)
  if(NOT DEFINED PARALLAX_OFFLOAD_BACKEND)
    set(PARALLAX_OFFLOAD_BACKEND "multicore")
    message(WARNING "-DPARALLAX_OFFLOAD_BACKEND is not defined. "
                    "\"multicore\" is used as default.")
  else()
    string(REPLACE "," ";" PARALLAX_OFFLOAD_BACKEND
                   "${PARALLAX_OFFLOAD_BACKEND}")
    list(APPEND _available_parallax_offload_backend "multicore" "cuda" "hip"
      "amgx" "rocalution" "hypre")

    foreach(_backend ${PARALLAX_OFFLOAD_BACKEND})
      if(NOT ${_backend} IN_LIST _available_parallax_offload_backend)
        message(FATAL_ERROR
          "Unknown PARALLAX GPU offloading backend: \"${_backend}\"! "
          "Allowed backends are: ${_available_parallax_offload_backend}.")
      endif()
    endforeach()
  endif()

elseif(DEFINED PARALLAX_OFFLOAD_BACKEND AND
       NOT PARALLAX_OFFLOAD_BACKEND STREQUAL "multicore")
  message(FATAL_ERROR "Turn on -DENABLE_PARALLAX_GPU to specify "
                      "-DPARALLAX_OFFLOAD_BACKEND!")
endif()

# Switch for MKL (default: ON)
# NOTE: Only disable for experimental non-production build
option(ENABLE_MKL "Enable MKL" ON)

# Switch for profiler or other HPC tools (default: native)
list(APPEND _supported_profilers "native" "nvtx" "roctx" "dlb" "extrae" "caliper")

if(NOT DEFINED PROFILER_TOOL)
  # Use only the GENE-X native profiling tool by default
  set(PROFILER_TOOL "native" CACHE STRING "Profiler tool used in GENE-X")

elseif(DEFINED PROFILER_TOOL AND ENABLE_GPU)
  # Find NVIDIA Tools Extension (NVTX)
  if(PROFILER_TOOL MATCHES "nvtx")
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES NVHPC)
      message(FATAL_ERROR "-DPROFILER_TOOL=\"nvtx\" cannot be used without "
                          "using NVHPC SDK C++ compiler!")
    endif()

  # Find ROC-tracer and ROC Tools Extension (ROC-TX)
  elseif(PROFILER_TOOL MATCHES "roctx")
    find_package(rocprofiler-sdk-roctx REQUIRED)
    #find_package(ROCM REQUIRED)

  # Find Dynamic Load Balancing (DLB) tool
  elseif(PROFILER_TOOL MATCHES "dlb")
    set(DLB_DIR "" CACHE STRING "Directory of DLB installation")
    find_package(DLB REQUIRED)

  # Find Extrae tool
  elseif(PROFILER_TOOL MATCHES "extrae")
    set(EXTRAE_DIR $ENV{EXTRAE_HOME}
      CACHE STRING "Directory of Extrae installation")
    find_package(EXTRAE REQUIRED)

  # Find Caliper tool
  elseif(PROFILER_TOOL MATCHES "caliper")
    set(caliper_DIR "" CACHE STRING "Directory of Caliper installation")
    find_package(caliper REQUIRED)

  elseif(NOT PROFILER_TOOL IN_LIST _supported_profilers)
    message(WARNING "Unsupported profiler: \"${PROFILER_TOOL}\"! "
                    "Supported profilers are: ${_supported_profilers}.")
  endif()

elseif(NOT PROFILER_TOOL STREQUAL "native")
  message(FATAL_ERROR "Profiling features with \"${PROFILER_TOOL}\" "
                      "require the C++ layer: -DENABLE_GPU=ON")
endif()

# Switch for communication framework backend (default: mpi)
if(ENABLE_GPU)
  list(APPEND _supported_comm_backend "mpi" "mpigpu" "nccl" "mpigpu,nccl")

  if(NOT DEFINED ENV{GENEX_COMM_BACKEND})
    if(NOT DEFINED COMM_BACKEND)
      set(COMM_BACKEND "mpi"
          CACHE STRING "Communication backend used in GENE-X")
    endif()
  else()
    set(COMM_BACKEND "$ENV{GENEX_COMM_BACKEND}"
        CACHE STRING "Communication backend used in GENE-X")
  endif()

  if(COMM_BACKEND STREQUAL "mpi")
    message(STATUS "${BoldYellow}Indirect GPU-CPU-CPU-GPU MPI communications "
                   "are used.${ColourReset}")

  elseif(COMM_BACKEND STREQUAL "mpigpu")
    EXECUTE_PROCESS (COMMAND ompi_info --parsable --all
                     COMMAND "grep" "mpi_built_with_cuda_support:value"
                     OUTPUT_VARIABLE OMPI_INFO_OUTPUT
                     ERROR_QUIET)

    if(DEFINED OMPI_INFO_OUTPUT AND OMPI_INFO_OUTPUT MATCHES "true")
      list(APPEND GENEX_CXX_COMPILE_DEFINITIONS "-DENABLE_COMM_MPI_GPU")
      message(STATUS "${BoldGreen}Built-in GPU-enabled feature in the MPI "
                     "library is used to enable direct GPU-GPU "
                     "MPI communications.${ColourReset}")
    else()
      set(COMM_BACKEND "mpi")
      message(STATUS "${BoldYellow}GPU-aware MPI is not supported!"
                     "${ColourReset}")
      message(STATUS "${BoldYellow}Indirect GPU-CPU-CPU-GPU MPI communications "
                     "are used.${ColourReset}")
    endif()

  elseif(COMM_BACKEND STREQUAL "nccl")
    if(GPU_VENDOR MATCHES NVIDIA)
      list(APPEND GENEX_CXX_COMPILE_DEFINITIONS "ENABLE_COMM_NCCL")
      message(STATUS "${BoldGreen}NCCL is used for collective communications."
                     "${ColourReset}")
    elseif(GPU_VENDOR MATCHES AMD)
      list(APPEND GENEX_CXX_COMPILE_DEFINITIONS "ENABLE_COMM_NCCL")
      message(STATUS "${BoldGreen}RCCL is used for collective communications."
                     "${ColourReset}")
    else()
      set(COMM_BACKEND "mpi")
      message(STATUS "${BoldYellow}NCCL is not supported!${ColourReset}")
      message(STATUS "${BoldYellow}Indirect GPU-CPU-CPU-GPU MPI communications "
                     "are used.${ColourReset}")
    endif()

  elseif(COMM_BACKEND STREQUAL "mpigpu,nccl")
    EXECUTE_PROCESS (COMMAND ompi_info --parsable --all
                     COMMAND "grep" "mpi_built_with_cuda_support:value"
                     OUTPUT_VARIABLE OMPI_INFO_OUTPUT
                     ERROR_QUIET)

    if(DEFINED OMPI_INFO_OUTPUT AND OMPI_INFO_OUTPUT MATCHES "true" AND
       CMAKE_CXX_COMPILER_ID MATCHES NVHPC)
      list(APPEND GENEX_CXX_COMPILE_DEFINITIONS "-DENABLE_COMM_MPI_GPU")
      list(APPEND GENEX_CXX_COMPILE_DEFINITIONS "-DENABLE_COMM_NCCL")
      message(STATUS "${BoldYellow}GPU-aware MPI is used for point-to-point MPI "
                     "communications and NCCL is used for collective "
                     "communications. NOTE: This is still experimental!"
                     "${ColourReset}")
    else()
      set(COMM_BACKEND "mpi")
      message(STATUS "${BoldYellow}GPU-aware MPI or/and NCCL is not supported!"
                     "${ColourReset}")
      message(STATUS "${BoldYellow}Indirect GPU-CPU-CPU-GPU MPI communications "
                     "are used.${ColourReset}")
    endif()

  else()
    message(WARNING "Unsupported profiler: \"${PROFILER_TOOL}\"! "
                    "Supported profilers are: ${_supported_profilers}.")

  endif()
endif()

# Switch for build with the latest PARALLAX version
option(ENABLE_LATEST_PARALLAX
       "Use the latest commit of master branch of PARALLAX" OFF)

option(ENABLE_SHARED_GENEX_LIB "Enable shared library for GENEX" ON)
get_property(ARE_SHARED_LIBS_SUPPORTED GLOBAL
            PROPERTY TARGET_SUPPORTS_SHARED_LIBS)
if(NOT ARE_SHARED_LIBS_SUPPORTED AND ENABLE_SHARED_GENEX_LIB)
  message(WARNING "The target platform does not support dynamic linking, "
                  "we turn ENABLE_SHARED_GENEX_LIB off.")
  set(ENABLE_SHARED_GENEX_LIB OFF CACHE BOOL
    "Enable shared library for GENEX" FORCE)
  cmake_print_variables(ENABLE_SHARED_GENEX_LIB)
endif()

if(NOT PRECISION)
  set(PRECISION "double" CACHE STRING
    "PRECISION chosen by the user at CMake configure time")
endif()
list(APPEND _available_precisions "single" "double")
if(NOT PRECISION IN_LIST _available_precisions)
  message(FATAL_ERROR "Unknown precision: \"${PRECISION}\"! "
                      "Allowed precisions are: ${_available_precisions}.")
endif()
#[[Set_property has a nice effect that when configuring the project
using cmake-gui,
the user is presented with a selection field for FEATURES listing all
available fftlibs that we have defined already.]]
set_property(CACHE PRECISION PROPERTY STRINGS ${_available_precisions})

#------------- Compiler Related Options -----------------------------------

option(COMPILER_REPORTS "Enable Compiler Reports" OFF)
option(PRODUCTION_RUN "Set Production Run" OFF)
option(COVERAGE "Switch on Coverage Reporting" OFF)

if(NOT OPTLEVEL)
  set(OPTLEVEL "3" CACHE STRING
    "OPTLEVEL chosen by the user at CMake configure time")
endif()

list(APPEND _available_optlevels "1" "2" "3")

if(NOT OPTLEVEL IN_LIST _available_optlevels)
  message(FATAL_ERROR "Unknown optimization level: \"${OPTLEVEL}\"! "
                      "Allowed levels are: ${_available_optlevels}.")
endif()

set_property(CACHE OPTLEVEL PROPERTY STRINGS ${_available_optlevels})

#----------------------- CMake build specific options -----------------------

if(NOT INSTALL_OUTPUT)
  # by default -- Up-to-date: messages are ignored
  set(INSTALL_OUTPUT "LAZY" CACHE STRING
    "INSTALL_OUTPUT chosen by the user at CMake configure time")
endif()

# INSTALL_OUTPUT controls which messages are printed
# check CMake documentation (CMAKE_INSTALL_MESSAGE)
list(APPEND _available_install_outputs "ALWAYS" "LAZY" "NEVER")

set_property(CACHE INSTALL_OUTPUT PROPERTY STRINGS
            ${_available_install_outputs})

set(CMAKE_INSTALL_MESSAGE ${INSTALL_OUTPUT})

#-----------------------------------------------------------------------------

# setup compiler flags and primary GENEX definitions

if(PRECISION STREQUAL "double")
  set(GENEX_COMPILE_DEFINITIONS "-DDOUBLE_PREC")
endif()

if(ENABLE_GPU)
  list(APPEND GENEX_COMPILE_DEFINITIONS "ENABLE_GPU")
  set(GENEX_GPU_MESSAGE "multicore (CPU)")

  foreach(_backend ${OFFLOAD_BACKEND})
    # OpenACC
    if(${_backend} STREQUAL "openacc" OR ${_backend} STREQUAL "all")
      if(CMAKE_Fortran_COMPILER_ID MATCHES Intel)
        message(FATAL_ERROR "GPU offload backend with OpenACC is not supported"
                            " by Intel Fortran!")
      endif()

      if(NOT CMAKE_CXX_COMPILER_ID MATCHES NVHPC)
        message(FATAL_ERROR "GPU offload backend with OpenACC is only supported"
                            " by NVHPC C++ compiler!")
      endif()

      option(ENABLE_OPENACC "Enable_OPENACC" ON)
      list(APPEND GENEX_COMPILE_DEFINITIONS "ENABLE_OPENACC")
      set(GENEX_GPU_MESSAGE "${GENEX_GPU_MESSAGE}, OpenACC")
    endif()

    # OpenMP offload
    if(${_backend} STREQUAL "openmpx" OR ${_backend} STREQUAL "all")
      if(CMAKE_Fortran_COMPILER_ID MATCHES Intel)
        message(FATAL_ERROR "GPU offload backend with OpenMP offload is"
                            " not supported by Intel Fortran!")
      endif()

      set(OPENMPX_COMPILER_IDs NVHPC Clang)
      if(NOT CMAKE_CXX_COMPILER_ID IN_LIST OPENMPX_COMPILER_IDs)
        message(FATAL_ERROR "GPU offload backend with OpenMP offload is only"
                            " supported by NVHPC and Clang C++ compiler!")
      endif()

      option(ENABLE_OPENMPX "Enable_OPENMPX" ON)
      list(APPEND GENEX_COMPILE_DEFINITIONS "ENABLE_OPENMPX")
      set(GENEX_GPU_MESSAGE "${GENEX_GPU_MESSAGE}, OpenMP offload")
    endif()

    # CUDA
    if(${_backend} STREQUAL "cuda" OR ${_backend} STREQUAL "all")
      option(ENABLE_CUDA "Enable_CUDA" ON)
      list(APPEND GENEX_COMPILE_DEFINITIONS "ENABLE_CUDA")
      set(GENEX_GPU_MESSAGE "${GENEX_GPU_MESSAGE}, CUDA")
    endif()

  endforeach()

  message(STATUS "${BoldGreen}GENE-X GPU offloading features will be built "
                   "with ${GENEX_GPU_MESSAGE} backend/s${ColourReset}")
endif()

if(ENABLE_PARALLAX_GPU)
  list(APPEND GENEX_COMPILE_DEFINITIONS "ENABLE_PARALLAX_GPU")
endif()

include(SetFortranCompilerFlags)
# set Fortran_FLAGS, Fortran_FLAGS_DEBUG, Fortran_FLAGS_RELEASE
set_fortran_compiler_flags(
  ${PRECISION} ${OPTLEVEL} ${CHIP} ${PRODUCTION_RUN} ${COMPILER_REPORTS}
  ${COVERAGE})

if(ENABLE_GPU OR ENABLE_PARALLAX_GPU)
  include(SetCXXCompilerFlags)
  # set CXX_FLAGS, CXX_FLAGS_DEBUG, CXX_FLAGS_RELEASE
  set_cxx_compiler_flags(
    ${PRECISION} ${OPTLEVEL} ${CHIP} ${PRODUCTION_RUN} ${COMPILER_REPORTS}
    ${COVERAGE})
endif()

if(ENABLE_CUDA OR "cuda" IN_LIST PARALLAX_OFFLOAD_BACKEND)
  include(EnableCUDA)
  enable_language_cuda()
  include(SetCUDACompilerFlags)
  # set CUDA_FLAGS, CUDA_FLAGS_DEBUG, CUDA_FLAGS_RELEASE
  set_cuda_compiler_flags(
    ${PRECISION} ${OPTLEVEL} ${CHIP} ${PRODUCTION_RUN} ${COMPILER_REPORTS}
    ${COVERAGE})
endif()

message(STATUS "${BoldGreen}GENEX compile definitions: "
               "${GENEX_COMPILE_DEFINITIONS}${ColourReset}")

#-----------------------------------------------------------------------------

# Install GENE-X prebuilder python tools
include(InstallGENEXPrebuilderTools)
install_genex_prebuilder_tools(${PREBUILDER_REQUIRED})

#-----------------------------------------------------------------------------

# Set directory structure for subprojects, installation of base and subprojects

include(ExternalProject)
# set:
# CMAKE_ARCHIVE_OUTPUT_DIRECTORY
# CMAKE_LIBRARY_OUTPUT_DIRECTORY
# CMAKE_RUNTIME_OUTPUT_DIRECTORY
# CMAKE_Fortran_MODULE_DIRECTORY
# INSTALL_LIBDIR
# INSTALL_BINDIR
# INSTALL_INCLUDEDIR
# INSTALL_CMAKEDIR
# INSTALL_MODDIR
set(INSTALL_MODDIR mod)
set(INSTALL_CMAKEDIR cmake)
set(CMAKE_Fortran_MODULE_DIRECTORY
  ${CMAKE_BINARY_DIR}/mod)

#-----------------------------------------------------------------------------

if(ENABLE_UTESTS)
  set(MPIEXEC_EXECUTABLE "srun")

  if(DEFINED ENV{PFUNIT_HOME})
    message(STATUS "PFUNIT_HOME is defined to be $ENV{PFUNIT_HOME}")
    set(PFUNIT_DIR $ENV{PFUNIT_HOME}/cmake)
    find_package(PFUNIT 4.6.3 REQUIRED)
  endif()
  if(NOT PFUNIT_FOUND)
    message(STATUS "pFUnit can not be found, hence downloading and building it from scratch.")
    include(FetchContent)
    FetchContent_Declare(
      PFUNIT
      GIT_REPOSITORY https://github.com/Goddard-Fortran-Ecosystem/pFUnit.git
      GIT_TAG v4.10.0
      GIT_SHALLOW    TRUE
      GIT_SUBMODULES_RECURSE TRUE
      )

    set(SKIP_FHAMCREST YES)
    if(CPU_VENDOR MATCHES "AMD")
      set(PFUNIT_MPI_USE_MPIEXEC "mpirun")
    else()
      set(PFUNIT_MPI_USE_MPIEXEC "/usr/bin/srun")
    endif()
    FetchContent_MakeAvailable(PFUNIT)
    message(STATUS "main CMakeLists: PFUNIT_MPI_FOUND=${PFUNIT_MPI_FOUND}")
  endif()

  message(STATUS "using pFUnit with PFUNIT_DIR=${PFUNIT_DIR}")
  include(CTest)
  set(pFUnit4-Project-var PFUNIT::pfunit)
endif()

message(STATUS "CMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}")

set(PARALLAX_FOUND FALSE)
set(PARALLAX_HOME "" CACHE FILEPATH "Directory where PARALLAX can be found")
if (NOT ${PARALLAX_HOME} STREQUAL "")
  set(PARALLAX_DIR ${PARALLAX_HOME}/share/cmake/PARALLAX)
  message(STATUS "PARALLAX_DIR=${PARALLAX_DIR}")
  find_package(PARALLAX CONFIG QUIET)
endif()
if (NOT ${PARALLAX_FOUND})
  message(STATUS "${BoldYellow}PARALLAX was not found, "
                 "hence building from scratch${ColourReset}")

  if(ENABLE_PARALLAX_GPU)
    # Switches related to the GPU features of PARALLAX
    set(PARALLAX_GPU_MESSAGE "multicore (CPU)")
    option(PARALLAX_ENABLE_PACCX        "Enable PARALLAX C++ backend"                   ON)
    option(PACCX_ENABLE_CODE            "Enable base C++ code of PARALLAX"              ON)
    option(PACCX_ENABLE_DOC             "Enable documentation in PARALLAX C++"          OFF)
    option(PACCX_ENABLE_MKL_DIRECT      "Enable MKL direct solver in PARALLAX C++"      OFF)
    option(PACCX_ENABLE_CUSPARSE_DIRECT "Enable CUSPARSE direct solver in PARALLAX C++" OFF)
    option(PACCX_ENABLE_SLU_DIRECT      "Enable SuperLU direct solver in PARALLAX C++"  ON)
    option(PACCX_ENABLE_TESTING         "Enable testing in PARALLAX C++"                OFF)
    option(PACCX_ENABLE_DEBUG_MESSAGES  "Enable debug messages in PARALLAX C++"         OFF)

    if(PROFILER_TOOL MATCHES "nvtx")
      option(PACCX_ENABLE_NVTX "Enable NVTX annotation in PARALLAX C++" ON)
    endif()

    if("cuda" IN_LIST PARALLAX_OFFLOAD_BACKEND)
      set(PARALLAX_GPU_MESSAGE "${PARALLAX_GPU_MESSAGE}, CUDA")
      option(PACCX_ENABLE_CUDA       "Enable CUDA code of PARALLAX C++"         ON)
      option(PACCX_ENABLE_HIP        "Enable HIP code of PARALLAX C++"          OFF)
      option(PACCX_ENABLE_ROCALUTION "Enable RocALUTION solver in PARALLAX C++" OFF)
      option(PACCX_ENABLE_AMGX       "Enable AMGX solver in PARALLAX C++"       OFF)
      option(PACCX_ENABLE_HYPRE      "Enable HYPRE in PARALLAX C++"             OFF)
    elseif("hip" IN_LIST PARALLAX_OFFLOAD_BACKEND)
      set(PARALLAX_GPU_MESSAGE "${PARALLAX_GPU_MESSAGE}, HIP")
      option(PACCX_ENABLE_CUDA       "Enable CUDA code of PARALLAX C++"         OFF)
      option(PACCX_ENABLE_HIP        "Enable HIP code of PARALLAX C++"          ON)
      option(PACCX_ENABLE_ROCALUTION "Enable RocALUTION solver in PARALLAX C++" OFF)
      option(PACCX_ENABLE_AMGX       "Enable AMGX solver in PARALLAX C++"       OFF)
      option(PACCX_ENABLE_HYPRE      "Enable HYPRE in PARALLAX C++"             OFF)
    elseif("amgx" IN_LIST PARALLAX_OFFLOAD_BACKEND)
      set(PARALLAX_GPU_MESSAGE "${PARALLAX_GPU_MESSAGE}, AMGX")
      option(PACCX_ENABLE_CUDA       "Enable CUDA code of PARALLAX C++"         ON)
      option(PACCX_ENABLE_HIP        "Enable HIP code of PARALLAX C++"          OFF)
      option(PACCX_ENABLE_ROCALUTION "Enable RocALUTION solver in PARALLAX C++" OFF)
      option(PACCX_ENABLE_AMGX       "Enable AMGX solver in PARALLAX C++"       ON)
      option(PACCX_ENABLE_HYPRE      "Enable HYPRE in PARALLAX C++"             OFF)
    elseif("rocalution" IN_LIST PARALLAX_OFFLOAD_BACKEND)
      set(PARALLAX_GPU_MESSAGE "${PARALLAX_GPU_MESSAGE}, RocALUTION")
      option(PACCX_ENABLE_CUDA       "Enable CUDA code of PARALLAX C++"         OFF)
      option(PACCX_ENABLE_HIP        "Enable HIP code of PARALLAX C++"          ON)
      option(PACCX_ENABLE_ROCALUTION "Enable RocALUTION solver in PARALLAX C++" ON)
      option(PACCX_ENABLE_AMGX       "Enable AMGX solver in PARALLAX C++"       OFF)
      option(PACCX_ENABLE_HYPRE      "Enable HYPRE in PARALLAX C++"             OFF)
    elseif("hypre" IN_LIST PARALLAX_OFFLOAD_BACKEND)
      set(PARALLAX_GPU_MESSAGE "${PARALLAX_GPU_MESSAGE}, HYPRE")
      option(PACCX_ENABLE_CUDA       "Enable CUDA code of PARALLAX C++"         ON)
      option(PACCX_ENABLE_HIP        "Enable HIP code of PARALLAX C++"          OFF)
      option(PACCX_ENABLE_ROCALUTION "Enable RocALUTION solver in PARALLAX C++" OFF)
      option(PACCX_ENABLE_AMGX       "Enable AMGX solver in PARALLAX C++"       OFF)
      option(PACCX_ENABLE_HYPRE      "Enable HYPRE in PARALLAX C++"             ON)
    else()
      option(PACCX_ENABLE_CUDA       "Enable CUDA code of PARALLAX C++"         OFF)
      option(PACCX_ENABLE_HIP        "Enable HIP code of PARALLAX C++"          OFF)
      option(PACCX_ENABLE_ROCALUTION "Enable RocALUTION solver in PARALLAX C++" OFF)
      option(PACCX_ENABLE_AMGX       "Enable AMGX solver in PARALLAX C++"       OFF)
      option(PACCX_ENABLE_HYPRE      "Enable HYPRE in PARALLAX C++"             OFF)
    endif()

    message(STATUS "${BoldGreen}PARALLAX GPU offloading features will be built "
                   "with ${PARALLAX_GPU_MESSAGE} backend/s${ColourReset}")
  endif()

  include(FetchContent)
  if (ENABLE_LATEST_PARALLAX)
    # use PARALLAX via FetchContent from gitlab repository
    message(STATUS "Build latest PARALLAX from phoenix-public/parallax")
    FetchContent_Declare(PARALLAX
      GIT_REPOSITORY git@gitlab.mpcdf.mpg.de:phoenix-public/parallax.git
      GIT_TAG master
      )
  else()
    # Use parallax from submodule
    FetchContent_Declare(PARALLAX
      URL ${GENEX-Superbuild-Project_SOURCE_DIR}/parallax
      )
    set(FETCHCONTENT_SOURCE_DIR_PARALLAX ${GENEX-Superbuild-Project_SOURCE_DIR}/parallax)
  endif()

  set(PARALLAX_ENABLE_UTESTS OFF)
  if(NOT ENABLE_MKL)
      set(PARALLAX_ENABLE_MKL OFF)
  endif()

  FetchContent_MakeAvailable(PARALLAX)
  set(PARALLAX_CONFIG_DIR ${parallax_BINARY_DIR}/src)
else()
  set(PARALLAX_CONFIG_DIR ${PARALLAX_DIR})
  message(STATUS "PARALLAX_FOUND is TRUE")
  set(Install-PARALLAX-var "")
endif()
set(PARALLAX_HOME ${PARALLAX_CONFIG_DIR})
message(STATUS "PARALLAX Config file can be used from ${PARALLAX_CONFIG_DIR}")
message(STATUS " ")

FetchContent_Declare(REAX
  URL ${GENEX-Superbuild-Project_SOURCE_DIR}/reax
)
set(FETCHCONTENT_SOURCE_DIR_REAX ${GENEX-Superbuild-Project_SOURCE_DIR}/reax)
set(REAX_ENABLE_UTESTS OFF)
FetchContent_MakeAvailable(REAX)
set(REAX_CONFIG_DIR ${reax_BINARY_DIR}/src)
set(REAX_HOME ${REAX_CONFIG_DIR})
message(STATUS "REAX Config file can be used from ${REAX_CONFIG_DIR}")
message(STATUS " ")

add_subdirectory(src)

#-----------------------------------------------------------------------------

# Autogenerate unit and MMS testing setup in <BUILD_DIR>/unit-testing/ and
# <BUILD_DIR>/mms-testing/
include(AutogenerateTesting)
autogenerate_testing()

#-----------------------------------------------------------------------------

# [ConfiX] Print the important CMake variables to CMakeVars.txt
print_variables(
  QUIET
  OUTPUT_FILE CMakeVars.txt
  VARS CMAKE_BUILD_TYPE
       CMAKE_C_COMPILER
       CMAKE_CXX_COMPILER
       CMAKE_Fortran_COMPILER
       ENABLE_SHARED_GENEX_LIB
       PRECISION
       CMAKE_INSTALL_PREFIX
       PRODUCTION_RUN
       ENABLE_UTESTS
       COVERAGE
       ENABLE_GPU
       OFFLOAD_BACKEND
       PARALLAX_OFFLOAD_BACKEND
       NVHPC_CUDA_VERSION
       MACHINE_HOST
       MACHINE_NAME
       CPU_VENDOR
       CHIP
       GPU_VENDOR
       GPU_NAME
       CUDA_TARGET_ARCH
       PROFILER_TOOL
       COMM_BACKEND
)
