cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

project(GENEX-Project
  VERSION 1.0.0
  LANGUAGES CXX Fortran)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

#--------------------------------------------------------------------
# Internal GENE-X C++/GPU library
if(ENABLE_GPU)
  set(GENEX_CXX_PRECISION ${PRECISION} CACHE STRING "Precision for GENEX-CXX")
  set(GENEX_CXX_ENABLE_SHARED_LIB ${ENABLE_SHARED_GENEX_LIB} CACHE BOOL "Shared library")
  set(GENEX_CXX_ENABLE_UTESTS ${ENABLE_UTESTS} CACHE BOOL "Unit tests for GENEX-CXX")
  set(GENEX_CXX_ENABLE_GPU ${ENABLE_GPU} CACHE BOOL "Use the GPU backend")
  set(GENEX_CXX_OFFLOAD_BACKEND ${OFFLOAD_BACKEND} CACHE STRING "which backend: all,multicore,openacc,openmpx,cuda")
  add_subdirectory(cxx)
endif()

#--------------------------------------------------------------------

set(FORTRAN_COMPILE_OPTIONS
  "$<$<COMPILE_LANGUAGE:Fortran>:${Fortran_FLAGS}>"
  "$<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Debug>>:${Fortran_FLAGS_DEBUG}>"
  "$<$<AND:$<COMPILE_LANGUAGE:Fortran>,$<CONFIG:Release>>:${Fortran_FLAGS_RELEASE}>"
)
set(CXX_COMPILE_OPTIONS
  "$<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>"
  "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:${CXX_FLAGS_DEBUG}>"
  "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:${CXX_FLAGS_RELEASE}>"
)

# Add targets

# [ConfiX] Print a title banner for output readibility
print_title_banner()

add_executable(GENEX-Executable genex.f90)
add_executable(Benchmark-Operators benchmarks/benchmark_operators.f90)

set_target_properties(GENEX-Executable
  PROPERTIES
  OUTPUT_NAME "genex"
  INSTALL_RPATH_USE_LINK_PATH ON
  INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}" )

set_target_properties(Benchmark-Operators
  PROPERTIES
  OUTPUT_NAME "benchmark-operators"
  INSTALL_RPATH_USE_LINK_PATH ON
  INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}" )

target_compile_definitions(Benchmark-Operators
  PUBLIC
  ${GENEX_COMPILE_DEFINITIONS}
  )

# For usage of Intel Advisor annotations module
if(CMAKE_Fortran_COMPILER_ID MATCHES Intel)
    if(DEFINED ENV{ADVISOR_HOME})
        message(STATUS "Found Advisor: $ENV{ADVISOR_HOME}")

        target_link_libraries(Benchmark-Operators
          PRIVATE -L$ENV{ADVISOR_HOME}/lib64
          PRIVATE -ladvisor
        )
        target_compile_options(Benchmark-Operators
          PRIVATE -I$ENV{ADVISOR_HOME}/include/intel64
          PUBLIC ${FORTRAN_COMPILE_OPTIONS}
        )
        # Create ADV_ANNOTATE preprocessor directive to use in benchmark program
        target_compile_definitions(Benchmark-Operators
          PRIVATE -DADV_ANNOTATE)
        message(STATUS "Building benchmarks with annotations.")
    else()
        message(STATUS "Advisor not found. "
                       "Building benchmarks without annotations.")
    endif()
else()
    message(STATUS "Compiler is not Intel. "
                   "Building benchmarks without annotations.")
endif()

if(ENABLE_SHARED_GENEX_LIB)
  message(STATUS "We will build SHARED GENEXCore library")

  add_library(GENEXCore-Library SHARED "")
  set_target_properties(GENEXCore-Library
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "GENEXCore"
    DEBUG_POSTFIX "_d"
    MACOSX_RPATH ON
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    INSTALL_RPATH_USE_LINK_PATH ON
    INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}"
    )
else()
  message(STATUS "We will build STATIC GENEXCore library")

  add_library(GENEXCore-Library STATIC "")
  set_target_properties(GENEXCore-Library
    PROPERTIES
    POSITION_INDEPENDENT_CODE OFF
    ARCHIVE_OUTPUT_NAME "GENEXCore"
    DEBUG_POSTFIX "_d"
    )
endif()

target_link_libraries(GENEX-Executable
  PUBLIC GENEXCore-Library
)

target_link_libraries(Benchmark-Operators
  PRIVATE GENEXCore-Library
)

target_compile_options(GENEXCore-Library
  PUBLIC
  ${FORTRAN_COMPILE_OPTIONS}
  )

if(ENABLE_GPU)
  target_compile_options(GENEXCore-Library
    PUBLIC
    ${CXX_COMPILE_OPTIONS}
    )
endif()

target_link_options(GENEXCore-Library
  PUBLIC
  ${Linker_FLAGS}
  )

target_link_options(GENEX-Executable
  PRIVATE
  ${Linker_FLAGS}
)

target_link_options(Benchmark-Operators
  PRIVATE
  ${Linker_FLAGS}
  )

target_compile_definitions(GENEXCore-Library
  PUBLIC
  ${GENEX_COMPILE_DEFINITIONS}
  )

target_include_directories(GENEXCore-Library
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  INTERFACE $<BUILD_INTERFACE:${CMAKE_Fortran_MODULE_DIRECTORY}>
  )

# Create a separated library to be used in the utests only
add_library(GENEXTestUtils-Library STATIC "")
set_target_properties(GENEXTestUtils-Library
  PROPERTIES
  POSITION_INDEPENDENT_CODE OFF
  ARCHIVE_OUTPUT_NAME "GENEXTestUtils"
  DEBUG_POSTFIX "_d"
  )

target_compile_options(GENEXTestUtils-Library
  PUBLIC
  ${FORTRAN_COMPILE_OPTIONS}
  )

target_link_options(GENEXTestUtils-Library
  PUBLIC
  ${Linker_FLAGS}
  )

target_compile_definitions(GENEXTestUtils-Library
  PUBLIC
  ${GENEX_COMPILE_DEFINITIONS}
  )

target_include_directories(GENEXTestUtils-Library
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  INTERFACE $<BUILD_INTERFACE:${CMAKE_Fortran_MODULE_DIRECTORY}>
  )

#--------------------------------------------------------------------

# Find preinstalled packages

# Internal GENE-X C++/GPU Library
if(ENABLE_GPU)
  target_link_libraries(GENEXCore-Library PUBLIC GENEXCore-CXX::GENEXCore-CXX-Library)
endif()

# Enable MKL
if(ENABLE_MKL)
  set(MKL_DIR $ENV{MKL_HOME}/lib/cmake/mkl)
  set(MKL_INTERFACE lp64)
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "Intel")
    set(MKL_THREADING intel_thread)
  endif()
  if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
    set(MKL_THREADING gnu_thread)
  endif()
  find_package(MKL REQUIRED)
endif()

target_link_libraries(GENEXCore-Library PUBLIC PARALLAX::PARALLAX-Library)
target_link_libraries(GENEXCore-Library PUBLIC REAX::REAX-Library)

# Enable OpenMP for Fortran layer
find_package(OpenMP REQUIRED QUIET COMPONENTS Fortran)
target_link_libraries(GENEXCore-Library PUBLIC OpenMP::OpenMP_Fortran)

# Enable MPI
if(ENABLE_GPU)
  find_package(MPI REQUIRED QUIET COMPONENTS Fortran CXX)
else()
  find_package(MPI REQUIRED QUIET COMPONENTS Fortran)
endif()

if(MPI_Fortran_FOUND)
  message(STATUS "MPI for Fortran found: MPI_Fortran_INCLUDE_DIRS=${MPI_Fortran_INCLUDE_DIRS}")
  message(STATUS "MPI_Fortran_COMPILER=${MPI_Fortran_COMPILER}")
  message(STATUS "MPI_Fortran_COMPILE_OPTIONS=${MPI_Fortran_COMPILE_OPTIONS}")
  message(STATUS "MPI_Fortran_COMPILE_DEFINITIONS=${MPI_Fortran_COMPILE_DEFINITIONS}")
  message(STATUS "MPI_Fortran_LINK_FLAGS=${MPI_Fortran_LINK_FLAGS}")
  message(STATUS "MPI_Fortran_LIBRARIES=${MPI_Fortran_LIBRARIES}")
else()
  message(FATAL_ERROR "MPI for Fortran was not found.")
endif()

# Enable NetCDF
find_package(NetCDF REQUIRED)

# Enable pFUnit
if(ENABLE_UTESTS)
  enable_testing()
endif()

target_link_libraries(GENEXTestUtils-Library PUBLIC PARALLAX::PARALLAX-Library)
target_link_libraries(GENEXTestUtils-Library PUBLIC GENEXCore-Library)

#--------------------------------------------------------------------

# Collect all sources in the current directory and subdirectories

target_sources(GENEXCore-Library
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/dimensions_m.f90
    ${CMAKE_CURRENT_LIST_DIR}/genex_fortran_env_m.f90
)

# [ConfiX] Generate a build-related info module, i.e. git and path information
import_build_info_to_source(
  LANG Fortran
  COMPONENTS GIT PATH
  APP_NAME "genex"
  PREFIX "build"
  TARGET GENEXCore-Library
)

# Add all subdirectories

add_subdirectory(math)
add_subdirectory(test_utils)
add_subdirectory(communication)
add_subdirectory(datastructures)
add_subdirectory(devtools)
add_subdirectory(dist_initial)
add_subdirectory(error_handling)
add_subdirectory(files)
add_subdirectory(logger)
add_subdirectory(mesh)
add_subdirectory(operators)
add_subdirectory(parameters)
add_subdirectory(profiler)
add_subdirectory(profiles)
add_subdirectory(state_vector)
add_subdirectory(timestep)
add_subdirectory(sysutils)
add_subdirectory(bsg)

#--------------------------------------------------------------------

# [ConfiX] Create softlink of the executables into the bin folder
create_target_softlink(DESTINATION ${CMAKE_BINARY_DIR}/bin
                       TARGET INCLUDE "utests"
                                      "GENEX-Executable"
                                      "Benchmark-Operators")

# Set environment variable to GPU unit testing targets to prevent fallback
# to CPU and force fatal error when GPUs are not found

if(CMAKE_CXX_COMPILER_ID MATCHES NVHPC)
  set(OPENACC_ENV_VAR "NVCOMPILER_ACC_DEVICE_TYPE")
else()
  set(OPENACC_ENV_VAR "ACC_DEVICE_TYPE")
endif()

if(GPU_VENDOR MATCHES "NVIDIA")
  set(OPENACC_ENV_VAR "${OPENACC_ENV_VAR}=NVIDIA")
elseif(GPU_VENDOR MATCHES "AMD")
  set(OPENACC_ENV_VAR "${OPENACC_ENV_VAR}=RADEON")
endif()

# [ConfiX] Set environment variable to specific test targets
append_test_environment(TEST INCLUDE "utests-gpu-acc"
                        ENV ${OPENACC_ENV_VAR})

append_test_environment(TEST INCLUDE "utests-gpu-ompx"
                        ENV "OMP_TARGET_OFFLOAD=MANDATORY")

# Set environment variable to GPU unit testing targets to determine
# the GPU offload backend based on the naming convention of the targets

append_test_environment(TEST INCLUDE "utests-gpu-cxx"
                        ENV "GENEX_UTEST_GPU_BACKEND=CXX")

append_test_environment(TEST INCLUDE "utests-gpu-acc"
                        ENV "GENEX_UTEST_GPU_BACKEND=OPENACC")

append_test_environment(TEST INCLUDE "utests-gpu-ompx"
                        ENV "GENEX_UTEST_GPU_BACKEND=OPENMPX")

append_test_environment(TEST INCLUDE "utests-gpu-cuda"
                        ENV "GENEX_UTEST_GPU_BACKEND=CUDA")

#--------------------------------------------------------------------

# Preparations for installation

message(STATUS "${PROJECT_NAME} will be installed to ${CMAKE_INSTALL_PREFIX}")

target_include_directories(GENEXCore-Library
  INTERFACE
  $<INSTALL_INTERFACE:${INSTALL_MODDIR}>
  )

target_include_directories(GENEXTestUtils-Library
  INTERFACE
  $<INSTALL_INTERFACE:${INSTALL_MODDIR}>
  )

#--------------------------------------------------------------------

# Compile the GPU unit testing targets sequentially

# Sequential build is necessary to avoid compile-time error of the unit testing
# targets related to GPU when multiple targets share same source files.
# This issue always occurs with GNU compiler and rarely occurs with Intel
# compiler in CI jobs..

message(STATUS "The GPU unit testing targets are built sequentially.")

# [ConfiX] Filter all GPU unit testing targets
get_all_targets(OUTPUT gpu_utest_targets INCLUDE "utests-gpu")

# Add GENE-X specific compile definitions (macros)
# to all GPU unit testing targets
foreach(tgt ${gpu_utest_targets})
  target_compile_definitions(${tgt} PRIVATE ${GENEX_COMPILE_DEFINITIONS})
endforeach()

# Get suffixes of all GPU unit testing targets
set(gpu_suffixes)
foreach(tgt ${gpu_utest_targets})
  string(REPLACE "-gpu-" ";" tgt_list ${tgt})
  list(GET tgt_list -1 suffix)
  list(APPEND gpu_suffixes ${suffix})
endforeach()
list(REMOVE_DUPLICATES gpu_suffixes)

# Filter all CPU unit testing targets that has its GPU counterparts
set(common_targets ${gpu_utest_targets})
list(TRANSFORM common_targets APPEND "-endmarker")
foreach(suffix ${gpu_suffixes})
  list(TRANSFORM common_targets REPLACE "-gpu-${suffix}-endmarker" "")
endforeach()
list(REMOVE_DUPLICATES common_targets)

# Create the dependency list
set(gpu_utest_target_dependencies "")
foreach(tgt ${common_targets})
  list(APPEND gpu_utest_target_dependencies ${tgt})
  set(sublist ${gpu_utest_targets})
  list(FILTER sublist INCLUDE REGEX "(.*)${tgt}(.*)")
  list(POP_BACK sublist)
  list(APPEND gpu_utest_target_dependencies ${sublist})
endforeach()

# Sequentialize GPU utest targets
foreach(target dependency
        IN ZIP_LISTS gpu_utest_targets gpu_utest_target_dependencies)
    add_dependencies(${target} ${dependency})
endforeach()

#--------------------------------------------------------------------

# Installation

install(
  TARGETS
  GENEXCore-Library
  GENEXTestUtils-Library
  GENEX-Executable
  Benchmark-Operators
  EXPORT
  GENEXCoreTargets
  ARCHIVE
  DESTINATION ${INSTALL_LIBDIR}
  COMPONENT lib
  RUNTIME
  DESTINATION ${INSTALL_BINDIR}
  COMPONENT bin
  LIBRARY
  DESTINATION ${INSTALL_LIBDIR}
  COMPONENT lib
  PUBLIC_HEADER
  DESTINATION ${INSTALL_INCLUDEDIR}/GENEXCore
  COMPONENT dev
  )

install(DIRECTORY ${CMAKE_Fortran_MODULE_DIRECTORY}/
        DESTINATION ${INSTALL_MODDIR})

#--------------------------------------------------------------------

# Exporting targets for other CMake-projects

install(
  EXPORT
  GENEXCoreTargets
  NAMESPACE
  "GENEXCore::"
  DESTINATION
  ${INSTALL_CMAKEDIR}
  COMPONENT
  dev
  )

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/GENEXCoreConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
  )

configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/GENEXCoreConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/GENEXCoreConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CMAKEDIR}
  )

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/GENEXCoreConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/GENEXCoreConfigVersion.cmake
  DESTINATION
  ${INSTALL_CMAKEDIR}
  )
#--------------------------------------------------------------------
