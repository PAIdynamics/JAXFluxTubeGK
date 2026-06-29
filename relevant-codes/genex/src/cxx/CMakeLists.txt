cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(GENEXCore-CXX-Project
  VERSION 1.0.0
  LANGUAGES CXX)
# Require CMAKE_CUDA_ARCHITECTURES for NVIDIA GPUs
cmake_policy(SET CMP0105 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# [ConfiX] Print a title banner for output readibility
print_title_banner()

# Switches
option(GENEX_CXX_ENABLE_UTESTS "Enable unit tests for CXX library" ON)
option(GENEX_CXX_ENABLE_GPU "Enable GPU code for CXX library" ON)
option(GENEX_CXX_ENABLE_SHARED_LIB "Build shared library" ON)
option(GENEX_CXX_ENABLE_GTESTS "Enable Googletests for CXX library" ON)
if(NOT PROJECT_IS_TOP_LEVEL)
  set(GENEX_CXX_ENABLE_GTESTS OFF)
endif()

# Arguments with values
set(GENEX_CXX_PRECISION "double" CACHE STRING
  "Floating point precision used in CXX library")
set_property(CACHE GENEX_CXX_PRECISION PROPERTY STRINGS "single" "double")
set(GENEX_CXX_OFFLOAD_BACKEND "all" CACHE STRING
  "Compile backend code for which platform")
if (DEFINED ENV{AMDLLVM_HOME})
  message(STATUS "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
  set(CMAKE_PREFIX_PATH $ENV{AMDLLVM_HOME}/lib/cmake/hip)
  message(STATUS "CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
endif()
#set_property(CACHE GENEX_CXX_OFFLOAD_BACKEND PROPERTY STRINGS "all" "multicore" "openacc" "openmpx" "cuda")

# print debug statements
include(CMakePrintHelpers)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

#--------------------------------------------------------------------
# from input variables, set internal variables for steering the further CMake files
if (GENEX_CXX_ENABLE_GPU)
  set(_available_offload_backend "all;multicore;openacc;openmpx;cuda")
  foreach(_backend ${GENEX_CXX_OFFLOAD_BACKEND})
    if(NOT ${_backend} IN_LIST _available_offload_backend)
      message(FATAL_ERROR "${BoldRed}Unknown gpu offloading backend: \"${_backend}\"."
        " Allowed backends are: ${_available_offload_backend}.${ColourReset}")
    endif()
  endforeach()

  foreach(_backend ${OFFLOAD_BACKEND})
    if (${_backend} STREQUAL "openacc")
      message(STATUS "Building with OpenACC backend")
      option(ENABLE_OPENACC "Enable OpenACC" ON)
    endif()

    if (${_backend} STREQUAL "openmpx")
      option(ENABLE_OPENMPX "Enable OpenMP Target" ON)
      message(STATUS "Building with OpenMP Target backend")
    endif()

    if (${_backend} STREQUAL "cuda")
      option(ENABLE_CUDA "Enable CUDA" ON)
      message(STATUS "Building with CUDA backend")
    endif()
    if (${_backend} STREQUAL "all")
      option(ENABLE_OPENACC "Enable OpenACC" ON)
      option(ENABLE_OPENMPX "Enable OpenMP Target" ON)
      option(ENABLE_CUDA "Enable CUDA" ON)
      message(STATUS "Building for all backends")
    endif()
  endforeach()
  if (DEFINED ENV{GPU_TARGETS})
    set(GPU_TARGETS $ENV{GPU_TARGETS})
  elseif(DEFINED HIP_TARGET_ARCH)
    set(GPU_TARGETS ${HIP_TARGET_ARCH})
  endif()
  message(STATUS "Using GPU_TARGETS=${GPU_TARGETS} to find HIP.")
  find_package(hip CONFIG QUIET)
  if (TARGET hip::device)
    message(STATUS "Found HIP C++ compiler")
    option(ENABLE_HIP "Enable HIP" ON)
  else()
    message(STATUS "No HIP C++ compiler found")
  endif()
endif()

#--------------------------------------------------------------------

if(GENEX_CXX_ENABLE_GTESTS)
  include(CTest)
  enable_testing()
  include(FetchContent)

  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG 1ed6a8c67a0bd675149ece27bbec0ef1759854cf
    )
  FetchContent_MakeAvailable(googletest)
  include(GoogleTest)
endif()

# Add targets
if(GENEX_CXX_ENABLE_SHARED_LIB)
  message(STATUS "We will build SHARED GENEXCore-CXX library")

  add_library(GENEXCore-CXX-Library SHARED)
  set_target_properties(GENEXCore-CXX-Library
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    SOVERSION ${PROJECT_VERSION_MAJOR}
    OUTPUT_NAME "GENEXCore-CXX"
    DEBUG_POSTFIX "_d"
    INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}"
    )
else()
  message(STATUS "We will build STATIC GENEXCore-CXX library")

  add_library(GENEXCore-CXX-Library STATIC)
  set_target_properties(GENEXCore-CXX-Library
    PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_NAME "GENEXCore-CXX"
    DEBUG_POSTFIX "_d"
    INSTALL_RPATH "${CMAKE_INSTALL_FULL_LIBDIR}"
    )
endif()

target_compile_options(GENEXCore-CXX-Library
  PUBLIC
  $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:${CXX_FLAGS_DEBUG}>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:${CXX_FLAGS_RELEASE}>
  )
target_link_options(GENEXCore-CXX-Library
  PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Debug>>:${CXX_FLAGS_DEBUG}>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:${CXX_FLAGS_RELEASE}>
  )
if(ENABLE_CUDA)
  target_compile_options(GENEXCore-CXX-Library
    PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS}>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:${CUDA_FLAGS_DEBUG}>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:${CUDA_FLAGS_RELEASE}>
    )
  list(APPEND GENEX_CXX_COMPILE_DEFINITIONS "ENABLE_CUDA")
endif()

if(ENABLE_HIP)
  list(APPEND GENEX_CXX_COMPILE_DEFINITIONS "ENABLE_HIP")
  list(APPEND GENEX_CXX_COMPILE_DEFINITIONS "__HIP_PLATFORM_AMD__")
endif()

if(GENEX_CXX_PRECISION STREQUAL "double")
  list(APPEND GENEX_CXX_COMPILE_DEFINITIONS DOUBLE_PREC)
endif()

target_include_directories(GENEXCore-CXX-Library
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
  )
#--------------------------------------------------------------------

# Find preinstalled packages

# Link missing -lstdc++fs to use std::filesystem
if (CMAKE_CXX_COMPILER_ID MATCHES Intel OR CMAKE_CXX_COMPILER_ID MATCHES GNU)
  target_link_libraries(GENEXCore-CXX-Library PRIVATE stdc++fs)
endif()

# Enable OpenMP for C++ layer
# Note: This will be deprecated if propagated from PARALLAX
find_package(OpenMP REQUIRED QUIET COMPONENTS CXX)
if (OpenMP_CXX_FOUND)
  message(STATUS "OpenMP_VERSION=${OpenMP_VERSION}")
  message(STATUS "OpenMP_CXX_FLAGS=${OpenMP_CXX_FLAGS}")
  message(STATUS "OpenMP_CXX_INCLUDE_DIRS=${OpenMP_CXX_INCLUDE_DIRS}")
  message(STATUS "OpenMP_CXX_LIB_NAMES=${OpenMP_CXX_LIB_NAMES}")
  message(STATUS "OpenMP_CXX_LIBRARIES=${OpenMP_CXX_LIBRARIES}")
else()
  message(FATAL_ERROR "OpenMP for CXX not FOUND")
endif()
if (ENABLE_OPENMPX)
  list(APPEND GENEX_CXX_COMPILE_DEFINITIONS ENABLE_OPENMPX)
  if(DEFINED ENV{AMDLLVM_HOME})
    target_link_libraries(GENEXCore-CXX-Library PRIVATE
      $ENV{AMDLLVM_HOME}/lib/llvm/lib/libomptarget.so)
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
    if(DEFINED ENV{AMDLLVM_HOME})
      set(OpenMPTarget_ROOT $ENV{AMDLLVM_HOME}/lib/llvm/lib)
      set(OpenMPTarget_AMDGPU_ARCH "${HIP_TARGET_ARCH}")
      find_package(OpenMPTarget REQUIRED COMPONENTS "AMDGPU")
      set_target_properties(OpenMPTarget::OpenMPTarget_AMDGPU PROPERTIES
	INTERFACE_COMPILE_OPTIONS "-fopenmp" "--offload-arch=${HIP_TARGET_ARCH}"
	INTERFACE_LINK_OPTIONS "${OpenMPTarget_FLAGS} -fopenmp" "--offload-arch=${HIP_TARGET_ARCH}")
      #  set(OpenMP_CXX_FLAGS "-fopenmp" "--offload-arch=${HIP_TARGET_ARCH}")
      target_link_libraries(GENEXCore-CXX-Library PRIVATE OpenMPTarget::OpenMPTarget_AMDGPU)
    endif()
  else()
    target_link_libraries(GENEXCore-CXX-Library
      PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

# Enable MPI
find_package(MPI REQUIRED QUIET COMPONENTS CXX)
target_link_libraries(GENEXCore-CXX-Library
  PUBLIC MPI::MPI_CXX)
if (MPI_CXX_FOUND)
  message(STATUS "MPI for CXX found: MPI_CXX_INCLUDE_DIRS=${MPI_CXX_INCLUDE_DIRS}")
  message(STATUS "MPI_CXX_COMPILER=${MPI_CXX_COMPILER}")
  message(STATUS "MPI_CXX_COMPILE_OPTIONS=${MPI_CXX_COMPILE_OPTIONS}")
  message(STATUS "MPI_CXX_HEADER_DIR=${MPI_CXX_HEADER_DIR}")
  message(STATUS "MPI_CXX_COMPILE_DEFINITIONS=${MPI_CXX_COMPILE_DEFINITIONS}")
  message(STATUS "MPI_CXX_LINK_FLAGS=${MPI_CXX_LINK_FLAGS}")
  message(STATUS "MPI_CXX_LIBRARIES=${MPI_CXX_LIBRARIES}")
else()
  message(FATAL_ERROR "MPI for CXX not found")
endif()

target_include_directories(GENEXCore-CXX-Library PUBLIC ${MPI_CXX_INCLUDE_DIRS})

if(COMM_BACKEND STREQUAL "nccl")
  message(STATUS "  TRY to use NCCL or RCCL")
  if (GPU_VENDOR MATCHES AMD)
    find_package(rccl REQUIRED)
    if (TARGET roc::rccl)
      message(STATUS "  RCCL found")
      target_link_libraries(GENEXCore-CXX-Library PUBLIC roc::rccl)
    else()
      message(STATUS "  RCCL not found")
    endif()
  else()
    target_link_libraries(GENEXCore-CXX-Library PRIVATE NVHPC::NCCL)
  endif()
endif()

# Enable OpenACC for C++ layer
if(ENABLE_OPENACC)
  find_package(OpenACC REQUIRED)
  if(OpenACC_CXX_FOUND)
    message(STATUS "Found OpenACC_CXX: ${OpenACC_CXX_FLAGS}")
    target_link_libraries(GENEXCore-CXX-Library
      PRIVATE OpenACC::OpenACC_CXX)
    list(APPEND GENEX_CXX_COMPILE_DEFINITIONS "ENABLE_OPENACC")
  else()
    message(FATAL_ERROR "OpenACC_CXX is not found.")
  endif()
endif()

# Link and include the profiler target
if(PROFILER_TOOL MATCHES "nvtx")
  target_link_libraries(GENEXCore-CXX-Library PRIVATE NVHPC::PROFILER)
  list(APPEND GENEX_CXX_COMPILE_DEFINITIONS "ENABLE_NVTX")

elseif(PROFILER_TOOL MATCHES "roctx")
  target_link_libraries(GENEXCore-CXX-Library PRIVATE
    rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
  message(STATUS "Adding ${rocprofiler-sdk-roctx_INCLUDE_DIR} as include directory")
  target_include_directories(GENEXCore-CXX-Library PRIVATE
    ${rocprofiler-sdk-roctx_INCLUDE_DIR})
  list(APPEND GENEX_CXX_COMPILE_DEFINITIONS "ENABLE_ROCTX")

elseif(PROFILER_TOOL MATCHES "dlb")
  target_link_libraries(GENEXCore-CXX-Library PUBLIC DLB::DLB)
  list(APPEND GENEX_CXX_COMPILE_DEFINITIONS "ENABLE_DLB")

elseif(PROFILER_TOOL MATCHES "extrae")
  target_include_directories(GENEXCore-CXX-Library
    PRIVATE ${EXTRAE_INCLUDE_DIRS})
  target_link_libraries(GENEXCore-CXX-Library PUBLIC ${EXTRAE_LIBRARIES})
  list(APPEND GENEX_CXX_COMPILE_DEFINITIONS "ENABLE_EXTRAE")

elseif(PROFILER_TOOL MATCHES "caliper")
  target_link_libraries(GENEXCore-CXX-Library PUBLIC caliper)
  list(APPEND GENEX_CXX_COMPILE_DEFINITIONS "ENABLE_CALIPER")

endif()

#--------------------------------------------------------------------

message(STATUS "${BoldGreen}GENEX_CXX_COMPILE_DEFINITIONS: "
               "${GENEX_CXX_COMPILE_DEFINITIONS}${ColourReset}")
target_compile_definitions(GENEXCore-CXX-Library
  PRIVATE ${GENEX_CXX_COMPILE_DEFINITIONS})

#--------------------------------------------------------------------

# Add all subdirectories
add_subdirectory(communication)
add_subdirectory(datastructures)
add_subdirectory(devtools)
add_subdirectory(logger)
add_subdirectory(math)
add_subdirectory(mesh)
add_subdirectory(operators)
add_subdirectory(parameters)
add_subdirectory(profiler)
add_subdirectory(utils)

#--------------------------------------------------------------------

# Preparations for installation
message(STATUS "${PROJECT_NAME} will be installed to ${CMAKE_INSTALL_PREFIX}")

# Installation
install(
  TARGETS
  GENEXCore-CXX-Library
  EXPORT
  GENEXCore-CXXTargets
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

#--------------------------------------------------------------------

# Exporting targets for other CMake-projects
set(INSTALL_CMAKEDIR cmake)

install(
  EXPORT
  GENEXCore-CXXTargets
  NAMESPACE
  "GENEXCore-CXX::"
  DESTINATION
  ${INSTALL_CMAKEDIR}
  COMPONENT
  dev
  )
if (NOT TARGET GENEXCore-CXX::GENEXCore-CXX-Library)
  add_library(GENEXCore-CXX::GENEXCore-CXX-Library ALIAS GENEXCore-CXX-Library)
endif()

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/GENEXCore-CXXConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
  )

configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/GENEXCore-CXXConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/GENEXCore-CXXConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CMAKEDIR}
  )

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/GENEXCore-CXXConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/GENEXCore-CXXConfigVersion.cmake
  DESTINATION
  ${INSTALL_CMAKEDIR}
  )

#--------------------------------------------------------------------
