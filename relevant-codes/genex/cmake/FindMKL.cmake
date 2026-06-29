# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindMKL
-------

Finds the MKL library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported targets, if found:

``MKL::MKL``
  The MKL library

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``MKL_FOUND``
  True if the system has the MKL library.
``MKL_VERSION``
  The version of the MKL library which was found.
``MKL_VERSION_MAJOR``
  The major version of the MKL library found.
``MKL_VERSION_MINOR``
  The minor version of the MKL library found.
``MKL_VERSION_PATCH``
  The update version of the MKL library found.
``MKL_ROOT_DIR``
  The root directory of the MKL library.
``MKL_INCLUDE_DIRS``
  Include directories needed to use MKL.
``MKL_LIBRARIES``
  Libraries needed to link to MKL.
``MKL_DEFINITIONS``
  Definitions to use when compiling code that uses MKL
``MKL_COMPILE_OPTIONS``
  Compile options to use when compiling code that uses MKL
``MKL_LINK_OPTIONS``
  Link options to use when compiling code that uses MKL

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``MKL_ROOT_DIR``
  The root directory.
#]=======================================================================]

if(NOT MKL_ROOT_DIR)
  find_path(MKL_ROOT_DIR
    HINTS
    $ENV{MKLROOT}
    $ENV{MKL_HOME}
    NAMES
    include/mkl.h)
  if(NOT MKL_ROOT_DIR)
    if(MKL_FIND_REQUIRED)
      message(FATAL_ERROR "Couldn't set MKL_ROOT_DIR")
    else()
      if(NOT MKL_FIND_QUIETLY)
    message(STATUS "FindMKL: Couldn't find MKL_ROOT_DIR")
      endif()
      set(MKL_FOUND False)
      return()
    endif()
  else()
    set(MKL_FOUND True)
  endif()
endif()

if(NOT MKL_FIND_QUIETLY)
  message(STATUS "FindMKL: MKL_ROOT_DIR is set to ${MKL_ROOT_DIR}")
endif()

find_path(_MKL_INCLUDE_DIR NAMES mkl.h HINTS ${MKL_ROOT_DIR}/include)
set(_MKL_VERSION_H ${_MKL_INCLUDE_DIR}/mkl_version.h)

if(NOT MKL_FIND_QUIETLY)
  message(STATUS "FindMKL: MKL version will be checked in ${_MKL_VERSION_H}")
endif()

function(_mkl_ver_extract _MKL_VER_COMPONENT _MKL_VER_OUTPUT)
  set(CMAKE_MATCH_1 "0")
  set(_MKL_expr
    "^[ \\t]*#define[ \\t]+${_MKL_VER_COMPONENT}[ \\t]+([0-9]+)$")
  file(STRINGS "${_MKL_VERSION_H}" _MKL_ver REGEX "${_MKL_expr}")
  string(REGEX MATCH "${_MKL_expr}" MKL_ver "${_MKL_ver}")
  set(${_MKL_VER_OUTPUT} "${CMAKE_MATCH_1}" PARENT_SCOPE)
endfunction()

_mkl_ver_extract("__INTEL_MKL__" MKL_VERSION_MAJOR)
_mkl_ver_extract("__INTEL_MKL_MINOR__" MKL_VERSION_MINOR)
_mkl_ver_extract("__INTEL_MKL_UPDATE__" MKL_VERSION_PATCH)

set(MKL_VERSION "${MKL_VERSION_MAJOR}.${MKL_VERSION_MINOR}.${MKL_VERSION_PATCH}")

if(NOT MKL_FIND_QUIETLY)
  message(STATUS "FindMKL: MKL version is set to ${MKL_VERSION}")
endif()

function(_mkl_arch_check _ARCH_IN _ARCH_NAME_IN _ARCH_OUT _ARCH_NAME_OUT)
  string(FIND ${CMAKE_SYSTEM_PROCESSOR} ${_ARCH_IN} _POSITION)
  if(_POSITION GREATER -1)
    set(${_ARCH_OUT} "${_ARCH_IN}" PARENT_SCOPE)
    set(${_ARCH_NAME_OUT} "${_ARCH_NAME_IN}" PARENT_SCOPE)
  endif()
endfunction()

_mkl_arch_check("64" "intel64" _MKL_BIT _MKL_ARCH)
_mkl_arch_check("32" "ia32" _MKL_BIT _MKL_ARCH)

if(NOT MKL_FIND_QUIETLY)
  message(STATUS
    "FindMKL: MKL architecture (BIT, NAME) is (${_MKL_BIT}, ${_MKL_ARCH})")
  message(STATUS
    "FindMKL: MKL Libraries will be looked for in ${MKL_ROOT_DIR}/lib/${_MKL_ARCH}")
endif()

if(("BLAS95" IN_LIST MKL_FIND_COMPONENTS) OR
    ("LAPACK95" IN_LIST MKL_FIND_COMPONENTS))
  set(_MKL_INCLUDE_BLAS_LAPACK_DIR "${_MKL_INCLUDE_DIR}/${_MKL_ARCH}")
  if(_MKL_ARCH STREQUAL "intel64")
    set(_MKL_INCLUDE_BLAS_LAPACK_DIR "${_MKL_INCLUDE_BLAS_LAPACK_DIR}/lp64")
  endif()
endif()

include(FindPackageHandleStandardArgs)

if("BLAS95" IN_LIST MKL_FIND_COMPONENTS)
  find_library(_MKL_BLAS95
    NAMES
    mkl_blas95_lp64
    mkl_blas95
    HINTS
    ${MKL_ROOT_DIR}/lib/${_MKL_ARCH}
    )
  find_package_handle_standard_args(MKL_BLAS95 DEFAULT_MSG
    _MKL_BLAS95 _MKL_INCLUDE_BLAS_LAPACK_DIR)
endif()

if("LAPACK95" IN_LIST MKL_FIND_COMPONENTS)
  find_library(_MKL_LAPACK95
    NAMES
    mkl_lapack95_lp64
    mkl_lapack95
    HINTS
    ${MKL_ROOT_DIR}/lib/${_MKL_ARCH}
    )
  find_package_handle_standard_args(MKL_LAPACK95 DEFAULT_MSG
    _MKL_LAPACK95 _MKL_INCLUDE_BLAS_LAPACK_DIR)
endif()

if("ScaLAPACK" IN_LIST MKL_FIND_COMPONENTS)
  find_library(_MKL_SCALAPACK
    NAMES
    mkl_scalapack_lp64
    mkl_scalapack_core
    HINTS
    ${MKL_ROOT_DIR}/lib/${_MKL_ARCH}
    )
  find_package_handle_standard_args(MKL_ScaLAPACK DEFAULT_MSG _MKL_SCALAPACK)
endif()

set(MKL_COMPILE_OPTIONS)
set(MKL_LINK_OPTIONS)
set(_MKL_STATIC False)

if(CMAKE_Fortran_COMPILER_ID MATCHES Intel)
  find_library(_MKL_COMPILER_SPECIFIC
    NAMES
    mkl_intel_lp64
    mkl_intel
    HINTS
    ${MKL_ROOT_DIR}/lib/${_MKL_ARCH}
    )
endif()

if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
  find_library(_MKL_COMPILER_SPECIFIC
    mkl_gf_lp64
    mkl_gf
    HINTS
    ${MKL_ROOT_DIR}/lib/${_MKL_ARCH}
    )
  list(APPEND MKL_COMPILE_OPTIONS -m64)
  list(APPEND MKL_LINK_OPTIONS -Wl,--no-as-needed)
endif()

if((CMAKE_Fortran_COMPILER_ID MATCHES PGI) OR
   (CMAKE_Fortran_COMPILER_ID MATCHES NVHPC))
  find_library(_MKL_COMPILER_SPECIFIC
    mkl_intel_lp64
    mkl_intel
    HINTS
    ${MKL_ROOT_DIR}/lib/${_MKL_ARCH}
    )
  list(APPEND MKL_COMPILE_OPTIONS -m64)
  list(APPEND MKL_LINK_OPTIONS -Wl,--no-as-needed)

  find_library(_MKL_THREAD
    NAMES
    mkl_intel_thread
    HINTS
    ${MKL_ROOT_DIR}/lib/${_MKL_ARCH}
    )
endif()

if(CMAKE_Fortran_COMPILER_ID MATCHES LLVMFlang)
  find_library(_MKL_COMPILER_SPECIFIC
    mkl_intel_lp64
    mkl_intel
    HINTS
    ${MKL_ROOT_DIR}/lib/${_MKL_ARCH}
  )
  #list(APPEND MKL_COMPILE_OPTIONS )
  list(APPEND MKL_LINK_OPTIONS -Wl,--no-as-needed)

  find_library(_MKL_THREAD
    NAMES
    mkl_intel_thread
    HINTS
    ${MKL_ROOT_DIR}/lib/${_MKL_ARCH}
  )
endif()

get_filename_component(_lib_ext ${_MKL_COMPILER_SPECIFIC} EXT)
if(_lib_ext STREQUAL ".a")
  set(_MKL_STATIC True)
endif()

if(CMAKE_Fortran_COMPILER_ID MATCHES Intel)
    find_library(_MKL_THREAD
      NAMES
      mkl_intel_thread
      HINTS
      ${MKL_ROOT_DIR}/lib/${_MKL_ARCH}
      )
endif()

if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
    find_library(_MKL_THREAD
      NAMES
      mkl_gnu_thread
      HINTS
      ${MKL_ROOT_DIR}/lib/${_MKL_ARCH}
      )
endif()

find_library(_MKL_CORE
  NAMES
  mkl_core
  HINTS
  ${MKL_ROOT_DIR}/lib/${_MKL_ARCH}
  )

if(("CDFT" IN_LIST MKL_FIND_COMPONENTS))
  find_library(_MKL_CDFT
    NAMES
    mkl_cdft_core
    HINTS
    ${MKL_ROOT_DIR}/lib/${_MKL_ARCH}
    )
  find_package_handle_standard_args(MKL_CDFT DEFAULT_MSG _MKL_CDFT)
endif()

if(("BLACS" IN_LIST MKL_FIND_COMPONENTS) OR
    ("ScaLAPACK" IN_LIST MKL_FIND_COMPONENTS)  OR
    ("CDFT" IN_LIST MKL_FIND_COMPONENTS))
  find_library(_MKL_BLACS
    NAMES
    mkl_blacs_intelmpi_lp64
    mkl_blacs_intelmpi
    HINTS
    ${MKL_ROOT_DIR}/lib/${_MKL_ARCH}
    )
  find_package_handle_standard_args(MKL_BLACS DEFAULT_MSG _MKL_BLACS)
endif()

# find_library(_PTHREAD
#   NAMES
#   pthread
#   )

# find_library(_M
#   NAMES
#   m
#   )

# find_library(_DL
#   NAMES
#   dl
#   )

if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
    set(_pthread_m_dl -lgomp -lpthread -lm -ldl)
endif()

if(CMAKE_Fortran_COMPILER_ID MATCHES Intel)
    set(_pthread_m_dl -liomp5 -lpthread -lm -ldl)
endif()

find_package_handle_standard_args(MKL
  REQUIRED_VARS
  MKL_ROOT_DIR
  _MKL_INCLUDE_DIR
  _MKL_COMPILER_SPECIFIC
  _MKL_THREAD
  _MKL_CORE
  # _PTHREAD
  # _M
  # _DL
  HANDLE_COMPONENTS
  VERSION_VAR MKL_VERSION
)

if(_MKL_STATIC)
  set(_START_GROUP "-Wl,--start-group")
  set(_END_GROUP "-Wl,--end-group")
endif()

if(MKL_FOUND)
  set(MKL_LIBRARIES
    ${_MKL_BLAS95}
    ${_MKL_LAPACK95}
    ${_MKL_SCALAPACK}
    ${_START_GROUP}
    ${_MKL_CDFT}
    ${_MKL_COMPILER_SPECIFIC}
    ${_MKL_THREAD}
    ${_MKL_CORE}
    ${_MKL_BLACS}
    ${_END_GROUP}
    ${_pthread_m_dl}
    # ${_PTHREAD}
    # ${_M}
    # ${_DL}
    )
  set(MKL_INCLUDE_DIRS ${_MKL_INCLUDE_DIR} ${_MKL_INCLUDE_BLAS_LAPACK_DIR})
  set(MKL_DEFINITIONS ${MKL_FFLAGS_OTHER})
endif()

if(NOT MKL_FIND_QUIETLY)
  message(STATUS "FindMKL: MKL_LIBRARIES are set to ${MKL_LIBRARIES}")
  message(STATUS "FindMKL: MKL_INCLUDE_DIRS are set to ${MKL_INCLUDE_DIRS}")
  message(STATUS "FindMKL: MKL_COMPILE_OPTIONS are set to ${MKL_COMPILE_OPTIONS}")
  message(STATUS "FindMKL: MKL_LINK_OPTIONS are set to ${MKL_LINK_OPTIONS}")
endif()

if(MKL_FOUND AND NOT TARGET MKL::MKL)
  add_library(MKL::MKL INTERFACE IMPORTED)
  set_property(TARGET MKL::MKL PROPERTY INTERFACE_COMPILE_DEFINITIONS
    "${MKL_DEFINITIONS}")
  set_property(TARGET MKL::MKL PROPERTY INTERFACE_LINK_LIBRARIES "${MKL_LIBRARIES}")
  set_property(TARGET MKL::MKL PROPERTY INTERFACE_INCLUDE_DIRECTORIES
    "${MKL_INCLUDE_DIRS}")
  set_property(TARGET MKL::MKL PROPERTY INTERFACE_COMPILE_OPTIONS
    "${MKL_COMPILE_OPTIONS}")
  set_property(TARGET MKL::MKL PROPERTY INTERFACE_LINK_OPTIONS
    "${MKL_LINK_OPTIONS}")
endif()

mark_as_advanced(MKL_ROOT_DIR)
