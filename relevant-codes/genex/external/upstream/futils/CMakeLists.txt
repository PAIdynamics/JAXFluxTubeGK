enable_language(C)

set(FUTILS_NAME futils-gene_hdf5-1.8.13) #futils-2.0

set(FUTILS_SOURCE_DIR
  ${CMAKE_SOURCE_DIR}/external/futils-gene-mod)
set(FUTILS_URL
  http://documents.epfl.ch/users/t/tt/ttran/www/${FUTILS_NAME}.tar.gz)

if(CMAKE_MAKE_PROGRAM MATCHES make)
  set(FUTILS_MAKE ${CMAKE_MAKE_PROGRAM})
else()
  find_program(FUTILS_MAKE
    NAMES make gmake
    DOC "Make command for FUTILS-Project"
    # [HINTS path1 [path2 ... ENV var]] ## you can set the full path to the file here
    )
  if(NOT FUTILS_MAKE)
    message(FATAL_ERROR "Couldn't find make command, please provide hints")
  endif()
endif()

find_package(MPI REQUIRED QUIET COMPONENTS Fortran C)
set(FUTILS_F90 ${MPI_Fortran_COMPILER})
set(FUTILS_C ${MPI_C_COMPILER})

if(CMAKE_BUILD_TYPE MATCHES Debug)
  string(REPLACE ";" " " FUTILS_F90_FLAGS "${Fortran_FLAGS_DEBUG}")
  string(REPLACE ";" " " FUTILS_C_FLAGS "${CMAKE_C_FLAGS_DEBUG}")
endif()
if(CMAKE_BUILD_TYPE MATCHES Release)
  string(REPLACE ";" " " FUTILS_F90_FLAGS "${Fortran_FLAGS_RELEASE}")
  string(REPLACE ";" " " FUTILS_C_FLAGS "${CMAKE_C_FLAGS_RELEASE}")
endif()

string(REPLACE "SHELL:" "" FUTILS_F90_FLAGS "${FUTILS_F90_FLAGS}")
string(REPLACE "SHELL:" "" FUTILS_C_FLAGS "${FUTILS_C_FLAGS}")

set(FUTILS_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS})

if(ENABLE_SHARED_GENE3D_LIB)
  set(FUTILS_F90_FLAGS "${FUTILS_F90_FLAGS} -fPIC")
  set(FUTILS_C_FLAGS "${FUTILS_C_FLAGS} -fPIC")
  set(FUTILS_EXE_LINKER_FLAGS "${FUTILS_EXE_LINKER_FLAGS} -fPIC")
endif()

# enable preprocessing for .f90 files with gfortran
if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
  set(FUTILS_F90_FLAGS "${FUTILS_F90_FLAGS} -cpp")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES Intel)
  set(FUTILS_F90_FLAGS "${FUTILS_F90_FLAGS} -fpp")
endif()

find_package(HDF5 QUIET REQUIRED COMPONENTS Fortran C)
string(REPLACE ";" " " FUTILS_HDF5_LIBS "${HDF5_LIBRARIES}")
set(FUTILS_HDF5_PATH ${HDF5_ROOT})
if(NOT FUTILS_HDF5_PATH)
  find_path(FUTILS_HDF5_PATH
    HINTS
    $ENV{HDF5_ROOT}
    $ENV{HDF5_HOME}
    NAMES
    include/hdf5.h)
endif()

set(FUTILS_AR "${CMAKE_AR} rc")

message(STATUS "${Yellow}FUTILS_NAME: ${FUTILS_NAME}")
message(STATUS "FUTILS_SOURCE_DIR: ${FUTILS_SOURCE_DIR}")
message(STATUS "FUTILS_URL: ${FUTILS_URL}")
message(STATUS "FUTILS_MAKE: ${FUTILS_MAKE}")
message(STATUS "FUTILS_F90: ${FUTILS_F90}")
message(STATUS "FUTILS_F90_FLAGS: ${FUTILS_F90_FLAGS}")
message(STATUS "FUTILS_C: ${FUTILS_C}")
message(STATUS "FUTILS_C_FLAGS: ${FUTILS_C_FLAGS}")
message(STATUS "FUTILS_HDF5_LIBS: ${FUTILS_HDF5_LIBS}")
message(STATUS "FUTILS_HDF5_PATH: ${FUTILS_HDF5_PATH}")
message(STATUS "FUTILS_AR: ${FUTILS_AR}")
message(STATUS "FUTILS_EXE_LINKER_FLAGS: ${FUTILS_EXE_LINKER_FLAGS}${ColourReset}")

include(ExternalProject)
ExternalProject_Add(FUTILS-Project
  #URL ${FUTILS_URL}
  SOURCE_DIR ${FUTILS_SOURCE_DIR}
  INSTALL_DIR ${STAGED_INSTALL_PREFIX}
  CONFIGURE_COMMAND
  COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR> <BINARY_DIR>
  BUILD_COMMAND
  COMMAND
  ${FUTILS_MAKE} -C <BINARY_DIR>/src -j
  F90=${FUTILS_F90} OPT=${FUTILS_F90_FLAGS}
  CC=${FUTILS_C} C_FLAGS=${FUTILS_C_FLAGS}
  HDF5LIBS=${FUTILS_HDF5_LIBS} HDF5=${FUTILS_HDF5_PATH}
  AR=${FUTILS_AR} LD_FLAGS=${FUTILS_EXE_LINKER_FLAGS}
  libfutils.a
  INSTALL_COMMAND
  COMMAND ${CMAKE_COMMAND} -E make_directory <INSTALL_DIR>/${INSTALL_LIBDIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory <INSTALL_DIR>/${INSTALL_MODDIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  <BINARY_DIR>/src/libfutils.a <INSTALL_DIR>/${INSTALL_LIBDIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  <BINARY_DIR>/src/futils.mod <INSTALL_DIR>/${INSTALL_MODDIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  <BINARY_DIR>/src/hashtable.mod <INSTALL_DIR>/${INSTALL_MODDIR}
  BUILD_IN_SOURCE OFF
  BUILD_ALWAYS OFF
  LOG_DOWNLOAD ON
  LOG_BUILD ON
  LOG_INSTALL ON
  )

# cache futils-package directory
ExternalProject_Get_property(FUTILS-Project INSTALL_DIR)
set(FUTILS_DOCSTRING "Path to internally built FUTILS-Project")
set(FUTILS_DIR ${INSTALL_DIR} CACHE PATH "${FUTILS_DOCSTRING}" FORCE)
message(STATUS "${ColourBold}${FUTILS_DOCSTRING}: ${FUTILS_DIR}${ColourReset}")
