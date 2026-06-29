find_package(PETSc QUIET)

if(PETSc_FOUND)
  message(STATUS "Found PETSc version ${PETSc_VERSION}")
  add_library(PETSc-Project INTERFACE)
else()
  set(PETSc_VERSION 3.11.0)
  set(PETSc_NAME "petsc-lite-${PETSc_VERSION}")
  set(PETSc_URL
    http://ftp.mcs.anl.gov/pub/petsc/release-snapshots/${PETSc_NAME}.tar.gz
    )

  message(STATUS "${ColourBold}"
    "PETSc could not be located, building PETSc ${PETSc_VERSION} instead."
    ${ColourReset})

  enable_language(C)
  enable_language(CXX)

  set(PETSc_CONFIGURE "configure")

  string( TOLOWER "${CMAKE_SYSTEM_NAME}" PETSc_SYSTEM_NAME )
  set(PETSc_ARCH ${PETSc_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR})

  if(CMAKE_MAKE_PROGRAM MATCHES make)
    set(PETSc_MAKE ${CMAKE_MAKE_PROGRAM})
  else()
    find_program(PETSc_MAKE
      NAMES make gmake
      DOC "Make command for PETSc-Project"
      # [HINTS path1 [path2 ... ENV var]] ## you can set the full path to the file here
      )
    if(NOT PETSc_MAKE)
      message(FATAL_ERROR "Couldn't find make command, please provide hints")
    endif()
  endif()

  find_package(MPI REQUIRED QUIET COMPONENTS Fortran C CXX)
  set(PETSc_FC ${MPI_Fortran_COMPILER})
  set(PETSc_C ${MPI_C_COMPILER})
  set(PETSc_CXX ${MPI_CXX_COMPILER})

  set(PETSc_WITH_PRECISION ${PRECISION})

  if(CMAKE_BUILD_TYPE MATCHES Debug)
    string(REPLACE ";" " " PETSc_FC_FLAGS "${Fortran_FLAGS_DEBUG}")
    string(REPLACE ";" " " PETSc_C_FLAGS "${CMAKE_C_FLAGS_DEBUG}")
    string(REPLACE ";" " " PETSc_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG}")
    set(PETSc_WITH_DEBUGGING "1")
  endif()
  if(CMAKE_BUILD_TYPE MATCHES Release)
    string(REPLACE ";" " " PETSc_FC_FLAGS "${Fortran_FLAGS_RELEASE}")
    string(REPLACE ";" " " PETSc_C_FLAGS "${CMAKE_C_FLAGS_RELEASE}")
    string(REPLACE ";" " " PETSc_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
    set(PETSc_WITH_DEBUGGING "0")
  endif()

  string(REPLACE "SHELL:" "" FUTILS_FC_FLAGS "${FUTILS_FC_FLAGS}")
  string(REPLACE "SHELL:" "" FUTILS_C_FLAGS "${FUTILS_C_FLAGS}")
  string(REPLACE "SHELL:" "" FUTILS_CXX_FLAGS "${FUTILS_CXX_FLAGS}")

  # PETSc build system generates useless and buggy CMake
  # configuration scripts, but it necessary for some
  # subprojects, e.g. METIS
  set(PETSc_CMAKE_ROOT ${CMAKE_ROOT})
  set(PETSc_WITH_CMAKE "1")

  if(CMAKE_Fortran_COMPILER_ID MATCHES GNU)
    set(PETSc_WITH_VENDOR_COMPILERS "gnu")
    set(PETSc_WITH_GNU_COMPILERS "1")
  elseif(CMAKE_Fortran_COMPILER_ID MATCHES Intel)
    set(PETSc_WITH_VENDOR_COMPILERS "intel")
    set(PETSc_WITH_GNU_COMPILERS "0")
  elseif(CMAKE_Fortran_COMPILER_ID MATCHES PGI)
    set(PETSc_WITH_VENDOR_COMPILERS "pgi")
    set(PETSc_WITH_GNU_COMPILERS "0")
  elseif(CMAKE_Fortran_COMPILER_ID MATCHES Cray)
    set(PETSc_WITH_VENDOR_COMPILERS "cray")
    set(PETSc_WITH_GNU_COMPILERS "0")
  elseif(CMAKE_Fortran_COMPILER_ID MATCHES XL)
    set(PETSc_WITH_VENDOR_COMPILERS "ibm")
    set(PETSc_WITH_GNU_COMPILERS "0")
  else()
    set(PETSc_WITH_GNU_COMPILERS "0")
  endif()

  find_package(MKL REQUIRED QUIET)
  set(PETSc_MKL_ROOT ${MKL_ROOT_DIR})

  # Workaround for not found -lintlc with Intel compilers when
  # --known-64-bit-blas-indices=0
  if(CMAKE_Fortran_COMPILER_ID MATCHES Intel)
    find_library(PETSc_C_LINK_LIBRARIES
      NAMES intlc
      PATHS ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES}
      )
  endif()

  find_program(SLURM_SRUN_COMMAND srun DOC "Path to the SLURM srun executable")
  if(SLURM_SRUN_COMMAND)
    find_program(TIMEOUT_COMMAND timeout DOC "Path to timeout command")
    if(TIMEOUT_COMMAND)
      set(WAIT_TIME "30s")
    endif()
    set(SRUN_NPROC  "-n 1")
    set(PETSc_CONFTEST "conftest-${PETSc_ARCH}")
    set(PETSc_RECONFIGURE_COMMAND "reconfigure-${PETSc_ARCH}.py")
    set(PETSc_WITH_BATCH 1)
  else()
    set(PETSc_WITH_BATCH 0)
  endif()

  if(APPLE)
    set(PETSc_RPATH_LINK "-Wl,-rpath,@loader_path/../lib")
  else()
    # expression for the relative path was found through try and error (and a lot of pain)
    # petsc homemade build system is a combination of python and gnu make, it is hard to pass
    # special symbols, because python and make interpret them differently :-(
    set(PETSc_RPATH_LINK "-Wl,-rpath,\\\\\$\$ORIGIN/../lib")
  endif()

  message(STATUS "${Yellow}PETSc_VERSION: ${PETSc_VERSION}")
  message(STATUS "PETSc_NAME: ${PETSc_NAME}")
  message(STATUS "PETSc_URL: ${PETSc_URL}")
  message(STATUS "PETSc_ARCH: ${PETSc_ARCH}")
  message(STATUS "PETSc_CONFIGURE: ${PETSc_CONFIGURE}")
  message(STATUS "PETSc_MAKE: ${PETSc_MAKE}")
  message(STATUS "PETSc_WITH_PRECISION: ${PETSc_WITH_PRECISION}")
  message(STATUS "PETSc_WITH_VENDOR_COMPILERS: ${PETSc_WITH_VENDOR_COMPILERS}")
  message(STATUS "PETSc_WITH_GNU_COMPILERS: ${PETSc_WITH_GNU_COMPILERS}")
  message(STATUS "PETSc_FC: ${PETSc_FC}")
  message(STATUS "PETSc_FC_FLAGS: ${PETSc_FC_FLAGS}")
  message(STATUS "PETSc_C: ${PETSc_C}")
  message(STATUS "PETSc_C_FLAGS: ${PETSc_C_FLAGS}")
  message(STATUS "PETSc_CXX: ${PETSc_CXX}")
  message(STATUS "PETSc_CXX_FLAGS: ${PETSc_CXX_FLAGS}")
  message(STATUS "PETSc_WITH_DEBUGGING: ${PETSc_WITH_DEBUGGING}")
  message(STATUS "PETSc_C_LINK_LIBRARIES: ${PETSc_C_LINK_LIBRARIES}")
  message(STATUS "PETSc_CMAKE_ROOT: ${PETSc_CMAKE_ROOT}")
  message(STATUS "PETSc_WITH_CMAKE: ${PETSc_WITH_CMAKE}")
  message(STATUS "PETSc_MKL_ROOT: ${PETSc_MKL_ROOT}")
  message(STATUS "PETSc_WITH_BATCH: ${PETSc_WITH_BATCH}")
  message(STATUS "PETSc_RPATH_LINK: ${PETSc_RPATH_LINK}")
  message(STATUS "PETSc_CONFTEST: ${PETSc_CONFTEST}")
  message(STATUS "PETSc_BATCH_COMMAND:"
    " ${TIMEOUT_COMMAND} ${WAIT_TIME}"
    " ${SLURM_SRUN_COMMAND} ${SRUN_NPROC} ${PETSc_CONFTEST}")
  message(STATUS "PETSc_RECONFIGURE_COMMAND: ${PETSc_RECONFIGURE_COMMAND}${ColourReset}")

  include(ExternalProject)
  ExternalProject_Add(PETSc-Project
    URL ${PETSc_URL}
    INSTALL_DIR ${STAGED_INSTALL_PREFIX}/petsc
    CONFIGURE_COMMAND
    COMMAND
    <SOURCE_DIR>/${PETSc_CONFIGURE}
    --prefix=<INSTALL_DIR>
    --PETSC_ARCH=${PETSc_ARCH}
    --MAKEFLAGS=-j
    --LIBS=${PETSc_C_LINK_LIBRARIES}
    --with-cc=${PETSc_C}
    --COPTFLAGS=${PETSc_C_FLAGS}
    --CC_LINKER_FLAGS=${PETSc_RPATH_LINK}
    --with-fc=${PETSc_FC}
    --FOPTFLAGS=${PETSc_FC_FLAGS}
    --FC_LINKER_FLAGS=${PETSc_RPATH_LINK}
    --with-fortran=1
    --with-cxx=${PETSc_CXX}
    --CXXOPTFLAGS=${PETSc_CXX_FLAGS}
    --CXX_LINKER_FLAGS=${PETSc_RPATH_LINK}
    --with-debugging=${PETSc_WITH_DEBUGGING}
    --with-mpi-compilers=1
    --with-vendor-compilers=${PETSc_WITH_VENDOR_COMPILERS}
    --with-gnu-compilers=${PETSc_WITH_GNU_COMPILERS}
    --known-mpi-shared-libraries=1
    --with-precision=${PETSc_WITH_PRECISION}
    --with-scalar-type=real
    --with-blaslapack-dir=${PETSc_MKL_ROOT}
    --known-64-bit-blas-indices=0
    --download-scalapack=1
    --download-mumps=1
    --download-hypre=1
    --with-cxx-dialect=C++11
    --download-superlu_dist=1
    --download-parmetis=1
    --download-metis=1
    --with-batch=${PETSc_WITH_BATCH}
    --with-64-bit-pointers=1
    --with-x=0
    --with-cmake-dir=${PETSc_CMAKE_ROOT}
    --with-cmake=${PETSc_WITH_CMAKE}
    --with-clib-autodetect=1
    COMMAND
    ${TIMEOUT_COMMAND} ${WAIT_TIME} ${SLURM_SRUN_COMMAND} ${SRUN_NPROC} ${PETSc_CONFTEST}
    COMMAND $<$<BOOL:PETSc_WITH_BATCH>:<SOURCE_DIR>/${PETSc_RECONFIGURE_COMMAND}>
    BUILD_COMMAND
    COMMAND ${PETSc_MAKE}
    INSTALL_COMMAND
    COMMAND ${PETSc_MAKE} install
    BUILD_IN_SOURCE ON
    BUILD_ALWAYS OFF
    LOG_DOWNLOAD ON
    LOG_BUILD ON
    LOG_INSTALL ON
    )

  set(PETSc_ROOT_DIR ${STAGED_INSTALL_PREFIX}/petsc
    CACHE PATH "Path to internally build PETSc installation root"
    FORCE
    )
  message(STATUS "${ColourBold}Path to internally build PETSc installation root:"
    " ${PETSc_ROOT_DIR}${ColourReset}")
endif()
