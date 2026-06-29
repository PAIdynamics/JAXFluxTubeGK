# GKW makefile
#============================================================================
### EXECUTABLE name
### (change the name here if you wish)
EXEC=../gkw.x

#============================================================================
### PREPROCESSOR OPTIONS
### (it is unlikely you need to modify anything beyond the first 2 options 
###  below)

## --- FFT library ---
## The preferred FFT library is FFTW3; interfaces for some others may
## exist in the fft module of the code. To switch off FFT's altogther (since
## they are not needed for linear runs) set `FFTLIB=nofft'.
FFTLIB=fft_w3
# FFTLIB=nofft

## --- Parallel switch ---
## For compiling without MPI, select `PARALLEL=nompi'.
PARALLEL=mpi
#PARALLEL=nompi

## --- Set the *real* precision ---
## This option is to aid with calling MPI and FFTW3 routines in single
## precision. If you do not wish to use either of these, but wish to run in
## single precision, you should be able to do so by changing only the compiler
## flags in the compiler options section below. When using MPI and/or FFTW3,
## Set the option here to `real_precision_default' to enable single precision
## (in addition to setting the compiler flags below). To link with the single
## precision FFTW3, the linking flag in compiler options may need to be change
## to -lfftw3f. It appears that GKW will run fine in single precision, saving
## some time and memory. However, we give no assurances with regards to the
## solutions obtained.
REAL_PRECISION=real_precision_double
#REAL_PRECISION=real_precision_default

## NO NEED TO MODIFY THE LINE BELOW UNLESS MORE OPTIONS ARE ADDED
FPREPROCS = -D$(FFTLIB)  \
			-D$(PARALLEL) \
			-D$(REAL_PRECISION)

#============================================================================
### COMPILER OPTIONS

## --- FORTRAN compiler ---
## The default here is a wrapper provided by most MPI implementations.
FC=mpif90
#FC=gfortran

## --- FORTRAN compiler flags ---
## We use double precision as the default, so please replace `-r8' with the
## appropriate switch for your compiler (gfortran uses -fdefault-real-8).
## Most compilers support `-O2' to turn on reasonable optimisation.
FFLAGS= -r8 -O2
#FFLAGS= -fdefault-real-8 -O2 #gfortran

## --- Include PATH ---
## You may need to add something here if the MPI, FFTW or other library header
## files are installed in an unusual location.
INC=
#INC= -I/usr/include
#INC= -I/usr/local/some_directory/include/

## --- LINKER ---
## (usually, the FORTRAN compiler should be fine for linking)
LD=$(FC) $(FFLAGS)

## --- linking flags ---
## It may be necessary to add flags here if you did not use the `mpif90', or
## similar wrapper for FC. If you do not wish to use FFTW3 you
## can (and should) remove the default flag here. As with the include PATH,
## if your library is in an unusual place, you may need to set that path here.
LDFLAGS=-lfftw3
#LDFLAGS= -L/usr/local -lfftw3
#LDFLAGS= -L/usr/local/some_directory/lib -lfftw3f

#############################################################################
#############################################################################
#---------- YOU SHOULD NOT NEED TO EDIT ANYTHING BEYOND HERE ---------------#
#           (unless more sources are added to the code)                     #
#############################################################################
#############################################################################

#============================================================================
### rules

.SUFFIXES:
.SUFFIXES: .f90 .F90 .o

.f90.o:
	    $(FC) $(FFLAGS) $(INC) -o $*.o -c $*.f90
.F90.o:
	    $(FC) $(FFLAGS) $(INC) $(FPREPROCS) -o $*.o -c $*.F90

#============================================================================
### required FORTRAN objects

F90OBJ =  init.o \
          constants.o \
          mpicomms.o \
		  components.o \
          diagnostic.o \
          io.o \
          exp_integration.o \
          geom.o \
          matdat.o \
          control.o \
          general.o \
          grid.o \
          global.o \
          dist.o \
          fft.o \
          mode.o \
          linear_terms.o \
          non_linear_terms.o \
          normalise.o \
	  mpiinterface.o \
      index_function.o \
    mpidatatypes.o \
          collisionop.o \
          functions.o \
          velocitygrid.o \
          rotation.o \
		  specfun.o \
		  marsaglia.o

GKWOBJ=linart.o $(F90OBJ)

#============================================================================
### various targets 

default: $(EXEC)

$(EXEC) : $(GKWOBJ)
	$(LD) $(LDFLAGS) $(GKWOBJ) -o $(EXEC)

clean:
	  rm -f *.o *.mod work.pc work.cc1 $(EXEC)

#============================================================================
### Deps.

collisionop.o : collisionop.F90 components.o constants.o control.o dist.o general.o geom.o global.o grid.o matdat.o mpicomms.o mpiinterface.o specfun.o velocitygrid.o
components.o : components.F90 control.o dist.o general.o global.o grid.o io.o mpiinterface.o rotation.o
constants.o : constants.f90
control.o : control.F90 fft.o general.o global.o mpiinterface.o
diagnostic.o : diagnostic.F90 components.o control.o dist.o fft.o functions.o general.o geom.o global.o grid.o index_function.o io.o mode.o mpicomms.o mpidatatypes.o mpiinterface.o non_linear_terms.o normalise.o rotation.o velocitygrid.o
dist.o : dist.F90 control.o general.o grid.o index_function.o mpidatatypes.o mpiinterface.o
exp_integration.o : exp_integration.F90 collisionop.o components.o control.o dist.o general.o global.o grid.o matdat.o mpicomms.o mpidatatypes.o mpiinterface.o non_linear_terms.o normalise.o rotation.o
fft.o : fft.F90
functions.o : functions.F90 components.o geom.o mode.o mpidatatypes.o mpiinterface.o specfun.o velocitygrid.o
general.o : general.F90 global.o mpiinterface.o
geom.o : geom.F90 constants.o control.o general.o global.o grid.o mpiinterface.o
global.o : global.f90
grid.o : grid.F90 control.o general.o global.o mpicomms.o mpiinterface.o
index_function.o : index_function.f90 general.o global.o grid.o
init.o : init.f90 collisionop.o components.o constants.o control.o diagnostic.o dist.o exp_integration.o general.o geom.o grid.o marsaglia.o matdat.o mode.o mpiinterface.o non_linear_terms.o rotation.o velocitygrid.o
io.o : io.F90 global.o mpiinterface.o
linart.o : linart.f90 control.o diagnostic.o exp_integration.o general.o init.o linear_terms.o matdat.o mpiinterface.o
linear_terms.o : linear_terms.F90 collisionop.o components.o constants.o control.o dist.o functions.o general.o geom.o global.o grid.o index_function.o matdat.o mode.o mpicomms.o mpiinterface.o rotation.o specfun.o velocitygrid.o
marsaglia.o : marsaglia.f90
matdat.o : matdat.F90 control.o dist.o general.o grid.o mpiinterface.o
mode.o : mode.F90 constants.o control.o fft.o general.o geom.o global.o grid.o mpiinterface.o
mpicomms.o : mpicomms.f90
mpidatatypes.o : mpidatatypes.F90 general.o global.o grid.o index_function.o mpiinterface.o
mpiinterface.o : mpiinterface.F90 global.o
non_linear_terms.o : non_linear_terms.F90 components.o constants.o control.o dist.o fft.o functions.o general.o geom.o grid.o mode.o mpiinterface.o rotation.o velocitygrid.o
normalise.o : normalise.F90 constants.o control.o dist.o global.o grid.o mode.o mpicomms.o mpiinterface.o
rotation.o : rotation.F90 constants.o control.o dist.o fft.o general.o geom.o global.o grid.o mode.o mpiinterface.o
specfun.o : specfun.f90
velocitygrid.o : velocitygrid.F90 control.o general.o geom.o grid.o mpiinterface.o
