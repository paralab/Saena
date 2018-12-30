############################################################################
#
#  Program:         SuperLU_DIST
#
#  Module:          make.inc
#
#  Purpose:         Top-level Definitions
#
#  Creation date:   March 1, 2016	version 5.0.0
#
#  Modified:	    October 13, 2017    version 5.2.1
#		    
#
############################################################################
#
#  The name of the libraries to be created/linked to
#
SuperLUroot = /home/majidrp/Dropbox/Projects/Saena/build/build_superlu
DSUPERLULIB = $(SuperLUroot)/SRC/libsuperlu_dist.a
INCLUDEDIR  = $(SuperLUroot)/include

LIBS 	    = $(DSUPERLULIB) /usr/lib/x86_64-linux-gnu/libblas.so /home/majidrp/Dropbox/Projects/Saena/build/build_superlu/../build_parmetis/libparmetis/libparmetis.a /home/majidrp/Dropbox/Projects/Saena/build/build_superlu/../build_parmetis/libmetis/libmetis.a
LIBS 	    += 

#
#  The archiver and the flag(s) to use when building archive (library)
#  If your system has no ranlib, set RANLIB = echo.
#
ARCH         = /usr/bin/ar
ARCHFLAGS    = cr
RANLIB       = /usr/bin/ranlib

CC           = /usr/bin/cc
CFLAGS 	     = -O3 -DNDEBUG -I/home/majidrp/Dropbox/Projects/Saena/external/SuperLU_DIST_5.4.0/../parmetis-4.0.3/metis/include -I/home/majidrp/Dropbox/Projects/Saena/external/SuperLU_DIST_5.4.0/../parmetis-4.0.3/include -DUSE_VENDOR_BLAS -fopenmp   -fPIC
#CFLAGS      += -D
# CFLAGS     +=  
CXX          = 
CXXFLAGS     = -O3    -fPIC
XSDK_INDEX_SIZE=32
HAVE_PARMETIS=TRUE
HAVE_COMBBLAS=
NOOPTS       = -O0
FORTRAN	     = /usr/bin/gfortran

LOADER       = $(CC)
LOADOPTS     =  -Wl,-rpath -Wl,/usr/local/lib -Wl,--enable-new-dtags -fopenmp
