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
SuperLUroot = /uufs/chpc.utah.edu/common/home/u6025515/Saena/build/build_superlu
DSUPERLULIB = $(SuperLUroot)/SRC/libsuperlu_dist.a
INCLUDEDIR  = $(SuperLUroot)/include

LIBS 	    = $(DSUPERLULIB) /uufs/chpc.utah.edu/sys/installdir/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64_lin/libmkl_gf_lp64.so /uufs/chpc.utah.edu/sys/installdir/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64_lin/libmkl_gnu_thread.so /uufs/chpc.utah.edu/sys/installdir/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64_lin/libmkl_core.so /usr/lib/gcc/x86_64-redhat-linux/4.8.5/libgomp.so -lm -ldl /uufs/chpc.utah.edu/common/home/u6025515/Saena/build/build_superlu/../build_parmetis/libparmetis/libparmetis.a /uufs/chpc.utah.edu/common/home/u6025515/Saena/build/build_superlu/../build_parmetis/libmetis/libmetis.a
LIBS 	    += 

#
#  The archiver and the flag(s) to use when building archive (library)
#  If your system has no ranlib, set RANLIB = echo.
#
ARCH         = /bin/ar
ARCHFLAGS    = cr
RANLIB       = /bin/ranlib

CC           = /usr/lib64/ccache/cc
CFLAGS 	     = -O3 -DNDEBUG -I/uufs/chpc.utah.edu/common/home/u6025515/Saena/external/SuperLU_DIST_5.4.0/../parmetis-4.0.3/metis/include -I/uufs/chpc.utah.edu/common/home/u6025515/Saena/external/SuperLU_DIST_5.4.0/../parmetis-4.0.3/include -DUSE_VENDOR_BLAS -fopenmp     -std=c99 -fPIC
#CFLAGS      += -D
# CFLAGS     +=  
CXX          = 
CXXFLAGS     = -O3    -std=c99 -fPIC
XSDK_INDEX_SIZE=32
HAVE_PARMETIS=TRUE
HAVE_COMBBLAS=
NOOPTS       = -O0
FORTRAN	     = /bin/gfortran

LOADER       = $(CC)
LOADOPTS     =  -Xlinker --enable-new-dtags -Xlinker -rpath -Xlinker /uufs/chpc.utah.edu/sys/installdir/intel/compilers_and_libraries_2019.5.281/linux/mpi/intel64/lib/release -Xlinker -rpath -Xlinker /uufs/chpc.utah.edu/sys/installdir/intel/compilers_and_libraries_2019.5.281/linux/mpi/intel64/lib -Xlinker --enable-new-dtags -L/uufs/chpc.utah.edu/sys/installdir/intel/compilers_and_libraries_2019.5.281/linux/mpi/intel64/lib -fopenmp
