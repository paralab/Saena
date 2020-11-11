# Saena

Saena is the name of falcon, the fastest animal, in Avesta (an old Persian book), and this library is supposed to be the falcon in multigrid solvers.

### Dependencies
There are some dependencies which are installed automatically by running the `install.sh` file.
Some of those dependencies are `ZFP` and `SuperLU`. `SuperLU` also depends on `ParMETIS`.
The other ones that may be needed to be installed before compiling Saena are the following.

#### 1- Boost:
For `Linux` this may help:

`sudo apt-get install libboost-all-dev`

#### 2- Intel MKL:
If you don't already have `Intel MKL` on your machine, it can be installed in a couple of ways explained on its website:

`https://software.intel.com/en-us/get-started-with-mkl-for-linux`

The easiest way may be using the newer intel compilers, which includes mkl; Or just installing the standalone version:

`https://software.seek.intel.com/performance-libraries`

### Datatypes
The following datatype are used in Saena:

    typedef int    index_t; // index type
    typedef long   nnz_t;   // nonzero type
    typedef double value_t; // value type
    
They can be changed in `include/data_type.h`.

ParMETIS and SuperLU index sizes are set to 32. They can be changed too (they should be changed together):
- ParMETIS: set `XSDK_INDEX_SIZE` in `external/SuperLU_DIST_5.4.0/CMakeLists.txt`.
- SuperLU: set `IDXTYPEWIDTH` in `external/parmetis-4.0.3/metis/include/metis.h`.
