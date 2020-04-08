# Saena

Saena is the name of falcon, the fastest animal, in Avesta (an old Persian book), and this library is supposed to be the falcon in multigrid solvers.

### Dependencies

#### Boost:
For `Linux` this may help:

`sudo apt-get install libboost-all-dev`

#### Intel MKL:
If you don't already have `Intel MKL` on your machine, it can be installed in a couple of ways explained on its website:

`https://software.intel.com/en-us/get-started-with-mkl-for-linux`

The easiest way may be installing the standalone version:

`https://software.seek.intel.com/performance-libraries`

There are other dependencies which are installed automatically by running the `install.sh` file. Some of those dependencies are `ZFP` and also `SuperLU`, which depends on `ParMETIS`.