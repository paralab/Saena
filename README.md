# Saena

Saena is the name of falcon, the fastest animal, in Avesta (an old Persian book), and this library is supposed to be the falcon in multigrid solvers.

To see an example, check the Examples folder.

By using "cmake .."  without any argument, Elemental will be installed in system folders, so it needs root password. To avoid that, use "-D CMAKE_INSTALL_PREFIX" argument to set where it should be installed in a folder called "install" inside the build folder:
cmake -D CMAKE_INSTALL_PREFIX="./elemental_install" ..
