if [ -d "build" ]; then
    rm -rf build
fi
mkdir build && cd build

# metis
make config prefix=`pwd` -C ../external/parmetis-4.0.3/metis
cd build_metis
make install
cd ..

#parmetis
make config prefix=`pwd` -C ../external/parmetis-4.0.3
cd build_parmetis
make install
cd ..

#send_zfp
mkdir build_zfp && cd build_zfp
cmake ../../external/zfp-0.5.3
make -j28
cd ..

# SuperLU
mkdir build_superlu && cd build_superlu
cmake ../../external/SuperLU_DIST_5.4.0 \
-Denable_blaslib=OFF \
-DCMAKE_INSTALL_PREFIX=. ;\
make -j28 install
cd ..

# Saena
cmake ..
make -j28
cd ..
