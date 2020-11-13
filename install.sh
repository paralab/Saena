if [ -d "build" ]; then
    rm -rf build
fi
mkdir build && cd build

# metis
make config prefix=`pwd` -C ../external/parmetis-4.0.3/metis
cd build_metis
make -j$(nproc)
make -j$(nproc) install
cd ..

#parmetis
make config prefix=`pwd` -C ../external/parmetis-4.0.3
cd build_parmetis
make -j$(nproc)
make -j$(nproc) install
cd ..

# SuperLU
mkdir build_superlu && cd build_superlu
cmake ../../external/SuperLU_DIST_5.4.0 \
-Denable_blaslib=OFF \
-DCMAKE_INSTALL_PREFIX=. ;\
make -j$(nproc)
make -j$(nproc) install
cd ..

#zfp
#mkdir build_zfp && cd build_zfp
#cmake ../../external/zfp
#cmake --build . --config Release
#cd ..

# Saena
cmake ..
make -j$(nproc)
cd ..
