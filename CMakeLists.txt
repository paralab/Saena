# ccmake -D CMAKE_INSTALL_PREFIX="./elemental_install" -D CMAKE_BUILD_TYPE="Release" ..
# or
# ccmake -D CMAKE_INSTALL_PREFIX="./elemental_install" -D CMAKE_BUILD_TYPE="Release" -D GFORTRAN_LIB="/usr/lib64/libgfortran.so.3" ..

cmake_minimum_required(VERSION 3.3)
project(Saena)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)
if(OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

include_directories(
        include
        external/ietl
        external/usort/include
        external/pugixml/include

#        build/elemental_install/include
#        build/external/Elemental-0.87.7/include
#        external/Elemental-0.87.7/include
#        external/Elemental-0.87.7/external/suite_sparse/include
)

#add_subdirectory(external/Elemental-0.87.7)
#include_directories("${PROJECT_BINARY_DIR}/external/Elemental-0.87.7/elemental_install/include")
#include_directories("${PROJECT_BINARY_DIR}/external/El/include")
#include_directories(${MPI_CXX_INCLUDE_PATH})

set(SOURCE_FILES
        include/saena.hpp
        include/saena_matrix.h
        include/saena_matrix_dense.h
        include/saena_object.h
        include/strength_matrix.h
        include/prolong_matrix.h
        include/restrict_matrix.h
        include/aux_functions.h
        include/grid.h

        src/main.cpp
        src/saena_matrix.cpp
        src/saena_matrix_dense.cpp
        src/saena_object.cpp
        src/strength_matrix.cpp
        src/prolong_matrix.cpp
        src/restrict_matrix.cpp
        src/aux_functions.cpp
        src/grid.cpp
        src/saena.cpp

        external/pugixml/src/pugixml.cpp

        external/usort/include/binUtils.h
        external/usort/include/dtypes.h
        external/usort/include/indexHolder.h
        external/usort/include/ompUtils.h
        external/usort/include/ompUtils.tcc
        external/usort/include/parUtils.h
        external/usort/include/parUtils.tcc
        external/usort/include/seqUtils.h
        external/usort/include/seqUtils.tcc
        external/usort/include/sort_profiler.h
        external/usort/src/binUtils.cpp
        external/usort/src/parUtils.cpp
        external/usort/src/sort_profiler.cpp
        include/ietl_saena.h)

add_executable(Saena ${SOURCE_FILES})

if(MPI_COMPILE_FLAGS)
    set_target_properties(Saena PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(Saena PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

#target_link_libraries(Saena ${MPI_LIBRARIES} El lapack blas)
target_link_libraries(Saena ${MPI_LIBRARIES} lapack blas)
#target_link_libraries(Saena ${MPI_LIBRARIES} El)
#target_link_libraries(Saena ${MPI_LIBRARIES})
