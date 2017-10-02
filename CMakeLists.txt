#SET(CMAKE_CXX_COMPILER mpicxx)
#SET(CMAKE_C_COMPILER mpicc)

cmake_minimum_required(VERSION 2.8)
project(Saena)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
#set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

include_directories(
        include
        include/usort
)

set(SOURCE_FILES
        src/main.cpp
        include/saena_matrix.h
        src/saena_matrix.cpp
        include/saena_object.h
        src/saena_object.cpp
        include/strength_matrix.h
        src/strength_matrix.cpp
        include/prolong_matrix.h
        src/prolong_matrix.cpp
        include/restrict_matrix.h
        src/restrict_matrix.cpp
        include/aux_functions.h
        src/aux_functions.cpp
        include/grid.h
        src/grid.cpp
        include/saena.hpp
        src/saena.cpp
        src/pugixml.cpp

        include/usort/binUtils.h
        include/usort/dtypes.h
        include/usort/indexHolder.h
        include/usort/ompUtils.h
        include/usort/ompUtils.tcc
        include/usort/parUtils.h
        include/usort/parUtils.tcc
        include/usort/seqUtils.h
        include/usort/seqUtils.tcc
        include/usort/sort_profiler.h
        src/usort/binUtils.cpp
        src/usort/parUtils.cpp
        src/usort/sort_profiler.cpp
        )

add_executable(Saena ${SOURCE_FILES})

if(MPI_COMPILE_FLAGS)
    set_target_properties(Saena PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(Saena PROPERTIES
            LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

target_link_libraries(Saena ${MPI_LIBRARIES})
