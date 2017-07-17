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
        include/coomatrix.h
        src/coomatrix.cpp
        include/AMGClass.h
        src/AMGClass.cpp
        include/strengthmatrix.h
        src/strengthmatrix.cpp
        src/usort/binUtils.cpp
        src/usort/parUtils.cpp
        include/prolongmatrix.h
        src/prolongmatrix.cpp
        include/restrictmatrix.h
        src/restrictmatrix.cpp
        include/auxFunctions.h
        src/auxFunctions.cpp
        include/grid.h
        src/grid.cpp
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
