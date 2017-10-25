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
        external/usort/include
)

add_subdirectory(external/Elemental-0.87.7)
include_directories("${PROJECT_BINARY_DIR}/external/El/include")
include_directories(${MPI_CXX_INCLUDE_PATH})

include_directories("build/install/include")
include_directories("build/external/Elemental-0.87.7/include")
include_directories("external/Elemental-0.87.7/include")
include_directories("external/Elemental-0.87.7/external/suite_sparse/include")

include_directories("external/pugixml/include")
include_directories("external/usort/include")

set(SOURCE_FILES
        src/main.cpp
        src/saena_matrix.cpp
        src/saena_object.cpp
        src/strength_matrix.cpp
        src/prolong_matrix.cpp
        src/restrict_matrix.cpp
        src/aux_functions.cpp
        src/grid.cpp
        src/saena.cpp

        external/pugixml/src/pugixml.cpp

        external/usort/src/binUtils.cpp
        external/usort/src/parUtils.cpp
        external/usort/src/sort_profiler.cpp
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

target_link_libraries(Saena ${MPI_LIBRARIES} El)
