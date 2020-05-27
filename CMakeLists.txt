cmake_minimum_required(VERSION 3.10)
project(Saena)

set(CMAKE_CXX_STANDARD 14) #CombBLAS requires 14
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -----------------------------------------------------------------------
# Build Type: Debug or Release

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

IF(CMAKE_BUILD_TYPE STREQUAL Debug)
    add_definitions(-D__DEBUG1__)
ENDIF()

message(--\ Saena:\ CMAKE_BUILD_TYPE:\ ${CMAKE_BUILD_TYPE})

# -----------------------------------------------------------------------
# OpenMP

find_package(OpenMP)
if(OPENMP_FOUND)
    set (CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_definitions(-D_USE_OPENMP_)
endif()

# -----------------------------------------------------------------------
# MPI

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

SET(SAENA_LIB_DEPENDENCIES ${SAENA_LIB_DEPENDENCIES} ${MPI_LIBRARIES})

# -----------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------
# SuperLU

set(SuperLU_DIST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/SuperLU_DIST_5.4.0)
set(SuperLU_DIST_LIB ${CMAKE_BINARY_DIR}/build_superlu/lib/libsuperlu_dist.a)
include_directories(
        ${SuperLU_DIST_DIR}/build/include
        ${SuperLU_DIST_DIR}/SRC
)

SET(SAENA_LIB_DEPENDENCIES ${SAENA_LIB_DEPENDENCIES} ${SuperLU_DIST_LIB})

# -----------------------------------------------------------------------
# METIS and PARMETIS

#set(METIS_INCLUDE_DIR    ${CMAKE_BINARY_DIR}/build_metis/include)
#set(PARMETIS_INCLUDE_DIR ${CMAKE_BINARY_DIR}/build_parmetis/include)
set(METIS_LIBRARY        ${CMAKE_BINARY_DIR}/build_metis/lib/libmetis.so)
set(PARMETIS_LIBRARY     ${CMAKE_BINARY_DIR}/build_parmetis/lib/libparmetis.a)

SET(SAENA_LIB_DEPENDENCIES ${SAENA_LIB_DEPENDENCIES} ${PARMETIS_LIBRARY} ${METIS_LIBRARY})

# -----------------------------------------------------------------------
# BOOST

#find_package(BOOST REQUIRED)
#message(STATUS BOOST found: ${BOOST_LIBRARIES})
#include_directories(${Boost_INCLUDE_DIRS})

# For Colorado Summit cluster:
#include_directories("/curc/sw/intel/17.4/mkl/include")
# For Frontera:
#include_directories(/opt/apps/intel19/python2_7/boost/1.69/include)

# -----------------------------------------------------------------------
# BLAS

find_package(BLAS REQUIRED)
message(STATUS BLAS found: ${BLAS_LIBRARIES})

SET(SAENA_LIB_DEPENDENCIES ${SAENA_LIB_DEPENDENCIES} ${BLAS_LIBRARIES})

# -----------------------------------------------------------------------
# LAPACK

find_package(LAPACK REQUIRED)
message( STATUS LAPACK found: ${LAPACK_LIBRARIES} )
include_directories(${LAPACK_INCLUDE_PATH})
#set(LAPACK_LIB /usr/lib/openblas-base/liblapack.a)

SET(SAENA_LIB_DEPENDENCIES ${SAENA_LIB_DEPENDENCIES} ${LAPACK_LIBRARIES})

# -----------------------------------------------------------------------
# MKL

set(MKLROOT $ENV{MKLROOT})
include_directories(${MKLROOT}/include)

set(MKL_LIB_DIR    "${MKLROOT}/lib/intel64_lin")

set(MKL_LIB_CORE   "${MKL_LIB_DIR}/libmkl_core.so")
set(MKL_LIB_LP64   "${MKL_LIB_DIR}/libmkl_intel_lp64.so")

set(MKL_LIB_THREAD "${MKL_LIB_DIR}/libmkl_gnu_thread.so")
#set(MKL_LIB_THREAD "${MKL_LIB_DIR}/libmkl_intel_thread.so")

#set(MKL_LIB_BLACS "${MKL_LIB_DIR}/libmkl_blacs_intelmpi_lp64.so")
#set(MKL_LIB_BLACS "${MKL_LIB_DIR}/libmkl_blacs_openmpi_lp64.so")

#set(MKL_OMP "/opt/intel/compilers_and_libraries_2020.0.166/linux/compiler/lib/intel64_lin/libiomp5.so")

SET(SAENA_LIB_DEPENDENCIES ${SAENA_LIB_DEPENDENCIES} ${MKL_LIB_CORE} ${MKL_LIB_LP64} ${MKL_LIB_THREAD})

# -----------------------------------------------------------------------
# IETL: eigensolver

include_directories(external/ietl)
set(HEADER_FILES ${HEADER_FILES}
        include/ietl_saena.h
        )

# -----------------------------------------------------------------------
# usort: to sort elements in parallel

include_directories(external/usort/include)
set(SOURCE_FILES ${SOURCE_FILES}
        external/usort/src/binUtils.cpp
        external/usort/src/parUtils.cpp
        external/usort/src/sort_profiler.cpp
        )

# -----------------------------------------------------------------------
# pugixml: to read xml files for options

#include_directories(external/pugixml/include)
#set(SOURCE_FILES ${SOURCE_FILES}
#        external/pugixml/src/pugixml.cpp
#        )

# -----------------------------------------------------------------------
# ZFP: for compressing arrays

set(ZFP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/zfp)
set(ZFP_LIB ${CMAKE_CURRENT_BINARY_DIR}/build_zfp/lib/libzfp.so)
include_directories(
        ${ZFP_DIR}/array
        ${ZFP_DIR}/array/zfp
        ${ZFP_DIR}/include
)

SET(SAENA_LIB_DEPENDENCIES ${SAENA_LIB_DEPENDENCIES} ${ZFP_LIB})

# -----------------------------------------------------------------------
#PETSc

option(ENABLE_PETSC OFF)

if(ENABLE_PETSC)
    if(NOT DEFINED ENV{PETSC_DIR})
        message( FATAL_ERROR "PETSC_DIR is not set! CMake will exit." )
    endif(NOT DEFINED ENV{PETSC_DIR})

    if(NOT DEFINED ENV{PETSC_ARCH})
        message( FATAL_ERROR "PETSC_ARCH is not set! CMake will exit." )
    endif(NOT DEFINED ENV{PETSC_ARCH})

    set(PETSC_DIR $ENV{PETSC_DIR})
    set(PETSC_ARCH $ENV{PETSC_ARCH})
    message(--\ PETSC_DIR:\ ${PETSC_DIR})
    message(--\ PETSC_ARCH:\ ${PETSC_ARCH})

    set(PETSC_LIB ${PETSC_DIR}/${PETSC_ARCH}/lib/libpetsc.so)
    include_directories(
            ${PETSC_DIR}/include
            ${PETSC_DIR}/${PETSC_ARCH}/include
    )

    SET(SAENA_LIB_DEPENDENCIES ${SAENA_LIB_DEPENDENCIES} ${PETSC_LIB})

    set(HEADER_FILES ${HEADER_FILES} include/petsc_functions.h)
    set(SOURCE_FILES ${SOURCE_FILES} src/petsc_functions.cpp)

    add_compile_definitions(_USE_PETSC_)

endif(ENABLE_PETSC)

# -----------------------------------------------------------------------
#CombBLAS

option(ENABLE_COMBBLAS OFF)

if(ENABLE_COMBBLAS)
    if(NOT DEFINED ENV{COMBBLAS_DIR})
        message( FATAL_ERROR "COMBBLAS_DIR is not set! CMake will exit." )
    endif(NOT DEFINED ENV{COMBBLAS_DIR})

    set(COMBBLAS_DIR $ENV{COMBBLAS_DIR})
    message(--\ COMBBLAS_DIR:\ ${COMBBLAS_DIR})

    include_directories("${COMBBLAS_DIR}/install/include")
    set(COMBBLAS_LIB ${COMBBLAS_DIR}/install/lib/libCombBLAS.a)

    SET(SAENA_LIB_DEPENDENCIES ${SAENA_LIB_DEPENDENCIES} ${COMBBLAS_LIB})

    set(HEADER_FILES ${HEADER_FILES} include/combblas_functions.h)
    set(SOURCE_FILES ${SOURCE_FILES} src/combblas_functions.cpp)

    add_compile_definitions(_USE_COMBBLAS_)

endif(ENABLE_COMBBLAS)

# -----------------------------------------------------------------------
# dollar: for profiling
# https://github.com/r-lyeh-archived/dollar

set(DOLLAR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/dollar")
include_directories(
        ${DOLLAR_DIR}
)

# -----------------------------------------------------------------------
# trsl: for sampling

#set(TRSL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/trsl-0.2.2)
#include_directories(
#        ${TRSL_DIR}
#)

# -----------------------------------------------------------------------
# Main Files

include_directories(include)

set(HEADER_FILES ${HEADER_FILES}
        include/saena.hpp
        include/aux_functions.h
        include/data_struct.h
        include/saena_matrix.h
        include/saena_matrix_dense.h
        include/saena_object.h
        include/strength_matrix.h
        include/prolong_matrix.h
        include/restrict_matrix.h
        include/grid.h
        include/saena_vector.h
        include/GR_encoder.h
        )

set(SOURCE_FILES ${SOURCE_FILES}
        ${HEADER_FILES}
        src/saena.cpp
        src/aux_functions.cpp
        src/data_struct.cpp
        src/strength_matrix.cpp
        src/prolong_matrix.cpp
        src/restrict_matrix.cpp
        src/saena_matrix_dense.cpp
        src/saena_vector.cpp
        src/GR_encoder.cpp

        src/saena_matrix.cpp
        src/saena_matrix_repart.cpp
        src/saena_matrix_setup.cpp
        src/saena_matrix_matvec.cpp
        src/saena_matrix_shrink.cpp
        src/saena_matrix_zfp.cpp

        src/saena_object.cpp
        src/saena_object_setup1.cpp
        src/saena_object_setup2.cpp
        src/saena_object_setup3.cpp
        src/saena_object_setup_matmat.cpp
        src/saena_object_solve.cpp
        src/saena_object_lazy.cpp
        src/saena_object_repart_shrink.cpp
        src/saena_object_sparsify.cpp
        src/saena_object_elemental.cpp
        src/saena_object_write2file.cpp
        )

# -----------------------------------------------------------------------

# Intel flags:
# -c                see Output, Debug, and PCH Options
# -g                see Output, Debug, and PCH Options
# -xCORE-AVX512     for Cascade Lake Processors (like Frontera) and Skylake
# -xCORE-AVX2       for Broadwell and Haswell
# -xAVX             for Ivy Bridge and Sandy Bridge

# Enable Intel Optimization flags
#IF(CMAKE_BUILD_TYPE STREQUAL Release)
#    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -O3 -axCORE-AVX512,CORE-AVX2,AVX -xSSE4.2")
#    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -axCORE-AVX512,CORE-AVX2,AVX -xSSE4.2")
#ENDIF()

IF(CMAKE_BUILD_TYPE STREQUAL Release)
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
ENDIF()

# -----------------------------------------------------------------------

# Enable the Intel Optimization Reports
# -qopt-report[=N]   N=1-5 specifies an increasing level of detail
# -qopt-report-file   =stdout, stderr, filename
# -qopt-report-phase  =loop, vec, par, cg, openmp, ipo, pgo, offload, all
# -qopt-report-routine=fn1[,fn2,…]

#set(REPORT_ARG1 "-qopt-report=5 -qopt-report-file=stdout -qopt-report-phase=loop,vec,par,cg")
#set(REPORT_ROUTINE "-qopt-report-routine=reorder_split")
#set(REPORT_FILES "src/saena_object_setup_matmat.cpp")
#set(REPORT_FILE "-qopt-report-filter=${REPORT_FILES}")

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${REPORT_ARG1} ${REPORT_ROUTINE}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${REPORT_ARG1} ${REPORT_ROUTINE}")

# -----------------------------------------------------------------------

#Compile
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -std=c++14 -DTHREADED -fopenmp -DNDEBUG -w -DGRAPH_GENERATOR_SEQ -O2 -DMPICH_IGNORE_CXX_SEEK")
#SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DNDEBUG -Drestrict=__restrict__ -w -DGRAPH_GENERATOR_SEQ -O2 -DMPICH_IGNORE_CXX_SEEK")

#add_executable(Saena src/main.cpp             ${SOURCE_FILES} ${HEADER_FILES})
#add_executable(Saena src/main_read_file.cpp   ${SOURCE_FILES} ${HEADER_FILES})
#add_executable(Saena src/main_read_2file.cpp  ${SOURCE_FILES} ${HEADER_FILES})
#add_executable(Saena src/main_band_matrix.cpp ${SOURCE_FILES} ${HEADER_FILES})
#add_executable(Saena src/main_lazy.cpp        ${SOURCE_FILES} ${HEADER_FILES})
add_executable(Saena src/main_zfp.cpp         ${SOURCE_FILES} ${HEADER_FILES})
#add_executable(Saena src/main_matmat.cpp      ${SOURCE_FILES} ${HEADER_FILES})
#add_executable(Saena src/main_matmat_read_2file.cpp ${SOURCE_FILES} ${HEADER_FILES})

# -----------------------------------------------------------------------

#Link
#message("SAENA_LIB_DEPENDENCIES:\ " ${SAENA_LIB_DEPENDENCIES})
target_link_libraries(Saena ${SAENA_LIB_DEPENDENCIES})