#message("SAENA_LIB_DEPENDENCIES:\ " ${SAENA_LIB_DEPENDENCIES})

#add_executable(basic basic.cpp ${SAENA_SOURCE_FILES} ${SAENA_HEADER_FILES})
#target_link_libraries(basic ${SAENA_LIB_DEPENDENCIES})

#add_executable(file file.cpp ${SAENA_SOURCE_FILES} ${SAENA_HEADER_FILES})
#target_link_libraries(file ${SAENA_LIB_DEPENDENCIES})

#add_executable(file2 file2.cpp ${SAENA_SOURCE_FILES} ${SAENA_HEADER_FILES})
#target_link_libraries(file2 ${SAENA_LIB_DEPENDENCIES})

#add_executable(banded banded.cpp ${SAENA_SOURCE_FILES} ${SAENA_HEADER_FILES})
#target_link_libraries(banded ${SAENA_LIB_DEPENDENCIES})

#add_executable(lazy lazy.cpp ${SAENA_SOURCE_FILES} ${SAENA_HEADER_FILES})
#target_link_libraries(lazy ${SAENA_LIB_DEPENDENCIES})

#add_executable(zfp zfp.cpp ${SAENA_SOURCE_FILES} ${SAENA_HEADER_FILES})
#target_link_libraries(zfp ${SAENA_LIB_DEPENDENCIES})

#add_executable(zfp_dense zfp_dense.cpp  ${SAENA_SOURCE_FILES} ${SAENA_HEADER_FILES})
#target_link_libraries(zfp_dense ${SAENA_LIB_DEPENDENCIES})

#add_executable(gemm gemm.cpp ${SAENA_SOURCE_FILES} ${SAENA_HEADER_FILES})
#target_link_libraries(gemm ${SAENA_LIB_DEPENDENCIES})

#add_executable(gemm_2file gemm_2file.cpp ${SAENA_SOURCE_FILES} ${SAENA_HEADER_FILES})
#target_link_libraries(gemm_2file ${SAENA_LIB_DEPENDENCIES})

#add_executable(gemm_MG gemm_MG.cpp ${SAENA_SOURCE_FILES} ${SAENA_HEADER_FILES})
#target_link_libraries(gemm_MG ${SAENA_LIB_DEPENDENCIES})

add_executable(poisson Poisson.cpp ${SAENA_SOURCE_FILES} ${SAENA_HEADER_FILES})
target_link_libraries(poisson ${SAENA_LIB_DEPENDENCIES})

#add_executable(poisson_petsc Poisson_PETSc.cpp ${SAENA_SOURCE_FILES} ${SAENA_HEADER_FILES})
#target_link_libraries(poisson_petsc ${SAENA_LIB_DEPENDENCIES})

add_executable(profile_f profile_file.cpp ${SAENA_SOURCE_FILES} ${SAENA_HEADER_FILES})
target_link_libraries(profile_f ${SAENA_LIB_DEPENDENCIES})

#add_executable(profile_f_petsc profile_file_PETSc.cpp ${SAENA_SOURCE_FILES} ${SAENA_HEADER_FILES})
#target_link_libraries(profile_f_petsc ${SAENA_LIB_DEPENDENCIES})