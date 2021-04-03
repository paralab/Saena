#pragma once

#include "saena_matrix.h"

inline void saena_matrix::matvec(value_t *v, value_t *w){
    if(use_dense){
        dense_matrix->matvec(v, w);
    }else{
        if(use_double) matvec_sparse(v, w);
        else matvec_sparse_float(v, w);
//        matvec_sparse_zfp(v, w);
    }
}

// Vector res = A * u - rhs;
inline void saena_matrix::residual(value_t *u, const value_t *rhs, value_t *res){
    matvec(&u[0], &res[0]);
#pragma omp parallel for
    for(index_t i = 0; i < M; ++i){
        res[i] -= rhs[i];
    }
}

// Vector res = rhs - A * u
inline void saena_matrix::residual_negative(value_t *u, const value_t *rhs, value_t *res){
    matvec(&u[0], &res[0]);
#pragma omp parallel for
    for(index_t i = 0; i < M; i++){
        res[i] = rhs[i] - res[i];
    }
}