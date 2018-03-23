//
// Created by Majid on 3/22/18.
//

#ifndef MATVEC003_SAENA_MATRIX_DENSE_H
#define MATVEC003_SAENA_MATRIX_DENSE_H

#include <mpi.h>
#include "vector"

typedef unsigned int index_t;
typedef unsigned long nnz_t;
typedef double value_t;

class saena_matrix_dense {

private:

public:

    index_t Mbig = 0, Nbig = 0, M = 0;
    value_t **entry;
    MPI_Comm comm;

//    index_t col_local_start, col_local_end;

    std::vector<index_t> split; // (row-wise) partition of the matrix between processes

    saena_matrix_dense();
    saena_matrix_dense(index_t M, index_t Nbig);
    saena_matrix_dense(index_t M, index_t Nbig, MPI_Comm comm);
//    saena_matrix_dense(char* Aname, MPI_Comm com);
    ~saena_matrix_dense();

    int set(index_t row, index_t col, value_t val);
//    int set(index_t* row, index_t* col, value_t* val, nnz_t nnz_local);
//    int set2(index_t row, index_t col, value_t val);
//    int set2(index_t* row, index_t* col, value_t* val, nnz_t nnz_local);

    int print(int ran);

    int matvec(std::vector<value_t>& v, std::vector<value_t>& w);

};

#endif //MATVEC003_SAENA_MATRIX_DENSE_H
