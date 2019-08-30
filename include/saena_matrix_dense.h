#ifndef MATVEC003_SAENA_MATRIX_DENSE_H
#define MATVEC003_SAENA_MATRIX_DENSE_H

#include "vector"
#include <mpi.h>
#include <cstdlib>

typedef unsigned int index_t;
typedef unsigned long nnz_t;
typedef double value_t;

class saena_matrix;

class saena_matrix_dense {

//private:

public:

    bool     allocated = false;
    index_t  M         = 0;
    index_t  Nbig      = 0;
    value_t  **entry   = nullptr;
    MPI_Comm comm      = MPI_COMM_WORLD;

    std::vector<index_t> split; // (row-wise) partition of the matrix between processes

    saena_matrix_dense();
    saena_matrix_dense(index_t M, index_t Nbig);
    saena_matrix_dense(index_t M, index_t Nbig, MPI_Comm comm);
    saena_matrix_dense(const saena_matrix_dense &B); // copy constructor
//    saena_matrix_dense(char* Aname, MPI_Comm com);
    ~saena_matrix_dense();

    saena_matrix_dense& operator=(const saena_matrix_dense &B);

    int erase();

    value_t get(index_t row, index_t col){
        if(row >= M || col >= Nbig){
            printf("\ndense matrix get out of range!\n");
            exit(EXIT_FAILURE);
        }else{
            return entry[row][col];
        }
    }

    void set(index_t row, index_t col, value_t val){
        if(row >= M || col >= Nbig){
            printf("\ndense matrix set out of range!\n");
            exit(EXIT_FAILURE);
        }else{
            entry[row][col] = val;
        }
    }

//    int set(index_t* row, index_t* col, value_t* val, nnz_t nnz_local);
//    int set2(index_t row, index_t col, value_t val);
//    int set2(index_t* row, index_t* col, value_t* val, nnz_t nnz_local);

    int print(int ran);
    int matvec(std::vector<value_t>& v, std::vector<value_t>& w);
    int convert_saena_matrix(saena_matrix *A);
};

#endif //MATVEC003_SAENA_MATRIX_DENSE_H
