#ifndef SAENA_STRENGTH_MATRIX_H
#define SAENA_STRENGTH_MATRIX_H

#include <vector>
#include "mpi.h"


typedef unsigned int index_t;
typedef unsigned long nnz_t;
typedef double value_t;

class strength_matrix {

public:

//    MPI_Comm comm = MPI_COMM_WORLD;
    index_t M    = 0;
    index_t Mbig = 0;
    nnz_t nnz_l  = 0;
//    long nnz_g;
//    double average_sparsity;

    nnz_t nnz_l_local  = 0;
    nnz_t nnz_l_remote = 0;
    index_t col_remote_size = 0; // this is the same as vElement_remote.size()

    index_t vIndexSize = 0;
    std::vector<index_t> vIndex;
    std::vector<unsigned long> vSend;
    std::vector<unsigned long> vecValues;

    std::vector<index_t> split;

    std::vector<value_t> values_local;
    std::vector<value_t> values_remote;
    std::vector<index_t> row_local;
    std::vector<index_t> row_remote;
    std::vector<index_t> col_local;
    std::vector<index_t> col_remote; // index starting from 0, instead of the original column index
    std::vector<index_t> col_remote2; //original col index
    std::vector<nnz_t> nnzPerRow;
    std::vector<nnz_t> nnzPerRow_local;
//    std::vector<unsigned int> nnzPerRow_remote;
    std::vector<nnz_t> nnz_col_remote;
    std::vector<index_t> vElement_remote;
    std::vector<index_t> vElementRep_local;
    std::vector<index_t> vElementRep_remote;
    std::vector<index_t> indicesP_local;
    std::vector<index_t> indicesP_remote;

    std::vector<int> vdispls;
    std::vector<int> rdispls;
    nnz_t recvSize = 0;
    int numRecvProc = 0;
    int numSendProc = 0;
    std::vector<int> recvProcRank;
    std::vector<int> recvProcCount;
    std::vector<int> sendProcRank;
    std::vector<int> sendProcCount;

    MPI_Comm comm;

//    strength_matrix(){}
    int strength_matrix_set(std::vector<index_t>& row, std::vector<index_t>& col, std::vector<value_t >& values,
                            index_t M, index_t Mbig, nnz_t nnzl, std::vector<index_t>& split, MPI_Comm com);
    ~strength_matrix();
    int erase();
    void print(int rank);
};

#endif //SAENA_STRENGTH_MATRIX_H
