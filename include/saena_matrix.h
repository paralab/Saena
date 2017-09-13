#ifndef SAENA_SAENA_MATRIX_H
#define SAENA_SAENA_MATRIX_H

#include <iostream>
#include <vector>
#include <set>
#include <mpi.h>
#include "aux_functions.h"

/**
 * @author Majid
 * @breif Contains the basic structure to define coo matrices
 *
 * */
class saena_matrix {
// A matrix of this class is ordered first column-wise, then row-wise.

private:
    unsigned int initial_nnz_l;
    bool freeBoolean = false; // use this parameter to know if deconstructor for SaenaMatrix class should free the variables or not.
    std::set<cooEntry> data_coo;
    std::vector<unsigned long> data; // todo: change data from vector to malloc. then free it, when you are done repartitioning.

public:
    std::vector<cooEntry> entry;

    unsigned int M;
    unsigned int Mbig;
    unsigned int nnz_g;
    unsigned int nnz_l;
    std::vector<unsigned long> split;

    unsigned int nnz_l_local;
    unsigned int nnz_l_remote;
    unsigned long col_remote_size; // number of remote columns
    std::vector<double> values_local;
    std::vector<double> values_remote;
    std::vector<unsigned long> row_local;
    std::vector<unsigned long> row_remote;
    std::vector<unsigned long> col_local;
    std::vector<unsigned long> col_remote; // index starting from 0, instead of the original column index
    std::vector<unsigned long> col_remote2; //original col index
    std::vector<unsigned int> nnzPerRow_local; // todo: this is used for openmp part of SaenaMatrix.cpp
    std::vector<unsigned int> nnzPerCol_remote; // todo: replace this. nnz Per Column is expensive.
//    std::vector<unsigned int> nnzPerRow;
//    std::vector<unsigned int> nnzPerRow_remote;
//    std::vector<unsigned int> nnzPerRowScan_local;
//    std::vector<unsigned int> nnzPerRowScan_remote;
//    std::vector<unsigned int> nnzPerCol_local;
//    std::vector<unsigned int> nnzPerColScan_local;
//    std::vector<unsigned int> nnz_row_remote;

    std::vector<double> invDiag;
    double norm1, normInf, rhoDA;

    int vIndexSize;
    long *vIndex;
    double *vSend;
    unsigned long *vSendULong;
    double* vecValues;
    unsigned long* vecValuesULong;

//    unsigned long* indicesP;
    unsigned long* indicesP_local;
    unsigned long* indicesP_remote;

    std::vector<int> vdispls;
    std::vector<int> rdispls;
    int recvSize;
    int numRecvProc;
    int numSendProc;
    std::vector<int> recvProcRank;
    std::vector<int> recvProcCount;
    std::vector<int> sendProcRank;
    std::vector<int> sendProcCount;

    unsigned long num_threads;
    unsigned int* iter_local_array;
    unsigned int* iter_remote_array;
    std::vector<unsigned long> vElement_remote;
    std::vector<unsigned long> vElementRep_local;
    std::vector<unsigned long> vElementRep_remote;

    MPI_Comm comm;

    saena_matrix();
//    SaenaMatrix(MPI_Comm com);
    saena_matrix(MPI_Comm com);
    /**
     * @param[in] Aname is the pointer to the matrix
     * @param[in] Mbig Number of rows in the matrix
     * */
    saena_matrix(char* Aname, MPI_Comm com);
    ~saena_matrix();
    // difference between set and set2 is that if there is a repetition, set will erase the previous one
    // and insert the new one, but in set2, the values of those entries will be added.
    int set(unsigned int row, unsigned int col, double val);
    int set(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local);
    int set2(unsigned int row, unsigned int col, double val);
    int set2(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local);
    int setup_initial_data();
    int repartition();
    int matrix_setup();
    int matvec(double* v, double* w);
    int jacobi(std::vector<double>& u, std::vector<double>& rhs);
    int inverse_diag(double* x);
    int destroy();
};

#endif //SAENA_SAENA_MATRIX_H


