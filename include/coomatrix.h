#ifndef SAENA_COOMATRIX_H
#define SAENA_COOMATRIX_H

#include <iostream>
#include <vector>
#include <mpi.h>
#include "auxFunctions.h"

using namespace std;

//#define MATRIX_TOL 1.e-6

/**
 * @author Majid
 * @breif Contains the basic structure to define coo matrices
 *
 * */
class COOMatrix {
// A matrix of this class is order first column-wise, then row-wise.

private:
    unsigned int initial_nnz_l;
    int nprocs, rank;

    std::vector<unsigned long> data; // todo: change data from vector to malloc. then free it, when you are done repartitiong.

public:
//    std::vector<unsigned long> row;
//    std::vector<unsigned long> col;
//    std::vector<double> values;
//    std::vector<cooEntry> entry;
    cooEntry* entryP;

    unsigned int M;
    unsigned int Mbig;
    unsigned int nnz_g;
    unsigned int nnz_l;
    std::vector<unsigned long> split;

    int vIndexSize;
    long *vIndex;
    double *vSend;
    unsigned long *vSendULong;
    double* vecValues;
    unsigned long* vecValuesULong;
    std::vector<double> values_local;
    std::vector<double> values_remote;
    std::vector<unsigned long> row_local;
    std::vector<unsigned long> row_remote;
    std::vector<unsigned long> col_local;
    std::vector<unsigned long> col_remote; // index starting from 0, instead of the original column index
    std::vector<unsigned long> col_remote2; //original col index
//    std::vector<unsigned int> nnzPerRow;
    std::vector<unsigned int> nnzPerRow_local; // todo: this is used for openmp part of coomatrix.cpp
//    std::vector<unsigned int> nnzPerRow_remote;
//    std::vector<unsigned int> nnzPerRowScan_local;
//    std::vector<unsigned int> nnzPerRowScan_remote;
//    std::vector<unsigned int> nnzPerCol_local;
//    std::vector<unsigned int> nnzPerColScan_local;
//    std::vector<unsigned int> nnz_row_remote;
    std::vector<unsigned int> nnz_col_remote;
    std::vector<double> invDiag;

    //    int* indicesP;
    unsigned long* indicesP_local;
    unsigned long* indicesP_remote;

    std::vector<int> splitOffset;
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

    unsigned int nnz_l_local;
    unsigned int nnz_l_remote;
    unsigned long col_remote_size;

    // functions
    void MatrixSetup(MPI_Comm comm);
    void matvec(double* v, double* w, double time[4], MPI_Comm comm);
    void jacobi(double* v, double* w, MPI_Comm comm);
    void inverseDiag(double* x);
    void SaenaSetup();
    void SaenaSolve();
    void print();

    //COOMatrix();
    /**
     * @param[in] Aname is the pointer to the matrix
     * @param[in] Mbig Number of rows in the matrix
     * */
    COOMatrix(char* Aname, unsigned int Mbig, MPI_Comm comm);
    ~COOMatrix();

    // for sparsifying:
    //COOMatrix& newsize(int M, int N, int nnz);
    //double& set(int i, int j);

};

#endif //SAENA_COOMATRIX_H


