#ifndef SAENA_SaenaMatrix_H
#define SAENA_SaenaMatrix_H

#include <iostream>
#include <vector>
#include <set>
#include <mpi.h>
#include "auxFunctions.h"

using namespace std;

//#define MATRIX_TOL 1.e-6

/**
 * @author Majid
 * @breif Contains the basic structure to define coo matrices
 *
 * */
class SaenaMatrix {
// A matrix of this class is ordered first column-wise, then row-wise.

private:
    unsigned int initial_nnz_l;
    bool freeBoolean = false; // use this parameter to know if deconstructor for SaenaMatrix class should free the variables or not.
    std::set<cooEntry> data_coo;
    std::vector<unsigned long> data; // todo: change data from vector to malloc. then free it, when you are done repartitioning.

public:
//    std::vector<unsigned long> row;
//    std::vector<unsigned long> col;
//    std::vector<double> values;
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


    SaenaMatrix();
    SaenaMatrix(unsigned int num_rows_global);
    /**
     * @param[in] Aname is the pointer to the matrix
     * @param[in] Mbig Number of rows in the matrix
     * */
    SaenaMatrix(char* Aname, unsigned int Mbig, MPI_Comm comm);
    ~SaenaMatrix();
//    int reserve(unsigned int nnz_local, unsigned int num_rows_global);
    int set(unsigned int row, unsigned int col, double val);
    int set(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local);
    int setup_initial_data(MPI_Comm comm);
    int repartition(MPI_Comm comm);
    int matrixSetup(MPI_Comm comm);
    int matvec(double* v, double* w, MPI_Comm comm);
    int jacobi(std::vector<double>& u, std::vector<double>& rhs, MPI_Comm comm);
    int inverseDiag(double* x, MPI_Comm comm);
    int SaenaSetup();
    int SaenaSolve();
    int print();
    int Destroy();

    // for sparsifying:
    //SaenaMatrix& newsize(int M, int N, int nnz);
    //double& set(int i, int j);

};

#endif //SAENA_SaenaMatrix_H


