#ifndef SAENA_COOMATRIX_H
#define SAENA_COOMATRIX_H

#include <iostream>
#include <vector>

using namespace std;

#define MATRIX_TOL 1.e-6
/**
 * @author Majid
 * @breif Contains the basic structure to define coo matrices
 *
 * */
class COOMatrix {

private:
    unsigned int initial_nnz_l;
    int nprocs, rank;

    std::vector<long> data;
    std::vector<long> row;
    std::vector<long> col;
    std::vector<double> values;
    double *vSend;
    double* vecValues;
    std::vector<double> values_local;
    std::vector<double> values_remote;
    std::vector<long> col_local;
    std::vector<long> col_remote;
    std::vector<unsigned int> nnz_row_local;
    std::vector<unsigned int> nnz_row_remote;
    std::vector<double> invDiag;

    //    int* indicesP;
    int* indicesP_local;
    int* indicesP_remote;

    long *vIndex;
    std::vector<int> splitOffset;
    std::vector<int> vdispls;
    std::vector<int> rdispls;
    int recvSize;
    int vIndexSize;
    int numRecvProc;
    int numSendProc;
    std::vector<int> recvProcRank;
    std::vector<int> recvProcCount;
    std::vector<int> sendProcRank;
    std::vector<int> sendProcCount;
    std::vector<long> row_local;
    std::vector<long> row_remote;

    long num_threads;
    unsigned int* iter_local_array;
    unsigned int* iter_remote_array;
    std::vector<long> vElement_remote;
    std::vector<long> vElementRep_local;
    std::vector<long> vElementRep_remote;

    unsigned int nnz_l_local;
    unsigned int nnz_l_remote;
    long col_remote_size;

public:
    unsigned int M;
    unsigned int Mbig;
    unsigned int nnz_g;
    unsigned int nnz_l;
    std::vector<long> split;

    // functions
    void MatrixSetup();

    /**
     * */
    void matvec(double* v, double* w, double time[4]);
    void jacobi(double* v, double* w);
    void inverseDiag(double* x);
    void SaenaSetup();
    void SaenaSolve();
    void print();

    //COOMatrix();
    /**
     * @param[in] Aname is the pointer to the matrix
     * @param[in] Mbig Number of rows in the matrix
     * */
    COOMatrix(char* Aname, unsigned int Mbig);
    ~COOMatrix();

    // for sparsifying:
    //COOMatrix& newsize(int M, int N, int nnz);
    //double& set(int i, int j);

};

#endif //SAENA_COOMATRIX_H


