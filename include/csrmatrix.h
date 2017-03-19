#ifndef SAENA_CSRMATRIX_H
#define SAENA_CSRMATRIX_H

#include <vector>

class CSRMatrix {
public:
//    std::vector<long>   rowIndex;
//    std::vector<long>   col;
//    std::vector<double> values;

//    long*   rowIndex;
//    long*   col;
//    double* values;

    long* split;

    long M;
    long Mbig;
    long nnz_l;
    long nnz_g;
    double average_sparsity;



    unsigned int nnz_l_local;
    unsigned int nnz_l_remote;
    long col_remote_size;

    int vIndexSize;
    long *vSend;
    long *vIndex;
    long* vecValues;
    std::vector<double> values_local;
    std::vector<double> values_remote;
    std::vector<long> row_local;
    std::vector<long> row_remote;
    std::vector<long> col_local;
    std::vector<long> col_remote; // index starting from 0, instead of the original column index
    std::vector<long> col_remote2; //original col index
    std::vector<unsigned int> nnz_row_local;
    std::vector<unsigned int> nnz_row_remote;
    std::vector<long> vElement_remote;
    std::vector<long> vElementRep_local;
    std::vector<long> vElementRep_remote;
    long* indicesP_local;
    long* indicesP_remote;

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

//    CSRMatrix(){}
    int CSRMatrixSet(long* row, long* col, double* values, long M, long Mbig, long nnzl, long* split);
    ~CSRMatrix();
    void print(int rank);
};

#endif //SAENA_CSRMATRIX_H
