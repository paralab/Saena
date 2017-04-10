#ifndef SAENA_STRENGTHMATRIX_H
#define SAENA_STRENGTHMATRIX_H

#include <vector>

class StrengthMatrix {
public:
//    std::vector<long>   rowIndex;
//    std::vector<long>   col;
//    std::vector<double> values;

//    long*   rowIndex;
//    long*   col;
//    double* values;

    unsigned long* split;

    long M;
    long Mbig;
    long nnz_l;
//    long nnz_g;
//    double average_sparsity;

    unsigned int nnz_l_local;
    unsigned int nnz_l_remote;
    unsigned long col_remote_size; // this is the same as vElement_remote.size()

    int vIndexSize;
    unsigned long* vSend;
//    int* vSend2;
    long* vIndex;
    unsigned long* vecValues;
//    int* vecValues2;
    std::vector<double> values_local;
    std::vector<double> values_remote;
    std::vector<unsigned long> row_local;
    std::vector<unsigned long> row_remote;
    std::vector<unsigned long> col_local;
    std::vector<unsigned long> col_remote; // index starting from 0, instead of the original column index
    std::vector<unsigned long> col_remote2; //original col index
    std::vector<unsigned int> nnz_row_local;
//    std::vector<unsigned int> nnz_row_remote;
    std::vector<unsigned int> nnz_col_remote;
    std::vector<unsigned long> vElement_remote;
    std::vector<unsigned long> vElementRep_local;
    std::vector<unsigned long> vElementRep_remote;
    unsigned long* indicesP_local;
    unsigned long* indicesP_remote;

//    std::vector<int> splitOffset;
    std::vector<int> vdispls;
    std::vector<int> rdispls;
    int recvSize;
    int numRecvProc;
    int numSendProc;
    std::vector<int> recvProcRank;
    std::vector<int> recvProcCount;
    std::vector<int> sendProcRank;
    std::vector<int> sendProcCount;

//    StrengthMatrix(){}
    int StrengthMatrixSet(unsigned long* row, unsigned long* col, double* values, long M, long Mbig, long nnzl, unsigned long* split);
    ~StrengthMatrix();
    void print(int rank);
};

#endif //SAENA_STRENGTHMATRIX_H
