#ifndef SAENA_PROLONGMATRIX_H
#define SAENA_PROLONGMATRIX_H

#include <vector>

class prolongMatrix {
private:

public:
    unsigned long Mbig;
    unsigned long Nbig;
    unsigned long M;
    unsigned long nnz_g;
    unsigned long nnz_l;

    std::vector<unsigned long> row;
    std::vector<unsigned long> col;
    std::vector<double> values;


    unsigned int nnz_l_local;
    unsigned int nnz_l_remote;
    unsigned long col_remote_size; // this is the same as vElement_remote.size()

    unsigned long* split;

    int vIndexSize;
    int vIndexSize_t;
    unsigned long* vSend;
    double* vSend_t;
    unsigned long* vIndex;
    unsigned long* recvIndex_t;
    unsigned long* vecValues;
    double* vecValues_t;
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
    std::vector<unsigned int> nnz_col_remote_t;
    std::vector<unsigned long> vElement_remote;
    std::vector<unsigned long> vElement_remote_t;
    std::vector<unsigned long> vElementRep_local;
    std::vector<unsigned long> vElementRep_remote;
    unsigned long* indicesP_local;
//    unsigned long* indicesP_remote;

//    std::vector<int> splitOffset;
    std::vector<int> vdispls;
    std::vector<int> vdispls_t;
    std::vector<int> rdispls;
    std::vector<int> rdispls_t;
    int recvSize;
    int recvSize_t;
    int numRecvProc;
    int numSendProc;
    std::vector<int> recvProcRank;
    std::vector<int> recvProcCount;
    std::vector<int> recvProcCount_t;
    std::vector<int> sendProcRank;
    std::vector<int> sendProcCount;
    std::vector<int> sendProcCount_t;


    prolongMatrix();
//    prolongMatrix(unsigned long Mbig, unsigned long Nbig, unsigned long nnz_g, unsigned long nnz_l, unsigned long* row, unsigned long* col, double* values);
    ~prolongMatrix();
    int findLocalRemote(unsigned long* row, unsigned long* col, double* values);
};


#endif //SAENA_PROLONGMATRIX_H
