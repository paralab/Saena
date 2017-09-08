#ifndef SAENA_RESTRICTMATRIX_H
#define SAENA_RESTRICTMATRIX_H

#include <vector>

class restrictMatrix {
// A matrix of this class is ordered first column-wise, then row-wise, using std:sort with cooEntry class "< operator".
// It is sorted in restrictMatrix constructor function in restrictmatrix.cpp.
private:

public:
    unsigned int M;
    unsigned int Mbig;
    unsigned int Nbig;
    unsigned long nnz_g;
    unsigned long nnz_l;
    unsigned long nnz_l_local;
    unsigned long nnz_l_remote;

    std::vector<cooEntry> entry;
    std::vector<cooEntry> entry_local;
    std::vector<cooEntry> entry_remote;
    std::vector<unsigned long> row_local;
    std::vector<unsigned long> row_remote;
    std::vector<unsigned long> col_remote; // index starting from 0, instead of the original column index

    std::vector<unsigned long> split;
    std::vector<unsigned long> splitNew;

    unsigned long* vIndex;
    double* vSend;
    double* vecValues;

    unsigned long col_remote_size; // number of remote columns. this is the same as vElement_remote.size()
    std::vector<unsigned int> nnzPerRow_local;
    std::vector<unsigned long> vElement_remote;
    std::vector<unsigned long> vElementRep_local;
    std::vector<unsigned long> vElementRep_remote;
    std::vector<unsigned int> nnzPerCol_remote; //todo: number of columns is large!
    std::vector<unsigned int> nnzPerRowScan_local;
    std::vector<int> vdispls;
    std::vector<int> rdispls;
    std::vector<int> recvProcRank;
    std::vector<int> recvProcCount;
    std::vector<int> sendProcRank;
    std::vector<int> sendProcCount;
    int vIndexSize;
    int recvSize;
    int numRecvProc;
    int numSendProc;

    unsigned long* indicesP_local;
    unsigned long* indicesP_remote;

    bool arrays_defined = false; // set to true if transposeP function is called. it will be used for destructor.

    MPI_Comm comm;

    restrictMatrix();
    int transposeP(prolongMatrix* P);
    ~restrictMatrix();
    int matvec(double* v, double* w);
};

#endif //SAENA_RESTRICTMATRIX_H
