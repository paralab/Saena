#ifndef SAENA_RESTRICT_MATRIX_H
#define SAENA_RESTRICT_MATRIX_H

#include <vector>

class prolong_matrix;

typedef unsigned int index_t;
typedef unsigned long nnz_t;
typedef double value_t;



class restrict_matrix {
// A matrix of this class is ordered first column-wise, then row-wise, using std:sort with cooEntry class "< operator".
// It is sorted in restrictMatrix constructor function in restrictmatrix.cpp.
private:

public:
    index_t M;
    index_t Mbig;
    index_t Nbig;
    nnz_t nnz_g;
    nnz_t nnz_l;
    nnz_t nnz_l_local;
    nnz_t nnz_l_remote;

    std::vector<cooEntry> entry;
    std::vector<cooEntry> entry_local;
    std::vector<cooEntry> entry_remote;
    std::vector<index_t> row_local;
    std::vector<index_t> row_remote;
    std::vector<index_t> col_remote; // index starting from 0, instead of the original column index

    std::vector<index_t> split;
    std::vector<index_t> splitNew;

    std::vector<index_t> vIndex;
    std::vector<value_t> vSend;
    std::vector<value_t> vecValues;

    index_t col_remote_size; // number of remote columns. this is the same as vElement_remote.size()
    std::vector<index_t> nnzPerRow_local;
    std::vector<index_t> vElement_remote;
    std::vector<index_t> vElementRep_local;
    std::vector<index_t> vElementRep_remote;
    std::vector<index_t> nnzPerCol_remote; //todo: number of columns is large!
    std::vector<nnz_t> nnzPerRowScan_local;
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

    std::vector<nnz_t> indicesP_local;
    std::vector<nnz_t> indicesP_remote;

    bool arrays_defined = false; // set to true if transposeP function is called. it will be used for destructor.

    MPI_Comm comm;

    restrict_matrix();
    ~restrict_matrix();
    int transposeP(prolong_matrix* P);
    int matvec(std::vector<value_t>& v, std::vector<value_t>& w);
};

#endif //SAENA_RESTRICT_MATRIX_H