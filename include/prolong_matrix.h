#ifndef SAENA_PROLONG_MATRIX_H
#define SAENA_PROLONG_MATRIX_H

#include <vector>
#include <mpi.h>
#include "aux_functions.h"

typedef unsigned int index_t;
typedef unsigned long nnz_t;
typedef double value_t;


class prolong_matrix {
// A matrix of this class is ordered <<ONLY>> if it is defined in createProlongation function in saena_object.cpp.
// Otherwise it can be ordered using the following line:
//#include <algorithm>
//std::sort(P->entry.begin(), P->entry.end());
// It is ordered first column-wise, then row-wise, using std:sort with cooEntry class "< operator".
// Duplicates are removed in createProlongation function in saena_object.cpp.
private:

public:
    index_t Mbig = 0;
    index_t Nbig = 0;
    index_t M = 0;
    nnz_t nnz_g = 0;
    nnz_t nnz_l = 0;

    nnz_t nnz_l_local  = 0;
    nnz_t nnz_l_remote = 0;
    index_t col_remote_size = 0; // this is the same as vElement_remote.size()

    std::vector<index_t> split;
    std::vector<index_t> splitNew;

//    std::vector<unsigned long> row;
//    std::vector<unsigned long> col;
//    std::vector<double> values;

    std::vector<cooEntry> entry;
    std::vector<cooEntry> entry_local;
    std::vector<cooEntry> entry_remote;
    std::vector<index_t> row_local;
    std::vector<index_t> row_remote;
    std::vector<index_t> col_remote; // index starting from 0, instead of the original column index
//    std::vector<unsigned long> col_remote2; //original col index
//    std::vector<double> values_local;
//    std::vector<double> values_remote;
//    std::vector<unsigned long> col_local;

    std::vector<index_t> nnzPerRow_local;
    std::vector<index_t> nnzPerRowScan_local;
    std::vector<index_t> nnzPerCol_remote;
    std::vector<index_t> vElement_remote;
    std::vector<index_t> vElement_remote_t;
    std::vector<index_t> vElementRep_local;
    std::vector<index_t> vElementRep_remote;
//    std::vector<unsigned int> nnz_row_remote;

    bool arrays_defined = false; // set to true if findLocalRemote function is called. it will be used for destructor.
    int vIndexSize;
    int vIndexSize_t;
    std::vector<index_t> vIndex;
    std::vector<value_t> vSend;
    std::vector<cooEntry> vSend_t;
    std::vector<value_t> vecValues;
    std::vector<cooEntry> vecValues_t;

    std::vector<int> vdispls;
    std::vector<int> vdispls_t;
    std::vector<int> rdispls;
    std::vector<int> rdispls_t;
    std::vector<int> recvProcRank;
    std::vector<int> recvProcRank_t;
    std::vector<int> recvProcCount;
    std::vector<int> recvProcCount_t;
    std::vector<int> sendProcRank;
    std::vector<int> sendProcRank_t;
    std::vector<int> sendProcCount;
    std::vector<int> sendProcCount_t;
    int recvSize;
    int recvSize_t;
    int numRecvProc;
    int numRecvProc_t;
    int numSendProc;
    int numSendProc_t;

    unsigned int num_threads;
    std::vector<nnz_t> iter_local_array;
    std::vector<nnz_t> iter_remote_array;
    std::vector<value_t> w_buff; // for matvec

    std::vector<nnz_t> indicesP_local;
    std::vector<nnz_t> indicesP_remote;

    MPI_Comm comm;

    bool verbose_prolong_setup = false;

    prolong_matrix();
    prolong_matrix(MPI_Comm com);
    ~prolong_matrix();
    int findLocalRemote();
    int openmp_setup();
    int matvec(std::vector<value_t>& v, std::vector<value_t>& w);
    int print(int ran);
};

#endif //SAENA_PROLONG_MATRIX_H