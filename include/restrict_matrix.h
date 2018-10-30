#ifndef SAENA_RESTRICT_MATRIX_H
#define SAENA_RESTRICT_MATRIX_H

#include "aux_functions.h"

#include <vector>
#include <mpi.h>

class prolong_matrix;

typedef unsigned int index_t;
typedef unsigned long nnz_t;
typedef double value_t;


class restrict_matrix {
// A matrix of this class is ordered first column-wise, then row-wise, using std:sort with cooEntry class "< operator".
// It is sorted in restrictMatrix constructor function in restrictmatrix.cpp.
private:

public:
    index_t M    = 0;
    index_t Mbig = 0;
    index_t Nbig = 0;
    nnz_t nnz_g  = 0;
    nnz_t nnz_l  = 0;
    nnz_t nnz_l_local  = 0;
    nnz_t nnz_l_remote = 0;

    std::vector<cooEntry> entry; // local row indices (not global)
    std::vector<cooEntry> entry_local;
    std::vector<cooEntry> entry_remote;
    std::vector<index_t> row_local;  // needed for finding sorting
    std::vector<index_t> row_remote; // needed for finding sorting
    std::vector<index_t> col_remote; // index starting from 0, instead of the original column index

    std::vector<index_t> split;
    std::vector<index_t> splitNew;

    std::vector<index_t> vIndex;
    std::vector<value_t> vSend;
    std::vector<value_t> vecValues;

    index_t col_remote_size = 0; // number of remote columns. this is the same as vElement_remote.size()
    std::vector<index_t> nnzPerRow_local;
    std::vector<index_t> vElement_remote;
    std::vector<index_t> vElementRep_local;
    std::vector<index_t> vElementRep_remote;
    std::vector<index_t> nnzPerCol_remote;
    std::vector<nnz_t> nnzPerRowScan_local;
    std::vector<int> vdispls;
    std::vector<int> rdispls;
    std::vector<int> recvProcRank;
    std::vector<int> recvProcCount;
    std::vector<int> sendProcRank;
    std::vector<int> sendProcCount;
    index_t vIndexSize  = 0;
    index_t recvSize    = 0;
    int numRecvProc = 0;
    int numSendProc = 0;

    unsigned int num_threads = 1;
    std::vector<nnz_t> iter_local_array;
    std::vector<nnz_t> iter_remote_array;
    std::vector<value_t> w_buff; // for matvec

    std::vector<nnz_t> indicesP_local;
    std::vector<nnz_t> indicesP_remote;

//    bool arrays_defined = false; // set to true if transposeP function is called. it will be used for destructor.

    MPI_Comm comm;

    bool verbose_restrict_setup = false;
    bool verbose_transposeP = false;

    restrict_matrix();
    ~restrict_matrix();
    int transposeP(prolong_matrix* P);
    int openmp_setup();
    int matvec(std::vector<value_t>& v, std::vector<value_t>& w);
    int print_entry(int ran);
    int print_info(int ran);
};

#endif //SAENA_RESTRICT_MATRIX_H