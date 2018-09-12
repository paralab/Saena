#ifndef SAENA_STRENGTH_MATRIX_H
#define SAENA_STRENGTH_MATRIX_H

#include "aux_functions.h"

#include <vector>
#include "mpi.h"


typedef unsigned int index_t;
typedef unsigned long nnz_t;
typedef double value_t;

class strength_matrix {

public:

    index_t M    = 0;
    index_t Mbig = 0;
    nnz_t nnz_l  = 0;
    nnz_t nnz_g  = 0;

    nnz_t nnz_l_local  = 0;
    nnz_t nnz_l_remote = 0;
    index_t col_remote_size = 0; // this is the same as vElement_remote.size()

    index_t vIndexSize = 0;
    std::vector<index_t> vIndex;
    std::vector<unsigned long> vSend;
    std::vector<unsigned long> vecValues;

    std::vector<index_t> split;

    std::vector<value_t> values_local;
    std::vector<value_t> values_remote;
    std::vector<index_t> row_local;
    std::vector<index_t> row_remote;
    std::vector<index_t> col_local;
    std::vector<index_t> col_remote;  // index starting from 0, instead of the original column index
    std::vector<index_t> col_remote2; // original col index
    std::vector<nnz_t> nnzPerRow;
    std::vector<nnz_t> nnzPerRow_local;
//    std::vector<unsigned int> nnzPerRow_remote;
    std::vector<nnz_t> nnzPerCol_remote;
    std::vector<index_t> vElement_remote;
    std::vector<index_t> vElementRep_local;
    std::vector<index_t> vElementRep_remote;
    std::vector<index_t> indicesP_local;
    std::vector<index_t> indicesP_remote;

    std::vector<int> vdispls;
    std::vector<int> rdispls;
    nnz_t recvSize = 0;
    int numRecvProc = 0;
    int numSendProc = 0;
    std::vector<int> recvProcRank;
    std::vector<int> recvProcCount;
    std::vector<int> sendProcRank;
    std::vector<int> sendProcCount;

    MPI_Comm comm;

    std::vector<cooEntry> entry;
    std::vector<cooEntry> entryT; // transpose entries
//    std::vector<index_t> Si, Sj;
//    std::vector<value_t> Sval, STval;

    int set_parameters(index_t M, index_t Mbig, std::vector<index_t> &split, MPI_Comm com);
    int setup_matrix(float connStrength);
    ~strength_matrix();
    int erase();
    int erase_update(); // only erase the parameters needed to update the matrix

    // this function is inefficient, since first the entry vector should be created based on local and remote entries.
    void print_entry(int rank);
    // try to use the following print functions
    void print_diagonal_block(int rank);
    void print_off_diagonal(int rank);
    int print_info(int ran);
    int save_to_disk();

};

#endif //SAENA_STRENGTH_MATRIX_H
