#ifndef SAENA_STRENGTH_MATRIX_H
#define SAENA_STRENGTH_MATRIX_H

#include "aux_functions.h"

class strength_matrix {
private:

public:
    index_t M     = 0;
    index_t Mbig  = 0;
    nnz_t   nnz_l = 0;
    nnz_t   nnz_g = 0;

    MPI_Comm comm;

    bool compute_val = true; // store values of strength matrix

    nnz_t   nnz_l_local     = 0;
    nnz_t   nnz_l_remote    = 0;
    index_t col_remote_size = 0; // this is the same as vElement_remote.size()

    index_t              vIndexSize = 0;
    std::vector<index_t> vIndex;

    std::vector<index_t> split;

    std::vector<cooEntry> entry;
    std::vector<cooEntry> entryT; // transpose entries

//    std::vector<value_t> values_local;
//    std::vector<value_t> values_remote;
    std::vector<index_t> row_local;
    std::vector<index_t> row_remote;
    std::vector<index_t> col_local;
    std::vector<index_t> col_remote;  // index starting from 0, instead of the original column index
    std::vector<index_t> col_remote2; // original col index
    std::vector<value_t> val_local;
//    std::vector<nnz_t>   nnzPerRow;
    std::vector<nnz_t>   nnzPerRow_local;
    std::vector<nnz_t>   nnzPerCol_remote;
    std::vector<index_t> vElement_remote;
    std::vector<index_t> vElementRep_local;
//    std::vector<index_t> indicesP_local;

    nnz_t recvSize = 0;
    int   numRecvProc = 0;
    int   numSendProc = 0;
    std::vector<int> vdispls;
    std::vector<int> rdispls;
    std::vector<int> recvProcRank;
    std::vector<int> recvProcCount;
    std::vector<int> sendProcRank;
    std::vector<int> sendProcCount;

    int set_parameters(index_t M, index_t Mbig, std::vector<index_t> &split, MPI_Comm com);
    int setup_matrix(float connStrength);
    int setup_matrix_test(float connStrength);
    ~strength_matrix() = default;
    int erase();
    int erase_update(); // only erase the parameters needed to update the matrix
    int destroy();

    int print_info(int ran) const;
    // this function is inefficient, since first the entry vector should be created based on local and remote entries.
    void print_entry(int rank);
    // try to use the following print functions
    void print_diagonal_block(int rank) const;
    void print_off_diagonal(int rank) const;

//    int set_weight(std::vector<unsigned long>& V);
//    int randomVector(std::vector<unsigned long>& V, long size);
//    int randomVector2(std::vector<double>& V);
//    int randomVector4(std::vector<unsigned long>& V, long size);
};

#endif //SAENA_STRENGTH_MATRIX_H
