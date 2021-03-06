#ifndef SAENA_PROLONG_MATRIX_H
#define SAENA_PROLONG_MATRIX_H

#include "aux_functions.h"


class prolong_matrix {
// A matrix of this class is ordered <<ONLY>> if it is defined in createProlongation function in saena_object.cpp.
// Otherwise it can be ordered using the following line:
//#include <algorithm>
//std::sort(P->entry.begin(), P->entry.end());
// It is ordered first column-wise, then row-wise, using std:sort with cooEntry class "< operator".
// Duplicates are removed in createProlongation function in saena_object.cpp.
private:

public:
    MPI_Comm comm = MPI_COMM_WORLD;

    index_t Mbig  = 0;
    index_t Nbig  = 0;
    index_t M     = 0;
    nnz_t   nnz_g = 0;
    nnz_t   nnz_l = 0;

    nnz_t nnz_l_local  = 0;
    nnz_t nnz_l_remote = 0;
    index_t col_remote_size = 0; // this is the same as vElement_remote.size()

    std::vector<index_t> split;
    std::vector<index_t> splitNew;

    std::vector<cooEntry> entry;
    std::vector<index_t> row_local;
    std::vector<index_t> col_local;
    std::vector<value_t> val_local;
    std::vector<index_t> row_remote;
    std::vector<index_t> col_remote;
    std::vector<value_t> val_remote;
//    std::vector<unsigned long> col_remote2; //original col index
//    std::vector<unsigned long> col_local;

    std::vector<index_t> nnzPerRow_local;
//    std::vector<index_t> nnzPerRowScan_local;
    std::vector<index_t> nnzPerCol_remote;
    std::vector<index_t> vElement_remote;
//    std::vector<index_t> vElement_remote_t;
//    std::vector<index_t> vElementRep_local;
//    std::vector<index_t> vElementRep_remote;
//    std::vector<unsigned int> nnz_row_remote;
    std::vector<nnz_t> nnzPerProcScan; // number of remote nonzeros on each proc. used in matvec

    int vIndexSize   = 0;
    int vIndexSize_t = 0;
    std::vector<index_t> vIndex;
    std::vector<value_t> vSend;
    std::vector<cooEntry> vSend_t;
    std::vector<value_t> vecValues;
    std::vector<cooEntry> vecValues_t;
    std::vector<float> vSend_f;           // float version
    std::vector<float> vecValues_f;       // float version

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
    std::vector<int> recvCount;

    int recvSize      = 0;
    int recvSize_t    = 0;
    int numRecvProc   = 0;
    int numRecvProc_t = 0;
    int numSendProc   = 0;
    int numSendProc_t = 0;

    unsigned int num_threads = 1;
    std::vector<nnz_t> iter_local_array;
    std::vector<nnz_t> iter_remote_array;
    std::vector<value_t> w_buff; // for matvec

//    std::vector<nnz_t> indicesP_local;
//    std::vector<nnz_t> indicesP_remote;

    vector<MPI_Request> mv_req;
    vector<MPI_Status>  mv_stat;

    bool verbose_prolong_setup = false;

    double tloc = 0, trem = 0, tcomm = 0, ttot = 0;    // for timing matvec
    index_t matvec_comm_sz = 0;                        // for profiling matvec communication size (average on all procs)

    bool use_double = true; // to determine the precision for matvec

    prolong_matrix();
    explicit prolong_matrix(MPI_Comm com);
    ~prolong_matrix();

    int findLocalRemote();
    int openmp_setup();
    inline void matvec(std::vector<value_t>& v, std::vector<value_t>& w){
        if(use_double) matvec_sparse(v, w);
        else matvec_sparse_float(v, w);
    }
    void matvec_sparse(std::vector<value_t>& v, std::vector<value_t>& w);
    void matvec_sparse_float(std::vector<value_t>& v, std::vector<value_t>& w);
    void matvec2(std::vector<value_t>& v, std::vector<value_t>& w);
    void matvec_omp(std::vector<value_t>& v, std::vector<value_t>& w);

    int print_entry(int ran);
    int print_info(int ran);

    int writeMatrixToFile(const std::string &name = "") const;
};

#endif //SAENA_PROLONG_MATRIX_H