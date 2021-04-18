#ifndef SAENA_RESTRICT_MATRIX_H
#define SAENA_RESTRICT_MATRIX_H

#include "aux_functions.h"

class prolong_matrix;


class restrict_matrix {
// A matrix of this class is ordered first column-wise, then row-wise, using std:sort with cooEntry class "< operator".
// It is sorted in restrictMatrix constructor function in restrictmatrix.cpp.
private:

public:
    MPI_Comm comm = MPI_COMM_WORLD;

    index_t M            = 0;
    index_t Mbig         = 0;
    index_t Nbig         = 0;
    nnz_t   nnz_g        = 0;
    nnz_t   nnz_l        = 0;
    nnz_t   nnz_l_local  = 0;
    nnz_t   nnz_l_remote = 0;
    nnz_t   nnz_max      = 0;
    index_t M_max        = 0;

    std::vector<cooEntry> entry; // local row indices (not global)
//    std::vector<cooEntry> entry_local;
//    std::vector<cooEntry> entry_remote;
//    std::vector<index_t>  row_local;
    std::vector<index_t>  col_local;
    std::vector<value_t>  val_local;
    std::vector<index_t>  row_remote;
    std::vector<value_t>  val_remote;
//    std::vector<index_t>  col_remote; // index starting from 0, instead of the original column index

    std::vector<index_t> split;
    std::vector<index_t> splitNew;
    std::vector<nnz_t>   nnz_list; // number of nonzeros on each process. To be used for mat-mat product.

    std::vector<index_t> vIndex;
    std::vector<value_t> vSend;
    std::vector<value_t> vecValues;
    std::vector<float> vSend_f;           // float version
    std::vector<float> vecValues_f;       // float version

    index_t col_remote_size = 0; // number of remote columns. this is the same as vElement_remote.size()
    std::vector<index_t> nnzPerRow_local;
    std::vector<index_t> vElement_remote;
//    std::vector<index_t> vElementRep_local;
//    std::vector<index_t> vElementRep_remote;
    std::vector<index_t> nnzPerCol_remote;
//    std::vector<nnz_t> nnzPerRowScan_local;
    std::vector<nnz_t>   nnzPerProcScan; // number of remote nonzeros on each proc. used in matvec

    std::vector<int> vdispls;
    std::vector<int> rdispls;
    std::vector<int> recvProcRank;
    std::vector<int> recvProcCount;
    std::vector<int> sendProcRank;
    std::vector<int> sendProcCount;
    std::vector<int> recvCount;
    index_t vIndexSize  = 0;
    index_t recvSize    = 0;
    int numRecvProc = 0;
    int numSendProc = 0;

    int num_threads   = 1;
    int matvec_levels = 1;

    std::vector<nnz_t> iter_local_array;
    std::vector<nnz_t> iter_remote_array;
    std::vector<value_t> w_buff; // for matvec

//    std::vector<nnz_t> indicesP_local;
//    std::vector<nnz_t> indicesP_remote;

    vector<MPI_Request> mv_req;
    vector<MPI_Status>  mv_stat;

//    bool arrays_defined = false; // set to true if transposeP function is called. it will be used for destructor.

    bool verbose_restrict_setup = false;
    bool verbose_transposeP     = false;

    double tloc = 0, trem = 0, tcomm = 0, ttot = 0;    // for timing matvec
    index_t matvec_comm_sz = 0;                        // for profiling matvec communication size (average on all procs)

    bool use_double = true; // to determine the precision for matvec

    restrict_matrix();
    ~restrict_matrix();

    int transposeP(prolong_matrix* P);
    int openmp_setup();
    inline void matvec(const value_t *v, value_t *w){
        if(use_double) matvec_sparse(v, w);
        else matvec_sparse_float(v, w);
    }
    void matvec_sparse(const value_t *v, value_t *w);
    void matvec_sparse_float(const value_t *v, value_t *w);
    void matvec2(std::vector<value_t>& v, std::vector<value_t>& w);
    void matvec_omp(std::vector<value_t>& v, std::vector<value_t>& w);

    int print_entry(int ran) const;
    int print_info(int ran) const;

    int writeMatrixToFile(const std::string &name = "") const;
};

#endif //SAENA_RESTRICT_MATRIX_H