#ifndef SAENA_SAENA_MATRIX_H
#define SAENA_SAENA_MATRIX_H

#include <iostream>
#include <vector>
#include <set>
#include <mpi.h>
#include "aux_functions.h"

/**
 * @author Majid
 * @breif Contains the basic structure for the Saena matrix calss (saena_matrix).
 *
 * */

typedef unsigned int index_t;
typedef unsigned long nnz_t;
typedef double value_t;

class saena_matrix {
// A matrix of this class is ordered first column-wise, then row-wise.

private:
    std::set<cooEntry> data_coo;
    std::vector<cooEntry> data_unsorted;
    std::vector<cooEntry> data;

    nnz_t initial_nnz_l;
    bool read_from_file = false;
    bool freeBoolean = false; // use this parameter to know if destructor for SaenaMatrix class should free the variables or not.

    bool verbose_saena_matrix = false;
    bool repartition_verbose  = false;
    bool verbose_matrix_setup = false;

public:
    std::vector<cooEntry> entry;

    index_t Mbig = 0; // global number of rows
    index_t M    = 0; // local number of rows
    nnz_t nnz_g  = 0; // global nnz
    nnz_t nnz_l  = 0; // local nnz
    std::vector<index_t> split; // (row-wise) partition of the matrix between processes
    std::vector<index_t> split_old;

    nnz_t nnz_l_local;
    nnz_t nnz_l_remote;
    index_t col_remote_size; // number of remote columns
    std::vector<value_t> values_local;
    std::vector<value_t> values_remote;
    std::vector<index_t> row_local;
    std::vector<index_t> row_remote;
    std::vector<index_t> col_local;
    std::vector<index_t> col_remote; // index starting from 0, instead of the original column index
    std::vector<index_t> col_remote2; //original col index
    std::vector<nnz_t> nnzPerRow_local; // todo: this is used for openmp part of saena_matrix.cpp
    std::vector<nnz_t> nnzPerRow_local2; // todo: this is used for openmp part of saena_matrix.cpp
    std::vector<nnz_t> nnzPerCol_remote;

    std::vector<value_t> invDiag;
//    double norm1, normInf, rhoDA;

    index_t vIndexSize;
    std::vector<index_t> vIndex;
    std::vector<value_t> vSend;
    std::vector<value_t> vecValues;
    std::vector<unsigned long> vSendULong;
    std::vector<unsigned long> vecValuesULong;

    std::vector<nnz_t> indicesP_local;
    std::vector<nnz_t> indicesP_remote;

    int recvSize;
    int numRecvProc;
    int numSendProc;
    std::vector<int> vdispls;
    std::vector<int> rdispls;
    std::vector<int> recvCount;
    std::vector<int> recvCountScan;
    std::vector<int> sendCount;
    std::vector<int> sendCountScan;
    std::vector<int> recvProcRank;
    std::vector<int> recvProcCount;
    std::vector<int> sendProcRank;
    std::vector<int> sendProcCount;

    unsigned int num_threads;
    std::vector<nnz_t> iter_local_array;
    std::vector<nnz_t> iter_remote_array;
    std::vector<nnz_t> iter_local_array2;
    std::vector<nnz_t> iter_remote_array2;
    std::vector<index_t> vElement_remote;
    std::vector<index_t> vElementRep_local;
    std::vector<index_t> vElementRep_remote;
    value_t *w_buff; // for matvec3()

    bool add_duplicates = false;
    bool assembled = false; // use this parameter to determine which matrix.set() function to use.

    MPI_Comm comm;
    MPI_Comm comm_horizontal;
    MPI_Comm comm_old;

    bool active = true;
    bool active_old_comm = false; // this is used for prolong and post-smooth

    bool enable_shrink = false;
    bool shrinked = false;
    int cpu_shrink_thre1 = 2; // Ac->last_M_shrink >= (Ac->Mbig * A->cpu_shrink_thre1)
    int cpu_shrink_thre2 = 2;
    index_t last_M_shrink;

    float jacobi_omega = float(2.0/3);
    double eig_max_of_invdiagXA = 0; // the biggest eigenvalue of (A * invDiag(A)) to be used in chebyshev smoother
    float density = -1;
//    double double_machine_prec = 1e-12; // it is hard-coded in aux_functions.h

    saena_matrix();
    saena_matrix(MPI_Comm com);
    /**
     * @param[in] Aname is the pointer to the matrix
     * @param[in] Mbig Number of rows in the matrix
     * */
    saena_matrix(char* Aname, MPI_Comm com);
    ~saena_matrix();

    void set_comm(MPI_Comm com);

    // The difference between set and set2 is that if there is a repetition, set will erase the previous one
    // and insert the new one, but in set2, the values of those entries will be added together.
    int set(index_t row, index_t col, value_t val);
    int set(index_t* row, index_t* col, value_t* val, nnz_t nnz_local);
    int set2(index_t row, index_t col, value_t val);
    int set2(index_t* row, index_t* col, value_t* val, nnz_t nnz_local);
//    int set3(unsigned int row, unsigned int col, double val);
//    int set3(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local);

    int setup_initial_data();
    int repartition(); // based on nnz.
    int matrix_setup();

    // these versions are used after matrix is assembled and needs to be updated again.
    int setup_initial_data2();
    int repartition2(); // based on nnz.
    int matrix_setup2();

    int repartition3(); // based on nnz. use this for repartitioning A's after they are created.
    int repartition4(); // based on M. use this for repartitioning A's after they are created.
    int shrink_cpu();

    int matvec(std::vector<value_t>& v, std::vector<value_t>& w);
    int matvec_timing1(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);
    int matvec_timing2(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);
    int matvec_timing3(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);
    int matvec_timing3_alltoall(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);
    int matvec_timing4(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);
    int matvec_timing5(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);
    int matvec_timing5_alltoall(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);

    int residual(std::vector<value_t>& u, std::vector<value_t>& rhs, std::vector<value_t>& res);
    int inverse_diag(std::vector<value_t>& x);
    int jacobi(int iter, std::vector<value_t>& u, std::vector<value_t>& rhs, std::vector<value_t>& temp);
    int chebyshev(int iter, std::vector<value_t>& u, std::vector<value_t>& rhs, std::vector<value_t>& temp, std::vector<value_t>& temp2);

    int set_zero();
    int erase();
    int erase_keep_remote(); // use this for coarsen2()
    int destroy();
};

#endif //SAENA_SAENA_MATRIX_H