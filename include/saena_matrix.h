#ifndef SAENA_SAENA_MATRIX_H
#define SAENA_SAENA_MATRIX_H

#include <iostream>
#include <vector>
#include <set>
#include <mpi.h>
#include "aux_functions.h"

/**
 * @author Majid
 * @breif Contains the basic structure to define coo matrices
 *
 * */

//template <class T>
//T lower_bound2(T *left, T *right, T val);

//class cooEntry;

class saena_matrix {
// A matrix of this class is ordered first column-wise, then row-wise.

private:
    bool read_from_file = false;
    unsigned int initial_nnz_l;
    bool freeBoolean = false; // use this parameter to know if destructor for SaenaMatrix class should free the variables or not.
    std::set<cooEntry> data_coo;
    std::vector<cooEntry> data_unsorted;
    std::vector<unsigned long> data;

public:
    std::vector<cooEntry> entry;

    unsigned int M     = 0; // local number of rows
    unsigned int Mbig  = 0; // global number of rows
    unsigned int nnz_g = 0; // global nnz
    unsigned int nnz_l = 0; // local nnz
    std::vector<unsigned long> split; // (row-wise) partition of the matrix between processes
    std::vector<unsigned long> split_old;

    unsigned int nnz_l_local;
    unsigned int nnz_l_remote;
    unsigned long col_remote_size; // number of remote columns
    std::vector<double> values_local;
    std::vector<double> values_remote;
    std::vector<unsigned long> row_local;
    std::vector<unsigned long> row_remote;
    std::vector<unsigned long> col_local;
    std::vector<unsigned long> col_remote; // index starting from 0, instead of the original column index
    std::vector<unsigned long> col_remote2; //original col index
    std::vector<unsigned int> nnzPerRow_local; // todo: this is used for openmp part of saena_matrix.cpp
    std::vector<unsigned int> nnzPerCol_remote; // todo: replace this. nnz Per Column is expensive.
//    std::vector<unsigned int> nnzPerRow;
//    std::vector<unsigned int> nnzPerRow_remote;
//    std::vector<unsigned int> nnzPerRowScan_local;
//    std::vector<unsigned int> nnzPerRowScan_remote;
//    std::vector<unsigned int> nnzPerCol_local;
//    std::vector<unsigned int> nnzPerColScan_local;
//    std::vector<unsigned int> nnz_row_remote;

    std::vector<double> invDiag;
//    double norm1, normInf, rhoDA;

    int vIndexSize;
    long *vIndex;
    double *vSend;
    unsigned long *vSendULong;
    double* vecValues;
    unsigned long* vecValuesULong;

//    unsigned long* indicesP;
    unsigned long* indicesP_local;
    unsigned long* indicesP_remote;

    std::vector<int> vdispls;
    std::vector<int> rdispls;
    int recvSize;
    int numRecvProc;
    int numSendProc;
    std::vector<int> recvProcRank;
    std::vector<int> recvProcCount;
    std::vector<int> sendProcRank;
    std::vector<int> sendProcCount;

    unsigned long num_threads;
    unsigned int* iter_local_array;
    unsigned int* iter_remote_array;
    std::vector<unsigned long> vElement_remote;
    std::vector<unsigned long> vElementRep_local;
    std::vector<unsigned long> vElementRep_remote;

    MPI_Comm comm;
    MPI_Comm comm_horizontal;
    MPI_Comm comm_old;
    bool shrinked = false;

    bool add_duplicates = false;
    bool assembled = false; // use this parameter to determine which matrix.set() function to use.

    bool active = true;
    bool active_old_comm = false; // this is used for prolong and post-smooth
    int cpu_shrink_thre1 = 60; // Ac->last_M_shrink >= (Ac->Mbig * A->cpu_shrink_thre1)
    int cpu_shrink_thre2 = 15;
    unsigned int last_M_shrink;

    float jacobi_omega = float(2.0/3);
    double eig_max_diagxA;

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
    int set(unsigned int row, unsigned int col, double val);
    int set(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local);
    int set2(unsigned int row, unsigned int col, double val);
    int set2(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local);
//    int set3(unsigned int row, unsigned int col, double val);
//    int set3(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local);

    int setup_initial_data();
    int repartition();
    int matrix_setup();

    // these versions are used after matrix is assembled and needs to be updated again.
    int setup_initial_data2();
    int repartition2();
    int matrix_setup2();

    int matvec(const std::vector<double>& v, std::vector<double>& w);
    int matvec_timing(const std::vector<double>& v, std::vector<double>& w, std::vector<double>& time);
    int matvec_timing2(const std::vector<double>& v, std::vector<double>& w, std::vector<double>& time);

    int residual(std::vector<double>& u, std::vector<double>& rhs, std::vector<double>& res);
    int inverse_diag(std::vector<double>& x);
    int jacobi(int iter, std::vector<double>& u, std::vector<double>& rhs, std::vector<double>& temp);
    int chebyshev(int iter, std::vector<double>& u, std::vector<double>& rhs, std::vector<double>& temp, std::vector<double>& temp2);
    int find_eig();

    int set_zero();
    int erase();
    int erase_keep_remote(); // use this for coarsen2()
    int destroy();
};

#endif //SAENA_SAENA_MATRIX_H