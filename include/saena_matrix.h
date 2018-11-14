#ifndef SAENA_SAENA_MATRIX_H
#define SAENA_SAENA_MATRIX_H

#include "saena_matrix_dense.h"
#include "aux_functions.h"
#include "zfparray1.h"

#include <iostream>
#include <vector>
#include <set>
#include <mpi.h>


/**
 * @author Majid
 * @breif Contains the basic structure for the Saena matrix class (saena_matrix).
 *
 * */

typedef unsigned int index_t;
typedef unsigned long nnz_t;
typedef double value_t;

class saena_matrix {

// A matrix of this class has column-major order.

    // Steps of creating a matrix of this class:
//    parameter	        type	                    reason
//    -----------------------------------------------------------------------------------
//    data_coo	        std::set<cooEntry_row>		add entries by set()
//    data_unsorted	    std::vector<cooEntry_row>	switch from std::set to std::vector
//    data_sorted_row   std::vector<cooEntry_row>	sort row-major
//    data_sorted	    std::vector<cooEntry>		switch from cooEntry_row to cooEntry
//    data		        std::vector<cooEntry>		remove duplicates

private:
    std::vector<cooEntry_row> data_unsorted;
    std::vector<cooEntry> data;

    nnz_t initial_nnz_l = 0;
    bool read_from_file = false;
    bool freeBoolean = false; // use this parameter to know if destructor for saena_matrix class should free the variables or not.

    bool verbose_saena_matrix = false;
    bool verbose_repartition  = false;
    bool verbose_matrix_setup = false;
    bool verbose_repartition_update = false;

public:
    std::set<cooEntry_row> data_coo;
    std::vector<cooEntry> entry;
    std::vector<cooEntry> entry_temp; // is used for updating the matrix

    index_t Mbig  = 0; // global number of rows
    index_t M     = 0; // local number of rows
    index_t M_old = 0; // local number of rows, before being repartitioned.
    nnz_t nnz_g   = 0; // global nnz
    nnz_t nnz_l   = 0; // local nnz
    std::vector<index_t> split; // (row-wise) partition of the matrix between processes
    std::vector<index_t> split_old;

    nnz_t nnz_l_local  = 0;
    nnz_t nnz_l_remote = 0;
    index_t col_remote_size = 0; // number of remote columns
    std::vector<value_t> values_local;
    std::vector<value_t> values_remote;
    std::vector<index_t> row_local;
    std::vector<index_t> row_remote;
    std::vector<index_t> col_local;
    std::vector<index_t> col_remote; // index starting from 0, instead of the original column index
    std::vector<index_t> col_remote2; //original col index
    std::vector<index_t> nnzPerRow_local;  // todo: this is used for openmp part of saena_matrix.cpp
    std::vector<index_t> nnzPerRow_local2; // todo: remove this. this is used for openmp part of saena_matrix.cpp
    std::vector<index_t> nnzPerRow_remote; // It is also used for PETSc function: MatMPIAIJSetPreallocation()
    std::vector<index_t> nnzPerCol_remote;

//    std::vector<index_t> row_local_temp;
//    std::vector<index_t> col_local_temp;
//    std::vector<value_t> values_local_temp;

    std::vector<value_t> inv_diag;
    std::vector<value_t> inv_diag_original;
    std::vector<value_t> inv_sq_diag;
//    double norm1, normInf, rhoDA;

    index_t vIndexSize = 0;
    index_t recvSize   = 0;
    std::vector<index_t> vIndex;
    std::vector<value_t> vSend;
    std::vector<value_t> vecValues;
    std::vector<unsigned long> vSendULong;
    std::vector<unsigned long> vecValuesULong;
//    zfp::array1<double> vSend_zfp;
//    zfp::array1<double> vecValues_zfp;

    std::vector<nnz_t> indicesP_local;
    std::vector<nnz_t> indicesP_remote;

    int numRecvProc = 0;
    int numSendProc = 0;
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

    unsigned int num_threads = 1;
    std::vector<nnz_t> iter_local_array;
    std::vector<nnz_t> iter_remote_array;
    std::vector<nnz_t> iter_local_array2;
    std::vector<nnz_t> iter_remote_array2;
    std::vector<index_t> vElement_remote;
//    std::vector<index_t> vElementRep_local;
    std::vector<index_t> vElementRep_remote;
    std::vector<value_t> w_buff; // for matvec3()

    bool add_duplicates = true;
    bool assembled = false; // use this parameter to determine which matrix.set() function to use.

    MPI_Comm comm;
    MPI_Comm comm_horizontal;
    MPI_Comm comm_old;

    bool active = true;
    bool active_old_comm = false; // this is used for prolong and post-smooth

    bool enable_shrink = false;  // default = true
    bool do_shrink     = false; // default = false
    bool shrinked      = false; // default = false. if shrinking happens for the matrix, set this to true.
    bool enable_dummy_matvec = true; // default = true
    std::vector<double> matvec_dummy_time;
    unsigned int total_active_procs = 0;
    int cpu_shrink_thre1 = 1; // default = 1. set 0 to shrink at every level. density >= (last_density_shrink * cpu_shrink_thre1)
    int cpu_shrink_thre2 = 1; // default = 1. number of procs after shrinking = nprocs / cpu_shrink_thre2
    int cpu_shrink_thre2_next_level = -1;
    float shrink_total_thre     = 1.25;
    float shrink_local_thre     = 1.25;
    float shrink_communic_thre  = 1.5;
    index_t last_M_shrink       = 0;
    nnz_t   last_nnz_shrink     = 0;
    double  last_density_shrink = 0;
    // use these two parameters to decide shrinking for the level of multigrid
    bool enable_shrink_next_level = false; // default is false. set it to true in the setup() function if it is required.
//    int cpu_shrink_thre1_next = 0; // set 0 to shrink at every level. density >= (last_density_shrink * cpu_shrink_thre1)

    // shrink_minor: if there no entry for the coarse matrix on this proc, then shrink.
    bool active_minor = true;    // default = true
//    index_t M_old_minor = 0; // local number of rows, before being repartitioned.
    std::vector<index_t> split_old_minor;
    bool shrinked_minor = false; // default = false
//    MPI_Comm comm_old_minor;

    double density = -1.0;
    float jacobi_omega = float(2.0/3);
    double eig_max_of_invdiagXA = 0; // the biggest eigenvalue of (A * inv_diag(A)) to be used in chebyshev smoother
    double highest_diag_val = 1e-10; // todo: check if this is still required.
//    double double_machine_prec = 1e-12; // it is hard-coded in aux_functions.h

    saena_matrix_dense dense_matrix;
    bool switch_to_dense = false;
    bool dense_matrix_generated = false;
    float dense_threshold = 0.1; // 0<dense_threshold<=1 decide when to also generate dense_matrix for this matrix.

    // zfp parameters
    zfp_field* field;  /* array meta data */
    zfp_stream* zfp;   /* compressed stream */
    bitstream* stream; /* bit stream to write to or read from */
//    unsigned char *send_buffer; /* storage for compressed stream */
//    unsigned char *recv_buffer;
    bool free_zfp_buff = false;
    double *zfp_send_buffer; /* storage for compressed stream */
    double *zfp_recv_buffer;
    unsigned rate = 64;
    unsigned zfp_send_bufsize = 0, zfp_recv_bufsize = 0;
    zfp_field* field2;  /* array meta data */
    zfp_stream* zfp2;   /* compressed stream */
    bitstream* stream2; /* bit stream to write to or read from */

    saena_matrix();
    saena_matrix(MPI_Comm com);
    int read_file(const char* Aname);
    int read_file(const char* Aname, const std::string &input_type);
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

    int assemble();
    int setup_initial_data();
    int remove_duplicates();
    int repartition_nnz_initial(); // based on nnz.
    int matrix_setup();
    int matrix_setup_no_scale();

    // these versions are used after matrix is assembled and needs to be updated again.
    int setup_initial_data2();
    int repartition_nnz_update(); // based on nnz.
    int matrix_setup_update();
    int matrix_setup_update_no_scale();

    int repartition_nnz(); // based on nnz. use this for repartitioning A's after they are created.
    int repartition_row(); // based on M.   use this for repartitioning A's after they are created.

    int repartition_nnz_update_Ac(); // based on nnz.

//    int set_rho();
    int set_off_on_diagonal();
    int find_sortings();
    int openmp_setup();
    int scale_matrix();
    int scale_back_matrix();

    int set_off_on_diagonal_dummy();
//    int find_sortings_dummy();
    int matrix_setup_dummy();
    int matvec_dummy(std::vector<value_t>& v, std::vector<value_t>& w);
    int compute_matvec_dummy_time();
    int decide_shrinking(std::vector<double> &prev_time);
    int shrink_cpu();
    int shrink_cpu_minor();

    int matvec(std::vector<value_t>& v, std::vector<value_t>& w);
    int matvec_sparse(std::vector<value_t>& v, std::vector<value_t>& w);
    int matvec_sparse_zfp(std::vector<value_t>& v, std::vector<value_t>& w);
    int matvec_timing1(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);
    int matvec_timing2(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);
    int matvec_timing3(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);
    int matvec_timing4(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);
    int matvec_timing4_alltoall(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);
    int matvec_timing5(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);
    int matvec_timing5_alltoall(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);

    int residual(std::vector<value_t>& u, std::vector<value_t>& rhs, std::vector<value_t>& res);
    int inverse_diag();
    int jacobi(int iter, std::vector<value_t>& u, std::vector<value_t>& rhs, std::vector<value_t>& temp);
    int chebyshev(int iter, std::vector<value_t>& u, std::vector<value_t>& rhs, std::vector<value_t>& temp, std::vector<value_t>& temp2);

    int generate_dense_matrix();

    int print_entry(int ran);
    int print_info(int ran);
    int writeMatrixToFile();
    int writeMatrixToFile(const char *folder_name);

    int set_zero();
    int erase();
    int erase2();
    int erase_update_local(); // use this for coarsen_update_Ac()
    int erase_keep_remote2(); // use this for coarsen_update_Ac()
    int erase_after_shrink();
    int erase_after_decide_shrinking();
    int erase_lazy_update();
    int destroy();
};

#endif //SAENA_SAENA_MATRIX_H

