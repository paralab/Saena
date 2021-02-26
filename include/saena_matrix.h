#ifndef SAENA_SAENA_MATRIX_H
#define SAENA_SAENA_MATRIX_H

/**
 * @author Majid Rasouli
 * @breif Contains the basic structure for the Saena matrix class (saena_matrix).
 *
 * */

#include "saena_matrix_dense.h"
#include "aux_functions.h"
#include "parUtils.h"

#ifdef SAENA_USE_ZFP
#include "zfparray1.h"
#endif

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
    std::vector<cooEntry>     data;             // entries after removing boundary nodes
    std::vector<cooEntry>     data_with_bound;  // entries including boundary nodes
    std::vector<cooEntry_row> data_unsorted;
    std::set<cooEntry_row>    data_coo;         // is set in set() functions. gets erased in setup_initial_data().
//    std::vector<cooEntry>     entry_temp;     // for updating the matrix

    nnz_t initial_nnz_l = 0;
    bool read_from_file = false;

    std::vector<int> recvCount;
    std::vector<int> sendCount;

    int MPI_flag = 0;
    vector<MPI_Request> requests;
    vector<MPI_Status>  statuses;

    std::vector<value_t> temp1;      // to be used in smoothers
    std::vector<value_t> temp2;      // to be used in smoothers

    int num_threads   = 1;
    int matvec_levels = 1;
    std::vector<nnz_t> iter_local_array;
    std::vector<nnz_t> iter_remote_array;
//    std::vector<nnz_t> iter_local_array2;
//    std::vector<nnz_t> iter_remote_array2;
    std::vector<index_t> vElement_remote;           // indices that should be received during matvec
    std::vector<value_t> w_buff; // for matvec with openmp support

    int matvec_iter_dummy = 5;

    bool verbose_saena_matrix      = false;
    bool verbose_repart            = false;
    bool verbose_matrix_setup      = false;
    bool verbose_matrix_setup_sh   = false;      // shrink
    bool verbose_repart_update     = false;
    bool verbose_matvec_dummy      = false;
    bool verbose_comp_matvec_dummy = false;
    bool verbose_shrink            = false;

public:
    MPI_Comm comm            = MPI_COMM_WORLD;
    MPI_Comm comm_old        = MPI_COMM_WORLD;
    MPI_Comm comm_horizontal = MPI_COMM_WORLD;

    index_t Mbig    = 0; // global number of rows
    index_t Nbig    = 0; // global number of columns
    index_t M       = 0; // local number of rows
    index_t M_old   = 0; // local number of rows, before being repartitioned.
    nnz_t   nnz_g   = 0; // global nnz
    nnz_t   nnz_l   = 0; // local nnz
    nnz_t   nnz_max = 0; // biggest nnz on all the processors
    index_t M_max   = 0; // biggest M on all the processors

    index_t M_orig    = 0; // before removing boundary nodes
    index_t Mbig_orig = 0; // before removing boundary nodes

    int p_order = 1;
    int prodim = 2;

    std::vector<cooEntry>  entry;

    std::vector<index_t> split;             // (row-wise) partition of the matrix between processes
    std::vector<index_t> split_b;           // (row-wise) partition of the matrix between processes, before removing boundary nodes
    std::vector<index_t> split_old;
    std::vector<nnz_t>   nnz_list;          // number of nonzeros on each process.
                                            // todo: Since it is local to each processor, int is enough. nnz_l should be changed too.

    bool remove_boundary = true;
    std::vector<index_t> bound_row; // boundary node row index
    std::vector<value_t> bound_val; // boundary node value

    nnz_t   nnz_l_local     = 0;
    nnz_t   nnz_l_remote    = 0;
    index_t col_remote_size = 0;            // number of remote columns
    std::vector<value_t> values_local;
    std::vector<value_t> values_remote;
    std::vector<index_t> row_local;
    std::vector<index_t> row_remote;
    std::vector<index_t> col_local;
    std::vector<index_t> col_remote;        // index starting from 0, instead of the original column index
    std::vector<index_t> col_remote2;       // original col index
    std::vector<index_t> nnzPerRow_local;   // todo: this is used for openmp part of saena_matrix.cpp
//    std::vector<index_t> nnzPerRow_local2;  // todo: remove this. this is used for openmp part of saena_matrix.cpp
    std::vector<index_t> nnzPerRow_remote;  // It is also used for PETSc function: MatMPIAIJSetPreallocation()
    std::vector<index_t> nnzPerCol_remote;
    std::vector<nnz_t>   nnzPerProcScan; // number of remote nonzeros on each proc. used in matvec_comp.

    std::vector<value_t> inv_diag;
    std::vector<value_t> inv_sq_diag;
    std::vector<value_t> inv_diag_orig;
    std::vector<value_t> inv_sq_diag_orig;
//    double norm1, normInf, rhoDA;

    index_t vIndexSize = 0;
    index_t recvSize   = 0;
    std::vector<index_t> vIndex;        // indices that should be sent during matvec
    std::vector<value_t> vSend;
    std::vector<value_t> vecValues;
//    std::vector<value_t> vSend2;
//    std::vector<value_t> vecValues2;    // for compressed matvec

    std::vector<nnz_t> indicesP_local;

    int numRecvProc = 0;
    int numSendProc = 0;
    std::vector<int> vdispls;
    std::vector<int> rdispls;
    std::vector<int> recvProcRank;
    std::vector<int> recvProcCount;
    std::vector<int> sendProcRank;
    std::vector<int> sendProcCount;

    bool add_duplicates = true;
    bool assembled      = false; // use this parameter to determine which matrix.set() function to use.

    bool active = false;

    // shrink_minor: if there is no entry for the coarse matrix on this proc, then shrink.
    bool active_minor = false;    // default = false

    bool enable_shrink   = true; // default = true
    bool enable_shrink_c = true;  // default = true. enables shrinking for the coarsest level.
    bool do_shrink       = false; // default = false
    bool shrinked        = false; // default = false. if shrinking happens for the matrix, set this to true.

    std::vector<double> matvec_dummy_time;
    int total_active_procs = 0;

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

    double density = -1.0;
    float jacobi_omega = float(2.0/3);
    double eig_max_of_invdiagXA = 0; // the biggest eigenvalue of (A * inv_diag(A)) to be used in chebyshev smoother
//    double highest_diag_val = 1e-10; // todo: check if this is still required.
//    double double_machine_prec = 1e-12; // it is hard-coded in aux_functions.h

    // dense matrix parameters
    // ***********************************************************
    // search for "uncomment to enable DENSE" to enable the dense part
    saena_matrix_dense dense_matrix;  // uncomment to enable DENSE
    bool use_dense = false;   // this will be set to true inside generate_dense_matrix()

    int generate_dense_matrix();

    // ***********************************************************
    // zfp parameters
    // ***********************************************************
#ifdef SAENA_USE_ZFP
    zfp_type    zfptype = zfp_type_double;

    zfp_field*  send_field;  // array meta data
    zfp_stream* send_zfp;    // compressed stream
    bitstream*  send_stream; // bit stream to write to or read from

    zfp_field*  recv_field;  // array meta data
    zfp_stream* recv_zfp;    // compressed stream
    bitstream*  recv_stream; // bit stream to write to or read from

    bool          use_zfp          = false;
    bool          free_zfp_buff    = false;
    unsigned char *zfp_send_buff   = nullptr, // storage for compressed stream to be sent
                  *zfp_send_buff2  = nullptr,
                  *zfp_recv_buff   = nullptr, // storage for compressed stream to be received
                  *zfp_recv_buff2  = nullptr; // storage for compressed stream to be received
    unsigned      zfp_send_buff_sz = 0,
                  zfp_send_comp_sz = 0,
                  zfp_recv_buff_sz = 0;

    unsigned zfp_rate      = 32;
    double   zfp_precision = 32;

//    double  *zfp_send_buff = nullptr, // storage for compressed send_stream
//            *zfp_recv_buff = nullptr;
//    unsigned char *send_buffer; // storage for compressed send_stream
//    unsigned char *recv_buffer;

    int allocate_zfp();
    int deallocate_zfp();

    int matvec_sparse_comp(std::vector<value_t>& v, std::vector<value_t>& w);
    int matvec_sparse_comp2(std::vector<value_t>& v, std::vector<value_t>& w);
    int matvec_sparse_comp3(std::vector<value_t>& v, std::vector<value_t>& w);
    int matvec_sparse_comp_omp(std::vector<value_t>& v, std::vector<value_t>& w);
#endif

    // ***********************************************************

    saena_matrix();
    explicit saena_matrix(MPI_Comm com);
    ~saena_matrix();

    int read_file(const string &filename, const std::string &input_type = "");

    void set_comm(MPI_Comm com);

    // The difference between set and set2 is that if there is a duplicate, set will erase the previous one
    // and insert the new one, but in set2, the values of those entries will be added together.
    int set(index_t row, index_t col, value_t val);
    int set(index_t* row, index_t* col, value_t* val, nnz_t nnz_local);
    int set2(index_t row, index_t col, value_t val);
    int set2(index_t* row, index_t* col, value_t* val, nnz_t nnz_local);
//    int set3(unsigned int row, unsigned int col, double val);
//    int set3(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local);

    void set_p_order(int _p_order);
    void set_prodim(int _prodim);

    int assemble(bool scale = false);
    int setup_initial_data();
    int remove_duplicates();
    int remove_boundary_nodes();
    int repartition_nnz_initial(); // based on nnz.
    int matrix_setup(bool scale = false);

    // these versions are used after matrix is assembled and needs to be updated again.
    int setup_initial_data2();
    int repartition_nnz_update(); // based on nnz.
    int matrix_setup_update(bool scale = false);

    int repart(bool repart_row = false);
//    int repartition_nnz(); // based on nnz. use this for repartitioning A's after they are created.
//    int repartition_row(); // based on M.   use this for repartitioning A's after they are created.

    int repartition_nnz_update_Ac(); // based on nnz.

    // for update3 for lazy-update
    int matrix_setup_lazy_update();
    int update_diag_lazy();

    int set_off_on_diagonal();
    int find_sortings();
    int openmp_setup();
    int scale_matrix(bool full_scale = false);
    int scale_back_matrix(bool full_scale = false);

    // dummy functions to decide if shrinking should happen
    void set_off_on_diagonal_dummy();
    void find_sortings_dummy();
    void matrix_setup_dummy();
    void matvec_dummy(std::vector<value_t>& v, std::vector<value_t>& w);
    void compute_matvec_dummy_time();

    // shrinking
    int decide_shrinking(std::vector<double> &prev_time);
    int decide_shrinking_c(); // for the coarsest level
    int shrink_set_params(std::vector<int> &send_size_array);
    int shrink_cpu();
    int shrink_cpu_minor();
    int shrink_cpu_c(); // for the coarsest level

    inline void matvec(std::vector<value_t>& v, std::vector<value_t>& w);
    void matvec_sparse(std::vector<value_t>& v, std::vector<value_t>& w);
    void matvec_sparse_array(value_t *v, value_t *w);    // to be used in ietl.
    void matvec_sparse_array2(value_t *v, value_t *w);   // to be used in ietl.

    // for profiling
    unsigned long matvec_iter = 0;
    double part1 = 0, part2 = 0, part3 = 0, part4 = 0, part5 = 0, part6 = 0;
    void matvec_time_init();
    void matvec_time_print(const int &opt = 1) const; // opt: pass 2 for the zfp version
    void matvec_time_print2() const;
    void matvec_time_print3() const;

    void matvec_sparse_test_orig(std::vector<value_t>& v, std::vector<value_t>& w);
    void matvec_sparse_test1(std::vector<value_t>& v, std::vector<value_t>& w);
    void matvec_sparse_test2(std::vector<value_t>& v, std::vector<value_t>& w);
    void matvec_sparse_test3(std::vector<value_t>& v, std::vector<value_t>& w);
    void matvec_sparse_test4(std::vector<value_t>& v, std::vector<value_t>& w);
    void matvec_sparse_test_omp(std::vector<value_t>& v, std::vector<value_t>& w); // openmp version

    // matvec timing functions for the matvec paper
//    int matvec_timing1(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);
//    int matvec_timing2(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);
//    int matvec_timing3(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);
//    int matvec_timing4(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);
//    int matvec_timing4_alltoall(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);
//    int matvec_timing5(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);
//    int matvec_timing5_alltoall(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time);

    // Vector res = A * u - rhs;
    inline void residual(std::vector<value_t>& u, std::vector<value_t>& rhs, std::vector<value_t>& res);
    // Vector res = rhs - A * u
    inline void residual_negative(std::vector<value_t>& u, std::vector<value_t>& rhs, std::vector<value_t>& res);

    int inverse_diag();

    // smoothers
    void jacobi(int iter, std::vector<value_t>& u, std::vector<value_t>& rhs);
    void chebyshev(const int &iter, std::vector<value_t>& u, std::vector<value_t>& rhs);

    // I/O functions
    int print_entry(int ran, const std::string &name = "") const;
    int print_info(int ran, const std::string &name = "") const;
    int writeMatrixToFile(const std::string &name = "") const;

    // erase and destroy
    int set_zero();
    int erase();
    int erase2();
    int erase_update_local(); // use this for compute_coarsen_update_Ac()
    int erase_keep_remote2(); // use this for compute_coarsen_update_Ac()
    int erase_after_shrink();
    int erase_after_decide_shrinking();
    int erase_lazy_update();
    int erase_no_shrink_to_fit();
    int destroy();
};

#endif //SAENA_SAENA_MATRIX_H

#include <saena_matrix.tpp>