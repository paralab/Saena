#ifndef SAENA_SAENA_OBJECT_H
#define SAENA_SAENA_OBJECT_H

//#include "superlu_ddefs.h"
#include "superlu_defs.h"

#include "aux_functions.h"
#include "saena_vector.h"

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <bitset>
//#include <spp.h> //sparsepp

// set one of the following to set fast_mm split based on nnz or matrix size
//#define SPLIT_NNZ
#define SPLIT_SIZE

//#define FAST_MM_MAP
//#define FAST_MM_VECTOR

typedef unsigned int  index_t;
typedef unsigned long nnz_t;
typedef double        value_t;

class strength_matrix;
class saena_matrix;
class prolong_matrix;
class restrict_matrix;
class Grid;

class saena_object {
public:

    int max_level = 10; // fine grid is level 0.
    // coarsening will stop if the number of rows on one processor goes below 10.
    unsigned int least_row_threshold = 20;
    // coarsening will stop if the number of rows of last level divided by previous level is higher this value,
    // which means the number of rows was not reduced much.
    double row_reduction_up_thrshld = 0.90;
    double row_reduction_down_thrshld = 0.10;
    int vcycle_num = 500;
    double relative_tolerance = 1e-10;
    std::string smoother = "chebyshev"; // choices: "jacobi", "chebyshev"
    int preSmooth  = 3;
    int postSmooth = 3;
    std::string direct_solver = "SuperLU"; // options: 1- CG, 2- SuperLU
    std::vector<Grid> grids;
    float connStrength = 0.2; // connection strength parameter: control coarsening aggressiveness
    int CG_max_iter = 150; //150
    double CG_tol = 1e-14;
    bool repartition = false; // this parameter will be set to true if the partition of input matrix changed. it will be decided in set_repartition_rhs().
//    bool shrink_cpu = true;
    bool dynamic_levels = true;
    bool adaptive_coarsening = false;

    // SuperLU parameters
    SuperMatrix A_SLU2; // save matrix in SuperLU for solve_coarsest_SuperLU()
    gridinfo_t superlu_grid;

    std::string coarsen_method = "recursive"; // 1-basic, 2-recursive, 3-no_overlap
    const index_t matmat_size_thre1        = 50000000; // if(row * col < matmat_size_thre1) decide to do case1 or not. default 20M, last 50M
    static const index_t matmat_size_thre2 = 20000000; // if(nnz_row * nnz_col < matmat_size_thre2) do case1. default 1M
    const index_t matmat_size_thre3        = 20000000;  // if(nnz_row * nnz_col < matmat_size_thre3) do dense, otherwise map. default 1M
//    const index_t min_size_threshold = 50; //default 50
    const index_t matmat_nnz_thre = 200; //default 200
    std::bitset<matmat_size_thre2> mapbit; // todo: is it possible to clear memory for this (after setup phase)?

    // memory pool used in compute_coarsen
    value_t *mempool1;
    index_t *mempool2;
    cooEntry *mempool3;
    std::unordered_map<index_t, value_t> map_matmat;
//    spp::sparse_hash_map<index_t, value_t> map_matmat;
//    std::unique_ptr<value_t[]> mempool1; // todo: try to use these smart pointers
//    std::unique_ptr<index_t[]> mempool2;
//    std::unique_ptr<value_t[]> mempool3;

    bool doSparsify = false;
    std::string sparsifier = "majid"; // options: 1- TRSL, 2- drineas, majid
    double sparse_epsilon = 1;
    double sample_sz_percent = 1.0;
    double sample_sz_percent_final = 1.0; // = sample_prcnt_numer / sample_prcnt_denom
    nnz_t sample_prcnt_numer = 0;
    nnz_t sample_prcnt_denom = 0;

    int set_shrink_levels(std::vector<bool> sh_lev_vec);
    std::vector<bool> shrink_level_vector;
    int set_shrink_values(std::vector<int> sh_val_vec);
    std::vector<int> shrink_values_vector;

    bool switch_repartition = false;
    int set_repartition_threshold(float thre);
    float repartition_threshold = 0.1;
    bool switch_to_dense = false;
    float dense_threshold = 0.1; // 0<dense_threshold<=1 decide when to switch to the dense structure.
    // dense_threshold should be greater than repartition_threshold, since it is more efficient on repartition based on the number of rows.

    bool verbose                  = false;
    bool verbose_setup            = true;
    bool verbose_setup_steps      = false;
    bool verbose_level_setup      = false;
    bool verbose_triple_mat_mult  = false;
    bool verbose_matmat           = false;
    bool verbose_matmat_recursive = false;
    bool verbose_matmat_A         = false;
    bool verbose_matmat_B         = false;
    bool verbose_solve            = false;
    bool verbose_vcycle           = false;
    bool verbose_vcycle_residuals = false;
    bool verbose_solve_coarse     = false;

    bool verbose_triple_mat_mult_test = false;

    saena_object();
    ~saena_object();
    int destroy();

    MPI_Comm get_orig_comm();

    void set_parameters(int vcycle_num, double relative_tolerance, std::string smoother, int preSmooth, int postSmooth);
    int setup(saena_matrix* A);
    int coarsen(Grid *grid);
    int compute_coarsen(Grid *grid);
    int compute_coarsen_old(Grid *grid);
    int compute_coarsen_update_Ac(Grid *grid, std::vector<cooEntry> &diff);
    int triple_mat_mult(Grid *grid, std::vector<cooEntry_row> &RAP_row_sorted);
    int triple_mat_mult_old_RAP(Grid *grid, std::vector<cooEntry_row> &RAP_row_sorted);
    int triple_mat_mult_no_overlap(Grid *grid, std::vector<cooEntry_row> &RAP_row_sorted);
    int triple_mat_mult_basic(Grid *grid, std::vector<cooEntry_row> &RAP_row_sorted);
    int matmat(Grid *grid);
    int matmat(saena_matrix *A, saena_matrix *B, saena_matrix *C);

    // matmat_ave: transpose of B is used.
    // matmat_ave_orig_B: original B is used.
    int matmat_ave(saena_matrix *A, saena_matrix *B, double &matmat_time); // this version is only for experiments.
    int matmat_ave_orig_B(saena_matrix *A, saena_matrix *B, double &matmat_time); // this version is only for experiments.
    int reorder_split(vecEntry *arr, index_t low, index_t high, index_t pivot);
    int reorder_split(vecEntry *arr, index_t *Ac1, index_t *Ac2, index_t col_sz, index_t threshold);

    // for fast_mm experiments
    int compute_coarsen_test(Grid *grid);
    int triple_mat_mult_test(Grid *grid, std::vector<cooEntry_row> &RAP_row_sorted);

    void fast_mm(vecEntry *A, vecEntry *B, std::vector<cooEntry> &C,
                 index_t A_row_size, index_t A_row_offset, index_t A_col_size, index_t A_col_offset,
                 index_t B_col_size, index_t B_col_offset,
                 index_t *Ac, index_t *Bc, MPI_Comm comm);

//    void fast_mm(const cooEntry *A, const cooEntry *B, std::vector<cooEntry> &C,
//                 nnz_t A_nnz, nnz_t B_nnz,
//                 index_t A_row_size, index_t A_row_offset, index_t A_col_size, index_t A_col_offset,
//                 index_t B_col_size, index_t B_col_offset,
//                 const index_t *nnzPerColScan_leftStart,  const index_t *nnzPerColScan_leftEnd,
//                 const index_t *nnzPerColScan_rightStart, const index_t *nnzPerColScan_rightEnd,
//                 std::unordered_map<index_t, value_t> &map_matmat, MPI_Comm comm);

    void fast_mm_basic(const cooEntry *A, const cooEntry *B, std::vector<cooEntry> &C,
                       nnz_t A_nnz, nnz_t B_nnz,
                       index_t A_row_size, index_t A_row_offset, index_t A_col_size, index_t A_col_offset,
                       index_t B_col_size, index_t B_col_offset,
                       const index_t *nnzPerColScan_leftStart,  const index_t *nnzPerColScan_leftEnd,
                       const index_t *nnzPerColScan_rightStart, const index_t *nnzPerColScan_rightEnd,
                       MPI_Comm comm);

//    void fast_mm_basic(const cooEntry *A, const cooEntry *B, std::vector<cooEntry> &C,
//                 nnz_t A_nnz, nnz_t B_nnz, index_t A_row_size, index_t A_col_size, index_t B_col_size,
//                 const index_t *nnzPerRowScan_left, const index_t *nnzPerColScan_right, MPI_Comm comm);

    int find_aggregation(saena_matrix* A, std::vector<unsigned long>& aggregate, std::vector<index_t>& splitNew);
    int create_strength_matrix(saena_matrix* A, strength_matrix* S);
    int aggregation_1_dist(strength_matrix *S, std::vector<unsigned long> &aggregate, std::vector<unsigned long> &aggArray);
    int aggregation_2_dist(strength_matrix *S, std::vector<unsigned long> &aggregate, std::vector<unsigned long> &aggArray);
    int aggregate_index_update(strength_matrix* S, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& aggArray, std::vector<index_t>& splitNew);
    int create_prolongation(saena_matrix* A, std::vector<unsigned long>& aggregate, prolong_matrix* P);
//    int sparsify_trsl1(std::vector<cooEntry_row> & A, std::vector<cooEntry>& A_spars, double norm_frob_sq, nnz_t sample_size, MPI_Comm comm);
//    int sparsify_trsl2(std::vector<cooEntry> & A, std::vector<cooEntry>& A_spars, double norm_frob_sq, nnz_t sample_size, MPI_Comm comm);
//    int sparsify_drineas(std::vector<cooEntry_row> & A, std::vector<cooEntry>& A_spars, double norm_frob_sq, nnz_t sample_size, MPI_Comm comm);
    int sparsify_majid(std::vector<cooEntry_row> & A, std::vector<cooEntry>& A_spars, double norm_frob_sq, nnz_t sample_size, double max_val, MPI_Comm comm);
//    int sparsify_majid(std::vector<cooEntry_row> & A, std::vector<cooEntry>& A_spars, double norm_frob_sq, nnz_t sample_size, double max_val, std::vector<index_t> &splitNew, MPI_Comm comm);
    int sparsify_majid_serial(std::vector<cooEntry> & A, std::vector<cooEntry>& A_spars, double norm_frob_sq, nnz_t sample_size, double max_val, MPI_Comm comm);
    int sparsify_majid_with_dup(std::vector<cooEntry> & A, std::vector<cooEntry>& A_spars, double norm_frob_sq, nnz_t sample_size, double max_val, MPI_Comm comm);
//    double spars_prob(cooEntry a, double norm_frob_sq);
    double spars_prob(cooEntry a);

    int solve(std::vector<value_t>& u);
    int solve_pcg(std::vector<value_t>& u);
    int vcycle(Grid* grid, std::vector<value_t>& u, std::vector<value_t>& rhs);
    int smooth(Grid* grid, std::string smoother, std::vector<value_t>& u, std::vector<value_t>& rhs, int iter);
    int solve_coarsest_CG(saena_matrix* A, std::vector<value_t>& u, std::vector<value_t>& rhs);
    int solve_coarsest_SuperLU(saena_matrix* A, std::vector<value_t>& u, std::vector<value_t>& rhs);
//    int solve_coarsest_Elemental(saena_matrix* A, std::vector<value_t>& u, std::vector<value_t>& rhs);

//    int set_repartition_rhs(std::vector<value_t> rhs);
    int set_repartition_rhs(saena_vector *rhs);

    // if Saena needs to repartition the input A and rhs, then call repartition_u() at the start of the solving function.
    // then, repartition_back_u() at the end of the solve function to convert the solution to the initial partition.
    int repartition_u(std::vector<value_t>& u);
    int repartition_back_u(std::vector<value_t>& u);

    // if shrinking happens, u and rhs should be shrunk too.
    int repartition_u_shrink_prepare(Grid *grid);
    int repartition_u_shrink(std::vector<value_t> &u, Grid &grid);
    int repartition_back_u_shrink(std::vector<value_t> &u, Grid &grid);

    // if minor shrinking happens, u and rhs should be shrunk too.
//    int repartition_u_shrink_minor_prepare(Grid *grid);
//    int repartition_u_shrink_minor(std::vector<value_t> &u, Grid &grid);
//    int repartition_back_u_shrink_minor(std::vector<value_t> &u, Grid &grid);

    int shrink_cpu_A(saena_matrix* Ac, std::vector<index_t>& P_splitNew);
    int shrink_u_rhs(Grid* grid, std::vector<value_t>& u, std::vector<value_t>& rhs);
    int unshrink_u(Grid* grid, std::vector<value_t>& u);
    bool active(int l);

    int find_eig(saena_matrix& A);
//    int find_eig_Elemental(saena_matrix& A);

    int local_diff(saena_matrix &A, saena_matrix &B, std::vector<cooEntry> &C);
    int scale_vector(std::vector<value_t>& v, std::vector<value_t>& w);
    int transpose_locally(cooEntry *A, nnz_t size);
    int transpose_locally(cooEntry *A, nnz_t size, cooEntry *B);
    int transpose_locally(cooEntry *A, nnz_t size, index_t row_offset, cooEntry *B);

    // lazy update functions
    int update1(saena_matrix* A_new);
    int update2(saena_matrix* A_new);
    int update3(saena_matrix* A_new);
//    int solve_pcg_update1(std::vector<value_t>& u);
//    int solve_pcg_update2(std::vector<value_t>& u);
//    int solve_pcg_update3(std::vector<value_t>& u);

//    to write saena matrix to a file use related function from saena_matrix class.
//    int writeMatrixToFileA(saena_matrix* A, std::string name);
    int writeMatrixToFile(std::vector<cooEntry>& A, const std::string &folder_name, MPI_Comm comm);
    int writeMatrixToFileP(prolong_matrix* P, std::string name);
    int writeMatrixToFileR(restrict_matrix* R, std::string name);
    int writeVectorToFileul(std::vector<unsigned long>& v, std::string name, MPI_Comm comm);
    int writeVectorToFileul2(std::vector<unsigned long>& v, std::string name, MPI_Comm comm);
    int writeVectorToFileui(std::vector<unsigned int>& v, std::string name, MPI_Comm comm);
//    template <class T>
//    int writeVectorToFile(std::vector<T>& v, unsigned long vSize, std::string name, MPI_Comm comm);
    int change_aggregation(saena_matrix* A, std::vector<index_t>& aggregate, std::vector<index_t>& splitNew);

    // double versions
//    int vcycle_d(Grid* grid, std::vector<value_d_t>& u_d, std::vector<value_d_t>& rhs_d);
//    int scale_vector_d(std::vector<value_d_t>& v, std::vector<value_t>& w);
//    int solve_coarsest_CG_d(saena_matrix* A, std::vector<value_t>& u, std::vector<value_t>& rhs);
};

#endif //SAENA_SAENA_OBJECT_H
