#ifndef SAENA_SAENA_OBJECT_H
#define SAENA_SAENA_OBJECT_H

#include "data_struct.h"
#include "superlu_ddefs.h"
#include "saena_vector.h"
#include "saena_matrix_dense.h"
#include "grid.h"

#include <memory>
#include <unordered_map>
#include <bitset>

// set one of the following to set fast_mm split based on nnz or matrix size
//#define SPLIT_NNZ
#define SPLIT_SIZE

// number of update steps for lazy-update
#define ITER_LAZY 20

// uncomment to enable timing
// to have a more accurate timing for PROFILE_TOTAL_PCG, comment out PROFILE_PCG and PROFILE_VCYCLE, since they
// have barriers and are inside PROFILE_TOTAL_PCG.
//#define PROFILE_PCG
//#define PROFILE_TOTAL_PCG
//#define PROFILE_VCYCLE

class strength_matrix;
class saena_matrix;
class prolong_matrix;
class restrict_matrix;
class Grid;


class saena_object {
public:

    // setup
    // **********************************************

    int          max_level                  = 20; // fine grid is level 0.
    // if dynamic_levels == true: coarsening will stop if the total number of rows goes below this parameter.
    unsigned int least_row_threshold        = 100;
    // if dynamic_levels == true: coarsening will stop if the number of rows of last level divided by previous level is
    // higher than this parameter, which means the number of rows was not reduced much through coarsening.
    double       row_reduction_up_thrshld   = 0.90;
//    double       row_reduction_down_thrshld = 0.10;

    bool repartition         = false; // this parameter will be set to true if the partition of input matrix changed. it will be decided in set_repartition_rhs().
    bool dynamic_levels      = true;
    bool adaptive_coarsening = false;

    bool scale = false;

    saena_matrix *A_coarsest = nullptr;

    // enable deciding the number of levels in the multigrid hierarchy automatically.
    // if new_size <= least_row_threshold, then stop coarsening.
    // Also, if new_size / prev_size > row_reduction_up_thrshld, the number of rows was not reduced much through coarsening.
    void set_dynamic_levels(const bool &dl = true);

    bool remove_boundary = false;
    std::vector<index_t> bound_row; // boundary node row index
    std::vector<value_t> bound_val; // boundary node value
    std::vector<value_t> bound_sol; // solution corresponding to boundary nodes
    void remove_boundary_rhs(std::vector<value_t> &rhs_large, std::vector<value_t> &rhs0, MPI_Comm comm);
    void add_boundary_sol(std::vector<value_t> &u);

    int float_level = 3; // any matrix after this level will use single-precision matvec

    // *****************
    // matmat
    // *****************

    // stop condition for matmat split:
    // A.row_sz * B.col_sz < matmat_thre1   &&   B.col_sz < matmat_thre2
    // matmat_thre1 is upper bound for nonzeros of C = A * B.
    // matmat_thre2 is upper bound for columns of C.
    // matmat_thre3 is for when we want to have an exact number of splits. (case2_iter + case3_iter == matmat_thre3)

    std::string          coarsen_method  = "recursive"; // 1-basic, 2-recursive, 3-no_overlap
    const int            matmat_thre1    = 20000000;
    index_t              matmat_thre2    = 0;
    static const index_t matmat_thre3    = 40;
    const index_t        matmat_nnz_thre = 200; //default 200

//    std::bitset<matmat_size_thre2> mapbit; // todo: is it possible to clear memory for this (after setup phase)?

    long   zfp_orig_sz = 0l;   // the original size of array that was compressed with zfp
    long   zfp_comp_sz = 0l;   // the size of the compressed array as the output of zfp
    double zfp_rate    = 0.0;  // = zfp_comp_sz / zfp_orig_sz
    double zfp_thrshld = 0.0;  // if (zfp_rate < zfp_thrshld) zfp_perform = true;
                               // set zfp_thrshld = 0 to disable zfp compression
    bool   zfp_perform = true;

    // memory pool used in compute_coarsen
//    value_t *mempool1   = nullptr;
//    index_t *mempool2   = nullptr;
    index_t *mempool3 = nullptr;
    index_t *mempool4 = nullptr;
    value_t *mempool5 = nullptr;
    uchar   *mempool6 = nullptr;
    uchar   *mempool7 = nullptr;

    nnz_t    mempool3_sz     = 0;
    nnz_t    mempool4and5_sz = 0;
    nnz_t    mempool6_sz     = 0;
    nnz_t    mempool7_sz     = 0;

    index_t *Cmkl_r      = nullptr;
    index_t *Cmkl_c_scan = nullptr;
    value_t *Cmkl_v      = nullptr;
    bool    use_dcsrmultcsr = true;

    std::vector<cooEntry> C_temp;

    index_t case1_iter = 0,       case2_iter = 0,       case3_iter = 0;
    double  case1_iter_ave = 0.0, case2_iter_ave = 0.0, case3_iter_ave = 0.0;

    // *****************
    // shrink
    // *****************

    int set_shrink_levels(std::vector<bool> sh_lev_vec);
    int set_shrink_values(std::vector<int>  sh_val_vec);
    std::vector<bool> shrink_level_vector;
    std::vector<int>  shrink_values_vector;

    // *****************
    // repartition
    // *****************

    bool  switch_repart = false;
    float repart_thre   = 0.1;
    int   set_repart_thre(float thre);

    // *****************
    // dense
    // *****************

    // dense_thre should be greater than repart_thre, since it is more efficient on repartition based on the number of rows.
    bool    switch_to_dense = true;
    float   dense_thre      = 0.1; // (0 < dense_thre <= 1) decide when to switch to the dense data structure.
    index_t dense_sz_thre   = 5000;

    // *****************
    // solve parameters
    // **********************************************

    int    solver_max_iter      = 500; // 500
    double solver_tol           = 1e-12;
    int    CG_coarsest_max_iter = 150; // 150
    double CG_coarsest_tol      = 1e-12;

    // ****************
    // AMG parameters
    // ****************

    int         preSmooth     = 2;
    int         postSmooth    = 2;
    std::string smoother      = "chebyshev";    // choices: "jacobi", "chebyshev"
    std::string direct_solver = "SuperLU";      // choices: "CG", "SuperLU"
    float       connStrength  = 0.3;            // connection strength parameter: control coarsening aggressiveness
    std::string PSmoother     = "jacobi";       // "jacobi", "SPAI"
    double      Pomega        = 2.0 / 3;        // For jacobi it is usually 2/3 for 2D and 6/7 for 3D.

    // ****************
    // SuperLU
    // ****************

    SuperMatrix            A_SLU2;         // save matrix in SuperLU format to be used in solve_coarsest_SuperLU()
    gridinfo_t             superlu_grid;
    index_t                fst_row;
    ScalePermstruct_t      ScalePermstruct;
    LUstruct_t             LUstruct;
    SOLVEstruct_t          SOLVEstruct;
    superlu_dist_options_t options;

    bool first_solve       = true;
    bool superlu_active    = true;
    bool superlu_allocated = false;
    bool lu_created        = false;

    // **********************************************

    std::vector<Grid> grids;

    // **********************************************
    // sparsification
    // **********************************************

    bool   doSparsify               = false;
    double sparse_epsilon           = 1;
    double sample_sz_percent        = 1.0;
    double sample_sz_percent_final  = 1.0; // = sample_prcnt_numer / sample_prcnt_denom
    nnz_t  sample_prcnt_numer       = 0;
    nnz_t  sample_prcnt_denom       = 0;
    std::string sparsifier          = "majid"; // options: 1- TRSL, 2- drineas, majid

    double filter_thre  = 1e-12;
    double filter_max   = 1e-8;
    int    filter_it    = 0; // to count the levels for filtering
    int    filter_start = 1; // filtering starts at this level. cannot filter level 0. so it should be >= 1
    int    filter_rate  = 2; // filter_thre will get larger with rate 10^filter_rate.
                             // So levels 1, 2, ... will be filtered by filter_thre, filter_thre * 10^filter_rate, ...
                             // and the threshold does not get largre than filter_max.
                             // e.g. 1e-14, 1e-12, 1e-10, ... , filter_max, filter_max

    // **********************************************
    // verbose
    // **********************************************
    bool verbose                  = false;
    bool verbose_setup            = true;
    bool verbose_setup_steps      = false;
    bool verbose_coarsen          = false;
    bool verbose_pcoarsen         = false;
    bool verbose_compute_coarsen  = false;
    bool verbose_compute_coarsen2 = false;
    bool verbose_triple_mat_mult  = false;
    bool verbose_matmat           = false;
    bool verbose_fastmm           = false;
    bool verbose_matmat_recursive = false;
    bool verbose_matmat_A         = false;
    bool verbose_matmat_B         = false;
    bool verbose_matmat_assemble  = false;
    bool verbose_matmat_timing    = false;
    bool verbose_setup_coarse     = false;
    bool verbose_set_rhs          = false;
    bool verbose_repart_vec       = false;

    bool verbose_solve            = false;
    bool verbose_vcycle           = false;
    bool verbose_vcycle_residuals = false;
    bool verbose_solve_coarse     = false;
    bool verbose_update           = false;

    // **********************************************
    // setup functions
    // **********************************************

    saena_object()  = default;
    ~saena_object() = default;

    void destroy_mpi_comms();

    void destroy(){
        destroy_SuperLU();
        destroy_mpi_comms();
    }

    MPI_Comm get_orig_comm();

    void set_parameters(int max_iter = 100, double relative_tol = 1e-8, std::string smoother = "chebyshev",
                        int preSmooth = 3, int postSmooth = 3, std::string PSmoother = "jacobi",
                        float connStrength = 0.25, bool dynamic_lev = true, int max_lev = 10, int float_lev = 3,
                        double fil_thr = 1e-14, double fil_max = 1e-8, int fil_st = 1, int fil_rate = 2,
                        bool switch_to_den = false, float dense_thr = 0.1, int dense_sz_thr = 5000);

    void set_solve_params(int max_iter = 100, double relative_tol = 1e-8, std::string smoother = "chebyshev",
                          int preSmooth = 3, int postSmooth = 3);

    void print_parameters(saena_matrix *A) const;
    void print_lev_info(const Grid &g, const int porder) const;

    int setup(saena_matrix* A, std::vector<std::vector<int>> &m_l2g, std::vector<int> &m_g2u, int m_bdydof, std::vector<int> &order_dif);
    int coarsen(Grid *grid,std::vector< std::vector< std::vector<int> > > &map_all, std::vector< std::vector<int> > &g2u_all, std::vector<int> &order_dif);
    int SA(Grid *grid);
    int pcoarsen(Grid *grid, std::vector< std::vector< std::vector<int> > > &map_all, std::vector< std::vector<int> > &g2u_all, std::vector<int> &order_dif);
    int compute_coarsen(Grid *grid);
    int compute_coarsen_update_Ac(Grid *grid, std::vector<cooEntry> &diff);
    int triple_mat_mult(Grid *grid, bool symm = true);
    void filter(vector<cooEntry> &v);

    int matmat(saena_matrix *A, saena_matrix *B, saena_matrix *C, bool assemble = true, bool print_timing = false, bool B_trans = true);
    int matmat_CSC(CSCMat &Acsc, CSCMat &Bcsc, saena_matrix &C, bool trans = false);
    int matmat_memory_alloc(CSCMat &Acsc, CSCMat &Bcsc);
    int matmat_memory_free();
    int matmat_assemble(saena_matrix *A, saena_matrix *B, saena_matrix *C);

    int matmat_grid(Grid *grid);
//    int matmat(CSCMat &Acsc, CSCMat &Bcsc, saena_matrix &C, nnz_t send_size_max, double &matmat_time);
//    int matmat_COO(saena_matrix *A, saena_matrix *B, saena_matrix *C);

//    int reorder_split(vecEntry *arr, index_t low, index_t high, index_t pivot);
    void reorder_split(CSCMat_mm &A, CSCMat_mm &A1, CSCMat_mm &A2);
    void reorder_back_split(CSCMat_mm &A, CSCMat_mm &A1, CSCMat_mm &A2);
    int reorder_counter = 0; // use this to make sure the same number of calls to reorder_split() and reorder_back_split()

    // for fast_mm experiments
//    int compute_coarsen_test(Grid *grid);
//    int triple_mat_mult_test(Grid *grid, std::vector<cooEntry_row> &RAP_row_sorted);

    void fast_mm(CSCMat_mm &A, CSCMat_mm &B, std::vector<cooEntry> &C, MPI_Comm comm);

    int find_aggregation(saena_matrix* A, std::vector<index_t>& aggregate, std::vector<index_t>& splitNew);
    int create_strength_matrix(saena_matrix* A, strength_matrix* S) const;
    int create_strength_matrix_test(saena_matrix* A, strength_matrix* S) const;
    int aggregation_1_dist(strength_matrix *S, std::vector<index_t> &aggregate, std::vector<index_t> &aggArray) const;
    int aggregation_1_dist_orig(strength_matrix *S, std::vector<index_t> &aggregate, std::vector<index_t> &aggArray) const;
    int aggregation_1_dist_new(strength_matrix *S, std::vector<index_t> &aggregates, std::vector<index_t> &aggArray) const;
//    int aggregation_2_dist(strength_matrix *S, std::vector<unsigned long> &aggregate, std::vector<unsigned long> &aggArray);
    int aggregate_index_update(strength_matrix* S, std::vector<index_t>& aggregate, std::vector<index_t>& aggArray, std::vector<index_t>& splitNew);
    int create_prolongation(Grid *gird, std::vector< std::vector< std::vector<int> > > &map_all, std::vector< std::vector<int> > &g2u_all, std::vector<int> &order_dif);

    int set_repartition_rhs(saena_vector *rhs);
    int repart_vector(vector<value_t> &v, vector<index_t> &split, MPI_Comm comm);

    // if Saena needs to repartition the input A and rhs, then call repartition_u() at the start of the solving function.
    // then, repartition_back_u() at the end of the solve function to convert the solution to the initial partition.
    int repartition_u(std::vector<value_t>& u);
    int repartition_back_u(std::vector<value_t>& u);

    // if minor shrinking happens, u and rhs should be shrunk too.
//    int repartition_u_shrink_minor_prepare(Grid *grid);
//    int repartition_u_shrink_minor(std::vector<value_t> &u, Grid &grid);
//    int repartition_back_u_shrink_minor(std::vector<value_t> &u, Grid &grid);

//    int shrink_cpu_A(saena_matrix* Ac, std::vector<index_t>& P_splitNew);
//    int shrink_u_rhs(Grid* grid, std::vector<value_t>& u, std::vector<value_t>& rhs);
//    int unshrink_u(Grid* grid, std::vector<value_t>& u);

    int find_eig(saena_matrix& A) const;
//    int find_eig_Elemental(saena_matrix& A);

    // *****************
    // sparsification functions
    // *****************

//    int sparsify_trsl1(std::vector<cooEntry_row> & A, std::vector<cooEntry>& A_spars, double norm_frob_sq, nnz_t sample_size, MPI_Comm comm);
//    int sparsify_trsl2(std::vector<cooEntry> & A, std::vector<cooEntry>& A_spars, double norm_frob_sq, nnz_t sample_size, MPI_Comm comm);
//    int sparsify_drineas(std::vector<cooEntry_row> & A, std::vector<cooEntry>& A_spars, double norm_frob_sq, nnz_t sample_size, MPI_Comm comm);
    int sparsify_majid(std::vector<cooEntry> & A, std::vector<cooEntry>& A_spars, double norm_frob_sq, nnz_t sample_size, double max_val, MPI_Comm comm);
//    int sparsify_majid(std::vector<cooEntry_row> & A, std::vector<cooEntry>& A_spars, double norm_frob_sq, nnz_t sample_size, double max_val, std::vector<index_t> &splitNew, MPI_Comm comm);
    int sparsify_majid_serial(std::vector<cooEntry> & A, std::vector<cooEntry>& A_spars, double norm_frob_sq, nnz_t sample_size, double max_val, MPI_Comm comm);
    int sparsify_majid_with_dup(std::vector<cooEntry> & A, std::vector<cooEntry>& A_spars, double norm_frob_sq, nnz_t sample_size, double max_val, MPI_Comm comm);
//    double spars_prob(cooEntry a, double norm_frob_sq);
    double spars_prob(cooEntry a);

    // *****************
    // solve functions
    // **********************************************

    int solve(std::vector<value_t>& u);
    int solve_smoother(std::vector<value_t>& u);
    int solve_CG(std::vector<value_t>& u);
	int solve_petsc(std::vector<value_t>& u);
    int solve_pCG(std::vector<value_t>& u);
    int setup_vcycle_memory();
    void vcycle(Grid* grid, std::vector<value_t>& u, std::vector<value_t>& rhs);
    void inline smooth(Grid *grid, std::vector<value_t> &u, std::vector<value_t> &rhs, int iter) const;

    // *****************
    // GMRES functions
    // *****************

    int  GMRES(std::vector<double> &u);
    int  pGMRES(std::vector<double> &u);
    void GMRES_update(std::vector<double> &x, index_t k, saena_matrix_dense &h, std::vector<double> &s, std::vector<std::vector<double>> &v);
    void GeneratePlaneRotation(double &dx, double &dy, double &cs, double &sn);
    void ApplyPlaneRotation(double &dx, double &dy, const double &cs, const double &sn);
//    int GMRES(const Operator &A, std::vector<double> &x, const std::vector<double> &b,
//                            const Preconditioner &M, Matrix &H, int &m, int &max_iter, double &tol);

    // *****************
    // direct solvers
    // *****************
    int setup_SuperLU();
    int solve_coarsest_SuperLU(saena_matrix* A, std::vector<value_t>& u, std::vector<value_t>& rhs);
    int destroy_SuperLU();
    int solve_coarsest_CG(saena_matrix* A, std::vector<value_t>& u, std::vector<value_t>& rhs) const;
//    int solve_coarsest_Elemental(saena_matrix* A, std::vector<value_t>& u, std::vector<value_t>& rhs);

    // *****************
    // lazy update functions
    // **********************************************

    int update1(saena_matrix* A_new);
    int update2(saena_matrix* A_new);
    int update3(saena_matrix* A_new);

    // *****************
    // I/O functions
    // **********************************************

//    to write saena matrix to a file use related function from saena_matrix class.
//    int writeMatrixToFileA(saena_matrix* A, std::string name);
    int writeMatrixToFile(std::vector<cooEntry>& A, const std::string &name, MPI_Comm comm);
    int writeMatrixToFileP(prolong_matrix* P, std::string name);
    int writeMatrixToFileR(restrict_matrix* R, std::string name);

    template <class T>
    int writeVectorToFile(std::vector<T>& v, const std::string &name, MPI_Comm comm = MPI_COMM_WORLD,
                          bool mat_market = false, index_t OFST = 0);

    // *****************
    // Misc functions
    // **********************************************

    bool active(int l);

    int local_diff(saena_matrix &A, saena_matrix &B, std::vector<cooEntry> &C);
    int scale_vector(std::vector<value_t>& v, std::vector<value_t>& w);
    void transpose_locally(cooEntry *A, nnz_t size);
    void transpose_locally(cooEntry *A, nnz_t size, cooEntry *B);
    void transpose_locally(cooEntry *A, nnz_t size, const index_t &row_offset, cooEntry *B);

    int change_aggregation(saena_matrix* A, std::vector<index_t>& aggregate, std::vector<index_t>& splitNew);

    // double versions
//    int vcycle_d(Grid* grid, std::vector<value_d_t>& u_d, std::vector<value_d_t>& rhs_d);
//    int scale_vector_d(std::vector<value_d_t>& v, std::vector<value_t>& w);
//    int solve_coarsest_CG_d(saena_matrix* A, std::vector<value_t>& u, std::vector<value_t>& rhs);

    template <class T>
    int scale_vector_scalar(std::vector<T> &v, T a, std::vector<T> &w, bool add = false);

    // *****************
    // profiling parameters
    // **********************************************

    int rank_v = 0;

    double superlu_time       = 0.0;
    double Rtransfer_time     = 0.0;
    double Ptransfer_time     = 0.0;
    double vcycle_smooth_time = 0.0;
    double vcycle_other       = 0.0;
    double vcycle_resid       = 0.0;
    double vcycle_repart      = 0.0;

    void print_vcycle_time(int i, int k, MPI_Comm comm);

    void profile_matvecs();
    void profile_matvecs_breakdown();

    // *****************
    // pcoarsen functions
    // **********************************************

    int bdydof;
    int next_bdydof;
    int elemno;
    int nodeno_fine;
    int nodeno_coarse;
    int next_order;
    int prodim;

    int  next_p_level_random(const std::vector<int>& ind_fine, int order, vector<int> &ind, int *type = nullptr);
    void set_P_from_mesh(int order, std::vector<cooEntry_row> &P_temp, MPI_Comm comm, std::vector< std::vector<int> > &g2u_all, std::vector< std::vector< std::vector<int> > > &map_all);
    int  coarse_p_node_arr(std::vector< std::vector<int> > &map, int order, vector<int> &ind);
    inline int findloc(std::vector<int> &arr, int a);
    inline int mesh_info(int order, std::vector< std::vector< std::vector<int> > > &map_all, MPI_Comm comm);
    void g2umap(int order, std::vector< std::vector<int> > &g2u_all, std::vector< std::vector< std::vector<int> > > &map, MPI_Comm comm);
};

#endif //SAENA_SAENA_OBJECT_H

#include <saena_object.tpp>
