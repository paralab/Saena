#ifdef _USE_PETSC_

#ifndef SAENA_PETSC_FUNCTIONS_H
#define SAENA_PETSC_FUNCTIONS_H

#include <petsc.h>
#include "saena_matrix.h"
#include "saena_vector.h"
#include "restrict_matrix.h"
#include "prolong_matrix.h"


// info:
// pc_gamg_threshold: Relative threshold to use for dropping edges in aggregation graph
//                    Increasing the threshold decreases the rate of coarsening. Conversely reducing the threshold increases the rate of coarsening (aggressive coarsening) and thereby reduces the complexity of the coarse grids, and generally results in slower solver converge rates. Reducing coarse grid complexity reduced the complexity of Galerkin coarse grid construction considerably.
//                    Before coarsening or aggregating the graph, GAMG removes small values from the graph with this threshold, and thus reducing the coupling in the graph and a different (perhaps better) coarser set of points.
//                    0.0 means keep all nonzero entries in the graph; negative means keep even zero entries in the graph

// the following info is from the PETSc code gamg.c:
/*      PCGAMG - Geometric algebraic multigrid (AMG) preconditioner
       Options Database Keys:
    +   -pc_gamg_type <type> - one of agg, geo, or classical
    .   -pc_gamg_repartition  <true,default=false> - repartition the degrees of freedom accross the coarse grids as they are determined
    .   -pc_gamg_reuse_interpolation <true,default=false> - when rebuilding the algebraic multigrid preconditioner reuse the previously computed interpolations
    .   -pc_gamg_asm_use_agg <true,default=false> - use the aggregates from the coasening process to defined the subdomains on each level for the PCASM smoother
    .   -pc_gamg_process_eq_limit <limit, default=50> - GAMG will reduce the number of MPI processes used directly on the coarse grids so that there are around <limit>
                                            equations on each process that has degrees of freedom
    .   -pc_gamg_coarse_eq_limit <limit, default=50> - Set maximum number of equations on coarsest grid to aim for.
    .   -pc_gamg_threshold[] <thresh,default=0> - Before aggregating the graph GAMG will remove small values from the graph on each level
    -   -pc_gamg_threshold_scale <scale,default=1> - Scaling of threshold on each coarser grid if not specified

       Options Database Keys for default Aggregation:
    +  -pc_gamg_agg_nsmooths <nsmooth, default=1> - number of smoothing steps to use with smooth aggregation
    .  -pc_gamg_sym_graph <true,default=false> - symmetrize the graph before computing the aggregation
    -  -pc_gamg_square_graph <n,default=1> - number of levels to square the graph before aggregating it

       Multigrid options:
    +  -pc_mg_cycles <v> - v or w, see PCMGSetCycleType()
    .  -pc_mg_distinct_smoothup - configure the up and down (pre and post) smoothers separately, see PCMGSetDistinctSmoothUp()
    .  -pc_mg_type <multiplicative> - (one of) additive multiplicative full kascade
    -  -pc_mg_levels <levels> - Number of levels of multigrid to use.
 */

/*		petsc_option =  "-ksp_type cg -pc_type gamg"
						" -pc_gamg_type agg -pc_gamg_agg_nsmooths 1"
					    " -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -mg_levels_ksp_max_it 2"
						" -pc_gamg_threshold 0.015 -pc_gamg_sym_graph false -pc_gamg_square_graph 0"
						" -pc_gamg_coarse_eq_limit 500 -pc_gamg_sym_graph false -pc_gamg_square_graph 2"
						" -ksp_monitor_true_residual -ksp_norm_type unpreconditioned -ksp_max_it 2000 -ksp_rtol 1e-6 -ksp_converged_reason -ksp_view -log_view";
*/

string gamg_opts = "-ksp_type cg -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1"
                   " -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -mg_levels_ksp_max_it 3"
                   " -pc_gamg_threshold 0.01 -pc_gamg_sym_graph false -pc_gamg_square_graph 0 -pc_gamg_coarse_eq_limit 100"
                   " -ksp_monitor_true_residual -ksp_norm_type unpreconditioned -ksp_max_it 500 -ksp_rtol 1e-8 -ksp_converged_reason -ksp_view -log_view";

/*		petsc_option =  "-ksp_type cg -pc_type ml"
						" -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -mg_levels_ksp_max_it 2"
						" -pc_ml_maxNlevels 7"
						" -pc_ml_Threshold 0.125 -pc_ml_CoarsenScheme MIS -pc_ml_maxCoarseSize 1000"
						" -ksp_monitor_true_residual -ksp_norm_type unpreconditioned -ksp_max_it 2000 -ksp_rtol 1e-6 -ksp_converged_reason -ksp_view -log_view";
*/
string ml_opts = "-ksp_type cg -pc_type ml"
                 " -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -mg_levels_ksp_max_it 3"
                 " -pc_ml_maxNlevels 10 -pc_ml_Threshold 0.0 -pc_ml_CoarsenScheme Uncoupled -pc_ml_maxCoarseSize 100"
                 " -ksp_monitor_true_residual -ksp_norm_type unpreconditioned -ksp_max_it 500 -ksp_rtol 1e-8 -ksp_converged_reason -ksp_view -log_view";;

/*		petsc_option = 	"-ksp_type cg -pc_type hypre -pc_hypre_type boomeramg"
       " -pc_hypre_boomeramg_max_levels 7 -pc_hypre_boomeramg_relax_type_all Chebyshev -pc_hypre_boomeramg_grid_sweeps_all 2"
       " -pc_hypre_boomeramg_strong_threshold 0.11 -pc_hypre_boomeramg_coarsen_type Falgout"
       " -pc_hypre_boomeramg_agg_nl 3 -pc_hypre_boomeramg_agg_num_paths 3 -pc_hypre_boomeramg_truncfactor 0"
       " -ksp_monitor_true_residual -ksp_norm_type unpreconditioned -ksp_max_it 2000 -ksp_rtol 1e-6 -ksp_converged_reason -ksp_view"
       " -pc_hypre_boomeramg_print_statistics -log_view";
       //" -pc_hypre_boomeramg_print_debug";// -log_view";
*/

string hypre_opts = "-ksp_type cg -pc_type hypre -pc_hypre_type boomeramg -pc_hypre_boomeramg_max_levels 6"
                    " -pc_hypre_boomeramg_relax_type_all Chebyshev -pc_hypre_boomeramg_grid_sweeps_all 3"
                    " -pc_hypre_boomeramg_strong_threshold 0.0 -pc_hypre_boomeramg_coarsen_type Falgout"
                    " -pc_hypre_boomeramg_agg_nl 3 -pc_hypre_boomeramg_agg_num_paths 4"
                    " -ksp_monitor_true_residual -ksp_norm_type unpreconditioned -ksp_max_it 500 -ksp_rtol 1e-10 -ksp_converged_reason -ksp_view -log_view";// -pc_hypre_boomeramg_print_statistics";

string dcg_opts = "-ksp_type cg -pc_type jacobi"
                  " -ksp_monitor_true_residual -ksp_norm_type unpreconditioned -ksp_max_it 500 -ksp_rtol 1e-8 -ksp_converged_reason -ksp_view -log_view";

PetscErrorCode ComputeMatrix(KSP ksp, Mat J, Mat jac, void *ctx);
PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx);

int petsc_write_mat_file(const saena_matrix *A1);
int petsc_viewer(const Mat &A);
int petsc_viewer(const saena_matrix *A);
int petsc_viewer(const prolong_matrix *P);
int petsc_viewer(const restrict_matrix *R);

int petsc_prolong_matrix(const prolong_matrix *P, Mat &B);
int petsc_restrict_matrix(const restrict_matrix *R, Mat &B);
int petsc_saena_matrix(const saena_matrix *A, Mat &B);

int petsc_std_vector(const std::vector<value_t> &v, Vec &w, const int &OFST, MPI_Comm comm);
int petsc_std_vector(const value_t *&v, Vec &w, const int &OFST, MPI_Comm comm);
int petsc_saena_vector(const saena_vector *v, Vec &w);        // NOTE: not tested

int petsc_matmat(saena_matrix *A, saena_matrix *B);
int petsc_matmat_ave(saena_matrix *A, saena_matrix *B, int matmat_iter);
int petsc_matmat_ave2(saena_matrix *A, saena_matrix *B, int matmat_iter);
int petsc_check_matmat(saena_matrix *A, saena_matrix *B, saena_matrix *AB);
int petsc_mat_diff(Mat &A, Mat &B, saena_matrix *B_saena);

int petsc_coarsen(restrict_matrix *R, saena_matrix *A, prolong_matrix *P);
int petsc_coarsen_PtAP(restrict_matrix *R, saena_matrix *A, prolong_matrix *P);
int petsc_coarsen_2matmult(restrict_matrix *R, saena_matrix *A, prolong_matrix *P);
int petsc_check_matmatmat(restrict_matrix *R, saena_matrix *A, prolong_matrix *P, saena_matrix *Ac);

int petsc_solve(const saena_matrix *A, const vector<value_t> &rhs, vector<value_t> &u, const double &rel_tol);
int petsc_solve(saena_matrix *A1, vector<value_t> &b1, vector<value_t> &x1, const double &rel_tol, const char in_str[], string pc_type);
int petsc_solve(saena_matrix *A1, value_t *&b1, value_t *&x1, const double &rel_tol, const char in_str[], const string &pc_type);

#endif //SAENA_PETSC_FUNCTIONS_H

#endif //_USE_PETSC_
