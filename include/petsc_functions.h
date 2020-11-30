#ifdef _USE_PETSC_

#ifndef SAENA_PETSC_FUNCTIONS_H
#define SAENA_PETSC_FUNCTIONS_H

#include <petsc.h>
#include "saena_matrix.h"
#include "saena_vector.h"
#include "restrict_matrix.h"
#include "prolong_matrix.h"

PetscErrorCode ComputeMatrix(KSP ksp, Mat J, Mat jac, void *ctx);
PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx);

int petsc_viewer(const Mat &A);
int petsc_viewer(const saena_matrix *A);
int petsc_viewer(const prolong_matrix *P);
int petsc_viewer(const restrict_matrix *R);

int petsc_prolong_matrix(const prolong_matrix *P, Mat &B);
int petsc_restrict_matrix(const restrict_matrix *R, Mat &B);
int petsc_saena_matrix(const saena_matrix *A, Mat &B);

int petsc_std_vector(const std::vector<value_t> &v, Vec &w, const int &OFST, MPI_Comm comm);
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

#endif //SAENA_PETSC_FUNCTIONS_H

#endif //_USE_PETSC_