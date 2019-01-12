#ifndef SAENA_PETSC_FUNCTIONS_H
#define SAENA_PETSC_FUNCTIONS_H

#include <petsc.h>
#include <saena_matrix.h>
#include "restrict_matrix.h"
#include "prolong_matrix.h"

PetscErrorCode ComputeMatrix(KSP ksp, Mat J, Mat jac, void *ctx);

PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx);

int petsc_viewer(Mat &A);
int petsc_viewer(saena_matrix *A);
int petsc_prolong_matrix(prolong_matrix *P, Mat &B);
int petsc_restrict_matrix(restrict_matrix *R, Mat &B);
int petsc_saena_matrix(saena_matrix *A, Mat &B);
int petsc_coarsen(restrict_matrix *R, saena_matrix *A, prolong_matrix *P);
int petsc_coarsen_PtAP(restrict_matrix *R, saena_matrix *A, prolong_matrix *P);
int petsc_coarsen_2matmult(restrict_matrix *R, saena_matrix *A, prolong_matrix *P);
int petsc_check_matmatmat(restrict_matrix *R, saena_matrix *A, prolong_matrix *P);

#endif //SAENA_PETSC_FUNCTIONS_H
