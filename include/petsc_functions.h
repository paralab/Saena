#ifndef SAENA_PETSC_FUNCTIONS_H
#define SAENA_PETSC_FUNCTIONS_H

#include <petsc.h>
#include <saena_matrix.h>
#include "restrict_matrix.h"
#include "prolong_matrix.h"

PetscErrorCode ComputeMatrix(KSP ksp, Mat J, Mat jac, void *ctx);

PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx);

int petsc_viewer(saena_matrix *A);
int petsc_viewer(saena_matrix *A);

int petsc_coarsen(restrict_matrix *R, saena_matrix *A, prolong_matrix *P);

#endif //SAENA_PETSC_FUNCTIONS_H
