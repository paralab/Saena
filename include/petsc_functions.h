#ifndef SAENA_PETSC_FUNCTIONS_H
#define SAENA_PETSC_FUNCTIONS_H

#include <petsc.h>
#include <saena_matrix.h>

PetscErrorCode ComputeMatrix(KSP ksp, Mat J, Mat jac, void *ctx);

PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx);

int petsc_viewer(saena_matrix *A);

int petsc_coarsen(saena_matrix *A, saena_matrix *B, saena_matrix *C);

#endif //SAENA_PETSC_FUNCTIONS_H
