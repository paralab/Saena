#ifndef SAENA_PETSC_FUNCTIONS_H
#define SAENA_PETSC_FUNCTIONS_H

#include <petsc.h>

PetscErrorCode ComputeMatrix(KSP ksp, Mat J, Mat jac, void *ctx);

PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx);

#endif //SAENA_PETSC_FUNCTIONS_H
