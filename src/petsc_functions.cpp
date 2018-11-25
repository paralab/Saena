#include <petsc_functions.h>


PetscErrorCode ComputeMatrix(KSP ksp, Mat J, Mat jac, void *ctx)
{
    PetscErrorCode ierr;
    PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs,num, numi, numj, numk;
    PetscScalar    v[7],Hx,Hy,Hz,HyHzdHx,HxHzdHy,HxHydHz;
    MatStencil     row, col[7];
    DM             da;
    MatNullSpace   nullspace;

    PetscFunctionBeginUser;
    ierr    = KSPGetDM(ksp,&da);CHKERRQ(ierr);
    ierr    = DMDAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
    Hx      = 1.0 / (PetscReal)(mx);
    Hy      = 1.0 / (PetscReal)(my);
    Hz      = 1.0 / (PetscReal)(mz);
    HyHzdHx = Hy*Hz/Hx;
    HxHzdHy = Hx*Hz/Hy;
    HxHydHz = Hx*Hy/Hz;
    ierr    = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
    for (k=zs; k<zs+zm; k++) {
        for (j=ys; j<ys+ym; j++) {
            for (i=xs; i<xs+xm; i++) {
                row.i = i; row.j = j; row.k = k;
                if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1) {
                    num = 0; numi=0; numj=0; numk=0;
                    if (k!=0) {
                        v[num]     = -HxHydHz;
                        col[num].i = i;
                        col[num].j = j;
                        col[num].k = k-1;
                        num++; numk++;
                    }
                    if (j!=0) {
                        v[num]     = -HxHzdHy;
                        col[num].i = i;
                        col[num].j = j-1;
                        col[num].k = k;
                        num++; numj++;
                    }
                    if (i!=0) {
                        v[num]     = -HyHzdHx;
                        col[num].i = i-1;
                        col[num].j = j;
                        col[num].k = k;
                        num++; numi++;
                    }
                    if (i!=mx-1) {
                        v[num]     = -HyHzdHx;
                        col[num].i = i+1;
                        col[num].j = j;
                        col[num].k = k;
                        num++; numi++;
                    }
                    if (j!=my-1) {
                        v[num]     = -HxHzdHy;
                        col[num].i = i;
                        col[num].j = j+1;
                        col[num].k = k;
                        num++; numj++;
                    }
                    if (k!=mz-1) {
                        v[num]     = -HxHydHz;
                        col[num].i = i;
                        col[num].j = j;
                        col[num].k = k+1;
                        num++; numk++;
                    }
                    v[num]     = (PetscReal)(numk)*HxHydHz + (PetscReal)(numj)*HxHzdHy + (PetscReal)(numi)*HyHzdHx;
                    col[num].i = i;   col[num].j = j;   col[num].k = k;
                    num++;
                    ierr = MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
                } else {
                    v[0] = -HxHydHz;                          col[0].i = i;   col[0].j = j;   col[0].k = k-1;
                    v[1] = -HxHzdHy;                          col[1].i = i;   col[1].j = j-1; col[1].k = k;
                    v[2] = -HyHzdHx;                          col[2].i = i-1; col[2].j = j;   col[2].k = k;
                    v[3] = 2.0*(HyHzdHx + HxHzdHy + HxHydHz); col[3].i = i;   col[3].j = j;   col[3].k = k;
                    v[4] = -HyHzdHx;                          col[4].i = i+1; col[4].j = j;   col[4].k = k;
                    v[5] = -HxHzdHy;                          col[5].i = i;   col[5].j = j+1; col[5].k = k;
                    v[6] = -HxHydHz;                          col[6].i = i;   col[6].j = j;   col[6].k = k+1;
                    ierr = MatSetValuesStencil(jac,1,&row,7,col,v,INSERT_VALUES);CHKERRQ(ierr);
                }
            }
        }
    }
    ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatSetNullSpace(J,nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
{
    PetscErrorCode ierr;
    PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs;
    PetscScalar    Hx,Hy,Hz;
    PetscScalar    ***array;
    DM             da;
    MatNullSpace   nullspace;

    PetscFunctionBeginUser;
    ierr = KSPGetDM(ksp,&da);CHKERRQ(ierr);
    ierr = DMDAGetInfo(da, 0, &mx, &my, &mz, 0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
    Hx   = 1.0 / (PetscReal)(mx);
    Hy   = 1.0 / (PetscReal)(my);
    Hz   = 1.0 / (PetscReal)(mz);
    ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, b, &array);CHKERRQ(ierr);
    for (k=zs; k<zs+zm; k++) {
        for (j=ys; j<ys+ym; j++) {
            for (i=xs; i<xs+xm; i++) {
                array[k][j][i] = 12 * PETSC_PI * PETSC_PI
                                 * PetscCosScalar(2*PETSC_PI*(((PetscReal)i+0.5)*Hx))
                                 * PetscCosScalar(2*PETSC_PI*(((PetscReal)j+0.5)*Hy))
                                 * PetscCosScalar(2*PETSC_PI*(((PetscReal)k+0.5)*Hz))
                                 * Hx * Hy * Hz;
            }
        }
    }
    ierr = DMDAVecRestoreArray(da, b, &array);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

    /* force right hand side to be consistent for singular matrix */
    /* note this is really a hack, normally the model would provide you with a consistent right handside */

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullspace,b);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}


int petsc_viewer(saena_matrix *A){

    PetscInitialize(0, nullptr, nullptr, nullptr);
    MPI_Comm comm = A->comm;
    PetscViewer    viewer;

    Mat B;
    MatCreate(comm, &B);
    MatSetSizes(B, A->M, A->M, A->Mbig, A->Mbig);

    // for serial
//    MatSetType(B,MATSEQAIJ);
//    MatSeqAIJSetPreallocation(B, 7, NULL);

    MatSetType(B, MATMPIAIJ);
    MatMPIAIJSetPreallocation(B, 80, NULL, 80, NULL);

    for(unsigned long i = 0; i < A->nnz_l; i++){
        MatSetValue(B, A->entry[i].row, A->entry[i].col, A->entry[i].val, INSERT_VALUES);
    }

    MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);

    PetscViewerDrawOpen(PETSC_COMM_WORLD, 0, "", 300, 0, 1000, 1000, &viewer);
    MatView(B,viewer);
    PetscViewerDestroy(&viewer);

    MatDestroy(&B);
    PetscFinalize();
    return 0;
}


int petsc_saena_matrix(saena_matrix *A, Mat &B){

    PetscInitialize(0, nullptr, nullptr, nullptr);
    MPI_Comm comm = A->comm;

    MatCreate(comm, &B);
    MatSetSizes(B, A->M, A->M, A->Mbig, A->Mbig);

    // for serial
//    MatSetType(B,MATSEQAIJ);
//    MatSeqAIJSetPreallocation(B, 7, NULL);

    MatSetType(B, MATMPIAIJ);
    MatMPIAIJSetPreallocation(B, 80, NULL, 80, NULL);

    for(unsigned long i = 0; i < A->nnz_l; i++){
        MatSetValue(B, A->entry[i].row, A->entry[i].col, A->entry[i].val, INSERT_VALUES);
    }

    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B,   MAT_FINAL_ASSEMBLY);

    PetscFinalize();
    return 0;
}
