// this is based on the following example:
// from: /home/boss/softwares/petsc-3.8.0/src/ksp/ksp/examples/tutorials/ex34.c

/*T
   Concepts: KSP^solving a system of linear equations
   Concepts: KSP^Laplacian, 3d
   Processors: n
T*/

/*
Laplacian in 3D. Modeled by the partial differential equation

   div  grad u = f,  0 < x,y,z < 1,

with pure Neumann boundary conditions

   u = 0 for x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.

The functions are cell-centered

This uses multigrid to solve the linear system

       Contributed by Jianming Yang <jianming-yang@uiowa.edu>
*/

static char help[] = "Solves 3D Laplacian using multigrid.\n\n";

//#include <vector>

#include <petscdm.h>
#include <petscdmda.h>
#include <petscksp.h>

#include <petsc.h>
#include "saena.hpp"
//#include <include/saena_object.h>
//#include "string"
#include "grid.h"

extern PetscErrorCode ComputeMatrix(KSP,Mat,Mat,void*);
extern PetscErrorCode ComputeRHS(KSP,Vec,void*);

int laplacian3D_PETSc(saena::matrix* A, unsigned int mx, unsigned int my, unsigned int mz, MPI_Comm comm);


int main(int argc,char **argv)
{
    KSP            ksp;
    DM             da;
    PetscReal      norm;
    PetscErrorCode ierr;
    PetscInt       i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs;
    PetscScalar    Hx,Hy,Hz;
    PetscScalar    ***array;
    Vec            x,b,r;
    Mat            J;
    Mat            A;
    PetscViewer    viewer;

    ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

//    if(argc != 4)
//        if(rank==0) printf("error: input: <x_grid_size> <y_grid_size> <z_grid_size>\n\n");

//    int mx_read(std::stoi(argv[1]));
//    int my_read(std::stoi(argv[2]));
//    int mz_read(std::stoi(argv[3]));
//    mx = mx_read;
//    my = my_read;
//    mz = mz_read;

//    if(argc != 2){
//        if(rank == 0){
//            std::cout << "Usage: ./Saena <MatrixA>" << std::endl;
//            std::cout << "Matrix file should be in triples format." << std::endl;}
//        MPI_Finalize();
//        return -1;}

/*
    ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
    ierr = DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,mx,my,mz,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,0,&da);CHKERRQ(ierr);
    ierr = DMSetFromOptions(da);CHKERRQ(ierr);
    ierr = DMSetUp(da);CHKERRQ(ierr);
    ierr = DMDASetInterpolationType(da, DMDA_Q0);CHKERRQ(ierr);

    ierr = KSPSetDM(ksp,da);CHKERRQ(ierr);

    ierr = KSPSetComputeRHS(ksp,ComputeRHS,NULL);CHKERRQ(ierr);
    ierr = KSPSetComputeOperators(ksp,ComputeMatrix,NULL);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,NULL,NULL);CHKERRQ(ierr);
    ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);
    ierr = KSPGetRhs(ksp,&b);CHKERRQ(ierr);
    ierr = KSPGetOperators(ksp,NULL,&J);CHKERRQ(ierr);
    ierr = VecDuplicate(b,&r);CHKERRQ(ierr);
*/
    //******************************* View Matrix *******************************

//    ierr = KSPGetOperators(ksp,&A,NULL);CHKERRQ(ierr);
//    ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,1000,1000,&viewer);
//    MatView(A,viewer);

    //******************************* The rest of the PETSc example *******************************

/*
    ierr = MatMult(J,x,r);CHKERRQ(ierr);
    ierr = VecAXPY(r,-1.0,b);CHKERRQ(ierr);
    ierr = VecNorm(r,NORM_2,&norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g\n",(double)norm);CHKERRQ(ierr);

    ierr = DMDAGetInfo(da, 0, &mx, &my, &mz, 0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
    Hx   = 1.0 / (PetscReal)(mx);
    Hy   = 1.0 / (PetscReal)(my);
    Hz   = 1.0 / (PetscReal)(mz);
    ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, x, &array);CHKERRQ(ierr);

    for (k=zs; k<zs+zm; k++) {
        for (j=ys; j<ys+ym; j++) {
            for (i=xs; i<xs+xm; i++) {
                array[k][j][i] -=
                        PetscCosScalar(2*PETSC_PI*(((PetscReal)i+0.5)*Hx))*
                        PetscCosScalar(2*PETSC_PI*(((PetscReal)j+0.5)*Hy))*
                        PetscCosScalar(2*PETSC_PI*(((PetscReal)k+0.5)*Hz));
            }
        }
    }
    ierr = DMDAVecRestoreArray(da, x, &array);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

    ierr = VecNorm(x,NORM_INFINITY,&norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error norm %g\n",(double)norm);CHKERRQ(ierr);
    ierr = VecNorm(x,NORM_1,&norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error norm %g\n",(double)(norm/((PetscReal)(mx)*(PetscReal)(my)*(PetscReal)(mz))));CHKERRQ(ierr);
    ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error norm %g\n",(double)(norm/((PetscReal)(mx)*(PetscReal)(my)*(PetscReal)(mz))));CHKERRQ(ierr);
*/

    //******************************* Print Matrix Data *******************************
/*
    if(rank==0) printf("mx = %d, my = %d, mz = %d\n", mx, my, mz);

    PetscInt m, n;
    MatGetSize(A, &m, &n);
    if(rank==0) printf("m = %d, n = %d\n", m, n);

    PetscInt m_loc, n_loc;
    MatGetLocalSize(A, &m_loc, &n_loc);
//    printf("\nm_local = %d, n_local = %d\n", m_loc, n_loc);

    // compute nnz
    MatInfo info;
    double  nz_allocated, nz_used;
    MatGetInfo(A,MAT_LOCAL,&info);
    nz_allocated = info.nz_allocated;
    nz_used = info.nz_used;
//    printf("nz_allocated = %d, nz_used = %d\n\n", (PetscInt)nz_allocated, (PetscInt)nz_used);
    if(rank==0) printf("nnz = %d\n\n", (PetscInt)nz_used);
*/

    //******************************* Destroy PETSc Elements *******************************
/*
//    ierr = VecDestroy(&b);CHKERRQ(ierr); // b will be destroyed in KSPDestroy().
    ierr = VecDestroy(&r);CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    ierr = DMDestroy(&da);CHKERRQ(ierr);
*/
    //*****************************************************************************************
    //******************************* Create Matrix in Saena *******************************
    //*****************************************************************************************

    int mx2(std::stoi(argv[1]));
    int my2(std::stoi(argv[2]));
    int mz2(std::stoi(argv[3]));

    saena::matrix A_saena(comm);
    saena::laplacian3D(&A_saena, mx2, my2, mz2);

//    char *file_name(argv[1]);
//    A_saena.read_file(file_name);
//    A_saena.assemble();

//    saena_matrix* A_saena2 = A_saena.get_internal_matrix();
//    if(rank==0) printf("\nSaena Laplacian:\nMbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu\n\n",
//            A_saena2->Mbig, A_saena2->M, A_saena2->nnz_g, A_saena2->nnz_l);

    saena::options opts;
    saena::amg solver;
    solver.set_matrix(&A_saena, &opts);

    //******************************* Compare Laplacian in Saena with PETSc *******************************

//    saena_object* A_ob = solver.get_object();
//    Grid gg = A_ob->grids[5];
//    saena_matrix *A_saena2 = solver.get_object()->grids[5].A;
    saena_matrix *A_saena2 = A_saena.get_internal_matrix();

    Mat C_p2;
    MatCreate(comm, &C_p2);
    MatSetSizes(C_p2, A_saena2->M, A_saena2->M, A_saena2->Mbig, A_saena2->Mbig);

    // for serial
//    MatSetType(C_p2,MATSEQAIJ);
//    MatSeqAIJSetPreallocation(C_p2, 7, NULL);

    MatSetType(C_p2, MATMPIAIJ);
    MatMPIAIJSetPreallocation(C_p2, 80, NULL, 80, NULL);

    for(unsigned long i = 0; i < A_saena2->nnz_l; i++){
        MatSetValue(C_p2, A_saena2->entry[i].row, A_saena2->entry[i].col, A_saena2->entry[i].val, INSERT_VALUES);
    }

    MatAssemblyBegin(C_p2,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(C_p2,MAT_FINAL_ASSEMBLY);

    ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,1000,1000,&viewer);
    MatView(C_p2,viewer);

    // PetscViewer    viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "mat.m", &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);
    MatView(C_p2, viewer);
    PetscViewerPopFormat(viewer);
    PetscViewerDestroy(&viewer);

    //******************************* destroy and finalize *******************************

//    ierr = PetscViewerDestroy(&viewer);
    ierr = PetscFinalize();
    return ierr;
}

//PetscErrorCode ComputeRHS(KSP ksp,Vec b,void *ctx)
/*
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

    // force right hand side to be consistent for singular matrix
    // note this is really a hack, normally the model would provide you with a consistent right handside

    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullspace,b);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
*/

//PetscErrorCode ComputeMatrix(KSP ksp, Mat J,Mat jac, void *ctx)
/*
PetscErrorCode ComputeMatrix(KSP ksp, Mat J,Mat jac, void *ctx)
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
//    printf("PETSc:\nmx = %d, my = %d, mz = %d, Hx = %f, Hy = %f, Hz = %f\n", mx, my, mz, Hx, Hy, Hz);

    HyHzdHx = Hy*Hz/Hx;
    HxHzdHy = Hx*Hz/Hy;
    HxHydHz = Hx*Hy/Hz;
    ierr    = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
//    printf("corners: \nxs = %d, ys = %d, zs = %d, xm = %d, ym = %d, zm = %d\n", xs, ys, zs, xm, ym, zm);

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

//    PetscViewer    viewer;
//    PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,1000,1000,&viewer);
//    MatView(J,viewer);
//    MatView(jac,viewer);

    PetscFunctionReturn(0);
}
*/

//int laplacian3D_PETSc(saena::matrix* A, unsigned int mx, unsigned int my, unsigned int mz, MPI_Comm comm)
/*
int laplacian3D_PETSc(saena::matrix* A, unsigned int mx, unsigned int my, unsigned int mz, MPI_Comm comm)
{

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    int       i,j,k,xm,ym,zm,xs,ys,zs,num, numi, numj, numk;
    double    v[7],Hx,Hy,Hz,HyHzdHx,HxHzdHy,HxHydHz;
    unsigned int col_index[7];
    unsigned int node;
//    int     row, col[7];

    Hx      = 1.0 / (double)(mx);
    Hy      = 1.0 / (double)(my);
    Hz      = 1.0 / (double)(mz);
//    printf("\nSaena:\nmx = %d, my = %d, mz = %d, Hx = %f, Hy = %f, Hz = %f\n", mx, my, mz, Hx, Hy, Hz);

    HyHzdHx = Hy*Hz/Hx;
    HxHzdHy = Hx*Hz/Hy;
    HxHydHz = Hx*Hy/Hz;

    // split the 3D grid by only the z axis. So put the whole x and y grids on processors, by split z by the number of processors.
    xs = 0;
    xm = mx;
    ys = 0;
    ym = my;
    zm = (int)floor(mz / nprocs);
    zs = rank * zm;

    if(rank == nprocs - 1)
        zm = mz - ( (nprocs - 1) * zm);
//    printf("corners: \nxs = %d, ys = %d, zs = %d, xm = %d, ym = %d, zm = %d\n", xs, ys, zs, xm, ym, zm);

    for (k=zs; k<zs+zm; k++) {
        for (j=ys; j<ys+ym; j++) {
            for (i=xs; i<xs+xm; i++) {
                node = mx * my * k + mx * j + i; // for 2D it should be = mx * j + i
//                printf("node = %u\n", node);

                if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1) {
                    num = 0; numi=0; numj=0; numk=0;
                    if (k!=0) {
                        v[num]     = -HxHydHz;
//                        col[num].i = i;
//                        col[num].j = j;
//                        col[num].k = k-1;
                        col_index[num] = node - (mx * my);
                        num++; numk++;
                    }
                    if (j!=0) {
                        v[num]     = -HxHzdHy;
//                        col[num].i = i;
//                        col[num].j = j-1;
//                        col[num].k = k;
                        col_index[num] = node - mx;
                        num++; numj++;
                    }
                    if (i!=0) {
                        v[num]     = -HyHzdHx;
//                        col[num].i = i-1;
//                        col[num].j = j;
//                        col[num].k = k;
                        col_index[num] = node - 1;
                        num++; numi++;
                    }
                    if (i!=mx-1) {
                        v[num]     = -HyHzdHx;
//                        col[num].i = i+1;
//                        col[num].j = j;
//                        col[num].k = k;
                        col_index[num] = node + 1;
                        num++; numi++;
                    }
                    if (j!=my-1) {
                        v[num]     = -HxHzdHy;
//                        col[num].i = i;
//                        col[num].j = j+1;
//                        col[num].k = k;
                        col_index[num] = node + mx;
                        num++; numj++;
                    }
                    if (k!=mz-1) {
                        v[num]     = -HxHydHz;
//                        col[num].i = i;
//                        col[num].j = j;
//                        col[num].k = k+1;
                        col_index[num] = node + (mx * my);
                        num++; numk++;
                    }
                    v[num]     = (double)(numk)*HxHydHz + (double)(numj)*HxHzdHy + (double)(numi)*HyHzdHx;
//                    col[num].i = i;   col[num].j = j;   col[num].k = k;
                    col_index[num] = node;
                    num++;
//                    ierr = MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);

                    for(int l = 0; l < num; l++)
                        A->set(node, col_index[l], v[l]);

                } else {
                    v[0] = -HxHydHz;
//                    col[0].i = i;   col[0].j = j;   col[0].k = k-1;
                    col_index[0] = node - (mx * my);
                    A->set(node, col_index[0], v[0]);

                    v[1] = -HxHzdHy;
//                    col[1].i = i;   col[1].j = j-1; col[1].k = k;
                    col_index[1] = node - mx;
                    A->set(node, col_index[1], v[1]);

                    v[2] = -HyHzdHx;
//                    col[2].i = i-1; col[2].j = j;   col[2].k = k;
                    col_index[2] = node - 1;
                    A->set(node, col_index[2], v[2]);

                    v[3] = 2.0*(HyHzdHx + HxHzdHy + HxHydHz);
//                    col[3].i = i;   col[3].j = j;   col[3].k = k;
                    col_index[3] = node;
                    A->set(node, col_index[3], v[3]);

                    v[4] = -HyHzdHx;
//                    col[4].i = i+1; col[4].j = j;   col[4].k = k;
                    col_index[4] = node + 1;
                    A->set(node, col_index[4], v[4]);

                    v[5] = -HxHzdHy;
//                    col[5].i = i;   col[5].j = j+1; col[5].k = k;
                    col_index[5] = node + mx;
                    A->set(node, col_index[5], v[5]);

                    v[6] = -HxHydHz;
//                    col[6].i = i;   col[6].j = j;   col[6].k = k+1;
                    col_index[6] = node + (mx * my);
                    A->set(node, col_index[6], v[6]);

//                    ierr = MatSetValuesStencil(jac,1,&row,7,col,v,INSERT_VALUES);CHKERRQ(ierr);
                }
            }
        }
    }
//    ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
//    ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
//    ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,0,&nullspace);CHKERRQ(ierr);
//    ierr = MatSetNullSpace(J,nullspace);CHKERRQ(ierr);
//    ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);

    A->assemble();

    return 0;
}
 */