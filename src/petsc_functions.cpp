#ifdef _USE_PETSC_

#include <petsc_functions.h>
#include <assert.h>


// Read this about MPI communicators in PETSc
// https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Sys/PETSC_COMM_WORLD.html

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


int petsc_viewer(Mat &A){

    PetscViewer viewer;
    PetscViewerDrawOpen(PETSC_COMM_WORLD, 0, "", 300, 0, 1000, 1000, &viewer);
    PetscViewerDrawSetPause(viewer, -1);
    MatView(A, viewer);
    PetscViewerDestroy(&viewer);

    return 0;
}


int petsc_viewer(saena_matrix *A){

    if(A->active) {
        MPI_Comm comm = A->comm;
        PETSC_COMM_WORLD = comm;
        PetscInitialize(nullptr, nullptr, nullptr, nullptr);

//        int rank = -1;
//        MPI_Comm_rank(comm, &rank);

        std::vector<int> nnz_per_row_diag(A->M, 0);
        for (nnz_t i = 0; i < A->nnz_l_local; i++) {
//            if(rank == 1) printf("%6d\t(loc)\n", A->row_local[i]);
            ++nnz_per_row_diag[A->row_local[i]];
        }

        std::vector<int> nnz_per_row_off_diag(A->M, 0);
        for (nnz_t i = 0; i < A->nnz_l_remote; i++) {
//            if(rank == 1) printf("%6d\t(rem)\n", A->row_remote[i]);
            ++nnz_per_row_off_diag[A->row_remote[i]];
        }

        Mat B;
        MatCreate(comm, &B);
        MatSetSizes(B, A->M, A->M, A->Mbig, A->Mbig);

        // for serial
//        MatSetType(B, MATSEQAIJ);
//        MatSeqAIJSetPreallocation(B, 7, NULL);

        MatSetType(B, MATMPIAIJ); // Documentation: A matrix type to be used for parallel sparse matrices
        MatMPIAIJSetPreallocation(B, 0, &nnz_per_row_diag[0], 0, &nnz_per_row_off_diag[0]);

        for (unsigned long i = 0; i < A->nnz_l; i++) {
//            if(rank == 1) printf("%6d\t%6d\t%6f\n", A->entry[i].row, A->entry[i].col, A->entry[i].val);
//            assert(A->entry[i].row >= A->split[rank] && A->entry[i].row < A->split[rank + 1]);
//            assert(A->entry[i].col >= 0 && A->entry[i].col < A->Mbig);
            MatSetValue(B, A->entry[i].row, A->entry[i].col, A->entry[i].val, INSERT_VALUES);
        }

        MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);

        PetscViewer viewer;
        PetscViewerDrawOpen(PETSC_COMM_WORLD, nullptr, "", 300, 0, 1000, 1000, &viewer);
        PetscViewerDrawSetPause(viewer, -1);
        MatView(B, viewer);
        PetscViewerDestroy(&viewer);

        MatDestroy(&B);
        PetscFinalize();
    }

    return 0;
}


int petsc_prolong_matrix(prolong_matrix *P, Mat &B){

    PetscBool petsc_init;
    PetscInitialized(&petsc_init);

    MPI_Comm comm = P->comm;
    PETSC_COMM_WORLD = comm;

    bool fin_petsc = true;
    if(!petsc_init) {
        PetscInitialize(nullptr, nullptr, nullptr, nullptr);
    }else{
        fin_petsc = false; // if PETSc is initialized in another function, that function should call PetscFinalize().
    }

    int rank = -1;
    MPI_Comm_rank(comm, &rank);

    std::vector<int> nnz_per_row_diag(P->M, 0);
    for(nnz_t i = 0; i < P->nnz_l_local; ++i){
        ++nnz_per_row_diag[P->row_local[i]];
    }

    std::vector<int> nnz_per_row_off_diag(P->M, 0);
    for(nnz_t i = 0; i < P->nnz_l_remote; ++i){
        ++nnz_per_row_off_diag[P->row_remote[i]];
    }

    MatCreate(comm, &B);
    MatSetSizes(B, P->M, P->splitNew[rank+1] - P->splitNew[rank], P->Mbig, P->Nbig);

    // for serial
//    MatSetType(B, MATSEQAIJ);
//    MatSeqAIJSetPreallocation(B, 7, NULL);

    MatSetType(B, MATMPIAIJ); // Documentation: A matrix type to be used for parallel sparse matrices
    MatMPIAIJSetPreallocation(B, 0, &nnz_per_row_diag[0], 0, &nnz_per_row_off_diag[0]);

    for(nnz_t i = 0; i < P->nnz_l; i++){
        MatSetValue(B, P->entry[i].row + P->split[rank], P->entry[i].col, P->entry[i].val, INSERT_VALUES);
    }

    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B,   MAT_FINAL_ASSEMBLY);

/*
    MatInfo info;
    MatGetInfo(B, MAT_GLOBAL_SUM,&info);
    if(!rank){
        std::cout << "\nmalloc = " << info.mallocs << ", nz_a = " << info.nz_allocated << ", nz_u = " << info.nz_used
                  << ", block size = " << info.block_size << std::endl;

        PetscInt m, n;
        MatGetSize(B, &m,&n);
        printf("\nm = %d, n = %d\n", m, n);
    }
*/

//    petsc_viewer(B);

    if(fin_petsc){
        PetscFinalize();
    }

    return 0;
}

int petsc_viewer(prolong_matrix *P){

    PetscBool petsc_init;
    PetscInitialized(&petsc_init);

    MPI_Comm comm = P->comm;
    PETSC_COMM_WORLD = comm;

    bool fin_petsc = true;
    if(!petsc_init){
        PetscInitialize(nullptr, nullptr, nullptr, nullptr);
    }else{
        fin_petsc = false; // if PETSc is initialized in another function, that function should call PetscFinalize().
    }

    Mat B;
    petsc_prolong_matrix(P, B);
    petsc_viewer(B);

    if(fin_petsc){
        PetscFinalize();
    }

    return 0;
}

int petsc_viewer(restrict_matrix *R){

    PetscBool petsc_init;
    PetscInitialized(&petsc_init);

    MPI_Comm comm = R->comm;
    PETSC_COMM_WORLD = comm;

    bool fin_petsc = true;
    if(!petsc_init){
        PetscInitialize(nullptr, nullptr, nullptr, nullptr);
    }else{
        fin_petsc = false; // if PETSc is initialized in another function, that function should call PetscFinalize().
    }

    Mat B;
    petsc_restrict_matrix(R, B);
    petsc_viewer(B);

    if(fin_petsc){
        PetscFinalize();
    }

    return 0;
}

int petsc_restrict_matrix(restrict_matrix *R, Mat &B){

//    PetscInitialize(0, nullptr, nullptr, nullptr);

    MPI_Comm comm = R->comm;
    PETSC_COMM_WORLD = comm;
    int rank;
    MPI_Comm_rank(comm, &rank);

    std::vector<int> nnz_per_row_diag(R->M, 0);
    for(nnz_t i = 0; i < R->nnz_l_local; i++){
        nnz_per_row_diag[R->row_local[i]]++;
    }

    std::vector<int> nnz_per_row_off_diag(R->M, 0);
    for(nnz_t i = 0; i < R->nnz_l_remote; i++){
        nnz_per_row_off_diag[R->row_remote[i]]++;
    }

    MatCreate(comm, &B);
    MatSetSizes(B, R->M, R->split[rank+1] - R->split[rank], R->Mbig, R->Nbig);

    // for serial
//    MatSetType(B,MATSEQAIJ);
//    MatSeqAIJSetPreallocation(B, 7, NULL);

    MatSetType(B, MATMPIAIJ); // Documentation: A matrix type to be used for parallel sparse matrices
    MatMPIAIJSetPreallocation(B, 0, &nnz_per_row_diag[0], 0, &nnz_per_row_off_diag[0]);

    for(unsigned long i = 0; i < R->nnz_l; i++){
        MatSetValue(B, R->entry[i].row + R->splitNew[rank], R->entry[i].col, R->entry[i].val, INSERT_VALUES);
    }

    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B,   MAT_FINAL_ASSEMBLY);

//    PetscFinalize();
    return 0;
}


int petsc_saena_matrix(saena_matrix *A, Mat &B){

//    PetscInitialize(0, nullptr, nullptr, nullptr);

    MPI_Comm comm = A->comm;
    PETSC_COMM_WORLD = comm;

    std::vector<int> nnz_per_row_diag(A->M, 0);
    for(nnz_t i = 0; i < A->nnz_l_local; i++){
        nnz_per_row_diag[A->row_local[i]]++;
    }

    std::vector<int> nnz_per_row_off_diag(A->M, 0);
    for(nnz_t i = 0; i < A->nnz_l_remote; i++){
        nnz_per_row_off_diag[A->row_remote[i]]++;
    }

    MatCreate(comm, &B);
    MatSetSizes(B, A->M, A->M, A->Mbig, A->Mbig);

    // for serial
//    MatSetType(B,MATSEQAIJ);
//    MatSeqAIJSetPreallocation(B, 7, NULL);

    MatSetType(B, MATMPIAIJ); // Documentation: A matrix type to be used for parallel sparse matrices
    MatMPIAIJSetPreallocation(B, 0, &nnz_per_row_diag[0], 0, &nnz_per_row_off_diag[0]);

    for(unsigned long i = 0; i < A->nnz_l; i++){
        MatSetValue(B, A->entry[i].row, A->entry[i].col, A->entry[i].val, INSERT_VALUES);
    }

    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B,   MAT_FINAL_ASSEMBLY);

//    int rank;
//    MPI_Comm_rank(comm, &rank);
//    MatInfo info;
//    MatGetInfo(B, MAT_GLOBAL_SUM,&info);
//    if(!rank){
//        std::cout << "\npetsc_saena_matrix:\nmalloc = " << info.mallocs << ", nz_a = " << info.nz_allocated << ", nz_u = " << info.nz_used
//                  << ", block size = " << info.block_size << std::endl;
//
//        PetscInt m, n;
//        MatGetSize(B, &m,&n);
//        printf("m = %d, n = %d\n", m, n);
//    }

//    PetscFinalize();
    return 0;
}


int petsc_coarsen(restrict_matrix *R, saena_matrix *A, prolong_matrix *P){

    // todo: petsc has a MatGalerkin() function for coarsening. check this link:
    // https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatGalerkin.html
    // manual for MatMatMatMult():
    // https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatMatMatMult.html

    MPI_Comm comm = A->comm;
    PETSC_COMM_WORLD = comm;
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);

    Mat R2, A2, P2, RAP;
    petsc_restrict_matrix(R, R2);
    petsc_saena_matrix(A, A2);
    petsc_prolong_matrix(P, P2);

    MPI_Barrier(comm);
    double t1 = MPI_Wtime();
    MatMatMatMult(R2, A2, P2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &RAP);
    t1 = MPI_Wtime() - t1;
    print_time_ave(t1, "PETSc MatMatMatMult", comm);

//    petsc_viewer(RAP);

    MatDestroy(&R2);
    MatDestroy(&A2);
    MatDestroy(&P2);
    MatDestroy(&RAP);
    PetscFinalize();
    return 0;
}


int petsc_coarsen_PtAP(restrict_matrix *R, saena_matrix *A, prolong_matrix *P){

    // todo: petsc has a MatGalerkin() function for coarsening. check this link:
    // https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatGalerkin.html
    // manual for MatMatMatMult():
    // https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatMatMatMult.html

    MPI_Comm comm = A->comm;
    PETSC_COMM_WORLD = comm;
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);

    Mat A2, P2, RAP;
//    petsc_restrict_matrix(R, R2);
    petsc_saena_matrix(A, A2);
    petsc_prolong_matrix(P, P2);

    MPI_Barrier(comm);
    double t1 = MPI_Wtime();
    MatPtAP(A2, P2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &RAP);
    t1 = MPI_Wtime() - t1;
    print_time_ave(t1, "PETSc MatPtAP", comm);

//    petsc_viewer(RAP);

//    int rank;
//    MPI_Comm_rank(comm, &rank);
//    MatInfo info;
//    MatGetInfo(RAP, MAT_GLOBAL_SUM,&info);
//    if(!rank){
//        std::cout << "\npetsc_saena_matrix:\nmalloc = " << info.mallocs << ", nz_a = " << info.nz_allocated << ", nz_u = " << info.nz_used
//                  << ", block size = " << info.block_size << std::endl;
//
//        PetscInt m, n;
//        MatGetSize(RAP, &m,&n);
//        printf("m = %d, n = %d\n", m, n);
//    }

//    MatDestroy(&R2);
    MatDestroy(&A2);
    MatDestroy(&P2);
    MatDestroy(&RAP);
    PetscFinalize();
    return 0;
}

int petsc_coarsen_2matmult(restrict_matrix *R, saena_matrix *A, prolong_matrix *P){

    // todo: petsc has a MatGalerkin() function for coarsening. check this link:
    // https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatGalerkin.html
    // manual for MatMatMatMult():
    // https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatMatMatMult.html

    MPI_Comm comm = A->comm;
    PETSC_COMM_WORLD = comm;
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);

    Mat R2, A2, P2, RA, RAP;
    petsc_restrict_matrix(R, R2);
    petsc_saena_matrix(A, A2);
    petsc_prolong_matrix(P, P2);

    MPI_Barrier(comm);
    double t1 = MPI_Wtime();
    MatMatMult(R2, A2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &RA);
    MatMatMult(RA, P2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &RAP);
    t1 = MPI_Wtime() - t1;
    print_time_ave(t1, "PETSc 2*MatMatMult", comm);

//    petsc_viewer(RAP);

    MatDestroy(&R2);
    MatDestroy(&A2);
    MatDestroy(&P2);
    MatDestroy(&RA);
    MatDestroy(&RAP);
    PetscFinalize();
    return 0;
}

int petsc_check_matmatmat(restrict_matrix *R, saena_matrix *A, prolong_matrix *P, saena_matrix *Ac){

    // Note: saena_matrix Ac should be scaled back for comparison.

    MPI_Comm comm = A->comm;
    PETSC_COMM_WORLD = comm;
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);

    int rank;
    MPI_Comm_rank(comm, &rank);

    Ac->scale_back_matrix();

    Mat R2, A2, Ac2, P2, RAP;
    petsc_restrict_matrix(R, R2);
    petsc_saena_matrix(A, A2);
    petsc_prolong_matrix(P, P2);
    petsc_saena_matrix(Ac, Ac2);

    // method1
    // =====================
//    MatMatMatMult(R2, A2, P2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &RAP);

    // method2
    // =====================
    MatPtAP(A2, P2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &RAP);

    // method3
    // =====================
//    Mat RA;
//    MatMatMult(R2, A2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &RA);
//    MatMatMult(RA, P2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &RAP);
//    MatDestroy(&RA);
    // =====================
    // debug info
    // =====================

//    petsc_viewer(RAP);
//    petsc_viewer(Ac2);

//    MatInfo info;
//    MatGetInfo(RAP, MAT_GLOBAL_SUM,&info);
//    if(!rank){
//        std::cout << "\nRAP:\nmalloc = " << info.mallocs << ", nz_a = " << info.nz_allocated << ", nz_u = " << info.nz_used
//                  << ", block size = " << info.block_size << std::endl;
//
//        PetscInt m, n;
//        MatGetSize(RAP, &m,&n);
//        printf("m = %d, n = %d\n", m, n);
//    }
//
//    MatGetInfo(Ac2, MAT_GLOBAL_SUM,&info);
//    if(!rank){
//        std::cout << "\nAc:\nmalloc = " << info.mallocs << ", nz_a = " << info.nz_allocated << ", nz_u = " << info.nz_used
//                  << ", block size = " << info.block_size << std::endl;
//
//        PetscInt m, n;
//        MatGetSize(Ac2, &m,&n);
//        printf("m = %d, n = %d\n", m, n);
//    }

    // ====================================
    // compute the norm of the difference
    // ====================================

    MatAXPY(RAP, -1, Ac2, DIFFERENT_NONZERO_PATTERN);

    double norm_frob;
    MatNorm(RAP, NORM_FROBENIUS, &norm_frob);
    if(rank==0) printf("\nnorm_frobenius(Ac_PETSc - Ac_Saena) = %.16f\n", norm_frob);

    // ====================================

//    petsc_viewer(Ac2);
//    petsc_viewer(RAP);

    // ====================================
    // destroy
    // ====================================

    MatDestroy(&RAP);
    MatDestroy(&P2);
    MatDestroy(&A2);
    MatDestroy(&Ac2);
    MatDestroy(&R2);

    Ac->scale_matrix();

    PetscFinalize();
    return 0;
}


int petsc_matmat(saena_matrix *A, saena_matrix *B){

    MPI_Comm comm = A->comm;
    PETSC_COMM_WORLD = comm;
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);

    Mat A2, B2, AB;
    petsc_saena_matrix(A, A2);
    petsc_saena_matrix(B, B2);

    // Turn on logging of objects and events.
//    PetscLogDefaultBegin();
//    float oldthresh;
//    PetscLogSetThreshold(0.1, &oldthresh);

    MPI_Barrier(comm);
    double t1 = MPI_Wtime();
    MatMatMult(A2, B2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &AB);
    t1 = MPI_Wtime() - t1;
    print_time_ave(t1, "\nPETSc MatMatMult", comm, true, false);

    // wrtie the log to file
//    int nprocs;
//    MPI_Comm_size(comm, &nprocs);
//    std::string filename = "petsclog_" + std::to_string(nprocs) + ".xml";
//    PetscViewer viewer;
//    PetscViewerASCIIOpen(PETSC_COMM_WORLD, filename.c_str(), &viewer);
//    PetscLogView(viewer);
//    PetscViewerDestroy(&viewer);

//    petsc_viewer(AB);

    MatDestroy(&A2);
    MatDestroy(&B2);
    MatDestroy(&AB);
    PetscFinalize();
    return 0;
}


int petsc_matmat_ave(saena_matrix *A, saena_matrix *B, int matmat_iter){

    MPI_Comm comm = A->comm;
    PETSC_COMM_WORLD = comm;
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);

    Mat A2, B2, AB;
    petsc_saena_matrix(A, A2);
    petsc_saena_matrix(B, B2);

    MPI_Barrier(comm);
    double t1 = MPI_Wtime();

/*
    MatMatMult(A2, B2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &AB);

    for(int i = 1; i < matmat_iter; i++){
        MatMatMult(A2, B2, MAT_REUSE_MATRIX, PETSC_DEFAULT, &AB);
    }
*/

    for(int i = 0; i < matmat_iter; i++){
        MatMatMult(A2, B2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &AB);
    }

    t1 = MPI_Wtime() - t1;
    print_time_ave(t1 / matmat_iter, "\nPETSc MatMatMult", comm, true, false);

//    petsc_viewer(AB);

    MatDestroy(&A2);
    MatDestroy(&B2);
    MatDestroy(&AB);
    PetscFinalize();
    return 0;
}


int petsc_matmat_ave2(saena_matrix *A, saena_matrix *B, int matmat_iter){

    MPI_Comm comm = A->comm;
    PETSC_COMM_WORLD = comm;
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);

    Mat A2, B2, AB;
    petsc_saena_matrix(A, A2);
    petsc_saena_matrix(B, B2);

    double t1 = 0.0, tmp = 0.0;

    MPI_Barrier(comm);
    for(int i = 0; i < matmat_iter; i++){
        tmp = MPI_Wtime();
        MatMatMult(A2, B2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &AB);
        tmp = MPI_Wtime() - tmp;
        t1 += tmp;
        MatDestroy(&AB);
    }

    print_time_ave(t1 / matmat_iter, "\nPETSc MatMatMult", comm, true, false);

//    petsc_viewer(AB);

    MatDestroy(&A2);
    MatDestroy(&B2);
//    MatDestroy(&AB);
    PetscFinalize();
    return 0;
}


int petsc_check_matmat(saena_matrix *A, saena_matrix *B, saena_matrix *AB){

    MPI_Comm comm = A->comm;
    PETSC_COMM_WORLD = comm;
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);

    int rank;
    MPI_Comm_rank(comm, &rank);

    AB->scale_back_matrix();

    Mat A2, B2, AB2, C;
    petsc_saena_matrix(A, A2);
    petsc_saena_matrix(B, B2);
    petsc_saena_matrix(AB, AB2);

//    MatView(A2,PETSC_VIEWER_STDOUT_WORLD);

//    MPI_Barrier(comm);
//    double t1 = MPI_Wtime();
    MatMatMult(A2, B2, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C);
//    t1 = MPI_Wtime() - t1;
//    print_time_ave(t1, "PETSc MatMatMult", comm);

//    petsc_viewer(AB2);
//    petsc_viewer(C);

    // ====================================
    // print the difference between the two result matrices
    // ====================================

//    petsc_mat_diff(C, AB2, AB);

    // ====================================
    // compute the norm of the difference
    // ====================================

    MatAXPY(C, -1, AB2, DIFFERENT_NONZERO_PATTERN);

    double norm_frob;
    MatNorm(C, NORM_FROBENIUS, &norm_frob);
    if(rank==0) printf("\nnorm_frobenius(AB_PETSc - AB_Saena) = %.16f\n\n", norm_frob);

    AB->scale_matrix();

    MatDestroy(&A2);
    MatDestroy(&B2);
    MatDestroy(&AB2);
    MatDestroy(&C);
    PetscFinalize();
    return 0;
}


int petsc_mat_diff(Mat &A, Mat &B, saena_matrix *B_saena){
//    PetscInitialize(nullptr, nullptr, nullptr, nullptr);

    MPI_Comm comm = B_saena->comm;
    int rank;
    MPI_Comm_rank(comm, &rank);

    PetscErrorCode ierr;
    PetscInt       i,j,nrsub,ncsub,*rsub,*csub,mystart,myend;
    PetscScalar    *vals;

//    MatView(A2,PETSC_VIEWER_STDOUT_WORLD);
//    petsc_viewer(AB2);
//    petsc_viewer(C);

//    ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

    // ====================================
    // Setup the matrix computed with PETSc
    // ====================================

    ierr  = MatGetOwnershipRange(A,&mystart,&myend);CHKERRQ(ierr);
    nrsub = myend - mystart;
    ncsub = B_saena->Mbig;         //todo : fix ncsub
//    if(rank == 1) printf("myend = %d, mystart = %d, nrsub = %d, ncsub = %d\n", mystart, myend, nrsub, ncsub);

    ierr  = PetscMalloc1(nrsub*ncsub,&vals);CHKERRQ(ierr);
    ierr  = PetscMalloc1(nrsub,&rsub);CHKERRQ(ierr);
    ierr  = PetscMalloc1(ncsub,&csub);CHKERRQ(ierr);

    for (i = 0; i < nrsub; ++i){
        rsub[i] = i + mystart;
    }

    for (i = 0; i < ncsub; ++i){
        csub[i] = i;
    }

    ierr = MatGetValues(A,nrsub,rsub,ncsub,csub,vals);CHKERRQ(ierr);

//    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"processor number %d: mystart=%D, myend=%D\n",rank,mystart,myend);CHKERRQ(ierr);
//    for (i=0; i<nrsub; i++) {
//        for (j=0; j<ncsub; j++) {
//            if (PetscImaginaryPart(vals[i*ncsub+j]) != 0.0) {
//                ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  C[%D, %D] = %g + %g i\n",rsub[i],csub[j],(double)PetscRealPart(vals[i*ncsub+j]),(double)PetscImaginaryPart(vals[i*ncsub+j]));CHKERRQ(ierr);
//            } else {
//                ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  C[%D, %D] = %g\n",rsub[i],csub[j],(double)PetscRealPart(vals[i*ncsub+j]));CHKERRQ(ierr);
//            }
//        }
//    }

    // ====================================
    // Setup the matrix computed with Saena
    // ====================================

    PetscInt    nrsub2,ncsub2,*rsub2,*csub2;
    PetscScalar *vals2;

    ierr  = MatGetOwnershipRange(A,&mystart,&myend);CHKERRQ(ierr);
    nrsub2 = myend - mystart;
    ncsub2 = B_saena->Mbig;
//    if(rank == 1) printf("myend = %d, mystart = %d, nrsub = %d, ncsub = %d\n", mystart, myend, nrsub, ncsub);

    assert( (nrsub == nrsub2) && (ncsub == ncsub2));

    ierr  = PetscMalloc1(nrsub2*ncsub2,&vals2);CHKERRQ(ierr);
    ierr  = PetscMalloc1(nrsub2,&rsub2);CHKERRQ(ierr);
    ierr  = PetscMalloc1(ncsub2,&csub2);CHKERRQ(ierr);

    for (i = 0; i < nrsub2; ++i){
        rsub2[i] = i + mystart;
    }

    for (i = 0; i < ncsub2; ++i){
        csub2[i] = i;
    }

    ierr = MatGetValues(A,nrsub2,rsub2,ncsub2,csub2,vals2);CHKERRQ(ierr);

//    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"processor number %d: mystart=%D, myend=%D\n",rank,mystart,myend);CHKERRQ(ierr);
//    for (i=0; i<nrsub2; i++) {
//        for (j=0; j<ncsub2; j++) {
//            if (PetscImaginaryPart(vals[i*ncsub2+j]) != 0.0) {
//                ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  C[%D, %D] = %g + %g i\n",rsub2[i],csub2[j],(double)PetscRealPart(vals2[i*ncsub2+j]),(double)PetscImaginaryPart(vals2[i*ncsub2+j]));CHKERRQ(ierr);
//            } else {
//                ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"  C[%D, %D] = %g\n",rsub2[i],csub2[j],(double)PetscRealPart(vals2[i*ncsub2+j]));CHKERRQ(ierr);
//            }
//        }
//    }

//    for (i=0; i<nrsub2; i++) {
//        for (j=0; j<ncsub2; j++) {
//            if((double) PetscRealPart(vals[i * ncsub + j]) - (double) PetscRealPart(vals2[i * ncsub2 + j]) < 1e-15) {
//                PetscSynchronizedPrintf(PETSC_COMM_WORLD, "  C[%D, %D] = (%g, %g)\n",
//                        rsub[i], csub[j], (double) PetscRealPart(vals[i * ncsub + j]),
//                        rsub2[i], csub2[j], (double) PetscRealPart(vals2[i * ncsub2 + j]));
//            }
//        }
//    }

    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
    ierr = PetscFree(rsub);CHKERRQ(ierr);
    ierr = PetscFree(csub);CHKERRQ(ierr);
    ierr = PetscFree(vals);CHKERRQ(ierr);

    ierr = PetscFree(rsub2);CHKERRQ(ierr);
    ierr = PetscFree(csub2);CHKERRQ(ierr);
    ierr = PetscFree(vals2);CHKERRQ(ierr);

//    PetscFinalize();
    return 0;
}


#endif //_USE_PETSC_