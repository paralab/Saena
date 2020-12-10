#ifdef _USE_PETSC_

#include <petsc_functions.h>
#include <cassert>


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
                array[k][j][i] = 12 * SAENA_PI * SAENA_PI
                                 * PetscCosScalar(2*SAENA_PI*(((PetscReal)i+0.5)*Hx))
                                 * PetscCosScalar(2*SAENA_PI*(((PetscReal)j+0.5)*Hy))
                                 * PetscCosScalar(2*SAENA_PI*(((PetscReal)k+0.5)*Hz))
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


int petsc_viewer(const Mat &A){

    int sz = 1800;

    int m, n;
    MatGetSize(A, &m, &n);

    // set the window size
    int w = 0, h = 0;   // width and height
    if(m == n){
        sz = 1000;
        w = sz;
        h = sz;
    }else if(m > n){
        w = (static_cast<double>(n) / m) * sz;
        h = sz;
    }else{
        h = (static_cast<double>(m) / n) * sz;
        w = sz;
    }

//    printf("m = %d, n = %d, w = %d, h = %d\n", m, n, w, h);

    PetscViewer viewer;
    PetscViewerDrawOpen(PETSC_COMM_WORLD, nullptr, "", 300, 0, w, h, &viewer);
    PetscViewerDrawSetPause(viewer, -1);
    MatView(A, viewer);
    PetscViewerDestroy(&viewer);

    return 0;
}


int petsc_viewer(const saena_matrix *A){

    if(A->active) {
        MPI_Comm comm = A->comm;
        PETSC_COMM_WORLD = comm;
        PetscInitialize(nullptr, nullptr, nullptr, nullptr);
        Mat B;
        petsc_saena_matrix(A, B);
        petsc_viewer(B);
        MatDestroy(&B);
        PetscFinalize();
    }

    return 0;
}


int petsc_prolong_matrix(const prolong_matrix *P, Mat &B){

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

int petsc_viewer(const prolong_matrix *P){

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

int petsc_viewer(const restrict_matrix *R){

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

int petsc_restrict_matrix(const restrict_matrix *R, Mat &B){

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


int petsc_saena_matrix(const saena_matrix *A, Mat &B){

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
//    MatSetType(B, MATSEQAIJ);
//    MatSeqAIJSetPreallocation(B, 7, NULL);

    MatSetType(B, MATMPIAIJ); // Documentation: A matrix type to be used for parallel sparse matrices
    MatMPIAIJSetPreallocation(B, 0, &nnz_per_row_diag[0], 0, &nnz_per_row_off_diag[0]);

    for (nnz_t i = 0; i < A->nnz_l; ++i) {
//        if(rank == 1) printf("%6d\t%6d\t%6f\n", A->entry[i].row, A->entry[i].col, A->entry[i].val);
//        assert(A->entry[i].row >= A->split[rank] && A->entry[i].row < A->split[rank + 1]);
//        assert(A->entry[i].col >= 0 && A->entry[i].col < A->Mbig);
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

// from petsc-3.13.5/src/vec/vec/tutorials/ex9.c
int petsc_std_vector(const std::vector<value_t> &v1, Vec &x, const int &OFST, MPI_Comm comm){

    PetscErrorCode ierr;
    PetscMPIInt    rank;
    PetscInt       i,istart,iend,nlocal, iglobal;
    PetscScalar    *array;
    PetscViewer    viewer;

    const int SZ = v1.size();

    PETSC_COMM_WORLD = comm;
//    PetscInitialize(nullptr, nullptr, nullptr, nullptr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);

    /*
       Create a vector, specifying only its global dimension.
       When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
       the vector format (currently parallel or sequential) is
       determined at runtime.  Also, the parallel partitioning of
       the vector is determined by PETSc at runtime.
    */
    ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
    ierr = VecSetType(x, VECMPI);CHKERRQ(ierr);
    ierr = VecSetSizes(x, SZ, PETSC_DECIDE);CHKERRQ(ierr);
//    VecMPISetGhost(x, nghost,ifrom);
//    ierr = VecSetFromOptions(x);CHKERRQ(ierr);

    /*
       PETSc parallel vectors are partitioned by
       contiguous chunks of rows across the processors.  Determine
       which vector are locally owned.
    */
//    ierr = VecGetOwnershipRange(x,&istart,&iend);CHKERRQ(ierr);

    /* --------------------------------------------------------------------
       Set the vector elements.
        - Always specify global locations of vector entries.
        - Each processor can insert into any location, even ones it does not own
        - In this case each processor adds values to all the entries,
           this is not practical, but is merely done as an example
     */
    for (i=0; i<SZ; i++) {
        iglobal = i + OFST;
        ierr = VecSetValues(x, 1, &iglobal, &v1[i], INSERT_VALUES);CHKERRQ(ierr);
    }

    /*
       Assemble vector, using the 2-step process:
         VecAssemblyBegin(), VecAssemblyEnd()
       Computations can be done while messages are in transition
       by placing code between these two statements.
    */
    ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

//    VecView(x, PETSC_VIEWER_STDOUT_WORLD);

//    ierr = PetscFinalize();
    return 0;
}

int petsc_saena_vector(saena_vector *v, Vec &w){
    // NOTE: not tested
    int rank = 0;
    MPI_Comm_rank(v->comm, &rank);
    std::vector<value_t> vstd;
    v->get_vec(vstd);
    petsc_std_vector(vstd, w, v->split[rank], v->comm);
    return 0;
}


int petsc_solve(const saena_matrix *A1, const vector<value_t> &b1, vector<value_t> &x1, const double &rel_tol){

    Vec            x,b;      /* approx solution, RHS */
    Mat            A;        /* linear system matrix */
    KSP            ksp;      /* linear solver context */
    PetscReal      norm;     /* norm of solution error */
    PetscInt       its;
    PetscScalar    *array;
    PC             pc;

    MPI_Comm comm = A1->comm;
    PETSC_COMM_WORLD = comm;
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    petsc_saena_matrix(A1, A);
    petsc_std_vector(b1, b, A1->split[rank], comm);
    MatCreateVecs(A, &x, NULL);
    VecSet(x, 0.0);

//    petsc_viewer(A);
//    MatView(A, PETSC_VIEWER_STDOUT_WORLD);
//    VecView(b, PETSC_VIEWER_STDOUT_WORLD);

    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetTolerances(ksp, rel_tol, 1.e-50, PETSC_DEFAULT, 1000);

//    KSPGetPC(ksp, &pc);
//    PCSetType(pc, PCHMG);
//    PCHMGSetInnerPCType(pc, PCGAMG);
//    PCHMGSetReuseInterpolation(pc,PETSC_TRUE);
//    PCHMGSetUseSubspaceCoarsening(pc,PETSC_TRUE);
//    PCHMGUseMatMAIJ(pc,PETSC_FALSE);
//    PCHMGSetCoarseningComponent(pc,0);

    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCGAMG);

    KSPSetFromOptions(ksp);
    KSPSolve(ksp,b,x);

//    VecView(x, PETSC_VIEWER_STDOUT_WORLD);

    VecGetArray(x, &array);
    for (int i = 0; i < x1.size(); i++)
        x1[i] = array[i];
    VecRestoreArray(x, &array);

//    VecAXPY(x,-1.0,b); // should we do norm(A * x - b)?
//    VecNorm(b,NORM_2,&norm);
//    KSPGetIterationNumber(ksp,&its);
//    PetscPrintf(PETSC_COMM_WORLD,"PETSc: Norm of error %g, iterations %D\n",(double)norm,its);

    KSPDestroy(&ksp);
    VecDestroy(&x);
    VecDestroy(&b);
    MatDestroy(&A);

    PetscFinalize();
    return 0;
}

int petsc_solve(saena_matrix *A1, vector<value_t> &b1, vector<value_t> &x1, const double &rel_tol, const char in_str[], string pc_type){

    Vec            x,b;      /* approx solution, RHS */
    Mat            A;        /* linear system matrix */
    KSP            ksp;      /* linear solver context */
    PetscReal      norm;     /* norm of solution error */
    PetscInt       its;
    PetscScalar    *array;
    PC             pc;

    MPI_Comm comm = A1->comm;
    PETSC_COMM_WORLD = comm;
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);
	PetscLogDefaultBegin();
	CHKERRQ(PetscOptionsInsertString(nullptr, in_str));
	//CHKERRQ(PetscOptionsInsertString(nullptr,"-ksp_type cg -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 3 -pc_gamg_threshold 0.01 0.01 0.01 -pc_gamg_square_graph 0 -ksp_monitor_true_residual -ksp_max_it 500 -ksp_rtol 1e-6 -ksp_converged_reason"));
	//CHKERRQ(PetscOptionsInsertString(nullptr,"-ksp_type cg -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 3 -pc_gamg_threshold 0.01 0.01 0.01  -ksp_monitor_true_residual -ksp_max_it 500 -ksp_rtol 1e-6"));
	//CHKERRQ(PetscOptionsInsertString(nullptr,"-ksp_type gmres -pc_type ml -pc_ml_Threshold 0.01 -pc_ml_CoarsenScheme MIS -pc_ml_maxCoarseSize 1000 -ksp_monitor_true_residual -ksp_max_it 500 -ksp_rtol 1e-6"));
	//CHKERRQ(PetscOptionsInsertString(nullptr,"-ksp_type gmres -pc_type hypre -pc_hypre_type boomeramg -ksp_monitor_true_residual -ksp_max_it 500 -ksp_rtol 1e-6"));
	
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    petsc_saena_matrix(A1, A);
    petsc_std_vector(b1, b, A1->split[rank], comm);
	A1->erase();
	if (!rank) std::cout << "destroy saena matrix" << std::endl;
    MatCreateVecs(A, &x, NULL);
    VecSet(x, 0.0);

//    petsc_viewer(A);
//    MatView(A, PETSC_VIEWER_STDOUT_WORLD);
//    VecView(b, PETSC_VIEWER_STDOUT_WORLD);

    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    //KSPSetTolerances(ksp, rel_tol, 1.e-6, PETSC_DEFAULT, 1000);

	// set ksp monitor
	/*PetscViewerAndFormat *vf;
	PetscViewerAndFormatCreate(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_DEFAULT,&vf);	
	KSPMonitorSet(ksp,(PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*))KSPMonitorDefault,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);
	KSPMonitorSet(ksp,(PetscErrorCode (*)(KSP,PetscInt,PetscReal,void*))KSPMonitorTrueResidualNorm,vf,(PetscErrorCode (*)(void**))PetscViewerAndFormatDestroy);*/

	// set ksp and pc type
	//KSPSetType(ksp, KSPGMRES);
    //KSPGetPC(ksp, &pc);

	// using petsc AMG
    /*PCSetType(pc, PCGAMG);
	PCGAMGSetType(pc, PCGAMGAGG);
	PCGAMGSetNSmooths(pc,3);
	PetscReal v[3] = {0.01, 0.01, 0.01};
	PCGAMGSetThreshold(pc, v, 3); 
	PCGAMGSetSquareGraph(pc,0);*/

	// using hypre
	//PCFactorSetShiftType(pc, MAT_SHIFT_POSITIVE_DEFINITE);
    //PCSetType(pc, PCHYPRE);
	//PCHYPRESetType(pc, "boomeramg");

	// using ml
	//PCFactorSetShiftType(pc, MAT_SHIFT_POSITIVE_DEFINITE);
    //PCSetType(pc, PCML);

    KSPSetFromOptions(ksp);

	// for nano case	
	if (pc_type == "gamg")
	{
    	KSPGetPC(ksp, &pc);
		if (!rank) cout << "set for additional gamg option" << endl;
		PCGAMGSetNlevels(pc, 6);
		double v[5] = {0.03, 0.03, 0.03, 0.03, 0.03};
		PCGAMGSetThreshold(pc, v, 5);
	}

	//PetscInt nn;
	//PCMGSetLevels(pc, 3, &comm);
	//std::cout << "levels: " << nn << std::endl;
    //KSPGetPC(ksp, &pc);
	//PCFactorSetShiftType(pc, MAT_SHIFT_POSITIVE_DEFINITE);
	
	//KSPType ksptype;
	//KSPGetType(ksp,&ksptype);
	//PetscPrintf(PETSC_COMM_WORLD,"KSPType: %s\n", ksptype);

	if (!rank) std::cout << "ksp solve" << std::endl;
	//double ksp_solve_t1 = omp_get_wtime();
    KSPSolve(ksp,b,x);
	//double ksp_solve_t2 = omp_get_wtime();
	//if (!rank) std::cout << "ksp solve time = " << ksp_solve_t2 - ksp_solve_t1 << std::endl;
	//KSPConvergedReason reason;
	//KSPGetConvergedReason(ksp,&reason);
	//PetscPrintf(PETSC_COMM_WORLD,"KSPConvergedReason: %D\n", reason);
//    VecView(x, PETSC_VIEWER_STDOUT_WORLD);

    VecGetArray(x, &array);
    for (int i = 0; i < x1.size(); i++)
        x1[i] = array[i];
    VecRestoreArray(x, &array);

    VecAXPY(x,-1.0,b);
    VecNorm(x,NORM_2,&norm);
    KSPGetIterationNumber(ksp,&its);
    PetscPrintf(PETSC_COMM_WORLD,"PETSc: Norm of error %g, iterations %D\n",(double)norm,its);

    KSPDestroy(&ksp);
    VecDestroy(&x);
    VecDestroy(&b);
    MatDestroy(&A);

    PetscFinalize();
    return 0;
}
#endif //_USE_PETSC_
