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


int petsc_write_mat_file(const saena_matrix *A1){
    // write matrix A to file in Matrix Market format
    Mat A;
    MPI_Comm comm = A1->comm;
    PETSC_COMM_WORLD = comm;
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);
    petsc_saena_matrix(A1, A);
    PetscViewer viewer;
    PetscViewerASCIIOpen(comm, "Amat.mtx", &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_MATRIXMARKET);
    MatView(A, viewer);
    PetscViewerPopFormat(viewer);
    PetscViewerDestroy(&viewer);
    MatDestroy(&A);
    PetscFinalize();
    return 0;
}


int petsc_viewer(const Mat &A){

    int sz = 1800;

    pindex_t m, n;
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

    if(!rank) printf("petsc_prolong_matrix: P.entry is cleared. This func. needs to be updated!\n");
    if(!rank) printf("petsc_prolong_matrix: P.split is cleared. This func. needs to be updated!\n");
    return 0;

    std::vector<pindex_t> nnz_per_row_diag(P->M, 0);
    for(nnz_t i = 0; i < P->nnz_l_local; ++i){
        ++nnz_per_row_diag[P->row_local[i]];
    }

    std::vector<pindex_t> nnz_per_row_off_diag(P->M, 0);
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
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    if(!rank) printf("petsc_restrict_matrix: R.row_local is deleted. This func. needs to be updated!\n");
    if(!rank) printf("petsc_restrict_matrix: R.splitNew is cleared. This func. needs to be updated!\n");
    return 0;

    std::vector<pindex_t> nnz_per_row_diag(R->M, 0);
    for(nnz_t i = 0; i < R->nnz_l_local; i++){
//        nnz_per_row_diag[R->row_local[i]]++;
    }

    std::vector<pindex_t> nnz_per_row_off_diag(R->M, 0);
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

    // todo: R->splitNew is cleared. update this!
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
    const pindex_t sz = A->M;

    std::vector<pindex_t> nnz_per_row_diag(sz, 0);
    const index_t iendl = A->nnz_l_local;
    for(nnz_t i = 0; i < iendl; ++i){
        ++nnz_per_row_diag[A->row_local[i]];
    }

    const index_t iendr = A->nnz_l_remote;
    std::vector<pindex_t> nnz_per_row_off_diag(sz, 0);
    for(nnz_t i = 0; i < iendr; ++i){
        ++nnz_per_row_off_diag[A->row_remote[i]];
    }

    MatCreate(comm, &B);
    MatSetSizes(B, sz, sz, A->Mbig, A->Mbig);

    // for serial
//    MatSetType(B, MATSEQAIJ);
//    MatSeqAIJSetPreallocation(B, 7, NULL);

    MatSetType(B, MATMPIAIJ); // Documentation: A matrix type to be used for parallel sparse matrices
    MatMPIAIJSetPreallocation(B, 0, &nnz_per_row_diag[0], 0, &nnz_per_row_off_diag[0]);

    const index_t iend = A->nnz_l;
    for (nnz_t i = 0; i < iend; ++i) {
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

int petsc_std_vector(value_t *&v1, Vec &x, const int SZ, const int &OFST, MPI_Comm comm){

    PetscErrorCode ierr;
    PetscMPIInt    rank;
    PetscInt       i,istart,iend,nlocal, iglobal;
    PetscScalar    *array;
    PetscViewer    viewer;

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
    value_t *vstd = nullptr;
    v->get_vec(vstd);
    const int SZ = v->get_size();
    petsc_std_vector(vstd, w, SZ, v->split[rank], v->comm);
    saena_free(vstd);
    return 0;
}


int petsc_solve(saena_matrix *A1, value_t *&b1, value_t *&x1, const double &rel_tol, const string &pc_type){
    Vec x,b;      /* approx solution, RHS */
    Mat A;        /* linear system matrix */

    MPI_Comm comm = A1->comm;
    PETSC_COMM_WORLD = comm;
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);
    PetscLogDefaultBegin();
//    CHKERRQ(PetscOptionsInsertString(nullptr, in_str));

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    const index_t sz = A1->M;
    petsc_saena_matrix(A1, A);
    petsc_std_vector(b1, b, sz, A1->split[rank], comm);
    A1->erase();
    if (!rank) std::cout << "destroy saena matrix" << std::endl;
    MatCreateVecs(A, &x, NULL);
    VecSet(x, 0.0);

    if (!rank) print_sep();
    petsc_solve(A, b, x, rel_tol, pc_type);
    if (!rank) print_sep();

    // extract the solution
    x1 = new value_t[sz];
    value_t *array = nullptr;
    VecGetArray(x, &array);
    for (int i = 0; i < sz; ++i){
        x1[i] = array[i];
    }
    VecRestoreArray(x, &array);

    VecDestroy(&x);
    VecDestroy(&b);
    MatDestroy(&A);
    PetscFinalize();
    return 0;
}

int petsc_solve_all(saena_matrix *A1, value_t *&b1, value_t *&x1, const double &rel_tol){
    Vec            x,b;      /* approx solution, RHS */
    Mat            A;        /* linear system matrix */

    MPI_Comm comm = A1->comm;
    PETSC_COMM_WORLD = comm;
    PetscInitialize(nullptr, nullptr, nullptr, nullptr);
    PetscLogDefaultBegin();
//    CHKERRQ(PetscOptionsInsertString(nullptr, in_str));

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    const index_t sz = A1->M;
    petsc_saena_matrix(A1, A);
    petsc_std_vector(b1, b, sz, A1->split[rank], comm);
    A1->erase();
    if (!rank) std::cout << "destroy saena matrix" << std::endl;
    MatCreateVecs(A, &x, NULL);
    VecSet(x, 0.0);

    if (!rank) print_sep();
    petsc_solve(A, b, x, rel_tol, "gamg");
    if (!rank) print_3sep();
    petsc_solve(A, b, x, rel_tol, "ml");
    if (!rank) print_3sep();
    petsc_solve(A, b, x, rel_tol, "boomerAMG");
    if (!rank) print_sep();

    VecDestroy(&x);
    VecDestroy(&b);
    MatDestroy(&A);
    PetscFinalize();
    return 0;
}

int petsc_solve(Mat &A, Vec &b, Vec &x, const double &rel_tol, const string &petsc_solver){
    KSP            ksp;      /* linear solver context */
    PetscLogEvent  SETUP,SOLVE;

//    MPI_Comm comm = A->comm;
//    PETSC_COMM_WORLD = comm;
//    PetscInitialize(nullptr, nullptr, nullptr, nullptr);
//    PetscLogDefaultBegin();

    string opts = return_petsc_opts(petsc_solver);
    CHKERRQ(PetscOptionsInsertString(nullptr, opts.c_str()));

    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetTolerances(ksp, rel_tol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    KSPSetFromOptions(ksp);
//    if (!rank) std::cout << "ksp setup" << std::endl;

    MPI_Barrier(PETSC_COMM_WORLD);
    double t1 = omp_get_wtime();

    string event = petsc_solver + " setup";
    PetscLogEventRegister(event.c_str(),0,&SETUP);
    PetscLogEventBegin(SETUP,0,0,0,0);
    KSPSetUp(ksp);
    PetscLogEventEnd(SETUP,0,0,0,0);

    double t2 = omp_get_wtime();

//    if (!rank) std::cout << "ksp solve" << std::endl;
    event = petsc_solver + " solve";
    PetscLogEventRegister(event.c_str(),0,&SOLVE);
    const int solve_iter = 10;

    MPI_Barrier(PETSC_COMM_WORLD);
    double t3 = omp_get_wtime();

    for(int i = 0; i < solve_iter; ++i){
        PetscLogEventBegin(SOLVE,0,0,0,0);
        KSPSolve(ksp,b,x);
        PetscLogEventEnd(SOLVE,0,0,0,0);
    }

    double t4 = omp_get_wtime();

//    VecGetArray(x, &array);
//    for (int i = 0; i < sz; ++i)
//        x1[i] = array[i];
//    VecRestoreArray(x, &array);

    // compute the norm of error
    PetscReal      norm, normb;     /* norm of solution error */
    PetscInt       its;
    PetscScalar    *array;
    Vec            Ax;

    MatCreateVecs(A, NULL, &Ax);
    MatMult(A,x,Ax);
    VecAXPY(Ax,-1.0,b);
    VecNorm(Ax,NORM_2,&norm);
    VecNorm(b,NORM_2,&normb);
    KSPGetIterationNumber(ksp,&its);
    PetscPrintf(PETSC_COMM_WORLD,"PETSc: Norm of error %g, iterations %D\n",(double)norm,its); // ||Ax - b||
    PetscPrintf(PETSC_COMM_WORLD,"PETSc: Relative Norm of error %g, iterations %D\n",(double)norm / normb,its); // ||Ax - b|| / ||b||

    KSPDestroy(&ksp);

    print_time(t1, t2, "PETSc Setup:", MPI_COMM_WORLD);
    print_time(t3 / solve_iter, t4 / solve_iter, "PETSc Solve:", MPI_COMM_WORLD);
    PetscLogView(PETSC_VIEWER_STDOUT_WORLD);
    return 0;
}


int petsc_solve_old1(const saena_matrix *A1, const vector<value_t> &b1, vector<value_t> &x1, const double &rel_tol){

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

int petsc_solve_old2(saena_matrix *A1, vector<value_t> &b1, vector<value_t> &x1, const double &rel_tol, const char in_str[], string pc_type){

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
    /*if (pc_type == "gamg")
    {
        KSPGetPC(ksp, &pc);
        if (!rank) cout << "set for additional gamg option" << endl;
        PCGAMGSetNlevels(pc, 6);
        double v[5] = {0.03, 0.03, 0.03, 0.03, 0.03};
        PCGAMGSetThreshold(pc, v, 5);
    }*/

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

int petsc_solve_old3(saena_matrix *A1, value_t *&b1, value_t *&x1, const double &rel_tol, const char in_str[]){

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

    const index_t sz = A1->M;
    petsc_saena_matrix(A1, A);
    petsc_std_vector(b1, b, sz, A1->split[rank], comm);
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
    /*if (pc_type == "gamg")
    {
        KSPGetPC(ksp, &pc);
        if (!rank) cout << "set for additional gamg option" << endl;
        PCGAMGSetNlevels(pc, 6);
        double v[5] = {0.03, 0.03, 0.03, 0.03, 0.03};
        PCGAMGSetThreshold(pc, v, 5);
    }*/

    //PetscInt nn;
    //PCMGSetLevels(pc, 3, &comm);
    //std::cout << "levels: " << nn << std::endl;
    //KSPGetPC(ksp, &pc);
    //PCFactorSetShiftType(pc, MAT_SHIFT_POSITIVE_DEFINITE);

    //KSPType ksptype;
    //KSPGetType(ksp,&ksptype);
    //PetscPrintf(PETSC_COMM_WORLD,"KSPType: %s\n", ksptype);

    if (!rank) std::cout << "ksp setup" << std::endl;
    KSPSetUp(ksp);

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
    for (int i = 0; i < sz; ++i)
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


string return_petsc_opts(const string &petsc_solver){
    string opts;
    if(petsc_solver == "gamg"){ // info at the bottom
        opts = "-ksp_type cg -pc_type gamg -pc_gamg_type agg -pc_gamg_agg_nsmooths 1"
               " -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -mg_levels_ksp_max_it 3"
               " -pc_gamg_threshold 0.01 -pc_gamg_sym_graph false -pc_gamg_square_graph 0 -pc_gamg_coarse_eq_limit 100"
               " -ksp_monitor_true_residual -ksp_norm_type unpreconditioned -ksp_max_it 500"
               " -ksp_converged_reason -ksp_view";
    } else if(petsc_solver == "ml"){
        opts = "-ksp_type cg -pc_type ml"
               " -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -mg_levels_ksp_max_it 3"
               " -pc_ml_maxCoarseSize 100 -pc_ml_CoarsenScheme MIS"
               " -ksp_monitor_true_residual -ksp_norm_type unpreconditioned -ksp_max_it 500"
               " -ksp_converged_reason -ksp_view";
//               " -pc_ml_maxNlevels 10 -pc_ml_Threshold 0.0"
    } else if(petsc_solver == "boomerAMG"){
        opts = "-ksp_type cg -pc_type hypre -pc_hypre_type boomeramg"
               " -pc_hypre_boomeramg_relax_type_all Chebyshev -pc_hypre_boomeramg_grid_sweeps_all 3"
               " -ksp_monitor_true_residual -ksp_norm_type unpreconditioned -ksp_max_it 500"
               " -ksp_converged_reason -ksp_view";
//               " -pc_hypre_boomeramg_strong_threshold 0.25 -pc_hypre_boomeramg_coarsen_type Falgout"
//               " -pc_hypre_boomeramg_max_levels 6"
//               " -pc_hypre_boomeramg_print_statistics"
//               " -pc_hypre_boomeramg_agg_nl 1 -pc_hypre_boomeramg_agg_num_paths 4"
    } else if(petsc_solver == "dcg"){
        opts = "-ksp_type cg -pc_type jacobi"
               " -ksp_monitor_true_residual -ksp_norm_type unpreconditioned -ksp_max_it 500"
               " -ksp_converged_reason -ksp_view -log_view";
    }else{
        printf("invalid petsc_solver!\n");
    }
    return opts;
}

/*
 * general info for petsc options:
    ksp_norm_type (KSPSetNormType): Sets the norm that is used for convergence testing.
            KSP_NORM_NONE - skips computing the norm
            KSP_NORM_PRECONDITIONED - the default for left preconditioned solves, uses the l2 norm
                of the preconditioned residual P^{-1}(b - A x)
            KSP_NORM_UNPRECONDITIONED - uses the l2 norm of the true b - Ax residual.
            KSP_NORM_NATURAL - supported  by KSPCG, KSPCR, KSPCGNE, KSPCGS
        from user's guide: For the conjugate gradient, Richardson, and Chebyshev methods the
                           true residual can be used by the options database command ksp_norm_type unpreconditioned.
 */

/*  info for gamg:
    from: the PETSc code gamg.c:
    PCGAMG - Geometric algebraic multigrid (AMG) preconditioner
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

    pc_gamg_threshold: Relative threshold to use for dropping edges in aggregation graph
        Increasing the threshold decreases the rate of coarsening. Conversely reducing the threshold increases the rate of coarsening (aggressive coarsening) and thereby reduces the complexity of the coarse grids, and generally results in slower solver converge rates. Reducing coarse grid complexity reduced the complexity of Galerkin coarse grid construction considerably.
        Before coarsening or aggregating the graph, GAMG removes small values from the graph with this threshold, and thus reducing the coupling in the graph and a different (perhaps better) coarser set of points.
        0.0 means keep all nonzero entries in the graph; negative means keep even zero entries in the graph.
 */


/*
info for ml:
from here: https://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/PC/PCML.html

Multigrid options(inherited)
-pc_mg_cycles <1>	- 1 for V cycle, 2 for W-cycle (MGSetCycles)
-pc_mg_distinct_smoothup	- Should one configure the up and down smoothers separately (PCMGSetDistinctSmoothUp)
-pc_mg_type <multiplicative>	- (one of) additive multiplicative full kascade

ML options
-pc_ml_PrintLevel <0>	- Print level (ML_Set_PrintLevel)
-pc_ml_maxNlevels <10>	- Maximum number of levels (None)
-pc_ml_maxCoarseSize <1>	- Maximum coarsest mesh size (ML_Aggregate_Set_MaxCoarseSize)
-pc_ml_CoarsenScheme <Uncoupled>	- (one of) Uncoupled Coupled MIS METIS
-pc_ml_DampingFactor <1.33333>	- P damping factor (ML_Aggregate_Set_DampingFactor)
-pc_ml_Threshold <0>	- Smoother drop tol (ML_Aggregate_Set_Threshold)
-pc_ml_SpectralNormScheme_Anorm <false>	- Method used for estimating spectral radius (ML_Set_SpectralNormScheme_Anorm)
-pc_ml_repartition <false>	- Allow ML to repartition levels of the heirarchy (ML_Repartition_Activate)
-pc_ml_repartitionMaxMinRatio <1.3>	- Acceptable ratio of repartitioned sizes (ML_Repartition_Set_LargestMinMaxRatio)
-pc_ml_repartitionMinPerProc <512>: Smallest repartitioned size (ML_Repartition_Set_MinPerProc)
-pc_ml_repartitionPutOnSingleProc <5000>	- Problem size automatically repartitioned to one processor (ML_Repartition_Set_PutOnSingleProc)
-pc_ml_repartitionType <Zoltan>	- Repartitioning library to use (ML_Repartition_Set_Partitioner)
-pc_ml_repartitionZoltanScheme <RCB>	- Repartitioning scheme to use (None)
-pc_ml_Aux <false>	- Aggregate using auxiliary coordinate-based laplacian (None)
-pc_ml_AuxThreshold <0.0>	- Auxiliary smoother drop tol (None)
*/


/*
info for hypre:
from here: https://mooseframework.inl.gov/application_development/hypre.html

HYPRE preconditioner options
  -pc_hypre_type <boomeramg> (choose one of) pilut parasails boomeramg ams (PCHYPRESetType)
HYPRE BoomerAMG Options
  -pc_hypre_boomeramg_cycle_type <V> (choose one of) V W (None)
  -pc_hypre_boomeramg_max_levels <25>: Number of levels (of grids) allowed (None)
  -pc_hypre_boomeramg_max_iter <1>: Maximum iterations used PER hypre call (None)
  -pc_hypre_boomeramg_tol <0.>: Convergence tolerance PER hypre call (0.0 = use a fixed number of iterations) (None)
  -pc_hypre_boomeramg_truncfactor <0.>: Truncation factor for interpolation (0=no truncation) (None)
  -pc_hypre_boomeramg_P_max <0>: Max elements per row for interpolation operator (0=unlimited) (None)
  -pc_hypre_boomeramg_agg_nl <0>: Number of levels of aggressive coarsening (None)
  -pc_hypre_boomeramg_agg_num_paths <1>: Number of paths for aggressive coarsening (None)
  -pc_hypre_boomeramg_strong_threshold <0.25>: Threshold for being strongly connected (None)
  -pc_hypre_boomeramg_max_row_sum <0.9>: Maximum row sum (None)
  -pc_hypre_boomeramg_grid_sweeps_all <1>: Number of sweeps for the up and down grid levels (None)
  -pc_hypre_boomeramg_nodal_coarsen <0>: Use a nodal based coarsening 1-6 (HYPRE_BoomerAMGSetNodal)
  -pc_hypre_boomeramg_vec_interp_variant <0>: Variant of algorithm 1-3 (HYPRE_BoomerAMGSetInterpVecVariant)
  -pc_hypre_boomeramg_grid_sweeps_down <1>: Number of sweeps for the down cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_up <1>: Number of sweeps for the up cycles (None)
  -pc_hypre_boomeramg_grid_sweeps_coarse <1>: Number of sweeps for the coarse level (None)
  -pc_hypre_boomeramg_smooth_type <Schwarz-smoothers> (choose one of) Schwarz-smoothers Pilut ParaSails Euclid (None)
  -pc_hypre_boomeramg_smooth_num_levels <25>: Number of levels on which more complex smoothers are used (None)
  -pc_hypre_boomeramg_eu_level <0>: Number of levels for ILU(k) in Euclid smoother (None)
  -pc_hypre_boomeramg_eu_droptolerance <0.>: Drop tolerance for ILU(k) in Euclid smoother (None)
  -pc_hypre_boomeramg_eu_bj: <FALSE> Use Block Jacobi for ILU in Euclid smoother? (None)
  -pc_hypre_boomeramg_relax_type_all <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_down <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_up <symmetric-SOR/Jacobi> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_type_coarse <Gaussian-elimination> (choose one of) Jacobi sequential-Gauss-Seidel seqboundary-Gauss-Seidel SOR/Jacobi backward-SOR/Jacobi  symmetric-SOR/Jacobi  l1scaled-SOR/Jacobi Gaussian-elimination      CG Chebyshev FCF-Jacobi l1scaled-Jacobi (None)
  -pc_hypre_boomeramg_relax_weight_all <1.>: Relaxation weight for all levels (0 = hypre estimates, -k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_relax_weight_level <1.>: Set the relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_outer_relax_weight_all <1.>: Outer relaxation weight for all levels (-k = determined with k CG steps) (None)
  -pc_hypre_boomeramg_outer_relax_weight_level <1.>: Set the outer relaxation weight for a particular level (weight,level) (None)
  -pc_hypre_boomeramg_no_CF: <FALSE> Do not use CF-relaxation (None)
  -pc_hypre_boomeramg_measure_type <local> (choose one of) local global (None)
  -pc_hypre_boomeramg_coarsen_type <Falgout> (choose one of) CLJP Ruge-Stueben  modifiedRuge-Stueben   Falgout  PMIS  HMIS (None)
  -pc_hypre_boomeramg_interp_type <classical> (choose one of) classical   direct multipass multipass-wts ext+i ext+i-cc standard standard-wts   FF FF1 (None)
  -pc_hypre_boomeramg_print_statistics: Print statistics (None)
  -pc_hypre_boomeramg_print_statistics <3>: Print statistics (None)
  -pc_hypre_boomeramg_print_debug: Print debug information (None)
  -pc_hypre_boomeramg_nodal_relaxation: <FALSE> Nodal relaxation via Schwarz (None)


Agressive Coarsening
Another option that can do a lot of coarsening is "Aggressive Coarsening". BoomerAMG actually has many parameters surrounding this - but currently only 2 are available to us as PETSc options: -pc_hypre_boomeramg_agg_nl and -pc_hypre_boomeramg_agg_num_paths.

-pc_hypre_boomeramg_agg_nl is the number of coarsening levels to apply "aggressive coarsening" to. Aggressive coarsening does just what you think it does: it tries even harder to remove matrix entries. The way it does this is looking at "second-order" connections: does there exist a path from one important entry to another important entry through several other entries. By looking at these pathways the algorithm will decide whether or not to keep an entry. Doing more aggressive coarsening will result in less time spent in BoomerAMG (and a lot less communication done) but will also impact the effectiveness of the preconditioner by quite a lot - so it's a balance.

-pc_hypre_boomeramg_agg_num_paths is the number of pathways to consider to find a connection and keep something. That means increasing this value will _reduce_ the ammount of aggressive coarsening happening in each aggressive coarsening level. What this means is that a higher -pc_hypre_boomeramg_agg_num_paths will improve accuracy/effectiveness but slow things down. So it's a balance.

By default aggressive coarsening is off (-pc_hypre_boomeramg_agg_nl 0), so to turn it on set -pc_hypre_boomeramg_agg_nl to something higher than zero. I recommend 2 or 3 to start with, but even 4 can be ok in 3D. -pc_hypre_boomeramg_agg_num_paths defaults to 1: which is the most aggressive setting. If the aggressive coarsening levels are causing too many linear iterations, try increasing the number of paths _first_. Go up to about 4,5 or 6 and see if it helps reduce the number of linear iterations. If it doesn't, then you may need to back off on the number of aggressive coarsening levels you are doing.
*/

/*		gamg_option =  "-ksp_type cg -pc_type gamg"
						" -pc_gamg_type agg -pc_gamg_agg_nsmooths 1"
					    " -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -mg_levels_ksp_max_it 2"
						" -pc_gamg_threshold 0.015 -pc_gamg_sym_graph false -pc_gamg_square_graph 0"
						" -pc_gamg_coarse_eq_limit 500 -pc_gamg_sym_graph false -pc_gamg_square_graph 2"
						" -ksp_monitor_true_residual -ksp_norm_type unpreconditioned -ksp_max_it 2000 -ksp_rtol 1e-6 -ksp_converged_reason -ksp_view -log_view";
*/

/*		ml_option =  "-ksp_type cg -pc_type ml"
						" -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -mg_levels_ksp_max_it 2"
						" -pc_ml_maxNlevels 7"
						" -pc_ml_Threshold 0.125 -pc_ml_CoarsenScheme MIS -pc_ml_maxCoarseSize 1000"
						" -ksp_monitor_true_residual -ksp_norm_type unpreconditioned -ksp_max_it 2000 -ksp_rtol 1e-6 -ksp_converged_reason -ksp_view -log_view";
*/

/*		hypre_option = 	"-ksp_type cg -pc_type hypre -pc_hypre_type boomeramg"
       " -pc_hypre_boomeramg_max_levels 7 -pc_hypre_boomeramg_relax_type_all Chebyshev -pc_hypre_boomeramg_grid_sweeps_all 2"
       " -pc_hypre_boomeramg_strong_threshold 0.11 -pc_hypre_boomeramg_coarsen_type Falgout"
       " -pc_hypre_boomeramg_agg_nl 3 -pc_hypre_boomeramg_agg_num_paths 3 -pc_hypre_boomeramg_truncfactor 0"
       " -ksp_monitor_true_residual -ksp_norm_type unpreconditioned -ksp_max_it 2000 -ksp_rtol 1e-6 -ksp_converged_reason -ksp_view"
       " -pc_hypre_boomeramg_print_statistics -log_view";
       //" -pc_hypre_boomeramg_print_debug";// -log_view";
*/

#endif //_USE_PETSC_
