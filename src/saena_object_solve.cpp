#include <cmath>
#include "superlu_ddefs.h"

#include "saena_object.h"
#include "saena_matrix.h"
#include "aux_functions.h"
#include "petsc_functions.h"

// uncomment to print info for the lazy update feature
// use this to store number of iterations for the lazy-update experiment.
//std::vector<int> iter_num_lazy;


int saena_object::solve_coarsest_CG(saena_matrix* A, std::vector<value_t>& u, std::vector<value_t>& rhs){
    // this is CG.
    // u is zero in the beginning. At the end, it is the solution.

    MPI_Comm comm = A->comm;
    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        MPI_Barrier(comm);
        if(rank==0) printf("start of solve_coarsest_CG()\n");
        MPI_Barrier(comm);
    }
#endif

    // since u is zero, res = -rhs, and the residual in this function is the negative of what I have in this library.
    std::vector<value_t> res = rhs;

    double initial_dot = 0.0;
    dotProduct(res, res, &initial_dot, comm);
//    if(rank==0) std::cout << "\nsolveCoarsest: initial norm(res) = " << sqrt(initial_dot) << std::endl;

    double thres = initial_dot * CG_coarsest_tol * CG_coarsest_tol;

    double dot = initial_dot;
    int max_iter = CG_coarsest_max_iter;
    if (dot < CG_coarsest_tol*CG_coarsest_tol)
        max_iter = 0;

    std::vector<value_t> dir(A->M);
    dir = res;

//    double dot2;
    std::vector<value_t> res2(A->M);

    double factor = 0.0, dot_prev = 0.0;
    std::vector<value_t> matvecTemp(A->M);
    int i = 1;
    while (i < max_iter) {
//        if(rank==0) std::cout << "starting iteration of CG = " << i << std::endl;
        // factor = sq_norm/ (dir' * A * dir)
        A->matvec(dir, matvecTemp);

        dotProduct(dir, matvecTemp, &factor, comm);
        factor = dot / factor;
//        if(rank==1) std::cout << "\nsolveCoarsest: factor = " << factor << std::endl;

        #pragma omp parallel for
        for(index_t j = 0; j < A->M; ++j){
            u[j]   += factor * dir[j];
            res[j] -= factor * matvecTemp[j];
        }

        dot_prev = dot;
        dotProduct(res, res, &dot, comm);
//        if(rank==0) std::cout << "absolute norm(res) = " << sqrt(dot) << "\t( r_i / r_0 ) = " << sqrt(dot)/initialNorm << "  \t( r_i / r_i-1 ) = " << sqrt(dot)/sqrt(dot_prev) << std::endl;
//        if(rank==0) std::cout << sqrt(dot)/initialNorm << std::endl;

#ifdef __DEBUG1__
        if(verbose_solve_coarse) {
            MPI_Barrier(comm);
            if(rank==0) printf("sqrt(dot)/sqrt(init_dot) = %.14f\tCG_tol = %.2e\n", sqrt(dot/initial_dot), CG_coarsest_tol);
            MPI_Barrier(comm);
        }
#endif

//        A->residual(u, rhs, res2);
//        dotProduct(res2, res2, &dot2, comm);
//        if(rank==0) std::cout << "norm(res) = " << sqrt(dot2) << std::endl;

        if (dot < thres) break;

        factor = dot / dot_prev;
//        if(rank==1) std::cout << "\nsolveCoarsest: update factor = " << factor << std::endl;

        // update direction
        #pragma omp parallel for
        for(index_t j = 0; j < A->M; j++)
            dir[j] = res[j] + factor * dir[j];

        i++;
    }

    if(i == max_iter && max_iter != 0)
        i--;

//    print_vector(u, -1, "u at the end of CG", comm);

    if(verbose_solve_coarse && rank==0) printf("end of solve_coarsest! it took CG iterations = %d\n \n", i);
//    if(rank==0) printf("end of solve_coarsest! it took CG iterations = %d \n\n", i);

    return 0;
}


int saena_object::setup_SuperLU() {
// Read "4.2.2 Distributed Input" of the SuperLU's User's Guide. From that section:
//    typedef struct {
//        int nnz_loc;      number of nonzeros in the local submatrix
//        int m_loc;        number of rows local to this process
//        int fst_row;      row number of the first row in the local submatrix
//        void *nzval;      pointer to array of nonzero values, packed by row
//        int *rowptr;      pointer to array of beginning of rows in nzval[] and colind[]
//        int *colind;      pointer to array of column indices of the nonzeros
//    } NRformat_loc;

//    saena_matrix *A_coarsest = &grids.back().Ac;

//    MPI_Barrier(A_coarsest->comm);
//    printf("A_coarsest->print_entry\n");
//    A_coarsest->print_entry(-1);
//    MPI_Barrier(A_coarsest->comm);

    MPI_Comm *comm_coarsest = &A_coarsest->comm;
    int nprocs_coarsest, rank_coarsest;
    MPI_Comm_size(*comm_coarsest, &nprocs_coarsest);
    MPI_Comm_rank(*comm_coarsest, &rank_coarsest);

    superlu_allocated = true;

    int m, n, m_loc, nnz_loc;
    int nprow, npcol;
    int iam, ldb;

    nprow = nprocs_coarsest; // Default process rows.
    npcol = 1;               // Default process columns.
//    nrhs  = 1;             // Number of right-hand side.

#if 0
    // ------------------------------------------------------------
    //   INITIALIZE MPI ENVIRONMENT.
    // ------------------------------------------------------------
    MPI_Init( &argc, &argv );

    char* file_name(argv[5]);
    saena::matrix A_saena (file_name, comm);
    A_saena.assemble();
    A_saena.print_entry(-1);
    if(rank==0) printf("after matrix assemble.\n");

    // Parse command line argv[].
    for (cpp = argv+1; *cpp; ++cpp) {
        if ( **cpp == '-' ) {
            c = *(*cpp+1);
            ++cpp;
            switch (c) {
                case 'h':
                    printf("Options:\n");
                    printf("\t-r <int>: process rows    (default %4d)\n", nprow);
                    printf("\t-c <int>: process columns (default %4d)\n", npcol);
                    exit(0);
                    break;
                case 'r': nprow = atoi(*cpp);
                    break;
                case 'c': npcol = atoi(*cpp);
                    break;
            }
        } else { // Last arg is considered a filename
//            if ( !(fp = fopen(*cpp, "r")) ) {
//                ABORT("File does not exist");
//            }

            saena::matrix A_saena (*cpp, comm);
            A_saena.assemble();
            A_saena.print_entry(-1);
            if(rank==0) printf("after matrix assemble.\n");
            break;
        }
    }
#endif

    // ------------------------------------------------------------
    //   INITIALIZE THE SUPERLU PROCESS GRID.
    // ------------------------------------------------------------

#ifdef __DEBUG1__
    if (verbose_setup_coarse) {
        MPI_Barrier(*comm_coarsest);
        if (rank_coarsest == 0) {
            printf("setup_SuperLU: start. \n");
            printf("INITIALIZE THE SUPERLU PROCESS GRID. \n");
        }
        MPI_Barrier(*comm_coarsest);
    }
#endif

    superlu_gridinit(*comm_coarsest, nprow, npcol, &superlu_grid);

    // Bail out if I do not belong in the grid.
    iam = superlu_grid.iam; // my process rank in this group
//    printf("iam = %d, nprow = %d, npcol = %d \n", iam, nprow, npcol);

    // tag for quitting processors that don't belong to the SuperLU's process grid.
    if ( iam >= nprow * npcol ){
        superlu_active = false;
        superlu_gridexit(&superlu_grid);
        return 0;
    }

#ifdef __DEBUG1__
    if (verbose_setup_coarse) {
        MPI_Barrier(*comm_coarsest);
        if (!iam) {
            int v_major, v_minor, v_bugfix;
            superlu_dist_GetVersionNumber(&v_major, &v_minor, &v_bugfix);
            printf("Library version:\t%d.%d.%d\n", v_major, v_minor, v_bugfix);
//            printf("Input matrix file:\t%s\n", *cpp);
            printf("Process grid:\t\t%d X %d\n", nprow, npcol);
            fflush(stdout);
        }
        MPI_Barrier(*comm_coarsest);
    }
#endif

#if (VAMPIR >= 1)
    VT_traceoff();
#endif

#if (DEBUGlevel >= 1)
    CHECK_MALLOC(iam, "Enter main()");
#endif

    // ------------------------------------------------------------
    //   PASS THE MATRIX FROM SAENA
    // ------------------------------------------------------------

    // Set up the local A_SLU in NR_loc format
//    dCreate_CompRowLoc_Matrix_dist(A_SLU, m, n, nnz_loc, m_loc, fst_row,
//                                   nzval_loc, colind, rowptr,
//                                   SLU_NR_loc, SLU_D, SLU_GE);

#ifdef __DEBUG1__
    if (verbose_setup_coarse) {
        MPI_Barrier(*comm_coarsest);
        if (rank_coarsest == 0) printf("PASS THE MATRIX FROM SAENA. \n");
        MPI_Barrier(*comm_coarsest);
    }
#endif

    m       = A_coarsest->Mbig;
    m_loc   = A_coarsest->M;
    n       = m;
    nnz_loc = A_coarsest->nnz_l;
    ldb     = m_loc;

#ifdef __DEBUG1__
    if (verbose_setup_coarse) {
        MPI_Barrier(*comm_coarsest);
        if (rank_coarsest == 0)
            printf("m = %d, m_loc = %d, n = %d, nnz_g = %ld, nnz_loc = %d, ldb = %d \n",
                   m, m_loc, n, A_coarsest->nnz_g, nnz_loc, ldb);
        MPI_Barrier(*comm_coarsest);
    }
#endif

    // Create the matrix in CSR format (compressed row) to pass to SuperLU
    // make a copy of entries and sort them in row-major order
    std::vector<cooEntry> entry_temp = A_coarsest->entry;
    std::sort(entry_temp.begin(), entry_temp.end(), row_major);
//    print_vector(entry_temp, -1, "entry_temp", *comm_coarsest);

//    index_t fst_row = A_coarsest->split[rank_coarsest]; // the offset for the first row
    fst_row = A_coarsest->split[rank_coarsest]; // the offset for the first row

    // these will be freed when calling Destroy_CompRowLoc_Matrix_dist on the matrix.
    auto *rowptr    = (int_t *) intMalloc_dist(m_loc + 1);      // scan on nonzeros per row
    auto *nzval_loc = (double *) doubleMalloc_dist(nnz_loc);    // values
    auto *colind    = (int_t *) intMalloc_dist(nnz_loc);        // column indices

    assert(rowptr    != nullptr);
    assert(nzval_loc != nullptr);
    assert(colind    != nullptr);

    std::fill(&rowptr[0], &rowptr[m_loc + 1], 0);

    // Do this line to avoid this subtraction for each entry in the next "for" loop.
    int *rowptr_p = &rowptr[1] - fst_row;

    for (nnz_t i = 0; i < nnz_loc; ++i) {
        ++rowptr_p[entry_temp[i].row];
        colind[i]    = entry_temp[i].col;
        nzval_loc[i] = entry_temp[i].val;
    }

    for (index_t i = 0; i < m_loc; ++i) {
        rowptr[i + 1] += rowptr[i];
    }

    assert(rowptr[m_loc] == nnz_loc);

#ifdef __DEBUG1__
/*
    MPI_Barrier(*comm_coarsest);
//    grids[0].A->print_entry(-1);
    if(rank_coarsest == 1) {
        printf("\nmatrix entries in row-major format to be passed to SuperLU:\n");
        for (nnz_t i = 0; i < nnz_loc; i++) {
            std::cout << entry_temp[i].row << "\t" << colind[i] << "\t" << nzval_loc[i] << std::endl;
//            printf("%ld \t%d \t%d \t%e \n", i, entry_temp[i].row, colind[i], nzval_loc[i]);
        }

//        printf("\nrowptr:\n");
//        for(nnz_t i = 0; i < m_loc+1; i++)
//            printf("%ld \t%d \n", i, rowptr[i]);
    }

    MPI_Barrier(*comm_coarsest);
    if(rank_coarsest == 1){
        printf("\n");
        for(nnz_t i = 0; i < m_loc; i++){
            for(nnz_t j = rowptr[i]; j < rowptr[i+1]; j++){
                std::cout << i + fst_row << "\t" << colind[j] << "\t" << nzval_loc[j] << std::endl;
            }
        }
    }
    MPI_Barrier(*comm_coarsest);
*/
#endif

    // create the matrix
    // =================
//    dcreate_matrix(&A_SLU, nrhs, &b, &ldb, &xtrue, &ldx, fp, &grid);

    // SLU_NR_loc  /* distributed compressed row format  */
    // SLU_D,      /* double */
    // SLU_GE,     /* general */ (there are options for symmetric and triangular)
    // TODO: set an option to set if the matrix is symmetric.

    // This function creates the matrix and puts it in the first argument.
    dCreate_CompRowLoc_Matrix_dist(&A_SLU2, m, n, nnz_loc, m_loc, fst_row,
                                   &nzval_loc[0], &colind[0], &rowptr[0],
                                   SLU_NR_loc, SLU_D, SLU_GE);

    // set the options
    // =================
    // TODO: check these info from SuperLU docs, source code and examples:
#if 0
//    SolveInitialized { YES | NO }
//    Specifies whether the initialization has been performed to the triangular solve.
//    (used only by the distributed input interface)
//    RefineInitialized { YES | NO }
//    Specifies whether the initialization has been performed to the sparse matrix-vector multiplication routine needed in the iterative refinement.
//    (used only by the distributed input interface)

    if ( options->SolveInitialized == NO ) { /* First time */
        dSolveInit(options, A, perm_r, perm_c, nrhs, LUstruct, grid,
                   SOLVEstruct);
        /* Inside this routine, SolveInitialized is set to YES.
       For repeated call to pdgssvx(), no need to re-initialilze
       the Solve data & communication structures, unless a new
       factorization with Fact == DOFACT or SamePattern is asked for. */
    }
#endif

    /* Set the default input options:
        options.Fact              = DOFACT;
        options.Equil             = YES;
        options.ParSymbFact       = NO;
        options.ColPerm           = METIS_AT_PLUS_A;
        options.RowPerm           = LargeDiag_MC64;
        options.ReplaceTinyPivot  = NO;
        options.IterRefine        = DOUBLE;
        options.Trans             = NOTRANS;
        options.SolveInitialized  = NO;
        options.RefineInitialized = NO;
        options.PrintStat         = YES; -> I changed this to NO.
     */

    set_default_options_dist(&options);
    options.ColPerm = NATURAL;
//    options.SymPattern = YES;

    // initialize the required parameters
    // =================
    ScalePermstructInit(m, n, &ScalePermstruct);
    LUstructInit(n, &LUstruct);

#ifdef __DEBUG1__
    if (verbose_setup_coarse) {
        options.PrintStat = YES;
        MPI_Barrier(*comm_coarsest);
        if (rank_coarsest == 0) printf("setup_SuperLU: done. \n");
        MPI_Barrier(*comm_coarsest);
    }
#endif

    return 0;
}

int saena_object::destroy_SuperLU(){

    if(superlu_allocated){
        superlu_allocated = false;

        if(superlu_active){
            Destroy_CompRowLoc_Matrix_dist(&A_SLU2);
            ScalePermstructFree(&ScalePermstruct);
            if(lu_created){
                Destroy_LU(A_coarsest->Mbig, &superlu_grid, &LUstruct);
            }
            LUstructFree(&LUstruct);
            superlu_gridexit(&superlu_grid);
            if ( options.SolveInitialized ) {
                dSolveFinalize(&options, &SOLVEstruct);
            }
        }
    }

    return 0;
}

// before checking if solve_coarsest_SuperLU() is called for the first time or not.
#if 0
int saena_object::solve_coarsest_SuperLU(saena_matrix *A, std::vector<value_t> &u, std::vector<value_t> &rhs){

//    MPI_Barrier(A->comm);
//    printf("A->print_entry\n");
//    A->print_entry(-1);
//    MPI_Barrier(A->comm);

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        MPI_Barrier(comm);
        if(rank==0){
            printf("\nstart of solve_coarsest_SuperLU()\n");
        }
        MPI_Barrier(comm);
//    print_vector(rhs, -1, "rhs passed to superlu", comm);
    }
#endif

    superlu_dist_options_t options;
    SuperLUStat_t stat;
//    SuperMatrix A_SLU;
    ScalePermstruct_t ScalePermstruct;
    LUstruct_t LUstruct;
    SOLVEstruct_t SOLVEstruct;
    gridinfo_t grid;
    double   *berr;
    double   *b;
    int      m, n, m_loc, nnz_loc;
    int      nprow, npcol;
    int      iam, info, ldb, nrhs;

//    double   *b, *xtrue;
//    int      iam, info, ldb, ldx, nrhs;
//    char     **cpp, c;
//    FILE *fp, *fopen();
//    FILE *fp;
//    int cpp_defs();

    nprow = 1;      /* Default process rows.      */
    npcol = nprocs; /* Default process columns.   */
    nrhs  = 1;      /* Number of right-hand side. */

#if 0
    // ------------------------------------------------------------
    //   INITIALIZE MPI ENVIRONMENT.
    // ------------------------------------------------------------
    MPI_Init( &argc, &argv );

    char* file_name(argv[5]);
    saena::matrix A_saena (file_name, comm);
    A_saena.assemble();
    A_saena.print_entry(-1);
    if(rank==0) printf("after matrix assemble.\n");

    // Parse command line argv[].
    for (cpp = argv+1; *cpp; ++cpp) {
        if ( **cpp == '-' ) {
            c = *(*cpp+1);
            ++cpp;
            switch (c) {
                case 'h':
                    printf("Options:\n");
                    printf("\t-r <int>: process rows    (default %4d)\n", nprow);
                    printf("\t-c <int>: process columns (default %4d)\n", npcol);
                    exit(0);
                    break;
                case 'r': nprow = atoi(*cpp);
                    break;
                case 'c': npcol = atoi(*cpp);
                    break;
            }
        } else { // Last arg is considered a filename
//            if ( !(fp = fopen(*cpp, "r")) ) {
//                ABORT("File does not exist");
//            }

            saena::matrix A_saena (*cpp, comm);
            A_saena.assemble();
            A_saena.print_entry(-1);
            if(rank==0) printf("after matrix assemble.\n");
            break;
        }
    }
#endif

    /* ------------------------------------------------------------
       INITIALIZE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------*/

#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        MPI_Barrier(comm);
        if(rank==0){
            printf("INITIALIZE THE SUPERLU PROCESS GRID. \n");
        }
        MPI_Barrier(comm);
    }
#endif

    superlu_gridinit(comm, nprow, npcol, &grid);

    // Bail out if I do not belong in the grid.
    iam = grid.iam; // my process rank in this group
//    iam = superlu_grid.iam;
//    printf("iam = %d, nprow = %d, npcol = %d \n", iam, nprow, npcol);

    // tag for quitting processors that don't belong to the SuperLU's process grid.
    if ( iam >= nprow * npcol ){
//        goto superlu_out2;
        superlu_gridexit(&grid);
        return 0;
    }

#if ( VAMPIR>=1 )
    VT_traceoff();
#endif

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter main()");
#endif

    /* ------------------------------------------------------------
       PASS THE MATRIX FROM SAENA
       ------------------------------------------------------------*/

    // Set up the local A_SLU in NR_loc format
//    dCreate_CompRowLoc_Matrix_dist(A_SLU, m, n, nnz_loc, m_loc, fst_row,
//                                   nzval_loc, colind, rowptr,
//                                   SLU_NR_loc, SLU_D, SLU_GE);

    m       = A->Mbig;
    m_loc   = A->M;
    n       = m;
    nnz_loc = A->nnz_l;
    ldb     = m_loc;

/*
#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        MPI_Barrier(comm);
        if(rank==0) printf("m = %d, m_loc = %d, n = %d, nnz_g = %ld, nnz_loc = %d, ldb = %d \n",
                           m, m_loc, n, A->nnz_g, nnz_loc, ldb);
        MPI_Barrier(comm);
    }
#endif

    // CSR format (compressed row)
    // sort entries in row-major
    std::vector<cooEntry> entry_temp = A->entry;
    std::sort(entry_temp.begin(), entry_temp.end(), row_major);
//    print_vector(entry_temp, -1, "entry_temp", comm);

    index_t fst_row = A->split[rank];
    std::vector<int> nnz_per_row(m_loc, 0);
//    std::vector<int> rowptr(m_loc+1);
//    std::vector<int> colind(nnz_loc);
//    std::vector<double> nzval_loc(nnz_loc);

    auto* rowptr    = (int_t *) intMalloc_dist(m_loc+1);
    auto* nzval_loc = (double *) doubleMalloc_dist(nnz_loc);
    auto* colind    = (int_t *) intMalloc_dist(nnz_loc);

    // Do this line to avoid this subtraction for each entry in the next "for" loop.
    int *nnz_per_row_p = &nnz_per_row[0] - fst_row;

    for(nnz_t i = 0; i < nnz_loc; i++){
        nnz_per_row_p[entry_temp[i].row]++;
        colind[i]    = entry_temp[i].col;
        nzval_loc[i] = entry_temp[i].val;
    }

    // rowtptr is scan of nnz_per_row.
    rowptr[0] = 0;
    for(index_t i = 0; i < m_loc; i++)
        rowptr[i+1] = rowptr[i] + nnz_per_row[i];

//    A->print_entry(-1);
//    print_vector(rowptr, -1, "rowptr", comm);
//    print_vector(colind, -1, "colind", comm);
//    print_vector(nzval_loc, -1, "nzval_loc", comm);
//    if(rank==0){
//        printf("\nmatrix entries in row-major format to be passed to SuperLU:\n");
//        for(nnz_t i = 0; i < nnz_loc; i++)
//            printf("%ld \t%d \t%lld \t%lf \n", i, entry_temp[i].row-fst_row, colind[i], nzval_loc[i]);
//        printf("\nrowptr:\n");
//        for(nnz_t i = 0; i < m_loc+1; i++)
//            printf("%ld \t%lld \n", i, rowptr[i]);
//    }

    dCreate_CompRowLoc_Matrix_dist(&A_SLU, m, n, nnz_loc, m_loc, fst_row,
                                   &nzval_loc[0], &colind[0], &rowptr[0],
                                   SLU_NR_loc, SLU_D, SLU_GE);

//    dcreate_matrix(&A_SLU, nrhs, &b, &ldb, &xtrue, &ldx, fp, &grid);
*/
    if ( !(berr = doubleMalloc_dist(nrhs)) )
        ABORT("Malloc fails for berr[].");

    /* ------------------------------------------------------------
       SET THE RIGHT HAND SIDE.
       ------------------------------------------------------------*/

    b = &rhs[0];
    u = rhs; // copy rhs to u. the solution will be save in b at the end. then, swap u and rhs.

    /* ------------------------------------------------------------
       .
       ------------------------------------------------------------*/

    /* Set the default input options:
        options.Fact              = DOFACT;
        options.Equil             = YES;
        options.ParSymbFact       = NO;
        options.ColPerm           = METIS_AT_PLUS_A;
        options.RowPerm           = LargeDiag_MC64;
        options.ReplaceTinyPivot  = NO;
        options.IterRefine        = DOUBLE;
        options.Trans             = NOTRANS;
        options.SolveInitialized  = NO;
        options.RefineInitialized = NO;
        options.PrintStat         = YES; -> I changed this to NO.
     */

    // I changed options->PrintStat default to NO.
    set_default_options_dist(&options);
    options.ColPerm = NATURAL;
//    options.SymPattern = YES;
//    options.PrintStat = YES;

#if 0
    options.RowPerm = NOROWPERM;
    options.RowPerm = LargeDiag_AWPM;
    options.IterRefine = NOREFINE;
    options.ColPerm = NATURAL;
    options.Equil = NO;
    options.ReplaceTinyPivot = YES;
#endif

#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        MPI_Barrier(comm);
        if(!iam){
            print_sp_ienv_dist(&options);
            print_options_dist(&options);
            fflush(stdout);
        }
        MPI_Barrier(comm);
        if(rank==0) printf("SOLVE THE LINEAR SYSTEM: step 1 \n");
        MPI_Barrier(comm);
    }
#endif

//    m = A_SLU.nrow;
//    n = A_SLU.ncol;

    // Initialize ScalePermstruct and LUstruct.
    ScalePermstructInit(m, n, &ScalePermstruct);
    LUstructInit(n, &LUstruct);

#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        MPI_Barrier(comm);
        if(rank==0) printf("SOLVE THE LINEAR SYSTEM: step 2 \n");
        MPI_Barrier(comm);
    }
#endif

    // Initialize the statistics variables.
    PStatInit(&stat);
    // Call the linear equation solver.
    // b points to rhs. after calling pdgssvx it will be the solution.
    pdgssvx(&options, &A_SLU2, &ScalePermstruct, b, ldb, nrhs, &grid,
            &LUstruct, &SOLVEstruct, berr, &stat, &info);

    // put the solution in u
    // b points to rhs. after calling pdgssvx it will be the solution.
    u.swap(rhs);

//    print_vector(u, -1, "u computed in superlu", comm);

    // Check the accuracy of the solution.
//    pdinf_norm_error(iam, ((NRformat_loc *)A_SLU.Store)->m_loc,
//                     nrhs, b, ldb, xtrue, ldx, &grid);

#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        PStatPrint(&options, &stat, &grid); // Print the statistics.
    }
#endif

    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------*/

#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        MPI_Barrier(comm);
        if(rank==0) printf("DEALLOCATE STORAGE. \n");
        MPI_Barrier(comm);
    }
#endif

    PStatFree(&stat);
    ScalePermstructFree(&ScalePermstruct);
    Destroy_LU(n, &grid, &LUstruct);
    LUstructFree(&LUstruct);
    if ( options.SolveInitialized ) {
        dSolveFinalize(&options, &SOLVEstruct);
    }
    SUPERLU_FREE(berr);
//    Destroy_CompRowLoc_Matrix_dist(&A_SLU);

    // don't need these two.
//    SUPERLU_FREE(b);
//    SUPERLU_FREE(xtrue);

    /* ------------------------------------------------------------
       RELEASE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------*/
//    superlu_out2:
//    superlu_gridexit(&grid);

    /* ------------------------------------------------------------
       TERMINATES THE MPI EXECUTION ENVIRONMENT.
       ------------------------------------------------------------*/
//    MPI_Finalize();

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit main()");
#endif

#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        MPI_Barrier(comm);
        if(rank==0) printf("end of solve_coarsest_SuperLU()\n\n");
        MPI_Barrier(comm);
    }
#endif

    return 0;
}
#endif

int saena_object::solve_coarsest_SuperLU(saena_matrix *A, std::vector<value_t> &u, std::vector<value_t> &rhs){
    // For a similar code, using the same matrix for mutiple rhs's, read SuperLU_DIST_5.4.0/EXAMPLE/pddrive1.c

    if(!superlu_active){
        return 0;
    }

    MPI_Comm comm = A->comm;
    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        MPI_Barrier(comm);
        if(rank==0) printf("\nsolve_coarsest_SuperLU(): start\n");
        MPI_Barrier(comm);
    }
//        A->print_info(-1);
//        A->print_entry(-1);
//        print_vector(rhs, -1, "rhs passed to superlu", comm);
//        print_vector(u, -1, "u passed to superlu", comm);
#endif

    SuperLUStat_t stat;
    double   *berr = nullptr;
    double   *b    = nullptr;
    int      m = 0, n = 0, m_loc = 0, nnz_loc = 0;
    int      nprow = 0, npcol = 0;
    int      iam = 0, info = 0, ldb = 0, nrhs = 0;

    /* ------------------------------------------------------------
       INITIALIZE SOME PARAMETERS
       ------------------------------------------------------------*/

    nprow = nprocs; /* Default process rows.      */
    npcol = 1;      /* Default process columns.   */
    nrhs  = 1;      /* Number of right-hand side. */

    iam = superlu_grid.iam;
//    printf("iam = %d, nprow = %d, npcol = %d \n", iam, nprow, npcol);

#if ( VAMPIR>=1 )
    VT_traceoff();
#endif

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter main()");
#endif

    m       = A->Mbig;
    m_loc   = A->M;
    n       = m;
    nnz_loc = A->nnz_l;
    ldb     = m_loc;

    if ( !(berr = doubleMalloc_dist(nrhs)) )
    ABORT("Malloc fails for berr[].");

#ifdef __DEBUG1__
    if (verbose_solve_coarse) {
//        MPI_Barrier(*comm_coarsest);
//        if (rank_coarsest == 0)
            printf("m = %d, m_loc = %d, n = %d, nnz_g = %ld, nnz_loc = %d, ldb = %d \n",
                   m, m_loc, n, A_coarsest->nnz_g, nnz_loc, ldb);
//        MPI_Barrier(*comm_coarsest);
    }
#endif

    /* ------------------------------------------------------------
       SET THE RIGHT HAND SIDE.
       ------------------------------------------------------------*/

    b = &rhs[0];
    u = rhs; // copy rhs to u. the solution will be saved in b at the end. then, swap u and rhs.

    /* ------------------------------------------------------------
       SOLVE THE LINEAR SYSTEM
       ------------------------------------------------------------*/

#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        MPI_Barrier(comm);
        if(!iam){
            print_sp_ienv_dist(&options);
            print_options_dist(&options);
            fflush(stdout);
        }
        MPI_Barrier(comm);
        if(rank==0) printf("SOLVE THE LINEAR SYSTEM\n");
        MPI_Barrier(comm);
    }
#endif

    // Initialize the statistics variables.
    PStatInit(&stat);

    // Call the linear equation solver.
    // on entry, b points to rhs. on return, it will be the solution.
    pdgssvx(&options, &A_SLU2, &ScalePermstruct, b, ldb, nrhs, &superlu_grid,
            &LUstruct, &SOLVEstruct, berr, &stat, &info);

    // put the solution in u
    // b points to rhs. after calling pdgssvx it will be the solution.
    u.swap(rhs);

//    print_vector(u, -1, "u computed in superlu", comm);

    // Check the accuracy of the solution.
//    pdinf_norm_error(iam, ((NRformat_loc *)A_SLU2.Store)->m_loc,
//                     nrhs, b, ldb, xtrue, ldb, &superlu_grid);

#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        PStatPrint(&options, &stat, &superlu_grid); // Print the statistics.
    }
#endif

    if(first_solve){
        options.Fact = FACTORED;
        lu_created   = TRUE;
        first_solve  = FALSE;
    }

    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------*/

#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        MPI_Barrier(comm);
        if(rank==0) printf("DEALLOCATE STORAGE. \n");
        MPI_Barrier(comm);
    }
#endif

    PStatFree(&stat);
    SUPERLU_FREE(berr);

    // don't need these two.
//    SUPERLU_FREE(b);
//    SUPERLU_FREE(xtrue);

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit main()");
#endif

#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        MPI_Barrier(comm);
        if(rank==0) printf("solve_coarsest_SuperLU(): done\n\n");
        MPI_Barrier(comm);
    }
#endif

    return 0;
}


int saena_object::setup_vcycle_memory(){
    for(int i = 0; i < grids.size() - 1; ++i){
        if(grids[i].active){
            grids[i].res.resize(grids[i].A->M);
            grids[i].uCorr.resize(grids[i].A->M);
//            grids[i].res_coarse.resize(max(grids[i].Ac.M_old, grids[i].Ac.M));
//            grids[i].uCorrCoarse.resize(max(grids[i].Ac.M_old, grids[i].Ac.M));
        }
    }
    return 0;
}


void saena_object::vcycle(Grid* grid, std::vector<value_t>& u, std::vector<value_t>& rhs) {

    if (!grid->A->active) {
        return;
    }

    MPI_Comm comm = grid->A->comm;
    int rank = -1, nprocs = -1;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    double t1 = 0, t2 = 0;
    value_t dot = 0.0;

#ifdef __DEBUG1__
    std::string func_name;
//    print_vector(rhs, -1, "rhs in vcycle", comm);

    if (verbose_vcycle) {
        MPI_Barrier(comm);
        if (!rank) printf("\n");
        MPI_Barrier(comm);
        printf("rank = %d: vcycle level = %d, A->M = %u, u.size = %lu, rhs.size = %lu \n",
               rank, grid->level, grid->A->M, u.size(), rhs.size());
        MPI_Barrier(comm);
    }
#endif

    // **************************************** 0. direct-solve the coarsest level ****************************************

    if (grid->level == max_level) {

#ifdef __DEBUG1__
        if (verbose_vcycle) {
            MPI_Barrier(comm);
            if (rank == 0) std::cout << "vcycle: solving the coarsest level using " << direct_solver << std::endl;
            MPI_Barrier(comm);
        }
        if (verbose) t1 = omp_get_wtime();
#endif

#ifdef PROFILE_VCYCLE
        MPI_Barrier(comm);
        double slu1 = omp_get_wtime();
#endif

        if (direct_solver == "CG") {
            solve_coarsest_CG(grid->A, u, rhs);
        } else if (direct_solver == "SuperLU") {
            solve_coarsest_SuperLU(grid->A, u, rhs);
        } else {
            if (!rank) printf("Error: Unknown direct solver! \n");
            exit(EXIT_FAILURE);
        }

#ifdef PROFILE_VCYCLE
        double slu2 = omp_get_wtime();
        superlu_time += slu2 - slu1;
#endif

#ifdef __DEBUG1__
        {
            if (verbose) {
                t2 = omp_get_wtime();
                func_name = "vcycle: level " + std::to_string(grid->level) + ": solve coarsest";
                print_time(t1, t2, func_name, comm);
            }

            if (verbose_vcycle_residuals) {
                std::vector<value_t> res(grid->A->M);
                grid->A->residual(u, rhs, res);
                dotProduct(res, res, &dot, comm);
                if (rank == 0)
                    std::cout << "\nlevel = " << grid->level
                              << ", after coarsest level = " << sqrt(dot) << std::endl;
            }

            // print the solution
            // ------------------
//            print_vector(u, -1, "solution from the direct solver", grid->A->comm);

            // check if the solution is correct
            // --------------------------------
//            std::vector<double> rhs_matvec(u.size(), 0);
//            grid->A->matvec(u, rhs_matvec);
//            if(rank==0){
//                printf("\nA*u - rhs:\n");
//                for(i = 0; i < rhs_matvec.size(); i++){
//                    if(rhs_matvec[i] - rhs[i] > 1e-6)
//                        printf("%lu \t%f - %f = \t%f \n", i, rhs_matvec[i], rhs[i], rhs_matvec[i] - rhs[i]);}
//                printf("-----------------------\n");
//            }
        }
#endif

        return;
    }

    std::vector<value_t> &res         = grid->res;
    std::vector<value_t> &uCorr       = grid->uCorr;
//    std::vector<value_t> &res_coarse  = grid->res_coarse;
//    std::vector<value_t> &uCorrCoarse = grid->uCorrCoarse;

    std::vector<value_t> res_coarse(grid->Ac.M_old);
    std::vector<value_t> uCorrCoarse(grid->Ac.M);
//    std::vector<value_t> res(grid->A->M);
//    std::vector<value_t> uCorr(grid->A->M);

#ifdef __DEBUG1__
    if (verbose_vcycle_residuals) {
        grid->A->residual(u, rhs, res);
        dotProduct(res, res, &dot, comm);
        if (!rank)
            std::cout << "\nlevel = " << grid->level << ", vcycle start      = " << sqrt(dot) << std::endl;
    }
#endif

    // **************************************** 1. pre-smooth ****************************************

#ifdef __DEBUG1__
    if (verbose_vcycle) {
        MPI_Barrier(comm);
        if (rank == 0) printf("vcycle level %d: presmooth\n", grid->level);
        MPI_Barrier(comm);
    }

    t1 = omp_get_wtime();
#endif

    double time_smooth_pre1 = 0.0, time_smooth_pre2 = 0.0;
//    if (grid->level == 0) {
#ifdef PROFILE_VCYCLE
        MPI_Barrier(comm);
        time_smooth_pre1 = omp_get_wtime();
#endif
//    }

    if (preSmooth) {
        smooth(grid, u, rhs, preSmooth);
    }

//    if (grid->level == 0) {
#ifdef PROFILE_VCYCLE
        time_smooth_pre2 = omp_get_wtime();
        vcycle_smooth_time += time_smooth_pre2 - time_smooth_pre1;
#endif
//    }

#ifdef __DEBUG1__
    t2 = omp_get_wtime();
    func_name = "Vcycle: level " + std::to_string(grid->level) + ": pre";
    if (verbose) print_time(t1, t2, func_name, comm);

//    print_vector(u, -1, "u in vcycle", comm);
//    if(rank==0) std::cout << "\n1. pre-smooth: u, level = " << grid->level << std::endl;
#endif

    // **************************************** 2. compute residual ****************************************

#ifdef __DEBUG1__
    if (verbose_vcycle) {
        MPI_Barrier(comm);
        if (rank == 0) printf("vcycle level %d: residual\n", grid->level);
        MPI_Barrier(comm);
    }
#endif

#ifdef PROFILE_VCYCLE
    MPI_Barrier(comm);
    double time_other1 = 0.0, time_other2 = 0.0;
    time_other1 = omp_get_wtime();
#endif

    grid->A->residual(u, rhs, res);

#ifdef PROFILE_VCYCLE
    time_other2 = omp_get_wtime();
    vcycle_other_time += time_other2 - time_other1;
#endif

#ifdef __DEBUG1__
//    print_vector(res, -1, "res", comm);

    if (verbose_vcycle_residuals) {
        dotProduct(res, res, &dot, comm);
        if (rank == 0)
            std::cout << "level = " << grid->level << ", after pre-smooth  = " << sqrt(dot) << std::endl;
    }
#endif

    // **************************************** 3. restrict ****************************************

#ifdef __DEBUG1__
    if (verbose_vcycle) {
        MPI_Barrier(comm);
        if (rank == 0) printf("vcycle level %d: restrict\n", grid->level);
//        printf("rank %d: grid->Ac.M_old = %u \n", rank, grid->Ac.M_old);
        MPI_Barrier(comm);
    }

    t1 = omp_get_wtime();
#endif

#ifdef PROFILE_VCYCLE
    MPI_Barrier(comm);
    double t_trans1 = omp_get_wtime();
#endif

    grid->R.matvec(res, res_coarse);

#ifdef PROFILE_VCYCLE
    double t_trans2 = omp_get_wtime();
    Rtransfer_time += t_trans2 - t_trans1;
#endif

#ifdef __DEBUG1__
//    grid->R.print_entry(-1);
//    print_vector(res_coarse, -1, "res_coarse in vcycle", comm);

    if (verbose_vcycle) {
        MPI_Barrier(comm);
        if (rank == 0) printf("vcycle level %d: repart_u_shrink\n", grid->level);
//        printf("rank %d: res.size() = %lu \tres_coarse.size() = %lu \n", rank, res.size(), res_coarse.size());
        MPI_Barrier(comm);
    }
#endif

    if (grid->Ac.active_minor) {

        if (nprocs > 1) {
            repartition_u_shrink(res_coarse, *grid);
        }

        if (grid->Ac.active) {

            comm = grid->Ac.comm;
            MPI_Comm_rank(comm, &rank);
            MPI_Comm_size(comm, &nprocs);

#ifdef __DEBUG1__

            {
//                MPI_Barrier(comm);
//                printf("rank %d: after  repart_u_shrink: res_coarse.size = %ld \n", rank, res_coarse.size());
//                MPI_Barrier(comm);
//                print_vector(res_coarse, 0, "res_coarse", comm);

                t2 = omp_get_wtime();
                func_name = "Vcycle: level " + std::to_string(grid->level) + ": restriction";
                if (verbose) print_time(t1, t2, func_name, comm);
            }
#endif

            // **************************************** 4. recurse ****************************************

#ifdef __DEBUG1__
            if (verbose_vcycle) {
                MPI_Barrier(comm);
                if (rank == 0) printf("vcycle level %d: recurse\n", grid->level);
                MPI_Barrier(comm);
            }
#endif

            // scale rhs of the next level
            if(scale) {
                scale_vector(res_coarse, grid->coarseGrid->A->inv_sq_diag_orig);
            }

//            uCorrCoarse.assign(grid->Ac.M, 0);
            fill(uCorrCoarse.begin(), uCorrCoarse.end(), 0);
            vcycle(grid->coarseGrid, uCorrCoarse, res_coarse);

            // scale uCorrCoarse
            if(scale) {
                scale_vector(uCorrCoarse, grid->coarseGrid->A->inv_sq_diag_orig);
            }

#ifdef __DEBUG1__
//            print_vector(uCorrCoarse, -1, "uCorrCoarse", grid->A->comm);
#endif

        } // if (grid->Ac.active)
    } // if (grid->Ac.active_minor)

    // **************************************** 5. prolong ****************************************

    comm = grid->A->comm;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    t1 = omp_get_wtime();

    if (verbose_vcycle) {
        MPI_Barrier(comm);
        if (rank == 0) printf("\nvcycle level %d: repart_back_u_shrink\n", grid->level);
        MPI_Barrier(comm);
    }
#endif

#ifdef PROFILE_VCYCLE
    MPI_Barrier(comm);
    time_other1 = omp_get_wtime();
#endif

    if(nprocs > 1 && grid->Ac.active_minor){
        repartition_back_u_shrink(uCorrCoarse, *grid);
    }

#ifdef PROFILE_VCYCLE
    time_other2 = omp_get_wtime();
    vcycle_other_time += time_other2 - time_other1;
#endif

#ifdef __DEBUG1__
//    MPI_Barrier(comm); printf("rank %d: after  repart_back_u_shrink: uCorrCoarse.size = %ld \n", rank, uCorrCoarse.size()); MPI_Barrier(comm);
//    print_vector(uCorrCoarse, -1, "uCorrCoarse", grid->A->comm);

    if(verbose_vcycle){
        MPI_Barrier(comm);
        if(rank==0) printf("vcycle level %d: prolong\n", grid->level);
        MPI_Barrier(comm);}
#endif

#ifdef PROFILE_VCYCLE
    MPI_Barrier(comm);
    t_trans1 = omp_get_wtime();
#endif

    grid->P.matvec(uCorrCoarse, uCorr);

#ifdef PROFILE_VCYCLE
    t_trans2 = omp_get_wtime();
    Ptransfer_time += t_trans2 - t_trans1;
#endif

#ifdef __DEBUG1__
    t2 = omp_get_wtime();
    func_name = "Vcycle: level " + std::to_string(grid->level) + ": prolongation";
    if (verbose) print_time(t1, t2, func_name, comm);

//    if(rank==0)
//        std::cout << "\n5. prolongation: uCorr = P*uCorrCoarse , level = " << grid->level
//                  << ", uCorr.size = " << uCorr.size() << std::endl;
//    print_vector(uCorr, -1, "uCorr", grid->A->comm);
#endif

    // **************************************** 6. correct ****************************************

#ifdef __DEBUG1__
    if(verbose_vcycle){
        MPI_Barrier(comm);
        if(rank==0) printf("vcycle level %d: correct\n", grid->level);
        MPI_Barrier(comm);}
#endif

    #pragma omp parallel for default(none) shared(u, uCorr)
    for (index_t i = 0; i < u.size(); i++)
        u[i] -= uCorr[i];

#ifdef __DEBUG1__
//    print_vector(u, 0, "u after correction", grid->A->comm);

    if(verbose_vcycle_residuals){
        grid->A->residual(u, rhs, res);
        dotProduct(res, res, &dot, comm);
        if(rank==0) std::cout << "level = " << grid->level << ", after correction  = " << sqrt(dot) << std::endl;
    }
#endif

    // **************************************** 7. post-smooth ****************************************

#ifdef __DEBUG1__
    if(verbose_vcycle){
        MPI_Barrier(comm);
        if(rank==0) printf("vcycle level %d: post-smooth\n", grid->level);
        MPI_Barrier(comm);}

    t1 = omp_get_wtime();
#endif

    double time_smooth_post1 = 0.0, time_smooth_post2 = 0.0;
//    if (grid->level == 0) {
#ifdef PROFILE_VCYCLE
        MPI_Barrier(comm);
        time_smooth_post1 = omp_get_wtime();
//    }
#endif

    if(postSmooth){
        smooth(grid, u, rhs, postSmooth);
    }

//    if (grid->level == 0) {
#ifdef PROFILE_VCYCLE
        time_smooth_post2 = omp_get_wtime();
        vcycle_smooth_time += time_smooth_post2 - time_smooth_post1;
#endif
//    }

#ifdef __DEBUG1__
    t2 = omp_get_wtime();
    func_name = "Vcycle: level " + std::to_string(grid->level) + ": post";
    if (verbose) print_time(t1, t2, func_name, comm);

    if(verbose_vcycle_residuals){

//        if(rank==1) std::cout << "\n7. post-smooth: u, level = " << grid->level << std::endl;
//        print_vector(u, 0, "u post-smooth", grid->A->comm);

        grid->A->residual(u, rhs, res);
        dotProduct(res, res, &dot, comm);
        if(rank==0) std::cout << "level = " << grid->level << ", after post-smooth = " << sqrt(dot) << std::endl;
    }
#endif
}


int saena_object::solve(std::vector<value_t>& u){

    auto *A = grids[0].A;
    vector<value_t> &rhs = grids[0].rhs;

    MPI_Comm comm = A->comm;
    int nprocs = -1, rank = -1;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
//    print_vector(rhs, -1, "rhs", comm);
    if(verbose_solve){
        MPI_Barrier(comm);
        if(rank == 0) printf("solve: start\n");
        MPI_Barrier(comm);
    }
#endif

    // ************** check u size **************
/*
    index_t u_size_local = u.size(), u_size_total;
    MPI_Allreduce(&u_size_local, &u_size_total, 1, MPI_UNSIGNED, MPI_SUM, A->comm);
    if(A->Mbig != u_size_total){
        if(rank==0) printf("Error: size of LHS (=%u) and the solution vector u (=%u) are not equal!\n", A->Mbig, u_size_total);
        MPI_Finalize();
        return -1;
    }
*/
    // ************** repartition u **************
/*
    if(repartition)
        repartition_u(u);
*/

    // ************** initialize u **************

    u.assign(A->M, 0);

    // ************** allocate memory for vcycle **************

    setup_vcycle_memory();

    // ************** solve **************

#ifdef __DEBUG1__
//    double temp;
//    current_dot(rhs, rhs, &temp, comm);
//    if(rank==0) std::cout << "norm(rhs) = " << sqrt(temp) << std::endl;
#endif

    std::vector<value_t> r(A->M);
    A->residual(u, rhs, r);
    double init_dot = 0.0, current_dot = 0.0;
    dotProduct(r, r, &init_dot, comm);
    if(!rank){
        print_sep();
        printf("\ninitial residual = %e \n\n", sqrt(init_dot));
    }

    const double THRSHLD = init_dot * solver_tol * solver_tol;

    // if max_level==0, it means only direct solver is being used.
    if(max_level == 0 && !rank){
        printf("\nonly using the direct solver! \n");
    }

    int i = 0;
    for(; i < solver_max_iter; ++i){
        vcycle(&grids[0], u, rhs);
        A->residual(u, rhs, r);
        dotProduct(r, r, &current_dot, comm);

#ifdef __DEBUG1__
//        if(rank==0) printf("Vcycle %d: \t%.10f \n", i, sqrt(current_dot));
//        if(rank==0) printf("vcycle iteration = %d, residual = %f \n\n", i, sqrt(current_dot));
#endif

        if(current_dot < THRSHLD)
            break;
    }

    // set number of iterations that took to find the solution
    // only do the following if the end of the previous for loop was reached.
    if(i == solver_max_iter)
        --i;

    if(!rank){
        print_sep();
        printf("\nfinal:\nstopped at iteration    = %d \nfinal absolute residual = %e"
                       "\nrelative residual       = %e \n", ++i, sqrt(current_dot), sqrt(current_dot / init_dot));
        print_sep();
    }

#ifdef __DEBUG1__
//    print_vector(u, -1, "u", comm);
#endif

    // ************** scale u **************

    if(scale){
        scale_vector(u, A->inv_sq_diag_orig);
    }

    // ************** repartition u back **************

//    if(repartition)
//        repartition_back_u(u);

#ifdef __DEBUG1__
//    print_vector(u, -1, "u", comm);
    if(verbose_solve){
        MPI_Barrier(comm);
        if(rank == 0) printf("solve_pcg: end\n");
        MPI_Barrier(comm);
    }
#endif

    return 0;
}


int saena_object::solve_smoother(std::vector<value_t>& u){

    auto *A = grids[0].A;
    vector<value_t> &rhs = grids[0].rhs;

    MPI_Comm comm = A->comm;
    int nprocs = -1, rank = -1;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // ************** check u size **************
/*
    index_t u_size_local = u.size(), u_size_total;
    MPI_Allreduce(&u_size_local, &u_size_total, 1, MPI_UNSIGNED, MPI_SUM, A->comm);
    if(A->Mbig != u_size_total){
        if(rank==0) printf("Error: size of LHS (=%u) and the solution vector u (=%u) are not equal!\n", A->Mbig, u_size_total);
        MPI_Finalize();
        return -1;
    }
*/
    // ************** repartition u **************
/*
    if(repartition)
        repartition_u(u);
*/

    // ************** initialize u **************

    u.assign(A->M, 0);

    // ************** solve **************

//    double temp;
//    current_dot(rhs, rhs, &temp, comm);
//    if(rank==0) std::cout << "norm(rhs) = " << sqrt(temp) << std::endl;

    std::vector<value_t> r(A->M);
    A->residual(u, rhs, r);
    double init_dot = 0.0, current_dot = 0.0;
    dotProduct(r, r, &init_dot, comm);
    if(!rank){
        print_sep();
        printf("\ninitial residual = %e \n\n", sqrt(init_dot));
    }

    const double THRSHLD = init_dot * solver_tol * solver_tol;

    int i = 0;
    for(i = 0; i < solver_max_iter; ++i){
        smooth(&grids[0], u, rhs, preSmooth);
        A->residual(u, rhs, r);
        dotProduct(r, r, &current_dot, comm);

//        if(rank==0) printf("Vcycle %d: \t%.10f \n", i, sqrt(current_dot));
//        if(rank==0) printf("vcycle iteration = %d, residual = %f \n\n", i, sqrt(current_dot));
        if(current_dot < THRSHLD)
            break;
    }

    // set number of iterations that took to find the solution
    // only do the following if the end of the previous for loop was reached.
    if(i == solver_max_iter)
        --i;

    if(rank==0){
        print_sep();
        printf("\nfinal:\nstopped at iteration    = %d \nfinal absolute residual = %e"
               "\nrelative residual       = %e \n", ++i, sqrt(current_dot), sqrt(current_dot / init_dot));
        print_sep();
    }

//    print_vector(u, -1, "u", comm);

    // ************** scale u **************

    if(scale){
        scale_vector(u, A->inv_sq_diag_orig);
    }

    // ************** repartition u back **************

//    if(repartition)
//        repartition_back_u(u);

//    print_vector(u, -1, "u", comm);

    return 0;
}


int saena_object::solve_CG(std::vector<value_t>& u){

    auto *A = grids[0].A;
    vector<value_t> &rhs = grids[0].rhs;

    MPI_Comm comm = A->comm;
    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(rank == 0) printf("solve_pcg: start!\n");
        MPI_Barrier(comm);
    }
#endif

    // ************** check u size **************
/*
    index_t u_size_local = u.size();
    index_t u_size_total;
    MPI_Allreduce(&u_size_local, &u_size_total, 1, MPI_UNSIGNED, MPI_SUM, A->comm);
    if(A->Mbig != u_size_total){
        if(rank==0) printf("Error: size of LHS (=%u) and the solution vector u (=%u) are not equal!\n", A->Mbig, u_size_total);
        MPI_Finalize();
        return -1;
    }

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(rank == 0) printf("solve_pcg: check u size!\n");
        MPI_Barrier(comm);
    }
#endif
*/

    // ************** repartition u **************
    // todo: using repartition(), give the user the option of passing an initial guess for u. in that case comment
    //  out "initialize u" part.

/*
    std::fill(u.begin(), u.end(), 0);
    if(repartition)
        repartition_u(u);

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(verbose_solve) if(rank == 0) printf("solve_pcg: repartition u!\n");
        MPI_Barrier(comm);
    }
#endif
*/

    // ************** initialize u **************

    u.assign(A->M, 0);

    // ************** solve **************

//    double t1 = MPI_Wtime();

//    double temp;
//    dot(rhs, rhs, &temp, comm);
//    if(rank==0) std::cout << "norm(rhs) = " << sqrt(temp) << std::endl;

    std::vector<value_t> r(A->M);
    A->residual(u, rhs, r);

    double init_dot = 0.0, current_dot = 0.0;
//    double previous_dot;
    dotProduct(r, r, &init_dot, comm);
    if(rank==0) printf("\ninitial residual = %e \n", sqrt(init_dot));

    // if max_level==0, it means only direct solver is being used inside the previous vcycle, and that is all needed.
/*
    if(max_level == 0){
        vcycle(&grids[0], u, rhs);
        A->residual(u, rhs, r);
        dotProduct(r, r, &current_dot, comm);

#ifdef __DEBUG1__
//        print_vector(r, -1, "res", comm);
//        if(rank==0) std::cout << "dot = " << current_dot << std::endl;
#endif

        if(rank==0){
            print_sep();
            printf("\nfinal:\nonly using the direct solver! \nfinal absolute residual = %e"
                   "\nrelative residual       = %e \n\n", sqrt(current_dot), sqrt(current_dot / init_dot));
            print_sep();
        }

        // scale the solution u
        scale_vector(u, A->inv_sq_diag);

        // repartition u back
//        if(repartition){
//            repartition_back_u(u);
//        }

        return 0;
    }
*/

//    std::vector<value_t> rho(A->M, 0);
//    vcycle(&grids[0], rho, r);
    std::vector<value_t> rho(r);

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(rank == 0) printf("solve_pcg: first vcycle!\n");
        MPI_Barrier(comm);
    }
//    for(i = 0; i < r.size(); i++)
//        printf("rho[%lu] = %f,\t r[%lu] = %f \n", i, rho[i], i, r[i]);

//    if(rank==0){
//        printf("Vcycle #: absolute residual \tconvergence factor\n");
//        printf("--------------------------------------------------------\n");
//    }
#endif

    std::vector<value_t> h(A->M);
    std::vector<value_t> p = rho;

    const double THRSHLD = init_dot * solver_tol * solver_tol;

    int i = 0;
    double rho_res = 0.0, pdoth = 0.0, alpha = 0.0, beta = 0.0;
    current_dot = init_dot;
//    previous_dot = init_dot;

    for(i = 0; i < solver_max_iter; i++){
        A->matvec(p, h);
        dotProduct(r, rho, &rho_res, comm);
        dotProduct(p, h,   &pdoth,   comm);
        alpha = rho_res / pdoth;

#pragma omp parallel for default(none) shared(u, r, p, h, alpha)
        for(index_t j = 0; j < u.size(); j++){
            u[j] -= alpha * p[j];
            r[j] -= alpha * h[j];
        }

        dotProduct(r, r, &current_dot, comm);

#ifdef __DEBUG1__
//        printf("rho_res = %e, pdoth = %e, alpha = %f \n", rho_res, pdoth, alpha);
//        print_vector(u, -1, "v inside solve_pcg", A->comm);
//        previous_dot = current_dot;

        // print the "absolute residual" and the "convergence factor":
//        if(rank==0) printf("Vcycle %d: %.10f  \t%.10f \n", i+1, sqrt(current_dot), sqrt(current_dot/previous_dot));
//        if(rank==0) printf("Vcycle %lu: aboslute residual = %.10f \n", i+1, sqrt(current_dot));
#endif

        if(current_dot < THRSHLD)
            break;

#ifdef __DEBUG1__
        if(verbose){
            MPI_Barrier(comm);
            if(!rank) printf("_______________________________ \n\n***** Vcycle %u *****\n", i+1);
            MPI_Barrier(comm);
        }
#endif

        // **************************************************************
        // Precondition
        // solve A * rho = r, in which rho is initialized to the 0 vector.
        // **************************************************************

//        std::fill(rho.begin(), rho.end(), 0);
//        vcycle(&grids[0], rho, r);
        rho = r;

        // **************************************************************

        dotProduct(r, rho, &beta, comm);
        beta /= rho_res;

//#pragma omp parallel for default(none) shared(u, p, rho, beta)
        for(index_t j = 0; j < u.size(); j++) {
            p[j] = rho[j] + beta * p[j];
        }
    } // for i

    // set number of iterations that took to find the solution.
    // only do the following if the end of the previous for loop was reached.
    if(i == solver_max_iter)
        i--;

//    double t_dif = MPI_Wtime() - t1;
//    print_time(t_dif, "solve_pcg", comm);

    if(rank==0){
        print_sep();
        printf("\nfinal:\nstopped at iteration    = %d \nfinal absolute residual = %e"
               "\nrelative residual       = %e \n", i+1, sqrt(current_dot), sqrt(current_dot / init_dot));
        print_sep();
    }

    // uncomment to print info for the lazy update feature
//    iter_num_lazy.emplace_back(i+1);
//    if(iter_num_lazy.size() == ITER_LAZY){
//        print_vector(iter_num_lazy, 0, "iter_num_lazy", comm);
//    }

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(verbose_solve) if(rank == 0) printf("solve_pcg: solve!\n");
        MPI_Barrier(comm);
    }
#endif

    // ************** scale u **************

    if(scale){
        scale_vector(u, A->inv_sq_diag_orig);
    }

    // ************** repartition u back **************

//    print_vector(u, 2, "final u before repartition_back_u", comm);

//    if(repartition){
//        repartition_back_u(u);
//    }

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(rank == 0) printf("solve_pcg: end!\n");
        MPI_Barrier(comm);

//        print_vector(u, 0, "final u", comm);
    }
#endif

//    if(rank==0) dollar::text(std::cout);

    return 0;
}


int saena_object::solve_pCG(std::vector<value_t>& u){

    auto *A = grids[0].A;
    vector<value_t> &rhs = grids[0].rhs;

    MPI_Comm comm = A->comm;
    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
//    print_vector(u, -1, "u", comm);
    if(verbose_solve){
        MPI_Barrier(comm);
        if(rank == 0) printf("solve_pcg: start!\n");
        MPI_Barrier(comm);
    }
#endif

    Rtransfer_time = 0;
    Ptransfer_time = 0;
    superlu_time = 0;
    vcycle_smooth_time = 0;
    vcycle_other_time = 0;
    double vcycle_time = 0;
    double matvec_time1 = 0;
    double dots = 0;

#ifdef PROFILE_TOTAL_PCG
    MPI_Barrier(comm);
    double t_pcg1 = omp_get_wtime();
#endif

    for(int l = 0; l < max_level; ++l){
        if(grids[l].active) {
            grids[l].P.tloc  = 0;
            grids[l].P.tcomm = 0;
            grids[l].P.trem  = 0;
            grids[l].P.ttot  = 0;
            grids[l].R.tloc  = 0;
            grids[l].R.tcomm = 0;
            grids[l].R.trem  = 0;
            grids[l].R.ttot  = 0;
        }
    }

    // ************** check u size **************
/*
    index_t u_size_local = u.size();
    index_t u_size_total;
    MPI_Allreduce(&u_size_local, &u_size_total, 1, MPI_UNSIGNED, MPI_SUM, grids[0].A->comm);
    if(grids[0].A->Mbig != u_size_total){
        if(rank==0) printf("Error: size of LHS (=%u) and the solution vector u (=%u) are not equal!\n", grids[0].A->Mbig, u_size_total);
        MPI_Finalize();
        return -1;
    }

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(rank == 0) printf("solve_pcg: check u size!\n");
        MPI_Barrier(comm);
    }
#endif
*/

    // ************** repartition u **************
    // todo: using repartition(), give the user the option of passing an initial guess for u. in that case comment
    //  out "initialize u" part.

/*
    std::fill(u.begin(), u.end(), 0);
    if(repartition)
        repartition_u(u);

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(verbose_solve) if(rank == 0) printf("solve_pcg: repartition u!\n");
        MPI_Barrier(comm);
    }
#endif
*/

    // ************** initialize u **************

    u.assign(A->M, 0);

    // ************** allocate memory for vcycle **************

    setup_vcycle_memory();

    // ************** solve **************

//    double t1 = MPI_Wtime();

//    double temp;
//    dot(rhs, rhs, &temp, comm);
//    if(rank==0) std::cout << "norm(rhs) = " << sqrt(temp) << std::endl;

    std::vector<value_t> r(A->M);
    A->residual(u, rhs, r);

    double init_dot = 0.0, current_dot = 0.0;
//    double previous_dot;
    dotProduct(r, r, &init_dot, comm);
    if(rank==0) printf("\ninitial residual = %e \n", sqrt(init_dot));

    // if max_level==0, it means only direct solver is being used inside the previous vcycle, and that is all needed.
    if(max_level == 0){
        vcycle(&grids[0], u, rhs);
        A->residual(u, rhs, r);
        dotProduct(r, r, &current_dot, comm);

#ifdef __DEBUG1__
//        print_vector(r, -1, "res", comm);
//        if(rank==0) std::cout << "dot = " << current_dot << std::endl;
#endif

        if(rank==0){
            print_sep();
            printf("\nfinal:\nonly using the direct solver! \nfinal absolute residual = %e"
                           "\nrelative residual       = %e \n", sqrt(current_dot), sqrt(current_dot / init_dot));
            print_sep();
        }

        // scale the solution u
        if(scale) {
            scale_vector(u, A->inv_sq_diag_orig);
        }

        // repartition u back
//        if(repartition){
//            repartition_back_u(u);
//        }

        return 0;
    }

    std::vector<value_t> rho(A->M, 0);
    vcycle(&grids[0], rho, r);

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(rank == 0) printf("solve_pcg: first vcycle!\n");
        MPI_Barrier(comm);
    }
//    for(i = 0; i < r.size(); i++)
//        printf("rho[%lu] = %f,\t r[%lu] = %f \n", i, rho[i], i, r[i]);

//    if(rank==0){
//        printf("Vcycle #: absolute residual \tconvergence factor\n");
//        printf("--------------------------------------------------------\n");
//    }
#endif

    std::vector<value_t> h(A->M);
    std::vector<value_t> p = rho;

    const double THRSHLD = init_dot * solver_tol * solver_tol;

    int i = 0;
    double rho_res = 0.0, pdoth = 0.0, alpha = 0.0, beta = 0.0;
    current_dot = init_dot;
//    previous_dot = init_dot;

    for(i = 0; i < solver_max_iter; i++){
#ifdef PROFILE_PCG
        MPI_Barrier(comm);
        double time_matvec1 = omp_get_wtime();
#endif

        A->matvec(p, h);

#ifdef PROFILE_PCG
        double time_matvec2 = omp_get_wtime();
        matvec_time1 += time_matvec2 - time_matvec1;
        MPI_Barrier(comm);
        double dot1 = omp_get_wtime();
#endif

        dotProduct(r, rho, &rho_res, comm);
        dotProduct(p, h,   &pdoth,   comm);

#ifdef PROFILE_PCG
        double dot2 = omp_get_wtime();
        dots += dot2 - dot1;
#endif

        alpha = rho_res / pdoth;

#pragma omp parallel for default(none) shared(u, r, p, h, alpha)
        for(index_t j = 0; j < u.size(); j++){
            u[j] -= alpha * p[j];
            r[j] -= alpha * h[j];
        }

#ifdef PROFILE_PCG
        MPI_Barrier(comm);
        dot1 = omp_get_wtime();
#endif

        dotProduct(r, r, &current_dot, comm);

#ifdef PROFILE_PCG
        dot2 = omp_get_wtime();
        dots += dot2 - dot1;
#endif

#ifdef __DEBUG1__
//        printf("rho_res = %e, pdoth = %e, alpha = %f \n", rho_res, pdoth, alpha);
//        print_vector(u, -1, "v inside solve_pcg", A->comm);
//        previous_dot = current_dot;

        // print the "absolute residual" and the "convergence factor":
//        if(rank==0) printf("%d: %.10f  \t%.10f \n", i+1, sqrt(current_dot), sqrt(current_dot/previous_dot));
//        if(rank==0) printf("%6d: aboslute = %.10f, relative = %.10f \n", i+1, sqrt(current_dot), sqrt(current_dot/init_dot));
#endif

//        if(rank==0) printf("%6d: aboslute = %.10f, relative = %.10f \n", i+1, sqrt(current_dot), sqrt(current_dot/init_dot));

        if(current_dot < THRSHLD)
            break;

#ifdef __DEBUG1__
        if(verbose){
            MPI_Barrier(comm);
            if(!rank) printf("_______________________________\n\n***** Vcycle %u *****\n", i+1);
            MPI_Barrier(comm);
        }
#endif

        // **************************************************************
        // Precondition
        // solve A * rho = r, in which rho is initialized to the 0 vector.
        // **************************************************************

#ifdef PROFILE_PCG
        double time_vcycle1 = omp_get_wtime();
#endif

        std::fill(rho.begin(), rho.end(), 0);
        vcycle(&grids[0], rho, r);

#ifdef PROFILE_PCG
        double time_vcycle2 = omp_get_wtime();
        vcycle_time += time_vcycle2 - time_vcycle1;
#endif

        // **************************************************************

#ifdef PROFILE_PCG
        MPI_Barrier(comm);
        dot1 = omp_get_wtime();
#endif

        dotProduct(r, rho, &beta, comm);

#ifdef PROFILE_PCG
        dot2 = omp_get_wtime();
        dots += dot2 - dot1;
#endif

        beta /= rho_res;

//#pragma omp parallel for default(none) shared(u, p, rho, beta)
        for(index_t j = 0; j < u.size(); j++) {
            p[j] = rho[j] + beta * p[j];
        }
    } // for i

    // set number of iterations that took to find the solution.
    // only do the following if the end of the previous for loop was reached.
    if(i == solver_max_iter)
        i--;

//    double t_dif = MPI_Wtime() - t1;
//    print_time(t_dif, "solve_pcg", comm);

    if(rank==0){
//        double t_pcg2 = omp_get_wtime();
        print_sep();
        printf("\nfinal:\nstopped at iteration    = %d \nfinal absolute residual = %e"
                       "\nrelative residual       = %e \n", i+1, sqrt(current_dot), sqrt(current_dot / init_dot));
//        printf("total   time per iteration = %e \n", (t_pcg2 - t_pcg1)/(i+1));
//        printf("vcycle  time per iteration = %e \n", vcycle_time/(i+1));
//        printf("superlu time per iteration = %e \n", superlu_time/(i+1));
//        printf("matvec1 time per iteration = %e \n", matvec_time1/(i+1));
//        printf("smooth  time per iteration = %e \n", vcycle_smooth_time/(i+1));
        print_sep();
    }

    // uncomment to print info for the lazy update feature
//    iter_num_lazy.emplace_back(i+1);
//    if(iter_num_lazy.size() == ITER_LAZY){
//        print_vector(iter_num_lazy, 0, "iter_num_lazy", comm);
//    }

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(verbose_solve) if(rank == 0) printf("solve_pcg: solve!\n");
        MPI_Barrier(comm);
    }
#endif

    // ************** scale u **************

//    writeVectorToFile(u, "sol", comm);

    if(scale){
        scale_vector(u, A->inv_sq_diag_orig);
    }

    // ************** repartition u back **************

//    print_vector(u, 2, "final u before repartition_back_u", comm);

//    if(repartition){
//        repartition_back_u(u);
//    }

#ifdef PROFILE_TOTAL_PCG
    double t_pcg2 = omp_get_wtime();
#endif

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(rank == 0) printf("solve_pcg: end!\n");
        MPI_Barrier(comm);

//        print_vector(u, 0, "final u", comm);
    }
#endif

    for(int k = 0; k < 3; ++k){
        // i = 0: average time
        // i = 1: min time
        // i = 2: max time

#ifdef PROFILE_VCYCLE
        if(!rank) printf("\nRtransfer\nPtransfer\nsmooth\nsuperlu\nvcycle_other\n");
        print_time(Rtransfer_time / (i+1),     "Rtransfer",    comm, true, false, k);
        print_time(Ptransfer_time / (i+1),     "Ptransfer",    comm, true, false, k);
        print_time(vcycle_smooth_time / (i+1), "smooth",       comm, true, false, k);
        print_time(superlu_time / (i+1),       "superlu",      comm, true, false, k);
        print_time(vcycle_other_time / (i+1),  "vcycle_other", comm, true, false, k);
#endif

#ifdef PROFILE_PCG
        if(!rank) printf("\nvcycle_pCG\nL0matvec\ndots\n");
        print_time(vcycle_time / (i+1),        "vcycle_pCG",   comm, true, false, k);
        print_time(matvec_time1 / (i+1),       "L0matvec",     comm, true, false, k);
        print_time(dots / (i+1),               "dots",         comm, true, false, k);
#endif

#ifdef PROFILE_TOTAL_PCG
        if(!rank) printf("\npCG_total\n");
        print_time((t_pcg2 - t_pcg1) / (i+1),  "pCG_total",    comm, true, false, k);
        if(!rank) print_sep();
#endif

    }

#if 0
    if(!rank) printf("\nP matvec:\n");
    if(!rank) printf("loc\ncomm\nrem\ntot\n");
    for(int l = 0; l < max_level; ++l){
//    for(int l = 0; l < 1; ++l){
        if(grids[l].active) {
            if(!rank) printf("\nlevel %d: \n", l);
            if(!rank) printf("matvec_comm_sz: %d\n", grids[l].P.matvec_comm_sz);
            print_time_ave(grids[l].P.tloc / (i+1),  "Ploc",  grids[l].A->comm, true, false);
            print_time_ave(grids[l].P.tcomm / (i+1), "Pcomm", grids[l].A->comm, true, false);
            print_time_ave(grids[l].P.trem / (i+1),  "Prem",  grids[l].A->comm, true, false);
            print_time_ave(grids[l].P.ttot / (i+1),  "Ptot",  grids[l].A->comm, true, false);
        }
    }

    if(!rank) printf("\nR matvec:\n");
    if(!rank) printf("loc\ncomm\nrem\ntot\n");
    for(int l = 0; l < max_level; ++l){
//    for(int l = 0; l < 1; ++l){
        if(grids[l].active) {
            if(!rank) printf("\nlevel %d: \n", l);
            if(!rank) printf("matvec_comm_sz: %d\n", grids[l].R.matvec_comm_sz);
            print_time_ave(grids[l].R.tloc / (i+1),  "Rloc",  grids[l].A->comm, true, false);
            print_time_ave(grids[l].R.tcomm / (i+1), "Rcomm", grids[l].A->comm, true, false);
            print_time_ave(grids[l].R.trem / (i+1),  "Rrem",  grids[l].A->comm, true, false);
            print_time_ave(grids[l].R.ttot / (i+1),  "Rtot",  grids[l].A->comm, true, false);
        }
    }
#endif

    // call petsc solver
//    std::vector<double> u_petsc(rhs.size());
//    petsc_solve(A, rhs, u_petsc, solver_tol);

    return 0;
}


// int saena_object::solve_pcg_update(std::vector<value_t>& u, saena_matrix* A_new)
/*
int saena_object::solve_pcg_update(std::vector<value_t>& u, saena_matrix* A_new){

    MPI_Comm comm = grids[0].A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    unsigned long i, j;
    bool solve_verbose = false;

    // ************** update A_new **************
    // this part is specific to solve_pcg_update(), in comparison to solve_pcg().
    // the difference between this function and solve_pcg(): residual in computed for A_new, instead of the original A,
    // so the solve stops when it reaches the same tolerance as the norm of the residual for A_new.

    grids[0].A_new = A_new;

    // ************** check u size **************

    unsigned int u_size_local = u.size();
    unsigned int u_size_total;
    MPI_Allreduce(&u_size_local, &u_size_total, 1, MPI_UNSIGNED, MPI_SUM, grids[0].A->comm);
    if(grids[0].A->Mbig != u_size_total){
        if(rank==0) printf("Error: size of LHS (=%u) and the solution vector u (=%u) are not equal!\n", grids[0].A->Mbig, u_size_total);
        MPI_Finalize();
        return -1;
    }

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: check u size!\n");

    // ************** repartition u **************

    if(repartition)
        repartition_u(u);

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: repartition u!\n");

    // ************** solve **************

//    double temp;
//    dot(rhs, rhs, &temp, comm);
//    if(rank==0) std::cout << "norm(rhs) = " << sqrt(temp) << std::endl;

    std::vector<double> res(grids[0].A->M);
    grids[0].A_new->residual(u, grids[0].rhs, res);
    double initial_dot, current_dot;
    dotProduct(res, res, &initial_dot, comm);
    if(rank==0) std::cout << "******************************************************" << std::endl;
    if(rank==0) printf("\ninitial residual = %e \n\n", sqrt(initial_dot));

    // if max_level==0, it means only direct solver is being used inside the previous vcycle, and that is all needed.
    if(max_level == 0){

        vcycle(&grids[0], u, grids[0].rhs);
        grids[0].A_new->residual(u, grids[0].rhs, res);
        dotProduct(res, res, &current_dot, comm);

        if(rank==0){
            std::cout << "******************************************************" << std::endl;
            printf("\nfinal:\nonly using the direct solver! \nfinal absolute residual = %e"
                           "\nrelative residual       = %e \n\n", sqrt(current_dot), sqrt(current_dot/initial_dot));
            std::cout << "******************************************************" << std::endl;
        }

        // repartition u back
        if(repartition)
            repartition_back_u(u);

        return 0;
    }

    std::vector<double> rho(grids[0].A->M, 0);
    vcycle(&grids[0], rho, res);

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: first vcycle!\n");

//    for(i = 0; i < res.size(); i++)
//        printf("rho[%lu] = %f,\t res[%lu] = %f \n", i, rho[i], i, res[i]);

    std::vector<double> h(grids[0].A->M);
    std::vector<double> p(grids[0].A->M);
    p = rho;

    double rho_res, pdoth, alpha, beta;
    for(i=0; i<solver_max_iter; i++){
        grids[0].A_new->matvec(p, h);
        dotProduct(res, rho, &rho_res, comm);
        dotProduct(p, h, &pdoth, comm);
        alpha = rho_res / pdoth;
//        printf("rho_res = %e, pdoth = %e, alpha = %f \n", rho_res, pdoth, alpha);

#pragma omp parallel for
        for(j = 0; j < u.size(); j++){
            u[j] -= alpha * p[j];
            res[j] -= alpha * h[j];
        }

        dotProduct(res, res, &current_dot, comm);
        if( current_dot/initial_dot < solver_tol * solver_tol )
            break;

        rho.assign(rho.size(), 0);
        vcycle(&grids[0], rho, res);
        dotProduct(res, rho, &beta, comm);
        beta /= rho_res;

#pragma omp parallel for
        for(j = 0; j < u.size(); j++)
            p[j] = rho[j] + beta * p[j];
//        printf("beta = %e \n", beta);
    }

    // set number of iterations that took to find the solution
    // only do the following if the end of the previous for loop was reached.
    if(i == solver_max_iter)
        i--;

    if(rank==0){
        std::cout << "******************************************************" << std::endl;
        printf("\nfinal:\nstopped at iteration    = %ld \nfinal absolute residual = %e"
                       "\nrelative residual       = %e \n\n", ++i, sqrt(current_dot), sqrt(current_dot/initial_dot));
        std::cout << "******************************************************" << std::endl;
    }

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: solve!\n");

    // ************** repartition u back **************

    if(repartition)
        repartition_back_u(u);

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: repartition back u!\n");

    return 0;
}
*/

// ************************************************
// GMRES related functions
// ************************************************

// The template is used from:
// https://math.nist.gov/iml++/gmres.h.txt
// The implementation is explained on page 17 (27 of pdf) of:
// http://www.netlib.org/templates/templates.pdf

// ************************************************

void saena_object::GMRES_update(std::vector<double> &x, index_t k, saena_matrix_dense &h, std::vector<double> &s, std::vector<std::vector<double>> &v){
    // ***************************************
    // GMRES_update(u, i, H, s, v)
    // x_i = x_0 + y_1 * v_1 + ... + y_i * v_i
    // x_0 = x_i
    // ***************************************

//    std::cout << __func__ << std::endl;

    // todo: pre-allocate y. it is being created and allocated each time this function is being called.
    std::vector<double> y = s;

    // Backsolve:
    for (long i = k; i >= 0; i--) {
//        std::cout << "\ni:" << i << ", h.get(i, i): " << h.get(i, i) << std::endl;
        y[i] /= h.get(i, i); //y[i] /= h(i,i);
        for (long j = i - 1; j >= 0; j--){
//            std::cout << "(i,j): " << i << "," << j << ", \th.get(j, i): " << h.get(j, i) << ", \ty[i]: " << y[i] << ", \ty[j]: " << y[j] << std::endl;
            y[j] -= h.get(j, i) * y[i];
        }
    }

    for (long j = 0; j <= k; j++){
        // x += v[j] * y[j];
        // scale each vector v[j] by scalar y[j]
        // x and v[j] are vectors. y[j] is a scalar.
        scale_vector_scalar(v[j], y[j], x, true);
    }
}


void saena_object::GeneratePlaneRotation(double &dx, double &dy, double &cs, double &sn){
    // set cs and sn

    if (dy == 0.0) {
        cs = 1.0;
        sn = 0.0;
    } else if (fabs(dy) > fabs(dx)) {
        double temp = dx / dy;
        sn = 1.0 / sqrt( 1.0 + temp*temp );
        cs = temp * sn;
    } else {
        double temp = dy / dx;
        cs = 1.0 / sqrt( 1.0 + temp*temp );
        sn = temp * cs;
    }
}


void saena_object::ApplyPlaneRotation(double &dx, double &dy, const double &cs, const double &sn){
    // set dx and dy

//    std::cout << "dx: " << dx << ", \tdy: " << dy << ", \tcs: " << cs << ", \tsn: " << sn << std::endl;

    double temp = cs * dx + sn * dy;
    dy = -sn * dx + cs * dy;
    dx = temp;
}


int saena_object::GMRES(std::vector<double> &u){
    // GMRES proconditioned with AMG
//    Preconditioner &M, Matrix &H;

    saena_matrix *A = grids[0].A;
    vector<value_t> &rhs = grids[0].rhs;

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(rank == 0) printf("pGMRES: start\n");
        MPI_Barrier(comm);
    }
#endif

    int     m        = 500; // when to restart.
    index_t size     = A->M;
    double  tol      = solver_tol;
    int     max_iter = solver_max_iter;

    double  resid, beta;
    long i, j, k;

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(rank == 0) printf("m: %u, \tsize: %u, \ttol: %e, \tmax_iter: %u \n", m, size, tol, max_iter);
        if(rank == 0) printf("pGMRES: AMG as preconditioner\n");
        MPI_Barrier(comm);
    }
#endif

    // use AMG as preconditioner
    // *************************
//    std::vector<double> r = M.solve(rhs - A * u);

    std::vector<double> res(size), r(size);
    u.assign(size, 0); // initial guess // todo: decide where to do this.
    A->residual_negative(u, rhs, res);
//    vcycle(&grids[0], r, res); //todo: M should be used here.
    r = res;

    // *************************

//    Vector *v = new Vector[m+1];
    std::vector<std::vector<value_t>> v(m + 1, std::vector<value_t>(size)); // todo: decide how to allocate for v.

//    double normb = norm(M.solve(rhs));
    double normb = pnorm(rhs, comm); // todo: this is different from the above line

    if (normb == 0.0){
        normb = 1;
    }

//    if(rank==0) printf("******************************************************");
    if(rank==0) printf("\ninitial residual = %e \n", normb);

    beta = pnorm(r, comm);
    resid = beta / normb;
    if (resid <= tol) {
        return 0;
    }

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(rank == 0) printf("pGMRES: Hessenberg matrix H(%u, %u)\n", m, m);
        MPI_Barrier(comm);
    }
#endif

    // initialize the Hessenberg matrix H
    // **********************************
    saena_matrix_dense H(m + 1, m + 1, comm);
//    #pragma omp parallel for
    for(i = 0; i < m; i++){
        std::fill(&H.entry[i][0], &H.entry[i][m + 1], 0);
    }

    // **********************************

    double tmp_scalar1 = 0, tmp_scalar2 = 0;
    std::vector<double> s(m + 1), cs(m + 1), sn(m + 1), w(size), temp(size);
    j = 1;
    while (j <= max_iter) {

#ifdef __DEBUG1__
        if (verbose_solve) {
            MPI_Barrier(comm);
            if (rank == 0) printf("pGMRES: j = %ld: v[0] \n", j);
            MPI_Barrier(comm);
        }
#endif

        // v[0] = r / beta
        scale_vector_scalar(r, 1.0 / beta, v[0]);

        // s = norm(r) * e_1
        std::fill(s.begin(), s.end(), 0.0); // s = 0.0;
        s[0] = beta;

        // this for loop is used to restart after m steps
        // **********************************************
        for (i = 0; i < m && j <= max_iter; i++, j++) {

#ifdef __DEBUG1__
            if (verbose_solve) {
                MPI_Barrier(comm);
                if (rank == 0) printf("pGMRES: j = %ld: for i = %ld: AMG \n", j, i);
                MPI_Barrier(comm);
            }
#endif

            // w = M.solve(A * v[i]);
            A->matvec(v[i], temp);
            std::fill(w.begin(), w.end(), 0); // todo
//            vcycle(&grids[0], w, temp);
            w = temp;

#ifdef __DEBUG1__
            if (verbose_solve) {
                MPI_Barrier(comm);
                if (rank == 0) printf("pGMRES: j = %ld: for i = %ld: for \n", j, i);
                MPI_Barrier(comm);
            }
#endif

            for (k = 0; k <= i; k++) {
                // compute H(k, i) = dot(w, v[k]);
                dotProduct(w, v[k], &H.entry[k][i], comm);

                // w -= H(k, i) * v[k];
                scale_vector_scalar(v[k], -H.get(k, i), w, true);
            }

#ifdef __DEBUG1__
            if (verbose_solve) {
                MPI_Barrier(comm);
                if (rank == 0) printf("pGMRES: j = %ld: for i = %ld: scale \n", j, i);
                MPI_Barrier(comm);
            }
#endif

//            MPI_Barrier(comm);
//            if(rank == 1) printf("%ld %ld %e", i+1, i, H.get(i + 1, i));
//            std::cout << i+1 << " " << i << " " << H.get(i + 1, i) << std::endl;
//            MPI_Barrier(comm);

            // compute H(i+1, i) = ||w||
            H.set(i + 1, i, pnorm(w, comm));

            if(fabs(H.get(i + 1, i)) < 1e-15){
                printf("EXIT_FAILURE: Division by zero inside pGMRES: H[%ld, %ld] = %e \n", i+1, i, H.get(i + 1, i));
                exit(EXIT_FAILURE);
            }

            // v[i+1] = w / H(i+1, i)
            scale_vector_scalar(w, 1.0 / H.get(i + 1, i), v[i + 1]);

#ifdef __DEBUG1__
            if (verbose_solve) {
                MPI_Barrier(comm);
                if (rank == 0) printf("pGMRES: j = %ld: for i = %ld: PlaneRotation \n", j, i);
                MPI_Barrier(comm);
            }
#endif

            for (k = 0; k < i; k++) {
                ApplyPlaneRotation(H.entry[k][i], H.entry[k + 1][i], cs[k], sn[k]);
            }

            GeneratePlaneRotation(H.entry[i][i], H.entry[i + 1][i], cs[i], sn[i]);
            ApplyPlaneRotation(H.entry[i][i], H.entry[i + 1][i], cs[i], sn[i]);
            ApplyPlaneRotation(s[i], s[i + 1], cs[i], sn[i]);

            resid = fabs(s[i + 1]) / normb;

#ifdef __DEBUG1__
            if (verbose_solve) {
                MPI_Barrier(comm);
                if (rank == 0) printf("%e\n", resid);
                if (rank == 0) printf("resid: %e \t1st resid\n", resid);
                MPI_Barrier(comm);
            }
#endif

            if (resid < tol) {
                GMRES_update(u, i, H, s, v);
                goto gmres_out;
            }
        }

#ifdef __DEBUG1__
        if(verbose_solve){
            MPI_Barrier(comm);
            if(rank == 0) printf("pGMRES: j = %ld: update \n", j);
            MPI_Barrier(comm);
        }
#endif

        GMRES_update(u, i - 1, H, s, v);

#ifdef __DEBUG1__
        if(verbose_solve){
            MPI_Barrier(comm);
            if(rank == 0) printf("pGMRES: j = %ld: AMG as preconditioner \n", j);
            MPI_Barrier(comm);
        }
#endif

        // r = M.solve(rhs - A * u);
        A->residual_negative(u, rhs, res);
//        vcycle(&grids[0], r, res);
        r = res;

        beta  = pnorm(r, comm);
        resid = beta / normb;

#ifdef __DEBUG1__
        if(verbose_solve){
            if(rank == 0) printf("resid: %e \t2nd resid\n", resid);
        }
#endif

        if (resid < tol) {
            goto gmres_out;
        }
    }

    // the exit label to be used by "goto".
    gmres_out:

    // ************** scale u **************

    if(scale) {
        scale_vector(u, A->inv_sq_diag_orig);
    }

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(rank == 0) printf("pGMRES: end");
//        if(rank == 0) printf("pGMRES: end. did not reach the accuracy. relative residual: %e, iter = %u \n", resid, j);
        MPI_Barrier(comm);
    }
#endif

    if(rank==0){
        printf("\n******************************************************\n");
        printf("\nfinal:\nstopped at iteration = %ld \n", j);
        printf("relative residual    = %e \n", resid);
//        printf("final absolute residual = %e", beta);
        printf("\n******************************************************\n");
    }

//    max_iter = j;
//    tol = resid;
    return 0;
}


//template < class Operator, class Preconditioner, class Matrix>
//int GMRES(std::vector<double> &u, std::vector<double> &rhs,
//                        const Preconditioner &M, Matrix &H, int &m, int &max_iter, double &tol){
int saena_object::pGMRES(std::vector<double> &u){
    // GMRES proconditioned with AMG
//    Preconditioner &M, Matrix &H;

    saena_matrix *A = grids[0].A;
    vector<value_t> &rhs = grids[0].rhs;

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(rank == 0) printf("pGMRES: start\n");
        MPI_Barrier(comm);
    }
#endif

    int     m        = 500; // when to restart.
    index_t size     = A->M;
    double  tol      = solver_tol;
    int     max_iter = solver_max_iter;

    double  resid, beta;
    long i, j, k;

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(rank == 0) printf("m: %u, \tsize: %u, \ttol: %e, \tmax_iter: %u \n", m, size, tol, max_iter);
        if(rank == 0) printf("pGMRES: AMG as preconditioner\n");
        MPI_Barrier(comm);
    }
#endif

    // use AMG as preconditioner
    // *************************
//    std::vector<double> r = M.solve(rhs - A * u);

    // allocate memory for vcycle
    setup_vcycle_memory();

    std::vector<double> res(size), r(size);
    u.assign(size, 0); // initial guess // todo: decide where to do this.
    A->residual_negative(u, rhs, res);
    vcycle(&grids[0], r, res); //todo: M should be used here.

    // *************************

//    Vector *v = new Vector[m+1];
    std::vector<std::vector<value_t>> v(m + 1, std::vector<value_t>(size)); // todo: decide how to allocate for v.

//    double normb = norm(M.solve(rhs));
    double normb = pnorm(res, comm); // todo: this is different from the above line

    if (normb == 0.0){
        normb = 1;
    }

//    if(rank==0) printf("******************************************************");
    if(rank==0) printf("\ninitial residual = %e \n", normb);

    beta = pnorm(r, comm);
    resid = beta / normb;
    if (resid <= tol) {
        return 0;
    }

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(rank == 0) printf("pGMRES: Hessenberg matrix H(%u, %u)\n", m, m);
        MPI_Barrier(comm);
    }
#endif

    // initialize the Hessenberg matrix H
    // **********************************
    saena_matrix_dense H(m + 1, m + 1, comm);
//    #pragma omp parallel for
    for(i = 0; i < m; i++){
        std::fill(&H.entry[i][0], &H.entry[i][m + 1], 0);
    }

    // **********************************

    double tmp_scalar1 = 0, tmp_scalar2 = 0;
    std::vector<double> s(m + 1), cs(m + 1), sn(m + 1), w(size), temp(size);
    j = 1;
    while (j <= max_iter) {

#ifdef __DEBUG1__
        if (verbose_solve) {
            MPI_Barrier(comm);
            if (rank == 0) printf("pGMRES: j = %ld: v[0] \n", j);
            MPI_Barrier(comm);
        }
#endif

        // v[0] = r / beta
        scale_vector_scalar(r, 1.0 / beta, v[0]);

        // s = norm(r) * e_1
        std::fill(s.begin(), s.end(), 0.0); // s = 0.0;
        s[0] = beta;

        // this for loop is used to restart after m steps
        // **********************************************
        for (i = 0; i < m && j <= max_iter; i++, j++) {

#ifdef __DEBUG1__
            if (verbose_solve) {
                MPI_Barrier(comm);
                if (rank == 0) printf("pGMRES: j = %ld: for i = %ld: AMG \n", j, i);
                MPI_Barrier(comm);
            }
#endif

            // w = M.solve(A * v[i]);
            A->matvec(v[i], temp);
            std::fill(w.begin(), w.end(), 0); // todo
            vcycle(&grids[0], w, temp); //todo: M should be used here.

#ifdef __DEBUG1__
            if (verbose_solve) {
                MPI_Barrier(comm);
                if (rank == 0) printf("pGMRES: j = %ld: for i = %ld: for \n", j, i);
                MPI_Barrier(comm);
            }
#endif

            for (k = 0; k <= i; k++) {
                // compute H(k, i) = dot(w, v[k]);
                dotProduct(w, v[k], &H.entry[k][i], comm);

                // w -= H(k, i) * v[k];
                scale_vector_scalar(v[k], -H.get(k, i), w, true);
            }

#ifdef __DEBUG1__
            if (verbose_solve) {
                MPI_Barrier(comm);
                if (rank == 0) printf("pGMRES: j = %ld: for i = %ld: scale \n", j, i);
                MPI_Barrier(comm);
            }
#endif

//            MPI_Barrier(comm);
//            if(rank == 1) printf("%ld %ld %e", i+1, i, H.get(i + 1, i));
//            std::cout << i+1 << " " << i << " " << H.get(i + 1, i) << std::endl;
//            MPI_Barrier(comm);

            // compute H(i+1, i) = ||w||
            H.set(i + 1, i, pnorm(w, comm));

            if(fabs(H.get(i + 1, i)) < 1e-15){
                printf("EXIT_FAILURE: Division by zero inside pGMRES: H[%ld, %ld] = %e \n", i+1, i, H.get(i + 1, i));
                exit(EXIT_FAILURE);
            }

            // v[i+1] = w / H(i+1, i)
            scale_vector_scalar(w, 1.0 / H.get(i + 1, i), v[i + 1]);

#ifdef __DEBUG1__
            if (verbose_solve) {
                MPI_Barrier(comm);
                if (rank == 0) printf("pGMRES: j = %ld: for i = %ld: PlaneRotation \n", j, i);
                MPI_Barrier(comm);
            }
#endif

            for (k = 0; k < i; k++) {
                ApplyPlaneRotation(H.entry[k][i], H.entry[k + 1][i], cs[k], sn[k]);
            }

            GeneratePlaneRotation(H.entry[i][i], H.entry[i + 1][i], cs[i], sn[i]);
            ApplyPlaneRotation(H.entry[i][i], H.entry[i + 1][i], cs[i], sn[i]);
            ApplyPlaneRotation(s[i], s[i + 1], cs[i], sn[i]);

            resid = fabs(s[i + 1]) / normb;

#ifdef __DEBUG1__
            if (verbose_solve) {
                MPI_Barrier(comm);
                if (rank == 0) printf("%e\n", resid);
                if (rank == 0) printf("resid: %e \t1st resid\n", resid);
                MPI_Barrier(comm);
            }
#endif

            if (resid < tol) {
                GMRES_update(u, i, H, s, v);
                goto gmres_out;
            }
        }

#ifdef __DEBUG1__
        if(verbose_solve){
            MPI_Barrier(comm);
            if(rank == 0) printf("pGMRES: j = %ld: update \n", j);
            MPI_Barrier(comm);
        }
#endif

        GMRES_update(u, i - 1, H, s, v);

#ifdef __DEBUG1__
        if(verbose_solve){
            MPI_Barrier(comm);
            if(rank == 0) printf("pGMRES: j = %ld: AMG as preconditioner \n", j);
            MPI_Barrier(comm);
        }
#endif

        // r = M.solve(rhs - A * u);
        A->residual_negative(u, rhs, res);
        vcycle(&grids[0], r, res); //todo: M should be used here.

        beta  = pnorm(r, comm);
        resid = beta / normb;

#ifdef __DEBUG1__
        if(verbose_solve){
            if(rank == 0) printf("resid: %e \t2nd resid\n", resid);
        }
#endif

        if (resid < tol) {
            goto gmres_out;
        }
    }

    // the exit label to be used by "goto".
    gmres_out:

    // ************** scale u **************

    if(scale) {
        scale_vector(u, A->inv_sq_diag_orig);
    }

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(rank == 0) printf("pGMRES: end");
//        if(rank == 0) printf("pGMRES: end. did not reach the accuracy. relative residual: %e, iter = %u \n", resid, j);
        MPI_Barrier(comm);
    }
#endif

    if(rank==0){
        printf("\n******************************************************\n");
        printf("\nfinal:\nstopped at iteration = %ld \n", j);
        printf("relative residual    = %e \n", resid);
//        printf("final absolute residual = %e", beta);
        printf("\n******************************************************\n");
    }

//    max_iter = j;
//    tol = resid;
    return 0;
}

// ************************************************
// End of GMRES related functions
// ************************************************
