#include <cmath>
#include "superlu_ddefs.h"
//#include "superlu_defs.h"

#include "saena_object.h"
#include "saena_matrix.h"
#include "grid.h"
#include "aux_functions.h"
#include "dollar.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mpi.h>

// use this to store number of iterations for the lazy-update experiment.
std::vector<int> iter_num_lazy;


int saena_object::solve_coarsest_CG(saena_matrix* A, std::vector<value_t>& u, std::vector<value_t>& rhs){
    // this is CG.
    // u is zero in the beginning. At the end, it is the solution.

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(verbose_solve_coarse && rank==0) printf("start of solve_coarsest_CG()\n");
#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        MPI_Barrier(comm);
        if(rank==0)
        MPI_Barrier(comm);
    }
#endif

    // since u is zero, res = -rhs, and the residual in this function is the negative of what I have in this library.
    std::vector<value_t> res = rhs;

    double initial_dot;
    dotProduct(res, res, &initial_dot, comm);
//    if(rank==0) std::cout << "\nsolveCoarsest: initial norm(res) = " << sqrt(initial_dot) << std::endl;

    double dot = initial_dot;
    int max_iter = CG_max_iter;
    if (dot < CG_tol*CG_tol)
        max_iter = 0;

    std::vector<value_t> dir(A->M);
    dir = res;

//    double dot2;
    std::vector<value_t> res2(A->M);

    double factor, dot_prev;
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
        for(index_t j = 0; j < A->M; j++){
            u[j]   += factor * dir[j];
            res[j] -= factor * matvecTemp[j];
        }

        dot_prev = dot;
        dotProduct(res, res, &dot, comm);
//        if(rank==0) std::cout << "absolute norm(res) = " << sqrt(dot) << "\t( r_i / r_0 ) = " << sqrt(dot)/initialNorm << "  \t( r_i / r_i-1 ) = " << sqrt(dot)/sqrt(dot_prev) << std::endl;
//        if(rank==0) std::cout << sqrt(dot)/initialNorm << std::endl;

        if(verbose_solve_coarse && rank==0)
            std::cout << "sqrt(dot)/sqrt(initial_dot) = " << sqrt(dot/initial_dot) << "  \tCG_tol = " << CG_tol << std::endl;
#ifdef __DEBUG1__
        if(verbose_solve_coarse) {
            MPI_Barrier(comm);
            if(rank==0)
                MPI_Barrier(comm);
        }
#endif

//        A->residual(u, rhs, res2);
//        dotProduct(res2, res2, &dot2, comm);
//        if(rank==0) std::cout << "norm(res) = " << sqrt(dot2) << std::endl;

        if (dot/initial_dot < CG_tol*CG_tol)
            break;

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


int saena_object::solve_coarsest_SuperLU(saena_matrix *A, std::vector<value_t> &u, std::vector<value_t> &rhs){

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        MPI_Barrier(comm);
        if(rank==0){
            printf("start of solve_coarsest_SuperLU()\n");
        }
        MPI_Barrier(comm);
//    print_vector(rhs, -1, "rhs passed to superlu", comm);
    }
#endif

    superlu_dist_options_t options;
    SuperLUStat_t stat;
    SuperMatrix A_SLU;
    ScalePermstruct_t ScalePermstruct;
    LUstruct_t LUstruct;
    SOLVEstruct_t SOLVEstruct;
    gridinfo_t grid;
    double   *berr;
//    double   *b, *xtrue;
    double   *b;
    int      m, n, m_loc, nnz_loc;
    int      nprow, npcol;
//    int      iam, info, ldb, ldx, nrhs;
    int      iam, info, ldb, nrhs;
//    char     **cpp, c;
//    FILE *fp, *fopen();
//    FILE *fp;
//    int cpp_defs();

    nprow = nprocs;  /* Default process rows.      */
    npcol = 1;  /* Default process columns.   */
    nrhs  = 1;  /* Number of right-hand side. */

    /* ------------------------------------------------------------
       INITIALIZE MPI ENVIRONMENT.
       ------------------------------------------------------------*/
//    MPI_Init( &argc, &argv );

//    char* file_name(argv[5]);
//    saena::matrix A_saena (file_name, comm);
//    A_saena.assemble();
//    A_saena.print_entry(-1);
//    if(rank==0) printf("after matrix assemble.\n");

    /*
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
*/
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
//    printf("iam = %d, nprow = %d, npcol = %d \n", iam, nprow, npcol);
//    if ( iam >= nprow * npcol )	goto out;

#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        MPI_Barrier(comm);
        if(!iam){
            int v_major, v_minor, v_bugfix;
            superlu_dist_GetVersionNumber(&v_major, &v_minor, &v_bugfix);
            printf("Library version:\t%d.%d.%d\n", v_major, v_minor, v_bugfix);

//        printf("Input matrix file:\t%s\n", *cpp);
            printf("Process grid:\t\t%d X %d\n", nprow, npcol);
            fflush(stdout);
        }
        MPI_Barrier(comm);
    }
#endif

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

#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        MPI_Barrier(comm);
        if(rank==0) printf("PASS THE MATRIX FROM SAENA. \n");
        MPI_Barrier(comm);
    }
#endif

    m = A->Mbig;
    m_loc = A->M;
    n = m;
    nnz_loc = A->nnz_l;
    ldb = m_loc;

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

    auto* rowptr = (int_t *) intMalloc_dist(m_loc+1);
    auto* nzval_loc = (double *) doubleMalloc_dist(nnz_loc);
    auto* colind = (int_t *) intMalloc_dist(nnz_loc);

    // Do this line to avoid this subtraction for each entry in the next "for" loop.
    int *nnz_per_row_p = &nnz_per_row[0] - fst_row;

    for(nnz_t i = 0; i < nnz_loc; i++){
        nzval_loc[i] = entry_temp[i].val;
//        nnz_per_row[entry_temp[i].row - fst_row]++;
        nnz_per_row_p[entry_temp[i].row]++;
        colind[i] = entry_temp[i].col;
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

    if ( !(berr = doubleMalloc_dist(nrhs)) )
    ABORT("Malloc fails for berr[].");

    /* ------------------------------------------------------------
       SET THE RIGHT HAND SIDE.
       ------------------------------------------------------------*/

    b = &rhs[0];
    u = rhs;

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
        if(rank==0) printf("SOLVE THE LINEAR SYSTEM. \n");
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
        if(rank==0) printf("SOLVE THE LINEAR SYSTEM: step 1 \n");
        MPI_Barrier(comm);
    }
#endif

    // Initialize the statistics variables.
    PStatInit(&stat);
    // Call the linear equation solver.
    pdgssvx(&options, &A_SLU, &ScalePermstruct, b, ldb, nrhs, &grid,
            &LUstruct, &SOLVEstruct, berr, &stat, &info);

#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        MPI_Barrier(comm);
        if(rank==0) printf("SOLVE THE LINEAR SYSTEM: step 2 \n");
        MPI_Barrier(comm);
    }
#endif

    // put the solution in u
    // b points to rhs. after calling pdgssvx it will contain the solution.
    u.swap(rhs);

//    print_vector(u, -1, "u computed in superlu", comm);

#ifdef __DEBUG1__
    if(verbose_solve_coarse) {
        MPI_Barrier(comm);
        if(rank==0) printf("SOLVE THE LINEAR SYSTEM: step 3 \n");
        MPI_Barrier(comm);
    }
#endif

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
    Destroy_CompRowLoc_Matrix_dist(&A_SLU);
    ScalePermstructFree(&ScalePermstruct);
    Destroy_LU(n, &grid, &LUstruct);
    LUstructFree(&LUstruct);
    if ( options.SolveInitialized ) {
        dSolveFinalize(&options, &SOLVEstruct);
    }
    SUPERLU_FREE(berr);

    // don't need these two.
//    SUPERLU_FREE(b);
//    SUPERLU_FREE(xtrue);

    /* ------------------------------------------------------------
       RELEASE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------*/
    out:
    superlu_gridexit(&grid);

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
        if(rank==0) printf("end of solve_coarsest_SuperLU()\n");
        MPI_Barrier(comm);
    }
#endif

    return 0;
}


// int SaenaObject::solveCoarsest
/*
int SaenaObject::solveCoarsest(SaenaMatrix* A, std::vector<double>& x, std::vector<double>& b, int& max_iter, double& tol, MPI_Comm comm){
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    long i, j;

    double normb_l, normb;
    normb_l = 0;
    for(i=0; i<A->M; i++)
        normb_l += b[i] * b[i];
    MPI_Allreduce(&normb_l, &normb, 1, MPI_DOUBLE, MPI_SUM, comm);
    normb = sqrt(normb);
//    if(rank==1) std::cout << normb << std::endl;

//    Vector r = b - A*x;
    std::vector<double> matvecTemp(A->M);
    A->matvec(&*x.begin(), &*matvecTemp.begin(), comm);
//    if(rank==1)
//        for(i=0; i<matvecTemp.size(); i++)
//            std::cout << matvecTemp[i] << std::endl;

    std::vector<double> r(A->M);
    for(i=0; i<matvecTemp.size(); i++)
        r[i] = b[i] - matvecTemp[i];

    if (normb == 0.0)
        normb = 1;

    double resid_l, resid;
    resid_l = 0;
    for(i=0; i<A->M; i++)
        resid_l += r[i] * r[i];
    MPI_Allreduce(&resid_l, &resid, 1, MPI_DOUBLE, MPI_SUM, comm);
    resid = sqrt(resid_l);

    if ((resid / normb) <= tol) {
        tol = resid;
        max_iter = 0;
        return 0;
    }

    double alpha, beta, rho, rho1, tempDot;
    std::vector<double> z(A->M);
    std::vector<double> p(A->M);
    std::vector<double> q(A->M);
    for (i = 0; i < max_iter; i++) {
//        z = M.solve(r);
        // todo: write this part.

//        rho(0) = dot(r, z);
        rho = 0;
        for(j = 0; j < A->M; j++)
            rho += r[j] * z[j];

//        if (i == 1)
//            p = z;
//        else {
//            beta(0) = rho(0) / rho_1(0);
//            p = z + beta(0) * p;
//        }

        if(i == 0)
            p = z;
        else{
            beta = rho / rho1;
            for(j = 0; j < A->M; j++)
                p[j] = z[j] + (beta * p[j]);
        }

//        q = A*p;
        A->matvec(&*p.begin(), &*q.begin(), comm);

//        alpha(0) = rho(0) / dot(p, q);
        tempDot = 0;
        for(j = 0; j < A->M; j++)
            tempDot += p[j] * q[j];
        alpha = rho / tempDot;

//        x += alpha(0) * p;
//        r -= alpha(0) * q;
        for(j = 0; j < A->M; j++){
            x[j] += alpha * p[j];
            r[j] -= alpha * q[j];
        }

        resid_l = 0;
        for(j = 0; j < A->M; j++)
            resid_l += r[j] * r[j];
        MPI_Allreduce(&resid_l, &resid, 1, MPI_DOUBLE, MPI_SUM, comm);
        resid = sqrt(resid_l);

        if ((resid / normb) <= tol) {
            tol = resid;
            max_iter = i;
            return 0;
        }

        rho1 = rho;
    }

    return 0;
}
*/


int saena_object::smooth(Grid* grid, std::string smoother, std::vector<value_t>& u, std::vector<value_t>& rhs, int iter){
    std::vector<value_t> temp1(u.size());
    std::vector<value_t> temp2(u.size());

    if(smoother == "jacobi"){
        grid->A->jacobi(iter, u, rhs, temp1);
    }else if(smoother == "chebyshev"){
        grid->A->chebyshev(iter, u, rhs, temp1, temp2);
    }else{
        printf("Error: Unknown smoother");
        MPI_Finalize();
        return -1;
    }

    return 0;
}


int saena_object::vcycle(Grid* grid, std::vector<value_t>& u, std::vector<value_t>& rhs){

    if(grid->A->active) {
        MPI_Comm comm = grid->A->comm;
        int rank, nprocs;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);

        double t1, t2;
        value_t dot;
        std::string func_name;
        std::vector<value_t> res;
        std::vector<value_t> res_coarse;
        std::vector<value_t> uCorrCoarse;
        std::vector<value_t> uCorr;
        std::vector<value_t> temp;

#ifdef __DEBUG1__
//        print_vector(rhs, -1, "rhs in vcycle", comm);

        if(verbose_vcycle){
            MPI_Barrier(comm);
            if(rank==0) printf("rank = %d: vcycle level = %d, A->M = %u, u.size = %lu, rhs.size = %lu \n",
                               rank, grid->currentLevel, grid->A->M, u.size(), rhs.size());
            MPI_Barrier(comm);
        }
#endif

        // **************************************** 0. direct-solve the coarsest level ****************************************

        if (grid->currentLevel == max_level) {

#ifdef __DEBUG1__
            if(verbose_vcycle){
                MPI_Barrier(comm);
                if(rank==0) std::cout << "vcycle: solving the coarsest level using " << direct_solver << std::endl;
                MPI_Barrier(comm);
            }
#endif

            res.resize(grid->A->M);

            t1 = omp_get_wtime();

            if(direct_solver == "CG")
                solve_coarsest_CG(grid->A, u, rhs);
            else if(direct_solver == "SuperLU")
                solve_coarsest_SuperLU(grid->A, u, rhs);
            else {
                if (rank == 0) printf("Error: Unknown direct solver is chosen! \n");
                MPI_Finalize();
                return -1;
            }

            // scale the solution u
            // -------------------------
//            scale_vector(u, grid->A->inv_sq_diag);

#ifdef __DEBUG1__
            t2 = omp_get_wtime();
            func_name = "vcycle: level " + std::to_string(grid->currentLevel) + ": solve coarsest";
            if (verbose) print_time(t1, t2, func_name, comm);

            if(verbose_vcycle_residuals){
                grid->A->residual(u, rhs, res);
                dotProduct(res, res, &dot, comm);
                if(rank==0) std::cout << "\nlevel = " << grid->currentLevel
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
//                printf("-----------------------\n");}
#endif

            return 0;
        }

        res.resize(grid->A->M);
        uCorr.resize(grid->A->M);
        temp.resize(grid->A->M);

#ifdef __DEBUG1__
        if(verbose_vcycle_residuals){
            grid->A->residual(u, rhs, res);
            dotProduct(res, res, &dot, comm);
            if(rank==0) std::cout << "\nlevel = " << grid->currentLevel << ", vcycle start      = " << sqrt(dot) << std::endl;
        }
#endif

        // **************************************** 1. pre-smooth ****************************************

#ifdef __DEBUG1__
        if(verbose_vcycle){
            MPI_Barrier(comm);
            if(rank==0) printf("vcycle level %d: presmooth\n", grid->currentLevel);
            MPI_Barrier(comm);
        }

//        MPI_Barrier(grid->A->comm);
        t1 = omp_get_wtime();
#endif

        if(preSmooth) {
            smooth(grid, smoother, u, rhs, preSmooth);
        }

#ifdef __DEBUG1__
        t2 = omp_get_wtime();
        func_name = "Vcycle: level " + std::to_string(grid->currentLevel) + ": pre";
        if (verbose) print_time(t1, t2, func_name, comm);

//        print_vector(u, -1, "u in vcycle", comm);
//        if(rank==0) std::cout << "\n1. pre-smooth: u, currentLevel = " << grid->currentLevel << std::endl;
#endif

        // **************************************** 2. compute residual ****************************************

#ifdef __DEBUG1__
        if(verbose_vcycle){
            MPI_Barrier(comm);
            if(rank==0) printf("vcycle level %d: residual\n", grid->currentLevel);
            MPI_Barrier(comm);
        }
#endif

        grid->A->residual(u, rhs, res);

#ifdef __DEBUG1__
//        print_vector(res, -1, "res", comm);

        if(verbose_vcycle_residuals){
            dotProduct(res, res, &dot, comm);
            if(rank==0) std::cout << "level = " << grid->currentLevel << ", after pre-smooth  = " << sqrt(dot) << std::endl;
        }
#endif

        // **************************************** 3. restrict ****************************************

#ifdef __DEBUG1__
        if(verbose_vcycle){
            MPI_Barrier(comm);
            if(rank==0) printf("vcycle level %d: restrict\n", grid->currentLevel);
            printf("grid->Ac.M_old = %u \n", grid->Ac.M_old);
            MPI_Barrier(comm);
        }

        t1 = omp_get_wtime();
#endif

        res_coarse.resize(grid->Ac.M_old);
        grid->R.matvec(res, res_coarse);

#ifdef __DEBUG1__
//        grid->R.print_entry(-1);
//        print_vector(res_coarse, -1, "res_coarse in vcycle", comm);
//        MPI_Barrier(comm); printf(" res.size() = %lu \tres_coarse.size() = %lu \n", res.size(), res_coarse.size()); MPI_Barrier(comm);
#endif

        if(grid->Ac.active_minor) {
            comm = grid->Ac.comm;
            MPI_Comm_size(comm, &nprocs);
            MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
            if (verbose_vcycle) {
                MPI_Barrier(comm);
                if (rank == 0) printf("vcycle level %d: repartition_u_shrink\n", grid->currentLevel);
                MPI_Barrier(comm);
            }
//            MPI_Barrier(comm); printf("before repartition_u_shrink: res_coarse.size = %ld \n", res_coarse.size()); MPI_Barrier(comm);
#endif

//            if (grid->Ac.shrinked && nprocs > 1)
//                repartition_u_shrink(res_coarse, *grid);

            if (nprocs > 1){
                repartition_u_shrink(res_coarse, *grid);
            }

#ifdef __DEBUG1__
//            MPI_Barrier(comm); printf("after  repartition_u_shrink: res_coarse.size = %ld \n", res_coarse.size()); MPI_Barrier(comm);

            t2 = omp_get_wtime();
            func_name = "Vcycle: level " + std::to_string(grid->currentLevel) + ": restriction";
            if (verbose) print_time(t1, t2, func_name, comm);

//            print_vector(res_coarse, 0, "res_coarse", comm);
#endif

            // **************************************** 4. recurse ****************************************

#ifdef __DEBUG1__
            if (verbose_vcycle) {
                MPI_Barrier(comm);
                if (rank == 0) printf("vcycle level %d: recurse\n", grid->currentLevel);
                MPI_Barrier(comm);
            }
#endif

            // scale rhs of the next level
            scale_vector(res_coarse, grid->coarseGrid->A->inv_sq_diag);

            uCorrCoarse.assign(grid->Ac.M, 0);
            vcycle(grid->coarseGrid, uCorrCoarse, res_coarse);

            // scale u
            scale_vector(uCorrCoarse, grid->coarseGrid->A->inv_sq_diag);

#ifdef __DEBUG1__
//        if(rank==0) std::cout << "\n4. uCorrCoarse, currentLevel = " << grid->currentLevel
//                              << ", uCorrCoarse.size = " << uCorrCoarse.size() << std::endl;
//        print_vector(uCorrCoarse, -1, "uCorrCoarse", grid->A->comm);
#endif

        }

        // **************************************** 5. prolong ****************************************

        comm = grid->A->comm;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
        t1 = omp_get_wtime();

        if (verbose_vcycle) {
            MPI_Barrier(comm);
            if (rank == 0) printf("vcycle level %d: repartition_back_u_shrink\n", grid->currentLevel);
            MPI_Barrier(comm);
        }
#endif

        if(nprocs > 1 && grid->Ac.active_minor){
            repartition_back_u_shrink(uCorrCoarse, *grid);
        }

#ifdef __DEBUG1__
//        print_vector(uCorrCoarse, -1, "uCorrCoarse", grid->A->comm);

        if(verbose_vcycle){
            MPI_Barrier(comm);
            if(rank==0) printf("vcycle level %d: prolong\n", grid->currentLevel);
            MPI_Barrier(comm);}
#endif

        uCorr.resize(grid->A->M);
        grid->P.matvec(uCorrCoarse, uCorr);

#ifdef __DEBUG1__
        t2 = omp_get_wtime();
        func_name = "Vcycle: level " + std::to_string(grid->currentLevel) + ": prolongation";
        if (verbose) print_time(t1, t2, func_name, comm);

//        if(rank==0)
//            std::cout << "\n5. prolongation: uCorr = P*uCorrCoarse , currentLevel = " << grid->currentLevel
//                      << ", uCorr.size = " << uCorr.size() << std::endl;
//        print_vector(uCorr, -1, "uCorr", grid->A->comm);
#endif

        // **************************************** 6. correct ****************************************

#ifdef __DEBUG1__
        if(verbose_vcycle){
            MPI_Barrier(comm);
            if(rank==0) printf("vcycle level %d: correct\n", grid->currentLevel);
            MPI_Barrier(comm);}
#endif

        #pragma omp parallel for
        for (index_t i = 0; i < u.size(); i++)
            u[i] -= uCorr[i];

#ifdef __DEBUG1__
//        print_vector(u, 0, "u after correction", grid->A->comm);

        if(verbose_vcycle_residuals){
            grid->A->residual(u, rhs, res);
            dotProduct(res, res, &dot, comm);
            if(rank==0) std::cout << "level = " << grid->currentLevel << ", after correction  = " << sqrt(dot) << std::endl;
        }
#endif

        // **************************************** 7. post-smooth ****************************************

#ifdef __DEBUG1__
        if(verbose_vcycle){
            MPI_Barrier(comm);
            if(rank==0) printf("vcycle level %d: post-smooth\n", grid->currentLevel);
            MPI_Barrier(comm);}

        t1 = omp_get_wtime();
#endif

        if(postSmooth){
            smooth(grid, smoother, u, rhs, postSmooth);
        }

#ifdef __DEBUG1__
        t2 = omp_get_wtime();
        func_name = "Vcycle: level " + std::to_string(grid->currentLevel) + ": post";
        if (verbose) print_time(t1, t2, func_name, comm);

        if(verbose_vcycle_residuals){

//        if(rank==1) std::cout << "\n7. post-smooth: u, currentLevel = " << grid->currentLevel << std::endl;
//        print_vector(u, 0, "u post-smooth", grid->A->comm);

            grid->A->residual(u, rhs, res);
            dotProduct(res, res, &dot, comm);
            if(rank==0) std::cout << "level = " << grid->currentLevel << ", after post-smooth = " << sqrt(dot) << std::endl;
        }
#endif

    } // end of if(active)

    return 0;
}


int saena_object::solve(std::vector<value_t>& u){

    MPI_Comm comm = grids[0].A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // ************** check u size **************
/*
    index_t u_size_local = u.size(), u_size_total;
    MPI_Allreduce(&u_size_local, &u_size_total, 1, MPI_UNSIGNED, MPI_SUM, grids[0].A->comm);
    if(grids[0].A->Mbig != u_size_total){
        if(rank==0) printf("Error: size of LHS (=%u) and the solution vector u (=%u) are not equal!\n", grids[0].A->Mbig, u_size_total);
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

    u.assign(grids[0].A->M, 0);

    // ************** solve **************

//    double temp;
//    current_dot(rhs, rhs, &temp, comm);
//    if(rank==0) std::cout << "norm(rhs) = " << sqrt(temp) << std::endl;

    std::vector<value_t> r(grids[0].A->M);
    grids[0].A->residual(u, grids[0].rhs, r);
    double initial_dot, current_dot;
    dotProduct(r, r, &initial_dot, comm);
    if(rank==0) std::cout << "******************************************************" << std::endl;
    if(rank==0) printf("\ninitial residual = %e \n\n", sqrt(initial_dot));

    // if max_level==0, it means only direct solver is being used.
    if(max_level == 0)
        printf("\nonly using the direct solver! \n");

//    if(rank==0){
//        printf("Vcycle #: \tabsolute residual\n");
//        printf("-----------------------------\n");
//    }

    int i;
    for(i=0; i<vcycle_num; i++){
        vcycle(&grids[0], u, grids[0].rhs);
        grids[0].A->residual(u, grids[0].rhs, r);
        dotProduct(r, r, &current_dot, comm);

//        if(rank==0) printf("Vcycle %d: \t%.10f \n", i, sqrt(current_dot));
//        if(rank==0) printf("vcycle iteration = %d, residual = %f \n\n", i, sqrt(current_dot));
        if( current_dot/initial_dot < relative_tolerance * relative_tolerance )
            break;
    }

    // set number of iterations that took to find the solution
    // only do the following if the end of the previous for loop was reached.
    if(i == vcycle_num)
        i--;

    if(rank==0){
        std::cout << "******************************************************" << std::endl;
        printf("\nfinal:\nstopped at iteration    = %d \nfinal absolute residual = %e"
                       "\nrelative residual       = %e \n\n", ++i, sqrt(current_dot), sqrt(current_dot/initial_dot));
        std::cout << "******************************************************" << std::endl;
    }

//    print_vector(u, -1, "u", comm);

    // ************** scale u **************

    scale_vector(u, grids[0].A->inv_sq_diag);

    // ************** repartition u back **************

//    if(repartition)
//        repartition_back_u(u);

//    print_vector(u, -1, "u", comm);

    return 0;
}


int saena_object::solve_pcg(std::vector<value_t>& u){

    MPI_Comm comm = grids[0].A->comm;
    int nprocs, rank;
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

    u.assign(grids[0].A->M, 0);

    // ************** create matrix in SuperLU **************
/*
    saena_matrix *A_coarsest = &grids.back().Ac;

    if(A_coarsest->active) {

        // todo: bool active is not set correctly for inactive processors in the lower grids.
        // check how the following command is set in setup() and try to fix it:
        // if(!grids[i].Ac.active)
        //     break;

        MPI_Comm *comm_coarsest = &A_coarsest->comm;
        int nprocs_coarsest, rank_coarsest;
        MPI_Comm_size(*comm_coarsest, &nprocs_coarsest);
        MPI_Comm_rank(*comm_coarsest, &rank_coarsest);

//    superlu_dist_options_t options;
//    SuperLUStat_t stat;
//    SuperMatrix A_SLU;
//    ScalePermstruct_t ScalePermstruct;
//    LUstruct_t LUstruct;
//    SOLVEstruct_t SOLVEstruct;
//    gridinfo_t superlu_grid;
//    double   *berr;
//    double   *b;
        int m, n, m_loc, nnz_loc;
        int nprow, npcol;
//    int      iam, info, ldb, nrhs;
        int iam, ldb;

        nprow = nprocs_coarsest;  // Default process rows.
        npcol = 1;  // Default process columns.
//    nrhs  = 1;  // Number of right-hand side.

        // ------------------------------------------------------------
        //   INITIALIZE MPI ENVIRONMENT.
        // ------------------------------------------------------------
//    MPI_Init( &argc, &argv );

//    char* file_name(argv[5]);
//    saena::matrix A_saena (file_name, comm);
//    A_saena.assemble();
//    A_saena.print_entry(-1);
//    if(rank==0) printf("after matrix assemble.\n");


        // Parse command line argv[].
//        for (cpp = argv+1; *cpp; ++cpp) {
//            if ( **cpp == '-' ) {
//                c = *(*cpp+1);
//                ++cpp;
//                switch (c) {
//                    case 'h':
//                        printf("Options:\n");
//                        printf("\t-r <int>: process rows    (default %4d)\n", nprow);
//                        printf("\t-c <int>: process columns (default %4d)\n", npcol);
//                        exit(0);
//                        break;
//                    case 'r': nprow = atoi(*cpp);
//                        break;
//                    case 'c': npcol = atoi(*cpp);
//                        break;
//                }
//            } else { // Last arg is considered a filename
//    //            if ( !(fp = fopen(*cpp, "r")) ) {
//    //                ABORT("File does not exist");
//    //            }
//
//                saena::matrix A_saena (*cpp, comm);
//                A_saena.assemble();
//                A_saena.print_entry(-1);
//                if(rank==0) printf("after matrix assemble.\n");
//                break;
//            }
//        }

        // ------------------------------------------------------------
        //   INITIALIZE THE SUPERLU PROCESS GRID.
        // ------------------------------------------------------------

#ifdef __DEBUG1__
        if (verbose_solve_coarse) {
            MPI_Barrier(*comm_coarsest);
            if (rank_coarsest == 0) {
                printf("INITIALIZE THE SUPERLU PROCESS GRID. \n");
            }
            MPI_Barrier(*comm_coarsest);
        }
#endif

        superlu_gridinit(*comm_coarsest, nprow, npcol, &superlu_grid);

        // Bail out if I do not belong in the grid.
        iam = superlu_grid.iam; // my process rank in this group
//    printf("iam = %d, nprow = %d, npcol = %d \n", iam, nprow, npcol);
//    if ( iam >= nprow * npcol )	goto out;

#ifdef __DEBUG1__
        if (verbose_solve_coarse) {
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
        if (verbose_solve_coarse) {
            MPI_Barrier(*comm_coarsest);
            if (rank_coarsest == 0) printf("PASS THE MATRIX FROM SAENA. \n");
            MPI_Barrier(*comm_coarsest);
        }
#endif

        m = A_coarsest->Mbig;
        m_loc = A_coarsest->M;
        n = m;
        nnz_loc = A_coarsest->nnz_l;
        ldb = m_loc;

#ifdef __DEBUG1__
        if (verbose_solve_coarse) {
            MPI_Barrier(*comm_coarsest);
            if (rank_coarsest == 0)
                printf("m = %d, m_loc = %d, n = %d, nnz_g = %ld, nnz_loc = %d, ldb = %d \n",
                       m, m_loc, n, A_coarsest->nnz_g, nnz_loc, ldb);
            MPI_Barrier(*comm_coarsest);
        }
#endif

        // CSR format (compressed row)
        // sort entries in row-major
        std::vector<cooEntry> entry_temp = A_coarsest->entry;
        std::sort(entry_temp.begin(), entry_temp.end(), row_major);
//        print_vector(entry_temp, -1, "entry_temp", comm);

        index_t fst_row = A_coarsest->split[rank_coarsest];
        std::vector<int> nnz_per_row(m_loc, 0);

        auto *rowptr    = (int_t *) intMalloc_dist(m_loc + 1);
        auto *nzval_loc = (double *) doubleMalloc_dist(nnz_loc);
        auto *colind    = (int_t *) intMalloc_dist(nnz_loc);

        // Do this line to avoid this subtraction for each entry in the next "for" loop.
        int *nnz_per_row_p = &nnz_per_row[0] - fst_row;

        for (nnz_t i = 0; i < nnz_loc; i++) {
            nzval_loc[i] = entry_temp[i].val;
//            nnz_per_row[entry_temp[i].row - fst_row]++;
            nnz_per_row_p[entry_temp[i].row]++;
            colind[i] = entry_temp[i].col;
        }

        // rowtptr is scan of nnz_per_row.
        rowptr[0] = 0;
        for (index_t i = 0; i < m_loc; i++) {
            rowptr[i + 1] = rowptr[i] + nnz_per_row[i];
        }

//    grids[0].A->print_entry(-1);
//    if(rank==0){
//        printf("\nmatrix entries in row-major format to be passed to SuperLU:\n");
//        for(nnz_t i = 0; i < nnz_loc; i++)
//            printf("%ld \t%d \t%lld \t%lf \n", i, entry_temp[i].row-fst_row, colind[i], nzval_loc[i]);
//        printf("\nrowptr:\n");
//        for(nnz_t i = 0; i < m_loc+1; i++)
//            printf("%ld \t%lld \n", i, rowptr[i]);
//    }

        dCreate_CompRowLoc_Matrix_dist(&A_SLU2, m, n, nnz_loc, m_loc, fst_row,
                                       &nzval_loc[0], &colind[0], &rowptr[0],
                                       SLU_NR_loc, SLU_D, SLU_GE);

//    dcreate_matrix(&A_SLU, nrhs, &b, &ldb, &xtrue, &ldx, fp, &grid);
    }
*/

    // ************** solve **************

//    double t1 = MPI_Wtime();

//    double temp;
//    dot(rhs, rhs, &temp, comm);
//    if(rank==0) std::cout << "norm(rhs) = " << sqrt(temp) << std::endl;

    std::vector<value_t> r(grids[0].A->M);
    grids[0].A->residual(u, grids[0].rhs, r);
    double initial_dot, current_dot, previous_dot;
    dotProduct(r, r, &initial_dot, comm);
    if(rank==0) std::cout << "******************************************************" << std::endl;
    if(rank==0) printf("\ninitial residual = %e \n\n", sqrt(initial_dot));

    // if max_level==0, it means only direct solver is being used inside the previous vcycle, and that is all needed.
    if(max_level == 0){
        vcycle(&grids[0], u, grids[0].rhs);
//        grids[0].A->print_entry(-1);
        grids[0].A->residual(u, grids[0].rhs, r);
//        print_vector(r, -1, "res", comm);
        dotProduct(r, r, &current_dot, comm);
//        if(rank==0) std::cout << "dot = " << current_dot << std::endl;

        if(rank==0){
            std::cout << "******************************************************" << std::endl;
            printf("\nfinal:\nonly using the direct solver! \nfinal absolute residual = %e"
                           "\nrelative residual       = %e \n\n", sqrt(current_dot), sqrt(current_dot/initial_dot));
            std::cout << "******************************************************" << std::endl;
        }

        // scale the solution u
        scale_vector(u, grids[0].A->inv_sq_diag);

        // repartition u back
//        if(repartition){
//            repartition_back_u(u);
//        }

        return 0;
    }

    std::vector<value_t> rho(grids[0].A->M, 0);
    vcycle(&grids[0], rho, r);

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(verbose_solve) if(rank == 0) printf("solve_pcg: first vcycle!\n");
        MPI_Barrier(comm);
    }
#endif

//    for(i = 0; i < r.size(); i++)
//        printf("rho[%lu] = %f,\t r[%lu] = %f \n", i, rho[i], i, r[i]);

//    if(rank==0){
//        printf("Vcycle #: absolute residual \tconvergence factor\n");
//        printf("--------------------------------------------------------\n");
//    }

    std::vector<value_t> h(grids[0].A->M);
    std::vector<value_t> p(grids[0].A->M);
    p = rho;

    int i;
    previous_dot = initial_dot;
    current_dot  = initial_dot;
    double rho_res, pdoth, alpha, beta;
    for(i = 0; i < vcycle_num; i++){
        grids[0].A->matvec(p, h);
        dotProduct(r, rho, &rho_res, comm);
        dotProduct(p, h, &pdoth, comm);
        alpha = rho_res / pdoth;
//        printf("rho_res = %e, pdoth = %e, alpha = %f \n", rho_res, pdoth, alpha);

#pragma omp parallel for
        for(index_t j = 0; j < u.size(); j++){
//            if(rank==0) printf("before u = %.10f \tp = %.10f \talpha = %f \n", u[j], p[j], alpha);
            u[j] -= alpha * p[j];
            r[j] -= alpha * h[j];
//            if(rank==0) printf("after  u = %.10f \tp = %.10f \talpha = %f \n", u[j], p[j], alpha);
        }

//        print_vector(u, -1, "v inside solve_pcg", grids[0].A->comm);

        previous_dot = current_dot;
        dotProduct(r, r, &current_dot, comm);
        // print the "absolute residual" and the "convergence factor":
//        if(rank==0) printf("Vcycle %d: %.10f  \t%.10f \n", i+1, sqrt(current_dot), sqrt(current_dot/previous_dot));
//        if(rank==0) printf("Vcycle %lu: aboslute residual = %.10f \n", i+1, sqrt(current_dot));
        if( current_dot/initial_dot < relative_tolerance * relative_tolerance )
            break;

        if(verbose) if(rank==0) printf("_______________________________ \n\n***** Vcycle %u *****\n", i+1);
        std::fill(rho.begin(), rho.end(), 0);
        vcycle(&grids[0], rho, r);
        dotProduct(r, rho, &beta, comm);
        beta /= rho_res;

#pragma omp parallel for
        for(index_t j = 0; j < u.size(); j++)
            p[j] = rho[j] + beta * p[j];
//        printf("beta = %e \n", beta);
    }

    // set number of iterations that took to find the solution
    // only do the following if the end of the previous for loop was reached.
    if(i == vcycle_num)
        i--;

//    double t_dif = MPI_Wtime() - t1;
//    print_time(t_dif, "solve_pcg", comm);

    if(rank==0){
        std::cout << "\n******************************************************" << std::endl;
        printf("\nfinal:\nstopped at iteration    = %d \nfinal absolute residual = %e"
                       "\nrelative residual       = %e \n\n", i+1, sqrt(current_dot), sqrt(current_dot/initial_dot));
        std::cout << "******************************************************" << std::endl;
    }

    iter_num_lazy.emplace_back(i+1);
    if(iter_num_lazy.size() == 10){
        print_vector(iter_num_lazy, 0, "iter_num_lazy", comm);
    }

#ifdef __DEBUG1__
    if(verbose_solve){
        MPI_Barrier(comm);
        if(verbose_solve) if(rank == 0) printf("solve_pcg: solve!\n");
        MPI_Barrier(comm);
    }
#endif

    // ************** destroy matrix from SuperLU **************
/*
    if(A_coarsest->active) {
        Destroy_CompRowLoc_Matrix_dist(&A_SLU2);

//       RELEASE THE SUPERLU PROCESS GRID.
        out:
        superlu_gridexit(&superlu_grid);
    }
*/
    // ************** scale u **************

    scale_vector(u, grids[0].A->inv_sq_diag);

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
    for(i=0; i<vcycle_num; i++){
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
        if( current_dot/initial_dot < relative_tolerance * relative_tolerance )
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
    if(i == vcycle_num)
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
