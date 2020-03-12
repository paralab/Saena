#include "saena_object.h"
#include "saena_matrix.h"
#include "strength_matrix.h"
#include "grid.h"
#include "GR_encoder.h"
#include "aux_functions.h"
#include "parUtils.h"
#include "dollar.hpp"

#include <mkl_spblas.h>

#include <cstdio>
#include <fstream>
#include <algorithm>
#include <mpi.h>
#include <iomanip>


double case1 = 0, case2 = 0, case3 = 0; // for timing case parts of fast_mm
double t_init_prep = 0, t_mat = 0, t_comp = 0, t_decomp = 0, t_prep_iter = 0, t_fast_mm = 0, t_sort = 0, t_wait = 0;

// from an MKL example
/* To avoid constantly repeating the part of code that checks inbound SparseBLAS functions' status,
   use macro CALL_AND_CHECK_STATUS */
#define CALL_AND_CHECK_STATUS(function, error_message) do { \
          if(function != SPARSE_STATUS_SUCCESS)             \
          {                                                 \
          printf(error_message); fflush(0);                 \
          status = 1;                                       \
          goto memory_free;                                 \
          }                                                 \
} while(0)


void saena_object::fast_mm(CSCMat_mm &A, CSCMat_mm &B, std::vector<cooEntry> &C, MPI_Comm comm){

    // =============================================================
    // Compute: C = A * B
    // This function has three parts:
    // 1- Do multiplication when blocks of A and B are small enough. Put them in a sparse C, where C_ij = A_i * B_j
    // 2- A is horizontal (row < col)
    // 3- A is vertical
    // In all three parts, return the multiplication sorted in column-major order.

    //idea of matrix-matrix product:
    // ----------------------------
//    int fast_mm(coo A, coo B, coo &C) {
//        if ( small_enough ) {
//            C_tmp = dense(nxn);
//            C_tmp = A*B;
//            C = dense_to_coo(C_tmp); // C is sorted
//        } else if { horizontal_split } { // n log n
//            coo C1 = fast_mm(A[1], B[1]);
//            coo C2 = fast_mm(A[2], B[2]);
//            return merge_add(C1, C2);
//        } else { // vertical split (when it gets smaller) n^2
//            return sort (A1B1 + A2B1 + A1B2 + A2B2);
//        }
//    }
// =============================================================
// Case 2:
// -------------------  ---------
// |        |        |  |       |
// |        |        |  |       |
// -------------------  |       |
//          A           |-------|
//                      |       |
//                      |       |
//                      |       |
//                      ---------
//                          B
//
// Case 3:
// ---------   --------------------
// |       |   |         |        |
// |       |   |         |        |
// |       |   --------------------
// |-------|             B
// |       |
// |       |
// |       |
// ---------
//     A
// =============================================================

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    int verbose_rank = 0;
    if(rank==verbose_rank && verbose_fastmm){
        printf("\nfast_mm: start \n");
        std::cout << "case iters: " << case1_iter << "," << case2_iter << "," << case3_iter << std::endl;
    }

    // assert and debug
    {

        ASSERT(A.nnz == (A.col_scan[A.col_sz] - A.col_scan[0]), "rank: " << rank << ", A.nnz: " << A.nnz
               << ", A.col_scan[0]: " << A.col_scan[0] << ", A.col_scan[A.col_sz]: " << A.col_scan[A.col_sz]);

        ASSERT(B.nnz == (B.col_scan[B.col_sz] - B.col_scan[0]), "rank: " << rank << ", B.nnz: " << B.nnz
               << ", B.col_scan[0]: " << B.col_scan[0] << ", B.col_scan[B.col_sz]: " << B.col_scan[B.col_sz]);

        // assert A entries
        index_t col_idx;
        for (nnz_t i = 0; i < A.col_sz; i++) {
//            col_idx = i + A.col_offset;
            for (nnz_t j = A.col_scan[i]; j < A.col_scan[i + 1]; j++) {
//                std::cout << j << "\t" << A.r[j] << "\t" << col_idx << "\t" << A.v[j] << "\n";
                assert((A.r[j] >= 0) && (A.r[j] < A.row_sz));
                assert(i < A.col_sz);
                ASSERT(fabs(A.v[j]) > ALMOST_ZERO, "rank: " << rank << ", A.v[j]: " << A.v[j]);
            }
        }

        // assert B entries
        for (nnz_t i = 0; i < B.col_sz; i++) {
        col_idx = i + B.col_offset;
            for (nnz_t j = B.col_scan[i]; j < B.col_scan[i + 1]; j++) {
//                std::cout << j << "\t" << B.r[j] << "\t" << col_idx << "\t" << B.v[j] << "\n";
                assert((B.r[j] >= 0) && (B.r[j] < B.row_sz));
                assert(i < B.col_sz);
                ASSERT(fabs(B.v[j]) > ALMOST_ZERO, "rank " << rank << ": B(" << B.r[j] << ", " << col_idx << ") = " << B.v[j]);
            }
        }

        if (rank == verbose_rank) {

            if (verbose_matmat_A) {
                std::cout << "\nA: nnz = " << A.nnz
                          << ", A_row_size = " << A.row_sz << ", A_col_size = " << A.col_sz
                          << ", A_row_offset = " << A.row_offset << ", A_col_offset = " << A.col_offset << std::endl;

//                print_array(A.col_scan, A.col_sz+1, verbose_rank, "A.col_scan", comm);

                // print entries of A:
                std::cout << "\nA: nnz = " << A.nnz << std::endl;
                for (nnz_t i = 0; i < A.col_sz; i++) {
                    col_idx = i + A.col_offset;
                    for (nnz_t j = A.col_scan[i]; j < A.col_scan[i + 1]; j++) {
                        std::cout << j << "\t" << A.r[j] + A.row_offset << "\t" << col_idx << "\t" << A.v[j] << "\n";
                    }
                }
            }

            if (verbose_matmat_B) {
                std::cout << "\nB: nnz = " << B.nnz;
                std::cout << ", B_row_size = " << B.row_sz << ", B_col_size = " << B.col_sz
                          << ", B_row_offset = " << B.row_offset << ", B_col_offset = " << B.col_offset << std::endl;

//                print_array(B.col_scan, B.col_sz+1, 1, "B.col_scan", comm);

                // print entries of B:
                std::cout << "\nB: nnz = " << B.nnz << std::endl;
                for (nnz_t i = 0; i < B.col_sz; i++) {
                    col_idx = i + B.col_offset;
                    for (nnz_t j = B.col_scan[i]; j < B.col_scan[i + 1]; j++) {
                        std::cout << j << "\t" << B.r[j] + B.row_offset << "\t" << col_idx << "\t" << B.v[j] << "\n";
                    }
                }
            }

//        std::cout << "\nnnzPerColScan_leftStart:" << std::endl;
//        for(nnz_t i = 0; i < A_col_size; i++) {
//            std::cout << i << "\t" << nnzPerColScan_leftStart[i] << std::endl;
//        }
//        std::cout << "\nnnzPerColScan_leftEnd:" << std::endl;
//        for(nnz_t i = 0; i < A_col_size; i++) {
//            std::cout << i << "\t" << nnzPerColScan_leftEnd[i] << std::endl;
//        }
        }
    }
#endif

    // Case1
    // ==============================================================

    // A.row_sz * A.col_sz < matmat_size_thre1
//    if ( A.row_sz < (matmat_size_thre1 / A.col_sz) ) { //DOLLAR("case1")
    if ( case2_iter + case3_iter == matmat_size_thre2 || (A.row_sz < (matmat_size_thre1 / A.col_sz)) ) {

//        if (rank == verbose_rank) printf("fast_mm: case 1: start \n");

#ifdef __DEBUG1__
        if (rank == verbose_rank && (verbose_fastmm || verbose_matmat_recursive)) {
            printf("fast_mm: case 1: start \n");
        }
//        ++case1_iter;
#endif

        double t1 = MPI_Wtime();
        ++case1_iter;

        sparse_matrix_t Amkl = nullptr;
        mkl_sparse_d_create_csc(&Amkl, SPARSE_INDEX_BASE_ZERO, A.row_sz, A.col_sz, (int*)A.col_scan, (int*)(A.col_scan+1), (int*)A.r, A.v);

        // export and print A from the MKL data structure
/*
        double  *values_A = nullptr;
        MKL_INT *rows_A = nullptr, *columns_A = nullptr;
        MKL_INT *pointerB_A = nullptr, *pointerE_A = nullptr;
        MKL_INT  rowsA, colsA, iii;
        sparse_index_base_t indexingA;

        mkl_sparse_d_export_csc( Amkl, &indexingA, &rowsA, &colsA, &pointerB_A, &pointerE_A, &rows_A, &values_A );

        MPI_Barrier(comm);
        iii = 0;
        for (MKL_INT j = 0; j < A_col_size; ++j) {
            for (MKL_INT i = pointerB_A[j]; i < pointerE_A[j]; ++i) {
                if (rank == 1)
                    printf("%3d: (%3d , %3d) = %8f\n", iii, rows_A[iii], j + A_col_offset, values_A[iii]); fflush(nullptr);
                ++iii;
            }
        }
        if(rank==0) std::cout << "\nExtracted A info: " << indexingA << "\t" << "rows = " << rowsA << "\tcols = " << colsA << std::endl;
        MPI_Barrier(comm);
*/

        sparse_matrix_t Bmkl = nullptr;
        mkl_sparse_d_create_csc(&Bmkl, SPARSE_INDEX_BASE_ZERO, B.row_sz, B.col_sz, (int*)B.col_scan, (int*)(B.col_scan+1), (int*)B.r, B.v);

#ifdef __DEBUG1__
        {
            //        MPI_Barrier(comm);
//        if(rank==1) printf("\nPerform MKL matmult\n"); fflush(nullptr);
//        MPI_Barrier(comm);

//        auto mkl_res = mkl_sparse_spmm( SPARSE_OPERATION_NON_TRANSPOSE, Amkl, Bmkl, &Cmkl );

//        if(mkl_res != SPARSE_STATUS_SUCCESS){
//            goto memory_free;
//        }

//        goto export_c;

        }
#endif

        // Compute C = A * B
        sparse_matrix_t Cmkl = nullptr;
        mkl_sparse_spmm( SPARSE_OPERATION_NON_TRANSPOSE, Amkl, Bmkl, &Cmkl );

//        struct matrix_descr descr;
//        descr.type = SPARSE_MATRIX_TYPE_GENERAL;
//        mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE, descr, Amkl,
//                        SPARSE_OPERATION_NON_TRANSPOSE, descr, Bmkl,
//                        SPARSE_STAGE_FULL_MULT, &Cmkl);

//        export_c:

        double  *values_C = nullptr;
        MKL_INT *rows_C = nullptr, *columns_C = nullptr;
        MKL_INT *pointerB_C = nullptr, *pointerE_C = nullptr;
        MKL_INT  rows, cols, i, j, ii, status;
        sparse_index_base_t indexing;

#ifdef __DEBUG1__
        {
            //        MPI_Barrier(comm);
//        if(rank==1) printf("Export C\n"); fflush(nullptr);
//        MPI_Barrier(comm);

            CALL_AND_CHECK_STATUS(mkl_sparse_d_export_csc( Cmkl, &indexing, &rows, &cols, &pointerB_C, &pointerE_C, &rows_C, &values_C ),
                                  "Error after MKL_SPARSE_D_EXPORT_CSC  \n");

//        CALL_AND_CHECK_STATUS(mkl_sparse_d_export_csr( Cmkl, &indexing, &rows, &cols, &pointerB_C, &pointerE_C, &columns_C, &values_C ),
//                              "Error after MKL_SPARSE_D_EXPORT_CSR  \n");

//        MPI_Barrier(comm);
//        if(rank==1) std::cout << "Extracted C info: " << indexing << "\t" << "rows = " << rows << "\tcols = " << cols << std::endl;
//        if(rank==1) printf("Extract nonzeros\n"); fflush(nullptr);
//        MPI_Barrier(comm);

            goto extract_c;
        }
#endif

        mkl_sparse_d_export_csc( Cmkl, &indexing, &rows, &cols, &pointerB_C, &pointerE_C, &rows_C, &values_C );

        // Extract C when it is in CSC format.
        extract_c:
        ii = 0;
        for (j = 0; j < B.col_sz; ++j) {
            for (i = pointerB_C[j]; i < pointerE_C[j]; ++i) {
//                if (rank == 3) printf("%3d: (%3d , %3d) = %8f\n", ii, rows_C[ii] + 1, j + B_col_offset + 1, values_C[ii]); fflush(nullptr);
                C.emplace_back(rows_C[ii] + A.row_offset, j + B.col_offset, values_C[ii]);
//                C.emplace_back(rows_C[ii], j + B.col_offset, values_C[ii]);
                ++ii;
            }
        }

        // Extract C when it is in CSR format.
//        ii = 0;
//        for( i = 0; i < A_row_size; i++ ){
//            for( j = pointerB_C[i]; j < pointerE_C[i]; j++ ){
//                printf("%3d: (%3d , %3d) = %8f\n", ii, i + A_row_offset + 1, columns_C[ii] + 1, values_C[ii]); fflush(nullptr);
//                C.emplace_back(i+A_row_offset, columns_C[ii], values_C[ii]);
//                ii++;
//            }
//        }

        memory_free:

#ifdef __DEBUG1__
        //Release matrix handle. Not necessary to deallocate arrays for which we don't allocate memory:
        // values_C, columns_C, pointerB_C, and pointerE_C.
        //These arrays will be deallocated together with csrC structure.
        if( mkl_sparse_destroy( Cmkl ) != SPARSE_STATUS_SUCCESS)
        { printf(" Error after MKL_SPARSE_DESTROY, csrC \n");fflush(nullptr); }

        //Release matrix handle and deallocate arrays for which we allocate memory ourselves.
        if( mkl_sparse_destroy( Amkl ) != SPARSE_STATUS_SUCCESS)
        { printf(" Error after MKL_SPARSE_DESTROY, csrA \n");fflush(nullptr); }

        if( mkl_sparse_destroy( Bmkl ) != SPARSE_STATUS_SUCCESS)
        { printf(" Error after MKL_SPARSE_DESTROY, csrB \n");fflush(nullptr); }

        t1 = MPI_Wtime() - t1;
        case1 += t1;

        return;
#endif

        mkl_sparse_destroy(Cmkl);
        mkl_sparse_destroy(Bmkl);
        mkl_sparse_destroy(Amkl);

        t1 = MPI_Wtime() - t1;
        case1 += t1;

//        MPI_Barrier(comm);
//        if(rank==1) printf("rank %d: DONE\n", rank); fflush(nullptr);
//        MPI_Barrier(comm);

        return;
    }

    // ==============================================================
    // Case2
    // ==============================================================

    index_t A_col_size_half = A.col_sz/2;

    // if A_col_size_half == 0, it means A_col_size = 1. In this case it goes to case3.
    if ( (A.row_sz <= A.col_sz) && (A_col_size_half != 0) ){//DOLLAR("case2")
//    if (case2_iter == 0){

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_fastmm) { printf("fast_mm: case 2: start \n"); }
//        ++case2_iter;
#endif
//        if (rank == verbose_rank) { printf("fast_mm: case 2: start \n"); }

        double t2 = MPI_Wtime();
        ++case2_iter;

        // =======================================================
        // split based on matrix size
        // =======================================================

#ifdef SPLIT_SIZE
        // this part is common with the SPLIT_SIZE part, so it is moved to after SPLIT_SIZE.
#endif

        // =======================================================
        // split based on nnz
        // =======================================================

#ifdef SPLIT_NNZ
        // todo: this part is not updated after changing fast_mm().
        if(!rank) printf("The  SPLIT_NNZ part is not updated!\n");
        exit(EXIT_FAILURE);

        // prepare splits of matrix A by column
        auto A_half_nnz = (nnz_t) ceil(A.nnz / 2);
//        index_t A_col_size_half = A_col_size/2;

        if (A.nnz > matmat_nnz_thre) { // otherwise A_col_size_half will stay A_col_size/2
            for (nnz_t i = 0; i < A.col_sz; i++) {
                if( (Ac[i+1] - Ac[0]) >= A_half_nnz){
                    A_col_size_half = i;
                    break;
                }
            }
        }

        // if A is not being split at all following "half nnz method", then swtich to "half size method".
        if (A_col_size_half == A.col_sz) {
            A_col_size_half = A.col_sz / 2;
        }
#endif

        CSCMat_mm A1(A.row_sz, A.row_offset, A_col_size_half, A.col_offset, A.col_scan[A_col_size_half] - A.col_scan[0],
                      A.r, A.v, A.col_scan);

        CSCMat_mm A2(A.row_sz, A.row_offset, A.col_sz - A1.col_sz, A.col_offset + A1.col_sz, A.nnz - A1.nnz,
                      A.r, A.v, &A.col_scan[A_col_size_half]);

#ifdef __DEBUG1__
/*
        CSCMat_mm A1, A2;

        A1.r = &A.r[0];
        A1.v = &A.v[0];
        A2.r = &A.r[0];
        A2.v = &A.v[0];

        A1.col_scan = A.col_scan;
        A2.col_scan = &A.col_scan[A_col_size_half];

        A1.row_sz = A.row_sz;
        A2.row_sz = A.row_sz;

        A1.row_offset = A.row_offset;
        A2.row_offset = A.row_offset;

        A1.col_sz = A_col_size_half;
        A2.col_sz = A.col_sz - A1.col_sz;

        A1.col_offset = A.col_offset;
        A2.col_offset = A.col_offset + A1.col_sz;

        A1.nnz = A1.col_scan[A1.col_sz] - A1.col_scan[0];
        A2.nnz = A.nnz - A1.nnz;
*/
#endif

        // =======================================================

        // split B based on how A is split, so use A_col_size_half to split B. A_col_size_half can be different based on
        // choosing the splitting method (nnz or size).
        index_t B_row_size_half = A_col_size_half;

#ifdef SPLIT_SIZE
#ifdef __DEBUG1__
        assert(B_row_size_half == B.row_sz / 2);
#endif
#endif

        CSCMat_mm B1(B_row_size_half, B.row_offset, B.col_sz, B.col_offset, 0,
                     &B.r[0], &B.v[0], &B.col_scan[0]);

        CSCMat_mm B2(B.row_sz - B1.row_sz, B.row_offset + B1.row_sz, B.col_sz, B.col_offset, 0);

        reorder_split(B, B1, B2);

        t2 = MPI_Wtime() - t2;
        case2 += t2;

#ifdef __DEBUG1__
/*
        CSCMat_mm B1, B2;

        B1.row_sz = B_row_size_half;
        B2.row_sz = B.row_sz - B1.row_sz;

        B1.row_offset = B.row_offset;
        B2.row_offset = B.row_offset + B1.row_sz;

        B1.col_sz = B.col_sz;
        B2.col_sz = B.col_sz;

        B1.col_offset = B.col_offset;
        B2.col_offset = B.col_offset;

        B1.col_scan = B.col_scan;
//        B2.col_scan = new index_t[B.col_sz + 1];
//        B2.free_c   = true;

        reorder_split(B, B1, B2);

        if(B2.nnz == 0){
            delete []B2.col_scan;
            B2.free_c = false;
        }

        B1.r = &B.r[0];
        B1.v = &B.v[0];
        B2.r = &B.r[B1.col_scan[B.col_sz]];
        B2.v = &B.v[B1.col_scan[B.col_sz]];
*/
#endif

#if 0
        #ifdef __DEBUG1__
        {
            // check if reorder_split and reorder_back_split are working correctly
            std::vector<value_t> Bt(B.nnz);
            nnz_t iter = 0;
            for (nnz_t i = 0; i < B.col_sz; i++) {
                for (nnz_t j = B.col_scan[i]; j < B.col_scan[i + 1]; j++) {
                    Bt[iter] = B.v[j];
                    ++iter;
                }
            }

            reorder_split(B, B1, B2);
            reorder_back_split(B, B1, B2);

            iter = 0;
//            printf("\nB.nnz = %lu\n", B.nnz);
            for (nnz_t i = 0; i < B.col_sz; i++) {
                for (nnz_t j = B.col_scan[i]; j < B.col_scan[i + 1]; j++) {
//                    printf("%lu\t%lu\t%u\t%.16f\t%.16f\n", j, i, B.r[j], B.v[j], Bt[j]);
                    assert( fabs(B.v[j]) > ALMOST_ZERO );
                    assert(B.v[j] == Bt[iter]);
                    ++iter;
                }
            }
        }
#endif
#endif

#ifdef __DEBUG1__
        {
//        if(A1.nnz == 0 || A2.nnz == 0 || B1.nnz == 0 || B2.nnz == 0)
//            printf("rank %d: nnzs: %lu\t%lu\t%lu\t%lu\t\n", rank, A1.nnz, A2.nnz, B1.nnz, B2.nnz);

            // assert A1
//        std::cout << "\nCase2:\nA1: nnz = " << A1.nnz << std::endl;
            if (A1.nnz != 0) {
                for (nnz_t i = 0; i < A1.col_sz; i++) {
                    for (nnz_t j = A1.col_scan[i]; j < A1.col_scan[i + 1]; j++) {
                        assert((A1.r[j] >= 0) && (A1.r[j] < A1.row_sz));
//                std::cout << j << "\t" << A1.r[j] << "\t" << i + A1.col_offset << "\t" << A1.v[j] << std::endl;
                    }
                }
            }

            // assert A2
//        std::cout << "\nA2: nnz = " << A2.nnz << std::endl;
            if (A2.nnz != 0) {
                for (nnz_t i = 0; i < A2.col_sz; i++) {
                    for (nnz_t j = A2.col_scan[i]; j < A2.col_scan[i + 1]; j++) {
                        assert((A2.r[j] >= 0) && (A2.r[j] < A2.row_sz));
//                std::cout << j << "\t" << A2.r[j] << "\t" << i + A2.col_offset << "\t" << A2.v[j] << std::endl;
                    }
                }
            }

            // assert B1
//        std::cout << "\nCase2:\nB1: nnz = " << B1.nnz << std::endl;
            if (B1.nnz != 0) {
                for (nnz_t i = 0; i < B1.col_sz; i++) {
                    for (nnz_t j = B1.col_scan[i]; j < B1.col_scan[i + 1]; j++) {
//                if(B1.r[j] >= B1.row_sz)
//                    std::cout << "(rank: " << rank << ", " << j << "): \t(" << B1.r[j] << ", " << i << ")\t[(" <<
//                       B1.row_sz << ", " << B1.row_offset << ")(" << B1.col_sz << ", " << B1.col_offset << ")]\n";
//                std::cout << j << "\t" << B1.r[j] << "\t" << i + B1.col_offset << "\t" << B1.v[j] << std::endl;
                        assert((B1.r[j] >= 0) && (B1.r[j] < B1.row_sz));
                    }
                }
            }

            // assert B2
//        std::cout << "\nB2: nnz = " << B2.nnz << std::endl;
            if (B2.nnz != 0) {
                for (nnz_t i = 0; i < B2.col_sz; i++) {
                    for (nnz_t j = B2.col_scan[i]; j < B2.col_scan[i + 1]; j++) {
                        assert((B2.r[j] >= 0) && (B2.r[j] < B2.row_sz));
//                    std::cout << j << "\t" << B2.r[j] << "\t" << i + B2.col_offset << "\t" << B2.v[j] << "\n";
                    }
                }
            }

            if (rank == verbose_rank) {

//                MPI_Barrier(comm);
//                std::cout << "B_row_threshold: " << std::setw(3) << B_row_threshold << std::endl;
//                print_array(Bc1, B_col_size+1, 0, "Bc1", comm);
//                print_array(Bc2, B_col_size+1, 0, "Bc2", comm);
//                print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);

//                std::cout << "\ncase2:\nA.nnz: " << std::setw(3) << A.nnz << ", A1.nnz: " << std::setw(3) << A1.nnz
//                          << ", A2.nnz: " << std::setw(3) << A2.nnz << ", A_col_size: "   << std::setw(3) << A.col_sz
//                          << ", A_col_size_half: " << std::setw(3) << A_col_size_half << std::endl;

//                printf("fast_mm: case 2: \nA.nnz: (%lu, %lu, %lu), B.nnz: (%lu, %lu, %lu)\n"
//                       "A_size: (%u, %u, %u), B_size: (%u, %u) \n",
//                       A.nnz, A1.nnz, A2.nnz, B.nnz, B1.nnz, B2.nnz, A_row_size, A_col_size, A_col_size_half, A_col_size, B_col_size);

                if (verbose_matmat_A) {
//                std::cout << "\nCase2:\n"
//                << "A.row_sz: \t"       << A.row_sz      << "\tA1.row_sz: \t"     << A1.row_sz     << "\tA2.row_sz: \t"     << A2.row_sz
//                << "\nA.row_offset: \t" << A.row_offset  << "\tA1.row_offset: \t" << A1.row_offset << "\tA2.row_offset: \t" << A2.row_offset
//                << "\nA.col_sz: \t"     << A.col_sz      << "\tA1.col_sz: \t"     << A1.col_sz     << "\tA2.col_sz: \t"     << A2.col_sz
//                << "\nA.col_offset: \t" << A.col_offset  << "\tA1.col_offset: \t" << A1.col_offset << "\tA2.col_offset: \t" << A2.col_offset
//                << "\nA.nnz: \t\t"      << A.nnz         << "\tA1.nnz: \t"        << A1.nnz        << "\tA2.nnz: \t"        << A2.nnz << std::endl;

//                std::cout << "\nranges of A:" << std::endl;
//                for (nnz_t i = 0; i < A_col_size; i++) {
//                    std::cout << i << "\t" << Ac[i] << "\t" << Ac[i + 1] << std::endl;
//                }
//
//                std::cout << "\nranges of A1:" << std::endl;
//                for (nnz_t i = 0; i < A_col_size / 2; i++) {
//                    std::cout << i << "\t" << Ac[i] << "\t" << Ac[i + 1] << std::endl;
//                }
//
//                std::cout << "\nranges of A2:" << std::endl;
//                for (nnz_t i = 0; i < A_col_size - A_col_size / 2; i++) {
//                    std::cout << i << "\t" << Ac[A_col_size / 2 + i] << "\t" << Ac[A_col_size / 2 + i + 1] << "\n";
//                }

                    // print entries of A1:
                    std::cout << "\nCase2:\nA1: nnz = " << A1.nnz << std::endl;
                    if (A1.nnz != 0) {
                        for (nnz_t i = 0; i < A1.col_sz; i++) {
                            for (nnz_t j = A1.col_scan[i]; j < A1.col_scan[i + 1]; j++) {
                                std::cout << j << "\t" << A1.r[j] + A1.row_offset << "\t" << i + A1.col_offset << "\t"
                                          << A1.v[j] << "\n";
                            }
                        }
                    }

                    // print entries of A2:
                    std::cout << "\nA2: nnz = " << A2.nnz << std::endl;
                    if (A2.nnz != 0) {
                        for (nnz_t i = 0; i < A2.col_sz; i++) {
                            for (nnz_t j = A2.col_scan[i]; j < A2.col_scan[i + 1]; j++) {
                                std::cout << j << "\t" << A2.r[j] + A2.row_offset << "\t" << i + A2.col_offset << "\t"
                                          << A2.v[j] << "\n";
                            }
                        }
                    }
                }

                if (verbose_matmat_B) {
//                std::cout << "\nB.row_sz: \t"     << B.row_sz      << "\tB1.row_sz: \t"     << B1.row_sz     << "\tB2.row_sz: \t"     << B2.row_sz
//                          << "\nB.row_offset: \t" << B.row_offset  << "\tB1.row_offset: \t" << B1.row_offset << "\tB2.row_offset: \t" << B2.row_offset
//                          << "\nB.col_sz: \t"     << B.col_sz      << "\tB1.col_sz: \t"     << B1.col_sz     << "\tB2.col_sz: \t"     << B2.col_sz
//                          << "\nB.col_offset: \t" << B.col_offset  << "\tB1.col_offset: \t" << B1.col_offset << "\tB2.col_offset: \t" << B2.col_offset
//                          << "\nB.nnz: \t\t"      << B.nnz         << "\tB1.nnz: \t"        << B1.nnz        << "\tB2.nnz: \t"        << B2.nnz << std::endl;

//                std::cout << "\nranges of B, B1, B2::" << std::endl;
//                for (nnz_t i = 0; i < B_col_size; i++) {
//                    std::cout << i << "\t" << Bc[i] << "\t" << Bc[i + 1]
//                              << "\t" << Bc1[i] << "\t" << Bc1[i + 1]
//                              << "\t" << Bc2[i] << "\t" << Bc2[i + 1] << std::endl;
//                }

                    // print entries of B1:
                    std::cout << "\nCase2:\nB1: nnz = " << B1.nnz << std::endl;
                    if (B1.nnz != 0) {
                        for (nnz_t i = 0; i < B1.col_sz; i++) {
                            for (nnz_t j = B1.col_scan[i]; j < B1.col_scan[i + 1]; j++) {
                                std::cout << j << "\t" << B1.r[j] + B1.row_offset << "\t" << i + B1.col_offset << "\t"
                                          << B1.v[j] << "\n";
                            }
                        }
                    }

                    // print entries of B2:
                    std::cout << "\nB2: nnz = " << B2.nnz << std::endl;
                    if (B2.nnz != 0) {
                        for (nnz_t i = 0; i < B2.col_sz; i++) {
                            for (nnz_t j = B2.col_scan[i]; j < B2.col_scan[i + 1]; j++) {
                                std::cout << j << "\t" << B2.r[j] + B2.row_offset << "\t" << i + B2.col_offset << "\t"
                                          << B2.v[j] << "\n";
                            }
                        }
                    }
                }
            }
        }
#endif

        // =======================================================
        // Call two recursive functions here. Put the result of the first one in C1, and the second one in C2.
        // merge sort them and add the result to C.
//        std::vector<cooEntry> C_temp;

        // A1: start: nnzPerColScan_leftStart,                  end: nnzPerColScan_leftEnd
        // A2: start: nnzPerColScan_leftStart[A_col_size_half], end: nnzPerColScan_leftEnd[A_col_size_half]
        // B1: start: nnzPerColScan_rightStart,                 end: nnzPerColScan_middle
        // B2: start: nnzPerColScan_middle,                     end: nnzPerColScan_rightEnd

        // =======================================================

        // C1 = A1 * B1
#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 1 \n");
#endif

        if (A1.nnz != 0 && B1.nnz != 0) {
            fast_mm(A1, B1, C, comm);
        }

        // C2 = A2 * B2
#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 2 \n");
#endif

        if (A2.nnz != 0 && B2.nnz != 0) {
            fast_mm(A2, B2, C, comm);
        }

        // =======================================================
        // Finalize Case2
        // =======================================================

        t2 = MPI_Wtime();

        // return B to its original order.
        if(B2.nnz != 0) {
            reorder_back_split(B, B1, B2);
        }
#ifdef __DEBUG1__
        else{
            --reorder_counter;
        }
#endif

        t2 = MPI_Wtime() - t2;
        case2 += t2;

#ifdef __DEBUG1__
//        if(rank==0 && verbose_fastmm) printf("C1.size() = %lu, C2.size() = %lu \n", C1.size(), C2.size());
//        print_vector(C1, -1, "C1", comm);
//        print_vector(C2, -1, "C2", comm);
        if (rank == verbose_rank && verbose_fastmm) printf("fast_mm: case 2: end \n");
#endif

        return; // end of case 2 and fast_mm()

    }

    // ==============================================================
    // Case3
    // ==============================================================

    { //DOLLAR("case3") // (A_row_size > A_col_size)

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_fastmm) printf("fast_mm: case 3: start \n");
//        ++case3_iter;
#endif
//        if (rank == verbose_rank) printf("fast_mm: case 3: start \n");

        double t3 = MPI_Wtime();
        ++case3_iter;

        // split based on matrix size
        // =======================================================

#ifdef SPLIT_SIZE
        // prepare splits of matrix B by column

        index_t B_col_size_half = B.col_sz/2;

        CSCMat_mm B1(B.row_sz, B.row_offset, B_col_size_half, B.col_offset, B.col_scan[B_col_size_half] - B.col_scan[0],
                     B.r, B.v, B.col_scan);

        CSCMat_mm B2(B.row_sz, B.row_offset, B.col_sz - B1.col_sz, B.col_offset + B_col_size_half, B.nnz - B1.nnz,
                     B.r, B.v, &B.col_scan[B_col_size_half]);

#endif

#ifdef __DEBUG1__
/*
        CSCMat_mm B1, B2;

        B1.row_sz = B.row_sz;
        B2.row_sz = B.row_sz;

        B1.row_offset = B.row_offset;
        B2.row_offset = B.row_offset;

        B1.col_sz = B_col_size_half;
        B2.col_sz = B.col_sz - B1.col_sz;

        B1.col_offset = B.col_offset;
        B2.col_offset = B.col_offset + B_col_size_half;

        B1.nnz = B.col_scan[B_col_size_half] - B.col_scan[0];
        B2.nnz = B.nnz - B1.nnz;

        B1.r = &B.r[0];
        B1.v = &B.v[0];
        B2.r = &B.r[0];
        B2.v = &B.v[0];

        B1.col_scan = B.col_scan;
        B2.col_scan = &B.col_scan[B_col_size_half];
*/
#endif

#ifdef __DEBUG1__
        {
//        std::cout << "\ncase3:\nB.nnz: " << std::setw(3) << B.nnz << ", B1.nnz: " << std::setw(3) << B1.nnz
//                  << ", B2.nnz: " << std::setw(3) << B2.nnz << ", B_col_size: " << std::setw(3) << B.col_sz
//                  << ", B_col_size_half: " << std::setw(3) << B_col_size_half << std::endl;

//        std::cout << "\ncase3_part1: B_row_size: " << B_row_size << "\tB_row_offset: " << B_row_offset
//                  << "\tB1_col_size:"  << B1_col_size << "\tB2_col_size: " << B2_col_size
//                  << "\tB1_col_offset: " << B1_col_offset << "\tB2_col_offset: " << B2_col_offset << std::endl;

//        print_array(Bc, B_col_size+1, 0, "Bc", comm);
//
//        std::cout << "\nB1: nnz: " << B1.nnz << std::endl ;
//        for(index_t j = 0; j < B_col_size_half; j++){
//            for(index_t i = Bc[j]; i < Bc[j+1]; i++){
//                std::cout << i << "\t" << B[i].row << "\t" << j << "\t" << B[i].val << std::endl;
//            }
//        }
//
//        std::cout << "\nB2: nnz: " << B2.nnz << std::endl ;
//        for(index_t j = B_col_size_half; j < B_col_size; j++){
//            for(index_t i = Bc[j]; i < Bc[j+1]; i++){
//                std::cout << i + B_nnz_offset << "\t" << B[i].row << "\t" << j << "\t" << B[i].val << std::endl;
//            }
//        }
        }
#endif

        // =======================================================
        // split based on nnz
        // =======================================================

#ifdef SPLIT_NNZ
        if(rank==0) printf("NOTE: fix splitting based on nnz for case3!");

        // prepare splits of matrix B by column
        nnz_t B1.nnz = 0, B2.nnz;
        auto B_half_nnz = (nnz_t) ceil(B.nnz / 2);
        index_t B_col_size_half = B.col_sz / 2;

        if (B.nnz > matmat_nnz_thre) {
            for (nnz_t i = 0; i < B.col_sz; i++) {
                B1.nnz += nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i];

#ifdef __DEBUG1__
//                if(rank==verbose_rank)
//                    printf("B.nnz = %lu, B_half_nnz = %lu, B1.nnz = %lu, nnz on col %u: %u \n",
//                           B.nnz, B_half_nnz, B1.nnz, B[nnzPerColScan_rightStart[i]].col,
//                           nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i]);
#endif

                if (B1.nnz >= B_half_nnz) {
                    B_col_size_half = B[nnzPerColScan_rightStart[i]].col + 1 - B.col_offset;
                    break;
                }
            }
        } else {
            for (nnz_t i = 0; i < B_col_size_half; i++) {
                B1.nnz += nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i];
            }
        }

        // if B is not being splitted at all following "half nnz method", then swtich to "half col method".
        if (B_col_size_half == B.col_sz) {
            B_col_size_half = B.col_sz / 2;
            B1.nnz = 0;
            for (nnz_t i = 0; i < B_col_size_half; i++) {
                B1.nnz += nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i];
            }
        }

        B2.nnz = B.nnz - B1.nnz;
#endif

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_fastmm) printf("fast_mm: case 3: step 1 \n");
#endif

        // prepare splits of matrix A by row

        CSCMat_mm A1(A.row_sz / 2, A.row_offset, A.col_sz, A.col_offset, 0,
                    A.r, A.v, A.col_scan);

        CSCMat_mm A2(A.row_sz - A1.row_sz, A.row_offset + A1.row_sz, A.col_sz, A.col_offset, 0);

        reorder_split(A, A1, A2);

        t3 = MPI_Wtime() - t3;
        case3 += t3;

#ifdef __DEBUG1__
/*
        CSCMat_mm A1, A2;

        A1.row_sz = A_row_size_half;
        A2.row_sz = A.row_sz - A1.row_sz;

        A1.row_offset = A.row_offset;
        A2.row_offset = A.row_offset + A1.row_sz;

        A1.col_sz = A.col_sz;
        A2.col_sz = A.col_sz;

        A1.col_offset = A.col_offset;
        A2.col_offset = A.col_offset;

        A1.col_scan = A.col_scan;
        A2.col_scan = new index_t[A.col_sz + 1];
//        A2.free_c = true;

        reorder_split(A, A1, A2);
//        reorder_split(A.r, A.v, A1.col_scan, A2.col_scan, A.col_sz, A_row_threshold, A_row_size_half);

//        A1.nnz = A1.col_scan[A.col_sz] - A1.col_scan[0];
//        A2.nnz = A2.col_scan[A.col_sz] - A2.col_scan[0];

        if(A2.nnz == 0){
            delete []A2.col_scan;
//            A2.free_c = false;
        }

        A1.r = &A.r[0];
        A1.v = &A.v[0];
        A2.r = &A.r[A1.col_scan[A.col_sz]];
        A2.v = &A.v[A1.col_scan[A.col_sz]];
*/
#endif

#ifdef __DEBUG1__
        {
            index_t A_row_size_half = A.row_sz / 2;

//        std::cout << "A.nnz: " << std::setw(3) << A.nnz << ", A1.nnz: " << std::setw(3) << A1.nnz << ", A2.nnz: "
//                  << std::setw(3) << A2.nnz << ", A_row_size: " << std::setw(3) << A.row_sz
//                  << ", A_row_size_half: " << std::setw(3) << A_row_size_half << std::endl;

//        std::cout << "\ncase3_part2: A1_row_size: " << A1.row_sz << ", A2_row_size: " << A2.row_sz
//                  << ", A1_row_offset: " << A1.row_offset << ", A2_row_offset: " << A2.row_offset
//                  << ", A1_col_size: "   << A1.col_sz     << ", A2_col_size: "   << A2.col_sz
//                  << ", A1_col_offset: " << A1.col_offset << ", A2_col_offset: " << A2.col_offset
//                  << ", A_row_threshold: " << std::setw(3)<< A_row_threshold     << std::endl;

//        std::cout << "A_row_threshold: " << std::setw(3) << A_row_threshold << std::endl;

//        print_array(Ac1, A_col_size+1, 0, "Ac1", comm);
//        print_array(Ac2, A_col_size+1, 0, "Ac2", comm);

            if (rank == verbose_rank && verbose_fastmm) printf("fast_mm: case 3: step 2 \n");

//        MPI_Barrier(comm);
            if (rank == verbose_rank) {

//            printf("fast_mm: case 3: \nA.nnz: (%lu, %lu, %lu), B.nnz: (%lu, %lu, %lu)\n"
//                   "A_size: (%u, %u), B_size: (%u, %u, %u) \n",
//                   A.nnz, A1.nnz, A2.nnz, B.nnz, B1.nnz, B2.nnz, A_row_size, A_col_size, A_col_size, B_col_size, B_col_size_half);

                if (verbose_matmat_A) {
                    // print entries of A1:
                    std::cout << "\nCase3:\nA1: nnz = " << A1.nnz << std::endl;
                    if (A2.nnz != 0) {
                        for (nnz_t i = 0; i < A1.col_sz; i++) {
                            for (nnz_t j = A1.col_scan[i]; j < A1.col_scan[i + 1]; j++) {
                                std::cout << j << "\t" << A1.r[j] + A1.row_offset << "\t" << i + A1.col_offset << "\t"
                                          << A1.v[j] << "\n";
                            }
                        }
                    }

                    // print entries of A2:
                    std::cout << "\nA2: nnz = " << A2.nnz << std::endl;
                    if (A2.nnz != 0) {
                        for (nnz_t i = 0; i < A2.col_sz; i++) {
                            for (nnz_t j = A2.col_scan[i]; j < A2.col_scan[i + 1]; j++) {
                                std::cout << j << "\t" << A2.r[j] + A2.row_offset << "\t" << i + A2.col_offset << "\t"
                                          << A2.v[j] << "\n";
                            }
                        }
                    }
                }

                if (verbose_matmat_B) {
//                std::cout << "\nranges of B:" << std::endl;
//                for (nnz_t i = 0; i < B_col_size; i++) {
//                    std::cout << i << "\t" << nnzPerColScan_rightStart[i] << "\t" << nnzPerColScan_rightEnd[i]
//                              << std::endl;
//                }
//
//                std::cout << "\nranges of B1:" << std::endl;
//                for (nnz_t i = 0; i < B_col_size / 2; i++) {
//                    std::cout << i << "\t" << nnzPerColScan_rightStart[i] << "\t" << nnzPerColScan_rightEnd[i]
//                              << std::endl;
//                }
//
//                std::cout << "\nranges of B2:" << std::endl;
//                for (nnz_t i = 0; i < B_col_size - B_col_size / 2; i++) {
//                    std::cout << i << "\t" << nnzPerColScan_rightStart[B_col_size / 2 + i]
//                              << "\t" << nnzPerColScan_rightEnd[B_col_size / 2 + i] << std::endl;
//                }

                    // print entries of B1:
                    std::cout << "\nCase3:\nB1: nnz = " << B1.nnz << std::endl;
                    if (B1.nnz != 0) {
                        for (nnz_t i = 0; i < B1.col_sz; i++) {
                            for (nnz_t j = B1.col_scan[i]; j < B1.col_scan[i + 1]; j++) {
                                std::cout << j << "\t" << B1.r[j] + B1.row_offset << "\t" << i + B1.col_offset << "\t"
                                          << B1.v[j] << "\n";
                            }
                        }
                    }

                    // print entries of B2:
                    std::cout << "\nB2: nnz = " << B2.nnz << std::endl;
                    if (B2.nnz != 0) {
                        for (nnz_t i = 0; i < B2.col_sz; i++) {
                            for (nnz_t j = B2.col_scan[i]; j < B2.col_scan[i + 1]; j++) {
                                std::cout << j << "\t" << B2.r[j] + B2.row_offset << "\t" << i + B2.col_offset << "\t"
                                          << B2.v[j] << "\n";
                            }
                        }
                    }
                }
            }
        }
#endif

        // =======================================================

        // A1: start: nnzPerColScan_leftStart,                   end: nnzPerColScan_middle
        // A2: start: nnzPerColScan_middle,                      end: nnzPerColScan_leftEnd
        // B1: start: nnzPerColScan_rightStart,                  end: nnzPerColScan_rightEnd
        // B2: start: nnzPerColScan_rightStart[B_col_size_half], end: nnzPerColScan_rightEnd[B_col_size_half]

        // =======================================================

        // C1 = A1 * B1
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 1 \n");
#endif

        if (A1.nnz != 0 && B1.nnz != 0) {
            fast_mm(A1, B1, C, comm);
        }


        // C2 = A1 * B2:
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 2 \n");
#endif

        if (A1.nnz != 0 && B2.nnz != 0) {
            fast_mm(A1, B2, C, comm);
        }


        // C3 = A2 * B1
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 3 \n");
#endif

        if (A2.nnz != 0 && B1.nnz != 0) {
            fast_mm(A2, B1, C, comm);
        }


        // C4 = A2 * B2
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 4 \n");
#endif

        if (A2.nnz != 0 && B2.nnz != 0) {
            fast_mm(A2, B2, C, comm);
        }

        // =======================================================
        // Finalize Case3
        // =======================================================

        t3 = MPI_Wtime();

        // return A to its original order.
        if(A2.nnz != 0){
            reorder_back_split(A, A1, A2);
        }
#ifdef __DEBUG1__
        else{
            --reorder_counter;
        }
#endif

        t3 = MPI_Wtime() - t3;
        case3 += t3;

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_fastmm) printf("fast_mm: case 3: end \n");
#endif

    }

//    return;
}


int saena_object::matmat(saena_matrix *A, saena_matrix *B, saena_matrix *C, const bool assemble/* = true*/, const bool print_timing/* = false*/){
    // This version only works when B is symmetric, since local transpose of B is used.
    // Use B's row indices as column indices and vice versa.

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // =======================================
    // Timing Parameters
    // =======================================

    // use this parameter to time matmat, and print it if (print_timing = true)
    double t_temp = 0, t_matmat_prep = 0, t_matmat = 0, t_matmat_tot = 0;
    // perform matmat multiple times and take the average of them.
    int matmat_iter_warmup = 2, matmat_iter = 3;

    // =======================================
    // Convert A to CSC
    // =======================================

#ifdef __DEBUG1__
    for(nnz_t i = 0; i < A->nnz_l; i++){
        assert( (A->entry[i].row - A->split[rank] >= 0) && (A->entry[i].row - A->split[rank] < A->M) );
        assert( (A->entry[i].col >= 0) && (A->entry[i].col < A->Mbig) );
        assert( fabs(A->entry[i].val) > ALMOST_ZERO );
    }
#endif

//    auto Arv = new vecEntry[A->nnz_l]; // row and val
//    auto Ac  = new index_t[A->Mbig+1]; // col_idx

    // todo: change to smart pointers
//    auto Arv = std::make_unique<vecEntry[]>(A->nnz_l); // row and val
//    auto Ac  = std::make_unique<index_t[]>(A->Mbig+1); // col_idx

    CSCMat Acsc;
    Acsc.comm     = A->comm;
    Acsc.nnz      = A->nnz_l;
    Acsc.col_sz   = A->Mbig;
    Acsc.max_nnz  = A->nnz_max;
    Acsc.max_M    = A->M_max;
    Acsc.row      = new index_t[Acsc.nnz];
    Acsc.val      = new value_t[Acsc.nnz];
    Acsc.col_scan = new index_t[Acsc.col_sz + 1];

    std::fill(&Acsc.col_scan[0], &Acsc.col_scan[Acsc.col_sz + 1], 0);
    index_t *Ac_tmp = &Acsc.col_scan[1];
    for(nnz_t i = 0; i < Acsc.nnz; i++){
        Acsc.row[i] = A->entry[i].row - A->split[rank]; // make the rows start from 0. when done with multiply, add this to the result.
//        Acsc.row[i] = A->entry[i].row;
        Acsc.val[i] = A->entry[i].val;
        Ac_tmp[A->entry[i].col]++;
    }

    for(nnz_t i = 0; i < Acsc.col_sz; i++){
        Acsc.col_scan[i+1] += Acsc.col_scan[i];
    }

    Acsc.split    = A->split;
    Acsc.nnz_list = A->nnz_list;

#ifdef __DEBUG1__
    assert(Acsc.nnz == (Acsc.col_scan[Acsc.col_sz] - Acsc.col_scan[0]));

//    A->print_entry(0);
//    printf("A: nnz_l: %ld\tnnz_g: %ld\tM: %d\tM_big: %d\n", A->nnz_l, A->nnz_g, A->M, A->Mbig);
//    print_array(Acsc.row, Acsc.nnz, 0, "Acsc.row", comm);
//    print_array(Acsc.val, Acsc.nnz, 0, "Acsc.val", comm);
//    print_array(Acsc.col_scan, Acsc.col_sz + 1, 0, "Acsc.col_scan", comm);

//    std::cout << "\nA: nnz: " << Acsc.nnz << std::endl ;
//    for(index_t j = 0; j < Acsc.col_sz; j++){
//        for(index_t i = Acsc.col_scan[j]; i < Acsc.col_scan[j+1]; i++){
//            ASSERT(fabs(Acsc.val[i]) > ALMOST_ZERO, "rank: " << rank << ", Acsc.val[i]: " << Acsc.val[i]);
//            std::cout << std::setprecision(4) << Acsc.row[i] << "\t" << j << "\t" << Acsc.val[i] << std::endl;
//        }
//    }
#endif

    // =======================================
    // Convert the local transpose of B to CSC
    // =======================================

#ifdef __DEBUG1__
    for(nnz_t i = 0; i < B->nnz_l; i++){
        assert( (B->entry[i].row - B->split[rank] >= 0) && (B->entry[i].row - B->split[rank] < B->M) );
        assert( (B->entry[i].col >= 0) && (B->entry[i].col < B->Mbig) );
        assert( fabs(B->entry[i].val) > ALMOST_ZERO );
    }
#endif

    // make a copy of entries of B, then change their order to row-major
    std::vector<cooEntry> B_ent(B->entry);
    std::sort(B_ent.begin(), B_ent.end(), row_major);

    // todo: change to smart pointers
//    auto Brv = std::make_unique<vecEntry[]>(B->nnz_l); // row (actually col to have the transpose) and val
//    auto Bc  = std::make_unique<index_t[]>(B->M+1); // col_idx

//    auto Brv = new vecEntry[B->nnz_l]; // row and val
//    auto Bc  = new index_t[B->M+1];    // col_idx

    CSCMat Bcsc;
    Bcsc.comm     = B->comm;
    Bcsc.nnz      = B->nnz_l;
    Bcsc.col_sz   = B->M;
    Bcsc.max_nnz  = B->nnz_max;
    Bcsc.max_M    = B->M_max;
    Bcsc.row      = new index_t[Bcsc.nnz];
    Bcsc.val      = new value_t[Bcsc.nnz];
    Bcsc.col_scan = new index_t[Bcsc.col_sz + 1];

    std::fill(&Bcsc.col_scan[0], &Bcsc.col_scan[Bcsc.col_sz + 1], 0);
    index_t *Bc_tmp   = &Bcsc.col_scan[1];
    index_t *Bc_tmp_p = &Bc_tmp[0] - B->split[rank]; // use this to avoid subtracting a fixed number,

    for(nnz_t i = 0; i < Bcsc.nnz; ++i){
        Bcsc.row[i] = B_ent[i].col;
        Bcsc.val[i] = B_ent[i].val;
        ++Bc_tmp_p[B_ent[i].row];
    }

    for(nnz_t i = 0; i < Bcsc.col_sz; ++i){
        Bcsc.col_scan[i+1] += Bcsc.col_scan[i];
    }

    Bcsc.split    = B->split;
    Bcsc.nnz_list = B->nnz_list;

#ifdef __DEBUG1__
    assert(Bcsc.nnz == (Bcsc.col_scan[Bcsc.col_sz] - Bcsc.col_scan[0]));
    // print B info
    {
//        B->print_entry(0);
//        printf("B: nnz_l: %ld\tnnz_g: %ld\tM: %d\tM_big: %d\n", B->nnz_l, B->nnz_g, B->M, B->Mbig);
//        printf("rank %d: B: nnz_max: %ld\tM_max: %d\n", rank, B->nnz_max, B->M_max);
//        print_array(Bcsc.row, Bcsc.nnz, 0, "Bcsc.row", comm);
//        print_array(Bcsc.col_scan, Bcsc.col_sz+1, 0, "Bcsc.col_scan", comm);
//        print_array(Bcsc.val, Bcsc.nnz, 0, "Bcsc.val", comm);

//        if(rank == 0){
//            std::cout << "\nB: nnz: " << B->nnz_l << std::endl ;
//            for(index_t j = 0; j < B->M; ++j){
//                for(index_t i = Bcsc.col_scan[j]; i < Bcsc.col_scan[j+1]; ++i){
//                    std::cout << std::setprecision(4) << Bcsc.row[i] << "\t" << j << "\t" << Bcsc.val[i] << std::endl;
//                    assert( fabs(Bcsc.val[i]) > ALMOST_ZERO );
//                }
//            }
//        }
    }
#endif

    // =======================================
    // prepare B for compression
    // =======================================

    if(print_timing){
        MPI_Barrier(comm);
        t_temp = MPI_Wtime();
    }

    Bcsc.compress_prep();

    // =======================================
    // Preallocate Memory
    // =======================================

    matmat_memory(A, B, Bcsc.max_comp_sz);

    if(print_timing){
        t_temp = MPI_Wtime() - t_temp;
        t_matmat_prep = t_temp;
    }

    // =======================================
    // perform the multiplication
    // =======================================

    if(!print_timing){
        matmat_CSC(Acsc, Bcsc, *C);
        if(assemble){
            matmat_assemble(A, B, C);
        }
    }else{
        // warmup
        case1 = 0, case2 = 0, case3 = 0;
        t_init_prep = 0, t_mat = 0, t_comp = 0, t_decomp = 0, t_prep_iter = 0, t_fast_mm = 0, t_sort = 0, t_wait = 0;
        for (int i = 0; i < matmat_iter_warmup; ++i) {
            case1_iter = 0;
            case2_iter = 0;
            case3_iter = 0;
            saena_matrix C_tmp(A->comm);
            matmat_CSC(Acsc, Bcsc, C_tmp);
        }

        case1 = 0, case2 = 0, case3 = 0;
        t_init_prep = 0, t_mat = 0, t_comp = 0, t_decomp = 0, t_prep_iter = 0, t_fast_mm = 0, t_sort = 0, t_wait = 0;
        for (int i = 0; i < matmat_iter; ++i) {
            case1_iter = 0;
            case2_iter = 0;
            case3_iter = 0;
            saena_matrix C_tmp(A->comm);

            MPI_Barrier(comm);
            t_temp = MPI_Wtime();

            matmat_CSC(Acsc, Bcsc, C_tmp);

            t_temp = MPI_Wtime() - t_temp;
            t_matmat += t_temp;
        }

        t_matmat_tot = t_matmat_prep + (t_matmat / matmat_iter);

        //===============
        // print timings
        //===============

        if (!rank) printf("\n");
        print_time_ave(t_matmat_tot, "Saena matmat", comm, true, true);

        if(!rank){
            auto orig_sz = sizeof(value_t) * Bcsc.nnz;
            if(!rank) std::cout << "\nrank " << rank << ": orig sz = " << zfp_orig_sz << ", zfp comp sz = " << zfp_comp_sz
                                << ", saving " << ( 1.0f - ( (float)zfp_comp_sz / (float)orig_sz ) ) << std::endl;
        }

        if (!rank) printf("\ninit prep\ncomm\nfastmm\nsort\nprep_iter\nwait\nt_comp\nt_decomp\n");
        print_time_ave(t_init_prep / matmat_iter,                       "t_init_prep", comm, true, false);
        print_time_ave((t_mat - t_prep_iter - t_fast_mm) / matmat_iter, "comm", comm, true, false);
        print_time_ave(t_fast_mm / matmat_iter,                         "t_fast_mm", comm, true, false);
        print_time_ave(t_sort / matmat_iter,                            "t_sort", comm, true, false);
        print_time_ave(t_prep_iter / matmat_iter,                       "t_prep_iter", comm, true, false);
        print_time_ave(t_wait / matmat_iter,                            "t_wait", comm, true, false);
        print_time_ave(t_comp / matmat_iter,                            "t_comp", comm, true, false);
        print_time_ave(t_decomp / matmat_iter,                          "t_decomp", comm, true, false);
        if (!rank) printf("\n");

        if (!rank) printf("case1\ncase2\ncase3\n");
        print_time_ave(case1 / matmat_iter, "case1", comm, true, false);
        print_time_ave(case2 / matmat_iter, "case2", comm, true, false);
        print_time_ave(case3 / matmat_iter, "case3", comm, true, false);

    }

//    case1_iter_ave = average_iter(case1_iter, comm);
//    case2_iter_ave = average_iter(case2_iter, comm);
//    case3_iter_ave = average_iter(case3_iter, comm);
//    if(rank==0){
//        printf("case iters:\n%.0f\n%.0f\n%.0f\n", case1_iter_ave, case2_iter_ave, case3_iter_ave);
//        printf("\ncase1 = %.0f\ncase2 = %.0f\ncase3 = %.0f\n", case1_iter_ave, case2_iter_ave, case3_iter_ave);
//    }

    // =======================================
    // finalize
    // =======================================

//    mat_send.clear();
//    mat_send.shrink_to_fit();
//    AB_temp.clear();
//    AB_temp.shrink_to_fit();

    delete []Acsc.row;
    delete []Acsc.val;
    delete []Acsc.col_scan;
    delete []Bcsc.row;
    delete []Bcsc.val;
    delete []Bcsc.col_scan;

//    delete []mempool1;
//    delete []mempool2;
    delete []mempool3;
    delete []mempool4;
    delete []mempool5;
    delete []mempool6;
    delete []mempool7;

    return 0;
}


int saena_object::matmat_memory(saena_matrix *A, saena_matrix *B, nnz_t &comp_max_sz){

#ifdef __DEBUG1__
//    int rank;
//    MPI_Comm_rank(A->comm, &rank);
//    int verbose_rank = 0;
#endif

    // =======================================
    // Preallocate Memory
    // =======================================

    // mempool1: used for the dense buffer
    // mempool2: usages: for A: 1- nnzPerRow_left and 2- orig_row_idx. for B: 3- B_new_col_idx and 4- orig_col_idx
    // mempool2 size: 2 * A_row_size + 2 * B_col_size
    // important: it depends if B or transpose of B is being used for the multiplciation. if originl B is used then
    // B_col_size is B->Mbig, but for the transpos it is B->M, which is scalable.
    // mempool3: is used to store the the received part of B.
    // mempool3 size: it should store remote B, so we allocate the max size of B on all the procs.
    //                sizeof(row index) + sizeof(value) + sizeof(col_scan) =
    //                nnz * index_t + nnz * value_t + (col_size+1) * index_t

//    index_t A_row_size = A->M;
//    index_t B_col_size = B->Mbig; // for original B
//    index_t B_col_size = B->M;      // for when tranpose of B is used to do the multiplication.

//    mempool1 = new value_t[matmat_size_thre2];
//    mempool2 = new index_t[2 * A_row_size + 2 * Bcsc.max_M];

    // used to store mat_current in matmat()
    // valbyidx for value, (B->M_max + 1) for col_scan
    // r_cscan_buffer_sz_max is for both row and col_scan which have the same type.
    int   valbyidx              = sizeof(value_t) / sizeof(index_t);
    nnz_t v_buffer_sz_max       = valbyidx * B->nnz_max;
    nnz_t r_cscan_buffer_sz_max = B->nnz_max + B->M_max + 1;
          mempool3_sz           = v_buffer_sz_max + r_cscan_buffer_sz_max;

    try{
        mempool3 = new index_t[mempool3_sz]; // used to store mat_current in matmat()
    }catch(std::bad_alloc& ba){
        std::cerr << "bad_alloc caught: " << ba.what() << '\n';
    }

    mempool4and5_sz = std::max(A->nnz_max, B->nnz_max);

    try{
        mempool4 = new index_t[mempool4and5_sz]; //used in reorder_split and reorder_back_split
    }catch(std::bad_alloc& ba){
        std::cerr << "bad_alloc caught: " << ba.what() << '\n';
    }

    try{
        mempool5 = new value_t[mempool4and5_sz]; //used in reorder_split and reorder_back_split
    }catch(std::bad_alloc& ba){
        std::cerr << "bad_alloc caught: " << ba.what() << '\n';
    }

    assert(comp_max_sz != 0);
    mempool6_sz = 2 * (comp_max_sz + B->nnz_max * sizeof(value_t));

    try{
        mempool6 = new uchar[mempool6_sz]; // one for mat_send, one for mat_recv. both are compressed.
    }catch(std::bad_alloc& ba){
        std::cerr << "bad_alloc caught: " << ba.what() << '\n';
    }

    // the buffer size is sometimes bigger than the original array, so choosing 2 to be safe.
    mempool7_sz = 2 * B->nnz_max * sizeof(value_t);

    try{
        mempool7 = new uchar[mempool7_sz]; // used as the zfp compressor and decompressor buffer
    }catch(std::bad_alloc& ba){
        std::cerr << "bad_alloc caught: " << ba.what() << '\n';
    }

//    mempool1 = std::make_unique<value_t[]>(matmat_size_thre2);
//    mempool2 = std::make_unique<index_t[]>(A->Mbig * 4);

#ifdef __DEBUG1__
//    if(rank==0) std::cout << "vecbyint = " << vecbyint << std::endl;

//    print_vector(B->nnz_list, verbose_rank, "B->nnz_list", B->comm);
//    if(rank==verbose_rank) std::cout << "loc_nnz_max: " << loc_nnz_max << "\n";

//    if(rank==0){
//        std::cout << "mempool1 size = " << matmat_size_thre2 << std::endl;
//        std::cout << "mempool2 size = " << 2 * A_row_size + 2 * Bcsc.col_sz << std::endl;
//        std::cout << "mempool3 size = " << 2 * send_size_max << std::endl;
//        std::cout << "B->nnz_max = " << B->nnz_max << ", B->M_max = " << B->M_max << ", valbyidx = " << valbyidx << std::endl;
//    }
#endif

    return 0;
}


int saena_object::matmat_assemble(saena_matrix *A, saena_matrix *B, saena_matrix *C){

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // =======================================
    // set C parameters
    // =======================================

    C->Mbig  = A->Mbig;
    C->Nbig  = B->Mbig;
    C->split = A->split;
    C->M     = C->split[rank+1] - C->split[rank];
    C->M_old = C->M;

    C->nnz_l = C->entry.size();
    MPI_Allreduce(&C->nnz_l, &C->nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

    C->comm            = A->comm;
    C->comm_old        = A->comm;
    C->active_old_comm = true;
    C->active          = true;
    C->active_minor    = true;

    // set dense parameters
    C->density         = (float)C->nnz_g / C->Mbig / C->Mbig;
    C->switch_to_dense = switch_to_dense;
    C->dense_threshold = dense_threshold;

    // set shrink parameters
    C->last_M_shrink       = A->last_M_shrink;
    C->last_density_shrink = A->last_density_shrink;
    C->cpu_shrink_thre1    = A->cpu_shrink_thre1; //todo: is this required?

//    if(A->cpu_shrink_thre2_next_level != -1) // this is -1 by default.
//        C->cpu_shrink_thre2 = A->cpu_shrink_thre2_next_level;
    //return these to default, since they have been used in the above part.
//    A->cpu_shrink_thre2_next_level = -1;
//    A->enable_shrink_next_level    = false;

#ifdef __DEBUG1__
    if(verbose_matmat_assemble) {
        MPI_Barrier(comm);
        printf("C: rank = %d \tMbig = %u  \tNbig = %u \tM = %u \tnnz_g = %lu \tnnz_l = %lu \tdensity = %f\n",
               rank, C->Mbig, C->Nbig, C->M, C->nnz_g, C->nnz_l, C->density);
        MPI_Barrier(comm);
    }
#endif

    C->matrix_setup();

#ifdef __DEBUG1__
    if(verbose_matmat_assemble){
        MPI_Barrier(comm); printf("matmat_assemble: rank = %d done!\n", rank); MPI_Barrier(comm);}
#endif

    return 0;
}


int saena_object::matmat_CSC(CSCMat &Acsc, CSCMat &Bcsc, saena_matrix &C){

    MPI_Comm comm = C.comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    int verbose_rank = 0;
#ifdef __DEBUG1__
    if (verbose_matmat) {
        MPI_Barrier(comm);
        if (rank == verbose_rank) printf("\nstart of matmat: nprocs: %d \n", nprocs);
        MPI_Barrier(comm);
    }
//    assert(Acsc.col_sz == Bcsc.row_sz);
#endif

    double t_temp, t_temp2, t_temp3;
    t_temp = MPI_Wtime();

    // =======================================
    // perform the multiplication - serial implementation
    // =======================================
    // this needs to be updated.

//    std::vector<cooEntry> AB_temp;
//    fast_mm(Arv, Brv, AB_temp,
//            A->M, A->split[rank], A->Mbig, 0, B->M, B->split[rank],
//            &Ac[0], &Bc[0], A->comm);

#ifdef __DEBUG1__
//    print_vector(AB_temp, -1, "AB_temp", comm);
#endif

    // =======================================
    // communication and multiplication - parallel implementation
    // =======================================

//    int   valbyidx              = sizeof(value_t) / sizeof(index_t);
//    nnz_t v_buffer_sz_max       = valbyidx * Bcsc.max_nnz;
//    nnz_t r_cscan_buffer_sz_max = Bcsc.max_nnz + Bcsc.max_M + 1;
//    nnz_t send_size_max         = v_buffer_sz_max + r_cscan_buffer_sz_max;

#ifdef __DEBUG1__
//    ASSERT(send_size <= send_size_max, "send_size: " << send_size << ", send_size_max: " << send_size_max);
    if (verbose_matmat) {
        MPI_Barrier(comm);
        if (rank == verbose_rank) printf("matmat: step 1\n");
        MPI_Barrier(comm);
//        if (rank == verbose_rank) std::cout << "send_nnz: " << send_nnz      << ",\tsend_size: " << send_size
//                                    << ",\tsend_size_max: " << send_size_max << ",\tAcsc_M: "    << Acsc_M << std::endl;
//        MPI_Barrier(comm);
//        printf("row: r: %d, \tq: %d, \ttot: %d\n", Bcsc.comp_row.r, Bcsc.comp_row.q, Bcsc.comp_row.tot);
//        printf("col: r: %d, \tq: %d, \ttot: %d\n", Bcsc.comp_col.r, Bcsc.comp_col.q, Bcsc.comp_col.tot);
    }
#endif

    index_t Acsc_M = Acsc.split[rank+1] - Acsc.split[rank];
    CSCMat_mm A(Acsc_M, Acsc.split[rank], Acsc.col_sz, 0, Acsc.nnz, Acsc.row, Acsc.val, Acsc.col_scan);

    std::vector<cooEntry> AB_temp;

    if(nprocs > 1){
        // set the mat_send data
        // structure of mat_send:
        // 1- row:    type: index_t, size: send_nnz
        // 2- c_scan: type: index_t, size: Bcsc.col_sz + 1
        // 3- val:    type: value_t, size: send_nnz
        auto mat_send = &mempool6[0];

        t_temp3 = MPI_Wtime();

        assert(mempool6_sz != 0);
        std::fill(mempool6, &mempool6[mempool6_sz], 0);

        // compress row and col_scan of B, communicate it, decompress it and use it
        GR_encoder encoder(mempool6_sz / 2);
        encoder.compress(Bcsc.row, Bcsc.nnz, Bcsc.comp_row.k, mat_send);
        encoder.compress(Bcsc.col_scan, Bcsc.col_sz+1, Bcsc.comp_col.k, &mat_send[Bcsc.comp_row.tot]);

        // zfp: compress values
        zfp_field*  field = zfp_field_1d(&Bcsc.val[0], zfp_type_double, Bcsc.nnz);
        zfp_stream* zfp   = zfp_stream_open(nullptr);
        zfp_stream_set_reversible(zfp);

        size_t bufsize   = zfp_stream_maximum_size(zfp, field);
        void* zfp_buffer = mempool7;
//        void* zfp_buffer = malloc(bufsize);

        bitstream* stream = stream_open(zfp_buffer, bufsize);
        zfp_stream_set_bit_stream(zfp, stream);
        zfp_stream_rewind(zfp);

        size_t zfpsize = zfp_compress(zfp, field); // The number of bytes of compressed storage that is returned.
        if (!zfpsize) {
            fprintf(stderr, "rank %d: compression failed\n", rank);
        }

        zfp_orig_sz = static_cast<long>(sizeof(value_t)) * Bcsc.nnz;
        zfp_comp_sz = zfpsize;

#ifdef __DEBUG1__
        if(rank == verbose_rank){
//            auto orig_sz = sizeof(value_t) * Bcsc.nnz;
//            if(!rank) std::cout << "rank " << rank << ": orig sz = " << orig_sz << ", zfp comp sz = " << zfpsize
//                                << ", saving " << ( 1.0f - ( (float)zfpsize / (float)orig_sz ) ) << std::endl;
        }
#endif

//        zfp_field_free(field);
//        zfp_stream_close(zfp);
//        stream_close(stream);
//        free(zfp_buffer);

        t_temp3 = MPI_Wtime() - t_temp3;
        t_comp += t_temp3;

        // copy B.val at the end of the compressed array
        auto mat_send_v = reinterpret_cast<value_t*>(&mat_send[Bcsc.comp_row.tot + Bcsc.comp_col.tot]);
//        memcpy(mat_send_v, Bcsc.val, Bcsc.nnz * sizeof(value_t));
        memcpy(mat_send_v, zfp_buffer, zfpsize);

        std::vector<long> zfp_comp_szs(nprocs);
        MPI_Allgather(&zfpsize, 1, MPI_LONG, &zfp_comp_szs[0], 1, MPI_LONG, comm);

        nnz_t send_nnz   = Bcsc.nnz;
        nnz_t row_buf_sz = tot_sz(send_nnz,        Bcsc.comp_row.k, Bcsc.comp_row.q);
        nnz_t col_buf_sz = tot_sz(Bcsc.col_sz + 1, Bcsc.comp_col.k, Bcsc.comp_col.q);
        nnz_t send_size  = row_buf_sz + col_buf_sz + zfpsize; // in bytes

#ifdef __DEBUG1__
        {
            if (verbose_matmat) {
                MPI_Barrier(comm);
                if (rank == verbose_rank) printf("matmat: step 2\n");
                MPI_Barrier(comm);
            }

//            if(rank==verbose_rank){
//                for(int i = 0; i < Bcsc.nnz; ++i){
//                    std::cout << i << "\t" << mat_send_v[i] << std::endl;
//                }
//            }
        }
#endif

#if 0
        auto mat_current_r     = &mat_send[0];
        auto mat_current_cscan = &mat_send[send_nnz];
        auto mat_current_v     = reinterpret_cast<value_t*>(&mat_send[send_nnz + Bcsc.col_sz + 1]);

        memcpy(mat_current_r,     Bcsc.row,      Bcsc.nnz * sizeof(index_t));
        memcpy(mat_current_cscan, Bcsc.col_scan, (Bcsc.col_sz + 1) * sizeof(index_t));
        memcpy(mat_current_v,     Bcsc.val,      Bcsc.nnz * sizeof(value_t));

#ifdef __DEBUG1__
        if (verbose_matmat) {
            MPI_Barrier(comm);
            if (rank == verbose_rank) printf("matmat: step 2\n");
            MPI_Barrier(comm);
//            print_vector(Bcsc.split, 0, "Bcsc.split", comm);
//            print_vector(Bcsc.nnz_list, 0, "Bcsc.nnz_list", comm);
//            MPI_Barrier(comm);
        }
//        print_array(mat_send_cscan, Bcsc.col_sz+1, 1, "mat_send_cscan", comm);
//        MPI_Barrier(comm);
        assert(send_nnz == (mat_current_cscan[Bcsc.col_sz] - mat_current_cscan[0]));
#endif
#endif

        int right_neighbor = (rank + 1)%nprocs;
        int left_neighbor  = rank - 1;
        if (left_neighbor < 0){
            left_neighbor += nprocs;
        }

        // set the mat_recv data
        nnz_t recv_nnz, value_nnz;
        nnz_t recv_size;
        index_t row_comp_sz, col_comp_sz, current_comp_sz;
        index_t mat_recv_M = 0, mat_current_M = 0;
        auto mat_recv = &mempool6[mempool6_sz / 2];

        auto mat_current = &mempool3[0];
        index_t *mat_current_r, *mat_current_cscan;
        value_t *mat_current_v;

        auto mat_temp = mat_send;
        int  owner, next_owner;
        auto *requests = new MPI_Request[2];
        auto *statuses = new MPI_Status[2];

        CSCMat_mm S;

        t_temp = MPI_Wtime() - t_temp;
        t_init_prep += t_temp;
        t_temp2 = MPI_Wtime();

        for(int k = rank; k < rank+nprocs; ++k){
            // This is overlapped. Both local and remote loops are done here.
            // The first iteration is the local loop. The rest are remote.
            // Send R_tranpose to the left_neighbor processor, receive R_tranpose from the right_neighbor.
            // In the next step: send R_tranpose that was received in the previous step to the left_neighbor processor,
            // receive R_tranpose from the right_neighbor. And so on.
            // --------------------------------------------------------------------

            t_temp = MPI_Wtime();

            if(k != rank+nprocs-1) { // do not perform the communication for the last one, because it's just each processor's data.
                next_owner = (k + 1) % nprocs;
                mat_recv_M = Bcsc.split[next_owner + 1] - Bcsc.split[next_owner];
                recv_nnz = Bcsc.nnz_list[next_owner];
                value_nnz = zfp_comp_szs[next_owner];

                row_buf_sz = tot_sz(recv_nnz, Bcsc.comp_row.ks[next_owner], Bcsc.comp_row.qs[next_owner]);
                col_buf_sz = tot_sz(mat_recv_M + 1, Bcsc.comp_col.ks[next_owner], Bcsc.comp_col.qs[next_owner]);
                recv_size  = row_buf_sz + col_buf_sz + value_nnz; // in bytes

                t_temp = MPI_Wtime() - t_temp;
                t_prep_iter += t_temp;

#ifdef __DEBUG1__
                ASSERT(row_buf_sz + col_buf_sz <= Bcsc.max_comp_sz, "rank " << rank << ": row_buf:" << row_buf_sz
                                                                            << "\tcol_buf: " << col_buf_sz
                                                                            << ",\tBcsc.max_comp_sz: "
                                                                            << Bcsc.max_comp_sz);
                if (verbose_matmat) {
                    MPI_Barrier(comm);
                    if (rank == verbose_rank) printf("matmat: step 3 - in for loop\n");
                    MPI_Barrier(comm);
                    printf("rank %d: next_owner: %4d, recv_nnz: %4lu, recv_size: %4lu, send_nnz = %4lu, send_size: %4lu, mat_recv_M: %4u\n",
                           rank, next_owner, recv_nnz, recv_size, send_nnz, send_size, mat_recv_M);
                    MPI_Barrier(comm);
                }
#endif

                // communicate data
                MPI_Irecv(mat_recv, recv_size, MPI_CHAR, right_neighbor, right_neighbor, comm, requests);
                MPI_Isend(mat_send, send_size, MPI_CHAR, left_neighbor, rank, comm, requests + 1);

                int flag;
                MPI_Test(requests, &flag, statuses);
                MPI_Test(requests+1, &flag, statuses+1);
            }

            // =======================================
            // perform the multiplication
            // =======================================
            if(Acsc.nnz != 0 && send_nnz != 0){

                t_temp = MPI_Wtime();

                owner         = k%nprocs;
                mat_current_M = Bcsc.split[owner + 1] - Bcsc.split[owner]; //this is col_sz

                // decompress mat_send into mat_current
                row_comp_sz     = tot_sz(send_nnz,          Bcsc.comp_row.ks[owner], Bcsc.comp_row.qs[owner]);
                col_comp_sz     = tot_sz(mat_current_M + 1, Bcsc.comp_col.ks[owner], Bcsc.comp_col.qs[owner]);
                current_comp_sz = row_comp_sz + col_comp_sz;

                mat_current_r     = &mat_current[0];
                mat_current_cscan = &mat_current[send_nnz];
                mat_current_v     = reinterpret_cast<value_t*>(&mat_current[send_nnz + mat_current_M + 1]);

                t_temp3 = MPI_Wtime();

                encoder.decompress(mat_current_r,     send_nnz,          Bcsc.comp_row.ks[owner], Bcsc.comp_row.qs[owner], mat_send);
                encoder.decompress(mat_current_cscan, mat_current_M + 1, Bcsc.comp_col.ks[owner], Bcsc.comp_col.qs[owner], &mat_send[row_comp_sz]);

                zfp_field* field2 = zfp_field_1d(mat_current_v, zfp_type_double, Bcsc.nnz_list[owner]);
                stream = stream_open(&mat_send[current_comp_sz], zfp_comp_szs[owner]);
                zfp_stream_set_bit_stream(zfp, stream);
                zfp_stream_rewind(zfp);
                zfp_decompress(zfp, field2);

                t_temp3 = MPI_Wtime() - t_temp3;
                t_decomp += t_temp3;

//                memcpy(mat_current_v, &mat_send[current_comp_sz], Bcsc.nnz_list[owner] * sizeof(value_t));

#ifdef __DEBUG1__
                {
//                if(rank==verbose_rank) printf("row_comp_sz: %d, col_comp_sz: %d, current_comp_sz: %d\n", row_comp_sz, col_comp_sz, current_comp_sz);
//                MPI_Barrier(comm);
//                auto mat_send_vv = reinterpret_cast<value_t*>(&mat_send[current_comp_sz]);
//                if(rank==verbose_rank){
//                    std::cout << "Bcsc.nnz_list[owner]: " << Bcsc.nnz_list[owner] << std::endl;
//                    for(int i = 0; i < Bcsc.nnz_list[owner]; ++i){
//                        std::cout << i << "\t" << mat_current_r[i] << "\t" << mat_current_v[i] << "\t" << mat_send_vv[i] << std::endl;
//                    }
//                    std::cout << std::endl;
//                }

//                MPI_Barrier(comm);
//                if(rank==verbose_rank) {
//                    index_t ofst = Bcsc.split[owner], col_idx;
//                    std::cout << "\nmat that is received. original owner: " << owner << ", mat_current_M: " << mat_current_M << "\n";
//                    for (nnz_t i = 0; i < mat_current_M; i++) {
//                        col_idx = i + ofst;
//                        for (nnz_t j = mat_current_cscan[i]; j < mat_current_cscan[i + 1]; j++) {
//                            std::cout << j << "\t" << mat_current_r[j] << "\t" << col_idx << "\t" << mat_current_v[j] << std::endl;
//                        }
//                    }
//                }
//                MPI_Barrier(comm);
                }
#endif

                S.set_params(Acsc.col_sz, 0, mat_current_M, Bcsc.split[owner], Bcsc.nnz_list[owner],
                             mat_current_r, mat_current_v, mat_current_cscan);

                t_temp = MPI_Wtime() - t_temp;
                t_prep_iter += t_temp;

#ifdef __DEBUG1__
                // ===============
                // assert mat_send
                // ===============
                {
                    ASSERT(Bcsc.nnz_list[owner] == (mat_current_cscan[mat_current_M] - mat_current_cscan[0]),
                           "rank: " << rank << ", owner: " << owner << ", mat_current_M: " << mat_current_M
                                    << ", Bcsc.nnz_list[owner]: " << Bcsc.nnz_list[owner]
                                    << ", mat_current_cscan[0]: " << mat_current_cscan[0]
                                    << ", mat_current_cscan[mat_current_M]: " << mat_current_cscan[mat_current_M]);

//                index_t ofst  = Bcsc.split[owner], col_idx;
                    for (nnz_t i = 0; i < mat_current_M; ++i) {
//                    col_idx = i + ofst;
                        for (nnz_t j = mat_current_cscan[i]; j < mat_current_cscan[i + 1]; j++) {
//                        assert( (col_idx >= Bcsc.split[owner]) && (col_idx < Bcsc.split[owner+1]) ); //this is always true
                            assert( (mat_current_r[j] >= 0) && (mat_current_r[j] < Bcsc.split.back()) );
                        }
                    }
                    assert(S.nnz == (S.col_scan[S.col_sz] - S.col_scan[0]));
                }
#endif

                t_temp = MPI_Wtime();

                fast_mm(A, S, AB_temp, comm);

                t_temp = MPI_Wtime() - t_temp;
                t_fast_mm += t_temp;
#ifdef __DEBUG1__
                assert(S.nnz == (S.col_scan[S.col_sz] - S.col_scan[0]));
#endif
#if 0
                // =======================================
                // sort and remove duplicates
                // =======================================
                if(!AB_temp.empty()) {

                    t_temp = MPI_Wtime();

                    std::sort(AB_temp.begin(), AB_temp.end());

                    nnz_t AP_temp_size_minus1 = AB_temp.size() - 1;
                    for (nnz_t i = 0; i < AB_temp.size(); i++) {
                        C.entry.emplace_back(AB_temp[i]);
                        while (i < AP_temp_size_minus1 && AB_temp[i] == AB_temp[i + 1]) { // values of entries with the same row and col should be added.
//                            std::cout << AB_temp[i] << "\t" << AB_temp[i+1] << std::endl;
                            C.entry.back().val += AB_temp[++i].val;
                        }
                    }

                    AB_temp.clear();
                    t_temp = MPI_Wtime() - t_temp;
                    t_sort_dup += t_temp;

                }
#endif
            }

            if(k != rank+nprocs-1) {
                t_temp = MPI_Wtime();

                MPI_Waitall(2, requests, statuses);

                t_temp = MPI_Wtime() - t_temp;
                t_wait += t_temp;

#ifdef __DEBUG1__
                {
                    if (verbose_matmat) {
                        MPI_Barrier(comm);
                        if (rank == verbose_rank) printf("matmat: step 4 - in for loop\n");
                        MPI_Barrier(comm);
                    }

                    assert(reorder_counter == 0);
                }
#endif

                t_temp = MPI_Wtime();

                send_size = recv_size;
                send_nnz  = recv_nnz;

                // swap mat_send and mat_recv
//            mat_recv.swap(mat_send);
//            std::swap(mat_send, mat_recv);
                mat_temp = mat_send;
                mat_send = mat_recv;
                mat_recv = mat_temp;

                t_temp = MPI_Wtime() - t_temp;
                t_prep_iter += t_temp;

#ifdef __DEBUG1__
                if (verbose_matmat) {
                    MPI_Barrier(comm);
                    if (rank == verbose_rank) printf("matmat: step 5 - in for loop\n");
                    MPI_Barrier(comm);
                }

                // info about mat_recv and mat_send
                {
//                MPI_Barrier(comm);
//                if(rank==verbose_rank) {
//                    owner = (k+1)%nprocs;
//                    index_t ofst = Bcsc.split[owner], col_idx;
//                    std::cout << "\nmat that is received. original owner: " << owner << "\n";
//                    for (nnz_t i = 0; i < mat_recv_M; i++) {
//                        col_idx = i + ofst;
//                        for (nnz_t j = mat_send_cscan[i]; j < mat_send_cscan[i + 1]; j++) {
//                            std::cout << j << "\t" << mat_send_r[j] << "\t" << col_idx << "\t" << mat_send_v[j] << std::endl;
//                        }
//                    }
//                }
//
//                MPI_Barrier(comm);
//                if(rank==0){
//                    std::cout << "print received matrix: mat_recv_M: " << mat_recv_M << ", col_offset: "
//                              << B->split[k%nprocs] << std::endl;
//
//                    print_array(mat_send_cscan, mat_recv_M+1, 0, "mat_send_cscan", comm);
//                }
//                MPI_Barrier(comm);
//
//                print_vector(AB_temp, -1, "AB_temp", A->comm);
//                print_vector(mat_send, -1, "mat_send", A->comm);
//                print_vector(mat_recv, -1, "mat_recv", A->comm);
//                prev_owner = owner;
//                printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
                }
#endif
            }
        }

        delete [] requests;
        delete [] statuses;

        t_temp2 = MPI_Wtime() - t_temp2;
        t_mat += t_temp2;

#ifdef __DEBUG1__
        if (verbose_matmat) {
            MPI_Barrier(comm);
            if (rank == verbose_rank) printf("matmat: step 6\n");
            MPI_Barrier(comm);
        }
#endif

        zfp_field_free(field);
//        zfp_field_free(field2);
        zfp_stream_close(zfp);
        stream_close(stream);
//        free(zfp_buffer);

    } else { // nprocs == 1 -> serial

#ifdef __DEBUG1__
        if (verbose_matmat) {
            if (rank == verbose_rank) printf("matmat: step 2 serial\n");
        }
#endif

#ifdef __DEBUG1__
/*
// use this to debug the compression in serial case
        {
            auto mat_send = &mempool6[0];
            std::fill(mempool6, &mempool6[mempool6_sz], 0);

//        printf("row: r: %d, \tq: %d, \ttot: %d\n", Bcsc.comp_row.r, Bcsc.comp_row.q, Bcsc.comp_row.tot);
//        printf("col: r: %d, \tq: %d, \ttot: %d\n", Bcsc.comp_col.r, Bcsc.comp_col.q, Bcsc.comp_col.tot);

            // compress
            GR_encoder encoder(mempool6_sz / 2);
            encoder.compress(Bcsc.row, Bcsc.nnz, Bcsc.comp_row.k, mat_send);
            encoder.compress(Bcsc.col_scan, Bcsc.col_sz+1, Bcsc.comp_col.k, &mat_send[Bcsc.comp_row.tot]);
            auto mat_send_v = reinterpret_cast<value_t*>(&mat_send[Bcsc.comp_row.tot + Bcsc.comp_col.tot]);
            memcpy(mat_send_v, Bcsc.val, Bcsc.nnz * sizeof(value_t));

            index_t row_comp_sz, col_comp_sz, current_comp_sz;
            index_t mat_current_M = 0;

            auto mat_current = &mempool3[0];
            index_t *mat_current_r, *mat_current_cscan;
            value_t *mat_current_v;

            int k = rank;
            int owner = k%nprocs;
            mat_current_M = Bcsc.split[owner + 1] - Bcsc.split[owner]; //this is col_sz

            // decompress mat_send into mat_current
            row_comp_sz     = tot_sz(send_nnz, Bcsc.comp_row.ks[owner], Bcsc.comp_row.qs[owner]);
            col_comp_sz     = tot_sz(mat_current_M + 1, Bcsc.comp_col.ks[owner], Bcsc.comp_col.qs[owner]);
            current_comp_sz = row_comp_sz + col_comp_sz;

//            if(rank==verbose_rank) printf("row_comp_sz: %d, col_comp_sz: %d, current_comp_sz: %d\n", row_comp_sz, col_comp_sz, current_comp_sz);

            mat_current_r     = &mat_current[0];
            mat_current_cscan = &mat_current[send_nnz];
            mat_current_v     = reinterpret_cast<value_t*>(&mat_current[send_nnz + mat_current_M + 1]);

            encoder.decompress(mat_current_r, send_nnz, Bcsc.comp_row.ks[owner], Bcsc.comp_row.qs[owner], mat_send);
            encoder.decompress(mat_current_cscan, mat_current_M + 1, Bcsc.comp_col.ks[owner], Bcsc.comp_col.qs[owner], &mat_send[row_comp_sz]);
            memcpy(mat_current_v, &mat_send[current_comp_sz], Bcsc.nnz_list[owner] * sizeof(value_t));

//            MPI_Barrier(comm);
//            auto mat_send_vv = reinterpret_cast<value_t*>(&mat_send[current_comp_sz]);
//            if(rank==verbose_rank){
//                std::cout << "Bcsc.nnz_list[owner]: " << Bcsc.nnz_list[owner] << std::endl;
//                for(int i = 0; i < Bcsc.nnz_list[owner]; ++i){
//                    std::cout << i << "\t" << mat_current_r[i] << "\t" << mat_current_v[i] << "\t" << mat_send_vv[i] << std::endl;
//                }
//                std::cout << std::endl;
//            }

            CSCMat_mm S;
            S.set_params(Acsc.col_sz, 0, mat_current_M, Bcsc.split[owner], Bcsc.nnz_list[owner],
                         mat_current_r, mat_current_v, mat_current_cscan);
            fast_mm(A, S, AB_temp, comm);
        }
*/
#endif

        if(Acsc.nnz != 0 && Bcsc.nnz != 0){
            CSCMat_mm B(Acsc.col_sz, 0, Bcsc.col_sz, Bcsc.split[rank], Bcsc.nnz, Bcsc.row, Bcsc.val, Bcsc.col_scan);
            fast_mm(A, B, AB_temp, comm);
        }
    }

#ifdef __DEBUG1__
//    print_vector(AB_temp, -1, "AB_temp", comm);
#endif

    // =======================================
    // sort and remove duplicates
    // =======================================

    t_temp = MPI_Wtime();
//#if 0
    if(!AB_temp.empty()) {

        std::sort(AB_temp.begin(), AB_temp.end());

        nnz_t AP_temp_size_minus1 = AB_temp.size() - 1;
        for (nnz_t i = 0; i < AB_temp.size(); i++) {
            C.entry.emplace_back(AB_temp[i]);
            while (i < AP_temp_size_minus1 && AB_temp[i] == AB_temp[i + 1]) { // values of entries with the same row and col should be added.
//                std::cout << AB_temp[i] << "\t" << AB_temp[i+1] << std::endl;
                C.entry.back().val += AB_temp[++i].val;
            }
        }

    }
//#endif
//    std::sort(C.entry.begin(), C.entry.end());
    t_temp = MPI_Wtime() - t_temp;
    t_sort += t_temp;

#ifdef __DEBUG1__
    {
//        print_vector(C.entry, -1, "C.entry", comm);
//        writeMatrixToFile(C.entry, "matrix_folder/result", comm);

        if (verbose_matmat) {
            MPI_Barrier(comm);
            if (rank == verbose_rank) printf("end of matmat\n\n");
            MPI_Barrier(comm);
        }
    }
#endif

    return 0;
}


int saena_object::matmat_grid(Grid *grid){
/*
    saena_matrix *A    = grid->A;
    prolong_matrix *P  = &grid->P;
    restrict_matrix *R = &grid->R;
//    saena_matrix *Ac   = &grid->Ac;

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // *******************************************************
    // part 1: multiply: AP_temp = A_i * P_j. in which P_j = R_j_tranpose and 0 <= j < nprocs.
    // *******************************************************

//    double t_AP = MPI_Wtime();

    unsigned long send_size     = R->entry.size();
    unsigned long send_size_max = R->nnz_max;
//    MPI_Allreduce(&send_size, &send_size_max, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);

    // local transpose of R is being used to compute A*P. So R is transposed locally here.
//    std::vector<cooEntry> mat_send(R->entry.size());
    auto mat_send = new cooEntry[send_size_max];
//    transpose_locally(&R->entry[0], R->entry.size(), R->splitNew[rank], &mat_send[0]);

    memcpy(&mat_send[0], &P->entry[0], P->entry.size() * sizeof(cooEntry));
    std::sort(&mat_send[0], &mat_send[P->entry.size()], row_major);

//    std::vector<cooEntry> mat_send = P->entry;
//    std::sort(mat_send.begin(), mat_send.end(), row_major);

#ifdef __DEBUG1__
//    print_vector(nnzPerColScan_left, -1, "nnzPerColScan_left", comm);
//    print_vector(P->splitNew, -1, "P->splitNew", comm);

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 4: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // compute the maximum size for nnzPerCol_right and nnzPerColScan_right
    index_t *nnzPerColScan_left = &A->nnzPerColScan[0];
    index_t mat_recv_M_max = R->M_max;

    std::vector<index_t> nnzPerColScan_right(mat_recv_M_max + 1);
    index_t *nnzPerCol_right = &nnzPerColScan_right[1];
    index_t *nnzPerCol_right_p = &nnzPerCol_right[0]; // use this to avoid subtracting a fixed number,

    std::vector<cooEntry> AP_temp;

//    printf("\n");
    if(nprocs > 1){

        int right_neighbor = (rank + 1)%nprocs;
        int left_neighbor  = rank - 1;
        if (left_neighbor < 0){
            left_neighbor += nprocs;
        }

        int owner;
        unsigned long recv_size;
        auto mat_recv = new cooEntry[send_size_max];
        index_t mat_recv_M;

        auto *requests = new MPI_Request[4];
        auto *statuses = new MPI_Status[4];

        for(int k = rank; k < rank+nprocs; k++){
            // This is overlapped. Both local and remote loops are done here.
            // The first iteration is the local loop. The rest are remote.
            // Send R_tranpose to the left_neighbor processor, receive R_tranpose from the right_neighbor.
            // In the next step: send R_tranpose that was received in the previous step to the left_neighbor processor,
            // receive R_tranpose from the right_neighbor. And so on.
            // --------------------------------------------------------------------

            // communicate size
            MPI_Irecv(&recv_size, 1, MPI_UNSIGNED_LONG, right_neighbor, right_neighbor, comm, requests);
            MPI_Isend(&send_size, 1, MPI_UNSIGNED_LONG, left_neighbor,  rank,           comm, requests+1);
            MPI_Waitall(1, requests, statuses);
//          printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
//            mat_recv.resize(recv_size);

#ifdef __DEBUG1__
//          print_vector(mat_recv, -1, "mat_recv", A->comm);
//          print_vector(mat_send, -1, "mat_send", A->comm);
#endif
            // communicate data
            MPI_Irecv(&mat_recv[0], recv_size, cooEntry::mpi_datatype(), right_neighbor, right_neighbor, comm, requests+2);
            MPI_Isend(&mat_send[0], send_size, cooEntry::mpi_datatype(), left_neighbor,  rank,           comm, requests+3);

            owner = k%nprocs;
            mat_recv_M = P->splitNew[owner + 1] - P->splitNew[owner];
//          printf("rank %d: owner = %d, mat_recv_M = %d, B_col_offset = %u \n", rank, owner, mat_recv_M, P->splitNew[owner]);

            std::fill(&nnzPerCol_right[0], &nnzPerCol_right[mat_recv_M], 0);
            nnzPerCol_right_p = &nnzPerCol_right[0] - P->splitNew[owner];
            for(nnz_t i = 0; i < send_size; i++){
                nnzPerCol_right_p[mat_send[i].col]++;
            }

//            nnzPerColScan_right[0] = 0;
            for(nnz_t i = 0; i < mat_recv_M; i++){
                nnzPerColScan_right[i+1] += nnzPerColScan_right[i];
            }

#ifdef __DEBUG1__
//          print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
//          print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

            if(A->entry.empty() || send_size==0){ // skip!
#ifdef __DEBUG1__
                if(verbose_compute_coarsen){
                    if(A->entry.empty()){
                        printf("\nskip: A->entry.size() == 0\n\n");
                    } else {
                        printf("\nskip: mat_send == 0\n\n");
                    }
                }
#endif
            } else {

                fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), send_size,
                        A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[owner],
                        &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
                        &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//                fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), mat_send.size(),
//                        A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[owner],
//                        &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
//                        &nnzPerColScan_right[0], &nnzPerColScan_right[1], map_matmat, A->comm);

            }

            MPI_Waitall(3, requests+1, statuses+1);

            std::swap(mat_send, mat_recv);
            send_size = recv_size;

#ifdef __DEBUG1__
//          print_vector(AP_temp, -1, "AP_temp", A->comm);
//          print_vector(mat_send, -1, "mat_send", A->comm);
//          print_vector(mat_recv, -1, "mat_recv", A->comm);
//          prev_owner = owner;
//          printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
#endif

        }

        delete [] mat_recv;
        delete [] requests;
        delete [] statuses;

    } else { // nprocs == 1 -> serial

        index_t mat_recv_M = P->splitNew[rank + 1] - P->splitNew[rank];

        std::fill(&nnzPerCol_right[0], &nnzPerCol_right[mat_recv_M], 0);
        nnzPerCol_right_p = &nnzPerCol_right[0] - P->splitNew[rank];
        for(nnz_t i = 0; i < send_size; i++){
            nnzPerCol_right_p[mat_send[i].col]++;
        }

        for(nnz_t i = 0; i < mat_recv_M; i++){
            nnzPerColScan_right[i+1] += nnzPerColScan_right[i];
        }

#ifdef __DEBUG1__
//          print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
//          print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

        if(A->entry.empty() || send_size == 0){ // skip!
#ifdef __DEBUG1__
            if(verbose_compute_coarsen){
                if(A->entry.empty()){
                    printf("\nskip: A->entry.size() == 0\n\n");
                } else {
                    printf("\nskip: mat_send == 0\n\n");
                }
            }
#endif
        } else {

//            double t1 = MPI_Wtime();

            fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), send_size,
                    A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[rank],
                    &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
                    &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//            double t2 = MPI_Wtime();
//            printf("\nfast_mm of AP_temp = %f\n", t2-t1);
        }
    }

    std::sort(AP_temp.begin(), AP_temp.end());
    std::vector<cooEntry> AP;
    nnz_t AP_temp_size_minus1 = AP_temp.size()-1;
    for(nnz_t i = 0; i < AP_temp.size(); i++){
        AP.emplace_back(AP_temp[i]);
        while(i < AP_temp_size_minus1 && AP_temp[i] == AP_temp[i+1]){ // values of entries with the same row and col should be added.
            AP.back().val += AP_temp[++i].val;
        }
    }

    delete [] mat_send;
    AP_temp.clear();
    AP_temp.shrink_to_fit();

//    t_AP = MPI_Wtime() - t_AP;
//    print_time_ave(t_AP, "AP:\n", grid->A->comm);

#ifdef __DEBUG1__
//    print_vector(AP_temp, -1, "AP_temp", A->comm);
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 5: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

*/

    return 0;
}