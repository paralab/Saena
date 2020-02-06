#include "saena_object.h"
#include "saena_matrix.h"
#include "strength_matrix.h"
#include "grid.h"
#include "aux_functions.h"
#include "parUtils.h"
#include "dollar.hpp"

#include <mkl_spblas.h>

#include <cstdio>
#include <fstream>
#include <algorithm>
#include <mpi.h>
#include <iomanip>


const double ALMOST_ZERO = 1e-16;

double case0 = 0, case11 = 0, case12 = 0, case2 = 0, case3 = 0; // for timing case parts of fast_mm

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

//    if(!rank) std::cout << __func__ << std::endl;

//    nnz_t A_nnz = A.col_scan[A.col_sz] - A.col_scan[0];
//    nnz_t B.nnz = B.col_scan[B.col_sz] - B.col_scan[0];

    index_t A_col_size_half = A.col_sz/2;

    int verbose_rank = 1;

#ifdef __DEBUG1__
    if(rank==verbose_rank && verbose_fastmm) printf("\nfast_mm: start \n");

    // assert A entries
    index_t col_idx;
    for (nnz_t i = 0; i < A.col_sz; i++) {
//        col_idx = i + A.col_offset;
        for (nnz_t j = A.col_scan[i]; j < A.col_scan[i + 1]; j++) {
//            std::cout << j << "\t" << A.r[j] << "\t" << col_idx << "\t" << A.v[j] << "\n";
            assert( (A.r[j] >= 0) && (A.r[j] < A.row_sz) );
            assert( i < A.col_sz );
            assert( fabs(A.v[j]) > ALMOST_ZERO );

//            assert( (col_idx >= A.col_offset) && (col_idx < A.col_offset + A.col_sz) );
        }
    }

    // assert B entries
    for (nnz_t i = 0; i < B.col_sz; i++) {
//        col_idx = i + B.col_offset;
        for (nnz_t j = B.col_scan[i]; j < B.col_scan[i + 1]; j++) {
//            std::cout << j << "\t" << B.r[j] << "\t" << col_idx << "\t" << B.v[j] << "\n";
            assert( (B.r[j] >= 0) && (B.r[j] < B.row_sz) );
            assert( i < B.col_sz );
            assert( fabs(B.v[j]) > ALMOST_ZERO );

//            assert( (col_idx >= B.col_offset) && (col_idx < B.col_offset + B.col_sz) );
        }
    }

    if(rank==verbose_rank){

        if(verbose_matmat_A){
            std::cout << "\nA: nnz = "       << A.nnz
                      << ", A_row_size = "   << A.row_sz     << ", A_col_size = "   << A.col_sz
                      << ", A_row_offset = " << A.row_offset << ", A_col_offset = " << A.col_offset << std::endl;

//            print_array(A.col_scan, A.col_sz+1, verbose_rank, "A.col_scan", comm);

            // print entries of A:
            std::cout << "\nA: nnz = " << A.nnz << std::endl;
//            index_t col_idx;
            for(nnz_t i = 0; i < A.col_sz; i++){
                col_idx = i + A.col_offset;
                for(nnz_t j = A.col_scan[i]; j < A.col_scan[i+1]; j++) {
                    std::cout << j << "\t" << A.r[j]+A.row_offset << "\t" << col_idx << "\t" << A.v[j] << std::endl;
                }
            }
        }

        if(verbose_matmat_B) {
            std::cout << "\nB: nnz = "       << B.nnz;
            std::cout << ", B_row_size = "   << B.row_sz     << ", B_col_size = "   << B.col_sz
                      << ", B_row_offset = " << B.row_offset << ", B_col_offset = " << B.col_offset << std::endl;

//            print_array(B.col_scan, B_col_size+1, 1, "B.col_scan", comm);

            // print entries of B:
            std::cout << "\nB: nnz = " << B.nnz << std::endl;
            for (nnz_t i = 0; i < B.col_sz; i++) {
                col_idx = i + B.col_offset;
                for (nnz_t j = B.col_scan[i]; j < B.col_scan[i+1]; j++) {
                    std::cout << j << "\t" << B.r[j]+B.row_offset << "\t" << col_idx << "\t" << B.v[j] << std::endl;
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
#endif

    // case1
    // ==============================================================

    if (A.row_sz * B.col_sz < matmat_size_thre1) { //DOLLAR("case1")
//    if (case2_iter == 1) {

//        if (rank == verbose_rank) printf("fast_mm: case 1: start \n");

#ifdef __DEBUG1__
        if (rank == verbose_rank && (verbose_fastmm || verbose_matmat_recursive)) {
            printf("fast_mm: case 1: start \n");
        }
        ++case1_iter;
#endif

//        double t1 = MPI_Wtime();

        sparse_matrix_t Amkl = nullptr;
//        mkl_sparse_d_create_csc(&Amkl, SPARSE_INDEX_BASE_ZERO, A.row_sz + A.row_offset, A.col_sz, (int*)A.col_scan, (int*)(A.col_scan+1), (int*)Ar, Av);
//        mkl_sparse_d_create_csc(&Amkl, SPARSE_INDEX_BASE_ZERO, A.row_sz, A.col_sz, (int*)A.col_scan, (int*)(A.col_scan+1), (int*)(Ar - A.row_offset), Av);
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
//        mkl_sparse_d_create_csc(&Bmkl, SPARSE_INDEX_BASE_ZERO, B.row_sz + B.row_offset, B.col_sz, (int*)B.col_scan, (int*)(B.col_scan+1), (int*)Br, Bv);
//        mkl_sparse_d_create_csc(&Bmkl, SPARSE_INDEX_BASE_ZERO, B.row_sz, B.col_sz, (int*)B.col_scan, (int*)(B.col_scan+1), (int*)(Br - B.row_offset), Bv);
        mkl_sparse_d_create_csc(&Bmkl, SPARSE_INDEX_BASE_ZERO, B.row_sz, B.col_sz, (int*)B.col_scan, (int*)(B.col_scan+1), (int*)B.r, B.v);

#ifdef __DEBUG1__

//        MPI_Barrier(comm);
//        if(rank==1) printf("\nPerform MKL matmult\n"); fflush(nullptr);
//        MPI_Barrier(comm);

//        auto mkl_res = mkl_sparse_spmm( SPARSE_OPERATION_NON_TRANSPOSE, Amkl, Bmkl, &Cmkl );

//        if(mkl_res != SPARSE_STATUS_SUCCESS){
//            goto memory_free;
//        }

//        goto export_c;

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

        return;
#endif

        mkl_sparse_destroy(Cmkl);
        mkl_sparse_destroy(Bmkl);
        mkl_sparse_destroy(Amkl);

//        MPI_Barrier(comm);
//        if(rank==1) printf("rank %d: DONE\n", rank); fflush(nullptr);
//        MPI_Barrier(comm);

        return;
    }

    // ==============================================================
    // Case2
    // ==============================================================

    index_t A_col_scan_end, B_col_scan_end;
//    mat_info A1_info, A2_info, B1_info, B2_info;

    // if A_col_size_half == 0, it means A_col_size = 1. In this case it goes to case3.
    if (A.row_sz <= A.col_sz && A_col_size_half != 0){//DOLLAR("case2")
//    if (case2_iter == 0){

        double t2 = MPI_Wtime();

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_fastmm) { printf("fast_mm: case 2: start \n"); }
        ++case2_iter;
#endif
//        if (rank == verbose_rank) { printf("fast_mm: case 2: start \n"); }

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

//        CSCMat_mm A1(A.row_sz, A.row_offset, A_col_size_half, A.col_offset, A.col_scan[A_col_size_half] - A.col_scan[0],
//                      A.r, A.v, A.col_scan);

//        CSCMat_mm A2(A.row_sz, A.row_offset, A.col_sz - A1.col_sz, A.col_offset + A1.col_sz, A.nnz - A1.nnz,
//                      A.r, A.v, &A.col_scan[A_col_size_half]);

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

        // =======================================================

        // split B based on how A is split, so use A_col_size_half to split B. A_col_size_half can be different based on
        // choosing the splitting method (nnz or size).
        index_t B_row_size_half = A_col_size_half;
//        index_t B_row_threshold = B_row_size_half + B.row_offset;
//        index_t B_row_threshold = B_row_size_half;

#ifdef SPLIT_SIZE
#ifdef __DEBUG1__
        assert(B_row_size_half == B.row_sz / 2);
#endif
#endif

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
        B2.col_scan = new index_t[B.col_sz + 1];
        B2.free_c   = true;

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

        reorder_split(B, B1, B2);

        if(B2.nnz == 0){
            delete []B2.col_scan;
            B2.free_c = false;
        }

        B1.r = &B.r[0];
        B1.v = &B.v[0];
        B2.r = &B.r[B1.col_scan[B.col_sz]];
        B2.v = &B.v[B1.col_scan[B.col_sz]];

        t2 = MPI_Wtime() - t2;
        case2 += t2;

#ifdef __DEBUG1__

//        if(A1.nnz == 0 || A2.nnz == 0 || B1.nnz == 0 || B2.nnz == 0)
//            printf("rank %d: nnzs: %lu\t%lu\t%lu\t%lu\t\n", rank, A1.nnz, A2.nnz, B1.nnz, B2.nnz);

        // assert A1
//        std::cout << "\nCase2:\nA1: nnz = " << A1.nnz << std::endl;
        if(A1.nnz != 0) {
            for (nnz_t i = 0; i < A1.col_sz; i++) {
                for (nnz_t j = A1.col_scan[i]; j < A1.col_scan[i + 1]; j++) {
                    assert((A1.r[j] >= 0) && (A1.r[j] < A1.row_sz));
//                std::cout << j << "\t" << A1.r[j] << "\t" << i + A1.col_offset << "\t" << A1.v[j] << std::endl;
                }
            }
        }

        // assert A2
//        std::cout << "\nA2: nnz = " << A2.nnz << std::endl;
        if(A2.nnz != 0) {
            for (nnz_t i = 0; i < A2.col_sz; i++) {
                for (nnz_t j = A2.col_scan[i]; j < A2.col_scan[i + 1]; j++) {
                    assert((A2.r[j] >= 0) && (A2.r[j] < A2.row_sz));
//                std::cout << j << "\t" << A2.r[j] << "\t" << i + A2.col_offset << "\t" << A2.v[j] << std::endl;
                }
            }
        }

        // assert B1
//        std::cout << "\nCase2:\nB1: nnz = " << B1.nnz << std::endl;
        if(B1.nnz != 0) {
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
        if(B2.nnz != 0){
            for (nnz_t i = 0; i < B2.col_sz; i++) {
                for (nnz_t j = B2.col_scan[i]; j < B2.col_scan[i + 1]; j++) {
                    assert( (B2.r[j] >= 0) && (B2.r[j] < B2.row_sz) );
//                    std::cout << j << "\t" << B2.r[j] << "\t" << i + B2.col_offset << "\t" << B2.v[j] << "\n";
                }
            }
        }

        if (rank == verbose_rank) {

//        MPI_Barrier(comm);
//        std::cout << "B_row_threshold: " << std::setw(3) << B_row_threshold << std::endl;
//        print_array(Bc1, B_col_size+1, 0, "Bc1", comm);
//        print_array(Bc2, B_col_size+1, 0, "Bc2", comm);
//        print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);

//        std::cout << "\ncase2:\nA.nnz: " << std::setw(3) << A.nnz << ", A1.nnz: " << std::setw(3) << A1.nnz << ", A2.nnz: " << std::setw(3) << A2.nnz << ", A_col_size: "
//                  << std::setw(3) << A.col_sz << ", A_col_size_half: " << std::setw(3) << A_col_size_half << std::endl;

//        printf("fast_mm: case 2: \nA.nnz: (%lu, %lu, %lu), B.nnz: (%lu, %lu, %lu)\n"
//               "A_size: (%u, %u, %u), B_size: (%u, %u) \n",
//               A.nnz, A1.nnz, A2.nnz, B.nnz, B1.nnz, B2.nnz, A_row_size, A_col_size, A_col_size_half, A_col_size, B_col_size);

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
                if(A1.nnz != 0) {
                    for (nnz_t i = 0; i < A1.col_sz; i++) {
                        for (nnz_t j = A1.col_scan[i]; j < A1.col_scan[i + 1]; j++) {
                            std::cout << j << "\t" << A1.r[j]+A1.row_offset << "\t" << i + A1.col_offset << "\t" << A1.v[j] << "\n";
                        }
                    }
                }

                // print entries of A2:
                std::cout << "\nA2: nnz = " << A2.nnz << std::endl;
                if(A2.nnz != 0) {
                    for (nnz_t i = 0; i < A2.col_sz; i++) {
                        for (nnz_t j = A2.col_scan[i]; j < A2.col_scan[i + 1]; j++) {
                            std::cout << j << "\t" << A2.r[j]+A2.row_offset << "\t" << i + A2.col_offset << "\t" << A2.v[j] << "\n" ;
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
                if(B1.nnz != 0) {
                    for (nnz_t i = 0; i < B1.col_sz; i++) {
                        for (nnz_t j = B1.col_scan[i]; j < B1.col_scan[i + 1]; j++) {
                            std::cout << j << "\t" << B1.r[j]+B1.row_offset << "\t" << i + B1.col_offset << "\t" << B1.v[j] << "\n";
                        }
                    }
                }

                // print entries of B2:
                std::cout << "\nB2: nnz = " << B2.nnz << std::endl;
                if(B2.nnz != 0){
                    for (nnz_t i = 0; i < B2.col_sz; i++) {
                        for (nnz_t j = B2.col_scan[i]; j < B2.col_scan[i + 1]; j++) {
                            std::cout << j << "\t" << B2.r[j]+B2.row_offset << "\t" << i + B2.col_offset << "\t" << B2.v[j] << "\n";
                        }
                    }
                }
            }
        }
//        MPI_Barrier(comm);
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

        // Split Fact 1:
        // The last element of A1.col_scan is shared with the first element of A2.col_scan, and it may gets changed
        // during the recursive calls from the A1.col_scan side. So, save that and use it for the starting
        // point of A2.col_scan inside the recursive calls.

        if (A1.nnz != 0 && B1.nnz != 0) {

            // Check Split Fact 1 for this part.
            A_col_scan_end = A1.col_scan[A1.col_sz];
            B_col_scan_end = B1.col_scan[B1.col_sz];

            fast_mm(A1, B1, C, comm);

//            fast_mm(&A1.r[0] - A1.row_offset, &A1.v[0], &A1.col_scan[0],
//                    &B1.r[0] - B1.row_offset, &B1.v[0], &B1.col_scan[0],
//                    &A1_info, &B1_info,
//                    C, comm);

//            fast_mm(&A1.r[0], &A1.v[0], &A1.col_scan[0],
//                    &B1.r[0], &B1.v[0], &B1.col_scan[0],
//                    &A1_info, &B1_info,
//                    C, comm);

            A1.col_scan[A1.col_sz] = A_col_scan_end;
            B1.col_scan[B1.col_sz] = B_col_scan_end;

        }

        // C2 = A2 * B2
#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 2 \n");
#endif

        if (A2.nnz != 0 && B2.nnz != 0) {

            A_col_scan_end = A2.col_scan[A2.col_sz];
            B_col_scan_end = B2.col_scan[B2.col_sz];

            fast_mm(A2, B2, C, comm);

//            fast_mm(&A2.r[0] - A2.row_offset, &A2.v[0], &A2.col_scan[0],
//                    &B2.r[0] - B2.row_offset, &B2.v[0], &B2.col_scan[0],
//                    &A2_info, &B2_info,
//                    C, comm);

//            fast_mm(&A2.r[0], &A2.v[0], &A2.col_scan[0],
//                    &B2.r[0], &B2.v[0], &B2.col_scan[0],
//                    &A2_info, &B2_info,
//                    C, comm);

            A2.col_scan[A2.col_sz] = A_col_scan_end;
            B2.col_scan[B2.col_sz] = B_col_scan_end;

        }

        t2 = MPI_Wtime();

        // return B to its original order.
        if(B2.nnz != 0) {
//            reorder_back_split(B.r, B.v, B1.col_scan, B2.col_scan, B.col_sz, B_row_size_half);
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

        double t3 = MPI_Wtime();

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_fastmm) printf("fast_mm: case 3: start \n");
        ++case3_iter;
#endif
//        if (rank == verbose_rank) printf("fast_mm: case 3: start \n");

        // split based on matrix size
        // =======================================================

#ifdef SPLIT_SIZE
        // prepare splits of matrix B by column

//        index_t A_col_size_half = A_col_size/2;
        index_t B_col_size_half = B.col_sz/2;

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

#endif

#ifdef __DEBUG1__
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

        index_t A_row_size_half = A.row_sz / 2;
//        index_t A_row_threshold = A_row_size_half + A.row_offset;
//        index_t A_row_threshold = A_row_size_half;

        CSCMat_mm A1, A2;

        A1.row_sz = A_row_size_half;
        A2.row_sz = A.row_sz - A1.row_sz;

        A1.row_offset = A.row_offset;
        A2.row_offset = A.row_offset + A1.row_sz;

        A1.col_sz = A.col_sz;
        A2.col_sz = A.col_sz;

        A1.col_offset = A.col_offset;
        A2.col_offset = A.col_offset;

        A1.col_scan = A.col_scan; // col_scan
        A2.col_scan = new index_t[A.col_sz + 1]; // col_scan
        A2.free_c = true;

        reorder_split(A, A1, A2);
//        reorder_split(A.r, A.v, A1.col_scan, A2.col_scan, A.col_sz, A_row_threshold, A_row_size_half);

//        A1.nnz = A1.col_scan[A.col_sz] - A1.col_scan[0];
//        A2.nnz = A2.col_scan[A.col_sz] - A2.col_scan[0];

        if(A2.nnz == 0){
            delete []A2.col_scan;
            A2.free_c = false;
        }

        A1.r = &A.r[0];
        A1.v = &A.v[0];
        A2.r = &A.r[A1.col_scan[A.col_sz]];
        A2.v = &A.v[A1.col_scan[A.col_sz]];

        t3 = MPI_Wtime() - t3;
        case3 += t3;

#ifdef __DEBUG1__
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
                if(A2.nnz != 0) {
                    for (nnz_t i = 0; i < A1.col_sz; i++) {
                        for (nnz_t j = A1.col_scan[i]; j < A1.col_scan[i + 1]; j++) {
                            std::cout << j << "\t" << A1.r[j]+A1.row_offset << "\t" << i + A1.col_offset << "\t" << A1.v[j] << "\n";
                        }
                    }
                }

                // print entries of A2:
                std::cout << "\nA2: nnz = " << A2.nnz << std::endl;
                if(A2.nnz != 0) {
                    for (nnz_t i = 0; i < A2.col_sz; i++) {
                        for (nnz_t j = A2.col_scan[i]; j < A2.col_scan[i + 1]; j++) {
                            std::cout << j << "\t" << A2.r[j]+A2.row_offset << "\t" << i + A2.col_offset << "\t" << A2.v[j] << "\n";
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
                if(B1.nnz != 0){
                    for (nnz_t i = 0; i < B1.col_sz; i++) {
                        for (nnz_t j = B1.col_scan[i]; j < B1.col_scan[i + 1]; j++) {
                            std::cout << j << "\t" << B1.r[j]+B1.row_offset << "\t" << i + B1.col_offset << "\t" << B1.v[j] << "\n";
                        }
                    }
                }

                // print entries of B2:
                std::cout << "\nB2: nnz = " << B2.nnz << std::endl;
                if(B2.nnz != 0) {
                    for (nnz_t i = 0; i < B2.col_sz; i++) {
                        for (nnz_t j = B2.col_scan[i]; j < B2.col_scan[i + 1]; j++) {
                            std::cout << j << "\t" << B2.r[j]+B2.row_offset << "\t" << i + B2.col_offset << "\t" << B2.v[j] << "\n";
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

        // C1 = A1 * B1:
//        fast_mm(A1, B1, C_temp, A_row_size_half, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size_half, B_col_offset, comm);
        // C2 = A2 * B1:
//        fast_mm(A2, B1, C_temp, A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset, B_row_offset, B_col_size_half, B_col_offset, comm);
        // C3 = A1 * B2:
//        fast_mm(A1, B2, C_temp, A_row_size_half, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size-B_col_size_half, B_col_offset+B_col_size_half, comm);
        // C4 = A2 * B2
//        fast_mm(A2, B2, C_temp, A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset, B_row_offset, B_col_size-B_col_size_half, B_col_offset+B_col_size_half, comm);

        // =======================================================

        // C1 = A1 * B1
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 1 \n");
#endif

        if (A1.nnz != 0 && B1.nnz != 0) {

            A_col_scan_end = A1.col_scan[A1.col_sz];
            B_col_scan_end = B1.col_scan[B1.col_sz];

            fast_mm(A1, B1, C, comm);

//            fast_mm(&A1.r[0] - A1.row_offset, &A1.v[0], &A1.col_scan[0],
//                    &B1.r[0] - B1.row_offset, &B1.v[0], &B1.col_scan[0],
//                    &A1_info, &B1_info,
//                    C, comm);

//            fast_mm(&A1.r[0], &A1.v[0], &A1.col_scan[0],
//                    &B1.r[0], &B1.v[0], &B1.col_scan[0],
//                    &A1_info, &B1_info,
//                    C, comm);

            A1.col_scan[A1.col_sz] = A_col_scan_end;
            B1.col_scan[B1.col_sz] = B_col_scan_end;

        }


        // C2 = A1 * B2:
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 2 \n");
#endif

        if (A1.nnz != 0 && B2.nnz != 0) {

            A_col_scan_end = A1.col_scan[A1.col_sz];
            B_col_scan_end = B2.col_scan[B2.col_sz];

            fast_mm(A1, B2, C, comm);

//            fast_mm(&A1.r[0] - A1.row_offset, &A1.v[0], &A1.col_scan[0],
//                    &B2.r[0] - B2.row_offset, &B2.v[0], &B2.col_scan[0],
//                    &A1_info, &B2_info,
//                    C, comm);

//            fast_mm(&A1.r[0], &A1.v[0], &A1.col_scan[0],
//                    &B2.r[0], &B2.v[0], &B2.col_scan[0],
//                    &A1_info, &B2_info,
//                    C, comm);

//            fast_mm(&A1[0], &B2[0], C,
//                    A1_row_size, A1_row_offset, A1_col_size, A1_col_offset,
//                    B2_col_size, B2_col_offset,
//                    &Ac1[0], &Bc2[0], comm);

            A1.col_scan[A1.col_sz] = A_col_scan_end;
            B2.col_scan[B2.col_sz] = B_col_scan_end;

        }


        // C3 = A2 * B1
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 3 \n");
#endif

        if (A2.nnz != 0 && B1.nnz != 0) {

            A_col_scan_end = A2.col_scan[A2.col_sz];
            B_col_scan_end = B1.col_scan[B1.col_sz];

            fast_mm(A2, B1, C, comm);

//            fast_mm(&A2.r[0] - A2.row_offset, &A2.v[0], &A2.col_scan[0],
//                    &B1.r[0] - B1.row_offset, &B1.v[0], &B1.col_scan[0],
//                    &A2_info, &B1_info,
//                    C, comm);

//            fast_mm(&A2[0], &B1[0], C,
//                    A2_row_size, A2_row_offset, A2_col_size, A2_col_offset,
//                    B1_col_size, B1_col_offset,
//                    &Ac2[0], &Bc1[0], comm);

            A2.col_scan[A2.col_sz] = A_col_scan_end;
            B1.col_scan[B1.col_sz] = B_col_scan_end;

        }


        // C4 = A2 * B2
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 4 \n");
#endif

        if (A2.nnz != 0 && B2.nnz != 0) {

            A_col_scan_end = A2.col_scan[A2.col_sz];
            B_col_scan_end = B2.col_scan[B2.col_sz];

            fast_mm(A2, B2, C, comm);

//            fast_mm(&A2.r[0] - A2.row_offset, &A2.v[0], &A2.col_scan[0],
//                    &B2.r[0] - B2.row_offset, &B2.v[0], &B2.col_scan[0],
//                    &A2_info, &B2_info,
//                    C, comm);

//            fast_mm(&A2.r[0], &A2.v[0], &A2.col_scan[0],
//                    &B2.r[0], &B2.v[0], &B2.col_scan[0],
//                    &A2_info, &B2_info,
//                    C, comm);

//            fast_mm(&A2[0], &B2[0], C,
//                    A2_row_size, A2_row_offset, A2_col_size, A2_col_offset,
//                    B2_col_size, B2_col_offset,
//                    &Ac2[0], &Bc2[0], comm);

            A2.col_scan[A2.col_sz] = A_col_scan_end;
            B2.col_scan[B2.col_sz] = B_col_scan_end;

        }

        t3 = MPI_Wtime();

        // return A to its original order.
        if(A2.nnz != 0){
//            reorder_back_split(A.r, A.v, A1.col_scan, A2.col_scan, A.col_sz, A_row_size_half);
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


int saena_object::matmat(saena_matrix *A, saena_matrix *B, saena_matrix *C, const bool assemble){
    // This version only works when B is symmetric, since local transpose of B is used.
    // Use B's row indices as column indices and vice versa.

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // =======================================
    // Convert A to CSC
    // =======================================

#ifdef __DEBUG1__
    for(nnz_t i = 0; i < A->nnz_l; i++){
        assert( (A->entry[i].row - A->split[rank] >= 0) && (A->entry[i].row - A->split[rank] < A->M) );
        assert( (A->entry[i].col >= 0) && (A->entry[i].col < A->Mbig) );
//        assert( fabs(A->entry[i].val) > 1e-14 );
    }
#endif

//    auto Arv = new vecEntry[A->nnz_l]; // row and val
//    auto Ac  = new index_t[A->Mbig+1]; // col_idx

    // todo: change to smart pointers
//    auto Arv = std::make_unique<vecEntry[]>(A->nnz_l); // row and val
//    auto Ac  = std::make_unique<index_t[]>(A->Mbig+1); // col_idx

    CSCMat Acsc;
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
    assert(Acsc.col_scan[Acsc.col_sz] == Acsc.nnz);

//    A->print_entry(0);
//    printf("A: nnz_l: %ld\tnnz_g: %ld\tM: %d\tM_big: %d\n", A->nnz_l, A->nnz_g, A->M, A->Mbig);
//    print_array(Acsc.row, Acsc.nnz, 0, "Acsc.row", comm);
//    print_array(Acsc.val, Acsc.nnz, 0, "Acsc.val", comm);
//    print_array(Acsc.col_scan, Acsc.col_sz + 1, 0, "Acsc.col_scan", comm);

//    std::cout << "\nA: nnz: " << Acsc.nnz << std::endl ;
//    for(index_t j = 0; j < Acsc.col_sz; j++){
//        for(index_t i = Acsc.col_scan[j]; i < Acsc.col_scan[j+1]; i++){
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
//        assert( fabs(B->entry[i].val - 0) > ALMOST_ZERO );
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

    for(nnz_t i = 0; i < Bcsc.nnz; i++){
        Bcsc.row[i] = B_ent[i].col;
        Bcsc.val[i] = B_ent[i].val;
        Bc_tmp_p[B_ent[i].row]++;
    }

    for(nnz_t i = 0; i < Bcsc.col_sz; i++){
        Bcsc.col_scan[i+1] += Bcsc.col_scan[i];
    }

    Bcsc.split    = B->split;
    Bcsc.nnz_list = B->nnz_list;

#ifdef __DEBUG1__
    assert(Bcsc.col_scan[Bcsc.col_sz] == Bcsc.nnz);

//    B->print_entry(0);
//    printf("B: nnz_l: %ld\tnnz_g: %ld\tM: %d\tM_big: %d\n", B->nnz_l, B->nnz_g, B->M, B->Mbig);
//    printf("rank %d: B: nnz_max: %ld\tM_max: %d\n", rank, B->nnz_max, B->M_max);
//    print_array(Bc, B->M+1, 0, "Bc", comm);
//
//    std::cout << "\nB: nnz: " << B->nnz_l << std::endl ;
//    for(index_t j = 0; j < B->M; j++){
//        for(index_t i = Bcsc.col_scan[j]; i < Bcsc.col_scan[j+1]; i++){
//            std::cout << std::setprecision(4) << Bcsc.row[i] << "\t" << j << "\t" << Bcsc.val[i] << std::endl;
//        }
//    }
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

    // 2 for both send and receive buffer, valbyidx for value, (B->M_max + 1) for col_scan
    // r_cscan_buffer_sz_max is for both row and col_scan which have the same type.
    int   valbyidx              = sizeof(value_t) / sizeof(index_t);
    nnz_t v_buffer_sz_max       = valbyidx * B->nnz_max;
    nnz_t r_cscan_buffer_sz_max = B->nnz_max + B->M_max + 1;
    nnz_t send_size_max         = v_buffer_sz_max + r_cscan_buffer_sz_max;

    try{
        mempool3 = new index_t[2 * send_size_max];
    }catch(std::bad_alloc& ba){
        std::cerr << "bad_alloc caught: " << ba.what() << '\n';
    }

    loc_nnz_max = std::max(A->nnz_max, B->nnz_max);

    try{
        mempool4 = new index_t[2 * loc_nnz_max];
    }catch(std::bad_alloc& ba){
        std::cerr << "bad_alloc caught: " << ba.what() << '\n';
    }

    try{
        mempool5 = new value_t[2 * loc_nnz_max];
    }catch(std::bad_alloc& ba){
        std::cerr << "bad_alloc caught: " << ba.what() << '\n';
    }

//    mempool1 = std::make_unique<value_t[]>(matmat_size_thre2);
//    mempool2 = std::make_unique<index_t[]>(A->Mbig * 4);

#ifdef __DEBUG1__
//    if(rank==0) std::cout << "vecbyint = " << vecbyint << std::endl;

//    if(rank==0){
//        std::cout << "mempool1 size = " << matmat_size_thre2 << std::endl;
//        std::cout << "mempool2 size = " << 2 * A_row_size + 2 * Bcsc.col_sz << std::endl;
//        std::cout << "mempool3 size = " << 2 * send_size_max << std::endl;
//        std::cout << "B->nnz_max = " << B->nnz_max << ", B->M_max = " << B->M_max << ", valbyidx = " << valbyidx << std::endl;
//    }
#endif

    // =======================================
    // perform the multiplication
    // =======================================

    matmat(Acsc, Bcsc, *C, send_size_max);

    if(assemble){
        matmat_assemble(A, B, C);
    }

#ifdef __DEBUG1__
    if(rank==0){
        printf("\nrank %d: case1 = %u, case2 = %u, case3 = %u\n", rank, case1_iter, case2_iter, case3_iter);
    }
    case1_iter = 0;
    case2_iter = 0;
    case3_iter = 0;
#endif

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

int saena_object::matmat_ave(saena_matrix *A, saena_matrix *B, double &matmat_time){
    // This version only works on symmetric matrices, since local transpose of B is being used.
    // this version is only for experiments.
    // B1 should be symmetric. Because we need its transpose. Use its row indices as column indices and vice versa.

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // =======================================
    // Convert A to CSC
    // =======================================

//    auto Arv = new vecEntry[A->nnz_l]; // row and val
//    auto Ac  = new index_t[A->Mbig+1]; // col_idx

    // todo: change to smart pointers
//    auto Arv = std::make_unique<vecEntry[]>(A->nnz_l); // row and val
//    auto Ac  = std::make_unique<index_t[]>(A->Mbig+1); // col_idx

    CSCMat Acsc;
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
        Acsc.val[i] = A->entry[i].val;
        Ac_tmp[A->entry[i].col]++;
    }

    for(nnz_t i = 0; i < Acsc.col_sz; i++){
        Acsc.col_scan[i+1] += Acsc.col_scan[i];
    }

    Acsc.split    = A->split;
    Acsc.nnz_list = A->nnz_list;

#ifdef __DEBUG1__
//    A->print_entry(0);
//    printf("A: nnz_l: %ld\tnnz_g: %ld\tM: %d\tM_big: %d\n", A->nnz_l, A->nnz_g, A->M, A->Mbig);
//    print_array(Acsc.row, Acsc.nnz, 0, "Acsc.row", comm);
//    print_array(Acsc.val, Acsc.nnz, 0, "Acsc.val", comm);
//    print_array(Acsc.col_scan, Acsc.col_sz + 1, 0, "Acsc.col_scan", comm);

//    std::cout << "\nA: nnz: " << Acsc.nnz << std::endl ;
//    for(index_t j = 0; j < Acsc.col_sz; j++){
//        for(index_t i = Acsc.col_scan[j]; i < Acsc.col_scan[j+1]; i++){
//            std::cout << std::setprecision(4) << Acsc.row[i] << "\t" << j << "\t" << Acsc.val[i] << std::endl;
//        }
//    }
#endif

    // =======================================
    // Convert the local transpose of B to CSC
    // =======================================

    // make a copy of entries of B, then change their order to row-major
    std::vector<cooEntry> B_ent(B->entry);
    std::sort(B_ent.begin(), B_ent.end(), row_major);

    // todo: change to smart pointers
//    auto Brv = std::make_unique<vecEntry[]>(B->nnz_l); // row (actually col to have the transpose) and val
//    auto Bc  = std::make_unique<index_t[]>(B->M+1); // col_idx

//    auto Brv = new vecEntry[B->nnz_l]; // row and val
//    auto Bc  = new index_t[B->M+1];    // col_idx

    CSCMat Bcsc;
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

    for(nnz_t i = 0; i < Bcsc.nnz; i++){
        Bcsc.row[i] = B_ent[i].col;
        Bcsc.val[i] = B_ent[i].val;
        Bc_tmp_p[B_ent[i].row]++;
    }

    for(nnz_t i = 0; i < Bcsc.col_sz; i++){
        Bcsc.col_scan[i+1] += Bcsc.col_scan[i];
    }

    Bcsc.split    = B->split;
    Bcsc.nnz_list = B->nnz_list;

#ifdef __DEBUG1__
//    B->print_entry(0);
//    printf("B: nnz_l: %ld\tnnz_g: %ld\tM: %d\tM_big: %d\n", B->nnz_l, B->nnz_g, B->M, B->Mbig);
//    print_array(Bc, B->M+1, 0, "Bc", comm);
//
//    std::cout << "\nB: nnz: " << B->nnz_l << std::endl ;
//    for(index_t j = 0; j < B->M; j++){
//        for(index_t i = Bcsc.col_scan[j]; i < Bcsc.col_scan[j+1]; i++){
//            std::cout << std::setprecision(4) << Bcsc.row[i] << "\t" << j << "\t" << Bcsc.val[i] << std::endl;
//        }
//    }
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

    // 2 for both send and receive buffer, valbyidx for value, (B->M_max + 1) for col_scan
    // r_cscan_buffer_sz_max is for both row and col_scan which have the same type.
    int   valbyidx              = sizeof(value_t) / sizeof(index_t);
    nnz_t v_buffer_sz_max       = valbyidx * B->nnz_max;
    nnz_t r_cscan_buffer_sz_max = B->nnz_max + B->M_max + 1;
    nnz_t send_size_max         = v_buffer_sz_max + r_cscan_buffer_sz_max;

    try{
        mempool3 = new index_t[2 * send_size_max];
    }catch(std::bad_alloc& ba){
        std::cerr << "bad_alloc caught: " << ba.what() << '\n';
    }

    loc_nnz_max = std::max(A->nnz_max, B->nnz_max);

    try{
        mempool4 = new index_t[2 * loc_nnz_max];
    }catch(std::bad_alloc& ba){
        std::cerr << "bad_alloc caught: " << ba.what() << '\n';
    }

    try{
        mempool5 = new value_t[2 * loc_nnz_max];
    }catch(std::bad_alloc& ba){
        std::cerr << "bad_alloc caught: " << ba.what() << '\n';
    }

//    mempool1 = std::make_unique<value_t[]>(matmat_size_thre2);
//    mempool2 = std::make_unique<index_t[]>(A->Mbig * 4);

#ifdef __DEBUG1__
//    if(rank==0) std::cout << "vecbyint = " << vecbyint << std::endl;

//    if(rank==0){
//        std::cout << "mempool1 size = " << matmat_size_thre2 << std::endl;
//        std::cout << "mempool2 size = " << 2 * A_row_size + 2 * Bcsc.col_sz << std::endl;
//        std::cout << "mempool3 size = " << 2 * send_size_max << std::endl;
//        std::cout << "B->nnz_max = " << B->nnz_max << "\t, B->M_max = " << B->M_max << std::endl;
//    }
#endif

    // =======================================
    // perform the multiplication
    // =======================================

    saena_matrix C(A->comm);
    matmat(Acsc, Bcsc, C, send_size_max, matmat_time);

    // =======================================
    // finalize
    // =======================================

//    mat_send.clear();
//    mat_send.shrink_to_fit();
//    AB_temp.clear();
//    AB_temp.shrink_to_fit();

//    delete []Arv;
//    delete []Ac;
//    delete []Brv;
//    delete []Bc;

    delete []Acsc.row;
    delete []Acsc.val;
    delete []Acsc.col_scan;
    delete []Bcsc.row;
    delete []Bcsc.val;
    delete []Bcsc.col_scan;

//    delete[] mempool1;
//    delete[] mempool2;
    delete []mempool3;
    delete []mempool4;
    delete []mempool5;

    return 0;
}

int saena_object::matmat(CSCMat &Acsc, CSCMat &Bcsc, saena_matrix &C, nnz_t send_size_max, double &matmat_time){

    MPI_Comm comm = C.comm;

    //todo: comment out these 3 lines.
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    case0 = 0, case11 = 0, case12 = 0, case2 = 0, case3 = 0;

    MPI_Barrier(comm);
    double t_AP = MPI_Wtime();

    matmat(Acsc, Bcsc, C, send_size_max);

    t_AP = MPI_Wtime() - t_AP;
    matmat_time += print_time_ave_consecutive(t_AP, comm);

//    if (!rank) printf("\n");
//    if (!rank) printf("case0\ncase11\ncase12\ncase2\ncase3\n\n");
//    print_time_ave(case0,  "case0", comm, true);
//    print_time_ave(case11, "case11", comm, true);
//    print_time_ave(case12, "case12", comm, true);
//    print_time_ave(case2,  "case2", comm, true);
//    print_time_ave(case3,  "case3", comm, true);

//    print_time(case12, "case12", comm);

//    if(rank == 1){
//        printf("\ncase0  = %f\n", case0);
//        printf("case11 = %f\n", case11);
//        printf("case12 = %f\n", case12);
//    }

    return 0;
}

int saena_object::matmat(CSCMat &Acsc, CSCMat &Bcsc, saena_matrix &C, nnz_t send_size_max){

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

    int   valbyidx              = sizeof(value_t) / sizeof(index_t);
//    nnz_t v_buffer_sz_max       = valbyidx * Bcsc.max_nnz;
//    nnz_t r_cscan_buffer_sz_max = Bcsc.max_nnz + Bcsc.max_M + 1;
//    nnz_t send_size_max         = v_buffer_sz_max + r_cscan_buffer_sz_max;

    nnz_t send_nnz  = Bcsc.nnz;
    nnz_t send_size = (valbyidx + 1) * send_nnz + Bcsc.col_sz + 1; // valbyidx for val, 1 for row, Bcsc.col_sz + 1 for c_scan

    index_t Acsc_M = Acsc.split[rank+1] - Acsc.split[rank];

#ifdef __DEBUG1__
    if (verbose_matmat) {
        MPI_Barrier(comm);
        if (rank == verbose_rank) printf("matmat: step 1\n");
        MPI_Barrier(comm);
//        if (rank == verbose_rank) std::cout << "send_nnz: " << send_nnz      << ",\tsend_size: " << send_size
//                                    << ",\tsend_size_max: " << send_size_max << ",\tAcsc_M: "    << Acsc_M << std::endl;
//        MPI_Barrier(comm);
    }
#endif

    CSCMat_mm A(Acsc_M, Acsc.split[rank], Acsc.col_sz, 0, Acsc.nnz);
    A.r = Acsc.row;
    A.v = Acsc.val;
    A.col_scan = Acsc.col_scan;

    std::vector<cooEntry> AB_temp;

    if(nprocs > 1){
//    if(false){
        // set the mat_send data
        // structure of mat_send:
        // 1- row:    type: index_t, size: send_nnz
        // 2- c_scan: type: index_t, size: Bcsc.col_sz + 1
        // 3- val:    type: value_t, size: send_nnz
        auto mat_send       = &mempool3[0];
        auto mat_send_r     = &mat_send[0];
        auto mat_send_cscan = &mat_send[send_nnz];
        auto mat_send_v     = reinterpret_cast<value_t*>(&mat_send[send_nnz + Bcsc.col_sz + 1]);

        memcpy(mat_send_r,     Bcsc.row,      Bcsc.nnz * sizeof(index_t));
        memcpy(mat_send_cscan, Bcsc.col_scan, (Bcsc.col_sz + 1) * sizeof(index_t));
        memcpy(mat_send_v,     Bcsc.val,      Bcsc.nnz * sizeof(value_t));

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
#endif

        int right_neighbor = (rank + 1)%nprocs;
        int left_neighbor  = rank - 1;
        if (left_neighbor < 0){
            left_neighbor += nprocs;
        }

        // set the mat_recv data
        nnz_t recv_nnz;
        nnz_t recv_size;
        index_t mat_recv_M = 0, mat_current_M = 0;
        auto mat_recv = &mempool3[send_size_max];
//        auto mat_recv_rv    = reinterpret_cast<vecEntry*>(&mat_recv[0]);
//        auto mat_recv_cscan = &mat_recv[rv_buffer_sz_max];
//        auto mat_recv_cscan = reinterpret_cast<index_t*>(&mat_recv[rv_buffer_sz_max]);

        auto mat_temp = mat_send;
        int  owner, next_owner;
        auto *requests = new MPI_Request[2];
        auto *statuses = new MPI_Status[2];

        CSCMat_mm S;

        // todo: the last communication means each proc receives a copy of its already owned B, which is redundant,
        //   so the communication should be avoided but the multiplication should be done. consider this:
        //   change to k < rank+nprocs-1. Then, copy fast_mm after the end of the for loop to perform the last multiplication
        for(int k = rank; k < rank+nprocs; ++k){
            // This is overlapped. Both local and remote loops are done here.
            // The first iteration is the local loop. The rest are remote.
            // Send R_tranpose to the left_neighbor processor, receive R_tranpose from the right_neighbor.
            // In the next step: send R_tranpose that was received in the previous step to the left_neighbor processor,
            // receive R_tranpose from the right_neighbor. And so on.
            // --------------------------------------------------------------------

            next_owner = (k+1)%nprocs;
            mat_recv_M = Bcsc.split[next_owner + 1] - Bcsc.split[next_owner];
            recv_nnz   = Bcsc.nnz_list[next_owner];
            recv_size  = (valbyidx + 1) * recv_nnz + mat_recv_M + 1;

#ifdef __DEBUG1__
            if (verbose_matmat) {
                MPI_Barrier(comm);
                if (rank == verbose_rank) printf("matmat: step 3 - in for loop\n");
                MPI_Barrier(comm);
                printf("rank %d: next_owner: %d, recv_nnz: %lu, recv_size: %lu, send_nnz = %lu, send_size: %lu, mat_recv_M: %u\n",
                       rank, next_owner, recv_nnz, recv_size, send_nnz, send_size, mat_recv_M);
                MPI_Barrier(comm);
            }

//            printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
//            mat_recv.resize(recv_size);

            // ===============
            // assert mat_send
            // ===============
            owner         = k%nprocs;
            mat_current_M = Bcsc.split[owner + 1] - Bcsc.split[owner];
//            index_t ofst  = Bcsc.split[owner], col_idx;
            for (nnz_t i = 0; i < mat_current_M; ++i) {
//                col_idx = i + ofst;
                for (nnz_t j = mat_send_cscan[i]; j < mat_send_cscan[i + 1]; j++) {
//                    assert( (col_idx >= Bcsc.split[owner]) && (col_idx < Bcsc.split[owner+1]) ); //this is always true
                    assert( (mat_send_r[j] >= 0) && (mat_send_r[j] < Bcsc.split.back() ) );
                }
            }
#endif

            // communicate data
            MPI_Irecv(&mat_recv[0], recv_size, MPI_UNSIGNED, right_neighbor, right_neighbor, comm, requests);
            MPI_Isend(&mat_send[0], send_size, MPI_UNSIGNED, left_neighbor,  rank,           comm, requests+1);

            // =======================================
            // perform the multiplication
            // =======================================
            if(Acsc.nnz != 0 && send_nnz != 0){

                owner         = k%nprocs;
                mat_current_M = Bcsc.split[owner + 1] - Bcsc.split[owner];
                nnz_t S_nnz = mat_send_cscan[mat_current_M] - mat_send_cscan[0];
                S.set_params(Acsc.col_sz, 0, mat_current_M, Bcsc.split[owner], S_nnz, mat_send_r, mat_send_v, mat_send_cscan);

                fast_mm(A, S, AB_temp, comm);

//                fast_mm(&Acsc.row[0],   &Acsc.val[0],   &Acsc.col_scan[0],
//                        &mat_send_r[0], &mat_send_v[0], &mat_send_cscan[0],
//                        &A_info, &B_info, AB_temp, comm);

            }

            MPI_Waitall(2, requests, statuses);

#ifdef __DEBUG1__
            if (verbose_matmat) {
                MPI_Barrier(comm);
                if (rank == verbose_rank) printf("matmat: step 4 - in for loop\n");
                MPI_Barrier(comm);
            }

            assert(reorder_counter == 0);

//            auto mat_recv_cscan = &mat_recv[rv_buffer_sz_max];
//            MPI_Barrier(comm);
//            if(rank==0){
//                print_array(mat_recv_cscan, mat_recv_M+1, 0, "mat_recv_cscan", comm);
//            }
//            MPI_Barrier(comm);
#endif

            send_size = recv_size;
            send_nnz  = recv_nnz;

            // swap mat_send and mat_recv
//            mat_recv.swap(mat_send);
//            std::swap(mat_send, mat_recv);
            mat_temp = mat_send;
            mat_send = mat_recv;
            mat_recv = mat_temp;

            mat_send_r     = &mat_send[0];
            mat_send_cscan = &mat_send[send_nnz];
            mat_send_v     = reinterpret_cast<value_t*>(&mat_send[send_nnz + mat_recv_M + 1]);

//            mat_send_rv    = reinterpret_cast<vecEntry*>(&mat_send[0]);
//            mat_send_cscan = &mat_send[vecbyint * send_nnz];
//            mat_recv_rv    = reinterpret_cast<vecEntry*>(&mat_recv[0]);
//            mat_recv_cscan = &mat_recv[rv_buffer_sz_max];

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

        delete [] requests;
        delete [] statuses;

//        AB_temp.clear();
//        AB_temp.shrink_to_fit();

#ifdef __DEBUG1__
//        print_vector(AB_temp_no_dup, 0, "AB_temp_no_dup", comm);
//        print_vector(AB_nnz_start, -1, "AB_nnz_start", comm);
//        print_vector(AB_nnz_end,   -1, "AB_nnz_end",   comm);
#endif

//        C.entry.resize(AB_temp_no_dup.size());
//        nnz_t AB_nnz = 0;
//        for(index_t p = 0; p < nprocs; p++){
//            memcpy(&C.entry[AB_nnz], &AB_temp_no_dup[AB_nnz_start[p]], (AB_nnz_end[p] - AB_nnz_start[p]) * sizeof(cooEntry));
//            AB_nnz += AB_nnz_end[p] - AB_nnz_start[p];
//        }

#ifdef __DEBUG1__
        if (verbose_matmat) {
            MPI_Barrier(comm);
            if (rank == verbose_rank) printf("matmat: step 6\n");
            MPI_Barrier(comm);
        }
//        print_vector(AB, 0, "AB", comm);
#endif

    } else { // nprocs == 1 -> serial

#ifdef __DEBUG1__
        if (verbose_matmat) {
            if (rank == verbose_rank) printf("matmat: step 2 serial\n");
        }
#endif

        if(Acsc.nnz != 0 && send_nnz != 0){

//            double t1 = MPI_Wtime();

            CSCMat_mm B(Acsc.col_sz, 0, Bcsc.col_sz, Bcsc.split[rank], Bcsc.nnz);
            B.r = Bcsc.row;
            B.v = Bcsc.val;
            B.col_scan = Bcsc.col_scan;

            fast_mm(A, B, AB_temp, comm);

//            double t2 = MPI_Wtime();
//            printf("\nfast_mm of AB_temp = %f\n", t2-t1);

#ifdef __DEBUG1__
//            print_vector(AB_temp, -1, "AB_temp", comm);
#endif

        }
    }

#ifdef __DEBUG1__
//    print_vector(AB_temp, -1, "AB_temp", comm);
#endif

    // =======================================
    // sort and remove duplicates
    // =======================================

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

#ifdef __DEBUG1__
//    print_vector(C.entry, -1, "C.entry", comm);
//    writeMatrixToFile(C.entry, "matrix_folder/result", comm);

    if (verbose_matmat) {
        MPI_Barrier(comm);
        if (rank == verbose_rank) printf("end of matmat\n\n");
        MPI_Barrier(comm);
    }
#endif

    return 0;
}

int saena_object::matmat(Grid *grid){
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