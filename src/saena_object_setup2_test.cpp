#include "saena_object.h"
#include "saena_matrix.h"
#include "strength_matrix.h"
#include "prolong_matrix.h"
#include "restrict_matrix.h"
#include "grid.h"
#include "aux_functions.h"
#include "parUtils.h"
#include "dollar.hpp"

#include "petsc_functions.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mpi.h>

int saena_object::fast_mm(cooEntry *A, cooEntry *B, std::vector<cooEntry> &C, nnz_t A_nnz, nnz_t B_nnz,
                          index_t A_row_size, index_t A_row_offset, index_t A_col_size, index_t A_col_offset,
                          index_t B_col_size, index_t B_col_offset,
                          index_t *nnzPerColScan_leftStart, index_t *nnzPerColScan_leftEnd,
                          index_t *nnzPerColScan_rightStart, index_t *nnzPerColScan_rightEnd,
                          value_t *mempool1, index_t *mempool2, MPI_Comm comm){

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

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    int verbose_rank = 0;
#ifdef _DEBUG_
    if(rank==verbose_rank && verbose_matmat) printf("\nfast_mm: start \n");
#endif

    if(A_nnz == 0 || B_nnz == 0){
#ifdef _DEBUG_
        if(rank==verbose_rank && verbose_matmat) printf("\nskip: A_nnz == 0 || B_nnz == 0\n\n");
#endif
        return 0;
    }

#ifdef _DEBUG_
    //    print_vector(A, -1, "A", comm);
//    print_vector(B, -1, "B", comm);
//    MPI_Barrier(comm); printf("rank %d: A: %ux%u, B: %ux%u \n\n", rank, A_row_size, A_col_size, A_col_size, B_col_size); MPI_Barrier(comm);
//    MPI_Barrier(comm); printf("rank %d: A_row_size = %u, A_row_offset = %u, A_col_size = %u, A_col_offset = %u, B_row_offset = %u, B_col_size = %u, B_col_offset = %u \n\n",
//            rank, A_row_size, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size, B_col_offset);

//    MPI_Barrier(comm);
    if(rank==verbose_rank){

        if(verbose_matmat_A){
            std::cout << "\nA: nnz = " << A_nnz << std::endl;
            std::cout << "A_row_size = "     << A_row_size   << ", A_col_size = "   << A_col_size
                      << ", A_row_offset = " << A_row_offset << ", A_col_offset = " << A_col_offset << std::endl;

            // print entries of A:
            std::cout << "\nA: nnz = " << A_nnz << std::endl;
            for(nnz_t i = 0; i < A_col_size; i++){
                for(nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftEnd[i]; j++) {
                    std::cout << j << "\t" << A[j] << std::endl;
                }
            }
        }

        if(verbose_matmat_B) {
            std::cout << "\nB: nnz = " << B_nnz << std::endl;
            std::cout << "B_row_size = " << A_col_size << ", B_col_size = " << B_col_size
                      << ", B_row_offset = " << B_row_offset << ", B_col_offset = " << B_col_offset << std::endl;

            // print entries of B:
            std::cout << "\nB: nnz = " << B_nnz << std::endl;
            for (nnz_t i = 0; i < B_col_size; i++) {
                for (nnz_t j = nnzPerColScan_rightStart[i]; j < nnzPerColScan_rightEnd[i]; j++) {
                    std::cout << j << "\t" << B[j] << std::endl;
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
//    MPI_Barrier(comm);
#endif

//    index_t size_min = std::min(std::min(A_row_size, A_col_size), B_col_size);

    if( A_row_size * B_col_size < matmat_size_thre ){

        fast_mm_part1(&A[0], &B[0], C, A_nnz, B_nnz,
                      A_row_size, A_row_offset, A_col_size, A_col_offset,
                      B_col_size, B_col_offset,
                      nnzPerColScan_leftStart,  nnzPerColScan_leftEnd, // A1
                      nnzPerColScan_rightStart, nnzPerColScan_rightEnd, mempool1, mempool2, comm); // B1

    } else if( (A_row_size) <= A_col_size ) {

        fast_mm_part2(&A[0], &B[0], C, A_nnz, B_nnz,
                      A_row_size, A_row_offset, A_col_size, A_col_offset,
                      B_col_size, B_col_offset,
                      nnzPerColScan_leftStart,  nnzPerColScan_leftEnd, // A1
                      nnzPerColScan_rightStart, nnzPerColScan_rightEnd, mempool1, mempool2, comm); // B1

    } else { // A_row_size > A_col_size

        fast_mm_part3(&A[0], &B[0], C, A_nnz, B_nnz,
                      A_row_size, A_row_offset, A_col_size, A_col_offset,
                      B_col_size, B_col_offset,
                      nnzPerColScan_leftStart,  nnzPerColScan_leftEnd, // A1
                      nnzPerColScan_rightStart, nnzPerColScan_rightEnd, mempool1, mempool2, comm); // B1

    }

#ifdef _DEBUG_
    if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
#endif

    return 0;
}

// C_temp is of size A_row_size * B_col_size.
//int saena_object::fast_mm_part1
/*
int saena_object::fast_mm_part1(cooEntry *A, cooEntry *B, std::vector<cooEntry> &C, nnz_t A_nnz, nnz_t B_nnz,
                                index_t A_row_size, index_t A_row_offset, index_t A_col_size, index_t A_col_offset,
                                index_t B_col_size, index_t B_col_offset,
                                index_t *nnzPerColScan_leftStart, index_t *nnzPerColScan_leftEnd,
                                index_t *nnzPerColScan_rightStart, index_t *nnzPerColScan_rightEnd,
                                value_t *mempool, MPI_Comm comm) {

#ifdef _DEBUG_
    if(rank==verbose_rank && (verbose_matmat || verbose_matmat_recursive)){printf("fast_mm: case 1: start \n");}
#endif

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    index_t B_row_offset = A_col_offset;

    // initialize
    value_t *C_temp = mempool;
    std::fill(&C_temp[0], &C_temp[A_row_size * B_col_size], 0);

#ifdef _DEBUG_
    if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 1: step 1 \n");}
#endif

    index_t C_index=0;
    for(nnz_t j = 0; j < B_col_size; j++) { // columns of B
        for (nnz_t k = nnzPerColScan_rightStart[j]; k < nnzPerColScan_rightEnd[j]; k++) { // nonzeros in column j of B
            for (nnz_t i = nnzPerColScan_leftStart[B[k].row - B_row_offset];
                 i < nnzPerColScan_leftEnd[B[k].row - B_row_offset]; i++) { // nonzeros in column B[k].row of A

                C_index = (A[i].row - A_row_offset) + A_row_size * (B[k].col - B_col_offset);
                C_temp[C_index] += B[k].val * A[i].val;

#ifdef _DEBUG_
                //                    if (rank == 0) std::cout << "A: " << A[i] << "\tB: " << B[k] << "\tC_index: " << C_index
//                                   << "\tA_row_offset = " << A_row_offset
//                                   << "\tB_col_offset = " << B_col_offset << std::endl;

//                    if(rank==1 && A[i].row == 0 && B[j].col == 0) std::cout << "A: " << A[i] << "\tB: " << B[j]
//                         << "\tC: " << C_temp[(A[i].row-A_row_offset) + A_row_size * (B[j].col-B_col_offset)]
//                         << "\tA*B: " << B[j].val * A[i].val << std::endl;
#endif
            }
        }
    }

#ifdef _DEBUG_
    //        print_vector(C_temp, -1, "C_temp", comm);
        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 1: step 2 \n");}
#endif

    // add the new elements to C
    // add the entries in column-major order
    for(index_t j = 0; j < B_col_size; j++) {
        for(index_t i = 0; i < A_row_size; i++) {
//                if(rank==0) std::cout << i + A_row_size*j << "\t" << C_temp[i + A_row_size*j] << std::endl;
            if (C_temp[i + A_row_size*j] != 0) {
                C.emplace_back( cooEntry( i+A_row_offset, j+B_col_offset, C_temp[i + A_row_size*j] ) );
            }
        }
    }

#ifdef _DEBUG_
    //        print_vector(C, -1, "C", comm);
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 1: end \n");
#endif

    return 0;
}
*/


int saena_object::fast_mm_part1(cooEntry *A, cooEntry *B, std::vector<cooEntry> &C, nnz_t A_nnz, nnz_t B_nnz,
                                index_t A_row_size, index_t A_row_offset, index_t A_col_size, index_t A_col_offset,
                                index_t B_col_size, index_t B_col_offset,
                                index_t *nnzPerColScan_leftStart, index_t *nnzPerColScan_leftEnd,
                                index_t *nnzPerColScan_rightStart, index_t *nnzPerColScan_rightEnd,
                                value_t *mempool1, index_t *mempool2, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    index_t B_row_offset = A_col_offset;

#ifdef _DEBUG_
    if(rank==verbose_rank && (verbose_matmat || verbose_matmat_recursive)){printf("fast_mm: case 1: start \n");}
#endif

    std::vector<index_t> nnzPerRow_left(A_row_size, 0);
//    std::fill(&nnzPerRow_left[0], &nnzPerRow_left[A_row_size], 0);
    index_t *nnzPerRow_left_p = &nnzPerRow_left[0] - A_row_offset;
    for(nnz_t i = 0; i < A_col_size; i++){
        for(nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftEnd[i]; j++) {
            nnzPerRow_left_p[A[j].row]++;
        }
    }

//    print_vector(nnzPerRow_left, -1, "nnzPerRow_left", comm);

    index_t *new_row_idx = &nnzPerRow_left[0];
    index_t *new_row_idx_p = &new_row_idx[0] - A_row_offset;
    std::vector<index_t> orig_row_idx;
    index_t A_nnz_row_sz = 0;
    for(nnz_t i = 0; i < A_row_size; i++){
        if(new_row_idx[i]){
            new_row_idx[i] = A_nnz_row_sz;
            A_nnz_row_sz++;
            orig_row_idx.emplace_back(i + A_row_offset);
        }
    }

//    print_vector(new_row_idx, -1, "new_row_idx", comm);

    std::vector<index_t> new_col_idx(B_col_size);
    index_t *orig_col_idx_p = &new_col_idx[0] - B_col_offset;
    std::vector<index_t> orig_col_idx;
    index_t B_nnz_col_sz = 0;
    for(nnz_t i = 0; i < B_col_size; i++){
        if(nnzPerColScan_rightEnd[i] != nnzPerColScan_rightStart[i]){
            new_col_idx[i] = B_nnz_col_sz;
            B_nnz_col_sz++;
            orig_col_idx.emplace_back(i + B_col_offset);
        }
    }

//    printf("A_row_size = %u, \tA_nnz_row_sz = %u, \tB_col_size = %u, \tB_nnz_col_sz = %u \n",
//            A_row_size, A_nnz_row_sz, B_col_size, B_nnz_col_sz);

    // initialize
    value_t *C_temp = &mempool1[0];
//    std::fill(&C_temp[0], &C_temp[A_nnz_row_sz * B_col_size], 0);
    std::fill(&C_temp[0], &C_temp[A_nnz_row_sz * B_nnz_col_sz], 0);

#ifdef _DEBUG_
    if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 1: step 1 \n");}
#endif

    index_t *nnzPerColScan_leftStart_p = &nnzPerColScan_leftStart[0] - B_row_offset;
    index_t *nnzPerColScan_leftEnd_p   = &nnzPerColScan_leftEnd[0] - B_row_offset;
//    value_t *C_temp_p = C_temp - (A_nnz_row_sz * B_col_offset);

    for(nnz_t j = 0; j < B_col_size; j++) { // columns of B
        for (nnz_t k = nnzPerColScan_rightStart[j]; k < nnzPerColScan_rightEnd[j]; k++) { // nonzeros in column j of B
            for (nnz_t i = nnzPerColScan_leftStart_p[B[k].row];
                 i < nnzPerColScan_leftEnd_p[B[k].row]; i++) { // nonzeros in column B[k].row of A

//                if(rank==0) std::cout << new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B[k].col << "\t"
//                << new_row_idx[A[i].row - A_row_offset] << "\t" << B[k].col << "\t" << C_temp[new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B[k].col] << std::endl;

//                C_temp_p[new_row_idx_p[A[i].row] + A_nnz_row_sz * B[k].col] += B[k].val * A[i].val;
                C_temp[ new_row_idx_p[A[i].row] + A_nnz_row_sz * orig_col_idx_p[B[k].col] ] += B[k].val * A[i].val;

#ifdef _DEBUG_
                if (rank == 0) std::cout << "A: " << A[i] << "\tB: " << B[k] << "\tC_index: " << C_index
                                         << "\tA_row_offset = " << A_row_offset
                                         << "\tB_col_offset = " << B_col_offset << std::endl;

//                    if(rank==1 && A[i].row == 0 && B[j].col == 0) std::cout << "A: " << A[i] << "\tB: " << B[j]
//                         << "\tC: " << C_temp[(A[i].row-A_row_offset) + A_row_size * (B[j].col-B_col_offset)]
//                         << "\tA*B: " << B[j].val * A[i].val << std::endl;
#endif
            }
        }
    }

#ifdef _DEBUG_
    //        print_vector(C_temp, -1, "C_temp", comm);
    if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 1: step 2 \n");}
#endif

    for(index_t j = 0; j < B_nnz_col_sz; j++) {
        for(index_t i = 0; i < A_nnz_row_sz; i++) {
//            if(rank==0) std::cout << i + A_nnz_row_sz*j << "\t" << orig_row_idx[i] << "\t" << j << "\t" << C_temp[i + A_nnz_row_sz*j] << std::endl;
            if (C_temp[i + A_nnz_row_sz*j] != 0) {
                C.emplace_back( orig_row_idx[i] , orig_col_idx[j], C_temp[i + A_nnz_row_sz * j] );
            }
        }
    }

#ifdef _DEBUG_
    //       print_vector(C, -1, "C", comm);
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 1: end \n");
#endif

    return 0;
}


int saena_object::fast_mm_part2(cooEntry *A, cooEntry *B, std::vector<cooEntry> &C, nnz_t A_nnz, nnz_t B_nnz,
                                index_t A_row_size, index_t A_row_offset, index_t A_col_size, index_t A_col_offset,
                                index_t B_col_size, index_t B_col_offset,
                                index_t *nnzPerColScan_leftStart, index_t *nnzPerColScan_leftEnd,
                                index_t *nnzPerColScan_rightStart, index_t *nnzPerColScan_rightEnd,
                                value_t *mempool1, index_t *mempool2, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    index_t B_row_offset = A_col_offset;
    index_t A_col_size_half = A_col_size/2;
    index_t B_row_size_half = A_col_size_half;

#ifdef _DEBUG_
    if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: start \n");}
#endif

    // prepare splits of matrix A by column
    nnz_t A1_nnz = 0, A2_nnz;

//        for(nnz_t i = 0; i < A_col_size; i++){
//            for(nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftEnd[i]; j++) {
//                if(A[j].col - A_col_offset < A_col_size_half) {
//                    A1_nnz++;
//                }
//            }
//        }

    for(nnz_t i = 0; i < A_col_size_half; i++){
        A1_nnz += nnzPerColScan_leftEnd[i] - nnzPerColScan_leftStart[i];
    }

    A2_nnz = A_nnz - A1_nnz;

    // prepare splits of matrix B by row
    nnz_t B1_nnz = 0, B2_nnz;

    std::vector<index_t> nnzPerCol_middle(B_col_size, 0);
    // to avoid subtraction in the following for loop " - B_col_offset"
    index_t *nnzPerCol_middle_p = &nnzPerCol_middle[0] - B_col_offset;

    index_t B_row_threshold = B_row_size_half + B_row_offset;
    for(nnz_t i = 0; i < B_col_size; i++){
        for(nnz_t j = nnzPerColScan_rightStart[i]; j < nnzPerColScan_rightEnd[i]; j++) {
            if(B[j].row < B_row_threshold){ // B[j].row - B_row_offset < B_row_size_half
//                    nnzPerCol_middle[B[j].col - B_col_offset]++;
                nnzPerCol_middle_p[B[j].col]++;
                B1_nnz++;
            }
        }
    }

    B2_nnz = B_nnz - B1_nnz;

//        print_vector(nnzPerCol_middle, -1, "nnzPerCol_middle", comm);

    std::vector<index_t> nnzPerColScan_middle(B_col_size + 1);
//        nnzPerColScan_middle[0] = 0;
//        for(nnz_t i = 0; i < B_col_size; i++){
//            nnzPerColScan_middle[i+1] = nnzPerColScan_middle[i] + nnzPerCol_middle[i];
//        }

    for(nnz_t i = 0; i < B_col_size; i++){
        nnzPerColScan_middle[i] = nnzPerColScan_rightStart[i] + nnzPerCol_middle[i];
    }

    nnzPerCol_middle.clear();
    nnzPerCol_middle.shrink_to_fit();

#ifdef _DEBUG_
    //        print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);
//        if(rank==0) printf("rank %d: A_nnz = %lu, A1_nnz = %lu, A2_nnz = %lu, B_nnz = %lu, B1_nnz = %lu, B2_nnz = %lu \n",
//                rank, A_nnz, A1_nnz, A2_nnz, B_nnz, B1_nnz, B2_nnz);
        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 1 \n");}
#endif

//        for(nnz_t i = 0; i < B_col_size; i++){
//            nnzPerColScan_middle[i] = nnzPerColScan_middle[i+1] + nnzPerColScan_rightStart[i] - nnzPerColScan_middle[i];
//            if(rank==0) printf("nnzPerColScan_middle[%lu] = %u, \tnnzPerColScan_middle[%lu] = %u, \tnnzPerColScan_rightStart = %u \n",
//                    i, nnzPerColScan_middle[i], i+1, nnzPerColScan_middle[i+1], nnzPerColScan_rightStart[i]);
//        }

#ifdef _DEBUG_
    //        print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);
        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 2 \n");}
#endif

    // A1: start: nnzPerColScan_leftStart,                  end: nnzPerColScan_leftEnd
    // A2: start: nnzPerColScan_leftStart[A_col_size_half], end: nnzPerColScan_leftEnd[A_col_size_half]
    // B1: start: nnzPerColScan_rightStart,                 end: nnzPerColScan_middle
    // B2: start: nnzPerColScan_middle,                     end: nnzPerColScan_rightEnd

#ifdef _DEBUG_
    //        MPI_Barrier(comm);
        if(rank==verbose_rank){

            if(verbose_matmat_A) {
                std::cout << "\nranges of A:" << std::endl;
                for (nnz_t i = 0; i < A_col_size; i++) {
                    std::cout << i << "\t" << nnzPerColScan_leftStart[i] << "\t" << nnzPerColScan_leftEnd[i]
                              << std::endl;
                }

                std::cout << "\nranges of A1:" << std::endl;
                for (nnz_t i = 0; i < A_col_size / 2; i++) {
                    std::cout << i << "\t" << nnzPerColScan_leftStart[i] << "\t" << nnzPerColScan_leftStart[i + 1]
                              << std::endl;
                }

                std::cout << "\nranges of A2:" << std::endl;
                for (nnz_t i = 0; i < A_col_size - A_col_size / 2; i++) {
                    std::cout << i << "\t" << nnzPerColScan_leftStart[A_col_size / 2 + i]
                              << "\t" << nnzPerColScan_leftStart[A_col_size / 2 + i + 1] << std::endl;
                }

                // print entries of A1:
                std::cout << "\nA1: nnz = " << A1_nnz << std::endl;
                for (nnz_t i = 0; i < A_col_size / 2; i++) {
                    for (nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftStart[i + 1]; j++) {
                        std::cout << j << "\t" << A[j] << std::endl;
                    }
                }

                // print entries of A2:
                std::cout << "\nA2: nnz = " << A2_nnz << std::endl;
                for (nnz_t i = 0; i < A_col_size - A_col_size / 2; i++) {
                    for (nnz_t j = nnzPerColScan_leftStart[A_col_size / 2 + i];
                         j < nnzPerColScan_leftStart[A_col_size / 2 + i + 1]; j++) {
                        std::cout << j << "\t" << A[j] << std::endl;
                    }
                }
            }

            if(verbose_matmat_B) {
                std::cout << "\nranges of B, B1, B2::" << std::endl;
                for (nnz_t i = 0; i < B_col_size; i++) {
                    std::cout << i << "\t" << nnzPerColScan_rightStart[i] << "\t" << nnzPerColScan_rightEnd[i]
                              << "\t" << nnzPerColScan_rightStart[i] << "\t" << nnzPerColScan_middle[i]
                              << "\t" << nnzPerColScan_middle[i] << "\t" << nnzPerColScan_rightEnd[i] << std::endl;
                }

                // print entries of B1:
                std::cout << "\nB1: nnz = " << B1_nnz << std::endl;
                for (nnz_t i = 0; i < B_col_size; i++) {
                    for (nnz_t j = nnzPerColScan_rightStart[i]; j < nnzPerColScan_middle[i]; j++) {
                        std::cout << j << "\t" << B[j] << std::endl;
                    }
                }

                // print entries of B2:
                std::cout << "\nB2: nnz = " << B2_nnz << std::endl;
                for (nnz_t i = 0; i < B_col_size; i++) {
                    for (nnz_t j = nnzPerColScan_middle[i]; j < nnzPerColScan_rightEnd[i]; j++) {
                        std::cout << j << "\t" << B[j] << std::endl;
                    }
                }
            }
        }
//        print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);
        MPI_Barrier(comm);
#endif

    std::vector<cooEntry> C1, C2;

    // C1 = A1 * B1
#ifdef _DEBUG_
    if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 1 \n");
#endif

    fast_mm(&A[0], &B[0], C1, A1_nnz, B1_nnz,
            A_row_size, A_row_offset, A_col_size_half, A_col_offset,
            B_col_size, B_col_offset,
            nnzPerColScan_leftStart,  nnzPerColScan_leftEnd, // A1
            nnzPerColScan_rightStart, &nnzPerColScan_middle[0], mempool1, mempool2, comm); // B1

    // C2 = A2 * B2
#ifdef _DEBUG_
    if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 2 \n");
#endif

    fast_mm(&A[0], &B[0], C2, A2_nnz, B2_nnz,
            A_row_size, A_row_offset, A_col_size-A_col_size_half, A_col_offset+A_col_size_half,
            B_col_size, B_col_offset,
            &nnzPerColScan_leftStart[A_col_size_half], &nnzPerColScan_leftEnd[A_col_size_half], // A2
            &nnzPerColScan_middle[0], nnzPerColScan_rightEnd, mempool1, mempool2, comm); // B2

#ifdef _DEBUG_
    //        print_vector(C1, -1, "C1", comm);
//        print_vector(C2, -1, "C2", comm);

        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 3 \n");}
//        if(rank==0 && verbose_matmat) printf("C1.size() = %lu, C2.size() = %lu \n", C1.size(), C2.size());
#endif

    // take care of the special cases when either C1 or C2 is empty.
//        nnz_t i = 0;
//        if(C1.empty()){
//            while(i < C2.size()){
//                C.emplace_back(C2[i]);
//                i++;
//            }
//
//#ifdef _DEBUG_
//            if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
//#endif
//            return 0;
//        }
//
//        if(C2.empty()) {
//            while (i < C1.size()) {
//                C.emplace_back(C1[i]);
//                i++;
//            }
//#ifdef _DEBUG_
//            if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
//#endif
//            return 0;
//        }

    if(C1.empty()){
        nnz_t C_init_size = C.size();
        C.resize(C.size() + C2.size());
        memcpy(&C[C_init_size], &C2[0], C2.size() * sizeof(cooEntry));

#ifdef _DEBUG_
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
#endif
        return 0;
    }

    if(C2.empty()){
        nnz_t C_init_size = C.size();
        C.resize(C.size() + C1.size());
        memcpy(&C[C_init_size], &C1[0], C1.size() * sizeof(cooEntry));

#ifdef _DEBUG_
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
#endif
        return 0;
    }

#ifdef _DEBUG_
    if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 4 \n");}
#endif

    // merge C1 and C2
    nnz_t i = 0;
    nnz_t j = 0;
    while(i < C1.size() && j < C2.size()){
        if(C1[i] < C2[j]){
            C.emplace_back(C1[i]);
            i++;
        }else if(C1[i] == C2[j]){ // there is no duplicate in either C1 or C2. So there may be at most one duplicate when we add them.
            C.emplace_back(C1[i] + C2[j]);
            i++; j++;
        }else{ // C1[i] > C2[j]
            C.emplace_back(C2[j]);
            j++;
        }

        // when end of C1 (or C2) is reached, just add C2 (C1).
        if(i == C1.size()){
            while(j < C2.size()){
                C.emplace_back(C2[j]);
                j++;
            }

#ifdef _DEBUG_
            if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
#endif
            return 0;
        }else if(j == C2.size()) {
            while (i < C1.size()) {
                C.emplace_back(C1[i]);
                i++;
            }

#ifdef _DEBUG_
            if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
#endif
            return 0;
        }
    }

#ifdef _DEBUG_
    if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 2: end \n");
#endif

    return 0;
}

int saena_object::fast_mm_part3(cooEntry *A, cooEntry *B, std::vector<cooEntry> &C, nnz_t A_nnz, nnz_t B_nnz,
                                index_t A_row_size, index_t A_row_offset, index_t A_col_size, index_t A_col_offset,
                                index_t B_col_size, index_t B_col_offset,
                                index_t *nnzPerColScan_leftStart, index_t *nnzPerColScan_leftEnd,
                                index_t *nnzPerColScan_rightStart, index_t *nnzPerColScan_rightEnd,
                                value_t *mempool1, index_t *mempool2, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    index_t A_col_size_half = A_col_size/2;
    index_t B_col_size_half = B_col_size/2;


#ifdef _DEBUG_
    if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: start \n");
#endif

    index_t A_row_size_half = A_row_size/2;

    // prepare splits of matrix B by column
    nnz_t B1_nnz = 0, B2_nnz;

//        for(nnz_t i = 0; i < B_col_size; i++){
//            for(nnz_t j = nnzPerColScan_rightStart[i]; j < nnzPerColScan_rightEnd[i]; j++) {
//                if(B[j].col - B_col_offset < B_col_size_half){
//                    B1_nnz++;
//                }
//            }
//        }

    for(nnz_t i = 0; i < B_col_size_half; i++){
        B1_nnz += nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i];
    }

    B2_nnz = B_nnz - B1_nnz;

#ifdef _DEBUG_
    if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 1 \n");
#endif
    // prepare splits of matrix A by row
    nnz_t A1_nnz = 0, A2_nnz;
    index_t A_row_threshold = A_row_size_half + A_row_offset;

    std::vector<index_t> nnzPerCol_middle(A_col_size, 0);
    // to avoid subtraction in the following for loop " - B_col_offset"
    index_t *nnzPerCol_middle_p = &nnzPerCol_middle[0] - A_col_offset;

    for(nnz_t i = 0; i < A_col_size; i++){
        for(nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftEnd[i]; j++) {
            if(A[j].row < A_row_threshold){ // A[j].row - A_row_offset < A_row_size_half
                nnzPerCol_middle_p[A[j].col]++;
                A1_nnz++;
            }
        }
    }

    A2_nnz = A_nnz - A1_nnz;

    std::vector<index_t> nnzPerColScan_middle(A_col_size + 1);
//    nnzPerColScan_middle[0] = 0;
//    for(nnz_t i = 0; i < A_col_size; i++){
//        nnzPerColScan_middle[i+1] = nnzPerColScan_middle[i] + nnzPerCol_middle[i];
//    }

    for(nnz_t i = 0; i < A_col_size; i++){
        nnzPerColScan_middle[i] = nnzPerColScan_leftStart[i] + nnzPerCol_middle[i];
    }

    nnzPerCol_middle.clear();
    nnzPerCol_middle.shrink_to_fit();

#ifdef _DEBUG_
    if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 2 \n");
#endif

//    for(nnz_t i = 0; i < A_col_size; i++){
//        nnzPerColScan_middle[i] = nnzPerColScan_middle[i+1] + nnzPerColScan_leftStart[i] - nnzPerColScan_middle[i];
//            if(rank==0) printf("nnzPerColScan_middle[%lu] = %u \n", i, nnzPerColScan_middle[i]);
//    }

#ifdef _DEBUG_
    if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 3 \n");
#endif

    // A1: start: nnzPerColScan_leftStart,                   end: nnzPerColScan_middle
    // A2: start: nnzPerColScan_middle,                      end: nnzPerColScan_leftEnd
    // B1: start: nnzPerColScan_rightStart,                  end: nnzPerColScan_rightEnd
    // B2: start: nnzPerColScan_rightStart[B_col_size_half], end: nnzPerColScan_rightEnd[B_col_size_half]

#ifdef _DEBUG_
    //        MPI_Barrier(comm);
        if(rank==verbose_rank){

            if(verbose_matmat_A) {
                // print entries of A1:
                std::cout << "\nA1: nnz = " << A1_nnz << std::endl;
                for (nnz_t i = 0; i < A_col_size; i++) {
                    for (nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_middle[i]; j++) {
                        std::cout << j << "\t" << A[j] << std::endl;
                    }
                }

                // print entries of A2:
                std::cout << "\nA2: nnz = " << A2_nnz << std::endl;
                for (nnz_t i = 0; i < A_col_size; i++) {
                    for (nnz_t j = nnzPerColScan_middle[i]; j < nnzPerColScan_leftEnd[i]; j++) {
                        std::cout << j << "\t" << A[j] << std::endl;
                    }
                }
            }

            if(verbose_matmat_B) {
                std::cout << "\nranges of B:" << std::endl;
                for (nnz_t i = 0; i < B_col_size; i++) {
                    std::cout << i << "\t" << nnzPerColScan_rightStart[i] << "\t" << nnzPerColScan_rightEnd[i]
                              << std::endl;
                }

                std::cout << "\nranges of B1:" << std::endl;
                for (nnz_t i = 0; i < B_col_size / 2; i++) {
                    std::cout << i << "\t" << nnzPerColScan_rightStart[i] << "\t" << nnzPerColScan_rightEnd[i]
                              << std::endl;
                }

                std::cout << "\nranges of B2:" << std::endl;
                for (nnz_t i = 0; i < B_col_size - B_col_size / 2; i++) {
                    std::cout << i << "\t" << nnzPerColScan_rightStart[B_col_size / 2 + i]
                              << "\t" << nnzPerColScan_rightEnd[B_col_size / 2 + i] << std::endl;
                }

                // print entries of B1:
                std::cout << "\nB1: nnz = " << B1_nnz << std::endl;
                for (nnz_t i = 0; i < B_col_size / 2; i++) {
                    for (nnz_t j = nnzPerColScan_rightStart[i]; j < nnzPerColScan_rightEnd[i]; j++) {
                        std::cout << j << "\t" << B[j] << std::endl;
                    }
                }

                // print entries of B2:
                std::cout << "\nB2: nnz = " << B2_nnz << std::endl;
                for (nnz_t i = 0; i < B_col_size - B_col_size / 2; i++) {
                    for (nnz_t j = nnzPerColScan_rightStart[B_col_size / 2 + i];
                         j < nnzPerColScan_rightEnd[B_col_size / 2 + i]; j++) {
                        std::cout << j << "\t" << B[j] << std::endl;
                    }
                }
            }
        }
//        MPI_Barrier(comm);
#endif

    std::vector<cooEntry> C_temp;

    // C1 = A1 * B1
#ifdef _DEBUG_
    if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 1 \n");
#endif
    fast_mm(&A[0], &B[0], C_temp, A1_nnz, B1_nnz,
            A_row_size_half, A_row_offset, A_col_size, A_col_offset,
            B_col_size_half, B_col_offset,
            nnzPerColScan_leftStart,  &nnzPerColScan_middle[0], // A1
            nnzPerColScan_rightStart, nnzPerColScan_rightEnd, mempool1, mempool2, comm); // B1

    // C2 = A2 * B1:
#ifdef _DEBUG_
    if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 2 \n");
#endif
    fast_mm(&A[0], &B[0], C_temp, A2_nnz, B1_nnz,
            A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset,
            B_col_size_half, B_col_offset,
            &nnzPerColScan_middle[0], nnzPerColScan_leftEnd, // A2
            nnzPerColScan_rightStart, nnzPerColScan_rightEnd, mempool1, mempool2, comm); // B1

    // C3 = A1 * B2:
#ifdef _DEBUG_
    if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 3 \n");
#endif
    fast_mm(&A[0], &B[0], C_temp, A1_nnz, B2_nnz,
            A_row_size_half, A_row_offset, A_col_size, A_col_offset,
            B_col_size-B_col_size_half, B_col_offset+B_col_size_half,
            nnzPerColScan_leftStart,  &nnzPerColScan_middle[0], // A1
            &nnzPerColScan_rightStart[B_col_size_half], &nnzPerColScan_rightEnd[B_col_size_half], mempool1, mempool2, comm); // B2

    // C4 = A2 * B2
#ifdef _DEBUG_
    if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 4 \n");
#endif
    fast_mm(&A[0], &B[0], C_temp, A2_nnz, B2_nnz,
            A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset,
            B_col_size-B_col_size_half, B_col_offset+B_col_size_half,
            &nnzPerColScan_middle[0], nnzPerColScan_leftEnd, // A2
            &nnzPerColScan_rightStart[B_col_size_half], &nnzPerColScan_rightEnd[B_col_size_half], mempool1, mempool2, comm); // B2

    // C1 = A1 * B1:
//        fast_mm(A1, B1, C_temp, A_row_size_half, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size_half, B_col_offset, comm);
    // C2 = A2 * B1:
//        fast_mm(A2, B1, C_temp, A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset, B_row_offset, B_col_size_half, B_col_offset, comm);
    // C3 = A1 * B2:
//        fast_mm(A1, B2, C_temp, A_row_size_half, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size-B_col_size_half, B_col_offset+B_col_size_half, comm);
    // C4 = A2 * B2
//        fast_mm(A2, B2, C_temp, A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset, B_row_offset, B_col_size-B_col_size_half, B_col_offset+B_col_size_half, comm);

//        if(rank==0 && verbose_matmat) printf("fast_mm: case 3: step 4 \n");

    std::sort(C_temp.begin(), C_temp.end());

    nnz_t C_temp_size_minus1 = C_temp.size()-1;
    // remove duplicates.
    for(nnz_t i = 0; i < C_temp.size(); i++){
        C.emplace_back(C_temp[i]);
        while(i < C_temp_size_minus1 && C_temp[i] == C_temp[i+1]){ // values of entries with the same row and col should be added.
            C.back().val += C_temp[++i].val;
        }
    }

#ifdef _DEBUG_
    if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: end \n");
#endif

    return 0;
}