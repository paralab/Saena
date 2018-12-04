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


// this version splits the matrices to have half nnz on each side.
int saena_object::fast_mm_nnz(cooEntry *A, cooEntry *B, std::vector<cooEntry> &C, nnz_t A_nnz, nnz_t B_nnz,
                          index_t A_row_size, index_t A_row_offset, index_t A_col_size, index_t A_col_offset,
                          index_t B_col_size, index_t B_col_offset,
                          index_t *nnzPerColScan_leftStart, index_t *nnzPerColScan_leftEnd,
                          index_t *nnzPerColScan_rightStart, index_t *nnzPerColScan_rightEnd,
                          value_t *mempool, MPI_Comm comm){

    // This function has three parts:
    // 1- A is horizontal (row > col)
    // 2- A is vertical
    // 3- do multiplication when blocks of A and B are small enough. Put them in a sparse C, where C_ij = A_i * B_j
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

    index_t B_row_offset = A_col_offset;

    int verbose_rank = 0;
    if(rank==verbose_rank && verbose_matmat) printf("\nfast_mm: start \n");

    if(A_nnz == 0 || B_nnz == 0){
        if(rank==verbose_rank && verbose_matmat) printf("\nskip: A_nnz == 0 || B_nnz == 0\n\n");
        return 0;
    }

//    print_vector(A, -1, "A", comm);
//    print_vector(B, -1, "B", comm);
//    if(rank==0) printf("rank %d: A: %ux%u, B: %ux%u, Ar*Bc: %u \n\n", rank, A_row_size, A_col_size, A_col_size, B_col_size, A_row_size*B_col_size);
//    printf("rank %d: A_row_size = %u, A_row_offset = %u, A_col_size = %u, A_col_offset = %u, B_col_size = %u, B_col_offset = %u \n\n",
//            rank, A_row_size, A_row_offset, A_col_size, A_col_offset, B_col_size, B_col_offset);

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

//    index_t size_min = std::min(std::min(A_row_size, A_col_size), B_col_size);
//    if(rank==verbose_rank) printf("A_row_size = %u, A_col_size = %u, B_col_size = %u, size_min = %u, min_size_threshold = %u \n",
//                                  A_row_size, A_col_size, B_col_size, size_min, min_size_threshold);

    if( A_row_size * B_col_size < matmat_size_thre ){

        if(rank==verbose_rank && (verbose_matmat || verbose_matmat_recursive)){printf("fast_mm: case 1: start \n");}

        // initialize
//        std::vector<cooEntry> C_temp(A_row_size * B_col_size); // 1D array is better than 2D for many reasons.
//        for(nnz_t i = 0; i < A_row_size * B_col_size; i++){
//            C_temp[i] = cooEntry(0, 0, 0);
//        }

        // initialize
        value_t *C_temp = mempool;
        std::fill(&C_temp[0], &C_temp[A_row_size * B_col_size], 0);

        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 1: step 1 \n");}

        index_t C_index;
        for(nnz_t j = 0; j < B_col_size; j++) { // columns of B
            for (nnz_t k = nnzPerColScan_rightStart[j]; k < nnzPerColScan_rightEnd[j]; k++) { // nonzeros in column j of B
                for (nnz_t i = nnzPerColScan_leftStart[B[k].row - B_row_offset];
                     i < nnzPerColScan_leftEnd[B[k].row - B_row_offset]; i++) { // nonzeros in column B[k].row of A

                    C_index = (A[i].row - A_row_offset) + A_row_size * (B[k].col - B_col_offset);

//                    if (rank == 0) std::cout << "A: " << A[i] << "\tB: " << B[k] << "\tC_index: " << C_index
//                                   << "\tA_row_offset = " << A_row_offset
//                                   << "\tB_col_offset = " << B_col_offset << std::endl;

//                    if (rank == 1)
//                        printf("A[i].row = %u, \tB[j].col = %u, \tC_index = %u \n", A[i].row - A_row_offset,
//                               B[k].col - B_col_offset, C_index);

//                    C_temp[C_index] = cooEntry(A[i].row, B[k].col, B[k].val * A[i].val + C_temp[C_index].val);
                    C_temp[C_index] += B[k].val * A[i].val;

//                if(rank==0) printf("A[i].val = %f, B[j].val = %f, C_temp[-] = %f \n", A[i].val, B[j].val, C_temp[A[i].row * c_dense + B[j].col].val);
//                if(rank==1 && A[i].row == 0 && B[j].col == 0) std::cout << "A: " << A[i] << "\tB: " << B[j]
//                     << "\tC: " << C_temp[(A[i].row-A_row_offset) + A_row_size * (B[j].col-B_col_offset)]
//                     << "\tA*B: " << B[j].val * A[i].val << std::endl;
                }
            }
        }

//        print_vector(C_temp, -1, "C_temp", comm);
//        if(rank==0){
//            for(nnz_t i = 0; i < r_dense; i++) {
//                for(nnz_t j = 0; j < c_dense; j++){
//                    std::cout << i << "\t" << j << "\t" << C_temp[A.entry[i].row * c_dense +  B[j].row] << std::endl;
//                }
//            }
//        }

        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 1: step 2 \n");}

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

//        print_vector(C, -1, "C", comm);

        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 1: end \n");

    } else if(A_row_size <= A_col_size) {

        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: start \n");}

        // prepare splits of matrix A by column
        nnz_t A1_nnz = 0, A2_nnz;
        auto A_half_nnz = (nnz_t)ceil(A_nnz/2);
        index_t A_col_half = A_col_size/2;

        if(A_nnz > matmat_nnz_thre){
            for (nnz_t i = 0; i < A_col_size; i++) {
                A1_nnz += nnzPerColScan_leftEnd[i] - nnzPerColScan_leftStart[i];
                if (A1_nnz >= A_half_nnz) {
                    A_col_half = A[nnzPerColScan_leftStart[i]].col + 1 - A_col_offset;
                    break;
                }
            }
        } else { // A_col_half will stay A_col_size/2
            for (nnz_t i = 0; i < A_col_half; i++) {
                A1_nnz += nnzPerColScan_leftEnd[i] - nnzPerColScan_leftStart[i];
            }
        }

        // if A is not being splitted at all following "half nnz method", then swtich to "half col method".
        if(A_col_half == A_col_size){
            A_col_half = A_col_size/2;
            A1_nnz = 0;
            for (nnz_t i = 0; i < A_col_half; i++) {
                A1_nnz += nnzPerColScan_leftEnd[i] - nnzPerColScan_leftStart[i];
            }
        }

        A2_nnz = A_nnz - A1_nnz;

        // prepare splits of matrix B by row
        nnz_t B1_nnz = 0, B2_nnz;
        index_t B_row_half = A_col_half;

        std::vector<index_t> nnzPerCol_middle(B_col_size, 0);
        for(nnz_t i = 0; i < B_col_size; i++){
            for(nnz_t j = nnzPerColScan_rightStart[i]; j < nnzPerColScan_rightEnd[i]; j++) {
                if(B[j].row - B_row_offset < B_row_half){
                    nnzPerCol_middle[B[j].col - B_col_offset]++;
                    B1_nnz++;
                }
            }
        }

//        print_vector(nnzPerCol_middle, -1, "nnzPerCol_middle", comm);

        std::vector<index_t> nnzPerColScan_middle(B_col_size + 1);
        nnzPerColScan_middle[0] = 0;
        for(nnz_t i = 0; i < B_col_size; i++){
            nnzPerColScan_middle[i+1] = nnzPerColScan_middle[i] + nnzPerCol_middle[i];
        }

        B2_nnz = B_nnz - B1_nnz;

        nnzPerCol_middle.clear();
        nnzPerCol_middle.shrink_to_fit();

//        print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);

        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 1 \n");}

        for(nnz_t i = 0; i < B_col_size; i++){
            nnzPerColScan_middle[i] = nnzPerColScan_middle[i+1] + nnzPerColScan_rightStart[i] - nnzPerColScan_middle[i];
//            if(rank==0) printf("nnzPerColScan_middle[%lu] = %u, \tnnzPerColScan_middle[%lu] = %u, \tnnzPerColScan_rightStart = %u \n",
//                    i, nnzPerColScan_middle[i], i+1, nnzPerColScan_middle[i+1], nnzPerColScan_rightStart[i]);
        }

//        print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);
//        if(rank==verbose_rank) printf("rank %d: nnz: (A, A1, A2) = (%lu, %lu, %lu), (B, B1, B2) = (%lu, %lu, %lu), A_col_half = %u of %u \n",
//                           rank, A_nnz, A1_nnz, A2_nnz, B_nnz, B1_nnz, B2_nnz, A_col_half, A_col_size);

        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 2 \n");}

        // A1: start: nnzPerColScan_leftStart,               end: nnzPerColScan_leftEnd
        // A2: start: nnzPerColScan_leftStart[A_col_half],   end: nnzPerColScan_leftEnd[A_col_half]
        // B1: start: nnzPerColScan_rightStart,              end: nnzPerColScan_middle
        // B2: start: nnzPerColScan_middle,                  end: nnzPerColScan_rightEnd

//        MPI_Barrier(comm);
        if(rank==verbose_rank){

            printf("fast_mm: case 2: \nA_nnz: (%lu, %lu, %lu), B_nnz: (%lu, %lu, %lu)\n"
                   "A_size: (%u, %u, %u), B_size: (%u) \n",
                    A_nnz, A1_nnz, A2_nnz, B_nnz, B1_nnz, B2_nnz, A_row_size, A_col_size, A_col_half, B_col_size);

            if(verbose_matmat_A) {
                std::cout << "\nranges of A:" << std::endl;
                for (nnz_t i = 0; i < A_col_size; i++) {
                    std::cout << i << "\t" << nnzPerColScan_leftStart[i] << "\t" << nnzPerColScan_leftEnd[i]
                              << std::endl;
                }

                std::cout << "\nranges of A1:" << std::endl;
                for (nnz_t i = 0; i < A_col_half; i++) {
                    std::cout << i << "\t" << nnzPerColScan_leftStart[i] << "\t" << nnzPerColScan_leftStart[i + 1]
                              << std::endl;
                }

                std::cout << "\nranges of A2:" << std::endl;
                for (nnz_t i = 0; i < A_col_size - A_col_half; i++) {
                    std::cout << i << "\t" << nnzPerColScan_leftStart[A_col_half + i]
                              << "\t" << nnzPerColScan_leftStart[A_col_half + i + 1] << std::endl;
                }

                // print entries of A1:
                std::cout << "\nA1: nnz = " << A1_nnz << std::endl;
                for (nnz_t i = 0; i < A_col_half; i++) {
                    for (nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftStart[i + 1]; j++) {
                        std::cout << j << "\t" << A[j] << std::endl;
                    }
                }

                // print entries of A2:
                std::cout << "\nA2: nnz = " << A2_nnz << std::endl;
                for (nnz_t i = 0; i < A_col_size - A_col_half; i++) {
                    for (nnz_t j = nnzPerColScan_leftStart[A_col_half + i];
                         j < nnzPerColScan_leftStart[A_col_half + i + 1]; j++) {
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
//        MPI_Barrier(comm);

        std::vector<cooEntry> C1, C2;

        // C1 = A1 * B1
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 1 \n");
        fast_mm(&A[0], &B[0], C1, A1_nnz, B1_nnz,
                A_row_size, A_row_offset, A_col_half, A_col_offset,
                B_col_size, B_col_offset,
                nnzPerColScan_leftStart,  nnzPerColScan_leftEnd, // A1
                nnzPerColScan_rightStart, &nnzPerColScan_middle[0], mempool, comm); // B1

        // C2 = A2 * B2
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 2 \n");
        fast_mm(&A[0], &B[0], C2, A2_nnz, B2_nnz,
                A_row_size, A_row_offset, A_col_size-A_col_half, A_col_offset+A_col_half,
                B_col_size, B_col_offset,
                &nnzPerColScan_leftStart[A_col_half], &nnzPerColScan_leftEnd[A_col_half], // A2
                &nnzPerColScan_middle[0], nnzPerColScan_rightEnd, mempool, comm); // B2

//        print_vector(C1, -1, "C1", comm);
//        print_vector(C2, -1, "C2", comm);

        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 3 \n");}
//        if(rank==0 && verbose_matmat) printf("C1.size() = %lu, C2.size() = %lu \n", C1.size(), C2.size());

        // take care of the special cases when either C1 or C2 is empty.
//        if(C1.empty()){
//            C = C2;
//            return 0;
//        } else if(C2.empty()){
//            C = C1;
//            return 0;
//        }

        // take care of the special cases when either C1 or C2 is empty.
        nnz_t i=0;
        if(C1.empty()){
            while(i < C2.size()){
                C.emplace_back(C2[i]);
                i++;
            }
            if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
            return 0;
        }

        if(C2.empty()) {
            while (i < C1.size()) {
                C.emplace_back(C1[i]);
                i++;
            }
            if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
            return 0;
        }

        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 4 \n");}

        // merge C1 and C2
        i = 0;
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
                if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
                return 0;
            }else if(j == C2.size()) {
                while (i < C1.size()) {
                    C.emplace_back(C1[i]);
                    i++;
                }
                if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
                return 0;
            }
        }


        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 2: end \n");

    } else { // A_row_size > A_col_size

        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: start \n");

        // prepare splits of matrix B by column
        nnz_t B1_nnz = 0, B2_nnz;
        auto B_half_nnz = (nnz_t)ceil(B_nnz/2);
        index_t B_col_half = B_col_size/2;

        if(B_nnz > matmat_nnz_thre) {
            for (nnz_t i = 0; i < B_col_size; i++) {
                B1_nnz += nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i];

//                if(rank==verbose_rank) printf("B_nnz = %lu, B_half_nnz = %lu, B1_nnz = %lu, nnz on col %u: %u \n",
//                                              B_nnz, B_half_nnz, B1_nnz, B[nnzPerColScan_rightStart[i]].col,
//                                              nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i]);
                if (B1_nnz >= B_half_nnz) {
                    B_col_half = B[nnzPerColScan_rightStart[i]].col + 1 - B_col_offset;
//                    if(rank==verbose_rank) printf("B_nnz = %lu, B_half_nnz = %lu, B1_nnz = %lu, B_col_half = %u, B_col_size = %u \n",
//                                              B_nnz, B_half_nnz, B1_nnz, B_col_half, B_col_size);
                    break;
                }
            }
        } else {
            for (nnz_t i = 0; i < B_col_half; i++) {
                B1_nnz += nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i];
            }
        }

        // if B is not being splitted at all following "half nnz method", then swtich to "half col method".
        if(B_col_half == B_col_size){
            B_col_half = B_col_size/2;
            B1_nnz = 0;
            for (nnz_t i = 0; i < B_col_half; i++) {
                B1_nnz += nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i];
            }
        }

        B2_nnz = B_nnz - B1_nnz;

        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 1 \n");

        // prepare splits of matrix A by row
        nnz_t A1_nnz = 0, A2_nnz;
        index_t A_row_half = B_col_half;

        std::vector<index_t> nnzPerCol_middle(A_col_size, 0);
        for(nnz_t i = 0; i < A_col_size; i++){
            for(nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftEnd[i]; j++) {
                if(A[j].row - A_row_offset < A_row_half){
                    nnzPerCol_middle[A[j].col - A_col_offset]++;
                    A1_nnz++;
                }
            }
        }

        A2_nnz = A_nnz - A1_nnz;

        std::vector<index_t> nnzPerColScan_middle(A_col_size + 1);
        nnzPerColScan_middle[0] = 0;
        for(nnz_t i = 0; i < A_col_size; i++){
            nnzPerColScan_middle[i+1] = nnzPerColScan_middle[i] + nnzPerCol_middle[i];
        }

        nnzPerCol_middle.clear();
        nnzPerCol_middle.shrink_to_fit();

        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 2 \n");

        for(nnz_t i = 0; i < A_col_size; i++){
            nnzPerColScan_middle[i] = nnzPerColScan_middle[i+1] + nnzPerColScan_leftStart[i] - nnzPerColScan_middle[i];
//            if(rank==0) printf("nnzPerColScan_middle[%lu] = %u \n", i, nnzPerColScan_middle[i]);
        }

//        print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);
//        if(rank==verbose_rank) printf("rank %d: nnz: (A, A1, A2) = (%lu, %lu, %lu), (B, B1, B2) = (%lu, %lu, %lu), B_col_half = %u of %u \n",
//                           rank, A_nnz, A1_nnz, A2_nnz, B_nnz, B1_nnz, B2_nnz, B_col_half, B_col_size);

        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 3 \n");

        // A1: start: nnzPerColScan_leftStart,                end: nnzPerColScan_middle
        // A2: start: nnzPerColScan_middle,                   end: nnzPerColScan_leftEnd
        // B1: start: nnzPerColScan_rightStart,               end: nnzPerColScan_rightEnd
        // B2: start: nnzPerColScan_rightStart[B_col_half], end: nnzPerColScan_rightEnd[B_col_half]

//        MPI_Barrier(comm);
        if(rank==verbose_rank){

            printf("fast_mm: case 3: \nA_nnz: (%lu, %lu, %lu), B_nnz: (%lu, %lu, %lu)\n"
                   "A_size: (%u, %u), B_size: (%u, %u) \n",
                   A_nnz, A1_nnz, A2_nnz, B_nnz, B1_nnz, B2_nnz, A_row_size, A_col_size, B_col_size, B_col_half);

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
                for (nnz_t i = 0; i < B_col_half; i++) {
                    std::cout << i << "\t" << nnzPerColScan_rightStart[i] << "\t" << nnzPerColScan_rightEnd[i]
                              << std::endl;
                }

                std::cout << "\nranges of B2:" << std::endl;
                for (nnz_t i = 0; i < B_col_size - B_col_half; i++) {
                    std::cout << i << "\t" << nnzPerColScan_rightStart[B_col_half + i]
                              << "\t" << nnzPerColScan_rightEnd[B_col_half + i] << std::endl;
                }

                // print entries of B1:
                std::cout << "\nB1: nnz = " << B1_nnz << std::endl;
                for (nnz_t i = 0; i < B_col_half; i++) {
                    for (nnz_t j = nnzPerColScan_rightStart[i]; j < nnzPerColScan_rightEnd[i]; j++) {
                        std::cout << j << "\t" << B[j] << std::endl;
                    }
                }

                // print entries of B2:
                std::cout << "\nB2: nnz = " << B2_nnz << std::endl;
                for (nnz_t i = 0; i < B_col_size - B_col_half; i++) {
                    for (nnz_t j = nnzPerColScan_rightStart[B_col_half + i];
                         j < nnzPerColScan_rightEnd[B_col_half + i]; j++) {
                        std::cout << j << "\t" << B[j] << std::endl;
                    }
                }
            }
        }
//        MPI_Barrier(comm);

        std::vector<cooEntry> C_temp;

        // C1 = A1 * B1
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 1 \n");
        fast_mm(&A[0], &B[0], C_temp, A1_nnz, B1_nnz,
                A_row_half, A_row_offset, A_col_size, A_col_offset,
                B_col_half, B_col_offset,
                nnzPerColScan_leftStart,  &nnzPerColScan_middle[0], // A1
                nnzPerColScan_rightStart, nnzPerColScan_rightEnd, mempool, comm); // B1

        // C2 = A2 * B1:
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 2 \n");
        fast_mm(&A[0], &B[0], C_temp, A2_nnz, B1_nnz,
                A_row_size-A_row_half, A_row_offset+A_row_half, A_col_size, A_col_offset,
                B_col_half, B_col_offset,
                &nnzPerColScan_middle[0], nnzPerColScan_leftEnd, // A2
                nnzPerColScan_rightStart, nnzPerColScan_rightEnd, mempool, comm); // B1

        // C3 = A1 * B2:
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 3 \n");
        fast_mm(&A[0], &B[0], C_temp, A1_nnz, B2_nnz,
                A_row_half, A_row_offset, A_col_size, A_col_offset,
                B_col_size-B_col_half, B_col_offset+B_col_half,
                nnzPerColScan_leftStart,  &nnzPerColScan_middle[0], // A1
                &nnzPerColScan_rightStart[B_col_half], &nnzPerColScan_rightEnd[B_col_half], mempool, comm); // B2

        // C4 = A2 * B2
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 4 \n");
        fast_mm(&A[0], &B[0], C_temp, A2_nnz, B2_nnz,
                A_row_size-A_row_half, A_row_offset+A_row_half, A_col_size, A_col_offset,
                B_col_size-B_col_half, B_col_offset+B_col_half,
                &nnzPerColScan_middle[0], nnzPerColScan_leftEnd, // A2
                &nnzPerColScan_rightStart[B_col_half], &nnzPerColScan_rightEnd[B_col_half], mempool, comm); // B2

        // C1 = A1 * B1:
//        fast_mm(A1, B1, C_temp, A_row_size/2, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size/2, B_col_offset, comm);
        // C2 = A2 * B1:
//        fast_mm(A2, B1, C_temp, A_row_size-A_row_size/2, A_row_offset+A_row_size/2, A_col_size, A_col_offset, B_row_offset, B_col_size/2, B_col_offset, comm);
        // C3 = A1 * B2:
//        fast_mm(A1, B2, C_temp, A_row_size/2, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size-B_col_size/2, B_col_offset+B_col_size/2, comm);
        // C4 = A2 * B2
//        fast_mm(A2, B2, C_temp, A_row_size-A_row_size/2, A_row_offset+A_row_size/2, A_col_size, A_col_offset, B_row_offset, B_col_size-B_col_size/2, B_col_offset+B_col_size/2, comm);

//        if(rank==0 && verbose_matmat) printf("fast_mm: case 3: step 4 \n");

        std::sort(C_temp.begin(), C_temp.end());

        // remove duplicates.
        for(nnz_t i=0; i<C_temp.size(); i++){
            C.emplace_back(C_temp[i]);
            while(i<C_temp.size()-1 && C_temp[i] == C_temp[i+1]){ // values of entries with the same row and col should be added.
                C.back().val += C_temp[i+1].val;
                i++;
            }
        }

        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: end \n");
    }

    if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");

    return 0;
}


// this version splits the matrices by the middle row and column.
int saena_object::fast_mm(cooEntry *A, cooEntry *B, std::vector<cooEntry> &C, nnz_t A_nnz, nnz_t B_nnz,
                          index_t A_row_size, index_t A_row_offset, index_t A_col_size, index_t A_col_offset,
                          index_t B_col_size, index_t B_col_offset,
                          index_t *nnzPerColScan_leftStart, index_t *nnzPerColScan_leftEnd,
                          index_t *nnzPerColScan_rightStart, index_t *nnzPerColScan_rightEnd,
                          value_t *mempool, MPI_Comm comm){ $

    // This function has three parts:
    // 1- A is horizontal (row > col)
    // 2- A is vertical
    // 3- do multiplication when blocks of A and B are small enough. Put them in a sparse C, where C_ij = A_i * B_j
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

    index_t B_row_offset = A_col_offset;

    int verbose_rank = 0;
    if(rank==verbose_rank && verbose_matmat) printf("\nfast_mm: start \n");

    if(A_nnz == 0 || B_nnz == 0){
        if(rank==verbose_rank && verbose_matmat) printf("\nskip: A_nnz == 0 || B_nnz == 0\n\n");
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

        if(rank==verbose_rank && (verbose_matmat || verbose_matmat_recursive)){printf("fast_mm: case 1: start \n");}

        // initialize
//        std::vector<cooEntry> C_temp(A_row_size * B_col_size); // 1D array is better than 2D for many reasons.
//        for(nnz_t i = 0; i < A_row_size * B_col_size; i++){
//            C_temp[i] = cooEntry(0, 0, 0);
//        }

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

#ifdef _DEBUG_
//                    if (rank == 0) std::cout << "A: " << A[i] << "\tB: " << B[k] << "\tC_index: " << C_index
//                                   << "\tA_row_offset = " << A_row_offset
//                                   << "\tB_col_offset = " << B_col_offset << std::endl;
#endif
                    C_temp[C_index] += B[k].val * A[i].val;

#ifdef _DEBUG_
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

    } else if(A_row_size <= A_col_size) {

#ifdef _DEBUG_
        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: start \n");}
#endif

        // prepare splits of matrix A by column
        nnz_t A1_nnz = 0, A2_nnz;

        for(nnz_t i = 0; i < A_col_size; i++){
            for(nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftEnd[i]; j++) {
                if(A[j].col - A_col_offset < A_col_size/2) {
                    A1_nnz++;
                }
            }
        }

        A2_nnz = A_nnz - A1_nnz;

        // prepare splits of matrix B by row
        nnz_t B1_nnz = 0, B2_nnz;

        std::vector<index_t> nnzPerCol_middle(B_col_size, 0);

        for(nnz_t i = 0; i < B_col_size; i++){
            for(nnz_t j = nnzPerColScan_rightStart[i]; j < nnzPerColScan_rightEnd[i]; j++) {
                if(B[j].row - B_row_offset < A_col_size/2){ // A_col_size/2 is middle row of B too.
                    nnzPerCol_middle[B[j].col - B_col_offset]++;
                    B1_nnz++;
                }
            }
        }

//        print_vector(nnzPerCol_middle, -1, "nnzPerCol_middle", comm);

        std::vector<index_t> nnzPerColScan_middle(B_col_size + 1);
        nnzPerColScan_middle[0] = 0;
        for(nnz_t i = 0; i < B_col_size; i++){
            nnzPerColScan_middle[i+1] = nnzPerColScan_middle[i] + nnzPerCol_middle[i];
        }

        B2_nnz = B_nnz - B1_nnz;

        nnzPerCol_middle.clear();
        nnzPerCol_middle.shrink_to_fit();

#ifdef _DEBUG_
//        print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);
//        if(rank==0) printf("rank %d: A_nnz = %lu, A1_nnz = %lu, A2_nnz = %lu, B_nnz = %lu, B1_nnz = %lu, B2_nnz = %lu \n",
//                rank, A_nnz, A1_nnz, A2_nnz, B_nnz, B1_nnz, B2_nnz);
        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 1 \n");}
#endif

        for(nnz_t i = 0; i < B_col_size; i++){
            nnzPerColScan_middle[i] = nnzPerColScan_middle[i+1] + nnzPerColScan_rightStart[i] - nnzPerColScan_middle[i];
//            if(rank==0) printf("nnzPerColScan_middle[%lu] = %u, \tnnzPerColScan_middle[%lu] = %u, \tnnzPerColScan_rightStart = %u \n",
//                    i, nnzPerColScan_middle[i], i+1, nnzPerColScan_middle[i+1], nnzPerColScan_rightStart[i]);
        }

#ifdef _DEBUG_
//        print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);
        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 2 \n");}
#endif

        // A1: start: nnzPerColScan_leftStart,               end: nnzPerColScan_leftEnd
        // A2: start: nnzPerColScan_leftStart[A_col_size/2], end: nnzPerColScan_leftEnd[A_col_size/2]
        // B1: start: nnzPerColScan_rightStart,              end: nnzPerColScan_middle
        // B2: start: nnzPerColScan_middle,                  end: nnzPerColScan_rightEnd

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
                A_row_size, A_row_offset, A_col_size/2, A_col_offset,
                B_col_size, B_col_offset,
                nnzPerColScan_leftStart,  nnzPerColScan_leftEnd, // A1
                nnzPerColScan_rightStart, &nnzPerColScan_middle[0], mempool, comm); // B1

        // C2 = A2 * B2
#ifdef _DEBUG_
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 2 \n");
#endif

        fast_mm(&A[0], &B[0], C2, A2_nnz, B2_nnz,
                A_row_size, A_row_offset, A_col_size-A_col_size/2, A_col_offset+A_col_size/2,
                B_col_size, B_col_offset,
                &nnzPerColScan_leftStart[A_col_size/2], &nnzPerColScan_leftEnd[A_col_size/2], // A2
                &nnzPerColScan_middle[0], nnzPerColScan_rightEnd, mempool, comm); // B2

#ifdef _DEBUG_
//        print_vector(C1, -1, "C1", comm);
//        print_vector(C2, -1, "C2", comm);

        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 3 \n");}
//        if(rank==0 && verbose_matmat) printf("C1.size() = %lu, C2.size() = %lu \n", C1.size(), C2.size());
#endif

        // take care of the special cases when either C1 or C2 is empty.
        nnz_t i=0;
        if(C1.empty()){
            while(i < C2.size()){
                C.emplace_back(C2[i]);
                i++;
            }

#ifdef _DEBUG_
            if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
#endif
            return 0;
        }

        if(C2.empty()) {
            while (i < C1.size()) {
                C.emplace_back(C1[i]);
                i++;
            }
#ifdef _DEBUG_
            if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
#endif
            return 0;
        }

#ifdef _DEBUG_
        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 4 \n");}
#endif

        // merge C1 and C2
        i = 0;
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

    } else { // A_row_size > A_col_size

#ifdef _DEBUG_
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: start \n");
#endif
        // prepare splits of matrix B by column
        nnz_t B1_nnz = 0, B2_nnz;

        for(nnz_t i = 0; i < B_col_size; i++){
            for(nnz_t j = nnzPerColScan_rightStart[i]; j < nnzPerColScan_rightEnd[i]; j++) {
                if(B[j].col - B_col_offset < B_col_size/2){
                    B1_nnz++;
                }
            }
        }

        B2_nnz = B_nnz - B1_nnz;

#ifdef _DEBUG_
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 1 \n");
#endif
        // prepare splits of matrix A by row
        nnz_t A1_nnz = 0, A2_nnz;

        std::vector<index_t> nnzPerCol_middle(A_col_size, 0);

        for(nnz_t i = 0; i < A_col_size; i++){
            for(nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftEnd[i]; j++) {
                if(A[j].row - A_row_offset < A_row_size/2){
                    nnzPerCol_middle[A[j].col - A_col_offset]++;
                    A1_nnz++;
                }
            }
        }

        std::vector<index_t> nnzPerColScan_middle(A_col_size + 1);
        nnzPerColScan_middle[0] = 0;
        for(nnz_t i = 0; i < A_col_size; i++){
            nnzPerColScan_middle[i+1] = nnzPerColScan_middle[i] + nnzPerCol_middle[i];
        }

        A2_nnz = A_nnz - A1_nnz;

        nnzPerCol_middle.clear();
        nnzPerCol_middle.shrink_to_fit();

#ifdef _DEBUG_
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 2 \n");
#endif
        for(nnz_t i = 0; i < A_col_size; i++){
            nnzPerColScan_middle[i] = nnzPerColScan_middle[i+1] + nnzPerColScan_leftStart[i] - nnzPerColScan_middle[i];
//            if(rank==0) printf("nnzPerColScan_middle[%lu] = %u \n", i, nnzPerColScan_middle[i]);
        }

#ifdef _DEBUG_
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 3 \n");
#endif

        // A1: start: nnzPerColScan_leftStart,                end: nnzPerColScan_middle
        // A2: start: nnzPerColScan_middle,                   end: nnzPerColScan_leftEnd
        // B1: start: nnzPerColScan_rightStart,               end: nnzPerColScan_rightEnd
        // B2: start: nnzPerColScan_rightStart[B_col_size/2], end: nnzPerColScan_rightEnd[B_col_size/2]

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
                A_row_size/2, A_row_offset, A_col_size, A_col_offset,
                B_col_size/2, B_col_offset,
                nnzPerColScan_leftStart,  &nnzPerColScan_middle[0], // A1
                nnzPerColScan_rightStart, nnzPerColScan_rightEnd, mempool, comm); // B1

        // C2 = A2 * B1:
#ifdef _DEBUG_
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 2 \n");
#endif
        fast_mm(&A[0], &B[0], C_temp, A2_nnz, B1_nnz,
                A_row_size-A_row_size/2, A_row_offset+A_row_size/2, A_col_size, A_col_offset,
                B_col_size/2, B_col_offset,
                &nnzPerColScan_middle[0], nnzPerColScan_leftEnd, // A2
                nnzPerColScan_rightStart, nnzPerColScan_rightEnd, mempool, comm); // B1

        // C3 = A1 * B2:
#ifdef _DEBUG_
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 3 \n");
#endif
        fast_mm(&A[0], &B[0], C_temp, A1_nnz, B2_nnz,
                A_row_size/2, A_row_offset, A_col_size, A_col_offset,
                B_col_size-B_col_size/2, B_col_offset+B_col_size/2,
                nnzPerColScan_leftStart,  &nnzPerColScan_middle[0], // A1
                &nnzPerColScan_rightStart[B_col_size/2], &nnzPerColScan_rightEnd[B_col_size/2], mempool, comm); // B2

        // C4 = A2 * B2
#ifdef _DEBUG_
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 4 \n");
#endif
        fast_mm(&A[0], &B[0], C_temp, A2_nnz, B2_nnz,
                A_row_size-A_row_size/2, A_row_offset+A_row_size/2, A_col_size, A_col_offset,
                B_col_size-B_col_size/2, B_col_offset+B_col_size/2,
                &nnzPerColScan_middle[0], nnzPerColScan_leftEnd, // A2
                &nnzPerColScan_rightStart[B_col_size/2], &nnzPerColScan_rightEnd[B_col_size/2], mempool, comm); // B2

        // C1 = A1 * B1:
//        fast_mm(A1, B1, C_temp, A_row_size/2, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size/2, B_col_offset, comm);
        // C2 = A2 * B1:
//        fast_mm(A2, B1, C_temp, A_row_size-A_row_size/2, A_row_offset+A_row_size/2, A_col_size, A_col_offset, B_row_offset, B_col_size/2, B_col_offset, comm);
        // C3 = A1 * B2:
//        fast_mm(A1, B2, C_temp, A_row_size/2, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size-B_col_size/2, B_col_offset+B_col_size/2, comm);
        // C4 = A2 * B2
//        fast_mm(A2, B2, C_temp, A_row_size-A_row_size/2, A_row_offset+A_row_size/2, A_col_size, A_col_offset, B_row_offset, B_col_size-B_col_size/2, B_col_offset+B_col_size/2, comm);

//        if(rank==0 && verbose_matmat) printf("fast_mm: case 3: step 4 \n");

        std::sort(C_temp.begin(), C_temp.end());

        // remove duplicates.
        for(nnz_t i=0; i<C_temp.size(); i++){
            C.emplace_back(C_temp[i]);
            while(i<C_temp.size()-1 && C_temp[i] == C_temp[i+1]){ // values of entries with the same row and col should be added.
                C.back().val += C_temp[i+1].val;
                i++;
            }
        }

#ifdef _DEBUG_
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: end \n");
#endif
    }

#ifdef _DEBUG_
    if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
#endif

    return 0;
}


int saena_object::coarsen(Grid *grid) { $

    // Output: Ac = R * A * P
    // Steps:
    // 1- Compute AP = A * P. To do that use the transpose of R_i, instead of P. Pass all R_j's to all the processors,
    //    Then, multiply local A_i by R_jon each process.
    // 2- Compute RAP = R * AP. Use transpose of P_i instead of R. It is done locally. So multiply P_i * (AP)_i.
    // 3- Sort and remove local duplicates.
    // 4- Do a parallel sort based on row-major order. A modified version of par::sampleSort from usort is used here.
    //    Again, remove duplicates.
    // 5- Not complete yet: Sparsify Ac.

    saena_matrix *A    = grid->A;
    prolong_matrix *P  = &grid->P;
    restrict_matrix *R = &grid->R;
    saena_matrix *Ac   = &grid->Ac;

    MPI_Comm comm = A->comm;
//    Ac->active_old_comm = true;

//    int rank1, nprocs1;
//    MPI_Comm_size(comm, &nprocs1);
//    MPI_Comm_rank(comm, &rank1);
//    if(A->active_old_comm)
//        printf("rank = %d, nprocs = %d active\n", rank1, nprocs1);

//    print_vector(A->entry, -1, "A->entry", comm);
//    print_vector(P->entry, -1, "P->entry", comm);
//    print_vector(R->entry, -1, "R->entry", comm);

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef _DEBUG_
    if (verbose_coarsen) {
        MPI_Barrier(comm);
        if (rank == 0) printf("start of coarsen nprocs: %d \n", nprocs);
        MPI_Barrier(comm);
        printf("rank %d: A.Mbig = %u, \tA.M = %u, \tA.nnz_g = %lu, \tA.nnz_l = %lu \n", rank, A->Mbig, A->M, A->nnz_g,
               A->nnz_l);
        MPI_Barrier(comm);
        printf("rank %d: P.Mbig = %u, \tP.M = %u, \tP.nnz_g = %lu, \tP.nnz_l = %lu \n", rank, P->Mbig, P->M, P->nnz_g,
               P->nnz_l);
        MPI_Barrier(comm);
        printf("rank %d: R.Mbig = %u, \tR.M = %u, \tR.nnz_g = %lu, \tR.nnz_l = %lu \n", rank, R->Mbig, R->M, R->nnz_g,
               R->nnz_l);
        MPI_Barrier(comm);
    }
#endif

    Ac->Mbig = P->Nbig;
    Ac->M = P->splitNew[rank+1] - P->splitNew[rank];
    Ac->M_old = Ac->M;
    Ac->comm = A->comm;
    Ac->comm_old = A->comm;
    Ac->last_M_shrink = A->last_M_shrink;
    Ac->last_density_shrink = A->last_density_shrink;
//    Ac->last_nnz_shrink = A->last_nnz_shrink;
//    Ac->enable_shrink = A->enable_shrink;
//    Ac->enable_shrink = A->enable_shrink_next_level;
    Ac->active_old_comm = true;
    Ac->density = float(Ac->nnz_g) / (Ac->Mbig * Ac->Mbig);
    Ac->switch_to_dense = switch_to_dense;
    Ac->dense_threshold = dense_threshold;

    Ac->cpu_shrink_thre1 = A->cpu_shrink_thre1; //todo: is this required?
    if(A->cpu_shrink_thre2_next_level != -1) // this is -1 by default.
        Ac->cpu_shrink_thre2 = A->cpu_shrink_thre2_next_level;

    //return these to default, since they have been used in the above part.
    A->cpu_shrink_thre2_next_level = -1;
    A->enable_shrink_next_level = false;
    Ac->split = P->splitNew;

//    MPI_Barrier(comm);
//    printf("Ac: rank = %d \tMbig = %u \tM = %u \tnnz_g = %lu \tnnz_l = %lu \tdensity = %f\n",
//           rank, Ac->Mbig, Ac->M, Ac->nnz_g, Ac->nnz_l, Ac->density);
//    MPI_Barrier(comm);

//    if(verbose_coarsen){
//        printf("\nrank = %d, Ac->Mbig = %u, Ac->M = %u, Ac->nnz_l = %lu, Ac->nnz_g = %lu \n", rank, Ac->Mbig, Ac->M, Ac->nnz_l, Ac->nnz_g);}

#ifdef _DEBUG_
    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 2: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // ********** minor shrinking **********
    for(index_t i = 0; i < Ac->split.size()-1; i++){
        if(Ac->split[i+1] - Ac->split[i] == 0){
//            printf("rank %d: shrink minor in coarsen: i = %d, split[i] = %d, split[i+1] = %d\n", rank, i, Ac->split[i], Ac->split[i+1]);
            Ac->shrink_cpu_minor();
            break;
        }
    }

//    int nprocs_updated;
//    MPI_Comm_size(Ac->comm, &nprocs_updated);

#ifdef _DEBUG_
    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 3: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // *******************************************************
    // multiply: AP = A_i * P_j. in which P_j = R_j_tranpose and 0 <= j < nprocs.
    // *******************************************************
    // this is an overlapped version of this multiplication, similar to dense_matvec.

    // local transpose of R is being used to compute A*P. So R is transposed locally here.
    std::vector<cooEntry> R_tranpose(R->entry.size());
    transpose_locally(R->entry, R->entry.size(), R->splitNew[rank], R_tranpose);

//    print_vector(R->entry, -1, "R->entry", comm);
//    print_vector(R_tranpose, -1, "R_tranpose", comm);

    std::vector<index_t> nnzPerCol_left(A->Mbig, 0);
//    unsigned int *AnnzPerCol_p = &nnzPerCol_left[0] - A[0].col;
    for(nnz_t i = 0; i < A->entry.size(); i++){
//        if(rank==0) printf("A[i].col = %u, \tA_col_size = %u \n", A[i].col - A_col_offset, A_col_size);
        nnzPerCol_left[A->entry[i].col]++;
    }
//    print_vector(A->entry, 1, "A->entry", comm);
//    print_vector(nnzPerCol_left, 1, "nnzPerCol_left", comm);

    std::vector<index_t> nnzPerColScan_left(A->Mbig+1);
    nnzPerColScan_left[0] = 0;
    for(nnz_t i = 0; i < A->Mbig; i++){
        nnzPerColScan_left[i+1] = nnzPerColScan_left[i] + nnzPerCol_left[i];
    }

    nnzPerCol_left.clear();
//    nnzPerCol_left.shrink_to_fit();

//    print_vector(nnzPerColScan_left, -1, "nnzPerColScan_left", comm);

    // this is done in the for loop for all R_i's, including the local one.
//    std::vector<unsigned int> nnzPerCol_right(R->M, 0); // range of rows of R is range of cols of R_transpose.
//    for(nnz_t i = 0; i < R_tranpose.size(); i++){
//        nnzPerCol_right[R_tranpose[i].col]++;
//    }
//    std::vector<nnz_t> nnzPerColScan_right(P->M+1);
//    nnzPerColScan_right[0] = 0;
//    for(nnz_t i = 0; i < P->M; i++){
//        nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
//    }
    std::vector<index_t> nnzPerCol_right; // range of rows of R is range of cols of R_transpose.
    std::vector<index_t> nnzPerColScan_right;

//    print_vector(P->splitNew, -1, "P->splitNew", comm);

#ifdef _DEBUG_
    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 4: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // mempool to be used for dense matmat in fast_mm
    value_t *mempool = new value_t[matmat_size_thre];

    std::vector<cooEntry> AP;

    if(nprocs > 1){
        int right_neighbor = (rank + 1)%nprocs;
        int left_neighbor  = rank - 1;
        if (left_neighbor < 0)
            left_neighbor += nprocs;
//        if(rank==0) printf("left_neighbor = %d, right_neighbor = %d\n", left_neighbor, right_neighbor);

        int owner;
        unsigned long send_size = R_tranpose.size();
        unsigned long recv_size;
        index_t mat_recv_M;

        std::vector<cooEntry> mat_recv = R_tranpose;
        std::vector<cooEntry> mat_send = R_tranpose;

        R_tranpose.clear();
        R_tranpose.shrink_to_fit();

        auto *requests = new MPI_Request[4];
        auto *statuses = new MPI_Status[4];

        for(int k = rank; k < rank+nprocs; k++){
            // Both local and remote loops are done here. The first iteration is the local loop. The rest are remote.
            // Send R_tranpose to the left_neighbor processor, receive R_tranpose from the right_neighbor processor.
            // In the next step: send R_tranpose that was received in the previous step to the left_neighbor processor,
            // receive R_tranpose from the right_neighbor processor. And so on.
            // --------------------------------------------------------------------

            // communicate size
            MPI_Irecv(&recv_size, 1, MPI_UNSIGNED_LONG, right_neighbor, right_neighbor, comm, requests);
            MPI_Isend(&send_size, 1, MPI_UNSIGNED_LONG, left_neighbor,  rank,           comm, requests+1);
            MPI_Waitall(1, requests, statuses);
//          printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
            mat_recv.resize(recv_size);

//          print_vector(mat_recv, -1, "mat_recv", A->comm);
//          print_vector(mat_send, -1, "mat_send", A->comm);

            // communicate data
            MPI_Irecv(&mat_recv[0], recv_size, cooEntry::mpi_datatype(), right_neighbor, right_neighbor, comm, requests+2);
            MPI_Isend(&mat_send[0], send_size, cooEntry::mpi_datatype(), left_neighbor,  rank,           comm, requests+3);

            owner = k%nprocs;
            mat_recv_M = P->splitNew[owner + 1] - P->splitNew[owner];
//          printf("rank %d: owner = %d, mat_recv_M = %d, B_col_offset = %u \n", rank, owner, mat_recv_M, P->splitNew[owner]);

            nnzPerCol_right.assign(mat_recv_M, 0);
            for(nnz_t i = 0; i < mat_send.size(); i++){
                nnzPerCol_right[mat_send[i].col - P->splitNew[owner]]++;
            }
//          print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);

            nnzPerColScan_right.resize(mat_recv_M+1);
            nnzPerColScan_right[0] = 0;
            for(nnz_t i = 0; i < mat_recv_M; i++){
                nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
            }
//          print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);

            fast_mm(&A->entry[0], &mat_send[0], AP, A->entry.size(), mat_send.size(),
                    A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[owner],
                    &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
                    &nnzPerColScan_right[0], &nnzPerColScan_right[1], mempool, A->comm);

//          print_vector(AP, -1, "AP", A->comm);

            MPI_Waitall(3, requests+1, statuses+1);

            mat_recv.swap(mat_send);
            send_size = recv_size;
//          print_vector(mat_send, -1, "mat_send", A->comm);
//          print_vector(mat_recv, -1, "mat_recv", A->comm);
//          prev_owner = owner;
//          printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
        }

        mat_send.clear();
        mat_send.shrink_to_fit();
        mat_recv.clear();
        mat_recv.shrink_to_fit();

        delete [] requests;
        delete [] statuses;

    } else { // nprocs == 1 -> serial

        index_t mat_recv_M = P->splitNew[rank + 1] - P->splitNew[rank];

        nnzPerCol_right.assign(mat_recv_M, 0);
        for(nnz_t i = 0; i < R_tranpose.size(); i++){
            nnzPerCol_right[R_tranpose[i].col - P->splitNew[rank]]++;
        }
//        print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);

        nnzPerColScan_right.resize(mat_recv_M+1);
        nnzPerColScan_right[0] = 0;
        for(nnz_t i = 0; i < mat_recv_M; i++){
            nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
        }

        fast_mm(&A->entry[0], &R_tranpose[0], AP, A->entry.size(), R_tranpose.size(),
                A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[rank],
                &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
                &nnzPerColScan_right[0], &nnzPerColScan_right[1], mempool, A->comm);

        R_tranpose.clear();
        R_tranpose.shrink_to_fit();
    }

    std::sort(AP.begin(), AP.end());
//    print_vector(AP, -1, "AP", A->comm);

//    nnzPerCol_right.clear();
//    nnzPerColScan_left.clear();
//    nnzPerColScan_right.clear();
//    nnzPerCol_right.shrink_to_fit();
//    nnzPerColScan_left.shrink_to_fit();
//    nnzPerColScan_right.shrink_to_fit();

#ifdef _DEBUG_
    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 5: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // *******************************************************
    // multiply: R_i * (AP)_i. in which R_i = P_i_tranpose
    // *******************************************************

    // local transpose of P is being used to compute R*(AP). So P is transposed locally here.
    std::vector<cooEntry> P_tranpose(P->entry.size());
    transpose_locally(P->entry, P->entry.size(), P_tranpose);

    // convert the indices to global
    for(nnz_t i = 0; i < P_tranpose.size(); i++){
        P_tranpose[i].col += P->split[rank];
    }

#ifdef _DEBUG_
//    print_vector(P->entry, -1, "P->entry", comm);
//    print_vector(P_tranpose, -1, "P_tranpose", comm);
#endif

    // compute nnzPerColScan_left for P_tranpose
    nnzPerCol_left.assign(P->M, 0);
    for(nnz_t i = 0; i < P_tranpose.size(); i++){
        nnzPerCol_left[P_tranpose[i].col - P->split[rank]]++;
    }

    nnzPerColScan_left.resize(P->M+1);
    nnzPerColScan_left[0] = 0;
    for(nnz_t i = 0; i < P->M; i++){
        nnzPerColScan_left[i+1] = nnzPerColScan_left[i] + nnzPerCol_left[i];
    }

    nnzPerCol_left.clear();
    nnzPerCol_left.shrink_to_fit();

#ifdef _DEBUG_
//    print_vector(nnzPerColScan_left, -1, "nnzPerColScan_left", comm);
#endif

    // compute nnzPerColScan_left for AP
    nnzPerCol_right.assign(P->Nbig, 0);
    for(nnz_t i = 0; i < AP.size(); i++){
        nnzPerCol_right[AP[i].col]++;
    }

#ifdef _DEBUG_
//    print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
#endif

    nnzPerColScan_right.resize(P->Nbig+1);
    nnzPerColScan_right[0] = 0;
    for(nnz_t i = 0; i < P->Nbig; i++){
        nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
    }

    nnzPerCol_right.clear();
    nnzPerCol_right.shrink_to_fit();

#ifdef _DEBUG_
//    print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

    // multiply: R_i * (AP)_i. in which R_i = P_i_tranpose
    std::vector<cooEntry> RAP_temp;
    fast_mm(&P_tranpose[0], &AP[0], RAP_temp, P_tranpose.size(), AP.size(),
            P->Nbig, 0, P->M, P->split[rank], P->Nbig, 0,
            &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
            &nnzPerColScan_right[0], &nnzPerColScan_right[1], mempool, A->comm);

    AP.clear();
    AP.shrink_to_fit();
    P_tranpose.clear();
    P_tranpose.shrink_to_fit();
    nnzPerColScan_left.clear();
    nnzPerColScan_left.shrink_to_fit();
    nnzPerColScan_right.clear();
    nnzPerColScan_right.shrink_to_fit();
    delete[] mempool;

#ifdef _DEBUG_
//    print_vector(RAP_temp, -1, "RAP_temp", A->comm);
    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 6: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // remove local duplicates.
    // Entries should be sorted in row-major order first, since the matrix should be partitioned based on rows.
    // So cooEntry_row is used here. Remove local duplicates and put them in RAP_temp_row.
    std::vector<cooEntry_row> RAP_temp_row;
    for(nnz_t i=0; i<RAP_temp.size(); i++){
        RAP_temp_row.emplace_back(cooEntry_row( RAP_temp[i].row, RAP_temp[i].col, RAP_temp[i].val ));
        while(i<RAP_temp.size()-1 && RAP_temp[i] == RAP_temp[i+1]){ // values of entries with the same row and col should be added.
            RAP_temp_row.back().val += RAP_temp[i+1].val;
            i++;
        }
    }

    RAP_temp.clear();
    RAP_temp.shrink_to_fit();

#ifdef _DEBUG_
//    MPI_Barrier(comm); printf("rank %d: RAP_temp_row.size = %lu \n", rank, RAP_temp_row.size()); MPI_Barrier(comm);
//    print_vector(RAP_temp_row, -1, "RAP_temp_row", comm);
//    print_vector(P->splitNew, -1, "P->splitNew", comm);
#endif

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 7: rank = %d\n", rank); MPI_Barrier(comm);}

    std::vector<cooEntry_row> RAP_row_sorted;
    par::sampleSort(RAP_temp_row, RAP_row_sorted, P->splitNew, comm);

    RAP_temp_row.clear();
    RAP_temp_row.shrink_to_fit();

#ifdef _DEBUG_
//    print_vector(RAP_row_sorted, -1, "RAP_row_sorted", A->comm);
//    MPI_Barrier(comm); printf("rank %d: RAP_row_sorted.size = %lu \n", rank, RAP_row_sorted.size()); MPI_Barrier(comm);

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 8: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

//    std::vector<cooEntry> RAP_sorted(RAP_row_sorted.size());
//    memcpy(&RAP_sorted[0], &RAP_row_sorted[0], RAP_row_sorted.size() * sizeof(cooEntry));
//    RAP_row_sorted.clear();
//    RAP_row_sorted.shrink_to_fit();

    // *******************************************************
    // form Ac
    // *******************************************************

    if(!doSparsify){

        // *******************************************************
        // version 1: without sparsification
        // *******************************************************
        // since RAP_row_sorted is sorted in row-major order, Ac->entry will be the same.

        // remove duplicates.
        cooEntry temp;
        for(nnz_t i = 0; i < RAP_row_sorted.size(); i++){
            temp = cooEntry(RAP_row_sorted[i].row, RAP_row_sorted[i].col, RAP_row_sorted[i].val);
            while(i<RAP_row_sorted.size()-1 && RAP_row_sorted[i] == RAP_row_sorted[i+1]){ // values of entries with the same row and col should be added.
                temp.val += RAP_row_sorted[i+1].val;
                i++;
            }
            Ac->entry.emplace_back( temp );
        }

        RAP_row_sorted.clear();
        RAP_row_sorted.shrink_to_fit();

    }else{

        // remove duplicates.
        // compute Frobenius norm squared (norm_frob_sq).
        cooEntry_row temp;
        double max_val = 0;
        double norm_frob_sq_local = 0, norm_frob_sq = 0;
        std::vector<cooEntry_row> Ac_orig;
//        nnz_t no_sparse_size = 0;
        for(nnz_t i=0; i<RAP_row_sorted.size(); i++){
            temp = cooEntry_row(RAP_row_sorted[i].row, RAP_row_sorted[i].col, RAP_row_sorted[i].val);
            while(i<RAP_row_sorted.size()-1 && RAP_row_sorted[i] == RAP_row_sorted[i+1]){ // values of entries with the same row and col should be added.
                temp.val += RAP_row_sorted[i+1].val;
                i++;
            }

//            if( fabs(val_temp) > sparse_epsilon / 2 / Ac->Mbig)
//            if(temp.val * temp.val > sparse_epsilon * sparse_epsilon / (4 * Ac->Mbig * Ac->Mbig) ){
            Ac_orig.emplace_back( temp );
            norm_frob_sq_local += temp.val * temp.val;
            if( fabs(temp.val) > max_val){
                max_val = temp.val;
            }
//            }
//            no_sparse_size++; //todo: just for test. delete this later!
        }

        MPI_Allreduce(&norm_frob_sq_local, &norm_frob_sq, 1, MPI_DOUBLE, MPI_SUM, comm);

//        if(rank==0) printf("\noriginal size   = %lu\n", Ac_orig.size());
//        if(rank==0) printf("\noriginal size without sparsification   \t= %lu\n", no_sparse_size);
//        if(rank==0) printf("filtered Ac size before sparsification \t= %lu\n", Ac_orig.size());

//        std::sort(Ac_orig.begin(), Ac_orig.end());
//        print_vector(Ac_orig, -1, "Ac_orig", A->comm);

        RAP_row_sorted.clear();
        RAP_row_sorted.shrink_to_fit();

//        auto sample_size = Ac_orig.size();
        auto sample_size_local = nnz_t(sample_sz_percent * Ac_orig.size());
//        auto sample_size = nnz_t(Ac->Mbig * Ac->Mbig * A->density);
//        if(rank==0) printf("sample_size     = %lu \n", sample_size);
        nnz_t sample_size;
        MPI_Allreduce(&sample_size_local, &sample_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

/*
        if(sparsifier == "TRSL"){

            sparsify_trsl1(Ac_orig, Ac->entry, norm_frob_sq, sample_size, comm);

        }else if(sparsifier == "drineas"){

            sparsify_drineas(Ac_orig, Ac->entry, norm_frob_sq, sample_size, comm);

        }else if(sparsifier == "majid"){

            sparsify_majid(Ac_orig, Ac->entry, norm_frob_sq, sample_size, max_val, comm);

        }else{
            printf("\nerror: wrong sparsifier!");
        }
*/

        if(Ac->active_minor) {
            if (sparsifier == "majid") {
                sparsify_majid(Ac_orig, Ac->entry, norm_frob_sq, sample_size, max_val, Ac->comm);
            } else {
                printf("\nerror: wrong sparsifier!");
            }
        }

    }

#ifdef _DEBUG_
//    print_vector(Ac->entry, -1, "Ac->entry", A->comm);
    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 9: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // *******************************************************
    // use this part to print data to be used in Julia, to check the solution.
    // *******************************************************
#ifdef _DEBUG_
/*
//    std::cout << "\n";
//    for(nnz_t i = 0; i < A->entry.size(); i++){
//        std::cout << A->entry[i].row+1 << ", ";
//    }
//    std::cout << "\n";
//    for(nnz_t i = 0; i < A->entry.size(); i++){
//        std::cout << A->entry[i].col+1 << ", ";
//    }
//    std::cout << "\n";
//    for(nnz_t i = 0; i < A->entry.size(); i++){
//        std::cout << A->entry[i].val << ", ";
//    }
//    std::cout << "\n";
//    for(nnz_t i = 0; i < R_tranpose.size(); i++){
//        std::cout << R_tranpose[i].row+1 << ", ";
//    }
//    std::cout << "\n";
//    for(nnz_t i = 0; i < R_tranpose.size(); i++){
//        std::cout << R_tranpose[i].col+1 << ", ";
//    }
//    std::cout << "\n";
//    for(nnz_t i = 0; i < R_tranpose.size(); i++){
//        std::cout << R_tranpose[i].val << ", ";
//    }
//    std::cout << "\n";
//    print_vector(A->entry, -1, "A->entry", A->comm);
//    print_vector(R_tranpose, -1, "R_tranpose", A->comm);
//    std::cout << "\n";
//    for(nnz_t i = 0; i < P_tranpose.size(); i++){
//        std::cout << P_tranpose[i].row+1 << ", ";
//    }
//    std::cout << "\n";
//    for(nnz_t i = 0; i < P_tranpose.size(); i++){
//        std::cout << P_tranpose[i].col+1 << ", ";
//    }
//    std::cout << "\n";
//    for(nnz_t i = 0; i < P_tranpose.size(); i++){
//        std::cout << P_tranpose[i].val << ", ";
//    }
//
//    do this for Julia:
//    I = [1, 2, 3, 5, 1, 2, 4, 6, 1, 3, 4, 7, 2, 3, 4, 8, 1, 5, 6, 7, 2, 5, 6, 8, 3, 5, 7, 8, 4, 6, 7, 8];
//    J = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8];
//    V = [1, -0.333333, -0.333333, -0.333333, -0.333333, 1, -0.333333, -0.333333, -0.333333, 1, -0.333333, -0.333333, -0.333333, -0.333333, 1, -0.333333, -0.333333, 1, -0.333333, -0.333333, -0.333333, -0.333333, 1, -0.333333, -0.333333, -0.333333, 1, -0.333333, -0.333333, -0.333333, -0.333333, 1];
//    A = sparse(I, J ,V)
//    and so on. then compare the multiplication from Julia with the following:
//    print_vector(Ac->entry, -1, "Ac->entry", A->comm);
*/
#endif

    // *******************************************************
    // setup matrix
    // *******************************************************
    // Update this description: Shrinking gets decided inside repartition_nnz() or repartition_row() functions,
    // then repartition happens.
    // Finally, shrink_cpu() and matrix_setup() are called. In this way, matrix_setup is called only once.

    Ac->nnz_l = Ac->entry.size();
    MPI_Allreduce(&Ac->nnz_l, &Ac->nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

#ifdef _DEBUG_
    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 10: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    if(Ac->active_minor){
        comm = Ac->comm;
        int rank_new;
        MPI_Comm_rank(Ac->comm, &rank_new);
//        Ac->print_info(-1);
//        Ac->print_entry(-1);

        // ********** decide about shrinking **********

        if(Ac->enable_shrink && Ac->enable_dummy_matvec && nprocs > 1){
//            MPI_Barrier(Ac->comm); if(rank_new==0) printf("start decide shrinking\n"); MPI_Barrier(Ac->comm);
            Ac->matrix_setup_dummy();
            Ac->compute_matvec_dummy_time();
            Ac->decide_shrinking(A->matvec_dummy_time);
            Ac->erase_after_decide_shrinking();
//            MPI_Barrier(Ac->comm); if(rank_new==0) printf("finish decide shrinking\n"); MPI_Barrier(Ac->comm);
        }

#ifdef _DEBUG_
        if(verbose_coarsen){
            MPI_Barrier(comm); printf("coarsen: step 11: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

        // decide to partition based on number of rows or nonzeros.
//    if(switch_repartition && Ac->density >= repartition_threshold)
        if(switch_repartition && Ac->density >= repartition_threshold){
            if(rank==0) printf("equi-ROW partition for the next level: density = %f, repartition_threshold = %f \n", Ac->density, repartition_threshold);
            Ac->repartition_row(); // based on number of rows
        }else{
            Ac->repartition_nnz(); // based on number of nonzeros
        }

//        if(Ac->shrinked_minor){
//            repartition_u_shrink_minor_prepare(grid);
//        }

#ifdef _DEBUG_
        if(verbose_coarsen){
            MPI_Barrier(comm); printf("coarsen: step 12: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

        repartition_u_shrink_prepare(grid);

        if(Ac->shrinked){
            Ac->shrink_cpu();
        }

#ifdef _DEBUG_
        if(verbose_coarsen){
            MPI_Barrier(comm); printf("coarsen: step 13: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

        if(Ac->active){
            Ac->matrix_setup();

            if(Ac->shrinked && Ac->enable_dummy_matvec)
                Ac->compute_matvec_dummy_time();

            if(switch_to_dense && Ac->density > dense_threshold){
                if(rank==0) printf("Switch to dense: density = %f, dense_threshold = %f \n", Ac->density, dense_threshold);
                Ac->generate_dense_matrix();
            }
        }

//        Ac->print_info(-1);
//        Ac->print_entry(-1);
    }
    comm = grid->A->comm;

#ifdef _DEBUG_
    if(verbose_coarsen){MPI_Barrier(comm); printf("end of coarsen: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

//    grid->A->writeMatrixToFile("Dropbox/Projects/Saena/test_results/37_compare_matmult");
//    grid->P.writeMatrixToFile("Dropbox/Projects/Saena/test_results/37_compare_matmult");
//    grid->R.writeMatrixToFile("Dropbox/Projects/Saena/test_results/37_compare_matmult");

//    petsc_viewer(Ac);
//    petsc_coarsen(&grid->R, grid->A, &grid->P);

    return 0;
} // coarsen()


int saena_object::coarsen_old(Grid *grid){

    // todo: to improve the performance of this function, consider using the arrays used for RA also for RAP.
    // todo: this way allocating and freeing memory will be halved.

    saena_matrix *A    = grid->A;
    prolong_matrix *P  = &grid->P;
    restrict_matrix *R = &grid->R;
    saena_matrix *Ac   = &grid->Ac;

    MPI_Comm comm = A->comm;
//    Ac->active_old_comm = true;

//    int rank1, nprocs1;
//    MPI_Comm_size(comm, &nprocs1);
//    MPI_Comm_rank(comm, &rank1);
//    if(A->active_old_comm)
//        printf("rank = %d, nprocs = %d active\n", rank1, nprocs1);

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(verbose_coarsen){
        MPI_Barrier(comm);
        if(rank==0) printf("\nstart of coarsen nprocs: %d \n", nprocs);
        MPI_Barrier(comm);
        printf("rank %d: A.Mbig = %u, A.M = %u, A.nnz_g = %lu, A.nnz_l = %lu \n", rank, A->Mbig, A->M, A->nnz_g, A->nnz_l);
        MPI_Barrier(comm);
        printf("rank %d: P.Mbig = %u, P.M = %u, P.nnz_g = %lu, P.nnz_l = %lu \n", rank, P->Mbig, P->M, P->nnz_g, P->nnz_l);
        MPI_Barrier(comm);
        printf("rank %d: R.Mbig = %u, R.M = %u, R.nnz_g = %lu, R.nnz_l = %lu \n", rank, R->Mbig, R->M, R->nnz_g, R->nnz_l);
        MPI_Barrier(comm);
    }

    prolong_matrix RA_temp(comm); // RA_temp is being used to remove duplicates while pushing back to RA.

    // ************************************* RA_temp - R on-diag and A local (on and off-diag) *************************************
    // Some local and remote elements of RA_temp are computed here using local R and local A.
    // Note: A local means whole entries of A on this process, not just the diagonal block.

    nnz_t AMaxNnz;
    index_t AMaxM;
    MPI_Allreduce(&A->nnz_l, &AMaxNnz, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);
    MPI_Allreduce(&A->M,     &AMaxM,   1, MPI_UNSIGNED,      MPI_MAX, comm);
//    MPI_Barrier(comm); printf("\nrank=%d, AMaxNnz=%d, AMaxM = %d \n", rank, AMaxNnz, AMaxM); MPI_Barrier(comm);

    // alloacted memory for AMaxM, instead of A.M to avoid reallocation of memory for when receiving data from other procs.
//    std::vector<unsigned int> AnnzPerRow(AMaxM, 0);
//    unsigned int *AnnzPerRow_p = &AnnzPerRow[0] - A->split[rank];
//    for(nnz_t i=0; i<A->nnz_l; i++)
//        AnnzPerRow_p[A->entry[i].row]++;

    std::vector<unsigned int> AnnzPerRow = A->nnzPerRow_local;
//    for(nnz_t i=0; i<A->nnz_l; i++)
//        AnnzPerRow[A->row_remote[i]]++;

    if(nprocs > 1) {
        for (nnz_t i = 0; i < A->M; i++)
            AnnzPerRow[i] += A->nnzPerRow_remote[i];
    }

//    print_vector(AnnzPerRow, -1, "AnnzPerRow", comm);

    // alloacted memory for AMaxM+1, instead of A.M+1 to avoid reallocation of memory for when receiving data from other procs.
    std::vector<unsigned long> AnnzPerRowScan(AMaxM+1);
    AnnzPerRowScan[0] = 0;
    for(index_t i=0; i<A->M; i++){
        AnnzPerRowScan[i+1] = AnnzPerRowScan[i] + AnnzPerRow[i];
//        if(rank==1) printf("i=%lu, AnnzPerRow=%d, AnnzPerRowScan = %d\n", i+A->split[rank], AnnzPerRow[i], AnnzPerRowScan[i+1]);
    }

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 1: rank = %d\n", rank); MPI_Barrier(comm);}

    // find row-wise ordering for A and save it in indicesP
    std::vector<nnz_t> indices_row_wise(A->nnz_l);
    for(nnz_t i=0; i<A->nnz_l; i++)
        indices_row_wise[i] = i;
    std::sort(&indices_row_wise[0], &indices_row_wise[A->nnz_l], sort_indices2(&A->entry[0]));

    nnz_t jstart, jend;
    if(!R->entry_local.empty()) {
        for (index_t i = 0; i < R->nnz_l_local; i++) {
            jstart = AnnzPerRowScan[R->entry_local[i].col - P->split[rank]];
            jend   = AnnzPerRowScan[R->entry_local[i].col - P->split[rank] + 1];
            if(jend - jstart == 0) continue;
            for (nnz_t j = jstart; j < jend; j++) {
//            if(rank==0) std::cout << A->entry[indicesP[j]].row << "\t" << A->entry[indicesP[j]].col << "\t" << A->entry[indicesP[j]].val
//                             << "\t" << R->entry_local[i].col << "\t" << R->entry_local[i].col - P->split[rank] << std::endl;
                RA_temp.entry.emplace_back(cooEntry(R->entry_local[i].row,
                                                    A->entry[indices_row_wise[j]].col,
                                                    R->entry_local[i].val * A->entry[indices_row_wise[j]].val));
            }
        }
    }

//    if(rank==0){
//        std::cout << "\nRA_temp.entry.size = " << RA_temp.entry.size() << std::endl;
//        for(i=0; i<RA_temp.entry.size(); i++)
//            std::cout << RA_temp.entry[i].row + R->splitNew[rank] << "\t" << RA_temp.entry[i].col << "\t" << RA_temp.entry[i].val << std::endl;}

//    printf("rank %d: RA_temp.entry.size_local = %lu \n", rank, RA_temp.entry.size());
//    MPI_Barrier(comm); printf("rank %d: local RA.size = %lu \n", rank, RA_temp.entry.size()); MPI_Barrier(comm);
    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 2: rank = %d\n", rank); MPI_Barrier(comm);}

    // ************************************* RA_temp - R off-diag and A remote (on and off-diag) *************************************

    // find the start and end nnz iterator of each block of R.
    // use A.split for this part to find each block corresponding to each processor's A.
//    unsigned int* left_block_nnz = (unsigned int*)malloc(sizeof(unsigned int)*(nprocs));
//    std::fill(left_block_nnz, &left_block_nnz[nprocs], 0);
    std::vector<nnz_t> left_block_nnz(nprocs, 0);

//    MPI_Barrier(comm); printf("rank=%d entry = %ld \n", rank, R->entry_remote[0].col); MPI_Barrier(comm);

    // find the owner of the first R.remote element.
    long procNum = 0;
//    unsigned int nnzIter = 0;
    if(!R->entry_remote.empty()){
        for (nnz_t i = 0; i < R->entry_remote.size(); i++) {
            procNum = lower_bound2(&*A->split.begin(), &*A->split.end(), R->entry_remote[i].col);
            left_block_nnz[procNum]++;
//        if(rank==1) printf("rank=%d, col = %lu, procNum = %ld \n", rank, R->entry_remote[0].col, procNum);
//        if(rank==1) std::cout << "\nprocNum = " << procNum << "   \tcol = " << R->entry_remote[0].col
//                              << "  \tnnzIter = " << nnzIter << "\t first" << std::endl;
//        nnzIter++;
        }
    }

//    unsigned int* left_block_nnz_scan = (unsigned int*)malloc(sizeof(unsigned int)*(nprocs+1));
    std::vector<nnz_t> left_block_nnz_scan(nprocs+1);
    left_block_nnz_scan[0] = 0;
    for(int i = 0; i < nprocs; i++)
        left_block_nnz_scan[i+1] = left_block_nnz_scan[i] + left_block_nnz[i];

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 3: rank = %d\n", rank); MPI_Barrier(comm);}

//    print_vector(left_block_nnz_scan, -1, "left_block_nnz_scan", comm);

//    printf("rank=%d A.nnz=%u \n", rank, A->nnz_l);
    AnnzPerRow.resize(AMaxM);
    indices_row_wise.resize(AMaxNnz);
    std::vector<cooEntry> Arecv(AMaxNnz);
    int left, right;
    nnz_t nnzSend, nnzRecv;
    long ARecvM;
    MPI_Status sendRecvStatus;
    nnz_t R_block_nnz_own;
    bool send_data = true;
    bool recv_data;
    nnz_t k, kstart, kend;
//    MPI_Barrier(comm); printf("\n\n rank = %d, loop starts! \n", rank); MPI_Barrier(comm);

//    print_vector(R->entry_remote, -1, "R->entry_remote", comm);

    // todo: after adding "R_remote block size" part, the current idea seems more efficient than this idea:
    // todo: change the algorithm so every processor sends data only to the next one and receives from the previous one in each iteration.
    long tag1 = 0;
    for(int i = 1; i < nprocs; i++) {
        // send A to the right processor, receive A from the left processor.
        // "left" decreases by one in each iteration. "right" increases by one.
        right = (rank + i) % nprocs;
        left = rank - i;
        if (left < 0)
            left += nprocs;

        // *************************** RA_temp - A remote - sendrecv(size RBlock) ****************************
        // if the part of R_remote corresponding to this neighbor doesn't have any nonzeros,
        // don't receive A on it and don't send A to it. so communicate #nnz of that R_remote block to decide that.
        // todo: change sendrecv to separate send and recv to skip some of them.

        nnzSend = A->nnz_l;
        jstart  = left_block_nnz_scan[left];
        jend    = left_block_nnz_scan[left + 1];
        recv_data = true;
        if(jend - jstart == 0)
            recv_data = false;
        MPI_Sendrecv(&recv_data, 1, MPI_CXX_BOOL, left, rank,
                     &send_data, 1, MPI_CXX_BOOL, right, right, comm, &sendRecvStatus);

        if(!send_data)
            nnzSend = 0;

        // *************************** RA_temp - A remote - sendrecv(size A) ****************************

        // use sender rank for send and receive tags.
        MPI_Sendrecv(&nnzSend, 1, MPI_UNSIGNED_LONG, right, rank,
                     &nnzRecv, 1, MPI_UNSIGNED_LONG, left,  left, comm, &sendRecvStatus);

//        printf("i=%d, rank=%d, left=%d, right=%d \n", i, rank, left, right);
//        printf("i=%d, rank = %d own A->nnz_l = %u    \tnnzRecv = %u \n", i, rank, A->nnz_l, nnzRecv);

//        if(!R_block_nnz_own)
//            nnzRecv = 0;

//        if(rank==0) printf("i=%d, rank=%d, left=%d, right=%d, nnzSend=%u, nnzRecv = %u, recv_data = %d, send_data = %d \n",
//                           i, rank, left, right, nnzSend, nnzRecv, recv_data, send_data);

        // *************************** RA_temp - A remote - sendrecv(A) ****************************

        // use sender rank for send and receive tags.
        MPI_Sendrecv(&A->entry[0], nnzSend, cooEntry::mpi_datatype(), right, rank,
                     &Arecv[0],    nnzRecv, cooEntry::mpi_datatype(), left,  left, comm, &sendRecvStatus);

//        for(unsigned int j=0; j<nnzRecv; j++)
//                        printf("rank = %d, j=%d \t %lu \t %lu \t %f \n", rank, j, Arecv[j].row , Arecv[j].col, Arecv[j].val);

        // *************************** RA_temp - A remote - multiplication ****************************

        // if #nnz of R_remote block is zero, then skip.
        R_block_nnz_own = jend - jstart;
        if(R_block_nnz_own == 0) continue;

        ARecvM = A->split[left+1] - A->split[left];
        std::fill(&AnnzPerRow[0], &AnnzPerRow[ARecvM], 0);
        unsigned int *AnnzPerRow_p = &AnnzPerRow[0] - A->split[left];
        for(index_t j=0; j<nnzRecv; j++){
            AnnzPerRow_p[Arecv[j].row]++;
//            if(rank==2)
//                printf("%lu \tArecv[j].row[i] = %lu, Arecv[j].row - A->split[left] = %lu \n", j, Arecv[j].row, Arecv[j].row - A->split[left]);
        }

//        print_vector(AnnzPerRow, -1, "AnnzPerRow", comm);

        AnnzPerRowScan[0] = 0;
        for(index_t j=0; j<ARecvM; j++){
            AnnzPerRowScan[j+1] = AnnzPerRowScan[j] + AnnzPerRow[j];
//            if(rank==2) printf("i=%d, AnnzPerRow=%d, AnnzPerRowScan = %d\n", i, AnnzPerRow[i], AnnzPerRowScan[i]);
        }

        // find row-wise ordering for Arecv and save it in indicesPRecv
        for(nnz_t i=0; i<nnzRecv; i++)
            indices_row_wise[i] = i;
        std::sort(&indices_row_wise[0], &indices_row_wise[nnzRecv], sort_indices2(&Arecv[0]));

//        if(rank==1) std::cout << "block start = " << RBlockStart[left] << "\tend = " << RBlockStart[left+1] << "\tleft rank = " << left << "\t i = " << i << std::endl;
        // jstart is the starting entry index of R corresponding to this neighbor.
        for (index_t j = jstart; j < jend; j++) {
//            if(rank==1) std::cout << "R = " << R->entry_remote[j] << std::endl;
//            if(rank==1) std::cout << "col = " << R->entry_remote[j].col << "\tcol-split = " << R->entry_remote[j].col - P->split[left] << "\tstart = " << AnnzPerRowScan[R->entry_remote[j].col - P->split[left]] << "\tend = " << AnnzPerRowScan[R->entry_remote[j].col - P->split[left] + 1] << std::endl;
            // kstart is the starting row entry index of A corresponding to the R entry column.
            kstart = AnnzPerRowScan[R->entry_remote[j].col - P->split[left]];
            kend   = AnnzPerRowScan[R->entry_remote[j].col - P->split[left] + 1];
            if(kend - kstart == 0) continue; // if there isno nonzero on this row of A, then skip.
            for (k = kstart; k < kend; k++) {
//                if(rank==1) std::cout << "R = " << R->entry_remote[j] << "\tA = " << Arecv[indicesPRecv[k]] << std::endl;
                RA_temp.entry.push_back(cooEntry(R->entry_remote[j].row,
                                                 Arecv[indices_row_wise[k]].col,
                                                 R->entry_remote[j].val * Arecv[indices_row_wise[k]].val));
            }
        }

    } //for i
//    MPI_Barrier(comm); printf("\n\n rank = %d, loop ends! \n", rank); MPI_Barrier(comm);

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 4-1: rank = %d\n", rank); MPI_Barrier(comm);}

    // todo: check this: since entries of RA_temp with these row indices only exist on this processor,
    // todo: duplicates happen only on this processor, so sorting should be done locally.
    std::sort(RA_temp.entry.begin(), RA_temp.entry.end());

//    printf("rank %d: RA_temp.entry.size_total = %lu \n", rank, RA_temp.entry.size());
//    print_vector(RA_temp.entry, -1, "RA_temp.entry", comm);

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 4-2: rank = %d\n", rank); MPI_Barrier(comm);}

    prolong_matrix RA(comm);
    RA.entry.resize(RA_temp.entry.size());

    // remove duplicates.
    unsigned long entry_size = 0;
    for(nnz_t i=0; i<RA_temp.entry.size(); i++){
        RA.entry[entry_size] = RA_temp.entry[i];
//        if(rank==1) std::cout << RA_temp.entry[i] << std::endl;
        while(i < RA_temp.entry.size()-1 && RA_temp.entry[i] == RA_temp.entry[i+1]){ // values of entries with the same row and col should be added.
            RA.entry[entry_size].val += RA_temp.entry[i+1].val;
            i++;
//            if(rank==1) std::cout << RA_temp.entry[i+1].val << std::endl;
        }
//        if(rank==1) std::cout << std::endl << "final: " << std::endl << RA.entry[RA.entry.size()-1].val << std::endl;
        entry_size++;
        // todo: pruning. don't hard code tol. does this make the matrix non-symmetric?
//        if( abs(RA.entry.back().val) < 1e-6)
//            RA.entry.pop_back();
//        if(rank==1) std::cout << "final: " << std::endl << RA.entry.back().val << std::endl;
    }

    RA_temp.entry.clear();
    RA_temp.entry.shrink_to_fit();
    RA.entry.resize(entry_size);
    RA.entry.shrink_to_fit();

//    print_vector(RA.entry, -1, "RA.entry", comm);

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 4-3: rank = %d\n", rank); MPI_Barrier(comm);}

    // ************************************* RAP_temp - P local *************************************
    // Some local and remote elements of RAP_temp are computed here.
    // Note: P local means whole entries of P on this process, not just the diagonal block.

    prolong_matrix RAP_temp(comm); // RAP_temp is being used to remove duplicates while pushing back to RAP.
    index_t P_max_M;
    MPI_Allreduce(&P->M, &P_max_M, 1, MPI_UNSIGNED, MPI_MAX, comm);
//    MPI_Barrier(comm); printf("rank=%d, PMaxNnz=%d \n", rank, PMaxNnz); MPI_Barrier(comm);

    std::vector<unsigned int> PnnzPerRow(P_max_M, 0);
    for(nnz_t i=0; i<P->nnz_l; i++){
        PnnzPerRow[P->entry[i].row]++;
    }

//    print_vector(PnnzPerRow, -1, "PnnzPerRow", comm);

//    unsigned int* PnnzPerRowScan = (unsigned int*)malloc(sizeof(unsigned int)*(P_max_M+1));
    std::vector<unsigned long> PnnzPerRowScan(P_max_M+1);
    PnnzPerRowScan[0] = 0;
    for(nnz_t i = 0; i < P->M; i++){
        PnnzPerRowScan[i+1] = PnnzPerRowScan[i] + PnnzPerRow[i];
//        if(rank==2) printf("i=%lu, PnnzPerRow=%d, PnnzPerRowScan = %d\n", i, PnnzPerRow[i], PnnzPerRowScan[i]);
    }

    std::fill(&left_block_nnz[0], &left_block_nnz[nprocs], 0);
    if(!RA.entry.empty()){
        for (nnz_t i = 0; i < RA.entry.size(); i++) {
            procNum = lower_bound2(&P->split[0], &P->split[nprocs], RA.entry[i].col);
            left_block_nnz[procNum]++;
//        if(rank==1) printf("rank=%d, col = %lu, procNum = %ld \n", rank, R->entry_remote[0].col, procNum);
        }
    }

    left_block_nnz_scan[0] = 0;
    for(int i = 0; i < nprocs; i++)
        left_block_nnz_scan[i+1] = left_block_nnz_scan[i] + left_block_nnz[i];

//    print_vector(left_block_nnz_scan, -1, "left_block_nnz_scan", comm);

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 4-4: rank = %d\n", rank); MPI_Barrier(comm);}

    // todo: combine indicesP_Prolong and indicesP_ProlongRecv together.
    // find row-wise ordering for A and save it in indicesP
    indices_row_wise.resize(P->nnz_l);
    for(nnz_t i=0; i<P->nnz_l; i++)
        indices_row_wise[i] = i;

    std::sort(&indices_row_wise[0], &indices_row_wise[P->nnz_l], sort_indices2(&*P->entry.begin()));

    for(nnz_t i=left_block_nnz_scan[rank]; i<left_block_nnz_scan[rank+1]; i++){
        for(nnz_t j = PnnzPerRowScan[RA.entry[i].col - P->split[rank]]; j < PnnzPerRowScan[RA.entry[i].col - P->split[rank] + 1]; j++){

//            if(rank==3) std::cout << RA.entry[i].row + P->splitNew[rank] << "\t" << P->entry[indicesP_Prolong[j]].col << "\t" << RA.entry[i].val * P->entry[indicesP_Prolong[j]].val << std::endl;

            RAP_temp.entry.emplace_back(cooEntry(RA.entry[i].row + P->splitNew[rank],  // Ac.entry should have global indices at the end.
                                                 P->entry[indices_row_wise[j]].col,
                                                 RA.entry[i].val * P->entry[indices_row_wise[j]].val));
        }
    }

//    print_vector(RAP_temp.entry, -1, "RAP_temp.entry", comm);

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 5: rank = %d\n", rank); MPI_Barrier(comm);}

    // ************************************* RAP_temp - P remote *************************************

    nnz_t PMaxNnz;
    MPI_Allreduce(&P->nnz_l, &PMaxNnz, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);

    indices_row_wise.resize(PMaxNnz);
    std::vector<cooEntry> Precv(PMaxNnz);
    nnz_t PrecvM;

    for(int i = 1; i < nprocs; i++) {
        // send P to the right processor, receive P from the left processor. "left" decreases by one in each iteration. "right" increases by one.
        right = (rank + i) % nprocs;
        left = rank - i;
        if (left < 0)
            left += nprocs;

        // *************************** RAP_temp - P remote - sendrecv(size) ****************************

        // use sender rank for send and receive tags.
        MPI_Sendrecv(&P->nnz_l, 1, MPI_UNSIGNED_LONG, right, rank, &nnzRecv, 1, MPI_UNSIGNED_LONG, left, left, comm, &sendRecvStatus);
//        int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf,
//                         int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status)

//        printf("i=%d, rank=%d, left=%d, right=%d \n", i, rank, left, right);
//        printf("i=%d, rank = %d own P->nnz_l = %lu    \tnnzRecv = %u \n", i, rank, P->nnz_l, nnzRecv);

        // *************************** RAP_temp - P remote - sendrecv(P) ****************************

        // use sender rank for send and receive tags.
        MPI_Sendrecv(&P->entry[0], P->nnz_l, cooEntry::mpi_datatype(), right, rank,
                     &Precv[0],    nnzRecv,  cooEntry::mpi_datatype(), left,  left, comm, &sendRecvStatus);

//        if(rank==1) for(int j=0; j<P->nnz_l; j++)
//                        printf("j=%d \t %lu \t %lu \t %f \n", j, P->entry[j].row, P->entry[j].col, P->entry[j].val);
//        if(rank==1) for(int j=0; j<nnzRecv; j++)
//                        printf("j=%d \t %lu \t %lu \t %f \n", j, Precv[j].row, Precv[j].col, Precv[j].val);

        // *************************** RAP_temp - P remote - multiplication ****************************

        PrecvM = P->split[left+1] - P->split[left];
        std::fill(&PnnzPerRow[0], &PnnzPerRow[PrecvM], 0);
        for(nnz_t j=0; j<nnzRecv; j++)
            PnnzPerRow[Precv[j].row]++;

//        if(rank==1)
//            for(j=0; j<PrecvM; j++)
//                std::cout << PnnzPerRow[i] << std::endl;

        PnnzPerRowScan[0] = 0;
        for(nnz_t j=0; j<PrecvM; j++){
            PnnzPerRowScan[j+1] = PnnzPerRowScan[j] + PnnzPerRow[j];
//            if(rank==1) printf("j=%lu, PnnzPerRow=%d, PnnzPerRowScan = %d\n", j, PnnzPerRow[j], PnnzPerRowScan[j]);
        }

        // find row-wise ordering for Arecv and save it in indicesPRecv
        for(nnz_t i=0; i<nnzRecv; i++)
            indices_row_wise[i] = i;

        std::sort(&indices_row_wise[0], &indices_row_wise[nnzRecv], sort_indices2(&Precv[0]));

//        if(rank==1) std::cout << "block start = " << RBlockStart[left] << "\tend = " << RBlockStart[left+1] << "\tleft rank = " << left << "\t i = " << i << std::endl;
        if(!RA.entry.empty()) {
            for (nnz_t j = left_block_nnz_scan[left]; j < left_block_nnz_scan[left + 1]; j++) {
//            if(rank==1) std::cout << "col = " << R->entry_remote[j].col << "\tcol-split = " << R->entry_remote[j].col - P->split[left] << "\tstart = " << AnnzPerRowScan[R->entry_remote[j].col - P->split[left]] << "\tend = " << AnnzPerRowScan[R->entry_remote[j].col - P->split[left] + 1] << std::endl;
                for (nnz_t k = PnnzPerRowScan[RA.entry[j].col - P->split[left]];
                     k < PnnzPerRowScan[RA.entry[j].col - P->split[left] + 1]; k++) {
//                if(rank==0) std::cout << Precv[indicesP_ProlongRecv[k]].row << "\t" << Precv[indicesP_ProlongRecv[k]].col << "\t" << Precv[indicesP_ProlongRecv[k]].val << std::endl;
                    RAP_temp.entry.emplace_back(cooEntry(
                            RA.entry[j].row + P->splitNew[rank], // Ac.entry should have global indices at the end.
                            Precv[indices_row_wise[k]].col,
                            RA.entry[j].val * Precv[indices_row_wise[k]].val));
                }
            }
        }

    } //for i

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 6-1: rank = %d\n", rank); MPI_Barrier(comm);}

    std::sort(RAP_temp.entry.begin(), RAP_temp.entry.end());

//    print_vector(RAP_temp.entry, -1, "RAP_temp.entry", comm);

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 6-2: rank = %d\n", rank); MPI_Barrier(comm);}

    Ac->entry.resize(RAP_temp.entry.size());

    // remove duplicates.
//    std::vector<cooEntry> Ac_temp;
    entry_size = 0;
    for(nnz_t i=0; i<RAP_temp.entry.size(); i++){
//        Ac->entry.push_back(RAP_temp.entry[i]);
        Ac->entry[entry_size] = RAP_temp.entry[i];
        while(i<RAP_temp.entry.size()-1 && RAP_temp.entry[i] == RAP_temp.entry[i+1]){ // values of entries with the same row and col should be added.
//            Ac->entry.back().val += RAP_temp.entry[i+1].val;
            Ac->entry[entry_size].val += RAP_temp.entry[i+1].val;
            i++;
        }
        entry_size++;
        // todo: pruning. don't hard code tol. does this make the matrix non-symmetric?
//        if( abs(Ac->entry.back().val) < 1e-6)
//            Ac->entry.pop_back();
    }

//    if(rank==0) printf("rank %d: entry_size = %lu \n", rank, entry_size);

    RAP_temp.entry.clear();
    RAP_temp.entry.shrink_to_fit();
    Ac->entry.resize(entry_size);
    Ac->entry.shrink_to_fit();

//    MPI_Barrier(comm); printf("rank %d: %lu \n", rank, Ac->entry.size()); MPI_Barrier(comm);
//    print_vector(Ac->entry, -1, "Ac->entry", comm);

//    par::sampleSort(Ac_temp, Ac->entry, comm);

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 7: rank = %d\n", rank); MPI_Barrier(comm);}

    Ac->nnz_l = entry_size;
    MPI_Allreduce(&Ac->nnz_l, &Ac->nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
    Ac->Mbig = P->Nbig;
    Ac->M = P->splitNew[rank+1] - P->splitNew[rank];
    Ac->M_old = Ac->M;
    Ac->split = P->splitNew;
    Ac->last_M_shrink = A->last_M_shrink;
    Ac->last_density_shrink = A->last_density_shrink;
//    Ac->last_nnz_shrink = A->last_nnz_shrink;
//    Ac->enable_shrink = A->enable_shrink;
//    Ac->enable_shrink = A->enable_shrink_next_level;
    Ac->comm = A->comm;
    Ac->comm_old = A->comm;
    Ac->active_old_comm = true;
    Ac->density = float(Ac->nnz_g) / (Ac->Mbig * Ac->Mbig);
    Ac->switch_to_dense = switch_to_dense;
    Ac->dense_threshold = dense_threshold;

    Ac->cpu_shrink_thre1 = A->cpu_shrink_thre1; //todo: is this required?
    if(A->cpu_shrink_thre2_next_level != -1) // this is -1 by default.
        Ac->cpu_shrink_thre2 = A->cpu_shrink_thre2_next_level;

    //return these to default, since they have been used in the above part.
    A->cpu_shrink_thre2_next_level = -1;
    A->enable_shrink_next_level = false;

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 8: rank = %d\n", rank); MPI_Barrier(comm);}

//    print_vector(Ac->split, -1, "Ac->split", comm);
    for(index_t i = 0; i < Ac->split.size()-1; i++){
        if(Ac->split[i+1] - Ac->split[i] == 0){
//            printf("rank %d: shrink minor in coarsen: i = %d, split[i] = %d, split[i+1] = %d\n", rank, i, Ac->split[i], Ac->split[i+1]);
            Ac->shrink_cpu_minor();
            break;
        }
    }

//    MPI_Barrier(comm);
//    printf("Ac: rank = %d \tMbig = %u \tM = %u \tnnz_g = %lu \tnnz_l = %lu \tdensity = %f\n",
//           rank, Ac->Mbig, Ac->M, Ac->nnz_g, Ac->nnz_l, Ac->density);
//    MPI_Barrier(comm);

//    if(verbose_coarsen){
//        printf("\nrank = %d, Ac->Mbig = %u, Ac->M = %u, Ac->nnz_l = %lu, Ac->nnz_g = %lu \n", rank, Ac->Mbig, Ac->M, Ac->nnz_l, Ac->nnz_g);}

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 9: rank = %d\n", rank); MPI_Barrier(comm);}


    if(Ac->active_minor){
        comm = Ac->comm;
        int rank_new;
        MPI_Comm_rank(Ac->comm, &rank_new);
//        Ac->print_info(-1);

        // ********** decide about shrinking **********

        if(Ac->enable_shrink && Ac->enable_dummy_matvec && nprocs > 1){
//            MPI_Barrier(Ac->comm); if(rank_new==0) printf("start decide shrinking\n"); MPI_Barrier(Ac->comm);
            Ac->matrix_setup_dummy();
            Ac->compute_matvec_dummy_time();
            Ac->decide_shrinking(A->matvec_dummy_time);
            Ac->erase_after_decide_shrinking();
//            MPI_Barrier(Ac->comm); if(rank_new==0) printf("finish decide shrinking\n"); MPI_Barrier(Ac->comm);
        }

        if(verbose_coarsen){
            MPI_Barrier(comm); printf("coarsen: step 10: rank = %d\n", rank); MPI_Barrier(comm);}

        // ********** setup matrix **********
        // Shrinking gets decided inside repartition_nnz() or repartition_row() functions, then repartition happens.
        // Finally, shrink_cpu() and matrix_setup() are called. In this way, matrix_setup is called only once.

        // decide to partition based on number of rows or nonzeros.
//    if(switch_repartition && Ac->density >= repartition_threshold)
        if(switch_repartition && Ac->density >= repartition_threshold){
            if(rank==0) printf("equi-ROW partition for the next level: density = %f, repartition_threshold = %f \n", Ac->density, repartition_threshold);
            Ac->repartition_row(); // based on number of rows
        }else{
            Ac->repartition_nnz(); // based on number of nonzeros
        }

//        if(Ac->shrinked_minor){
//            repartition_u_shrink_minor_prepare(grid);
//        }

        if(verbose_coarsen){
            MPI_Barrier(comm); printf("coarsen: step 11: rank = %d\n", rank); MPI_Barrier(comm);}

        repartition_u_shrink_prepare(grid);

        if(Ac->shrinked){
            Ac->shrink_cpu();
        }

        if(verbose_coarsen){
            MPI_Barrier(comm); printf("coarsen: step 12: rank = %d\n", rank); MPI_Barrier(comm);}

        if(Ac->active){
            Ac->matrix_setup();

            if(Ac->shrinked && Ac->enable_dummy_matvec)
                Ac->compute_matvec_dummy_time();

            if(switch_to_dense && Ac->density > dense_threshold){
                if(rank==0) printf("Switch to dense: density = %f, dense_threshold = %f \n", Ac->density, dense_threshold);
                Ac->generate_dense_matrix();
            }
        }

//        Ac->print_info(-1);
//        Ac->print_entry(-1);
    }

    comm = grid->A->comm;
    if(verbose_coarsen){MPI_Barrier(comm); printf("end of coarsen: rank = %d\n", rank); MPI_Barrier(comm);}

    return 0;
} // coarsen_old()


int saena_object::transpose_locally(std::vector<cooEntry> &A, nnz_t size){

    transpose_locally(A, size, 0, A);

    return 0;
}


int saena_object::transpose_locally(std::vector<cooEntry> &A, nnz_t size, std::vector<cooEntry> &B){

    transpose_locally(A, size, 0, B);

    return 0;
}


int saena_object::transpose_locally(std::vector<cooEntry> &A, nnz_t size, index_t row_offset, std::vector<cooEntry> &B){

    for(nnz_t i = 0; i < size; i++){
        B[i] = cooEntry(A[i].col, A[i].row+row_offset, A[i].val);
    }

    std::sort(B.begin(), B.end());

    return 0;
}

