#include "saena_object.h"
#include "saena_matrix.h"
#include "strength_matrix.h"
//#include "prolong_matrix.h"
//#include "restrict_matrix.h"
#include "grid.h"
#include "aux_functions.h"
#include "parUtils.h"
#include "dollar.hpp"

#include <cstdio>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <mpi.h>



// remove this
#include <unistd.h>


double case0 = 0, case11 = 0, case12 = 0, case2 = 0, case3 = 0; // for timing case parts of fast_mm

void saena_object::fast_mm(index_t *Ar, value_t *Av, index_t *Ac_scan,
                           index_t *Br, value_t *Bv, index_t *Bc_scan,
                           index_t A_row_size, index_t A_row_offset, index_t A_col_size, index_t A_col_offset,
                           index_t B_col_size, index_t B_col_offset,
                           std::vector<cooEntry> &C, MPI_Comm comm){

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

    nnz_t A_nnz = Ac_scan[A_col_size] - Ac_scan[0];
    nnz_t B_nnz = Bc_scan[B_col_size] - Bc_scan[0];

    // check Split Fact 1
//    index_t A_col_scan_start_tmp = Ac[0];
//    index_t B_col_scan_start_tmp = Bc[0];
//    Ac[0] = A_col_scan_start;
//    Bc[0] = B_col_scan_start;

    index_t B_row_size      = A_col_size;
    index_t B_row_offset    = A_col_offset;
    index_t A_col_size_half = A_col_size/2;
//    index_t B_row_size_half = A_col_size_half;
//    index_t B_col_size_half = B_col_size/2;

    int verbose_rank = 0;
#ifdef __DEBUG1__
//    if(rank==verbose_rank) std::cout << "\n==========================" << __func__ << "==========================\n";
    if(rank==verbose_rank && verbose_fastmm) printf("\nfast_mm: start \n");

//    MPI_Barrier(comm);
    if(rank==verbose_rank){

        if(verbose_matmat_A){
            std::cout << "\nA: nnz = "       << A_nnz
                      << ", A_row_size = "   << A_row_size   << ", A_col_size = "   << A_col_size
                      << ", A_row_offset = " << A_row_offset << ", A_col_offset = " << A_col_offset << std::endl;

//            print_array(Ac_scan, A_col_size+1, 1, "Ac_scan", comm);

            // print entries of A:
            std::cout << "\nA: nnz = " << A_nnz << std::endl;
            for(nnz_t i = 0; i < A_col_size; i++){
                for(nnz_t j = Ac_scan[i]; j < Ac_scan[i+1]; j++) {
                    std::cout << j << "\t" << Ar[j] << "\t" << i + A_col_offset << "\t" << Av[j] << std::endl;
                }
            }
        }

        if(verbose_matmat_B) {
            std::cout << "\nB: nnz = "       << B_nnz;
            std::cout << ", B_row_size = "   << B_row_size   << ", B_col_size = "   << B_col_size
                      << ", B_row_offset = " << B_row_offset << ", B_col_offset = " << B_col_offset << std::endl;

//            print_array(Bc_scan, B_col_size+1, 1, "Bc_scan", comm);

            // print entries of B:
            std::cout << "\nB: nnz = " << B_nnz << std::endl;
            for (nnz_t i = 0; i < B_col_size; i++) {
                for (nnz_t j = Bc_scan[i]; j < Bc_scan[i+1]; j++) {
                    std::cout << j << "\t" << Br[j] << "\t" << i + B_col_offset << "\t" << Bv[j] << std::endl;
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

    // case1
    // ==============================================================

    if (A_row_size * B_col_size < matmat_size_thre1) { //DOLLAR("case0")

#ifdef __DEBUG1__
        if (rank == verbose_rank && (verbose_fastmm || verbose_matmat_recursive)) {
            printf("fast_mm: case 0: start \n");
        }
#endif

//        double t1 = MPI_Wtime();
        double t0 = MPI_Wtime();
        double t01 = MPI_Wtime();

        index_t *nnzPerRow_left = &mempool2[0];
        std::fill(&nnzPerRow_left[0], &nnzPerRow_left[A_row_size], 0);
        index_t *nnzPerRow_left_p = &nnzPerRow_left[0] - A_row_offset;
//        index_t *nnzPerRow_left_p = &nnzPerRow_left[0];

//        std::cout << "\nA_row_offset = " << A_row_offset << std::endl;
        for (nnz_t i = 0; i < A_col_size; i++) {
            for (nnz_t j = Ac_scan[i]; j < Ac_scan[i+1]; j++) {
//                if(rank==1) std::cout << j << "\t" << Ar[j] << "\t" << i + A_col_offset << "\t" << Av[j] << std::endl;
                nnzPerRow_left_p[Ar[j]]++;
            }
        }

        t01 = MPI_Wtime() - t01;

#ifdef __DEBUG1__
//        print_array(nnzPerRow_left, A_row_size, verbose_rank, "nnzPerRow_left", comm);
#endif

        double t02 = MPI_Wtime();

        index_t *A_new_row_idx   = &nnzPerRow_left[0];
        index_t *A_new_row_idx_p = &A_new_row_idx[0] - A_row_offset;
        index_t *orig_row_idx    = &mempool2[A_row_size];
        index_t A_nnz_row_sz     = 0;

        for (index_t i = 0; i < A_row_size; i++) {
            if (A_new_row_idx[i]) {
                A_new_row_idx[i] = A_nnz_row_sz;
                orig_row_idx[A_nnz_row_sz] = i + A_row_offset;
                A_nnz_row_sz++;
            }
        }

        t02 = MPI_Wtime() - t02;

#ifdef __DEBUG1__
//        print_array(orig_row_idx,  A_nnz_row_sz, verbose_rank, "orig_row_idx",  comm);
//        print_array(A_new_row_idx, A_row_size,   verbose_rank, "A_new_row_idx", comm);
#endif

        double t03 = MPI_Wtime();

        index_t *B_new_col_idx   = &mempool2[A_row_size * 2];
//        index_t *B_new_col_idx_p = &B_new_col_idx[0] - B_col_offset;
        index_t *orig_col_idx    = &mempool2[A_row_size * 2 + B_col_size];
        index_t B_nnz_col_sz     = 0;

        for (index_t i = 0; i < B_col_size; i++) {
            if (Bc_scan[i+1] != Bc_scan[i]) {
//                if(rank==0) printf("orig_col_idx index = %u\n", (A_row_size * 2 + B_col_size) + B_nnz_col_sz);
                B_new_col_idx[i] = B_nnz_col_sz;
                orig_col_idx[B_nnz_col_sz] = i + B_col_offset;
                B_nnz_col_sz++;
            }
        }

        t03 = MPI_Wtime() - t03;

//        MPI_Barrier(comm);
//        if(!rank) printf("\n");
//        MPI_Barrier(comm);
//        printf("rank %d: A_row_size = %u, \tA_nnz_row_sz = %u, \tB_col_size = %u, \tB_nnz_col_sz = %u \n",
//               rank, A_row_size, A_nnz_row_sz, B_col_size, B_nnz_col_sz);
//        if(!rank)
//            printf("A_row_size = %u, \tA_nnz = %ld, \tB_col_size = %u, \tB_nnz = %ld\n", A_row_size, A_nnz, B_col_size, B_nnz);
//        MPI_Barrier(comm);




#ifdef __DEBUG1__
//        std::cout << "orig_col_idx max: " << A_row_size * 2 + B_col_size + B_nnz_col_sz - 1 << std::endl;

//        print_array(orig_col_idx,  B_nnz_col_sz, verbose_rank, "B orig_col_idx", comm);
//        print_array(B_new_col_idx, B_col_size,   verbose_rank, "B_new_col_idx",  comm);

//        printf("rank %d: A_row_size = %u, \tA_nnz_row_sz = %u, \tB_col_size = %u, \tB_nnz_col_sz = %u \n",
//            rank, A_row_size, A_nnz_row_sz, B_col_size, B_nnz_col_sz);
#endif

        t0 = MPI_Wtime() - t0;
//        if(!rank) printf("\n");
//        print_time_ave(t0, "case0", comm, true);
        case0 += t0;



        if(rank == 1){
            printf("%f\t%f\t%f\n", t01, t02, t03);
        }



        // check if A_nnz_row_sz * B_nnz_col_sz < matmat_size_thre1, then do dense multiplication. otherwise, do case2 or 3.
        if(A_nnz_row_sz * B_nnz_col_sz < matmat_size_thre2) {

            double t11 = MPI_Wtime();

            // initialize
            value_t *C_temp = &mempool1[0];
//            std::fill(&C_temp[0], &C_temp[A_nnz_row_sz * B_nnz_col_sz], 0);

#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_fastmm) { printf("fast_mm: case 1: step 1 \n"); }
#endif

//            if (rank == verbose_rank) { printf("\nfast_mm: case 1: step 1 \n"); }
//            printf("\n");

            mapbit.reset();
            index_t C_index;
            value_t C_val;
            index_t temp;
//            bool C_not_zero = false;
            const index_t *Ac_p = &Ac_scan[0] - B_row_offset;

            for (nnz_t j = 0; j < B_col_size; ++j) { // columns of B
                temp = A_nnz_row_sz * B_new_col_idx[j];

                for (nnz_t k = Bc_scan[j]; k < Bc_scan[j+1]; ++k) { // nonzeros in column j of B

//                    if(rank==1) std::cout << "\n" << Br[k] << "\t" << Br[k] - B_row_offset
//                                          << "\t" << Ac_p[Br[k]] << "\t" << Ac_p[Br[k]+1] << std::endl;

                    for (nnz_t i = Ac_p[Br[k]]; i < Ac_p[Br[k] + 1]; ++i) { // nonzeros in column (Br[k]) of A

#ifdef __DEBUG1__
//                        std::cout << Ar[i] << "\t" << j+B_col_offset << "\t" << Av[i] << "\t" << Bv[k] << std::endl;

//                        if(Ar[i] == 9 && j+B_col_offset==23)
//                            std::cout << "===========" << Ar[i] << "\t" << Br[k]          << "\t" << Av[i] << "\t"
//                                      << Br[k] << "\t" << j+B_col_offset << "\t" << Bv[k] << std::endl;

//                            if(rank==0) std::cout << B[k].row << "\t" << B[k].row - B_row_offset << "\t" << Ac_p[B[k].row] << std::endl;

//                            if(rank==0) std::cout << Ar[i] << "\t" << A_row_offset << "\t" << Ar[i] - A_row_offset
//                                        << "\t" << A_new_row_idx[Ar[i] - A_row_offset] << "\t" << j << "\t" << B_new_col_idx[j]
//                                        << "\t" << A_new_row_idx[Ar[i] - A_row_offset] + temp
//                                        << "\t" << C_temp[A_new_row_idx[Ar[i] - A_row_offset] + temp]
//                                        << std::endl;

//                            if(rank==0) std::cout << A_new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B_new_col_idx[j] << "\t"
//                                                  << A_new_row_idx[A[i].row - A_row_offset] << "\t" << B_new_col_idx[j] << "\t"
//                                                  << C_temp[A_new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B_new_col_idx[j]] << std::endl;
#endif

//                        C_temp_p[A_new_row_idx_p[A[i].row] + A_nnz_row_sz * B[k].col] += B[k].val * A[i].val;
//                        C_temp[A_new_row_idx_p[A[i].row] + temp] += B[k].val * A[i].val;
//                        C_not_zero = true;

                        C_index = A_new_row_idx_p[Ar[i]] + temp;
                        C_val   = Bv[k] * Av[i];

//                        if(C_index==0) std::cout << C_temp[C_index] << "\t" << C_val << std::endl;
//                        if(rank==0) std::cout << C_index << "\t" << Av[i] << "\t" << Bv[k] << "\t" << C_val << std::endl;

                        if(mapbit[C_index]) {
                            C_temp[C_index] += C_val;
                        } else {
                            C_temp[C_index] = C_val;
                            mapbit[C_index] = true;
                        }

//                        if(C_index==0) std::cout << C_temp[C_index] << std::endl;

#ifdef __DEBUG1__
//                        if(rank==0) std::cout << A_new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B_new_col_idx[j] << "\t"
//                                              << A_new_row_idx[A[i].row - A_row_offset] << "\t" << B_new_col_idx[j] << "\t"
//                                              << C_temp[C_index] << "\t" << C_val << std::endl;

//                        if(rank == 0) std::cout << "A: " << A[i] << "\tB: " << B[k] << "\tC_index: " << A_new_row_idx_p[A[i].row] + temp
//                             << "\tA_row_offset = " << A_row_offset << "\tB_col_offset = " << B_col_offset << std::endl;

//                        if(rank==1 && A[i].row == 0 && B[j].col == 0) std::cout << "A: " << A[i] << "\tB: " << B[j]
//                             << "\tC: " << C_temp[(A[i].row-A_row_offset) + A_row_size * (B[j].col-B_col_offset)]
//                             << "\tA*B: " << B[j].val * A[i].val << std::endl;
#endif
                    }
                }
            }

#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_fastmm) { printf("fast_mm: case 1: step 2 \n"); }
//            print_array(C_temp, A_nnz_row_sz * B_nnz_col_sz, -1, "C_temp", comm);
#endif

            t11 = MPI_Wtime() - t11;
//            print_time_ave(t11, "case11", comm, true);
            case11 += t11;

            // =======================================
            // Extract nonzeros
            // =======================================
            // Go through a dense matrix of size (A_nnz_row_sz * B_nnz_col_sz) and check if mapbit of that entry
            // is true. If so, then extract that entry.

            if(mapbit.count() == 0){
                return;
            }

//            if(rank == 1){
//                printf("Arow = %d,\tBcol = %d,\tAr*Bc = %d,\tmapbit = %ld\n",
//                       A_nnz_row_sz, B_nnz_col_sz, A_nnz_row_sz*B_nnz_col_sz, mapbit.count());
//            }
//            MPI_Barrier(comm);

            double t12 = MPI_Wtime();

//            sleep(1);

//            C.reserve(C.size() + mapbit.count());

            nnz_t temp2;
            if(mapbit.count()){
                for (index_t j = 0; j < B_nnz_col_sz; j++) {
                    temp = A_nnz_row_sz * j;
                    for (index_t i = 0; i < A_nnz_row_sz; i++) {
                        temp2 = i + temp;
                        if(mapbit[temp2]){
//                            if(rank==verbose_rank) std::cout << i << "\t" << j << "\t" << temp2 << "\t" << orig_row_idx[i] << "\t" << orig_col_idx[j] << "\t" << C_temp[i + temp] << std::endl;
                            C.emplace_back(orig_row_idx[i], orig_col_idx[j], C_temp[temp2]);
                        }
                    }
                }
            }

//            t11 = MPI_Wtime() - t11;
            t12 = MPI_Wtime() - t12;
//            print_time_ave(t12, "case12", comm, true);
            case12 += t12;

//            if(rank == 1) printf("%f\t%f\n", t12, case12);

#ifdef __DEBUG1__
//                nnz_t C_nnz = 0; // not required
//                if(C_not_zero) {
//                    for (index_t j = 0; j < B_nnz_col_sz; j++) {
//                        temp = A_nnz_row_sz * j;
//                        for (index_t i = 0; i < A_nnz_row_sz; i++) {
//                            temp2 = i + temp;
//                            if (C_temp[temp2] != 0) {
//                                //if(rank==0) std::cout << i << "\t" << j << "\t" << temp2 << "\t" << orig_row_idx[i] << "\t" << orig_col_idx[j] << "\t" << C_temp[i + temp] << std::endl;
//                                C.emplace_back(orig_row_idx[i], orig_col_idx[j], C_temp[temp2]);
//                                C_nnz++; // not required
//                            }
//                        }
//                    }
//                }

            if (rank == verbose_rank && verbose_fastmm) printf("fast_mm: case 1: end \n");
//                printf("C_nnz = %lu\tA: %u, %u\tB: %u, %u\ttime = %f\t\tvec\n", C_nnz, A_row_size, A_nnz_row_sz,
//                       B_col_size, B_nnz_col_sz, t1);
//                printf("C_nnz: %lu \tA_nnz: %lu \t(%f) \tB_nnz: %lu \t(%f) \tA_row: %u (%u) \tB_col: %u (%u) \tt: %.3f \n",
//                       C_nnz, A_nnz, (double(A_nnz)/A_row_size/A_col_size), B_nnz,
//                       (double(B_nnz)/A_col_size/B_col_size), A_row_size, A_nnz_row_sz, B_col_size, B_nnz_col_sz, t11*1000);
//                print_vector(C, -1, "C", comm);
#endif

//            }

            return;
        }

    }





    //todo: delete this
    MPI_Barrier(comm);
    if(!rank) std::cout << "case2 is being performed! exit!" << std::endl;
    MPI_Barrier(comm);
    exit(EXIT_FAILURE);





    // ==============================================================
    // Case2
    // ==============================================================

    index_t A_col_scan_end, B_col_scan_end;

    // if A_col_size_half == 0, it means A_col_size = 1. In this case it goes to case3.
    if (A_row_size <= A_col_size && A_col_size_half != 0){//DOLLAR("case2")

        double t2 = MPI_Wtime();

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_fastmm) { printf("fast_mm: case 2: start \n"); }
#endif

//        index_t B_row_offset = A_col_offset;
//        index_t A_col_size_half = A_col_size/2;

        // =======================================================
        // split based on matrix size
        // =======================================================

#ifdef SPLIT_SIZE

        // this part is common with part for SPLIT_SIZE, so it is moved to after SPLIT_SIZE.

#endif

        // =======================================================
        // split based on nnz
        // =======================================================

#ifdef SPLIT_NNZ

        // todo: this part is not updated after changing fast_mm().

        // prepare splits of matrix A by column
        auto A_half_nnz = (nnz_t) ceil(A_nnz / 2);
//        index_t A_col_size_half = A_col_size/2;

        if (A_nnz > matmat_nnz_thre) { // otherwise A_col_size_half will stay A_col_size/2
            for (nnz_t i = 0; i < A_col_size; i++) {
                if( (Ac[i+1] - Ac[0]) >= A_half_nnz){
                    A_col_size_half = i;
                    break;
                }
            }
        }

        // if A is not being split at all following "half nnz method", then swtich to "half size method".
        if (A_col_size_half == A_col_size) {
            A_col_size_half = A_col_size / 2;
        }

#endif

//        print_array(Ac_scan, A_col_size+1, 0, "Ac_scan", comm);

        auto A1r = &Ar[0];
        auto A1v = &Av[0];
        auto A2r = &Ar[0];
        auto A2v = &Av[0];

        auto A1c_scan = Ac_scan;
        auto A2c_scan = &Ac_scan[A_col_size_half];

        // Split Fact 1:
        // The last element of A1c_scan is shared with the first element of A2c_scan, and it may gets changed
        // during the recursive calls from the A1c_scan side. So, save that and use it for the starting
        // point of A2c_scan inside the recursive calls.
//        index_t A1_col_scan_start = Ac1[0];
//        index_t A2_col_scan_start = Ac2[0];

        auto A1_row_size = A_row_size;
        auto A2_row_size = A_row_size;

        auto A1_row_offset = A_row_offset;
        auto A2_row_offset = A_row_offset;

        auto A1_col_size = A_col_size_half;
        auto A2_col_size = A_col_size - A1_col_size;

        auto A1_col_offset = A_col_offset;
        auto A2_col_offset = A_col_offset + A1_col_size;

        nnz_t A1_nnz = A1c_scan[A1_col_size] - A1c_scan[0];
        nnz_t A2_nnz = A_nnz - A1_nnz;

//        std::cout << "A1_nnz: " << A1_nnz << ", A2_nnz: " << A2_nnz << ", A_col_size: " << A_col_size
//                  << ", A_col_size_half: " << A_col_size_half << std::endl;

        // =======================================================

        // split B based on how A is split, so use A_col_size_half to split B. A_col_size_half can be different based on
        // choosing the splitting method (nnz or size).
        index_t B_row_size_half = A_col_size_half;
        index_t B_row_threshold = B_row_size_half + B_row_offset;

        auto B1c_scan = Bc_scan; // col_scan
        auto B2c_scan = new index_t[B_col_size + 1]; // col_scan

        reorder_split(Br, Bv, B1c_scan, B2c_scan, B_col_size, B_row_threshold);

        nnz_t B1_nnz = B1c_scan[B_col_size] - B1c_scan[0];
        nnz_t B2_nnz = B2c_scan[B_col_size] - B2c_scan[0];

        auto B1r = &Br[0];
        auto B1v = &Bv[0];
        auto B2r = &Br[B1c_scan[B_col_size]];
        auto B2v = &Bv[B1c_scan[B_col_size]];

        auto B1_row_size = B_row_size_half;
        auto B2_row_size = A_col_size - B1_row_size;

        auto B1_row_offset = B_row_offset;
        auto B2_row_offset = B_row_offset + B_row_size_half;

        auto B1_col_size = B_col_size;
        auto B2_col_size = B_col_size;

        auto B1_col_offset = B_col_offset;
        auto B2_col_offset = B_col_offset;

        t2 = MPI_Wtime() - t2;
//        print_time_ave(t2, "case2", comm, true);
        case2 += t2;

        // Check Split Fact 1
//        index_t B1_col_scan_start = Bc1[0];
//        index_t B2_col_scan_start = Bc2[0];

//        std::cout << "\ncase2: B_nnz: " << B_nnz << "\tB1_nnz: " << B1_nnz << "\tB2_nnz: " << B2_nnz
//                  << "\tB_row_size: " << B_row_size << "\tB_row_size_half: " << B_row_size_half << std::endl;
//
//        std::cout << "\ncase2_part2: B1_row_size: " << B1_row_size << "\tB2_row_size: " << B2_row_size
//                  << "\tB1_row_offset: " << B1_row_offset << "\tB2_row_offset: " << B2_row_offset
//                  << "\tB1_col_size:"  << B1_col_size << "\tB2_col_size: " << B2_col_size
//                  << "\tB1_col_offset: " << B1_col_offset << "\tB2_col_offset: " << B2_col_offset << std::endl;

#ifdef __DEBUG1__
//        print_array(Bc1, B_col_size+1, 0, "Bc1", comm);
//        print_array(Bc2, B_col_size+1, 0, "Bc2", comm);

//        std::cout << "\nB1: nnz: " << B1_nnz << std::endl ;
//        for(index_t j = 0; j < B_col_size; j++){
//            for(index_t i = Bc1[j]; i < Bc1[j+1]; i++){
//                std::cout << std::setprecision(4) << B[i].row << "\t" << j << "\t" << B[i].val << std::endl;
//            }
//        }
//
//        std::cout << "\nB2: nnz: " << B2_nnz << std::endl ;
//        for(index_t j = 0; j < B_col_size; j++){
//            for(index_t i = Bc2[j]+B1_nnz; i < Bc2[j+1]+B1_nnz; i++){
//                std::cout << std::setprecision(4) << B[i].row << "\t" << j << "\t" << B[i].val << std::endl;
//            }
//        }


#endif

        // A1: start: nnzPerColScan_leftStart,                  end: nnzPerColScan_leftEnd
        // A2: start: nnzPerColScan_leftStart[A_col_size_half], end: nnzPerColScan_leftEnd[A_col_size_half]
        // B1: start: nnzPerColScan_rightStart,                 end: nnzPerColScan_middle
        // B2: start: nnzPerColScan_middle,                     end: nnzPerColScan_rightEnd

#ifdef __DEBUG1__
//        MPI_Barrier(comm);
        if (rank == verbose_rank) {

//        printf("fast_mm: case 2: \nA_nnz: (%lu, %lu, %lu), B_nnz: (%lu, %lu, %lu)\n"
//               "A_size: (%u, %u, %u), B_size: (%u, %u) \n",
//               A_nnz, A1_nnz, A2_nnz, B_nnz, B1_nnz, B2_nnz, A_row_size, A_col_size, A_col_size_half, A_col_size, B_col_size);

            if (verbose_matmat_A) {
//                std::cout << "\nranges of A:" << std::endl;
//                for (nnz_t i = 0; i < A_col_size; i++) {
//                    std::cout << i << "\t" << Ac[i] << "\t" << Ac[i + 1]
//                              << std::endl;
//                }
//
//                std::cout << "\nranges of A1:" << std::endl;
//                for (nnz_t i = 0; i < A_col_size / 2; i++) {
//                    std::cout << i << "\t" << Ac[i] << "\t" << Ac[i + 1]
//                              << std::endl;
//                }
//
//                std::cout << "\nranges of A2:" << std::endl;
//                for (nnz_t i = 0; i < A_col_size - A_col_size / 2; i++) {
//                    std::cout << i << "\t" << Ac[A_col_size / 2 + i]
//                              << "\t" << Ac[A_col_size / 2 + i + 1] << std::endl;
//                }

                // print entries of A1:
                std::cout << "\nCase2:\nA1: nnz = " << A1_nnz << std::endl;
                for (nnz_t i = 0; i < A1_col_size; i++) {
                    for (nnz_t j = A1c_scan[i]; j < A1c_scan[i + 1]; j++) {
                        std::cout << j << "\t" << A1r[j] << "\t" << i + A1_col_offset << "\t" << A1v[j] << std::endl;
                    }
                }

                // print entries of A2:
                std::cout << "\nA2: nnz = " << A2_nnz << std::endl;
                for (nnz_t i = 0; i < A2_col_size; i++) {
                    for (nnz_t j = A2c_scan[i]; j < A2c_scan[i + 1]; j++) {
                        std::cout << j << "\t" << A2r[j] << "\t" << i + A2_col_offset << "\t" << A2v[j] << std::endl;
                    }
                }
            }

            if (verbose_matmat_B) {
//                std::cout << "\nranges of B, B1, B2::" << std::endl;
//                for (nnz_t i = 0; i < B_col_size; i++) {
//                    std::cout << i << "\t" << Bc[i] << "\t" << Bc[i + 1]
//                              << "\t" << Bc1[i] << "\t" << Bc1[i + 1]
//                              << "\t" << Bc2[i] << "\t" << Bc2[i + 1]
//                              << std::endl;
//                }

                // print entries of B1:
                std::cout << "\nCase2:\nB1: nnz = " << B1_nnz << std::endl;
                for (nnz_t i = 0; i < B1_col_size; i++) {
                    for (nnz_t j = B1c_scan[i]; j < B1c_scan[i + 1]; j++) {
                        std::cout << j << "\t" << B1r[j] << "\t" << i + B1_col_offset << "\t" << B1v[j] << std::endl;
                    }
                }

                // print entries of B2:
                std::cout << "\nB2: nnz = " << B2_nnz << std::endl;
                for (nnz_t i = 0; i < B2_col_size; i++) {
                    for (nnz_t j = B2c_scan[i]; j < B2c_scan[i + 1]; j++) {
                        std::cout << j << "\t" << B2r[j] << "\t" << i + B2_col_offset << "\t" << B2v[j] << std::endl;
                    }
                }
            }
        }
//        print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);
//        MPI_Barrier(comm);
#endif

        // =======================================================
        // Call two recursive functions here. Put the result of the first one in C1, and the second one in C2.
        // merge sort them and add the result to C.
//        std::vector<cooEntry> C_temp;

        // C1 = A1 * B1
#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 1 \n");
#endif

        if (A1_nnz == 0 || B1_nnz == 0) { // skip!
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat_recursive) {
                if (A1_nnz == 0) {
                    printf("skip: A1_nnz == 0\n");
                } else {
                    printf("skip: B1_nnz == 0\n");
                }
            }
#endif
        } else {

            // Check Split Fact 1 for this part.
            A_col_scan_end = A1c_scan[A1_col_size];
            B_col_scan_end = B1c_scan[B1_col_size];

//            auto A1r_temp      = new index_t[A1_nnz];
//            auto A1v_temp      = new value_t[A1_nnz];
//            auto A1c_scan_temp = new index_t[A1_col_size + 1];
//
//            memcpy(A1r_temp,      A1r,      sizeof(index_t) * A1_nnz);
//            memcpy(A1v_temp,      A1v,      sizeof(value_t) * A1_nnz);
//            memcpy(A1c_scan_temp, A1c_scan, sizeof(index_t) * (A1_col_size + 1));
//
//            auto B1r_temp      = new index_t[B1_nnz];
//            auto B1v_temp      = new value_t[B1_nnz];
//            auto B1c_scan_temp = new index_t[B1_col_size + 1];
//
//            memcpy(B1r_temp,      B1r,      sizeof(index_t) * B1_nnz);
//            memcpy(B1v_temp,      B1v,      sizeof(value_t) * B1_nnz);
//            memcpy(B1c_scan_temp, B1c_scan, sizeof(index_t) * (B1_col_size + 1));

            fast_mm(&A1r[0], &A1v[0], &A1c_scan[0],
                    &B1r[0], &B1v[0], &B1c_scan[0],
                    A1_row_size, A1_row_offset, A1_col_size, A1_col_offset,
                    B1_col_size, B1_col_offset,
                    C, comm);

            A1c_scan[A1_col_size] = A_col_scan_end;
            B1c_scan[B1_col_size] = B_col_scan_end;

//            memcpy(A1r,      A1r_temp,      sizeof(index_t) * A1_nnz);
//            memcpy(A1v,      A1v_temp,      sizeof(value_t) * A1_nnz);
//            memcpy(A1c_scan, A1c_scan_temp, sizeof(index_t) * (A1_col_size + 1));
//
//            delete []A1r_temp;
//            delete []A1v_temp;
//            delete []A1c_scan_temp;
//
//            memcpy(B1r,      B1r_temp,      sizeof(index_t) * B1_nnz);
//            memcpy(B1v,      B1v_temp,      sizeof(value_t) * B1_nnz);
//            memcpy(B1c_scan, B1c_scan_temp, sizeof(index_t) * (B1_col_size + 1));
//
//            delete []B1r_temp;
//            delete []B1v_temp;
//            delete []B1c_scan_temp;
        }

        // C2 = A2 * B2
#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 2 \n");
#endif

        if (A2_nnz == 0 || B2_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat_recursive) {
                if (A2_nnz == 0) {
                    printf("skip: A2_nnz == 0\n");
                } else {
                    printf("skip: B2_nnz == 0\n");
                }
            }
#endif
        } else {

            A_col_scan_end = A2c_scan[A2_col_size];
            B_col_scan_end = B2c_scan[B2_col_size];

//            auto A2r_temp      = new index_t[A2_nnz];
//            auto A2v_temp      = new value_t[A2_nnz];
//            auto A2c_scan_temp = new index_t[A2_col_size + 1];
//
//            memcpy(A2r_temp,      A2r,      sizeof(index_t) * A2_nnz);
//            memcpy(A2v_temp,      A2v,      sizeof(value_t) * A2_nnz);
//            memcpy(A2c_scan_temp, A2c_scan, sizeof(index_t) * (A2_col_size + 1));
//
//            auto B2r_temp      = new index_t[B2_nnz];
//            auto B2v_temp      = new value_t[B2_nnz];
//            auto B2c_scan_temp = new index_t[B2_col_size + 1];
//
//            memcpy(B2r_temp,      B2r,      sizeof(index_t) * B2_nnz);
//            memcpy(B2v_temp,      B2v,      sizeof(value_t) * B2_nnz);
//            memcpy(B2c_scan_temp, B2c_scan, sizeof(index_t) * (B2_col_size + 1));

            fast_mm(&A2r[0], &A2v[0], &A2c_scan[0],
                    &B2r[0], &B2v[0], &B2c_scan[0],
                    A2_row_size, A2_row_offset, A2_col_size, A2_col_offset,
                    B2_col_size, B2_col_offset,
                    C, comm);

            A2c_scan[A2_col_size] = A_col_scan_end;
            B2c_scan[B2_col_size] = B_col_scan_end;

//            memcpy(A2r,      A2r_temp,      sizeof(index_t) * A2_nnz);
//            memcpy(A2v,      A2v_temp,      sizeof(value_t) * A2_nnz);
//            memcpy(A2c_scan, A2c_scan_temp, sizeof(index_t) * (A2_col_size + 1));
//
//            delete []A2r_temp;
//            delete []A2v_temp;
//            delete []A2c_scan_temp;
//
//            memcpy(B2r,      B2r_temp,      sizeof(index_t) * B2_nnz);
//            memcpy(B2v,      B2v_temp,      sizeof(value_t) * B2_nnz);
//            memcpy(B2c_scan, B2c_scan_temp, sizeof(index_t) * (B2_col_size + 1));
//
//            delete []B2r_temp;
//            delete []B2v_temp;
//            delete []B2c_scan_temp;

//        fast_mm(&A[0], &B[0], C, A2_nnz, B2_nnz,
//                A_row_size, A_row_offset, A_col_size - A_col_size_half, A_col_offset + A_col_size_half,
//                B_col_size, B_col_offset,
//                &nnzPerColScan_leftStart[A_col_size_half], &nnzPerColScan_leftEnd[A_col_size_half], // A2
//                &nnzPerColScan_middle[0], nnzPerColScan_rightEnd, comm); // B2
        }

        t2 = MPI_Wtime();

        // return B to its original order.
        reorder_back_split(Br, Bv, B1c_scan, B2c_scan, B_col_size);
        delete []B2c_scan;

        t2 = MPI_Wtime() - t2;
        case2 += t2;

#ifdef __DEBUG1__
//        print_vector(C1, -1, "C1", comm);
//        print_vector(C2, -1, "C2", comm);

//        if(rank==verbose_rank && verbose_fastmm) {printf("fast_mm: case 2: step 3 \n");}
//        if(rank==0 && verbose_fastmm) printf("C1.size() = %lu, C2.size() = %lu \n", C1.size(), C2.size());
#endif

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_fastmm) printf("fast_mm: case 2: end \n");
#endif

        return; // end of case 2 and fast_mm()

    }


    // ==============================================================
    // case3
    // ==============================================================

    { //DOLLAR("case3") // (A_row_size > A_col_size)

        double t3 = MPI_Wtime();

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_fastmm) printf("fast_mm: case 3: start \n");
#endif

        // split based on matrix size
        // =======================================================

#ifdef SPLIT_SIZE
        // prepare splits of matrix B by column

//        index_t A_col_size_half = A_col_size/2;
        index_t B_col_size_half = B_col_size/2;

        nnz_t B1_nnz = Bc_scan[B_col_size_half] - Bc_scan[0];
        nnz_t B2_nnz = B_nnz - B1_nnz;

        auto B1r = &Br[0];
        auto B1v = &Bv[0];
        auto B2r = &Br[0];
        auto B2v = &Bv[0];

        auto B1c_scan = Bc_scan;
        auto B2c_scan = &Bc_scan[B_col_size_half];

        auto B1_row_size = A_col_size;
        auto B2_row_size = A_col_size;

        auto B1_row_offset = B_row_offset;
        auto B2_row_offset = B_row_offset;

        auto B1_col_size = B_col_size_half;
        auto B2_col_size = B_col_size - B1_col_size;

        auto B1_col_offset = B_col_offset;
        auto B2_col_offset = B_col_offset + B_col_size_half;

//        std::cout << "\ncase2: B_nnz: " << B_nnz << "\tB1_nnz: " << B1_nnz << "\tB2_nnz: " << B2_nnz
//                  << "\tB_col_size: " << B_col_size << "\tB_col_size_half: " << B_col_size_half << std::endl;
//
//        std::cout << "\ncase2_part2: B_row_size: " << B_row_size << "\tB_row_offset: " << B_row_offset
//                  << "\tB1_col_size:"  << B1_col_size << "\tB2_col_size: " << B2_col_size
//                  << "\tB1_col_offset: " << B1_col_offset << "\tB2_col_offset: " << B2_col_offset << std::endl;
#endif

#ifdef __DEBUG1__
//        print_array(Bc, B_col_size+1, 0, "Bc", comm);
//
//        std::cout << "\nB1: nnz: " << B1_nnz << std::endl ;
//        for(index_t j = 0; j < B_col_size_half; j++){
//            for(index_t i = Bc[j]; i < Bc[j+1]; i++){
//                std::cout << i << "\t" << B[i].row << "\t" << j << "\t" << B[i].val << std::endl;
//            }
//        }
//
//        std::cout << "\nB2: nnz: " << B2_nnz << std::endl ;
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
        nnz_t B1_nnz = 0, B2_nnz;
        auto B_half_nnz = (nnz_t) ceil(B_nnz / 2);
        index_t B_col_size_half = B_col_size / 2;

        if (B_nnz > matmat_nnz_thre) {
            for (nnz_t i = 0; i < B_col_size; i++) {
                B1_nnz += nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i];

#ifdef __DEBUG1__
//                if(rank==verbose_rank)
//                    printf("B_nnz = %lu, B_half_nnz = %lu, B1_nnz = %lu, nnz on col %u: %u \n",
//                           B_nnz, B_half_nnz, B1_nnz, B[nnzPerColScan_rightStart[i]].col,
//                           nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i]);
#endif

                if (B1_nnz >= B_half_nnz) {
                    B_col_size_half = B[nnzPerColScan_rightStart[i]].col + 1 - B_col_offset;
                    break;
                }
            }
        } else {
            for (nnz_t i = 0; i < B_col_size_half; i++) {
                B1_nnz += nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i];
            }
        }

        // if B is not being splitted at all following "half nnz method", then swtich to "half col method".
        if (B_col_size_half == B_col_size) {
            B_col_size_half = B_col_size / 2;
            B1_nnz = 0;
            for (nnz_t i = 0; i < B_col_size_half; i++) {
                B1_nnz += nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i];
            }
        }

        B2_nnz = B_nnz - B1_nnz;
#endif

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_fastmm) printf("fast_mm: case 3: step 1 \n");
#endif

        // prepare splits of matrix A by row

        index_t A_row_size_half = A_row_size / 2;
        index_t A_row_threshold = A_row_size_half + A_row_offset;

        auto A1c_scan = Ac_scan; // col_scan
        auto A2c_scan = new index_t[A_col_size + 1]; // col_scan

        reorder_split(Ar, Av, A1c_scan, A2c_scan, A_col_size, A_row_threshold);

        nnz_t A1_nnz = A1c_scan[A_col_size] - A1c_scan[0];
        nnz_t A2_nnz = A2c_scan[A_col_size] - A2c_scan[0];

        auto A1r = &Ar[0];
        auto A1v = &Av[0];
        auto A2r = &Ar[A1c_scan[A_col_size]];
        auto A2v = &Av[A1c_scan[A_col_size]];

        auto A1_row_size = A_row_size_half;
        auto A2_row_size = A_row_size - A1_row_size;

        auto A1_row_offset = A_row_offset;
        auto A2_row_offset = A_row_offset + A1_row_size;

        auto A1_col_size = A_col_size;
        auto A2_col_size = A_col_size;

        auto A1_col_offset = A_col_offset;
        auto A2_col_offset = A_col_offset;

//        std::cout << "\ncase3: A_nnz: " << A_nnz << "\tA1_nnz: " << A1_nnz << "\tA2_nnz: " << A2_nnz
//                  << "\tA_row_size: " << A_row_size << "\tA_row_size_half: " << A_row_size_half
//                  << "\tA_row_threshold: " << A_row_threshold << std::endl;
//
//        std::cout << "\ncase3_part2: A1_row_size: " << A1_row_size << "\tA2_row_size: " << A2_row_size
//                  << "\tA1_row_offset: " << A1_row_offset << "\tA2_row_offset: " << A2_row_offset
//                  << "\tA1_col_size: " << A1_col_size << "\tA2_col_size: " << A2_col_size
//                  << "\tA1_col_offset: " << A1_col_offset << "\tA2_col_offset: " << A2_col_offset << std::endl;

    t3 = MPI_Wtime() - t3;
    case3 += t3;

#ifdef __DEBUG1__
//        print_array(Ac1, A_col_size+1, 0, "Ac1", comm);
//        print_array(Ac2, A_col_size+1, 0, "Ac2", comm);

//        std::cout << "\nCase3:\nA1: nnz: " << A1_nnz << std::endl ;
//        for(index_t j = 0; j < A1_col_size; j++){
//            for(index_t i = Ac1[j]; i < Ac1[j+1]; i++){
//                std::cout << std::setprecision(4) << A1[i].row << "\t" << j + A1_col_offset << "\t" << A1[i].val << std::endl;
//            }
//        }
//        std::cout << "\nA2: nnz: " << A2_nnz << std::endl ;
//        for(index_t j = 0; j < A2_col_size; j++){
//            for(index_t i = Ac2[j]; i < Ac2[j+1]; i++){
//                std::cout << std::setprecision(4) << A2[i].row << "\t" << j + A2_col_offset << "\t" << A2[i].val << std::endl;
//            }
//        }
#endif

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_fastmm) printf("fast_mm: case 3: step 2 \n");
#endif

        // A1: start: nnzPerColScan_leftStart,                   end: nnzPerColScan_middle
        // A2: start: nnzPerColScan_middle,                      end: nnzPerColScan_leftEnd
        // B1: start: nnzPerColScan_rightStart,                  end: nnzPerColScan_rightEnd
        // B2: start: nnzPerColScan_rightStart[B_col_size_half], end: nnzPerColScan_rightEnd[B_col_size_half]

#ifdef __DEBUG1__
//        MPI_Barrier(comm);
        if (rank == verbose_rank) {

//            printf("fast_mm: case 3: \nA_nnz: (%lu, %lu, %lu), B_nnz: (%lu, %lu, %lu)\n"
//                   "A_size: (%u, %u), B_size: (%u, %u, %u) \n",
//                   A_nnz, A1_nnz, A2_nnz, B_nnz, B1_nnz, B2_nnz, A_row_size, A_col_size, A_col_size, B_col_size, B_col_size_half);

            if (verbose_matmat_A) {
                // print entries of A1:
                std::cout << "\nCase3:\nA1: nnz = " << A1_nnz << std::endl;
                for (nnz_t i = 0; i < A1_col_size; i++) {
                    for (nnz_t j = A1c_scan[i]; j < A1c_scan[i + 1]; j++) {
                        std::cout << j << "\t" << A1r[j] << "\t" << i + A1_col_offset << "\t" << A1v[j] << std::endl;
                    }
                }

                // print entries of A2:
                std::cout << "\nA2: nnz = " << A2_nnz << std::endl;
                for (nnz_t i = 0; i < A2_col_size; i++) {
                    for (nnz_t j = A2c_scan[i]; j < A2c_scan[i + 1]; j++) {
                        std::cout << j << "\t" << A2r[j] << "\t" << i + A2_col_offset << "\t" << A2v[j] << std::endl;
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
                std::cout << "\nCase3:\nB1: nnz = " << B1_nnz << std::endl;
                for (nnz_t i = 0; i < B1_col_size; i++) {
                    for (nnz_t j = B1c_scan[i]; j < B1c_scan[i + 1]; j++) {
                        std::cout << j << "\t" << B1r[j] << "\t" << i + B1_col_offset << "\t" << B1v[j] << std::endl;
                    }
                }

                // print entries of B2:
                std::cout << "\nB2: nnz = " << B2_nnz << std::endl;
                for (nnz_t i = 0; i < B2_col_size; i++) {
                    for (nnz_t j = B2c_scan[i]; j < B2c_scan[i + 1]; j++) {
                        std::cout << j << "\t" << B2r[j] << "\t" << i + B2_col_offset << "\t" << B2v[j] << std::endl;
                    }
                }
            }
        }
//        MPI_Barrier(comm);
#endif

        // =======================================================

        // C1 = A1 * B1
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 1 \n");
#endif

        if (A1_nnz == 0 || B1_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat_recursive) {
                if (A1_nnz == 0) {
                    printf("skip: A1_nnz == 0\n");
                } else {
                    printf("skip: B1_nnz == 0\n");
                }
            }
#endif
        } else {

            A_col_scan_end = A1c_scan[A1_col_size];
            B_col_scan_end = B1c_scan[B1_col_size];

//            auto Ar_temp      = new index_t[A_nnz];
//            auto Av_temp      = new value_t[A_nnz];
//            auto Ac_scan_temp = new index_t[A_col_size + 1];
//
//            memcpy(Ar_temp,      Ar,      sizeof(index_t) * A_nnz);
//            memcpy(Av_temp,      Av,      sizeof(value_t) * A_nnz);
//            memcpy(Ac_scan_temp, Ac_scan, sizeof(index_t) * (A_col_size + 1));
//
//            auto Br_temp      = new index_t[B_nnz];
//            auto Bv_temp      = new value_t[B_nnz];
//            auto Bc_scan_temp = new index_t[B_col_size + 1];
//
//            memcpy(Br_temp,      Br,      sizeof(index_t) * B_nnz);
//            memcpy(Bv_temp,      Bv,      sizeof(value_t) * B_nnz);
//            memcpy(Bc_scan_temp, Bc_scan, sizeof(index_t) * (B_col_size + 1));

            fast_mm(&A1r[0], &A1v[0], &A1c_scan[0],
                    &B1r[0], &B1v[0], &B1c_scan[0],
                    A1_row_size, A1_row_offset, A1_col_size, A1_col_offset,
                    B1_col_size, B1_col_offset,
                    C, comm);

            A1c_scan[A1_col_size] = A_col_scan_end;
            B1c_scan[B1_col_size] = B_col_scan_end;

//            memcpy(Ar,      Ar_temp,      sizeof(index_t) * A_nnz);
//            memcpy(Av,      Av_temp,      sizeof(value_t) * A_nnz);
//            memcpy(Ac_scan, Ac_scan_temp, sizeof(index_t) * (A_col_size + 1));
//
//            delete []Ar_temp;
//            delete []Av_temp;
//            delete []Ac_scan_temp;
//
//            memcpy(Br,      Br_temp,      sizeof(index_t) * B_nnz);
//            memcpy(Bv,      Bv_temp,      sizeof(value_t) * B_nnz);
//            memcpy(Bc_scan, Bc_scan_temp, sizeof(index_t) * (B_col_size + 1));
//
//            delete []Br_temp;
//            delete []Bv_temp;
//            delete []Bc_scan_temp;

        }


        // C2 = A1 * B2:
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 2 \n");
#endif

        if (A1_nnz == 0 || B2_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat_recursive) {
                if (A1_nnz == 0) {
                    printf("skip: A1_nnz == 0\n");
                } else {
                    printf("skip: B2_nnz == 0\n");
                }
            }
#endif
        } else {

            A_col_scan_end = A1c_scan[A1_col_size];
            B_col_scan_end = B2c_scan[B2_col_size];

//            auto Ar_temp      = new index_t[A_nnz];
//            auto Av_temp      = new value_t[A_nnz];
//            auto Ac_scan_temp = new index_t[A_col_size + 1];
//
//            memcpy(Ar_temp,      Ar,      sizeof(index_t) * A_nnz);
//            memcpy(Av_temp,      Av,      sizeof(value_t) * A_nnz);
//            memcpy(Ac_scan_temp, Ac_scan, sizeof(index_t) * (A_col_size + 1));
//
//            auto Br_temp      = new index_t[B_nnz];
//            auto Bv_temp      = new value_t[B_nnz];
//            auto Bc_scan_temp = new index_t[B_col_size + 1];
//
//            memcpy(Br_temp,      Br,      sizeof(index_t) * B_nnz);
//            memcpy(Bv_temp,      Bv,      sizeof(value_t) * B_nnz);
//            memcpy(Bc_scan_temp, Bc_scan, sizeof(index_t) * (B_col_size + 1));

            fast_mm(&A1r[0], &A1v[0], &A1c_scan[0],
                    &B2r[0], &B2v[0], &B2c_scan[0],
                    A1_row_size, A1_row_offset, A1_col_size, A1_col_offset,
                    B2_col_size, B2_col_offset,
                    C, comm);

//            fast_mm(&A1[0], &B2[0], C,
//                    A1_row_size, A1_row_offset, A1_col_size, A1_col_offset,
//                    B2_col_size, B2_col_offset,
//                    &Ac1[0], &Bc2[0], comm);

            A1c_scan[A1_col_size] = A_col_scan_end;
            B2c_scan[B2_col_size] = B_col_scan_end;

//            memcpy(Ar,      Ar_temp,      sizeof(index_t) * A_nnz);
//            memcpy(Av,      Av_temp,      sizeof(value_t) * A_nnz);
//            memcpy(Ac_scan, Ac_scan_temp, sizeof(index_t) * (A_col_size + 1));
//
//            delete []Ar_temp;
//            delete []Av_temp;
//            delete []Ac_scan_temp;
//
//            memcpy(Br,      Br_temp,      sizeof(index_t) * B_nnz);
//            memcpy(Bv,      Bv_temp,      sizeof(value_t) * B_nnz);
//            memcpy(Bc_scan, Bc_scan_temp, sizeof(index_t) * (B_col_size + 1));
//
//            delete []Br_temp;
//            delete []Bv_temp;
//            delete []Bc_scan_temp;
        }


        // C3 = A2 * B1
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 3 \n");
#endif

        if (A2_nnz == 0 || B1_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat_recursive) {
                if (A2_nnz == 0) {
                    printf("skip: A2_nnz == 0\n");
                } else {
                    printf("skip: B1_nnz == 0\n");
                }
            }
#endif
        } else {

            A_col_scan_end = A2c_scan[A2_col_size];
            B_col_scan_end = B1c_scan[B1_col_size];

//            auto Ar_temp      = new index_t[A_nnz];
//            auto Av_temp      = new value_t[A_nnz];
//            auto Ac_scan_temp = new index_t[A_col_size + 1];
//
//            memcpy(Ar_temp,      Ar,      sizeof(index_t) * A_nnz);
//            memcpy(Av_temp,      Av,      sizeof(value_t) * A_nnz);
//            memcpy(Ac_scan_temp, Ac_scan, sizeof(index_t) * (A_col_size + 1));
//
//            auto Br_temp      = new index_t[B_nnz];
//            auto Bv_temp      = new value_t[B_nnz];
//            auto Bc_scan_temp = new index_t[B_col_size + 1];
//
//            memcpy(Br_temp,      Br,      sizeof(index_t) * B_nnz);
//            memcpy(Bv_temp,      Bv,      sizeof(value_t) * B_nnz);
//            memcpy(Bc_scan_temp, Bc_scan, sizeof(index_t) * (B_col_size + 1));

            fast_mm(&A2r[0], &A2v[0], &A2c_scan[0],
                    &B1r[0], &B1v[0], &B1c_scan[0],
                    A2_row_size, A2_row_offset, A2_col_size, A2_col_offset,
                    B1_col_size, B1_col_offset,
                    C, comm);

//            fast_mm(&A2[0], &B1[0], C,
//                    A2_row_size, A2_row_offset, A2_col_size, A2_col_offset,
//                    B1_col_size, B1_col_offset,
//                    &Ac2[0], &Bc1[0], comm);

            A2c_scan[A2_col_size] = A_col_scan_end;
            B1c_scan[B1_col_size] = B_col_scan_end;

//            memcpy(Ar,      Ar_temp,      sizeof(index_t) * A_nnz);
//            memcpy(Av,      Av_temp,      sizeof(value_t) * A_nnz);
//            memcpy(Ac_scan, Ac_scan_temp, sizeof(index_t) * (A_col_size + 1));
//
//            delete []Ar_temp;
//            delete []Av_temp;
//            delete []Ac_scan_temp;
//
//            memcpy(Br,      Br_temp,      sizeof(index_t) * B_nnz);
//            memcpy(Bv,      Bv_temp,      sizeof(value_t) * B_nnz);
//            memcpy(Bc_scan, Bc_scan_temp, sizeof(index_t) * (B_col_size + 1));
//
//            delete []Br_temp;
//            delete []Bv_temp;
//            delete []Bc_scan_temp;

        }


        // C4 = A2 * B2
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 4 \n");
#endif

        if (A2_nnz == 0 || B2_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat_recursive) {
                if (A2_nnz == 0) {
                    printf("skip: A2_nnz == 0\n");
                } else {
                    printf("skip: B2_nnz == 0\n");
                }
            }
#endif
        } else {

            A_col_scan_end = A2c_scan[A2_col_size];
            B_col_scan_end = B2c_scan[B2_col_size];

//            auto A2r_temp      = new index_t[A2_nnz];
//            auto A2v_temp      = new value_t[A2_nnz];
//            auto A2c_scan_temp = new index_t[A2_col_size + 1];
//
//            memcpy(A2r_temp,      A2r,      sizeof(index_t) * A2_nnz);
//            memcpy(A2v_temp,      A2v,      sizeof(value_t) * A2_nnz);
//            memcpy(A2c_scan_temp, A2c_scan, sizeof(index_t) * (A2_col_size + 1));
//
//            auto B2r_temp      = new index_t[B2_nnz];
//            auto B2v_temp      = new value_t[B2_nnz];
//            auto B2c_scan_temp = new index_t[B2_col_size + 1];
//
//            memcpy(B2r_temp,      B2r,      sizeof(index_t) * B2_nnz);
//            memcpy(B2v_temp,      B2v,      sizeof(value_t) * B2_nnz);
//            memcpy(B2c_scan_temp, B2c_scan, sizeof(index_t) * (B2_col_size + 1));

            fast_mm(&A2r[0], &A2v[0], &A2c_scan[0],
                    &B2r[0], &B2v[0], &B2c_scan[0],
                    A2_row_size, A2_row_offset, A2_col_size, A2_col_offset,
                    B2_col_size, B2_col_offset,
                    C, comm);

//            fast_mm(&A2[0], &B2[0], C,
//                    A2_row_size, A2_row_offset, A2_col_size, A2_col_offset,
//                    B2_col_size, B2_col_offset,
//                    &Ac2[0], &Bc2[0], comm);

            A2c_scan[A2_col_size] = A_col_scan_end;
            B2c_scan[B2_col_size] = B_col_scan_end;

//            memcpy(A2r,      A2r_temp,      sizeof(index_t) * A2_nnz);
//            memcpy(A2v,      A2v_temp,      sizeof(value_t) * A2_nnz);
//            memcpy(A2c_scan, A2c_scan_temp, sizeof(index_t) * (A2_col_size + 1));
//
//            delete []A2r_temp;
//            delete []A2v_temp;
//            delete []A2c_scan_temp;
//
//            memcpy(B2r,      B2r_temp,      sizeof(index_t) * B2_nnz);
//            memcpy(B2v,      B2v_temp,      sizeof(value_t) * B2_nnz);
//            memcpy(B2c_scan, B2c_scan_temp, sizeof(index_t) * (B2_col_size + 1));
//
//            delete []B2r_temp;
//            delete []B2v_temp;
//            delete []B2c_scan_temp;
        }

        t3 = MPI_Wtime();

        // return A to its original order.
        reorder_back_split(Ar, Av, A1c_scan, A2c_scan, A_col_size);
        delete []A2c_scan;

        t3 = MPI_Wtime() - t3;
        case3 += t3;

        // C1 = A1 * B1:
//        fast_mm(A1, B1, C_temp, A_row_size_half, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size_half, B_col_offset, comm);
        // C2 = A2 * B1:
//        fast_mm(A2, B1, C_temp, A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset, B_row_offset, B_col_size_half, B_col_offset, comm);
        // C3 = A1 * B2:
//        fast_mm(A1, B2, C_temp, A_row_size_half, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size-B_col_size_half, B_col_offset+B_col_size_half, comm);
        // C4 = A2 * B2
//        fast_mm(A2, B2, C_temp, A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset, B_row_offset, B_col_size-B_col_size_half, B_col_offset+B_col_size_half, comm);

//        if(rank==0 && verbose_fastmm) printf("fast_mm: case 3: step 4 \n");

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_fastmm) printf("fast_mm: case 3: end \n");
#endif

    }

//    Ac[0] = A_col_scan_start_tmp;
//    Bc[0] = B_col_scan_start_tmp;

//    return;
}

void saena_object::fast_mm_basic(const cooEntry *A, const cooEntry *B, std::vector<cooEntry> &C,
                                 nnz_t A_nnz, nnz_t B_nnz,
                                 index_t A_row_size, index_t A_row_offset, index_t A_col_size, index_t A_col_offset,
                                 index_t B_col_size, index_t B_col_offset,
                                 const index_t *nnzPerColScan_leftStart, const index_t *nnzPerColScan_leftEnd,
                                 const index_t *nnzPerColScan_rightStart, const index_t *nnzPerColScan_rightEnd,
                                 MPI_Comm comm){

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

    index_t B_row_offset = A_col_offset;
    index_t A_col_size_half = A_col_size/2;
//    index_t B_row_size_half = A_col_size_half;
//    index_t B_col_size_half = B_col_size/2;

    int verbose_rank = 0;
#ifdef __DEBUG1__
    if(rank==verbose_rank && verbose_fastmm) printf("\nfast_mm: start \n");
#endif

//    if(A_nnz == 0 || B_nnz == 0){
//#ifdef __DEBUG1__
//        if(rank==verbose_rank && verbose_fastmm) printf("\nskip: A_nnz == 0 || B_nnz == 0\n\n");
//#endif
//        return 0;
//    }

#ifdef __DEBUG1__
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


    // case1
    // ==============================================================


#ifdef __DEBUG1__
    if (rank == verbose_rank && (verbose_fastmm || verbose_matmat_recursive)) {
        printf("fast_mm: case 1: start \n");
    }
#endif

    std::unordered_map<index_t, value_t> map_matmat;
    map_matmat.reserve(A_nnz + 2*B_nnz);

    index_t C_index;
    value_t C_val;
    const index_t *nnzPerColScan_leftStart_p = &nnzPerColScan_leftStart[0] - B_row_offset;
    const index_t *nnzPerColScan_leftEnd_p = &nnzPerColScan_leftEnd[0] - B_row_offset;
    for (nnz_t j = 0; j < B_col_size; j++) { // columns of B
        for (nnz_t k = nnzPerColScan_rightStart[j]; k < nnzPerColScan_rightEnd[j]; k++) { // nonzeros in column j of B
            for (nnz_t i = nnzPerColScan_leftStart_p[B[k].row]; i < nnzPerColScan_leftEnd_p[B[k].row]; i++) { // nonzeros in column B[k].row of A

                C_index = (A[i].row - A_row_offset) + A_row_size * (B[k].col - B_col_offset);
                C_val = B[k].val * A[i].val;
                auto it = map_matmat.emplace(C_index, C_val);
                if (!it.second) it.first->second += C_val;

            }
        }
    }

    C.reserve(C.size() + map_matmat.size());
    for (auto it1 = map_matmat.begin(); it1 != map_matmat.end(); ++it1) {
//        std::cout << it1->first.first << "\t" << it1->first.second << "\t" << it1->second << std::endl;
        C.emplace_back( (it1->first % A_row_size) + A_row_offset, (it1->first / A_row_size) + B_col_offset, it1->second);
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
        Acsc.row[i] = A->entry[i].row;
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

    index_t A_row_size = A->M;
//    index_t B_col_size = B->Mbig; // for original B
//    index_t B_col_size = B->M;      // for when tranpose of B is used to do the multiplication.

    mempool1 = new value_t[matmat_size_thre2];
    mempool2 = new index_t[2 * A_row_size + 2 * Bcsc.max_M];

    // 2 for both send and receive buffer, valbyidx for value, (B->M_max + 1) for col_scan
    // r_cscan_buffer_sz_max is for both row and col_scan which have the same type.
    int   valbyidx              = sizeof(value_t) / sizeof(index_t);
    nnz_t v_buffer_sz_max       = valbyidx * B->nnz_max;
    nnz_t r_cscan_buffer_sz_max = B->nnz_max + B->M_max + 1;
    nnz_t send_size_max         = v_buffer_sz_max + r_cscan_buffer_sz_max;
    mempool3              = new index_t[2 * send_size_max];

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

    matmat(Acsc, Bcsc, *C, send_size_max);

    if(assemble){
        matmat_assemble(A, B, C);
    }

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

    delete[] mempool1;
    delete[] mempool2;
    delete[] mempool3;

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
    C->density         = float(C->nnz_g) / (C->Mbig * C->Mbig);
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
        Acsc.row[i] = A->entry[i].row;
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

    index_t A_row_size = A->M;
//    index_t B_col_size = B->Mbig; // for original B
//    index_t B_col_size = B->M;      // for when tranpose of B is used to do the multiplication.

    mempool1 = new value_t[matmat_size_thre2];
    mempool2 = new index_t[2 * A_row_size + 2 * Bcsc.max_M];

    // 2 for both send and receive buffer, valbyidx for value, (B->M_max + 1) for col_scan
    // r_cscan_buffer_sz_max is for both row and col_scan which have the same type.
    int   valbyidx              = sizeof(value_t) / sizeof(index_t);
    nnz_t v_buffer_sz_max       = valbyidx * B->nnz_max;
    nnz_t r_cscan_buffer_sz_max = B->nnz_max + B->M_max + 1;
    nnz_t send_size_max         = v_buffer_sz_max + r_cscan_buffer_sz_max;
    mempool3              = new index_t[2 * send_size_max];

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

    delete[] mempool1;
    delete[] mempool2;
    delete[] mempool3;

    return 0;
}

// older version
//int saena_object::matmat_ave(saena_matrix *A, saena_matrix *B, double &matmat_time)
/*
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
        Acsc.row[i] = A->entry[i].row;
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

    index_t A_row_size = A->M;
//    index_t B_col_size = B->Mbig; // for original B
//    index_t B_col_size = B->M;      // for when tranpose of B is used to do the multiplication.

    mempool1 = new value_t[matmat_size_thre2];
    mempool2 = new index_t[2 * A_row_size + 2 * Bcsc.max_M];

    // 2 for both send and receive buffer, valbyidx for value, (B->M_max + 1) for col_scan
    // r_cscan_buffer_sz_max is for both row and col_scan which have the same type.
    int valbyidx                = sizeof(value_t) / sizeof(index_t);
    nnz_t v_buffer_sz_max       = valbyidx * B->nnz_max;
    nnz_t r_cscan_buffer_sz_max = B->nnz_max + B->M_max + 1;
    nnz_t send_size_max         = v_buffer_sz_max + r_cscan_buffer_sz_max;
    mempool3                    = new index_t[2 * send_size_max];

//    mempool1 = std::make_unique<value_t[]>(matmat_size_thre2);
//    mempool2 = std::make_unique<index_t[]>(A->Mbig * 4);

#ifdef __DEBUG1__
//    if(rank==0) std::cout << "valbyidx = " << valbyidx << std::endl;

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

    delete[] mempool1;
    delete[] mempool2;
    delete[] mempool3;

    return 0;
}
*/

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

    return 0;
}

//original
/*
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

    std::vector<cooEntry> AB_temp;

    if(nprocs > 1){

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
        index_t mat_recv_M, mat_current_M;
//        auto mat_recv = &mempool3[send_size_max];

        auto mat_recv = &mempool3[send_size_max];
//        auto mat_recv_rv    = reinterpret_cast<vecEntry*>(&mat_recv[0]);
//        auto mat_recv_cscan = &mat_recv[rv_buffer_sz_max];
//        auto mat_recv_cscan = reinterpret_cast<index_t*>(&mat_recv[rv_buffer_sz_max]);

        auto mat_temp = mat_send;
        int  owner, next_owner;
        auto *requests = new MPI_Request[2];
        auto *statuses = new MPI_Status[2];

//        std::vector<cooEntry> AB_temp_no_dup;
//        std::vector<nnz_t> AB_nnz_start(nprocs), AB_nnz_end(nprocs);

        // todo: the last communication means each proc receives a copy of its already owned B, which is redundant,
        //   so the communication should be avoided but the multiplication should be done. consider this:
        //   change to k < rank+nprocs-1. Then, copy fast_mm after the end of the for loop to perform the last multiplication
        for(int k = rank; k < rank+nprocs; k++){
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
#endif

            // communicate data
            MPI_Irecv(&mat_recv[0], recv_size, MPI_UNSIGNED, right_neighbor, right_neighbor, comm, requests);
            MPI_Isend(&mat_send[0], send_size, MPI_UNSIGNED, left_neighbor,  rank,           comm, requests+1);

//            owner = k%nprocs;
//            mat_recv_M = B->split[owner + 1] - B->split[owner];

            if(Acsc.nnz == 0 || send_nnz==0){ // skip!
#ifdef __DEBUG1__
                if(verbose_fastmm){
                    if(Acsc.nnz == 0){
                        printf("\nskip: A->entry.size() == 0\n\n");
                    } else {
                        printf("\nskip: send_nnz == 0\n\n");
                    }
                }
#endif
            } else {

                // =======================================
                // perform the multiplication
                // =======================================

                owner         = k%nprocs;
                mat_current_M = Bcsc.split[owner + 1] - Bcsc.split[owner];

//                AB_temp.clear();
                fast_mm(&Acsc.row[0],   &Acsc.val[0],   &Acsc.col_scan[0],
                        &mat_send_r[0], &mat_send_v[0], &mat_send_cscan[0],
                        Acsc_M, Acsc.split[rank], Acsc.col_sz, 0, mat_current_M, Bcsc.split[owner],
                        AB_temp, comm);

                // =======================================
                // sort and remove duplicates
                // =======================================

//                std::sort(AB_temp.begin(), AB_temp.end());

//                print_vector(AB_temp, -1, "AB_temp", comm);

//                AB_nnz_start[owner] = AB_temp_no_dup.size();

//                if(!AB_temp.empty()) {
//                    nnz_t AP_temp_size_minus1 = AB_temp.size() - 1;
//                    for (nnz_t i = 0; i < AB_temp.size(); i++) {
//                        AB_temp_no_dup.emplace_back(AB_temp[i]);
//                        while (i < AP_temp_size_minus1 && AB_temp[i] == AB_temp[i + 1]) { // values of entries with the same row and col should be added.
//                            AB_temp_no_dup.back().val += AB_temp[++i].val;
//                        }
//                    }
//                }

//                AB_nnz_end[owner] = AB_temp_no_dup.size();
            }

            MPI_Waitall(2, requests, statuses);

#ifdef __DEBUG1__
            if (verbose_matmat) {
                MPI_Barrier(comm);
                if (rank == verbose_rank) printf("matmat: step 4 - in for loop\n");
                MPI_Barrier(comm);
            }

//            auto mat_recv_cscan = &mat_recv[rv_buffer_sz_max];
//            MPI_Barrier(comm);
//            if(rank==0){
//                print_array(mat_recv_cscan, mat_recv_M+1, 0, "mat_recv_cscan", comm);
//            }
//            MPI_Barrier(comm);

//            if(rank==1) {
//                for (nnz_t i = 0; i < mat_recv_M; i++) {
//                    for (nnz_t j = mat_send_cscan[i]; j < mat_send_cscan[i + 1]; j++) {
//                        std::cout << j << "\t" << mat_send_r[j] << "\t" << i + Bcsc.split[k % nprocs] << "\t"
//                                  << mat_send_v[j] << std::endl;
//                    }
//                }
//            }
#endif

            send_size = recv_size;
            send_nnz  = recv_nnz;

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

//            MPI_Barrier(comm);
//            if(rank==1) {
//                for (nnz_t i = 0; i < mat_recv_M; i++) {
//                    for (nnz_t j = mat_send_cscan[i]; j < mat_send_cscan[i + 1]; j++) {
//                        std::cout << j << "\t" << mat_send_r[j] << "\t" << i + Bcsc.split[k % nprocs] << "\t"
//                                  << mat_send_v[j] << std::endl;
//                    }
//                }
//            }

//            MPI_Barrier(comm);
//            if(rank==0){
//                std::cout << "print received matrix: mat_recv_M: " << mat_recv_M << ", col_offset: "
//                          << B->split[k%nprocs] << std::endl;

//                print_array(mat_send_cscan, mat_recv_M+1, 0, "mat_send_cscan", comm);
//            }
//            MPI_Barrier(comm);

//          print_vector(AB_temp, -1, "AB_temp", A->comm);
//          print_vector(mat_send, -1, "mat_send", A->comm);
//          print_vector(mat_recv, -1, "mat_recv", A->comm);
//          prev_owner = owner;
//          printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
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

        if(Acsc.nnz == 0 || send_nnz == 0){ // skip!
#ifdef __DEBUG1__
            if(verbose_fastmm){
                if(Acsc.nnz == 0){
                    printf("\nskip: A->entry.size() == 0\n\n");
                } else {
                    printf("\nskip: send_nnz == 0\n\n");
                }
            }
#endif
        } else {

//            index_t mat_recv_M = B->split[rank + 1] - B->split[rank];

//            double t1 = MPI_Wtime();

            fast_mm(&Acsc.row[0], &Acsc.val[0], &Acsc.col_scan[0],
                    &Bcsc.row[0], &Bcsc.val[0], &Bcsc.col_scan[0],
                    Acsc_M, Acsc.split[rank], Acsc.col_sz, 0, Bcsc.col_sz, Bcsc.split[rank],
                    AB_temp, comm);

//            double t2 = MPI_Wtime();
//            printf("\nfast_mm of AB_temp = %f\n", t2-t1);

#ifdef __DEBUG1__
            if (verbose_matmat) {
                if (rank == verbose_rank) printf("matmat: step 2 serial\n");
            }
//            print_vector(AB_temp, -1, "AB_temp", comm);
#endif

            // =======================================
            // sort and remove duplicates
            // =======================================

//            if(!AB_temp.empty()) {
//
//                std::sort(AB_temp.begin(), AB_temp.end());
//
//                nnz_t AP_temp_size_minus1 = AB_temp.size() - 1;
//                for (nnz_t i = 0; i < AB_temp.size(); i++) {
//                    C.entry.emplace_back(AB_temp[i]);
//                    while (i < AP_temp_size_minus1 && AB_temp[i] == AB_temp[i + 1]) { // values of entries with the same row and col should be added.
//                    std::cout << AB_temp[i] << "\t" << AB_temp[i+1] << std::endl;
//                        C.entry.back().val += AB_temp[++i].val;
//                    }
//                }
//
//            }
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
*/

int saena_object::matmat(CSCMat &Acsc, CSCMat &Bcsc, saena_matrix &C, nnz_t send_size_max){

    MPI_Comm comm = C.comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(nprocs == 1){
        printf("this experimental version of matmat does not work in serial!");
        exit(EXIT_FAILURE);
    }

    double t1 = MPI_Wtime();

    int valbyidx = sizeof(value_t) / sizeof(index_t);

    nnz_t send_nnz  = Bcsc.nnz;
    nnz_t send_size = (valbyidx + 1) * send_nnz + Bcsc.col_sz + 1; // valbyidx for val, 1 for row, Bcsc.col_sz + 1 for c_scan

    index_t Acsc_M = Acsc.split[rank+1] - Acsc.split[rank];

    std::vector<cooEntry> AB_temp;

    auto mat_send       = &mempool3[0];
    auto mat_send_r     = &mat_send[0];
    auto mat_send_cscan = &mat_send[send_nnz];
    auto mat_send_v     = reinterpret_cast<value_t*>(&mat_send[send_nnz + Bcsc.col_sz + 1]);

    memcpy(mat_send_r,     Bcsc.row,      Bcsc.nnz * sizeof(index_t));
    memcpy(mat_send_cscan, Bcsc.col_scan, (Bcsc.col_sz + 1) * sizeof(index_t));
    memcpy(mat_send_v,     Bcsc.val,      Bcsc.nnz * sizeof(value_t));

    int right_neighbor = (rank + 1)%nprocs;
    int left_neighbor  = rank - 1;
    if (left_neighbor < 0){
        left_neighbor += nprocs;
    }

    nnz_t recv_nnz;
    nnz_t recv_size;
    index_t mat_recv_M, mat_current_M;

    auto mat_recv = &mempool3[send_size_max];

    auto mat_temp = mat_send;
    int  owner, next_owner;
    auto *requests = new MPI_Request[2];
    auto *statuses = new MPI_Status[2];

    t1 = MPI_Wtime() - t1;

    double t2 = MPI_Wtime();
    double tf, tf_tot = 0; // for timing fast_mm

    for(int k = rank; k < rank+nprocs; k++){
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

        // communicate data
        MPI_Irecv(&mat_recv[0], recv_size, MPI_UNSIGNED, right_neighbor, right_neighbor, comm, requests);
        MPI_Isend(&mat_send[0], send_size, MPI_UNSIGNED, left_neighbor,  rank,           comm, requests+1);

        tf = MPI_Wtime();

        if(Acsc.nnz != 0 && send_nnz != 0){
            owner         = k%nprocs;
            mat_current_M = Bcsc.split[owner + 1] - Bcsc.split[owner];

            fast_mm(&Acsc.row[0],   &Acsc.val[0],   &Acsc.col_scan[0],
                    &mat_send_r[0], &mat_send_v[0], &mat_send_cscan[0],
                    Acsc_M, Acsc.split[rank], Acsc.col_sz, 0, mat_current_M, Bcsc.split[owner],
                    AB_temp, comm);
        }

        tf = MPI_Wtime() - tf;
        tf_tot += tf;

        MPI_Waitall(2, requests, statuses);

        send_size = recv_size;
        send_nnz  = recv_nnz;

        mat_temp = mat_send;
        mat_send = mat_recv;
        mat_recv = mat_temp;

        mat_send_r     = &mat_send[0];
        mat_send_cscan = &mat_send[send_nnz];
        mat_send_v     = reinterpret_cast<value_t*>(&mat_send[send_nnz + mat_recv_M + 1]);
    }

    delete [] requests;
    delete [] statuses;

    t2 = MPI_Wtime() - t2;

    // =======================================
    // sort and remove duplicates
    // =======================================

    double t3 = MPI_Wtime();

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

    t3 = MPI_Wtime() - t3;

//    if (!rank) printf("\nprepare\nfast_mm\ncomm\nsort\n\n");
//    print_time_ave(t1,          "prepare", comm, true);
//    print_time_ave(tf_tot,      "fast_mm", comm, true);
//    print_time_ave(t2 - tf_tot, "comm",    comm, true);
//    print_time_ave(t3,          "sort",    comm, true);

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