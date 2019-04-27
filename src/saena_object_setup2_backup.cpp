// Version before switching to CSC matrix format for A and B. here r and v are stored together.
/*
void saena_object::fast_mm(vecEntry *A, vecEntry *B, std::vector<cooEntry> &C,
                           index_t A_row_size, index_t A_row_offset, index_t A_col_size, index_t A_col_offset,
                           index_t B_col_size, index_t B_col_offset,
                           index_t *Ac, index_t *Bc, MPI_Comm comm){

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

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    nnz_t A_nnz = Ac[A_col_size] - Ac[0];
    nnz_t B_nnz = Bc[B_col_size] - Bc[0];

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
    if(rank==verbose_rank && verbose_matmat) printf("\nfast_mm: start \n");

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
                for(nnz_t j = Ac[i]; j < Ac[i+1]; j++) {
                    std::cout << j << "\t" << A[j].row << "\t" << i + A_col_offset << "\t" << A[j].val << std::endl;
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
                for (nnz_t j = Bc[i]; j < Bc[i+1]; j++) {
                    std::cout << j << "\t" << B[j].row << "\t" << i + B_col_offset << "\t" << B[j].val << std::endl;
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
        if (rank == verbose_rank && (verbose_matmat || verbose_matmat_recursive)) {
            printf("fast_mm: case 0: start \n");
        }
#endif

//        double t1 = MPI_Wtime();

        index_t *nnzPerRow_left = &mempool2[0];
        std::fill(&nnzPerRow_left[0], &nnzPerRow_left[A_row_size], 0);
        index_t *nnzPerRow_left_p = &nnzPerRow_left[0] - A_row_offset;

//        std::cout << "\nA_row_offset = " << A_row_offset << std::endl;
        for (nnz_t i = 0; i < A_col_size; i++) {
            for (nnz_t j = Ac[i]; j < Ac[i+1]; j++) {
//                std::cout << i << "\t" << A[j].row << "\t" << A[j].row - A_row_offset << std::endl;
                nnzPerRow_left_p[A[j].row]++;
            }
        }

#ifdef __DEBUG1__
//        print_array(nnzPerRow_left, A_row_size, 0, "nnzPerRow_left", comm);
#endif

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

#ifdef __DEBUG1__
//        print_array(orig_row_idx, A_nnz_row_sz, 0, "orig_row_idx", comm);
//        print_array(A_new_row_idx, A_row_size, 0, "A_new_row_idx", comm);
#endif

        index_t *B_new_col_idx   = &mempool2[A_row_size * 2];
//        index_t *B_new_col_idx_p = &B_new_col_idx[0] - B_col_offset;
        index_t *orig_col_idx    = &mempool2[A_row_size * 2 + B_col_size];
        index_t B_nnz_col_sz     = 0;

        for (index_t i = 0; i < B_col_size; i++) {
            if (Bc[i+1] != Bc[i]) {
                B_new_col_idx[i] = B_nnz_col_sz;
                orig_col_idx[B_nnz_col_sz] = i + B_col_offset;
                B_nnz_col_sz++;
            }
        }

#ifdef __DEBUG1__
//        print_array(orig_col_idx, B_nnz_col_sz, 0, "orig_col_idx", comm);
//        print_array(B_new_col_idx, B_col_size, 0, "B_new_col_idx", comm);

//        printf("A_row_size = %u, \tA_nnz_row_sz = %u, \tB_col_size = %u, \tB_nnz_col_sz = %u \n",
//            A_row_size, A_nnz_row_sz, B_col_size, B_nnz_col_sz);
#endif



        // check if A_nnz_row_sz * B_nnz_col_sz < matmat_size_thre1, then do dense multiplication. otherwise, do case2 or 3.
        if(A_nnz_row_sz * B_nnz_col_sz < matmat_size_thre2) {

            // initialize
            value_t *C_temp = &mempool1[0];
//                std::fill(&C_temp[0], &C_temp[A_nnz_row_sz * B_nnz_col_sz], 0);

#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) { printf("fast_mm: case 1: step 1 \n"); }
#endif

//                if (rank == verbose_rank) { printf("\nfast_mm: case 1: step 1 \n"); }

            mapbit.reset();
            index_t C_index;
            value_t C_val;
            index_t temp;
//                bool C_not_zero = false;
            const index_t *Ac_p = &Ac[0] - B_row_offset;

            for (nnz_t j = 0; j < B_col_size; j++) { // columns of B

                for (nnz_t k = Bc[j]; k < Bc[j+1]; k++) { // nonzeros in column j of B

                    temp = A_nnz_row_sz * B_new_col_idx[j];

//                        if(rank==0) std::cout << B[k].row << "\t" << B[k].row - B_row_offset
//                                              << "\t" << Ac_p[B[k].row] << "\t" << Ac_p[B[k].row+1] << std::endl;

                    for (nnz_t i = Ac_p[B[k].row]; i < Ac_p[B[k].row + 1]; i++) { // nonzeros in column (B[k].row) of A

#ifdef __DEBUG1__
//                            if(rank==0) std::cout << B[k].row << "\t" << B[k].row - B_row_offset << "\t" << Ac_p[B[k].row] << std::endl;

//                            if(rank==0) std::cout << A[i].row << "\t" << A_row_offset << "\t" << A[i].row - A_row_offset
//                                        << "\t" << A_new_row_idx[A[i].row - A_row_offset] << "\t" << j << "\t" << B_new_col_idx[j]
//                                        << "\t" << A_new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B_new_col_idx[j]
//                                        << "\t" << C_temp[A_new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B_new_col_idx[j]]
//                                        << std::endl;

//                            if(rank==0) std::cout << A_new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B_new_col_idx[j] << "\t"
//                                                  << A_new_row_idx[A[i].row - A_row_offset] << "\t" << B_new_col_idx[j] << "\t"
//                                                  << C_temp[A_new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B_new_col_idx[j]] << std::endl;

//                            if(rank==0) std::cout << B[k].val << "\t" << A[i].val << std::endl;
#endif

//                            C_temp_p[A_new_row_idx_p[A[i].row] + A_nnz_row_sz * B[k].col] += B[k].val * A[i].val;
//                            C_temp[A_new_row_idx_p[A[i].row] + temp] += B[k].val * A[i].val;
//                            C_not_zero = true;

                        C_index = A_new_row_idx_p[A[i].row] + temp;
                        C_val = B[k].val * A[i].val;

//                            std::cout << C_index << "\t" << C_val << std::endl;
                        if(mapbit[C_index]) {
                            C_temp[C_index] += C_val;
                        } else {
                            C_temp[C_index] = C_val;
                            mapbit[C_index] = true;
                        }

#ifdef __DEBUG1__
//                            if(rank==0) std::cout << A_new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B_new_col_idx[j] << "\t"
//                                                  << A_new_row_idx[A[i].row - A_row_offset] << "\t" << B_new_col_idx[j] << "\t"
//                                                  << C_temp[C_index] << "\t" << C_val << std::endl;

//                            if(rank == 0) std::cout << "A: " << A[i] << "\tB: " << B[k] << "\tC_index: " << A_new_row_idx_p[A[i].row] + temp
//                                 << "\tA_row_offset = " << A_row_offset << "\tB_col_offset = " << B_col_offset << std::endl;

//                            if(rank==1 && A[i].row == 0 && B[j].col == 0) std::cout << "A: " << A[i] << "\tB: " << B[j]
//                                 << "\tC: " << C_temp[(A[i].row-A_row_offset) + A_row_size * (B[j].col-B_col_offset)]
//                                 << "\tA*B: " << B[j].val * A[i].val << std::endl;
#endif
                    }
                }
            }

#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) { printf("fast_mm: case 1: step 2 \n"); }
//                print_vector(C_temp, -1, "C_temp", comm);
#endif

            // =======================================
            // Extract nonzeros
            // =======================================

            nnz_t temp2;
            if(mapbit.count()){
                for (index_t j = 0; j < B_nnz_col_sz; j++) {
                    temp = A_nnz_row_sz * j;
                    for (index_t i = 0; i < A_nnz_row_sz; i++) {
                        temp2 = i + temp;
                        if(mapbit[temp2]){
//                                if(rank==0) std::cout << i << "\t" << j << "\t" << temp2 << "\t" << orig_row_idx[i] << "\t" << orig_col_idx[j] << "\t" << C_temp[i + temp] << std::endl;
                            C.emplace_back(orig_row_idx[i], orig_col_idx[j], C_temp[temp2]);
                        }
                    }
                }
            }

//                t11 = MPI_Wtime() - t11;

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

            if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 1: end \n");
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
    // ==============================================================
    // Case2
    // ==============================================================

    if (A_row_size <= A_col_size) { //DOLLAR("case2")

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat) { printf("fast_mm: case 2: start \n"); }
#endif

//        index_t B_row_offset = A_col_offset;
//        index_t A_col_size_half = A_col_size/2;

        // =======================================================
        // split based on matrix size
        // =======================================================

#ifdef SPLIT_SIZE

        // this part is common with part for SPLIT_SIZE, so it moved after SPLIT_SIZE.

#endif

        // =======================================================
        // split based on nnz
        // =======================================================

#ifdef SPLIT_NNZ

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

        auto A1 = &A[0];
        auto A2 = &A[0];

        auto Ac1 = Ac;
        auto Ac2 = &Ac[A_col_size_half];

        // Split Fact 1:
        // The last element of Ac1 is shared with the first element of Ac2, and it may gets changed
        // during the recursive calls from the Ac1 side. So, save that and use it for the starting
        // point of Ac2 inside the recursive calls.
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

        nnz_t A1_nnz = Ac1[A1_col_size + 1] - Ac1[0];
        nnz_t A2_nnz = A_nnz - A1_nnz;

        // =======================================================

        // split B based on how A is split, so use A_col_size_half to split B. A_col_size_half is different based on
        // choosing the splitting method (nnz or size).
        index_t B_row_size_half = A_col_size_half;
        index_t B_row_threshold = B_row_size_half + B_row_offset;

        auto Bc1 = Bc; // col_idx
        auto Bc2 = new index_t[B_col_size + 1]; // col_idx

        reorder_split(B, Bc1, Bc2, B_col_size, B_row_threshold);

        nnz_t B1_nnz = Bc1[B_col_size] - Bc1[0];
        nnz_t B2_nnz = Bc2[B_col_size] - Bc2[0];

        auto B1 = &B[0];
        auto B2 = &B[Bc1[B_col_size]];

        auto B1_row_size = B_row_size_half;
        auto B2_row_size = A_col_size - B1_row_size;

        auto B1_row_offset = B_row_offset;
        auto B2_row_offset = B_row_offset + B_row_size_half;

        auto B1_col_size = B_col_size;
        auto B2_col_size = B_col_size;

        auto B1_col_offset = B_col_offset;
        auto B2_col_offset = B_col_offset;

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
                    for (nnz_t j = Ac1[i]; j < Ac1[i + 1]; j++) {
                        std::cout << j << "\t" << A1[j].row << "\t" << i + A1_col_offset << "\t" << A1[j].val << std::endl;
                    }
                }

                // print entries of A2:
                std::cout << "\nA2: nnz = " << A2_nnz << std::endl;
                for (nnz_t i = 0; i < A2_col_size; i++) {
                    for (nnz_t j = Ac2[i]; j < Ac2[i + 1]; j++) {
                        std::cout << j << "\t" << A2[j].row << "\t" << i + A2_col_offset << "\t" << A2[j].val << std::endl;
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
                    for (nnz_t j = Bc1[i]; j < Bc1[i + 1]; j++) {
                        std::cout << j << "\t" << B1[j].row << "\t" << i + B1_col_offset << "\t" << B1[j].val << std::endl;
                    }
                }

                // print entries of B2:
                std::cout << "\nB2: nnz = " << B2_nnz << std::endl;
                for (nnz_t i = 0; i < B2_col_size; i++) {
                    for (nnz_t j = Bc2[i]; j < Bc2[i + 1]; j++) {
                        std::cout << j << "\t" << B2[j].row << "\t" << i + B2_col_offset << "\t" << B2[j].val << std::endl;
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
            if (rank == verbose_rank && verbose_matmat) {
                if (A1_nnz == 0) {
                    printf("\nskip: A1_nnz == 0\n\n");
                } else {
                    printf("\nskip: B1_nnz == 0\n\n");
                }
            }
#endif
        } else {

            index_t A_col_scan_end = Ac1[A1_col_size];
            index_t B_col_scan_end = Bc1[B1_col_size];

            fast_mm(&A1[0], &B1[0], C,
                    A1_row_size, A1_row_offset, A1_col_size, A1_col_offset,
                    B1_col_size, B1_col_offset,
                    &Ac1[0], &Bc1[0], comm);

            Ac1[A1_col_size] = A_col_scan_end;
            Bc1[B1_col_size] = B_col_scan_end;

        }


        // C2 = A2 * B2
#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 2 \n");
#endif

        if (A2_nnz == 0 || B2_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A2_nnz == 0) {
                    printf("\nskip: A2_nnz == 0\n\n");
                } else {
                    printf("\nskip: B2_nnz == 0\n\n");
                }
            }
#endif
        } else {

            index_t A_col_scan_end = Ac2[A2_col_size];
            index_t B_col_scan_end = Bc2[B2_col_size];

            fast_mm(&A2[0], &B2[0], C,
                    A2_row_size, A2_row_offset, A2_col_size, A2_col_offset,
                    B2_col_size, B2_col_offset,
                    &Ac2[0], &Bc2[0], comm);

            Ac2[A2_col_size] = A_col_scan_end;
            Bc2[B2_col_size] = B_col_scan_end;

//        fast_mm(&A[0], &B[0], C, A2_nnz, B2_nnz,
//                A_row_size, A_row_offset, A_col_size - A_col_size_half, A_col_offset + A_col_size_half,
//                B_col_size, B_col_offset,
//                &nnzPerColScan_leftStart[A_col_size_half], &nnzPerColScan_leftEnd[A_col_size_half], // A2
//                &nnzPerColScan_middle[0], nnzPerColScan_rightEnd, comm); // B2
        }

        delete []Bc2;

#ifdef __DEBUG1__
//        print_vector(C1, -1, "C1", comm);
//        print_vector(C2, -1, "C2", comm);

//        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 3 \n");}
//        if(rank==0 && verbose_matmat) printf("C1.size() = %lu, C2.size() = %lu \n", C1.size(), C2.size());
#endif

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 2: end \n");
#endif

    } else { //DOLLAR("case3") // (A_row_size > A_col_size)

        // ==============================================================
        // case3
        // ==============================================================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 3: start \n");
#endif

        // split based on matrix size
        // =======================================================

#ifdef SPLIT_SIZE
        // prepare splits of matrix B by column

//        index_t A_col_size_half = A_col_size/2;
        index_t B_col_size_half = B_col_size/2;

        nnz_t B1_nnz = Bc[B_col_size_half] - Bc[0];
        nnz_t B2_nnz = B_nnz - B1_nnz;

        auto B1 = &B[0];
        auto B2 = &B[0];

        auto Bc1 = Bc;
        auto Bc2 = &Bc[B_col_size_half];

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
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 1 \n");
#endif

        // prepare splits of matrix A by row

        index_t A_row_size_half = A_row_size / 2;
        index_t A_row_threshold = A_row_size_half + A_row_offset;

        auto Ac1 = Ac; // col_idx
        auto Ac2 = new index_t[A_col_size + 1]; // col_idx

        reorder_split(A, Ac1, Ac2, A_col_size, A_row_threshold);

        nnz_t A1_nnz = Ac1[A_col_size] - Ac1[0];
        nnz_t A2_nnz = Ac2[A_col_size] - Ac2[0];

        auto A1 = &A[0];
        auto A2 = &A[Ac1[A_col_size]];

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
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 2 \n");
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
                    for (nnz_t j = Ac1[i]; j < Ac1[i + 1]; j++) {
                        std::cout << j << "\t" << A1[j].row << "\t" << i + A1_col_offset << "\t" << A1[j].val << std::endl;
                    }
                }

                // print entries of A2:
                std::cout << "\nA2: nnz = " << A2_nnz << std::endl;
                for (nnz_t i = 0; i < A2_col_size; i++) {
                    for (nnz_t j = Ac2[i]; j < Ac2[i + 1]; j++) {
                        std::cout << j << "\t" << A2[j].row << "\t" << i + A2_col_offset << "\t" << A2[j].val << std::endl;
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
                    for (nnz_t j = Bc1[i]; j < Bc1[i + 1]; j++) {
                        std::cout << j << "\t" << B1[j].row << "\t" << i + B1_col_offset << "\t" << B1[j].val << std::endl;
                    }
                }

                // print entries of B2:
                std::cout << "\nB2: nnz = " << B2_nnz << std::endl;
                for (nnz_t i = 0; i < B2_col_size; i++) {
                    for (nnz_t j = Bc2[i]; j < Bc2[i + 1]; j++) {
                        std::cout << j << "\t" << B2[j].row << "\t" << i + B2_col_offset << "\t" << B2[j].val << std::endl;
                    }
                }
            }
        }
//        MPI_Barrier(comm);
#endif

        // =======================================================
        // Save the result of 4 recursive functions in C_temp.

        // C1 = A1 * B1
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 1 \n");
#endif

        if (A1_nnz == 0 || B1_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A1_nnz == 0) {
                    printf("\nskip: A1_nnz == 0\n\n");
                } else {
                    printf("\nskip: B1_nnz == 0\n\n");
                }
            }
#endif
        } else {

            index_t A_col_scan_end = Ac1[A1_col_size];
            index_t B_col_scan_end = Bc1[B1_col_size];

            fast_mm(&A1[0], &B1[0], C,
                    A1_row_size, A1_row_offset, A1_col_size, A1_col_offset,
                    B1_col_size, B1_col_offset,
                    &Ac1[0], &Bc1[0], comm);

            Ac1[A1_col_size] = A_col_scan_end;
            Bc1[B1_col_size] = B_col_scan_end;

        }


        // C2 = A1 * B2:
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 2 \n");
#endif

        if (A1_nnz == 0 || B2_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A1_nnz == 0) {
                    printf("\nskip: A1_nnz == 0\n\n");
                } else {
                    printf("\nskip: B2_nnz == 0\n\n");
                }
            }
#endif
        } else {

            index_t A_col_scan_end = Ac1[A1_col_size];
            index_t B_col_scan_end = Bc2[B2_col_size];

            fast_mm(&A1[0], &B2[0], C,
                    A1_row_size, A1_row_offset, A1_col_size, A1_col_offset,
                    B2_col_size, B2_col_offset,
                    &Ac1[0], &Bc2[0], comm);

            Ac1[A1_col_size] = A_col_scan_end;
            Bc2[B2_col_size] = B_col_scan_end;
        }


        // C3 = A2 * B1
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 3 \n");
#endif

        if (A2_nnz == 0 || B1_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A2_nnz == 0) {
                    printf("\nskip: A2_nnz == 0\n\n");
                } else {
                    printf("\nskip: B1_nnz == 0\n\n");
                }
            }
#endif
        } else {

            index_t A_col_scan_end = Ac2[A2_col_size];
            index_t B_col_scan_end = Bc1[B1_col_size];

            fast_mm(&A2[0], &B1[0], C,
                    A2_row_size, A2_row_offset, A2_col_size, A2_col_offset,
                    B1_col_size, B1_col_offset,
                    &Ac2[0], &Bc1[0], comm);

            Ac2[A2_col_size] = A_col_scan_end;
            Bc1[B1_col_size] = B_col_scan_end;

        }


        // C4 = A2 * B2
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 4 \n");
#endif

        if (A2_nnz == 0 || B2_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A2_nnz == 0) {
                    printf("\nskip: A2_nnz == 0\n\n");
                } else {
                    printf("\nskip: B2_nnz == 0\n\n");
                }
            }
#endif
        } else {

            index_t A_col_scan_end = Ac2[A2_col_size];
            index_t B_col_scan_end = Bc2[B2_col_size];

            fast_mm(&A2[0], &B2[0], C,
                    A2_row_size, A2_row_offset, A2_col_size, A2_col_offset,
                    B2_col_size, B2_col_offset,
                    &Ac2[0], &Bc2[0], comm);

            Ac2[A2_col_size] = A_col_scan_end;
            Bc2[B2_col_size] = B_col_scan_end;
        }

        delete []Ac2;

        // C1 = A1 * B1:
//        fast_mm(A1, B1, C_temp, A_row_size_half, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size_half, B_col_offset, comm);
        // C2 = A2 * B1:
//        fast_mm(A2, B1, C_temp, A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset, B_row_offset, B_col_size_half, B_col_offset, comm);
        // C3 = A1 * B2:
//        fast_mm(A1, B2, C_temp, A_row_size_half, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size-B_col_size_half, B_col_offset+B_col_size_half, comm);
        // C4 = A2 * B2
//        fast_mm(A2, B2, C_temp, A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset, B_row_offset, B_col_size-B_col_size_half, B_col_offset+B_col_size_half, comm);

//        if(rank==0 && verbose_matmat) printf("fast_mm: case 3: step 4 \n");

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 3: end \n");
#endif

    }

//    Ac[0] = A_col_scan_start_tmp;
//    Bc[0] = B_col_scan_start_tmp;

//    return;
}
*/


// Version before switching to CSC matrix format for A and B
//void saena_object::fast_mm
/*
void saena_object::fast_mm(const cooEntry *A, const cooEntry *B, std::vector<cooEntry> &C,
                           const nnz_t A_nnz, const nnz_t B_nnz,
                           const index_t A_row_size, const index_t A_row_offset, const index_t A_col_size, const index_t A_col_offset,
                           const index_t B_col_size, const index_t B_col_offset,
                           const index_t *nnzPerColScan_leftStart,  const index_t *nnzPerColScan_leftEnd,
                           const index_t *nnzPerColScan_rightStart, const index_t *nnzPerColScan_rightEnd,
                           const MPI_Comm comm){

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

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    index_t B_row_offset = A_col_offset;
    index_t A_col_size_half = A_col_size/2;
//    index_t B_row_size_half = A_col_size_half;
//    index_t B_col_size_half = B_col_size/2;

    int verbose_rank = 0;
#ifdef __DEBUG1__
    if(rank==verbose_rank && verbose_matmat) printf("\nfast_mm: start \n");

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

    if (A_row_size * B_col_size < matmat_size_thre1) { //DOLLAR("case0")

#ifdef __DEBUG1__
        if (rank == verbose_rank && (verbose_matmat || verbose_matmat_recursive)) {
            printf("fast_mm: case 0: start \n");
        }
#endif

//        double t1 = MPI_Wtime();

        index_t *nnzPerRow_left = &mempool2[0];
        std::fill(&nnzPerRow_left[0], &nnzPerRow_left[A_row_size], 0);
        index_t *nnzPerRow_left_p = &nnzPerRow_left[0] - A_row_offset;

        for (nnz_t i = 0; i < A_col_size; i++) {
            for (nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftEnd[i]; j++) {
                nnzPerRow_left_p[A[j].row]++;
            }
        }

#ifdef __DEBUG1__
        //    for(nnz_t i = 0; i < A_row_size; i++){
//        printf("%u\n", nnzPerRow_left[i]);
//    }
#endif

        index_t *A_new_row_idx   = &nnzPerRow_left[0];
        index_t *A_new_row_idx_p = &A_new_row_idx[0] - A_row_offset;
        index_t *orig_row_idx    = &mempool2[A_row_size];
        index_t A_nnz_row_sz = 0;

        for (index_t i = 0; i < A_row_size; i++) {
            if (A_new_row_idx[i]) {
                A_new_row_idx[i] = A_nnz_row_sz;
                orig_row_idx[A_nnz_row_sz] = i + A_row_offset;
                A_nnz_row_sz++;
            }
        }

#ifdef __DEBUG1__
//                for (index_t i = 0; i < A_nnz_row_sz; i++) {
//                    std::cout << orig_row_idx[i] << std::endl;
//                }
#endif

        index_t *B_new_col_idx   = &mempool2[A_row_size * 2];
        index_t *B_new_col_idx_p = &B_new_col_idx[0] - B_col_offset;
        index_t *orig_col_idx    = &mempool2[A_row_size * 2 + B_col_size];
        index_t B_nnz_col_sz = 0;
        for (index_t i = 0; i < B_col_size; i++) {
            if (nnzPerColScan_rightEnd[i] != nnzPerColScan_rightStart[i]) {
                B_new_col_idx[i] = B_nnz_col_sz;
                orig_col_idx[B_nnz_col_sz] = i + B_col_offset;
                B_nnz_col_sz++;
            }
        }

#ifdef __DEBUG1__
//        printf("A_row_size = %u, \tA_nnz_row_sz = %u, \tB_col_size = %u, \tB_nnz_col_sz = %u \n",
//            A_row_size, A_nnz_row_sz, B_col_size, B_nnz_col_sz);
#endif

        // check if A_nnz_row_sz * B_nnz_col_sz < matmat_size_thre1, then do dense multiplication. otherwise, do case2 or 3.
        if(A_nnz_row_sz * B_nnz_col_sz < matmat_size_thre2) {

            // initialize
            value_t *C_temp = &mempool1[0];
//                std::fill(&C_temp[0], &C_temp[A_nnz_row_sz * B_nnz_col_sz], 0);

#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) { printf("fast_mm: case 1: step 1 \n"); }
#endif

            mapbit.reset();
            index_t C_index;
            value_t C_val;
            index_t temp;
//                bool C_not_zero = false;
            const index_t *nnzPerColScan_leftStart_p = &nnzPerColScan_leftStart[0] - B_row_offset;
            const index_t *nnzPerColScan_leftEnd_p   = &nnzPerColScan_leftEnd[0] - B_row_offset;

            for (nnz_t j = 0; j < B_col_size; j++) { // columns of B
                for (nnz_t k = nnzPerColScan_rightStart[j]; k < nnzPerColScan_rightEnd[j]; k++) { // nonzeros in column j of B
                    temp = A_nnz_row_sz * B_new_col_idx_p[B[k].col];
                    for (nnz_t i = nnzPerColScan_leftStart_p[B[k].row]; i < nnzPerColScan_leftEnd_p[B[k].row]; i++) { // nonzeros in column B[k].row of A

#ifdef __DEBUG1__
//                            if(rank==0) std::cout << A_new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B_new_col_idx_p[B[k].col] << "\t"
//                            << A_new_row_idx[A[i].row - A_row_offset] << "\t" << B_new_col_idx_p[B[k].col] << "\t"
//                            << C_temp[A_new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B_new_col_idx_p[B[k].col]] << std::endl;
//
//                            if(rank==0) std::cout << B[k].val << "\t" << A[i].val << std::endl;
#endif

                        C_index = A_new_row_idx_p[A[i].row] + temp;
                        C_val = B[k].val * A[i].val;

//                            std::cout << C_index << "\t" << C_val << std::endl;
                        if(mapbit[C_index]) {
                            C_temp[C_index] += C_val;
                        } else {
                            C_temp[C_index] = C_val;
                            mapbit[C_index] = true;
                        }

#ifdef __DEBUG1__
//                            if(rank == 0) std::cout << "A: " << A[i] << "\tB: " << B[k] << "\tC_index: " << A_new_row_idx_p[A[i].row] + temp
//                                 << "\tA_row_offset = " << A_row_offset << "\tB_col_offset = " << B_col_offset << std::endl;

//                            if(rank==1 && A[i].row == 0 && B[j].col == 0) std::cout << "A: " << A[i] << "\tB: " << B[j]
//                                 << "\tC: " << C_temp[(A[i].row-A_row_offset) + A_row_size * (B[j].col-B_col_offset)]
//                                 << "\tA*B: " << B[j].val * A[i].val << std::endl;
#endif
                    }
                }
            }

#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) { printf("fast_mm: case 1: step 2 \n"); }
            //    print_vector(C_temp, -1, "C_temp", comm);
#endif

            nnz_t temp2;
            if(mapbit.count()){
                for (index_t j = 0; j < B_nnz_col_sz; j++) {
                    temp = A_nnz_row_sz * j;
                    for (index_t i = 0; i < A_nnz_row_sz; i++) {
                        temp2 = i + temp;
                        if(mapbit[temp2]){
//                                if(rank==0) std::cout << i << "\t" << j << "\t" << temp2 << "\t" << orig_row_idx[i] << "\t" << orig_col_idx[j] << "\t" << C_temp[i + temp] << std::endl;
                            C.emplace_back(orig_row_idx[i], orig_col_idx[j], C_temp[temp2]);
                        }
                    }
                }
            }

//                t11 = MPI_Wtime() - t11;

#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 1: end \n");
//                printf("C_nnz = %lu\tA: %u, %u\tB: %u, %u\ttime = %f\t\tvec\n", C_nnz, A_row_size, A_nnz_row_sz,
//                       B_col_size, B_nnz_col_sz, t1);
//                printf("C_nnz: %lu \tA_nnz: %lu \t(%f) \tB_nnz: %lu \t(%f) \tA_row: %u (%u) \tB_col: %u (%u) \tt: %.3f \n",
//                       C_nnz, A_nnz, (double(A_nnz)/A_row_size/A_col_size), B_nnz,
//                       (double(B_nnz)/A_col_size/B_col_size), A_row_size, A_nnz_row_sz, B_col_size, B_nnz_col_sz, t11*1000);
//                print_vector(C, -1, "C", comm);
#endif

            return;
        }

    }

    // case2
    // ==============================================================

    if (A_row_size <= A_col_size) { //DOLLAR("case2")

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat) { printf("fast_mm: case 2: start \n"); }
#endif

//        index_t B_row_offset = A_col_offset;
//        index_t A_col_size_half = A_col_size/2;

        // split based on matrix size
        // =======================================================

#ifdef SPLIT_SIZE
        // prepare splits of matrix A by column
        nnz_t A1_nnz = 0, A2_nnz;
        for(nnz_t i = 0; i < A_col_size_half; i++){
            A1_nnz += nnzPerColScan_leftEnd[i] - nnzPerColScan_leftStart[i];
        }

        A2_nnz = A_nnz - A1_nnz;
#endif

        // =======================================================
        // split based on nnz
        // =======================================================

#ifdef SPLIT_NNZ

        // prepare splits of matrix A by column
        nnz_t A1_nnz = 0, A2_nnz;
        auto A_half_nnz = (nnz_t) ceil(A_nnz / 2);
//    index_t A_col_size_half = A_col_size/2;

        if (A_nnz > matmat_nnz_thre) {
            for (nnz_t i = 0; i < A_col_size; i++) {
                A1_nnz += nnzPerColScan_leftEnd[i] - nnzPerColScan_leftStart[i];
                if (A1_nnz >= A_half_nnz) {
                    A_col_size_half = A[nnzPerColScan_leftStart[i]].col + 1 -
                                      A_col_offset; // this is called once! don't optimize.
                    break;
                }
            }
        } else { // A_col_half will stay A_col_size/2
            for (nnz_t i = 0; i < A_col_size_half; i++) {
                A1_nnz += nnzPerColScan_leftEnd[i] - nnzPerColScan_leftStart[i];
            }
        }

        // if A is not being splitted at all following "half nnz method", then swtich to "half size method".
        if (A_col_size_half == A_col_size) {
            A_col_size_half = A_col_size / 2;
            A1_nnz = 0;
            for (nnz_t i = 0; i < A_col_size_half; i++) {
                A1_nnz += nnzPerColScan_leftEnd[i] - nnzPerColScan_leftStart[i];
            }
        }

        A2_nnz = A_nnz - A1_nnz;
#endif

        // =======================================================

        // prepare splits of matrix B by row
        index_t B_row_size_half = A_col_size_half;
        index_t B_row_threshold = B_row_size_half + B_row_offset;
        nnz_t B1_nnz = 0, B2_nnz;

        index_t *nnzPerCol_middle = &mempool2[0];
        std::fill(&nnzPerCol_middle[0], &nnzPerCol_middle[B_col_size], 0);
        // to avoid subtraction in the following for loop " - B_col_offset"
        index_t *nnzPerCol_middle_p = &nnzPerCol_middle[0] - B_col_offset;

        for (nnz_t i = 0; i < B_col_size; i++) {
            for (nnz_t j = nnzPerColScan_rightStart[i]; j < nnzPerColScan_rightEnd[i]; j++) {
                if (B[j].row < B_row_threshold) { // B[j].row - B_row_offset < B_row_size_half
                    nnzPerCol_middle_p[B[j].col]++;
                    B1_nnz++;
                }
            }
        }

        B2_nnz = B_nnz - B1_nnz;

#ifdef __DEBUG1__
//    print_vector(nnzPerCol_middle, -1, "nnzPerCol_middle", comm);
#endif

        std::vector<index_t> nnzPerColScan_middle(B_col_size + 1);
        for (nnz_t i = 0; i < B_col_size; i++) {
            nnzPerColScan_middle[i] = nnzPerColScan_rightStart[i] + nnzPerCol_middle[i];
        }

//    nnzPerCol_middle.clear();
//    nnzPerCol_middle.shrink_to_fit();

#ifdef __DEBUG1__
//        print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);
//        if(rank==0) printf("rank %d: A_nnz = %lu, A1_nnz = %lu, A2_nnz = %lu, B_nnz = %lu, B1_nnz = %lu, B2_nnz = %lu \n",
//                rank, A_nnz, A1_nnz, A2_nnz, B_nnz, B1_nnz, B2_nnz);
        if (rank == verbose_rank && verbose_matmat) { printf("fast_mm: case 2: step 1 \n"); }
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

            if (verbose_matmat_B) {
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
#endif

        // =======================================================
        // Call two recursive functions here. Put the result of the first one in C1, and the second one in C2.
        // merge sort them and add the result to C.
        std::vector<cooEntry> C_temp;

        // C1 = A1 * B1
#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 1 \n");
#endif

        if (A1_nnz == 0 || B1_nnz == 0) { // skip!
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A1_nnz == 0) {
                    printf("\nskip: A1_nnz == 0\n\n");
                } else {
                    printf("\nskip: B1_nnz == 0\n\n");
                }
            }
#endif
        } else {

            fast_mm(&A[0], &B[0], C, A1_nnz, B1_nnz,
                    A_row_size, A_row_offset, A_col_size_half, A_col_offset,
                    B_col_size, B_col_offset,
                    nnzPerColScan_leftStart, nnzPerColScan_leftEnd, // A1
                    nnzPerColScan_rightStart, &nnzPerColScan_middle[0], comm); // B1

        }


        // C2 = A2 * B2
#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 2 \n");
#endif

        if (A2_nnz == 0 || B2_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A2_nnz == 0) {
                    printf("\nskip: A2_nnz == 0\n\n");
                } else {
                    printf("\nskip: B2_nnz == 0\n\n");
                }
            }
#endif
            return;
        }

        fast_mm(&A[0], &B[0], C, A2_nnz, B2_nnz,
                A_row_size, A_row_offset, A_col_size - A_col_size_half, A_col_offset + A_col_size_half,
                B_col_size, B_col_offset,
                &nnzPerColScan_leftStart[A_col_size_half], &nnzPerColScan_leftEnd[A_col_size_half], // A2
                &nnzPerColScan_middle[0], nnzPerColScan_rightEnd, comm); // B2


#ifdef __DEBUG1__
//        print_vector(C1, -1, "C1", comm);
//        print_vector(C2, -1, "C2", comm);

//        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 3 \n");}
//        if(rank==0 && verbose_matmat) printf("C1.size() = %lu, C2.size() = %lu \n", C1.size(), C2.size());
#endif

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 2: end \n");
#endif

        // case3
        // ==============================================================

    } else { //DOLLAR("case3") // (A_row_size > A_col_size)

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 3: start \n");
#endif

        // split based on matrix size
        // =======================================================

#ifdef SPLIT_SIZE
        // prepare splits of matrix B by column
//        index_t A_col_size_half = A_col_size/2;
        index_t B_col_size_half = B_col_size/2;
        nnz_t B1_nnz = 0, B2_nnz;

        for(nnz_t i = 0; i < B_col_size_half; i++){
            B1_nnz += nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i];
        }

        B2_nnz = B_nnz - B1_nnz;
#endif

        // =======================================================
        // split based on nnz
        // =======================================================

#ifdef SPLIT_NNZ
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
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 1 \n");
#endif

        // prepare splits of matrix A by row
        nnz_t A1_nnz = 0, A2_nnz;
        index_t A_row_size_half = A_row_size / 2;
        index_t A_row_threshold = A_row_size_half + A_row_offset;

//    std::vector<index_t> nnzPerCol_middle(A_col_size, 0);
        index_t *nnzPerCol_middle = &mempool2[0];
        std::fill(&nnzPerCol_middle[0], &nnzPerCol_middle[A_col_size], 0);
        // to avoid subtraction in the following for loop " - B_col_offset"
        index_t *nnzPerCol_middle_p = &nnzPerCol_middle[0] - A_col_offset;

        for (nnz_t i = 0; i < A_col_size; i++) {
            for (nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftEnd[i]; j++) {
                if (A[j].row < A_row_threshold) { // A[j].row - A_row_offset < A_row_size_half
                    nnzPerCol_middle_p[A[j].col]++;
                    A1_nnz++;
                }
            }
        }

        A2_nnz = A_nnz - A1_nnz;

        std::vector<index_t> nnzPerColScan_middle(A_col_size);
        for (nnz_t i = 0; i < A_col_size; i++) {
            nnzPerColScan_middle[i] = nnzPerColScan_leftStart[i] + nnzPerCol_middle[i];
        }

//    nnzPerCol_middle.clear();
//    nnzPerCol_middle.shrink_to_fit();

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 2 \n");
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

            if (verbose_matmat_B) {
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

        // =======================================================
        // Save the result of 4 recursive functions in C_temp.

        // C1 = A1 * B1
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 1 \n");
#endif

        if (A1_nnz == 0 || B1_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A1_nnz == 0) {
                    printf("\nskip: A1_nnz == 0\n\n");
                } else {
                    printf("\nskip: B1_nnz == 0\n\n");
                }
            }
#endif
        } else {

            fast_mm(&A[0], &B[0], C, A1_nnz, B1_nnz,
                    A_row_size_half, A_row_offset, A_col_size, A_col_offset,
                    B_col_size_half, B_col_offset,
                    nnzPerColScan_leftStart, &nnzPerColScan_middle[0], // A1
                    nnzPerColScan_rightStart, nnzPerColScan_rightEnd, comm); // B1

        }


        // C2 = A1 * B2:
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 2 \n");
#endif

        if (A1_nnz == 0 || B2_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A1_nnz == 0) {
                    printf("\nskip: A1_nnz == 0\n\n");
                } else {
                    printf("\nskip: B2_nnz == 0\n\n");
                }
            }
#endif
        } else {

            fast_mm(&A[0], &B[0], C, A1_nnz, B2_nnz,
                    A_row_size_half, A_row_offset, A_col_size, A_col_offset,
                    B_col_size - B_col_size_half, B_col_offset + B_col_size_half,
                    nnzPerColScan_leftStart, &nnzPerColScan_middle[0], // A1
                    &nnzPerColScan_rightStart[B_col_size_half], &nnzPerColScan_rightEnd[B_col_size_half],
                    comm); // B2

        }


        // C3 = A2 * B1
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 3 \n");
#endif

        if (A2_nnz == 0 || B1_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A2_nnz == 0) {
                    printf("\nskip: A2_nnz == 0\n\n");
                } else {
                    printf("\nskip: B1_nnz == 0\n\n");
                }
            }
#endif
        } else {

            fast_mm(&A[0], &B[0], C, A2_nnz, B1_nnz,
                    A_row_size - A_row_size_half, A_row_offset + A_row_size_half, A_col_size, A_col_offset,
                    B_col_size_half, B_col_offset,
                    &nnzPerColScan_middle[0], nnzPerColScan_leftEnd, // A2
                    nnzPerColScan_rightStart, nnzPerColScan_rightEnd, comm); // B1

        }


        // C4 = A2 * B2
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 4 \n");
#endif

        if (A2_nnz == 0 || B2_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A2_nnz == 0) {
                    printf("\nskip: A2_nnz == 0\n\n");
                } else {
                    printf("\nskip: B2_nnz == 0\n\n");
                }
            }
#endif
        } else {

            fast_mm(&A[0], &B[0], C, A2_nnz, B2_nnz,
                    A_row_size - A_row_size_half, A_row_offset + A_row_size_half, A_col_size, A_col_offset,
                    B_col_size - B_col_size_half, B_col_offset + B_col_size_half,
                    &nnzPerColScan_middle[0], nnzPerColScan_leftEnd, // A2
                    &nnzPerColScan_rightStart[B_col_size_half], &nnzPerColScan_rightEnd[B_col_size_half],
                    comm); // B2

        }

        // C1 = A1 * B1:
//        fast_mm(A1, B1, C_temp, A_row_size_half, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size_half, B_col_offset, comm);
        // C2 = A2 * B1:
//        fast_mm(A2, B1, C_temp, A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset, B_row_offset, B_col_size_half, B_col_offset, comm);
        // C3 = A1 * B2:
//        fast_mm(A1, B2, C_temp, A_row_size_half, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size-B_col_size_half, B_col_offset+B_col_size_half, comm);
        // C4 = A2 * B2
//        fast_mm(A2, B2, C_temp, A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset, B_row_offset, B_col_size-B_col_size_half, B_col_offset+B_col_size_half, comm);

//        if(rank==0 && verbose_matmat) printf("fast_mm: case 3: step 4 \n");

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 3: end \n");
#endif

    }

//    return;
}
*/


// this version is before swtiching to only using bitvector
//void saena_object::fast_mm
/*
void saena_object::fast_mm(const cooEntry *A, const cooEntry *B, std::vector<cooEntry> &C,
                           const nnz_t A_nnz, const nnz_t B_nnz,
                           const index_t A_row_size, const index_t A_row_offset, const index_t A_col_size, const index_t A_col_offset,
                           const index_t B_col_size, const index_t B_col_offset,
                           const index_t *nnzPerColScan_leftStart,  const index_t *nnzPerColScan_leftEnd,
                           const index_t *nnzPerColScan_rightStart, const index_t *nnzPerColScan_rightEnd,
                           const MPI_Comm comm){

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

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    index_t B_row_offset = A_col_offset;
    index_t A_col_size_half = A_col_size/2;
//    index_t B_row_size_half = A_col_size_half;
//    index_t B_col_size_half = B_col_size/2;

    int verbose_rank = 0;
#ifdef __DEBUG1__
    if(rank==verbose_rank && verbose_matmat) printf("\nfast_mm: start \n");

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

    if (A_row_size * B_col_size < matmat_size_thre1) { //DOLLAR("case0")

#ifdef __DEBUG1__
        if (rank == verbose_rank && (verbose_matmat || verbose_matmat_recursive)) {
            printf("fast_mm: case 0: start \n");
        }
#endif

//        double t1 = MPI_Wtime();

        index_t *nnzPerRow_left = &mempool2[0];
        std::fill(&nnzPerRow_left[0], &nnzPerRow_left[A_row_size], 0);
        index_t *nnzPerRow_left_p = &nnzPerRow_left[0] - A_row_offset;

        for (nnz_t i = 0; i < A_col_size; i++) {
            for (nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftEnd[i]; j++) {
                nnzPerRow_left_p[A[j].row]++;
            }
        }

#ifdef __DEBUG1__
        //    for(nnz_t i = 0; i < A_row_size; i++){
//        printf("%u\n", nnzPerRow_left[i]);
//    }
#endif

*/
/*
        index_t *A_new_row_idx = &nnzPerRow_left[0];
//        index_t *A_new_row_idx_p = &A_new_row_idx[0] - A_row_offset;
//        index_t *orig_row_idx = &mempool2[A_row_size];
        index_t A_nnz_row_sz = 0;

        for (index_t i = 0; i < A_row_size; i++) {
            if (A_new_row_idx[i]) {
//                A_new_row_idx[i] = A_nnz_row_sz;
//                orig_row_idx[A_nnz_row_sz] = i + A_row_offset;
                A_nnz_row_sz++;
            }
        }

#ifdef __DEBUG1__
//    print_vector(A_new_row_idx, -1, "A_new_row_idx", comm);
//        for (index_t i = 0; i < A_row_size; i++) {
//            std::cout << A_new_row_idx[i] << std::endl;
//        }
#endif

//        index_t *B_new_col_idx = &mempool2[A_row_size * 2];
//        index_t *B_new_col_idx_p = &B_new_col_idx[0] - B_col_offset;
//        index_t *orig_col_idx = &mempool2[A_row_size * 2 + B_col_size];
        index_t B_nnz_col_sz = 0;
        for (index_t i = 0; i < B_col_size; i++) {
            if (nnzPerColScan_rightEnd[i] != nnzPerColScan_rightStart[i]) {
//                B_new_col_idx[i] = B_nnz_col_sz;
//                orig_col_idx[B_nnz_col_sz] = i + B_col_offset;
                B_nnz_col_sz++;
            }
        }
*/
/*
        index_t *A_new_row_idx   = &nnzPerRow_left[0];
        index_t *A_new_row_idx_p = &A_new_row_idx[0] - A_row_offset;
        index_t *orig_row_idx    = &mempool2[A_row_size];
        index_t A_nnz_row_sz = 0;

        for (index_t i = 0; i < A_row_size; i++) {
            if (A_new_row_idx[i]) {
                A_new_row_idx[i] = A_nnz_row_sz;
                orig_row_idx[A_nnz_row_sz] = i + A_row_offset;
                A_nnz_row_sz++;
            }
        }

#ifdef __DEBUG1__
//                for (index_t i = 0; i < A_nnz_row_sz; i++) {
//                    std::cout << orig_row_idx[i] << std::endl;
//                }
#endif

        index_t *B_new_col_idx   = &mempool2[A_row_size * 2];
        index_t *B_new_col_idx_p = &B_new_col_idx[0] - B_col_offset;
        index_t *orig_col_idx    = &mempool2[A_row_size * 2 + B_col_size];
        index_t B_nnz_col_sz = 0;
        for (index_t i = 0; i < B_col_size; i++) {
            if (nnzPerColScan_rightEnd[i] != nnzPerColScan_rightStart[i]) {
                B_new_col_idx[i] = B_nnz_col_sz;
                orig_col_idx[B_nnz_col_sz] = i + B_col_offset;
                B_nnz_col_sz++;
            }
        }
*/
/*
#ifdef __DEBUG1__
//        printf("A_row_size = %u, \tA_nnz_row_sz = %u, \tB_col_size = %u, \tB_nnz_col_sz = %u \n",
//            A_row_size, A_nnz_row_sz, B_col_size, B_nnz_col_sz);
#endif

        // check if A_nnz_row_sz * B_nnz_col_sz < matmat_size_thre1, then do dense multiplication. otherwise, do case2 or 3.
        if(A_nnz_row_sz * B_nnz_col_sz < matmat_size_thre2) {

            if (A_nnz_row_sz * B_nnz_col_sz > matmat_size_thre3) { //DOLLAR("case1m")

//                double t11 = MPI_Wtime();

//                std::unordered_map<index_t, value_t> map_matmat;
//                spp::sparse_hash_map<index_t, value_t> map_matmat;
//                map_matmat.reserve(A_nnz + 2*B_nnz);
//                if(mapbit.count() == 1000000)
//                    map_matmat.clear();
//                printf("\nmap_matmat.size = %lu\n", map_matmat.size());
                mapbit.reset();
//                map_matmat.clear();

                index_t C_index;
                value_t C_val;
                index_t temp;
                const index_t *nnzPerColScan_leftStart_p = &nnzPerColScan_leftStart[0] - B_row_offset;
                const index_t *nnzPerColScan_leftEnd_p   = &nnzPerColScan_leftEnd[0] - B_row_offset;
                for (nnz_t j = 0; j < B_col_size; j++) { // columns of B
                    for (nnz_t k = nnzPerColScan_rightStart[j]; k < nnzPerColScan_rightEnd[j]; k++) { // nonzeros in column j of B
                        temp = A_nnz_row_sz * B_new_col_idx_p[B[k].col];
                        for (nnz_t i = nnzPerColScan_leftStart_p[B[k].row]; i < nnzPerColScan_leftEnd_p[B[k].row]; i++) { // nonzeros in column B[k].row of A

//                            C_index = (A[i].row - A_row_offset) + A_row_size * (B[k].col - B_col_offset);
                            C_index = A_new_row_idx_p[A[i].row] + temp;
                            C_val = B[k].val * A[i].val;
//                            auto it = map_matmat.emplace(C_index, C_val);
//                            if (!it.second) it.first->second += C_val;

//                            std::cout << C_index << "\t" << C_val << std::endl;
                            if(mapbit[C_index]) {
                                map_matmat[C_index] += C_val;
                            } else {
                                map_matmat[C_index] = C_val;
                                mapbit[C_index] = true;
                            }

                        }
                    }
                }

//                C.reserve(C.size() + map_matmat.size());
//                C.reserve(C.size() + mapbit.count());
//                for (auto it1 = map_matmat.begin(); it1 != map_matmat.end(); ++it1) {
//                    if(mapbit[it1->first])
//                        C.emplace_back( (it1->first % A_nnz_row_sz) + A_row_offset, (it1->first / A_nnz_row_sz) + B_col_offset, it1->second);
//                }

                nnz_t temp2;
                for (index_t j = 0; j < B_nnz_col_sz; j++) {
                    temp = A_nnz_row_sz * j;
                    for (index_t i = 0; i < A_nnz_row_sz; i++) {
                        temp2 = i + temp;
                        if(mapbit[temp2]){
//                            std::cout << i << "\t" << j << "\t" << i + temp << "\t" << orig_row_idx[i] << "\t"
//                                      << orig_col_idx[j] << "\t" << map_matmat[i + temp] << std::endl;
                            C.emplace_back(orig_row_idx[i], orig_col_idx[j], map_matmat[temp2]);
                        }
                    }
                }

#ifdef __DEBUG1__
                if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 1 (map): end \n");
//                print_vector(C, -1, "C", comm);
//                t11 = MPI_Wtime() - t11;
//                printf("C_nnz = %lu\tA: %u, %u\tB: %u, %u\ttime = %f\t\tmap\n", map_matmat.size(), A_row_size, A_nnz_row_sz,
//                       B_col_size, B_nnz_col_sz, t1);
//                printf("C_nnz: %lu \tA_nnz: %lu \t(%f) \tB_nnz: %lu \t(%f) \tA_row: %u (%u) \tB_col: %u (%u) \tt: %.3f \n",
//                       map_matmat.size(), A_nnz, (double(A_nnz)/A_row_size/A_col_size), B_nnz,
//                       (double(B_nnz)/A_col_size/B_col_size), A_row_size, A_nnz_row_sz, B_col_size, B_nnz_col_sz, t11*1000);

//                map_matmat.clear();
#endif

            } else { //DOLLAR("case1v")

//                double t11 = MPI_Wtime();
*/
/*
//                index_t *A_new_row_idx = &nnzPerRow_left[0];
                index_t *A_new_row_idx_p = &A_new_row_idx[0] - A_row_offset;
                index_t *orig_row_idx    = &mempool2[A_row_size];
                A_nnz_row_sz = 0;

                for (index_t i = 0; i < A_row_size; i++) {
                    if (A_new_row_idx[i]) {
                        A_new_row_idx[i] = A_nnz_row_sz;
                        orig_row_idx[A_nnz_row_sz] = i + A_row_offset;
                        A_nnz_row_sz++;
                    }
                }

#ifdef __DEBUG1__
//                for (index_t i = 0; i < A_nnz_row_sz; i++) {
//                    std::cout << orig_row_idx[i] << std::endl;
//                }
#endif

                index_t *B_new_col_idx = &mempool2[A_row_size * 2];
                index_t *B_new_col_idx_p = &B_new_col_idx[0] - B_col_offset;
                index_t *orig_col_idx = &mempool2[A_row_size * 2 + B_col_size];
                B_nnz_col_sz = 0;
                for (index_t i = 0; i < B_col_size; i++) {
                    if (nnzPerColScan_rightEnd[i] != nnzPerColScan_rightStart[i]) {
                        B_new_col_idx[i] = B_nnz_col_sz;
                        orig_col_idx[B_nnz_col_sz] = i + B_col_offset;
                        B_nnz_col_sz++;
                    }
                }

#ifdef __DEBUG1__
//                printf("A_row_size = %u, \tA_nnz_row_sz = %u, \tB_col_size = %u, \tB_nnz_col_sz = %u \n",
//                        A_row_size, A_nnz_row_sz, B_col_size, B_nnz_col_sz);
//                for (index_t i = 0; i < A_nnz_row_sz; i++) {
//                    std::cout << i << "\t" << orig_row_idx[i] << std::endl;
//                }
//                for (index_t i = 0; i < B_nnz_col_sz; i++) {
//                    std::cout << i << "\t" << orig_col_idx[i] << std::endl;
//                }
#endif
*/
/*
            // initialize
            value_t *C_temp = &mempool1[0];
//                std::fill(&C_temp[0], &C_temp[A_nnz_row_sz * B_nnz_col_sz], 0);

#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) { printf("fast_mm: case 1: step 1 \n"); }
#endif

            mapbit.reset();
            index_t C_index;
            value_t C_val;
            index_t temp;
//                bool C_not_zero = false;
            const index_t *nnzPerColScan_leftStart_p = &nnzPerColScan_leftStart[0] - B_row_offset;
            const index_t *nnzPerColScan_leftEnd_p   = &nnzPerColScan_leftEnd[0] - B_row_offset;

            for (nnz_t j = 0; j < B_col_size; j++) { // columns of B
                for (nnz_t k = nnzPerColScan_rightStart[j]; k < nnzPerColScan_rightEnd[j]; k++) { // nonzeros in column j of B
                    temp = A_nnz_row_sz * B_new_col_idx_p[B[k].col];
                    for (nnz_t i = nnzPerColScan_leftStart_p[B[k].row]; i < nnzPerColScan_leftEnd_p[B[k].row]; i++) { // nonzeros in column B[k].row of A

#ifdef __DEBUG1__
//                            if(rank==0) std::cout << A_new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B_new_col_idx_p[B[k].col] << "\t"
//                            << A_new_row_idx[A[i].row - A_row_offset] << "\t" << B_new_col_idx_p[B[k].col] << "\t"
//                            << C_temp[A_new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B_new_col_idx_p[B[k].col]] << std::endl;
//
//                            if(rank==0) std::cout << B[k].val << "\t" << A[i].val << std::endl;
#endif

//                            C_temp_p[A_new_row_idx_p[A[i].row] + A_nnz_row_sz * B[k].col] += B[k].val * A[i].val;
//                            C_temp[A_new_row_idx_p[A[i].row] + temp] += B[k].val * A[i].val;
//                            C_not_zero = true;

                        C_index = A_new_row_idx_p[A[i].row] + temp;
                        C_val = B[k].val * A[i].val;

//                            std::cout << C_index << "\t" << C_val << std::endl;
                        if(mapbit[C_index]) {
                            C_temp[C_index] += C_val;
                        } else {
                            C_temp[C_index] = C_val;
                            mapbit[C_index] = true;
                        }

#ifdef __DEBUG1__
//                            if(rank == 0) std::cout << "A: " << A[i] << "\tB: " << B[k] << "\tC_index: " << A_new_row_idx_p[A[i].row] + temp
//                                 << "\tA_row_offset = " << A_row_offset << "\tB_col_offset = " << B_col_offset << std::endl;

//                            if(rank==1 && A[i].row == 0 && B[j].col == 0) std::cout << "A: " << A[i] << "\tB: " << B[j]
//                                 << "\tC: " << C_temp[(A[i].row-A_row_offset) + A_row_size * (B[j].col-B_col_offset)]
//                                 << "\tA*B: " << B[j].val * A[i].val << std::endl;
#endif
                    }
                }
            }

#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) { printf("fast_mm: case 1: step 2 \n"); }
            //    print_vector(C_temp, -1, "C_temp", comm);
#endif

//                nnz_t C_nnz = 0; //todo: delete this
//                if(C_not_zero) {
            nnz_t temp2;
            if(mapbit.count()){
                for (index_t j = 0; j < B_nnz_col_sz; j++) {
                    temp = A_nnz_row_sz * j;
                    for (index_t i = 0; i < A_nnz_row_sz; i++) {
//                            if (C_temp[i + temp] != 0) {
                        temp2 = i + temp;
                        if(mapbit[temp2]){
//                                if(rank==0) std::cout << i << "\t" << j << "\t" << temp2 << "\t" << orig_row_idx[i] << "\t" << orig_col_idx[j] << "\t" << C_temp[i + temp] << std::endl;
                            C.emplace_back(orig_row_idx[i], orig_col_idx[j], C_temp[temp2]);
//                                C_nnz++; //todo: delete this
                        }
                    }
                }
            }

//                t11 = MPI_Wtime() - t11;

#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 1: end \n");
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
*/
/*
    // case2
    // ==============================================================

    if (A_row_size <= A_col_size) { //DOLLAR("case2")

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat) { printf("fast_mm: case 2: start \n"); }
#endif

//        index_t B_row_offset = A_col_offset;
//        index_t A_col_size_half = A_col_size/2;

        // split based on matrix size
        // =======================================================

#ifdef SPLIT_SIZE
        // prepare splits of matrix A by column
        nnz_t A1_nnz = 0, A2_nnz;
        for(nnz_t i = 0; i < A_col_size_half; i++){
            A1_nnz += nnzPerColScan_leftEnd[i] - nnzPerColScan_leftStart[i];
        }

        A2_nnz = A_nnz - A1_nnz;
#endif

        // =======================================================
        // split based on nnz
        // =======================================================

#ifdef SPLIT_NNZ

        // prepare splits of matrix A by column
        nnz_t A1_nnz = 0, A2_nnz;
        auto A_half_nnz = (nnz_t) ceil(A_nnz / 2);
//    index_t A_col_size_half = A_col_size/2;

        if (A_nnz > matmat_nnz_thre) {
            for (nnz_t i = 0; i < A_col_size; i++) {
                A1_nnz += nnzPerColScan_leftEnd[i] - nnzPerColScan_leftStart[i];
                if (A1_nnz >= A_half_nnz) {
                    A_col_size_half = A[nnzPerColScan_leftStart[i]].col + 1 -
                                      A_col_offset; // this is called once! don't optimize.
                    break;
                }
            }
        } else { // A_col_half will stay A_col_size/2
            for (nnz_t i = 0; i < A_col_size_half; i++) {
                A1_nnz += nnzPerColScan_leftEnd[i] - nnzPerColScan_leftStart[i];
            }
        }

        // if A is not being splitted at all following "half nnz method", then swtich to "half size method".
        if (A_col_size_half == A_col_size) {
            A_col_size_half = A_col_size / 2;
            A1_nnz = 0;
            for (nnz_t i = 0; i < A_col_size_half; i++) {
                A1_nnz += nnzPerColScan_leftEnd[i] - nnzPerColScan_leftStart[i];
            }
        }

        A2_nnz = A_nnz - A1_nnz;
#endif

        // =======================================================

        // prepare splits of matrix B by row
        index_t B_row_size_half = A_col_size_half;
        index_t B_row_threshold = B_row_size_half + B_row_offset;
        nnz_t B1_nnz = 0, B2_nnz;

        index_t *nnzPerCol_middle = &mempool2[0];
        std::fill(&nnzPerCol_middle[0], &nnzPerCol_middle[B_col_size], 0);
        // to avoid subtraction in the following for loop " - B_col_offset"
        index_t *nnzPerCol_middle_p = &nnzPerCol_middle[0] - B_col_offset;

        for (nnz_t i = 0; i < B_col_size; i++) {
            for (nnz_t j = nnzPerColScan_rightStart[i]; j < nnzPerColScan_rightEnd[i]; j++) {
                if (B[j].row < B_row_threshold) { // B[j].row - B_row_offset < B_row_size_half
                    nnzPerCol_middle_p[B[j].col]++;
                    B1_nnz++;
                }
            }
        }

        B2_nnz = B_nnz - B1_nnz;

#ifdef __DEBUG1__
//    print_vector(nnzPerCol_middle, -1, "nnzPerCol_middle", comm);
#endif

        std::vector<index_t> nnzPerColScan_middle(B_col_size + 1);
        for (nnz_t i = 0; i < B_col_size; i++) {
            nnzPerColScan_middle[i] = nnzPerColScan_rightStart[i] + nnzPerCol_middle[i];
        }

//    nnzPerCol_middle.clear();
//    nnzPerCol_middle.shrink_to_fit();

#ifdef __DEBUG1__
//        print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);
//        if(rank==0) printf("rank %d: A_nnz = %lu, A1_nnz = %lu, A2_nnz = %lu, B_nnz = %lu, B1_nnz = %lu, B2_nnz = %lu \n",
//                rank, A_nnz, A1_nnz, A2_nnz, B_nnz, B1_nnz, B2_nnz);
        if (rank == verbose_rank && verbose_matmat) { printf("fast_mm: case 2: step 1 \n"); }
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

            if (verbose_matmat_B) {
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
#endif

        // =======================================================
        // Call two recursive functions here. Put the result of the first one in C1, and the second one in C2.
        // merge sort them and add the result to C.
        std::vector<cooEntry> C_temp;

        // C1 = A1 * B1
#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 1 \n");
#endif

        if (A1_nnz == 0 || B1_nnz == 0) { // skip!
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A1_nnz == 0) {
                    printf("\nskip: A1_nnz == 0\n\n");
                } else {
                    printf("\nskip: B1_nnz == 0\n\n");
                }
            }
#endif
        } else {

            fast_mm(&A[0], &B[0], C, A1_nnz, B1_nnz,
                    A_row_size, A_row_offset, A_col_size_half, A_col_offset,
                    B_col_size, B_col_offset,
                    nnzPerColScan_leftStart, nnzPerColScan_leftEnd, // A1
                    nnzPerColScan_rightStart, &nnzPerColScan_middle[0], comm); // B1

        }


        // C2 = A2 * B2
#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 2 \n");
#endif

        if (A2_nnz == 0 || B2_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A2_nnz == 0) {
                    printf("\nskip: A2_nnz == 0\n\n");
                } else {
                    printf("\nskip: B2_nnz == 0\n\n");
                }
            }
#endif
            return;
        }

        fast_mm(&A[0], &B[0], C, A2_nnz, B2_nnz,
                A_row_size, A_row_offset, A_col_size - A_col_size_half, A_col_offset + A_col_size_half,
                B_col_size, B_col_offset,
                &nnzPerColScan_leftStart[A_col_size_half], &nnzPerColScan_leftEnd[A_col_size_half], // A2
                &nnzPerColScan_middle[0], nnzPerColScan_rightEnd, comm); // B2


#ifdef __DEBUG1__
//        print_vector(C1, -1, "C1", comm);
//        print_vector(C2, -1, "C2", comm);

//        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 3 \n");}
//        if(rank==0 && verbose_matmat) printf("C1.size() = %lu, C2.size() = %lu \n", C1.size(), C2.size());
#endif

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 2: end \n");
#endif

        // case3
        // ==============================================================

    } else { //DOLLAR("case3") // (A_row_size > A_col_size)

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 3: start \n");
#endif

        // split based on matrix size
        // =======================================================

#ifdef SPLIT_SIZE
        // prepare splits of matrix B by column
//        index_t A_col_size_half = A_col_size/2;
        index_t B_col_size_half = B_col_size/2;
        nnz_t B1_nnz = 0, B2_nnz;

        for(nnz_t i = 0; i < B_col_size_half; i++){
            B1_nnz += nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i];
        }

        B2_nnz = B_nnz - B1_nnz;
#endif

        // =======================================================
        // split based on nnz
        // =======================================================

#ifdef SPLIT_NNZ
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
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 1 \n");
#endif

        // prepare splits of matrix A by row
        nnz_t A1_nnz = 0, A2_nnz;
        index_t A_row_size_half = A_row_size / 2;
        index_t A_row_threshold = A_row_size_half + A_row_offset;

//    std::vector<index_t> nnzPerCol_middle(A_col_size, 0);
        index_t *nnzPerCol_middle = &mempool2[0];
        std::fill(&nnzPerCol_middle[0], &nnzPerCol_middle[A_col_size], 0);
        // to avoid subtraction in the following for loop " - B_col_offset"
        index_t *nnzPerCol_middle_p = &nnzPerCol_middle[0] - A_col_offset;

        for (nnz_t i = 0; i < A_col_size; i++) {
            for (nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftEnd[i]; j++) {
                if (A[j].row < A_row_threshold) { // A[j].row - A_row_offset < A_row_size_half
                    nnzPerCol_middle_p[A[j].col]++;
                    A1_nnz++;
                }
            }
        }

        A2_nnz = A_nnz - A1_nnz;

        std::vector<index_t> nnzPerColScan_middle(A_col_size);
        for (nnz_t i = 0; i < A_col_size; i++) {
            nnzPerColScan_middle[i] = nnzPerColScan_leftStart[i] + nnzPerCol_middle[i];
        }

//    nnzPerCol_middle.clear();
//    nnzPerCol_middle.shrink_to_fit();

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 2 \n");
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

            if (verbose_matmat_B) {
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

        // =======================================================
        // Save the result of 4 recursive functions in C_temp.

        // C1 = A1 * B1
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 1 \n");
#endif

        if (A1_nnz == 0 || B1_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A1_nnz == 0) {
                    printf("\nskip: A1_nnz == 0\n\n");
                } else {
                    printf("\nskip: B1_nnz == 0\n\n");
                }
            }
#endif
        } else {

            fast_mm(&A[0], &B[0], C, A1_nnz, B1_nnz,
                    A_row_size_half, A_row_offset, A_col_size, A_col_offset,
                    B_col_size_half, B_col_offset,
                    nnzPerColScan_leftStart, &nnzPerColScan_middle[0], // A1
                    nnzPerColScan_rightStart, nnzPerColScan_rightEnd, comm); // B1

        }


        // C2 = A1 * B2:
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 2 \n");
#endif

        if (A1_nnz == 0 || B2_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A1_nnz == 0) {
                    printf("\nskip: A1_nnz == 0\n\n");
                } else {
                    printf("\nskip: B2_nnz == 0\n\n");
                }
            }
#endif
        } else {

            fast_mm(&A[0], &B[0], C, A1_nnz, B2_nnz,
                    A_row_size_half, A_row_offset, A_col_size, A_col_offset,
                    B_col_size - B_col_size_half, B_col_offset + B_col_size_half,
                    nnzPerColScan_leftStart, &nnzPerColScan_middle[0], // A1
                    &nnzPerColScan_rightStart[B_col_size_half], &nnzPerColScan_rightEnd[B_col_size_half],
                    comm); // B2

        }


        // C3 = A2 * B1
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 3 \n");
#endif

        if (A2_nnz == 0 || B1_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A2_nnz == 0) {
                    printf("\nskip: A2_nnz == 0\n\n");
                } else {
                    printf("\nskip: B1_nnz == 0\n\n");
                }
            }
#endif
        } else {

            fast_mm(&A[0], &B[0], C, A2_nnz, B1_nnz,
                    A_row_size - A_row_size_half, A_row_offset + A_row_size_half, A_col_size, A_col_offset,
                    B_col_size_half, B_col_offset,
                    &nnzPerColScan_middle[0], nnzPerColScan_leftEnd, // A2
                    nnzPerColScan_rightStart, nnzPerColScan_rightEnd, comm); // B1

        }


        // C4 = A2 * B2
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 4 \n");
#endif

        if (A2_nnz == 0 || B2_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A2_nnz == 0) {
                    printf("\nskip: A2_nnz == 0\n\n");
                } else {
                    printf("\nskip: B2_nnz == 0\n\n");
                }
            }
#endif
        } else {

            fast_mm(&A[0], &B[0], C, A2_nnz, B2_nnz,
                    A_row_size - A_row_size_half, A_row_offset + A_row_size_half, A_col_size, A_col_offset,
                    B_col_size - B_col_size_half, B_col_offset + B_col_size_half,
                    &nnzPerColScan_middle[0], nnzPerColScan_leftEnd, // A2
                    &nnzPerColScan_rightStart[B_col_size_half], &nnzPerColScan_rightEnd[B_col_size_half],
                    comm); // B2

        }

        // C1 = A1 * B1:
//        fast_mm(A1, B1, C_temp, A_row_size_half, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size_half, B_col_offset, comm);
        // C2 = A2 * B1:
//        fast_mm(A2, B1, C_temp, A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset, B_row_offset, B_col_size_half, B_col_offset, comm);
        // C3 = A1 * B2:
//        fast_mm(A1, B2, C_temp, A_row_size_half, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size-B_col_size_half, B_col_offset+B_col_size_half, comm);
        // C4 = A2 * B2
//        fast_mm(A2, B2, C_temp, A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset, B_row_offset, B_col_size-B_col_size_half, B_col_offset+B_col_size_half, comm);

//        if(rank==0 && verbose_matmat) printf("fast_mm: case 3: step 4 \n");

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 3: end \n");
#endif

    }

//    return;
}
*/




// this version is before copying optimized parts back to this function.
//void saena_object::fast_mm
/*
int saena_object::fast_mm_orig(const cooEntry *A, const cooEntry *B, std::vector<cooEntry> &C,
                               const nnz_t A_nnz, const nnz_t B_nnz,
                               const index_t A_row_size, const index_t A_row_offset, const index_t A_col_size, const index_t A_col_offset,
                               const index_t B_col_size, const index_t B_col_offset,
                               const index_t *nnzPerColScan_leftStart,  const index_t *nnzPerColScan_leftEnd,
                               const index_t *nnzPerColScan_rightStart, const index_t *nnzPerColScan_rightEnd, const MPI_Comm comm){

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
    index_t B_row_size_half = A_col_size_half;
    index_t B_col_size_half = B_col_size/2;

    int verbose_rank = 0;
#ifdef __DEBUG1__
    if(rank==verbose_rank && verbose_matmat) printf("\nfast_mm: start \n");
#endif

    if(A_nnz == 0 || B_nnz == 0){
#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat) printf("\nskip: A_nnz == 0 || B_nnz == 0\n\n");
#endif
        return 0;
    }

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

//    index_t size_min = std::min(std::min(A_row_size, A_col_size), B_col_size);

    if( A_row_size * B_col_size < matmat_size_thre2 ){

#ifdef __DEBUG1__
        if(rank==verbose_rank && (verbose_matmat || verbose_matmat_recursive)){printf("fast_mm: case 1: start \n");}
#endif

        // initialize
        value_t *C_temp = mempool1;
        std::fill(&C_temp[0], &C_temp[A_row_size * B_col_size], 0);

#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 1: step 1 \n");}
#endif

        index_t C_index=0;
        for(nnz_t j = 0; j < B_col_size; j++) { // columns of B
            for (nnz_t k = nnzPerColScan_rightStart[j]; k < nnzPerColScan_rightEnd[j]; k++) { // nonzeros in column j of B
                for (nnz_t i = nnzPerColScan_leftStart[B[k].row - B_row_offset];
                     i < nnzPerColScan_leftEnd[B[k].row - B_row_offset]; i++) { // nonzeros in column B[k].row of A

                    C_index = (A[i].row - A_row_offset) + A_row_size * (B[k].col - B_col_offset);
                    C_temp[C_index] += B[k].val * A[i].val;

#ifdef __DEBUG1__
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

#ifdef __DEBUG1__
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

#ifdef __DEBUG1__
        //        print_vector(C, -1, "C", comm);
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 1: end \n");
#endif

    } else if( A_row_size <= A_col_size) {

#ifdef __DEBUG1__
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
        index_t B_row_threshold = B_row_size_half + B_row_offset;

        std::vector<index_t> nnzPerCol_middle(B_col_size, 0);
        // to avoid subtraction in the following for loop " - B_col_offset"
        index_t *nnzPerCol_middle_p = &nnzPerCol_middle[0] - B_col_offset;

        for(nnz_t i = 0; i < B_col_size; i++){
            for(nnz_t j = nnzPerColScan_rightStart[i]; j < nnzPerColScan_rightEnd[i]; j++) {
                if(B[j].row < B_row_threshold){ // B[j].row - B_row_offset < B_row_size_half
//                    nnzPerCol_middle[B[j].col - B_col_offset]++;
                    nnzPerCol_middle_p[B[j].col]++;
                    B1_nnz++;
                }
            }
        }

//        print_vector(nnzPerCol_middle, -1, "nnzPerCol_middle", comm);

        std::vector<index_t> nnzPerColScan_middle(B_col_size + 1);
        nnzPerColScan_middle[0] = 0;
//        for(nnz_t i = 0; i < B_col_size; i++){
//            nnzPerColScan_middle[i+1] = nnzPerColScan_middle[i] + nnzPerCol_middle[i];
//        }

        for(nnz_t i = 0; i < B_col_size; i++){
            nnzPerColScan_middle[i] = nnzPerColScan_rightStart[i] + nnzPerCol_middle[i];
        }

        B2_nnz = B_nnz - B1_nnz;

        nnzPerCol_middle.clear();
        nnzPerCol_middle.shrink_to_fit();

#ifdef __DEBUG1__
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

#ifdef __DEBUG1__
        //        print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);
        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 2 \n");}
#endif

        // A1: start: nnzPerColScan_leftStart,                  end: nnzPerColScan_leftEnd
        // A2: start: nnzPerColScan_leftStart[A_col_size_half], end: nnzPerColScan_leftEnd[A_col_size_half]
        // B1: start: nnzPerColScan_rightStart,                 end: nnzPerColScan_middle
        // B2: start: nnzPerColScan_middle,                     end: nnzPerColScan_rightEnd

#ifdef __DEBUG1__
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
#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 1 \n");
#endif

        fast_mm(&A[0], &B[0], C1, A1_nnz, B1_nnz,
                A_row_size, A_row_offset, A_col_size_half, A_col_offset,
                B_col_size, B_col_offset,
                nnzPerColScan_leftStart,  nnzPerColScan_leftEnd, // A1
                nnzPerColScan_rightStart, &nnzPerColScan_middle[0], comm); // B1

        // C2 = A2 * B2
#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 2 \n");
#endif

        fast_mm(&A[0], &B[0], C2, A2_nnz, B2_nnz,
                A_row_size, A_row_offset, A_col_size-A_col_size_half, A_col_offset+A_col_size_half,
                B_col_size, B_col_offset,
                &nnzPerColScan_leftStart[A_col_size_half], &nnzPerColScan_leftEnd[A_col_size_half], // A2
                &nnzPerColScan_middle[0], nnzPerColScan_rightEnd, comm); // B2

#ifdef __DEBUG1__
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
//#ifdef __DEBUG1__
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
//#ifdef __DEBUG1__
//            if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
//#endif
//            return 0;
//        }

        if(C1.empty()){
            nnz_t C_init_size = C.size();
            C.resize(C.size() + C2.size());
            memcpy(&C[C_init_size], &C2[0], C2.size() * sizeof(cooEntry));

#ifdef __DEBUG1__
            if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
#endif
            return 0;
        }

        if(C2.empty()){
            nnz_t C_init_size = C.size();
            C.resize(C.size() + C1.size());
            memcpy(&C[C_init_size], &C1[0], C1.size() * sizeof(cooEntry));

#ifdef __DEBUG1__
            if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
#endif
            return 0;
        }

#ifdef __DEBUG1__
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

#ifdef __DEBUG1__
                if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
#endif
                return 0;
            }else if(j == C2.size()) {
                while (i < C1.size()) {
                    C.emplace_back(C1[i]);
                    i++;
                }

#ifdef __DEBUG1__
                if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
#endif
                return 0;
            }
        }

#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 2: end \n");
#endif

    } else { // A_row_size > A_col_size

#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: start \n");
#endif

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

#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 1 \n");
#endif
        // prepare splits of matrix A by row
        nnz_t A1_nnz = 0, A2_nnz;
        index_t A_row_size_half = A_row_size/2;
        index_t A_row_threshold = A_row_size_half + A_row_offset;

        std::vector<index_t> nnzPerCol_middle(A_col_size, 0);
        // to avoid subtraction in the following for loop " - B_col_offset"
        index_t *nnzPerCol_middle_p = &nnzPerCol_middle[0] - A_col_offset;

        for(nnz_t i = 0; i < A_col_size; i++){
            for(nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftEnd[i]; j++) {
                if(A[j].row < A_row_threshold){ // A[j].row - A_row_offset < A_row_size_half
//                    nnzPerCol_middle[A[j].col - A_col_offset]++;
                    nnzPerCol_middle_p[A[j].col]++;
                    A1_nnz++;
                }
            }
        }

        A2_nnz = A_nnz - A1_nnz;

        std::vector<index_t> nnzPerColScan_middle(A_col_size + 1);
//        nnzPerColScan_middle[0] = 0;
//        for(nnz_t i = 0; i < A_col_size; i++){
//            nnzPerColScan_middle[i+1] = nnzPerColScan_middle[i] + nnzPerCol_middle[i];
//        }

        for(nnz_t i = 0; i < A_col_size; i++){
            nnzPerColScan_middle[i] = nnzPerColScan_leftStart[i] + nnzPerCol_middle[i];
        }

        nnzPerCol_middle.clear();
        nnzPerCol_middle.shrink_to_fit();

#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 2 \n");
#endif

//        for(nnz_t i = 0; i < A_col_size; i++){
//            nnzPerColScan_middle[i] = nnzPerColScan_middle[i+1] + nnzPerColScan_leftStart[i] - nnzPerColScan_middle[i];
//            if(rank==0) printf("nnzPerColScan_middle[%lu] = %u \n", i, nnzPerColScan_middle[i]);
//        }

#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 3 \n");
#endif

        // A1: start: nnzPerColScan_leftStart,                   end: nnzPerColScan_middle
        // A2: start: nnzPerColScan_middle,                      end: nnzPerColScan_leftEnd
        // B1: start: nnzPerColScan_rightStart,                  end: nnzPerColScan_rightEnd
        // B2: start: nnzPerColScan_rightStart[B_col_size_half], end: nnzPerColScan_rightEnd[B_col_size_half]

#ifdef __DEBUG1__
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
#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 1 \n");
#endif
        fast_mm(&A[0], &B[0], C_temp, A1_nnz, B1_nnz,
                A_row_size_half, A_row_offset, A_col_size, A_col_offset,
                B_col_size_half, B_col_offset,
                nnzPerColScan_leftStart,  &nnzPerColScan_middle[0], // A1
                nnzPerColScan_rightStart, nnzPerColScan_rightEnd, comm); // B1

        // C2 = A2 * B1:
#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 2 \n");
#endif
        fast_mm(&A[0], &B[0], C_temp, A2_nnz, B1_nnz,
                A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset,
                B_col_size_half, B_col_offset,
                &nnzPerColScan_middle[0], nnzPerColScan_leftEnd, // A2
                nnzPerColScan_rightStart, nnzPerColScan_rightEnd, comm); // B1

        // C3 = A1 * B2:
#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 3 \n");
#endif
        fast_mm(&A[0], &B[0], C_temp, A1_nnz, B2_nnz,
                A_row_size_half, A_row_offset, A_col_size, A_col_offset,
                B_col_size-B_col_size_half, B_col_offset+B_col_size_half,
                nnzPerColScan_leftStart,  &nnzPerColScan_middle[0], // A1
                &nnzPerColScan_rightStart[B_col_size_half], &nnzPerColScan_rightEnd[B_col_size_half], comm); // B2

        // C4 = A2 * B2
#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 4 \n");
#endif
        fast_mm(&A[0], &B[0], C_temp, A2_nnz, B2_nnz,
                A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset,
                B_col_size-B_col_size_half, B_col_offset+B_col_size_half,
                &nnzPerColScan_middle[0], nnzPerColScan_leftEnd, // A2
                &nnzPerColScan_rightStart[B_col_size_half], &nnzPerColScan_rightEnd[B_col_size_half], comm); // B2

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

#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: end \n");
#endif
    }

#ifdef __DEBUG1__
    if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
#endif

    return 0;
}
*/


// this version splits the matrices to have half nnz on each side.
//int saena_object::fast_mm_nnz
/*
int saena_object::fast_mm_nnz(const cooEntry *A, const cooEntry *B, std::vector<cooEntry> &C,
                              const nnz_t A_nnz, const nnz_t B_nnz,
                              const index_t A_row_size, const index_t A_row_offset, const index_t A_col_size, const index_t A_col_offset,
                              const index_t B_col_size, const index_t B_col_offset,
                              const index_t *nnzPerColScan_leftStart,  const index_t *nnzPerColScan_leftEnd,
                              const index_t *nnzPerColScan_rightStart, const index_t *nnzPerColScan_rightEnd, const MPI_Comm comm){

    // =======================================================
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
    // =======================================================

#ifdef __DEBUG1__
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    int verbose_rank = 0;
    if(rank==verbose_rank && verbose_matmat) printf("\nfast_mm: start \n");
#endif

    index_t B_row_offset = A_col_offset;

    if(A_nnz == 0 || B_nnz == 0){
#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat) printf("\nskip: A_nnz == 0 || B_nnz == 0\n\n");
#endif
        return 0;
    }

#ifdef __DEBUG1__
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
#endif

    if( A_row_size * B_col_size < matmat_size_thre2 ){

#ifdef __DEBUG1__
        if(rank==verbose_rank && (verbose_matmat || verbose_matmat_recursive)){printf("fast_mm: case 1: start \n");}
#endif

        index_t *nnzPerRow_left = &mempool2[0];
        std::fill(&nnzPerRow_left[0], &nnzPerRow_left[A_row_size], 0);
        index_t *nnzPerRow_left_p = &nnzPerRow_left[0] - A_row_offset;

        for(nnz_t i = 0; i < A_col_size; i++){
            for(nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftEnd[i]; j++) {
                nnzPerRow_left_p[A[j].row]++;
            }
        }

#ifdef __DEBUG1__
        //    for(nnz_t i = 0; i < A_row_size; i++){
//        printf("%u\n", nnzPerRow_left[i]);
//    }
#endif

        index_t *A_new_row_idx   = &nnzPerRow_left[0];
        index_t *A_new_row_idx_p = &A_new_row_idx[0] - A_row_offset;
        index_t *orig_row_idx = &mempool2[A_row_size];
        index_t A_nnz_row_sz = 0;

        for(index_t i = 0; i < A_row_size; i++){
            if(A_new_row_idx[i]){
                A_new_row_idx[i] = A_nnz_row_sz;
                orig_row_idx[A_nnz_row_sz] = i + A_row_offset;
                A_nnz_row_sz++;
            }
        }

#ifdef __DEBUG1__
        //    print_vector(A_new_row_idx, -1, "A_new_row_idx", comm);
#endif

        index_t *B_new_col_idx   = &mempool2[A_row_size * 2];
        index_t *B_new_col_idx_p = &B_new_col_idx[0] - B_col_offset;
        index_t *orig_col_idx = &mempool2[A_row_size * 2 + B_col_size];
        index_t B_nnz_col_sz = 0;
        for(index_t i = 0; i < B_col_size; i++){
            if(nnzPerColScan_rightEnd[i] != nnzPerColScan_rightStart[i]){
                B_new_col_idx[i] = B_nnz_col_sz;
                orig_col_idx[B_nnz_col_sz] = i + B_col_offset;
                B_nnz_col_sz++;
            }
        }

#ifdef __DEBUG1__
        //    printf("A_row_size = %u, \tA_nnz_row_sz = %u, \tB_col_size = %u, \tB_nnz_col_sz = %u \n",
//            A_row_size, A_nnz_row_sz, B_col_size, B_nnz_col_sz);
#endif

        // initialize
        value_t *C_temp = &mempool1[0];
        std::fill(&C_temp[0], &C_temp[A_nnz_row_sz * B_nnz_col_sz], 0);

#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 1: step 1 \n");}
#endif

        index_t temp;
        const index_t *nnzPerColScan_leftStart_p = &nnzPerColScan_leftStart[0] - B_row_offset;
        const index_t *nnzPerColScan_leftEnd_p   = &nnzPerColScan_leftEnd[0] - B_row_offset;

        for(nnz_t j = 0; j < B_col_size; j++) { // columns of B

//        if(rank==0) std::cout << "\n" << j << "\tright: " << nnzPerColScan_rightStart[j] << "\t" << nnzPerColScan_rightEnd[j] << std::endl;

            for (nnz_t k = nnzPerColScan_rightStart[j]; k < nnzPerColScan_rightEnd[j]; k++) { // nonzeros in column j of B

//            if(rank==0) std::cout << k << "\tleft:  " << nnzPerColScan_leftStart_p[B[k].row] << "\t" << nnzPerColScan_leftEnd_p[B[k].row] << std::endl;

                temp = A_nnz_row_sz * B_new_col_idx_p[B[k].col];
                for (nnz_t i = nnzPerColScan_leftStart_p[B[k].row];
                     i < nnzPerColScan_leftEnd_p[B[k].row]; i++) { // nonzeros in column B[k].row of A

#ifdef __DEBUG1__
                    //                if(rank==0) std::cout << A_new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B_new_col_idx_p[B[k].col] << "\t"
//                << A_new_row_idx[A[i].row - A_row_offset] << "\t" << B_new_col_idx_p[B[k].col] << "\t"
//                << C_temp[A_new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B_new_col_idx_p[B[k].col]] << std::endl;

//                if(rank==0) std::cout << B[k].val << "\t" << A[i].val << std::endl;
#endif

//                C_temp_p[A_new_row_idx_p[A[i].row] + A_nnz_row_sz * B[k].col] += B[k].val * A[i].val;
                    C_temp[ A_new_row_idx_p[A[i].row] + temp ] += B[k].val * A[i].val;

#ifdef __DEBUG1__
                    //                if (rank == 0) std::cout << "A: " << A[i] << "\tB: " << B[k] << "\tC_index: " << A_new_row_idx_p[A[i].row] + temp
//                                         << "\tA_row_offset = " << A_row_offset
//                                         << "\tB_col_offset = " << B_col_offset << std::endl;

//                    if(rank==1 && A[i].row == 0 && B[j].col == 0) std::cout << "A: " << A[i] << "\tB: " << B[j]
//                         << "\tC: " << C_temp[(A[i].row-A_row_offset) + A_row_size * (B[j].col-B_col_offset)]
//                         << "\tA*B: " << B[j].val * A[i].val << std::endl;
#endif
                }
            }
        }

#ifdef __DEBUG1__
        //    print_vector(C_temp, -1, "C_temp", comm);
        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 1: step 2 \n");}
#endif

        for(index_t j = 0; j < B_nnz_col_sz; j++) {
            temp = A_nnz_row_sz * j;
            for(index_t i = 0; i < A_nnz_row_sz; i++) {
//            if(rank==0) std::cout << i + A_nnz_row_sz*j << "\t" << orig_row_idx[i] << "\t" << orig_col_idx[j] << "\t" << C_temp[i + A_nnz_row_sz*j] << std::endl;
                if (C_temp[i + A_nnz_row_sz * j] != 0) {
                    C.emplace_back( orig_row_idx[i] , orig_col_idx[j], C_temp[i + temp] );
                }
            }
        }

#ifdef __DEBUG1__
        //       print_vector(C, -1, "C", comm);
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 1: end \n");
#endif

    } else if(A_row_size <= A_col_size) {

#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: start \n");}
#endif

        // prepare splits of matrix A by column
        nnz_t A1_nnz = 0, A2_nnz;
        auto A_half_nnz = (nnz_t)ceil(A_nnz/2);
        index_t A_col_size_half = A_col_size/2;

        if(A_nnz > matmat_nnz_thre){
            for (nnz_t i = 0; i < A_col_size; i++) {
                A1_nnz += nnzPerColScan_leftEnd[i] - nnzPerColScan_leftStart[i];
                if (A1_nnz >= A_half_nnz) {
                    A_col_size_half = A[nnzPerColScan_leftStart[i]].col + 1 - A_col_offset; // this is called once! don't optimize.
                    break;
                }
            }
        } else { // A_col_half will stay A_col_size/2
            for (nnz_t i = 0; i < A_col_size_half; i++) {
                A1_nnz += nnzPerColScan_leftEnd[i] - nnzPerColScan_leftStart[i];
            }
        }

        // if A is not being splitted at all following "half nnz method", then swtich to "half size method".
        if(A_col_size_half == A_col_size){
            A_col_size_half = A_col_size/2;
            A1_nnz = 0;
            for (nnz_t i = 0; i < A_col_size_half; i++) {
                A1_nnz += nnzPerColScan_leftEnd[i] - nnzPerColScan_leftStart[i];
            }
        }

        A2_nnz = A_nnz - A1_nnz;

        // prepare splits of matrix B by row
        index_t B_row_size_half = A_col_size_half;
        index_t B_row_threshold = B_row_size_half + B_row_offset;
        nnz_t B1_nnz = 0, B2_nnz;

//        std::vector<index_t> nnzPerCol_middle(B_col_size, 0);
        index_t *nnzPerCol_middle = &mempool2[0];
        std::fill(&nnzPerCol_middle[0], &nnzPerCol_middle[B_col_size], 0);
        // to avoid subtraction in the following for loop " - B_col_offset"
        index_t *nnzPerCol_middle_p = &nnzPerCol_middle[0] - B_col_offset;

        for(nnz_t i = 0; i < B_col_size; i++){
            for(nnz_t j = nnzPerColScan_rightStart[i]; j < nnzPerColScan_rightEnd[i]; j++) {
                if(B[j].row < B_row_threshold){
//                    nnzPerCol_middle[B[j].col - B_col_offset]++;
                    nnzPerCol_middle_p[B[j].col]++;
                    B1_nnz++;
                }
            }
        }

        B2_nnz = B_nnz - B1_nnz;

#ifdef __DEBUG1__
//        print_vector(nnzPerCol_middle, -1, "nnzPerCol_middle", comm);
#endif

        std::vector<index_t> nnzPerColScan_middle(B_col_size + 1);
        for(nnz_t i = 0; i < B_col_size; i++){
            nnzPerColScan_middle[i] = nnzPerColScan_rightStart[i] + nnzPerCol_middle[i];
        }

//        std::vector<index_t> nnzPerColScan_middle(B_col_size + 1);
//        nnzPerColScan_middle[0] = 0;
//        for(nnz_t i = 0; i < B_col_size; i++){
//            nnzPerColScan_middle[i+1] = nnzPerColScan_middle[i] + nnzPerCol_middle[i];
//        }

//        nnzPerCol_middle.clear();
//        nnzPerCol_middle.shrink_to_fit();

#ifdef __DEBUG1__
//        print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);
        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 1 \n");}
#endif

//        for(nnz_t i = 0; i < B_col_size; i++){
//            nnzPerColScan_middle[i] = nnzPerColScan_middle[i+1] + nnzPerColScan_rightStart[i] - nnzPerColScan_middle[i];
//            if(rank==0) printf("nnzPerColScan_middle[%lu] = %u, \tnnzPerColScan_middle[%lu] = %u, \tnnzPerColScan_rightStart = %u \n",
//                    i, nnzPerColScan_middle[i], i+1, nnzPerColScan_middle[i+1], nnzPerColScan_rightStart[i]);
//        }

#ifdef __DEBUG1__
//        print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);
//        if(rank==verbose_rank) printf("rank %d: nnz: (A, A1, A2) = (%lu, %lu, %lu), (B, B1, B2) = (%lu, %lu, %lu), A_col_half = %u of %u \n",
//                           rank, A_nnz, A1_nnz, A2_nnz, B_nnz, B1_nnz, B2_nnz, A_col_half, A_col_size);
        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 2 \n");}
#endif

        // A1: start: nnzPerColScan_leftStart,               end: nnzPerColScan_leftEnd
        // A2: start: nnzPerColScan_leftStart[A_col_half],   end: nnzPerColScan_leftEnd[A_col_half]
        // B1: start: nnzPerColScan_rightStart,              end: nnzPerColScan_middle
        // B2: start: nnzPerColScan_middle,                  end: nnzPerColScan_rightEnd

#ifdef __DEBUG1__
//        MPI_Barrier(comm);
        if(rank==verbose_rank){

//            printf("fast_mm: case 2: \nA_nnz: (%lu, %lu, %lu), B_nnz: (%lu, %lu, %lu)\n"
//                   "A_size: (%u, %u, %u), B_size: (%u) \n",
//                    A_nnz, A1_nnz, A2_nnz, B_nnz, B1_nnz, B2_nnz, A_row_size, A_col_size, A_col_half, B_col_size);

            if(verbose_matmat_A) {
                std::cout << "\nranges of A:" << std::endl;
                for (nnz_t i = 0; i < A_col_size; i++) {
                    std::cout << i << "\t" << nnzPerColScan_leftStart[i] << "\t" << nnzPerColScan_leftEnd[i]
                              << std::endl;
                }

                std::cout << "\nranges of A1:" << std::endl;
                for (nnz_t i = 0; i < A_col_size_half; i++) {
                    std::cout << i << "\t" << nnzPerColScan_leftStart[i] << "\t" << nnzPerColScan_leftStart[i + 1]
                              << std::endl;
                }

                std::cout << "\nranges of A2:" << std::endl;
                for (nnz_t i = 0; i < A_col_size - A_col_size_half; i++) {
                    std::cout << i << "\t" << nnzPerColScan_leftStart[A_col_size_half + i]
                              << "\t" << nnzPerColScan_leftStart[A_col_size_half + i + 1] << std::endl;
                }

                // print entries of A1:
                std::cout << "\nA1: nnz = " << A1_nnz << std::endl;
                for (nnz_t i = 0; i < A_col_size_half; i++) {
                    for (nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftStart[i + 1]; j++) {
                        std::cout << j << "\t" << A[j] << std::endl;
                    }
                }

                // print entries of A2:
                std::cout << "\nA2: nnz = " << A2_nnz << std::endl;
                for (nnz_t i = 0; i < A_col_size - A_col_size_half; i++) {
                    for (nnz_t j = nnzPerColScan_leftStart[A_col_size_half + i];
                         j < nnzPerColScan_leftStart[A_col_size_half + i + 1]; j++) {
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
#endif

        std::vector<cooEntry> C1, C2;

        // C1 = A1 * B1
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 1 \n");
        fast_mm(&A[0], &B[0], C1, A1_nnz, B1_nnz,
                A_row_size, A_row_offset, A_col_size_half, A_col_offset,
                B_col_size, B_col_offset,
                nnzPerColScan_leftStart,  nnzPerColScan_leftEnd, // A1
                nnzPerColScan_rightStart, &nnzPerColScan_middle[0], comm); // B1

        // C2 = A2 * B2
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 2 \n");
        fast_mm(&A[0], &B[0], C2, A2_nnz, B2_nnz,
                A_row_size, A_row_offset, A_col_size-A_col_size_half, A_col_offset+A_col_size_half,
                B_col_size, B_col_offset,
                &nnzPerColScan_leftStart[A_col_size_half], &nnzPerColScan_leftEnd[A_col_size_half], // A2
                &nnzPerColScan_middle[0], nnzPerColScan_rightEnd, comm); // B2

#ifdef __DEBUG1__
//        print_vector(C1, -1, "C1", comm);
//        print_vector(C2, -1, "C2", comm);

        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 3 \n");}
//        if(rank==0 && verbose_matmat) printf("C1.size() = %lu, C2.size() = %lu \n", C1.size(), C2.size());
#endif

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

#ifdef __DEBUG1__
            if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
#endif

            return 0;
        }

        if(C2.empty()) {
            while (i < C1.size()) {
                C.emplace_back(C1[i]);
                i++;
            }

#ifdef __DEBUG1__
            if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
#endif

            return 0;
        }

#ifdef __DEBUG1__
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

#ifdef __DEBUG1__
                if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
#endif

                return 0;
            }else if(j == C2.size()) {
                while (i < C1.size()) {
                    C.emplace_back(C1[i]);
                    i++;
                }

#ifdef __DEBUG1__
                if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
#endif

                return 0;
            }
        }

#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 2: end \n");
#endif

    } else { // A_row_size > A_col_size

#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: start \n");
#endif

        // prepare splits of matrix B by column
        nnz_t B1_nnz = 0, B2_nnz;
        auto B_half_nnz = (nnz_t)ceil(B_nnz/2);
        index_t B_col_size_half = B_col_size/2;

        if(B_nnz > matmat_nnz_thre) {
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
        if(B_col_size_half == B_col_size){
            B_col_size_half = B_col_size/2;
            B1_nnz = 0;
            for (nnz_t i = 0; i < B_col_size_half; i++) {
                B1_nnz += nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i];
            }
        }

        B2_nnz = B_nnz - B1_nnz;

#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 1 \n");
#endif

        // prepare splits of matrix A by row
        index_t A_row_size_half = A_row_size/2;
        index_t A_row_threshold = A_row_size_half + A_row_offset;
        nnz_t A1_nnz = 0, A2_nnz;

//        std::vector<index_t> nnzPerCol_middle(A_col_size, 0);
        index_t *nnzPerCol_middle = &mempool2[0];
        std::fill(&nnzPerCol_middle[0], &nnzPerCol_middle[A_col_size], 0);
        // to avoid subtraction in the following for loop " - A_col_offset"
        index_t *nnzPerCol_middle_p = &nnzPerCol_middle[0] - A_col_offset;

        for(nnz_t i = 0; i < A_col_size; i++){
            for(nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftEnd[i]; j++) {
                if(A[j].row < A_row_threshold){
//                    nnzPerCol_middle[A[j].col - A_col_offset]++;
                    nnzPerCol_middle_p[A[j].col]++;
                    A1_nnz++;
                }
            }
        }

        A2_nnz = A_nnz - A1_nnz;

        std::vector<index_t> nnzPerColScan_middle(A_col_size);
        for(nnz_t i = 0; i < A_col_size; i++){
            nnzPerColScan_middle[i] = nnzPerColScan_leftStart[i] + nnzPerCol_middle[i];
        }

//        std::vector<index_t> nnzPerColScan_middle(A_col_size + 1);
//        nnzPerColScan_middle[0] = 0;
//        for(nnz_t i = 0; i < A_col_size; i++){
//            nnzPerColScan_middle[i+1] = nnzPerColScan_middle[i] + nnzPerCol_middle[i];
//        }

//        nnzPerCol_middle.clear();
//        nnzPerCol_middle.shrink_to_fit();

#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 2 \n");
#endif

//        for(nnz_t i = 0; i < A_col_size; i++){
//            nnzPerColScan_middle[i] = nnzPerColScan_middle[i+1] + nnzPerColScan_leftStart[i] - nnzPerColScan_middle[i];
//            if(rank==0) printf("nnzPerColScan_middle[%lu] = %u \n", i, nnzPerColScan_middle[i]);
//        }

#ifdef __DEBUG1__
//        print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);
//        if(rank==verbose_rank) printf("rank %d: nnz: (A, A1, A2) = (%lu, %lu, %lu), (B, B1, B2) = (%lu, %lu, %lu), B_col_size_half = %u of %u \n",
//                           rank, A_nnz, A1_nnz, A2_nnz, B_nnz, B1_nnz, B2_nnz, B_col_size_half, B_col_size);

        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 3 \n");
#endif

        // A1: start: nnzPerColScan_leftStart,                end: nnzPerColScan_middle
        // A2: start: nnzPerColScan_middle,                   end: nnzPerColScan_leftEnd
        // B1: start: nnzPerColScan_rightStart,               end: nnzPerColScan_rightEnd
        // B2: start: nnzPerColScan_rightStart[B_col_size_half], end: nnzPerColScan_rightEnd[B_col_size_half]

#ifdef __DEBUG1__
//        MPI_Barrier(comm);
        if(rank==verbose_rank){

//            printf("fast_mm: case 3: \nA_nnz: (%lu, %lu, %lu), B_nnz: (%lu, %lu, %lu)\n"
//                   "A_size: (%u, %u), B_size: (%u, %u) \n",
//                   A_nnz, A1_nnz, A2_nnz, B_nnz, B1_nnz, B2_nnz, A_row_size, A_col_size, B_col_size, B_col_size_half);

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
                for (nnz_t i = 0; i < B_col_size_half; i++) {
                    std::cout << i << "\t" << nnzPerColScan_rightStart[i] << "\t" << nnzPerColScan_rightEnd[i]
                              << std::endl;
                }

                std::cout << "\nranges of B2:" << std::endl;
                for (nnz_t i = 0; i < B_col_size - B_col_size_half; i++) {
                    std::cout << i << "\t" << nnzPerColScan_rightStart[B_col_size_half + i]
                              << "\t" << nnzPerColScan_rightEnd[B_col_size_half + i] << std::endl;
                }

                // print entries of B1:
                std::cout << "\nB1: nnz = " << B1_nnz << std::endl;
                for (nnz_t i = 0; i < B_col_size_half; i++) {
                    for (nnz_t j = nnzPerColScan_rightStart[i]; j < nnzPerColScan_rightEnd[i]; j++) {
                        std::cout << j << "\t" << B[j] << std::endl;
                    }
                }

                // print entries of B2:
                std::cout << "\nB2: nnz = " << B2_nnz << std::endl;
                for (nnz_t i = 0; i < B_col_size - B_col_size_half; i++) {
                    for (nnz_t j = nnzPerColScan_rightStart[B_col_size_half + i];
                         j < nnzPerColScan_rightEnd[B_col_size_half + i]; j++) {
                        std::cout << j << "\t" << B[j] << std::endl;
                    }
                }
            }
        }
//        MPI_Barrier(comm);
#endif

        std::vector<cooEntry> C_temp;

        // C1 = A1 * B1
#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 1 \n");
#endif

        fast_mm(&A[0], &B[0], C_temp, A1_nnz, B1_nnz,
                A_row_size_half, A_row_offset, A_col_size, A_col_offset,
                B_col_size_half, B_col_offset,
                nnzPerColScan_leftStart,  &nnzPerColScan_middle[0], // A1
                nnzPerColScan_rightStart, nnzPerColScan_rightEnd, comm); // B1

        // C2 = A2 * B1:
#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 2 \n");
#endif

        fast_mm(&A[0], &B[0], C_temp, A2_nnz, B1_nnz,
                A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset,
                B_col_size_half, B_col_offset,
                &nnzPerColScan_middle[0], nnzPerColScan_leftEnd, // A2
                nnzPerColScan_rightStart, nnzPerColScan_rightEnd, comm); // B1

        // C3 = A1 * B2:
#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 3 \n");
#endif

        fast_mm(&A[0], &B[0], C_temp, A1_nnz, B2_nnz,
                A_row_size_half, A_row_offset, A_col_size, A_col_offset,
                B_col_size-B_col_size_half, B_col_offset+B_col_size_half,
                nnzPerColScan_leftStart,  &nnzPerColScan_middle[0], // A1
                &nnzPerColScan_rightStart[B_col_size_half], &nnzPerColScan_rightEnd[B_col_size_half], comm); // B2

        // C4 = A2 * B2
#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 4 \n");
#endif

        fast_mm(&A[0], &B[0], C_temp, A2_nnz, B2_nnz,
                A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset,
                B_col_size-B_col_size_half, B_col_offset+B_col_size_half,
                &nnzPerColScan_middle[0], nnzPerColScan_leftEnd, // A2
                &nnzPerColScan_rightStart[B_col_size_half], &nnzPerColScan_rightEnd[B_col_size_half], comm); // B2

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
                i++;
                C.back().val += C_temp[i].val;
            }
        }

#ifdef __DEBUG1__
        if(rank==verbose_rank && verbose_matmat) printf("fast_mm: case 3: end \n");
#endif

    }

#ifdef __DEBUG1__
    if(rank==verbose_rank && verbose_matmat) printf("fast_mm: end \n\n");
#endif

    return 0;
}
*/


//void saena_object::fast_mm
/*
void saena_object::fast_mm(const cooEntry *A, const cooEntry *B, std::vector<cooEntry> &C,
                           const nnz_t A_nnz, const nnz_t B_nnz,
                           const index_t A_row_size, const index_t A_row_offset, const index_t A_col_size, const index_t A_col_offset,
                           const index_t B_col_size, const index_t B_col_offset,
                           const index_t *nnzPerColScan_leftStart,  const index_t *nnzPerColScan_leftEnd,
                           const index_t *nnzPerColScan_rightStart, const index_t *nnzPerColScan_rightEnd,
                           std::unordered_map<index_t, value_t> &map_matmat, const MPI_Comm comm){

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

    map_matmat.clear();

    index_t B_row_offset = A_col_offset;
    index_t A_col_size_half = A_col_size/2;
//    index_t B_row_size_half = A_col_size_half;
//    index_t B_col_size_half = B_col_size/2;

    int verbose_rank = 0;
#ifdef __DEBUG1__
    if(rank==verbose_rank && verbose_matmat) printf("\nfast_mm: start \n");
#endif

//    if(A_nnz == 0 || B_nnz == 0){
//#ifdef __DEBUG1__
//        if(rank==verbose_rank && verbose_matmat) printf("\nskip: A_nnz == 0 || B_nnz == 0\n\n");
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

    if (A_row_size * B_col_size < matmat_size_thre1) { //DOLLAR("case0")

#ifdef __DEBUG1__
        if (rank == verbose_rank && (verbose_matmat || verbose_matmat_recursive)) {
            printf("fast_mm: case 1: start \n");
        }
#endif

        double t1 = MPI_Wtime();

        index_t *nnzPerRow_left = &mempool2[0];
        std::fill(&nnzPerRow_left[0], &nnzPerRow_left[A_row_size], 0);
        index_t *nnzPerRow_left_p = &nnzPerRow_left[0] - A_row_offset;

        for (nnz_t i = 0; i < A_col_size; i++) {
            for (nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftEnd[i]; j++) {
                nnzPerRow_left_p[A[j].row]++;
            }
        }

#ifdef __DEBUG1__
        //    for(nnz_t i = 0; i < A_row_size; i++){
//        printf("%u\n", nnzPerRow_left[i]);
//    }
#endif

        index_t *A_new_row_idx = &nnzPerRow_left[0];
//        index_t *A_new_row_idx_p = &A_new_row_idx[0] - A_row_offset;
//        index_t *orig_row_idx = &mempool2[A_row_size];
        index_t A_nnz_row_sz = 0;

        for (index_t i = 0; i < A_row_size; i++) {
            if (A_new_row_idx[i]) {
//                A_new_row_idx[i] = A_nnz_row_sz;
//                orig_row_idx[A_nnz_row_sz] = i + A_row_offset;
                A_nnz_row_sz++;
            }
        }

#ifdef __DEBUG1__
        //    print_vector(A_new_row_idx, -1, "A_new_row_idx", comm);
#endif

//        index_t *B_new_col_idx = &mempool2[A_row_size * 2];
//        index_t *B_new_col_idx_p = &B_new_col_idx[0] - B_col_offset;
//        index_t *orig_col_idx = &mempool2[A_row_size * 2 + B_col_size];
        index_t B_nnz_col_sz = 0;
        for (index_t i = 0; i < B_col_size; i++) {
            if (nnzPerColScan_rightEnd[i] != nnzPerColScan_rightStart[i]) {
//                B_new_col_idx[i] = B_nnz_col_sz;
//                orig_col_idx[B_nnz_col_sz] = i + B_col_offset;
                B_nnz_col_sz++;
            }
        }

#ifdef __DEBUG1__
        //    printf("A_row_size = %u, \tA_nnz_row_sz = %u, \tB_col_size = %u, \tB_nnz_col_sz = %u \n",
//            A_row_size, A_nnz_row_sz, B_col_size, B_nnz_col_sz);
#endif

        // check if A_nnz_row_sz * B_nnz_col_sz < matmat_size_thre2, then do dense multiplication. otherwise, do case2 or 3.
        if(A_nnz_row_sz * B_nnz_col_sz < matmat_size_thre2) {

            if (A_nnz_row_sz * B_nnz_col_sz < matmat_size_thre3) { //DOLLAR("case1m")
//                std::unordered_map<index_t, value_t> map_matmat;

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
//                std::map<index_t, value_t>::iterator it1;
                for (auto it1 = map_matmat.begin(); it1 != map_matmat.end(); ++it1) {
//                std::cout << it1->first.first << "\t" << it1->first.second << "\t" << it1->second << std::endl;
                    C.emplace_back( (it1->first % A_row_size) + A_row_offset, (it1->first / A_row_size) + B_col_offset, it1->second);
                }

//                t1 = MPI_Wtime() - t1;
//                printf("C_nnz = %lu\tA: %u, %u\tB: %u, %u\ttime = %f\t\tmap\n", map_matmat.size(), A_row_size, A_nnz_row_sz,
//                       B_col_size, B_nnz_col_sz, t1);


#ifdef __DEBUG1__
//       print_vector(C, -1, "C", comm);
                if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 1: end \n");
#endif

                return;
            } else { //DOLLAR("case1v")
//                index_t *A_new_row_idx = &nnzPerRow_left[0];
                index_t *A_new_row_idx_p = &A_new_row_idx[0] - A_row_offset;
                index_t *orig_row_idx = &mempool2[A_row_size];
//                index_t A_nnz_row_sz = 0;
                A_nnz_row_sz = 0;

                for (index_t i = 0; i < A_row_size; i++) {
                    if (A_new_row_idx[i]) {
                        A_new_row_idx[i] = A_nnz_row_sz;
                        orig_row_idx[A_nnz_row_sz] = i + A_row_offset;
                        A_nnz_row_sz++;
                    }
                }

#ifdef __DEBUG1__
                //    print_vector(A_new_row_idx, -1, "A_new_row_idx", comm);
#endif

                index_t *B_new_col_idx = &mempool2[A_row_size * 2];
                index_t *B_new_col_idx_p = &B_new_col_idx[0] - B_col_offset;
                index_t *orig_col_idx = &mempool2[A_row_size * 2 + B_col_size];
//                index_t B_nnz_col_sz = 0;
                B_nnz_col_sz = 0;
                for (index_t i = 0; i < B_col_size; i++) {
                    if (nnzPerColScan_rightEnd[i] != nnzPerColScan_rightStart[i]) {
                        B_new_col_idx[i] = B_nnz_col_sz;
                        orig_col_idx[B_nnz_col_sz] = i + B_col_offset;
                        B_nnz_col_sz++;
                    }
                }

#ifdef __DEBUG1__
                //    printf("A_row_size = %u, \tA_nnz_row_sz = %u, \tB_col_size = %u, \tB_nnz_col_sz = %u \n",
//            A_row_size, A_nnz_row_sz, B_col_size, B_nnz_col_sz);
#endif

                    // initialize
                    value_t *C_temp = &mempool1[0];
                    std::fill(&C_temp[0], &C_temp[A_nnz_row_sz * B_nnz_col_sz], 0);

#ifdef __DEBUG1__
                    if (rank == verbose_rank && verbose_matmat) { printf("fast_mm: case 1: step 1 \n"); }
#endif

                    index_t temp;
                    const index_t *nnzPerColScan_leftStart_p = &nnzPerColScan_leftStart[0] - B_row_offset;
                    const index_t *nnzPerColScan_leftEnd_p = &nnzPerColScan_leftEnd[0] - B_row_offset;

                    for (nnz_t j = 0; j < B_col_size; j++) { // columns of B

//        if(rank==0) std::cout << "\n" << j << "\tright: " << nnzPerColScan_rightStart[j] << "\t" << nnzPerColScan_rightEnd[j] << std::endl;

                        for (nnz_t k = nnzPerColScan_rightStart[j];
                             k < nnzPerColScan_rightEnd[j]; k++) { // nonzeros in column j of B

//            if(rank==0) std::cout << k << "\tleft:  " << nnzPerColScan_leftStart_p[B[k].row] << "\t" << nnzPerColScan_leftEnd_p[B[k].row] << std::endl;

                            temp = A_nnz_row_sz * B_new_col_idx_p[B[k].col];
                            for (nnz_t i = nnzPerColScan_leftStart_p[B[k].row];
                                 i < nnzPerColScan_leftEnd_p[B[k].row]; i++) { // nonzeros in column B[k].row of A

#ifdef __DEBUG1__
                                //                if(rank==0) std::cout << A_new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B_new_col_idx_p[B[k].col] << "\t"
//                << A_new_row_idx[A[i].row - A_row_offset] << "\t" << B_new_col_idx_p[B[k].col] << "\t"
//                << C_temp[A_new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B_new_col_idx_p[B[k].col]] << std::endl;

//                if(rank==0) std::cout << B[k].val << "\t" << A[i].val << std::endl;
#endif

//                C_temp_p[A_new_row_idx_p[A[i].row] + A_nnz_row_sz * B[k].col] += B[k].val * A[i].val;
                                C_temp[A_new_row_idx_p[A[i].row] + temp] += B[k].val * A[i].val;

#ifdef __DEBUG1__
                                //                if (rank == 0) std::cout << "A: " << A[i] << "\tB: " << B[k] << "\tC_index: " << A_new_row_idx_p[A[i].row] + temp
//                                         << "\tA_row_offset = " << A_row_offset
//                                         << "\tB_col_offset = " << B_col_offset << std::endl;

//                    if(rank==1 && A[i].row == 0 && B[j].col == 0) std::cout << "A: " << A[i] << "\tB: " << B[j]
//                         << "\tC: " << C_temp[(A[i].row-A_row_offset) + A_row_size * (B[j].col-B_col_offset)]
//                         << "\tA*B: " << B[j].val * A[i].val << std::endl;
#endif
                            }
                        }
                    }

#ifdef __DEBUG1__
                    //    print_vector(C_temp, -1, "C_temp", comm);
                    if (rank == verbose_rank && verbose_matmat) { printf("fast_mm: case 1: step 2 \n"); }
#endif

                    nnz_t C_nnz = 0;
                    for (index_t j = 0; j < B_nnz_col_sz; j++) {
                        temp = A_nnz_row_sz * j;
                        for (index_t i = 0; i < A_nnz_row_sz; i++) {
//                if(rank==0) std::cout << i + A_nnz_row_sz*j << "\t" << orig_row_idx[i] << "\t" << orig_col_idx[j] << "\t" << C_temp[i + A_nnz_row_sz*j] << std::endl;
                            if (C_temp[i + A_nnz_row_sz * j] != 0) {
                                C.emplace_back(orig_row_idx[i], orig_col_idx[j], C_temp[i + temp]);
                                C_nnz++;
                            }
                        }
                    }

//                    t1 = MPI_Wtime() - t1;
//                    printf("C_nnz = %lu\tA: %u, %u\tB: %u, %u\ttime = %f\t\tvec\n", C_nnz, A_row_size, A_nnz_row_sz,
//                           B_col_size, B_nnz_col_sz, t1);

#ifdef __DEBUG1__
//       print_vector(C, -1, "C", comm);
                    if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 1: end \n");
#endif

                    return;

                }
            }

    }

    // case2
    // ==============================================================

    if (A_row_size <= A_col_size) { //DOLLAR("case2")

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat) { printf("fast_mm: case 2: start \n"); }
#endif

//        index_t B_row_offset = A_col_offset;
//        index_t A_col_size_half = A_col_size/2;

        // split based on matrix size
        // =======================================================

#ifdef SPLIT_SIZE
        // prepare splits of matrix A by column
        nnz_t A1_nnz = 0, A2_nnz;
        for(nnz_t i = 0; i < A_col_size_half; i++){
            A1_nnz += nnzPerColScan_leftEnd[i] - nnzPerColScan_leftStart[i];
        }

        A2_nnz = A_nnz - A1_nnz;
#endif

        // =======================================================
        // split based on nnz
        // =======================================================

#ifdef SPLIT_NNZ

        // prepare splits of matrix A by column
        nnz_t A1_nnz = 0, A2_nnz;
        auto A_half_nnz = (nnz_t) ceil(A_nnz / 2);
//    index_t A_col_size_half = A_col_size/2;

        if (A_nnz > matmat_nnz_thre) {
            for (nnz_t i = 0; i < A_col_size; i++) {
                A1_nnz += nnzPerColScan_leftEnd[i] - nnzPerColScan_leftStart[i];
                if (A1_nnz >= A_half_nnz) {
                    A_col_size_half = A[nnzPerColScan_leftStart[i]].col + 1 -
                                      A_col_offset; // this is called once! don't optimize.
                    break;
                }
            }
        } else { // A_col_half will stay A_col_size/2
            for (nnz_t i = 0; i < A_col_size_half; i++) {
                A1_nnz += nnzPerColScan_leftEnd[i] - nnzPerColScan_leftStart[i];
            }
        }

        // if A is not being splitted at all following "half nnz method", then swtich to "half size method".
        if (A_col_size_half == A_col_size) {
            A_col_size_half = A_col_size / 2;
            A1_nnz = 0;
            for (nnz_t i = 0; i < A_col_size_half; i++) {
                A1_nnz += nnzPerColScan_leftEnd[i] - nnzPerColScan_leftStart[i];
            }
        }

        A2_nnz = A_nnz - A1_nnz;
#endif

        // =======================================================

        // prepare splits of matrix B by row
        index_t B_row_size_half = A_col_size_half;
        index_t B_row_threshold = B_row_size_half + B_row_offset;
        nnz_t B1_nnz = 0, B2_nnz;

        index_t *nnzPerCol_middle = &mempool2[0];
        std::fill(&nnzPerCol_middle[0], &nnzPerCol_middle[B_col_size], 0);
        // to avoid subtraction in the following for loop " - B_col_offset"
        index_t *nnzPerCol_middle_p = &nnzPerCol_middle[0] - B_col_offset;

        for (nnz_t i = 0; i < B_col_size; i++) {
            for (nnz_t j = nnzPerColScan_rightStart[i]; j < nnzPerColScan_rightEnd[i]; j++) {
                if (B[j].row < B_row_threshold) { // B[j].row - B_row_offset < B_row_size_half
                    nnzPerCol_middle_p[B[j].col]++;
                    B1_nnz++;
                }
            }
        }

        B2_nnz = B_nnz - B1_nnz;

#ifdef __DEBUG1__
//    print_vector(nnzPerCol_middle, -1, "nnzPerCol_middle", comm);
#endif

        std::vector<index_t> nnzPerColScan_middle(B_col_size + 1);
        for (nnz_t i = 0; i < B_col_size; i++) {
            nnzPerColScan_middle[i] = nnzPerColScan_rightStart[i] + nnzPerCol_middle[i];
        }

//    nnzPerCol_middle.clear();
//    nnzPerCol_middle.shrink_to_fit();

#ifdef __DEBUG1__
//        print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);
//        if(rank==0) printf("rank %d: A_nnz = %lu, A1_nnz = %lu, A2_nnz = %lu, B_nnz = %lu, B1_nnz = %lu, B2_nnz = %lu \n",
//                rank, A_nnz, A1_nnz, A2_nnz, B_nnz, B1_nnz, B2_nnz);
        if (rank == verbose_rank && verbose_matmat) { printf("fast_mm: case 2: step 1 \n"); }
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

            if (verbose_matmat_B) {
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
#endif

        // =======================================================
        // Call two recursive functions here. Put the result of the first one in C1, and the second one in C2.
        // merge sort them and add the result to C.
        std::vector<cooEntry> C_temp;

        // C1 = A1 * B1
#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 1 \n");
#endif

        if (A1_nnz == 0 || B1_nnz == 0) { // skip!
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A1_nnz == 0) {
                    printf("\nskip: A1_nnz == 0\n\n");
                } else {
                    printf("\nskip: B1_nnz == 0\n\n");
                }
            }
#endif
        } else {

            fast_mm(&A[0], &B[0], C, A1_nnz, B1_nnz,
                    A_row_size, A_row_offset, A_col_size_half, A_col_offset,
                    B_col_size, B_col_offset,
                    nnzPerColScan_leftStart, nnzPerColScan_leftEnd, // A1
                    nnzPerColScan_rightStart, &nnzPerColScan_middle[0], map_matmat, comm); // B1

        }


        // C2 = A2 * B2
#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 2 \n");
#endif

        if (A2_nnz == 0 || B2_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A2_nnz == 0) {
                    printf("\nskip: A2_nnz == 0\n\n");
                } else {
                    printf("\nskip: B2_nnz == 0\n\n");
                }
            }
#endif
            return;
        }

        fast_mm(&A[0], &B[0], C, A2_nnz, B2_nnz,
                A_row_size, A_row_offset, A_col_size - A_col_size_half, A_col_offset + A_col_size_half,
                B_col_size, B_col_offset,
                &nnzPerColScan_leftStart[A_col_size_half], &nnzPerColScan_leftEnd[A_col_size_half], // A2
                &nnzPerColScan_middle[0], nnzPerColScan_rightEnd, map_matmat, comm); // B2


#ifdef __DEBUG1__
//        print_vector(C1, -1, "C1", comm);
//        print_vector(C2, -1, "C2", comm);

//        if(rank==verbose_rank && verbose_matmat) {printf("fast_mm: case 2: step 3 \n");}
//        if(rank==0 && verbose_matmat) printf("C1.size() = %lu, C2.size() = %lu \n", C1.size(), C2.size());
#endif

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 2: end \n");
#endif

        // case3
        // ==============================================================

    } else { //DOLLAR("case3") // (A_row_size > A_col_size)

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 3: start \n");
#endif

        // split based on matrix size
        // =======================================================

#ifdef SPLIT_SIZE
        // prepare splits of matrix B by column
//        index_t A_col_size_half = A_col_size/2;
        index_t B_col_size_half = B_col_size/2;
        nnz_t B1_nnz = 0, B2_nnz;

        for(nnz_t i = 0; i < B_col_size_half; i++){
            B1_nnz += nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i];
        }

        B2_nnz = B_nnz - B1_nnz;
#endif

        // =======================================================
        // split based on nnz
        // =======================================================

#ifdef SPLIT_NNZ
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
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 1 \n");
#endif

        // prepare splits of matrix A by row
        nnz_t A1_nnz = 0, A2_nnz;
        index_t A_row_size_half = A_row_size / 2;
        index_t A_row_threshold = A_row_size_half + A_row_offset;

//    std::vector<index_t> nnzPerCol_middle(A_col_size, 0);
        index_t *nnzPerCol_middle = &mempool2[0];
        std::fill(&nnzPerCol_middle[0], &nnzPerCol_middle[A_col_size], 0);
        // to avoid subtraction in the following for loop " - B_col_offset"
        index_t *nnzPerCol_middle_p = &nnzPerCol_middle[0] - A_col_offset;

        for (nnz_t i = 0; i < A_col_size; i++) {
            for (nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftEnd[i]; j++) {
                if (A[j].row < A_row_threshold) { // A[j].row - A_row_offset < A_row_size_half
                    nnzPerCol_middle_p[A[j].col]++;
                    A1_nnz++;
                }
            }
        }

        A2_nnz = A_nnz - A1_nnz;

        std::vector<index_t> nnzPerColScan_middle(A_col_size);
        for (nnz_t i = 0; i < A_col_size; i++) {
            nnzPerColScan_middle[i] = nnzPerColScan_leftStart[i] + nnzPerCol_middle[i];
        }

//    nnzPerCol_middle.clear();
//    nnzPerCol_middle.shrink_to_fit();

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 3: step 2 \n");
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

            if (verbose_matmat_B) {
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

        // =======================================================
        // Save the result of 4 recursive functions in C_temp.

        // C1 = A1 * B1
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 1 \n");
#endif

        if (A1_nnz == 0 || B1_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A1_nnz == 0) {
                    printf("\nskip: A1_nnz == 0\n\n");
                } else {
                    printf("\nskip: B1_nnz == 0\n\n");
                }
            }
#endif
        } else {

            fast_mm(&A[0], &B[0], C, A1_nnz, B1_nnz,
                    A_row_size_half, A_row_offset, A_col_size, A_col_offset,
                    B_col_size_half, B_col_offset,
                    nnzPerColScan_leftStart, &nnzPerColScan_middle[0], // A1
                    nnzPerColScan_rightStart, nnzPerColScan_rightEnd, map_matmat, comm); // B1

        }


        // C2 = A1 * B2:
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 2 \n");
#endif

        if (A1_nnz == 0 || B2_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A1_nnz == 0) {
                    printf("\nskip: A1_nnz == 0\n\n");
                } else {
                    printf("\nskip: B2_nnz == 0\n\n");
                }
            }
#endif
        } else {

            fast_mm(&A[0], &B[0], C, A1_nnz, B2_nnz,
                    A_row_size_half, A_row_offset, A_col_size, A_col_offset,
                    B_col_size - B_col_size_half, B_col_offset + B_col_size_half,
                    nnzPerColScan_leftStart, &nnzPerColScan_middle[0], // A1
                    &nnzPerColScan_rightStart[B_col_size_half], &nnzPerColScan_rightEnd[B_col_size_half],
                    map_matmat, comm); // B2

        }


        // C3 = A2 * B1
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 3 \n");
#endif

        if (A2_nnz == 0 || B1_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A2_nnz == 0) {
                    printf("\nskip: A2_nnz == 0\n\n");
                } else {
                    printf("\nskip: B1_nnz == 0\n\n");
                }
            }
#endif
        } else {

            fast_mm(&A[0], &B[0], C, A2_nnz, B1_nnz,
                    A_row_size - A_row_size_half, A_row_offset + A_row_size_half, A_col_size, A_col_offset,
                    B_col_size_half, B_col_offset,
                    &nnzPerColScan_middle[0], nnzPerColScan_leftEnd, // A2
                    nnzPerColScan_rightStart, nnzPerColScan_rightEnd, map_matmat, comm); // B1

        }


        // C4 = A2 * B2
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 4 \n");
#endif

        if (A2_nnz == 0 || B2_nnz == 0) {
#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_matmat) {
                if (A2_nnz == 0) {
                    printf("\nskip: A2_nnz == 0\n\n");
                } else {
                    printf("\nskip: B2_nnz == 0\n\n");
                }
            }
#endif
        } else {

            fast_mm(&A[0], &B[0], C, A2_nnz, B2_nnz,
                    A_row_size - A_row_size_half, A_row_offset + A_row_size_half, A_col_size, A_col_offset,
                    B_col_size - B_col_size_half, B_col_offset + B_col_size_half,
                    &nnzPerColScan_middle[0], nnzPerColScan_leftEnd, // A2
                    &nnzPerColScan_rightStart[B_col_size_half], &nnzPerColScan_rightEnd[B_col_size_half],
                    map_matmat, comm); // B2

        }

        // C1 = A1 * B1:
//        fast_mm(A1, B1, C_temp, A_row_size_half, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size_half, B_col_offset, comm);
        // C2 = A2 * B1:
//        fast_mm(A2, B1, C_temp, A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset, B_row_offset, B_col_size_half, B_col_offset, comm);
        // C3 = A1 * B2:
//        fast_mm(A1, B2, C_temp, A_row_size_half, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size-B_col_size_half, B_col_offset+B_col_size_half, comm);
        // C4 = A2 * B2
//        fast_mm(A2, B2, C_temp, A_row_size-A_row_size_half, A_row_offset+A_row_size_half, A_col_size, A_col_offset, B_row_offset, B_col_size-B_col_size_half, B_col_offset+B_col_size_half, comm);

//        if(rank==0 && verbose_matmat) printf("fast_mm: case 3: step 4 \n");

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat) printf("fast_mm: case 3: end \n");
#endif

    }

//    return;
}
*/


// before switching to CSC. here row and val are stored together.
/*
int saena_object::matmat_ave(saena_matrix *A, saena_matrix *B, double &matmat_time){
    // This version only works on symmetric matrices, since local transpose of B is being used.
    // this version is only for experiments.
    // B1 should be symmetric. Because we need its transpose. Treat its row indices as column indices and vice versa.

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // =======================================
    // Convert A to CSC
    // =======================================

    auto Arv = new vecEntry[A->nnz_l]; // row and val
    auto Ac  = new index_t[A->Mbig+1]; // col_idx

    // todo: change to smart pointers
//    auto Arv = std::make_unique<vecEntry[]>(A->nnz_l); // row and val
//    auto Ac  = std::make_unique<index_t[]>(A->Mbig+1); // col_idx

    for(nnz_t i = 0; i < A->entry.size(); i++){
        Arv[i] = vecEntry(A->entry[i].row, A->entry[i].val);
    }

    std::fill(&Ac[0], &Ac[A->Mbig+1], 0);
    index_t *Ac_tmp = &Ac[1];
    for(auto ent:A->entry){
        Ac_tmp[ent.col]++;
    }

    for(nnz_t i = 0; i < A->Mbig; i++){
        Ac[i+1] += Ac[i];
    }

    nnz_t A_nnz = A->nnz_l;

#ifdef __DEBUG1__
//    A->print_entry(0);
//    printf("A: nnz_l: %ld\tnnz_g: %ld\tM: %d\tM_big: %d\n", A->nnz_l, A->nnz_g, A->M, A->Mbig);
//    print_array(Ac, A->Mbig+1, 0, "Ac", comm);

//    std::cout << "\nA: nnz: " << A->nnz_l << std::endl ;
//    for(index_t j = 0; j < A->Mbig; j++){
//        for(index_t i = Ac[j]; i < Ac[j+1]; i++){
//            std::cout << std::setprecision(4) << Arv[i].row << "\t" << j << "\t" << Arv[i].val << std::endl;
//        }
//    }
#endif

    // =======================================
    // Convert the local transpose of B to CSC
    // =======================================

//    print_vector(B->entry, 0, "B col-major", comm);
    std::sort(B->entry.begin(), B->entry.end(), row_major);
//    print_vector(B->entry, 0, "B row-major", comm);

    auto Brv = new vecEntry[B->nnz_l]; // row and val
    auto Bc  = new index_t[B->M+1];    // col_idx

    // todo: change to smart pointers
//    auto Brv = std::make_unique<vecEntry[]>(B->nnz_l); // row (actually col to have the transpose) and val
//    auto Bc  = std::make_unique<index_t[]>(B->M+1); // col_idx

    for(nnz_t i = 0; i < B->entry.size(); i++){
        Brv[i] = vecEntry(B->entry[i].col, B->entry[i].val);
    }

    std::fill(&Bc[0], &Bc[B->M+1], 0);
    index_t *Bc_tmp   = &Bc[1];
    index_t *Bc_tmp_p = &Bc_tmp[0] - B->split[rank]; // use this to avoid subtracting a fixed number,
    for(auto ent:B->entry){
        Bc_tmp_p[ent.row]++;
    }

    for(nnz_t i = 0; i < B->M; i++){
        Bc[i+1] += Bc[i];
    }

    nnz_t B_nnz = B->nnz_l;

#ifdef __DEBUG1__
//    B->print_entry(0);
//    printf("B: nnz_l: %ld\tnnz_g: %ld\tM: %d\tM_big: %d\n", B->nnz_l, B->nnz_g, B->M, B->Mbig);
//    print_array(Bc, B->M+1, 0, "Bc", comm);
//
//    std::cout << "\nB: nnz: " << B->nnz_l << std::endl ;
//    for(index_t j = 0; j < B->M; j++){
//        for(index_t i = Bc[j]; i < Bc[j+1]; i++){
//            std::cout << std::setprecision(4) << Brv[i].row << "\t" << j << "\t" << Brv[i].val << std::endl;
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
    index_t B_col_size = B->M;      // for when tranpose of B is used to do the multiplication.

    mempool1 = new value_t[matmat_size_thre2];
    mempool2 = new index_t[2 * A_row_size + 2 * B_col_size];

    // 2 for both send and receive buffer, vecbyint for row and value, (B->max_M + 1) for col_scan
    int vecbyint              = sizeof(vecEntry) / sizeof(index_t);
    nnz_t rv_buffer_sz_max    = vecbyint * B->nnz_max;
    nnz_t cscan_buffer_sz_max = B->max_M + 1;
    nnz_t send_size_max       = rv_buffer_sz_max + cscan_buffer_sz_max;
    auto mempool3             = new index_t[2 * (send_size_max)];

//    mempool1 = std::make_unique<value_t[]>(matmat_size_thre2);
//    mempool2 = std::make_unique<index_t[]>(A->Mbig * 4);

    if(2 * sizeof(index_t) != sizeof(double)){
        if(rank==0) std::cout << __func__ << ": 2 * sizeof(index_t) != sizeof(double)" << std::endl;
    }

#ifdef __DEBUG1__
//    if(rank==0) std::cout << "vecbyint = " << vecbyint << std::endl;
#endif

    // =======================================
    // perform the multiplication - serial implementation
    // =======================================

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

    MPI_Barrier(comm);
    double t_AP = MPI_Wtime();

#ifdef __DEBUG1__
//    if(rank==0) std::cout << "sizeof(index_t) = " << sizeof(index_t) << ", sizeof(value_t) = " << sizeof(value_t)
//                          << ", sizeof(vecEntry) = "<< sizeof(vecEntry)
//                          << ", sizeof(cooEntry) = "<< sizeof(cooEntry)<< std::endl;
#endif

    // set the mat_send data
    nnz_t send_nnz  = B_nnz;
    nnz_t send_size = vecbyint * send_nnz + B_col_size + 1;
//    unsigned long send_size_max = send_buffer_sz_max;

    auto mat_send       = &mempool3[0];
    auto mat_send_rv    = reinterpret_cast<vecEntry*>(&mat_send[0]);
    auto mat_send_cscan = &mat_send[vecbyint * send_nnz];
//    auto mat_send_cscan = reinterpret_cast<index_t*>(&mempool3[rv_buffer_sz_max]);

//    for(nnz_t i = 0; i < B_nnz; i++){
//        mat_send_rv[i] = Brv[i];
//    }
    memcpy(mat_send_rv,    Brv, B_nnz * sizeof(vecEntry));
    memcpy(mat_send_cscan, Bc,  (B_col_size + 1) * sizeof(index_t));

#ifdef __DEBUG1__
//    MPI_Barrier(comm);
//    printf("send_size = %lu, send_size_max = %lu\n", send_size, send_size_max);
//    print_array(mat_send_cscan, B_col_size+1, 1, "mat_send_cscan", comm);
//    MPI_Barrier(comm);
#endif

//    double tt;
//    double t_swap = 0;
//    index_t *nnzPerColScan_left = &A->nnzPerColScan[0];
//    index_t mat_recv_M_max      = B->max_M;

//    std::vector<index_t> nnzPerColScan_right(mat_recv_M_max + 1);
//    index_t *nnzPerCol_right   = &nnzPerColScan_right[1];
//    index_t *nnzPerCol_right_p = &nnzPerCol_right[0]; // use this to avoid subtracting a fixed number,

    std::vector<cooEntry> AB_temp;
    std::vector<cooEntry> AB; // this won't be used.

//    printf("\n");
    if(nprocs > 1){
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
        int owner, next_owner;
        auto *requests = new MPI_Request[2];
        auto *statuses = new MPI_Status[2];

        // todo: the last communication means each proc receives a copy of its already owned B, which is redundant,
        // so the communication should be avoided but the multiplication should be done. consider this:
        // change to k < rank+nprocs-1. Then, copy fast_mm after the end of the for loop to perform the last multiplication
        for(int k = rank; k < rank+nprocs; k++){
            // This is overlapped. Both local and remote loops are done here.
            // The first iteration is the local loop. The rest are remote.
            // Send R_tranpose to the left_neighbor processor, receive R_tranpose from the right_neighbor.
            // In the next step: send R_tranpose that was received in the previous step to the left_neighbor processor,
            // receive R_tranpose from the right_neighbor. And so on.
            // --------------------------------------------------------------------

            next_owner = (k+1)%nprocs;
            mat_recv_M = B->split[next_owner + 1] - B->split[next_owner];
            recv_nnz   = B->nnz_list[next_owner];
            recv_size  = vecbyint * recv_nnz + mat_recv_M + 1;

#ifdef __DEBUG1__
//            if(rank==0) std::cout << "\n===================" << __func__ << ", k = " << k << "===================\n";
//            if(rank==0) std::cout << "\nnext_owner: " << next_owner << ", recv_size: " << recv_size << ", mat_recv_M: "
//                        << mat_recv_M << ", recv_nnz: " << recv_nnz << std::endl;

//            printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
//            mat_recv.resize(recv_size);
#endif

            // communicate data
            MPI_Irecv(&mat_recv[0], recv_size, MPI_UNSIGNED, right_neighbor, right_neighbor, comm, requests);
            MPI_Isend(&mat_send[0], send_size, MPI_UNSIGNED, left_neighbor,  rank,           comm, requests+1);

//            owner = k%nprocs;
//            mat_recv_M = B->split[owner + 1] - B->split[owner];

            if(A->entry.empty() || send_nnz==0){ // skip!
#ifdef __DEBUG1__
                if(verbose_matmat){
                    if(A->entry.empty()){
                        printf("\nskip: A->entry.size() == 0\n\n");
                    } else {
                        printf("\nskip: send_nnz == 0\n\n");
                    }
                }
#endif
            } else {

//                fast_mm(&A->entry[0], &mat_send[0], AB_temp, A->entry.size(), send_size,
//                        A->M, A->split[rank], A->Mbig, 0, mat_recv_M, B->split[owner],
//                        &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
//                        &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

                owner         = k%nprocs;
                mat_current_M = B->split[owner + 1] - B->split[owner];

                fast_mm(Arv, &mat_send_rv[0], AB_temp,
                        A->M, A->split[rank], A->Mbig, 0, mat_current_M, B->split[owner],
                        &Ac[0], &mat_send_cscan[0], A->comm);

            }

            MPI_Waitall(2, requests, statuses);


//            mat_recv_cscan = &mat_recv[rv_buffer_sz_max];
//            MPI_Barrier(comm);
//            if(rank==0){
//                print_array(mat_recv_cscan, mat_recv_M+1, 0, "mat_recv_cscan", comm);
//            }
//            MPI_Barrier(comm);

            send_size = recv_size;
            send_nnz  = recv_nnz;

//            mat_recv.swap(mat_send);
//            std::swap(mat_send, mat_recv);
            mat_temp = mat_send;
            mat_send = mat_recv;
            mat_recv = mat_temp;

            mat_send_rv    = reinterpret_cast<vecEntry*>(&mat_send[0]);
            mat_send_cscan = &mat_send[vecbyint * send_nnz];
//            mat_recv_rv    = reinterpret_cast<vecEntry*>(&mat_recv[0]);
//            mat_recv_cscan = &mat_recv[rv_buffer_sz_max];

#ifdef __DEBUG1__
//            MPI_Barrier(comm);
//            if(rank==0){
//                std::cout << "print received matrix: mat_recv_M: " << mat_recv_M << ", col_offset: "
//                          << B->split[k%nprocs] << std::endl;

//                print_array(mat_send_cscan, mat_recv_M+1, 0, "mat_send_cscan", comm);

//                for(nnz_t i = 0; i < mat_recv_M; i++){
//                    for(nnz_t j = mat_send_cscan[i]; j < mat_send_cscan[i+1]; j++) {
//                        std::cout << j << "\t" << mat_send_rv[j].row << "\t" << i + B->split[k%nprocs] << "\t" << mat_send_rv[j].val << std::endl;
//                    }
//                }
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

    } else { // nprocs == 1 -> serial

        if(A->entry.empty() || send_nnz == 0){ // skip!
#ifdef __DEBUG1__
            if(verbose_matmat){
                if(A->entry.empty()){
                    printf("\nskip: A->entry.size() == 0\n\n");
                } else {
                    printf("\nskip: send_nnz == 0\n\n");
                }
            }
#endif
        } else {

//            index_t mat_recv_M = B->split[rank + 1] - B->split[rank];

//            double t1 = MPI_Wtime();

//            fast_mm(&A->entry[0], &mat_send[0], AB_temp, A->entry.size(), send_size,
//                    A->M, A->split[rank], A->Mbig, 0, mat_recv_M, B->split[rank],
//                    &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
//                    &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

            fast_mm(Arv, Brv, AB_temp,
                    A->M, A->split[rank], A->Mbig, 0, B_col_size, B->split[rank],
                    &Ac[0], &Bc[0], A->comm);

//            double t2 = MPI_Wtime();
//            printf("\nfast_mm of AB_temp = %f\n", t2-t1);
        }
    }

    // =======================================
    // sort and remove duplicates
    // =======================================

//    MPI_Barrier(comm);
//    t1 = MPI_Wtime();

    std::sort(AB_temp.begin(), AB_temp.end());

    nnz_t AP_temp_size_minus1 = AB_temp.size()-1;
    for(nnz_t i = 0; i < AB_temp.size(); i++){
        AB.emplace_back(AB_temp[i]);
        while(i < AP_temp_size_minus1 && AB_temp[i] == AB_temp[i+1]){ // values of entries with the same row and col should be added.
//                std::cout << AB_temp[i] << "\t" << AB_temp[i+1] << std::endl;
            AB.back().val += AB_temp[++i].val;
        }
    }
//    AB_temp.clear();

//    t1 = MPI_Wtime() - t1;
//    print_time_ave(t1, "AB:", comm);

#ifdef __DEBUG1__
//    print_vector(AB, -1, "AB", comm);
//    writeMatrixToFile(AB, "matrix_folder/result", comm);
#endif

    t_AP = MPI_Wtime() - t_AP;
    matmat_time += print_time_ave_consecutive(t_AP, comm);
//    matmat_time += t_ave;

    // =======================================
    // finalize
    // =======================================

//    mat_send.clear();
//    mat_send.shrink_to_fit();
//    AB_temp.clear();
//    AB_temp.shrink_to_fit();

    delete []Arv;
    delete []Ac;
    delete []Brv;
    delete []Bc;

    return 0;
}
*/


// before switching to CSC
//int saena_object::matmat_ave(saena_matrix *A, saena_matrix *B, double &matmat_time)
/*
int saena_object::matmat_ave(saena_matrix *A, saena_matrix *B, double &matmat_time){
    // This version only works on symmetric matrices, since local transpose of B is being used.
    // this version is only for experiments.
    // B1 should be symmetric. Because we need its transpose. Treat its row indices as column indices and vice versa.

//    saena_matrix *A = A1.get_internal_matrix();
//    saena_matrix *B = B1.get_internal_matrix();

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    bool verbose_matmat = false;

    mempool1 = new value_t[matmat_size_thre2];
    mempool2 = new index_t[A->Mbig * 4];
    mempool3 = new cooEntry[B->nnz_max * 2];

//    MPI_Barrier(comm);
//    double t_AP = MPI_Wtime();

//    MPI_Barrier(comm);
//    double t1 = MPI_Wtime();

    unsigned long send_size     = B->entry.size();
    unsigned long send_size_max = B->nnz_max;
//    MPI_Allreduce(&send_size, &send_size_max, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);
//    printf("send_size = %lu, send_size_max = %lu\n", send_size, send_size_max);

    // local transpose of R is being used to compute A*P. So R is transposed locally here.
//    std::vector<cooEntry> mat_send(R->entry.size());
//    auto mat_send = new cooEntry[send_size_max];
    auto mat_send = &mempool3[0];
    auto mat_temp = mat_send; // use this to swap mat_send and mat_recv
//    transpose_locally(&R->entry[0], R->entry.size(), R->splitNew[rank], &mat_send[0]);
//    memcpy(&mat_send[0], &B->entry[0], B->entry.size() * sizeof(cooEntry));

//    std::sort(B->entry.begin(), B->entry.end(), row_major);

    // setting the local tranpose of B in mat_send
    // ===========================================
    for(nnz_t i = 0; i < B->entry.size(); i++){
        mat_send[i] = cooEntry(B->entry[i].col, B->entry[i].row, B->entry[i].val);
//        if(rank==0) std::cout << mat_send[i] << std::endl;
    }

    std::sort(&mat_send[0], &mat_send[B->entry.size()]);

//    t1 = MPI_Wtime() - t1;
//    print_time_ave(t1, "mat_send:", comm);

#ifdef __DEBUG1__
//    print_vector(A->entry, 0, "A->entry", comm);
//    print_vector(A->nnzPerColScan, 0, "A->nnzPerColScan", comm);
#endif

//    MPI_Barrier(comm);
//    t1 = MPI_Wtime() - t1;
    double tt;
    double t_swap = 0;
    index_t *nnzPerColScan_left = &A->nnzPerColScan[0];
    index_t mat_recv_M_max      = B->max_M;

    std::vector<index_t> nnzPerColScan_right(mat_recv_M_max + 1);
    index_t *nnzPerCol_right   = &nnzPerColScan_right[1];
    index_t *nnzPerCol_right_p = &nnzPerCol_right[0]; // use this to avoid subtracting a fixed number,

    std::vector<cooEntry> AB_temp;

//    printf("\n");
    if(nprocs > 1){

        int right_neighbor = (rank + 1)%nprocs;
        int left_neighbor  = rank - 1;
        if (left_neighbor < 0){
            left_neighbor += nprocs;
        }

        int owner;
        unsigned long recv_size;
//        std::vector<cooEntry> mat_recv;
//        auto mat_recv = new cooEntry[send_size_max];
        auto mat_recv = &mempool3[send_size_max];
        index_t mat_recv_M;

        auto *requests = new MPI_Request[2];
        auto *statuses = new MPI_Status[2];

        for(int k = rank; k < rank+nprocs; k++){
            // This is overlapped. Both local and remote loops are done here.
            // The first iteration is the local loop. The rest are remote.
            // Send R_tranpose to the left_neighbor processor, receive R_tranpose from the right_neighbor.
            // In the next step: send R_tranpose that was received in the previous step to the left_neighbor processor,
            // receive R_tranpose from the right_neighbor. And so on.
            // --------------------------------------------------------------------

            // communicate size
//            MPI_Irecv(&recv_size, 1, MPI_UNSIGNED_LONG, right_neighbor, right_neighbor, comm, requests);
//            MPI_Isend(&send_size, 1, MPI_UNSIGNED_LONG, left_neighbor,  rank,           comm, requests+1);
//            MPI_Waitall(1, requests, statuses);

            recv_size = B->nnz_list[(k+1) % nprocs];
//            printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
//            mat_recv.resize(recv_size);

            // communicate data
            MPI_Irecv(&mat_recv[0], recv_size, cooEntry::mpi_datatype(), right_neighbor, right_neighbor, comm, requests);
            MPI_Isend(&mat_send[0], send_size, cooEntry::mpi_datatype(), left_neighbor,  rank,           comm, requests+1);

            owner = k%nprocs;
            mat_recv_M = B->split[owner + 1] - B->split[owner];

            std::fill(&nnzPerCol_right[0], &nnzPerCol_right[mat_recv_M], 0);
            nnzPerCol_right_p = &nnzPerCol_right[0] - B->split[owner];
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
                if(verbose_matmat){
                    if(A->entry.empty()){
                        printf("\nskip: A->entry.size() == 0\n\n");
                    } else {
                        printf("\nskip: mat_send == 0\n\n");
                    }
                }
#endif
            } else {

                fast_mm(&A->entry[0], &mat_send[0], AB_temp, A->entry.size(), send_size,
                        A->M, A->split[rank], A->Mbig, 0, mat_recv_M, B->split[owner],
                        &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
                        &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

            }

//            mat_recv.swap(mat_send);
//            std::swap(mat_send, mat_recv);
            mat_temp = mat_send;
            mat_send = mat_recv;
            mat_recv = mat_temp;
            send_size = recv_size;

            MPI_Waitall(2, requests, statuses);

#ifdef __DEBUG1__
//          print_vector(AB_temp, -1, "AB_temp", A->comm);
//          print_vector(mat_send, -1, "mat_send", A->comm);
//          print_vector(mat_recv, -1, "mat_recv", A->comm);
//          prev_owner = owner;
//          printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
#endif

        }

//        mat_recv.clear();
//        mat_recv.shrink_to_fit();
//        delete [] mat_recv;
        delete [] requests;
        delete [] statuses;

    } else { // nprocs == 1 -> serial

        index_t mat_recv_M = B->split[rank + 1] - B->split[rank];

        std::fill(&nnzPerCol_right[0], &nnzPerCol_right[mat_recv_M], 0);
        nnzPerCol_right_p = &nnzPerCol_right[0] - B->split[rank];
        for(nnz_t i = 0; i < send_size; i++){
            nnzPerCol_right_p[mat_send[i].col]++;
        }

//        nnzPerColScan_right[0] = 0;
        for(nnz_t i = 0; i < mat_recv_M; i++){
            nnzPerColScan_right[i+1] += nnzPerColScan_right[i];
        }

#ifdef __DEBUG1__
//          print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
//          print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

        if(A->entry.empty() || send_size == 0){ // skip!
#ifdef __DEBUG1__
            if(verbose_matmat){
                if(A->entry.empty()){
                    printf("\nskip: A->entry.size() == 0\n\n");
                } else {
                    printf("\nskip: mat_send == 0\n\n");
                }
            }
#endif
        } else {

//            double t1 = MPI_Wtime();

            fast_mm(&A->entry[0], &mat_send[0], AB_temp, A->entry.size(), send_size,
                    A->M, A->split[rank], A->Mbig, 0, mat_recv_M, B->split[rank],
                    &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
                    &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//            double t2 = MPI_Wtime();
//            printf("\nfast_mm of AB_temp = %f\n", t2-t1);
        }
    }

//    print_vector(AB_temp, 0, "AB_temp", comm);

//    t1 = MPI_Wtime() - t1;
//    print_time_ave(t1, "AB_temp:", comm);

//    MPI_Barrier(comm);
//    t1 = MPI_Wtime();

    std::sort(AB_temp.begin(), AB_temp.end());
//    std::vector<cooEntry> AB;
//    nnz_t AP_temp_size_minus1 = AB_temp.size()-1;
//    for(nnz_t i = 0; i < AB_temp.size(); i++){
//        AB.emplace_back(AB_temp[i]);
//        while(i < AP_temp_size_minus1 && AB_temp[i] == AB_temp[i+1]){ // values of entries with the same row and col should be added.
//            AB.back().val += AB_temp[++i].val;
//        }
//    }

    nnz_t AP_temp_size_minus1 = AB_temp.size()-1;
    std::vector<cooEntry> AB; // this won't be used.
    for(nnz_t i = 0; i < AB_temp.size(); i++){
        AB.emplace_back(AB_temp[i]);
        while(i < AP_temp_size_minus1 && AB_temp[i] == AB_temp[i+1]){ // values of entries with the same row and col should be added.
//            std::cout << AB_temp[i] << "\t" << AB_temp[i+1] << std::endl;
            AB.back().val += AB_temp[++i].val;
        }
    }

//    t1 = MPI_Wtime() - t1;
//    print_time_ave(t1, "AB:", comm);

//    mat_send.clear();
//    mat_send.shrink_to_fit();
    AB_temp.clear();
    AB_temp.shrink_to_fit();
//    delete [] mat_send;

//    unsigned long AP_size_loc = AB.size(), AP_size;
//    MPI_Reduce(&AP_size_loc, &AP_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);
//    if(!rank) printf("A_nnz_g = %lu, \tP_nnz_g = %lu, \tAP_size = %lu\n", A->nnz_g, P->nnz_g, AP_size);

//    t_AP = MPI_Wtime() - t_AP;
//    matmat_time += print_time_ave_consecutive(t_AP, comm);
//    matmat_time += t_ave;

    delete[] mempool1;
    delete[] mempool2;
    delete[] mempool3;

#ifdef __DEBUG1__
//    print_vector(C->entry, -1, "AB = C", A->comm);
    if(verbose_matmat){
        MPI_Barrier(comm); printf("compute_coarsen: step 5: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

//    if(rank==0) printf("\nAB:\n");
//    if(rank==0) dollar::text(std::cout);
//    dollar::clear();

    // AB is a vector. Create a matrix for which AB is its entry() variable.
//    AB->repartition_nnz();
//    if(AB->active){
//        AB->matrix_setup();
//    }
//    petsc_viewer(AB);

    return 0;
}
*/


//void saena_object::matmat(const cooEntry *A, const cooEntry *B, std::vector<cooEntry> &C,
//                          nnz_t A_nnz, nnz_t B_nnz, index_t A_row_size, index_t A_col_size, index_t B_col_size,
//                          const index_t *nnzPerRowScan_left, const index_t *nnzPerColScan_right, MPI_Comm comm)
/*
void saena_object::matmat(const cooEntry *A, const cooEntry *B, std::vector<cooEntry> &C,
                          nnz_t A_nnz, nnz_t B_nnz, index_t A_row_size, index_t A_col_size, index_t B_col_size,
                          const index_t *nnzPerRowScan_left, const index_t *nnzPerColScan_right, MPI_Comm comm){

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
#ifdef __DEBUG1__
    if(rank==verbose_rank && verbose_matmat) printf("\nfast_mm_basic: start \n");
#endif

//    if(A_nnz == 0 || B_nnz == 0){
//#ifdef __DEBUG1__
//        if(rank==verbose_rank && verbose_matmat) printf("\nskip: A_nnz == 0 || B_nnz == 0\n\n");
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
            std::cout << "A_row_size = "     << A_row_size   << ", A_col_size = "   << A_col_size << std::endl;

            // print entries of A:
//            std::cout << "\nA: nnz = " << A_nnz << std::endl;
//            for(nnz_t i = 0; i < A_col_size; i++){
//                for(nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftEnd[i]; j++) {
//                    std::cout << j << "\t" << A[j] << std::endl;
//                }
//            }
        }

        if(verbose_matmat_B) {
            std::cout << "\nB: nnz = " << B_nnz << std::endl;
            std::cout << "B_row_size = " << A_col_size << ", B_col_size = " << B_col_size << std::endl;

            // print entries of B:
//            std::cout << "\nB: nnz = " << B_nnz << std::endl;
//            for (nnz_t i = 0; i < B_col_size; i++) {
//                for (nnz_t j = nnzPerColScan_rightStart[i]; j < nnzPerColScan_rightEnd[i]; j++) {
//                    std::cout << j << "\t" << B[j] << std::endl;
//                }
//            }
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

    std::vector<cooEntry> A_vec(A, A + A_nnz);
    std::sort(A_vec.begin(), A_vec.end(), row_major);

//    for(nnz_t i = 0; i < A_nnz; i++){
//        std::cout << A_vec[i] << std::endl;
//    }

    double C_val = 0;
    for(nnz_t i = 0; i < A_row_size; i++){
        for(nnz_t j = nnzPerRowScan_left[i]; j < nnzPerRowScan_left[i+1]; j++){
            for(nnz_t k = 0; k < B_col_size; k++){
                C_val = 0;
                for(nnz_t l = nnzPerColScan_right[k]; l < nnzPerColScan_right[k+1]; l++){
                    C_val += A[j].val * B[l].val;
                }
                C.emplace_back(A[i].row, B[k].col, C_val);
            }
        }
    }

//    return;
}
*/


// this version is based in in-place quicksort. The issue with this is that the final blocks are not ordered.
//int saena_object::reorder_split(vecEntry *arr, index_t *Ac1, index_t *Ac2, index_t col_sz, index_t pivot)
/*
int saena_object::reorder_split(vecEntry *arr, index_t *Ac1, index_t *Ac2, index_t col_sz, index_t pivot){
    // nnz is Ac1[col_sz]

    nnz_t nnz = Ac1[col_sz];
    vecEntry tmp = arr[0];
    bool cont = true;

    std::fill(&Ac2[0], &Ac2[col_sz+1], 0);

    auto col = new index_t[nnz]; // COO column idx

    nnz_t k   = 0;
    index_t i = Ac1[k];

    nnz_t l   = col_sz - 1;
    index_t j = Ac1[l+1] - 1;

#ifdef __DEBUG1__
//    print_array(Ac1, col_sz+1, 0, "Ac", MPI_COMM_WORLD);
#endif

    //    std::cout << "pivot: " << pivot << std::endl;
    while (i <= j) {

        cont = true;
        while(k < col_sz){ // col index
            while(i < Ac1[k+1]){ //nonzero index
//                std::cout << i << "\t" << arr[i].row << "\t" << k << "\t" << arr[i].val << std::endl;
                if(arr[i].row >= pivot){
                    cont = false;
                    break;
                }
                col[i] = k;
                i++;
            }

            if(!cont){
                break;
            }

            k++;
            i = Ac1[k];
        }
//        std::cout << "\nfinal i: " << i << "\t" << k << std::endl;

        cont = true;
        while(l >= 0){ // col index
            while(j >= Ac1[l]) { //nonzero index
//                 std::cout << j << "\t" << arr[j].row << "\t" << l << "\t" << arr[j].val << std::endl;
                if(arr[j].row < pivot){
                    cont = false;
                    break;
                }

                // since a scan operation will be done on this at the end, 1 is added:
                Ac2[l+1]++;
                col[j] = l;
                j--;
            }

            if(!cont){
                break;
            }

            l--;
            j = Ac1[l+1] - 1;
        }
//        std::cout << "final j: " << j << "\t" << l << std::endl;

        if (i <= j) {
//            std::cout << "i: " << i << "\tj: " << j << "\tk: " << k << "\tl+1: " << l+1 << std::endl;
            tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;

            Ac2[k+1]++;
            col[i] = l;
            col[j] = k;

            i++;
            j--;
        }

    }

#ifdef __DEBUG1__
//    print_array(Ac1, col_sz+1, 0, "Ac1", MPI_COMM_WORLD);
//    print_array(Ac2, col_sz+1, 0, "Ac2", MPI_COMM_WORLD);
#endif

    for(i = 1; i <= col_sz; i++){
        Ac2[i] += Ac2[i-1];
        Ac1[i] -= Ac2[i];
    }

#ifdef __DEBUG1__
//    print_array(Ac1, col_sz+1, 0, "Ac1", MPI_COMM_WORLD);
//    print_array(Ac2, col_sz+1, 0, "Ac2", MPI_COMM_WORLD);

    // ========================================================
    // this shows how to go through entries of A, A1 (top half) and A2 (bottom half) after changing order.
    // ========================================================
//    std::cout << "\nmatrix after reordering: " << std::endl;
//    for(nnz_t i = 0; i < nnz; i++){
//        std::cout << arr[i].row << "\t" << col[i] << "\t" << arr[i].val << std::endl;
//    }

    std::cout << "\ntop half: " << std::endl;
    for(nnz_t i = 0; i < Ac1[col_sz]; i++){
        std::cout << arr[i].row << "\t" << col[i] << "\t" << arr[i].val << std::endl;
    }

    std::cout << "\nbottom half: " << std::endl;
    for(nnz_t i = Ac1[col_sz]; i < Ac1[col_sz] + Ac2[col_sz]; i++){
        std::cout << arr[i].row << "\t" << col[i] << "\t" << arr[i].val << std::endl;
    }
    // ========================================================
#endif

    auto mat_vecCol = new vecCol[nnz];
    for(l = 0; l < nnz; l++){
        mat_vecCol[l] = vecCol(&arr[l], &col[l]);
    }

//    std::cout << "\nmatrix after reordering:" << std::endl;
//    for(l = 0; l < nnz; l++){
//        std::cout << mat_vecCol[l] << std::endl;
//    }

//    std::cout << "test: " << (mat_vecCol[0] > mat_vecCol[1]) << std::endl;

    std::sort(&mat_vecCol[0], &mat_vecCol[nnz], vecCol_col_major);

    for(l = 0; l < nnz; l++){
        arr[i].row = mat_vecCol[l].rv->row;
        arr[i].val = mat_vecCol[l].rv->val;
    }

//    std::cout << "\nafter sort:" << std::endl;
//    for(l = 0; l < nnz; l++){
//        std::cout << mat_vecCol[l] << std::endl;
//    }

//    vecCol mat(arr, col);
//    std::cout << mat << std::endl;

//    vecCol mat2(arr+1, col+1);
//    std::cout << mat2 << std::endl;

//    std::cout << "test: " << (mat == mat2) << std::endl;

//    for(i = 0; i < nnz; i++){
//        std::cout << mat << std::endl;
//    }

#ifdef __DEBUG1__
    // ========================================================
    // A, A1 (top half) and A2 (bottom half) after sorting
    // ========================================================
//    std::cout << "\nmatrix after sorting: " << std::endl;
//    std::cout << "\ntop half: " << std::endl;
//    for(index_t j = 0; j < col_sz; j++){
//        for(index_t i = Ac1[j]; i < Ac1[j+1]; i++){
//            std::cout << std::setprecision(4) << arr[i].row << "\t" << j << "\t" << arr[i].val << std::endl;
//        }
//    }
//    std::cout << "\nbottom half: " << std::endl;
//    for(index_t j = 0; j < col_sz; j++){
//        for(index_t i = Ac1[col_sz] + Ac2[j]; i < Ac1[col_sz] + Ac2[j+1]; i++){
//            std::cout << std::setprecision(4) << arr[i].row << "\t" << j << "\t" << arr[i].val << std::endl;
//        }
//    }

//    for(i = 0; i < nnz; i++){
//        std::cout << i << "\t" << arr[i].row << "\t" << col[i] << "\t" << arr[i].val << std::endl;
//    }
//
//    std::cout << "\ntop half: " << std::endl;
//    for(i = 0; i < Ac1[col_sz]; i++){
//        std::cout << i << "\t" << arr[i].row << "\t" << col[i] << "\t" << arr[i].val << std::endl;
//    }
//
//    std::cout << "\nbottom half: " << std::endl;
//    for(i = Ac1[col_sz]; i < Ac1[col_sz] + Ac2[col_sz]; i++){
//        std::cout << i << "\t" << arr[i].row << "\t" << col[i] << "\t" << arr[i].val << std::endl;
//    }
    // ========================================================
#endif

    delete []col;
    delete []mat_vecCol;
    return 0;
}
*/


//int saena_object::triple_mat_mult_basic(Grid *grid, std::vector<cooEntry_row> &RAP_row_sorted)
/*
int saena_object::triple_mat_mult_basic(Grid *grid, std::vector<cooEntry_row> &RAP_row_sorted){

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

    // local transpose of R is being used to compute A*P. So R is transposed locally here.
    std::vector<cooEntry> mat_send(R->entry.size());
    transpose_locally(R->entry, R->entry.size(), R->splitNew[rank], mat_send);

#ifdef __DEBUG1__
//    print_vector(R->entry, -1, "R->entry", comm);
//    print_vector(R_tranpose, -1, "mat_send", comm);
#endif

//    index_t *nnzPerRow_left = &mempool2[0];
//    std::fill(&nnzPerRow_left[0], &nnzPerRow_left[A_row_size], 0);
    std::vector<index_t> nnzPerRow_left(A->M, 0);
    index_t *nnzPerRow_left_p = &nnzPerRow_left[0] - A->split[rank];
    for(nnz_t i = 0; i < A->entry.size(); i++){
        nnzPerRow_left_p[A->entry[i].row]++;
    }

#ifdef __DEBUG1__
//    print_vector(A->entry, 1, "A->entry", comm);
//    print_vector(nnzPerCol_left, 1, "nnzPerCol_left", comm);
#endif

    std::vector<index_t> nnzPerRowScan_left(A->M+1);
    nnzPerRowScan_left[0] = 0;
    for(nnz_t i = 0; i < A->Mbig; i++){
        nnzPerRowScan_left[i+1] = nnzPerRowScan_left[i] + nnzPerRow_left[i];
    }

    nnzPerRow_left.clear();
//    nnzPerCol_left.shrink_to_fit();

#ifdef __DEBUG1__
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

//    print_vector(P->splitNew, -1, "P->splitNew", comm);

    if(verbose_triple_mat_mult){
        MPI_Barrier(comm); printf("compute_coarsen: step 4: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // compute the maximum size for nnzPerCol_right and nnzPerColScan_right
    index_t mat_recv_M_max = 0;
    for(index_t i = 0; i < nprocs; i++){
        if(P->splitNew[i+1] - P->splitNew[i] > mat_recv_M_max){
            mat_recv_M_max = P->splitNew[i+1] - P->splitNew[i];
        }
    }

    // use this for fast_mm case1
//    std::unordered_map<index_t, value_t> map_matmat;
//    map_matmat.reserve(matmat_size_thre2);

    std::vector<index_t> nnzPerCol_right(mat_recv_M_max); // range of rows of R is range of cols of R_transpose.
    index_t *nnzPerCol_right_p = &nnzPerCol_right[0]; // use this to avoid subtracting a fixed number,
    std::vector<index_t> nnzPerColScan_right(mat_recv_M_max + 1);
    std::vector<cooEntry> AP;

    if(nprocs > 1){
        int right_neighbor = (rank + 1)%nprocs;
        int left_neighbor  = rank - 1;
        if (left_neighbor < 0){
            left_neighbor += nprocs;
        }

        int owner;
        unsigned long send_size = mat_send.size();
        unsigned long recv_size;
        std::vector<cooEntry> mat_recv;
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
            mat_recv.resize(recv_size);

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
            for(nnz_t i = 0; i < mat_send.size(); i++){
                nnzPerCol_right_p[mat_send[i].col]++;
            }

            nnzPerColScan_right[0] = 0;
            for(nnz_t i = 0; i < mat_recv_M; i++){
                nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
            }

#ifdef __DEBUG1__
//          print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
//          print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

            if(A->entry.empty() || mat_send.empty()){ // skip!
#ifdef __DEBUG1__
                if(verbose_triple_mat_mult){
                    if(A->entry.empty()){
                        printf("\nskip: A->entry.size() == 0\n\n");
                    } else {
                        printf("\nskip: mat_send == 0\n\n");
                    }
                }
#endif
            } else {

//                fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), mat_send.size(),
//                        A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[owner],
//                        &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
//                        &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//                fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), mat_send.size(),
//                        A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[owner],
//                        &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
//                        &nnzPerColScan_right[0], &nnzPerColScan_right[1], map_matmat, A->comm);

//                matmat(&A->entry[0], &mat_send[0], AP, A->entry.size(), mat_send.size(),
//                       A->M, A->Mbig, mat_recv_M, &nnzPerRowScan_left[0], &nnzPerColScan_right[0], A->comm);

                matmat(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), mat_send.size(),
                        A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[owner],
                        &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
                        &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);
            }

            MPI_Waitall(3, requests+1, statuses+1);

            mat_recv.swap(mat_send);
            send_size = recv_size;

#ifdef __DEBUG1__
//          print_vector(AP_temp, -1, "AP_temp", A->comm);
//          print_vector(mat_send, -1, "mat_send", A->comm);
//          print_vector(mat_recv, -1, "mat_recv", A->comm);
//          prev_owner = owner;
//          printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
#endif

        }

        mat_recv.clear();
        mat_recv.shrink_to_fit();

        delete [] requests;
        delete [] statuses;

    } else { // nprocs == 1 -> serial

        index_t mat_recv_M = P->splitNew[rank + 1] - P->splitNew[rank];

        std::fill(&nnzPerCol_right[0], &nnzPerCol_right[mat_recv_M], 0);
        nnzPerCol_right_p = &nnzPerCol_right[0] - P->splitNew[rank];
        for(nnz_t i = 0; i < mat_send.size(); i++){
            nnzPerCol_right_p[mat_send[i].col]++;
        }

        nnzPerColScan_right[0] = 0;
        for(nnz_t i = 0; i < mat_recv_M; i++){
            nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
        }

#ifdef __DEBUG1__
//          print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
//          print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

        if(A->entry.empty() || mat_send.empty()){ // skip!
#ifdef __DEBUG1__
            if(verbose_triple_mat_mult){
                if(A->entry.empty()){
                    printf("\nskip: A->entry.size() == 0\n\n");
                } else {
                    printf("\nskip: mat_send == 0\n\n");
                }
            }
#endif
        } else {

            double t1 = MPI_Wtime();

//            fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), mat_send.size(),
//                    A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[rank],
//                    &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
//                    &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//            fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), mat_send.size(),
//                    A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[rank],
//                    &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
//                    &nnzPerColScan_right[0], &nnzPerColScan_right[1], map_matmat, A->comm);

            matmat(&A->entry[0], &mat_send[0], AP, A->entry.size(), mat_send.size(),
                    A->M, A->Mbig, mat_recv_M, &nnzPerRowScan_left[0],&nnzPerColScan_right[0], A->comm);

            std::sort(AP.begin(), AP.end()); //todo: move this to the end of this else statement.

            double t2 = MPI_Wtime();
            printf("\nmatmat of AP = %f\n", t2-t1);
        }
    }

//    std::sort(AP.begin(), AP.end());

//    std::vector<cooEntry> AP;
//    nnz_t AP_temp_size_minus1 = AP_temp.size()-1;
//    for(nnz_t i = 0; i < AP_temp.size(); i++){
//        AP.emplace_back(AP_temp[i]);
//        while(i < AP_temp_size_minus1 && AP_temp[i] == AP_temp[i+1]){ // values of entries with the same row and col should be added.
//            std::cout << AP_temp[i] << "\t" << AP_temp[i+1] << std::endl;
//            AP.back().val += AP_temp[++i].val;
//        }
//    }

    mat_send.clear();
    mat_send.shrink_to_fit();
//    AP_temp.clear();
//    AP_temp.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(AP_temp, -1, "AP_temp", A->comm);
    if(verbose_triple_mat_mult){
        MPI_Barrier(comm); printf("compute_coarsen: step 5: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

//    std::ofstream file("chrome.json");
//    dollar::chrome(file);
//    if(rank==0) printf("\nRA:\n");
//    if(rank==0) dollar::text(std::cout);
//    dollar::clear();

    // *******************************************************
    // part 2: multiply: R_i * (AP_temp)_i. in which R_i = P_i_tranpose
    // *******************************************************

    // local transpose of P is being used to compute R*(AP_temp). So P is transposed locally here.
    std::vector<cooEntry> P_tranpose(P->entry.size());
    transpose_locally(P->entry, P->entry.size(), P_tranpose);

    // convert the indices to global
    for(nnz_t i = 0; i < P_tranpose.size(); i++){
        P_tranpose[i].col += P->split[rank];
    }

#ifdef __DEBUG1__
//    print_vector(P->entry, -1, "P->entry", comm);
//    print_vector(P_tranpose, -1, "P_tranpose", comm);
#endif

    // compute nnzPerColScan_left for P_tranpose
//    nnzPerRow_left.assign(P->Mbig, 0);
//    *nnzPerRow_left_p = &nnzPerRow_left[0] - P->split[rank];
//    for(nnz_t i = 0; i < P_tranpose.size(); i++){
//        nnzPerRow_left[P_tranpose[i].row]++;
//    }

//    nnzPerRowScan_left.resize(P->M + 1);
//    nnzPerRowScan_left[0] = 0;
//    for(nnz_t i = 0; i < P->M; i++){
//        nnzPerRowScan_left[i+1] = nnzPerRowScan_left[i] + nnzPerRow_left[i];
//    }

//    nnzPerRow_left.clear();
//    nnzPerRow_left.shrink_to_fit();

    // compute nnzPerColScan_left for P_tranpose
    std::vector<index_t> nnzPerCol_left(P->M, 0);
//    nnzPerCol_left.assign(P->M, 0);
    index_t *nnzPerCol_left_p = &nnzPerCol_left[0] - P->split[rank];
    for(nnz_t i = 0; i < P_tranpose.size(); i++){
        nnzPerCol_left_p[P_tranpose[i].col]++;
//        nnzPerCol_left[P_tranpose[i].col - P->split[rank]]++;
    }

    std::vector<index_t> nnzPerColScan_left(P->M + 1);
    nnzPerColScan_left[0] = 0;
    for(nnz_t i = 0; i < P->M; i++){
        nnzPerColScan_left[i+1] = nnzPerColScan_left[i] + nnzPerCol_left[i];
    }

    nnzPerCol_left.clear();
    nnzPerCol_left.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(nnzPerColScan_left, -1, "nnzPerColScan_left", comm);
#endif

    // compute nnzPerColScan_left for AP_temp
    nnzPerCol_right.assign(P->Nbig, 0);
    for(nnz_t i = 0; i < AP.size(); i++){
        nnzPerCol_right[AP[i].col]++;
    }

#ifdef __DEBUG1__
//    print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
#endif

    nnzPerColScan_right.resize(P->Nbig+1);
    nnzPerColScan_right[0] = 0;
    for(nnz_t i = 0; i < P->Nbig; i++){
        nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
    }

    nnzPerCol_right.clear();
    nnzPerCol_right.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

    // multiply: R_i * (AP_temp)_i. in which R_i = P_i_tranpose
    // ===================================================

    std::vector<cooEntry> RAP_temp;
    if(rank==0) printf("\n");

    if(P_tranpose.empty() || AP.empty()){ // skip!
#ifdef __DEBUG1__
        if(verbose_triple_mat_mult){
            if(P_tranpose.empty()){
                printf("\nskip: P_tranpose.size() == 0\n\n");
            } else {
                printf("\nskip: AP == 0\n\n");
            }
        }
#endif
    } else {

//        double t1 = MPI_Wtime();
        fast_mm(&P_tranpose[0], &AP[0], RAP_temp, P_tranpose.size(), AP.size(),
                P->Nbig, 0, P->M, P->split[rank], P->Nbig, 0,
                &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
                &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//        fast_mm(&P_tranpose[0], &AP[0], RAP_temp, P_tranpose.size(), AP.size(),
//                P->Nbig, 0, P->M, P->split[rank], P->Nbig, 0,
//                &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
//                &nnzPerColScan_right[0], &nnzPerColScan_right[1], map_matmat, A->comm);
//        double t2 = MPI_Wtime();
//        printf("\nfast_mm of R(AP_temp) = %f \n", t2-t1);

//        fast_mm_basic(&P_tranpose[0], &AP[0], RAP_temp, P_tranpose.size(), AP.size(),
//                P->Nbig, 0, P->M, P->split[rank], P->Nbig, 0,
//                &nnzPerRowScan_left[0], &nnzPerColScan_right[0], A->comm);
    }

    // free memory
    // -----------
    AP.clear();
    AP.shrink_to_fit();
    P_tranpose.clear();
    P_tranpose.shrink_to_fit();
    nnzPerRowScan_left.clear();
    nnzPerRowScan_left.shrink_to_fit();
    nnzPerColScan_right.clear();
    nnzPerColScan_right.shrink_to_fit();

//    if(rank==0) printf("\nRAP:\n");
//    if(rank==0) dollar::text(std::cout);
//    dollar::clear();

#ifdef __DEBUG1__
//    print_vector(RAP_temp, -1, "RAP_temp", A->comm);
    if(verbose_triple_mat_mult){
        MPI_Barrier(comm); printf("compute_coarsen: step 6: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // remove local duplicates.
    // Entries should be sorted in row-major order first, since the matrix should be partitioned based on rows.
    // So cooEntry_row is used here. Remove local duplicates and put them in RAP_temp_row.
    nnz_t size_minus_1 = 0;
    if(!RAP_temp.empty()){
        size_minus_1 = RAP_temp.size() - 1;
    }

    std::vector<cooEntry_row> RAP_temp_row;
    for(nnz_t i = 0; i < RAP_temp.size(); i++){
        RAP_temp_row.emplace_back(cooEntry_row( RAP_temp[i].row, RAP_temp[i].col, RAP_temp[i].val ));
        while(i < size_minus_1 && RAP_temp[i] == RAP_temp[i+1]){ // values of entries with the same row and col should be added.
            i++;
            RAP_temp_row.back().val += RAP_temp[i].val;
        }
    }

    RAP_temp.clear();
    RAP_temp.shrink_to_fit();

#ifdef __DEBUG1__
//    MPI_Barrier(comm); printf("rank %d: RAP_temp_row.size = %lu \n", rank, RAP_temp_row.size()); MPI_Barrier(comm);
//    print_vector(RAP_temp_row, -1, "RAP_temp_row", comm);
//    print_vector(P->splitNew, -1, "P->splitNew", comm);
    if(verbose_triple_mat_mult){
        MPI_Barrier(comm); printf("compute_coarsen: step 7: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // sort globally
    // -------------
    par::sampleSort(RAP_temp_row, RAP_row_sorted, P->splitNew, comm);

    RAP_temp_row.clear();
    RAP_temp_row.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(RAP_row_sorted, -1, "RAP_row_sorted", A->comm);
//    MPI_Barrier(comm); printf("rank %d: RAP_row_sorted.size = %lu \n", rank, RAP_row_sorted.size()); MPI_Barrier(comm);

    if(verbose_triple_mat_mult){
        MPI_Barrier(comm); printf("compute_coarsen: step 8: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

//    std::vector<cooEntry> RAP_sorted(RAP_row_sorted.size());
//    memcpy(&RAP_sorted[0], &RAP_row_sorted[0], RAP_row_sorted.size() * sizeof(cooEntry));
//    RAP_row_sorted.clear();
//    RAP_row_sorted.shrink_to_fit();

    // clear map_matmat and free memory
//    map_matmat.clear();
//    std::unordered_map<index_t, value_t> map_temp;
//    std::swap(map_matmat, map_temp);

    // remove duplicates.
    std::vector<cooEntry> Ac_dummy;
    cooEntry temp;
    size_minus_1 = RAP_row_sorted.size() - 1;
    for(nnz_t i = 0; i < RAP_row_sorted.size(); i++){
        temp = cooEntry(RAP_row_sorted[i].row, RAP_row_sorted[i].col, RAP_row_sorted[i].val);
        while(i < size_minus_1 && RAP_row_sorted[i] == RAP_row_sorted[i+1]){ // values of entries with the same row and col should be added.
            ++i;
            temp.val += RAP_row_sorted[i].val;
        }
        Ac_dummy.emplace_back( temp );
    }

    return 0;
}
*/