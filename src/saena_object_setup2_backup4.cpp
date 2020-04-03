// this version has case1 written from scratch and it is updated to work with the new changes until Feb 5, 2020.
// It works in the following commit:
//commit ea2c19634231952d02c87f2445aecd50305df5ed (HEAD -> matmat2, origin/matmat2)
//Author: majidrp <u1013493@utah.edu>
//Date:   Wed Feb 5 20:36:16 2020 -0700
//return case1 from MKL to your old written code (temp)

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

    int verbose_rank = 0;

#ifdef __DEBUG1__
    if(rank==verbose_rank && verbose_fastmm) printf("\nfast_mm: start \n");

    const double ALMOST_ZERO = 1e-16;

    // assert A entries
    index_t col_idx;
    for (nnz_t i = 0; i < A.col_sz; i++) {
//        col_idx = i + A.col_offset;
        for (nnz_t j = A.col_scan[i]; j < A.col_scan[i + 1]; j++) {
//            std::cout << j << "\t" << A.r[j] << "\t" << col_idx << "\t" << A.v[j] << "\n";
//            assert( (A.r[j] >= 0) && (A.r[j] < A.row_sz) );
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
//            assert( (B.r[j] >= 0) && (B.r[j] < B.row_sz) );
            assert( i < B.col_sz );
//            assert( fabs(B.v[j]) > ALMOST_ZERO );

//            assert( (col_idx >= B.col_offset) && (col_idx < B.col_offset + B.col_sz) );
        }
    }

    if(rank==verbose_rank){

        if(verbose_matmat_A){
            std::cout << "\nA: nnz = "       << A.nnz
                      << ", A_row_size = "   << A.row_sz     << ", A_col_size = "   << A.col_sz
                      << ", A_row_offset = " << A.row_offset << ", A_col_offset = " << A.col_offset << std::endl;

//            print_array(A.col_scan, A.col_sz+1, 0, "A.col_scan", comm);

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
#if 0
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
#endif

    if (A.row_sz * B.col_sz < matmat_size_thre1) { //DOLLAR("case0")

#ifdef __DEBUG1__
        if (rank == verbose_rank && (verbose_fastmm || verbose_matmat_recursive)) {
            printf("fast_mm: case 0: start \n");
        }
#endif

//        double t1 = MPI_Wtime();

        index_t *nnzPerRow_left = &mempool2[0];
        std::fill(&nnzPerRow_left[0], &nnzPerRow_left[A.row_sz], 0);
        index_t *nnzPerRow_left_p = &nnzPerRow_left[0] - A.row_offset;
//        index_t *nnzPerRow_left_p = &nnzPerRow_left[0];

//        std::cout << "\nA_row_offset = " << A_row_offset << std::endl;
        for (nnz_t i = 0; i < A.col_sz; i++) {
            for (nnz_t j = A.col_scan[i]; j < A.col_scan[i+1]; j++) {
//                std::cout << i << "\t" << A.r[j] << "\t" << A.r[j] - A.row_offset << std::endl;
                nnzPerRow_left_p[A.r[j]]++;
            }
        }

#ifdef __DEBUG1__
//        print_array(nnzPerRow_left, A_row_size, 1, "nnzPerRow_left", comm);
#endif

        index_t *A_new_row_idx   = &nnzPerRow_left[0];
        index_t *A_new_row_idx_p = &A_new_row_idx[0] - A.row_offset;
        index_t *orig_row_idx    = &mempool2[A.row_sz];
        index_t A_nnz_row_sz     = 0;

        for (index_t i = 0; i < A.row_sz; i++) {
            if (A_new_row_idx[i]) {
                A_new_row_idx[i] = A_nnz_row_sz;
                orig_row_idx[A_nnz_row_sz] = i + A.row_offset;
                A_nnz_row_sz++;
            }
        }

#ifdef __DEBUG1__
//        print_array(orig_row_idx,  A_nnz_row_sz, 0, "orig_row_idx",  comm);
//        print_array(A_new_row_idx, A.row_sz,     0, "A_new_row_idx", comm);
#endif

        index_t *B_new_col_idx   = &mempool2[A.row_sz * 2];
//        index_t *B_new_col_idx_p = &B_new_col_idx[0] - B_col_offset;
        index_t *orig_col_idx    = &mempool2[A.row_sz * 2 + B.col_sz];
        index_t B_nnz_col_sz     = 0;

        for (index_t i = 0; i < B.col_sz; i++) {
            if (B.col_scan[i+1] != B.col_scan[i]) {
                B_new_col_idx[i] = B_nnz_col_sz;
                orig_col_idx[B_nnz_col_sz] = i + B.col_offset;
                B_nnz_col_sz++;
            }
        }


#ifdef __DEBUG1__
//        std::cout << "orig_col_idx max: " << A_row_size * 2 + B_col_size + B_nnz_col_sz - 1 << std::endl;

//        print_array(orig_col_idx,  B_nnz_col_sz, 1, "B orig_col_idx", comm);
//        print_array(B_new_col_idx, B_col_size,   1, "B_new_col_idx",  comm);

//        printf("A_row_size = %u, \tA_nnz_row_sz = %u, \tB_col_size = %u, \tB_nnz_col_sz = %u \n",
//            A_row_size, A_nnz_row_sz, B_col_size, B_nnz_col_sz);
#endif

        // check if A_nnz_row_sz * B_nnz_col_sz < matmat_size_thre1, then do dense multiplication. otherwise, do case2 or 3.
        if(A_nnz_row_sz * B_nnz_col_sz < matmat_size_thre2) {

            ++case1_iter;

            // initialize
            value_t *C_temp = &mempool1[0];
//                std::fill(&C_temp[0], &C_temp[A_nnz_row_sz * B_nnz_col_sz], 0);

#ifdef __DEBUG1__
            if (rank == verbose_rank && verbose_fastmm) { printf("fast_mm: case 1: step 1 \n"); }
#endif

            mapbit.reset();
            index_t C_index;
            value_t C_val;
            index_t temp;
            const index_t *Ac_p = &A.col_scan[0] - B.row_offset;
//            bool C_not_zero = false;

            for (nnz_t j = 0; j < B.col_sz; j++) { // columns of B

                for (nnz_t k = B.col_scan[j]; k < B.col_scan[j+1]; k++) { // nonzeros in column j of B

                    temp = A_nnz_row_sz * B_new_col_idx[j];

//                    if(rank==0) std::cout << "\n" << B.r[k] << "\t" << B.r[k] - B.row_offset
//                                          << "\t" << Ac_p[B.r[k]] << "\t" << Ac_p[B.r[k]+1] << std::endl;

                    for (nnz_t i = Ac_p[B.r[k]]; i < Ac_p[B.r[k] + 1]; i++) { // nonzeros in column (Br[k]) of A

#ifdef __DEBUG1__
//                            std::cout << Ar[i] << "\t" << j+B_col_offset << "\t" << Av[i] << "\t" << Bv[k] << std::endl;
//                            if(rank==0) std::cout << B[k].row << "\t" << B[k].row - B_row_offset << "\t" << Ac_p[B[k].row] << std::endl;

//                            if(rank==0) std::cout << A.r[i] << "\t" << A.row_offset << "\t" << A.r[i] - A.row_offset
//                                        << "\t" << A_new_row_idx[A.r[i] - A.row_offset] << "\t" << j << "\t" << B_new_col_idx[j]
//                                        << "\t" << A_new_row_idx[A.r[i] - A.row_offset] + temp
//                                        << "\t" << C_temp[A_new_row_idx[A.r[i] - A.row_offset] + temp]
//                                        << std::endl;

//                            if(rank==0) std::cout << A_new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B_new_col_idx[j] << "\t"
//                                                  << A_new_row_idx[A[i].row - A_row_offset] << "\t" << B_new_col_idx[j] << "\t"
//                                                  << C_temp[A_new_row_idx[A[i].row - A_row_offset] + A_nnz_row_sz * B_new_col_idx[j]] << std::endl;
#endif

//                            C_temp_p[A_new_row_idx_p[A[i].row] + A_nnz_row_sz * B[k].col] += B[k].val * A[i].val;
//                            C_temp[A_new_row_idx_p[A[i].row] + temp] += B[k].val * A[i].val;
//                            C_not_zero = true;

                        C_index = A_new_row_idx_p[A.r[i]] + temp;
                        C_val   = B.v[k] * A.v[i];

//                        if(rank==0) std::cout << C_index << "\t" << A.v[i] << "\t" << B.v[k] << "\t" << C_val << std::endl;
                        if(mapbit[C_index]) {
                            C_temp[C_index] += C_val;
                        } else {
                            C_temp[C_index] = C_val;
                            mapbit[C_index] = true;
                        }

//                            if(C_index==0) std::cout << C_temp[C_index] << std::endl;

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
            if (rank == verbose_rank && verbose_fastmm) { printf("fast_mm: case 1: step 2 \n"); }
//            print_array(C_temp, A_nnz_row_sz * B_nnz_col_sz, -1, "C_temp", comm);
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
        B2.row_offset = B.row_offset + B_row_size_half;
//        B2.row_offset = B.row_offset;

        B1.col_sz = B.col_sz;
        B2.col_sz = B.col_sz;

        B1.col_offset = B.col_offset;
        B2.col_offset = B.col_offset;

        B1.col_scan = B.col_scan;
        B2.col_scan = new index_t[B.col_sz + 1];
        B2.free_c   = true;

#if 0
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

        reorder_split(B, B1, B2);
//        reorder_split(B.r, B.v, B1.col_scan, B2.col_scan, B.col_sz, B_row_threshold, B_row_size_half);

//        B1.nnz = B1.col_scan[B.col_sz] - B1.col_scan[0];
//        B2.nnz = B2.col_scan[B.col_sz] - B2.col_scan[0];

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
//                    assert((A1.r[j] >= 0) && (A1.r[j] < A1.row_sz));
//                std::cout << j << "\t" << A1.r[j] << "\t" << i + A1.col_offset << "\t" << A1.v[j] << std::endl;
                }
            }
        }

        // assert A2
//        std::cout << "\nA2: nnz = " << A2.nnz << std::endl;
        if(A2.nnz != 0) {
            for (nnz_t i = 0; i < A2.col_sz; i++) {
                for (nnz_t j = A2.col_scan[i]; j < A2.col_scan[i + 1]; j++) {
//                    assert((A2.r[j] >= 0) && (A2.r[j] < A2.row_sz));
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
//                    assert((B1.r[j] >= 0) && (B1.r[j] < B1.row_sz));
                }
            }
        }

        // assert B2
//        std::cout << "\nB2: nnz = " << B2.nnz << std::endl;
        if(B2.nnz != 0){
            for (nnz_t i = 0; i < B2.col_sz; i++) {
                for (nnz_t j = B2.col_scan[i]; j < B2.col_scan[i + 1]; j++) {
//                    assert( (B2.r[j] >= 0) && (B2.r[j] < B2.row_sz) );
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

//            std::cout << "\nCase2:\n"
//                      << "A.row_sz: \t"       << A.row_sz      << "\tA1.row_sz: \t"     << A1.row_sz     << "\tA2.row_sz: \t"     << A2.row_sz
//                      << "\nA.row_offset: \t" << A.row_offset  << "\tA1.row_offset: \t" << A1.row_offset << "\tA2.row_offset: \t" << A2.row_offset
//                      << "\nA.col_sz: \t"     << A.col_sz      << "\tA1.col_sz: \t"     << A1.col_sz     << "\tA2.col_sz: \t"     << A2.col_sz
//                      << "\nA.col_offset: \t" << A.col_offset  << "\tA1.col_offset: \t" << A1.col_offset << "\tA2.col_offset: \t" << A2.col_offset
//                      << "\nA.nnz: \t\t"      << A.nnz         << "\tA1.nnz: \t"        << A1.nnz        << "\tA2.nnz: \t"        << A2.nnz << std::endl;
//
//            std::cout << "\nB.row_sz: \t"     << B.row_sz      << "\tB1.row_sz: \t"     << B1.row_sz     << "\tB2.row_sz: \t"     << B2.row_sz
//                      << "\nB.row_offset: \t" << B.row_offset  << "\tB1.row_offset: \t" << B1.row_offset << "\tB2.row_offset: \t" << B2.row_offset
//                      << "\nB.col_sz: \t"     << B.col_sz      << "\tB1.col_sz: \t"     << B1.col_sz     << "\tB2.col_sz: \t"     << B2.col_sz
//                      << "\nB.col_offset: \t" << B.col_offset  << "\tB1.col_offset: \t" << B1.col_offset << "\tB2.col_offset: \t" << B2.col_offset
//                      << "\nB.nnz: \t\t"      << B.nnz         << "\tB1.nnz: \t"        << B1.nnz        << "\tB2.nnz: \t"        << B2.nnz << std::endl;

            if (verbose_matmat_A) {
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
//        A2.row_offset = A.row_offset;

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

void saena_object::fast_mm(index_t *Ar, value_t *Av, index_t *Ac_scan,
                           index_t *Br, value_t *Bv, index_t *Bc_scan,
                           mat_info *A_info, mat_info *B_info,
//                           index_t A_row_size, index_t A_row_offset, index_t A_col_size, index_t A_col_offset,
//                           index_t B_col_size, index_t B_col_offset,
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

    nnz_t A_nnz = Ac_scan[A_info->col_sz] - Ac_scan[0];
    nnz_t B_nnz = Bc_scan[B_info->col_sz] - Bc_scan[0];

    index_t A_col_size_half = A_info->col_sz/2;

/*
    if(rank==0){

        std::cout << "\nA: nnz = "       << A_nnz
                  << ", A_row_size = "   << A_info->row_sz     << ", A_col_size = "   << A_info->col_sz
                  << ", A_row_offset = " << A_info->row_offset << ", A_col_offset = " << A_info->col_offset << std::endl;

        std::cout << "\nA: nnz = " << A_nnz << std::endl;
        for(nnz_t i = 0; i < A_info->col_sz; i++){
            for(nnz_t j = Ac_scan[i]; j < Ac_scan[i+1]; j++) {
                std::cout << j << "\t" << Ar[j] << "\t" << i << "\t" << Av[j] << std::endl;
            }
        }

        std::cout << "\nB: nnz = "       << B_nnz;
        std::cout << ", B_row_size = "   << B_info->row_sz     << ", B_col_size = "   << B_info->col_sz
                  << ", B_row_offset = " << B_info->row_offset << ", B_col_offset = " << B_info->col_offset << std::endl;

        // print entries of B:
        std::cout << "\nB: nnz = " << B_nnz << std::endl;
        for (nnz_t i = 0; i < B_info->col_sz; i++) {
            for (nnz_t j = Bc_scan[i]; j < Bc_scan[i+1]; j++) {
                std::cout << j << "\t" << Br[j] << "\t" << i << "\t" << Bv[j] << std::endl;
            }
        }
    }
*/

    int verbose_rank = 1;
#ifdef __DEBUG1__
//    if(rank==verbose_rank) std::cout << "\n==========================" << __func__ << "==========================\n";
    if(rank==verbose_rank && verbose_fastmm) printf("\nfast_mm: start \n");

    if(rank==verbose_rank){
/*
        std::cout << "\nA: nnz = "       << A_nnz
                  << ", A_row_size = "   << A_info->row_sz   << ", A_col_size = "   << A_info->col_sz
                  << ", A_row_offset = " << A_info->row_offset << ", A_col_offset = " << A_info->col_offset << std::endl;

        std::cout << "\nA: nnz = " << A_nnz << std::endl;
        index_t col_idx;
        for(nnz_t i = 0; i < A_info->col_sz; i++){
            col_idx = i + A_info->col_offset;
            for(nnz_t j = Ac_scan[i]; j < Ac_scan[i+1]; j++) {
                assert( (Ar[j] >= 0) && (Ar[j] < A_info->row_sz) );
                assert( (col_idx > A_info->col_offset) && (col_idx < A_info->col_offset + A_info->col_sz) );
                std::cout << j << "\t" << Ar[j] << "\t" << col_idx << "\t" << Av[j] << std::endl;
            }
        }
*/
        if(verbose_matmat_A){
            std::cout << "\nA: nnz = "       << A_nnz
                      << ", A_row_size = "   << A_info->row_sz     << ", A_col_size = "   << A_info->col_sz
                      << ", A_row_offset = " << A_info->row_offset << ", A_col_offset = " << A_info->col_offset << std::endl;

//            print_array(Ac_scan, A_col_size+1, 1, "Ac_scan", comm);

            // print entries of A:
            std::cout << "\nA: nnz = " << A_nnz << std::endl;
            index_t col_idx;
            for(nnz_t i = 0; i < A_info->col_sz; i++){
                col_idx = i + A_info->col_offset;
                for(nnz_t j = Ac_scan[i]; j < Ac_scan[i+1]; j++) {
                    std::cout << j << "\t" << Ar[j] << "\t" << col_idx << "\t" << Av[j] << std::endl;
                }
            }
        }

        if(verbose_matmat_B) {
            std::cout << "\nB: nnz = "       << B_nnz;
            std::cout << ", B_row_size = "   << B_info->row_sz     << ", B_col_size = "   << B_info->col_sz
                      << ", B_row_offset = " << B_info->row_offset << ", B_col_offset = " << B_info->col_offset << std::endl;

//            print_array(Bc_scan, B_col_size+1, 1, "Bc_scan", comm);

            // print entries of B:
            std::cout << "\nB: nnz = " << B_nnz << std::endl;
            for (nnz_t i = 0; i < B_info->col_sz; i++) {
                for (nnz_t j = Bc_scan[i]; j < Bc_scan[i+1]; j++) {
                    std::cout << j << "\t" << Br[j] << "\t" << i + B_info->col_offset << "\t" << Bv[j] << std::endl;
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

    if (A_info->row_sz * B_info->col_sz < matmat_size_thre1) { //DOLLAR("case0")
//    if (true) {

//        if (rank == verbose_rank) printf("fast_mm: case 1: start \n");

#ifdef __DEBUG1__
        if (rank == verbose_rank && (verbose_fastmm || verbose_matmat_recursive)) {
            printf("fast_mm: case 0: start \n");
        }
#endif

//        double t1 = MPI_Wtime();

        sparse_matrix_t Amkl = nullptr;
//        mkl_sparse_d_create_csc(&Amkl, SPARSE_INDEX_BASE_ZERO, A_info->row_sz + A_info->row_offset, A_info->col_sz, (int*)Ac_scan, (int*)(Ac_scan+1), (int*)Ar, Av);
//        mkl_sparse_d_create_csc(&Amkl, SPARSE_INDEX_BASE_ZERO, A_info->row_sz, A_info->col_sz, (int*)Ac_scan, (int*)(Ac_scan+1), (int*)(Ar - A_info->row_offset), Av);
        mkl_sparse_d_create_csc(&Amkl, SPARSE_INDEX_BASE_ZERO, A_info->row_sz, A_info->col_sz, (int*)Ac_scan, (int*)(Ac_scan+1), (int*)Ar, Av);

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
//        mkl_sparse_d_create_csc(&Bmkl, SPARSE_INDEX_BASE_ZERO, B_info->row_sz + B_info->row_offset, B_info->col_sz, (int*)Bc_scan, (int*)(Bc_scan+1), (int*)Br, Bv);
//        mkl_sparse_d_create_csc(&Bmkl, SPARSE_INDEX_BASE_ZERO, B_info->row_sz, B_info->col_sz, (int*)Bc_scan, (int*)(Bc_scan+1), (int*)(Br - B_info->row_offset), Bv);
        mkl_sparse_d_create_csc(&Bmkl, SPARSE_INDEX_BASE_ZERO, B_info->row_sz, B_info->col_sz, (int*)Bc_scan, (int*)(Bc_scan+1), (int*)Br, Bv);

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
        for (j = 0; j < B_info->col_sz; ++j) {
            for (i = pointerB_C[j]; i < pointerE_C[j]; ++i) {
//                if (rank == 3) printf("%3d: (%3d , %3d) = %8f\n", ii, rows_C[ii] + 1, j + B_col_offset + 1, values_C[ii]); fflush(nullptr);
                C.emplace_back(rows_C[ii] + A_info->row_offset, j + B_info->col_offset, values_C[ii]);
//                C.emplace_back(rows_C[ii], j + B_info->col_offset, values_C[ii]);
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
    mat_info A1_info, A2_info, B1_info, B2_info;

    // if A_col_size_half == 0, it means A_col_size = 1. In this case it goes to case3.
    if (A_info->row_sz <= A_info->col_sz && A_col_size_half != 0){//DOLLAR("case2")

        double t2 = MPI_Wtime();

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_fastmm) { printf("fast_mm: case 2: start \n"); }
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
        auto A_half_nnz = (nnz_t) ceil(A_nnz / 2);
//        index_t A_col_size_half = A_col_size/2;

        if (A_nnz > matmat_nnz_thre) { // otherwise A_col_size_half will stay A_col_size/2
            for (nnz_t i = 0; i < A_info->col_sz; i++) {
                if( (Ac[i+1] - Ac[0]) >= A_half_nnz){
                    A_col_size_half = i;
                    break;
                }
            }
        }

        // if A is not being split at all following "half nnz method", then swtich to "half size method".
        if (A_col_size_half == A_info->col_sz) {
            A_col_size_half = A_info->col_sz / 2;
        }
#endif

        auto A1r = &Ar[0];
        auto A1v = &Av[0];
        auto A2r = &Ar[0];
        auto A2v = &Av[0];

        auto A1c_scan = Ac_scan;
        auto A2c_scan = &Ac_scan[A_col_size_half];

        A1_info.row_sz = A_info->row_sz;
        A2_info.row_sz = A_info->row_sz;

        A1_info.row_offset = A_info->row_offset;
        A2_info.row_offset = A_info->row_offset;

        A1_info.col_sz = A_col_size_half;
        A2_info.col_sz = A_info->col_sz - A1_info.col_sz;

        A1_info.col_offset = A_info->col_offset;
        A2_info.col_offset = A_info->col_offset + A1_info.col_sz;

        nnz_t A1_nnz = A1c_scan[A1_info.col_sz] - A1c_scan[0];
        nnz_t A2_nnz = A_nnz - A1_nnz;

        // =======================================================

        // split B based on how A is split, so use A_col_size_half to split B. A_col_size_half can be different based on
        // choosing the splitting method (nnz or size).
        index_t B_row_size_half = A_col_size_half;
//        index_t B_row_threshold = B_row_size_half + B_info->row_offset;
        index_t B_row_threshold = B_row_size_half;

        auto B1c_scan = Bc_scan; // col_scan
        auto B2c_scan = new index_t[B_info->col_sz + 1]; // col_scan

        reorder_split(Br, Bv, B1c_scan, B2c_scan, B_info->col_sz, B_row_threshold, B_row_size_half);

        nnz_t B1_nnz = B1c_scan[B_info->col_sz] - B1c_scan[0];
        nnz_t B2_nnz = B2c_scan[B_info->col_sz] - B2c_scan[0];

        auto B1r = &Br[0];
        auto B1v = &Bv[0];
        auto B2r = &Br[B1c_scan[B_info->col_sz]];
        auto B2v = &Bv[B1c_scan[B_info->col_sz]];

        B1_info.row_sz = B_row_size_half;
        B2_info.row_sz = B_info->row_sz - B1_info.row_sz;

        B1_info.row_offset = B_info->row_offset;
        B2_info.row_offset = B_info->row_offset + B_row_size_half;

        B1_info.col_sz = B_info->col_sz;
        B2_info.col_sz = B_info->col_sz;

        B1_info.col_offset = B_info->col_offset;
        B2_info.col_offset = B_info->col_offset;

        t2 = MPI_Wtime() - t2;
        case2 += t2;

#ifdef __DEBUG1__

//        MPI_Barrier(comm);
//        std::cout << "\ncase2:\nA_nnz: " << std::setw(3) << A_nnz << ", A1_nnz: " << std::setw(3) << A1_nnz << ", A2_nnz: " << std::setw(3) << A2_nnz << ", A_col_size: "
//                  << std::setw(3) << A_info->col_sz << ", A_col_size_half: " << std::setw(3) << A_col_size_half << std::endl;

//        std::cout << "B_nnz: " << std::setw(3) << B_nnz << ", B1_nnz: " << std::setw(3) << B1_nnz
//                  << ", B2_nnz: " << std::setw(3) << B2_nnz << ", B_row_size: " << std::setw(3) << B_info->row_sz
//                  << ", B_row_size_half: " << std::setw(3) << B_row_size_half << ", B_row_size/2: " << B_info->row_sz/2
//                  << ", B_row_threshold: " << std::setw(3) << B_row_threshold << std::endl;
//
//        std::cout << "\ncase2_part2: B1_row_size: " << B1_row_size << "\tB2_row_size: " << B2_row_size
//                  << "\tB1_row_offset: " << B1_row_offset << "\tB2_row_offset: " << B2_row_offset
//                  << "\tB1_col_size:"  << B1_col_size << "\tB2_col_size: " << B2_col_size
//                  << "\tB1_col_offset: " << B1_col_offset << "\tB2_col_offset: " << B2_col_offset << std::endl;

//        std::cout << "B_row_threshold: " << std::setw(3) << B_row_threshold << std::endl;
//        print_array(Bc1, B_col_size+1, 0, "Bc1", comm);
//        print_array(Bc2, B_col_size+1, 0, "Bc2", comm);

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
                for (nnz_t i = 0; i < A1_info.col_sz; i++) {
                    for (nnz_t j = A1c_scan[i]; j < A1c_scan[i + 1]; j++) {
                        std::cout << j << "\t" << A1r[j] << "\t" << i + A1_info.col_offset << "\t" << A1v[j] << std::endl;
                    }
                }

                // print entries of A2:
                std::cout << "\nA2: nnz = " << A2_nnz << std::endl;
                for (nnz_t i = 0; i < A2_info.col_sz; i++) {
                    for (nnz_t j = A2c_scan[i]; j < A2c_scan[i + 1]; j++) {
                        std::cout << j << "\t" << A2r[j] << "\t" << i + A2_info.col_offset << "\t" << A2v[j] << std::endl;
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
                for (nnz_t i = 0; i < B1_info.col_sz; i++) {
                    for (nnz_t j = B1c_scan[i]; j < B1c_scan[i + 1]; j++) {
                        std::cout << j << "\t" << B1r[j] << "\t" << i + B1_info.col_offset << "\t" << B1v[j] << std::endl;
                    }
                }

                // print entries of B2:
                std::cout << "\nB2: nnz = " << B2_nnz << std::endl;
                for (nnz_t i = 0; i < B2_info.col_sz; i++) {
                    for (nnz_t j = B2c_scan[i]; j < B2c_scan[i + 1]; j++) {
                        std::cout << j << "\t" << B2r[j] << "\t" << i + B2_info.col_offset << "\t" << B2v[j] << std::endl;
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
        // The last element of A1c_scan is shared with the first element of A2c_scan, and it may gets changed
        // during the recursive calls from the A1c_scan side. So, save that and use it for the starting
        // point of A2c_scan inside the recursive calls.

        if (A1_nnz != 0 && B1_nnz != 0) {

            // Check Split Fact 1 for this part.
            A_col_scan_end = A1c_scan[A1_info.col_sz];
            B_col_scan_end = B1c_scan[B1_info.col_sz];

//            fast_mm(&A1r[0] - A1_info.row_offset, &A1v[0], &A1c_scan[0],
//                    &B1r[0] - B1_info.row_offset, &B1v[0], &B1c_scan[0],
//                    &A1_info, &B1_info,
//                    C, comm);

            fast_mm(&A1r[0], &A1v[0], &A1c_scan[0],
                    &B1r[0], &B1v[0], &B1c_scan[0],
                    &A1_info, &B1_info,
                    C, comm);

            A1c_scan[A1_info.col_sz] = A_col_scan_end;
            B1c_scan[B1_info.col_sz] = B_col_scan_end;

        }

        // C2 = A2 * B2
#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 2 \n");
#endif

        if (A2_nnz != 0 && B2_nnz != 0) {

            A_col_scan_end = A2c_scan[A2_info.col_sz];
            B_col_scan_end = B2c_scan[B2_info.col_sz];

//            fast_mm(&A2r[0] - A2_info.row_offset, &A2v[0], &A2c_scan[0],
//                    &B2r[0] - B2_info.row_offset, &B2v[0], &B2c_scan[0],
//                    &A2_info, &B2_info,
//                    C, comm);

            fast_mm(&A2r[0], &A2v[0], &A2c_scan[0],
                    &B2r[0], &B2v[0], &B2c_scan[0],
                    &A2_info, &B2_info,
                    C, comm);

            A2c_scan[A2_info.col_sz] = A_col_scan_end;
            B2c_scan[B2_info.col_sz] = B_col_scan_end;

//        fast_mm(&A[0], &B[0], C, A2_nnz, B2_nnz,
//                A_row_size, A_row_offset, A_col_size - A_col_size_half, A_col_offset + A_col_size_half,
//                B_col_size, B_col_offset,
//                &nnzPerColScan_leftStart[A_col_size_half], &nnzPerColScan_leftEnd[A_col_size_half], // A2
//                &nnzPerColScan_middle[0], nnzPerColScan_rightEnd, comm); // B2
        }

        t2 = MPI_Wtime();

        // return B to its original order.
        reorder_back_split(Br, Bv, B1c_scan, B2c_scan, B_info->col_sz, B_row_size_half);
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
//        if (rank == verbose_rank) printf("fast_mm: case 3: start \n");

        // split based on matrix size
        // =======================================================

#ifdef SPLIT_SIZE
        // prepare splits of matrix B by column

//        index_t A_col_size_half = A_col_size/2;
        index_t B_col_size_half = B_info->col_sz/2;

        nnz_t B1_nnz = Bc_scan[B_col_size_half] - Bc_scan[0];
        nnz_t B2_nnz = B_nnz - B1_nnz;

        auto B1r = &Br[0];
        auto B1v = &Bv[0];
        auto B2r = &Br[0];
        auto B2v = &Bv[0];

        auto B1c_scan = Bc_scan;
        auto B2c_scan = &Bc_scan[B_col_size_half];

        B1_info.row_sz = B_info->row_sz;
        B2_info.row_sz = B_info->row_sz;

        B1_info.row_offset = B_info->row_offset;
        B2_info.row_offset = B_info->row_offset;

        B1_info.col_sz = B_col_size_half;
        B2_info.col_sz = B_info->col_sz - B1_info.col_sz;

        B1_info.col_offset = B_info->col_offset;
        B2_info.col_offset = B_info->col_offset + B_col_size_half;

#endif

#ifdef __DEBUG1__
//        std::cout << "\ncase3:\nB_nnz: " << std::setw(3) << B_nnz << ", B1_nnz: " << std::setw(3) << B1_nnz
//                  << ", B2_nnz: " << std::setw(3) << B2_nnz << ", B_col_size: " << std::setw(3) << B_info->col_sz
//                  << ", B_col_size_half: " << std::setw(3) << B_col_size_half << std::endl;

//        std::cout << "\ncase3_part1: B_row_size: " << B_row_size << "\tB_row_offset: " << B_row_offset
//                  << "\tB1_col_size:"  << B1_col_size << "\tB2_col_size: " << B2_col_size
//                  << "\tB1_col_offset: " << B1_col_offset << "\tB2_col_offset: " << B2_col_offset << std::endl;

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
        index_t B_col_size_half = B_info->col_sz / 2;

        if (B_nnz > matmat_nnz_thre) {
            for (nnz_t i = 0; i < B_info->col_sz; i++) {
                B1_nnz += nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i];

#ifdef __DEBUG1__
//                if(rank==verbose_rank)
//                    printf("B_nnz = %lu, B_half_nnz = %lu, B1_nnz = %lu, nnz on col %u: %u \n",
//                           B_nnz, B_half_nnz, B1_nnz, B[nnzPerColScan_rightStart[i]].col,
//                           nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i]);
#endif

                if (B1_nnz >= B_half_nnz) {
                    B_col_size_half = B[nnzPerColScan_rightStart[i]].col + 1 - B_info->col_offset;
                    break;
                }
            }
        } else {
            for (nnz_t i = 0; i < B_col_size_half; i++) {
                B1_nnz += nnzPerColScan_rightEnd[i] - nnzPerColScan_rightStart[i];
            }
        }

        // if B is not being splitted at all following "half nnz method", then swtich to "half col method".
        if (B_col_size_half == B_info->col_sz) {
            B_col_size_half = B_info->col_sz / 2;
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

        index_t A_row_size_half = A_info->row_sz / 2;
//        index_t A_row_threshold = A_row_size_half + A_info->row_offset;
        index_t A_row_threshold = A_row_size_half;

        auto A1c_scan = Ac_scan; // col_scan
        auto A2c_scan = new index_t[A_info->col_sz + 1]; // col_scan

        reorder_split(Ar, Av, A1c_scan, A2c_scan, A_info->col_sz, A_row_threshold, A_row_size_half);

        nnz_t A1_nnz = A1c_scan[A_info->col_sz] - A1c_scan[0];
        nnz_t A2_nnz = A2c_scan[A_info->col_sz] - A2c_scan[0];

        auto A1r = &Ar[0];
        auto A1v = &Av[0];
        auto A2r = &Ar[A1c_scan[A_info->col_sz]];
        auto A2v = &Av[A1c_scan[A_info->col_sz]];

        A1_info.row_sz = A_row_size_half;
        A2_info.row_sz = A_info->row_sz - A1_info.row_sz;

        A1_info.row_offset = A_info->row_offset;
        A2_info.row_offset = A_info->row_offset + A1_info.row_sz;

        A1_info.col_sz = A_info->col_sz;
        A2_info.col_sz = A_info->col_sz;

        A1_info.col_offset = A_info->col_offset;
        A2_info.col_offset = A_info->col_offset;

        t3 = MPI_Wtime() - t3;
        case3 += t3;

#ifdef __DEBUG1__
//        std::cout << "A_nnz: " << std::setw(3) << A_nnz << ", A1_nnz: " << std::setw(3) << A1_nnz << ", A2_nnz: "
//                  << std::setw(3) << A2_nnz << ", A_row_size: " << std::setw(3) << A_info->row_sz
//                  << ", A_row_size_half: " << std::setw(3) << A_row_size_half << std::endl;

//        std::cout << "\ncase3_part2: A1_row_size: " << A1_row_size << "\tA2_row_size: " << A2_row_size
//                  << "\tA1_row_offset: " << A1_row_offset << "\tA2_row_offset: " << A2_row_offset
//                  << "\tA1_col_size: " << A1_col_size << "\tA2_col_size: " << A2_col_size
//                  << "\tA1_col_offset: " << A1_col_offset << "\tA2_col_offset: " << A2_col_offset << std::endl;

//        std::cout << "A_row_threshold: " << std::setw(3) << A_row_threshold << std::endl;

//        print_array(Ac1, A_col_size+1, 0, "Ac1", comm);
//        print_array(Ac2, A_col_size+1, 0, "Ac2", comm);

        if (rank == verbose_rank && verbose_fastmm) printf("fast_mm: case 3: step 2 \n");

//        MPI_Barrier(comm);
        if (rank == verbose_rank) {

//            printf("fast_mm: case 3: \nA_nnz: (%lu, %lu, %lu), B_nnz: (%lu, %lu, %lu)\n"
//                   "A_size: (%u, %u), B_size: (%u, %u, %u) \n",
//                   A_nnz, A1_nnz, A2_nnz, B_nnz, B1_nnz, B2_nnz, A_row_size, A_col_size, A_col_size, B_col_size, B_col_size_half);

            if (verbose_matmat_A) {
                // print entries of A1:
                std::cout << "\nCase3:\nA1: nnz = " << A1_nnz << std::endl;
                for (nnz_t i = 0; i < A1_info.col_sz; i++) {
                    for (nnz_t j = A1c_scan[i]; j < A1c_scan[i + 1]; j++) {
                        std::cout << j << "\t" << A1r[j] << "\t" << i + A1_info.col_offset << "\t" << A1v[j] << std::endl;
                    }
                }

                // print entries of A2:
                std::cout << "\nA2: nnz = " << A2_nnz << std::endl;
                for (nnz_t i = 0; i < A2_info.col_sz; i++) {
                    for (nnz_t j = A2c_scan[i]; j < A2c_scan[i + 1]; j++) {
                        std::cout << j << "\t" << A2r[j] << "\t" << i + A2_info.col_offset << "\t" << A2v[j] << std::endl;
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
                for (nnz_t i = 0; i < B1_info.col_sz; i++) {
                    for (nnz_t j = B1c_scan[i]; j < B1c_scan[i + 1]; j++) {
                        std::cout << j << "\t" << B1r[j] << "\t" << i + B1_info.col_offset << "\t" << B1v[j] << std::endl;
                    }
                }

                // print entries of B2:
                std::cout << "\nB2: nnz = " << B2_nnz << std::endl;
                for (nnz_t i = 0; i < B2_info.col_sz; i++) {
                    for (nnz_t j = B2c_scan[i]; j < B2c_scan[i + 1]; j++) {
                        std::cout << j << "\t" << B2r[j] << "\t" << i + B2_info.col_offset << "\t" << B2v[j] << std::endl;
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

        if (A1_nnz != 0 && B1_nnz != 0) {

            A_col_scan_end = A1c_scan[A1_info.col_sz];
            B_col_scan_end = B1c_scan[B1_info.col_sz];

//            fast_mm(&A1r[0] - A1_info.row_offset, &A1v[0], &A1c_scan[0],
//                    &B1r[0] - B1_info.row_offset, &B1v[0], &B1c_scan[0],
//                    &A1_info, &B1_info,
//                    C, comm);

            fast_mm(&A1r[0], &A1v[0], &A1c_scan[0],
                    &B1r[0], &B1v[0], &B1c_scan[0],
                    &A1_info, &B1_info,
                    C, comm);

            A1c_scan[A1_info.col_sz] = A_col_scan_end;
            B1c_scan[B1_info.col_sz] = B_col_scan_end;

        }


        // C2 = A1 * B2:
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 2 \n");
#endif

        if (A1_nnz != 0 && B2_nnz != 0) {

            A_col_scan_end = A1c_scan[A1_info.col_sz];
            B_col_scan_end = B2c_scan[B2_info.col_sz];

//            fast_mm(&A1r[0] - A1_info.row_offset, &A1v[0], &A1c_scan[0],
//                    &B2r[0] - B2_info.row_offset, &B2v[0], &B2c_scan[0],
//                    &A1_info, &B2_info,
//                    C, comm);

            fast_mm(&A1r[0], &A1v[0], &A1c_scan[0],
                    &B2r[0], &B2v[0], &B2c_scan[0],
                    &A1_info, &B2_info,
                    C, comm);

//            fast_mm(&A1[0], &B2[0], C,
//                    A1_row_size, A1_row_offset, A1_col_size, A1_col_offset,
//                    B2_col_size, B2_col_offset,
//                    &Ac1[0], &Bc2[0], comm);

            A1c_scan[A1_info.col_sz] = A_col_scan_end;
            B2c_scan[B2_info.col_sz] = B_col_scan_end;

        }


        // C3 = A2 * B1
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 3 \n");
#endif

        if (A2_nnz != 0 && B1_nnz != 0) {

            A_col_scan_end = A2c_scan[A2_info.col_sz];
            B_col_scan_end = B1c_scan[B1_info.col_sz];

//            fast_mm(&A2r[0] - A2_info.row_offset, &A2v[0], &A2c_scan[0],
//                    &B1r[0] - B1_info.row_offset, &B1v[0], &B1c_scan[0],
//                    &A2_info, &B1_info,
//                    C, comm);

            fast_mm(&A2r[0], &A2v[0], &A2c_scan[0],
                    &B1r[0], &B1v[0], &B1c_scan[0],
                    &A2_info, &B1_info,
                    C, comm);

//            fast_mm(&A2[0], &B1[0], C,
//                    A2_row_size, A2_row_offset, A2_col_size, A2_col_offset,
//                    B1_col_size, B1_col_offset,
//                    &Ac2[0], &Bc1[0], comm);

            A2c_scan[A2_info.col_sz] = A_col_scan_end;
            B1c_scan[B1_info.col_sz] = B_col_scan_end;

        }


        // C4 = A2 * B2
        //=========================

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 4 \n");
#endif

        if (A2_nnz != 0 && B2_nnz != 0) {

            A_col_scan_end = A2c_scan[A2_info.col_sz];
            B_col_scan_end = B2c_scan[B2_info.col_sz];

//            fast_mm(&A2r[0] - A2_info.row_offset, &A2v[0], &A2c_scan[0],
//                    &B2r[0] - B2_info.row_offset, &B2v[0], &B2c_scan[0],
//                    &A2_info, &B2_info,
//                    C, comm);

            fast_mm(&A2r[0], &A2v[0], &A2c_scan[0],
                    &B2r[0], &B2v[0], &B2c_scan[0],
                    &A2_info, &B2_info,
                    C, comm);

//            fast_mm(&A2[0], &B2[0], C,
//                    A2_row_size, A2_row_offset, A2_col_size, A2_col_offset,
//                    B2_col_size, B2_col_offset,
//                    &Ac2[0], &Bc2[0], comm);

            A2c_scan[A2_info.col_sz] = A_col_scan_end;
            B2c_scan[B2_info.col_sz] = B_col_scan_end;

        }

        t3 = MPI_Wtime();

        // return A to its original order.
        reorder_back_split(Ar, Av, A1c_scan, A2c_scan, A_info->col_sz, A_row_size_half);
        delete []A2c_scan;

        t3 = MPI_Wtime() - t3;
        case3 += t3;

#ifdef __DEBUG1__
        if (rank == verbose_rank && verbose_fastmm) printf("fast_mm: case 3: end \n");
#endif

    }

//    return;
}


int saena_object::reorder_split(index_t *Ar, value_t *Av, index_t *Ac1, index_t *Ac2, index_t col_sz, index_t threshold, index_t partial_offset){

#ifdef __DEBUG1__
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int verbose_rank = 1;

    if(rank == verbose_rank){
//        std::cout << "\nstart of " << __func__ << std::endl;
//        std::cout << "\n=========================================================================" << std::endl ;
//        std::cout << "\nA: nnz: " << Ac1[col_sz] - Ac1[0] << ", col_sz: " << col_sz << ", threshold: " << threshold << std::endl ;
//        print_array(Ac1, col_sz+1, 0, "Ac", MPI_COMM_WORLD);

        // ========================================================
        // this shows how to go through entries of A before changing order.
        // NOTE: column is not correct. col_offset should be added to it.
        // ========================================================

//        std::cout << "\nA: nnz: " << Ac1[col_sz] - Ac1[0] << "\tcol is not correct." << std::endl ;
//        for(index_t j = 0; j < col_sz; j++){
//            for(index_t i = Ac1[j]; i < Ac1[j+1]; i++){
//                std::cout << std::setprecision(4) << Ar[i] << "\t" << j << "\t" << Av[i] << std::endl;
//            }
//        }

        // ========================================================
    }

#endif

    // ========================================================
    // IMPORTANT: An offset should be used to access Ar and Av.
    // ========================================================
    nnz_t offset = Ac1[0];

    index_t *A1r = &mempool4[0];
    index_t *A2r = &mempool4[loc_nnz_max];
    value_t *A1v = &mempool5[0];
    value_t *A2v = &mempool5[loc_nnz_max];

    std::fill(&Ac2[0], &Ac2[col_sz+1], 0);
    auto Ac2_p = &Ac2[1]; // to do scan on it at the end.

    nnz_t A1_nnz = 0, A2_nnz = 0;
    for(index_t j = 0; j < col_sz; j++){
        for(nnz_t i = Ac1[j]; i < Ac1[j+1]; i++){
            if(Ar[i] < threshold){
                A1r[A1_nnz] = Ar[i];
                A1v[A1_nnz] = Av[i];
                ++A1_nnz;
//                if(rank==verbose_rank) std::cout << std::setprecision(4) << Ar[i] << "\t" << j << "\t" << Av[i] << "\ttop half" << std::endl;
            }else{
                A2r[A2_nnz] = Ar[i] - partial_offset;
//                A2r[A2_nnz] = Ar[i];
                A2v[A2_nnz] = Av[i];
                ++A2_nnz;
                Ac2_p[j]++;
//                if(rank==verbose_rank) std::cout << std::setprecision(4) << Ar[i] << "\t" << j << "\t" << Av[i] << "\tbottom half" << "\t" << partial_offset << std::endl;
            }
        }
    }

    // if Ac2 does not have any nonzero, then just return.
    if(Ac2[col_sz] == Ac2[0]){
        return 0;
    }

    for(index_t i = 1; i <= col_sz; i++){
        Ac2[i] += Ac2[i-1]; // scan on Ac2
        Ac1[i] -= Ac2[i];   // subtract Ac2 from Ac1 to have the correct scan for A1
    }

#ifdef __DEBUG1__
//    print_array(Ac1, col_sz+1, 0, "Ac1", MPI_COMM_WORLD);
//    print_array(Ac2, col_sz+1, 0, "Ac2", MPI_COMM_WORLD);
#endif

    // First put A1 at the beginning of A, then put A2 at the end A.
    memcpy(&Ar[offset],          &A1r[0], A1_nnz * sizeof(index_t));
    memcpy(&Av[offset],          &A1v[0], A1_nnz * sizeof(value_t));

    memcpy(&Ar[offset + A1_nnz], &A2r[0], A2_nnz * sizeof(index_t));
    memcpy(&Av[offset + A1_nnz], &A2v[0], A2_nnz * sizeof(value_t));

#if 0
    // Equivalent to the previous part. Uses for loops instead of memcpy.
    nnz_t arr_idx = offset;
    for(nnz_t i = 0; i < A1r.size(); i++){
        Ar[arr_idx] = A1r[i];
        Av[arr_idx] = A1v[i];
        arr_idx++;
    }
    for(nnz_t i = 0; i < A2r.size(); i++){
        Ar[arr_idx] = A2r[i];
        Av[arr_idx] = A2v[i];
        arr_idx++;
    }
#endif

#ifdef __DEBUG1__
//    print_array(Ac1, col_sz+1, 0, "Ac1", MPI_COMM_WORLD);
//    print_array(Ac2, col_sz+1, 0, "Ac2", MPI_COMM_WORLD);

    // ========================================================
    // this shows how to go through entries of A1 (top half) and A2 (bottom half) after changing order.
    // NOTE: column is not correct. col_offset should be added to it.
    // ========================================================
//    if(rank == verbose_rank) {
//        std::cout << "\nA1: nnz: " << Ac1[col_sz] - Ac1[0] << "\tcol is not correct." << std::endl;
//        for (index_t j = 0; j < col_sz; j++) {
//            for (index_t i = Ac1[j]; i < Ac1[j + 1]; i++) {
//                std::cout << std::setprecision(4) << Ar[i] << "\t" << j << "\t" << Av[i] << std::endl;
//            }
//        }
//        std::cout << "\nA2: nnz: " << Ac2[col_sz] - Ac2[0] << "\tcol is not correct." << std::endl;
//        for (index_t j = 0; j < col_sz; j++) {
//            for (index_t i = Ac2[j] + Ac1[col_sz]; i < Ac2[j + 1] + Ac1[col_sz]; i++) {
//                std::cout << std::setprecision(4) << Ar[i] + partial_offset << "\t" << j << "\t" << Av[i] << std::endl;
//            }
//        }
//    }
    // ========================================================
#endif

    return 0;
}

int saena_object::reorder_back_split(index_t *Ar, value_t *Av, index_t *Ac1, index_t *Ac2, index_t col_sz, index_t partial_offset){

#ifdef __DEBUG1__
//    int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    if(rank==0) std::cout << "\nstart of " << __func__ << std::endl;
#endif

    // ========================================================
    // IMPORTANT: An offset should be used to access Ar and Av.
    // ========================================================
    nnz_t offset = Ac1[0];

    nnz_t nnz1 = Ac1[col_sz] - Ac1[0];
    nnz_t nnz2 = Ac2[col_sz] - Ac2[0];
    nnz_t nnz  = nnz1 + nnz2;

    auto *Ar_temp = &mempool4[0];
    auto *Av_temp = &mempool5[0];

    memcpy(&Ar_temp[0], &Ar[offset], sizeof(index_t) * nnz);
    memcpy(&Av_temp[0], &Av[offset], sizeof(value_t) * nnz);

//    for(index_t i = offset; i < offset + nnz; i++){
//        Ar_temp_p[i] = Ar[i];
//        Av_temp_p[i] = Av[i];
//    }

#ifdef __DEBUG1__
//    print_array(Ac1, col_sz+1, 0, "Ac1", MPI_COMM_WORLD);
//    print_array(Ac2, col_sz+1, 0, "Ac2", MPI_COMM_WORLD);

#if 0
    // ========================================================
    // this shows how to go through entries of A1 (top half) and A2 (bottom half) after changing order.
    // NOTE: column is not correct. col_offset should be added to it.
    // ========================================================
    std::cout << "\nA1: nnz: " << Ac1[col_sz] - Ac1[0] << "\tcol is not correct." << std::endl ;
    for(index_t j = 0; j < col_sz; j++){
        for(index_t i = Ac1[j]; i < Ac1[j+1]; i++){
            std::cout << std::setprecision(4) << Ar[i] << "\t" << j << "\t" << Av[i] << std::endl;
//            std::cout << std::setprecision(4) << Ar[i] << "\t" << j << "\t" << Av[i] << "\ttemp: \t" << Ar_temp_p[i] << "\t" << j << "\t" << Av_temp_p[i] << std::endl;
        }
    }

    std::cout << "\nA2: nnz: " << Ac2[col_sz] - Ac2[0] << "\tcol is not correct." << std::endl ;
    for(index_t j = 0; j < col_sz; j++){
        for(index_t i = Ac2[j]+Ac1[col_sz]; i < Ac2[j+1]+Ac1[col_sz]; i++){
            std::cout << std::setprecision(4) << Ar[i] << "\t" << j << "\t" << Av[i] << std::endl;
//            std::cout << std::setprecision(4) << Ar[i] << "\t" << j << "\t" << Av[i] << "\ttemp: \t" << Ar_temp_p[i] << "\t" << j << "\t" << Av_temp_p[i] << std::endl;
        }
    }
#endif

    // ========================================================
#endif

    index_t *Ac = Ac1; // Will add Ac2 to Ac1 for each column to have Ac.

    nnz_t i, iter0 = offset, iter1 = 0, iter2 = Ac1[col_sz] - offset;
    nnz_t nnz_col;
    for(index_t j = 0; j < col_sz; j++){
        nnz_col = Ac1[j+1] - Ac1[j];
        if(nnz_col){
            memcpy(&Ar[iter0], &Ar_temp[iter1], sizeof(index_t) * nnz_col);
            memcpy(&Av[iter0], &Av_temp[iter1], sizeof(value_t) * nnz_col);
            iter1 += nnz_col;
            iter0 += nnz_col;
        }

        nnz_col = Ac2[j+1] - Ac2[j];
        if(nnz_col){

            for(i = 0; i < nnz_col; ++i){
//                Ar[iter0 + i] = Ar_temp[iter2 + i];
                Ar[iter0 + i] = Ar_temp[iter2 + i] + partial_offset;
//                if(rank==1) std::cout << Ar_temp[iter2 + i] << "\t" << j << "\t" << Av_temp[iter2 + i] << "\t" << partial_offset << std::endl;
            }

//            memcpy(&Ar[iter0], &Ar_temp[iter2], sizeof(index_t) * nnz_col);
            memcpy(&Av[iter0], &Av_temp[iter2], sizeof(value_t) * nnz_col);
            iter2 += nnz_col;
            iter0 += nnz_col;
        }

        Ac[j] += Ac2[j];
    }

    Ac[col_sz] += Ac2[col_sz];

#if 0
    // Equivalent to the previous part. Uses for loops instead of memcpy.
    index_t iter = offset;
    for(index_t j = 0; j < col_sz; j++){

        for(index_t i = Ac1[j]; i < Ac1[j+1]; i++) {
//            printf("%u \t%u \t%f\n", Ar_temp_p[i], j, Av_temp_p[i]);
            Ar[iter] = Ar_temp_p[i];
            Av[iter] = Av_temp_p[i];
            iter++;
        }

        for(index_t i = Ac2[j]+Ac1[col_sz]; i < Ac2[j+1]+Ac1[col_sz]; i++){
//            printf("%u \t%u \t%f\n", Ar_temp_p[i], j, Av_temp_p[i]);
            Ar[iter] = Ar_temp_p[i];
            Av[iter] = Av_temp_p[i];
            iter++;
        }

        Ac[j] += Ac2[j];
    }

    Ac[col_sz] += Ac2[col_sz];
#endif

#ifdef __DEBUG1__
//    print_array(Ac, col_sz+1, 0, "Ac", MPI_COMM_WORLD);

    // ========================================================
    // this shows how to go through entries of A before changing order.
    // NOTE: column is not correct. col_offset should be added to it.
    // ========================================================
//    std::cout << "\nA: nnz: " << Ac[col_sz] - Ac[0] << "\tcol is not correct." << std::endl ;
//    for(index_t j = 0; j < col_sz; j++){
//        for(index_t i = Ac[j]; i < Ac[j+1]; i++){
//            std::cout << std::setprecision(4) << Ar[i] << "\t" << j << "\t" << Av[i] << std::endl;
//        }
//    }

    // ========================================================
#endif

    return 0;
}


//before updating to the new matmat, which uses MKL
int saena_object::triple_mat_mult(Grid *grid){

    saena_matrix    *A  = grid->A;
//    prolong_matrix  *P  = &grid->P;
    restrict_matrix *R  = &grid->R;
    saena_matrix    *Ac = &grid->Ac;

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
//    print_vector(A->entry, -1, "A->entry", comm);
//    print_vector(P->entry, -1, "P->entry", comm);
//    print_vector(R->entry, -1, "R->entry", comm);

    if (verbose_triple_mat_mult) {
        MPI_Barrier(comm);
        if (rank == 0) printf("\nstart of triple_mat_mult: nprocs: %d \n", nprocs);
        MPI_Barrier(comm);
        printf("rank %d: A: Mbig = %u, \tM = %u, \tnnz_g = %lu, \tnnz_l = %lu \n", rank, A->Mbig, A->M, A->nnz_g,
               A->nnz_l);
//        MPI_Barrier(comm);
//        printf("rank %d: P: Mbig = %u, \tM = %u, \tnnz_g = %lu, \tnnz_l = %lu \n", rank, P->Mbig, P->M, P->nnz_g,
//               P->nnz_l);
        MPI_Barrier(comm);
        printf("rank %d: R: Mbig = %u, \tM = %u, \tnnz_g = %lu, \tnnz_l = %lu \n", rank, R->Mbig, R->M, R->nnz_g,
               R->nnz_l);
        MPI_Barrier(comm);
//        print_vector(A->split,    1, "A->split",    comm);
//        print_vector(R->splitNew, 1, "R->splitNew", comm);
//        print_vector(A->nnz_list, 1, "A->nnz_list", comm);
//        print_vector(R->nnz_list, 1, "R->nnz_list", comm);
    }
#endif

    // =======================================
    // Convert R to CSC
    // =======================================

    CSCMat Rcsc;
    Rcsc.nnz      = R->nnz_l;
    Rcsc.col_sz   = R->Nbig;
    Rcsc.max_nnz  = R->nnz_max;
//    Rcsc.max_M    = R->M_max; // we only need this for the right-hand side matrix in matmat.
    Rcsc.row      = new index_t[Rcsc.nnz];
    Rcsc.val      = new value_t[Rcsc.nnz];
    Rcsc.col_scan = new index_t[Rcsc.col_sz + 1];

    Rcsc.split = R->splitNew;
//    Rcsc.nnz_list = R->nnz_list;

    std::fill(&Rcsc.col_scan[0], &Rcsc.col_scan[Rcsc.col_sz + 1], 0);
    index_t *Rc_tmp = &Rcsc.col_scan[1];
    for(nnz_t i = 0; i < Rcsc.nnz; i++){
        Rcsc.row[i] = R->entry[i].row + Rcsc.split[rank];
        Rcsc.val[i] = R->entry[i].val;
        Rc_tmp[R->entry[i].col]++;
    }

    if(Rcsc.nnz != 0) {
        for (nnz_t i = 0; i < Rcsc.col_sz; i++) {
            Rcsc.col_scan[i + 1] += Rcsc.col_scan[i];
        }
    }

#ifdef __DEBUG1__
    {
        if (verbose_triple_mat_mult) {
            MPI_Barrier(comm);
            if (rank == 0) printf("triple_mat_mult: step 1\n");
            MPI_Barrier(comm);
//        printf("R: rank %d: nnz: %lu, \tmax_nnz: %lu, \tcol_sz: %u, \tmax_M: %u\n",
//               rank, Rcsc.nnz, Rcsc.max_nnz, Rcsc.col_sz, Rcsc.max_M);
//        MPI_Barrier(comm);
        }

//    R->print_entry(0);
//    printf("A: nnz_l: %ld\tnnz_g: %ld\tM: %d\tM_big: %d\tN_big: %d\n", R->nnz_l, R->nnz_g, R->M, R->Mbig, R->Nbig);
//    print_array(Rcsc.row, Rcsc.nnz, 0, "Rcsc.row", comm);
//    print_array(Rcsc.val, Rcsc.nnz, 0, "Rcsc.val", comm);
//    print_array(Rcsc.col_scan, Rcsc.col_sz + 1, 0, "Rcsc.col_scan", comm);

//    std::cout << "\nR: nnz: " << Rcsc.nnz << std::endl ;
//    for(index_t j = 0; j < Rcsc.col_sz; j++){
//        for(index_t i = Rcsc.col_scan[j]; i < Rcsc.col_scan[j+1]; i++){
//            std::cout << std::setprecision(4) << Rcsc.row[i] << "\t" << j << "\t" << Rcsc.val[i] << std::endl;
//        }
//    }
    }
#endif

    // =======================================
    // Convert the local transpose of A to CSC
    // =======================================

    // make a copy of A entries, then change their order to row-major
    std::vector<cooEntry> A_ent(A->entry);
    std::sort(A_ent.begin(), A_ent.end(), row_major);

    CSCMat Acsc;
    Acsc.nnz      = A->nnz_l;
    Acsc.col_sz   = A->M;
    Acsc.max_nnz  = A->nnz_max;
    Acsc.max_M    = A->M_max;
    Acsc.row      = new index_t[Acsc.nnz];
    Acsc.val      = new value_t[Acsc.nnz];
    Acsc.col_scan = new index_t[Acsc.col_sz + 1];

    std::fill(&Acsc.col_scan[0], &Acsc.col_scan[Acsc.col_sz + 1], 0);
//    index_t *Ac_tmp   = &Acsc.col_scan[1];
    index_t *Ac_tmp_p = &Acsc.col_scan[1] - A->split[rank]; // use this to avoid subtracting a fixed number

    for(nnz_t i = 0; i < Acsc.nnz; i++){
        Acsc.row[i] = A_ent[i].col;
        Acsc.val[i] = A_ent[i].val;
        Ac_tmp_p[A_ent[i].row]++;
    }

    for(nnz_t i = 0; i < Acsc.col_sz; i++){
        Acsc.col_scan[i+1] += Acsc.col_scan[i];
    }

    Acsc.split    = A->split;
    Acsc.nnz_list = A->nnz_list;

#ifdef __DEBUG1__
    if (verbose_triple_mat_mult) {
        MPI_Barrier(comm);
        if (rank == 0) printf("triple_mat_mult: step 2\n");
        MPI_Barrier(comm);
//        printf("A: rank %d: nnz: %lu, \tmax_nnz: %lu, \tcol_sz: %u, \tmax_M: %u\n",
//               rank, Acsc.nnz, Acsc.max_nnz, Acsc.col_sz, Acsc.max_M);
//        MPI_Barrier(comm);
    }

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
    // Preallocate Memory
    // =======================================

// note: this guide is written as when the left matrix is called "A" and the right one as "B":
// mempool1: used for the dense buffer
// mempool2: usages: for A: 1- nnzPerRow_left and 2- orig_row_idx. for B: 3- B_new_col_idx and 4- orig_col_idx
// mempool2 size: 2 * A_row_size + 2 * B_col_size
// important: it depends if B or transpose of B is being used for the multiplciation. if originl B is used then
// B_col_size is B->Mbig, but for the transpos it is B->M, which is scalable.
// mempool3: is used to store the the received part of B.
// mempool3 size: it should store remote B, so we allocate the max size of B on all the procs.
//                sizeof(row index) + sizeof(value) + sizeof(col_scan) =
//                nnz * index_t + nnz * value_t + (col_size+1) * index_t

//    mempool1 = new value_t[matmat_size_thre2];

    // compute the maximum column size for both RA and (RA)P,
    // allocate mempool2 once and use it for both RA and (RA)P
    // ---------------------------------------------------
    index_t max_col_sz  = std::max(A->M_max, R->M_max);

    index_t R_row_size = R->M;
//    mempool2 = new index_t[2 * R_row_size + 2 * max_col_sz];

#ifdef __DEBUG1__
    if (verbose_triple_mat_mult) {
        MPI_Barrier(comm);
        if (rank == 0) printf("triple_mat_mult: step 3\n");
        MPI_Barrier(comm);
//        if (rank == 0) std::cout << "R_row_size: "      << R_row_size  << ", A->M_max: "   << A->M_max
//                                 << ", R->M_max: "      << R->M_max    << ", max_col_sz: " << max_col_sz
//                                 << ", mempool2 size: " << 2 * R_row_size + 2 * max_col_sz << std::endl;
//        MPI_Barrier(comm);
    }
#endif

    // 2 for both send and receive buffer, valbyidx for value, (B->max_M + 1) for col_scan
    // r_cscan_buffer_sz_max is for both row and col_scan which have the same type.
    int valbyidx = sizeof(value_t) / sizeof(index_t);

    // compute send_size_max for A
    nnz_t v_buffer_sz_max_RA       = valbyidx * A->nnz_max;
    nnz_t r_cscan_buffer_sz_max_RA = A->nnz_max + A->M_max + 1;
    nnz_t send_size_max_RA         = v_buffer_sz_max_RA + r_cscan_buffer_sz_max_RA;

    // compute send_size_max for P (which is actually R)
    nnz_t v_buffer_sz_max_RAP       = valbyidx * R->nnz_max;
    nnz_t r_cscan_buffer_sz_max_RAP = R->nnz_max + R->M_max + 1;
    nnz_t send_size_max_RAP         = v_buffer_sz_max_RAP + r_cscan_buffer_sz_max_RAP;

    nnz_t send_size_max = std::max(send_size_max_RA, send_size_max_RAP);
    mempool3            = new index_t[2 * send_size_max];

//    mempool1 = std::make_unique<value_t[]>(matmat_size_thre2);
//    mempool2 = std::make_unique<index_t[]>(A->Mbig * 4);

//    index_t B_col_size = B->Mbig; // for original B
//    index_t B_col_size = B->M;    // for when tranpose of B is used to do the multiplication.

#ifdef __DEBUG1__
    if (verbose_triple_mat_mult) {
        MPI_Barrier(comm);
        if (rank == 0) printf("triple_mat_mult: step 4\n");
        MPI_Barrier(comm);
//        if (rank == 0) std::cout << "send_size_max_RA: "     << send_size_max_RA
//                                 << ",\tsend_size_max_RAP: " << send_size_max_RAP
//                                 << ",\tsend_size_max: "     << send_size_max << ",\tvalbyidx:" << valbyidx << std::endl;
//        if(rank==0){
//            std::cout << "mempool1 size = " << matmat_size_thre2 << std::endl;
//            std::cout << "mempool2 size = " << 2 * R_row_size + 2 * max_col_sz << std::endl;
//            std::cout << "mempool3 size = " << 2 * send_size_max << std::endl;
//        }
//        MPI_Barrier(comm);
    }
//    if(rank==0) std::cout << "valbyidx = " << valbyidx << std::endl;
#endif

    // =======================================
    // perform the multiplication R * A
    // =======================================

    saena_matrix RA(comm);
    matmat_CSC(Rcsc, Acsc, RA);

#ifdef __DEBUG1__
    if (verbose_triple_mat_mult) {
        MPI_Barrier(comm);
        if (rank == 0) printf("triple_mat_mult: step 5\n");
        MPI_Barrier(comm);
    }
//    print_vector(RA.entry, -1, "RA.entry", comm);
#endif

    // =======================================
    // free memory from the previous part
    // =======================================

    // keep the mempool parameters for the next part

    delete []Rcsc.row;
    delete []Rcsc.val;
    delete []Rcsc.col_scan;

    delete []Acsc.row;
    delete []Acsc.val;
    delete []Acsc.col_scan;

#ifdef __DEBUG1__
    if (verbose_triple_mat_mult) {
        MPI_Barrier(comm);
        if (rank == 0) printf("triple_mat_mult: step 6\n");
        MPI_Barrier(comm);
    }
#endif

    // =======================================
    // Convert RA to CSC
    // =======================================

    CSCMat RAcsc;
    RAcsc.nnz      = RA.entry.size();
    RAcsc.col_sz   = A->Mbig;
//    RAcsc.max_M    = R->M_max; // we only need this for the right-hand side matrix in matmat.
    RAcsc.row      = new index_t[RAcsc.nnz];
    RAcsc.val      = new value_t[RAcsc.nnz];
    RAcsc.col_scan = new index_t[RAcsc.col_sz + 1];

    // compute nnz_max
    MPI_Allreduce(&RAcsc.nnz, &RAcsc.max_nnz, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);

    std::fill(&RAcsc.col_scan[0], &RAcsc.col_scan[RAcsc.col_sz + 1], 0);
    index_t *RAc_tmp = &RAcsc.col_scan[1];
    for(nnz_t i = 0; i < RAcsc.nnz; i++){
        RAcsc.row[i] = RA.entry[i].row;
        RAcsc.val[i] = RA.entry[i].val;
        RAc_tmp[RA.entry[i].col]++;
    }

    for(nnz_t i = 0; i < RAcsc.col_sz; i++){
        RAcsc.col_scan[i+1] += RAcsc.col_scan[i];
    }

    RAcsc.split = R->splitNew;
//    Rcsc.nnz_list = R->nnz_list;

    // compute nnz_list
//    nnz_list.resize(nprocs);
//    MPI_Allgather(&nnz_l, 1, MPI_UNSIGNED_LONG, &nnz_list[0], 1, MPI_UNSIGNED_LONG, comm);
//        print_vector(nnz_list, 1, "nnz_list", comm);

#ifdef __DEBUG1__
    if (verbose_triple_mat_mult) {
        MPI_Barrier(comm);
        if (rank == 0) printf("triple_mat_mult: step 7\n");
        MPI_Barrier(comm);
//        printf("RA: rank %d: nnz: %lu, \tmax_nnz: %lu, \tcol_sz: %u, \tmax_M: %u\n",
//                rank, RAcsc.nnz, RAcsc.max_nnz, RAcsc.col_sz, RAcsc.max_M);
//        MPI_Barrier(comm);
    }

//    R->print_entry(0);
//    printf("A: nnz_l: %ld\tnnz_g: %ld\tM: %d\tM_big: %d\n", R->nnz_l, R->nnz_g, R->M, R->Mbig);
//    print_array(Rcsc.row, Rcsc.nnz, 0, "Rcsc.row", comm);
//    print_array(Rcsc.val, Rcsc.nnz, 0, "Rcsc.val", comm);
//    print_array(Rcsc.col_scan, Rcsc.col_sz + 1, 0, "Rcsc.col_scan", comm);

//    if(rank==1){
//        std::cout << "\nRA: nnz: " << RAcsc.nnz << std::endl ;
//        for(index_t j = 0; j < RAcsc.col_sz; j++){
//            for(index_t i = RAcsc.col_scan[j]; i < RAcsc.col_scan[j+1]; i++){
//                std::cout << std::setprecision(4) << RAcsc.row[i] << "\t" << j << "\t" << RAcsc.val[i] << std::endl;
//            }
//        }
//    }
#endif

    // =======================================
    // Convert the local transpose of R to CSC
    // =======================================

    // Instead of using P, we use the local tranpose of R.

    // make a copy of R entries, then change their order to row-major
    std::vector<cooEntry> P_ent(R->entry);
    std::sort(P_ent.begin(), P_ent.end(), row_major);

    CSCMat Pcsc;
    Pcsc.nnz      = R->nnz_l;
    Pcsc.col_sz   = R->M;
    Pcsc.max_nnz  = R->nnz_max;
    Pcsc.max_M    = R->M_max;
    Pcsc.row      = new index_t[Pcsc.nnz];
    Pcsc.val      = new value_t[Pcsc.nnz];
    Pcsc.col_scan = new index_t[Pcsc.col_sz + 1];

    Pcsc.split    = R->splitNew;
    Pcsc.nnz_list = R->nnz_list;

    std::fill(&Pcsc.col_scan[0], &Pcsc.col_scan[Pcsc.col_sz + 1], 0);
    index_t *Pc_tmp   = &Pcsc.col_scan[1];
//    index_t *Pc_tmp_p = &Pc_tmp[0] - Pcsc.split[rank]; // use this to avoid subtracting a fixed number

    for(nnz_t i = 0; i < Pcsc.nnz; i++){
        Pcsc.row[i] = P_ent[i].col;
        Pcsc.val[i] = P_ent[i].val;
        Pc_tmp[P_ent[i].row]++;
    }

    for(nnz_t i = 0; i < Pcsc.col_sz; i++){
        Pcsc.col_scan[i+1] += Pcsc.col_scan[i];
    }

#ifdef __DEBUG1__
    if (verbose_triple_mat_mult) {
        MPI_Barrier(comm);
        if (rank == 0) printf("triple_mat_mult: step 8\n");
        MPI_Barrier(comm);
//        printf("P: rank %d: nnz: %lu, \tmax_nnz: %lu, \tcol_sz: %u, \tmax_M: %u\n",
//                rank, Pcsc.nnz, Pcsc.max_nnz, Pcsc.col_sz, Pcsc.max_M);
//        MPI_Barrier(comm);
//        print_array(Pcsc.col_scan, Pcsc.col_sz+1, 1, "Pcsc.col_scan", comm);
    }

//    R->print_entry(0);
//    printf("R: nnz_l: %ld\tnnz_g: %ld\tM: %d\tM_big: %d\n", R->nnz_l, R->nnz_g, R->M, R->Mbig);

//    if (rank == 1) {
//        std::cout << "\nP transpose: nnz: " << R->nnz_l << std::endl;
//        for (index_t j = 0; j < R->M; j++) {
//            for (index_t i = Pcsc.col_scan[j]; i < Pcsc.col_scan[j + 1]; i++) {
//                std::cout << std::setprecision(4) << Pcsc.row[i] << "\t" << j + Pcsc.split[rank] << "\t" << Pcsc.val[i] << std::endl;
//            }
//        }
//    }
#endif

    // =======================================
    // perform the multiplication RA * P
    // =======================================

//    saena_matrix RAP;
    MPI_Comm comm_temp = Ac->comm;
    Ac->comm           = A->comm;
    matmat_CSC(RAcsc, Pcsc, *Ac);
    Ac->comm           = comm_temp;

#ifdef __DEBUG1__
//    print_vector(Ac->entry, -1, "Ac->entry", comm);
//    printf("triple_mat_mult: step 7777\n");
#endif

    // =======================================
    // finalize
    // =======================================

#ifdef __DEBUG1__
    if (verbose_triple_mat_mult) {
        MPI_Barrier(comm);
        if (rank == 0) printf("triple_mat_mult: step 9\n");
        MPI_Barrier(comm);
    }
#endif

    delete []RAcsc.row;
    delete []RAcsc.val;
    delete []RAcsc.col_scan;

    delete []Pcsc.row;
    delete []Pcsc.val;
    delete []Pcsc.col_scan;

//    delete[] mempool1;
//    delete[] mempool2;
    delete[] mempool3;

#ifdef __DEBUG1__
    if (verbose_triple_mat_mult) {
        MPI_Barrier(comm);
        if (rank == 0) printf("end of triple_mat_mult\n\n");
        MPI_Barrier(comm);
    }
#endif

    return 0;
}
