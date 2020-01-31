
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
