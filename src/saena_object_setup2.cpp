#include "saena_object.h"
#include "saena_matrix.h"
#include "strength_matrix.h"
#include "prolong_matrix.h"
#include "restrict_matrix.h"


int saena_object::compute_coarsen(Grid *grid) {
    // Output: Ac = R * A * P

    saena_matrix    *A  = grid->A;
    prolong_matrix  *P  = &grid->P;
    restrict_matrix *R  = &grid->R;
    saena_matrix    *Ac = &grid->Ac;

    MPI_Comm comm = A->comm;
    int nprocs = -1, rank = -1;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
//    print_vector(A->entry, -1, "A->entry", comm);
//    print_vector(P->entry, -1, "P->entry", comm);
//    print_vector(R->entry, 0, "R->entry", comm);

//    Ac->active_old_comm = true;

//    int rank1, nprocs1;
//    MPI_Comm_size(comm, &nprocs1);
//    MPI_Comm_rank(comm, &rank1);
//    if(A->active_old_comm)
//        printf("rank = %d, nprocs = %d active\n", rank1, nprocs1);

//#ifdef SPLIT_NNZ
//    if(rank==0) printf("\nfast_mm: split based on nnz\n");
//#endif
//#ifdef SPLIT_SIZE
//    if(rank==0) printf("\nfast_mm: split based on matrix size\n");
//#endif

    if (verbose_compute_coarsen) {
        MPI_Barrier(comm);
        if (rank == 0) printf("start of compute_coarsen nprocs: %d \n", nprocs);
        MPI_Barrier(comm);
    }

    if (verbose_compute_coarsen2) {
        MPI_Barrier(comm);
        printf("rank %d: A.Mbig = %u, \tA.M = %u, \tA.nnz_g = %lu, \tA.nnz_l = %lu \n",
                rank, A->Mbig, A->M, A->nnz_g, A->nnz_l);
        MPI_Barrier(comm);
        printf("rank %d: P.Mbig = %u, \tP.M = %u, \tP.nnz_g = %lu, \tP.nnz_l = %lu \n",
                rank, P->Mbig, P->M, P->nnz_g, P->nnz_l);
        MPI_Barrier(comm);
        printf("rank %d: R.Mbig = %u, \tR.M = %u, \tR.nnz_g = %lu, \tR.nnz_l = %lu \n",
                rank, R->Mbig, R->M, R->nnz_g, R->nnz_l);
        MPI_Barrier(comm);
    }
#endif

    // set some of Ac parameters
    // -------------------------
    Ac->comm  = A->comm;
    Ac->Mbig  = P->Nbig;
    Ac->Nbig  = P->Nbig;
    Ac->split = P->splitNew;
    Ac->M     = Ac->split[rank+1] - Ac->split[rank];
    Ac->M_old = Ac->M;

    // set dense parameters
    Ac->density = (static_cast<float>(Ac->nnz_g) / Ac->Mbig) / Ac->Mbig;

    // set shrink parameters
    Ac->last_M_shrink       = A->last_M_shrink;
    Ac->last_density_shrink = A->last_density_shrink;
    Ac->cpu_shrink_thre1    = A->cpu_shrink_thre1; //todo: is this required?
    if(A->cpu_shrink_thre2_next_level != -1) // this is -1 by default.
        Ac->cpu_shrink_thre2 = A->cpu_shrink_thre2_next_level;
    //return these to default, since they have been used in the above part.
    A->cpu_shrink_thre2_next_level = -1;
    A->enable_shrink_next_level = false;

    Ac->comm_old = Ac->comm;

#ifdef __DEBUG1__
//    MPI_Barrier(comm);
//    printf("Ac: rank = %d \tMbig = %u \tM = %u \tnnz_g = %lu \tnnz_l = %lu \tdensity = %f\n",
//           rank, Ac->Mbig, Ac->M, Ac->nnz_g, Ac->nnz_l, Ac->density);
//    MPI_Barrier(comm);

//    if(verbose_compute_coarsen){
//        printf("\nrank = %d, Ac->Mbig = %u, Ac->M = %u, Ac->nnz_l = %lu, Ac->nnz_g = %lu \n", rank, Ac->Mbig, Ac->M, Ac->nnz_l, Ac->nnz_g);}

//    print_vector(Ac->split, 0, "Ac->split", Ac->comm);
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); if(!rank) printf("compute_coarsen: step 2: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // minor shrinking
    // -------------------------------------
    Ac->active       = true;
    Ac->active_minor = true;
    for(index_t i = 0; i < Ac->split.size()-1; ++i){
        if(Ac->split[i+1] - Ac->split[i] == 0){
//            printf("rank %d: shrink minor in compute_coarsen: i = %d, split[i] = %d, split[i+1] = %d\n",
//                    rank, i, Ac->split[i], Ac->split[i+1]);
            Ac->shrink_cpu_minor();
            break;
        }
    }

#ifdef __DEBUG1__
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); if(!rank) printf("compute_coarsen: step 3: rank = %d\n", rank); MPI_Barrier(comm);
    }
#endif

    // *******************************************************
    // perform the triple multiplication: R*A*P
    // *******************************************************

    // reserve memory for matmat_size_thre2 used in fast_mm case1
//    map_matmat.reserve(2*matmat_size_thre2);
//    for(nnz_t i = 0; i < matmat_size_thre2; i++){
//        map_matmat[i] = 0;
//    }

//    MPI_Barrier(grid->A->comm);
//    double t11 = MPI_Wtime();

//    std::vector<cooEntry_row> RAP_row_sorted;

    if (coarsen_method == "recursive") {
        triple_mat_mult(grid);
    } else if (coarsen_method == "basic") {
        printf("Update coarsen_method = <basic>\n");
//        triple_mat_mult_basic(grid, RAP_row_sorted);
    } else if (coarsen_method == "no_overlap") {
        printf("Update coarsen_method = <no_overlap>\n");
//        triple_mat_mult_no_overlap(grid, RAP_row_sorted);
    } else {
        printf("wrong coarsen method!\n");
    }

#ifdef __DEBUG1__
    if (verbose_compute_coarsen) {
        MPI_Barrier(comm); if(!rank) printf("compute_coarsen: step 4: rank = %d\n", rank); MPI_Barrier(comm);
    }
#endif

    // *******************************************************
    // form Ac
    // *******************************************************

    filter(Ac->entry);

    if (doSparsify) {

        // *******************************************************
        // Sparsification
        // *******************************************************

        // compute Frobenius norm squared (norm_frob_sq).
        double max_val = 0;
        double norm_frob_sq_local = 0, norm_frob_sq = 0;
        for (nnz_t i = 0; i < Ac->nnz_l; i++) {
//            if( fabs(val_temp) > sparse_epsilon / 2 / Ac->Mbig)
//            if(temp.val * temp.val > sparse_epsilon * sparse_epsilon / (4 * Ac->Mbig * Ac->Mbig) ){
            norm_frob_sq_local += Ac->entry[i].val * Ac->entry[i].val;
            if (fabs(Ac->entry[i].val) > max_val) {
                max_val = Ac->entry[i].val;
            }
        }

        MPI_Allreduce(&norm_frob_sq_local, &norm_frob_sq, 1, MPI_DOUBLE, MPI_SUM, comm);

#ifdef __DEBUG1__
//        if(rank==0) printf("\noriginal size   = %lu\n", Ac_orig.size());
//        if(rank==0) printf("\noriginal size without sparsification   \t= %lu\n", no_sparse_size);
//        if(rank==0) printf("filtered Ac size before sparsification \t= %lu\n", Ac_orig.size());

//        std::sort(Ac_orig.begin(), Ac_orig.end());
//        print_vector(Ac_orig, -1, "Ac_orig", A->comm);
#endif

        auto sample_size_local = (nnz_t) (sample_sz_percent * Ac->nnz_l);
        nnz_t sample_size = 0;
        MPI_Allreduce(&sample_size_local, &sample_size, 1, par::Mpi_datatype<nnz_t>::value(), MPI_SUM, comm);

#ifdef __DEBUG1__
//        if(rank==0) printf("sample_size     = %lu \n", sample_size);
#endif

//        if(sparsifier == "TRSL"){
//            sparsify_trsl1(Ac_orig, Ac->entry, norm_frob_sq, sample_size, comm);
//        }else if(sparsifier == "drineas"){
//            sparsify_drineas(Ac_orig, Ac->entry, norm_frob_sq, sample_size, comm);
//        }else if(sparsifier == "majid"){
//            sparsify_majid(Ac_orig, Ac->entry, norm_frob_sq, sample_size, max_val, comm);
//        }else{
//            printf("\nerror: wrong sparsifier!");
//        }

        std::vector<cooEntry> Ac_orig(std::move(Ac->entry));
        if (Ac->active_minor) {
            if (sparsifier == "majid") {
                sparsify_majid(Ac_orig, Ac->entry, norm_frob_sq, sample_size, max_val, Ac->comm);
            } else {
                printf("\nerror: wrong sparsifier!");
            }
        }

    }

#ifdef __DEBUG1__
//    print_vector(Ac->entry, -1, "Ac->entry", A->comm);
    if (verbose_compute_coarsen) {
        MPI_Barrier(comm); if(!rank) printf("compute_coarsen: step 5: rank = %d\n", rank); MPI_Barrier(comm);
    }
#endif

    // *******************************************************
    // setup matrix
    // *******************************************************
    // Update this description: Shrinking gets decided inside repartition_nnz() or repartition_row() functions,
    // then repartition happens.
    // Finally, shrink_cpu() and matrix_setup() are called. In this way, matrix_setup is called only once.

    Ac->nnz_l = Ac->entry.size();
    MPI_Allreduce(&Ac->nnz_l, &Ac->nnz_g, 1, par::Mpi_datatype<nnz_t>::value(), MPI_SUM, comm);

#ifdef __DEBUG1__
    if (verbose_compute_coarsen) {
        MPI_Barrier(comm); if(!rank) printf("compute_coarsen: step 6: rank = %d\n", rank); MPI_Barrier(comm);
    }
//    Ac->print_info(-1);
//    Ac->print_entry(-1);
#endif

    if(Ac->active_minor) {

        comm = Ac->comm;
        int rank_new = 0;
        MPI_Comm_rank(Ac->comm, &rank_new);

        // ********** decide about shrinking **********
        //---------------------------------------------

        if(nprocs > 1){
            // decide if this is the last level, then execute decide_shrinking_c(), otherwise check for a normal shrink.
            bool shrink_c = false;

            if(Ac->enable_shrink_c){
                if(!dynamic_levels){
                    if(grid->level + 1 == max_level){
                        shrink_c = true;
                    }
                }else{
                    if(Ac->p_order == 1){
                        if((Ac->Mbig <= least_row_threshold) ||
                           (static_cast<float>(Ac->Mbig) / A->Mbig > row_reduction_up_thrshld)){
                            shrink_c = true;
                        }
                    }
                }
            }

            if(shrink_c){ //coarsest level
                Ac->decide_shrinking_c();
            }else if (Ac->enable_shrink) {
//                MPI_Barrier(Ac->comm); if(rank_new==0) printf("start decide shrinking\n"); MPI_Barrier(Ac->comm);
                Ac->matrix_setup_dummy();
                Ac->compute_matvec_dummy_time();
                Ac->decide_shrinking(A->matvec_dummy_time);
                Ac->erase_after_shrink();
            }
        }

#ifdef __DEBUG1__
        if (verbose_compute_coarsen) {
            MPI_Barrier(comm); if(!rank) printf("compute_coarsen: step 7: rank = %d\n", rank); MPI_Barrier(comm);
        }
#endif

        Ac->density = (Ac->nnz_g / double(Ac->Mbig)) / (Ac->Mbig);

        // decide to partition based on number of rows or nonzeros.
        if (switch_repart && Ac->density >= repart_thre) {
            Ac->repart_row = true;
//            if (!rank){
//                printf("equi-ROW partition for the next level: \ndensity = %.2f, repart_thre = %.2f \n",
//                       Ac->density, repart_thre);
//            }
        }

        Ac->repart();

#ifdef __DEBUG1__
        if (verbose_compute_coarsen) {
            MPI_Barrier(comm); if(!rank) printf("compute_coarsen: step 8: rank = %d\n", rank); MPI_Barrier(comm);
        }
#endif

        if(nprocs > 1){
            grid->repart_u_prepare();
        }

        Ac->active = true;
        if (Ac->shrinked) {
            if(!dynamic_levels && grid->level + 1 == max_level) {
                Ac->shrink_cpu_c();
            }else{
                Ac->shrink_cpu();
            }
        }

#ifdef __DEBUG1__
        if (verbose_compute_coarsen) {
            MPI_Barrier(comm); if(!rank) printf("compute_coarsen: step 9: rank = %d\n", rank); MPI_Barrier(comm);
        }
#endif

        if (Ac->active) {
            if (switch_to_dense && Ac->density > dense_thre && Ac->Mbig <= dense_sz_thre) {
                Ac->use_dense = true;
#ifdef __DEBUG1__
                if (verbose_compute_coarsen) {
//                    Ac->print_info(-1);
//                    Ac->print_entry(-1);
                    if (!rank)
                        printf("Switch to dense: density = %f, dense_thres = %f, dense_sz_thre= %d\n",
                               Ac->density, dense_thre, dense_sz_thre);
                }
#endif
            }

            Ac->matrix_setup(scale);
        }
    }

    comm = grid->A->comm;

#ifdef __DEBUG1__
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); if(!rank) printf("compute_coarsen: done! rank = %d\n", rank); MPI_Barrier(comm);
    }

    // view matrix Ac
    // --------------
//    petsc_viewer(Ac);
#endif

    return 0;
} // compute_coarsen()


int saena_object::triple_mat_mult(Grid *grid, bool symm /*=true*/){

    saena_matrix    *A  = grid->A;
    prolong_matrix  *P  = &grid->P;
    restrict_matrix *R  = &grid->R;
    saena_matrix    *Ac = &grid->Ac;

    MPI_Comm comm = A->comm;
    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
//    print_vector(A->entry, -1, "A->entry", comm);
//    print_vector(P->entry, -1, "P->entry", comm);
//    print_vector(R->entry, -1, "R->entry", comm);

    if (verbose_triple_mat_mult) {
//        MPI_Barrier(comm);
//        if (rank == 0) printf("\nstart of triple_mat_mult: nprocs: %d \n", nprocs);
//        MPI_Barrier(comm);
//        printf("rank %d: A: Mbig = %u, \tM = %u, \tnnz_g = %lu, \tnnz_l = %lu\n",
//                rank, A->Mbig, A->M, A->nnz_g, A->nnz_l);
//        MPI_Barrier(comm);
//        printf("rank %d: P: Mbig = %u, \tM = %u, \tnnz_g = %lu, \tnnz_l = %lu, \tNbig = %u\n",
//                rank, P->Mbig, P->M, P->nnz_g, P->nnz_l, P->Nbig);
//        MPI_Barrier(comm);
//        printf("rank %d: R: Mbig = %u, \tM = %u, \tnnz_g = %lu, \tnnz_l = %lu, \tNbig = %u\n",
//                rank, R->Mbig, R->M, R->nnz_g, R->nnz_l, R->Nbig);
//        MPI_Barrier(comm);
//        print_vector(A->split,    1, "A->split",    comm);
//        print_vector(R->splitNew, 1, "R->splitNew", comm);
//        print_vector(A->nnz_list, 1, "A->nnz_list", comm);
//        print_vector(R->nnz_list, 1, "R->nnz_list", comm);
    }
#endif

    // =======================================
    // Convert R to CSC
    // =======================================

#ifdef __DEBUG1__
    for(nnz_t i = 0; i < R->nnz_l; i++){
//        std::cout << i << "\t" << R->entry[i] << std::endl;
        assert( (R->entry[i].row >= 0) && (R->entry[i].row < R->M) );
        assert( (R->entry[i].col >= 0) && (R->entry[i].col < R->Nbig) );
//        ASSERT( fabs(R->entry[i].val) > ALMOST_ZERO, i << "\t" << R->entry[i] );
    }
#endif

    CSCMat Rcsc;
    Rcsc.comm     = R->comm;
    Rcsc.nnz      = R->nnz_l;
    Rcsc.col_sz   = R->Nbig;        //TODO: check if this is correct or Mbig should be used.
    Rcsc.max_nnz  = R->nnz_max;
    Rcsc.max_M    = R->M_max;
    Rcsc.row      = new index_t[Rcsc.nnz];
    Rcsc.val      = new value_t[Rcsc.nnz];
    Rcsc.col_scan = new index_t[Rcsc.col_sz + 1];

    std::fill(&Rcsc.col_scan[0], &Rcsc.col_scan[Rcsc.col_sz + 1], 0);
    Rcsc.col_scan[0] = 1; // starts from 1, not 0, because the mkl function indexing starts from 1.

    index_t *Rc_tmp = &Rcsc.col_scan[1];
    for(nnz_t i = 0; i < Rcsc.nnz; i++){
        Rcsc.row[i] = R->entry[i].row + 1; // make the rows start from 1. when done with multiply, add this to the result.
        Rcsc.val[i] = R->entry[i].val;
        Rc_tmp[R->entry[i].col]++;
    }

    if(Rcsc.nnz != 0) {
        for (nnz_t i = 0; i < Rcsc.col_sz; i++) {
            Rcsc.col_scan[i + 1] += Rcsc.col_scan[i];
        }
    }

    Rcsc.split    = R->splitNew;        //TODO: check if this is correct or "split" should be used.
    Rcsc.nnz_list = R->nnz_list;

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
//    print_array(Rcsc.row, Rcsc.nnz, 1, "Rcsc.row", comm);
//    print_array(Rcsc.val, Rcsc.nnz, 1, "Rcsc.val", comm);
//    print_array(Rcsc.col_scan, Rcsc.col_sz + 1, 1, "Rcsc.col_scan", comm);

//    std::cout << "\nR: nnz: " << Rcsc.nnz << std::endl ;
//    for(index_t j = 0; j < Rcsc.col_sz; j++){
//        for(index_t i = Rcsc.col_scan[j] - 1; i < Rcsc.col_scan[j+1] - 1; i++){ // indexing starts from 1, not 0.
//            std::cout << std::setprecision(4) << Rcsc.row[i] << "\t" << j << "\t" << Rcsc.val[i] << std::endl;
//        }
//    }
    }
#endif

    // =======================================
    // Convert the local transpose of A to CSC
    // =======================================

#ifdef __DEBUG1__
    for(nnz_t i = 0; i < A->nnz_l; i++){
        assert( (A->entry[i].row - A->split[rank] >= 0) && (A->entry[i].row - A->split[rank] < A->M) );
        assert( (A->entry[i].col >= 0) && (A->entry[i].col < A->Mbig) );
//        ASSERT( fabs(A->entry[i].val) > ALMOST_ZERO, i << A->entry[i] );
    }
#endif

    // make a copy of A entries, then change their order to row-major
    std::vector<cooEntry> A_ent(A->entry);
    std::sort(A_ent.begin(), A_ent.end(), row_major);

    CSCMat Acsc;
    Acsc.comm     = A->comm;
    Acsc.nnz      = A->nnz_l;
    Acsc.col_sz   = A->M;
    Acsc.max_nnz  = A->nnz_max;
    Acsc.max_M    = A->M_max;
    Acsc.row      = new index_t[Acsc.nnz];
    Acsc.val      = new value_t[Acsc.nnz];
    Acsc.col_scan = new index_t[Acsc.col_sz + 1];

    std::fill(&Acsc.col_scan[0], &Acsc.col_scan[Acsc.col_sz + 1], 0);
    Acsc.col_scan[0] = 1;

//    index_t *Ac_tmp   = &Acsc.col_scan[1];
    index_t *Ac_tmp_p = &Acsc.col_scan[1] - A->split[rank]; // use this to avoid subtracting a fixed number

    for(nnz_t i = 0; i < Acsc.nnz; ++i){
        Acsc.row[i] = A_ent[i].col + 1;
        Acsc.val[i] = A_ent[i].val;
        Ac_tmp_p[A_ent[i].row]++;
    }

    for(nnz_t i = 0; i < Acsc.col_sz; ++i){
        Acsc.col_scan[i+1] += Acsc.col_scan[i];
    }

    Acsc.split    = A->split;
    Acsc.nnz_list = A->nnz_list;

    Acsc.use_trans = true;

#ifdef __DEBUG1__
    {
        if (verbose_triple_mat_mult) {
            MPI_Barrier(comm);
            if (rank == 0) printf("triple_mat_mult: step 2\n");
            MPI_Barrier(comm);
//            printf("A: rank %d: nnz: %lu, \tmax_nnz: %lu, \tcol_sz: %u, \tmax_M: %u\n",
//                   rank, Acsc.nnz, Acsc.max_nnz, Acsc.col_sz, Acsc.max_M);
//            MPI_Barrier(comm);
        }

//        A->print_entry(0);
//        printf("A: nnz_l: %ld\tnnz_g: %ld\tM: %d\tM_big: %d\n", A->nnz_l, A->nnz_g, A->M, A->Mbig);
//        print_array(Acsc.row, Acsc.nnz, 0, "Acsc.row", comm);
//        print_array(Acsc.val, Acsc.nnz, 0, "Acsc.val", comm);
//        print_array(Acsc.col_scan, Acsc.col_sz + 1, 0, "Acsc.col_scan", comm);

//        std::cout << "\nA: nnz: " << Acsc.nnz << std::endl ;
        for(index_t j = 0; j < Acsc.col_sz; j++){
            for(index_t i = Acsc.col_scan[j] - 1; i < Acsc.col_scan[j+1] - 1; i++){
                ASSERT(Acsc.row[i] >= 0, Acsc.row[i] << "\t" << j << "\t" << Acsc.val[i]);
//                std::cout << std::setprecision(4) << Acsc.row[i] << "\t" << j << "\t" << Acsc.val[i] << std::endl;
            }
        }
    }
#endif

    // =======================================
    // prepare A for compression
    // =======================================

    if(nprocs > 1) {
        Acsc.compress_prep();
    }

#ifdef __DEBUG1__
    if (verbose_triple_mat_mult) {
        MPI_Barrier(comm);
        if (rank == 0) printf("triple_mat_mult: step 3\n");
        MPI_Barrier(comm);
    }
#endif

    // =======================================
    // Preallocate Memory
    // =======================================

    matmat_memory_alloc(Rcsc, Acsc);
    C_temp.reserve(10 * std::max(Acsc.nnz, Rcsc.nnz));

    saena_matrix RA(comm);
    RA.entry.reserve(std::max(Acsc.nnz, Rcsc.nnz));

#ifdef __DEBUG1__
    if (verbose_triple_mat_mult) {
        MPI_Barrier(comm);
        if (rank == 0) printf("triple_mat_mult: step 4\n");
        MPI_Barrier(comm);
    }
#endif

    // =======================================
    // perform the multiplication R * A
    // =======================================

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

    Rcsc.row = nullptr;
    Rcsc.val = nullptr;
    Rcsc.col_scan = nullptr;

    delete []Acsc.row;
    delete []Acsc.val;
    delete []Acsc.col_scan;

    Acsc.row = nullptr;
    Acsc.val = nullptr;
    Acsc.col_scan = nullptr;

    matmat_memory_free();

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
    RAcsc.comm     = R->comm;
    RAcsc.nnz      = RA.entry.size();
    RAcsc.col_sz   = A->Mbig;
//    RAcsc.max_M    = R->M_max;    // we only need this for the right-hand side matrix in matmat.
    RAcsc.row      = new index_t[RAcsc.nnz];
    RAcsc.val      = new value_t[RAcsc.nnz];
    RAcsc.col_scan = new index_t[RAcsc.col_sz + 1];

    // compute nnz_max
    MPI_Allreduce(&RAcsc.nnz, &RAcsc.max_nnz, 1, par::Mpi_datatype<nnz_t>::value(), MPI_MAX, comm);

    std::fill(&RAcsc.col_scan[0], &RAcsc.col_scan[RAcsc.col_sz + 1], 0);
    RAcsc.col_scan[0] = 1;

    // make the rows start from 1. when done with multiply, add this to the result.
    const index_t OFST = R->splitNew[rank] - 1;

    index_t *RAc_tmp = &RAcsc.col_scan[1];
    for(nnz_t i = 0; i < RAcsc.nnz; ++i){
        RAcsc.row[i] = RA.entry[i].row - OFST;
        RAcsc.val[i] = RA.entry[i].val;
        ++RAc_tmp[RA.entry[i].col];
    }

    if(RAcsc.nnz != 0) {
        for (nnz_t i = 0; i < RAcsc.col_sz; i++) {
            RAcsc.col_scan[i + 1] += RAcsc.col_scan[i];
        }
    }

    RAcsc.split = R->splitNew;

    // compute nnz_list
    RAcsc.nnz_list.resize(nprocs);
    MPI_Allgather(&RAcsc.nnz, 1, par::Mpi_datatype<nnz_t>::value(), &RAcsc.nnz_list[0], 1, par::Mpi_datatype<nnz_t>::value(), comm);

#ifdef __DEBUG1__
    {
        if (verbose_triple_mat_mult) {
            MPI_Barrier(comm);
            if (rank == 0) printf("triple_mat_mult: step 7\n");
            MPI_Barrier(comm);
//            printf("RA: rank %d: nnz: %lu, \tmax_nnz: %lu, \tcol_sz: %u, \tmax_M: not needed for lhs of matmat\n",
//                   rank, RAcsc.nnz, RAcsc.max_nnz, RAcsc.col_sz);
//            MPI_Barrier(comm);
        }

//    RA.print_entry(0);
//    printf("A: nnz_l: %ld\tnnz_g: %ld\tM: %d\tM_big: %d\n", RA->nnz_l, RA.nnz_g, RA.M, RA.Mbig);
//    print_array(RAcsc.row, RAcsc.nnz, 1, "RAcsc.row", comm);
//    print_array(RAcsc.val, RAcsc.nnz, 1, "RAcsc.val", comm);
//    print_array(RAcsc.col_scan, RAcsc.col_sz + 1, 1, "RAcsc.col_scan", comm);

//    print_vector(RAcsc.nnz_list, 1, "RAcsc.nnz_list", comm);

//    if(rank==1){
//        std::cout << "\nRA: nnz: " << RAcsc.nnz << std::endl ;
//        for(index_t j = 0; j < RAcsc.col_sz; j++){
//            for(index_t i = RAcsc.col_scan[j] - 1; i < RAcsc.col_scan[j+1] - 1; i++){
//                std::cout << std::setprecision(4) << RAcsc.row[i] << "\t" << j + 1 << "\t" << RAcsc.val[i] << std::endl;
//            }
//        }
//    }
    }
#endif

    // =======================================
    // Convert the local transpose of R to CSC
    // =======================================

    // Instead of using P, we use the local tranpose of R.

    // make a copy of R entries, then change their order to row-major
    std::vector<cooEntry> P_ent(R->entry);
    std::sort(P_ent.begin(), P_ent.end(), row_major);

    CSCMat Pcsc;
    Pcsc.comm     = R->comm;
    Pcsc.nnz      = R->nnz_l;
    Pcsc.col_sz   = R->M;
    Pcsc.max_nnz  = R->nnz_max;
    Pcsc.max_M    = R->M_max;
    Pcsc.row      = new index_t[Pcsc.nnz];
    Pcsc.val      = new value_t[Pcsc.nnz];
    Pcsc.col_scan = new index_t[Pcsc.col_sz + 1];

    std::fill(&Pcsc.col_scan[0], &Pcsc.col_scan[Pcsc.col_sz + 1], 0);
    Pcsc.col_scan[0] = 1;

    Pcsc.split = R->splitNew;

    index_t *Pc_tmp = &Pcsc.col_scan[1];
    for(nnz_t i = 0; i < Pcsc.nnz; ++i){
        Pcsc.row[i] = P_ent[i].col + 1;
        Pcsc.val[i] = P_ent[i].val;
        ++Pc_tmp[P_ent[i].row];
//        if(rank==1) std::cout << Pcsc.row[i] << "\t" << P_ent[i].row << std::endl;
    }

    for(nnz_t i = 0; i < Pcsc.col_sz; ++i){
        Pcsc.col_scan[i+1] += Pcsc.col_scan[i];
    }

    Pcsc.nnz_list = R->nnz_list;

    Pcsc.use_trans = true;

#ifdef __DEBUG1__
    {
        if (verbose_triple_mat_mult) {
            MPI_Barrier(comm);
            if (rank == 0) printf("triple_mat_mult: step 8\n");
            MPI_Barrier(comm);
//            printf("P: rank %d: nnz: %lu, \tmax_nnz: %lu, \tcol_sz: %u, \tmax_M: %u\n",
//                    rank, Pcsc.nnz, Pcsc.max_nnz, Pcsc.col_sz, Pcsc.max_M);
//            MPI_Barrier(comm);
//            print_array(Pcsc.col_scan, Pcsc.col_sz+1, 1, "Pcsc.col_scan", comm);
        }

//        R->print_entry(0);
//        printf("R: nnz_l: %ld\tnnz_g: %ld\tM: %d\tM_big: %d\n", R->nnz_l, R->nnz_g, R->M, R->Mbig);

//    if (rank == 1) {
//        std::cout << "\nP transpose: nnz: " << R->nnz_l << std::endl;
        for (index_t j = 0; j < R->M; j++) {
            for (index_t i = Pcsc.col_scan[j] - 1; i < Pcsc.col_scan[j + 1] - 1; ++i) {
                ASSERT(Pcsc.row[i] >= 0, Pcsc.row[i] << "\t" << j + Pcsc.split[rank] << "\t" << Pcsc.val[i]);
//                std::cout << std::setprecision(4) << Pcsc.row[i] << "\t" << j + Pcsc.split[rank] << "\t" << Pcsc.val[i] << std::endl;
            }
        }
//    }
    }
#endif

    // =======================================
    // prepare P for compression
    // =======================================

    if(nprocs > 1) {
        Pcsc.compress_prep();
    }

#ifdef __DEBUG1__
    if (verbose_triple_mat_mult) {
        MPI_Barrier(comm);
        if (rank == 0) printf("triple_mat_mult: step 9\n");
        MPI_Barrier(comm);
    }
#endif

    // =======================================
    // Preallocate Memory
    // =======================================

    matmat_memory_alloc(RAcsc, Pcsc);
    Ac->entry.reserve(std::max(RAcsc.nnz, Pcsc.nnz));

#ifdef __DEBUG1__
    if (verbose_triple_mat_mult) {
        MPI_Barrier(comm);
        if (rank == 0) printf("triple_mat_mult: step 10\n");
        MPI_Barrier(comm);
    }
#endif

    // =======================================
    // perform the multiplication RA * P
    // =======================================

    MPI_Comm comm_temp = Ac->comm;
    Ac->comm = A->comm;
    if(symm){
        matmat_CSC(RAcsc, Pcsc, *Ac);
    }else{
        matmat_CSC(RAcsc, Pcsc, *Ac, true); // transpose the result
    }
    Ac->comm = comm_temp;

//    assert(!Ac->entry.empty());

#ifdef __DEBUG1__
//    print_vector(Ac->entry, -1, "Ac->entry", comm);
    if (verbose_triple_mat_mult) {
        MPI_Barrier(comm);
        if (rank == 0) printf("triple_mat_mult: step 11\n");
        MPI_Barrier(comm);
    }
#endif

    // =======================================
    // finalize
    // =======================================

    delete []RAcsc.row;
    delete []RAcsc.val;
    delete []RAcsc.col_scan;

    RAcsc.row = nullptr;
    RAcsc.val = nullptr;
    RAcsc.col_scan = nullptr;

    delete []Pcsc.row;
    delete []Pcsc.val;
    delete []Pcsc.col_scan;

    Pcsc.row = nullptr;
    Pcsc.val = nullptr;
    Pcsc.col_scan = nullptr;

    matmat_memory_free();

#ifdef __DEBUG1__
    if (verbose_triple_mat_mult) {
        MPI_Barrier(comm);
        if (rank == 0) printf("end of triple_mat_mult\n\n");
        MPI_Barrier(comm);
    }
#endif

    return 0;
}


void saena_object::filter(vector<cooEntry> &v) {
    // filter out entries less than THRE
    // keep the diagonal entries

    if(++filter_it < filter_start){
        return;
    }

//    printf("\nfilter next level: filter_it = %d,filter_start = %d\n", filter_it, filter_start);

    filter_thre *= pow(10, filter_rate);
    if(filter_thre > filter_max){
        filter_thre = filter_max ;
    }
    const double THRE = filter_thre;

    vector<cooEntry> w;
    for(const auto &a : v){
        if(fabs(a.val) > THRE || a.row == a.col)
            w.emplace_back(a);
    }
    w.swap(v);
}

// from here: http://www.algolist.net/Algorithms/Sorting/Quicksort
/*
int saena_object::reorder_split(vecEntry *arr, index_t left, index_t right, index_t pivot){
    //pivot is a row instead of an entry of arr

    vecEntry tmp = arr[left];
    index_t i = left, j = right;

//    std::cout << "pivot: " << pivot << std::endl;
    while (i <= j) {

//        std::cout << "\ni:" << "\t" << i << std::endl;
        while (arr[i].row < pivot){
//            std::cout << i << "\t" << arr[i] << std::endl;
            i++;
        }
//        std::cout << i << "\t" << arr[i] << "\tfinal" << std::endl;

//        std::cout << "\nj:" << "\t" << j << std::endl;
        while (arr[j].row >= pivot){
//            std::cout << j << "\t" << arr[j] << std::endl;
            j--;
        }
//        std::cout << j << "\t" << arr[j] << "\tfinal" << std::endl;

        if (i <= j) {
            tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
            i++;
            j--;
        }

    }

    return 0;
}
*/

// This version moves entries of A1 to the begining and A2 to the end of the input array.
void saena_object::reorder_split(CSCMat_mm &A, CSCMat_mm &A1, CSCMat_mm &A2){

#ifdef __DEBUG1__
    int rank = 0, nprocs = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int verbose_rank = 0;

    ++reorder_counter;

    assert(A.nnz != 0);
    ASSERT(A.nnz == (A.col_scan[A.col_sz] - A.col_scan[0]), "A.nnz: " << A.nnz << ", A.col_scan[0]: " << A.col_scan[0]
           << ", A.col_scan[A.col_sz]: " << A.col_scan[A.col_sz]);

//    std::cout << "\nA: nnz: " << A.nnz << std::endl;
    for (int j = 0; j < A.col_sz; ++j) {
        for (index_t i = A.col_scan[j] - 1; i < A.col_scan[j + 1] - 1; ++i) {
            assert( A.r[i] > 0 );
            assert( A.r[i] <= A.row_sz );
//            ASSERT(fabs(A.v[i]) > ALMOST_ZERO, "rank: " << rank << ", A.v[i]: " << A.v[i]);
//            std::cout << std::setprecision(4) << i << "\t" << A.r[i] << "\t" << j+A.col_offset+1 << "\t" << A.v[i] << std::endl;
        }
    }

    if(rank == verbose_rank){
//        std::cout << "\nstart of " << __func__ << std::endl;
//        std::cout << "\n=========================================================================" << std::endl ;
//        std::cout << "\nA: nnz: " << A.nnz << ", size: " << A.row_sz << ", " << A.col_sz
//                  << ", \toffset: " << A.row_offset << ", " << A.col_offset << std::endl;

//        print_array(A.col_scan, A.col_sz+1, verbose_rank, "Ac", MPI_COMM_WORLD);

        // ========================================================
        // this shows how to go through entries of A before changing order.
        // NOTE: column is not correct. col_offset should be added to it.
        // ========================================================

//        printf("=============================================\n");
//        std::cout << "\nA: nnz: " << A.nnz << std::endl;
//        for (index_t j = 0; j < A.col_sz; j++) {
//            for (index_t i = A1.col_scan[j]; i < A1.col_scan[j + 1]; i++) {
//                std::cout << std::setprecision(4) << A.r[i] << "\t" << j+A.col_offset << "\t" << A.v[i] << std::endl;
//            }
//        }
        // ========================================================
    }

#endif

    // ========================================================
    // IMPORTANT: An offset should be used to access Ar and Av.
    // ========================================================

    nnz_t offset = A.col_scan[0] - 1;

    index_t *A1r = &A.r[offset];
    value_t *A1v = &A.v[offset];
    index_t *A2r = &mempool4[0];
    value_t *A2v = &mempool5[0];

    // ========================================================
    // allocate memory and initialize the second half's col_scan
    // ========================================================

    A2.col_scan = new index_t[A2.col_sz + 1];
    A2.free_c   = true;
    auto *Ac2_p = &A2.col_scan[1]; // to do scan on it at the end.

#ifdef SANEA_USE_PSTL
    std::fill(pstl::execution::par_unseq, &A2.col_scan[0], &A2.col_scan[A2.col_sz+1], 0);
#else
    std::fill(&A2.col_scan[0], &A2.col_scan[A2.col_sz+1], 0);
#endif

    // ========================================================
    // form A1 and A2
    // ========================================================

#ifdef __DEBUG1__
    // check memory allocation
    assert(A2.col_scan);

    // print A in top and bottom split
/*    for(index_t j = 0; j < A.col_sz; ++j) {
        if(rank==verbose_rank) std::cout << std::endl;
        for(nnz_t i = A.col_scan[j]; i < A.col_scan[j+1]; ++i) {
            if (A.r[i] < A1.row_sz) {
                if(rank==verbose_rank) std::cout << std::setprecision(4) << A.r[i] << "\t" << j << "\t" << A.v[i] << "\ttop half" << std::endl;
            } else {
                if (rank == verbose_rank)  std::cout << std::setprecision(4) << A.r[i] << "\t" << j << "\t" << A.v[i] << "\tbottom half" << std::endl;
            }
        }
    }*/
#endif

    const int THRSHLD = A1.row_sz;
    int i = 0, j = 0;
    A1.nnz = 0, A2.nnz = 0;
    for(j = 0; j < A.col_sz; ++j){
//        if(rank==verbose_rank) std::cout << "\n" << A.col_scan[j]-1 << "\t" << A.col_scan[j+1]-1 << std::endl;
        for(i = A.col_scan[j] - 1; i < A.col_scan[j+1] - 1; ++i){
//            std::cout << i << "\t" << A.r[i] << "\t" << j << "\t" << A.v[i] << std::endl;

            if(A.r[i] <= THRSHLD){
                A1r[A1.nnz] = A.r[i];
                A1v[A1.nnz] = A.v[i];
                ++A1.nnz;
//                if(rank==verbose_rank) std::cout << std::setprecision(4) << A.r[i] << "\t" << j+1 << "\t" << A.v[i] << "\ttop half: " << std::endl;
            }else{
                A2r[A2.nnz] = A.r[i] - THRSHLD;
                A2v[A2.nnz] = A.v[i];
                ++A2.nnz;
                ++Ac2_p[j];
//                if(rank==verbose_rank) std::cout << std::setprecision(4) << A.r[i] - A1.row_sz << "\t" << j+1 << "\t" << A.v[i] << "\tbottom half: " << std::endl;
            }

        }
/*
        col_nnz = A.col_scan[j+1] - A.col_scan[j];
        if(col_nnz != 0){

            if(A.r[A.col_scan[j+1] - 1] < A1.row_sz){
                memcpy(&A1r[A1.nnz], &A.r[iter0], col_nnz * sizeof(index_t));
                memcpy(&A1v[A1.nnz], &A.v[iter0], col_nnz * sizeof(value_t));
                iter0    += col_nnz;
                A1.nnz   += col_nnz;
            }else if(A.r[A.col_scan[j]] >= A1.row_sz){
//                memcpy(&A2r[A2.nnz], &A.r[iter0], col_nnz * sizeof(index_t));
                memcpy(&A2v[A2.nnz], &A.v[iter0], col_nnz * sizeof(value_t));
//                iter0    += col_nnz;
//                A2.nnz   += col_nnz;
                Ac2_p[j] += col_nnz;

                for(; iter0 < A.col_scan[j+1]; ++iter0, ++A2.nnz){
                    A2r[A2.nnz] = A.r[iter0] - A1.row_sz;
                }

            }else{

                mid = std::distance(&A.r[A.col_scan[j]], std::lower_bound(&A.r[A.col_scan[j]], &A.r[A.col_scan[j + 1]], A1.row_sz));

//                if(rank == verbose_rank) std::cout << "\nrow_sz:" << A.row_sz << ", row half: " << A1.row_sz
//                               << ", mid distance: " << mid << ", mid val: " << A.r[A.col_scan[j] + mid] << "\n";

                // copy to A1
                memcpy(&A1r[A1.nnz], &A.r[iter0], mid * sizeof(index_t));
                memcpy(&A1v[A1.nnz], &A.v[iter0], mid * sizeof(value_t));
                iter0  += mid;
                A1.nnz += mid;

                // copy to A2
//                memcpy(&A2r[A2.nnz], &A.r[iter0], (col_nnz - mid) * sizeof(index_t));
                memcpy(&A2v[A2.nnz], &A.v[iter0], (col_nnz - mid) * sizeof(value_t));
//                iter0    += col_nnz - mid;
//                A2.nnz   += col_nnz - mid;
                Ac2_p[j] += col_nnz - mid;

                for(; iter0 < A.col_scan[j+1]; ++iter0, ++A2.nnz){
                    A2r[A2.nnz] = A.r[iter0] - A1.row_sz;
                }
            }

        } //if(col_nnz != 0){
*/
//        if(rank==verbose_rank) std::cout << "A1.nnz: " << A1.nnz << ", A2.nnz: " << A2.nnz << "\n\n";
    }

#ifdef __DEBUG1__

    if(nprocs > 1) {
        ASSERT(A1.nnz <= mempool4and5_sz, "A1.nnz: " << A1.nnz << ", loc_nnz_max: " << mempool4and5_sz);
        ASSERT(A2.nnz <= mempool4and5_sz, "A2.nnz: " << A2.nnz << ", loc_nnz_max: " << mempool4and5_sz);
    }
    ASSERT(A1.nnz + A2.nnz == A.nnz, "A.nnz: " << A.nnz << ", A1.nnz: " << A1.nnz << ", A2.nnz: " << A2.nnz);

    // older versions
#if 0
    A1.nnz = 0, A2.nnz = 0;
    for(index_t j = 0; j < A.col_sz; ++j){
        for(nnz_t i = A.col_scan[j]; i < A.col_scan[j+1]; ++i){
            if(A.r[i] < A1.row_sz){
                A1r[A1.nnz] = A.r[i];
                A1v[A1.nnz] = A.v[i];
                ++A1.nnz;
//                if(rank==verbose_rank) std::cout << std::setprecision(4) << A.r[i] << "\t" << j << "\t" << A.v[i] << "\ttop half" << std::endl;
            }else{
                A2r[A2.nnz] = A.r[i] - A1.row_sz;
                A2v[A2.nnz] = A.v[i];
                ++A2.nnz;
                ++Ac2_p[j];

//                assert(A.r[i] >= A1.row_sz);
//                assert(j < A2.col_sz);
//                if(rank==verbose_rank) std::cout << std::setprecision(4) << A.r[i] << "\t" << j << "\t" << A.v[i] << "\tbottom half" << std::endl;
            }
        }
    }

    std::vector<index_t> A1r, A2r;
    std::vector<value_t> A1v, A2v;

    A1.nnz = 0, A2.nnz = 0;
    for(index_t j = 0; j < A.col_sz; j++){
        for(nnz_t i = A.col_scan[j]; i < A.col_scan[j+1]; ++i){
            if(A.r[i] < A1.row_sz + A1.row_offset){
                A1r.emplace_back(A.r[i]);
                A1v.emplace_back(A.v[i]);
//                if(rank==verbose_rank) std::cout << std::setprecision(4) << A.r[i] << "\t" << j << "\t" << A.v[i] << "\ttop half" << std::endl;
            }else{
                A2r.emplace_back(A.r[i] - A1.row_sz);
                A2v.emplace_back(A.v[i]);
                Ac2_p[j]++;
//                if(rank==verbose_rank) std::cout << std::setprecision(4) << A.r[i] << "\t" << j << "\t" << A.v[i] << "\tbottom half" << std::endl;
            }
        }
    }
#endif
#endif

    // if A1 does not have any nonzero, then free A2's memory and make A2.col_scan point to A.col_scan and return.
    if(A1.nnz == 0){
        delete [] A2.col_scan;
        A2.col_scan = nullptr;
        A2.free_c   = false;
        A2.col_scan = A1.col_scan;
        A2.r = &A.r[0];
        A2.v = &A.v[0];

#pragma omp simd
        for (i = A2.col_scan[0] - 1; i < A2.col_scan[A2.col_sz] - 1; ++i) {
            A2.r[i] -= THRSHLD;
        }

//        for (j = 0; j < A2.col_sz; ++j) {
//            for (i = A2.col_scan[j] - 1; i < A2.col_scan[j + 1] - 1; ++i) {
//                A2.r[i] -= A1.row_sz;
//            }
//        }

#ifdef __DEBUG1__
        goto reorder_split_end;
#endif
        return;
    }

    // if A2 does not have any nonzero, then free its memory and return.
    if(A2.nnz == 0){
        delete [] A2.col_scan;
        A2.col_scan = nullptr;
        A2.free_c   = false;

#ifdef __DEBUG1__
        goto reorder_split_end;
#endif
        return;
    }

    A2.col_scan[0] += 1;
    for(i = 1; i <= A.col_sz; ++i){
        A2.col_scan[i] += A2.col_scan[i-1]; // scan on A2.col_scan
    }

#pragma omp simd
    for(i = 1; i <= A.col_sz; ++i){
        A1.col_scan[i] -= A2.col_scan[i] - 1; // subtract A2.col_scan from A1.col_scan to have the correct scan for A1
    }

#ifdef __DEBUG1__
//    print_array(A1.col_scan, A1.col_sz+1, verbose_rank, "A1.col_scan", MPI_COMM_WORLD);
//    print_array(A2.col_scan, A2.col_sz+1, verbose_rank, "A2.col_scan", MPI_COMM_WORLD);
#endif

    // First put A1 at the beginning of A, then put A2 at the end A.
    // A1 points to A, so no need to copy A1 into A.
//    memcpy(&A.r[offset], &A1r[0], A1.nnz * sizeof(index_t));
//    memcpy(&A.v[offset], &A1v[0], A1.nnz * sizeof(value_t));

#ifdef SANEA_USE_PSTL
    const int ST = offset + A1.nnz;
    std::copy(pstl::execution::par_unseq, &A.r[ST], &A.r[ST + A2.nnz], &A2r[0]);
    std::copy(pstl::execution::par_unseq, &A.v[ST], &A.v[ST + A2.nnz], &A2v[0]);
#else
    memcpy(&A.r[offset + A1.nnz], &A2r[0], A2.nnz * sizeof(index_t));
    memcpy(&A.v[offset + A1.nnz], &A2v[0], A2.nnz * sizeof(value_t));
#endif

    // set r and v for A2
    A2.r = &A.r[A1.col_scan[A.col_sz] - 1];
    A2.v = &A.v[A1.col_scan[A.col_sz] - 1];

//    std::cout << "\nA2: nnz: " << A2.nnz << std::endl ;
//    for(j = 0; j < A2.col_sz; ++j){
//        for(i = A2.col_scan[j]; i < A2.col_scan[j+1]; ++i){
//            if(rank==verbose_rank) std::cout << std::setprecision(4) << A2.r[i] << "\t" << j+A2.col_offset << "\t" << A2.v[i] << std::endl;
//            A2.r[i] -= A1.row_sz;
//        }
//    }

//    std::transform(&A2.r[0], &A2.r[A2.nnz], &A2.r[0], decrement(A1.row_sz));

#if 0
    // Equivalent to the previous part. Uses for loops instead of memcpy.
    nnz_t arr_idx = offset;
    for(nnz_t i = 0; i < A1r.size(); i++){
        A.r[arr_idx] = A1r[i];
        A.v[arr_idx] = A1v[i];
        arr_idx++;
    }
    for(nnz_t i = 0; i < A2r.size(); i++){
        A.r[arr_idx] = A2r[i];
        A.v[arr_idx] = A2v[i];
        arr_idx++;
    }
#endif

#ifdef __DEBUG1__
    reorder_split_end:

    {
        // assert A1
//        std::cout << "\nA1: nnz: " << A1.nnz << std::endl;
//        nnz_t iter = 0;
        if(A1.nnz != 0) {
            for (j = 0; j < A1.col_sz; ++j) {
                for (i = A1.col_scan[j] - 1; i < A1.col_scan[j + 1] - 1; ++i) {
//                    if(rank == verbose_rank) std::cout << std::setprecision(4) << A1.r[i] << "\t" << j+A1.col_offset+1 << "\t" << A1.v[i] << std::endl;
                    assert(A1.r[i] > 0);
                    assert(A.r[i] <= A1.row_sz);

//                    std::cout << "(rank: " << rank << ", " << i << "): \t(" << A.r[i] << ", " << j << ")\t[("
//                              << A.row_sz << ", " << A.row_offset << ")(" << A.col_sz << ", " << A.col_offset
//                              << ")], A1r: " << A1r[iter] << "\n";
//                    ++iter;
                }
            }
        }

        // assert A2
//        std::cout << "\nA2: nnz: " << A2.nnz << std::endl;
        if(A2.nnz != 0){
            for (j = 0; j < A2.col_sz; ++j) {
                for (i = A2.col_scan[j] - 1; i < A2.col_scan[j + 1] - 1; ++i) {
//                    if(rank == verbose_rank) std::cout << std::setprecision(4) << A2.r[i] << "\t" << j+A2.col_offset+1 << "\t" << A2.v[i] << std::endl;
                    ASSERT(A2.r[i] > 0, "rank: " << rank << ", A2.r[i]: " << A2.r[i]);
                    ASSERT(A2.r[i] <= A2.row_sz, "rank: " << rank << ", A2.r[i]: " << A2.r[i] << ", A2.row_sz: " << A2.row_sz);
//                    assert(A.r[i + A1.col_scan[A.col_sz]] >= 0);
//                    assert(A.r[i + A1.col_scan[A.col_sz]] < A2.row_sz);
                }
            }
        }

        // ========================================================
        // this shows how to go through entries of A1 (top half) and A2 (bottom half) after changing order.
        // ========================================================
        if(rank == verbose_rank) {
//            print_array(A1.col_scan, A.col_sz+1, verbose_rank, "A1.col_scan", MPI_COMM_WORLD);
//            print_array(A2.col_scan, A.col_sz+1, verbose_rank, "A2.col_scan", MPI_COMM_WORLD);

//            nnz_t it = 0;
//            std::cout << "\nA1: nnz: " << A1.nnz << std::endl;
//            for (j = 0; j < A1.col_sz; j++) {
//                for (i = A1.col_scan[j] - 1; i < A1.col_scan[j + 1] - 1; i++) {
//                    std::cout << std::setprecision(4) << it++ << "\t" << A1.r[i] << "\t" << j+A1.col_offset+1 << "\t" << A1.v[i] << std::endl;
//                }
//            }
//            it = 0;
//            std::cout << "\nA2: nnz: " << A2.nnz << std::endl;
//            for (j = 0; j < A2.col_sz; j++) {
//                for (i = A2.col_scan[j] - 1; i < A2.col_scan[j + 1] - 1; i++) {
//                    std::cout << std::setprecision(4) << it++ << "\t"  << A2.r[i] << "\t" << j+A2.col_offset+1 << "\t" << A2.v[i] << std::endl;
//                }
//            }
        }
        // ========================================================
    }
#endif

}

void saena_object::reorder_back_split(CSCMat_mm &A, CSCMat_mm &A1, CSCMat_mm &A2){

#ifdef __DEBUG1__
//    int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    if(rank==0) std::cout << "\nstart of " << __func__ << std::endl;

    --reorder_counter;

//    nnz_t nnz1 = A1.col_scan[A.col_sz] - A1.col_scan[0];
//    nnz_t nnz2 = A2.col_scan[A.col_sz] - A2.col_scan[0];
//    assert(A.nnz == (nnz1 + nnz2));
    assert(A.nnz == (A1.nnz + A2.nnz));
#endif

    const int THRSHLD = A1.row_sz;

    // if A1.nnz==0 it means A2 was the whole A, so we only need to return row indices to their original values.
    if(A1.nnz == 0) {

#pragma omp simd
        for (int i = A2.col_scan[0] - 1; i < A2.col_scan[A2.col_sz] - 1; ++i) {
            A2.r[i] += THRSHLD;
        }

//        for (int j = 0; j < A2.col_sz; ++j) {
//            for (int i = A2.col_scan[j] - 1; i < A2.col_scan[j + 1] - 1; ++i) {
//                A2.r[i] += THRSHLD;
//            }
//        }

        return;
    }

    // ========================================================
    // IMPORTANT: An offset should be used to access Ar and Av.
    // ========================================================
    nnz_t offset = A1.col_scan[0] - 1;

    auto *Ar_temp = &mempool4[0];
    auto *Av_temp = &mempool5[0];

#ifdef SANEA_USE_PSTL
    std::copy(pstl::execution::par_unseq, &Ar_temp[0], &Ar_temp[0 + A.nnz], &A.r[offset]);
    std::copy(pstl::execution::par_unseq, &Av_temp[0], &Av_temp[0 + A.nnz], &A.v[offset]);
#else
    memcpy(&Ar_temp[0], &A.r[offset], sizeof(index_t) * A.nnz);
    memcpy(&Av_temp[0], &A.v[offset], sizeof(value_t) * A.nnz);
#endif

#ifdef __DEBUG1__
//    for(index_t i = offset; i < offset + nnz; i++){
//        Ar_temp_p[i] = Ar[i];
//        Av_temp_p[i] = Av[i];
//    }

//    print_array(A1.col_scan, A.col_sz+1, 0, "A1.col_scan", MPI_COMM_WORLD);
//    print_array(A2.col_scan, A.col_sz+1, 0, "A2.col_scan", MPI_COMM_WORLD);

#if 0
    // ========================================================
    // this shows how to go through entries of A1 (top half) and A2 (bottom half) after changing order.
    // NOTE: column is not correct. col_offset should be added to it.
    // ========================================================
    std::cout << "\nA1: nnz: " << A1.col_scan[A.col_sz] - A1.col_scan[0] << "\tcol is not correct." << std::endl ;
    for(index_t j = 0; j < A.col_sz; j++){
        for(index_t i = A1.col_scan[j]; i < A1.col_scan[j+1]; i++){
            std::cout << std::setprecision(4) << Ar[i] << "\t" << j << "\t" << Av[i] << std::endl;
//            std::cout << std::setprecision(4) << Ar[i] << "\t" << j << "\t" << Av[i] << "\ttemp: \t" << Ar_temp_p[i] << "\t" << j << "\t" << Av_temp_p[i] << std::endl;
        }
    }

    std::cout << "\nA2: nnz: " << A2.col_scan[A.col_sz] - A2.col_scan[0] << "\tcol is not correct." << std::endl ;
    for(index_t j = 0; j < A.col_sz; j++){
        for(index_t i = A2.col_scan[j]+A1.col_scan[A.col_sz]; i < A2.col_scan[j+1]+A1.col_scan[A.col_sz]; i++){
            std::cout << std::setprecision(4) << Ar[i] << "\t" << j << "\t" << Av[i] << std::endl;
//            std::cout << std::setprecision(4) << Ar[i] << "\t" << j << "\t" << Av[i] << "\ttemp: \t" << Ar_temp_p[i] << "\t" << j << "\t" << Av_temp_p[i] << std::endl;
        }
    }
#endif

    // ========================================================
#endif

    index_t *Ac = A1.col_scan; // Will add A2.col_scan to A1.col_scan for each column to have Ac.

    int i = 0;
    nnz_t nnz_col = 0, iter0 = offset, iter1 = 0, iter2 = A1.col_scan[A.col_sz] - offset - 1;
    for(int j = 0; j < A.col_sz; ++j){
        nnz_col = A1.col_scan[j+1] - A1.col_scan[j];
        if(nnz_col != 0){
#ifdef SANEA_USE_PSTL
            std::copy(pstl::execution::par_unseq, &A.r[iter0], &A.r[iter0 + nnz_col], &Ar_temp[iter1]);
            std::copy(pstl::execution::par_unseq, &A.v[iter0], &A.v[iter0 + nnz_col], &Av_temp[iter1]);
#else
            memcpy(&A.r[iter0], &Ar_temp[iter1], sizeof(index_t) * nnz_col);
            memcpy(&A.v[iter0], &Av_temp[iter1], sizeof(value_t) * nnz_col);
#endif
            iter1 += nnz_col;
            iter0 += nnz_col;
        }

        nnz_col = A2.col_scan[j+1] - A2.col_scan[j];
        if(nnz_col != 0){

            #pragma omp simd
            for(i = 0; i < nnz_col; ++i){
//                A.r[iter0 + i] = Ar_temp[iter2 + i];
                A.r[iter0 + i] = Ar_temp[iter2 + i] + THRSHLD;
//                A.v[iter0 + i] = Av_temp[iter2 + i];
//                std::cout << Ar_temp[iter2 + i] << "\t" << j+1 << "\t" << Av_temp[iter2 + i] << std::endl;
            }

//            memcpy(&Ar[iter0],  &Ar_temp[iter2], sizeof(index_t) * nnz_col);

#ifdef SANEA_USE_PSTL
            std::copy(pstl::execution::par_unseq, &A.v[iter0], &A.v[iter0 + nnz_col], &Av_temp[iter2]);
#else
            memcpy(&A.v[iter0], &Av_temp[iter2], sizeof(value_t) * nnz_col);
#endif
            iter2 += nnz_col;
            iter0 += nnz_col;
        }

        Ac[j] += A2.col_scan[j] - 1;
    }

    Ac[A.col_sz] += A2.col_scan[A2.col_sz] - 1;

#if 0
    // Equivalent to the previous part. Uses for loops instead of memcpy.
    index_t iter = offset;
    for(index_t j = 0; j < A.col_sz; j++){

        for(index_t i = A1.col_scan[j]; i < A1.col_scan[j+1]; i++) {
//            printf("%u \t%u \t%f\n", Ar_temp_p[i], j, Av_temp_p[i]);
            Ar[iter] = Ar_temp_p[i];
            Av[iter] = Av_temp_p[i];
            iter++;
        }

        for(index_t i = A2.col_scan[j]+A1.col_scan[A.col_sz]; i < A2.col_scan[j+1]+A1.col_scan[A.col_sz]; i++){
//            printf("%u \t%u \t%f\n", Ar_temp_p[i], j, Av_temp_p[i]);
            Ar[iter] = Ar_temp_p[i];
            Av[iter] = Av_temp_p[i];
            iter++;
        }

        Ac[j] += A2.col_scan[j];
    }

    Ac[A.col_sz] += A2.col_scan[A.col_sz];
#endif

#ifdef __DEBUG1__
//    print_array(Ac, A.col_sz+1, 0, "Ac", MPI_COMM_WORLD);
    assert(A.nnz == Ac[A.col_sz] - Ac[0]);

    // ========================================================
    // this shows how to go through entries of A before changing order.
    // NOTE: column is not correct. col_offset should be added to it.
    // ========================================================
//    std::cout << "\nA: nnz: " << A.nnz << std::endl ;
//    for(index_t j = 0; j < A.col_sz; ++j){
//        for(i = Ac[j] - 1; i < Ac[j+1] - 1; ++i){
//            std::cout << std::setprecision(4) << A.r[i] << "\t" << j+A.col_offset+1 << "\t" << A.v[i] << std::endl;
//        }
//    }

    // ========================================================
#endif
}


void saena_object::transpose_locally(cooEntry *A, nnz_t size){
    transpose_locally(A, size, 0, A);
}

void saena_object::transpose_locally(cooEntry *A, nnz_t size, cooEntry *B){
    transpose_locally(A, size, 0, B);
}

void saena_object::transpose_locally(cooEntry *A, nnz_t size, const index_t &row_offset, cooEntry *B){
    for(nnz_t i = 0; i < size; i++){
        B[i] = cooEntry(A[i].col, A[i].row + row_offset, A[i].val);
    }
    std::sort(&B[0], &B[size]);
}
