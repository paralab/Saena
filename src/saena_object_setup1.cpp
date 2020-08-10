#include "saena_object.h"
#include "saena_matrix.h"
#include "strength_matrix.h"
#include "prolong_matrix.h"
#include "grid.h"
#include "aux_functions.h"
#include "parUtils.h"


int saena_object::SA(Grid *grid){
    // Smoothed Aggregation (SA)
    // return value:
    // 0: normal
    // 1: this will be last level.
    // 2: stop the coarsening. the previous level will be set as the last level.

    // formula for the prolongation matrix from Irad Yavneh's paper:
    // P = (I - 4/(3*rhoDA) * DA) * P_t

    // todo: check when you should update new aggregate values: before creating prolongation or after.

    // Here P is computed: P = A_w * P_t; in which P_t is aggregate, and A_w = I - w*Q*A, Q is inverse of diagonal of A.
    // Here A_w is computed on the fly, while adding values to P. Diagonal entries of A_w are 0, so they are skipped.
    // todo: think about A_F which is A filtered.
    // todo: think about smoothing preconditioners other than damped jacobi. check the following paper:
    // todo: Eran Treister and Irad Yavneh, Non-Galerkin Multigrid based on Sparsified Smoothed Aggregation. page22.

    saena_matrix   *A = grid->A;
    prolong_matrix *P = &grid->P;

    MPI_Comm comm = A->comm;
    int nprocs = -1, rank = -1;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    P->comm = A->comm;
//    MPI_Comm_dup(A->comm, &P->comm);

#ifdef __DEBUG1__
    if(verbose_setup_steps) {
        MPI_Barrier(comm); if (!rank) printf("SA: start\n"); MPI_Barrier(comm);
    }
#endif

    nnz_t i = 0, j = 0;
    double omega = A->jacobi_omega; // todo: receive omega as user input. it is usually 2/3 for 2d and 6/7 for 3d.

    std::vector<index_t> aggregate(grid->A->M);
    int ret_val = find_aggregation(grid->A, aggregate, grid->P.splitNew);

#ifdef __DEBUG1__
//    print_vector(aggregate, -1, "aggregate", comm);
    if(verbose_setup_steps) {
        MPI_Barrier(comm); if (!rank) printf("SA: step 1\n"); MPI_Barrier(comm);
    }
#endif

    if(ret_val == 2){ // stop the coarsening
        return ret_val;
    }

    std::vector<index_t> vSendAgg(A->vIndexSize);
    std::vector<index_t> vecValuesAgg(A->recvSize);

    P->M    = A->M;
    P->Mbig = A->Mbig;
    P->Nbig = P->splitNew[nprocs]; // This is the number of aggregates, which is the number of columns of P.

    // store remote elements from aggregate in vSend to be sent to other processes.
    // todo: is it ok to use vSend instead of vSendAgg? vSend is double and vSendAgg is unsigned long.
    // todo: the same question for vecValues and Isend and Ireceive.
    for(i = 0; i < A->vIndexSize; i++){
        vSendAgg[i] = aggregate[A->vIndex[i]];
//        std::cout << A->vIndex[i] << "\t" << vSendAgg[i] << std::endl;
    }

#ifdef __DEBUG1__
    if(verbose_setup_steps) {
        MPI_Barrier(comm); if (!rank) printf("SA: step 2\n"); MPI_Barrier(comm);
    }
#endif

    int flag = 0;
    auto* requests = new MPI_Request[A->numSendProc + A->numRecvProc];
    auto* statuses = new MPI_Status[A->numSendProc + A->numRecvProc];

    for(i = 0; i < A->numRecvProc; ++i){
        MPI_Irecv(&vecValuesAgg[A->rdispls[A->recvProcRank[i]]], A->recvProcCount[i], par::Mpi_datatype<index_t>::value(), A->recvProcRank[i], 1, comm, &requests[i]);
        MPI_Test(&requests[i], &flag, &statuses[i]);
    }

    for(i = 0; i < A->numSendProc; ++i){
        MPI_Isend(&vSendAgg[A->vdispls[A->sendProcRank[i]]], A->sendProcCount[i], par::Mpi_datatype<index_t>::value(), A->sendProcRank[i], 1, comm, &requests[A->numRecvProc+i]);
        MPI_Test(&requests[A->numRecvProc + i], &flag, &statuses[A->numRecvProc + i]);
    }

    std::vector<cooEntry> PEntryTemp;

    // P = (I - 4/(3*rhoDA) * DA) * P_t
    // aggreagte is used as P_t in the following "for" loop.
    // local
    // -----

    // use these to avoid subtracting A->split[rank] from each case
    auto *aggregate_p = &aggregate[0] - A->split[rank];

    // go through A row-wise using indicesP_local (there is no order for the columns, only rows are ordered.)
    value_t vtmp = 0;
    const double ONE_M_OMEGA = 1 - omega;
    long iter = 0;
    for (i = 0; i < A->M; ++i) {
        const auto r_idx     = i + A->split[rank];      // row index
        const auto ninv_diag = -A->inv_diag[i];     // negative inverse of the diagonal element
        for (j = 0; j < A->nnzPerRow_local[i]; ++j, ++iter) {
            const auto idx = A->indicesP_local[iter];

            if(r_idx == A->col_local[idx]){ // diagonal element
                vtmp = ONE_M_OMEGA;
            }else{
                vtmp = omega * A->values_local[idx] * ninv_diag;
            }

            PEntryTemp.emplace_back(cooEntry(i, aggregate_p[A->col_local[idx]], vtmp));

//            if(rank==1) std::cout << i + A->split[rank] << "\t" << A->col_local[idx] << "\t" <<
//               aggregate_p[A->col_local[idx]] << "\t" << A->values_local[idx] * A->inv_diag[A->row_local[idx]] << "\n";
        }
    }

    MPI_Waitall(A->numRecvProc, requests, statuses);

#ifdef __DEBUG1__
    if(verbose_setup_steps) {
        MPI_Barrier(comm); if (!rank) printf("SA: step 3\n"); MPI_Barrier(comm);
    }
#endif

    // remote
    iter = 0;
    for (i = 0; i < A->col_remote_size; ++i) {
        const index_t c_idx = vecValuesAgg[A->col_remote[iter]];
        for (j = 0; j < A->nnzPerCol_remote[i]; ++j, ++iter) {
            PEntryTemp.emplace_back(cooEntry(A->row_remote[iter],
                                             c_idx,
                                             -omega * A->values_remote[iter] * A->inv_diag[A->row_remote[iter]]));
//            if(rank==3) std::cout << A->row_remote[iter] + A->split[rank] << "\t" << vecValuesAgg[A->col_remote[iter]] << "\t"
//                      << A->values_remote[iter] * A->inv_diag[A->row_remote[iter]] << "\t" << A->col_remote[iter] << std::endl;
        }
    }

#ifdef __DEBUG1__
//    printf("rank %d: PEntryTemp.size = %ld\n", rank, PEntryTemp.size());
//    print_vector(PEntryTemp, -1, "PEntryTemp", comm);
    if(verbose_setup_steps) {
        MPI_Barrier(comm); if (!rank) printf("SA: step 4\n"); MPI_Barrier(comm);
    }
#endif

    // add duplicates.
    // values of entries with the same row and col should be added together.

    std::sort(PEntryTemp.begin(), PEntryTemp.end());

    const int SZ_M1 = static_cast<int>(PEntryTemp.size()) - 1;
    cooEntry tmp(0, 0, 0.0);
    for(i = 0; i < PEntryTemp.size(); ++i){
        tmp = PEntryTemp[i];
        while(i < SZ_M1 && PEntryTemp[i] == PEntryTemp[i+1]){
            tmp.val += PEntryTemp[++i].val;
        }

        if(fabs(tmp.val) > ALMOST_ZERO){
            P->entry.emplace_back(tmp);
        }
    }

    P->nnz_l = P->entry.size();
    MPI_Allreduce(&P->nnz_l, &P->nnz_g, 1, par::Mpi_datatype<nnz_t>::value(), MPI_SUM, comm);
    P->split = A->split;
    P->findLocalRemote();

    MPI_Waitall(A->numSendProc, A->numRecvProc + requests, A->numRecvProc + statuses);
    delete [] requests;
    delete [] statuses;

#ifdef __DEBUG1__
//    print_vector(P->entry, -1, "P->entry", comm);
//    printf("rank %d: P->nnz_l = %ld, P->nnz_g = %ld\n", rank, P->nnz_l, P->nnz_g);
    if(verbose_setup_steps) {
        MPI_Barrier(comm); if (!rank) printf("SA: end\n"); MPI_Barrier(comm);
    }
#endif

    return ret_val;
}


int saena_object::find_aggregation(saena_matrix* A, std::vector<index_t>& aggregate, std::vector<index_t>& splitNew){
    // finding aggregation is written in an adaptive way. An aggregation_2_dist is being created first. If it is too small,
    // or too big it will be recreated until an aggregation_2_dist with size within the acceptable range is produced.

    // return value:
    // 0: normal
    // 1: this will be last level.
    // 2: stop the coarsening. the previous level will be set as the last level.

    MPI_Comm comm = A->comm;
    int rank;
    MPI_Comm_rank(comm, &rank);
//    MPI_Comm_size(A->comm, &nprocs);

#ifdef __DEBUG1__
    if(verbose_setup_steps) {
        MPI_Barrier(comm); if (!rank) printf("find_agg: start\n"); MPI_Barrier(comm);
    }
#endif

    strength_matrix S;
    create_strength_matrix(A, &S);

#ifdef __DEBUG1__
//    S.print_entry(-1);
    if(verbose_setup_steps) {
        MPI_Barrier(comm); if (!rank) printf("find_agg: step 1\n"); MPI_Barrier(comm);
    }
#endif

    std::vector<index_t> aggArray; // vector of root nodes.

    aggregation_1_dist(&S, aggregate, aggArray);
//    aggregation_2_dist(&S, aggregate, aggArray);

#ifdef __DEBUG1__
//    print_vector(aggArray, -1, "aggArray", comm);
    if(verbose_setup_steps) {
        MPI_Barrier(comm); if (!rank) printf("find_agg: step 2\n"); MPI_Barrier(comm);
    }
#endif

    double  division = 0;
    index_t new_size = 0;
    index_t new_size_local = aggArray.size(); // new_size is the size of the new coarse matrix.

    MPI_Allreduce(&new_size_local, &new_size, 1, par::Mpi_datatype<index_t>::value(), MPI_SUM, comm);
    division = (double) A->Mbig / new_size;

#ifdef __DEBUG1__
//    MPI_Barrier(comm);
//    if(verbose_setup_steps && rank==0)
//        printf("\nfind_aggregation: connStrength = %.2f, current size = %u, next level's size = %u, division = %.2f\n",
//               connStrength, A->Mbig, new_size, division);
//    MPI_Barrier(comm);
#endif

    if(adaptive_coarsening){

        float connStrength_temp = connStrength;
        bool continue_agg = false;

        if(division < 1.5 || division > 8) {
            continue_agg = true;
        }

        while(continue_agg){

            if( division > 8 ){

                connStrength_temp += 0.05;
                aggArray.clear();
                S.erase_update();
                S.setup_matrix(connStrength_temp);
//                create_strength_matrix(A, &S);

            } else if( division < 1.5 ){

                connStrength_temp -= 0.05;
                aggArray.clear();
                S.erase_update();
                S.setup_matrix(connStrength_temp);
//                create_strength_matrix(A, &S);

            }

            aggregation_1_dist(&S, aggregate, aggArray);
//            aggregation_2_dist(&S, aggregate, aggArray);

            // new_size is the size of the new coarse matrix.
            new_size_local = (index_t) aggArray.size(); // new_size is the size of the new coarse matrix.
            MPI_Allreduce(&new_size_local, &new_size, 1, par::Mpi_datatype<index_t>::value(), MPI_SUM, comm);
            division = (double)A->Mbig / new_size;

#ifdef __DEBUG1__
            if(verbose_setup_steps){
                MPI_Barrier(comm);
                if(!rank)
                    printf("\nfind_agg: adaptive coarsening: connStrength = %.2f \ncurrent size = %u \nnew size     = %u \ndivision     = %.2f\n",
                           connStrength_temp, A->Mbig, new_size, division);
                MPI_Barrier(comm);
            }
#endif

            if (connStrength_temp < 0.2 || connStrength_temp > 0.95){
                continue_agg = false;
            }

        }

    }

#ifdef __DEBUG1__
//    if(!rank) printf("\nfinal: connStrength = %f, current size = %u, new size = %u,  division = %d\n",
//                       connStrength_temp, A->Mbig, new_size, division);
    if(verbose_setup_steps) {
        MPI_Barrier(comm); if (!rank) printf("find_agg: step 3\n"); MPI_Barrier(comm);
    }
#endif

    // decide if next level for multigrid is required or not.
    // threshold to set maximum multigrid level
    int ret_val = 0;
    if(dynamic_levels){
//        MPI_Allreduce(&grids[i].Ac.M, &M_current, 1, MPI_UNSIGNED, MPI_MIN, grids[i].Ac.comm);
//        total_row_reduction = (float) grids[0].A->Mbig / grids[i].Ac.Mbig;
        float row_reduc_min = static_cast<float>(new_size) / A->Mbig;

//        if(grids[i].Ac.active){ MPI_Barrier(grids[i].Ac.comm); printf("row_reduction_min = %f, row_reduction_up_thrshld = %f, least_row_threshold = %u \n", grids[i+1].row_reduction_min, row_reduction_up_thrshld, least_row_threshold); MPI_Barrier(grids[i].Ac.comm);}
//        if(rank==0) if(row_reduction_min < 0.1) printf("\nWarning: Coarsening is too aggressive! Increase connStrength in saena_object.h\n");
//        row_reduction_local = (float) grids[i].Ac.M / grids[i].A->M;
//        MPI_Allreduce(&row_reduction_local, &row_reduction_min, 1, MPI_FLOAT, MPI_MIN, grids[i].Ac.comm);
//        if(rank==0) printf("row_reduction_min = %f, row_reduction_up_thrshld = %f \n", row_reduction_min, row_reduction_up_thrshld);

#ifdef __DEBUG1__
        if(verbose_setup_steps) {
            MPI_Barrier(comm);
            if (!rank) printf("find_agg: dynamic levels: next level's size / current size = %d / %d = %f\n",
                               new_size, A->Mbig, row_reduc_min);
            MPI_Barrier(comm);
        }
#endif

        if ( (new_size < least_row_threshold) ||
             (row_reduc_min > row_reduction_up_thrshld) ||
             (row_reduc_min < row_reduction_down_thrshld) ) {

            if(new_size < least_row_threshold) {
                ret_val = 1; // this will be the last level.
            }else{
                return 2; // stop the coarsening. the previous level will be set as the last level.
            }

        }
    }

#ifdef __DEBUG1__
    if(verbose_setup_steps) {
        MPI_Barrier(comm); if (!rank) printf("find_agg: step 4\n"); MPI_Barrier(comm);
    }
#endif

    aggregate_index_update(&S, aggregate, aggArray, splitNew);

    // destroy the strength matrix, since it is not needed anymore.
    S.destroy();

#ifdef __DEBUG1__
//    updateAggregation(aggregate, &aggSize);
//    print_vector(aggArray, -1, "aggArray", comm);
    if(verbose_setup_steps) {
        MPI_Barrier(comm); if (!rank) printf("find_agg: end\n"); MPI_Barrier(comm);
    }
#endif

    return ret_val;
} // end of SaenaObject::findAggregation


int saena_object::create_strength_matrix(saena_matrix* A, strength_matrix* S){

    // based on the following paper by Irad Yavneh:
    // Non-Galerkin Multigrid Based on Sparsified Smoothed Aggregation - page: A51

    // todo: add openmp to this function.

    MPI_Comm comm = A->comm;
    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
//    printf("create_strength_matrix: start: rank %d: A->M = %d \tA->nnz_l = %d \n", rank, A->M, A->nnz_l);
//    A->print_entry(-1);
#endif

    // ******************************** compute max per row ********************************

    std::vector<value_t> maxPerRow(A->M, 0);
    value_t *maxPerRow_p = &maxPerRow[0] - A->split[rank]; // use split to convert the index from global to local.

    for(nnz_t i = 0; i < A->nnz_l; ++i){
        if( A->entry[i].row != A->entry[i].col ){
            if(maxPerRow_p[A->entry[i].row] == 0 ||
              (maxPerRow_p[A->entry[i].row] < -A->entry[i].val))

                maxPerRow_p[A->entry[i].row] = -A->entry[i].val;
        }
    }

#ifdef __DEBUG1__
//    print_vector(maxPerRow, -1, "maxPerRow", comm);
#endif

    // ******************************** compute S ********************************

    S->entry.resize(A->nnz_l);
    value_t val_temp = 0;
    for(nnz_t i = 0; i < A->nnz_l; i++){
//        if(rank==1) std::cout << A->entry[i] << "\t maxPerRow = " << maxPerRow_p[A->entry[i].row] << std::endl;
        if(A->entry[i].row == A->entry[i].col) {
            val_temp = 1;
        } else {
#ifdef __DEBUG1__
    assert(maxPerRow_p[A->entry[i].row] != 0);
#endif
            val_temp = -A->entry[i].val / (maxPerRow_p[A->entry[i].row]);
        }

        S->entry[i] = cooEntry(A->entry[i].row, A->entry[i].col, val_temp);
    }

//    print_vector(S->entry, -1, "S->entry", comm);

    // ******************************** compute max per column - version 1 - for general matrices ********************************
/*
    std::vector<value_t> local_maxPerCol(A->Mbig, 0);

    for(nnz_t i=0; i<A->nnz_l; i++){
        if( A->entry[i].row != A->entry[i].col ){
            if( (local_maxPerCol[A->entry[i].col] == 0) || (local_maxPerCol[A->entry[i].col] < -A->entry[i].val) )
                local_maxPerCol[A->entry[i].col] = -A->entry[i].val;
        }
    }

    std::vector<value_t> maxPerCol(A->Mbig);
    MPI_Allreduce(&*local_maxPerCol.begin(), &*maxPerCol.begin(), A->Mbig, MPI_DOUBLE, MPI_MAX, comm);

//    print_vector(maxPerCol, -1, "maxPerCol", comm);

    // ******************************** compute ST - version 1 ********************************
    // row and columns indices are commented out, because only the value vector is required.

    nnz_t ST_size = 0;
    S->entryT.resize(A->nnz_l);
    for(nnz_t i = 0; i < A->nnz_l; i++){

        if(A->entry[i].row == A->entry[i].col) {
            val_temp = 1;
        }else{
            val_temp = -A->entry[i].val / maxPerCol[A->entry[i].col];
        }

        S->entryT[S_size] = cooEntry(A->entry[i].row - A->split[rank], A->entry[i].col - A->split[rank], val_temp);
        ST_size++;
    }

    S->entryT.resize(ST_size);
    S->entryT.shrink_to_fit();

//    if(rank==1)
//        for(i=0; i<STi.size(); i++)
//            std::cout << "ST: " << "[" << STi[i]+1 << "," << STj[i]+1 << "] = " << STval[i] << std::endl;
*/
    // ******************************** compute max per column - version 2 - if A is symmetric ********************************

    // TODO: assuming A is symmteric!
    // since A is symmetric, use maxPerRow for local entries on each process. receive the remote ones like matvec.

    //vSend are maxPerCol for remote elements that should be sent to other processes.
    for(nnz_t i = 0; i < A->vIndexSize; ++i)
        A->vSend[i] = maxPerRow[A->vIndex[i]];

    int flag = 0;
    auto *requests = new MPI_Request[A->numSendProc + A->numRecvProc];
    auto *statuses = new MPI_Status[A->numSendProc + A->numRecvProc];

    // vecValues are maxPerRow for remote elements that are received from other processes.
    for(nnz_t i = 0; i < A->numRecvProc; ++i){
        MPI_Irecv(&A->vecValues[A->rdispls[A->recvProcRank[i]]], A->recvProcCount[i], par::Mpi_datatype<value_t>::value(), A->recvProcRank[i], 1, comm, &requests[i]);
        MPI_Test(&requests[i], &flag, &statuses[i]);
    }

    for(nnz_t i = 0; i < A->numSendProc; ++i){
        MPI_Isend(&A->vSend[A->vdispls[A->sendProcRank[i]]], A->sendProcCount[i], par::Mpi_datatype<value_t>::value(), A->sendProcRank[i], 1, comm, &(requests[A->numRecvProc+i]));
        MPI_Test(&requests[A->numRecvProc + i], &flag, &statuses[A->numRecvProc + i]);
    }

    // ******************************** compute ST - version 2 ********************************

/*
    long iter2 = 0;
    for (i = 0; i < A->M; ++i, iter2++) {
        for (unsigned int j = 0; j < A->nnzPerRow_local[i]; ++j, ++iter) {

            // diagonal entry
            if(i == A->col_local[A->indicesP_local[iter]]){
                STi.emplace_back(iter2); // iter2 is actually i, but it was giving an error for using i.
                STj.emplace_back(A->col_local[A->indicesP_local[iter]]);
                STval.emplace_back(1);
                continue;
            }

            STi.emplace_back(iter2); // iter2 is actually i, but it was giving an error for using i.
            STj.emplace_back(A->col_local[A->indicesP_local[iter]]);
            STval.emplace_back( -A->values_local[A->indicesP_local[iter]] / maxPerRow[A->col_local[A->indicesP_local[iter]]] );
        }
    }
*/

    // TODO: iter2 is not being used later.
    // local ST values
    S->entryT.resize(A->nnz_l);
    for (nnz_t i = 0; i < A->nnz_l_local; ++i) {
        if(A->row_local[i] + A->split[rank] == A->col_local[i]) // diagonal entry
            val_temp = 1;
        else
            val_temp = -A->values_local[i] / maxPerRow_p[A->col_local[i]];

//        if(rank==3) printf("%u \t%u \t%f \n", A->row_local[i] + A->split[rank], A->col_local[i], val_temp);
        S->entryT[i] = cooEntry(A->row_local[i], A->col_local[i], val_temp);
    }

    MPI_Waitall(A->numRecvProc, requests, statuses);

/*
    iter = 0;
    for (i = 0; i < A->col_remote_size; ++i) {
        for (unsigned int j = 0; j < A->nnzPerCol_remote[i]; ++j, ++iter) {
            STi.emplace_back(A->row_remote[A->indicesP_remote[iter]]);
            STj.emplace_back(A->col_remote2[A->indicesP_remote[iter]]);
            STval.emplace_back( -A->values_remote[A->indicesP_remote[iter]] / A->vecValues[A->col_remote[A->indicesP_remote[iter]]] );
        }
    }
*/

    // remote ST values
    // todo: add OpenMP just like matvec.
    // TODO: double check values for entryT.
    nnz_t iter = 0;
    double valtmp = 0;
    for (index_t i = 0; i < A->col_remote_size; ++i) {
        valtmp = -1 / A->vecValues[i];
        for (index_t j = 0; j < A->nnzPerCol_remote[i]; ++j, ++iter) {
//            if(rank==1) printf("%u \t%u \t%f \n", A->row_remote[iter], A->col_remote2[iter], -A->values_remote[iter] / A->vecValues[i]);
            S->entryT[iter + A->nnz_l_local] = cooEntry(A->row_remote[iter], A->col_remote2[iter], A->values_remote[iter] * valtmp);
        }
    }

    std::sort(S->entryT.begin(), S->entryT.end());
//    print_vector(S->entryT, -1, "S->entryT", comm);

    // ******************************** setup the matrix S ********************************

    // S indices are local on each process, which means it starts from 0 on each process.
    S->set_parameters(A->M, A->Mbig, A->split, comm);
    S->setup_matrix(connStrength);

    MPI_Waitall(A->numSendProc, A->numRecvProc+requests, A->numRecvProc+statuses);
    delete [] requests;
    delete [] statuses;

//    MPI_Barrier(comm); printf("rank %d: end of saena_object::create_strength_matrix \n", rank); MPI_Barrier(comm);
    return 0;
} // end of saena_object::create_strength_matrix


// Using MIS(1) from the following paper by Luke Olson:
// EXPOSING FINE-GRAINED PARALLELISM IN ALGEBRAIC MULTIGRID METHODS
int saena_object::aggregation_1_dist(strength_matrix *S, std::vector<index_t> &aggregate,
                                     std::vector<index_t> &aggArray) {
    // aggregate: of size dof. at the end it will show to what root node (aggregate) each node is assigned.
    // aggArray: the root nodes of the coarse matrix.

    // For each node, first assign it to a 1-distance root.
    // If there is not any root in distance-1, that node will become a root.

    MPI_Comm comm = S->comm;
    int nprocs = -1, rank = -1;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    if(verbose_setup_steps){
        MPI_Barrier(comm); if(!rank) printf("aggregation_1_dist: start\n"); MPI_Barrier(comm);
    }
//    S->print_entry(-1);
#endif

    index_t size = S->M;
    std::vector<index_t> aggregate2(size);

    auto *decided = new bool[size];
    assert(decided != nullptr);
    fill(&decided[0], &decided[size], false);

    auto *dec_nei = new bool[size]; // decided_neighbor
    assert(dec_nei != nullptr);

    auto *is_root = new bool[size];
    assert(is_root != nullptr);
    fill(&is_root[0], &is_root[size], false);

    auto *is_root_nei = new bool[size];
    assert(is_root_nei != nullptr);

    auto *vSend = new bool[2 * S->vIndexSize];
    assert(vSend != nullptr);

    auto *vecValues = new bool[2 * S->recvSize];
    assert(vecValues != nullptr);

    bool continueAggLocal = true, continueAgg = true;
    index_t i = 0, j = 0, i_remote = 0, j_remote = 0, it = 0, col_idx = 0;

    auto *requests = new MPI_Request[S->numSendProc + S->numRecvProc];
    auto *statuses = new MPI_Status[S->numSendProc + S->numRecvProc];

    // initialization -> this part is merged to the first "for" loop in the following "while".
    for(i = 0; i < size; ++i) {
        aggregate[i] = i + S->split[rank];
    }

#ifdef __DEBUG1__
    {
//    int whileiter = 0;
//    S->print_entry(-1);
//    S->print_diagonal_block(-1);
//    S->print_off_diagonal(-1);
//    print_vector(S->split, 0, "split", comm);
//    print_vector(S->nnzPerCol_remote, 0, "nnzPerCol_remote", comm);
    }
#endif

    while(continueAgg) {
        // local part
        // ==========
        // aggregate2 is used instead of aggregate, because the assigning a node to an aggregate happens at the end of
        // the while loop.
        // local entries are sorted in row-major order, but columns are not ordered.

        // use these to avoid subtracting S->split[rank] from each case
        auto *aggregate_p = &aggregate[0] - S->split[rank];
        auto *is_root_p   = &is_root[0] - S->split[rank];
        auto *decided_p   = &decided[0] - S->split[rank];

        it = 0;
        for (i = 0; i < size; ++i) {
            if(!decided[i]) {
                aggregate2[i]  = aggregate[i];
                dec_nei[i]     = true;
                is_root_nei[i] = false;
                for (j = 0; j < S->nnzPerRow_local[i]; ++j, ++it) {
//                    col_idx = S->col_local[it] - S->split[rank];
                    col_idx = S->col_local[it];
#ifdef __DEBUG1__
//                    if(rank==1) printf("%d:\taggregate[i] = %3d,\taggregate2[i] = %3d,\taggregate[col_idx] = %3d,"
//                                       "\tdecided[col_idx] = %d,\tis_root[col_idx] = %d,\tcol_idx = %d \n",
//                                       i, aggregate[i], aggregate2[i], aggregate_p[col_idx], decided_p[col_idx], is_root_p[col_idx], col_idx);
#endif

                        if( (aggregate_p[col_idx] < aggregate2[i]) &&
                        (!decided_p[col_idx] || is_root_p[col_idx])){ // equivalent: if(dec_nei && !is_root) skip;

#ifdef __DEBUG1__
//                        if(rank==1) printf("%d:\taggregate[i] = %d,\taggregate2[i] = %d,\tdec_nei[i] = %d,\tis_root_nei[i] = %d,\tcol_idx = %d \n",
//                                i, aggregate[i], aggregate2[i], dec_nei[i], is_root_nei[i], col_idx);
#endif

                            aggregate2[i]  = aggregate_p[col_idx];
                            dec_nei[i]     = decided_p[col_idx];
                            is_root_nei[i] = is_root_p[col_idx];
                        }
                }
#ifdef __DEBUG1__
//                if(rank==1) printf("%d:\taggregate[i] = %d,\taggregate2[i] = %d,\tdec_nei[i] = %d,\tis_root_nei[i] = %d\n",
//                        i, aggregate[i], aggregate2[i], dec_nei[i], is_root_nei[i]);
#endif

            }else{
                it += S->nnzPerRow_local[i];
            }
        }

        // remote part
        // ===========
        if(nprocs > 1){
            for (i = 0; i < S->vIndexSize; ++i){
                vSend[2 * i]     = decided[S->vIndex[i]];
                vSend[2 * i + 1] = is_root[S->vIndex[i]];
            }

            for (i = 0; i < S->numRecvProc; ++i)
                MPI_Irecv(&vecValues[S->rdispls[S->recvProcRank[i]]], S->recvProcCount[i], MPI_CXX_BOOL,
                          S->recvProcRank[i], 1, comm, &requests[i]);

            for (i = 0; i < S->numSendProc; ++i)
                MPI_Isend(&vSend[S->vdispls[S->sendProcRank[i]]], S->sendProcCount[i], MPI_CXX_BOOL,
                          S->sendProcRank[i], 1, comm, &requests[S->numRecvProc + i]);

            MPI_Waitall(S->numRecvProc, requests, statuses);

            it = 0;
            for (i = 0; i < S->col_remote_size; ++i) {
                for (j = 0; j < S->nnzPerCol_remote[i]; ++j, ++it) {
                    i_remote = S->row_remote[it];
                    j_remote = S->col_remote[it];
//                    if(rank==1) printf("%d:\trow_remote = %4d,\tcol_remote2 = %4d,\tvecValues[2*it] = %4d,\tvecValues[2*it+1] = %4d\n",
//                                       it, S->row_remote[it], S->col_remote2[it], vecValues[2 * j_remote], vecValues[2 * j_remote + 1]);

                    if (!decided[i_remote] &&
                       (S->col_remote2[it] < aggregate2[i_remote]) &&
                       (!vecValues[2 * j_remote] || vecValues[2 * j_remote + 1]) ){ // vecValues[2 * it]:     decided
                                                                                    // vecValues[2 * it + 1]: is_root
                                                                             // equivalent: if(dec_nei && !is_root) skip
                        aggregate2[i_remote]  = S->col_remote2[it];
                        dec_nei[i_remote]     = vecValues[2 * j_remote];
                        is_root_nei[i_remote] = vecValues[2 * j_remote + 1];
                    }
                }
            }

            MPI_Waitall(S->numSendProc, S->numRecvProc+requests, S->numRecvProc+statuses);
        }

        // put weight2 in weight and aggregate2 in aggregate.
        // if a row does not have a remote element then (weight2[i]&weightMax) == (weight[i]&weightMax)
        // update aggStatus of remote elements at the same time
        for(i = 0; i < size; ++i){
//            if(rank==1) printf("%d:\taggregate[i] = %d,\taggregate2[i] = %d,\tdec_nei[i] = %d,\tis_root_nei[i] = %d\n",
//                    i, aggregate[i], aggregate2[i], dec_nei[i], is_root_nei[i]);

            if(!decided[i] && dec_nei[i]){
                decided[i] = true;
                if(aggregate[i] == aggregate2[i]){
                    is_root[i] = true;
                    aggArray.emplace_back(aggregate[i]);
                }else if(is_root_nei[i]){
                    aggregate[i] = aggregate2[i];
                }
            }
        }

        continueAggLocal = false;
        for (i = 0; i < size; ++i) {
            // if any un-assigned node is available, continue aggregating.
            if(!decided[i]) {
                continueAggLocal = true;
                break;
            }
        }

        // check if every processor does not have any non-assigned node, otherwise all the processors should continue aggregating.
        if(nprocs > 1){
            MPI_Allreduce(&continueAggLocal, &continueAgg, 1, MPI_CXX_BOOL, MPI_LOR, comm);
        }else{
            continueAgg = continueAggLocal;
        }

#ifdef __DEBUG1__
        {
//            ++whileiter;
//            if(rank==0) printf("\nwhileiter = %d\n", whileiter);
//            print_vector(aggregate, -1, "aggregate", comm);
//            print_vector(aggArray, -1, "aggArray", comm);
        }
#endif
    } //while(continueAgg)

    delete [] requests;
    delete [] statuses;
    delete [] decided;
    delete [] dec_nei;
    delete [] is_root;
    delete [] is_root_nei;
    delete [] vSend;
    delete [] vecValues;

#ifdef __DEBUG1__
    {
//    for(i = 0; i < size; ++i)
//        if(rank==0) std::cout << "V[" << i+S->split[rank] << "] = " << initialWeight[i] << ";" << std::endl;

        // *************************** update aggregate to new indices ****************************
//    if(rank==0)
//        std::cout << std::endl << "S.M = " << S->M << ", S.nnz_l = " << S->nnz_l << ", S.nnz_l_local = " << S->nnz_l_local
//             << ", S.nnz_l_remote = " << S->nnz_l_remote << std::endl << std::endl;

//    if(rank==1){
//        std::cout << "aggregate:" << std::endl;
//        for(i=0; i<size; i++)
//            std::cout << i+S->split[rank] << "\t" << aggregate[i] << std::endl;
//        std::cout << std::endl;}
    }
#endif

    // aggArray is the set of root nodes.
    if(!aggArray.empty())
        std::sort(aggArray.begin(), aggArray.end());

#ifdef __DEBUG1__
    {
//        print_vector(aggArray, -1, "aggArray", comm);
//        print_vector(aggregate, -1, "aggregate", comm);
        if(verbose_setup_steps) {
            MPI_Barrier(comm); if(!rank) printf("aggregation_1_dist: end\n"); MPI_Barrier(comm);
        }

    // ************* write the aggregate nodes to a file *************

    // use this command to concatenate the output files:
    // cat aggArraySaena0.txt aggArraySaena1.txt > aggArraySaena.txt
//    unsigned long aggArray_size, aggArray_size_total;
//    aggArray_size = aggArray.size();
//    MPI_Allreduce(&aggArray_size, &aggArray_size_total, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
//    if(rank==1) std::cout << "total aggArray size = " << aggArray_size_total << std::endl << std::endl;
//    for(i=0; i<aggArray.size(); i++)
//        aggArray[i]++;
//    writeVectorToFileul(aggArray, aggArray_size_total, "aggArraySaena", comm);
//    for(i=0; i<aggArray.size(); i++)
//        aggArray[i]--;
    }
#endif

    return 0;
}


// Using MIS(2) from the following paper by Luke Olson:
// EXPOSING FINE-GRAINED PARALLELISM IN ALGEBRAIC MULTIGRID METHODS
// This function needs to be updated.
#if 0
int saena_object::aggregation_2_dist(strength_matrix *S, std::vector<unsigned long> &aggregate,
                                     std::vector<unsigned long> &aggArray) {

    // For each node, first assign it to a 1-distance root. If there is not any root in distance-1, find a distance-2 root.
    // If there is not any root in distance-2, that node should become a root.

    // aggregate: of size dof at the end will shows to what root node (aggregate) each node is assigned.
    // aggArray: the root nodes of the coarse matrix.

    // variables used in this function:
    // weight[i]: the two most left bits store the status of node i, the other 62 bits store weight assigned to that node.
    //            status of a node: 1 for 01 not assigned, 0 for 00 assigned, 2 for 10 root
    //            the max value for weight is 2^63 - 1
    //            weight is first generated randomly by randomVector function and saved in initialWeight. During the
    //            aggregation_2_dist process, it becomes the weight of the node's aggregate.

    MPI_Comm comm = S->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    unsigned long i, j;
    unsigned long size = S->M;

    std::vector<unsigned long> aggregate2(size);
    std::vector<unsigned long> weight(size);
    std::vector<unsigned long> weight2(size);
    std::vector<unsigned long> initialWeight(size);
//    std::vector<unsigned long> aggStatus2(size); // 1 for 01 not assigned, 0 for 00 assigned, 2 for 10 root

    S->set_weight(initialWeight);
//    S->randomVector(initialWeight, S->Mbig, comm);
//    S->randomVector4(initialWeight, S->Mbig);

//    print_vector(initialWeight, -1, "initialWeight", comm);

    const int wOffset = 62;
    // 1UL<<wOffset: the status of the node is set to 1 which is "not assigned".
    // 2UL<<wOffset: the status of the node is set to 2 which is "root".

    const unsigned long weightMax = (1UL<<wOffset) - 1;
    const unsigned long UNDECIDED = 1UL<<wOffset;
    const unsigned long ROOT =      1UL<<(wOffset+1);
//    const unsigned long UNDECIDED_OR_ROOT = 3UL<<wOffset;

    unsigned long weightTemp, aggregateTemp, aggStatusTemp;
    int* root_distance = (int*)malloc(sizeof(int)*size);
    // root_distance is initialized to 3(11). root = 0 (00), 1-distance root = 1 (01), 2-distance root = 2 (10).
    bool* dist1or2undecided = (bool*)malloc(sizeof(bool)*size);
    // if there is a distance-2 neighbor which is undecided, set this to true.
    bool continueAggLocal;
    bool continueAgg = true;
    unsigned long i_remote, j_remote, weight_neighbor, agg_neighbor, iter, col_index;
    int whileiter = 0;

    MPI_Request *requests = new MPI_Request[S->numSendProc + S->numRecvProc];
    MPI_Status *statuses  = new MPI_Status[S->numSendProc + S->numRecvProc];

//    if(rank==0) std::cout << "set boundary points: " << std::endl;

    // initialization -> this part is merged to the first "for" loop in the following "while".
    for(i=0; i<size; i++) {
        aggregate[i] = i + S->split[rank];
//        aggStatus2[i] = 1;
        // Boundary nodes are the ones that only have one neighbor which is itself (so one nnzPerRow),
        // and it is the diagonal element. They are roots for every coarse-grid.
//        if(rank==0) std::cout << i << "\t" << S->nnzPerRow[i] << std::endl;
        if(S->nnzPerRow[i] == 1){
            weight[i] = ( 2UL<<wOffset | initialWeight[i] );
            root_distance[i] = 0;
            aggArray.emplace_back(aggregate[i]);
//            if(rank==0) std::cout << "boundary: " << i+S->split[rank] << std::endl;
        }else{
            weight[i] = ( 1UL<<wOffset | initialWeight[i] ); // status of each node is initialized to 1 and its weight to initialWeight.
//            if(rank==0) std::cout << "V[" << i+S->split[rank] << "] = " << initialWeight[i] << ";" << std::endl;
        }
    }

//    S->print_entry(-1);
//    S->print_diagonal_block(-1);
//    S->print_off_diagonal(-1);
//    print_vector(weight, -1, "weight", comm);
//    print_vector(S->split, 0, "split", comm);
//    print_vector(S->nnzPerCol_remote, 0, "nnzPerCol_remote", comm);

    while(continueAgg) {
        // ******************************* first round of max computation *******************************
        // first "compute max" is local. The second one is both local and remote.
        // for loop is of size "number of rows". it checks if a node is UNDECIDED. Then, it goes over its neighbors,
        // which are nonzeros on that row. If the neighbor is UNDECIDED or ROOT, and its weight is higher than
        // weightTemp, then that node will be chosen for root for now.
        // UNDECIDED is also considered because it may become a root later, and it may be a better root for that node.
        // In the next "for" loop, weight and aggregate are updated, but not the status.

        // local part - distance-1 aggregate
        // aggregateTemp and weightTemp are used instead of aggregate and weight, because the complete set of
        // parameters for the nodes will be set at the end of the while loop, so we don't want to use the updated
        // weight and aggregate of a node for other nodes. In other words, if weight and aggregate for node i is set
        // here and we are checking neighbors of another node and i is one of them, we don't want to use the new
        // weight and aggregate for node i, since they are not finalized yet.
        iter = 0;
        for (i = 0; i < size; ++i) {
            if(weight[i]&UNDECIDED) {
                root_distance[i] = 3; // initialization
                dist1or2undecided[i] = false; // initialization
                aggregateTemp = aggregate[i];
                weightTemp = weight[i]&weightMax;
//                weight2[i] = weight[i]&weightMax;
//                aggStatusTemp = 1UL; // this will be used for aggStatus2, and aggStatus2 will be used for the remote part.
                for (j = 0; j < S->nnzPerRow_local[i]; ++j, ++iter) {
                    col_index = S->col_local[S->indicesP_local[iter]] - S->split[rank];
//                    if(rank==0 && i==3) printf("col_index = %lu, status = %lu, aggregate = %u \n", col_index, weight[col_index]>>wOffset, S->col_local[S->indicesP_local[iter]]);
                    if(weight[col_index] & ROOT){
                        weight[i] = (0UL << wOffset | (weight[col_index] & weightMax)); // 0UL << wOffset is not required.
                        weightTemp = weight[i];
                        aggregate[i] = S->col_local[S->indicesP_local[iter]];
                        aggregateTemp = aggregate[i];
                        root_distance[i] = 1;
                        dist1or2undecided[i] = false;

//                        if(rank==0) std::cout << i+S->split[rank] << "\t assigned to = " << aggregate[i] << " distance-1 local \t weight = " << weight[i] << std::endl;
//                        break; todo: try to add this break. You can define nnzPerRowScan_local.
                    } else if( (weight[col_index] & UNDECIDED) ){
                        // if there is an UNDECIDED neighbor, and its weight is bigger than this node, this node should
                        // wait until the next round of the "while" loop to see if this neighbor becomes root or assigned.
//                        if(rank==0) std::cout << i+S->split[rank] << "\t neighbor = " << S->col_local[S->indicesP_local[iter]] << std::endl;

                        if( (initialWeight[col_index] > weightTemp) ||
                            ((initialWeight[col_index] == weightTemp) && (aggregate[col_index] > i+S->split[rank])) ){
//                            if(rank==0 && i==10) std::cout << col_index << "\tweight = " << (weight[col_index]&weightMax) << "\tagg = " << aggregate[col_index] << std::endl;
                            weightTemp = (weight[col_index] & weightMax);
                            aggregateTemp = S->col_local[S->indicesP_local[iter]];
                            root_distance[i] = 1;
                            dist1or2undecided[i] = true;
                        }
                    }
                }
                weight2[i]    = weightTemp;
                aggregate2[i] = aggregateTemp;
//                if(rank==1) std::cout << i+S->split[rank] << "," << aggregate2[i] << "\t\t";
            }else{
                iter += S->nnzPerRow_local[i];
            }
        }

        // todo: for distance-1 it is probably safe to remove this for loop, and change weight2 to weight and aggregate2 to aggregate at the end of the previous for loop.
        for (i = 0; i < size; ++i) {
            if( (S->nnzPerRow_local[i]!=0) && (weight[i]&UNDECIDED) && (root_distance[i]==1)) {
                weight[i] = (1UL << wOffset | weight2[i] & weightMax );
                aggregate[i] = aggregate2[i];
//                if(rank==0) std::cout << i+S->split[rank] << "\t" << aggregate[i] << "\t" << aggregate2[i] << std::endl;
            }
        }

//        if(rank==0 && weight[1]&UNDECIDED) printf("node1: local 1: weight = %lu, aggregate = %lu \n", weight2[1]&weightMax, aggregate[1]);
//        if(rank==0 && weight[3]&UNDECIDED) printf("node3: local 1: weight = %lu, aggregate = %lu \n", weight2[3]&weightMax, aggregate[3]);
//        if(rank==0 && weight[7]&UNDECIDED) printf("node7: local 1: weight = %lu, aggregate = %lu \n", weight2[7]&weightMax, aggregate[7]);

        //    if(rank==0){
        //        std::cout << std::endl << "after first max computation!" << std::endl;
        //        for (i = 0; i < size; ++i)
        //            std::cout << i << "\tweight = " << weight[i] << "\tindex = " << aggregate[i] << std::endl;
        //    }





/*
        // remote part - distance-1 aggregate
        iter = 0;
        for (i = 0; i < S->col_remote_size; ++i) {
            for (j = 0; j < S->nnzPerCol_remote[i]; ++j, ++iter) {
                i_remote = S->row_remote[iter];
                j_remote = S->col_remote[iter];
                weight_neighbor = S->vecValues[2 * j_remote];
                agg_neighbor    = S->vecValues[2 * j_remote + 1];

                if (weight[i_remote] & UNDECIDED) {

                    //distance-1 aggregate
                    if (weight_neighbor & ROOT) {
                        weight[i_remote] = (0UL << wOffset | (weight_neighbor & weightMax));
//                        weight2[i_remote] = weight[i_remote];
                        aggregate[i_remote] = (agg_neighbor & weightMax);
                        root_distance[i_remote] = 1;
                        dist1or2undecided[i_remote] = false;
//                        if(rank==0) std::cout << i+S->split[rank] << "\t assigned to = " << aggregate[i_remote] << " distance-1 remote \t weight = " << weight[i_remote] << std::endl;
                    } else if (weight_neighbor & UNDECIDED) {

                        if (root_distance[i_remote] == 1)
                            weightTemp = (weight2[i_remote] & weightMax);
                        else{
                            weightTemp = initialWeight[i_remote];
                        }

                        if ( ( (weight_neighbor & weightMax) > weightTemp) ||
                             ( ( (weight_neighbor & weightMax) == weightTemp) && ( ((agg_neighbor & weightMax) > i_remote+S->split[rank]) )) ) {
//                            if(rank==1) std::cout << "first  before\t" << "row = " << S->row_remote[iter]+S->split[rank] << "  \tinitial weight = " << initialWeight[S->row_remote[iter]] << ",\taggregate = " << aggregate2[S->row_remote[iter]] << "\tremote weight = " << (weight2[S->row_remote[iter]]&weightMax) << "\tremote status = " << (S->vecValues[2*S->col_remote[iter]+1]>>wOffset) << "\taggStat2 = " << aggStatus2[S->row_remote[iter]] << std::endl;
                            weight2[i_remote] = (weight_neighbor & weightMax);
                            aggregate2[i_remote] = (agg_neighbor & weightMax);
                            root_distance[i_remote] = 1;
                            dist1or2undecided[i_remote] = true;
//                            aggStatus2[S->row_remote[iter]] = (S->vecValues[2 * S->col_remote[iter]] >> wOffset);
//                            if(rank==1) std::cout << "first  after \t" << "row = " << S->row_remote[iter]+S->split[rank] << "  \tinitial weight = " << initialWeight[S->row_remote[iter]] << ",\taggregate = " << aggregate2[S->row_remote[iter]] << "\t\tremote weight = " << (weight2[S->row_remote[iter]]&weightMax) << "\tremote status = " << (S->vecValues[2*S->col_remote[iter]+1]>>wOffset) << "\taggStat2 = " << aggStatus2[S->row_remote[iter]] << std::endl;
                        }
                    }

                    // if the node is assigned and root_distance is 2, check to see if there is a root in distance-1
                } else if( (weight[i_remote]>>wOffset == 0) && (root_distance[i_remote] == 2) ){
                    if (weight_neighbor & ROOT){
                        weight[i_remote] = (0UL << wOffset | (weight_neighbor & weightMax));
                        aggregate[i_remote] = (agg_neighbor & weightMax);
                        root_distance[i_remote] = 1;
                    }
                }
            }
        }

        MPI_Waitall(S->numSendProc, S->numRecvProc+requests, S->numRecvProc+statuses);

        // put weight2 in weight and aggregate2 in aggregate.
        // if a row does not have a remote element then (weight2[i]&weightMax) == (weight[i]&weightMax)
        // update aggStatus of remote elements at the same time
        for(i=0; i<size; i++){
            if( (weight[i]&UNDECIDED) && aggregate[i] != aggregate2[i] ){
                aggregate[i] = aggregate2[i];
                weight[i] = ( 1UL<<wOffset | weight2[i] );
//                if(aggStatus2[i] != 1) // if this is 1, it should go to the next aggregation_2_dist round.
//                    weight[i] = (0UL<<wOffset | weight2[i]&weightMax);
            }
        }
*/




        // ******************************* exchange remote max values for the second round of max computation *******************************

        // vSend[2*i]:   the first right 62 bits of vSend is maxPerCol for remote elements that should be sent to other processes.
        //               the first left 2 bits of vSend are aggStatus.
        // vSend[2*i+1]: the first right 62 bits of vSend is aggregate for remote elements that should be sent to other processes.
        //               the first left 2 bits are root_distance.
        //               root_distance is initialized to 3(11). root = 0 (00), 1-distance root = 1 (01), 2-distance root = 2 (10).

        // the following shows how the data is being stored in vecValues:
//        iter = 0;
//        if(rank==1)
//            for (i = 0; i < S->col_remote_size; ++i)
//                for (j = 0; j < S->nnzPerCol_remote[i]; ++j, ++iter){
//                    std::cout << "row:" << S->row_remote[iter]+S->split[rank] << "\tneighbor(col) = " << S->col_remote2[iter]
//                         << "\t weight of neighbor = "        << (S->vecValues[2*S->col_remote[iter]]&weightMax)
//                         << "\t\t status of neighbor = "      << (S->vecValues[2*S->col_remote[iter]]>>wOffset)
//                         << "\t root_distance of neighbor = " << (S->vecValues[2*S->col_remote[iter]+1]&weightMax)
//                         << "\t status of agg = "             << (S->vecValues[2*S->col_remote[iter]+1]>>wOffset)
//                         << std::endl;
//                }

        for (i = 0; i < S->vIndexSize; i++){
            S->vSend[2*i] = weight[S->vIndex[i]];
            aggStatusTemp = (unsigned long)root_distance[S->vIndex[i]]; // this is root_distance of the neighbor's aggregate.
            S->vSend[2*i+1] = ( (aggStatusTemp<<wOffset) | (aggregate[S->vIndex[i]]&weightMax) );
//            if(rank==1) std::cout << "vsend: " << S->vIndex[i]+S->split[rank] << "\tw = " << (S->vSend[2*i]&weightMax) << "\tst = " << (S->vSend[2*i]>>wOffset) << "\tagg = " << (S->vSend[2*i+1]&weightMax) << "\t oneDis = " << (S->vSend[2*i+1]>>wOffset) << std::endl;
        }

        // not using adaptive Mpitype for the following commands, since they should be unsigned long.

        for (i = 0; i < S->numRecvProc; i++)
            MPI_Irecv(&S->vecValues[S->rdispls[S->recvProcRank[i]]], S->recvProcCount[i], MPI_UNSIGNED_LONG,
                      S->recvProcRank[i], 1, comm, &(requests[i]));

        for (i = 0; i < S->numSendProc; i++)
            MPI_Isend(&S->vSend[S->vdispls[S->sendProcRank[i]]], S->sendProcCount[i], MPI_UNSIGNED_LONG,
                      S->sendProcRank[i], 1, comm, &(requests[S->numRecvProc + i]));

        // ******************************* second round of max computation *******************************
        // "for" loop is of size "number of rows". It checks if a node is UNDECIDED and also if it does not have a
        // root of distance one. Roots of distance 1 have priority. Then, it goes over its neighbors, which are
        // nonzeros on that row. The neighbor should be UNDECIDED or assigned. Then, it should have a 1-distance root,
        // because we are looking for a distance-2 root. Finally, its weight should be higher than weightTemp,
        // then than node will be chosen for root. UNDECIDED is also considered because it may find a 1-distance root
        // later. In the next "for" loop, weight and aggregate are updated, but not the status.

        // local part - distance-2 aggregate
        iter = 0;
        for (i = 0; i < size; ++i) {
            if( (weight[i]&UNDECIDED) && root_distance[i]!=1) { // root_distance cannot be 2 or 0 here.
                aggregateTemp = aggregate[i];
                weightTemp    = weight[i];
//                aggStatusTemp = 1UL; // this will be used for aggStatus2, and aggStatus2 will be used for the remote part.
                for (j = 0; j < S->nnzPerRow_local[i]; ++j, ++iter) {
                    col_index = S->col_local[S->indicesP_local[iter]] - S->split[rank];
                    if( root_distance[col_index]==1 ){
                        if( ((weight[col_index]&weightMax) > (weightTemp&weightMax)) ||
                            ( ((weight[col_index]&weightMax) == (weightTemp&weightMax)) && (aggregate[col_index] > aggregateTemp) ) ){

//                            if(rank==1 && i==2) std::cout << col_index << "\tweight = " << (weight[col_index]&weightMax) << "\tagg = " << aggregate[col_index] << std::endl;
                            aggregateTemp = aggregate[col_index];
                            weightTemp    = weight[col_index];
                            root_distance[i] = 2;
                            if( weight[col_index]>>wOffset == 0) // assigned neighbor
                                dist1or2undecided[i] = false;
                            else // UNDECIDED neighbor. It won't be ROOT because of root_distance[col_index]==1 condition.
                                dist1or2undecided[i] = true;

                        }
                    }
                }
                weight2[i]    = weightTemp;
                aggregate2[i] = aggregateTemp;
//                aggStatus2[i] = aggStatusTemp; // this is stored only to be compared with the remote one in the remote part.
            }else
                iter += S->nnzPerRow_local[i];
        }

        for (i = 0; i < size; ++i) {
            if( (S->nnzPerRow_local[i]!=0) && (weight[i]&UNDECIDED) && (root_distance[i]==2) ) {
                aggregate[i] = aggregate2[i];
                aggStatusTemp = (weight[i]>>wOffset);
                weight[i] = (aggStatusTemp<<wOffset | (weight2[i]&weightMax) );
//                if (aggregate[i] < S->split[rank] || aggregate[i] >= S->split[rank+1]) // this is distance-2 to a remote root.
//                    aggStatusTemp = 0;
//                std::cout << "before: " << (weight[i]>>wOffset) << ",after: " << aggStatusTemp << std::endl;
            }
        }

//        if(rank==0 && weight[1]&UNDECIDED) printf("node1: local 2: weight = %lu, aggregate = %lu \n", weight2[1]&weightMax, aggregate[1]);
//        if(rank==0 && weight[3]&UNDECIDED) printf("node3: local 2: weight = %lu, aggregate = %lu \n", weight2[3]&weightMax, aggregate[3]);
//        if(rank==0 && weight[7]&UNDECIDED) printf("node7: local 2: weight = %lu, aggregate = %lu \n", weight2[7]&weightMax, aggregate[7]);

//        if(rank==1){
//            std::cout << std::endl << "after second max computation!" << std::endl;
//            for (i = 0; i < size; ++i)
//                std::cout << i << "\tweight = " << weight[i] << "\tindex = " << aggregate[i] << "\taggStatus = " << aggStatus[i] << std::endl;
//        }

        MPI_Waitall(S->numRecvProc, requests, statuses);

//        MPI_Barrier(comm);
//        iter = 0;
//        if(rank==1)
//            for (i = 0; i < S->col_remote_size; ++i)
//                for (j = 0; j < S->nnzPerCol_remote[i]; ++j, ++iter){
//                    std::cout << "row:" << S->row_remote[iter]+S->split[rank] << "\tneighbor(col) = " << S->col_remote2[iter]
//                         << "\tweight of neighbor = "          << (S->vecValues[2*S->col_remote[iter]]&weightMax)
//                         << "\t\tstatus of neighbor = "        << (S->vecValues[2*S->col_remote[iter]]>>wOffset)
//                         << "\t root_distance of neighbor = "  << (S->vecValues[2*S->col_remote[iter]+1]&weightMax) --> I think this is aggregate of neighbor.
//                         << "\tstatus of agg = "               << (S->vecValues[2*S->col_remote[iter]+1]>>wOffset)  --> I think this is root_distance of neighbor.
//                         << std::endl;
//                }
//        MPI_Barrier(comm);

//        if(rank==0 && i==3) printf("col_index = %lu, status = %lu, aggregate = %u \n", col_index, weight[col_index]>>wOffset, S->col_local[S->indicesP_local[iter]]);
//        if(rank==1) printf("\n");

        // remote part
        // only when there is a 1-distance root, set that as the root, so use weight[] and aggregate[], instead of
        // weight2[] and aggregate2[], since everything for this node gets finalized here. in other cases, store the
        // root candidate in weight2 and aggregate2. then update the parameters after this for loop.
        iter = 0;
        for (i = 0; i < S->col_remote_size; ++i) {
            for (j = 0; j < S->nnzPerCol_remote[i]; ++j, ++iter) {
                i_remote = S->row_remote[iter];
                j_remote = S->col_remote[iter];
                weight_neighbor = S->vecValues[2 * j_remote];
                agg_neighbor    = S->vecValues[2 * j_remote + 1];
//                if(rank==1 && i_remote+S->split[rank]==9) printf("i_remote = %lu \tj_remote = %u, weight = %lu, status = %lu, aggregate = %lu, root_dist = %lu, \t\tstatus of this node = %lu \n",
//                                   i_remote+S->split[rank], S->col_remote2[iter], weight_neighbor&weightMax, weight_neighbor>>wOffset, agg_neighbor&weightMax, agg_neighbor>>wOffset, weight[i_remote]>>wOffset);
//                if(rank==0 && i_remote==3) printf("i_remote = %lu \tj_remote = %u, weight = %lu, status = %lu, aggregate = %lu, root_dist = %lu, \t\tstatus of this node = %lu \n",
//                                                                 i_remote+S->split[rank], S->col_remote2[iter], weight_neighbor&weightMax, weight_neighbor>>wOffset, agg_neighbor&weightMax, agg_neighbor>>wOffset, weight[i_remote]>>wOffset);

                if (weight[i_remote] & UNDECIDED) {

                    weightTemp    = weight2[i_remote] & weightMax;
                    aggregateTemp = aggregate2[i_remote] & weightMax;

                    // distance-1 aggregate
                    // there should be at most one root in distance 1.
                    // only in this case set the status to assigned(0).
                    if (weight_neighbor & ROOT) {
                        weight[i_remote] = 0UL << wOffset | (weight_neighbor & weightMax);
                        aggregate[i_remote] = agg_neighbor & weightMax;
                        root_distance[i_remote] = 1;
                        dist1or2undecided[i_remote] = false;
//                        weight2[i_remote] = weight[i_remote];
//                        if(rank==0) std::cout << i+S->split[rank] << "\t assigned to = " << aggregate[i_remote] << " distance-1 remote \t weight = " << weight[i_remote] << std::endl;
                    } else if (weight_neighbor & UNDECIDED) {

/*
                        if (root_distance[i_remote] == 1)
                            weightTemp = (weight2[i_remote] & weightMax);
                        else{
                            weightTemp = initialWeight[i_remote];
                        }

                        if ( ( (weight_neighbor & weightMax) > weightTemp) ||
                             ( ( (weight_neighbor & weightMax) == weightTemp) && ( ((agg_neighbor & weightMax) > i_remote+S->split[rank]) )) ) {
//                            if(rank==1) std::cout << "first  before\t" << "row = " << S->row_remote[iter]+S->split[rank] << "  \tinitial weight = " << initialWeight[S->row_remote[iter]] << ",\taggregate = " << aggregate2[S->row_remote[iter]] << "\tremote weight = " << (weight2[S->row_remote[iter]]&weightMax) << "\tremote status = " << (S->vecValues[2*S->col_remote[iter]+1]>>wOffset) << "\taggStat2 = " << aggStatus2[S->row_remote[iter]] << std::endl;
                            weight2[i_remote] = (weight_neighbor & weightMax);
                            aggregate2[i_remote] = (agg_neighbor & weightMax);
                            root_distance[i_remote] = 1;
                            dist1or2undecided[i_remote] = true;
//                            aggStatus2[S->row_remote[iter]] = (S->vecValues[2 * S->col_remote[iter]] >> wOffset);
//                            if(rank==1) std::cout << "first  after \t" << "row = " << S->row_remote[iter]+S->split[rank] << "  \tinitial weight = " << initialWeight[S->row_remote[iter]] << ",\taggregate = " << aggregate2[S->row_remote[iter]] << "\t\tremote weight = " << (weight2[S->row_remote[iter]]&weightMax) << "\tremote status = " << (S->vecValues[2*S->col_remote[iter]+1]>>wOffset) << "\taggStat2 = " << aggStatus2[S->row_remote[iter]] << std::endl;
                        }
*/

                        if ( ( (weight_neighbor & weightMax) > weightTemp) ||
                             ( ( (weight_neighbor & weightMax) == weightTemp) && ( ((agg_neighbor & weightMax) > aggregateTemp) )) ) {
//                            if(rank==1) std::cout << "first  before\t" << "row = " << S->row_remote[iter]+S->split[rank] << "  \tinitial weight = " << initialWeight[S->row_remote[iter]] << ",\taggregate = " << aggregate2[S->row_remote[iter]] << "\tremote weight = " << (weight2[S->row_remote[iter]]&weightMax) << "\tremote status = " << (S->vecValues[2*S->col_remote[iter]+1]>>wOffset) << "\taggStat2 = " << aggStatus2[S->row_remote[iter]] << std::endl;
                            weight2[i_remote] = (weight_neighbor & weightMax);
                            aggregate2[i_remote] = (agg_neighbor & weightMax);
                            root_distance[i_remote] = 1;
                            dist1or2undecided[i_remote] = true;
//                            aggStatus2[S->row_remote[iter]] = (S->vecValues[2 * S->col_remote[iter]] >> wOffset);
//                            if(rank==1) std::cout << "first  after \t" << "row = " << S->row_remote[iter]+S->split[rank] << "  \tinitial weight = " << initialWeight[S->row_remote[iter]] << ",\taggregate = " << aggregate2[S->row_remote[iter]] << "\t\tremote weight = " << (weight2[S->row_remote[iter]]&weightMax) << "\tremote status = " << (S->vecValues[2*S->col_remote[iter]+1]>>wOffset) << "\taggStat2 = " << aggStatus2[S->row_remote[iter]] << std::endl;
                        }
                    }

                    // distance-2 aggregate
                    if (root_distance[i_remote] != 1 && ((agg_neighbor >> wOffset) == 1) ){ // this is root_distance of the neighbor.

                        if( ( (weight_neighbor & weightMax) > weightTemp ) ||
                            (( (weight_neighbor & weightMax) == weightTemp ) && (agg_neighbor & weightMax) > aggregateTemp ) ){

                            weight2[i_remote] = (weight_neighbor & weightMax);
                            aggregate2[i_remote] = (agg_neighbor & weightMax);
                            root_distance[i_remote] = 2;
                            if(weight_neighbor >> wOffset == 0) // the neighbor is assigned (00).
                                dist1or2undecided[i_remote] = false;
                            else
                                dist1or2undecided[i_remote] = true;

//                            aggStatus2[S->row_remote[iter]] = (S->vecValues[2 * S->col_remote[iter]]>>wOffset); // this is always 0.

                        }
                    }
                    // if the node is assigned and root_distance is 2, check to see if there is a root in distance-1
                } else if( (weight[i_remote]>>wOffset == 0) && (root_distance[i_remote] == 2) ){
                    if (weight_neighbor & ROOT){
                        weight[i_remote] = (0UL << wOffset | (weight_neighbor & weightMax));
                        aggregate[i_remote] = (agg_neighbor & weightMax);
                        root_distance[i_remote] = 1;
                        dist1or2undecided[i_remote] = false;
                    }
                }
            }
        }

        MPI_Waitall(S->numSendProc, S->numRecvProc+requests, S->numRecvProc+statuses);

        // put weight2 in weight and aggregate2 in aggregate.
        // if a row does not have a remote element then (weight2[i]&weightMax) == (weight[i]&weightMax)
        // update aggStatus of remote elements at the same time
        for(i=0; i<size; i++){
            if( (weight[i]&UNDECIDED) && aggregate[i] != aggregate2[i] ){
                aggregate[i] = aggregate2[i];
                weight[i] = ( 1UL<<wOffset | weight2[i] );
//                if(aggStatus2[i] != 1) // if this is 1, it should go to the next aggregation_2_dist round.
//                    weight[i] = (0UL<<wOffset | weight2[i]&weightMax);
            }
        }

//        if(rank==0 && weight[1]&UNDECIDED) printf("node1: remote:  weight = %lu, aggregate = %lu \n", weight2[1]&weightMax, aggregate[1]);
//        if(rank==1 && weight[9-S->split[rank]]&UNDECIDED) printf("node9: remote:  weight = %lu, aggregate = %lu \n", weight2[9-S->split[rank]]&weightMax, aggregate[9-S->split[rank]]);

        // ******************************* Update Status *******************************
        // "for" loop is of size "number of rows". it checks if a node is UNDECIDED.
        // If aggregate of a node equals its index, that's a root.

//        if(rank==0) std::cout << "******************** Update Status ********************" << std::endl;
        for (i = 0; i < size; ++i) {
            if(weight[i]&UNDECIDED) {
//                if(rank==1) std::cout << "checking " << i+S->split[rank] << "\taggregate[i] = " << aggregate[i] << std::endl;
//                if(rank==1) std::cout << "i = " << i << "\taggregate[i] = " << aggregate[i] << "\taggStatus[aggregate[i]] = " << aggStatus[aggregate[i]] << std::endl;
//                if(rank==1) std::cout << "i = " << i+S->split[rank] << "\taggregate[i] = " << aggregate[i] << "\taggStatus[i] = " << (weight[i]>>wOffset) << std::endl;
                if (aggregate[i] == i + S->split[rank]) { // this node is a root.
                    weight[i] = ( (2UL<<wOffset) | (weight[i]&weightMax) ); // change aggStatus of a root to 2.
                    root_distance[i] = 0;
                    aggArray.emplace_back(aggregate[i]);
//                    if(rank==0) std::cout << "\nroot " << "i = " << i+S->split[rank] << "\t weight = " << (weight[i]&weightMax) << std::endl;
                } else if ( !dist1or2undecided[i] ){
//                    if(rank==0) std::cout << "assign " << "i = " << i+S->split[rank] << "\taggregate[i] = " << aggregate[i] << "\taggStatus[i] = " << (weight[i]>>wOffset) << std::endl;
                    weight[i] = ( (0UL<<wOffset) | (weight[i]&weightMax) );
//                    if(rank==0) std::cout << i+S->split[rank] << "\t assigned to = " << aggregate[i] << " distance-1 or 2 local final step, root_distance = " << root_distance[i] << "\t weight = " << weight[i]  << std::endl;
//                    if (root_distance[aggregate[i] - S->split[rank]] == 0) root_distance[i] = 1; // todo: this is WRONG!
                }
            }
        }

        // here for every node that is assigned to a 2-distance root check if there is a 1-distance root.
        // if there is, then assign this node to it.
        iter = 0;
        for (i = 0; i < size; ++i) {
            if ( (weight[i] >> wOffset == 0) && (root_distance[i] == 2) ) {
                for (j = 0; j < S->nnzPerRow_local[i]; ++j, ++iter) {
                    col_index = S->col_local[S->indicesP_local[iter]] - S->split[rank];
                    if (weight[col_index] & ROOT) {
//                        std::cout << i << "\t col_index = " << col_index << "\t weight[col_index] = " << (weight[col_index] & weightMax) << "\t aggregate = " << S->col_local[S->indicesP_local[iter]] << std::endl;
                        weight[i] = (0UL << wOffset | (weight[col_index] & weightMax));
                        aggregate[i] = S->col_local[S->indicesP_local[iter]];
                        root_distance[i] = 1;
                    }
//                    break; todo: try to add this break.
                }
            }else {
                iter += S->nnzPerRow_local[i];
            }
        }

//        if(rank==0 && weight[1]&UNDECIDED) printf("node1: final:   weight = %lu, aggregate = %lu \n", weight2[1]&weightMax, aggregate[1]);
//        if(rank==0 && weight[3]&UNDECIDED) printf("node3: final:   weight = %lu, aggregate = %lu \n", weight2[3]&weightMax, aggregate[3]);
//        if(rank==0 && weight[7]&UNDECIDED) printf("node7: final:   weight = %lu, aggregate = %lu \n", weight2[7]&weightMax, aggregate[7]);
//        if(rank==0) printf("\n");

//        for(int k=0; k<nprocs; k++){
//            MPI_Barrier(comm);
//            if(rank==k){
//                std::cout << "final aggregate! rank:" << rank << ", iter = " << whileiter << std::endl;
//                for (i = 0; i < size; ++i){
//                    std::cout << "i = " << i+S->split[rank] << "\t\taggregate = " << aggregate[i] << "\t\taggStatus = "
//                         << (weight[i]>>wOffset) << "\t\tinitial weight = " << initialWeight[i]
//                         << "\t\tcurrent weight = " << (weight[i]&weightMax) << "\t\taggStat2 = " << aggStatus2[i]
//                         << "\t\toneDistanceRoot = " << oneDistanceRoot[i] << std::endl;
//                }
//            }
//            MPI_Barrier(comm);
//        }

        // todo: merge this loop with the previous one.
        continueAggLocal = false;
        for (i = 0; i < size; ++i) {
            // if any un-assigned node is available, continue aggregating.
            if(weight[i]&UNDECIDED) {
                continueAggLocal = true;
                break;
            }
        }

//        whileiter++;
//        if(whileiter==15) continueAggLocal = false;

        // check if every processor does not have any non-assigned node, otherwise all the processors should continue aggregating.
        MPI_Allreduce(&continueAggLocal, &continueAgg, 1, MPI_CXX_BOOL, MPI_LOR, comm);

        if(continueAgg){
            for (i = 0; i < size; ++i) {
//                aggStatus2[i] = 1;
                if(weight[i]&UNDECIDED){
//                    printf("%lu\n", i+S->split[rank]);
                    weight[i] = ( 1UL<<wOffset | initialWeight[i] );
                    aggregate[i] = i+S->split[rank];
                }
            }
        }

//        print_vector(aggregate, -1, "aggregate", comm);
//        print_vector(aggArray, -1, "aggArray", comm);
//        if(rank==0){
//            printf("\nnode \tagg \tstatus \troot_distance\n");
//            for(index_t i = 0; i < aggregate.size(); i++){
//                printf("%u \t%lu \t%lu \t%d \n", i+S->split[rank], aggregate[i], (weight[i]>>wOffset), root_distance[i]);
//            }
//        }

    } //while(continueAgg)

//    MPI_Barrier(comm);
//    if(rank==nprocs-1) std::cout << "number of loops to find aggregation_2_dist: " << whileiter << std::endl;
//    MPI_Barrier(comm);

//    for(i=0; i<size;i++)
//        if(rank==0) std::cout << "V[" << i+S->split[rank] << "] = " << initialWeight[i] << ";" << std::endl;

    delete [] requests;
    delete [] statuses;
    free(root_distance);
    free(dist1or2undecided);

    // *************************** avoid P.size == 0  ****************************

    // check if there is not any root nodes on a processor make its first node, a root node

    // keep at least one root node on each proc
//    if(aggArray.empty()){
//        printf("rank %d = aggArray.empty \n", rank);
//        aggArray.emplace_back(0+S->split[rank]);
//        aggregate[0] = 0+S->split[rank];}

    // *************************** update aggregate to new indices ****************************

//    if(rank==0)
//        std::cout << std::endl << "S.M = " << S->M << ", S.nnz_l = " << S->nnz_l << ", S.nnz_l_local = " << S->nnz_l_local
//             << ", S.nnz_l_remote = " << S->nnz_l_remote << std::endl << std::endl;

//    if(rank==1){
//        std::cout << "aggregate:" << std::endl;
//        for(i=0; i<size; i++)
//            std::cout << i+S->split[rank] << "\t" << aggregate[i] << std::endl;
//        std::cout << std::endl;}

    // ************* write the aggregate values of all the nodes to a file *************

    // use this command to concatenate the output files:
    // cat aggregateSaena0.txt aggregateSaena1.txt > aggregateSaena.txt
//    for(i=0; i<aggregate.size(); i++)
//        aggregate[i]++;
//    writeVectorToFileul(aggregate, S->Mbig, "aggregateSaena", comm);
//    for(i=0; i<aggregate.size(); i++)
//        aggregate[i]--;

    // aggArray is the set of root nodes.
    if(!aggArray.empty()) sort(aggArray.begin(), aggArray.end());

//    if(rank==1){
//        std::cout << "aggArray:" << aggArray.size() << std::endl;
//        for(auto i:aggArray)
//            std::cout << i << std::endl;
//        std::cout << std::endl;}

    // ************* write the aggregate nodes to a file *************

    // use this command to concatenate the output files:
    // cat aggArraySaena0.txt aggArraySaena1.txt > aggArraySaena.txt
//    unsigned long aggArray_size, aggArray_size_total;
//    aggArray_size = aggArray.size();
//    MPI_Allreduce(&aggArray_size, &aggArray_size_total, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
//    if(rank==1) std::cout << "total aggArray size = " << aggArray_size_total << std::endl << std::endl;
//    for(i=0; i<aggArray.size(); i++)
//        aggArray[i]++;
//    writeVectorToFileul(aggArray, aggArray_size_total, "aggArraySaena", comm);
//    for(i=0; i<aggArray.size(); i++)
//        aggArray[i]--;

    // ************* print info *************
//    if(rank==0) printf("\n\nfinal: \n");
//    print_vector(aggArray, -1, "aggArray", comm);
//    print_vector(aggregate, -1, "aggregate", comm);
//    for(index_t i = 0; i < aggregate.size(); i++){
//        std::cout << i << "\t" << aggregate[i] << "\t" << (weight[i]>>wOffset) << std::endl;
//    }
//    writeVectorToFileul2(aggregate, "agg2", comm);

    return 0;
}
#endif


int saena_object::change_aggregation(saena_matrix* A, std::vector<index_t>& aggregate, std::vector<index_t>& splitNew){

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    index_t i = 0;

    MPI_Status status;
    MPI_File fh;
    MPI_Offset offset;

    std::string aggName = "agg.bin";
    int mpiopen = MPI_File_open(comm, aggName.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if(mpiopen){
        if (rank==0) std::cout << "Unable to open the vector file!" << std::endl;
        MPI_Finalize();
        return -1;
    }

    // vector should have the following format: first line shows the value in row 0, second line shows the value in row 1
    offset = A->split[rank] * sizeof(double);
    MPI_File_read_at(fh, offset, &*aggregate.begin(), A->M, MPI_UNSIGNED_LONG, &status);
    MPI_File_close(&fh);

//    for(auto i:aggregate)
//        std::cout << i << std::endl;

//    MPI_Status status2;
    MPI_File fh2;
    MPI_Offset offset2;

    std::string aggName2 = "/home/abaris/Dropbox/Projects/Saena/build/juliaAggArray.bin";
    int mpiopen2 = MPI_File_open(comm, aggName2.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh2);
    if(mpiopen2){
        if (rank==0) std::cout << "Unable to open the vector file!" << std::endl;
        MPI_Finalize();
        return -1;
    }

    std::vector<unsigned long> aggArray(A->M);
    // vector should have the following format: first line shows the value in row 0, second line shows the value in row 1
    offset2 = A->split[rank] * sizeof(double);
    MPI_File_read_at(fh2, offset2, &*aggArray.begin(), A->M, MPI_UNSIGNED_LONG, &status);
    MPI_File_close(&fh2);

//    for(auto i:aggArray)
//        std::cout << i << std::endl;

    unsigned long newSize = 0;
    for(auto i:aggArray)
        if(i == 1)
            newSize++;

//    if(rank==0)
//        std::cout << "newSize = " << newSize << std::endl;

    // set splitNew
    fill(splitNew.begin(), splitNew.end(), 0);
    splitNew[rank] = newSize;

    auto* splitNewTemp = new index_t[nprocs];
    MPI_Allreduce(&splitNew[0], splitNewTemp, nprocs, par::Mpi_datatype<index_t>::value(), MPI_SUM, comm);

    // do scan on splitNew
    splitNew[0] = 0;
    for(i = 1; i < nprocs+1; ++i)
        splitNew[i] = splitNew[i-1] + splitNewTemp[i-1];

//    for(i=0; i<nprocs+1; i++)
//        std::cout << splitNew[i] << std::endl;

    delete []splitNewTemp;

    return 0;
}


int saena_object::aggregate_index_update(strength_matrix* S, std::vector<index_t>& aggregate,
                                         std::vector<index_t>& aggArray, std::vector<index_t>& splitNew){
    // ************* update aggregates' indices *************
    // check each node to see if it is assigned to a local or remote node.
    // if it is local, then aggregate[i] will be to the root's new index,
    // and if it is remote, then it will be add to aggregateRemote to communicate the new index for its root.
    // **********************************************************

    MPI_Comm comm = S->comm;

    int nprocs = -1, rank = -1;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    index_t size    = S->M;
    index_t procNum = 0;
    std::vector<index_t> aggregateRemote;
    std::vector<index_t> recvProc;

    // ************* compute splitNew *************

    index_t agg_sz = aggArray.size();
    splitNew.resize(nprocs+1);
    MPI_Allgather(&agg_sz,      1, par::Mpi_datatype<index_t>::value(),
                  &splitNew[1], 1, par::Mpi_datatype<index_t>::value(), comm);

    // do scan on splitNew
    splitNew[0] = 0;
    for(index_t i = 1; i < nprocs+1; ++i)
        splitNew[i] += splitNew[i-1];

#ifdef __DEBUG1__
//    print_vector(splitNew, 0, "splitNew", comm);
//    if(rank==0){
//        std::cout << "split and splitNew:" << std::endl;
//        for(i=0; i<nprocs+1; i++)
//            std::cout << S->split[i] << "\t" << splitNew[i] << std::endl;
//        std::cout << std::endl;}
#endif

    // local update
    // --------------
    std::vector<bool> isAggRemote(size);
    for(index_t i = 0; i < size; ++i){
        if(aggregate[i] >= S->split[rank] && aggregate[i] < S->split[rank+1]){
            aggregate[i] = lower_bound2(&*aggArray.begin(), &*aggArray.end(), aggregate[i]) + splitNew[rank];
//            if(rank==1) std::cout << aggregate[i] << std::endl;
            isAggRemote[i] = false;
        }else{
            isAggRemote[i] = true;
            aggregateRemote.emplace_back(aggregate[i]);
        }
    }

//    set<unsigned long> aggregateRemote2(aggregateRemote.begin(), aggregateRemote.end());
//    if(rank==1) std::cout << "i and procNum:" << std::endl;
//    for(auto i:aggregateRemote2){
//        procNum = lower_bound2(&S->split[0], &S->split[nprocs+1], i);
//        if(rank==1) std::cout << i << "\t" << procNum << std::endl;
//        recvCount[procNum]++;
//    }

    // remote update
    // ------------
    sort(aggregateRemote.begin(), aggregateRemote.end());
    auto last = unique(aggregateRemote.begin(), aggregateRemote.end()); // Unique() Removes consecutive duplicates.
    aggregateRemote.erase(last, aggregateRemote.end());

//    MPI_Barrier(comm); printf("rank %d: aggregateRemote size = %ld \n", rank, aggregateRemote.size()); MPI_Barrier(comm);

    std::vector<int> recvCount(nprocs, 0);
    for(auto i:aggregateRemote){
        procNum = lower_bound2(&S->split[0], &S->split[nprocs], index_t(i));
        ++recvCount[procNum];
//        if(rank==0) std::cout << i << "\t" << procNum << std::endl;
    }

#ifdef __DEBUG1__
//    print_vector(recvCount, -1, "recvCount", comm);
#endif

    int recvSize = 0;
    int vIndexSize = 0;

    if(nprocs > 1) {
        std::vector<int> vIndexCount(nprocs);
        MPI_Alltoall(&recvCount[0], 1, MPI_INT, &vIndexCount[0], 1, MPI_INT, comm);

#ifdef __DEBUG1__
//        print_vector(vIndexCount, -1, "vIndexCount", comm);
#endif

        // this part is for isend and ireceive.
        std::vector<int> recvProcRank;
        std::vector<int> recvProcCount;
        std::vector<int> sendProcRank;
        std::vector<int> sendProcCount;
        int numRecvProc = 0;
        int numSendProc = 0;
        for (int i = 0; i < nprocs; ++i) {
            if (recvCount[i] != 0) {
                ++numRecvProc;
                recvProcRank.emplace_back(i);
                recvProcCount.emplace_back(recvCount[i]);
            }
            if (vIndexCount[i] != 0) {
                ++numSendProc;
                sendProcRank.emplace_back(i);
                sendProcCount.emplace_back(vIndexCount[i]);
            }
        }

        std::vector<int> vdispls(nprocs);
        std::vector<int> rdispls(nprocs);
        vdispls[0] = 0;
        rdispls[0] = 0;

        for (int i = 1; i < nprocs; ++i) {
            vdispls[i] = vdispls[i - 1] + vIndexCount[i - 1];
            rdispls[i] = rdispls[i - 1] + recvCount[i - 1];
        }
        vIndexSize = vdispls[nprocs - 1] + vIndexCount[nprocs - 1];
        recvSize   = rdispls[nprocs - 1] + recvCount[nprocs - 1];

#ifdef __DEBUG1__
//        MPI_Barrier(comm); printf("rank %d: vIndexSize = %d, recvSize = %d \n", rank, vIndexSize, recvSize); MPI_Barrier(comm);
#endif

        std::vector<index_t> vIndex(vIndexSize);
        MPI_Alltoallv(&aggregateRemote[0], &recvCount[0],   &rdispls[0], par::Mpi_datatype<index_t>::value(),
                      &vIndex[0],          &vIndexCount[0], &vdispls[0], par::Mpi_datatype<index_t>::value(), comm);
//        MPI_Alltoallv(&*aggregateRemote2.begin(), recvCount, &*rdispls.begin(), MPI_UNSIGNED_LONG, vIndex, vIndexCount, &*vdispls.begin(), MPI_UNSIGNED_LONG, comm);

        std::vector<index_t> aggSend(vIndexSize);
        std::vector<index_t> aggRecv(recvSize);

        auto *aggregate_p = &aggregate[-S->split[rank]];
        for (index_t i = 0; i < vIndexSize; ++i) {
            aggSend[i] = aggregate_p[vIndex[i]];
//            if(rank==1) std::cout << "vIndex = " << vIndex[i] << "\taggSend = " << aggSend[i] << std::endl;
        }

        // replace this alltoallv with isend and irecv.
//        MPI_Alltoallv(aggSend, vIndexCount, &*(vdispls.begin()), MPI_UNSIGNED_LONG, aggRecv, recvCount, &*(rdispls.begin()), MPI_UNSIGNED_LONG, comm);

        auto* requests2 = new MPI_Request[numSendProc + numRecvProc];
        auto* statuses2 = new MPI_Status[numSendProc + numRecvProc];

        for (int i = 0; i < numRecvProc; ++i)
            MPI_Irecv(&aggRecv[rdispls[recvProcRank[i]]], recvProcCount[i], par::Mpi_datatype<index_t>::value(),
                    recvProcRank[i], 1, comm, &requests2[i]);

        for (int i = 0; i < numSendProc; i++)
            MPI_Isend(&aggSend[vdispls[sendProcRank[i]]], sendProcCount[i], par::Mpi_datatype<index_t>::value(),
                    sendProcRank[i], 1, comm, &requests2[numRecvProc + i]);

        MPI_Waitall(numRecvProc, requests2, statuses2);

//        if(rank==1) std::cout << "aggRemote received:" << std::endl;
//        set<unsigned long>::iterator it;
//        for(i=0; i<size; i++){
//            if(isAggRemote[i]){
//                it = aggregateRemote2.find(aggregate[i]);
//                if(rank==1) std::cout << aggRecv[ distance(aggregateRemote2.begin(), it) ] << std::endl;
//                aggregate[i] = aggRecv[ distance(aggregateRemote2.begin(), it) ];
//            }
//        }

        // remote
        for (nnz_t i = 0; i < size; ++i) {
            if (isAggRemote[i]) {
                aggregate[i] = aggRecv[lower_bound2(&*aggregateRemote.begin(), &*aggregateRemote.end(), aggregate[i])];
//                if(rank==1) std::cout << i << "\t" << aggRecv[ lower_bound2(&*aggregateRemote.begin(), &*aggregateRemote.end(), aggregate[i]) ] << std::endl;
            }
        }

#ifdef __DEBUG1__
//        print_vector(aggregate, -1, "aggregate", comm);
#endif

        MPI_Waitall(numSendProc, numRecvProc + requests2, numRecvProc + statuses2);
        delete[] requests2;
        delete[] statuses2;
    }

    return 0;
}


// Decoupled Aggregation - not complete
/*
int SaenaObject::Aggregation(CSRMatrix* S){
    // At the end just set P here, which is in SaenaObject.

    std::vector<long> P(S->M);
    P.assign(S->M,-1);
    long nc = -1;
    bool isInN1[S->M];
    bool aggregated;
    unsigned int i;
    double tau_sbar = tau * S->average_sparsity;
    for(i=0; i<S->M; i++){
        if( S->rowIndex[i+1] - S->rowIndex[i] <= tau_sbar ){
            isInN1[1] = true;
        }
    }

    // ************************************* first pass *************************************

    for (i=0; i<S->M; i++){
        aggregated = true;
        if(isInN1[i] == false)
            continue;

        for(long j=S->rowIndex[i]; j<S->rowIndex[i+1]; j++){
            if(P[S->col[j]] == -1)
                break;
            aggregated = false;
        }

        if(aggregated==false) {
            nc++;
            for (long j = S->rowIndex[i]; j < S->rowIndex[i+1]; j++) {
                if(isInN1[S->col[i]] == true){
                    P[S->col[i]] = nc;
                }
            }
        }
    }

    // ************************************* second pass *************************************

    for (i=0; i<S->M; i++){
        aggregated = true;
        if(isInN1[i] == true)
            continue;

        for(long j=S->rowIndex[i]; j<S->rowIndex[i+1]; j++){
            if(P[S->col[j]] == -1)
                break;
            aggregated = false;
        }

        if(aggregated==false) {
            nc++;
            for (long j = S->rowIndex[i]; j < S->rowIndex[i+1]; j++) {
                P[S->col[i]] = nc;
            }
        }
    }

    // ************************************* third pass *************************************

    nc++;

    return 0;
};
 */