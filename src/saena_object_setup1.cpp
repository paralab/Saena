#include "saena_object.h"
#include "saena_matrix.h"
#include "strength_matrix.h"
#include "prolong_matrix.h"
#include "restrict_matrix.h"
#include "grid.h"
#include "aux_functions.h"
#include "parUtils.h"
#include "dollar.hpp"
#include "superlu_ddefs.h"
#include <superlu_defs.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <set>
#include <random>
#include <mpi.h>


int saena_object::create_prolongation(saena_matrix* A, std::vector<unsigned long>& aggregate, prolong_matrix* P){
    // formula for the prolongation matrix from Irad Yavneh's paper:
    // P = (I - 4/(3*rhoDA) * DA) * P_t

    // todo: check when you should update new aggregate values: before creating prolongation or after.

    // Here P is computed: P = A_w * P_t; in which P_t is aggregate, and A_w = I - w*Q*A, Q is inverse of diagonal of A.
    // Here A_w is computed on the fly, while adding values to P. Diagonal entries of A_w are 0, so they are skipped.
    // todo: think about A_F which is A filtered.
    // todo: think about smoothing preconditioners other than damped jacobi. check the following paper:
    // todo: Eran Treister and Irad Yavneh, Non-Galerkin Multigrid based on Sparsified Smoothed Aggregation. page22.

//    MPI_Comm_dup(A->comm, &P->comm);
    P->comm = A->comm;
    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
//    unsigned int i, j;
    float omega = A->jacobi_omega; // todo: receive omega as user input. it is usually 2/3 for 2d and 6/7 for 3d.

    P->Mbig = A->Mbig;
    P->Nbig = P->splitNew[nprocs]; // This is the number of aggregates, which is the number of columns of P.
    P->M = A->M;

    // store remote elements from aggregate in vSend to be sent to other processes.
    // todo: is it ok to use vSend instead of vSendULong? vSend is double and vSendULong is unsigned long.
    // todo: the same question for vecValues and Isend and Ireceive.
    for(index_t i = 0; i < A->vIndexSize; i++){
        A->vSendULong[i] = aggregate[( A->vIndex[i] )];
//        std::cout <<  A->vIndex[i] << "\t" << A->vSendULong[i] << std::endl;
    }

    MPI_Request* requests = new MPI_Request[A->numSendProc+A->numRecvProc];
    MPI_Status*  statuses = new MPI_Status[A->numSendProc+A->numRecvProc];

    // todo: here
    // todo: are vSendULong and vecValuesULong required to be a member of the class.

    for(index_t i = 0; i < A->numRecvProc; i++)
        MPI_Irecv(&A->vecValuesULong[A->rdispls[A->recvProcRank[i]]], A->recvProcCount[i], MPI_UNSIGNED_LONG, A->recvProcRank[i], 1, comm, &(requests[i]));

    for(index_t i = 0; i < A->numSendProc; i++)
        MPI_Isend(&A->vSendULong[A->vdispls[A->sendProcRank[i]]], A->sendProcCount[i], MPI_UNSIGNED_LONG, A->sendProcRank[i], 1, comm, &(requests[A->numRecvProc+i]));

    std::vector<cooEntry> PEntryTemp;

    // P = (I - 4/(3*rhoDA) * DA) * P_t
    // aggreagte is used as P_t in the following "for" loop.
    // local
    // -----
    long iter = 0;
    for (index_t i = 0; i < A->M; ++i) {
        for (index_t j = 0; j < A->nnzPerRow_local[i]; ++j, ++iter) {
            if(A->row_local[A->indicesP_local[iter]] == A->col_local[A->indicesP_local[iter]]-A->split[rank]){ // diagonal element
                PEntryTemp.emplace_back(cooEntry(A->row_local[A->indicesP_local[iter]],
                                                 aggregate[ A->col_local[A->indicesP_local[iter]] - A->split[rank] ],
                                                 1 - omega));
            }else{
                PEntryTemp.emplace_back(cooEntry(A->row_local[A->indicesP_local[iter]],
                                                 aggregate[ A->col_local[A->indicesP_local[iter]] - A->split[rank] ],
                                                 -omega * A->values_local[A->indicesP_local[iter]] * A->inv_diag[A->row_local[A->indicesP_local[iter]]]));
            }
//            std::cout << A->row_local[A->indicesP_local[iter]] << "\t" << aggregate[A->col_local[A->indicesP_local[iter]] - A->split[rank]] << "\t" << A->values_local[A->indicesP_local[iter]] * A->inv_diag[A->row_local[A->indicesP_local[iter]]] << std::endl;
        }
    }

    MPI_Waitall(A->numRecvProc, requests, statuses);

    // remote
    // ------saena_object
    iter = 0;
    for (index_t i = 0; i < A->col_remote_size; ++i) {
        for (index_t j = 0; j < A->nnzPerCol_remote[i]; ++j, ++iter) {
            PEntryTemp.emplace_back(cooEntry(A->row_remote[iter],
                                             A->vecValuesULong[A->col_remote[iter]],
                                             -omega * A->values_remote[iter] * A->inv_diag[A->row_remote[iter]]));
//            P->values.emplace_back(A->values_remote[iter]);
//            std::cout << A->row_remote[iter] << "\t" << A->vecValuesULong[A->col_remote[iter]] << "\t"
//                      << A->values_remote[iter] * A->inv_diag[A->row_remote[iter]] << std::endl;
        }
    }

    std::sort(PEntryTemp.begin(), PEntryTemp.end());

//    print_vector(PEntryTemp, 0, "PEntryTemp", comm);

    // todo: here
//    P->entry.resize(PEntryTemp.size());
    // remove duplicates.
    for(index_t i=0; i<PEntryTemp.size(); i++){
        P->entry.emplace_back(PEntryTemp[i]);
        while(i<PEntryTemp.size()-1 && PEntryTemp[i] == PEntryTemp[i+1]){ // values of entries with the same row and col should be added.
            P->entry.back().val += PEntryTemp[i+1].val;
            i++;
        }
    }

//    print_vector(P->entry, 0, "P->entry", comm);

    P->nnz_l = P->entry.size();
    MPI_Allreduce(&P->nnz_l, &P->nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
    P->split = A->split;
    P->findLocalRemote();

    MPI_Waitall(A->numSendProc, A->numRecvProc+requests, A->numRecvProc+statuses);
    delete [] requests;
    delete [] statuses;

    return 0;
}// end of saena_object::create_prolongation


int saena_object::find_aggregation(saena_matrix* A, std::vector<unsigned long>& aggregate, std::vector<index_t>& splitNew){
    // finding aggregation is written in an adaptive way. An aggregation_2_dist is being created first. If it is too small,
    // or too big it will be recreated until an aggregation_2_dist with size within the acceptable range is produced.

    MPI_Comm comm = A->comm;
    int rank;
    MPI_Comm_rank(comm, &rank);
//    MPI_Comm_size(A->comm, &nprocs);

    strength_matrix S;
    create_strength_matrix(A, &S);
//    S.print_entry(-1);

    std::vector<unsigned long> aggArray; // vector of root nodes.

    aggregation_1_dist(&S, aggregate, aggArray);
//    aggregation_2_dist(&S, aggregate, aggArray);

    double division = 0;
    unsigned int new_size_local, new_size=0;

    // new_size is the size of the new coarse matrix.
    new_size_local = unsigned(aggArray.size());
    MPI_Allreduce(&new_size_local, &new_size, 1, MPI_UNSIGNED, MPI_SUM, comm);
    division = (double)A->Mbig / new_size;
//    if(rank==0) {
//        printf("\nconnStrength = %.2f \ncurrent size = %u \nnew size     = %u \ndivision     = %.2f\n",
//               connStrength, A->Mbig, new_size, division);
//    }

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
            new_size_local = unsigned(aggArray.size());
            MPI_Allreduce(&new_size_local, &new_size, 1, MPI_UNSIGNED, MPI_SUM, comm);
            division = (double)A->Mbig / new_size;
            if(rank==0) printf("\nconnStrength = %.2f \ncurrent size = %u \nnew size     = %u \ndivision     = %.2f\n",
                               connStrength_temp, A->Mbig, new_size, division);

            if (connStrength_temp < 0.2 || connStrength_temp > 0.95){
                continue_agg = false;
            }

        }

    }





//    if(rank==0) printf("\nfinal: connStrength = %f, current size = %u, new size = %u,  division = %d\n",
//                       connStrength_temp, A->Mbig, new_size, division);

//    connStrength = connStrength_temp;
    aggregate_index_update(&S, aggregate, aggArray, splitNew);
//    updateAggregation(aggregate, &aggSize);

//    print_vector(aggArray, -1, "aggArray", comm);

    return 0;
} // end of SaenaObject::findAggregation


int saena_object::create_strength_matrix(saena_matrix* A, strength_matrix* S){

    // based on the following paper by Irad Yavneh:
    // Non-Galerkin Multigrid Based on Sparsified Smoothed Aggregation - page: A51

    // todo: add openmp to this function.

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    printf("inside strength: rank %d: A->M = %d \tA->nnz_l = %d \n", rank, A->M, A->nnz_l);

    // ******************************** compute max per row ********************************

    std::vector<value_t> maxPerRow(A->M, 0);
    index_t row_local;
    for(nnz_t i=0; i<A->nnz_l; i++){
        row_local = A->entry[i].row - A->split[rank];  // use split to convert the index from global to local.
        if( A->entry[i].row != A->entry[i].col ){
            if(maxPerRow[row_local] == 0 || maxPerRow[row_local] < -A->entry[i].val)
                maxPerRow[row_local] = -A->entry[i].val;
        }
    }

//    print_vector(maxPerRow, -1, "maxPerRow", comm);

    // ******************************** compute S ********************************

    S->entry.resize(A->nnz_l);
    value_t val_temp = 0;
    for(nnz_t i = 0; i < A->nnz_l; i++){
//        if(rank==0) std::cout << A->entry[i] << "\t maxPerRow = " << maxPerRow[A->entry[i].row - A->split[rank]] << std::endl;
        if(A->entry[i].row == A->entry[i].col) {
            val_temp = 1;
        } else {
            if(maxPerRow[A->entry[i].row - A->split[rank]] == 0)
                printf("\n\nerror in saena_object::create_strength_matrix: maxPerRow[%lu] == 0 on rank %d \n\n", i, rank);
            val_temp = -A->entry[i].val / (maxPerRow[A->entry[i].row - A->split[rank]]);
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

    // since A is symmetric, use maxPerRow for local entries on each process. receive the remote ones like matvec.

    //vSend are maxPerCol for remote elements that should be sent to other processes.
    for(nnz_t i = 0; i<A->vIndexSize; i++)
        A->vSend[i] = maxPerRow[( A->vIndex[i] )];

    MPI_Request* requests = new MPI_Request[A->numSendProc+A->numRecvProc];
    MPI_Status* statuses  = new MPI_Status[A->numSendProc+A->numRecvProc];

    //vecValues are maxperCol for remote elements that are received from other processes.
    // Do not recv from self.
    for(nnz_t i = 0; i < A->numRecvProc; i++)
        MPI_Irecv(&A->vecValues[A->rdispls[A->recvProcRank[i]]], A->recvProcCount[i], MPI_DOUBLE, A->recvProcRank[i], 1, comm, &(requests[i]));

    // Do not send to self.
    for(nnz_t i = 0; i < A->numSendProc; i++)
        MPI_Isend(&A->vSend[A->vdispls[A->sendProcRank[i]]], A->sendProcCount[i], MPI_DOUBLE, A->sendProcRank[i], 1, comm, &(requests[A->numRecvProc+i]));

    // ******************************** compute ST - version 2 ********************************

//    std::vector<long> STi;
//    std::vector<long> STj;
//    std::vector<double> STval;

    // add OpenMP just like matvec.
    long iter = 0;
    long iter2 = 0;
//    for (i = 0; i < A->M; ++i, iter2++) {
//        for (unsigned int j = 0; j < A->nnzPerRow_local[i]; ++j, ++iter) {
//
//            // diagonal entry
//            if(i == A->col_local[A->indicesP_local[iter]]){
//                STi.emplace_back(iter2); // iter2 is actually i, but it was giving an error for using i.
//                STj.emplace_back(A->col_local[A->indicesP_local[iter]]);
//                STval.emplace_back(1);
//                continue;
//            }
//
//            STi.emplace_back(iter2); // iter2 is actually i, but it was giving an error for using i.
//            STj.emplace_back(A->col_local[A->indicesP_local[iter]]);
//            STval.emplace_back( -A->values_local[A->indicesP_local[iter]] / maxPerRow[A->col_local[A->indicesP_local[iter]]] );
//        }
//    }

    // local ST values
    S->entryT.resize(A->nnz_l);
    for (nnz_t i = 0; i < A->nnz_l_local; ++i, iter2++) {

        if(A->row_local[i] == A->col_local[i]) // diagonal entry
            val_temp = 1;
        else
            val_temp = -A->values_local[i] / maxPerRow[A->col_local[i] - A->split[rank]];

//        if(rank==0) printf("%u \t%u \t%f \n", A->row_local[i], A->col_local[i], val_temp);
        S->entryT[i] = cooEntry(A->row_local[i], A->col_local[i], val_temp);
    }

    MPI_Waitall(A->numRecvProc, requests, statuses);

    // add OpenMP just like matvec.
//    iter = 0;
//    for (i = 0; i < A->col_remote_size; ++i) {
//        for (unsigned int j = 0; j < A->nnzPerCol_remote[i]; ++j, ++iter) {
//            STi.emplace_back(A->row_remote[A->indicesP_remote[iter]]);
//            STj.emplace_back(A->col_remote2[A->indicesP_remote[iter]]);
//            STval.emplace_back( -A->values_remote[A->indicesP_remote[iter]] / A->vecValues[A->col_remote[A->indicesP_remote[iter]]] );
//        }
//    }

    // remote ST values
    // add OpenMP just like matvec.
    iter = 0;
    for (nnz_t i = 0; i < A->vElement_remote.size(); ++i) {
        for (unsigned int j = 0; j < A->vElementRep_remote[i]; ++j, ++iter) {
//            if(rank==1) printf("%u \t%u \t%f \n", A->row_remote[iter], A->col_remote2[iter], -A->values_remote[iter] / A->vecValues[i]);
//            w[A->row_remote[A->indicesP_remote[iter]]] += A->values_remote[A->indicesP_remote[iter]] * A->vecValues[A->col_remote[A->indicesP_remote[iter]]];
            S->entryT[iter + A->nnz_l_local] = cooEntry(A->row_remote[iter], A->col_remote2[iter], -A->values_remote[iter] / A->vecValues[i]);
        }
    }

    std::sort(S->entryT.begin(), S->entryT.end());
//    print_vector(S->entryT, -1, "S->entryT", comm);

    // ******************************** setup the matrix S ********************************

    // S indices are local on each process, which means it starts from 0 on each process.
//    S->setup_matrix(Si2, Sj2, Sval2, A->M, A->Mbig, Si2.size(), A->split, comm);
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
int saena_object::aggregation_1_dist(strength_matrix *S, std::vector<unsigned long> &aggregate,
                                     std::vector<unsigned long> &aggArray) {

    // todo: update the comments for 1-distance independent set.
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

//    S->randomVector(initialWeight, S->Mbig, comm);
    S->randomVector3(initialWeight, comm);
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
    unsigned long i_remote, j_remote, weight_neighbor, agg_neighbor, aggStatus_neighbor, iter, col_index;
    int whileiter = 0;

    MPI_Request *requests = new MPI_Request[S->numSendProc + S->numRecvProc];
    MPI_Status *statuses  = new MPI_Status[S->numSendProc + S->numRecvProc];

//    if(rank==0) std::cout << "set boundary points: " << std::endl;

    // initialization -> this part is merged to the first "for" loop in the following "while".
    for(i=0; i<size; i++) {
        aggregate[i] = i + S->split[rank];
//        aggStatus2[i] = 1;
        // Some boundary nodes are the ones that only have one neighbor which is itself (so one nnzPerRow),
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
        // todo: WARNING: S->col_local[S->indicesP_local[iter]] is not ordered row-major! is it fine?
        iter = 0;
        for (i = 0; i < size; ++i) {
            if(weight[i]&UNDECIDED) {
//                if(rank==0) printf("\n");
                root_distance[i] = 3; // initialization
                dist1or2undecided[i] = false; // initialization
                aggregate2[i] = aggregate[i];
                weight2[i] = weight[i]&weightMax;
//                weight2[i] = weight[i]&weightMax;
//                aggStatusTemp = 1UL; // this will be used for aggStatus2, and aggStatus2 will be used for the remote part.
                for (j = 0; j < S->nnzPerRow_local[i]; ++j, ++iter) {
                    col_index = S->col_local[S->indicesP_local[iter]] - S->split[rank];
                    aggStatus_neighbor = weight[col_index]>>wOffset;
//                    if(rank==0) printf("node = %lu, \tmy_init_wei = %lu, \tweight2 = %lu, \tcol_ind = %lu, \tneighbor_stat = %lu, \tneighbor_init_wei = %lu, \tneighbor_agg = %lu \n",
//                                       i, initialWeight[i], weight2[i], col_index, aggStatus_neighbor, initialWeight[col_index], aggregate[col_index]);
                    if( aggStatus_neighbor != 0 ){ // neighbor being ROOT or UNDECIDED (not ASSIGNED).
                        if( (initialWeight[col_index] > weight2[i]) ||
                            ((initialWeight[col_index] == weight2[i]) && (aggregate[col_index] > aggregate2[i])) ){

                            weight2[i] = (weight[col_index] & weightMax);
                            aggregate2[i] = S->col_local[S->indicesP_local[iter]];
                            root_distance[i] = 1;
                            if(aggStatus_neighbor == 2){ // ROOT
                                dist1or2undecided[i] = false;
                            }else{ // UNDECIDED
                                dist1or2undecided[i] = true;
                            }

//                            weight[i] = (0UL << wOffset | (weight[col_index] & weightMax)); // 0UL << wOffset is not required.
//                            aggregate[i] = S->col_local[S->indicesP_local[iter]];
//                            if(rank==0) std::cout << i+S->split[rank] << "\t assigned to = " << aggregate[i] << " distance-1 local \t weight = " << weight[i] << std::endl;
                        }
                    }
                }
//                if(rank==1) std::cout << i+S->split[rank] << "," << aggregate2[i] << "\t\t";
            }else{
                iter += S->nnzPerRow_local[i];
            }
        }

/*
        // todo: for distance-1 it is probably safe to remove this for loop, and change weight2 to weight and aggregate2 to aggregate at the end of the previous for loop.
        for (i = 0; i < size; ++i) {
            if( (S->nnzPerRow_local[i]!=0) && (weight[i]&UNDECIDED) && (root_distance[i]==1)) {
                weight[i] = (1UL << wOffset | weight2[i] & weightMax );
                aggregate[i] = aggregate2[i];
//                if(rank==0) std::cout << i+S->split[rank] << "\t" << aggregate[i] << "\t" << aggregate2[i] << std::endl;
            }
        }
*/

//        if(rank==0 && weight[1]&UNDECIDED) printf("node1: local 1: weight = %lu, aggregate = %lu \n", weight2[1]&weightMax, aggregate[1]);
//        if(rank==0 && weight[3]&UNDECIDED) printf("node3: local 1: weight = %lu, aggregate = %lu \n", weight2[3]&weightMax, aggregate[3]);
//        if(rank==0 && weight[7]&UNDECIDED) printf("node7: local 1: weight = %lu, aggregate = %lu \n", weight2[7]&weightMax, aggregate[7]);

        //    if(rank==0){
        //        std::cout << std::endl << "after first max computation!" << std::endl;
        //        for (i = 0; i < size; ++i)
        //            std::cout << i << "\tweight = " << weight[i] << "\tindex = " << aggregate[i] << std::endl;
        //    }

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

                if (weight[i_remote] & UNDECIDED) {

                    j_remote = S->col_remote[iter];
                    weight_neighbor    = S->vecValues[2 * j_remote] & weightMax;
                    aggStatus_neighbor = S->vecValues[2 * j_remote] >> wOffset;
                    agg_neighbor       = S->vecValues[2 * j_remote + 1] & weightMax;

                    weightTemp    = weight2[i_remote] & weightMax;
                    aggregateTemp = aggregate2[i_remote] & weightMax;

//                if(rank==1 && i_remote+S->split[rank]==9) printf("i_remote = %lu \tj_remote = %u, weight = %lu, status = %lu, aggregate = %lu, root_dist = %lu, \t\tstatus of this node = %lu \n",
//                                   i_remote+S->split[rank], S->col_remote2[iter], weight_neighbor&weightMax, weight_neighbor>>wOffset, agg_neighbor&weightMax, agg_neighbor>>wOffset, weight[i_remote]>>wOffset);
//                if(rank==0 && i_remote==3) printf("i_remote = %lu \tj_remote = %u, weight = %lu, status = %lu, aggregate = %lu, root_dist = %lu, \t\tstatus of this node = %lu \n",
//                                                                 i_remote+S->split[rank], S->col_remote2[iter], weight_neighbor&weightMax, weight_neighbor>>wOffset, agg_neighbor&weightMax, agg_neighbor>>wOffset, weight[i_remote]>>wOffset);

                    // distance-1 aggregate
                    // there should be at most one root in distance 1.
                    // only in this case set the status to assigned(0).
                    if (aggStatus_neighbor != 0) { // neighbor being ROOT or UNDECIDED (not ASSIGNED).
                        if ( ( weight_neighbor > weightTemp) ||
                             ( ( weight_neighbor == weightTemp) && ( (agg_neighbor > aggregateTemp) )) ){

                            weight2[i_remote] = weight_neighbor;
                            aggregate2[i_remote] = agg_neighbor;
                            root_distance[i_remote] = 1;

                            if (aggStatus_neighbor == 2) // ROOT
                                dist1or2undecided[i_remote] = false;
                            else // UNDECIDED
                                dist1or2undecided[i_remote] = true;
                        }
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
                weight[i] = ( 1UL<<wOffset | weight2[i] ); // keep it undecided.
//                if(aggStatus2[i] != 1) // if this is 1, it should go to the next aggregation_2_dist round.
//                    weight[i] = (0UL<<wOffset | weight2[i]&weightMax);
            }
        }

//        if(rank==0 && weight[1]&UNDECIDED) printf("node1: remote:  weight = %lu, aggregate = %lu \n", weight2[1]&weightMax, aggregate[1]);
//        if(rank==0 && weight[3]&UNDECIDED) printf("node3: remote:  weight = %lu, aggregate = %lu \n", weight2[3]&weightMax, aggregate[3]);
//        if(rank==0 && weight[7]&UNDECIDED) printf("node7: remote:  weight = %lu, aggregate = %lu \n", weight2[7]&weightMax, aggregate[7]);
//        if(rank==1 && weight[9-S->split[rank]]&UNDECIDED) printf("node9: remote:  weight = %lu, aggregate = %lu \n", weight2[9-S->split[rank]]&weightMax, aggregate[9-S->split[rank]]);

        // ******************************* Update Status *******************************
        // "for" loop is of size "number of rows". it first checks if a node is UNDECIDED.
        // If aggregate of a node equals its index, that's a root.

//        if(rank==0) std::cout << "******************** Update Status ********************" << std::endl;
        for (i = 0; i < size; ++i) {
            if(weight[i]&UNDECIDED) {
//                if(rank==1) std::cout << "checking " << i+S->split[rank] << "\taggregate[i] = " << aggregate[i] << std::endl;

                if (aggregate[i] == i + S->split[rank]) {
                    weight[i] = ( (2UL<<wOffset) | (weight[i]&weightMax) ); // change aggStatus of a root to 2.
                    root_distance[i] = 0;
                    aggArray.emplace_back(aggregate[i]);
//                        if(rank==0) std::cout << "\nroot " << "i = " << i+S->split[rank] << "\t weight = " << (weight[i]&weightMax) << std::endl;

                    // this node should become an "ASSIGNED".
                } else if ( !dist1or2undecided[i] ){
//                        if(rank==0) std::cout << "assign " << "i = " << i+S->split[rank] << "\taggregate[i] = " << aggregate[i] << "\taggStatus[i] = " << (weight[i]>>wOffset) << std::endl;
                    weight[i] = ( (0UL<<wOffset) | (weight[i]&weightMax) );
//                        if(rank==0) std::cout << i+S->split[rank] << "\t assigned to = " << aggregate[i] << " distance-1 or 2 local final step, root_distance = " << root_distance[i] << "\t weight = " << weight[i]  << std::endl;
                }
            }
        }

/*
        // here for every node that is assigned to a 2-distance root check if there is a 1-distance root.
        // if there is, then assign to it.
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
*/

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
    if(!aggArray.empty()) std::sort(aggArray.begin(), aggArray.end());

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
//    if(rank==0){
//        printf("\nnode \tagg\n");
//        for(index_t i = 0; i < aggregate.size(); i++){
//            printf("%u \t%lu \n", i+S->split[rank]+1, aggregate[i]+1);
//        }
//    }
//    writeVectorToFileul2(aggregate, "agg1", comm);

    return 0;
}


// Using MIS(2) from the following paper by Luke Olson:
// EXPOSING FINE-GRAINED PARALLELISM IN ALGEBRAIC MULTIGRID METHODS
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

//    S->randomVector(initialWeight, S->Mbig, comm);
    S->randomVector3(initialWeight, comm);
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
//        if(rank==0 && weight[3]&UNDECIDED) printf("node3: remote:  weight = %lu, aggregate = %lu \n", weight2[3]&weightMax, aggregate[3]);
//        if(rank==0 && weight[7]&UNDECIDED) printf("node7: remote:  weight = %lu, aggregate = %lu \n", weight2[7]&weightMax, aggregate[7]);
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


int saena_object::change_aggregation(saena_matrix* A, std::vector<index_t>& aggregate, std::vector<index_t>& splitNew){

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    int i;

    MPI_Status status;
    MPI_File fh;
    MPI_Offset offset;

    std::string aggName = "/home/abaris/Dropbox/Projects/Saena/build/juliaAgg.bin";
    int mpiopen = MPI_File_open(comm, aggName.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if(mpiopen){
        if (rank==0) std::cout << "Unable to open the vector file!" << std::endl;
        MPI_Finalize();
        return -1;
    }

    // vector should have the following format: first line shows the value in row 0, second line shows the value in row 1
    offset = A->split[rank] * 8; // value(long=8)
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
    offset2 = A->split[rank] * 8; // value(long=8)
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

    unsigned long* splitNewTemp = (unsigned long*)malloc(sizeof(unsigned long)*nprocs);
    MPI_Allreduce(&splitNew[0], splitNewTemp, nprocs, MPI_UNSIGNED, MPI_SUM, comm);

    // do scan on splitNew
    splitNew[0] = 0;
    for(i=1; i<nprocs+1; i++)
        splitNew[i] = splitNew[i-1] + splitNewTemp[i-1];

//    for(i=0; i<nprocs+1; i++)
//        std::cout << splitNew[i] << std::endl;

    free(splitNewTemp);

    return 0;
}


int saena_object::aggregate_index_update(strength_matrix* S, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& aggArray, std::vector<index_t>& splitNew){
    // ************* update aggregates' indices *************
    // check each node to see if it is assigned to a local or remote node.
    // if it is local, then aggregate[i] will be to the root's new index,
    // and if it is remote, then it will be add to aggregateRemote to communicate the new index for its root.
    // **********************************************************

    MPI_Comm comm = S->comm;

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    unsigned long size = S->M;

    unsigned long procNum;
    std::vector<unsigned long> aggregateRemote;
    std::vector<unsigned int> recvProc;

    // ************* compute splitNew *************

    splitNew.assign(nprocs+1, 0);
    splitNew[rank] = aggArray.size();

    std::vector<index_t> splitNewTemp(nprocs);
    MPI_Allreduce(&splitNew[0], &splitNewTemp[0], nprocs, MPI_UNSIGNED, MPI_SUM, comm);

    // do scan on splitNew
    splitNew[0] = 0;
    for(unsigned long i = 1; i < nprocs+1; ++i)
        splitNew[i] = splitNew[i-1] + splitNewTemp[i-1];

//    print_vector(splitNew, 0, "splitNew", comm);
//    if(rank==0){
//        std::cout << "split and splitNew:" << std::endl;
//        for(i=0; i<nprocs+1; i++)
//            std::cout << S->split[i] << "\t" << splitNew[i] << std::endl;
//        std::cout << std::endl;}

    // local update
    // --------------
    std::vector<bool> isAggRemote(size);
    for(unsigned long i = 0; i < size; ++i){
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
        recvCount[procNum]++;
//        if(rank==0) std::cout << i << "\t" << procNum << std::endl;
    }

//    print_vector(recvCount, -1, "recvCount", comm);

    int recvSize = 0;
    int vIndexSize = 0;

    if(nprocs > 1) {
        std::vector<int> vIndexCount(nprocs);
        MPI_Alltoall(&recvCount[0], 1, MPI_INT, &vIndexCount[0], 1, MPI_INT, comm);

//        print_vector(vIndexCount, -1, "vIndexCount", comm);

        // this part is for isend and ireceive.
        std::vector<int> recvProcRank;
        std::vector<int> recvProcCount;
        std::vector<int> sendProcRank;
        std::vector<int> sendProcCount;
        int numRecvProc = 0;
        int numSendProc = 0;
        for (int i = 0; i < nprocs; i++) {
            if (recvCount[i] != 0) {
                numRecvProc++;
                recvProcRank.emplace_back(i);
                recvProcCount.emplace_back(recvCount[i]);
            }
            if (vIndexCount[i] != 0) {
                numSendProc++;
                sendProcRank.emplace_back(i);
                sendProcCount.emplace_back(vIndexCount[i]);
            }
        }

        std::vector<int> vdispls;
        std::vector<int> rdispls;
        vdispls.resize(nprocs);
        rdispls.resize(nprocs);
        vdispls[0] = 0;
        rdispls[0] = 0;

        for (int i = 1; i < nprocs; ++i) {
            vdispls[i] = vdispls[i - 1] + vIndexCount[i - 1];
            rdispls[i] = rdispls[i - 1] + recvCount[i - 1];
        }
        vIndexSize = vdispls[nprocs - 1] + vIndexCount[nprocs - 1];
        recvSize   = rdispls[nprocs - 1] + recvCount[nprocs - 1];

//        MPI_Barrier(comm); printf("rank %d: vIndexSize = %d, recvSize = %d \n", rank, vIndexSize, recvSize); MPI_Barrier(comm);

        std::vector<unsigned long> vIndex(vIndexSize);
        MPI_Alltoallv(&*aggregateRemote.begin(), &recvCount[0], &*rdispls.begin(), MPI_UNSIGNED_LONG, &vIndex[0],
                      &vIndexCount[0], &*vdispls.begin(), MPI_UNSIGNED_LONG, comm);
//        MPI_Alltoallv(&*aggregateRemote2.begin(), recvCount, &*rdispls.begin(), MPI_UNSIGNED_LONG, vIndex, vIndexCount, &*vdispls.begin(), MPI_UNSIGNED_LONG, comm);

        std::vector<unsigned long> aggSend(vIndexSize);
        std::vector<unsigned long> aggRecv(recvSize);

//        if(rank==0) std::cout << std::endl << "vSend:\trank:" << rank << std::endl;
        for (long i = 0; i < vIndexSize; i++) {
            aggSend[i] = aggregate[(vIndex[i] - S->split[rank])];
//            if(rank==0) std::cout << "vIndex = " << vIndex[i] << "\taggSend = " << aggSend[i] << std::endl;
        }

        // replace this alltoallv with isend and irecv.
//        MPI_Alltoallv(aggSend, vIndexCount, &*(vdispls.begin()), MPI_UNSIGNED_LONG, aggRecv, recvCount, &*(rdispls.begin()), MPI_UNSIGNED_LONG, comm);

        auto requests2 = new MPI_Request[numSendProc + numRecvProc];
        auto statuses2 = new MPI_Status[numSendProc + numRecvProc];

        for (int i = 0; i < numRecvProc; i++)
            MPI_Irecv(&aggRecv[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_UNSIGNED_LONG, recvProcRank[i], 1, comm,
                      &(requests2[i]));

        //Next send the messages. Do not send to self.
        for (int i = 0; i < numSendProc; i++)
            MPI_Isend(&aggSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_UNSIGNED_LONG, sendProcRank[i], 1, comm,
                      &(requests2[numRecvProc + i]));

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
        for (unsigned long i = 0; i < size; ++i) {
            if (isAggRemote[i]) {
                aggregate[i] = aggRecv[lower_bound2(&*aggregateRemote.begin(), &*aggregateRemote.end(), aggregate[i])];
//                if(rank==1) std::cout << i << "\t" << aggRecv[ lower_bound2(&*aggregateRemote.begin(), &*aggregateRemote.end(), aggregate[i]) ] << std::endl;
            }
        }

//        print_vector(aggregate, -1, "aggregate", comm);

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