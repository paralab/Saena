#include <cmath>
#include "superlu_ddefs.h"

#include "saena_object.h"
#include "saena_matrix.h"
#include "strength_matrix.h"
#include "prolong_matrix.h"
#include "restrict_matrix.h"
#include "grid.h"
#include "aux_functions.h"
#include "ietl_saena.h"
#include <parUtils.h>
//#include "El.hpp"
#include "dollar.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <set>
#include <random>
#include <mpi.h>
#include <superlu_defs.h>


saena_object::saena_object(){}


saena_object::~saena_object(){}


int saena_object::destroy(){
    return 0;
}


void saena_object::set_parameters(int vcycle_n, double relT, std::string sm, int preSm, int postSm){
//    maxLevel = l-1; // maxLevel does not include fine level. fine level is 0.
    vcycle_num = vcycle_n;
    relative_tolerance  = relT;
    smoother = sm;
    preSmooth = preSm;
    postSmooth = postSm;
}


int saena_object::setup(saena_matrix* A) { $
    int nprocs, rank, rank_new;
    MPI_Comm_size(A->comm, &nprocs);
    MPI_Comm_rank(A->comm, &rank);
    A->active_old_comm = true;

    #pragma omp parallel
    if(rank==0 && omp_get_thread_num()==0)
        printf("\nnumber of processes = %d, number of threads = %d\n\n", nprocs, omp_get_num_threads());

    int i;
//    float row_reduction_min;
//    float total_row_reduction;
//    index_t M_current;

    if(verbose_setup)
        if(rank==0){
            printf("_____________________________\n\n");
            printf("level = 0 \nnumber of procs = %d \nmatrix size \t= %d \nnonzero \t= %lu \ndensity \t= %.6f \n",
                   nprocs, A->Mbig, A->nnz_g, A->density);}

    if(verbose_setup_steps && rank==0) printf("setup: find_eig()\n");
    if(smoother=="chebyshev"){
//        double t1 = omp_get_wtime();
        find_eig(*A);
//        double t2 = omp_get_wtime();
//        if(verbose_level_setup) print_time(t1, t2, "find_eig() level 0: ", A->comm);
    }

    if(verbose_setup_steps && rank==0) printf("setup: generate_dense_matrix()\n");
    A->switch_to_dense = switch_to_dense;
    A->dense_threshold = dense_threshold;
    if(switch_to_dense && A->density > dense_threshold)
        A->generate_dense_matrix();

    if(verbose_setup_steps && rank==0) printf("setup: first Grid\n");
    grids.resize(max_level+1);
    grids[0] = Grid(A, max_level, 0); // pass A to grids[0]

    if(verbose_setup_steps && rank==0) printf("setup: other Grids\n");
    for(i = 0; i < max_level; i++){
//        MPI_Barrier(grids[0].A->comm); printf("rank = %d, level setup; before if\n", rank); MPI_Barrier(grids[0].A->comm);
        if(grids[i].A->active) {
            if (shrink_level_vector.size()>i+1) if(shrink_level_vector[i+1]) grids[i].A->enable_shrink_next_level = true;
            if (shrink_values_vector.size()>i+1) grids[i].A->cpu_shrink_thre2_next_level = shrink_values_vector[i+1];
            level_setup(&grids[i]); // create P, R and Ac for grid[i]
            grids[i + 1] = Grid(&grids[i].Ac, max_level, i + 1); // Pass A to grids[i+1] (created as Ac in grids[i])
            grids[i].coarseGrid = &grids[i + 1]; // connect grids[i+1] to grids[i]

            if(grids[i].Ac.active) {
                if (smoother == "chebyshev") {
//                    printf("find_eig() start!\n");
//                    double t1 = omp_get_wtime();
                    find_eig(grids[i].Ac);
//                    double t2 = omp_get_wtime();
//                    if(verbose_level_setup) print_time(t1, t2, "find_eig(): ", A->comm);
                }
            }

            if (verbose_setup) {
                if (rank == 0) {
//                    MPI_Comm_size(grids[i].Ac.comm, &nprocs);
                    printf("_____________________________\n\n");
                    printf("level = %d \nnumber of procs = %d \nmatrix size \t= %d \nnonzero \t= %lu \ndensity \t= %.6f \n",
                           grids[i + 1].currentLevel, grids[i + 1].A->total_active_procs, grids[i + 1].A->Mbig, grids[i + 1].A->nnz_g,
                           grids[i + 1].A->density);
                }
            }

            // decide if next level for multigrid is required or not.
            // threshold to set maximum multigrid level
            if(dynamic_levels){
//                MPI_Allreduce(&grids[i].Ac.M, &M_current, 1, MPI_UNSIGNED, MPI_MIN, grids[i].Ac.comm);
//                total_row_reduction = (float) grids[0].A->Mbig / grids[i].Ac.Mbig;
                grids[i+1].row_reduction_min = (float) grids[i].Ac.Mbig / grids[i].A->Mbig;

//                if(rank==0) printf("row_reduction_min = %f, total_row_reduction = %f\n", row_reduction_min, total_row_reduction);
//                if(rank==0) if(row_reduction_min < 0.1) printf("\nWarning: Coarsening is too aggressive! Increase connStrength in saena_object.h\n");
//                row_reduction_local = (float) grids[i].Ac.M / grids[i].A->M;
//                MPI_Allreduce(&row_reduction_local, &row_reduction_min, 1, MPI_FLOAT, MPI_MIN, grids[i].Ac.comm);
//                if(rank==0) printf("row_reduction_min = %f, row_reduction_threshold = %f \n", row_reduction_min, row_reduction_threshold);
//                if(rank==0) printf("grids[i].Ac.Mbig = %d, grids[0].A->Mbig = %d, inequality = %d \n", grids[i].Ac.Mbig, grids[0].A->Mbig, (grids[i].Ac.Mbig*1000 < grids[0].A->Mbig));

                if ( (grids[i].Ac.Mbig < least_row_threshold) || (grids[i+1].row_reduction_min > row_reduction_threshold) ) {
                    max_level = grids[i].currentLevel + 1;
//                    grids.resize(max_level);
                }
            }
        }
        if(!grids[i].Ac.active)
            break;
    }

    // max_level is the lowest on the active processors in the last grid. So MPI_MIN is used in the following MPI_Allreduce.
    int max_level_send = max_level;
    MPI_Allreduce(&max_level_send, &max_level, 1, MPI_INT, MPI_MIN, grids[0].A->comm);
    grids.resize(max_level);
//    printf("rank = %d, max_level = %d\n", rank, max_level);
//    printf("i = %u, max_level = %u \n", i, max_level);

    // grids[i+1].row_reduction_min is 0 by default. for the active processors in the last grid, it will be non-zero.
    // that's why MPI_MAX is used in the following MPI_Allreduce.
    float row_reduction_min_send = grids[i].row_reduction_min;
    MPI_Allreduce(&row_reduction_min_send, &grids[i].row_reduction_min, 1, MPI_FLOAT, MPI_MAX, grids[0].A->comm);
    // delete the coarsest level, if the size is not reduced much.
    if (grids[i].row_reduction_min > row_reduction_threshold) {
        grids.pop_back();
        max_level--;
        // todo: when destroy() is written, delete P and R by that.
//        grids[i].P.destroy();
//        grids[i].R.destroy();
    }

    if(verbose_setup && rank==0){
        printf("_____________________________\n\n");
        printf("number of levels = << %d >> (the finest level is 0)\n", max_level);
        printf("\n******************************************************\n");
    }

//    MPI_Barrier(grids[0].A->comm); printf("rank %d: setup done!\n", rank); MPI_Barrier(grids[0].A->comm);

    return 0;
}


int saena_object::level_setup(Grid* grid){$

    int nprocs, rank;
    MPI_Comm_size(grid->A->comm, &nprocs);
    MPI_Comm_rank(grid->A->comm, &rank);

//    if(verbose_level_setup){
//        MPI_Barrier(grid->A->comm);
//        printf("rank = %d, start of level_setup: level = %d \n", rank, grid->currentLevel);
//        MPI_Barrier(grid->A->comm);
//    }

//    grid->A->print_info(-1);

    // **************************** find_aggregation ****************************

    std::vector<unsigned long> aggregate(grid->A->M);
    double t1 = omp_get_wtime();
    find_aggregation(grid->A, aggregate, grid->P.splitNew);
    double t2 = omp_get_wtime();
    if(verbose_level_setup) print_time(t1, t2, "Aggregation: level "+std::to_string(grid->currentLevel), grid->A->comm);

//    MPI_Barrier(grid->A->comm); printf("rank %d: here after find_aggregation!!! \n", rank); MPI_Barrier(grid->A->comm);
//    print_vector(aggregate, -1, "aggregate", grid->A->comm);

    // **************************** changeAggregation ****************************

    // use this to read aggregation from file and replace the aggregation_2_dist computed here.
//    changeAggregation(grid->A, aggregate, grid->P.splitNew, grid->A->comm);

    // **************************** create_prolongation ****************************

    t1 = omp_get_wtime();
    create_prolongation(grid->A, aggregate, &grid->P);
    t2 = omp_get_wtime();
    if(verbose_level_setup) print_time(t1, t2, "Prolongation: level "+std::to_string(grid->currentLevel), grid->A->comm);

//    MPI_Barrier(grid->A->comm); printf("rank %d: here after create_prolongation!!! \n", rank); MPI_Barrier(grid->A->comm);
//    print_vector(grid->P.split, 0, "grid->P.split", grid->A->comm);
//    print_vector(grid->P.splitNew, 0, "grid->P.splitNew", grid->A->comm);
//    grid->P.print_info(-1);
//    grid->P.print_entry(-1);

    // **************************** restriction ****************************

    t1 = omp_get_wtime();
    grid->R.transposeP(&grid->P);
    t2 = omp_get_wtime();
    if(verbose_level_setup) print_time(t1, t2, "Restriction: level "+std::to_string(grid->currentLevel), grid->A->comm);

//    MPI_Barrier(grid->A->comm); printf("rank %d: here after transposeP!!! \n", rank); MPI_Barrier(grid->A->comm);
//    grid->R.print_info(-1);
//    grid->R.print_entry(-1);
//    print_vector(grid->R.entry_local, -1, "grid->R.entry_local", grid->A->comm);
//    print_vector(grid->R.entry_remote, -1, "grid->R.entry_remote", grid->A->comm);

    // **************************** coarsen ****************************

    t1 = omp_get_wtime();
    coarsen(grid);
//    coarsen_old(grid);
    t2 = omp_get_wtime();
    if(verbose_level_setup) print_time(t1, t2, "Coarsening: level "+std::to_string(grid->currentLevel), grid->A->comm);

//    MPI_Barrier(grid->A->comm); printf("rank %d: here after coarsen!!! \n", rank); MPI_Barrier(grid->A->comm);
//    if(grid->Ac.active) print_vector(grid->Ac.split, 1, "grid->Ac.split", grid->Ac.comm);
//    if(grid->Ac.active) print_vector(grid->Ac.entry, 1, "grid->Ac.entry", grid->A->comm);

//    printf("rank = %d, M = %u, nnz_l = %lu, nnz_g = %lu, Ac.M = %u, Ac.nnz_l = %lu \n",
//           rank, grid->A->M, grid->A->nnz_l, grid->A->nnz_g, grid->Ac.M, grid->Ac.nnz_l);

    return 0;
}


int saena_object::find_aggregation(saena_matrix* A, std::vector<unsigned long>& aggregate, std::vector<index_t>& splitNew){$
    // finding aggregation is written in an adaptive way. An aggregation_2_dist is being created first. If it is too small,
    // or too big it will be recreated until an aggregation_2_dist with size within the acceptable range is produced.

    MPI_Comm comm = A->comm;
    int rank;
    MPI_Comm_rank(comm, &rank);
//    MPI_Comm_size(A->comm, &nprocs);

    strength_matrix S;
    create_strength_matrix(A, &S);
//    S.print_entry(-1);

    float connStrength_temp = connStrength;
    std::vector<unsigned long> aggArray; // vector of root nodes.
    bool continue_agg = true;

//    aggregation_2_dist(&S, aggregate, aggArray);

    // new_size is the size of the new coarse matrix.
    unsigned int new_size_local, new_size=0;
    double division = 0;
    while(continue_agg){
        aggregation_1_dist(&S, aggregate, aggArray);
//        aggregation_2_dist(&S, aggregate, aggArray);
        continue_agg = false;

        new_size_local = aggArray.size();
        MPI_Allreduce(&new_size_local, &new_size, 1, MPI_UNSIGNED, MPI_SUM, comm);
        division = (double)A->Mbig / new_size;
        if(rank==0) printf("\nconnStrength = %.2f \ncurrent size = %u \nnew size     = %u \ndivision     = %.2f\n",
                           connStrength_temp, A->Mbig, new_size, division);

        if( division > 8 ){
            connStrength_temp += 0.05;
            if(connStrength_temp > 0.95)
                continue_agg = false;
            else{
                aggArray.clear();
                continue_agg = true;
                S.erase_update();
                S.setup_matrix(connStrength_temp);
//                create_strength_matrix(A, &S);
            }
        } else if( division < 1.5 ){
            connStrength_temp -= 0.05;
            if(connStrength_temp < 0.2)
                continue_agg = false;
            else{
                aggArray.clear();
                continue_agg = true;
                S.erase_update();
                S.setup_matrix(connStrength_temp);
//                create_strength_matrix(A, &S);
            }
        }
        if(!adaptive_coarsening)
            continue_agg = false;
    }

//    if(rank==0) printf("\nfinal: connStrength = %f, current size = %u, new size = %u,  division = %d\n",
//                       connStrength_temp, A->Mbig, new_size, division);

//    connStrength = connStrength_temp;
    aggregate_index_update(&S, aggregate, aggArray, splitNew);
//    updateAggregation(aggregate, &aggSize);

//    print_vector(aggArray, -1, "aggArray", comm);

    return 0;
} // end of SaenaObject::findAggregation


int saena_object::create_strength_matrix(saena_matrix* A, strength_matrix* S){$

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
//                STi.push_back(iter2); // iter2 is actually i, but it was giving an error for using i.
//                STj.push_back(A->col_local[A->indicesP_local[iter]]);
//                STval.push_back(1);
//                continue;
//            }
//
//            STi.push_back(iter2); // iter2 is actually i, but it was giving an error for using i.
//            STj.push_back(A->col_local[A->indicesP_local[iter]]);
//            STval.push_back( -A->values_local[A->indicesP_local[iter]] / maxPerRow[A->col_local[A->indicesP_local[iter]]] );
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
//            STi.push_back(A->row_remote[A->indicesP_remote[iter]]);
//            STj.push_back(A->col_remote2[A->indicesP_remote[iter]]);
//            STval.push_back( -A->values_remote[A->indicesP_remote[iter]] / A->vecValues[A->col_remote[A->indicesP_remote[iter]]] );
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
                                     std::vector<unsigned long> &aggArray) {$

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

//    randomVector(initialWeight, S->Mbig, S, comm);
    randomVector3(initialWeight, S->Mbig, S, comm);
//    randomVector4(initialWeight, S->Mbig);

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
            aggArray.push_back(aggregate[i]);
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
                    aggArray.push_back(aggregate[i]);
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
//        aggArray.push_back(0+S->split[rank]);
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

//    randomVector(initialWeight, S->Mbig, S, comm);
    randomVector3(initialWeight, S->Mbig, S, comm);
//    randomVector4(initialWeight, S->Mbig);

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
            aggArray.push_back(aggregate[i]);
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
                    aggArray.push_back(aggregate[i]);
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
//        aggArray.push_back(0+S->split[rank]);
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

    unsigned long i;
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
    for(i=1; i<nprocs+1; i++)
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
    for(i=0; i<size; i++){
        if(aggregate[i] >= S->split[rank] && aggregate[i] < S->split[rank+1]){
            aggregate[i] = lower_bound2(&*aggArray.begin(), &*aggArray.end(), aggregate[i]) + splitNew[rank];
//            if(rank==1) std::cout << aggregate[i] << std::endl;
            isAggRemote[i] = false;
        }else{
            isAggRemote[i] = true;
            aggregateRemote.push_back(aggregate[i]);
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
                recvProcRank.push_back(i);
                recvProcCount.push_back(recvCount[i]);
            }
            if (vIndexCount[i] != 0) {
                numSendProc++;
                sendProcRank.push_back(i);
                sendProcCount.push_back(vIndexCount[i]);
            }
        }

        std::vector<int> vdispls;
        std::vector<int> rdispls;
        vdispls.resize(nprocs);
        rdispls.resize(nprocs);
        vdispls[0] = 0;
        rdispls[0] = 0;

        for (int i = 1; i < nprocs; i++) {
            vdispls[i] = vdispls[i - 1] + vIndexCount[i - 1];
            rdispls[i] = rdispls[i - 1] + recvCount[i - 1];
        }
        vIndexSize = vdispls[nprocs - 1] + vIndexCount[nprocs - 1];
        recvSize = rdispls[nprocs - 1] + recvCount[nprocs - 1];

//    MPI_Barrier(comm); printf("rank %d: vIndexSize = %d, recvSize = %d \n", rank, vIndexSize, recvSize); MPI_Barrier(comm);

        std::vector<unsigned long> vIndex(vIndexSize);
        MPI_Alltoallv(&*aggregateRemote.begin(), &recvCount[0], &*rdispls.begin(), MPI_UNSIGNED_LONG, &vIndex[0],
                      &vIndexCount[0], &*vdispls.begin(), MPI_UNSIGNED_LONG, comm);
//    MPI_Alltoallv(&*aggregateRemote2.begin(), recvCount, &*rdispls.begin(), MPI_UNSIGNED_LONG, vIndex, vIndexCount, &*vdispls.begin(), MPI_UNSIGNED_LONG, comm);

        std::vector<unsigned long> aggSend(vIndexSize);
        std::vector<unsigned long> aggRecv(recvSize);

//    if(rank==0) std::cout << std::endl << "vSend:\trank:" << rank << std::endl;
        for (long i = 0; i < vIndexSize; i++) {
            aggSend[i] = aggregate[(vIndex[i] - S->split[rank])];
//        if(rank==0) std::cout << "vIndex = " << vIndex[i] << "\taggSend = " << aggSend[i] << std::endl;
        }

        // replace this alltoallv with isend and irecv.
//    MPI_Alltoallv(aggSend, vIndexCount, &*(vdispls.begin()), MPI_UNSIGNED_LONG, aggRecv, recvCount, &*(rdispls.begin()), MPI_UNSIGNED_LONG, comm);

        MPI_Request *requests2 = new MPI_Request[numSendProc + numRecvProc];
        MPI_Status *statuses2 = new MPI_Status[numSendProc + numRecvProc];

        for (int i = 0; i < numRecvProc; i++)
            MPI_Irecv(&aggRecv[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_UNSIGNED_LONG, recvProcRank[i], 1, comm,
                      &(requests2[i]));

        //Next send the messages. Do not send to self.
        for (int i = 0; i < numSendProc; i++)
            MPI_Isend(&aggSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_UNSIGNED_LONG, sendProcRank[i], 1, comm,
                      &(requests2[numRecvProc + i]));

        MPI_Waitall(numRecvProc, requests2, statuses2);

//    if(rank==1) std::cout << "aggRemote received:" << std::endl;
//    set<unsigned long>::iterator it;
//    for(i=0; i<size; i++){
//        if(isAggRemote[i]){
//            it = aggregateRemote2.find(aggregate[i]);
//            if(rank==1) std::cout << aggRecv[ distance(aggregateRemote2.begin(), it) ] << std::endl;
//            aggregate[i] = aggRecv[ distance(aggregateRemote2.begin(), it) ];
//        }
//    }

        // remote
        for (i = 0; i < size; i++) {
            if (isAggRemote[i]) {
                aggregate[i] = aggRecv[lower_bound2(&*aggregateRemote.begin(), &*aggregateRemote.end(), aggregate[i])];
//            if(rank==1) std::cout << i << "\t" << aggRecv[ lower_bound2(&*aggregateRemote.begin(), &*aggregateRemote.end(), aggregate[i]) ] << std::endl;
            }
        }

//    print_vector(aggregate, -1, "aggregate", comm);

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


int saena_object::create_prolongation(saena_matrix* A, std::vector<unsigned long>& aggregate, prolong_matrix* P){$
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
                PEntryTemp.push_back(cooEntry(A->row_local[A->indicesP_local[iter]],
                                              aggregate[ A->col_local[A->indicesP_local[iter]] - A->split[rank] ],
                                              1 - omega));
            }else{
                PEntryTemp.push_back(cooEntry(A->row_local[A->indicesP_local[iter]],
                                              aggregate[ A->col_local[A->indicesP_local[iter]] - A->split[rank] ],
                                              -omega * A->values_local[A->indicesP_local[iter]] * A->inv_diag[A->row_local[A->indicesP_local[iter]]]));
            }
//            std::cout << A->row_local[A->indicesP_local[iter]] << "\t" << aggregate[A->col_local[A->indicesP_local[iter]] - A->split[rank]] << "\t" << A->values_local[A->indicesP_local[iter]] * A->inv_diag[A->row_local[A->indicesP_local[iter]]] << std::endl;
        }
    }

    MPI_Waitall(A->numRecvProc, requests, statuses);

    // remote
    // ------
    iter = 0;
    for (index_t i = 0; i < A->col_remote_size; ++i) {
        for (index_t j = 0; j < A->nnzPerCol_remote[i]; ++j, ++iter) {
            PEntryTemp.push_back(cooEntry(A->row_remote[iter],
                                          A->vecValuesULong[A->col_remote[iter]],
                                          -omega * A->values_remote[iter] * A->inv_diag[A->row_remote[iter]]));
//            P->values.push_back(A->values_remote[iter]);
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
        P->entry.push_back(PEntryTemp[i]);
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


int saena_object::fast_mm(cooEntry *A, cooEntry *B, std::vector<cooEntry> &C, nnz_t A_nnz, nnz_t B_nnz,
                          index_t A_row_size, index_t A_row_offset, index_t A_col_size, index_t A_col_offset,
                          index_t B_row_offset, index_t B_col_size, index_t B_col_offset,
                          index_t *nnzPerColScan_leftStart, index_t *nnzPerColScan_leftEnd,
                          index_t *nnzPerColScan_rightStart, index_t *nnzPerColScan_rightEnd, MPI_Comm comm){$
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
    if(rank==0 && verbose_matmat) printf("\nfast_mm: start \n");

    if(A_nnz == 0 || B_nnz == 0){
        printf("\nskip: A_nnz == 0 || B_nnz == 0\n");
        return 0;
    }

//    print_vector(A, -1, "A", comm);
//    print_vector(B, -1, "B", comm);
//    MPI_Barrier(comm); printf("rank %d: A: %ux%u, B: %ux%u \n\n", rank, A_row_size, A_col_size, A_col_size, B_col_size); MPI_Barrier(comm);
//    MPI_Barrier(comm); printf("rank %d: A_row_size = %u, A_row_offset = %u, A_col_size = %u, A_col_offset = %u, B_row_offset = %u, B_col_size = %u, B_col_offset = %u \n\n",
//            rank, A_row_size, A_row_offset, A_col_size, A_col_offset, B_row_offset, B_col_size, B_col_offset);

    MPI_Barrier(comm);
    if(rank==0){

        if(verbose_fastmm_A){
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

        if(verbose_fastmm_B) {
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
    MPI_Barrier(comm);

    index_t size_min = std::min(std::min(A_row_size, A_col_size), B_col_size);

    index_t r_dense = 10, c_dense = 10; //default 100. todo: fix this.
    if( (A_row_size < r_dense && A_col_size < c_dense) || size_min < 4 ){ //default 30. todo: fix this.

        if(rank==0 && verbose_matmat){printf("fast_mm: case 1: start \n");}
/*
        std::vector<unsigned int> AnnzPerCol(A_col_size, 0);
//        unsigned int *AnnzPerCol_p = &AnnzPerCol[0] - A[0].col;
        for(nnz_t i = 0; i < A.size(); i++){
//            if(rank==0) printf("A[i].col = %u, \tA_col_size = %u \n", A[i].col - A_col_offset, A_col_size);
            AnnzPerCol[A[i].col - A_col_offset]++;
        }

        std::vector<nnz_t> AnnzPerColScan(A_col_size+1);
        AnnzPerColScan[0] = 0;
        for(nnz_t i = 0; i < A_col_size; i++){
            AnnzPerColScan[i+1] = AnnzPerColScan[i] + AnnzPerCol[i];
        }

//        print_vector(AnnzPerColScan, -1, "AnnzPerColScan", comm);
*/
        // initialize
        std::vector<cooEntry> C_temp(A_row_size * B_col_size); // 1D array is better than 2D for many reasons.
        for(nnz_t i = 0; i < A_row_size * B_col_size; i++){
            C_temp[i] = cooEntry(0, 0, 0);
        }

        if(rank==0 && verbose_matmat) {printf("fast_mm: case 1: step 1 \n");}

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

                    C_temp[C_index] = cooEntry(A[i].row, B[k].col, B[k].val * A[i].val + C_temp[C_index].val);

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

        if(rank==0 && verbose_matmat) {printf("fast_mm: case 1: step 2 \n");}

        // add the new elements to C
        // add the entries in column-major order
        for(nnz_t j = 0; j < B_col_size; j++) {
            for(nnz_t i = 0; i < A_row_size; i++) {
//                if(rank==0) std::cout << i + A_row_size*j << "\t" << C_temp[i + A_row_size*j] << std::endl;
                if (C_temp[i + A_row_size*j].val != 0) {
                    C.emplace_back(C_temp[i + A_row_size*j]);
                }
            }
        }

//        print_vector(C, -1, "C", comm);

        if(rank==0 && verbose_matmat) printf("fast_mm: case 1: end \n");

    } else if(A_row_size <= A_col_size) { //todo: fix this.

        if(rank==0 && verbose_matmat) {printf("fast_mm: case 2: start \n");}

        // prepare splits of matrix A by column
        nnz_t A1_nnz = 0, A2_nnz;

//        for(nnz_t i = 0; i < A_nnz; i++){
//            if(A[i].col-A_col_offset < A_col_size/2){
//                A1_nnz++;
//            }
//        }
        for(nnz_t i = 0; i < A_col_size; i++){
            for(nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftEnd[i]; j++) {
                if(A[j].col-A_col_offset < A_col_size/2) {
                    A1_nnz++;
                }
            }
        }
        A2_nnz = A_nnz - A1_nnz;

        // prepare splits of matrix B by row
        nnz_t B1_nnz = 0, B2_nnz;

        std::vector<index_t> nnzPerCol_middle(B_col_size, 0);
//        for(nnz_t i = 0; i < B_nnz; i++){
//            if(rank==1) std::cout << B[i] << "\t" << A_col_size/2 << "\t" << B_row_offset << "\t" << B_col_offset << std::endl;
//            if(B[i].row < A_col_size/2){ // A_col_size/2 is middle row of B too.
//                nnzPerCol_middle[B[i].col - B_col_offset]++;
//                B1_nnz++;
//            }
//        }
        for(nnz_t i = 0; i < B_col_size; i++){
            for(nnz_t j = nnzPerColScan_rightStart[i]; j < nnzPerColScan_rightEnd[i]; j++) {
                if(B[j].row < A_col_size/2){ // A_col_size/2 is middle row of B too.
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

//        print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);
//        if(rank==0) printf("rank %d: A_nnz = %lu, A1_nnz = %lu, A2_nnz = %lu, B_nnz = %lu, B1_nnz = %lu, B2_nnz = %lu \n",
//                rank, A_nnz, A1_nnz, A2_nnz, B_nnz, B1_nnz, B2_nnz);

        if(rank==0 && verbose_matmat) {printf("fast_mm: case 2: step 1 \n");}

        for(nnz_t i = 0; i < B_col_size; i++){
            nnzPerColScan_middle[i] = nnzPerColScan_middle[i+1] + nnzPerColScan_rightStart[i] - nnzPerColScan_middle[i];
//            if(rank==0) printf("nnzPerColScan_middle[%lu] = %u, \tnnzPerColScan_middle[%lu] = %u, \tnnzPerColScan_rightStart = %u \n",
//                    i, nnzPerColScan_middle[i], i+1, nnzPerColScan_middle[i+1], nnzPerColScan_rightStart[i]);
        }

//        print_vector(nnzPerColScan_middle, -1, "nnzPerColScan_middle", comm);

        if(rank==0 && verbose_matmat) {printf("fast_mm: case 2: step 2 \n");}

        // A1: start: nnzPerColScan_leftStart,               end: nnzPerColScan_leftEnd
        // A2: start: nnzPerColScan_leftStart[A_col_size/2], end: nnzPerColScan_leftEnd[A_col_size/2]
        // B1: start: nnzPerColScan_rightStart,              end: nnzPerColScan_middle
        // B2: start: nnzPerColScan_middle,                  end: nnzPerColScan_rightEnd

        MPI_Barrier(comm);
        if(rank==0){

            if(verbose_fastmm_A) {
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

            if(verbose_fastmm_B) {
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

        std::vector<cooEntry> C1, C2;

        // C1 = A1 * B1
        if(rank==0 && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 1 \n");
        fast_mm(&A[0], &B[0], C1, A1_nnz, B1_nnz,
                A_row_size, A_row_offset, A_col_size/2, A_col_offset,
                B_row_offset, B_col_size, B_col_offset,
                nnzPerColScan_leftStart,  nnzPerColScan_leftEnd, // A1
                nnzPerColScan_rightStart, &nnzPerColScan_middle[0], comm); // B1
//        fast_mm(&A[0], &B[0], C1, A1_nnz, B1_nnz,
//                A_row_size/2, A_row_offset, A_col_size, A_col_offset,
//                B_row_offset, B_col_size/2, B_col_offset,
//                nnzPerColScan_leftStart,  &nnzPerColScan_leftStart[1], // A1
//                nnzPerColScan_rightStart, &nnzPerColScan_middle[0], comm); // B1

        // C2 = A2 * B2
        if(rank==0 && verbose_matmat_recursive) printf("fast_mm: case 2: recursive 2 \n");
        fast_mm(&A[0], &B[0], C2, A2_nnz, B2_nnz,
                A_row_size, A_row_offset, A_col_size-A_col_size/2, A_col_offset+A_col_size/2,
                B_row_offset+A_col_size/2, B_col_size, B_col_offset,
                &nnzPerColScan_leftStart[A_col_size/2], &nnzPerColScan_leftEnd[A_col_size/2], // A2
                &nnzPerColScan_middle[0], nnzPerColScan_rightEnd, comm); // B2
//        fast_mm(&A[0], &B[B_col_offset/2], C2, A2_nnz, B2_nnz,
//                A_row_size-A_row_size/2, A_row_offset+A_row_size/2, A_col_size, A_col_offset,
//                B_row_offset, B_col_size-B_col_size/2, B_col_offset+B_col_size/2,
//                &nnzPerColScan_leftStart[A_col_size/2 + 1], &nnzPerColScan_leftStart[A_col_size/2 + 2], // A2
//                &nnzPerColScan_middle[0], &nnzPerColScan_rightStart[1], // B2
//                comm);

//        print_vector(C1, -1, "C1", comm);
//        print_vector(C2, -1, "C2", comm);

        if(rank==0 && verbose_matmat) {printf("fast_mm: case 2: step 3 \n");}
//        if(rank==0 && verbose_matmat) printf("C1.size() = %lu, C2.size() = %lu \n", C1.size(), C2.size());

        // take care of the special cases when either C1 or C2 is empty.
//        if(C1.empty()){
//            C = C2;
//            return 0;
//        } else if(C2.empty()){
//            C = C1;
//            return 0;
//        }
        nnz_t i=0;
        if(C1.empty()){
            while(i < C2.size()){
                C.emplace_back(C2[i]);
                i++;
            }
            if(rank==0 && verbose_matmat) printf("fast_mm: end \n\n");
            return 0;
        }

        if(C2.empty()) {
            while (i < C1.size()) {
                C.emplace_back(C1[i]);
                i++;
            }
            if(rank==0 && verbose_matmat) printf("fast_mm: end \n\n");
            return 0;
        }

        if(rank==0 && verbose_matmat) {printf("fast_mm: case 2: step 4 \n");}

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
                if(rank==0 && verbose_matmat) printf("fast_mm: end \n\n");
                return 0;
            }else if(j == C2.size()) {
                while (i < C1.size()) {
                    C.emplace_back(C1[i]);
                    i++;
                }
                if(rank==0 && verbose_matmat) printf("fast_mm: end \n\n");
                return 0;
            }
        }

        if(rank==0 && verbose_matmat) printf("fast_mm: case 2: end \n");

    } else { // A_row_size > A_col_size

        if(rank==0 && verbose_matmat) printf("fast_mm: case 3: start \n");

        // prepare splits of matrix B by column
        nnz_t B1_nnz = 0, B2_nnz;

//        for(nnz_t i = 0; i < B_nnz; i++){
//            if(B[i].col < B_col_size/2){
//                B1_nnz++;
//            }
//        }
        for(nnz_t i = 0; i < B_col_size; i++){
            for(nnz_t j = nnzPerColScan_rightStart[i]; j < nnzPerColScan_rightEnd[i]; j++) {
                if(B[j].col-B_col_offset < B_col_size/2){
                    B1_nnz++;
                }
            }
        }
        B2_nnz = B_nnz - B1_nnz;

        if(rank==0 && verbose_matmat) printf("fast_mm: case 3: step 1 \n");

        // prepare splits of matrix A by row
        nnz_t A1_nnz = 0, A2_nnz;

        std::vector<index_t> nnzPerCol_middle(A_col_size, 0);
//        for(nnz_t i = 0; i < A_nnz; i++){
//            if(A[i].row < A_row_size/2){
//                nnzPerCol_middle[A[i].col - A_col_offset]++;
//                A1_nnz++;
//            }
//        }
        for(nnz_t i = 0; i < A_col_size; i++){
            for(nnz_t j = nnzPerColScan_leftStart[i]; j < nnzPerColScan_leftEnd[i]; j++) {
                if(A[j].row < A_row_size/2){
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

        if(rank==0 && verbose_matmat) printf("fast_mm: case 3: step 2 \n");

        for(nnz_t i = 0; i < A_col_size; i++){
            nnzPerColScan_middle[i] = nnzPerColScan_middle[i+1] + nnzPerColScan_leftStart[i] - nnzPerColScan_middle[i];
//            if(rank==0) printf("nnzPerColScan_middle[%lu] = %u \n", i, nnzPerColScan_middle[i]);
        }

        if(rank==0 && verbose_matmat) printf("fast_mm: case 3: step 3 \n");

        // A1: start: nnzPerColScan_leftStart,                end: nnzPerColScan_middle
        // A2: start: nnzPerColScan_middle,                   end: nnzPerColScan_leftEnd
        // B1: start: nnzPerColScan_rightStart,               end: nnzPerColScan_rightEnd
        // B2: start: nnzPerColScan_rightStart[B_col_size/2], end: nnzPerColScan_rightEnd[B_col_size/2]


        MPI_Barrier(comm);
        if(rank==0){

            if(verbose_fastmm_A) {
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

            if(verbose_fastmm_B) {
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
        MPI_Barrier(comm);

        std::vector<cooEntry> C_temp;

        // C1 = A1 * B1
        if(rank==0 && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 1 \n");
        fast_mm(&A[0], &B[0], C_temp, A1_nnz, B1_nnz,
                A_row_size/2, A_row_offset, A_col_size, A_col_offset,
                B_row_offset, B_col_size/2, B_col_offset,
                nnzPerColScan_leftStart,  &nnzPerColScan_middle[0], // A1
                nnzPerColScan_rightStart, nnzPerColScan_rightEnd, comm); // B1

        // C2 = A2 * B1:
        if(rank==0 && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 2 \n");
        fast_mm(&A[0], &B[0], C_temp, A2_nnz, B1_nnz,
                A_row_size-A_row_size/2, A_row_offset+A_row_size/2, A_col_size, A_col_offset,
                B_row_offset, B_col_size/2, B_col_offset,
                &nnzPerColScan_middle[0], nnzPerColScan_leftEnd, // A2
                nnzPerColScan_rightStart, nnzPerColScan_rightEnd, comm); // B1

        // C3 = A1 * B2:
        if(rank==0 && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 3 \n");
        fast_mm(&A[0], &B[0], C_temp, A1_nnz, B2_nnz,
                A_row_size/2, A_row_offset, A_col_size, A_col_offset,
                B_row_offset, B_col_size-B_col_size/2, B_col_offset+B_col_size/2,
                nnzPerColScan_leftStart,  &nnzPerColScan_middle[0], // A1
                &nnzPerColScan_rightStart[B_col_size/2], &nnzPerColScan_rightEnd[B_col_size/2], comm); // B2

        // C4 = A2 * B2
        if(rank==0 && verbose_matmat_recursive) printf("fast_mm: case 3: recursive 4 \n");
        fast_mm(&A[0], &B[0], C_temp, A2_nnz, B2_nnz,
                A_row_size-A_row_size/2, A_row_offset+A_row_size/2, A_col_size, A_col_offset,
                B_row_offset, B_col_size-B_col_size/2, B_col_offset+B_col_size/2,
                &nnzPerColScan_middle[0], nnzPerColScan_leftEnd, // A2
                &nnzPerColScan_rightStart[B_col_size/2], &nnzPerColScan_rightEnd[B_col_size/2], comm); // B2

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

        if(rank==0 && verbose_matmat) printf("fast_mm: case 3: end \n");
    }

    if(rank==0 && verbose_matmat) printf("fast_mm: end \n\n");

    return 0;
}


int saena_object::coarsen(Grid *grid) {$

    // Output: Ac = R * A * P
    // Steps:
    // 1- Compute AP = A * P. To do that use the transpose of R_i, instead of P. Pass all R_j's to all the processors,
    //    Then, multiply local A_i by R_jon each process.
    // 2- Compute RAP = R * AP. Use transpose of P_i instead of R. It is done locally. So multiply P_i * (AP)_i.
    // 3- Sort and remove local duplicates.
    // 4- Do a parallel sort based on row-major order. A modified version of par::sampleSort from usort is used here.
    //    Again, remove duplicates.
    // 5- Not complete yet: Sparsify Ac.

    saena_matrix *A = grid->A;
    prolong_matrix *P = &grid->P;
    restrict_matrix *R = &grid->R;
    saena_matrix *Ac = &grid->Ac;

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

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 2: rank = %d\n", rank); MPI_Barrier(comm);}

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

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 3: rank = %d\n", rank); MPI_Barrier(comm);}

    // *******************************************************
    // multiply: AP = A_i * P_j. in which P_j = R_j_tranpose and 0 <= j < nprocs.
    // *******************************************************
    // this is an overlapped version of this multiplication, similar to dense_matvec.

    // local transpose of R is being used to compute A*P. So R is transposed locally here.
    std::vector<cooEntry> R_tranpose(R->entry.size());
    transpose_locally(R->entry, R->entry.size(), R_tranpose);

    // convert the indices to global
    for(nnz_t i = 0; i < R_tranpose.size(); i++){
        R_tranpose[i].col += R->splitNew[rank];
    }
//    print_vector(R->entry, -1, "R->entry", comm);
//    print_vector(R_tranpose, -1, "R_tranpose", comm);

    std::vector<index_t> nnzPerCol_left(A->Mbig, 0);
//    unsigned int *AnnzPerCol_p = &nnzPerCol_left[0] - A[0].col;
    for(nnz_t i = 0; i < A->entry.size(); i++){
//        if(rank==0) printf("A[i].col = %u, \tA_col_size = %u \n", A[i].col - A_col_offset, A_col_size);
        nnzPerCol_left[A->entry[i].col]++;
    }

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

//    print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 4: rank = %d\n", rank); MPI_Barrier(comm);}

    int right_neighbor = (rank + 1)%nprocs;
    int left_neighbor = rank - 1;
    if (left_neighbor < 0)
        left_neighbor += nprocs;
//    if(rank==0) printf("left_neighbor = %d, right_neighbor = %d\n", left_neighbor, right_neighbor);

    int owner;
    unsigned long send_size = R_tranpose.size();
    unsigned long recv_size;
    index_t mat_recv_M;

    std::vector<cooEntry> mat_recv = R_tranpose;
    std::vector<cooEntry> mat_send = R_tranpose;

    R_tranpose.clear();
    R_tranpose.shrink_to_fit();

    MPI_Request *requests = new MPI_Request[4];
    MPI_Status  *statuses = new MPI_Status[4];

    std::vector<cooEntry> AP;

    for(index_t k = rank; k < rank+nprocs; k++){
        // Both local and remote loops are done here. The first iteration is the local loop. The rest are remote.
        // Send R_tranpose to the left_neighbor processor, receive R_tranpose from the right_neighbor processor.
        // In the next step: send R_tranpose that was received in the previous step to the left_neighbor processor,
        // receive R_tranpose from the right_neighbor processor. And so on.
        // --------------------------------------------------------------------

        // communicate size
        MPI_Irecv(&recv_size, 1, MPI_UNSIGNED_LONG, right_neighbor, right_neighbor, comm, requests);
        MPI_Isend(&send_size, 1, MPI_UNSIGNED_LONG, left_neighbor,  rank,           comm, requests+1);
        MPI_Waitall(1, requests, statuses);
        printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
//        print_vector(mat_recv, 0, "mat_recv", A->comm);
        mat_recv.resize(recv_size);

        MPI_Irecv(&mat_recv[0], recv_size, cooEntry::mpi_datatype(), right_neighbor, right_neighbor, comm, requests+2);
        MPI_Isend(&mat_send[0], send_size, cooEntry::mpi_datatype(), left_neighbor,  rank,           comm, requests+3);

        owner = k%nprocs;
        mat_recv_M = P->splitNew[owner + 1] - P->splitNew[owner];

        nnzPerCol_right.assign(mat_recv_M, 0);
        for(nnz_t i = 0; i < mat_send.size(); i++){
            nnzPerCol_right[mat_send[i].col - P->splitNew[owner]]++;
        }
//        print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
        nnzPerColScan_right.resize(mat_recv_M+1);
        nnzPerColScan_right[0] = 0;
        for(nnz_t i = 0; i < mat_recv_M; i++){
            nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
        }
//        print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);

        fast_mm(&A->entry[0], &mat_send[0], AP, A->entry.size(), mat_send.size(),
                A->M, A->split[rank], A->Mbig, 0, 0, mat_recv_M, P->splitNew[owner],
                &nnzPerColScan_left[0], &nnzPerColScan_left[1],
                &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

        MPI_Waitall(3, requests+1, statuses+1);

        mat_recv.swap(mat_send);
        send_size = recv_size;
//        print_vector(mat_send, -1, "mat_send", A->comm);
//        print_vector(mat_recv, -1, "mat_recv", A->comm);
//        prev_owner = owner;
    }

    delete [] requests;
    delete [] statuses;

    std::sort(AP.begin(), AP.end());
//    print_vector(AP, -1, "AP", A->comm);

    mat_send.clear();
    mat_send.shrink_to_fit();
    mat_recv.clear();
    mat_recv.shrink_to_fit();
//    nnzPerCol_right.clear();
//    nnzPerColScan_left.clear();
//    nnzPerColScan_right.clear();
//    nnzPerCol_right.shrink_to_fit();
//    nnzPerColScan_left.shrink_to_fit();
//    nnzPerColScan_right.shrink_to_fit();

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 5: rank = %d\n", rank); MPI_Barrier(comm);}


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
//    print_vector(P->entry, -1, "P->entry", comm);
//    print_vector(P_tranpose, -1, "P_tranpose", comm);


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

//    print_vector(nnzPerColScan_left, -1, "nnzPerColScan_left", comm);

    // compute nnzPerColScan_left for AP
    nnzPerCol_right.assign(P->Nbig, 0);
    for(nnz_t i = 0; i < AP.size(); i++){
        nnzPerCol_right[AP[i].col]++;
    }

//    print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);

    nnzPerColScan_right.resize(P->Nbig+1);
    nnzPerColScan_right[0] = 0;
    for(nnz_t i = 0; i < P->Nbig; i++){
        nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
    }

    nnzPerCol_right.clear();
    nnzPerCol_right.shrink_to_fit();

//    print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);

    // multiply: R_i * (AP)_i. in which R_i = P_i_tranpose
    std::vector<cooEntry> RAP_temp;
    fast_mm(&P_tranpose[0], &AP[0], RAP_temp, P_tranpose.size(), AP.size(),
            P->Nbig, 0, P->M, P->split[rank], A->split[rank], P->Nbig, 0,
            &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
            &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

    AP.clear();
    AP.shrink_to_fit();
    P_tranpose.clear();
    P_tranpose.shrink_to_fit();
    nnzPerColScan_left.clear();
    nnzPerColScan_left.shrink_to_fit();
    nnzPerColScan_right.clear();
    nnzPerColScan_right.shrink_to_fit();

//    print_vector(RAP_temp, -1, "RAP_temp", A->comm);
    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 6: rank = %d\n", rank); MPI_Barrier(comm);}

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

//    MPI_Barrier(comm); printf("rank %d: RAP_temp_row.size = %lu \n", rank, RAP_temp_row.size()); MPI_Barrier(comm);
//    print_vector(RAP_temp_row, -1, "RAP_temp_row", comm);
//    print_vector(P->splitNew, 0, "P->splitNew", comm);

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 7: rank = %d\n", rank); MPI_Barrier(comm);}

    std::vector<cooEntry_row> RAP_row_sorted;
    par::sampleSort(RAP_temp_row, RAP_row_sorted, P->splitNew, comm);

    RAP_temp_row.clear();
    RAP_temp_row.shrink_to_fit();

//    print_vector(RAP_row_sorted, -1, "RAP_row_sorted", A->comm);
//    MPI_Barrier(comm); printf("rank %d: RAP_row_sorted.size = %lu \n", rank, RAP_row_sorted.size()); MPI_Barrier(comm);

//    std::vector<cooEntry> RAP_sorted(RAP_row_sorted.size());
//    memcpy(&RAP_sorted[0], &RAP_row_sorted[0], RAP_row_sorted.size() * sizeof(cooEntry));
//    RAP_row_sorted.clear();
//    RAP_row_sorted.shrink_to_fit();

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 8: rank = %d\n", rank); MPI_Barrier(comm);}

    // *******************************************************
    // form Ac
    // *******************************************************
    // version 1: without sparsification
    // *******************************************************

    // remove duplicates.
    double val_temp;
    for(nnz_t i = 0; i < RAP_row_sorted.size(); i++){
        val_temp = RAP_row_sorted[i].val;
        while(i<RAP_row_sorted.size()-1 && RAP_row_sorted[i] == RAP_row_sorted[i+1]){ // values of entries with the same row and col should be added.
            val_temp += RAP_row_sorted[i+1].val;
            i++;
        }
        Ac->entry.emplace_back( cooEntry(RAP_row_sorted[i].row, RAP_row_sorted[i].col, val_temp) );
    }

    RAP_row_sorted.clear();
    RAP_row_sorted.shrink_to_fit();

//    print_vector(Ac->entry, -1, "Ac->entry", A->comm);
    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 9: rank = %d\n", rank); MPI_Barrier(comm);}

    // *******************************************************
    // version 2: with sparsification
    // *******************************************************
/*
    nnz_t no_sparse_size = 0;

    // remove duplicates.
    // compute Frobenius norm squared (norm_frob_sq).
    double val_temp;
    double norm_frob_sq = 0;
    std::vector<cooEntry> Ac_orig;
    for(nnz_t i=0; i<RAP_row_sorted.size(); i++){
        val_temp = RAP_row_sorted[i].val;
        while(i<RAP_row_sorted.size()-1 && RAP_row_sorted[i] == RAP_row_sorted[i+1]){ // values of entries with the same row and col should be added.
            val_temp += RAP_row_sorted[i+1].val;
            i++;
        }

//        if( fabs(val_temp) > sparse_epsilon / 2 / Ac->Mbig)
        if(val_temp * val_temp > sparse_epsilon * sparse_epsilon / (4 * Ac->Mbig * Ac->Mbig) ){
            Ac_orig.emplace_back( cooEntry(RAP_row_sorted[i].row, RAP_row_sorted[i].col, val_temp) );
            norm_frob_sq += val_temp * val_temp;
        }
        no_sparse_size++; //todo: just for test. delete this later!
    }

    if(rank==0) printf("\noriginal size without sparsification   \t= %lu\n", no_sparse_size);
    if(rank==0) printf("filtered Ac size before sparsification \t= %lu\n", Ac_orig.size());
//    std::sort(Ac_orig.begin(), Ac_orig.end());
//    print_vector(Ac_orig, -1, "Ac_orig", A->comm);

    RAP_row_sorted.clear();
    RAP_row_sorted.shrink_to_fit();

    // *******************************************************
    // sparsification
    // *******************************************************

    //Type of random number distribution
    std::uniform_real_distribution<double> dist(0.0,1.0); //(min, max)

    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng;

    //Initialize with non-deterministic seeds
    rng.seed(std::random_device{}());

    // s = 28nln(sqrt(2)*n) / epsilon^2
    nnz_t sample_size = nnz_t( (double)28 * Ac->Mbig * log(sqrt(2) * Ac->Mbig) * norm_frob_sq / (sparse_epsilon * sparse_epsilon) );
    if(rank==0) printf("sample size \t\t\t\t= %lu\n", sample_size);
//    if(rank==0) printf("norm_frob_sq = %f, \tsparse_epsilon = %f, \tAc->Mbig = %u \n", norm_frob_sq, sparse_epsilon, Ac->Mbig);

    std::vector<cooEntry> Ac_sample(sample_size);
    double norm_temp = 0, criteria;
    for(nnz_t i = 0; i < Ac_orig.size(); i++){
        norm_temp += Ac_orig[i].val * Ac_orig[i].val;

        criteria = (Ac_orig[i].val * Ac_orig[i].val) / norm_temp;
        for(nnz_t j = 0; j < sample_size; j++){
            if(dist(rng) < criteria){
//                std::cout << "dist(rng) = " << dist(rng) << "\tcriteria = " << criteria << "\tAc_sample[j] = " << Ac_sample[j] << std::endl;
//                Ac_sample[j] = cooEntry(Ac_orig[i].row, Ac_orig[i].col, Ac_orig[i].val);
                Ac_sample[j] = Ac_orig[i];
            }
        }
    }

//    if(rank==0) printf("Ac_sample.size() = %lu\n", Ac_sample.size());
//    print_vector(Ac_sample, -1, "Ac_sample", A->comm);
    std::sort(Ac_sample.begin(), Ac_sample.end());

    // remove duplicates and change the values based on Algorithm 1 of Drineas' paper.
    double factor = norm_frob_sq / sample_size;
    for(nnz_t i=0; i<Ac_sample.size(); i++){
        val_temp = Ac_sample[i].val;
        while(i<Ac_sample.size()-1 && Ac_sample[i] == Ac_sample[i+1]){ // values of entries with the same row and col should be added.
            val_temp += Ac_sample[i+1].val;
            i++;
        }
//        Ac->entry.emplace_back( cooEntry(Ac_sample[i].row, Ac_sample[i].col, factor / val_temp) );
        Ac->entry.emplace_back( cooEntry(Ac_sample[i].row, Ac_sample[i].col, val_temp) );
    }

    if(rank==0) printf("Ac size after sparsification \t\t= %lu\n", Ac->entry.size());
    print_vector(Ac->entry, -1, "Ac->entry", A->comm);

    Ac_sample.clear();
    Ac_sample.shrink_to_fit();
*/
    // *******************************************************
    // use this part to print data to be used in Julia, to check the solution.
    // *******************************************************

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

    // *******************************************************
    // setup matrix
    // *******************************************************
    // Update this description: Shrinking gets decided inside repartition_nnz() or repartition_row() functions,
    // then repartition happens.
    // Finally, shrink_cpu() and matrix_setup() are called. In this way, matrix_setup is called only once.

    Ac->nnz_l = Ac->entry.size();
    MPI_Allreduce(&Ac->nnz_l, &Ac->nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

    if(verbose_coarsen){
        MPI_Barrier(comm); printf("coarsen: step 10: rank = %d\n", rank); MPI_Barrier(comm);}

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
            MPI_Barrier(comm); printf("coarsen: step 11: rank = %d\n", rank); MPI_Barrier(comm);}

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
            MPI_Barrier(comm); printf("coarsen: step 12: rank = %d\n", rank); MPI_Barrier(comm);}

        repartition_u_shrink_prepare(grid);

        if(Ac->shrinked){
            Ac->shrink_cpu();
        }

        if(verbose_coarsen){
            MPI_Barrier(comm); printf("coarsen: step 13: rank = %d\n", rank); MPI_Barrier(comm);}

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
} // coarsen()


int saena_object::coarsen_old(Grid *grid){$

    // todo: to improve the performance of this function, consider using the arrays used for RA also for RAP.
    // todo: this way allocating and freeing memory will be halved.

    saena_matrix *A = grid->A;
    prolong_matrix *P = &grid->P;
    restrict_matrix *R = &grid->R;
    saena_matrix *Ac = &grid->Ac;

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
        if(rank==0) printf("start of coarsen nprocs: %d \n", nprocs);
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
    for(nnz_t i=0; i<A->M; i++)
        AnnzPerRow[i] += A->nnzPerRow_remote[i];

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


int saena_object::coarsen_update_Ac(Grid *grid, std::vector<cooEntry> &diff){

    // This function computes delta_Ac = RAP in which A is diff = diag_block(A) - diag_block(A_new) at each level.
    // Then, updates Ac with the delta_Ac. Finally, it saves delta_Ac in diff for doing the same operation
    // for the next level.

    saena_matrix *A = grid->A;
    prolong_matrix *P = &grid->P;
    restrict_matrix *R = &grid->R;
    saena_matrix *Ac = &grid->Ac;

    MPI_Comm comm = A->comm;
//    Ac->active_old_comm = true;

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(verbose_coarsen2){
        MPI_Barrier(comm);
        printf("start of coarsen2: rank = %d, nprocs: %d, P.nnz_l = %lu, P.nnz_g = %lu, R.nnz_l = %lu,"
               " R.nnz_g = %lu, R.M = %u, R->nnz_l_local = %lu, R->nnz_l_remote = %lu \n\n", rank, nprocs,
               P->nnz_l, P->nnz_g, R->nnz_l, R->nnz_g, R->M, R->nnz_l_local, R->nnz_l_remote);
    }

    prolong_matrix RA_temp(comm); // RA_temp is being used to remove duplicates while pushing back to RA.

    // ************************************* RA_temp - A local *************************************
    // Some local and remote elements of RA_temp are computed here using local R and local A.
    // Note: A local means whole entries of A on this process, not just the diagonal block.

    // alloacted memory for AMaxM, instead of A.M to avoid reallocation of memory for when receiving data from other procs.
    unsigned int* AnnzPerRow = (unsigned int*)malloc(sizeof(unsigned int)*A->M);
    std::fill(&AnnzPerRow[0], &AnnzPerRow[A->M], 0);
    for(nnz_t i=0; i<diff.size(); i++){
//        if(rank==0) printf("%u\n", diff[i].row);
        AnnzPerRow[diff[i].row]++;
    }

//    MPI_Barrier(A->comm);
//    if(rank==0){
//        printf("rank = %d, AnnzPerRow: \n", rank);
//        for(long i=0; i<A->M; i++)
//            printf("%lu \t%u \n", i, AnnzPerRow[i]);}

    // alloacted memory for AMaxM+1, instead of A.M+1 to avoid reallocation of memory for when receiving data from other procs.
    unsigned int* AnnzPerRowScan = (unsigned int*)malloc(sizeof(unsigned int)*(A->M+1));
    AnnzPerRowScan[0] = 0;
    for(index_t i=0; i<A->M; i++){
        AnnzPerRowScan[i+1] = AnnzPerRowScan[i] + AnnzPerRow[i];
//        if(rank==1) printf("i=%lu, AnnzPerRow=%d, AnnzPerRowScan = %d\n", i+A->split[rank], AnnzPerRow[i], AnnzPerRowScan[i+1]);
    }

    if(verbose_coarsen2){
        MPI_Barrier(comm); printf("coarsen2: step 1: rank = %d\n", rank); MPI_Barrier(comm);}

    // find row-wise ordering for A and save it in indicesP
//    std::vector<nnz_t> indicesP(A->nnz_l_local);
//    for(nnz_t i=0; i<A->nnz_l; i++)
//        indicesP[i] = i;
//    std::sort(&indicesP[0], &indicesP[A->nnz_l_local], sort_indices2(&*A->entry.begin()));

    std::sort(diff.begin(), diff.end(), row_major);

    index_t jstart, jend;
    if(!R->entry_local.empty()) {
        for (index_t i = 0; i < R->nnz_l_local; i++) {
//            if(rank==0) std::cout << "i=" << i << "\tR[" << R->entry_local[i].row << ", " << R->entry_local[i].col
//                                  << "]=" << R->entry_local[i].val << std::endl;
            jstart = AnnzPerRowScan[R->entry_local[i].col - P->split[rank]];
            jend   = AnnzPerRowScan[R->entry_local[i].col - P->split[rank] + 1];
            if(jend - jstart == 0) continue;
            for (index_t j = jstart; j < jend; j++) {
//                if(rank==0) std::cout << "i=" << i << ", j=" << j
//                                      << "   \tdiff[" << diff[j].row
//                                      << ", " << diff[j].col << "]=\t" << diff[j].val
//                                      << "         \tR[" << R->entry_local[i].row << ", " << R->entry_local[i].col
//                                      << "]=\t" << R->entry_local[i].val << std::endl;
                RA_temp.entry.push_back(cooEntry(R->entry_local[i].row,
                                                 diff[j].col,
                                                 R->entry_local[i].val * diff[j].val));
            }
        }
    }

//    print_vector(RA_temp.entry, -1, "RA_temp.entry", comm);

    free(AnnzPerRow);
    free(AnnzPerRowScan);

    if(verbose_coarsen2){
        MPI_Barrier(comm); printf("coarsen2: step 2: rank = %d\n", rank); MPI_Barrier(comm);}

    std::sort(RA_temp.entry.begin(), RA_temp.entry.end());

//    print_vector(RA_temp.entry, -1, "RA_temp.entry: after sort", comm);

    prolong_matrix RA(comm);
    RA.entry.resize(RA_temp.entry.size());

    // remove duplicates.
    unsigned long entry_size = 0;
    for(nnz_t i=0; i<RA_temp.entry.size(); i++){
//        RA.entry.push_back(RA_temp.entry[i]);
        RA.entry[entry_size] = RA_temp.entry[i];
//        if(rank==1) std::cout << RA_temp.entry[i] << std::endl;
        while(i<RA_temp.entry.size()-1 && RA_temp.entry[i] == RA_temp.entry[i+1]){ // values of entries with the same row and col should be added.
//            RA.entry.back().val += RA_temp.entry[i+1].val;
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

    RA.entry.resize(entry_size);
    RA.entry.shrink_to_fit();

//    print_vector(RA.entry, -1, "RA.entry", comm);

    if(verbose_coarsen2){
        MPI_Barrier(comm); printf("coarsen2: step 3: rank = %d\n", rank); MPI_Barrier(comm);}

    // ************************************* RAP_temp - P local *************************************
    // Some local and remote elements of RAP_temp are computed here.
    // Note: P local means whole entries of P on this process, not just the diagonal block.

    prolong_matrix RAP_temp(comm); // RAP_temp is being used to remove duplicates while pushing back to RAP.

    unsigned int* PnnzPerRow = (unsigned int*)malloc(sizeof(unsigned int)*P->M);
    std::fill(&PnnzPerRow[0], &PnnzPerRow[P->M], 0);
    for(nnz_t i=0; i<P->nnz_l_local; i++){
//        if(rank==1) printf("%u\n", P->entry_local[i].row);
        PnnzPerRow[P->entry_local[i].row]++;
    }

//    if(rank==1) for(long i=0; i<P->M; i++) std::cout << PnnzPerRow[i] << std::endl;

    unsigned int* PnnzPerRowScan = (unsigned int*)malloc(sizeof(unsigned int)*(P->M+1));
    PnnzPerRowScan[0] = 0;
    for(nnz_t i = 0; i < P->M; i++){
        PnnzPerRowScan[i+1] = PnnzPerRowScan[i] + PnnzPerRow[i];
//        if(rank==2) printf("i=%lu, PnnzPerRow=%d, PnnzPerRowScan = %d\n", i, PnnzPerRow[i], PnnzPerRowScan[i]);
    }

    long procNum = 0;
    std::vector<nnz_t> left_block_nnz(nprocs, 0);
    if(!RA.entry.empty()){
        for (nnz_t i = 0; i < RA.entry.size(); i++) {
            procNum = lower_bound2(&P->split[0], &P->split[nprocs], RA.entry[i].col);
            left_block_nnz[procNum]++;
//        if(rank==1) printf("rank=%d, col = %lu, procNum = %ld \n", rank, R->entry_remote[0].col, procNum);
        }
    }

    std::vector<nnz_t> left_block_nnz_scan(nprocs+1);
    left_block_nnz_scan[0] = 0;
    for(int i = 0; i < nprocs; i++)
        left_block_nnz_scan[i+1] = left_block_nnz_scan[i] + left_block_nnz[i];

//    print_vector(left_block_nnz_scan, -1, "left_block_nnz_scan", comm);

    // find row-wise ordering for A and save it in indicesP
//    std::vector<nnz_t> indicesP_Prolong(P->nnz_l_local);
//    for(nnz_t i=0; i<P->nnz_l_local; i++)
//        indicesP_Prolong[i] = i;
//    std::sort(&indicesP_Prolong[0], &indicesP_Prolong[P->nnz_l], sort_indices2(&*P->entry.begin()));

    //....................
    // note: Here we want to multiply local RA by local P, but whole RA.entry is local because of how it was made earlier in this function.
    //....................

    for(nnz_t i=left_block_nnz_scan[rank]; i<left_block_nnz_scan[rank+1]; i++){
        for(nnz_t j = PnnzPerRowScan[RA.entry[i].col - P->split[rank]]; j < PnnzPerRowScan[RA.entry[i].col - P->split[rank] + 1]; j++){

//            if(rank==3) std::cout << RA.entry[i].row + P->splitNew[rank] << "\t" << P->entry[indicesP_Prolong[j]].col << "\t" << RA.entry[i].val * P->entry[indicesP_Prolong[j]].val << std::endl;

            RAP_temp.entry.emplace_back(cooEntry(RA.entry[i].row + P->splitNew[rank],  // Ac.entry should have global indices at the end.
                                                 P->entry_local[P->indicesP_local[j]].col,
                                                 RA.entry[i].val * P->entry_local[P->indicesP_local[j]].val));
        }
    }

//    print_vector(RAP_temp.entry, 0, "RAP_temp.entry", comm);

    free(PnnzPerRow);
    free(PnnzPerRowScan);

    if(verbose_coarsen2){
        MPI_Barrier(comm); printf("coarsen2: step 4: rank = %d\n", rank); MPI_Barrier(comm);}

    std::sort(RAP_temp.entry.begin(), RAP_temp.entry.end());

//    print_vector(RAP_temp.entry, 0, "RAP_temp.entry: after sort", comm);

    // erase_keep_remote() was called on Ac, so the remote elements are already in Ac.
    // Now resize it to store new local entries.
    Ac->entry_temp.resize(RAP_temp.entry.size());

    // remove duplicates.
    entry_size = 0;
    for(nnz_t i=0; i<RAP_temp.entry.size(); i++){
//        std::cout << RAP_temp.entry[i] << std::endl;
//        Ac->entry.push_back(RAP_temp.entry[i]);
        Ac->entry_temp[entry_size] = RAP_temp.entry[i];
        while(i<RAP_temp.entry.size()-1 && RAP_temp.entry[i] == RAP_temp.entry[i+1]){ // values of entries with the same row and col should be added.
//            if(rank==0) std::cout << Ac->entry_temp[entry_size] << std::endl;
            Ac->entry_temp[entry_size].val += RAP_temp.entry[i+1].val;
            i++;
        }
        entry_size++;
        // todo: pruning. don't hard code tol. does this make the matrix non-symmetric?
//        if( abs(Ac->entry.back().val) < 1e-6)
//            Ac->entry.pop_back();
    }

    Ac->entry_temp.resize(entry_size);
    Ac->entry_temp.shrink_to_fit();
//    print_vector(Ac->entry_temp, -1, "Ac->entry_temp", Ac->comm);
//    if(rank==0) printf("rank %d: entry_size = %lu \n", rank, entry_size);

    if(verbose_coarsen2){
        MPI_Barrier(comm); printf("coarsen2: step 5: rank = %d\n", rank); MPI_Barrier(comm);}

    // ********** setup matrix **********

    // decide to partition based on number of rows or nonzeros.
//    if(switch_repartition && Ac->density >= repartition_threshold)
//        Ac->repartition4(); // based on number of rows
//    else
        Ac->repartition_nnz_update_Ac(); // based on number of nonzeros

    diff.clear();
    diff.swap(Ac->entry_temp);

    if(verbose_coarsen2){
        MPI_Barrier(comm); printf("coarsen2: step 6: rank = %d\n", rank); MPI_Barrier(comm);}

//    repartition_u_shrink_prepare(grid);

//    if(Ac->shrinked)
//        Ac->shrink_cpu();

    // todo: try to reduce matrix_setup() for this case.
    if(Ac->active)
        Ac->matrix_setup();

    if(verbose_coarsen2){
        MPI_Barrier(comm); printf("end of coarsen2:  rank = %d\n", rank); MPI_Barrier(comm);}

    return 0;
} // end of coarsen_update_Ac()


// int saena_object::coarsen2
/*
int saena_object::coarsen2(saena_matrix* A, prolong_matrix* P, restrict_matrix* R, saena_matrix* Ac){
    // this function is similar to the coarsen(), but does R*A*P for only local (diagonal) blocks.

    // todo: to improve the performance of this function, consider using the arrays used for RA also for RAP.
    // todo: this way allocating and freeing memory will be halved.

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

    if(verbose_coarsen2){
        MPI_Barrier(comm);
        printf("start of coarsen: rank = %d, nprocs: %d, A->M = %u, A.nnz_l = %lu, A.nnz_g = %lu, P.nnz_l = %lu, P.nnz_g = %lu, R.nnz_l = %lu,"
                       " R.nnz_g = %lu, R.M = %u, R->nnz_l_local = %lu, R->nnz_l_remote = %lu \n\n", rank, nprocs, A->M, A->nnz_l,
               A->nnz_g, P->nnz_l, P->nnz_g, R->nnz_l, R->nnz_g, R->M, R->nnz_l_local, R->nnz_l_remote);
    }

//    unsigned long i, j;
    prolong_matrix RA_temp(comm); // RA_temp is being used to remove duplicates while pushing back to RA.

    // ************************************* RA_temp - A local *************************************
    // Some local and remote elements of RA_temp are computed here using local R and local A.

    unsigned int AMaxNnz, AMaxM;
    MPI_Allreduce(&A->nnz_l, &AMaxNnz, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);
    MPI_Allreduce(&A->M, &AMaxM, 1, MPI_UNSIGNED, MPI_MAX, comm);
//    MPI_Barrier(comm); printf("\nrank=%d, AMaxNnz=%d, AMaxM = %d \n", rank, AMaxNnz, AMaxM); MPI_Barrier(comm);
    // todo: is this way better than using the previous Allreduce? reduce on processor 0, then broadcast to other processors.

    // alloacted memory for AMaxM, instead of A.M to avoid reallocation of memory for when receiving data from other procs.
//    unsigned int* AnnzPerRow = (unsigned int*)malloc(sizeof(unsigned int)*AMaxM);
//    std::fill(&AnnzPerRow[0], &AnnzPerRow[AMaxM], 0);
    std::vector<index_t> AnnzPerRow(AMaxM, 0);
    for(nnz_t i=0; i<A->nnz_l; i++)
        AnnzPerRow[A->entry[i].row - A->split[rank]]++;

//    MPI_Barrier(A->comm);
//    if(rank==0){
//        printf("rank = %d, AnnzPerRow: \n", rank);
//        for(i=0; i<A->M; i++)
//            printf("%lu \t%u \n", i, AnnzPerRow[i]);
//    }

    // alloacted memory for AMaxM+1, instead of A.M+1 to avoid reallocation of memory for when receiving data from other procs.
//    unsigned int* AnnzPerRowScan = (unsigned int*)malloc(sizeof(unsigned int)*(AMaxM+1));
    std::vector<nnz_t> AnnzPerRowScan(AMaxM+1);
    AnnzPerRowScan[0] = 0;
    for(index_t i=0; i<A->M; i++){
        AnnzPerRowScan[i+1] = AnnzPerRowScan[i] + AnnzPerRow[i];
//        if(rank==1) printf("i=%lu, AnnzPerRow=%d, AnnzPerRowScan = %d\n", i+A->split[rank], AnnzPerRow[i], AnnzPerRowScan[i+1]);
    }

    if(verbose_coarsen2){
        MPI_Barrier(comm); printf("coarsen: step 1: rank = %d", rank); MPI_Barrier(comm);}

    // todo: combine indicesP and indicesPRecv together.
    // find row-wise ordering for A and save it in indicesP
//    unsigned long* indicesP = (unsigned long*)malloc(sizeof(unsigned long)*A->nnz_l);
    std::vector<nnz_t> indicesP(A->nnz_l);
    for(nnz_t i=0; i<A->nnz_l; i++)
        indicesP[i] = i;
    std::sort(&indicesP[0], &indicesP[A->nnz_l], sort_indices2(&*A->entry.begin()));

    unsigned long jstart, jend;
    if(!R->entry_local.empty()) {
        for (nnz_t i = 0; i < R->nnz_l_local; i++) {
            jstart = AnnzPerRowScan[R->entry_local[i].col - P->split[rank]];
            jend   = AnnzPerRowScan[R->entry_local[i].col - P->split[rank] + 1];
            if(jend - jstart == 0) continue;
            for (nnz_t j = jstart; j < jend; j++) {
//            if(rank==0) std::cout << A->entry[indicesP[j]].row << "\t" << A->entry[indicesP[j]].col << "\t" << A->entry[indicesP[j]].val
//                             << "\t" << R->entry_local[i].col << "\t" << R->entry_local[i].col - P->split[rank] << std::endl;
                RA_temp.entry.push_back(cooEntry(R->entry_local[i].row,
                                                 A->entry[indicesP[j]].col,
                                                 R->entry_local[i].val * A->entry[indicesP[j]].val));
            }
        }
    }

//    free(indicesP);
    indicesP.clear();
    indicesP.shrink_to_fit();

//    if(rank==0){
//        std::cout << "\nRA_temp.entry.size = " << RA_temp.entry.size() << std::endl;
//        for(i=0; i<RA_temp.entry.size(); i++)
//            std::cout << RA_temp.entry[i].row + R->splitNew[rank] << "\t" << RA_temp.entry[i].col << "\t" << RA_temp.entry[i].val << std::endl;}

    if(verbose_coarsen2){
        MPI_Barrier(comm); printf("coarsen: step 2: rank = %d", rank); MPI_Barrier(comm);}

    // todo: check this: since entries of RA_temp with these row indices only exist on this processor,
    // todo: duplicates happen only on this processor, so sorting should be done locally.
    std::sort(RA_temp.entry.begin(), RA_temp.entry.end());

//    MPI_Barrier(A->comm);
//    if(rank==1)
//        for(j=0; j<RA_temp.entry.size(); j++)
//            std::cout << RA_temp.entry[j].row + P->splitNew[rank] << "\t" << RA_temp.entry[j].col << "\t" << RA_temp.entry[j].val << std::endl;

    prolong_matrix RA(comm);

    // todo: here
    // remove duplicates.
    for(nnz_t i=0; i<RA_temp.entry.size(); i++){
        RA.entry.push_back(RA_temp.entry[i]);
//        if(rank==1) std::cout << std::endl << "start:" << std::endl << RA_temp.entry[i].val << std::endl;
        while(i<RA_temp.entry.size()-1 && RA_temp.entry[i] == RA_temp.entry[i+1]){ // values of entries with the same row and col should be added.
            RA.entry.back().val += RA_temp.entry[i+1].val;
            i++;
//            if(rank==1) std::cout << RA_temp.entry[i+1].val << std::endl;
        }
//        if(rank==1) std::cout << std::endl << "final: " << std::endl << RA.entry[RA.entry.size()-1].val << std::endl;
        // todo: pruning. don't hard code tol. does this make the matrix non-symmetric?
//        if( abs(RA.entry.back().val) < 1e-6)
//            RA.entry.pop_back();
//        if(rank==1) std::cout << "final: " << std::endl << RA.entry.back().val << std::endl;
    }

//    MPI_Barrier(comm);
//    if(rank==0){
//        std::cout << "RA.entry.size = " << RA.entry.size() << std::endl;
//        for(j=0; j<RA.entry.size(); j++)
//            std::cout << RA.entry[j].row + P->splitNew[rank] << "\t" << RA.entry[j].col << "\t" << RA.entry[j].val << std::endl;}
//    MPI_Barrier(comm);

    // find the start and end nnz iterator of each block of R.
    // use A.split for this part to find each block corresponding to each processor's A.
//    unsigned int* left_block_nnz = (unsigned int*)malloc(sizeof(unsigned int)*(nprocs));
//    std::fill(left_block_nnz, &left_block_nnz[nprocs], 0);
    std::vector<int> left_block_nnz(nprocs, 0);

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

    // ************************************* RAP_temp - P local *************************************
    // Some local and remote elements of RAP_temp are computed here.

    prolong_matrix RAP_temp(comm); // RAP_temp is being used to remove duplicates while pushing back to RAP.
    index_t P_max_M;
    MPI_Allreduce(&P->M, &P_max_M, 1, MPI_UNSIGNED, MPI_MAX, comm);
//    MPI_Barrier(comm); printf("rank=%d, PMaxNnz=%d \n", rank, PMaxNnz); MPI_Barrier(comm);
    // todo: is this way better than using the previous Allreduce? reduce on processor 0, then broadcast to other processors.

//    unsigned int* PnnzPerRow = (unsigned int*)malloc(sizeof(unsigned int)*P_max_M);
//    std::fill(&PnnzPerRow[0], &PnnzPerRow[P->M], 0);
    std::vector<index_t> PnnzPerRow(P_max_M, 0);
    for(nnz_t i=0; i<P->nnz_l; i++){
        PnnzPerRow[P->entry[i].row]++;
    }

//    if(rank==1)
//        for(i=0; i<P->M; i++)
//            std::cout << PnnzPerRow[i] << std::endl;

//    unsigned int* PnnzPerRowScan = (unsigned int*)malloc(sizeof(unsigned int)*(P_max_M+1));
    std::vector<nnz_t> PnnzPerRowScan(P_max_M+1);
    PnnzPerRowScan[0] = 0;
    for(index_t i = 0; i < P->M; i++){
        PnnzPerRowScan[i+1] = PnnzPerRowScan[i] + PnnzPerRow[i];
//        if(rank==2) printf("i=%lu, PnnzPerRow=%d, PnnzPerRowScan = %d\n", i, PnnzPerRow[i], PnnzPerRowScan[i]);
    }

//    std::fill(left_block_nnz, &left_block_nnz[nprocs], 0);
    left_block_nnz.assign(nprocs, 0);
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

//    if(rank==1){
//        std::cout << "RABlockStart: " << std::endl;
//        for(i=0; i<nprocs+1; i++)
//            std::cout << R_block_nnz_scan[i] << std::endl;}

    // todo: combine indicesP_Prolong and indicesP_ProlongRecv together.
    // find row-wise ordering for A and save it in indicesP
//    unsigned long* indicesP_Prolong = (unsigned long*)malloc(sizeof(unsigned long)*P->nnz_l);
    std::vector<nnz_t> indicesP_Prolong(P->nnz_l);
    for(nnz_t i=0; i<P->nnz_l; i++)
        indicesP_Prolong[i] = i;
    std::sort(&indicesP_Prolong[0], &indicesP_Prolong[P->nnz_l], sort_indices2(&*P->entry.begin()));

    for(nnz_t i=left_block_nnz_scan[rank]; i<left_block_nnz_scan[rank+1]; i++){
        for(nnz_t j = PnnzPerRowScan[RA.entry[i].col - P->split[rank]]; j < PnnzPerRowScan[RA.entry[i].col - P->split[rank] + 1]; j++){

//            if(rank==3) std::cout << RA.entry[i].row + P->splitNew[rank] << "\t" << P->entry[indicesP_Prolong[j]].col << "\t" << RA.entry[i].val * P->entry[indicesP_Prolong[j]].val << std::endl;

            RAP_temp.entry.emplace_back(cooEntry(RA.entry[i].row + P->splitNew[rank],  // Ac.entry should have global indices at the end.
                                                 P->entry[indicesP_Prolong[j]].col,
                                                 RA.entry[i].val * P->entry[indicesP_Prolong[j]].val));
        }
    }

//    if(rank==1)
//        for(i=0; i<RAP_temp.entry.size(); i++)
//            std::cout << RAP_temp.entry[i].row << "\t" << RAP_temp.entry[i].col << "\t" << RAP_temp.entry[i].val << std::endl;

//    free(indicesP_Prolong);
//    free(PnnzPerRow);
//    free(PnnzPerRowScan);
//    free(left_block_nnz);
//    free(left_block_nnz_scan);

    indicesP_Prolong.clear();
    PnnzPerRow.clear();
    PnnzPerRowScan.clear();
    left_block_nnz.clear();
    left_block_nnz_scan.clear();

    indicesP_Prolong.shrink_to_fit();
    PnnzPerRow.shrink_to_fit();
    PnnzPerRowScan.shrink_to_fit();
    left_block_nnz.shrink_to_fit();
    left_block_nnz_scan.shrink_to_fit();

    if(verbose_coarsen2){
        MPI_Barrier(comm); printf("coarsen: step 3: rank = %d", rank); MPI_Barrier(comm);}

    std::sort(RAP_temp.entry.begin(), RAP_temp.entry.end());

//    if(rank==2)
//        for(j=0; j<RAP_temp.entry.size(); j++)
//            std::cout << RAP_temp.entry[j].row << "\t" << RAP_temp.entry[j].col << "\t" << RAP_temp.entry[j].val << std::endl;

    // todo:here
    // remove duplicates.
    for(nnz_t i=0; i<RAP_temp.entry.size(); i++){
        Ac->entry.push_back(RAP_temp.entry[i]);
        while(i<RAP_temp.entry.size()-1 && RAP_temp.entry[i] == RAP_temp.entry[i+1]){ // values of entries with the same row and col should be added.
            Ac->entry.back().val += RAP_temp.entry[i+1].val;
            i++;
        }
        // todo: pruning. don't hard code tol. does this make the matrix non-symmetric?
//        if( abs(Ac->entry.back().val) < 1e-6)
//            Ac->entry.pop_back();
    }
//    MPI_Barrier(comm); printf("rank=%d here6666666666666!!!!!!!! \n", rank); MPI_Barrier(comm);

//    par::sampleSort(Ac_temp, Ac->entry, comm);
//    Ac->entry = Ac_temp;

//    if(rank==1){
//        std::cout << "after sort:" << std::endl;
//        for(j=0; j<Ac->entry.size(); j++)
//            std::cout << Ac->entry[j] << std::endl;
//    }

    Ac->nnz_l = Ac->entry.size();
    MPI_Allreduce(&Ac->nnz_l, &Ac->nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
    Ac->Mbig = P->Nbig;
    Ac->M = P->splitNew[rank+1] - P->splitNew[rank];
    Ac->split = P->splitNew;
    Ac->cpu_shrink_thre1 = A->cpu_shrink_thre1;
    Ac->last_M_shrink = A->last_M_shrink;
//    Ac->last_nnz_shrink = A->last_nnz_shrink;
    Ac->last_density_shrink = A->last_density_shrink;
    Ac->comm = A->comm;
    Ac->comm_old = A->comm;
    Ac->active_old_comm = true;
//    printf("\nrank = %d, Ac->Mbig = %u, Ac->M = %u, Ac->nnz_l = %u, Ac->nnz_g = %u \n", rank, Ac->Mbig, Ac->M, Ac->nnz_l, Ac->nnz_g);

//    if(verbose_coarsen){
//        printf("\nrank = %d, Ac->Mbig = %u, Ac->M = %u, Ac->nnz_l = %u, Ac->nnz_g = %u \n", rank, Ac->Mbig, Ac->M, Ac->nnz_l, Ac->nnz_g);}

    if(verbose_coarsen2){
        MPI_Barrier(comm); printf("coarsen: step 4: rank = %d", rank); MPI_Barrier(comm);}

//    MPI_Barrier(comm);
//    if(rank==0){
//        for(i = 0; i < Ac->nnz_l; i++)
//            std::cout << i << "\t" << Ac->entry[i] << std::endl;
//        std::cout << std::endl;}
//    MPI_Barrier(comm);
//    if(rank==1){
//        for(i = 0; i < Ac->nnz_l; i++)
//            std::cout << i << "\t" << Ac->entry[i] << std::endl;
//        std::cout << std::endl;}
//    MPI_Barrier(comm);
//    if(rank==2){
//        for(i = 0; i < Ac->nnz_l; i++)
//            std::cout << i << "\t" << Ac->entry[i] << std::endl;
//        std::cout << std::endl;}
//    MPI_Barrier(comm);

//    printf("rank=%d \tA: Mbig=%u, nnz_g = %u, nnz_l = %u, M = %u \tAc: Mbig=%u, nnz_g = %u, nnz_l = %u, M = %u \n",
//            rank, A->Mbig, A->nnz_g, A->nnz_l, A->M, Ac->Mbig, Ac->nnz_g, Ac->nnz_l, Ac->M);
//    MPI_Barrier(comm);
//    if(rank==1)
//        for(i=0; i<nprocs+1; i++)
//            std::cout << Ac->split[i] << std::endl;


    // ********** check for cpu shrinking **********
    // if number of rows on Ac < threshold*number of rows on A, then shrink.
    // redistribute Ac from processes 4k+1, 4k+2 and 4k+3 to process 4k.

    // todo: is this part required for coarsen2()?
//    if( (nprocs >= Ac->cpu_shrink_thre2) && (Ac->last_M_shrink >= (Ac->Mbig * A->cpu_shrink_thre1)) ){

//        shrink_cpu_A(Ac, P->splitNew);

//        MPI_Barrier(comm);
//        if(rank==0) std::cout << "\nafter shrink: Ac->last_M_shrink = " << Ac->last_M_shrink << ", Ac->Mbig = " << Ac->Mbig
//                              << ", mult = " << Ac->Mbig * A->cpu_shrink_thre1 << std::endl;
//        MPI_Barrier(comm);
//    }

    // ********** setup matrix **********

    if(verbose_coarsen2){
        MPI_Barrier(comm); printf("coarsen: step 5: rank = %d", rank); MPI_Barrier(comm);}

    if(Ac->active) // there is another if(active) in matrix_setup().
        Ac->matrix_setup();

    if(verbose_coarsen2){
        MPI_Barrier(comm); printf("end of coarsen: step 6: rank = %d", rank); MPI_Barrier(comm);}

    return 0;
} // end of SaenaObject::coarsen
*/


int saena_object::solve_coarsest_CG(saena_matrix* A, std::vector<value_t>& u, std::vector<value_t>& rhs){
    // this is CG.
    // u is zero in the beginning. At the end, it is the solution.

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(verbose_solve_coarse && rank==0) printf("start of solve_coarsest_CG()\n");

    // since u is zero, res = -rhs, and the residual in this function is the negative of what I have in this library.
    std::vector<value_t> res = rhs;

    double initial_dot;
    dotProduct(res, res, &initial_dot, comm);
//    if(rank==0) std::cout << "\nsolveCoarsest: initial norm(res) = " << sqrt(initial_dot) << std::endl;

    double dot = initial_dot;
    int max_iter = CG_max_iter;
    if (dot < CG_tol*CG_tol)
        max_iter = 0;

    std::vector<value_t> dir(A->M);
    dir = res;

    double dot2;
    std::vector<value_t> res2(A->M);

    double factor, dot_prev;
    std::vector<value_t> matvecTemp(A->M);
    int i = 1;
    while (i < max_iter) {
//        if(rank==0) std::cout << "starting iteration of CG = " << i << std::endl;
        // factor = sq_norm/ (dir' * A * dir)
        A->matvec(dir, matvecTemp);

        dotProduct(dir, matvecTemp, &factor, comm);
        factor = dot / factor;
//        if(rank==1) std::cout << "\nsolveCoarsest: factor = " << factor << std::endl;

        #pragma omp parallel for
        for(index_t j = 0; j < A->M; j++){
            u[j]   += factor * dir[j];
            res[j] -= factor * matvecTemp[j];
        }

        dot_prev = dot;
        dotProduct(res, res, &dot, comm);
//        if(rank==0) std::cout << "absolute norm(res) = " << sqrt(dot) << "\t( r_i / r_0 ) = " << sqrt(dot)/initialNorm << "  \t( r_i / r_i-1 ) = " << sqrt(dot)/sqrt(dot_prev) << std::endl;
//        if(rank==0) std::cout << sqrt(dot)/initialNorm << std::endl;

        if(verbose_solve_coarse && rank==0)
            std::cout << "sqrt(dot)/sqrt(initial_dot) = " << sqrt(dot/initial_dot) << "  \tCG_tol = " << CG_tol << std::endl;

//        A->residual(u, rhs, res2);
//        dotProduct(res2, res2, &dot2, comm);
//        if(rank==0) std::cout << "norm(res) = " << sqrt(dot2) << std::endl;

        if (dot/initial_dot < CG_tol*CG_tol)
            break;

        factor = dot / dot_prev;
//        if(rank==1) std::cout << "\nsolveCoarsest: update factor = " << factor << std::endl;

        // update direction
        #pragma omp parallel for
        for(index_t j = 0; j < A->M; j++)
            dir[j] = res[j] + factor * dir[j];

        i++;
    }

    if(i == max_iter && max_iter != 0)
        i--;

//    print_vector(u, -1, "u at the end of CG", comm);

    if(verbose_solve_coarse && rank==0) printf("end of solve_coarsest! it took CG iterations = %d\n \n", i);
//    if(rank==0) printf("end of solve_coarsest! it took CG iterations = %d \n\n", i);

    return 0;
}


int saena_object::solve_coarsest_SuperLU(saena_matrix *A, std::vector<value_t> &u, std::vector<value_t> &rhs){$

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(verbose_solve_coarse && rank==0) printf("start of solve_coarsest_SuperLU()\n");
//    print_vector(rhs, -1, "rhs passed to superlu", comm);

    superlu_dist_options_t options;
    SuperLUStat_t stat;
    SuperMatrix A_SLU;
    ScalePermstruct_t ScalePermstruct;
    LUstruct_t LUstruct;
    SOLVEstruct_t SOLVEstruct;
    gridinfo_t grid;
    double   *berr;
//    double   *b, *xtrue;
    double   *b;
    int      m, n, m_loc, nnz_loc;
    int      nprow, npcol;
//    int      iam, info, ldb, ldx, nrhs;
    int      iam, info, ldb, nrhs;
//    char     **cpp, c;
//    FILE *fp, *fopen();
//    FILE *fp;
//    int cpp_defs();

    nprow = nprocs;  /* Default process rows.      */
    npcol = 1;  /* Default process columns.   */
    nrhs  = 1;  /* Number of right-hand side. */

    /* ------------------------------------------------------------
       INITIALIZE MPI ENVIRONMENT.
       ------------------------------------------------------------*/
//    MPI_Init( &argc, &argv );

//    char* file_name(argv[5]);
//    saena::matrix A_saena (file_name, comm);
//    A_saena.assemble();
//    A_saena.print_entry(-1);
//    if(rank==0) printf("after matrix assemble.\n");

    /*
    // Parse command line argv[].
    for (cpp = argv+1; *cpp; ++cpp) {
        if ( **cpp == '-' ) {
            c = *(*cpp+1);
            ++cpp;
            switch (c) {
                case 'h':
                    printf("Options:\n");
                    printf("\t-r <int>: process rows    (default %4d)\n", nprow);
                    printf("\t-c <int>: process columns (default %4d)\n", npcol);
                    exit(0);
                    break;
                case 'r': nprow = atoi(*cpp);
                    break;
                case 'c': npcol = atoi(*cpp);
                    break;
            }
        } else { // Last arg is considered a filename
//            if ( !(fp = fopen(*cpp, "r")) ) {
//                ABORT("File does not exist");
//            }

            saena::matrix A_saena (*cpp, comm);
            A_saena.assemble();
            A_saena.print_entry(-1);
            if(rank==0) printf("after matrix assemble.\n");
            break;
        }
    }
*/
    /* ------------------------------------------------------------
       INITIALIZE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------*/
    if(verbose_solve_coarse && rank==0) printf("INITIALIZE THE SUPERLU PROCESS GRID. \n");

    superlu_gridinit(comm, nprow, npcol, &grid);

    // Bail out if I do not belong in the grid.
    iam = grid.iam; // my process rank in this group
//    printf("iam = %d, nprow = %d, npcol = %d \n", iam, nprow, npcol);
//    if ( iam >= nprow * npcol )	goto out;

    if ( verbose_solve_coarse && !iam ) {
        int v_major, v_minor, v_bugfix;
        superlu_dist_GetVersionNumber(&v_major, &v_minor, &v_bugfix);
        printf("Library version:\t%d.%d.%d\n", v_major, v_minor, v_bugfix);

//        printf("Input matrix file:\t%s\n", *cpp);
        printf("Process grid:\t\t%d X %d\n", nprow, npcol);
        fflush(stdout);
    }

#if ( VAMPIR>=1 )
    VT_traceoff();
#endif

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Enter main()");
#endif

    /* ------------------------------------------------------------
       PASS THE MATRIX FROM SAENA
       ------------------------------------------------------------*/

    // Set up the local A_SLU in NR_loc format
//    dCreate_CompRowLoc_Matrix_dist(A_SLU, m, n, nnz_loc, m_loc, fst_row,
//                                   nzval_loc, colind, rowptr,
//                                   SLU_NR_loc, SLU_D, SLU_GE);

    if(verbose_solve_coarse && rank==0) printf("PASS THE MATRIX FROM SAENA. \n");

    m = A->Mbig;
    m_loc = A->M;
    n = m;
    nnz_loc = A->nnz_l;
    ldb = m_loc;
    if(verbose_solve_coarse && rank==0)
        printf("m = %d, m_loc = %d, n = %d, nnz_g = %ld, nnz_loc = %d, ldb = %d \n",
               m, m_loc, n, A->nnz_g, nnz_loc, ldb);

    // CSR format (compressed row)
    // sort entries in row-major
    std::vector<cooEntry> entry_temp = A->entry;
    std::sort(entry_temp.begin(), entry_temp.end(), row_major);
//    print_vector(entry_temp, -1, "entry_temp", comm);

    index_t fst_row = A->split[rank];
    std::vector<int> nnz_per_row(m_loc, 0);
//    std::vector<int> rowptr(m_loc+1);
//    std::vector<int> colind(nnz_loc);
//    std::vector<double> nzval_loc(nnz_loc);

    auto* rowptr = (int_t *) intMalloc_dist(m_loc+1);
    auto* nzval_loc = (double *) doubleMalloc_dist(nnz_loc);
    auto* colind = (int_t *) intMalloc_dist(nnz_loc);

    for(nnz_t i = 0; i < nnz_loc; i++){
        nzval_loc[i] = entry_temp[i].val;
        nnz_per_row[entry_temp[i].row - fst_row]++;
        colind[i] = entry_temp[i].col;
    }

    // rowtptr is scan of nnz_per_row.
    rowptr[0] = 0;
    for(index_t i = 0; i < m_loc; i++)
        rowptr[i+1] = rowptr[i] + nnz_per_row[i];

//    A->print_entry(-1);
//    print_vector(rowptr, -1, "rowptr", comm);
//    print_vector(colind, -1, "colind", comm);
//    print_vector(nzval_loc, -1, "nzval_loc", comm);
//    if(rank==0){
//        printf("\nmatrix entries in row-major format to be passed to SuperLU:\n");
//        for(nnz_t i = 0; i < nnz_loc; i++)
//            printf("%ld \t%d \t%lld \t%lf \n", i, entry_temp[i].row-fst_row, colind[i], nzval_loc[i]);
//        printf("\nrowptr:\n");
//        for(nnz_t i = 0; i < m_loc+1; i++)
//            printf("%ld \t%lld \n", i, rowptr[i]);
//    }

    dCreate_CompRowLoc_Matrix_dist(&A_SLU, m, n, nnz_loc, m_loc, fst_row,
                                   &nzval_loc[0], &colind[0], &rowptr[0],
                                   SLU_NR_loc, SLU_D, SLU_GE);

//    dcreate_matrix(&A_SLU, nrhs, &b, &ldb, &xtrue, &ldx, fp, &grid);

    if ( !(berr = doubleMalloc_dist(nrhs)) )
        ABORT("Malloc fails for berr[].");

    /* ------------------------------------------------------------
       SET THE RIGHT HAND SIDE.
       ------------------------------------------------------------*/

    b = &rhs[0];
    u = rhs;

    /* ------------------------------------------------------------
       .
       ------------------------------------------------------------*/

    /* Set the default input options:
        options.Fact              = DOFACT;
        options.Equil             = YES;
        options.ParSymbFact       = NO;
        options.ColPerm           = METIS_AT_PLUS_A;
        options.RowPerm           = LargeDiag_MC64;
        options.ReplaceTinyPivot  = NO;
        options.IterRefine        = DOUBLE;
        options.Trans             = NOTRANS;
        options.SolveInitialized  = NO;
        options.RefineInitialized = NO;
        options.PrintStat         = YES; -> I changed this to NO.
     */

    // I changed options->PrintStat default to NO.
    set_default_options_dist(&options);
    options.ColPerm = NATURAL;
//    options.SymPattern = YES;

#if 0
    options.RowPerm = NOROWPERM;
    options.RowPerm = LargeDiag_AWPM;
    options.IterRefine = NOREFINE;
    options.ColPerm = NATURAL;
    options.Equil = NO;
    options.ReplaceTinyPivot = YES;
#endif

    if (verbose_solve_coarse && !iam) {
        print_sp_ienv_dist(&options);
        print_options_dist(&options);
        fflush(stdout);
    }

//    m = A_SLU.nrow;
//    n = A_SLU.ncol;

    if(verbose_solve_coarse && rank==0) printf("SOLVE THE LINEAR SYSTEM. \n");

    // Initialize ScalePermstruct and LUstruct.
    ScalePermstructInit(m, n, &ScalePermstruct);
    LUstructInit(n, &LUstruct);

    if(verbose_solve_coarse && rank==0) printf("SOLVE THE LINEAR SYSTEM: step 1 \n");

    // Initialize the statistics variables.
    PStatInit(&stat);
    // Call the linear equation solver.
    pdgssvx(&options, &A_SLU, &ScalePermstruct, b, ldb, nrhs, &grid,
            &LUstruct, &SOLVEstruct, berr, &stat, &info);

    if(verbose_solve_coarse && rank==0) printf("SOLVE THE LINEAR SYSTEM: step 2 \n");

    // put the solution in u
    // b points to rhs. after calling pdgssvx it will contain the solution.
    u.swap(rhs);

//    print_vector(u, -1, "u computed in superlu", comm);

    if(verbose_solve_coarse && rank==0) printf("SOLVE THE LINEAR SYSTEM: step 3 \n");

    // Check the accuracy of the solution.
//    pdinf_norm_error(iam, ((NRformat_loc *)A_SLU.Store)->m_loc,
//                     nrhs, b, ldb, xtrue, ldx, &grid);

    if(verbose_solve_coarse) PStatPrint(&options, &stat, &grid); // Print the statistics.

    /* ------------------------------------------------------------
       DEALLOCATE STORAGE.
       ------------------------------------------------------------*/

    if(verbose_solve_coarse && rank==0) printf("DEALLOCATE STORAGE. \n");

    PStatFree(&stat);
    Destroy_CompRowLoc_Matrix_dist(&A_SLU);
    ScalePermstructFree(&ScalePermstruct);
    Destroy_LU(n, &grid, &LUstruct);
    LUstructFree(&LUstruct);
    if ( options.SolveInitialized ) {
        dSolveFinalize(&options, &SOLVEstruct);
    }
    SUPERLU_FREE(berr);

    // don't need these two.
//    SUPERLU_FREE(b);
//    SUPERLU_FREE(xtrue);

    /* ------------------------------------------------------------
       RELEASE THE SUPERLU PROCESS GRID.
       ------------------------------------------------------------*/
    out:
    superlu_gridexit(&grid);

    /* ------------------------------------------------------------
       TERMINATES THE MPI EXECUTION ENVIRONMENT.
       ------------------------------------------------------------*/
//    MPI_Finalize();

#if ( DEBUGlevel>=1 )
    CHECK_MALLOC(iam, "Exit main()");
#endif

    if(verbose_solve_coarse && rank==0) printf("end of solve_coarsest_SuperLU()\n");

    return 0;
}


// int SaenaObject::solveCoarsest
/*
int SaenaObject::solveCoarsest(SaenaMatrix* A, std::vector<double>& x, std::vector<double>& b, int& max_iter, double& tol, MPI_Comm comm){
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    long i, j;

    double normb_l, normb;
    normb_l = 0;
    for(i=0; i<A->M; i++)
        normb_l += b[i] * b[i];
    MPI_Allreduce(&normb_l, &normb, 1, MPI_DOUBLE, MPI_SUM, comm);
    normb = sqrt(normb);
//    if(rank==1) std::cout << normb << std::endl;

//    Vector r = b - A*x;
    std::vector<double> matvecTemp(A->M);
    A->matvec(&*x.begin(), &*matvecTemp.begin(), comm);
//    if(rank==1)
//        for(i=0; i<matvecTemp.size(); i++)
//            std::cout << matvecTemp[i] << std::endl;

    std::vector<double> r(A->M);
    for(i=0; i<matvecTemp.size(); i++)
        r[i] = b[i] - matvecTemp[i];

    if (normb == 0.0)
        normb = 1;

    double resid_l, resid;
    resid_l = 0;
    for(i=0; i<A->M; i++)
        resid_l += r[i] * r[i];
    MPI_Allreduce(&resid_l, &resid, 1, MPI_DOUBLE, MPI_SUM, comm);
    resid = sqrt(resid_l);

    if ((resid / normb) <= tol) {
        tol = resid;
        max_iter = 0;
        return 0;
    }

    double alpha, beta, rho, rho1, tempDot;
    std::vector<double> z(A->M);
    std::vector<double> p(A->M);
    std::vector<double> q(A->M);
    for (i = 0; i < max_iter; i++) {
//        z = M.solve(r);
        // todo: write this part.

//        rho(0) = dot(r, z);
        rho = 0;
        for(j = 0; j < A->M; j++)
            rho += r[j] * z[j];

//        if (i == 1)
//            p = z;
//        else {
//            beta(0) = rho(0) / rho_1(0);
//            p = z + beta(0) * p;
//        }

        if(i == 0)
            p = z;
        else{
            beta = rho / rho1;
            for(j = 0; j < A->M; j++)
                p[j] = z[j] + (beta * p[j]);
        }

//        q = A*p;
        A->matvec(&*p.begin(), &*q.begin(), comm);

//        alpha(0) = rho(0) / dot(p, q);
        tempDot = 0;
        for(j = 0; j < A->M; j++)
            tempDot += p[j] * q[j];
        alpha = rho / tempDot;

//        x += alpha(0) * p;
//        r -= alpha(0) * q;
        for(j = 0; j < A->M; j++){
            x[j] += alpha * p[j];
            r[j] -= alpha * q[j];
        }

        resid_l = 0;
        for(j = 0; j < A->M; j++)
            resid_l += r[j] * r[j];
        MPI_Allreduce(&resid_l, &resid, 1, MPI_DOUBLE, MPI_SUM, comm);
        resid = sqrt(resid_l);

        if ((resid / normb) <= tol) {
            tol = resid;
            max_iter = i;
            return 0;
        }

        rho1 = rho;
    }

    return 0;
}
*/


int saena_object::smooth(Grid* grid, std::string smoother, std::vector<value_t>& u, std::vector<value_t>& rhs, int iter){$
    std::vector<value_t> temp1(u.size());
    std::vector<value_t> temp2(u.size());

    if(smoother == "jacobi"){
        grid->A->jacobi(iter, u, rhs, temp1);
    }else if(smoother == "chebyshev"){
        grid->A->chebyshev(iter, u, rhs, temp1, temp2);
    }else{
        printf("Error: Unknown smoother");
        MPI_Finalize();
        return -1;
    }

    return 0;
}


int saena_object::vcycle(Grid* grid, std::vector<value_t>& u, std::vector<value_t>& rhs){$

    if(grid->A->active) {
        MPI_Comm comm = grid->A->comm;
        int rank, nprocs;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);

//        print_vector(rhs, -1, "rhs in vcycle", comm);

        double t1, t2;
        value_t dot;
        std::string func_name;
        std::vector<value_t> res;
        std::vector<value_t> res_coarse;
        std::vector<value_t> uCorrCoarse;
        std::vector<value_t> uCorr;
        std::vector<value_t> temp;

        if(verbose_vcycle){
            MPI_Barrier(comm);
            if(rank==0) printf("rank = %d: vcycle level = %d, A->M = %u, u.size = %lu, rhs.size = %lu \n",
                   rank, grid->currentLevel, grid->A->M, u.size(), rhs.size());
            MPI_Barrier(comm);}

        // **************************************** 0. direct-solve the coarsest level ****************************************

        if (grid->currentLevel == max_level) {
            if(verbose_vcycle){
                MPI_Barrier(comm);
                if(rank==0) std::cout << "vcycle: solving the coarsest level using " << direct_solver << std::endl;
                MPI_Barrier(comm);}

            res.resize(grid->A->M);

            t1 = omp_get_wtime();

            if(direct_solver == "CG")
                solve_coarsest_CG(grid->A, u, rhs);
            else if(direct_solver == "SuperLU")
                solve_coarsest_SuperLU(grid->A, u, rhs);
            else {
                if (rank == 0) printf("Error: Unknown direct solver is chosen! \n");
                MPI_Finalize();
                return -1;
            }

            // scale the solution u
            // -------------------------
//            scale_vector(u, grid->A->inv_sq_diag);

            t2 = omp_get_wtime();
            func_name = "vcycle: level " + std::to_string(grid->currentLevel) + ": solve coarsest";
            if (verbose) print_time(t1, t2, func_name, comm);

            if(verbose_vcycle_residuals){
                grid->A->residual(u, rhs, res);
                dotProduct(res, res, &dot, comm);
                if(rank==0) std::cout << "\nlevel = " << grid->currentLevel
                                      << ", after coarsest level = " << sqrt(dot) << std::endl;
            }

            // print the solution
            // ------------------
//            print_vector(u, -1, "solution from the direct solver", grid->A->comm);

            // check if the solution is correct
            // --------------------------------
//            std::vector<double> rhs_matvec(u.size(), 0);
//            grid->A->matvec(u, rhs_matvec);
//            if(rank==0){
//                printf("\nA*u - rhs:\n");
//                for(i = 0; i < rhs_matvec.size(); i++){
//                    if(rhs_matvec[i] - rhs[i] > 1e-6)
//                        printf("%lu \t%f - %f = \t%f \n", i, rhs_matvec[i], rhs[i], rhs_matvec[i] - rhs[i]);}
//                printf("-----------------------\n");}

            return 0;
        }

        res.resize(grid->A->M);
        uCorr.resize(grid->A->M);
        temp.resize(grid->A->M);

        if(verbose_vcycle_residuals){
            grid->A->residual(u, rhs, res);
            dotProduct(res, res, &dot, comm);
            if(rank==0) std::cout << "\nlevel = " << grid->currentLevel << ", vcycle start      = " << sqrt(dot) << std::endl;
        }

        // **************************************** 1. pre-smooth ****************************************

        if(verbose_vcycle){
            MPI_Barrier(comm);
            if(rank==0) printf("vcycle: presmooth\n");
            MPI_Barrier(comm);}

//        MPI_Barrier(grid->A->comm);
        t1 = omp_get_wtime();

        if(preSmooth)
            smooth(grid, smoother, u, rhs, preSmooth);

        t2 = omp_get_wtime();
        func_name = "Vcycle: level " + std::to_string(grid->currentLevel) + ": pre";
        if (verbose) print_time(t1, t2, func_name, comm);

//        print_vector(u, -1, "u in vcycle", comm);
//        if(rank==0) std::cout << "\n1. pre-smooth: u, currentLevel = " << grid->currentLevel << std::endl;

        // **************************************** 2. compute residual ****************************************

        if(verbose_vcycle){
            MPI_Barrier(comm);
            if(rank==0) printf("vcycle: residual\n");
            MPI_Barrier(comm);}

        grid->A->residual(u, rhs, res);

//        print_vector(res, -1, "res", comm);

        if(verbose_vcycle_residuals){
            dotProduct(res, res, &dot, comm);
            if(rank==0) std::cout << "level = " << grid->currentLevel << ", after pre-smooth  = " << sqrt(dot) << std::endl;
        }

        // **************************************** 3. restrict ****************************************

        if(verbose_vcycle){
            MPI_Barrier(comm);
            if(rank==0) printf("vcycle: restrict\n");
            printf("grid->Ac.M_old = %u \n", grid->Ac.M_old);
            MPI_Barrier(comm);}

        t1 = omp_get_wtime();

        res_coarse.resize(grid->Ac.M_old);
        grid->R.matvec(res, res_coarse);

//        grid->R.print_entry(-1);
//        print_vector(res_coarse, -1, "res_coarse in vcycle", comm);
//        MPI_Barrier(comm); printf(" res.size() = %lu \tres_coarse.size() = %lu \n", res.size(), res_coarse.size()); MPI_Barrier(comm);

        if(grid->Ac.active_minor) {
            comm = grid->Ac.comm;
            MPI_Comm_size(comm, &nprocs);
            MPI_Comm_rank(comm, &rank);

            if (verbose_vcycle) {
                MPI_Barrier(comm);
                if (rank == 0) printf("vcycle: repartition_u_shrink\n");
                MPI_Barrier(comm);
            }

//            MPI_Barrier(comm); printf("before repartition_u_shrink: res_coarse.size = %ld \n", res_coarse.size()); MPI_Barrier(comm);
//            if (grid->Ac.shrinked && nprocs > 1)
//                repartition_u_shrink(res_coarse, *grid);
            if (nprocs > 1)
                repartition_u_shrink(res_coarse, *grid);
//            MPI_Barrier(comm); printf("after  repartition_u_shrink: res_coarse.size = %ld \n", res_coarse.size()); MPI_Barrier(comm);

            t2 = omp_get_wtime();
            func_name = "Vcycle: level " + std::to_string(grid->currentLevel) + ": restriction";
            if (verbose) print_time(t1, t2, func_name, comm);

//	      print_vector(res_coarse, 0, "res_coarse", comm);

            // **************************************** 4. recurse ****************************************

            if (verbose_vcycle) {
                MPI_Barrier(comm);
                if (rank == 0) printf("vcycle: recurse\n");
                MPI_Barrier(comm);
            }

            // scale rhs of the next level
            scale_vector(res_coarse, grid->coarseGrid->A->inv_sq_diag);

            uCorrCoarse.assign(grid->Ac.M, 0);
            vcycle(grid->coarseGrid, uCorrCoarse, res_coarse);

            // scale u
            scale_vector(uCorrCoarse, grid->coarseGrid->A->inv_sq_diag);

//        if(rank==0) std::cout << "\n4. uCorrCoarse, currentLevel = " << grid->currentLevel
//                              << ", uCorrCoarse.size = " << uCorrCoarse.size() << std::endl;
//        print_vector(uCorrCoarse, -1, "uCorrCoarse", grid->A->comm);
        }
        // **************************************** 5. prolong ****************************************

        comm = grid->A->comm;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);

        t1 = omp_get_wtime();

        if (verbose_vcycle) {
            MPI_Barrier(comm);
            if (rank == 0) printf("vcycle: repartition_back_u_shrink\n");
            MPI_Barrier(comm);
        }

        if(nprocs > 1 && grid->Ac.active_minor)
            repartition_back_u_shrink(uCorrCoarse, *grid);

//        print_vector(uCorrCoarse, -1, "uCorrCoarse", grid->A->comm);

        if(verbose_vcycle){
            MPI_Barrier(comm);
            if(rank==0) printf("vcycle: prolong\n");
            MPI_Barrier(comm);}

        uCorr.resize(grid->A->M);
        grid->P.matvec(uCorrCoarse, uCorr);

        t2 = omp_get_wtime();
        func_name = "Vcycle: level " + std::to_string(grid->currentLevel) + ": prolongation";
        if (verbose) print_time(t1, t2, func_name, comm);

//        if(rank==0)
//            std::cout << "\n5. prolongation: uCorr = P*uCorrCoarse , currentLevel = " << grid->currentLevel
//                      << ", uCorr.size = " << uCorr.size() << std::endl;
//        print_vector(uCorr, -1, "uCorr", grid->A->comm);

        // **************************************** 6. correct ****************************************

        if(verbose_vcycle){
            MPI_Barrier(comm);
            if(rank==0) printf("vcycle: correct\n");
            MPI_Barrier(comm);}

#pragma omp parallel for
        for (index_t i = 0; i < u.size(); i++)
            u[i] -= uCorr[i];

//        print_vector(u, 0, "u after correction", grid->A->comm);

        if(verbose_vcycle_residuals){
            grid->A->residual(u, rhs, res);
            dotProduct(res, res, &dot, comm);
            if(rank==0) std::cout << "level = " << grid->currentLevel << ", after correction  = " << sqrt(dot) << std::endl;
        }

        // **************************************** 7. post-smooth ****************************************

        if(verbose_vcycle){
            MPI_Barrier(comm);
            if(rank==0) printf("vcycle: post-smooth\n");
            MPI_Barrier(comm);}

        t1 = omp_get_wtime();

        if(postSmooth)
            smooth(grid, smoother, u, rhs, postSmooth);

        t2 = omp_get_wtime();
        func_name = "Vcycle: level " + std::to_string(grid->currentLevel) + ": post";
        if (verbose) print_time(t1, t2, func_name, comm);

//        if(rank==1) std::cout << "\n7. post-smooth: u, currentLevel = " << grid->currentLevel << std::endl;
//        print_vector(u, 0, "u post-smooth", grid->A->comm);

        if(verbose_vcycle_residuals){
            grid->A->residual(u, rhs, res);
            dotProduct(res, res, &dot, comm);
            if(rank==0) std::cout << "level = " << grid->currentLevel << ", after post-smooth = " << sqrt(dot) << std::endl;
        }

    } // end of if(active)

    return 0;
}


int saena_object::solve(std::vector<value_t>& u){

    MPI_Comm comm = grids[0].A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // ************** check u size **************

    index_t u_size_local = u.size(), u_size_total;
    MPI_Allreduce(&u_size_local, &u_size_total, 1, MPI_UNSIGNED, MPI_SUM, grids[0].A->comm);
    if(grids[0].A->Mbig != u_size_total){
        if(rank==0) printf("Error: size of LHS (=%u) and the solution vector u (=%u) are not equal!\n", grids[0].A->Mbig, u_size_total);
        MPI_Finalize();
        return -1;
    }

    // ************** repartition u **************

    if(repartition)
        repartition_u(u);

    // ************** solve **************

//    double temp;
//    current_dot(rhs, rhs, &temp, comm);
//    if(rank==0) std::cout << "norm(rhs) = " << sqrt(temp) << std::endl;

    std::vector<value_t > r(grids[0].A->M);
    grids[0].A->residual(u, grids[0].rhs, r);
    double initial_dot, current_dot;
    dotProduct(r, r, &initial_dot, comm);
    if(rank==0) std::cout << "******************************************************" << std::endl;
    if(rank==0) printf("\ninitial residual = %e \n\n", sqrt(initial_dot));

    // if max_level==0, it means only direct solver is being used.
    if(max_level == 0)
        printf("\nonly using the direct solver! \n");

//    if(rank==0){
//        printf("Vcycle #: \tabsolute residual\n");
//        printf("-----------------------------\n");
//    }

    int i;
    for(i=0; i<vcycle_num; i++){
        vcycle(&grids[0], u, grids[0].rhs);
        grids[0].A->residual(u, grids[0].rhs, r);
        dotProduct(r, r, &current_dot, comm);

//        if(rank==0) printf("Vcycle %d: \t%.10f \n", i, sqrt(current_dot));
//        if(rank==0) printf("vcycle iteration = %d, residual = %f \n\n", i, sqrt(current_dot));
        if( current_dot/initial_dot < relative_tolerance * relative_tolerance )
            break;
    }

    // set number of iterations that took to find the solution
    // only do the following if the end of the previous for loop was reached.
    if(i == vcycle_num)
        i--;

    if(rank==0){
        std::cout << "******************************************************" << std::endl;
        printf("\nfinal:\nstopped at iteration    = %d \nfinal absolute residual = %e"
                       "\nrelative residual       = %e \n\n", ++i, sqrt(current_dot), sqrt(current_dot/initial_dot));
        std::cout << "******************************************************" << std::endl;
    }

//    print_vector(u, -1, "u", comm);

    // ************** scale u **************

    scale_vector(u, grids[0].A->inv_sq_diag);

    // ************** repartition u back **************

    if(repartition)
        repartition_back_u(u);

//    print_vector(u, -1, "u", comm);

    return 0;
}


int saena_object::solve_pcg(std::vector<value_t>& u){$

    MPI_Comm comm = grids[0].A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // ************** check u size **************

    index_t u_size_local = u.size(), u_size_total;
    MPI_Allreduce(&u_size_local, &u_size_total, 1, MPI_UNSIGNED, MPI_SUM, grids[0].A->comm);
    if(grids[0].A->Mbig != u_size_total){
        if(rank==0) printf("Error: size of LHS (=%u) and the solution vector u (=%u) are not equal!\n", grids[0].A->Mbig, u_size_total);
        MPI_Finalize();
        return -1;
    }

    if(verbose_solve) if(rank == 0) printf("verbose:solve_pcg: check u size!\n");

    // ************** repartition u **************

    if(repartition)
        repartition_u(u);

    if(verbose_solve) if(rank == 0) printf("verbose:solve_pcg: repartition u!\n");

    // ************** solve **************

//    double temp;
//    dot(rhs, rhs, &temp, comm);
//    if(rank==0) std::cout << "norm(rhs) = " << sqrt(temp) << std::endl;

    std::vector<value_t> r(grids[0].A->M);
    grids[0].A->residual(u, grids[0].rhs, r);
    double initial_dot, current_dot, previous_dot;
    dotProduct(r, r, &initial_dot, comm);
    if(rank==0) std::cout << "******************************************************" << std::endl;
    if(rank==0) printf("\ninitial residual = %e \n\n", sqrt(initial_dot));

    // if max_level==0, it means only direct solver is being used inside the previous vcycle, and that is all needed.
    if(max_level == 0){
        vcycle(&grids[0], u, grids[0].rhs);
//        grids[0].A->print_entry(-1);
        grids[0].A->residual(u, grids[0].rhs, r);
//        print_vector(r, -1, "res", comm);
        dotProduct(r, r, &current_dot, comm);
//        if(rank==0) std::cout << "dot = " << current_dot << std::endl;

        if(rank==0){
            std::cout << "******************************************************" << std::endl;
            printf("\nfinal:\nonly using the direct solver! \nfinal absolute residual = %e"
                           "\nrelative residual       = %e \n\n", sqrt(current_dot), sqrt(current_dot/initial_dot));
            std::cout << "******************************************************" << std::endl;
        }

        // scale the solution u
        scale_vector(u, grids[0].A->inv_sq_diag);

        // repartition u back
        if(repartition)
            repartition_back_u(u);

        return 0;
    }

    std::vector<value_t> rho(grids[0].A->M, 0);
    vcycle(&grids[0], rho, r);

    if(verbose_solve) if(rank == 0) printf("verbose:solve_pcg: first vcycle!\n");

//    for(i = 0; i < r.size(); i++)
//        printf("rho[%lu] = %f,\t r[%lu] = %f \n", i, rho[i], i, r[i]);

//    if(rank==0){
//        printf("Vcycle #: absolute residual \tconvergence factor\n");
//        printf("--------------------------------------------------------\n");
//    }

    std::vector<value_t> h(grids[0].A->M);
    std::vector<value_t> p(grids[0].A->M);
    p = rho;

    int i;
    previous_dot = initial_dot;
    current_dot  = initial_dot;
    double rho_res, pdoth, alpha, beta;
    for(i = 0; i < vcycle_num; i++){
        grids[0].A->matvec(p, h);
        dotProduct(r, rho, &rho_res, comm);
        dotProduct(p, h, &pdoth, comm);
        alpha = rho_res / pdoth;
//        printf("rho_res = %e, pdoth = %e, alpha = %f \n", rho_res, pdoth, alpha);

#pragma omp parallel for
        for(index_t j = 0; j < u.size(); j++){
//            if(rank==0) printf("before u = %.10f \tp = %.10f \talpha = %f \n", u[j], p[j], alpha);
            u[j] -= alpha * p[j];
            r[j] -= alpha * h[j];
//            if(rank==0) printf("after  u = %.10f \tp = %.10f \talpha = %f \n", u[j], p[j], alpha);
        }

//        print_vector(u, -1, "v inside solve_pcg", grids[0].A->comm);

        previous_dot = current_dot;
        dotProduct(r, r, &current_dot, comm);
        // print the "absolute residual" and the "convergence factor":
        if(rank==0) printf("Vcycle %d: %.10f  \t%.10f \n", i+1, sqrt(current_dot), sqrt(current_dot/previous_dot));
//        if(rank==0) printf("Vcycle %lu: aboslute residual = %.10f \n", i+1, sqrt(current_dot));
        if( current_dot/initial_dot < relative_tolerance * relative_tolerance )
            break;

        if(verbose) if(rank==0) printf("_______________________________ \n\n***** Vcycle %u *****\n", i+1);
        std::fill(rho.begin(), rho.end(), 0);
        vcycle(&grids[0], rho, r);
        dotProduct(r, rho, &beta, comm);
        beta /= rho_res;

#pragma omp parallel for
        for(index_t j = 0; j < u.size(); j++)
            p[j] = rho[j] + beta * p[j];
//        printf("beta = %e \n", beta);
    }

    // set number of iterations that took to find the solution
    // only do the following if the end of the previous for loop was reached.
    if(i == vcycle_num)
        i--;

    if(rank==0){
        std::cout << "\n******************************************************" << std::endl;
        printf("\nfinal:\nstopped at iteration    = %d \nfinal absolute residual = %e"
                       "\nrelative residual       = %e \n\n", i+1, sqrt(current_dot), sqrt(current_dot/initial_dot));
        std::cout << "******************************************************" << std::endl;
    }

    if(verbose_solve) if(rank == 0) printf("verbose:solve_pcg: solve!\n");

    // ************** scale u **************

    scale_vector(u, grids[0].A->inv_sq_diag);

    // ************** repartition u back **************

//    print_vector(u, 2, "final u before repartition_back_u", comm);

    if(repartition)
        repartition_back_u(u);

    if(verbose_solve) if(rank == 0) printf("verbose:solve_pcg: repartition back u!\n");

//     print_vector(u, 0, "final u", comm);

    return 0;
}


// int saena_object::solve_pcg_update(std::vector<value_t>& u, saena_matrix* A_new)
/*
int saena_object::solve_pcg_update(std::vector<value_t>& u, saena_matrix* A_new){

    MPI_Comm comm = grids[0].A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    unsigned long i, j;
    bool solve_verbose = false;

    // ************** update A_new **************
    // this part is specific to solve_pcg_update(), in comparison to solve_pcg().
    // the difference between this function and solve_pcg(): residual in computed for A_new, instead of the original A,
    // so the solve stops when it reaches the same tolerance as the norm of the residual for A_new.

    grids[0].A_new = A_new;

    // ************** check u size **************

    unsigned int u_size_local = u.size();
    unsigned int u_size_total;
    MPI_Allreduce(&u_size_local, &u_size_total, 1, MPI_UNSIGNED, MPI_SUM, grids[0].A->comm);
    if(grids[0].A->Mbig != u_size_total){
        if(rank==0) printf("Error: size of LHS (=%u) and the solution vector u (=%u) are not equal!\n", grids[0].A->Mbig, u_size_total);
        MPI_Finalize();
        return -1;
    }

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: check u size!\n");

    // ************** repartition u **************

    if(repartition)
        repartition_u(u);

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: repartition u!\n");

    // ************** solve **************

//    double temp;
//    dot(rhs, rhs, &temp, comm);
//    if(rank==0) std::cout << "norm(rhs) = " << sqrt(temp) << std::endl;

    std::vector<double> res(grids[0].A->M);
    grids[0].A_new->residual(u, grids[0].rhs, res);
    double initial_dot, current_dot;
    dotProduct(res, res, &initial_dot, comm);
    if(rank==0) std::cout << "******************************************************" << std::endl;
    if(rank==0) printf("\ninitial residual = %e \n\n", sqrt(initial_dot));

    // if max_level==0, it means only direct solver is being used inside the previous vcycle, and that is all needed.
    if(max_level == 0){

        vcycle(&grids[0], u, grids[0].rhs);
        grids[0].A_new->residual(u, grids[0].rhs, res);
        dotProduct(res, res, &current_dot, comm);

        if(rank==0){
            std::cout << "******************************************************" << std::endl;
            printf("\nfinal:\nonly using the direct solver! \nfinal absolute residual = %e"
                           "\nrelative residual       = %e \n\n", sqrt(current_dot), sqrt(current_dot/initial_dot));
            std::cout << "******************************************************" << std::endl;
        }

        // repartition u back
        if(repartition)
            repartition_back_u(u);

        return 0;
    }

    std::vector<double> rho(grids[0].A->M, 0);
    vcycle(&grids[0], rho, res);

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: first vcycle!\n");

//    for(i = 0; i < res.size(); i++)
//        printf("rho[%lu] = %f,\t res[%lu] = %f \n", i, rho[i], i, res[i]);

    std::vector<double> h(grids[0].A->M);
    std::vector<double> p(grids[0].A->M);
    p = rho;

    double rho_res, pdoth, alpha, beta;
    for(i=0; i<vcycle_num; i++){
        grids[0].A_new->matvec(p, h);
        dotProduct(res, rho, &rho_res, comm);
        dotProduct(p, h, &pdoth, comm);
        alpha = rho_res / pdoth;
//        printf("rho_res = %e, pdoth = %e, alpha = %f \n", rho_res, pdoth, alpha);

#pragma omp parallel for
        for(j = 0; j < u.size(); j++){
            u[j] -= alpha * p[j];
            res[j] -= alpha * h[j];
        }

        dotProduct(res, res, &current_dot, comm);
        if( current_dot/initial_dot < relative_tolerance * relative_tolerance )
            break;

        rho.assign(rho.size(), 0);
        vcycle(&grids[0], rho, res);
        dotProduct(res, rho, &beta, comm);
        beta /= rho_res;

#pragma omp parallel for
        for(j = 0; j < u.size(); j++)
            p[j] = rho[j] + beta * p[j];
//        printf("beta = %e \n", beta);
    }

    // set number of iterations that took to find the solution
    // only do the following if the end of the previous for loop was reached.
    if(i == vcycle_num)
        i--;

    if(rank==0){
        std::cout << "******************************************************" << std::endl;
        printf("\nfinal:\nstopped at iteration    = %ld \nfinal absolute residual = %e"
                       "\nrelative residual       = %e \n\n", ++i, sqrt(current_dot), sqrt(current_dot/initial_dot));
        std::cout << "******************************************************" << std::endl;
    }

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: solve!\n");

    // ************** repartition u back **************

    if(repartition)
        repartition_back_u(u);

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: repartition back u!\n");

    return 0;
}
*/


int saena_object::solve_pcg_update1(std::vector<value_t>& u, saena_matrix* A_new){

    MPI_Comm comm = grids[0].A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    bool solve_verbose = false;

    // ************** update grids[0].A **************
// this part is specific to solve_pcg_update2(), in comparison to solve_pcg().
// the difference between this function and solve_pcg(): the finest level matrix (original LHS) is updated with
// the new one.

    // first set A_new.eig_max_of_invdiagXA equal to the previous A's. Since we only need an upper bound, this is good enough.
    A_new->eig_max_of_invdiagXA = grids[0].A->eig_max_of_invdiagXA;

    grids[0].A = A_new;

    // ************** check u size **************

    index_t u_size_local = u.size(), u_size_total;
    MPI_Allreduce(&u_size_local, &u_size_total, 1, MPI_UNSIGNED, MPI_SUM, grids[0].A->comm);
    if(grids[0].A->Mbig != u_size_total){
        if(rank==0) printf("Error: size of LHS (=%u) and the solution vector u (=%u) are not equal!\n", grids[0].A->Mbig, u_size_total);
        MPI_Finalize();
        return -1;
    }

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: check u size!\n");

    // ************** repartition u **************

    if(repartition)
        repartition_u(u);

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: repartition u!\n");

    // ************** solve **************

//    double temp;
//    dot(rhs, rhs, &temp, comm);
//    if(rank==0) std::cout << "norm(rhs) = " << sqrt(temp) << std::endl;

    std::vector<value_t> r(grids[0].A->M);
    grids[0].A->residual(u, grids[0].rhs, r);
    double initial_dot, current_dot, previous_dot;
    dotProduct(r, r, &initial_dot, comm);
    if(rank==0) std::cout << "******************************************************" << std::endl;
    if(rank==0) printf("\nsolve_pcg_update1\n");
    if(rank==0) printf("\ninitial residual = %e \n\n", sqrt(initial_dot));

    // if max_level==0, it means only direct solver is being used inside the previous vcycle, and that is all needed.
    if(max_level == 0){
        vcycle(&grids[0], u, grids[0].rhs);
        grids[0].A->residual(u, grids[0].rhs, r);
        dotProduct(r, r, &current_dot, comm);

        if(rank==0){
            std::cout << "******************************************************" << std::endl;
            printf("\nfinal:\nonly using the direct solver! \nfinal absolute residual = %e"
                           "\nrelative residual       = %e \n\n", sqrt(current_dot), sqrt(current_dot/initial_dot));
            std::cout << "******************************************************" << std::endl;
        }

        // scale the solution u
        scale_vector(u, grids[0].A->inv_sq_diag);

        // repartition u back
        if(repartition)
            repartition_back_u(u);

        return 0;
    }

    std::vector<value_t> rho(grids[0].A->M, 0);
    vcycle(&grids[0], rho, r);

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: first vcycle!\n");

//    for(i = 0; i < r.size(); i++)
//        printf("rho[%lu] = %f,\t r[%lu] = %f \n", i, rho[i], i, r[i]);

    std::vector<value_t> h(grids[0].A->M);
    std::vector<value_t> p(grids[0].A->M);
    p = rho;

    int i;
    previous_dot = initial_dot;
    current_dot  = initial_dot;
    double rho_res, pdoth, alpha, beta;
    for(i = 0; i < vcycle_num; i++){
        grids[0].A->matvec(p, h);
        dotProduct(r, rho, &rho_res, comm);
        dotProduct(p, h, &pdoth, comm);
        alpha = rho_res / pdoth;
//        printf("rho_res = %e, pdoth = %e, alpha = %f \n", rho_res, pdoth, alpha);

#pragma omp parallel for
        for(index_t j = 0; j < u.size(); j++){
            u[j] -= alpha * p[j];
            r[j] -= alpha * h[j];
        }

        previous_dot = current_dot;
        dotProduct(r, r, &current_dot, comm);
        // this prints the "absolute residual" and the "convergence factor":
//        if(rank==0) printf("Vcycle %d: %.10f  \t%.10f \n", i+1, sqrt(current_dot), sqrt(current_dot/previous_dot));
//        if(rank==0) printf("Vcycle %lu: aboslute residual = %.10f \n", i+1, sqrt(current_dot));
        if( current_dot/initial_dot < relative_tolerance * relative_tolerance )
            break;

        if(verbose) if(rank==0) printf("_______________________________ \n\n***** Vcycle %u *****\n", i+1);
        std::fill(rho.begin(), rho.end(), 0);
        vcycle(&grids[0], rho, r);
        dotProduct(r, rho, &beta, comm);
        beta /= rho_res;

#pragma omp parallel for
        for(index_t j = 0; j < u.size(); j++)
            p[j] = rho[j] + beta * p[j];
//        printf("beta = %e \n", beta);
    }

    // set number of iterations that took to find the solution
    // only do the following if the end of the previous for loop was reached.
    if(i == vcycle_num)
        i--;

    if(rank==0){
        std::cout << "******************************************************" << std::endl;
        printf("\nfinal:\nstopped at iteration    = %d \nfinal absolute residual = %e"
                       "\nrelative residual       = %e \n\n", i+1, sqrt(current_dot), sqrt(current_dot/initial_dot));
        std::cout << "******************************************************" << std::endl;
    }

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: solve!\n");

    // ************** scale u **************

    scale_vector(u, grids[0].A->inv_sq_diag);

    // ************** repartition u back **************

    if(repartition)
        repartition_back_u(u);

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: repartition back u!\n");

    return 0;
}


int saena_object::solve_pcg_update2(std::vector<value_t>& u, saena_matrix* A_new){

    MPI_Comm comm = grids[0].A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    unsigned long i, j;
    bool solve_verbose = false;

    // ************** update grids[i].A for all levels i **************

    // first set A_new.eig_max_of_invdiagXA equal to the previous A's. Since we only need an upper bound, this is good enough.
    // do the same for the next level matrices.
    A_new->eig_max_of_invdiagXA = grids[0].A->eig_max_of_invdiagXA;

    double eigen_temp;
    grids[0].A = A_new;
    for(i = 0; i < max_level; i++){
        if(grids[i].A->active) {
            eigen_temp = grids[i].Ac.eig_max_of_invdiagXA;
            grids[i].Ac.erase();
            coarsen(&grids[i]);
            grids[i + 1].A = &grids[i].Ac;
            grids[i].Ac.eig_max_of_invdiagXA = eigen_temp;
//            Grid(&grids[i].Ac, max_level, i + 1);
        }
    }

    // ************** check u size **************

    unsigned int u_size_local = u.size();
    unsigned int u_size_total;
    MPI_Allreduce(&u_size_local, &u_size_total, 1, MPI_UNSIGNED, MPI_SUM, grids[0].A->comm);
    if(grids[0].A->Mbig != u_size_total){
        if(rank==0) printf("Error: size of LHS (=%u) and the solution vector u (=%u) are not equal!\n", grids[0].A->Mbig, u_size_total);
        MPI_Finalize();
        return -1;
    }

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: check u size!\n");

    // ************** repartition u **************
    if(repartition)
        repartition_u(u);

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: repartition u!\n");

    // ************** solve **************

//    double temp;
//    dot(rhs, rhs, &temp, comm);
//    if(rank==0) std::cout << "norm(rhs) = " << sqrt(temp) << std::endl;

    std::vector<double> r(grids[0].A->M);
    grids[0].A->residual(u, grids[0].rhs, r);
    double initial_dot, current_dot, previous_dot;
    dotProduct(r, r, &initial_dot, comm);
    if(rank==0) std::cout << "******************************************************" << std::endl;
    if(rank==0) printf("\nsolve_pcg_update2\n");
    if(rank==0) printf("\ninitial residual = %e \n\n", sqrt(initial_dot));

    // if max_level==0, it means only direct solver is being used inside the previous vcycle, and that is all needed.
    if(max_level == 0){

        vcycle(&grids[0], u, grids[0].rhs);
        grids[0].A->residual(u, grids[0].rhs, r);
        dotProduct(r, r, &current_dot, comm);

        if(rank==0){
            std::cout << "******************************************************" << std::endl;
            printf("\nfinal:\nonly using the direct solver! \nfinal absolute residual = %e"
                           "\nrelative residual       = %e \n\n", sqrt(current_dot), sqrt(current_dot/initial_dot));
            std::cout << "******************************************************" << std::endl;
        }

        // scale the solution u
        scale_vector(u, grids[0].A->inv_sq_diag);

        // repartition u back
        if(repartition)
            repartition_back_u(u);

        return 0;
    }

    std::vector<double> rho(grids[0].A->M, 0);
    vcycle(&grids[0], rho, r);

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: first vcycle!\n");

//    for(i = 0; i < r.size(); i++)
//        printf("rho[%lu] = %f,\t r[%lu] = %f \n", i, rho[i], i, r[i]);

    std::vector<double> h(grids[0].A->M);
    std::vector<double> p(grids[0].A->M);
    p = rho;

    previous_dot = initial_dot;
    current_dot  = initial_dot;
    double rho_res, pdoth, alpha, beta;
    for(i=0; i<vcycle_num; i++){
        grids[0].A->matvec(p, h);
        dotProduct(r, rho, &rho_res, comm);
        dotProduct(p, h, &pdoth, comm);
        alpha = rho_res / pdoth;
//        printf("rho_res = %e, pdoth = %e, alpha = %f \n", rho_res, pdoth, alpha);

#pragma omp parallel for
        for(j = 0; j < u.size(); j++){
            u[j] -= alpha * p[j];
            r[j] -= alpha * h[j];
        }

        previous_dot = current_dot;
        dotProduct(r, r, &current_dot, comm);
        if( current_dot/initial_dot < relative_tolerance * relative_tolerance )
            break;

        if(verbose) if(rank==0) printf("_______________________________ \n\n***** Vcycle %lu *****\n", i+1);
        // this prints the "absolute residual" and the "convergence factor":
//        if(rank==0) printf("Vcycle %lu: %.10f  \t%.10f \n", i+1, sqrt(current_dot), sqrt(current_dot/previous_dot));
        std::fill(rho.begin(), rho.end(), 0);
        vcycle(&grids[0], rho, r);
        dotProduct(r, rho, &beta, comm);
        beta /= rho_res;

#pragma omp parallel for
        for(j = 0; j < u.size(); j++)
            p[j] = rho[j] + beta * p[j];
//        printf("beta = %e \n", beta);
    }

    // set number of iterations that took to find the solution
    // only do the following if the end of the previous for loop was reached.
    if(i == vcycle_num)
        i--;

    if(rank==0){
        std::cout << "******************************************************" << std::endl;
        printf("\nfinal:\nstopped at iteration    = %ld \nfinal absolute residual = %e"
                       "\nrelative residual       = %e \n\n", ++i, sqrt(current_dot), sqrt(current_dot/initial_dot));
        std::cout << "******************************************************" << std::endl;
    }

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: solve!\n");

    // ************** scale u **************

    scale_vector(u, grids[0].A->inv_sq_diag);

    // ************** repartition u back **************

    if(repartition)
        repartition_back_u(u);

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: repartition back u!\n");

    return 0;
}


int saena_object::solve_pcg_update3(std::vector<value_t>& u, saena_matrix* A_new){

    MPI_Comm comm = grids[0].A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    unsigned long i, j;
    bool solve_verbose = false;

    // ************** update grids[i].A for all levels i **************

    // first set A_new.eig_max_of_invdiagXA equal to the previous A's. Since we only need an upper bound, this is good enough.
    // do the same for the next level matrices.
    A_new->eig_max_of_invdiagXA = grids[0].A->eig_max_of_invdiagXA;

    std::vector<cooEntry> A_diff;
    local_diff(*grids[0].A, *A_new, A_diff);
//    print_vector(A_diff, -1, "A_diff", grids[0].A->comm);
//    print_vector(grids[0].A->split, 0, "split", grids[0].A->comm);

    grids[0].A = A_new;
    for(i = 0; i < max_level; i++){
        if(grids[i].A->active) {
//            if(rank==0) printf("_____________________________________\nlevel = %lu \n", i);
//            grids[i].Ac.print_entry(-1);
            coarsen_update_Ac(&grids[i], A_diff);
//            grids[i].Ac.print_entry(-1);
//            print_vector(A_diff, -1, "A_diff", grids[i].Ac.comm);
//            print_vector(grids[i+1].A->split, 0, "split", grids[i+1].A->comm);
        }
    }

//    saena_matrix* B = grids[0].Ac->get_internal_matrix();

    // ************** check u size **************

    unsigned int u_size_local = u.size();
    unsigned int u_size_total;
    MPI_Allreduce(&u_size_local, &u_size_total, 1, MPI_UNSIGNED, MPI_SUM, grids[0].A->comm);
    if(grids[0].A->Mbig != u_size_total){
        if(rank==0) printf("Error: size of LHS (=%u) and the solution vector u (=%u) are not equal!\n", grids[0].A->Mbig, u_size_total);
        MPI_Finalize();
        return -1;
    }

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: check u size!\n");

    // ************** repartition u **************

    if(repartition)
        repartition_u(u);

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: repartition u!\n");

    // ************** solve **************

//    double temp;
//    dot(rhs, rhs, &temp, comm);
//    if(rank==0) std::cout << "norm(rhs) = " << sqrt(temp) << std::endl;

    std::vector<double> r(grids[0].A->M);
    grids[0].A->residual(u, grids[0].rhs, r);
    double initial_dot, current_dot, previous_dot;
    dotProduct(r, r, &initial_dot, comm);
    if(rank==0) std::cout << "******************************************************" << std::endl;
    if(rank==0) printf("\nsolve_pcg_update3\n");
    if(rank==0) printf("\ninitial residual = %e \n\n", sqrt(initial_dot));

    // if max_level==0, it means only direct solver is being used inside the previous vcycle, and that is all needed.
    if(max_level == 0){

        vcycle(&grids[0], u, grids[0].rhs);
        grids[0].A->residual(u, grids[0].rhs, r);
        dotProduct(r, r, &current_dot, comm);

        if(rank==0){
            std::cout << "******************************************************" << std::endl;
            printf("\nfinal:\nonly using the direct solver! \nfinal absolute residual = %e"
                           "\nrelative residual       = %e \n\n", sqrt(current_dot), sqrt(current_dot/initial_dot));
            std::cout << "******************************************************" << std::endl;
        }

        // scale the solution u
        scale_vector(u, grids[0].A->inv_sq_diag);

        // repartition u back
        if(repartition)
            repartition_back_u(u);

        return 0;
    }

    std::vector<double> rho(grids[0].A->M, 0);
    vcycle(&grids[0], rho, r);

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: first vcycle!\n");

//    for(i = 0; i < r.size(); i++)
//        printf("rho[%lu] = %f,\t r[%lu] = %f \n", i, rho[i], i, r[i]);

    std::vector<double> h(grids[0].A->M);
    std::vector<double> p(grids[0].A->M);
    p = rho;

    previous_dot = initial_dot;
    current_dot = initial_dot;
    double rho_res, pdoth, alpha, beta;
    for(i=0; i<vcycle_num; i++){
        grids[0].A->matvec(p, h);
        dotProduct(r, rho, &rho_res, comm);
        dotProduct(p, h, &pdoth, comm);
        alpha = rho_res / pdoth;
//        printf("rho_res = %e, pdoth = %e, alpha = %f \n", rho_res, pdoth, alpha);

#pragma omp parallel for
        for(j = 0; j < u.size(); j++){
            u[j] -= alpha * p[j];
            r[j] -= alpha * h[j];
        }

        previous_dot = current_dot;
        dotProduct(r, r, &current_dot, comm);
        if( current_dot/initial_dot < relative_tolerance * relative_tolerance )
            break;

        if(verbose || solve_verbose) if(rank==0) printf("_______________________________ \n\n***** Vcycle %lu *****\n", i+1);
        // this prints the "absolute residual" and the "convergence factor":
//        if(rank==0) printf("Vcycle %lu: %.10f  \t%.10f \n", i+1, sqrt(current_dot), sqrt(current_dot/previous_dot));
        std::fill(rho.begin(), rho.end(), 0);
        vcycle(&grids[0], rho, r);
        dotProduct(r, rho, &beta, comm);
        beta /= rho_res;

#pragma omp parallel for
        for(j = 0; j < u.size(); j++)
            p[j] = rho[j] + beta * p[j];
//        printf("beta = %e \n", beta);
    }

    // set number of iterations that took to find the solution
    // only do the following if the end of the previous for loop was reached.
    if(i == vcycle_num)
        i--;

    if(rank==0){
        std::cout << "******************************************************" << std::endl;
        printf("\nfinal:\nstopped at iteration    = %ld \nfinal absolute residual = %e"
                       "\nrelative residual       = %e \n\n", ++i, sqrt(current_dot), sqrt(current_dot/initial_dot));
        std::cout << "******************************************************" << std::endl;
    }

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: solve!\n");

    // ************** scale u **************

    scale_vector(u, grids[0].A->inv_sq_diag);

    // ************** repartition u back **************

    if(repartition)
        repartition_back_u(u);

    if(solve_verbose) if(rank == 0) printf("verbose: solve_pcg_update: repartition back u!\n");

    return 0;
}


int saena_object::set_repartition_rhs(std::vector<value_t>& rhs0){

    MPI_Comm comm = grids[0].A->comm;
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    // ************** check rhs size **************

    index_t rhs_size_local = rhs0.size(), rhs_size_total;
    MPI_Allreduce(&rhs_size_local, &rhs_size_total, 1, MPI_UNSIGNED, MPI_SUM, comm);
    if(grids[0].A->Mbig != rhs_size_total){
        if(rank==0) printf("Error: size of LHS (=%u) and RHS (=%u) are not equal!\n", grids[0].A->Mbig,rhs_size_total);
        MPI_Finalize();
        return -1;
    }

//    print_vector(grids[0].A->split, 1, "split", comm);

    // ************** repartition rhs, based on A.split **************

    std::vector<index_t> rhs_init_partition(nprocs);
    rhs_init_partition[rank] = rhs0.size();
    index_t temp = rhs0.size();

    MPI_Allgather(&temp, 1, MPI_UNSIGNED, &*rhs_init_partition.begin(), 1, MPI_UNSIGNED, comm);
//    MPI_Alltoall(&*grids[0].rhs_init_partition.begin(), 1, MPI_INT, &*grids[0].rhs_init_partition.begin(), 1, MPI_INT, grids[0].comm);

//    print_vector(rhs_init_partition, 1, "rhs_init_partition", comm);

    std::vector<index_t> init_partition_scan(nprocs+1);
    init_partition_scan[0] = 0;
    for(int i = 1; i < nprocs+1; i++)
        init_partition_scan[i] = init_partition_scan[i-1] + rhs_init_partition[i-1];

//    print_vector(init_partition_scan, 1, "init_partition_scan", comm);

    index_t start, end, start_proc, end_proc;
    start = grids[0].A->split[rank];
    end   = grids[0].A->split[rank+1];
    start_proc = lower_bound2(&*init_partition_scan.begin(), &*init_partition_scan.end(), start);
    end_proc   = lower_bound2(&*init_partition_scan.begin(), &*init_partition_scan.end(), end);
    if(init_partition_scan[rank+1] == grids[0].A->split[rank+1])
        end_proc--;
//    if(rank == 1) printf("\nstart_proc = %u, end_proc = %u \n", start_proc, end_proc);

    grids[0].rcount.assign(nprocs, 0);
    if(start_proc < end_proc){
//        if(rank==1) printf("start_proc = %u, end_proc = %u\n", start_proc, end_proc);
//        if(rank==1) printf("init_partition_scan[start_proc+1] = %u, grids[0].A->split[rank] = %u\n", init_partition_scan[start_proc+1], grids[0].A->split[rank]);
        grids[0].rcount[start_proc] = init_partition_scan[start_proc+1] - grids[0].A->split[rank];
        grids[0].rcount[end_proc] = grids[0].A->split[rank+1] - init_partition_scan[end_proc];

        for(int i = start_proc+1; i < end_proc; i++){
//            if(rank==ran) printf("init_partition_scan[i+1] = %lu, init_partition_scan[i] = %lu\n", init_partition_scan[i+1], init_partition_scan[i]);
            grids[0].rcount[i] = init_partition_scan[i+1] - init_partition_scan[i];
        }
    }else if(start_proc == end_proc){
//        grids[0].rcount[start_proc] = grids[0].A->split[start_proc + 1] - grids[0].A->split[start_proc];
        grids[0].rcount[start_proc] = grids[0].A->split[rank + 1] - grids[0].A->split[rank];
    }else{
        printf("error in set_repartition_rhs function: start_proc > end_proc\n");
        MPI_Finalize();
        return -1;
    }

//    print_vector(grids[0].rcount, -1, "grids[0].rcount", comm);

    start = init_partition_scan[rank];
    end   = init_partition_scan[rank+1];
    start_proc = lower_bound2(&*grids[0].A->split.begin(), &*grids[0].A->split.end(), start);
    end_proc   = lower_bound2(&*grids[0].A->split.begin(), &*grids[0].A->split.end(), end);
    if(init_partition_scan[rank+1] == grids[0].A->split[rank+1])
        end_proc--;
//    if(rank == ran) printf("\nstart_proc = %lu, end_proc = %lu \n", start_proc, end_proc);

    grids[0].scount.assign(nprocs, 0);
    if(end_proc > start_proc){
//        if(rank==1) printf("start_proc = %u, end_proc = %u\n", start_proc, end_proc);
//        if(rank==1) printf("init_partition_scan[rank+1] = %u, grids[0].A->split[end_proc] = %u\n", init_partition_scan[rank+1], grids[0].A->split[end_proc]);
        grids[0].scount[start_proc] = grids[0].A->split[start_proc+1] - init_partition_scan[rank];
        grids[0].scount[end_proc] = init_partition_scan[rank+1] - grids[0].A->split[end_proc];

        for(int i = start_proc+1; i < end_proc; i++)
            grids[0].scount[i] = grids[0].A->split[i+1] - grids[0].A->split[i];
    } else if(start_proc == end_proc)
        grids[0].scount[start_proc] = init_partition_scan[rank+1] - init_partition_scan[rank];
    else{
        printf("error in set_repartition_rhs function: start_proc > end_proc\n");
        MPI_Finalize();
        return -1;
    }

//    print_vector(grids[0].scount, -1, "grids[0].scount", comm);

//    std::vector<int> rdispls(nprocs);
    grids[0].rdispls.resize(nprocs);
    grids[0].rdispls[0] = 0;
    for(int i = 1; i < nprocs; i++)
        grids[0].rdispls[i] = grids[0].rcount[i-1] + grids[0].rdispls[i-1];

//    print_vector(grids[0].rdispls, -1, "grids[0].rdispls", comm);

//    std::vector<int> sdispls(nprocs);
    grids[0].sdispls.resize(nprocs);
    grids[0].sdispls[0] = 0;
    for(int i = 1; i < nprocs; i++)
        grids[0].sdispls[i] = grids[0].sdispls[i-1] + grids[0].scount[i-1];

//    print_vector(grids[0].sdispls, -1, "grids[0].sdispls", comm);

    // check if repartition is required. it is not required if the number of rows on all processors does not change.
    bool repartition_local = true;
    if(start_proc == end_proc)
        repartition_local = false;
    MPI_Allreduce(&repartition_local, &repartition, 1, MPI_CXX_BOOL, MPI_LOR, comm);
//    printf("rank = %d, repartition_local = %d, repartition = %d \n", rank, repartition_local, repartition);

    if(repartition){
        grids[0].rhs.resize(grids[0].A->split[rank+1] - grids[0].A->split[rank]);
        MPI_Alltoallv(&*rhs0.begin(), &grids[0].scount[0], &grids[0].sdispls[0], MPI_DOUBLE,
                      &*grids[0].rhs.begin(), &grids[0].rcount[0], &grids[0].rdispls[0], MPI_DOUBLE, comm);
    } else{
        grids[0].rhs = rhs0;
    }

//    print_vector(grids[0].rhs, -1, "rhs after repartition", comm);

    // scale rhs
    // ---------
    scale_vector(grids[0].rhs, grids[0].A->inv_sq_diag);

    return 0;
}


int saena_object::repartition_u(std::vector<value_t>& u0){

    int rank;
    MPI_Comm_rank(grids[0].A->comm, &rank);
//    MPI_Comm_size(grids[0].A->comm, &nprocs);

    // make a copy of u0 to be used in Alltoallv as sendbuf. u0 itself will be recvbuf there.
    std::vector<value_t> u_temp = u0;

    // ************** repartition u, based on A.split **************

    u0.resize(grids[0].A->split[rank+1] - grids[0].A->split[rank]);
    MPI_Alltoallv(&u_temp[0], &grids[0].scount[0], &grids[0].sdispls[0], MPI_DOUBLE,
                  &u0[0],     &grids[0].rcount[0], &grids[0].rdispls[0], MPI_DOUBLE, grids[0].A->comm);

    return 0;
}


int saena_object::repartition_back_u(std::vector<value_t>& u0){

    MPI_Comm comm = grids[0].A->comm;
    int rank, nprocs;
    MPI_Comm_rank(grids[0].A->comm, &rank);
    MPI_Comm_size(grids[0].A->comm, &nprocs);

//    print_vector(grids[0].A->split, 0, "split", comm);

    // rdispls should be the opposite of the initial repartition function. So, rdispls should be the scan of scount.
    // the same for sdispls.
    std::vector<int> rdispls(nprocs);
    rdispls[0] = 0;
    for(int i = 1; i < nprocs; i++)
        rdispls[i] = rdispls[i-1] + grids[0].scount[i-1];

//    print_vector(grids[0].scount, -1, "rec size", comm);
//    print_vector(rdispls, -1, "rdispls", grids[0].A->comm);

    std::vector<int> sdispls(nprocs);
    sdispls[0] = 0;
    for(int i = 1; i < nprocs; i++)
        sdispls[i] = sdispls[i-1] + grids[0].rcount[i-1];

//    print_vector(grids[0].rcount, -1, "send size", comm);
//    print_vector(sdispls, -1, "sdispls", grids[0].A->comm);

    index_t rhs_init_size = rdispls[nprocs-1] + grids[0].scount[nprocs-1]; // this is the summation over all rcount values on each proc.
//    printf("rank = %d, rhs_init_size = %lu \n", rank, rhs_init_size);

    // make a copy of u0 to be used in Alltoall as sendbuf. u0 itself will be recvbuf there.
    std::vector<value_t> u_temp = u0;
//    u0.clear();
    u0.resize(rhs_init_size);
//    std::fill(u0.begin(), u0.end(), -111);
//    print_vector(u_temp, 2, "u_temp", grids[0].A->comm);
    MPI_Alltoallv(&u_temp[0], &grids[0].rcount[0], &sdispls[0], MPI_DOUBLE,
                  &u0[0],     &grids[0].scount[0], &rdispls[0], MPI_DOUBLE, comm);

//    MPI_Barrier(grids[0].A->comm);
//    print_vector(u0, -1, "u after repartition_back_u", grids[0].A->comm);

    return 0;
}


int saena_object::shrink_cpu_A(saena_matrix* Ac, std::vector<index_t>& P_splitNew){

    // if number of rows on Ac < threshold*number of rows on A, then shrink.
    // redistribute Ac from processes 4k+1, 4k+2 and 4k+3 to process 4k.
    MPI_Comm comm = Ac->comm;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    index_t i;
    bool verbose_shrink = false;

//    MPI_Barrier(comm);
//    if(rank==0) printf("\n****************************\n");
//    if(rank==0) printf("***********SHRINK***********\n");
//    if(rank==0) printf("****************************\n\n");
//    MPI_Barrier(comm);

//    MPI_Barrier(comm); printf("rank = %d \tnnz_l = %u \n", rank, Ac->nnz_l); MPI_Barrier(comm);

//    MPI_Barrier(comm);
//    if(rank == 0){
//        std::cout << "\nbefore shrinking!!!" <<std::endl;
//        std::cout << "\nrank = " << rank << ", size = " << Ac->entry.size() <<std::endl;
//        for(i=0; i < Ac->entry.size(); i++)
//            std::cout << i << "\t" << Ac->entry[i]  <<std::endl;
//    }

    // assume cpu_shrink_thre2 is 4 (it is simpler to explain)
    // 1 - create a new comm, consisting only of processes 4k, 4k+1, 4k+2 and 4k+3 (with new ranks 0,1,2,3)
    int color = rank / Ac->cpu_shrink_thre2;
    MPI_Comm_split(comm, color, rank, &Ac->comm_horizontal);

    int rank_new, nprocs_new;
    MPI_Comm_size(Ac->comm_horizontal, &nprocs_new);
    MPI_Comm_rank(Ac->comm_horizontal, &rank_new);

//    MPI_Barrier(Ac->comm_horizontal);
//    printf("rank = %d, rank_new = %d on Ac->comm_horizontal \n", rank, rank_new);

    // 2 - update the number of rows on process 4k, and resize "entry".
    index_t Ac_M_neighbors_total = 0;
    unsigned int Ac_nnz_neighbors_total = 0;
    MPI_Reduce(&Ac->M, &Ac_M_neighbors_total, 1, MPI_UNSIGNED, MPI_SUM, 0, Ac->comm_horizontal);
    MPI_Reduce(&Ac->nnz_l, &Ac_nnz_neighbors_total, 1, MPI_UNSIGNED, MPI_SUM, 0, Ac->comm_horizontal);

    if(rank_new == 0){
        Ac->M = Ac_M_neighbors_total;
        Ac->entry.resize(Ac_nnz_neighbors_total);
//        printf("rank = %d, Ac_M_neighbors = %d \n", rank, Ac_M_neighbors_total);
//        printf("rank = %d, Ac_nnz_neighbors = %d \n", rank, Ac_nnz_neighbors_total);
    }

    // last cpu that its right neighbors are going be shrinked to.
    auto last_root_cpu = (unsigned int)floor(nprocs/Ac->cpu_shrink_thre2) * Ac->cpu_shrink_thre2;
//    printf("last_root_cpu = %u\n", last_root_cpu);

    int neigbor_rank;
    nnz_t A_recv_nnz = 0; // set to 0 just to avoid "not initialized" warning
    unsigned long offset = Ac->nnz_l; // put the data on root from its neighbors at the end of entry[] which is of size nnz_l
    if(nprocs_new > 1) { // if there is no neighbor, skip.
        for (neigbor_rank = 1; neigbor_rank < Ac->cpu_shrink_thre2; neigbor_rank++) {

//            if( rank == last_root_cpu && (rank + neigbor_rank >= nprocs) )
//                stop_forloop = true;
            // last row of cpus should stop to avoid passing the last neighbor cpu.
//            if( rank >= last_root_cpu){
//                MPI_Bcast(&stop_forloop, 1, MPI_CXX_BOOL, 0, Ac->comm_horizontal);
//                printf("rank = %d, neigbor_rank = %d, stop_forloop = %d \n", rank, neigbor_rank, stop_forloop);
//                if (stop_forloop)
//                    break;}

            if( rank == last_root_cpu && (rank + neigbor_rank >= nprocs) )
                break;

            // 3 - send and receive size of Ac.
            if (rank_new == 0)
                MPI_Recv(&A_recv_nnz, 1, MPI_UNSIGNED_LONG, neigbor_rank, 0, Ac->comm_horizontal, MPI_STATUS_IGNORE);

            if (rank_new == neigbor_rank)
                MPI_Send(&Ac->nnz_l, 1, MPI_UNSIGNED_LONG, 0, 0, Ac->comm_horizontal);

            // 4 - send and receive Ac.
            if (rank_new == 0) {
//                printf("rank = %d, neigbor_rank = %d, recv size = %u, offset = %lu \n", rank, neigbor_rank, A_recv_nnz, offset);
                MPI_Recv(&*(Ac->entry.begin() + offset), A_recv_nnz, cooEntry::mpi_datatype(), neigbor_rank, 1,
                         Ac->comm_horizontal, MPI_STATUS_IGNORE);
                offset += A_recv_nnz; // set offset for the next iteration
            }

            if (rank_new == neigbor_rank) {
//                printf("rank = %d, neigbor_rank = %d, send size = %u, offset = %lu \n", rank, neigbor_rank, Ac->nnz_l, offset);
                MPI_Send(&*Ac->entry.begin(), Ac->nnz_l, cooEntry::mpi_datatype(), 0, 1, Ac->comm_horizontal);
            }

            // update local index for rows
//        if(rank_new == 0)
//            for(i=0; i < A_recv_nnz; i++)
//                Ac->entry[offset + i].row += Ac->split[rank + neigbor_rank] - Ac->split[rank];
        }
    }

    // even though the entries are sorted before shrinking, after shrinking they still need to be sorted locally,
    // because remote elements damage sorting after shrinking.
    std::sort(Ac->entry.begin(), Ac->entry.end());

//    MPI_Barrier(comm); MPI_Barrier(Ac->comm_horizontal);
//    if(rank == 2){
//        std::cout << "\nafter shrinking: rank = " << rank << ", size = " << Ac->entry.size() <<std::endl;
//        for(i=0; i < Ac->entry.size(); i++)
//            std::cout << i << "\t" << Ac->entry[i]  <<std::endl;}
//    MPI_Barrier(comm); MPI_Barrier(Ac->comm_horizontal);

    Ac->active = false;
//    Ac->active_old_comm = true; // this is used for prolong and post-smooth
    if(rank_new == 0){
        Ac->active = true;
//        printf("active: rank = %d, rank_new = %d \n", rank, rank_new);
    }

    // 5 - update 4k.nnz_l and split. nnz_g stays the same, so no need to update.
/*
    if(Ac->active){
        Ac->nnz_l = Ac->entry.size();
        Ac->split_old = Ac->split; // save the old split for shrinking rhs and u
        Ac->split.clear();
        for(i = 0; i < nprocs+1; i++){
//            if(rank==0) printf("P->splitNew[i] = %lu\n", P_splitNew[i]);
            if( i % Ac->cpu_shrink_thre2 == 0){
                Ac->split.push_back( P_splitNew[i] );
            }
        }
        Ac->split.push_back( P_splitNew[nprocs] );
        // assert M == split[rank+1] - split[rank]

        print_vector(Ac->split, 0, "Ac->split after shrinking:", Ac->comm);

//        if(rank==0) {
//            printf("Ac split after shrinking: \n");
//            for (int i = 0; i < Ac->split.size(); i++)
//                printf("%lu \n", Ac->split[i]);}
    }
*/

    // 6 - create a new comm including only processes with 4k rank.
    MPI_Group bigger_group;
    MPI_Comm_group(comm, &bigger_group);
    auto total_active_procs = (unsigned int)ceil((double)nprocs / Ac->cpu_shrink_thre2); // note: this is ceiling, not floor.
    std::vector<int> ranks(total_active_procs);
    for(i = 0; i < total_active_procs; i++)
        ranks[i] = Ac->cpu_shrink_thre2 * i;

//    printf("total_active_procs = %u \n", total_active_procs);
//    print_vector(ranks, 0, "ranks", Ac->comm);

    MPI_Group group_new;
    MPI_Group_incl(bigger_group, total_active_procs, &*ranks.begin(), &group_new);
    MPI_Comm_create_group(comm, group_new, 0, &Ac->comm);

//    if(Ac->active) {
//        int rankkk;
//        MPI_Comm_rank(Ac->comm, &rankkk);
//        MPI_Barrier(Ac->comm);
//        if (rankkk == 4) {
//                std::cout << "\ninside cpu_shrink, after shrinking" << std::endl;
//            std::cout << "\nrank = " << rank << ", size = " << Ac->entry.size() << std::endl;
//            for (i = 0; i < Ac->entry.size(); i++)
//                std::cout << i << "\t" << Ac->entry[i] << std::endl;}
//        MPI_Barrier(Ac->comm);
//    }

//    Ac->split_old = Ac->split; // save the old split for shrinking rhs and u
    std::vector<index_t> split_temp = Ac->split;
    Ac->split.clear();
    if(Ac->active){
        Ac->nnz_l = Ac->entry.size();

        Ac->split.resize(total_active_procs+1);
        Ac->split.shrink_to_fit();
        Ac->split[0] = 0;
        Ac->split[total_active_procs] = Ac->Mbig;
        for(unsigned int i = 1; i < total_active_procs; i++){
//            if(rank==0) printf("%u \t%lu \n", i, split_old[ranks[i]]);
            Ac->split[i] = split_temp[ranks[i]];
        }
//        print_vector(Ac->split, 0, "Ac->split after shrinking:", comm);
    }

    // 7 - update 4k.nnz_g
//    if(Ac->active)
//        MPI_Allreduce(&Ac->nnz_l, &Ac->nnz_g, 1, MPI_UNSIGNED, MPI_SUM, Ac->comm);

//    if(Ac->active){
//        MPI_Comm_size(Ac->comm, &nprocs);
//        MPI_Comm_rank(Ac->comm, &rank);
//        printf("\n\nrank = %d, nprocs = %d, M = %u, nnz_l = %u, nnz_g = %u, Ac->split[rank+1] = %lu, Ac->split[rank] = %lu \n",
//               rank, nprocs, Ac->M, Ac->nnz_l, Ac->nnz_g, Ac->split[rank+1], Ac->split[rank]);
//    }

    // todo: how should these be freed?
//    free(&bigger_group);
//    free(&group_new);
//    free(&comm_new2);

    Ac->last_M_shrink = Ac->Mbig;
    Ac->shrinked = true;

    MPI_Barrier(comm); if(rank==0) printf("shrinking done!\n"); MPI_Barrier(comm);

    return 0;
}


// int saena_object::repartition_u_shrink_prepare
/*
int saena_object::repartition_u_shrink_prepare(std::vector<value_t> &u, Grid &grid){

    MPI_Comm comm = grid.A->comm;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    MPI_Barrier(comm);
//    if(rank == 0) printf("\nsplit_old: \n");
//    print_vector(grid.A->split_old, 0, comm);
//    MPI_Barrier(comm);
//    if(rank == 0) printf("\nsplit: \n");
//    print_vector(grid.A->split, 0, comm);
//    MPI_Barrier(comm);

    std::vector<int> scount(nprocs, 0);

    long least_proc;
    least_proc = lower_bound2(&grid.A->split[0], &grid.A->split[nprocs], 0 + grid.A->split_old[rank]);
    scount[least_proc]++;
//    printf("rank %d: 0 + A.split_old[rank] = %u, least_proc = %ld \n", rank, 0 + A.split_old[rank], least_proc);

    long curr_proc = least_proc;
    for(index_t i = 1; i < u.size(); i++){
        if(i + grid.A->split_old[rank] >= grid.A->split[curr_proc+1])
            curr_proc++; //todo: if shrinked==true then curr_proc += A.shrink_thre2
        scount[curr_proc]++;
//        if(rank==0) printf("i + A.split_old[rank] = %u, curr_proc = %ld \n", i + A.split_old[rank], curr_proc);
    }

//    print_vector(send_size_array, -1, comm);

    std::vector<int> rcount(nprocs);
    MPI_Alltoall(&scount[0], 1, MPI_INT, &rcount[0], 1, MPI_INT, comm);
//    print_vector(recv_size_array, -1, comm);

    std::vector<int> send_offset(nprocs);
    send_offset[0] = 0;
    for (int i=1; i<nprocs; i++)
        send_offset[i] = scount[i-1] + send_offset[i-1];

//    print_vector(send_offset, -1, comm);

    std::vector<int> recv_offset(nprocs);
    recv_offset[0] = 0;
    for (int i=1; i<nprocs; i++)
        recv_offset[i] = rcount[i-1] + recv_offset[i-1];

//    print_vector(recv_offset, 2, comm);

    std::vector<value_t> u_old = u;
    u.resize(grid.A->M);
    MPI_Alltoallv(&u_old[0], &scount[0], &send_offset[0], MPI_DOUBLE,
                  &u[0],     &rcount[0], &recv_offset[0], MPI_DOUBLE, comm);

    return 0;
}
*/


// new version of int saena_object::repartition_u_shrink_prepare(Grid *grid)
// it only consider repartition when shrinking has happened.
/*
int saena_object::repartition_u_shrink_prepare(Grid *grid){

    MPI_Comm comm = grid->Ac.comm;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // Note: A is grid->Ac!
    // --------------------
    saena_matrix *A = &grid->Ac;

//    print_vector(A->split_old, 0, "split_old", comm);
//    print_vector(A->split, 0, "split", comm);
//    printf("rank %d: A->M = %u, A->M_old = %u \n", rank, A->M, A->M_old);
//    MPI_Barrier(comm);

    grid->scount2.assign(nprocs, 0);

    long least_proc;
    least_proc = lower_bound3(&A->split[0], &A->split[nprocs], 0 + A->split_old[rank]);
    grid->scount2[least_proc]++;
//    printf("rank %d: 0 + A.split_old[rank] = %u, least_proc = %ld \n", rank, 0 + A->split_old[rank], least_proc);

    long curr_proc = least_proc;
    for(index_t i = 1; i < A->M_old; i++){
        if(i + A->split_old[rank] >= A->split[curr_proc+1]){
//            if(A->shrinked)
                curr_proc += A->cpu_shrink_thre2;
//            else
//                curr_proc++;
        }
        grid->scount2[curr_proc]++;
//        if(rank==2) printf("i + A.split_old[rank] = %u, curr_proc = %ld \n", i + A->split_old[rank], curr_proc);
    }

//    print_vector(grid->scount2, -1, comm);

    // instead of only resizing rcount2, it is put equal to scount2 in case of nprocs = 1.
    grid->rcount2 = grid->scount2;
    if(nprocs > 1)
        MPI_Alltoall(&grid->scount2[0], 1, MPI_INT, &grid->rcount2[0], 1, MPI_INT, comm);

//    print_vector(grid->rcount2, -1, comm);

//    std::vector<int> sdispls2(nprocs);
    grid->sdispls2.resize(nprocs);
    grid->sdispls2[0] = 0;
    for (int i=1; i<nprocs; i++)
        grid->sdispls2[i] = grid->scount2[i-1] + grid->sdispls2[i-1];

//    print_vector(grid->sdispls2, -1, comm);

//    std::vector<int> rdispls2(nprocs);
    grid->rdispls2.resize(nprocs);
    grid->rdispls2[0] = 0;
    for (int i=1; i<nprocs; i++)
        grid->rdispls2[i] = grid->rcount2[i-1] + grid->rdispls2[i-1];

//    print_vector(grid->rdispls2, -1, comm);

    return 0;
}
*/


int saena_object::repartition_u_shrink_prepare(Grid *grid){

//    MPI_Comm comm = grid->A->comm;
    MPI_Comm comm = grid->Ac.comm;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // Note: A is grid->Ac!
    // --------------------
    saena_matrix *A = &grid->Ac;

//    print_vector(A->split_old, 0, "split_old", comm);
//    print_vector(A->split, 0, "split", comm);
//    MPI_Barrier(comm); printf("rank %d: A->M = %u, A->M_old = %u \n", rank, A->M, A->M_old); MPI_Barrier(comm);

    grid->scount2.assign(nprocs, 0);

    long least_proc = 0, curr_proc;
    if(A->M_old != 0){
        least_proc = lower_bound3(&A->split[0], &A->split[nprocs], 0 + A->split_old[rank]);
        grid->scount2[least_proc]++;
//    printf("rank %d: 0 + A.split_old[rank] = %u, least_proc = %ld \n", rank, 0 + A->split_old[rank], least_proc);

        curr_proc = least_proc;
        for(index_t i = 1; i < A->M_old; i++){
            if(i + A->split_old[rank] >= A->split[curr_proc+1]){
                if(A->shrinked)
                    curr_proc += A->cpu_shrink_thre2;
                else
                    curr_proc++;
            }
            grid->scount2[curr_proc]++;
    //        if(rank==2) printf("i + A.split_old[rank] = %u, curr_proc = %ld \n", i + A->split_old[rank], curr_proc);
        }
    }

//    print_vector(grid->scount2, -1, "scount2", comm);

    // instead of only resizing rcount2, it is put equal to scount2 in case of nprocs = 1.
    grid->rcount2 = grid->scount2;
    if(nprocs > 1)
        MPI_Alltoall(&grid->scount2[0], 1, MPI_INT, &grid->rcount2[0], 1, MPI_INT, comm);

//    print_vector(grid->rcount2, -1, "rcount2", comm);

//    std::vector<int> sdispls2(nprocs);
    grid->sdispls2.resize(nprocs);
    grid->sdispls2[0] = 0;
    for (int i=1; i<nprocs; i++)
        grid->sdispls2[i] = grid->scount2[i-1] + grid->sdispls2[i-1];

//    print_vector(grid->sdispls2, "sdispls2, -1, comm);

//    std::vector<int> rdispls2(nprocs);
    grid->rdispls2.resize(nprocs);
    grid->rdispls2[0] = 0;
    for (int i=1; i<nprocs; i++)
        grid->rdispls2[i] = grid->rcount2[i-1] + grid->rdispls2[i-1];

//    print_vector(grid->rdispls2, -1, "rdispls2, comm);

    return 0;
}


int saena_object::repartition_u_shrink(std::vector<value_t> &u, Grid &grid){

    MPI_Comm comm = grid.Ac.comm;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    MPI_Barrier(grid.A->comm);
//    printf("rank %d: A->M = %u, A->M_old = %u \n", rank, grid.Ac.M, grid.Ac.M_old);
//    MPI_Barrier(grid.A->comm);
//    print_vector(u, 1, "u inside repartition_u_shrink", comm);

    std::vector<value_t> u_old = u;
    u.resize(grid.Ac.M);
    MPI_Alltoallv(&u_old[0], &grid.scount2[0], &grid.sdispls2[0], MPI_DOUBLE,
                  &u[0],     &grid.rcount2[0], &grid.rdispls2[0], MPI_DOUBLE, comm);

    return 0;
}


int saena_object::repartition_back_u_shrink(std::vector<value_t> &u, Grid &grid){

    MPI_Comm comm = grid.Ac.comm;
//    MPI_Comm comm = grid.A->comm;
//    int rank, nprocs;
//    MPI_Comm_size(comm, &nprocs);
//    MPI_Comm_rank(comm, &rank);

    std::vector<value_t> u_old = u;
    u.resize(grid.Ac.M_old);
    MPI_Alltoallv(&u_old[0], &grid.rcount2[0], &grid.rdispls2[0], MPI_DOUBLE,
                  &u[0],     &grid.scount2[0], &grid.sdispls2[0], MPI_DOUBLE, comm);

    return 0;
}


//int saena_object::repartition_u_shrink_minor_prepare(Grid *grid)
/*
int saena_object::repartition_u_shrink_minor_prepare(Grid *grid){

    MPI_Comm comm = grid->Ac.comm;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // Note: A is grid->Ac!
    // --------------------
    saena_matrix *A = &grid->Ac;

//    print_vector(A->split_old, 0, "split_old", comm);
//    print_vector(A->split, 0, "split", comm);
//    printf("rank %d: A->M = %u, A->M_old = %u \n", rank, A->M, A->M_old);
//    MPI_Barrier(comm);

    grid->scount3.assign(nprocs, 0);

    long least_proc;
    least_proc = lower_bound3(&A->split[0], &A->split[nprocs], 0 + A->split_old_minor[rank]);
    grid->scount3[least_proc]++;
//    printf("rank %d: 0 + A.split_old[rank] = %u, least_proc = %ld \n", rank, 0 + A->split_old[rank], least_proc);

    long curr_proc = least_proc;
    for(index_t i = 1; i < A->M; i++){
        while(i + A->split_old_minor[rank] >= A->split[curr_proc+1]){
             curr_proc++;
        }
        grid->scount3[curr_proc]++;
//        if(rank==2) printf("i + A.split_old[rank] = %u, curr_proc = %ld \n", i + A->split_old[rank], curr_proc);
    }

//    print_vector(grid->scount3, -1, comm);

    // instead of only resizing rcount2, it is put equal to scount2 in case of nprocs = 1.
    grid->rcount3 = grid->scount3;
    if(nprocs > 1)
        MPI_Alltoall(&grid->scount3[0], 1, MPI_INT, &grid->rcount3[0], 1, MPI_INT, comm);

//    print_vector(grid->rcount3, -1, comm);

//    std::vector<int> sdispls3(nprocs);
    grid->sdispls3.resize(nprocs);
    grid->sdispls3[0] = 0;
    for (int i=1; i<nprocs; i++)
        grid->sdispls3[i] = grid->scount3[i-1] + grid->sdispls3[i-1];

//    print_vector(grid->sdispls3, -1, comm);

//    std::vector<int> rdispls2(nprocs);
    grid->rdispls3.resize(nprocs);
    grid->rdispls3[0] = 0;
    for (int i=1; i<nprocs; i++)
        grid->rdispls3[i] = grid->rcount3[i-1] + grid->rdispls3[i-1];

//    print_vector(grid->rdispls3, -1, comm);

    return 0;
}
*/


//int saena_object::repartition_u_shrink_minor(std::vector<value_t> &u, Grid &grid)
/*
int saena_object::repartition_u_shrink_minor(std::vector<value_t> &u, Grid &grid){

    MPI_Comm comm = grid.Ac.comm;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    MPI_Barrier(grid.A->comm);
//    printf("rank %d: A->M = %u, A->M_old = %u \n", rank, grid.Ac.M, grid.Ac.M_old);
//    MPI_Barrier(grid.A->comm);

    std::vector<value_t> u_old = u;
    u.resize(grid.Ac.M);
    MPI_Alltoallv(&u_old[0], &grid.scount3[0], &grid.sdispls3[0], MPI_DOUBLE,
                  &u[0],     &grid.rcount3[0], &grid.rdispls3[0], MPI_DOUBLE, comm);

    return 0;
}
*/


// int saena_object::repartition_back_u_shrink
/*
int saena_object::repartition_back_u_shrink(std::vector<value_t> &u, saena_matrix &A){

    MPI_Comm comm = A.comm; //todo: after shrinking check if it is still true
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    MPI_Barrier(comm);
//    if(rank == 0) printf("\nsplit_old: \n");
//    print_vector(A.split_old, 0, comm);
//    MPI_Barrier(comm);
//    if(rank == 0) printf("\nsplit: \n");
//    print_vector(A.split, 0, comm);
//    MPI_Barrier(comm);

    std::vector<int> send_size_array(nprocs, 0);

    long least_proc;
    least_proc = lower_bound2(&A.split_old[0], &A.split_old[nprocs], 0 + A.split[rank]);
    send_size_array[least_proc]++;
//    printf("rank %d: 0 + A.split[rank] = %u, least_proc = %ld \n", rank, 0 + A.split[rank], least_proc);

    long curr_proc = least_proc;
    for(index_t i = 1; i < u.size(); i++){
        if(i + A.split[rank] >= A.split_old[curr_proc+1])
            curr_proc++; //todo: if shrinked==true then curr_proc += A.shrink_thre2
        send_size_array[curr_proc]++;
//        if(rank==0) printf("i + A.split[rank] = %u, curr_proc = %ld \n", i + A.split[rank], curr_proc);
    }

//    print_vector(send_size_array, -1, comm);

    std::vector<int> recv_size_array(nprocs);
    MPI_Alltoall(&send_size_array[0], 1, MPI_INT, &recv_size_array[0], 1, MPI_INT, comm);

//    print_vector(recv_size_array, -1, comm);

    std::vector<int> send_offset(nprocs);
    send_offset[0] = 0;
    for (int i=1; i<nprocs; i++)
        send_offset[i] = send_size_array[i-1] + send_offset[i-1];

//    print_vector(send_offset, -1, comm);

    std::vector<int> recv_offset(nprocs);
    recv_offset[0] = 0;
    for (int i=1; i<nprocs; i++)
        recv_offset[i] = recv_size_array[i-1] + recv_offset[i-1];

//    print_vector(recv_offset, 2, comm);

    std::vector<value_t> u_old = u;
    u.resize(A.M_old);
    MPI_Alltoallv(&u_old[0], &send_size_array[0], &send_offset[0], MPI_DOUBLE,
                  &u[0],     &recv_size_array[0], &recv_offset[0], MPI_DOUBLE, comm);

    return 0;
}
*/


int saena_object::shrink_u_rhs(Grid* grid, std::vector<value_t>& u, std::vector<value_t>& rhs){

    int rank, nprocs;
    MPI_Comm_size(grid->A->comm, &nprocs);
    MPI_Comm_rank(grid->A->comm, &rank);

    int rank_horizontal = -1;
    int nprocs_horizontal = -1;
//    if(grid->A->active){
    MPI_Comm_size(grid->Ac.comm_horizontal, &nprocs_horizontal);
    MPI_Comm_rank(grid->Ac.comm_horizontal, &rank_horizontal);
//    }

//    if(rank==0) {
//        printf("\nbefore shrinking: rank = %d, level = %d, rhs.size = %lu \n", rank, grid->currentLevel,
//               rhs.size());
//        for(unsigned long i=0; i<rhs.size(); i++)
//            printf("rhs[%lu] = %f \n", i, rhs[i]);}
//    MPI_Barrier(grid->A->comm);

    unsigned long offset = 0;
    if(grid->Ac.active){
        offset = grid->Ac.split_old[rank + 1] - grid->Ac.split_old[rank];
        u.assign(grid->Ac.M,0); // u is just zero, so no communication is required.
        rhs.resize(grid->Ac.M); // it is already of size grid->A->M. this is redundant.
    }

    // last cpu that its right neighbors are going be shrinked to.
    auto last_root_cpu = (unsigned int)floor(nprocs/grid->A->cpu_shrink_thre2) * grid->A->cpu_shrink_thre2;
//    printf("last_root_cpu = %u\n", last_root_cpu);

    int neigbor_rank;
    index_t recv_size = 0;
//    index_t send_size = rhs.size();
    for(neigbor_rank = 1; neigbor_rank < grid->A->cpu_shrink_thre2; neigbor_rank++){

        if( rank == last_root_cpu && (rank + neigbor_rank >= nprocs) )
            break;

//        if(rank_horizontal == 0 && (rank + neigbor_rank >= nprocs) )
//            break;

        // send and receive size of rhs.
//        if(rank_horizontal == 0)
//            MPI_Recv(&recv_size, 1, MPI_UNSIGNED, neigbor_rank, 0, comm_new, MPI_STATUS_IGNORE);
//        if(rank_horizontal == neigbor_rank)
//            MPI_Send(&local_size, 1, MPI_UNSIGNED, 0, 0, comm_new);

        // send and receive rhs and u.

        if(rank_horizontal == 0){
//            printf("rank = %d, grid->A->split_old[rank + neigbor_rank + 1] = %lu, grid->A->split_old[rank + neigbor_rank] = %lu \n",
//                     rank, grid->A->split_old[rank + neigbor_rank + 1], grid->A->split_old[rank + neigbor_rank]);

            recv_size = grid->Ac.split_old[rank + neigbor_rank + 1] - grid->Ac.split_old[rank + neigbor_rank];
//            printf("rank = %d, neigbor_rank = %d, recv_size = %u, offset = %lu \n", rank, neigbor_rank, recv_size, offset);
//            MPI_Recv(&*(u.begin() + offset),   recv_size, MPI_DOUBLE, neigbor_rank, 0, grid->Ac.comm_horizontal, MPI_STATUS_IGNORE);
            MPI_Recv(&*(rhs.begin() + offset), recv_size, MPI_DOUBLE, neigbor_rank, 1, grid->Ac.comm_horizontal, MPI_STATUS_IGNORE);
            offset += recv_size; // set offset for the next iteration
        }

        if(rank_horizontal == neigbor_rank){
//            printf("rank = %d, neigbor_rank = %d, local_size = %lu \n", rank, neigbor_rank, rhs.size());
//            MPI_Send(&*u.begin(),   send_size, MPI_DOUBLE, 0, 0, grid->Ac.comm_horizontal);
            MPI_Send(&*rhs.begin(), rhs.size(), MPI_DOUBLE, 0, 1, grid->Ac.comm_horizontal);
        }
    }

//    MPI_Barrier(grid->Ac.comm_horizontal);
//    if(rank==0){
//        printf("\nafter shrinking: rank = %d, level = %d, rhs.size = %lu \n", rank, grid->currentLevel, rhs.size());
//        for(unsigned long i=0; i<rhs.size(); i++)
//            printf("rhs[%lu] = %f \n", i, rhs[i]);}
//    MPI_Barrier(grid->Ac.comm_horizontal);
//    if(rank==1){
//        printf("\nafter shrinking: rank = %d, level = %d, rhs.size = %lu \n", rank, grid->currentLevel, rhs.size());
//        for(unsigned long i=0; i<rhs.size(); i++)
//            printf("rhs[%lu] = %f \n", i, rhs[i]);}
//    MPI_Barrier(grid->Ac.comm_horizontal);

    return 0;
}


int saena_object::unshrink_u(Grid* grid, std::vector<value_t>& u) {

    int rank, nprocs;
    MPI_Comm_size(grid->A->comm, &nprocs);
    MPI_Comm_rank(grid->A->comm, &rank);

//    int nprocs_horizontal = -1;
//    MPI_Comm_size(grid->Ac.comm_horizontal, &nprocs_horizontal);
    int rank_horizontal = -1;
    MPI_Comm_rank(grid->Ac.comm_horizontal, &rank_horizontal);

//    MPI_Barrier(grid->A->comm);
//    if(rank==0) {
//        printf("\nbefore un-shrinking: rank = %d, level = %d, u.size = %lu \n", rank, grid->currentLevel,
//               u.size());
//        for(unsigned long i=0; i<u.size(); i++)
//            printf("u[%lu] = %f \n", i, u[i]);}
//    MPI_Barrier(grid->A->comm);

//    if(rank==1){
//        printf("\nbefore un-shrinking: rank_new = %d, level = %d, u.size = %lu \n", rank, grid->currentLevel,
//               u.size());
//        for(unsigned long i=0; i<u.size(); i++)
//            printf("u[%lu] = %f \n", i, u[i]);}

    unsigned long offset = 0;
    // initialize offset to the size of u on the sender processor
    if(grid->Ac.active)
        offset = grid->Ac.split_old[rank + 1] - grid->Ac.split_old[rank];

    // last cpu that its right neighbors are going be shrinked to.
    auto last_root_cpu = (unsigned int)floor(nprocs/grid->A->cpu_shrink_thre2) * grid->A->cpu_shrink_thre2;
//    printf("last_root_cpu = %u\n", last_root_cpu);


    int neighbor_size = (nprocs % grid->A->cpu_shrink_thre2) - 1;
    int requests_size;
    if(rank_horizontal != 0)
        requests_size = 1;
    else
        requests_size = grid->A->cpu_shrink_thre2 - 1;

    if(rank == last_root_cpu)
        requests_size = neighbor_size;

//    std::vector<MPI_Request> reqs;
//    MPI_Request req;
    MPI_Request* requests = new MPI_Request[requests_size]; // 2 is for send and also for receive
    MPI_Status *statuses  = new MPI_Status[requests_size];

    int neigbor_rank;
    index_t send_size, recv_size;
    for(neigbor_rank = 1; neigbor_rank < grid->A->cpu_shrink_thre2; neigbor_rank++){
//        printf("rank = %d, rank_horizontal = %d, nprocs = %d, neigbor_rank = %d, recv_size = %lu, offset = %lu \n",
//               rank, rank_horizontal, nprocs, neigbor_rank, u.size(), offset);

        if(rank == last_root_cpu && (rank + neigbor_rank >= nprocs) )
            break;

        // send and receive size of u.
//        if(nprocs_horizontal == 0)
//            MPI_Recv(&recv_size, 1, MPI_UNSIGNED, neigbor_rank, 0, comm_new, MPI_STATUS_IGNORE);
//        if(nprocs_horizontal == neigbor_rank)
//            MPI_Send(&local_size, 1, MPI_UNSIGNED, 0, 0, comm_new);

        // send and receive u.
        // ------------------
        if(rank_horizontal == neigbor_rank){
            recv_size = grid->A->split[rank + 1] - grid->A->split[rank];
            u.resize(recv_size);
//            printf("un-shrink: rank = %d, neigbor_rank = %d, recv_size = %lu, offset = %lu \n", rank, neigbor_rank, u.size(), offset);
            MPI_Recv(&*u.begin(), recv_size, MPI_DOUBLE, 0, 0, grid->Ac.comm_horizontal, MPI_STATUS_IGNORE);
//            MPI_Recv(&*u.begin(), u.size(), MPI_DOUBLE, 0, 0, grid->Ac.comm_horizontal, MPI_STATUS_IGNORE);
//            MPI_Irecv(&*u.begin(), u.size(), MPI_DOUBLE, 0, 0, grid->Ac.comm_horizontal, &requests[0]);
//            reqs.push_back(req);
        }

        if(rank_horizontal == 0){
            send_size = grid->Ac.split_old[rank + neigbor_rank + 1] - grid->Ac.split_old[rank + neigbor_rank];
//            printf("un-shrink: rank = %d, neigbor_rank = %d, send_size = %u, offset = %lu \n", rank, neigbor_rank, send_size, offset);
            MPI_Send(&*(u.begin() + offset), send_size, MPI_DOUBLE, neigbor_rank, 0, grid->Ac.comm_horizontal);
//            MPI_Isend(&*u.begin() + offset, send_size, MPI_DOUBLE, neigbor_rank, 0, grid->Ac.comm_horizontal, &requests[neigbor_rank - 1]);
//            reqs.push_back(req);
            offset += send_size; // set offset for the next iteration
        }
    }

//    MPI_Barrier(grid->A->comm);
//    printf("done un-shrink: rank = %d\n", rank);
//    MPI_Barrier(grid->A->comm);

//    MPI_Waitall(requests_size, requests, statuses);

    if(rank_horizontal == 0){
        u.resize(grid->Ac.split_old[rank + 1] - grid->Ac.split_old[rank]);
        // todo: is shrink_to_fit required, or it's better to keep the memory for u to be used for the next vcycle iterations?
//        u.shrink_to_fit();
    }

//    MPI_Barrier(grid->A->comm);
//    if(rank_horizontal==0){
//        printf("\nafter un-shrinking: rank = %d, level = %d, u.size = %lu \n", rank, grid->currentLevel, u.size());
//        for(unsigned long i=0; i<u.size(); i++)
//            printf("u[%lu] = %f \n", i, u[i]);}
//    MPI_Barrier(grid->A->comm);
//    if(rank==1){
//        printf("\nafter un-shrinking: rank_horizontal = %d, level = %d, u.size = %lu \n", rank_horizontal, grid->currentLevel, u.size());
//        for(unsigned long i=0; i<u.size(); i++)
//            printf("u[%lu] = %f \n", i, u[i]);}
//    MPI_Barrier(grid->A->comm);
//    if(rank_horizontal==2){
//        printf("\nafter un-shrinking: rank = %d, level = %d, u.size = %lu \n", rank, grid->currentLevel, u.size());
//        for(unsigned long i=0; i<u.size(); i++)
//            printf("u[%lu] = %f \n", i, u[i]);}
//    MPI_Barrier(grid->A->comm);
//    if(rank_horizontal==3){
//        printf("\nafter un-shrinking: rank = %d, level = %d, u.size = %lu \n", rank, grid->currentLevel, u.size());
//        for(unsigned long i=0; i<u.size(); i++)
//            printf("u[%lu] = %f \n", i, u[i]);}
//    MPI_Barrier(grid->A->comm);

    delete [] requests;
    delete [] statuses;

    return 0;
}


int saena_object::writeMatrixToFileA(saena_matrix* A, std::string name){
    // Create txt files with name Ac-r0.txt for processor 0, Ac-r1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat Ac-r0.txt Ac-r1.txt > Ac.txt
    // row and column indices of txt files should start from 1, not 0.

    // todo: check global or local index and see if A->split[rank] is required for rows.

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::ofstream outFileTxt;
    std::string outFileNameTxt = "/home/boss/Dropbox/Projects/Saena_base/build/writeMatrix/";
    outFileNameTxt += name;
    outFileNameTxt += "-r";
    outFileNameTxt += std::to_string(rank);
    outFileNameTxt += ".txt";
    outFileTxt.open(outFileNameTxt);

//    if(rank==0)
//        outFileTxt << A->Mbig << "\t" << A->Mbig << "\t" << A->nnz_g << std::endl;

    for (long i = 0; i < A->nnz_l; i++) {
        if(A->entry[i].row == A->entry[i].col)
            continue;

//        std::cout       << A->entry[i].row + 1 << "\t" << A->entry[i].col + 1 << "\t" << A->entry[i].val << std::endl;
        outFileTxt << A->entry[i].row + 1 << "\t" << A->entry[i].col + 1 << "\t" << A->entry[i].val << std::endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

/*
    // this is the code for writing the result of jacobi to a file.
    char* outFileNameTxt = "jacobi_saena.bin";
    MPI_Status status2;
    MPI_File fh2;
    MPI_Offset offset2;
    MPI_File_open(comm, outFileNameTxt, MPI_MODE_CREATE| MPI_MODE_WRONLY, MPI_INFO_NULL, &fh2);
    offset2 = A.split[rank] * 8; // value(double=8)
    MPI_File_write_at(fh2, offset2, xp, A.M, MPI_UNSIGNED_LONG, &status2);
    int count2;
    MPI_Get_count(&status2, MPI_UNSIGNED_LONG, &count2);
    //printf("process %d wrote %d lines of triples\n", rank, count2);
    MPI_File_close(&fh2);
*/

/*
    // failed try to write this part.
    MPI_Status status;
    MPI_File fh;
    MPI_Offset offset;

//    const char* fileName = "/home/abaris/Dropbox/Projects/Saena/build/Ac.txt";
//    const char* fileName = (const char*)malloc(sizeof(const char)*49);
//    fileName = "/home/abaris/Dropbox/Projects/Saena/build/Ac" + "1.txt";
    std::string fileName = "/home/abaris/Acoarse/Ac";
    fileName += std::to_string(7);
    fileName += ".txt";

    int mpierror = MPI_File_open(comm, fileName.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if (mpierror) {
        if (rank == 0) std::cout << "Unable to open the matrix file!" << std::endl;
        MPI_Finalize();
    }

//    std::vector<unsigned int> nnzScan(nprocs);
//    mpierror = MPI_Allgather(&A->nnz_l, 1, MPI_UNSIGNED, &nnzScan[1], 1, MPI_UNSIGNED, comm);
//    if (mpierror) {
//        if (rank == 0) std::cout << "Unable to gather!" << std::endl;
//        MPI_Finalize();
//    }

//    nnzScan[0] = 0;
//    for(unsigned long i=0; i<nprocs; i++){
//        nnzScan[i+1] = nnzScan[i] + nnzScan[i+1];
//        if(rank==1) std::cout << nnzScan[i] << std::endl;
//    }
//    offset = nnzScan[rank];
//    MPI_File_write_at_all(fh, rank, &nnzScan[rank])
//    unsigned int a = 1;
//    double b = 3;
//    MPI_File_write_at(fh, rank, &rank, 1, MPI_INT, &status);
//    MPI_File_write_at_all(fh, offset, &A->entry[0], A->nnz_l, cooEntry::mpi_datatype(), &status);
//    MPI_File_write_at_all(fh, &A->entry[0], A->nnz_l, cooEntry::mpi_datatype(), &status);
//    if (mpierror) {
//        if (rank == 0) std::cout << "Unable to write to the matrix file!" << std::endl;
//        MPI_Finalize();
//    }

//    int count;
//    MPI_Get_count(&status, MPI_UNSIGNED_LONG, &count);
    //printf("process %d read %d lines of triples\n", rank, count);
    MPI_File_close(&fh);
*/

    return 0;
}


int saena_object::writeMatrixToFileP(prolong_matrix* P, std::string name) {
    // Create txt files with name P0.txt for processor 0, P1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat P0.txt P1.txt > P.txt
    // row and column indices of txt files should start from 1, not 0.

    MPI_Comm comm = P->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::ofstream outFileTxt;
    std::string outFileNameTxt = "/home/abaris/Dropbox/Projects/Saena/build/writeMatrix/";
    outFileNameTxt += name;
    outFileNameTxt += std::to_string(rank);
    outFileNameTxt += ".txt";
    outFileTxt.open(outFileNameTxt);

    if (rank == 0)
        outFileTxt << P->Mbig << "\t" << P->Mbig << "\t" << P->nnz_g << std::endl;
    for (long i = 0; i < P->nnz_l; i++) {
//        std::cout       << P->entry[i].row + 1 + P->split[rank] << "\t" << P->entry[i].col + 1 << "\t" << P->entry[i].val << std::endl;
        outFileTxt << P->entry[i].row + 1 + P->split[rank] << "\t" << P->entry[i].col + 1 << "\t" << P->entry[i].val << std::endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

    return 0;
}


int saena_object::writeMatrixToFileR(restrict_matrix* R, std::string name) {
    // Create txt files with name R0.txt for processor 0, R1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat R0.txt R1.txt > R.txt
    // row and column indices of txt files should start from 1, not 0.

    MPI_Comm comm = R->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::ofstream outFileTxt;
    std::string outFileNameTxt = "/home/abaris/Dropbox/Projects/Saena/build/writeMatrix/";
    outFileNameTxt += name;
    outFileNameTxt += std::to_string(rank);
    outFileNameTxt += ".txt";
    outFileTxt.open(outFileNameTxt);

    if (rank == 0)
        outFileTxt << R->Mbig << "\t" << R->Mbig << "\t" << R->nnz_g << std::endl;
    for (long i = 0; i < R->nnz_l; i++) {
//        std::cout       << R->entry[i].row + 1 + R->splitNew[rank] << "\t" << R->entry[i].col + 1 << "\t" << R->entry[i].val << std::endl;
        outFileTxt << R->entry[i].row + 1 +  R->splitNew[rank] << "\t" << R->entry[i].col + 1 << "\t" << R->entry[i].val << std::endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

    return 0;
}


int saena_object::writeVectorToFileul(std::vector<unsigned long>& v, std::string name, MPI_Comm comm) {

    // Create txt files with name name-r0.txt for processor 0, name-r1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat name-r0.txt name-r1.txt > V.txt

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::ofstream outFileTxt;
    std::string outFileNameTxt = "/home/boss/Dropbox/Projects/Saena_base/build/writeMatrix/";
    outFileNameTxt += name;
    outFileNameTxt += "-r";
    outFileNameTxt += std::to_string(rank);
    outFileNameTxt += ".txt";
    outFileTxt.open(outFileNameTxt);

    if (rank == 0)
        outFileTxt << v.size() << std::endl;
    for (long i = 0; i < v.size(); i++) {
//        std::cout       << R->entry[i].row + 1 + R->splitNew[rank] << "\t" << R->entry[i].col + 1 << "\t" << R->entry[i].val << std::endl;
        outFileTxt << v[i]+1 << std::endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

    return 0;
}


int saena_object::writeVectorToFileul2(std::vector<unsigned long>& v, std::string name, MPI_Comm comm) {

    // Create txt files with name name-r0.txt for processor 0, name-r1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat name-r0.txt name-r1.txt > V.txt
    // This version also writes the index number, so it has two columns, instead of 1.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::ofstream outFileTxt;
    std::string outFileNameTxt = "/home/boss/Dropbox/Projects/Saena_base/build/writeMatrix/";
    outFileNameTxt += name;
    outFileNameTxt += "-r";
    outFileNameTxt += std::to_string(rank);
    outFileNameTxt += ".txt";
    outFileTxt.open(outFileNameTxt);

//    if (rank == 0)
//        outFileTxt << v.size() << std::endl;
    for (long i = 0; i < v.size(); i++) {
        outFileTxt << i+1 << "\t" << v[i]+1 << std::endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

    return 0;
}


int saena_object::writeVectorToFileui(std::vector<unsigned int>& v, std::string name, MPI_Comm comm) {

    // Create txt files with name name-r0.txt for processor 0, name-r1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat name-r0.txt name-r1.txt > V.txt

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::ofstream outFileTxt;
    std::string outFileNameTxt = "/home/boss/Dropbox/Projects/Saena_base/build/writeMatrix/";
    outFileNameTxt += name;
    outFileNameTxt += "-r";
    outFileNameTxt += std::to_string(rank);
    outFileNameTxt += ".txt";
    outFileTxt.open(outFileNameTxt);

    if (rank == 0)
        outFileTxt << v.size() << std::endl;
    for (long i = 0; i < v.size(); i++) {
//        std::cout << v[i] + 1 + split[rank] << std::endl;
        outFileTxt << v[i]+1 << std::endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

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


bool saena_object::active(int l){
    return grids[l].A->active;
}


int saena_object::set_shrink_levels(std::vector<bool> sh_lev_vec) {
    shrink_level_vector = sh_lev_vec;
    return 0;
}


int saena_object::set_shrink_values(std::vector<int> sh_val_vec) {
    shrink_values_vector = sh_val_vec;
    return 0;
}


int saena_object::set_repartition_threshold(float thre) {
    repartition_threshold = thre;
    return 0;
}


int saena_object::local_diff(saena_matrix &A, saena_matrix &B, std::vector<cooEntry> &C){

    if(A.active){

        MPI_Comm comm = A.comm;
        int nprocs, rank;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);

        if(A.nnz_g != B.nnz_g)
            if(rank==0) std::cout << "error: local_diff(): A.nnz_g != B.nnz_g" << std::endl;

        C.clear();
        C.resize(A.nnz_l_local);
        index_t loc_size = 0;
        for(nnz_t i = 0; i < A.nnz_l_local; i++){
            if(!almost_zero(A.values_local[i]-B.values_local[i])){
//                if(rank==1) printf("%u \t%u \t%f \n", A.row_local[i], A.col_local[i], A.values_local[i]-B.values_local[i]);
                C[loc_size] = cooEntry(A.row_local[i], A.col_local[i], B.values_local[i]-A.values_local[i]);
                loc_size++;
            }
        }
        C.resize(loc_size);

        // this part sets the parameters needed to be set until the end of repartition().
//        C.Mbig = A.Mbig;
//        C.M = A.M;
//        C.split = A.split;
//        C.nnz_l = loc_size;
//        C.nnz_l_local = C.nnz_l;
//        MPI_Allreduce(&C.nnz_l, &C.nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, C.comm);

        // the only part needed from matrix_setup() for coarsen2().
//        C.indicesP_local.resize(C.nnz_l_local);
//#pragma omp parallel for
//        for (nnz_t i = 0; i < C.nnz_l_local; i++)
//            C.indicesP_local[i] = i;
//        index_t *row_localP = &*C.row_local.begin();
//        std::sort(&C.indicesP_local[0], &C.indicesP_local[C.nnz_l_local], sort_indices(row_localP));

//        C.matrix_setup();
    }

    return 0;
}


int saena_object::scale_vector(std::vector<value_t>& v, std::vector<value_t>& w) {

#pragma omp parallel for
    for(index_t i = 0; i < v.size(); i++)
        v[i] *= w[i];

    return 0;
}


int saena_object::find_eig(saena_matrix& A){

//    find_eig_Elemental(A);
    find_eig_ietl(A);

    return 0;
}


// saena_object::find_eig_Elemental
/*
int saena_object::find_eig_Elemental(saena_matrix& A) {

    int argc = 0;
    char** argv = {NULL};
//    El::Environment env( argc, argv );
    El::Initialize( argc, argv );

    int rank, nprocs;
    MPI_Comm_rank(A.comm, &rank);
    MPI_Comm_size(A.comm, &nprocs);

    const El::Int n = A.Mbig;

    // *************************** serial ***************************

//    El::Matrix<double> A(n,n);
//    El::Zero( A );
//    for(unsigned long i = 0; i<nnz_l; i++)
//        A(entry[i].row, entry[i].col) = entry[i].val * inv_diag[entry[i].row];

//    El::Print( A, "\nGlobal Elemental matrix (serial):\n" );

//    El::Matrix<El::Complex<double>> w(n,1);

    // *************************** parallel ***************************

    El::DistMatrix<value_t> B(n,n);
    El::Zero( B );
    B.Reserve(A.nnz_l);
    for(nnz_t i = 0; i < A.nnz_l; i++){
//        if(rank==0) printf("%lu \t%u \t%f \t%f \t%f \telemental\n",
//                           i, A.entry[i].row, A.entry[i].val, A.inv_diag[A.entry[i].row - A.split[rank]], A.entry[i].val*A.inv_diag[A.entry[i].row - A.split[rank]]);
//        B.QueueUpdate(A.entry[i].row, A.entry[i].col, A.entry[i].val * A.inv_diag[A.entry[i].row - A.split[rank]]); // this is not A! each entry is multiplied by the same-row diagonal value.
        B.QueueUpdate(A.entry[i].row, A.entry[i].col, A.entry[i].val);
    }
    B.ProcessQueues();
//    El::Print( A, "\nGlobal Elemental matrix:\n" );

    El::DistMatrix<El::Complex<value_t>> w(n,1);

    // *************************** common part between serial and parallel ***************************

    El::SchurCtrl<double> schurCtrl;
    schurCtrl.time = false;
//    schurCtrl.hessSchurCtrl.progress = true;
//    El::Schur( A, w, V, schurCtrl ); //  eigenvectors will be saved in V.

//    printf("before Schur!\n");
    El::Schur( B, w, schurCtrl ); // eigenvalues will be saved in w.
//    printf("after Schur!\n");
//    MPI_Barrier(comm); El::Print( w, "eigenvalues:" ); MPI_Barrier(comm);

//    A.eig_max_of_invdiagXA = w.Get(0,0).real();
//    for(unsigned long i = 1; i < n; i++)
//        if(w.Get(i,0).real() > A.eig_max_of_invdiagXA)
//            A.eig_max_of_invdiagXA = w.Get(i,0).real();

    // todo: if the matrix is not symmetric, the eigenvalue will be a complex number.
    A.eig_max_of_invdiagXA = fabs(w.Get(0,0).real());
    for(index_t i = 1; i < n; i++) {
//       std::cout << i << "\t" << w.Get(i, 0) << std::endl;
        if (fabs(w.Get(i, 0).real()) > A.eig_max_of_invdiagXA)
            A.eig_max_of_invdiagXA = fabs(w.Get(i, 0).real());
    }

//    if(rank==0) printf("\nthe biggest eigenvalue of A is %f (Elemental) \n", A.eig_max_of_invdiagXA);

    El::Finalize();

    return 0;
}
*/


// saena_object::solve_coarsest_Elemental
/*
int saena_object::solve_coarsest_Elemental(saena_matrix *A_S, std::vector<value_t> &u, std::vector<value_t> &rhs){

    int argc = 0;
    char** argv = {NULL};
//    El::Environment env( argc, argv );
    El::Initialize( argc, argv );

    int rank, nprocs;
    MPI_Comm_rank(A_S->comm, &rank);
    MPI_Comm_size(A_S->comm, &nprocs);

//    printf("solve_coarsest_Elemental!\n");

    const El::Unsigned n = A_S->Mbig;
//    printf("size = %d\n", n);

    // set the matrix
    // --------------
    El::DistMatrix<value_t> A(n,n);
    El::Zero( A );
    A.Reserve(A_S->nnz_l);
    for(nnz_t i = 0; i < A_S->nnz_l; i++){
//        if(rank==1) printf("%lu \t%lu \t%f \n", A_S->entry[i].row, A_S->entry[i].col, A_S->entry[i].val);
        A.QueueUpdate(A_S->entry[i].row, A_S->entry[i].col, A_S->entry[i].val);
    }
    A.ProcessQueues();
//    El::Print( A, "\nGlobal Elemental matrix:\n" );

    // set the rhs
    // --------------
    El::DistMatrix<value_t> w(n,1);
    El::Zero( w );
    w.Reserve(n);
    for(index_t i = 0; i < rhs.size(); i++){
//        if(rank==0) printf("%lu \t%f \n", i+A_S->split[rank], rhs[i]);
        w.QueueUpdate(i+A_S->split[rank], 0, rhs[i]);
    }
    w.ProcessQueues();
//    El::Print( w, "\nrhs (w):\n" );

    // solve the system
    // --------------
    // w is the rhs. after calling the solve function, it will be the solution.
//    El::DistMatrix<double> C(n,n);
//    El::SymmetricSolve(El::LOWER, El::NORMAL, &A, &);
    El::LinearSolve(A, w);
//    El::Print( w, "\nsolution (w):\n" );

//    double temp;
//    if(rank==1) printf("w solution:\n");
//    for(unsigned long i = A_S->split[rank]; i < A_S->split[rank+1]; i++){
//        if(rank==1) printf("before: %lu \t%f \n", i, w.Get(i,0));
//        temp = w.Get(i,0);
//        u[i-A_S->split[rank]] = temp;
//        if(rank==0) printf("rank = %d \t%lu \t%f \n", rank, i, u[i-A_S->split[rank]]);
//        if(rank==1) printf("rank = %d \t%lu \t%f \n", rank, i, u[i-A_S->split[rank]]);
//        if(rank==0) printf("rank = %d \t%lu \t%f \n", rank, i, temp);
//        if(rank==1) printf("rank = %d \t%lu \t%f \n", rank, i, temp);
//    }

    std::vector<value_t> temp(n);
    for(index_t i = 0; i < n; i++){
        temp[i] = w.Get(i,0);
//        if(rank==1) printf("rank = %d \t%lu \t%f \n", rank, i, temp[i]);
    }

    for(index_t i = A_S->split[rank]; i < A_S->split[rank+1]; i++)
        u[i-A_S->split[rank]] = temp[i];

    El::Finalize();

    return 0;
}
*/


int saena_object::transpose_locally(std::vector<cooEntry> &A, nnz_t size){

    for(nnz_t i = 0; i < size; i++){
        A[i] = cooEntry(A[i].col, A[i].row, A[i].val);
    }

    std::sort(A.begin(), A.end());

    return 0;
}


int saena_object::transpose_locally(std::vector<cooEntry> &A, nnz_t size, std::vector<cooEntry> &B){

    for(nnz_t i = 0; i < size; i++){
        B[i] = cooEntry(A[i].col, A[i].row, A[i].val);
    }

    std::sort(B.begin(), B.end());

    return 0;
}


int saena_object::sparsify(std::vector<cooEntry>& A, MPI_Comm comm) {

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);



    return 0;
}

