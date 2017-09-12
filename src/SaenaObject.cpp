//
// Created by abaris on 3/14/17.
//

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <set>
#include <mpi.h>

#include "SaenaObject.h"

SaenaObject::SaenaObject(){
//    maxLevel = max_lev-1;
} //SaenaObject


SaenaObject::~SaenaObject(){}


int SaenaObject::destroy(){
    return 0;
}


void SaenaObject::set_parameters(int vcycle_n, double relT, std::string sm, int preSm, int postSm){
//    maxLevel = l-1; // maxLevel does not include fine level. fine level is 0.
    vcycle_num = vcycle_n;
    relative_tolerance  = relT;
    smoother = sm;
    preSmooth = preSm;
    postSmooth = postSm;
}


int SaenaObject::setup(SaenaMatrix* A) {
    MPI_Comm comm = MPI_COMM_WORLD; // todo: fix this.
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    int i;

    grids.resize(maxLevel+1);
//    MPI_Barrier(comm); if(rank==1) printf("----------start of AMGSetup----------\n"); MPI_Barrier(comm);
    grids[0] = Grid(A, maxLevel, 0);
    for(i = 0; i < maxLevel; i++){
//        MPI_Barrier(comm); if(rank==2) printf("\n\n----------111 AMGSetup----------\n"); MPI_Barrier(comm);
        levelSetup(&grids[i]);
        grids[i+1] = Grid(&grids[i].Ac, maxLevel, i+1);
        grids[i].coarseGrid = &grids[i+1];
    }

    return 0;
}


int SaenaObject::levelSetup(Grid* grid){

//    int nprocs, rank;
//    MPI_Comm_size(comm, &nprocs);
//    MPI_Comm_rank(comm, &rank);
    unsigned long i;

//    MPI_Barrier(comm); if(rank==1) cout << endl << "current level = " << grid->currentLevel << endl; MPI_Barrier(comm);

    // todo: think about a parameter for making the aggregation less or more aggressive.
    std::vector<unsigned long> aggregate(grid->A->M);
    findAggregation(grid->A, aggregate, grid->P.splitNew);
//    if(rank==0)
//        for(auto i:aggregate)
//            cout << i << endl;

    // todo: delete this!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!*****************&&&&&&&&&&&&&&&&&&&&&&&&&&&$$$$$$$$$
//    changeAggregation(grid->A, aggregate, grid->P.splitNew, comm);

//    if(rank==0) cout << "after changeAggregation" << endl;
//    if(rank==0)
//        for(i=0; i<aggregate.size(); i++)
//            cout << i << "\t" << aggregate[i] << endl;

//    MPI_Barrier(comm); if(rank==0) printf("----------1 aggregate----------\n"); MPI_Barrier(comm);

/*
    std::vector<long> aggregateSorted(A->M);
//    long* aggregateSorted_p = &(*aggregateSorted.begin());
    par::sampleSort(aggregate, aggregateSorted, comm);
    if(rank==0) cout << "\nafter:" << endl;
    if(rank==0)
        for(long i=0; i<A->M; i++)
            cout << i << "\t" << aggregate[i] << "\t" << aggregateSorted[i] << endl;
    if(rank==1)
        for(long i=0; i<A->M; i++)
            cout << i << "\t" << aggregate[i] << "\t" << aggregateSorted[i] << endl;
*/
//    par::sampleSort(aggregate, comm);
/*
    if(rank==0) cout << "\nafter:" << endl;
    if(rank==0)
        for(long i=0; i<A->M; i++)
            cout << i << "\t" << aggregate[i] << endl;
    if(rank==1)
        for(long i=0; i<A->M; i++)
            cout << i << "\t" << aggregate[i] << endl;
*/

    createProlongation(grid->A, aggregate, &grid->P);
//    MPI_Barrier(comm); if(rank==0) printf("----------2 createProlongation----------\n"); MPI_Barrier(comm);
    grid->R.transposeP(&grid->P);
//    MPI_Barrier(comm); if(rank==0) printf("----------3 transposeP----------\n"); MPI_Barrier(comm);
    coarsen(grid->A, &grid->P, &grid->R, &grid->Ac);
//    MPI_Barrier(comm); if(rank==0) printf("----------4 coarsen----------\n"); MPI_Barrier(comm);
    return 0;
}


// ******************** shrink cpus ********************
// this part was located in SaenaObject::AMGSetup function after findAggregation.
// todo: shrink only if it is required to go to the next multigrid level.
// todo: move the shrinking to after aggregation.

/*
// check if the cpus need to be shrinked
int threshold1 = 1000*nprocs;
int threshold2 = 100*nprocs;

if(R.Mbig < threshold1){
    int color = rank/4;
    MPI_Comm comm2;
    MPI_Comm_split(comm, color, rank, &comm2);

    int nprocs2, rank2;
    MPI_Comm_size(comm2, &nprocs2);
    MPI_Comm_rank(comm2, &rank2);

    bool active = false;
    if(rank2 == 0)
        active = true;
//        if(active) printf("rank=%d\n", rank2);
//        printf("rank=%d\trank2=%d\n", rank, rank2);

    // ******************** send the size from children to parent ********************

    unsigned long* nnzGroup = NULL;
    if(active)
        nnzGroup = (unsigned long*)malloc(sizeof(unsigned long)*4);
    MPI_Gather(&R.nnz_l, 1, MPI_UNSIGNED_LONG, nnzGroup, 1, MPI_UNSIGNED_LONG, 0, comm2);

    unsigned long* rdispls = NULL;
    if(active)
        rdispls = (unsigned long*)malloc(sizeof(unsigned long)*3);

    unsigned long nnzNew = 0;
    unsigned long nnzRecv = 0;
    if(active){
        rdispls[0] = 0;
        rdispls[1] = nnzGroup[1];
        rdispls[2] = rdispls[1] + nnzGroup[2];

        for(i=0; i<4; i++)
            nnzNew += nnzGroup[i];

        nnzRecv = nnzNew - R.nnz_l;
//            if(rank==0){
//                cout << "rdispls:" << endl;
//                for(i=0; i<3; i++)
//                    cout << rdispls[i] << endl;
//            }
    }

//        printf("rank=%d\tnnzNew=%lu\tnnzRecv=%lu\n", rank, nnzNew, nnzRecv);

    // ******************** allocate memory ********************

    cooEntry* sendData = NULL;
    if(!active){
        sendData = (cooEntry*)malloc(sizeof(cooEntry)*R.nnz_l);
        for(i=0; i<R.entry_local.size(); i++)
            sendData[i] = R.entry_local[i];
        for(i=0; i<R.entry_remote.size(); i++)
            sendData[i + R.entry_local.size()] = R.entry_remote[i];
    }

//        MPI_Barrier(comm2);
//        if(rank2==2) cout << "sendData:" << endl;
//        for(i=0; i<R.entry_local.size()+R.entry_remote.size(); i++)
//            if(rank2==2) cout << sendData[i].row << "\t" << sendData[i].col << "\t" << sendData[i].val << endl;

    cooEntry* recvData = NULL;
    if(active)
        recvData = (cooEntry*)malloc(sizeof(cooEntry)*nnzRecv);

    // ******************** send data from children to parent ********************

    int numRecvProc = 0;
    int numSendProc = 1;
    if(active){
        numRecvProc = 3;
        numSendProc = 0;
    }

    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses = new MPI_Status[numSendProc+numRecvProc];

    for(int i = 0; i < numRecvProc; i++)
        MPI_Irecv(&recvData[rdispls[i]], (int)nnzGroup[i+1], cooEntry::mpi_datatype(), i+1, 1, comm2, &requests[i]);

    if(!active)
        MPI_Isend(sendData, (int)R.nnz_l, cooEntry::mpi_datatype(), 0, 1, comm2, &requests[numRecvProc]);

    MPI_Waitall(numSendProc+numRecvProc, requests, statuses);

//        if(active)
//            for(i=0; i<nnzRecv; i++)
//                cout << i << "\t" << recvData[i].row << "\t" << recvData[i].col << "\t" << recvData[i].val << endl;

    if(active)
        free(nnzGroup);
    if(!active)
        free(sendData);

    // update split?

    // update threshol1

    if(active){
        free(recvData);
        free(rdispls);
    }
    MPI_Comm_free(&comm2);
}//end of cpu shrinking
*/


int SaenaObject::findAggregation(SaenaMatrix* A, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew){
//    int nprocs, rank;
//    MPI_Comm_size(comm, &nprocs);
//    MPI_Comm_rank(comm, &rank);

    StrengthMatrix S;
    createStrengthMatrix(A, &S);
//    S.print(0);

//    unsigned long aggSize = 0;
    aggregation(&S, aggregate, splitNew);
//    updateAggregation(aggregate, &aggSize);
//    printf("rank = %d \n", rank);

//    if(rank==0)
//        for(long i=0; i<S.M; i++)
//            cout << i << "\t" << aggregate[i] << endl;

    return 0;
} // end of SaenaObject::findAggregation


int SaenaObject::createStrengthMatrix(SaenaMatrix* A, StrengthMatrix* S){

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if(rank==0) cout << "M = " << A->M << ", nnz_l = " << A->nnz_l << endl;

    // ******************************** compute max per row ********************************

    unsigned int i;
//    double maxPerRow[A->M];
    std::vector<double> maxPerRow(A->M);
    fill(&maxPerRow[0], &maxPerRow[A->M], 0);
    for(i=0; i<A->nnz_l; i++){
        if( A->entry[i].row != A->entry[i].col ){
            if(maxPerRow[A->entry[i].row - A->split[rank]] == 0) // use split to convert the index from global to local.
                maxPerRow[A->entry[i].row - A->split[rank]] = -A->entry[i].val;
            else if(maxPerRow[A->entry[i].row - A->split[rank]] < -A->entry[i].val)
                maxPerRow[A->entry[i].row - A->split[rank]] = -A->entry[i].val;
        }
    }

//    if(rank==0)
//        for(i=0; i<A->M; i++)
//            cout << i << "\t" << maxPerRow[i] << endl;

    // ******************************** compute S ********************************

    std::vector<unsigned long> Si;
    std::vector<unsigned long> Sj;
    std::vector<double> Sval;
    for(i=0; i<A->nnz_l; i++){
        if(A->entry[i].row == A->entry[i].col) {
            Si.push_back(A->entry[i].row);
            Sj.push_back(A->entry[i].col);
            Sval.push_back(1);
        }
        else if(maxPerRow[A->entry[i].row - A->split[rank]] != 0) {
//            if ( -A->values[i] / (maxPerRow[A->row[i] - A->split[rank]] ) > connStrength) {
            Si.push_back(A->entry[i].row);
            Sj.push_back(A->entry[i].col);
            Sval.push_back(  -A->entry[i].val / (maxPerRow[A->entry[i].row - A->split[rank]])  );
//                if(rank==0) cout << "A.val = " << -A->values[i] << ", max = " << maxPerRow[A->row[i] - A->split[rank]] << ", divide = " << (-A->values[i] / (maxPerRow[A->row[i] - A->split[rank]])) << endl;
//            }
        }
    }

/*    if(rank==0)
        for (i=0; i<Si.size(); i++)
            cout << "val = " << Sval[i] << endl;*/

    // ******************************** compute max per column - version 1 - for general matrices ********************************

//    double local_maxPerCol[A->Mbig];
    std::vector<double> local_maxPerCol(A->Mbig);
    double* local_maxPerCol_p = &(*local_maxPerCol.begin());
    local_maxPerCol.assign(A->Mbig,0);
//    fill(&local_maxPerCol[0], &local_maxPerCol[A->Mbig], 0);

    for(i=0; i<A->nnz_l; i++){
        if( A->entry[i].row != A->entry[i].col ){
            if(local_maxPerCol[A->entry[i].col] == 0)
                local_maxPerCol[A->entry[i].col] = -A->entry[i].val;
            else if(local_maxPerCol[A->entry[i].col] < -A->entry[i].val)
                local_maxPerCol[A->entry[i].col] = -A->entry[i].val;
        }
    }

//    double maxPerCol[A->Mbig];
    std::vector<double> maxPerCol(A->Mbig);
    double* maxPerCol_p = &(*maxPerCol.begin());
//    MPI_Allreduce(&local_maxPerCol, &maxPerCol, A->Mbig, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(local_maxPerCol_p, maxPerCol_p, A->Mbig, MPI_DOUBLE, MPI_MAX, comm);

//    if(rank==0)
//        for(i=0; i<A->Mbig; i++)
//            cout << i << "\t" << maxPerCol[i] << endl;

    // ******************************** compute ST - version 1 ********************************

    std::vector<long> STi;
    std::vector<long> STj;
    std::vector<double> STval;
    for(i=0; i<A->nnz_l; i++){
        if(A->entry[i].row == A->entry[i].col) {
            STi.push_back(A->entry[i].row - A->split[rank]);
            STj.push_back(A->entry[i].col - A->split[rank]);
            STval.push_back(1);
        }
        else{
//            if ( (-A->values[i] / maxPerCol[A->col[i]]) > connStrength) {
            STi.push_back(A->entry[i].row - A->split[rank]);
            STj.push_back(A->entry[i].col - A->split[rank]);
            STval.push_back( -A->entry[i].val / maxPerCol[A->entry[i].col] );
//            }
        }
    }

//    if(rank==1)
//        for(i=0; i<STi.size(); i++)
//            cout << "ST: " << "[" << STi[i]+1 << "," << STj[i]+1 << "] = " << STval[i] << endl;

    // ******************************** compute max per column - version 2 - of A is symmetric matrices ********************************

/*
    // since A is symmetric, use maxPerRow for local entries on each process. receive the remote ones like matvec.

    //vSend are maxPerCol for remote elements that should be sent to other processes.
    for(i=0;i<A->vIndexSize;i++)
        A->vSend[i] = maxPerRow[( A->vIndex[i] )];

    MPI_Request* requests = new MPI_Request[A->numSendProc+A->numRecvProc];
    MPI_Status* statuses = new MPI_Status[A->numSendProc+A->numRecvProc];

    //vecValues are maxperCol for remote elements that are received from other processes.
    // Do not recv from self.
    for(i = 0; i < A->numRecvProc; i++)
        MPI_Irecv(&A->vecValues[A->rdispls[A->recvProcRank[i]]], A->recvProcCount[i], MPI_DOUBLE, A->recvProcRank[i], 1, comm, &(requests[i]));

    // Do not send to self.
    for(i = 0; i < A->numSendProc; i++)
        MPI_Isend(&A->vSend[A->vdispls[A->sendProcRank[i]]], A->sendProcCount[i], MPI_DOUBLE, A->sendProcRank[i], 1, comm, &(requests[A->numRecvProc+i]));

    // ******************************** compute ST - version 2 ********************************

    std::vector<long> STi;
    std::vector<long> STj;
    std::vector<double> STval;

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
    for (i = 0; i < A->nnz_l_local; ++i, iter2++) {
        // diagonal entry
        if(A->row_local[i] == A->col_local[i]){
            STi.push_back(A->row_local[i]);
            STj.push_back(A->col_local[i]);
            STval.push_back(1);
            continue;
        }
        STi.push_back(A->row_local[i]);
        STj.push_back(A->col_local[i]);
        STval.push_back( -A->values_local[i] / maxPerRow[A->col_local[i]] );
    }

    MPI_Waitall(A->numSendProc+A->numRecvProc, requests, statuses);

    // add OpenMP just like matvec.
//    iter = 0;
//    for (i = 0; i < A->col_remote_size; ++i) {
//        for (unsigned int j = 0; j < A->nnz_col_remote[i]; ++j, ++iter) {
//            STi.push_back(A->row_remote[A->indicesP_remote[iter]]);
//            STj.push_back(A->col_remote2[A->indicesP_remote[iter]]);
//            STval.push_back( -A->values_remote[A->indicesP_remote[iter]] / A->vecValues[A->col_remote[A->indicesP_remote[iter]]] );
//        }
//    }

    // remote ST values
    // add OpenMP just like matvec.
    iter = 0;
    for (i = 0; i < A->vElement_remote.size(); ++i) {
        for (unsigned int j = 0; j < A->vElementRep_remote[i]; ++j, ++iter) {
//            w[A->row_remote[A->indicesP_remote[iter]]] += A->values_remote[A->indicesP_remote[iter]] * A->vecValues[A->col_remote[A->indicesP_remote[iter]]];
            STi.push_back(A->row_remote[iter]);
            STj.push_back(A->col_remote2[iter]);
            STval.push_back( -A->values_remote[iter] / A->vecValues[i] );
        }
    }

//    if(rank==0)
//        for(i=0; i<STi.size(); i++)
//            cout << "ST: " << "[" << STi[i]+1 << "," << STj[i]+1 << "] = " << STval[i] << endl;
*/

    // *************************** make S symmetric and apply the connection strength parameter ****************************

    std::vector<unsigned long> Si2;
    std::vector<unsigned long> Sj2;
    std::vector<double> Sval2;

    for(i=0; i<Si.size(); i++){
        if (Sval[i] <= connStrength && STval[i] <= connStrength)
            continue;
        else if (Sval[i] > connStrength && STval[i] <= connStrength){
            Si2.push_back(Si[i]);
            Sj2.push_back(Sj[i]);
            Sval2.push_back(0.5*Sval[i]);
        }
        else if (Sval[i] <= connStrength && STval[i] > connStrength){
            Si2.push_back(Si[i]);
            Sj2.push_back(Sj[i]);
            Sval2.push_back(0.5*STval[i]);
        }
        else{
            Si2.push_back(Si[i]);
            Sj2.push_back(Sj[i]);
            Sval2.push_back(0.5*(Sval[i] + STval[i]));
        }
    }

//    if(rank==1)
//        for(i=0; i<Si2.size(); i++){
//            cout << "S:  " << "[" << (Si2[i] - A->split[rank]) << "," << Sj2[i] << "] = \t" << Sval2[i] << endl;
//        }

    // S indices are local on each process, which means it starts from 0 on each process.
    S->StrengthMatrixSet(&(*(Si2.begin())), &(*(Sj2.begin())), &(*(Sval2.begin())), A->M, A->Mbig, Si2.size(), &(*(A->split.begin())), comm);

    return 0;
} // end of SaenaObject::createStrengthMatrix


// Using MIS(2) from the following paper by Luke Olson:
// EXPOSING FINE-GRAINED PARALLELISM IN ALGEBRAIC MULTIGRID METHODS
int SaenaObject::aggregation(StrengthMatrix* S, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew) {

    // For each node, first assign it to a 1-distance root. If there is not any root in distance-1, find a distance-2 root.
    // If there is not any root in distance-2, that node should become a root.

    // variables used in this function:
    // weight[i]: the two most left bits store the status of node i, the other 62 bits store weight assigned to that node.
    //            status of a node: 1 for 01 not assigned, 0 for 00 assigned, 2 for 10 root
    //            the max value for weight is 2^63 - 1
    //            weight is first generated randomly by randomVector function and saved in initialWeight. During the
    //            aggregation process, it becomes the weight of the node's aggregate.

    MPI_Comm comm = S->comm;

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    unsigned long i, j;
    unsigned long size = S->M;

    std::vector<unsigned long> aggArray; // root nodes.
    std::vector<unsigned long> aggregate2(size);
//    std::vector<unsigned long> aggStatus2(size); // 1 for 01 not assigned, 0 for 00 assigned, 2 for 10 root
    std::vector<unsigned long> weight(size);
    std::vector<unsigned long> weight2(size);
    std::vector<unsigned long> initialWeight(size);

    randomVector(initialWeight, S->Mbig, S, comm);
//    randomVector3(initialWeight, S->Mbig, S, comm);
//    randomVector4(initialWeight, S->Mbig);

//    if(rank==1){
//        cout << endl << "after initialization!" << endl;
//        for (i = 0; i < size; ++i)
//            cout << i << "\tinitialWeight = " << initialWeight[i] << endl;}

    const int wOffset = 62;
    const unsigned long weightMax = (1UL<<wOffset) - 1;
    const unsigned long UNDECIDED = 1UL<<wOffset;
    const unsigned long ROOT = 1UL<<(wOffset+1);
    const unsigned long UNDECIDED_OR_ROOT = 3UL<<wOffset;
    unsigned long weightTemp, aggregateTemp, aggStatusTemp;
    int* root_distance = (int*)malloc(sizeof(int)*size);
    // root_distance is initialized to 3(11). root = 0 (00), 1-distance root = 1 (01), 2-distance root = 2 (10).
    bool* dist1or2undecided = (bool*)malloc(sizeof(bool)*size);
    // if there is a distance-2 neighbor which is undecided, set this to true.
//    fill(&dist2undecided[0], &dist2undecided[size], 0);
    bool continueAggLocal;
    bool continueAgg = true;
    unsigned long i_remote, j_remote, weight_remote, agg_remote;
    unsigned long iter, col_index;
    int whileiter = 0;

    MPI_Request *requests = new MPI_Request[S->numSendProc + S->numRecvProc];
    MPI_Status *statuses  = new MPI_Status[S->numSendProc + S->numRecvProc];

    // initialization -> this part is merged to the first "for" loop in the following "while".
    for(i=0; i<size; i++) {
        aggregate[i] = i + S->split[rank];
//        aggStatus2[i] = 1;
        // Boundary nodes are the ones which only have one neighbor (so one nnzPerRow), which is the diagonal element. They are roots for every coarse-grid.
        if(S->nnzPerRow[i] == 1){
            weight[i] = ( 2UL<<wOffset | initialWeight[i] );
            root_distance[i] = 0;
            aggArray.push_back(aggregate[i]);
//            if(rank==0) cout << "boundary: " << i+S->split[rank] << endl;
        }
        else{
            weight[i] = ( 1UL<<wOffset | initialWeight[i] ); // status of each node is initialized to 1 and its weight to initialWeight.
//            if(rank==0) cout << "V[" << i+S->split[rank] << "] = " << initialWeight[i] << ";" << endl;
        }
    }

//    if(rank==0){
//        cout << "if(rank==0){" << endl;
//        for(i=0; i<size; i++)
//            cout << "V[" << i << "] = " << initialWeight[i] << ";" << endl;
//        cout << "}" << endl;
//    }
//    MPI_Barrier(comm);
//    if(rank==1){
//        cout << "if(rank==1){" << endl;
//        for(i=0; i<size; i++)
//            cout << "V[" << i << "] = " << initialWeight[i] << ";" << endl;
//        cout << "}" << endl;
//    }
//    MPI_Barrier(comm);

    while(continueAgg) {
        // ******************************* first round of max computation *******************************
        // first "compute max" is local. The second one is both local and remote.
        // for loop is of size "number of rows". it checks if a node is UNDECIDED. Then, it goes over its neighbors, which are nonzeros on that row.
        // if the neighbor is UNDECIDED or ROOT, and its weight is higher than weightTemp, then than node will be chosen for root.
        // UNDECIDED is also considered because it may become a root later, and it may be a better root for that node.
        // In the next "for" loop, weight and aggregate are updated, but not the status.

        iter = 0;
        for (i = 0; i < size; ++i) {
            if ( (weight[i] >> wOffset == 0) && (root_distance[i] == 2) ) {
                for (j = 0; j < S->nnzPerRow_local[i]; ++j, ++iter) {
                    col_index = S->col_local[S->indicesP_local[iter]] - S->split[rank];
                    if (weight[col_index] & ROOT) {
//                        cout << "$$$$$$$$$$$$$$$$$$$$$$$$$" << i << "\t col_index = " << col_index << "\t weight[col_index] = " << (weight[col_index] & weightMax) << "\t aggregate = " << S->col_local[S->indicesP_local[iter]] << endl;
                        weight[i] = (0UL << wOffset | (weight[col_index] & weightMax));
                        aggregate[i] = S->col_local[S->indicesP_local[iter]];
                        root_distance[i] = 1;
                    }
//                    break; todo: try to add this break.
                }
            }else
                iter += S->nnzPerRow_local[i];
        }

        // todo: check the aggregation for deadlock: two neighbors wait for each other to have a status other than 1, while both have status 1.
        //distance-1 aggregate
        iter = 0;
        for (i = 0; i < size; ++i) {
            if(weight[i]&UNDECIDED) {
//                if(i==25) cout << ">>>>>>>>>>>>>>>>>25 root ==================== " << root_distance[25] << "\taggregate = " << aggregate[25] << endl;
//            if(weight[i]>>wOffset <= 1) {
                root_distance[i] = 3; // initialization
                dist1or2undecided[i] = false; // initialization
                aggregateTemp = aggregate[i];
                weightTemp = weight[i]&weightMax;
//                weight2[i] = weight[i]&weightMax;
//                aggStatusTemp = 1UL; // this will be used for aggStatus2, and aggStatus2 will be used for the remote part.
                for (j = 0; j < S->nnzPerRow_local[i]; ++j, ++iter) {
                    col_index = S->col_local[S->indicesP_local[iter]] - S->split[rank];
                    if(weight[col_index] & ROOT){
                        weight[i] = (0UL << wOffset | (weight[col_index] & weightMax)); // 0UL << wOffset is not required.
                        weightTemp = weight[i];
                        aggregate[i] = S->col_local[S->indicesP_local[iter]];
                        aggregateTemp = aggregate[i];
                        root_distance[i] = 1;
                        dist1or2undecided[i] = false;

//                        if(rank==0) cout << i+S->split[rank] << "\t assigned to = " << aggregate[i] << " distance-1 local \t weight = " << weight[i] << endl;
//                        break; todo: try to add this break. You can define nnzPerRowScan_local.
                    } else if( (weight[col_index] & UNDECIDED) && (initialWeight[col_index] >= weightTemp) ){
                        // if there is an UNDECIDED neighbor, and its weight is bigger than this node, this node should
                        // wait until the next round of the "while" loop to see if that neighbor becomes root or assigned.
//                        if(rank==0) cout << i+S->split[rank] << "\t neighbor = " << S->col_local[S->indicesP_local[iter]] << endl;

                        if((initialWeight[col_index] > weightTemp) || ((initialWeight[col_index] == weightTemp) && (aggregate[col_index] > i+S->split[rank])) ){
//                            if(rank==0 && i==10) cout << "??????????? " << col_index << "\tweight = " << (weight[col_index]&weightMax) << "\tagg = " << aggregate[col_index] << endl;
                            weightTemp = (weight[col_index] & weightMax);
                            aggregateTemp = S->col_local[S->indicesP_local[iter]];
                            root_distance[i] = 1;
                            dist1or2undecided[i] = true;
                        }
                    }
                }
                weight2[i]    = weightTemp;
                aggregate2[i] = aggregateTemp;
//                if(rank==1) cout << i+S->split[rank] << "," << aggregate2[i] << "\t\t";
            }else
                iter += S->nnzPerRow_local[i];
        }
//        if(rank==0) cout << "1>>>>>>>>>>>>>>>>>10 root ==================== " << root_distance[10] << "\taggregate = " << aggregate[10] << endl;

        // todo: for distance-1 it is probably safe to remove this for loop, and change weight2 to weight and aggregate2 to aggregate at the end of the previous for loop.
        for (i = 0; i < size; ++i) {
            if( (S->nnzPerRow_local[i]!=0) && (weight[i]&UNDECIDED) && (root_distance[i]==1)) {
                weight[i] = (1UL << wOffset | weight2[i] & weightMax );
                aggregate[i] = aggregate2[i];
//                if(rank==0) cout << i+S->split[rank] << "\t" << aggregate[i] << "\t" << aggregate2[i] << endl;
            }
        }
//        if(rank==0) cout << "2>>>>>>>>>>>>>>>>>10 root ==================== " << root_distance[10] << "\taggregate = " << aggregate[10] << endl;

        //    if(rank==0){
        //        cout << endl << "after first max computation!" << endl;
        //        for (i = 0; i < size; ++i)
        //            cout << i << "\tweight = " << weight[i] << "\tindex = " << aggregate[i] << endl;
        //    }

        // ******************************* exchange remote max values for the second round of max computation *******************************

        // vSend[2*i]:   the first right 62 bits of vSend is maxPerCol for remote elements that should be sent to other processes.
        //               the first left 2 bits of vSend are aggStatus.
        // vSend[2*i+1]: the first right 62 bits of vSend is aggregate for remote elements that should be sent to other processes.
        //               the first left 2 bits are root_distance.
        //               root_distance is initialized to 3(11). root = 0 (00), 1-distance root = 1 (01), 2-distance root = 2 (10).

        // the following shows how the data are being stored in vecValues:
//        iter = 0;
//        if(rank==1)
//            for (i = 0; i < S->col_remote_size; ++i)
//                for (j = 0; j < S->nnz_col_remote[i]; ++j, ++iter){
//                    cout << "row:" << S->row_remote[iter]+S->split[rank] << "\tneighbor(col) = " << S->col_remote2[iter]
//                         << "\t weight of neighbor = "          << (S->vecValues[2*S->col_remote[iter]]&weightMax)
//                         << "\t\t status of neighbor = "        << (S->vecValues[2*S->col_remote[iter]]>>wOffset)
//                         << "\t root_distance of neighbor = " << (S->vecValues[2*S->col_remote[iter]+1]&weightMax)
//                         << "\t status of agg = "               << (S->vecValues[2*S->col_remote[iter]+1]>>wOffset)
//                         << endl;
//                }

//        if(rank==1) cout << endl << endl;
        for (i = 0; i < S->vIndexSize; i++){
//            S->vSend[i] = weight[(S->vIndex[i])];
            S->vSend[2*i] = weight[S->vIndex[i]];
            aggStatusTemp = (unsigned long)root_distance[S->vIndex[i]]; // this is root_distance of the neighbor's aggregate.
            S->vSend[2*i+1] = ( (aggStatusTemp<<wOffset) | (aggregate[S->vIndex[i]]&weightMax) );
//            if(rank==1) cout << "vsend: " << S->vIndex[i]+S->split[rank] << "\tw = " << (S->vSend[2*i]&weightMax) << "\tst = " << (S->vSend[2*i]>>wOffset) << "\tagg = " << (S->vSend[2*i+1]&weightMax) << "\t oneDis = " << (S->vSend[2*i+1]>>wOffset) << endl;
        }

        for (i = 0; i < S->numRecvProc; i++)
            MPI_Irecv(&S->vecValues[S->rdispls[S->recvProcRank[i]]], S->recvProcCount[i], MPI_UNSIGNED_LONG,
                      S->recvProcRank[i], 1, comm, &(requests[i]));

        for (i = 0; i < S->numSendProc; i++)
            MPI_Isend(&S->vSend[S->vdispls[S->sendProcRank[i]]], S->sendProcCount[i], MPI_UNSIGNED_LONG,
                      S->sendProcRank[i], 1, comm, &(requests[S->numRecvProc + i]));

        // ******************************* second round of max computation *******************************
        // "for" loop is of size "number of rows". it checks if a node is UNDECIDED and also if it does not have a root of distance one.
        // Roots of distance 1 have priority. Then, it goes over its neighbors, which are nonzeros on that row.
        // The neighbor should be UNDECIDED or assigned. Then, it should have a 1-distance root, because we are looking for a distance-2 root.
        // Finally, its weight should be higher than weightTemp, then than node will be chosen for root.
        // UNDECIDED is also considered because it may become have a 1-distance root later.
        // In the next "for" loop, weight and aggregate are updated, but not the status.

        // local part - distance-2 aggregate
        iter = 0;
        for (i = 0; i < size; ++i) {
//            if(i==34) cout << ">>>>>>>>>>>>>>>>>34 root ==================== " << root_distance[34] << "\taggregate = " << aggregate[34] << endl;
            if( (weight[i]&UNDECIDED) && root_distance[i]!=1) { // root_distance cannot be 2 or 0 here.
//                oneDistanceRoot[i] = false;
                aggregateTemp = aggregate[i];
                weightTemp    = weight[i];
//                aggStatusTemp = 1UL; // this will be used for aggStatus2, and aggStatus2 will be used for the remote part.
                for (j = 0; j < S->nnzPerRow_local[i]; ++j, ++iter) {
                    col_index = S->col_local[S->indicesP_local[iter]] - S->split[rank];
                    if( (weight[col_index]>>wOffset) <= 1 && root_distance[col_index]==1){
                        if( ((weight[col_index]&weightMax) > (weightTemp&weightMax)) ||
                                ( ((weight[col_index]&weightMax) == (weightTemp&weightMax)) && (aggregate[col_index] > i+S->split[rank]) ) ){

//                            if(rank==1 && i==2) cout << "??????????? " << col_index << "\tweight = " << (weight[col_index]&weightMax) << "\tagg = " << aggregate[col_index] << endl;
                            aggregateTemp = aggregate[col_index];
                            weightTemp    = weight[col_index];
                            root_distance[i] = 2;
                            if(weight[col_index]&UNDECIDED)
                                dist1or2undecided[i] = true;
                            else
                                dist1or2undecided[i] = false;

                        }
                    }
                }
                weight2[i]    = weightTemp;
                aggregate2[i] = aggregateTemp;
//                aggStatus2[i] = aggStatusTemp; // this is stored only to be compared with the remote one in the remote part.
            }else
                iter += S->nnzPerRow_local[i];
        }
//        if(rank==1) cout << "3>>>>>>>>>>>>>>>>>2 root ==================== " << root_distance[2] << "\taggregate = " << aggregate[2] << endl;

        for (i = 0; i < size; ++i) {
            if( (S->nnzPerRow_local[i]!=0) && (weight[i]&UNDECIDED) && (root_distance[i]==2) ) {
                aggregate[i] = aggregate2[i];
                aggStatusTemp = (weight[i]>>wOffset);
//                if (aggregate[i] < S->split[rank] || aggregate[i] >= S->split[rank+1]) // this is distance-2 to a remote root.
//                    aggStatusTemp = 0;
//                cout << "before: " << (weight[i]>>wOffset) << ",after: " << aggStatusTemp << endl;
                weight[i] = (aggStatusTemp<<wOffset | (weight2[i]&weightMax) );
            }
        }
//        if(rank==1) cout << "4>>>>>>>>>>>>>>>>>2 root ==================== " << root_distance[2] << "\taggregate = " << aggregate[2] << endl;

//        if(rank==1){
//            cout << endl << "after second max computation!" << endl;
//            for (i = 0; i < size; ++i)
//                cout << i << "\tweight = " << weight[i] << "\tindex = " << aggregate[i] << "\taggStatus = " << aggStatus[i] << endl;
//        }

        MPI_Waitall(S->numSendProc + S->numRecvProc, requests, statuses);

//        delete requests; // todo: delete requests and statuses in whole project, if it is required.
//        delete statuses;

//        MPI_Barrier(comm);
//        iter = 0;
//        if(rank==1)
//            for (i = 0; i < S->col_remote_size; ++i)
//                for (j = 0; j < S->nnz_col_remote[i]; ++j, ++iter){
//                    cout << "row:" << S->row_remote[iter]+S->split[rank] << "\tneighbor(col) = " << S->col_remote2[iter]
//                         << "\tweight of neighbor = "          << (S->vecValues[2*S->col_remote[iter]]&weightMax)
//                         << "\t\tstatus of neighbor = "        << (S->vecValues[2*S->col_remote[iter]]>>wOffset)
//                         << "\t root_distance of neighbor = "  << (S->vecValues[2*S->col_remote[iter]+1]&weightMax)
//                         << "\tstatus of agg = "               << (S->vecValues[2*S->col_remote[iter]+1]>>wOffset)
//                         << endl;
//                }
//        MPI_Barrier(comm);

        // remote part
        // store the max of rows of remote elements in weight2 and aggregate2.
        iter = 0;
        for (i = 0; i < S->col_remote_size; ++i) {
            for (j = 0; j < S->nnz_col_remote[i]; ++j, ++iter) {
                i_remote = S->row_remote[iter];
                j_remote = S->col_remote[iter];
                weight_remote = S->vecValues[2 * j_remote];
                agg_remote    = S->vecValues[2 * j_remote + 1];

                if (weight[i_remote] & UNDECIDED) {

                    //distance-1 aggregate
                    if (weight_remote & ROOT) {
                        weight[i_remote] = (0UL << wOffset | (weight_remote & weightMax));
//                        weight2[i_remote] = weight[i_remote];
                        aggregate[i_remote] = (agg_remote & weightMax);
                        root_distance[i_remote] = 1;
                        dist1or2undecided[i_remote] = false;
//                        if(rank==0) cout << i+S->split[rank] << "\t assigned to = " << aggregate[i_remote] << " distance-1 remote \t weight = " << weight[i_remote] << endl;
                    } else if (weight_remote & UNDECIDED) {

                        if (root_distance[i_remote] == 1)
                            weightTemp = (weight2[i_remote] & weightMax);
                        else{
                            weightTemp = initialWeight[i_remote];
                        }

                        if ( ( (weight_remote & weightMax) > weightTemp) ||
                                ( ( (weight_remote & weightMax) == weightTemp) && ( ((agg_remote & weightMax) > i_remote+S->split[rank]) )) ) {
//                            if(rank==1) cout << "first  before\t" << "row = " << S->row_remote[iter]+S->split[rank] << "  \tinitial weight = " << initialWeight[S->row_remote[iter]] << ",\taggregate = " << aggregate2[S->row_remote[iter]] << "\tremote weight = " << (weight2[S->row_remote[iter]]&weightMax) << "\tremote status = " << (S->vecValues[2*S->col_remote[iter]+1]>>wOffset) << "\taggStat2 = " << aggStatus2[S->row_remote[iter]] << endl;
                            weight2[i_remote] = (weight_remote & weightMax);
                            aggregate2[i_remote] = (agg_remote & weightMax);
                            root_distance[i_remote] = 1;
                            dist1or2undecided[i_remote] = true;
//                            aggStatus2[S->row_remote[iter]] = (S->vecValues[2 * S->col_remote[iter]] >> wOffset);
//                            if(rank==1) cout << "first  after \t" << "row = " << S->row_remote[iter]+S->split[rank] << "  \tinitial weight = " << initialWeight[S->row_remote[iter]] << ",\taggregate = " << aggregate2[S->row_remote[iter]] << "\t\tremote weight = " << (weight2[S->row_remote[iter]]&weightMax) << "\tremote status = " << (S->vecValues[2*S->col_remote[iter]+1]>>wOffset) << "\taggStat2 = " << aggStatus2[S->row_remote[iter]] << endl;
                        }
                    }

                    //distance-2 aggregate
                    if (weight_remote >> wOffset == 0) { // the neighbor is assigned (00).
                        if (root_distance[i_remote] != 1 && ((agg_remote >> wOffset) == 1) ){ // this is root_distance of the neighbor.

                            if( ( (weight_remote & weightMax) > (weight2[i_remote] & weightMax) ) ||
                                    (( (weight_remote & weightMax) == (weight2[i_remote] & weightMax) ) && (agg_remote & weightMax) > i_remote+S->split[rank] ) ){

                                weight2[i_remote] = (weight_remote & weightMax);
                                aggregate2[i_remote] = (agg_remote & weightMax);
//                            aggStatus2[S->row_remote[iter]] = (S->vecValues[2 * S->col_remote[iter]]>>wOffset); // this is always 0.
                                root_distance[i_remote] = 2;
                                dist1or2undecided[i_remote] = false;

                            }

                        }
                    } else if (weight_remote & UNDECIDED) { // the neighbor is UNDECIDED (01).
                        if (root_distance[i_remote] != 1 ) {

                            if( (weight_remote & weightMax) > (weight2[i_remote] & weightMax) ||
                                    ((weight_remote & weightMax) > (weight2[i_remote] & weightMax)) && (agg_remote & weightMax) > i_remote+S->split[rank] ){

                                weight2[i_remote] = (weight_remote & weightMax);
                                aggregate2[i_remote] = (agg_remote & weightMax);
//                            aggStatus2[S->row_remote[iter]] = (S->vecValues[2 * S->col_remote[iter]]>>wOffset); // this is always 0.
                                root_distance[i_remote] = 2;
                                dist1or2undecided[i_remote] = true;

                            }
                        }
                    }
                    // if the node is assigned and root_distance is 2, check to see if there is a root in distance-1
                } else if( (weight[i_remote]>>wOffset == 0) && (root_distance[i_remote] == 2) ){
                    if (weight_remote & ROOT){
                        weight[i_remote] = (0UL << wOffset | (weight_remote & weightMax));
                        aggregate[i_remote] = (agg_remote & weightMax);
                        root_distance[i_remote] = 1;
                    }
                }
            }
        }
//        if(rank==1) cout << "5>>>>>>>>>>>>>>>>>2 root ==================== " << root_distance[2] << "\taggregate = " << aggregate[2] << endl;

        // put weight2 in weight and aggregate2 in aggregate.
        // if a row does not have a remote element then (weight2[i]&weightMax) == (weight[i]&weightMax)
        // update aggStatus of remote elements at the same time
        for(i=0; i<size; i++){
            if( (weight[i]&UNDECIDED) && aggregate[i] != aggregate2[i] ){
                aggregate[i] = aggregate2[i];
                weight[i] = ( 1UL<<wOffset | weight2[i] );
//                if(aggStatus2[i] != 1) // if this is 1, it should go to the next aggregation round.
//                    weight[i] = (0UL<<wOffset | weight2[i]&weightMax);
            }
        }
//        if(rank==1) cout << "6>>>>>>>>>>>>>>>>>2 root ==================== " << root_distance[2] << "\taggregate = " << aggregate[2] << endl;

        // ******************************* Update Status *******************************
        // "for" loop is of size "number of rows". it checks if a node is UNDECIDED.
        // If aggregate of a node equals its index, that's a root.

//        if(rank==0) cout << "******************** Update Status ********************" << endl;
        for (i = 0; i < size; ++i) {
            if(weight[i]&UNDECIDED) {
//                if(rank==0) cout << "checking " << i << "\taggregate[i] = " << aggregate[i] << endl;
                // local
                if (aggregate[i] >= S->split[rank] && aggregate[i] < S->split[rank+1]) {
//                    if(rank==1) cout << "i = " << i << "\taggregate[i] = " << aggregate[i] << "\taggStatus[aggregate[i]] = " << aggStatus[aggregate[i]] << endl;
//                    if(rank==1) cout << "i = " << i+S->split[rank] << "\taggregate[i] = " << aggregate[i] << "\taggStatus[i] = " << (weight[i]>>wOffset) << endl;
                    if (aggregate[i] == i + S->split[rank]) { // this node is a root.
                        weight[i] = ( (2UL<<wOffset) | (weight[i]&weightMax) ); // change aggStatus of a root to 2.
                        root_distance[i] = 0;
                        aggArray.push_back(aggregate[i]);
//                        (*aggSize)++;
//                        if(rank==0) cout << "root " << "i = " << i+S->split[rank] << "\t weight = " << (weight[i]&weightMax) << endl;
                    } else if ( (root_distance[i] == 1 && (weight[aggregate[i] - S->split[rank]]&ROOT) && (!dist1or2undecided[i])) ||
                                (root_distance[i] == 2 && !dist1or2undecided[i]) ) {
//                        if(rank==0) cout << "assign " << "i = " << i+S->split[rank] << "\taggregate[i] = " << aggregate[i] << "\taggStatus[i] = " << (weight[i]>>wOffset) << endl;
                        weight[i] = ( (0UL<<wOffset) | (weight[i]&weightMax) );
//                        if(rank==0) cout << i+S->split[rank] << "\t assigned to = " << aggregate[i] << " distance-1 or 2 local final step, root_distance = " << root_distance[i] << "\t weight = " << weight[i]  << endl;
//                        if (root_distance[aggregate[i] - S->split[rank]] == 0) root_distance[i] = 1; // todo: this is WRONG!
                    }

                // remote
                }else{
//                    if(root_distance[i] == 1)
//                        continue;
                    if(root_distance[i] == 2 && !dist1or2undecided[i]){
                        weight[i] = (0UL<<wOffset | weight[i]&weightMax);
//                        if(rank==0) cout << i+S->split[rank] << "\t assigned to = " << aggregate[i] << " distance-2 remote final step" << endl;
                    }
//                    if(aggStatus2[i] != 1) // if it is 1, it should go to the next aggregation round.
//                        weight[i] = (0UL<<wOffset | weight2[i]&weightMax);
                }
            }
        }
//        if(rank==1) cout << "7>>>>>>>>>>>>>>>>>2 root ==================== " << root_distance[2] << "\taggregate = " << aggregate[2] << endl;

//        for(int k=0; k<nprocs; k++){
//            MPI_Barrier(comm);
//            if(rank==k){
//                cout << "final aggregate! rank:" << rank << ", iter = " << whileiter << endl;
//                for (i = 0; i < size; ++i){
//                    cout << "i = " << i+S->split[rank] << "\t\taggregate = " << aggregate[i] << "\t\taggStatus = "
//                         << (weight[i]>>wOffset) << "\t\tinitial weight = " << initialWeight[i]
//                         << "\t\tcurrent weight = " << (weight[i]&weightMax) << "\t\taggStat2 = " << aggStatus2[i]
//                         << "\t\toneDistanceRoot = " << oneDistanceRoot[i] << endl;
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
//        MPI_Barrier(comm);
//        cout << rank << "\tcontinueAgg = " << continueAgg << endl;
//        MPI_Barrier(comm);

//        MPI_Barrier(comm); if(rank==0) cout << "UNDECIDED: " << whileiter << endl; MPI_Barrier(comm);
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

    } //while(continueAgg)

//    MPI_Barrier(comm);
//    if(rank==nprocs-1) cout << "number of loops to find aggregation: " << whileiter << endl;
//    MPI_Barrier(comm);

//    for(i=0; i<size;i++)
//        if(rank==0) cout << "V[" << i+S->split[rank] << "] = " << initialWeight[i] << ";" << endl;

    free(root_distance);
    free(dist1or2undecided);

    // *************************** update aggregate to new indices ****************************

//    if(rank==2)
//        cout << endl << "S.M = " << S->M << ", S.nnz_l = " << S->nnz_l << ", S.nnz_l_local = " << S->nnz_l_local
//             << ", S.nnz_l_remote = " << S->nnz_l_remote << endl << endl;

//    if(rank==1){
//        cout << "aggregate:" << endl;
//        for(i=0; i<size; i++)
//            cout << i+S->split[rank] << "\t" << aggregate[i] << endl;
//        cout << endl;}

    // write the aggregate values of all the nodes to a file:
    // use this command to concatenate the output files:
    // cat aggregateSaena0.txt aggregateSaena1.txt > aggregateSaena.txt
    for(i=0; i<aggregate.size(); i++)
        aggregate[i]++;
    writeVectorToFileul(aggregate, S->Mbig, "aggregateSaena", comm);
    for(i=0; i<aggregate.size(); i++)
        aggregate[i]--;

    // aggArray is the set of root nodes.
    sort(aggArray.begin(), aggArray.end());

//    if(rank==1){
//        cout << "aggArray:" << aggArray.size() << endl;
//        for(auto i:aggArray)
//            cout << i << endl;
//        cout << endl;}

    // write the aggregate nodes to a file:
    // use this command to concatenate the output files:
    // cat aggArraySaena0.txt aggArraySaena1.txt > aggArraySaena.txt
    unsigned long aggArray_size, aggArray_size_total;
    aggArray_size = aggArray.size();
    MPI_Allreduce(&aggArray_size, &aggArray_size_total, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
    for(i=0; i<aggArray.size(); i++)
        aggArray[i]++;
    writeVectorToFileul(aggArray, aggArray_size_total, "aggArraySaena", comm);
    for(i=0; i<aggArray.size(); i++)
        aggArray[i]--;

    splitNew.resize(nprocs+1);
    fill(splitNew.begin(), splitNew.end(), 0);
    splitNew[rank] = aggArray.size();

    unsigned long* splitNewTemp = (unsigned long*)malloc(sizeof(unsigned long)*nprocs);
    MPI_Allreduce(&splitNew[0], splitNewTemp, nprocs, MPI_UNSIGNED_LONG, MPI_SUM, comm);

    // do scan on splitNew
    splitNew[0] = 0;
    for(i=1; i<nprocs+1; i++)
        splitNew[i] = splitNew[i-1] + splitNewTemp[i-1];

    free(splitNewTemp);

//    if(rank==0){
//        cout << "splitNew:" << endl;
//        for(i=0; i<nprocs+1; i++)
//            cout << S->split[i] << "\t" << splitNew[i] << endl;
//        cout << endl;}

    unsigned long procNum;
    vector<unsigned long> aggregateRemote;
    vector<unsigned long> recvProc;
    int* recvCount = (int*)malloc(sizeof(int)*nprocs);
    std::fill(recvCount, recvCount + nprocs, 0);

//    if(rank==1) cout << endl;
    bool* isAggRemote = (bool*)malloc(sizeof(bool)*size);
    // local: aggregate update to new values.
    for(i=0; i<size; i++){
        if(aggregate[i] >= S->split[rank] && aggregate[i] < S->split[rank+1]){
            aggregate[i] = lower_bound2(&*aggArray.begin(), &*aggArray.end(), aggregate[i]) + splitNew[rank];
//            if(rank==1) cout << aggregate[i] << endl;
            isAggRemote[i] = false;
        }else{
            isAggRemote[i] = true;
            aggregateRemote.push_back(aggregate[i]);
        }
    }

//    set<unsigned long> aggregateRemote2(aggregateRemote.begin(), aggregateRemote.end());
//    if(rank==1) cout << "i and procNum:" << endl;
//    for(auto i:aggregateRemote2){
//        procNum = lower_bound2(&S->split[0], &S->split[nprocs+1], i);
//        if(rank==1) cout << i << "\t" << procNum << endl;
//        recvCount[procNum]++;
//    }

    sort(aggregateRemote.begin(), aggregateRemote.end());
    auto last = unique(aggregateRemote.begin(), aggregateRemote.end());
    aggregateRemote.erase(last, aggregateRemote.end());
//    if(rank==1) cout << "i and procNum:" << endl;
    for(auto i:aggregateRemote){
        procNum = lower_bound2(&S->split[0], &S->split[nprocs], i);
        recvCount[procNum]++;
//        if(rank==1) cout << i << "\t" << procNum << endl;
    }

    int* vIndexCount = (int*)malloc(sizeof(int)*nprocs);
    MPI_Alltoall(recvCount, 1, MPI_INT, vIndexCount, 1, MPI_INT, comm);

//    if(rank==0){
//        cout << "vIndexCount:\t" << rank << endl;
//        for(i=0; i<nprocs; i++)
//            cout << vIndexCount[i] << endl;
//    }

    // this part is for isend and ireceive.
    std::vector<int> recvProcRank;
    std::vector<int> recvProcCount;
    std::vector<int> sendProcRank;
    std::vector<int> sendProcCount;
    int numRecvProc = 0;
    int numSendProc = 0;
    for(int i=0; i<nprocs; i++){
        if(recvCount[i]!=0){
            numRecvProc++;
            recvProcRank.push_back(i);
            recvProcCount.push_back(recvCount[i]);
        }
        if(vIndexCount[i]!=0){
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

    for (int i=1; i<nprocs; i++){
        vdispls[i] = vdispls[i-1] + vIndexCount[i-1];
        rdispls[i] = rdispls[i-1] + recvCount[i-1];
    }
    int vIndexSize = vdispls[nprocs-1] + vIndexCount[nprocs-1];
    int recvSize   = rdispls[nprocs-1] + recvCount[nprocs-1];

    unsigned long* vIndex = (unsigned long*)malloc(sizeof(unsigned long)*vIndexSize); // indices to be sent. And aggregateRemote are indices to be received.
    MPI_Alltoallv(&*aggregateRemote.begin(), recvCount, &*rdispls.begin(), MPI_UNSIGNED_LONG, vIndex, vIndexCount, &*vdispls.begin(), MPI_UNSIGNED_LONG, comm);
//    MPI_Alltoallv(&*aggregateRemote2.begin(), recvCount, &*rdispls.begin(), MPI_UNSIGNED_LONG, vIndex, vIndexCount, &*vdispls.begin(), MPI_UNSIGNED_LONG, comm);

    unsigned long* aggSend = (unsigned long*)malloc(sizeof(unsigned long*) * vIndexSize);
    unsigned long* aggRecv = (unsigned long*)malloc(sizeof(unsigned long*) * recvSize);

//    if(rank==0) cout << endl << "vSend:\trank:" << rank << endl;
    for(long i=0;i<vIndexSize;i++){
        aggSend[i] = aggregate[( vIndex[i]-S->split[rank] )];
//        if(rank==0) cout << "vIndex = " << vIndex[i] << "\taggSend = " << aggSend[i] << endl;
    }

    // replace this alltoallv with isend and irecv.
//    MPI_Alltoallv(aggSend, vIndexCount, &*(vdispls.begin()), MPI_UNSIGNED_LONG, aggRecv, recvCount, &*(rdispls.begin()), MPI_UNSIGNED_LONG, comm);

    MPI_Request *requests2 = new MPI_Request[numSendProc + numRecvProc];
    MPI_Status  *statuses2 = new MPI_Status[numSendProc + numRecvProc];

    for(int i = 0; i < numRecvProc; i++)
        MPI_Irecv(&aggRecv[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_UNSIGNED_LONG, recvProcRank[i], 1, comm, &(requests2[i]));

    //Next send the messages. Do not send to self.
    for(int i = 0; i < numSendProc; i++)
        MPI_Isend(&aggSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_UNSIGNED_LONG, sendProcRank[i], 1, comm, &(requests2[numRecvProc+i]));

    MPI_Waitall(numSendProc+numRecvProc, requests2, statuses2);

//    if(rank==1) cout << "aggRemote received:" << endl;
//    set<unsigned long>::iterator it;
//    for(i=0; i<size; i++){
//        if(isAggRemote[i]){
//            it = aggregateRemote2.find(aggregate[i]);
//            if(rank==1) cout << aggRecv[ distance(aggregateRemote2.begin(), it) ] << endl;
//            aggregate[i] = aggRecv[ distance(aggregateRemote2.begin(), it) ];
//        }
//    }
//    if(rank==1) cout << endl;

//    if(rank==1) cout << "aggRemote received:" << endl;
    // remote
    for(i=0; i<size; i++){
        if(isAggRemote[i]){
            aggregate[i] = aggRecv[ lower_bound2(&*aggregateRemote.begin(), &*aggregateRemote.end(), aggregate[i]) ];
//            if(rank==1) cout << i << "\t" << aggRecv[ lower_bound2(&*aggregateRemote.begin(), &*aggregateRemote.end(), aggregate[i]) ] << endl;
        }
    }
//    if(rank==1) cout << endl;

//    set<unsigned long> aggArray2(&aggregate[0], &aggregate[size]);
//    if(rank==1){
//        cout << "aggArray2:" << endl;
//        for(auto i:aggArray2)
//            cout << i << endl;
//        cout << endl;
//    }
    // update aggregate to new indices
//    set<unsigned long>::iterator it;
//    for(i=0; i<size; i++){
//        it = aggArray.find(aggregate[i]);
//        aggregate[i] = distance(aggArray.begin(), it) + splitNew[rank];
//    }

    free(aggSend);
    free(aggRecv);
    free(isAggRemote);
    free(recvCount);
    free(vIndexCount);
    free(vIndex);
    return 0;
} // end of SaenaObject::Aggregation


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


int SaenaObject::createProlongation(SaenaMatrix* A, std::vector<unsigned long>& aggregate, prolongMatrix* P){
    // formula for the prolongation matrix from Irad Yavneh's paper:
    // P = (I - 4/(3*rhoDA) * DA) * P_t

    // todo: check when you should update new aggregate values: before creating prolongation or after.

    // Here P is computed: P = A_w * P_t; in which P_t is aggregate, and A_w = I - w*Q*A, Q is inverse of diagonal of A.
    // Here A_w is computed on the fly, while adding values to P. Diagonal entries of A_w are 0, so they are skipped.
    // todo: think about A_F which is A filtered.
    // todo: think about smoothing preconditioners other than damped jacobi. check the following paper:
    // todo: Eran Treister and Irad Yavneh, Non-Galerkin Multigrid based on Sparsified Smoothed Aggregation. page22.

    P->comm = A->comm;
    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    unsigned int i, j;
    float omega = 0.67; // todo: receive omega as user input. it is usually 2/3 for 2d and 6/7 for 3d.

    P->Mbig = A->Mbig;
    P->Nbig = P->splitNew[nprocs]; // This is the number of aggregates, which is the number of columns of P.
    P->M = A->M;

    // store remote elements from aggregate in vSend to be sent to other processes.
    // todo: is it ok to use vSend instead of vSendULong? vSend is double and vSendULong is unsigned long.
    // todo: the same question for vecValues and Isend and Ireceive.
    for(i=0; i<A->vIndexSize; i++){
        A->vSendULong[i] = aggregate[( A->vIndex[i] )];
//        if(rank==1) cout <<  A->vIndex[i] << "\t" << A->vSend[i] << endl;
    }

    MPI_Request* requests = new MPI_Request[A->numSendProc+A->numRecvProc];
    MPI_Status*  statuses = new MPI_Status[A->numSendProc+A->numRecvProc];

    for(i = 0; i < A->numRecvProc; i++)
        MPI_Irecv(&A->vecValuesULong[A->rdispls[A->recvProcRank[i]]], A->recvProcCount[i], MPI_UNSIGNED_LONG, A->recvProcRank[i], 1, comm, &(requests[i]));

    for(i = 0; i < A->numSendProc; i++)
        MPI_Isend(&A->vSendULong[A->vdispls[A->sendProcRank[i]]], A->sendProcCount[i], MPI_UNSIGNED_LONG, A->sendProcRank[i], 1, comm, &(requests[A->numRecvProc+i]));

    std::vector<cooEntry> PEntryTemp;

    // P = (I - 4/(3*rhoDA) * DA) * P_t
    // aggreagte is used as P_t in the following "for" loop.
    // local
    long iter = 0;
    for (i = 0; i < A->M; ++i) {
        for (j = 0; j < A->nnzPerRow_local[i]; ++j, ++iter) {
            if(A->row_local[A->indicesP_local[iter]] == A->col_local[A->indicesP_local[iter]]-A->split[rank]){ // diagonal element
                PEntryTemp.push_back(cooEntry(A->row_local[A->indicesP_local[iter]],
                                              aggregate[ A->col_local[A->indicesP_local[iter]] - A->split[rank] ],
                                              1 - omega));
            }else{
                PEntryTemp.push_back(cooEntry(A->row_local[A->indicesP_local[iter]],
                                              aggregate[ A->col_local[A->indicesP_local[iter]] - A->split[rank] ],
                                              -omega * A->values_local[A->indicesP_local[iter]] * A->invDiag[A->row_local[A->indicesP_local[iter]]]));
            }
//            if(rank==1) cout << A->row_local[A->indicesP_local[iter]] << "\t" << aggregate[A->col_local[A->indicesP_local[iter]] - A->split[rank]] << "\t" << A->values_local[A->indicesP_local[iter]] * A->invDiag[A->row_local[A->indicesP_local[iter]]] << endl;
        }
    }

    MPI_Waitall(A->numSendProc+A->numRecvProc, requests, statuses);

    // remote
    iter = 0;
    for (i = 0; i < A->col_remote_size; ++i) {
        for (j = 0; j < A->nnzPerCol_remote[i]; ++j, ++iter) {
            PEntryTemp.push_back(cooEntry(A->row_remote[iter],
                                          A->vecValuesULong[A->col_remote[iter]],
                                          -omega * A->values_remote[iter] * A->invDiag[A->row_remote[iter]]));
//            P->values.push_back(A->values_remote[iter]);
//            if(rank==1) cout << A->row_remote[iter] << "\t" << A->vecValuesULong[A->col_remote[iter]] << "\t" << A->values_remote[iter] * A->invDiag[A->row_remote[iter]] << endl;
        }
    }

    std::sort(PEntryTemp.begin(), PEntryTemp.end());

//    if(rank==1)
//        for(i=0; i<PEntryTemp.size(); i++)
//            cout << PEntryTemp[i].row << "\t" << PEntryTemp[i].col << "\t" << PEntryTemp[i].val << endl;

    // remove duplicates.
    for(i=0; i<PEntryTemp.size(); i++){
        P->entry.push_back(PEntryTemp[i]);
        while(i<PEntryTemp.size()-1 && PEntryTemp[i] == PEntryTemp[i+1]){ // values of entries with the same row and col should be added.
            P->entry.back().val += PEntryTemp[i+1].val;
            i++;
        }
    }

//    if(rank==1)
//        for(i=0; i<P->entry.size(); i++)
//            cout << P->entry[i].row << "\t" << P->entry[i].col << "\t" << P->entry[i].val << endl;

    PEntryTemp.clear();

    P->nnz_l = P->entry.size();
    MPI_Allreduce(&P->nnz_l, &P->nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

    P->split = A->split;

    P->findLocalRemote(&*P->entry.begin());
//    P->findLocalRemote(&*P->row.begin(), &*P->col.begin(), &*P->values.begin(), comm);

    return 0;
}// end of SaenaObject::createProlongation


int SaenaObject::coarsen(SaenaMatrix* A, prolongMatrix* P, restrictMatrix* R, SaenaMatrix* Ac){

    MPI_Comm comm = P->comm;

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    unsigned long i, j;
    prolongMatrix RATemp(comm); // RATemp is being used to remove duplicates while pushing back to RA.

    // ************************************* RATemp - A local *************************************
    // Some local and remote elements of RATemp are computed here using local R and local A.

    unsigned int AMaxNnz, AMaxM;
    MPI_Allreduce(&A->nnz_l, &AMaxNnz, 1, MPI_UNSIGNED, MPI_MAX, comm);
    MPI_Allreduce(&A->M, &AMaxM, 1, MPI_UNSIGNED, MPI_MAX, comm);
//    MPI_Barrier(comm); printf("rank=%d, AMaxNnz=%d \n", rank, AMaxNnz); MPI_Barrier(comm);
    // todo: is this way better than using the previous Allreduce? reduce on processor 0, then broadcast to other processors.

    unsigned int* AnnzPerRow = (unsigned int*)malloc(sizeof(unsigned int)*AMaxM);
    fill(&AnnzPerRow[0], &AnnzPerRow[A->M], 0);
    for(i=0; i<A->nnz_l; i++)
        AnnzPerRow[A->entry[i].row - A->split[rank]]++;

//    if(rank==0)
//        for(i=0; i<A->M; i++)
//            cout << i << "\t" << AnnzPerRow[i] << endl;

    unsigned int* AnnzPerRowScan = (unsigned int*)malloc(sizeof(unsigned int)*(AMaxM+1));
    AnnzPerRowScan[0] = 0;
    for(i=0; i<A->M; i++){
        AnnzPerRowScan[i+1] = AnnzPerRowScan[i] + AnnzPerRow[i];
//        if(rank==0) printf("i=%lu, AnnzPerRow=%d, AnnzPerRowScan = %d\n", i, AnnzPerRow[i], AnnzPerRowScan[i]);
    }

    // todo: combine indicesP and indicesPRecv together.
    // find row-wise ordering for A and save it in indicesP
    unsigned long* indicesP = (unsigned long*)malloc(sizeof(unsigned long)*A->nnz_l);
    for(unsigned long i=0; i<A->nnz_l; i++)
        indicesP[i] = i;
    std::sort(indicesP, &indicesP[A->nnz_l], sort_indices2(&*A->entry.begin()));

    for(i=0; i<R->nnz_l_local; i++){
        for(j = AnnzPerRowScan[R->entry_local[i].col - P->split[rank]]; j < AnnzPerRowScan[R->entry_local[i].col - P->split[rank] + 1]; j++){
//            if(rank==0) cout << A->entry[indicesP[j]].row << "\t" << A->entry[indicesP[j]].col << "\t" << A->entry[indicesP[j]].val
//                             << "\t" << R->entry_local[i].col << "\t" << R->entry_local[i].col - P->split[rank] << endl;
            RATemp.entry.push_back(cooEntry(R->entry_local[i].row,
                                        A->entry[indicesP[j]].col,
                                        R->entry_local[i].val * A->entry[indicesP[j]].val));
        }
    }

//    if(rank==1)
//        for(i=0; i<RATemp.entry.size(); i++)
//            cout << RATemp.entry[i].row << "\t" << RATemp.entry[i].col << "\t" << RATemp.entry[i].val << endl;

    free(indicesP);

    // ************************************* RATemp - A remote *************************************

    // find the start and end of each block of R.
    unsigned int* RBlockStart = (unsigned int*)malloc(sizeof(unsigned int)*(nprocs+1));
    fill(RBlockStart, &RBlockStart[nprocs], 0);


//    MPI_Barrier(comm); printf("rank=%d here!!!!!!!! \n", rank); MPI_Barrier(comm);
//    MPI_Barrier(comm); printf("rank=%d entry = %ld \n", rank, R->entry_remote[0].col); MPI_Barrier(comm);
    long procNum = -1;
    if(R->entry_remote.size() > 0)
        procNum = lower_bound2(&*A->split.begin(), &*A->split.end(), R->entry_remote[0].col);
//    MPI_Barrier(comm); printf("rank=%d procNum = %ld \n", rank, procNum); MPI_Barrier(comm);


    unsigned int nnzIter = 1;
    for(i=1; i<R->entry_remote.size(); i++){
        nnzIter++;
        if(R->entry_remote[i].col >= A->split[procNum+1]){
            RBlockStart[procNum+1] = nnzIter-1;
            procNum = lower_bound2(&*A->split.begin(), &*A->split.end(), R->entry_remote[i].col);
        }
//        if(rank==2) cout << "procNum = " << procNum << "\tcol = " << R->entry_remote[i].col << "\tnnzIter = " << nnzIter << endl;
    }
    RBlockStart[rank+1] = RBlockStart[rank]; // there is not any nonzero of R_remote on the local processor.
    fill(&RBlockStart[procNum+1], &RBlockStart[nprocs+1], nnzIter);

//    if(rank==1){
//        cout << "RBlockStart: " << endl;
//        for(i=0; i<nprocs+1; i++)
//            cout << RBlockStart[i] << endl;}

    unsigned long* indicesPRecv = (unsigned long*)malloc(sizeof(unsigned long)*AMaxNnz);

    //    printf("rank=%d A.nnz=%u \n", rank, A->nnz_l);
    cooEntry* Arecv = (cooEntry*)malloc(sizeof(cooEntry)*AMaxNnz);
    int left, right;
    unsigned int nnzRecv;
    long ARecvM;
    MPI_Status sendRecvStatus;

    // todo: change the algorithm so every processor sends data only to the next one and receives from the previous one in each iteration.
    for(int i = 1; i < nprocs; i++) {
        // send A to the right processor, recieve A from the left processor. "left" decreases by one in each iteration. "right" increases by one.
        right = (rank + i) % nprocs;
        left = rank - i;
        if (left < 0)
            left += nprocs;

        // *************************** RATemp - A remote - sendrecv(size) ****************************

        // use sender rank for send and receive tags.
        MPI_Sendrecv(&A->nnz_l, 1, MPI_UNSIGNED, right, rank, &nnzRecv, 1, MPI_UNSIGNED, left, left, comm, &sendRecvStatus);
//        int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf,
//                         int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status)
//        if(rank==1) printf("i=%d, rank=%d, left=%d, right=%d \n", i, rank, left, right);

//        if(rank==2) cout << "own A->nnz_l = " << A->nnz_l << "\tnnzRecv = " << nnzRecv << endl;

        // *************************** RATemp - A remote - sendrecv(A) ****************************

        // use sender rank for send and receive tags.
        MPI_Sendrecv(&A->entry[0], A->nnz_l, cooEntry::mpi_datatype(), right, rank, Arecv, nnzRecv, cooEntry::mpi_datatype(), left, left, comm, &sendRecvStatus);
//        if(rank==1) for(int j=0; j<nnzRecv; j++)
//                        printf("j=%d \t %lu \t %lu \t %f \n", j, Arecv[j].row, Arecv[j].col, Arecv[j].val);

        // *************************** RATemp - A remote - multiplication ****************************

        ARecvM = A->split[left+1] - A->split[left];
        fill(&AnnzPerRow[0], &AnnzPerRow[ARecvM], 0);
        for(j=0; j<nnzRecv; j++){
            AnnzPerRow[Arecv[j].row - A->split[left]]++;
        }

//    if(rank==2)
//        for(i=0; i<A->M; i++)
//            cout << AnnzPerRow[i] << endl;

        AnnzPerRowScan[0] = 0;
        for(j=0; j<ARecvM; j++){
            AnnzPerRowScan[j+1] = AnnzPerRowScan[j] + AnnzPerRow[j];
//            if(rank==2) printf("i=%d, AnnzPerRow=%d, AnnzPerRowScan = %d\n", i, AnnzPerRow[i], AnnzPerRowScan[i]);
        }

        // find row-wise ordering for Arecv and save it in indicesPRecv
        for(unsigned long i=0; i<nnzRecv; i++)
            indicesPRecv[i] = i;
        std::sort(indicesPRecv, &indicesPRecv[nnzRecv], sort_indices2(Arecv));

//        if(rank==1) cout << "block start = " << RBlockStart[left] << "\tend = " << RBlockStart[left+1] << "\tleft rank = " << left << "\t i = " << i << endl;
        for(j=RBlockStart[left]; j<RBlockStart[left+1]; j++){
//            if(rank==1) cout << "col = " << R->entry_remote[j].col << "\tcol-split = " << R->entry_remote[j].col - P->split[left] << "\tstart = " << AnnzPerRowScan[R->entry_remote[j].col - P->split[left]] << "\tend = " << AnnzPerRowScan[R->entry_remote[j].col - P->split[left] + 1] << endl;
            for(unsigned long k = AnnzPerRowScan[R->entry_remote[j].col - P->split[left]]; k < AnnzPerRowScan[R->entry_remote[j].col - P->split[left] + 1]; k++){
//                if(rank==0) cout << Arecv[indicesPRecv[k]].row << "\t" << Arecv[indicesPRecv[k]].col << "\t" << Arecv[indicesPRecv[k]].val << endl;
                RATemp.entry.push_back(cooEntry(R->entry_remote[j].row,
                                            Arecv[indicesPRecv[k]].col,
                                            R->entry_remote[j].val * Arecv[indicesPRecv[k]].val));
            }
        }


    } //for i
//    MPI_Barrier(comm); printf("rank=%d here!!!!!!!! \n", rank); MPI_Barrier(comm);

    free(indicesPRecv);
    free(AnnzPerRow);
    free(AnnzPerRowScan);
    free(Arecv);
    free(RBlockStart);

    std::sort(RATemp.entry.begin(), RATemp.entry.end());

//    if(rank==1)
//        for(j=0; j<RATemp.entry.size(); j++)
//            cout << RATemp.entry[j].row << "\t" << RATemp.entry[j].col << "\t" << RATemp.entry[j].val << endl;

    prolongMatrix RA(comm);

    // remove duplicates.
    for(i=0; i<RATemp.entry.size(); i++){
        RA.entry.push_back(RATemp.entry[i]);
//        if(rank==1) cout << endl << "start:" << endl << RATemp.entry[i].val << endl;
        while(i<RATemp.entry.size()-1 && RATemp.entry[i] == RATemp.entry[i+1]){ // values of entries with the same row and col should be added.
            RA.entry.back().val += RATemp.entry[i+1].val;
            i++;
//            if(rank==1) cout << RATemp.entry[i+1].val << endl;
        }
//        if(rank==1) cout << endl << "final: " << endl << RA.entry[RA.entry.size()-1].val << endl;
        // todo: pruning. don't hard code tol. it makes the matrix non-symmetric.
//        if( abs(RA.entry.back().val) < 1e-6)
//            RA.entry.pop_back();
//        if(rank==1) cout << "final: " << endl << RA.entry.back().val << endl;
    }

//    if(rank==1)
//        for(j=0; j<RA.entry.size(); j++)
//            cout << RA.entry[j].row << "\t" << RA.entry[j].col << "\t" << RA.entry[j].val << endl;

    // ************************************* RAPTemp - P local *************************************
    // Some local and remote elements of RAPTemp are computed here.

    prolongMatrix RAPTemp(comm); // RAPTemp is being used to remove duplicates while pushing back to RAP.
    unsigned int PMaxNnz;
    MPI_Allreduce(&P->nnz_l, &PMaxNnz, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);
//    MPI_Barrier(comm); printf("rank=%d, PMaxNnz=%d \n", rank, PMaxNnz); MPI_Barrier(comm);
    // todo: is this way better than using the previous Allreduce? reduce on processor 0, then broadcast to other processors.

    unsigned int* PnnzPerRow = (unsigned int*)malloc(sizeof(unsigned int)*PMaxNnz);
    fill(&PnnzPerRow[0], &PnnzPerRow[P->M], 0);
    for(i=0; i<P->nnz_l; i++){
        PnnzPerRow[P->entry[i].row]++;
    }

//    if(rank==1)
//        for(i=0; i<P->M; i++)
//            cout << PnnzPerRow[i] << endl;

    unsigned int* PnnzPerRowScan = (unsigned int*)malloc(sizeof(unsigned int)*(PMaxNnz+1));
    PnnzPerRowScan[0] = 0;
    for(i=0; i<P->M; i++){
        PnnzPerRowScan[i+1] = PnnzPerRowScan[i] + PnnzPerRow[i];
//        if(rank==2) printf("i=%lu, PnnzPerRow=%d, PnnzPerRowScan = %d\n", i, PnnzPerRow[i], PnnzPerRowScan[i]);
    }

    // find the start and end of each block of R.
    unsigned int* RABlockStart = (unsigned int*)malloc(sizeof(unsigned int)*(nprocs+1));
    fill(RABlockStart, &RABlockStart[nprocs+1], 0);
    procNum = lower_bound2(&P->split[0], &P->split[nprocs], RA.entry[0].col);
    nnzIter = 1;
    for(i=1; i<RA.entry.size(); i++){
        nnzIter++;
        if(RA.entry[i].col >= P->split[procNum+1]){
            RABlockStart[procNum+1] = nnzIter-1;
            procNum = lower_bound2(&P->split[0], &P->split[nprocs], RA.entry[i].col);
        }
//        if(rank==2) cout << "procNum = " << procNum << "\tcol = " << R->entry_remote[i].col << "\tnnzIter = " << nnzIter << endl;
    }
//    RABlockStart[rank+1] = RABlockStart[rank]; // there is not any nonzero of R_remote on the local processor.
    fill(&RABlockStart[procNum+1], &RABlockStart[nprocs+1], nnzIter);

//    if(rank==1){
//        cout << "RABlockStart: " << endl;
//        for(i=0; i<nprocs+1; i++)
//            cout << RABlockStart[i] << endl;}

    // todo: combine indicesP_Prolong and indicesP_ProlongRecv together.
    // find row-wise ordering for A and save it in indicesP
    unsigned long* indicesP_Prolong = (unsigned long*)malloc(sizeof(unsigned long)*P->nnz_l);
    for(unsigned long i=0; i<P->nnz_l; i++)
        indicesP_Prolong[i] = i;
    std::sort(indicesP_Prolong, &indicesP_Prolong[P->nnz_l], sort_indices2(&*P->entry.begin()));

    for(i=RABlockStart[rank]; i<RABlockStart[rank+1]; i++){
        for(j = PnnzPerRowScan[RA.entry[i].col - P->split[rank]]; j < PnnzPerRowScan[RA.entry[i].col - P->split[rank] + 1]; j++){
//            if(rank==0) cout << P->entry[indicesP[j]].row << "\t" << P->entry[indicesP[j]].col << "\t" << P->entry[indicesP[j]].val << endl;
            RAPTemp.entry.push_back(cooEntry(RA.entry[i].row + P->splitNew[rank],  // Ac.entry should have global indices at the end.
                                             P->entry[indicesP_Prolong[j]].col,
                                             RA.entry[i].val * P->entry[indicesP_Prolong[j]].val));
        }
    }

//    if(rank==1)
//        for(i=0; i<RAPTemp.entry.size(); i++)
//            cout << RAPTemp.entry[i].row << "\t" << RAPTemp.entry[i].col << "\t" << RAPTemp.entry[i].val << endl;

    free(indicesP_Prolong);

    // ************************************* RAPTemp - P remote *************************************

    unsigned long* indicesP_ProlongRecv = (unsigned long*)malloc(sizeof(unsigned long)*PMaxNnz);
    cooEntry* Precv = (cooEntry*)malloc(sizeof(cooEntry)*PMaxNnz);
    long PrecvM;

    for(int i = 1; i < nprocs; i++) {
        // send P to the right processor, receive P from the left processor. "left" decreases by one in each iteration. "right" increases by one.
        right = (rank + i) % nprocs;
        left = rank - i;
        if (left < 0)
            left += nprocs;

        // *************************** RATemp - A remote - sendrecv(size) ****************************

        // use sender rank for send and receive tags.
        MPI_Sendrecv(&P->nnz_l, 1, MPI_UNSIGNED_LONG, right, rank, &nnzRecv, 1, MPI_UNSIGNED_LONG, left, left, comm, &sendRecvStatus);
//        int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf,
//                         int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status)

//        if(rank==2) cout << "own P->nnz_l = " << P->nnz_l << "\tnnzRecv = " << nnzRecv << endl;

        // *************************** RATemp - A remote - sendrecv(P) ****************************

        // use sender rank for send and receive tags.
        MPI_Sendrecv(&P->entry[0], P->nnz_l, cooEntry::mpi_datatype(), right, rank, Precv, nnzRecv, cooEntry::mpi_datatype(), left, left, comm, &sendRecvStatus);

//        if(rank==1) for(int j=0; j<P->nnz_l; j++)
//                        printf("j=%d \t %lu \t %lu \t %f \n", j, P->entry[j].row, P->entry[j].col, P->entry[j].val);
//        if(rank==1) for(int j=0; j<nnzRecv; j++)
//                        printf("j=%d \t %lu \t %lu \t %f \n", j, Precv[j].row, Precv[j].col, Precv[j].val);

        // *************************** RATemp - A remote - multiplication ****************************

        PrecvM = P->split[left+1] - P->split[left];
        fill(&PnnzPerRow[0], &PnnzPerRow[PrecvM], 0);
        for(j=0; j<nnzRecv; j++)
            PnnzPerRow[Precv[j].row]++;

//        if(rank==1) cout << "PrecvM = " << PrecvM << endl;

//        if(rank==1)
//            for(j=0; j<PrecvM; j++)
//                cout << PnnzPerRow[i] << endl;

        PnnzPerRowScan[0] = 0;
        for(j=0; j<PrecvM; j++){
            PnnzPerRowScan[j+1] = PnnzPerRowScan[j] + PnnzPerRow[j];
//            if(rank==1) printf("j=%lu, PnnzPerRow=%d, PnnzPerRowScan = %d\n", j, PnnzPerRow[j], PnnzPerRowScan[j]);
        }

        // find row-wise ordering for Arecv and save it in indicesPRecv
        for(unsigned long i=0; i<nnzRecv; i++)
            indicesP_ProlongRecv[i] = i;
        std::sort(indicesP_ProlongRecv, &indicesP_ProlongRecv[nnzRecv], sort_indices2(Precv));

//        if(rank==1) cout << "block start = " << RBlockStart[left] << "\tend = " << RBlockStart[left+1] << "\tleft rank = " << left << "\t i = " << i << endl;
        for(j=RABlockStart[left]; j<RABlockStart[left+1]; j++){
//            if(rank==1) cout << "col = " << R->entry_remote[j].col << "\tcol-split = " << R->entry_remote[j].col - P->split[left] << "\tstart = " << AnnzPerRowScan[R->entry_remote[j].col - P->split[left]] << "\tend = " << AnnzPerRowScan[R->entry_remote[j].col - P->split[left] + 1] << endl;
            for(unsigned long k = PnnzPerRowScan[RA.entry[j].col - P->split[left]]; k < PnnzPerRowScan[RA.entry[j].col - P->split[left] + 1]; k++){
//                if(rank==0) cout << Precv[indicesP_ProlongRecv[k]].row << "\t" << Precv[indicesP_ProlongRecv[k]].col << "\t" << Precv[indicesP_ProlongRecv[k]].val << endl;
                RAPTemp.entry.push_back(cooEntry(RA.entry[j].row + P->splitNew[rank], // Ac.entry should have global indices at the end.
                                                Precv[indicesP_ProlongRecv[k]].col,
                                                RA.entry[j].val * Precv[indicesP_ProlongRecv[k]].val));
            }
        }

    } //for i

    free(indicesP_ProlongRecv);
    free(PnnzPerRow);
    free(PnnzPerRowScan);
    free(Precv);
    free(RABlockStart);

    std::sort(RAPTemp.entry.begin(), RAPTemp.entry.end());

//    if(rank==1)
//        for(j=0; j<RAPTemp.entry.size(); j++)
//            cout << RAPTemp.entry[j].row << "\t" << RAPTemp.entry[j].col << "\t" << RAPTemp.entry[j].val << endl;

    // remove duplicates.
    for(i=0; i<RAPTemp.entry.size(); i++){
        Ac->entry.push_back(RAPTemp.entry[i]);
        while(i<RAPTemp.entry.size()-1 && RAPTemp.entry[i] == RAPTemp.entry[i+1]){ // values of entries with the same row and col should be added.
            Ac->entry.back().val += RAPTemp.entry[i+1].val;
            i++;
        }
        // todo: pruning. don't hard code tol. it makes the matrix non-symmetric.
//        if( abs(Ac->entry.back().val) < 1e-6)
//            Ac->entry.pop_back();
    }

//    if(rank==1)
//        for(j=0; j<Ac->entry.size(); j++)
//            cout << Ac->entry[j].row << "\t" << Ac->entry[j].col << "\t" << Ac->entry[j].val << endl;

    Ac->nnz_l = Ac->entry.size();
    MPI_Allreduce(&Ac->nnz_l, &Ac->nnz_g, 1, MPI_UNSIGNED, MPI_SUM, comm);
    Ac->Mbig = P->Nbig;
    Ac->split = P->splitNew;
    Ac->M = Ac->split[rank+1] - Ac->split[rank];
    Ac->comm = P->comm;

//    printf("rank=%d \tA: Mbig=%u, nnz_g = %u, nnz_l = %u, M = %u \tAc: Mbig=%u, nnz_g = %u, nnz_l = %u, M = %u \n", rank, A->Mbig, A->nnz_g, A->nnz_l, A->M, Ac->Mbig, Ac->nnz_g, Ac->nnz_l, Ac->M);
//    MPI_Barrier(comm);
//    if(rank==1)
//        for(i=0; i<nprocs+1; i++)
//            cout << Ac->split[i] << endl;

    Ac->matrixSetup();

    return 0;
} // end of SaenaObject::coarsen


int SaenaObject::solveCoarsest(SaenaMatrix* A, std::vector<double>& u, std::vector<double>& rhs, int& maxIter, double& tol){
    // this is CG.
    // u is zero in the beginning. At the end, it is the solution.

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    long i, j;

    // res = A*u - rhs
    std::vector<double> res(A->M);
    residual(A, u, rhs, res);

    // make res = rhs - A*u
    for(i=0; i<res.size(); i++)
        res[i] = -res[i];

//    if(rank==1){
//        cout << "\nsolveCoarsest: initial res" << endl;
//        for(auto i:res)
//            cout << i << endl;}

    double dot;
    dotProduct(res, res, &dot, comm);
    double initialNorm = sqrt(dot);
//    if(rank==0) cout << "\nsolveCoarsest: initial norm(res) = " << initialNorm << endl;

    if (dot < tol*tol)
        maxIter = 0;

    std::vector<double> dir(A->M);
    dir = res;

    double factor, dot_prev;
    std::vector<double> matvecTemp(A->M);
    i = 0;
    while (i < maxIter) {
//        if(rank==0) cout << "starting iteration of CG = " << i << endl;
        // factor = sq_norm/ (dir' * A * dir)
        A->matvec(&*dir.begin(), &*matvecTemp.begin());
//        if(rank==1){
//            cout << "\nsolveCoarsest: A*dir" << endl;
//            for(auto i:matvecTemp)
//                cout << i << endl;}

//        factor = 0;
//        for(j = 0; j < A->M; j++)
//            factor += dir[j] * matvecTemp[j];
        dotProduct(dir, matvecTemp, &factor, comm);
        factor = dot / factor;
//        if(rank==1) cout << "\nsolveCoarsest: factor = " << factor << endl;

        for(j = 0; j < A->M; j++)
            u[j] += factor * dir[j];
//        if(rank==1){
//            cout << "\nsolveCoarsest: u" << endl;
//            for(auto i:u)
//                cout << i << endl;}

        // update residual
        for(j = 0; j < A->M; j++)
            res[j] -= factor * matvecTemp[j];
//        if(rank==1){
//            cout << "\nsolveCoarsest: update res" << endl;
//            for(auto i:res)
//                cout << i << endl;}

        dot_prev = dot;

        dotProduct(res, res, &dot, comm);
//        if(rank==0) cout << "absolute norm(res) = " << sqrt(dot) << "\t( r_i / r_0 ) = " << sqrt(dot)/initialNorm << "  \t( r_i / r_i-1 ) = " << sqrt(dot)/sqrt(dot_prev) << endl;
//        if(rank==0) cout << sqrt(dot)/initialNorm << endl;

        if (dot < tol*tol)
            break;

        factor = dot / dot_prev;
//        if(rank==1) cout << "\nsolveCoarsest: update factor = " << factor << endl;

        // update direction
        for(j = 0; j < A->M; j++)
            dir[j] = res[j] + factor * dir[j];
//        if(rank==1){
//            cout << "\nsolveCoarsest: update dir" << endl;
//            for(auto i:dir)
//                cout << i << endl;}

        i++;
    }
//    if(rank==0) cout << endl;

    return 0;
}


// int SaenaObject::solveCoarsest(SaenaMatrix* A, std::vector<double>& x, std::vector<double>& b, int& max_iter, double& tol, MPI_Comm comm){
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
//    if(rank==1) cout << normb << endl;

//    Vector r = b - A*x;
    std::vector<double> matvecTemp(A->M);
    A->matvec(&*x.begin(), &*matvecTemp.begin(), comm);
//    if(rank==1)
//        for(i=0; i<matvecTemp.size(); i++)
//            cout << matvecTemp[i] << endl;

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


int SaenaObject::vcycle(Grid* grid, std::vector<double>& u, std::vector<double>& rhs, MPI_Comm comm){
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    long i;

//    printf("rank = %d, current level = %d here!!!!!!!!!! \n", rank, grid->currentLevel);

//    % handle for the coarsest level
//    if ( isempty( grid.Coarse ) )
//        u = grid.K \ rhs;
//    return;
//    end

    // todo: don't hard-code. receive them by another way.
    int maxIter = 20;
    double tol = 1e-12;
    if(grid->currentLevel == maxLevel){
//        if(rank==0) cout << "current level = " << grid->currentLevel << ", Solving the coarsest level!" << endl;
        solveCoarsest(grid->A, u, rhs, maxIter, tol);
        return 0;
    }

//    if(rank==0) cout << "******************************************************" << endl;

    double dot;
    std::vector<double> r(grid->A->M);
    residual(grid->A, u, rhs, r);
    dotProduct(r, r, &dot, comm);
//    if(rank==0) cout << "current level = " << grid->currentLevel << ", vcycle start      = " << sqrt(dot) << endl;

    // **************************************** 1. pre-smooth ****************************************
    // u = grid.smooth ( v1, rhs, u );

    for(i=0; i<preSmooth; i++)
        grid->A->jacobi(u, rhs);

//    if(rank==1) cout << "\n1. pre-smooth: u, currentLevel = " << grid->currentLevel << endl;
//    if(rank==1)
//        for(auto i:u)
//            cout << i << endl;

    // **************************************** 2. compute residual ****************************************
    // res = grid.residual( rhs, u );

    residual(grid->A, u, rhs, r);

//    if(rank==1) cout << "\n2. compute residual: res, currentLevel = " << grid->currentLevel << endl;
//    if(rank==1)
//        for(auto i:r)
//            cout << i << endl;

//    dotProduct(r, r, &dot, comm);
//    if(rank==0) cout << "current level = " << grid->currentLevel << ", after pre-smooth  = " << sqrt(dot) << endl;

    // **************************************** 3. restrict ****************************************
    // res_coarse = grid.R * res;

    std::vector<double> rCoarse(grid->Ac.M);
    grid->R.matvec(&*r.begin(), &*rCoarse.begin());

//    if(rank==0){
//        cout << "\n3. restriction: rCoarse = R*res, currentLevel = " << grid->currentLevel << endl;
//        for(auto i:rCoarse)
//            cout << i << endl;}

    // **************************************** 4. recurse ****************************************
    // u_corr_coarse = grid.Coarse.vcycle(v1, v2, res_coarse, zeros(size(res_coarse)));
    // function u = vcycle(grid, v1, v2, rhs, u)

//    if(rank==1) cout << "\n\n\nenter recursive vcycle = " << grid->currentLevel << endl;
    std::vector<double> uCorrCoarse(grid->Ac.M);
    uCorrCoarse.assign(grid->Ac.M, 0);
    vcycle(grid->coarseGrid, uCorrCoarse, rCoarse, comm);

//    if(rank==1) cout << "\n4. uCorrCoarse, currentLevel = " << grid->currentLevel << endl;
//    if(rank==1)
//        for(auto i:uCorrCoarse)
//            cout << i << endl;

    // **************************************** 5 & 6. prolong and correct ****************************************
    // u = u - grid.P * u_corr_coarse;

    std::vector<double> uCorr(grid->A->M);
    grid->P.matvec(&*uCorrCoarse.begin(), &*uCorr.begin());
//    if(rank==1) cout << "\n5. prolongation: uCorr = P*uCorrCoarse , currentLevel = " << grid->currentLevel << endl;
//    if(rank==1)
//        for(i=0; i<u.size(); i++)
//            cout << uCorr[i] << endl;

    for(i=0; i<u.size(); i++)
        u[i] -= uCorr[i];

//    if(rank==1) cout << "\n6. correct: u -= uCorr, currentLevel = " << grid->currentLevel << endl;
//    if(rank==1)
//        for(i=0; i<u.size(); i++)
//            cout << u[i] << endl;

    // todo: delete this part after debugging.
//    residual(grid->A, u, rhs, r);
//    dotProduct(r, r, &dot, comm);
//    if(rank==0) cout << "current level = " << grid->currentLevel << ", after correction  = " << sqrt(dot) << endl;

    // **************************************** 7. post-smooth ****************************************
    // u = grid.smooth ( v2, rhs, u );

    for(i=0; i<postSmooth; i++)
        grid->A->jacobi(u, rhs);

//    if(rank==1) cout << "\n7. post-smooth: u, currentLevel = " << grid->currentLevel << endl;
//    if(rank==1)
//        for(auto i:u)
//            cout << i << endl;

    // todo: delete this part after debugging.
//    residual(grid->A, u, rhs, r);
//    dotProduct(r, r, &dot, comm);
//    if(rank==0) cout << "current level = " << grid->currentLevel << ", after post-smooth = " << sqrt(dot) << endl;

    return 0;
}


int SaenaObject::solve(std::vector<double>& u, std::vector<double>& rhs){
    MPI_Comm comm = MPI_COMM_WORLD; // todo: fix this
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    long i;

//    double temp;
//    dot(rhs, rhs, &temp, comm);
//    if(rank==1) cout << "norm(rhs) = " << sqrt(temp) << endl;

    std::vector<double> r(grids[0].A->M);
    residual(grids[0].A, u, rhs, r);
    double initial_dot;
    dotProduct(r, r, &initial_dot, comm);
//    if(rank==0) cout << "******************************************************" << endl;
//    if(rank==0) cout << "\ninitial_norm = " << sqrt(initial_dot) << endl << endl;

    double dot = initial_dot;
    for(i=0; i<vcycle_num; i++){
        vcycle(&grids[0], u, rhs, comm);
        residual(grids[0].A, u, rhs, r);
        dotProduct(r, r, &dot, comm);
//        if(rank==0) printf("vcycle iteration = %ld, residual = %f \n\n", i, sqrt(dot));
        if(dot/initial_dot < relative_tolerance)
            break;
    }

    // set number of iterations that took to find the solution
    // only do the following if the end of the previous for loop was reached.
    if(i == vcycle_num)
        i--;

    if(rank==0) cout << "******************************************************" << endl;
    if(rank==0) printf("\nfinal:\niter = %ld, residual = %f \n\n", i, sqrt(dot));
    if(rank==0) cout << "******************************************************" << endl;

    return 0;
}


int SaenaObject::residual(SaenaMatrix* A, std::vector<double>& u, std::vector<double>& rhs, std::vector<double>& res){
    //    Vector res = A*u - rhs;

//    int nprocs, rank;
//    MPI_Comm_size(comm, &nprocs);
//    MPI_Comm_rank(comm, &rank);

    std::vector<double> matvecTemp(A->M);
    A->matvec(&*u.begin(), &*matvecTemp.begin());
//    if(rank==1)
//        for(long i=0; i<matvecTemp.size(); i++)
//            cout << matvecTemp[i] << endl;

    for(long i=0; i<A->M; i++)
        res[i] = matvecTemp[i] - rhs[i];
//    if(rank==1)
//        for(long i=0; i<res.size(); i++)
//            cout << res[i] << endl;

    return 0;
}


int SaenaObject::writeMatrixToFileA(SaenaMatrix* A, std::string name){
    // Create txt files with name Ac0.txt for processor 0, Ac1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat Ac0.txt Ac1.txt > Ac.txt
    // row and column indices of txt files should start from 1, not 0.

    // todo: check global or local index and see if A->split[rank] is required for rows.

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    ofstream outFileTxt;
    std::string outFileNameTxt = "/home/abaris/Dropbox/Projects/Saena/build/writeMatrix/";
    outFileNameTxt += name;
    outFileNameTxt += std::to_string(rank);
    outFileNameTxt += ".txt";
    outFileTxt.open(outFileNameTxt);

    if(rank==0)
        outFileTxt << A->Mbig << "\t" << A->Mbig << "\t" << A->nnz_g << endl;
    for (long i = 0; i < A->nnz_l; i++) {
//        cout       << A->entry[i].row + 1 << "\t" << A->entry[i].col + 1 << "\t" << A->entry[i].val << endl;
        outFileTxt << A->entry[i].row + 1 << "\t" << A->entry[i].col + 1 << "\t" << A->entry[i].val << endl;
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
        if (rank == 0) cout << "Unable to open the matrix file!" << endl;
        MPI_Finalize();
    }

//    std::vector<unsigned int> nnzScan(nprocs);
//    mpierror = MPI_Allgather(&A->nnz_l, 1, MPI_UNSIGNED, &nnzScan[1], 1, MPI_UNSIGNED, comm);
//    if (mpierror) {
//        if (rank == 0) cout << "Unable to gather!" << endl;
//        MPI_Finalize();
//    }

//    nnzScan[0] = 0;
//    for(unsigned long i=0; i<nprocs; i++){
//        nnzScan[i+1] = nnzScan[i] + nnzScan[i+1];
//        if(rank==1) cout << nnzScan[i] << endl;
//    }
//    offset = nnzScan[rank];
//    MPI_File_write_at_all(fh, rank, &nnzScan[rank])
//    unsigned int a = 1;
//    double b = 3;
//    MPI_File_write_at(fh, rank, &rank, 1, MPI_INT, &status);
//    MPI_File_write_at_all(fh, offset, &A->entry[0], A->nnz_l, cooEntry::mpi_datatype(), &status);
//    MPI_File_write_at_all(fh, &A->entry[0], A->nnz_l, cooEntry::mpi_datatype(), &status);
//    if (mpierror) {
//        if (rank == 0) cout << "Unable to write to the matrix file!" << endl;
//        MPI_Finalize();
//    }

//    int count;
//    MPI_Get_count(&status, MPI_UNSIGNED_LONG, &count);
    //printf("process %d read %d lines of triples\n", rank, count);
    MPI_File_close(&fh);
*/

    return 0;
}


int SaenaObject::writeMatrixToFileP(prolongMatrix* P, std::string name) {
    // Create txt files with name P0.txt for processor 0, P1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat P0.txt P1.txt > P.txt
    // row and column indices of txt files should start from 1, not 0.

    MPI_Comm comm = P->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    ofstream outFileTxt;
    std::string outFileNameTxt = "/home/abaris/Dropbox/Projects/Saena/build/writeMatrix/";
    outFileNameTxt += name;
    outFileNameTxt += std::to_string(rank);
    outFileNameTxt += ".txt";
    outFileTxt.open(outFileNameTxt);

    if (rank == 0)
        outFileTxt << P->Mbig << "\t" << P->Mbig << "\t" << P->nnz_g << endl;
    for (long i = 0; i < P->nnz_l; i++) {
//        cout       << P->entry[i].row + 1 + P->split[rank] << "\t" << P->entry[i].col + 1 << "\t" << P->entry[i].val << endl;
        outFileTxt << P->entry[i].row + 1 + P->split[rank] << "\t" << P->entry[i].col + 1 << "\t" << P->entry[i].val << endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

    return 0;
}


int SaenaObject::writeMatrixToFileR(restrictMatrix* R, std::string name) {
    // Create txt files with name R0.txt for processor 0, R1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat R0.txt R1.txt > R.txt
    // row and column indices of txt files should start from 1, not 0.

    MPI_Comm comm = R->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    ofstream outFileTxt;
    std::string outFileNameTxt = "/home/abaris/Dropbox/Projects/Saena/build/writeMatrix/";
    outFileNameTxt += name;
    outFileNameTxt += std::to_string(rank);
    outFileNameTxt += ".txt";
    outFileTxt.open(outFileNameTxt);

    if (rank == 0)
        outFileTxt << R->Mbig << "\t" << R->Mbig << "\t" << R->nnz_g << endl;
    for (long i = 0; i < R->nnz_l; i++) {
//        cout       << R->entry[i].row + 1 + R->splitNew[rank] << "\t" << R->entry[i].col + 1 << "\t" << R->entry[i].val << endl;
        outFileTxt << R->entry[i].row + 1 +  R->splitNew[rank] << "\t" << R->entry[i].col + 1 << "\t" << R->entry[i].val << endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

    return 0;
}

//template <class T>
//int SaenaObject::writeVectorToFile(std::vector<T>& v, unsigned long vSize, std::string name, MPI_Comm comm) {
int SaenaObject::writeVectorToFiled(std::vector<double>& v, unsigned long vSize, std::string name, MPI_Comm comm) {

    // Create txt files with name name0.txt for processor 0, name1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat name0.txt name1.txt > V.txt

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    ofstream outFileTxt;
    std::string outFileNameTxt = "/home/abaris/Dropbox/Projects/Saena/build/writeMatrix/";
    outFileNameTxt += name;
    outFileNameTxt += std::to_string(rank);
    outFileNameTxt += ".txt";
    outFileTxt.open(outFileNameTxt);

    if (rank == 0)
        outFileTxt << vSize << endl;
    for (long i = 0; i < v.size(); i++) {
//        cout       << R->entry[i].row + 1 + R->splitNew[rank] << "\t" << R->entry[i].col + 1 << "\t" << R->entry[i].val << endl;
        outFileTxt << v[i] << endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

    return 0;
}


int SaenaObject::writeVectorToFileul(std::vector<unsigned long>& v, unsigned long vSize, std::string name, MPI_Comm comm) {

    // Create txt files with name name0.txt for processor 0, name1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat name0.txt name1.txt > V.txt

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    ofstream outFileTxt;
    std::string outFileNameTxt = "/home/abaris/Dropbox/Projects/Saena/build/writeMatrix/";
    outFileNameTxt += name;
    outFileNameTxt += std::to_string(rank);
    outFileNameTxt += ".txt";
    outFileTxt.open(outFileNameTxt);

    if (rank == 0)
        outFileTxt << vSize << endl;
    for (long i = 0; i < v.size(); i++) {
//        cout       << R->entry[i].row + 1 + R->splitNew[rank] << "\t" << R->entry[i].col + 1 << "\t" << R->entry[i].val << endl;
        outFileTxt << v[i] << endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

    return 0;
}


int SaenaObject::changeAggregation(SaenaMatrix* A, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew){

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
        if (rank==0) cout << "Unable to open the vector file!" << endl;
        MPI_Finalize();
        return -1;
    }

    // vector should have the following format: first line shows the value in row 0, second line shows the value in row 1
    offset = A->split[rank] * 8; // value(long=8)
    MPI_File_read_at(fh, offset, &*aggregate.begin(), A->M, MPI_UNSIGNED_LONG, &status);
    MPI_File_close(&fh);

//    for(auto i:aggregate)
//        cout << i << endl;

    MPI_Status status2;
    MPI_File fh2;
    MPI_Offset offset2;

    std::string aggName2 = "/home/abaris/Dropbox/Projects/Saena/build/juliaAggArray.bin";
    int mpiopen2 = MPI_File_open(comm, aggName2.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh2);
    if(mpiopen2){
        if (rank==0) cout << "Unable to open the vector file!" << endl;
        MPI_Finalize();
        return -1;
    }

    std::vector<unsigned long> aggArray(A->M);
    // vector should have the following format: first line shows the value in row 0, second line shows the value in row 1
    offset2 = A->split[rank] * 8; // value(long=8)
    MPI_File_read_at(fh2, offset2, &*aggArray.begin(), A->M, MPI_UNSIGNED_LONG, &status);
    MPI_File_close(&fh2);

//    for(auto i:aggArray)
//        cout << i << endl;

    unsigned long newSize = 0;
    for(auto i:aggArray)
        if(i == 1)
            newSize++;

//    if(rank==0)
//        cout << "newSize = " << newSize << endl;

    // set splitNew
    fill(splitNew.begin(), splitNew.end(), 0);
    splitNew[rank] = newSize;

    unsigned long* splitNewTemp = (unsigned long*)malloc(sizeof(unsigned long)*nprocs);
    MPI_Allreduce(&splitNew[0], splitNewTemp, nprocs, MPI_UNSIGNED_LONG, MPI_SUM, comm);

    // do scan on splitNew
    splitNew[0] = 0;
    for(i=1; i<nprocs+1; i++)
        splitNew[i] = splitNew[i-1] + splitNewTemp[i-1];

//    for(i=0; i<nprocs+1; i++)
//        cout << splitNew[i] << endl;

    free(splitNewTemp);

    return 0;
}
