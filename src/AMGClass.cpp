//
// Created by abaris on 3/14/17.
//

#include <cstdio>
#include <algorithm>
#include <mpi.h>
#include <usort/parUtils.h>
#include <set>
#include <mpich/mpi.h>
#include "AMGClass.h"
#include "auxFunctions.h"
//#include <random>
//#include "prolongmatrix.h"
//#include "restrictmatrix.h"

int randomVector(unsigned long* V, unsigned long size){
//    int rank;
//    MPI_Comm_rank(comm, &rank);

    //Type of random number distribution
    std::uniform_int_distribution<unsigned long> dist(1, size); //(min, max)

    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng;

    //Initialize with non-deterministic seeds
    rng.seed(std::random_device{}());

    for (unsigned long i=0; i<size; i++)
        V[i] = dist(rng);

//    if(rank==0){
//        V[6] = 100;
//        V[11] = 100;
//    }

    return 0;
}


AMGClass::AMGClass(int l, int vcycle_n, double relT, string sm, int preSm, int postSm, float connStr, float ta){
    levels = l;
    vcycle_num = vcycle_n;
    relTol  = relT;
    smoother = sm;
    preSmooth = preSm;
    postSmooth = postSm;
    connStrength = connStr;
    tau = ta;
} //AMGClass


AMGClass::~AMGClass(){}


int AMGClass::AMGSetup(COOMatrix* A, bool doSparsify, MPI_Comm comm){

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    unsigned long i;

    // pointer for shrinking number of processors
    unsigned long* initialNumberOfRows;
    initialNumberOfRows = &A->split[0];

    std::vector<unsigned long> aggregate(A->M);
    unsigned long* aggregate_p = &(*aggregate.begin());
    unsigned long* splitNew = (unsigned long*)malloc(sizeof(unsigned long)*(nprocs+1));

    // todo: think about a parameter for making the aggregation less or more aggressive.
    findAggregation(A, aggregate_p, splitNew, comm);
//    MPI_Barrier(comm);
//    if(rank==0)
//        for(long i=0; i<A->M; i++)
//            cout << i << "\t" << aggregate[i] << endl;
//    MPI_Barrier(comm);

/*
    std::vector<long> aggregateSorted(A->M);
//    long* aggregateSorted_p = &(*aggregateSorted.begin());
    par::sampleSort(aggregate, aggregateSorted, comm);
    if(rank==0) cout << "\nafter:" << endl;
    if(rank==0)
        for(long i=0; i<A->M; i++)
            cout << i << "\t" << aggregate[i] << "\t" << aggregateSorted[i] << endl;
    MPI_Barrier(comm);
    if(rank==1)
        for(long i=0; i<A->M; i++)
            cout << i << "\t" << aggregate[i] << "\t" << aggregateSorted[i] << endl;
    MPI_Barrier(comm);
*/


//    par::sampleSort(aggregate, comm);

    /*
    if(rank==0) cout << "\nafter:" << endl;
    if(rank==0)
        for(long i=0; i<A->M; i++)
            cout << i << "\t" << aggregate[i] << endl;
    MPI_Barrier(comm);
    if(rank==1)
        for(long i=0; i<A->M; i++)
            cout << i << "\t" << aggregate[i] << endl;
    MPI_Barrier(comm);
*/

    // todo: shrink only if it is required to go to the next multigrid level.
    // todo: move the shrinking to after aggregation.
    // ******************** shrink cpus ********************

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

    prolongMatrix P;
    createProlongation(A, aggregate_p, splitNew, &P, comm);
    restrictMatrix R(&P, initialNumberOfRows, comm);

    prolongMatrix Ac; // A_coarse = R*A*P
    coarsen(A, &P, &R, &Ac, comm);

    free(splitNew);
    return 0;
}


int AMGClass::findAggregation(COOMatrix* A, unsigned long* aggregate, unsigned long* splitNew, MPI_Comm comm){
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    StrengthMatrix S;
    createStrengthMatrix(A, &S, comm);
//    S.print(0);

//    unsigned long aggSize = 0;
    Aggregation(&S, aggregate, splitNew, comm);
//    updateAggregation(aggregate, &aggSize);

//    if(rank==0)
//        for(long i=0; i<S.M; i++)
//            cout << i << "\t" << aggregate[i] << endl;

    return 0;
} // end of AMGClass::findAggregation


int AMGClass::createStrengthMatrix(COOMatrix* A, StrengthMatrix* S, MPI_Comm comm){
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
        if( A->entryP[i].row != A->entryP[i].col ){
            if(maxPerRow[A->entryP[i].row - A->split[rank]] == 0) // use split to convert the index from global to local.
                maxPerRow[A->entryP[i].row - A->split[rank]] = -A->entryP[i].val;
            else if(maxPerRow[A->entryP[i].row - A->split[rank]] < -A->entryP[i].val)
                maxPerRow[A->entryP[i].row - A->split[rank]] = -A->entryP[i].val;
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
        if(A->entryP[i].row == A->entryP[i].col) {
            Si.push_back(A->entryP[i].row);
            Sj.push_back(A->entryP[i].col);
            Sval.push_back(1);
        }
        else if(maxPerRow[A->entryP[i].row - A->split[rank]] != 0) {
//            if ( -A->values[i] / (maxPerRow[A->row[i] - A->split[rank]] ) > connStrength) {
            Si.push_back(A->entryP[i].row);
            Sj.push_back(A->entryP[i].col);
            Sval.push_back(  -A->entryP[i].val / (maxPerRow[A->entryP[i].row - A->split[rank]])  );
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
        if( A->entryP[i].row != A->entryP[i].col ){
            if(local_maxPerCol[A->entryP[i].col] == 0)
                local_maxPerCol[A->entryP[i].col] = -A->entryP[i].val;
            else if(local_maxPerCol[A->entryP[i].col] < -A->entryP[i].val)
                local_maxPerCol[A->entryP[i].col] = -A->entryP[i].val;
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
        if(A->entryP[i].row == A->entryP[i].col) {
            STi.push_back(A->entryP[i].row - A->split[rank]);
            STj.push_back(A->entryP[i].col - A->split[rank]);
            STval.push_back(1);
        }
        else{
//            if ( (-A->values[i] / maxPerCol[A->col[i]]) > connStrength) {
            STi.push_back(A->entryP[i].row - A->split[rank]);
            STj.push_back(A->entryP[i].col - A->split[rank]);
            STval.push_back( -A->entryP[i].val / maxPerCol[A->entryP[i].col] );
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
    S->StrengthMatrixSet(&(*(Si2.begin())), &(*(Sj2.begin())), &(*(Sval2.begin())), A->M, A->Mbig, Si2.size(), &(*(A->split.begin())));
    return 0;
} // end of AMGClass::createStrengthMatrix


// Using MIS(2) from the following paper by Luke Olson:
// EXPOSING FINE-GRAINED PARALLELISM IN ALGEBRAIC MULTIGRID METHODS
int AMGClass::Aggregation(StrengthMatrix* S, unsigned long* aggregate, unsigned long* splitNew, MPI_Comm comm) {

    // the first two bits of aggregate are being used for aggStatus: 1 for 01 not assigned, 0 for 00 assigned, 2 for 10 root
    // bits 0 up to 61 are for storing aggregate values.
    // the max value for aggregate is 2^63 - 1

    // variables used in this function:
    // weight[i]: the two most left bits store the status of node i, the other 62 bits store weight assigned to that node.
    //            weight is first generated randomly by randomVector function and saved in initialWeight. During the
    //            aggregation process, it becomes the weight of the node's aggregate.
    //

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    unsigned long i, j;
    unsigned long size = S->M;

    std::vector<unsigned long> aggArray;
    std::vector<unsigned long> aggregate2(size);
    std::vector<unsigned long> aggStatus2(size); // 1 for 01 not assigned, 0 for 00 assigned, 2 for 10 root
    std::vector<unsigned long> weight(size);
    std::vector<unsigned long> weight2(size);
    std::vector<unsigned long> initialWeight(size);
    unsigned long* initialWeight_p = &(*initialWeight.begin());

    randomVector(initialWeight_p, size);

//    if(rank==0){
//        cout << endl << "after initialization!" << endl;
//        for (i = 0; i < size; ++i)
//            cout << i << "\tinitialWeight = " << initialWeight[i] << endl;
//    }


    const int aggOffset = 62;
    const unsigned long weightMax = (1UL<<aggOffset) - 1;
    const unsigned long UNDECIDED = 1UL<<aggOffset;
    const unsigned long ROOT = 1UL<<(aggOffset+1);
    const unsigned long UNDECIDED_OR_ROOT = 3UL<<aggOffset;
    unsigned long weightTemp, aggregateTemp, aggStatusTemp;
    bool* oneDistanceRoot = (bool*)malloc(sizeof(bool)*size);
//    fill(&oneDistanceRoot[0], &oneDistanceRoot[size], 0);
    bool continueAggLocal;
    bool continueAgg = true;
    unsigned long iter;
    int whileiter = 0;

    MPI_Request *requests = new MPI_Request[S->numSendProc + S->numRecvProc];
    MPI_Status *statuses  = new MPI_Status[S->numSendProc + S->numRecvProc];

    // initialization -> this part is merged to the first "for" loop in the following "while".
    for(i=0; i<size; i++) {
        aggregate[i] = i + S->split[rank];
        weight[i] = (1UL<<aggOffset | (initialWeight[i]) ); // status of each node is initialized to 1 and its weight to initialWeight.
        aggStatus2[i] = 1;
    }

    while(continueAgg) {
        // ******************************* first round of max computation *******************************
        // first "compute max" is local. The second one is both local and remote.

        // todo: check the aggregation for deadlock: two neighbors for each other to have a status other than 1, while both have status 1.
        //distance-1 aggregate
        iter = 0;
        for (i = 0; i < size; ++i) {
            if(weight[i]&UNDECIDED) {
                oneDistanceRoot[i] = 0; // initialization
                aggregateTemp = aggregate[i];
                weightTemp = weight[i];
//                aggStatusTemp = 1UL; // this will be used for aggStatus2, and aggStatus2 will be used for the remote part.
                for (j = 0; j < S->nnzPerRow_local[i]; ++j, ++iter) {
                    if( weight[S->col_local[S->indicesP_local[iter]] - S->split[rank]]&UNDECIDED_OR_ROOT ) {
                        if (initialWeight[S->col_local[S->indicesP_local[iter]] - S->split[rank]] > (weightTemp & weightMax)) {
                            aggregateTemp = S->col_local[S->indicesP_local[iter]];
                            weightTemp = initialWeight[S->col_local[S->indicesP_local[iter]] - S->split[rank]];
                            // this will be used for aggStatus2, and aggStatus2 will be used for the remote part.
//                            aggStatusTemp = (weight[S->col_local[S->indicesP_local[iter]] - S->split[rank]] >> aggOffset);
                            oneDistanceRoot[i] = 1;
                        }
                    }
                }
                weight2[i]    = weightTemp;
                aggregate2[i] = aggregateTemp;
//                if(rank==1) cout << i+S->split[rank] << "," << aggregate2[i] << "\t\t";
            }else
                iter += S->nnzPerRow_local[i];
        }

        for (i = 0; i < size; ++i) {
            if(S->nnzPerRow_local[i] != 0) {
                aggStatusTemp = (weight[i]>>aggOffset);
                weight[i] = (aggStatusTemp<<aggOffset | (weight2[i]&weightMax) );
                aggregate[i] = aggregate2[i];
//                if(rank==1) cout << i+S->split[rank] << "\t" << aggregate[i] << "\t" << aggregate2[i] << endl;
            }
        }

        //    if(rank==0){
        //        cout << endl << "after first max computation!" << endl;
        //        for (i = 0; i < size; ++i)
        //            cout << i << "\tweight = " << weight[i] << "\tindex = " << aggregate[i] << endl;
        //    }

        // ******************************* exchange remote max values for the second round of max computation *******************************

        // vSend[2*i]: the first right 62 bits of vSend is maxPerCol for remote elements that should be sent to other processes.
        // the first left 2 bits of vSend are aggStatus.
        // vSend[2*i+1]: the first right 62 bits of vSend is aggregate for remote elements that should be sent to other processes.
        // the first left 2 bits are oneDistanceRoot.

        // the following shows how the data are being stored in vecValues:
//        iter = 0;
//        if(rank==1)
//            for (i = 0; i < S->col_remote_size; ++i)
//                for (j = 0; j < S->nnz_col_remote[i]; ++j, ++iter){
//                    cout << "row:" << S->row_remote[iter]+S->split[rank] << "\tneighbor(col) = " << S->col_remote2[iter]
//                         << "\tweight of neighbor = "          << (S->vecValues[2*S->col_remote[iter]]&weightMax)
//                         << "\t\tstatus of neighbor = "        << (S->vecValues[2*S->col_remote[iter]]>>aggOffset)
//                         << "\toneDistanceRoot of neighbor = " << (S->vecValues[2*S->col_remote[iter]+1]&weightMax)
//                         << "\tstatus of agg = "               << (S->vecValues[2*S->col_remote[iter]+1]>>aggOffset)
//                         << endl;
//                }

//        if(rank==1) cout << endl << endl;
        for (i = 0; i < S->vIndexSize; i++){
//            S->vSend[i] = weight[(S->vIndex[i])];
            S->vSend[2*i] = weight[S->vIndex[i]];
            aggStatusTemp = (unsigned long)oneDistanceRoot[S->vIndex[i]]; // this is oneDistanceRoot of the neighbor's aggregate.
            S->vSend[2*i+1] = ( (aggStatusTemp<<aggOffset) | (aggregate[S->vIndex[i]]&weightMax) );
//            if(rank==1) cout << "vsend: " << S->vIndex[i]+S->split[rank] << "\tw = " << (S->vSend[2*i]&weightMax) << "\tst = " << (S->vSend[2*i]>>aggOffset) << "\tagg = " << (S->vSend[2*i+1]&weightMax) << "\t oneDis = " << (S->vSend[2*i+1]>>aggOffset) << endl;
        }

        for (i = 0; i < S->numRecvProc; i++)
            MPI_Irecv(&S->vecValues[S->rdispls[S->recvProcRank[i]]], S->recvProcCount[i], MPI_UNSIGNED_LONG,
                      S->recvProcRank[i], 1, comm, &(requests[i]));

        for (i = 0; i < S->numSendProc; i++)
            MPI_Isend(&S->vSend[S->vdispls[S->sendProcRank[i]]], S->sendProcCount[i], MPI_UNSIGNED_LONG,
                      S->sendProcRank[i], 1, comm, &(requests[S->numRecvProc + i]));

        // ******************************* second round of max computation *******************************

        // local part - distance-2 aggregate
        iter = 0;
        for (i = 0; i < size; ++i) {
            if(weight[i]&UNDECIDED) {
//                oneDistanceRoot[i] = false;
                aggregateTemp = aggregate[i];
                weightTemp    = weight[i];
//                aggStatusTemp = 1UL; // this will be used for aggStatus2, and aggStatus2 will be used for the remote part.
                for (j = 0; j < S->nnzPerRow_local[i]; ++j, ++iter) {
                    if((weight[S->col_local[S->indicesP_local[iter]] - S->split[rank]]>>aggOffset) <= 1){ // todo: merge these two if statements.
                        if(oneDistanceRoot[i]==0 &&
                           oneDistanceRoot[S->col_local[S->indicesP_local[iter]] - S->split[rank]]==1 &&
                           ( (weight[S->col_local[S->indicesP_local[iter]] - S->split[rank]]&weightMax) > (weightTemp&weightMax) )){
                            aggregateTemp = aggregate[S->col_local[S->indicesP_local[iter]] - S->split[rank]];
                            weightTemp    = weight[S->col_local[S->indicesP_local[iter]] - S->split[rank]];
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
            if(S->nnzPerRow_local[i] != 0) {
                aggregate[i] = aggregate2[i];
                aggStatusTemp = (weight[i]>>aggOffset);
                if (aggregate[i] < S->split[rank] || aggregate[i] >= S->split[rank+1]) // this is distance-2 to a remote root.
                    aggStatusTemp = 0;
                weight[i] = (aggStatusTemp<<aggOffset | (weight2[i]&weightMax) );
            }
        }

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
//                         << "\t\tstatus of neighbor = "        << (S->vecValues[2*S->col_remote[iter]]>>aggOffset)
//                         << "\toneDistanceRoot of neighbor = " << (S->vecValues[2*S->col_remote[iter]+1]&weightMax)
//                         << "\tstatus of agg = "               << (S->vecValues[2*S->col_remote[iter]+1]>>aggOffset)
//                         << endl;
//                }
//        MPI_Barrier(comm);

        // remote part
        // store the max of rows of remote elements in weight2 and aggregate2.
        iter = 0;
        for (i = 0; i < S->col_remote_size; ++i) {
            for (j = 0; j < S->nnz_col_remote[i]; ++j, ++iter) {
                if(weight[S->row_remote[iter]]&UNDECIDED) {
//                    if(rank==0) cout << "remote: row = " << S->row_remote[iter]+S->split[rank] << " \tcol = " << S->col_remote2[iter] << "  \tinitial weight = " << initialWeight[S->row_remote[iter]] << ",\tcurrent updated aggregate = " << aggregate2[S->row_remote[iter]] << "\t\tremote aggregate = " << (S->vecValues[2*S->col_remote[iter]+1]&weightMax) << "\t\tremote weight = " << (weight2[S->row_remote[iter]]&weightMax) << "\tremote oneDistance = " << (S->vecValues[2*S->col_remote[iter]+1]>>aggOffset) << "\t\taggStat2 = " << aggStatus2[S->row_remote[iter]] << endl;

                    //distance-1 aggregate
                    if (S->vecValues[2 * S->col_remote[iter]] &UNDECIDED_OR_ROOT) {
                        if ((S->vecValues[2 * S->col_remote[iter]] & weightMax) > (weight2[S->row_remote[iter]] & weightMax)) {
//                            if(rank==1) cout << "first  before\t" << "row = " << S->row_remote[iter]+S->split[rank] << "  \tinitial weight = " << initialWeight[S->row_remote[iter]] << ",\taggregate = " << aggregate2[S->row_remote[iter]] << "\tremote weight = " << (weight2[S->row_remote[iter]]&weightMax) << "\tremote status = " << (S->vecValues[2*S->col_remote[iter]+1]>>aggOffset) << "\taggStat2 = " << aggStatus2[S->row_remote[iter]] << endl;
                            weight2[S->row_remote[iter]] = (S->vecValues[2 * S->col_remote[iter]] & weightMax);
                            aggregate2[S->row_remote[iter]] = (S->vecValues[2 * S->col_remote[iter] + 1] & weightMax);
                            aggStatus2[S->row_remote[iter]] = (S->vecValues[2 * S->col_remote[iter]] >> aggOffset);
                            oneDistanceRoot[S->row_remote[iter]] = 1;
//                            if(rank==1) cout << "first  after \t" << "row = " << S->row_remote[iter]+S->split[rank] << "  \tinitial weight = " << initialWeight[S->row_remote[iter]] << ",\taggregate = " << aggregate2[S->row_remote[iter]] << "\t\tremote weight = " << (weight2[S->row_remote[iter]]&weightMax) << "\tremote status = " << (S->vecValues[2*S->col_remote[iter]+1]>>aggOffset) << "\taggStat2 = " << aggStatus2[S->row_remote[iter]] << endl;
                        }

                        //distance-2 aggregate
                    } else {
                        if (oneDistanceRoot[S->row_remote[iter]] == 0 &&
                            (S->vecValues[2 * S->col_remote[iter] + 1] &UNDECIDED) &&
                            // this is oneDistanceRoot of the neighbor's aggregate.
                            (S->vecValues[2 * S->col_remote[iter]] & weightMax) > (weight2[S->row_remote[iter]] & weightMax)) {
//                            if(rank==1) cout << "second before\t" << "row = " << S->row_remote[iter]+S->split[rank] << "  \tinitial weight = " << initialWeight[S->row_remote[iter]] << ",\taggregate = " << aggregate2[S->row_remote[iter]] << "\tremote weight = " << (weight2[S->row_remote[iter]]&weightMax) << "\tremote status = " << (S->vecValues[2*S->col_remote[iter]+1]>>aggOffset) << "\taggStat2 = " << aggStatus2[S->row_remote[iter]] << endl;
                            weight2[S->row_remote[iter]] = (S->vecValues[2 * S->col_remote[iter]] & weightMax);
                            aggregate2[S->row_remote[iter]] = (S->vecValues[2 * S->col_remote[iter]+1]&weightMax);
                            aggStatus2[S->row_remote[iter]] = (S->vecValues[2 * S->col_remote[iter]]>>aggOffset); // this is always 0.
//                            if(rank==1) cout << "second after \t" << "row = " << S->row_remote[iter]+S->split[rank] << "  \tinitial weight = " << initialWeight[S->row_remote[iter]] << ",\taggregate = " << aggregate2[S->row_remote[iter]] << "\t\tremote weight = " << (weight2[S->row_remote[iter]]&weightMax) << "\tremote status = " << (S->vecValues[2*S->col_remote[iter]+1]>>aggOffset) << "\taggStat2 = " << aggStatus2[S->row_remote[iter]] << endl;
                        }
                    }
                }
            }
        }
        // put weight2 in weight and aggregate2 in aggregate.
        // if a row does not have a remote element then (weight2[i]&weightMax) == (weight[i]&weightMax)
        //update aggStatus of remote elements at the same time
        for(i=0; i<size; i++){
            if(aggregate[i] != aggregate2[i]){
                aggregate[i] = aggregate2[i];
                if(aggStatus2[i] != 1) // if this is 1, it should go to the next aggregation round.
                    weight[i] = (0UL<<aggOffset | weight2[i]&weightMax);
            }
        }

        // update aggStatus
        for (i = 0; i < size; ++i) {
            if(weight[i]&UNDECIDED) {
//                if(rank==0) cout << "i = " << i << "\taggregate[i] = " << aggregate[i] << "\taggStatus[aggregate[i]] = " << aggStatus[aggregate[i]-S->split[rank]] << endl;

                // local
                if (aggregate[i] >= S->split[rank] && aggregate[i] < S->split[rank+1]) {
//                    if(rank==1) cout << "i = " << i << "\taggregate[i] = " << aggregate[i] << "\taggStatus[aggregate[i]] = " << aggStatus[aggregate[i]] << endl;
//                    if(rank==1) cout << "i = " << i+S->split[rank] << "\taggregate[i] = " << aggregate[i] << "\taggStatus[i] = " << (weight[i]>>aggOffset) << endl;
                    if (aggregate[i] == i + S->split[rank]) {
                        weight[i] = (2UL << aggOffset | weight[i] & weightMax); // change aggStatus of a root to 2.
                        oneDistanceRoot[i] = 0;
                        aggArray.push_back(aggregate[i]);
//                        (*aggSize)++;
                    } else if (weight[aggregate[i] - S->split[rank]] &ROOT) {
                        weight[i] = (0UL << aggOffset | weight[i] & weightMax); // this node is assigned to another aggregate.
                        if (oneDistanceRoot[aggregate[i] - S->split[rank]] == 0) oneDistanceRoot[i] = 1;
//                        if(rank==1) cout << "i = " << i+S->split[rank] << "\taggregate[i] = " << aggregate[i] << "\taggStatus[i] = " << (weight[i]>>aggOffset) << endl;
                    }
                }
                // remote
//                }else{
//                    if(aggStatus2[i] != 1) // if it is 1, it should go to the next aggregation round.
//                        weight[i] = (0UL<<aggOffset | weight2[i]&weightMax);
//                }
            }
        }

//        for(int k=0; k<nprocs; k++){
//            MPI_Barrier(comm);
//            if(rank==k){
//                cout << "final aggregate! rank:" << rank << ", iter = " << whileiter << endl;
//                for (i = 0; i < size; ++i){
//                    cout << "i = " << i+S->split[rank] << "\t\taggregate = " << aggregate[i] << "\t\taggStatus = "
//                         << (weight[i]>>aggOffset) << "\t\tinitial weight = " << initialWeight[i]
//                         << "\t\tcurrent weight = " << (weight[i]&weightMax) << "\t\taggStat2 = " << aggStatus2[i]
//                         << "\t\toneDistanceRoot = " << oneDistanceRoot[i] << endl;
//                }
//            }
//            MPI_Barrier(comm);
//        }

        continueAggLocal = false;
        for (i = 0; i < size; ++i) {
            // if any un-assigned node is available, continue aggregating.
            if(weight[i]&UNDECIDED) {
                continueAggLocal = true;
                break;
            }
        }

//        whileiter++;
//        if(whileiter==20) continueAggLocal = false;

        // check if every processor does not have any non-assigned node, otherwise all the processors should continue aggregating.
        MPI_Allreduce(&continueAggLocal, &continueAgg, 1, MPI_CXX_BOOL, MPI_LOR, comm);
//        MPI_Barrier(comm);
//        cout << rank << "\tcontinueAgg = " << continueAgg << endl;
//        MPI_Barrier(comm);

        if(continueAgg){
            for (i = 0; i < size; ++i) {
                aggStatus2[i] = 1;
                if(weight[i]&UNDECIDED){
                    weight[i] = ( 1UL<<aggOffset | initialWeight[i] );
                    aggregate[i] = i+S->split[rank];
                }
            }
        }

    } //while(continueAgg)

//    MPI_Barrier(comm);
//    if(rank==nprocs-1) cout << "number of loops to find aggregation: " << whileiter << endl;
//    MPI_Barrier(comm);

    free(oneDistanceRoot);

    // *************************** update aggregate to new indices ****************************

//    if(rank==1){
//        cout << "split" << endl;
//        for(i=0; i<nprocs+1; i++)
//            cout << S->split[i] << endl;
//        cout << endl;
//    }

//    sort(aggregate, &aggregate[size]);

//    if(rank==1){
//        cout << "aggregate:" << endl;
//        for(i=0; i<size; i++)
//            cout << aggregate[i] << endl;
//        cout << endl;
//    }

    sort(aggArray.begin(), aggArray.end());

//    if(rank==1){
//        cout << "aggArray:" << endl;
//        for(auto i:aggArray)
//            cout << i << endl;
//        cout << endl;
//    }

//    unsigned long* splitNew = (unsigned long*)malloc(sizeof(unsigned long)*(nprocs+1));
    fill(splitNew, &splitNew[nprocs], 0);
    splitNew[rank] = aggArray.size();

    unsigned long* splitNewTemp = (unsigned long*)malloc(sizeof(unsigned long)*nprocs);
    MPI_Allreduce(splitNew, splitNewTemp, nprocs, MPI_UNSIGNED_LONG, MPI_SUM, comm);

    // do scan on splitNew
    splitNew[0] = 0;
    for(i=1; i<nprocs+1; i++)
        splitNew[i] = splitNew[i-1] + splitNewTemp[i-1];

    free(splitNewTemp);

    if(rank==1){
        cout << "splitNew:" << endl;
        for(i=0; i<nprocs+1; i++)
            cout << S->split[i] << "\t" << splitNew[i] << endl;
        cout << endl;
    }

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
        procNum = lower_bound2(&S->split[0], &S->split[nprocs+1], i);
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

//    delete requests2;
//    delete statuses2;

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

//    MPI_Barrier(comm);
//    if(rank==0){
//        cout << "aggregate:\t"<< rank << endl;
//        for(i=0; i<size; i++)
//            cout << aggregate[i] << endl;
//        cout << endl;
//    }
//    MPI_Barrier(comm);
//    if(rank==1){
//        cout << "\naggregate:\t"<< rank << endl;
//        for(i=0; i<size; i++)
//            cout << aggregate[i] << endl;
//        cout << endl;
//    }
//    MPI_Barrier(comm);
//    if(rank==2){
//        cout << "aggregate:\t"<< rank << endl;
//        for(i=0; i<size; i++)
//            cout << aggregate[i] << endl;
//        cout << endl;
//    }
//    MPI_Barrier(comm);

    free(aggSend);
    free(aggRecv);
    free(isAggRemote);
    free(recvCount);
    free(vIndexCount);
    free(vIndex);
    return 0;
}


// Decoupled Aggregation - not complete
/*
int AMGClass::Aggregation(CSRMatrix* S){
    // At the end just set P here, which is in AMGClass.

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


int AMGClass::createProlongation(COOMatrix* A, unsigned long* aggregate, unsigned long* splitNew, prolongMatrix* P, MPI_Comm comm){
    // todo: check when you should update new aggregate values: before creating prolongation or after.

    // Here P is computed: P = A_w * P_t; in which P_t is aggregate, and A_w = I - w*Q*A, Q is inverse of diagonal of A.
    // Here A_w is computed on the fly, while adding values to P. Diagonal entries of A_w are 0, so they are skipped.
    // todo: think about A_F which is A filtered.
    // todo: think about smoothing preconditioners other than damped jacobi. check the following paper:
    // todo: Eran Treister and Irad Yavneh, Non-Galerkin Multigrid based on Sparsified Smoothed Aggregation. page22.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    unsigned int i, j;
    float omega = 0.67; // todo: receive omega as user input. it is usually 2/3 for 2d and 6/7 for 3d.

    P->Mbig = A->Mbig;
    P->Nbig = splitNew[nprocs]; // This is the number of aggregates, which is the number of columns of P.
    P->M = A->M;
    P->splitNew = splitNew;

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
        for (j = 0; j < A->nnz_col_remote[i]; ++j, ++iter) {
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

    for(i=0; i<PEntryTemp.size(); i++){
        P->entry.push_back(PEntryTemp[i]);
        while(i<PEntryTemp.size()-1 && PEntryTemp[i] == PEntryTemp[i+1]){ // values of entries with the same row and col should be added.
            P->entry[P->entry.size()-1].val += PEntryTemp[i+1].val;
            i++;
        }
    }

//    if(rank==1)
//        for(i=0; i<P->entry.size(); i++)
//            cout << P->entry[i].row << "\t" << P->entry[i].col << "\t" << P->entry[i].val << endl;

    PEntryTemp.clear();

    P->nnz_l = P->entry.size();
    MPI_Allreduce(&P->nnz_l, &P->nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

    P->split = &*A->split.begin();

    P->findLocalRemote(&*P->entry.begin(), comm);
//    P->findLocalRemote(&*P->row.begin(), &*P->col.begin(), &*P->values.begin(), comm);

    return 0;
}// end of AMGClass::createProlongation


int AMGClass::coarsen(COOMatrix* A, prolongMatrix* P, restrictMatrix* R, prolongMatrix* Ac, MPI_Comm comm){

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    unsigned long i, j, start;
    prolongMatrix RA;

    // ************************************* RA - A local *************************************
    // Some local and remote elements of RA are computed here using local R and local A.

    unsigned int* AnnzPerRow = (unsigned int*)malloc(sizeof(unsigned int)*A->M);
    fill(&AnnzPerRow[0], &AnnzPerRow[A->M], 0);
    for(i=0; i<A->nnz_l; i++){
        AnnzPerRow[A->entryP[i].row - A->split[rank]]++;
    }

//    if(rank==2)
//        for(i=0; i<A->M; i++)
//            cout << AnnzPerRow[i] << endl;

    unsigned int* AnnzPerRowScan = (unsigned int*)malloc(sizeof(unsigned int)*(A->M+1));
    AnnzPerRowScan[0] = 0;
    for(i=0; i<A->M; i++){
        AnnzPerRowScan[i+1] = AnnzPerRowScan[i] + AnnzPerRow[i];
//        if(rank==2) printf("i=%lu, AnnzPerRow=%d, AnnzPerRowScan = %d\n", i, AnnzPerRow[i], AnnzPerRowScan[i]);
    }

    for(i=0; i<R->nnz_l_local; i++){
        for(j = AnnzPerRowScan[R->entry_local[i].col - P->split[rank]]; j < AnnzPerRowScan[R->entry_local[i].col - P->split[rank] + 1]; j++){
            RA.entry.push_back(cooEntry(R->entry_local[i].row,
                                        A->entryP[i].col,
                                        R->entry_local[i].val * A->entryP[i].val));
        }
    }

//    if(rank==1)
//        for(i=0; i<RA.entry.size(); i++)
//            cout << RA.entry[i].row << "\t" << RA.entry[i].col << "\t" << RA.entry[i].val << endl;

    free(AnnzPerRow);
    free(AnnzPerRowScan);




    // iterate through A column-wise and P row-wise.
//    for(i = 0; i < A->nnz_l_local; ++i) {
//        start = P->nnzPerRowScan_local[A->col_local[i] - A->split[rank]];
//        if(rank==1) printf("P.row=%lu, start=%lu \n", A->col_local[i]-A->split[rank], start);
//        for(j=0; j < P->nnzPerRow_local[A->col_local[i] - A->split[rank]]; j++){
//            if(rank==1) printf("A.row=%lu, A.col=%lu, A.val=%f \tP.row=%lu, P.col=%lu, P.val=%f \n", A->row_local[i], A->col_local[i], A->values_local[i], P->entry_local[P->indicesP_local[start + j]].row, P->entry_local[P->indicesP_local[start + j]].col, P->entry_local[P->indicesP_local[start + j]].val);
//            AP->entry_local.push_back(cooEntry(A->row_local[i],
//                                               P->entry_local[P->indicesP_local[start + j]].col,
//                                               A->values_local[i] * P->entry_local[P->indicesP_local[start + j]].val));
//        }
//    }

/*
    // todo: this is wrong: should switch rows of A with its columns. It means you should use nnzPerCol, which is expensive memory-wise.
    // iterate through R column-wise and A row-wise.
    // R is stored in local and remote formats. there does not exist general entries for R, so R is iterated first through local entries, then the remote ones.
    // R local, A first local then remote
    for(i = 0; i < R->nnz_l_local; ++i) {
        start = A->nnzPerRowScan_local[R->entry_local[i].col - P->split[rank]]; // todo: check if split should be replaced by splitNew.
//        if(rank==1) printf("R.col=%lu, start=%lu, A.AnnzPerRow=%d \n", R->entry_local[i].col - P->split[rank], start, A->nnzPerRow_local[R->entry_local[i].col - P->split[rank]]);
        // A local
        for(j=0; j < A->nnzPerRow_local[R->entry_local[i].col - P->split[rank]]; j++){
//            if(rank==1) printf("R.row=%lu, R.col=%lu, R.val=%f,\tA.row=%lu, A.col=%lu, A.val=%f \n", R->entry_local[i].row, (R->entry_local[i].col-P->split[rank]), R->entry_local[i].val, A->row_local[A->indicesP_local[start + j]], A->col_local[A->indicesP_local[start + j]], A->values_local[A->indicesP_local[start + j]]);
            RA.entry_local.push_back(cooEntry( R->entry_local[i].row,
                                               A->col_local[A->indicesP_local[start + j]],
                                               R->entry_local[i].val * A->values_local[A->indicesP_local[start + j]]));
        }

        // A remote
        start = A->nnzPerRowScan_remote[R->entry_local[i].col - P->split[rank]];
        for(j=0; j < A->nnzPerRow_remote[R->entry_local[i].col - P->split[rank]]; j++){
//            if(rank==1) printf("R.col=%lu, start=%lu, A.AnnzPerRow=%d \n", R->entry_local[i].col - P->split[rank], start, A->nnzPerRow_remote[R->entry_local[i].col - P->split[rank]]);
            RA.entry_local.push_back(cooEntry( R->entry_local[i].row,
                                               A->col_remote2[A->indicesP_remote[start + j]],
                                               R->entry_local[i].val * A->values_remote[A->indicesP_remote[start + j]]));
        }
    }
//    printf("rank=%d\thereeeeee!!!\n", rank);
    // iterate through R column-wise and A row-wise. (R remote is not sorted!)
    // R remote, A first local then remote
    for(i = 0; i < R->nnz_l_remote; ++i) {
        start = A->nnzPerRowScan_local[R->entry_remote[i].col - P->split[rank]];
        if(rank==2) printf("\nR.col=%lu, R.col-split=%lu, start=%lu, A.AnnzPerRow=%d, nnz_l_remote=%lu \n", R->entry_remote[i].col, R->entry_remote[i].col-P->split[rank], start, A->nnzPerRow_local[R->entry_remote[i].col - P->split[rank]], R->nnz_l_remote);
        // A local
        for(j=0; j < A->nnzPerRow_local[R->entry_remote[i].col - P->split[rank]]; j++){
            if(rank==2) printf("R.row=%lu, R.col=%lu, R.val=%f,\tA.row=%lu, A.col=%lu, A.val=%f \n", R->entry_remote[i].row, (R->entry_remote[i].col-P->split[rank]), R->entry_remote[i].val, A->row_local[A->indicesP_local[start + j]], A->col_local[A->indicesP_local[start + j]], A->values_local[A->indicesP_local[start + j]]);
//            RA.entry_local.push_back(cooEntry( R->entry_remote[i].row,
//                                               A->col_local[A->indicesP_local[start + j]],
//                                               R->entry_remote[i].val * A->values_local[A->indicesP_local[start + j]]));
        }
        // A remote
        start = A->nnzPerRowScan_remote[R->entry_remote[i].col - P->split[rank]];
        for(j=0; j < A->nnzPerRow_remote[R->entry_remote[i].col - P->split[rank]]; j++){
//            if(rank==1) printf("A.row=%lu, A.col=%lu, A.val=%f \tP.row=%lu, P.col=%lu, P.val=%f \n", A->row_local[i], A->col_local[i], A->values_local[i], P->entry_local[P->indicesP_local[start + j]].row, P->entry_local[P->indicesP_local[start + j]].col, P->entry_local[P->indicesP_local[start + j]].val);
            RA.entry_local.push_back(cooEntry( R->entry_remote[i].row,
                                               A->col_remote2[A->indicesP_remote[start + j]],
                                               R->entry_remote[i].val * A->values_remote[A->indicesP_remote[start + j]]));
        }
    }
*/

    // ************************************* RA - A remote *************************************

    unsigned int AMaxNnz;
    MPI_Allreduce(&A->nnz_l, &AMaxNnz, 1, MPI_UNSIGNED, MPI_MAX, comm);
//    MPI_Barrier(comm); printf("rank=%d, AMaxNnz=%d \n", rank, AMaxNnz); MPI_Barrier(comm);
    // todo: is this way better than using the previous Allreduce? reduce on processor 0, then broadcast to other processors.
    cooEntry* Arecv = (cooEntry*)malloc(sizeof(cooEntry)*AMaxNnz);

//    unsigned int* RStart = (unsigned int*)malloc(sizeof(unsigned int)*(nprocs+1));

    int left, right;
    unsigned int nnzRecv;
    MPI_Status sendRecvStatus;

    for(int i = 1; i < nprocs; i++) {
//        MPI_Barrier(comm); printf("rank=%d, i=%d , start! \n", rank, i); MPI_Barrier(comm);
        // Don't communicate with yourself.
//        if(i == rank)
//            continue;

        //sendrecv(size)
        // send A to the right processor, recieve A from the left processor. "left" decreases by one in each iteration. "right" increases by one.
        right = (rank + i) % nprocs;
        left = rank - i;
        if (left < 0)
            left += nprocs;
//        MPI_Barrier(comm); printf("rank=%d, left=%d, right=%d \n", rank, left, right); MPI_Barrier(comm);

        // use sender rank for send and receive tags.
        MPI_Sendrecv(&A->nnz_l, 1, MPI_UNSIGNED, right, rank, &nnzRecv, 1, MPI_UNSIGNED, left, left, comm, &sendRecvStatus);
//        int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf,
//                         int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status)

//        if(rank==2) cout << "own A->nnz_l = " << A->nnz_l << "\tnnzRecv = " << nnzRecv << endl;
//        MPI_Barrier(comm); printf("rank=%d, my A->nnz_l = %d, nnzRecv = %d \n", rank, A->nnz_l, nnzRecv); MPI_Barrier(comm);

        //sendrecv(data)
        // use sender rank for send and receive tags.
        MPI_Sendrecv(A->entryP, A->nnz_l, cooEntry::mpi_datatype(), right, rank, Arecv, nnzRecv, cooEntry::mpi_datatype(), left, left, comm, &sendRecvStatus);
//        if(rank==1)
//            for(int j=0; j<nnzRecv; j++)
//                printf("j=%d \t %lu \t %lu \t %f \n", j, Arecv[j].row, Arecv[j].col, Arecv[j].val);

//        MPI_Barrier(comm); printf("rank=%d, done! \n", rank); MPI_Barrier(comm);
        //multiplication

    }

//    if(rank==1)
//        for(int j=0; j<A->nnz_l; j++)
//            printf("j=%d \t %lu \t %lu \t %f \n", j, A->entryP[j].row, A->entryP[j].col, A->entryP[j].val);

    free(Arecv);
//    free(RStart);

    return 0;
} // end of AMGClass::coarsen