//
// Created by abaris on 3/14/17.
//

#include <cstdio>
#include <algorithm>
#include <mpi.h>
#include <random>
#include <usort/parUtils.h>
#include "AMGClass.h"
#include "prolongmatrix.h"
#include "restrictmatrix.h"
//#include "coomatrix.h"
//#include "strengthmatrix.h"


// sort indices and store the ordering.
class sort_indices
{
private:
    long* mparr;
public:
    sort_indices(long* parr) : mparr(parr) {}
    bool operator()(long i, long j) const { return mparr[i]<mparr[j]; }
};

// binary search tree using the lower bound
template <class T>
T lower_bound2(T *left, T *right, T val) {
    T* first = left;
    while (left < right) {
        T *middle = left + (right - left) / 2;
        if (*middle < val){
            left = middle + 1;
        }
        else{
            right = middle;
        }
    }
    if(val == *left){
        return distance(first, left);
    }
    else
        return distance(first, left-1);
}

int randomVector(unsigned long* V, unsigned long size){
//    int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Type of random number distribution
    std::uniform_int_distribution<unsigned long> dist(1, size); //(min, max)

    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng;

    //Initialize with non-deterministic seeds
    rng.seed(std::random_device{}());

    for (unsigned long i=0; i<size; i++)
        V[i] = dist(rng);

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


int AMGClass::AMGSetup(COOMatrix* A, bool doSparsify){

    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<unsigned long> aggregate(A->M);
    unsigned long* aggregate_p = &(*aggregate.begin());
    findAggregation(A, aggregate_p); // todo: think about a parameter for making the aggregation less or more aggressive.

//    MPI_Barrier(MPI_COMM_WORLD);
//    if(rank==0)
//        for(long i=0; i<A->M; i++)
//            cout << i << "\t" << aggregate[i] << endl;
//    MPI_Barrier(MPI_COMM_WORLD);

/*
    std::vector<long> aggregateSorted(A->M);
//    long* aggregateSorted_p = &(*aggregateSorted.begin());
    par::sampleSort(aggregate, aggregateSorted, MPI_COMM_WORLD);
    if(rank==0) cout << "\nafter:" << endl;
    if(rank==0)
        for(long i=0; i<A->M; i++)
            cout << i << "\t" << aggregate[i] << "\t" << aggregateSorted[i] << endl;
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==1)
        for(long i=0; i<A->M; i++)
            cout << i << "\t" << aggregate[i] << "\t" << aggregateSorted[i] << endl;
    MPI_Barrier(MPI_COMM_WORLD);
*/


//    par::sampleSort(aggregate, MPI_COMM_WORLD);

    /*
    if(rank==0) cout << "\nafter:" << endl;
    if(rank==0)
        for(long i=0; i<A->M; i++)
            cout << i << "\t" << aggregate[i] << endl;
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==1)
        for(long i=0; i<A->M; i++)
            cout << i << "\t" << aggregate[i] << endl;
    MPI_Barrier(MPI_COMM_WORLD);
*/

    prolongMatrix P;
    createProlongation(A, aggregate_p, &P);
    restrictMatrix R;
    createRestriction(&P, &R);

//    if(rank==0)
//        for(long i=0; i<A->nnz_l; i++)
//            cout << P.row[i] << "\t" << P.col[i] << "\t" << P.values[i] << endl;

    return 0;
}


int AMGClass::findAggregation(COOMatrix* A, unsigned long* aggregate){
    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    StrengthMatrix S;
    createStrengthMatrix(A, &S);
//    S.print(0);

    Aggregation(&S, aggregate);

//    if(rank==0)
//        for(long i=0; i<S.M; i++)
//            cout << i << "\t" << aggregate[i] << endl;

    return 0;
} // end of AMGClass::findAggregation


int AMGClass::createStrengthMatrix(COOMatrix* A, StrengthMatrix* S){
    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//    if(rank==0) cout << "M = " << A->M << ", nnz_l = " << A->nnz_l << endl;

    // ******************************** compute max per row ********************************

    unsigned int i;
//    double maxPerRow[A->M];
    std::vector<double> maxPerRow(A->M);
    fill(&maxPerRow[0], &maxPerRow[A->M], 0);
    for(i=0; i<A->nnz_l; i++){
        if( A->row[i] != A->col[i] ){
            if(maxPerRow[A->row[i] - A->split[rank]] == 0) // use split to convert the index from global to local.
                maxPerRow[A->row[i] - A->split[rank]] = -A->values[i];
            else if(maxPerRow[A->row[i] - A->split[rank]] < -A->values[i])
                maxPerRow[A->row[i] - A->split[rank]] = -A->values[i];
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
        if(A->row[i] == A->col[i]) {
            Si.push_back(A->row[i]);
            Sj.push_back(A->col[i]);
            Sval.push_back(1);
        }
        else if(maxPerRow[A->row[i] - A->split[rank]] != 0) {
//            if ( -A->values[i] / (maxPerRow[A->row[i] - A->split[rank]] ) > connStrength) {
            Si.push_back(A->row[i]);
            Sj.push_back(A->col[i]);
            Sval.push_back(  -A->values[i] / (maxPerRow[A->row[i] - A->split[rank]])  );
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
        if( A->row[i] != A->col[i] ){
            if(local_maxPerCol[A->col[i]] == 0)
                local_maxPerCol[A->col[i]] = -A->values[i];
            else if(local_maxPerCol[A->col[i]] < -A->values[i])
                local_maxPerCol[A->col[i]] = -A->values[i];
        }
    }

//    double maxPerCol[A->Mbig];
    std::vector<double> maxPerCol(A->Mbig);
    double* maxPerCol_p = &(*maxPerCol.begin());
//    MPI_Allreduce(&local_maxPerCol, &maxPerCol, A->Mbig, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(local_maxPerCol_p, maxPerCol_p, A->Mbig, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

//    if(rank==0)
//        for(i=0; i<A->Mbig; i++)
//            cout << i << "\t" << maxPerCol[i] << endl;

    // ******************************** compute ST - version 1 ********************************

    std::vector<long> STi;
    std::vector<long> STj;
    std::vector<double> STval;
    for(i=0; i<A->nnz_l; i++){
        if(A->row[i] == A->col[i]) {
            STi.push_back(A->row[i] - A->split[rank]);
            STj.push_back(A->col[i] - A->split[rank]);
            STval.push_back(1);
        }
        else{
//            if ( (-A->values[i] / maxPerCol[A->col[i]]) > connStrength) {
            STi.push_back(A->row[i] - A->split[rank]);
            STj.push_back(A->col[i] - A->split[rank]);
            STval.push_back( -A->values[i] / maxPerCol[A->col[i]] );
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
        MPI_Irecv(&A->vecValues[A->rdispls[A->recvProcRank[i]]], A->recvProcCount[i], MPI_DOUBLE, A->recvProcRank[i], 1, MPI_COMM_WORLD, &(requests[i]));

    // Do not send to self.
    for(i = 0; i < A->numSendProc; i++)
        MPI_Isend(&A->vSend[A->vdispls[A->sendProcRank[i]]], A->sendProcCount[i], MPI_DOUBLE, A->sendProcRank[i], 1, MPI_COMM_WORLD, &(requests[A->numRecvProc+i]));

    // ******************************** compute ST - version 2 ********************************

    std::vector<long> STi;
    std::vector<long> STj;
    std::vector<double> STval;

    // add OpenMP just like matvec.
    long iter = 0;
    long iter2 = 0;
//    for (i = 0; i < A->M; ++i, iter2++) {
//        for (unsigned int j = 0; j < A->nnz_row_local[i]; ++j, ++iter) {
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
//            cout << "S:  " << "[" << Si2[i] << "," << Sj2[i] << "] = " << Sval2[i] << endl;
//            cout << "S:  " << "[" << (Si2[i] - A->split[rank]) << "," << Sj2[i] << "] = \t" << Sval2[i] << endl;
//        }

    // S indices are local on each process, which means it starts from 0 on each process.
    S->StrengthMatrixSet(&(*(Si2.begin())), &(*(Sj2.begin())), &(*(Sval2.begin())), A->M, A->Mbig, Si2.size(), &(*(A->split.begin())));
    return 0;
} // end of AMGClass::createStrengthMatrix


// Using MIS(2) from the following paper by Luke Olson:
// EXPOSING FINE-GRAINED PARALLELISM IN ALGEBRAIC MULTIGRID METHODS
int AMGClass::Aggregation(StrengthMatrix* S, unsigned long* aggregate) {

    // the first two bits of aggregate are being used for aggStatus: 1 for 01 not assigned, 0 for 00 assigned, 2 for 10 root
    // bits 0 up to 61 are for storing aggregate values.
    // the max value for aggregate is 2^63 - 1

    // variables used in this function:
    // weight[i]: the two most left bits store the status of node i, the other 62 bits store weight assigned to that node.
    //            weight is first generated randomly by randomVector function and saved in initialWeight. During the
    //            aggregation process, it becomes the weight of the node's aggregate.
    //

    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    unsigned long i, j;
    unsigned long size = S->M;
//    long aggregate[size];
    std::vector<unsigned long> aggregate2(size);
//    std::vector<unsigned long> aggStatus(size);
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

    int aggOffset = 62;
    unsigned long weightMax = (1UL<<aggOffset) - 1;
    unsigned long weightTemp, aggregateTemp, aggStatusTemp;
    bool* oneDistanceRoot = (bool*)malloc(sizeof(bool)*size);
    fill(&oneDistanceRoot[0], &oneDistanceRoot[size], 0);
    bool continueAggLocal;
    bool continueAgg = true;
    unsigned long iter;
    int whileiter = 0;

    MPI_Request *requests = new MPI_Request[S->numSendProc + S->numRecvProc];
    MPI_Status *statuses  = new MPI_Status[S->numSendProc + S->numRecvProc];

    // initialization
    for(i=0; i<size; i++) {
        aggregate[i] = i + S->split[rank];
        weight[i] = (1UL<<aggOffset | (initialWeight[i]&weightMax) ); // status of each node is initialized to 1 and its weight to initialWeight.
        aggStatus2[i] = 0;
    }

    while(continueAgg) {
        // ******************************* first round of max computation *******************************
        // first "compute max" is local. The second one is both local and remote.

        iter = 0;
        for (i = 0; i < size; ++i) {
            oneDistanceRoot[i] = false;
            if(weight[i]>>aggOffset == 1) {
                aggregateTemp = aggregate[i];
                aggStatusTemp = (weight[i]>>aggOffset); // this will be used for aggStatus2, and aggStatus2 will be used for the remote part.
                weightTemp = weight[i];
                for (j = 0; j < S->nnz_row_local[i]; ++j, ++iter) {

                    //distance-1 aggregate
                    if( weight[S->col_local[S->indicesP_local[iter]]] > weightTemp ){ // by doing this, it first compares aggStatus, then if aggStatus are both equal, it compares weight.
                        aggregateTemp = aggregate[S->col_local[S->indicesP_local[iter]]];
                        weightTemp = weight[S->col_local[S->indicesP_local[iter]]];
                        aggStatusTemp = (weight[S->col_local[S->indicesP_local[iter]]]>>aggOffset);
                        if(aggStatusTemp==2) oneDistanceRoot[i] = true;

                    // distance-2 aggregate
                    }else if(!oneDistanceRoot[i] &&
                            (oneDistanceRoot[S->col_local[S->indicesP_local[iter]]]==1) &&
                            ( (weight[S->col_local[S->indicesP_local[iter]]]&weightMax) > (weightTemp&weightMax) ) )
                    {
                        aggregateTemp = aggregate[S->col_local[S->indicesP_local[iter]]];
                        weightTemp = weight[S->col_local[S->indicesP_local[iter]]];
                        oneDistanceRoot[i] = false;
                        aggStatusTemp = (weight[aggregate[S->col_local[S->indicesP_local[iter]]]]>>aggOffset); // this will be used for aggStatus2, and aggStatus2 will be used for thr remote part.
//                        if(rank==1) cout << "*** second: " << i << "\tagg = " << aggregateTemp << "\taggStatus = " << aggStatusTemp << endl;
                    }
                }
                weight2[i]    = weightTemp;
//            weight2[i]  = (aggStatusTemp<<aggOffset | weightTemp);
                aggregate2[i] = aggregateTemp;
                aggStatus2[i] = aggStatusTemp; // this is stored only to be compared with the remote one in the remote part.
//            if(rank==1) cout << i << "\t" << aggStatusTemp << endl;
            }else
                iter += S->nnz_row_local[i];
        }

        for (i = 0; i < size; ++i) {
            if(S->nnz_row_local[i] != 0) {
                aggStatusTemp = (weight[i]>>aggOffset);
                weight[i] = (aggStatusTemp<<aggOffset | (weight2[i]&weightMax) );
                aggregate[i] = aggregate2[i];
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

        for (i = 0; i < S->vIndexSize; i++){
//            S->vSend[i] = weight[(S->vIndex[i])];
            S->vSend[2*i] = weight[S->vIndex[i]];
            S->vSend[2*i+1] = ( (oneDistanceRoot[S->vIndex[i]]<<aggOffset) | (aggregate[S->vIndex[i]]&weightMax) );
        }

        for (i = 0; i < S->numRecvProc; i++)
            MPI_Irecv(&S->vecValues[S->rdispls[S->recvProcRank[i]]], S->recvProcCount[i], MPI_UNSIGNED_LONG,
                      S->recvProcRank[i], 1, MPI_COMM_WORLD, &(requests[i]));

        for (i = 0; i < S->numSendProc; i++)
            MPI_Isend(&S->vSend[S->vdispls[S->sendProcRank[i]]], S->sendProcCount[i], MPI_UNSIGNED_LONG,
                      S->sendProcRank[i], 1, MPI_COMM_WORLD, &(requests[S->numRecvProc + i]));

        // ******************************* second round of max computation *******************************

        // local part
        iter = 0;
        for (i = 0; i < size; ++i) {
            if(weight[i]>>aggOffset == 1) {
//                oneDistanceRoot[i] = false;
                aggregateTemp = aggregate[i];
//                weightTemp = (weight[i]&weightMax);
                aggStatusTemp = (weight[i]>>aggOffset); // this will be used for aggStatus2, and aggStatus2 will be used for the remote part.
                weightTemp = weight[i];
                for (j = 0; j < S->nnz_row_local[i]; ++j, ++iter) {

                    // distance-1 aggregate
                    if( weight[S->col_local[S->indicesP_local[iter]]] > weightTemp ){ // by doing this, it first compares aggStatus, then if aggStatus are both equal, it compares weight.
                        aggregateTemp = aggregate[S->col_local[S->indicesP_local[iter]]];
                        weightTemp = weight[S->col_local[S->indicesP_local[iter]]];
                        aggStatusTemp = (weight[S->col_local[S->indicesP_local[iter]]]>>aggOffset); // this will be used for aggStatus2, and aggStatus2 will be used for thr remote part.
                        if(aggStatusTemp==2) oneDistanceRoot[i] = true;

                    // distance-2 aggregate
                    }else if(!oneDistanceRoot[i] &&
                             (oneDistanceRoot[S->col_local[S->indicesP_local[iter]]]==1) &&
                             ( (weight[S->col_local[S->indicesP_local[iter]]]&weightMax) > (weightTemp&weightMax) ) )
                    {
                        aggregateTemp = aggregate[S->col_local[S->indicesP_local[iter]]];
                        weightTemp = weight[S->col_local[S->indicesP_local[iter]]];
                        aggStatusTemp = (weight[aggregate[S->col_local[S->indicesP_local[iter]]]]>>aggOffset); // this will be used for aggStatus2, and aggStatus2 will be used for thr remote part.
                        oneDistanceRoot[i] = false;
//                        if(rank==1) cout << "*** second: " << i << "\tagg = " << aggregateTemp << "\taggStatus = " << aggStatusTemp << endl;
                    }
                }
                weight2[i]    = weightTemp;
//                weight2[i]  = (aggStatusTemp<<aggOffset | weightTemp);
                aggregate2[i] = aggregateTemp;
                aggStatus2[i] = aggStatusTemp; // this is stored only to be compared with the remote one in the remote part.
            }else
                iter += S->nnz_row_local[i];
        }

        for (i = 0; i < size; ++i) {
            if(S->nnz_row_local[i] != 0) {
                aggStatusTemp = (weight[i]>>aggOffset);
                weight[i] = (aggStatusTemp<<aggOffset | (weight2[i]&weightMax) );
                aggregate[i] = aggregate2[i];
            }
        }

//        if(rank==1){
//            cout << endl << "after second max computation!" << endl;
//            for (i = 0; i < size; ++i)
//                cout << i << "\tweight = " << weight[i] << "\tindex = " << aggregate[i] << "\taggStatus = " << aggStatus[i] << endl;
//        }

        MPI_Waitall(S->numSendProc + S->numRecvProc, requests, statuses);

        MPI_Barrier(MPI_COMM_WORLD);
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
        MPI_Barrier(MPI_COMM_WORLD);

        // remote part
        // store the max of rows of remote elements in weight2 and aggregate2.
        iter = 0;
        for (i = 0; i < S->col_remote_size; ++i) {
            for (j = 0; j < S->nnz_col_remote[i]; ++j, ++iter) {
//                if(rank==1) if(S->row_remote[iter]==0) cout << "row:" << S->row_remote[iter]+S->split[rank] << "\taggregate = " << (S->vecValues[2*S->col_remote[iter]+1]&weightMax) << "\tstatus of agg: " << (S->vecValues[2*S->col_remote[iter]+1]>>aggOffset) << endl;
                if(weight[S->row_remote[iter]]>>aggOffset == 1){
                    if(  ((S->vecValues[2*S->col_remote[iter]]>>aggOffset) >  aggStatus2[S->row_remote[iter]]) ||
                        (((S->vecValues[2*S->col_remote[iter]]>>aggOffset) == aggStatus2[S->row_remote[iter]]) &&
                        ( (S->vecValues[2*S->col_remote[iter]]&weightMax)  > (weight2[S->row_remote[iter]]&weightMax) )))
                    {
//                        if(rank==1) cout << "first  before\t" << "row = " << S->row_remote[iter]+S->split[rank] << "  \tinitial weight = " << initialWeight[S->row_remote[iter]] << ",\taggregate = " << aggregate2[S->row_remote[iter]] << "\tremote weight = " << (weight2[S->row_remote[iter]]&weightMax) << "\tremote status = " << (S->vecValues[2*S->col_remote[iter]+1]>>aggOffset) << "\taggStat2 = " << aggStatus2[S->row_remote[iter]] << endl;

                        weight2[S->row_remote[iter]]    = (S->vecValues[2*S->col_remote[iter]]&weightMax);
                        aggStatus2[S->row_remote[iter]] = (S->vecValues[2*S->col_remote[iter]]>>aggOffset);
                        aggregate2[S->row_remote[iter]] = S->vecValues[2*S->col_remote[iter]+1]&weightMax;
                        if((S->vecValues[2*S->col_remote[iter]]>>aggOffset)==2) oneDistanceRoot[S->row_remote[iter]] = true;

//                        if(rank==1) cout << "first  after \t" << "row = " << S->row_remote[iter]+S->split[rank] << "  \tinitial weight = " << initialWeight[S->row_remote[iter]] << ",\taggregate = " << aggregate2[S->row_remote[iter]] << "\tremote weight = " << (weight2[S->row_remote[iter]]&weightMax) << "\tremote status = " << (S->vecValues[2*S->col_remote[iter]+1]>>aggOffset) << "\taggStat2 = " << aggStatus2[S->row_remote[iter]] << endl;
                    }else if(!oneDistanceRoot[S->row_remote[iter]] && (S->vecValues[2*S->col_remote[iter]+1]>>aggOffset) == 1) // the second value is oneDistanceRoot of the neighbor's aggregate.
//                             (( (S->vecValues[2*S->col_remote[iter]+1]>>aggOffset) >  aggStatus2[S->row_remote[iter]]) ||
//                             ( ((S->vecValues[2*S->col_remote[iter]+1]>>aggOffset) == aggStatus2[S->row_remote[iter]]) &&
//                                ( (S->vecValues[2*S->col_remote[iter]]&weightMax)  > (weight2[S->row_remote[iter]]&weightMax) ))))
                    {
//                        if(rank==1) cout << "second before\t" << "row = " << S->row_remote[iter]+S->split[rank] << "  \tinitial weight = " << initialWeight[S->row_remote[iter]] << ",\taggregate = " << aggregate2[S->row_remote[iter]] << "\tremote weight = " << (weight2[S->row_remote[iter]]&weightMax) << "\tremote status = " << (S->vecValues[2*S->col_remote[iter]+1]>>aggOffset) << "\taggStat2 = " << aggStatus2[S->row_remote[iter]] << endl;

                        weight2[S->row_remote[iter]]    = (S->vecValues[2*S->col_remote[iter]]&weightMax);
                        aggStatus2[S->row_remote[iter]] = 2;
//                        aggStatus2[S->row_remote[iter]] = (S->vecValues[2*S->col_remote[iter]+1]>>aggOffset);
//                        aggregate2[S->row_remote[iter]] = S->col_remote2[iter];
                        aggregate2[S->row_remote[iter]] = S->vecValues[2*S->col_remote[iter]+1]&weightMax;

//                        if(rank==1) cout << "second after \t" << "row = " << S->row_remote[iter]+S->split[rank] << "  \tinitial weight = " << initialWeight[S->row_remote[iter]] << ",\taggregate = " << aggregate2[S->row_remote[iter]] << "\tremote weight = " << (weight2[S->row_remote[iter]]&weightMax) << "\tremote status = " << (S->vecValues[2*S->col_remote[iter]+1]>>aggOffset) << "\taggStat2 = " << aggStatus2[S->row_remote[iter]] << endl;
                    }
                }
            }
        }

        // put weight2 in weight and aggregate2 in aggregate.
        // if a row does not have a remote element then (weight2[i]&weightMax) == (weight[i]&weightMax)
        //update aggStatus of remote elements at the same time
        for(i=0; i<size; i++){
            if(aggregate[i] != aggregate2[i]){
//                aggStatusTemp = (weight[i]>>aggOffset); // keep aggStatus in weight.
                weight[i] = (0UL<<aggOffset | weight2[i]&weightMax);
                aggregate[i] = aggregate2[i];
//                weight[S->row_remote[iter]] = ( (0UL<<aggOffset) | (weight[S->row_remote[iter]]&weightMax));
            }
        }

        // update aggStatus - local
        for (i = 0; i < size; ++i) {
            if(aggregate[i] >= S->split[rank] && aggregate[i] < S->split[rank+1]) {
//                if(rank==0) cout << "i = " << i << "\taggregate[i] = " << aggregate[i] << "\taggStatus[aggregate[i]] = " << aggStatus[aggregate[i]-S->split[rank]] << endl;
                if ( (weight[i]>>aggOffset) == 1) {
//                    if(rank==1) cout << "i = " << i << "\taggregate[i] = " << aggregate[i] << "\taggStatus[aggregate[i]] = " << aggStatus[aggregate[i]] << endl;
//                    if(rank==1) cout << "i = " << i+S->split[rank] << "\taggregate[i] = " << aggregate[i] << "\taggStatus[i] = " << (weight[i]>>aggOffset) << endl;
                    if (aggregate[i] == i+S->split[rank]) {
                        weight[i] = (2UL<<aggOffset | weight[i]&weightMax); // change aggStatus of a root to 2.
                        aggStatus2[i] = 2; // to bed used in vSend
                        oneDistanceRoot[i] = 0;
                    } else if ( (weight[aggregate[i]-S->split[rank]]>>aggOffset) == 2){
                        weight[i] = (0UL<<aggOffset | weight[i]&weightMax); // this node is assigned to another aggregate.
                        if(oneDistanceRoot[aggregate[i]-S->split[rank]] == 0) oneDistanceRoot[i] = 1;
//                        if(rank==1) cout << "i = " << i+S->split[rank] << "\taggregate[i] = " << aggregate[i] << "\taggStatus[i] = " << (weight[i]>>aggOffset) << endl;
                    }
                }
            }
        }

        continueAggLocal = false;
        for (i = 0; i < size; ++i) {
            // if any un-assigned node is available, continue aggregating.
            if( (weight[i]>>aggOffset) == 1) {
                continueAggLocal = true;
                break;
            }
        }

//        whileiter++;
//        if(whileiter==10) continueAggLocal = false;

        // check if every processor does not have any non-assigned node, otherwise all the processors should continue aggregating.
        MPI_Allreduce(&continueAggLocal, &continueAgg, 1, MPI_CXX_BOOL, MPI_LOR, MPI_COMM_WORLD);
//        MPI_Barrier(MPI_COMM_WORLD);
//        cout << rank << "\tcontinueAgg = " << continueAgg << endl;
//        MPI_Barrier(MPI_COMM_WORLD);

        if(continueAgg){
            for (i = 0; i < size; ++i) {
                aggStatusTemp = (weight[i]>>aggOffset);
                if(aggStatusTemp==1){
                    weight[i] = ( aggStatusTemp<<aggOffset | initialWeight[i] );
                    aggregate[i] = i+S->split[rank];
                }
            }
        }

    } //while(continueAgg)

//    MPI_Barrier(MPI_COMM_WORLD);
//    if(rank==nprocs-1) cout << "number of loops to find aggregation: " << whileiter << endl;
//    MPI_Barrier(MPI_COMM_WORLD);

    free(oneDistanceRoot);
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


int AMGClass::createProlongation(COOMatrix* A, unsigned long* aggregate, prolongMatrix* P){

    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    unsigned int i, j;

    P->Mbig = A->Mbig;
//    P->Nbig = size of aggregate todo: fix this after writing aggregation update function.
    P->M = A->M;

    // store remote elements from aggregate in vSend to be sent to other processes.
    // change datatype for vSend and send and receive from double to long
    for(i=0; i<A->vIndexSize; i++)
        A->vSend[i] = aggregate[( A->vIndex[i] )];

    MPI_Request* requests = new MPI_Request[A->numSendProc+A->numRecvProc];
    MPI_Status* statuses = new MPI_Status[A->numSendProc+A->numRecvProc];

    for(i = 0; i < A->numRecvProc; i++) {
        MPI_Irecv(&A->vecValues[A->rdispls[A->recvProcRank[i]]], A->recvProcCount[i], MPI_DOUBLE, A->recvProcRank[i], 1, MPI_COMM_WORLD, &(requests[i]));
    }

    for(i = 0; i < A->numSendProc; i++) {
        MPI_Isend(&A->vSend[A->vdispls[A->sendProcRank[i]]], A->sendProcCount[i], MPI_DOUBLE, A->sendProcRank[i], 1, MPI_COMM_WORLD, &(requests[A->numRecvProc+i]));
    }

    // local
    long iter = 0;
    for (i = 0; i < A->M; ++i) {
        for (j = 0; j < A->nnz_row_local[i]; ++j, ++iter) {
            P->row.push_back(A->row_local[A->indicesP_local[iter]]);
            P->col.push_back(aggregate[  A->col_local[A->indicesP_local[iter]]  ]);
            P->values.push_back(A->values_local[A->indicesP_local[iter]]);
        }
    }

    MPI_Waitall(A->numSendProc+A->numRecvProc, requests, statuses);

    // remote
    iter = 0;
    for (i = 0; i < A->col_remote_size; ++i) {
        for (j = 0; j < A->nnz_col_remote[i]; ++j, ++iter) {
            P->row.push_back(A->row_remote[A->indicesP_remote[iter]]);
//            P->col.push_back(A->vecValues[  A->col_remote[A->indicesP_remote[iter]]  ]);
            P->col.push_back((unsigned long)(A->vecValues[A->col_remote[A->indicesP_remote[iter]]]));
            P->values.push_back(A->values_remote[A->indicesP_remote[iter]]);
        }
    }

    P->nnz_l = P->row.size();
    MPI_Allreduce(&P->nnz_l, &P->nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);

    P->split = &*A->split.begin();

    P->findLocalRemote(&*P->row.begin(), &*P->col.begin(), &*P->values.begin());

    return 0;
}// end of AMGClass::createProlongation


int AMGClass::createRestriction(prolongMatrix* P, restrictMatrix* R){

    R->Mbig = P->Nbig;
    R->Nbig = P->Mbig;

    return 0;
}// end of AMGClass::createRestriction