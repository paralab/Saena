//
// Created by abaris on 3/14/17.
//

#include <cstdio>
#include <algorithm>
#include <mpi.h>
#include <random>

#include "AMGClass.h"
//#include "coomatrix.h"
//#include "csrmatrix.h"


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

int randomVector(long* V, long size){
    //Type of random number distribution
    std::uniform_real_distribution<double> dist(1, 10*size); //(min, max)

    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng;

    //Initialize with non-deterministic seeds
    rng.seed(std::random_device{}());

    for (long i=0; i<size; i++)
        V[i] = (long)dist(rng);

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

int AMGClass::AMGsetup(COOMatrix* A, bool doSparsify){

    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    findAggregation(A);

    return 0;
}

int AMGClass::findAggregation(COOMatrix* A){
    CSRMatrix S;
    createStrengthMatrix(A, &S);
//    S.print(0);
    long aggregate[S.M];
    Aggregation(&S, aggregate);
    return 0;
}

int AMGClass::createStrengthMatrix(COOMatrix* A, CSRMatrix* S){

    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//    if(rank==0) cout << "M = " << A->M << ", nnz_l = " << A->nnz_l << endl;

    // ******************************** compute max per row ********************************

    unsigned int i;
    double maxPerRow[A->M];
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

    std::vector<long> Si;
    std::vector<long> Sj;
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
//                if (rank==0) cout << Sval[Sval.size()-1] << "\t" << connStrength << endl;
//                if(rank==1) cout << "index = " << A->row[i] - A->split[rank] << ", max = " << maxPerRow[A->row[i] - A->split[rank]] << endl;
//                if(rank==0) cout << "A.val = " << -A->values[i] << ", max = " << maxPerRow[A->row[i] - A->split[rank]] << ", divide = " << (-A->values[i] / (maxPerRow[A->row[i] - A->split[rank]])) << endl;
//            }
        }
    }

/*    if(rank==0)
        for (i=0; i<Si.size(); i++)
            cout << "val = " << Sval[i] << endl;*/

    // ******************************** compute max per column - version 1 - for general matrices ********************************

    double local_maxPerCol[A->Mbig];
    fill(&local_maxPerCol[0], &local_maxPerCol[A->Mbig], 0);
    for(i=0; i<A->nnz_l; i++){
        if( A->row[i] != A->col[i] ){
            if(local_maxPerCol[A->col[i]] == 0)
                local_maxPerCol[A->col[i]] = -A->values[i];
            else if(local_maxPerCol[A->col[i]] < -A->values[i])
                local_maxPerCol[A->col[i]] = -A->values[i];
        }
    }

    double maxPerCol[A->Mbig];
    MPI_Allreduce(&local_maxPerCol, &maxPerCol, A->Mbig, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

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
//        for (unsigned int j = 0; j < A->nnz_row_remote[i]; ++j, ++iter) {
//            w[A->row_remote[A->indicesP_remote[iter]]] += A->values_remote[A->indicesP_remote[iter]] * A->vecValues[A->col_remote[A->indicesP_remote[iter]]];
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

    std::vector<long> Si2;
    std::vector<long> Sj2;
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

//    if(rank==3)
//        for(i=0; i<Si2.size(); i++){
//            cout << "S:  " << "[" << Si2[i] << "," << Sj2[i] << "] = " << Sval2[i] << endl;
//            cout << "S:  " << "[" << (Si2[i] - A->split[rank]) << "," << Sj2[i] << "] = \t" << Sval2[i] << endl;
//        }

    // S indices are local on each process, which means it starts from 0 on each process.
    S->CSRMatrixSet(&(*(Si2.begin())), &(*(Sj2.begin())), &(*(Sval2.begin())), A->M, A->Mbig, Si2.size(), &(*(A->split.begin())));
//    S->print(0);
    return 0;
}


// Using MIS(2) from the following paper by Luke Olson:
// EXPOSING FINE-GRAINED PARALLELISM IN ALGEBRAIC MULTIGRID METHODS
int AMGClass::Aggregation(CSRMatrix* S, long* aggregate) {

    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    long i, j;
    long size = S->M;
    long maxIndex[size];
    long maxIndex2[size];
    int aggregated[size];
//    long aggregate[size];
    long aggregate2[size];
    randomVector(aggregate, size);

//    if(rank==0){
//        cout << endl << "after initialization!" << endl;
//        for (i = 0; i < size; ++i)
//            cout << i << "\taggregate = " << aggregate[i] << endl;
//    }

    for(i=0; i<size; i++) {
        maxIndex[i] = i + S->split[rank];
//        maxIndex[i] = -1;
        aggregated[i] = 0;
    }

//    long maxTemp;
//    for(i=0; i<size; i++){
//        maxTemp = maxIndex[i];
//        for(j=S->rowIndex[i]; j<S->rowIndex[i+1]; j++){
//            if(j >= S->split[rank] && j < S->split[rank+1])
//                //store index of max too.
//                maxTemp = max(maxTemp, aggregate[j]);
//        }
//    }

    // ******************************* first round of max computation *******************************

    long maxWeightTemp, maxIndexTemp;
    // first "compute max" is local. second one is both local and remote.
    long iter = 0;
    for (i = 0; i < size; ++i) {
        maxWeightTemp = aggregate[i];
        maxIndexTemp  = maxIndex[i];
        for (j = 0; j < S->nnz_row_local[i]; ++j, ++iter) {
//            w[i] += values_local[indicesP_local[iter]] * v[col_local[indicesP_local[iter]]];
//            if(rank==0) cout << i << "\t" << S->col_local[S->indicesP_local[iter]] << endl;
            if(aggregate[S->col_local[S->indicesP_local[iter]]] > maxWeightTemp){
                maxWeightTemp = aggregate[S->col_local[S->indicesP_local[iter]]];
                maxIndexTemp  = maxIndex[S->col_local[S->indicesP_local[iter]]];
            }
        }
        aggregate2[i]   = maxWeightTemp;
        maxIndex2[i] = maxIndexTemp;
    }
    for (i = 0; i < size; ++i) {
        aggregate[i] = aggregate2[i];
        maxIndex[i] = maxIndex2[i];
    }

//    if(rank==0){
//        cout << endl << "after first max computation!" << endl;
//        for (i = 0; i < size; ++i)
//            cout << i << "\tweight = " << aggregate[i] << "\tindex = " << maxIndex[i] << endl;
//    }

    // ******************************* exchange remote max values for the second round of max computation *******************************

    //vSend are maxPerCol for remote elements that should be sent to other processes.
    for(i=0;i<S->vIndexSize;i++)
        S->vSend[i] = aggregate[( S->vIndex[i] )];

    MPI_Request* requests = new MPI_Request[S->numSendProc+S->numRecvProc];
    MPI_Status* statuses = new MPI_Status[S->numSendProc+S->numRecvProc];

    //vecValues are maxperCol for remote elements that are received from other processes.
    // Do not recv from self.
    for(i = 0; i < S->numRecvProc; i++)
        MPI_Irecv(&S->vecValues[S->rdispls[S->recvProcRank[i]]], S->recvProcCount[i], MPI_LONG, S->recvProcRank[i], 1, MPI_COMM_WORLD, &(requests[i]));

    // Do not send to self.
    for(i = 0; i < S->numSendProc; i++)
        MPI_Isend(&S->vSend[S->vdispls[S->sendProcRank[i]]], S->sendProcCount[i], MPI_LONG, S->sendProcRank[i], 1, MPI_COMM_WORLD, &(requests[S->numRecvProc+i]));

    // ******************************* second round of max computation *******************************

    // local part
    iter = 0;
    for (i = 0; i < size; ++i) {
        maxWeightTemp = aggregate[i];
        maxIndexTemp = maxIndex[i];
        for (j = 0; j < S->nnz_row_local[i]; ++j, ++iter) {
//            w[i] += values_local[indicesP_local[iter]] * v[col_local[indicesP_local[iter]]];
            if(aggregate[S->col_local[S->indicesP_local[iter]]] > maxWeightTemp){
                maxWeightTemp = aggregate[S->col_local[S->indicesP_local[iter]]];
                maxIndexTemp  = maxIndex[S->col_local[S->indicesP_local[iter]]];
            }
        }
        aggregate2[i]   = maxWeightTemp;
        maxIndex2[i] = maxIndexTemp;
    }
    for (i = 0; i < size; ++i) {
        aggregate[i] = aggregate2[i];
        maxIndex[i] = maxIndex2[i];
    }

//    if(rank==0){
//        cout << endl << "after second max computation!" << endl;
//        for (i = 0; i < size; ++i)
//            cout << i << "\tweight = " << aggregate[i] << "\tindex = " << maxIndex[i] << endl;
//    }

    MPI_Waitall(S->numSendProc+S->numRecvProc, requests, statuses);

    // remote part
    iter = 0;
    long iter2;
    for (i = 0; i < S->col_remote_size; ++i) {
        iter2 = iter;
        maxWeightTemp = aggregate[S->row_remote[S->indicesP_remote[iter]]]; // the weight of the row so far.
        maxIndexTemp  = maxIndex[S->row_remote[S->indicesP_remote[iter]]];
        for (j = 0; j < S->nnz_row_remote[i]; ++j, ++iter) {
            if(S->vecValues[S->col_remote[S->indicesP_remote[iter]]] > maxWeightTemp) {
                maxWeightTemp = S->vecValues[S->col_remote[S->indicesP_remote[iter]]];
                maxIndexTemp = S->col_remote2[S->indicesP_remote[iter]];
//            cout << "row = " << S->row_remote[S->indicesP_remote[iter]] << ", col = " << S->col_remote2[S->indicesP_remote[iter]] << "\tweight = " << S->vecValues[S->col_remote[S->indicesP_remote[iter]]] << endl;
            }
        }
        aggregate2[S->row_remote[S->indicesP_remote[iter2]]]   = maxWeightTemp;
        maxIndex2[S->row_remote[S->indicesP_remote[iter2]]] = maxIndexTemp;
//        if(rank==0) cout << S->row_remote[S->indicesP_remote[iter]] << "\t" << aggregate2[i] << endl;
    }

    for (i = S->row_remote[S->indicesP_remote[0]]; i < size; ++i) {
        aggregate[i] = aggregate2[i];
        maxIndex[i] = maxIndex2[i];
    }

//    if(rank==0){
//        cout << "final weight for remote part!" << endl;
//        for (i = S->row_remote[S->indicesP_remote[0]]; i < size; ++i)
//            cout << i << "\tweight = " << aggregate[i] << "\tindex = " << maxIndex[i] << endl;
//    }

//    if(rank==3){
//        cout << "final aggregate!" << endl;
//        for (i = 0; i < size; ++i){
//            cout << i << "\tweight = " << aggregate[i] << "\tindex = " << maxIndex[i] << endl;
//            cout << "i = " << i << "\tmax_index = " << maxIndex[i] << endl;
//        }
//    }

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