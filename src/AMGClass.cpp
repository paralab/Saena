//
// Created by abaris on 3/14/17.
//

#include <cstdio>
#include "AMGClass.h"
#include <mpi.h>
//#include "coomatrix.h"


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


AMGClass::AMGClass(int levels, int vcycle_num, double relTol, string relaxType, int preSmooth, int postSmooth, float connStrength, bool doSparsify){
    levels = levels;
    vcycle_num = vcycle_num;
    relTol  = relTol;
    relaxType = relaxType;
    preSmooth = preSmooth;
    postSmooth = postSmooth;
    connStrength = connStrength;
    doSparsify = doSparsify;
} //AMGClass

AMGClass::~AMGClass(){

}

int AMGClass::AMGsetup(COOMatrix* A, bool doSparsify){

    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double* S = NULL;
    findAggregation(A, connStrength, S);

    return 0;
}

int AMGClass::findAggregation(COOMatrix* A, float connStrength, double* S){
    createStrengthMatrix(A, S);
    return 0;
}

int AMGClass::createStrengthMatrix(COOMatrix* A, double* S_p){

    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(rank==1) cout << "M = " << A->M << ", nnz_l = " << A->nnz_l << endl;

    unsigned int i;
    double maxPerRow[A->M];
    fill(&maxPerRow[0], &maxPerRow[A->M], 0);
    for(i=0; i<A->nnz_l; i++){
        if( A->row[i] != A->col[i] ){
            if(maxPerRow[A->row[i] - A->split[rank]] == 0)
                maxPerRow[A->row[i] - A->split[rank]] = -A->values[i];
            else if(maxPerRow[A->row[i] - A->split[rank]] < -A->values[i])
                maxPerRow[A->row[i] - A->split[rank]] = -A->values[i]; // use split to convert the index from global to local.
        }
    }

/*    if(rank==0)
        for(i=0; i<A->M; i++)
            cout << "max = " << maxPerRow[i] << endl;*/

    std::vector<long> Si;
    std::vector<long> Sj;
    std::vector<double> Sval;
    for(i=0; i<A->nnz_l; i++){
        if(A->row[i] == A->col[i]) {
            Si.push_back(A->row[i]);
            Sj.push_back(A->col[i]);
            Sval.push_back(1);
        }
        else if(maxPerRow[A->row[i] - A->split[rank]] != 0)
            if(-A->values[i] / maxPerRow[A->row[i] - A->split[rank]] > connStrength){
                Si.push_back(A->row[i]);
                Sj.push_back(A->col[i]);
                Sval.push_back(-A->values[i] / maxPerRow[A->row[i] - A->split[rank]]);
//                if(rank==1) cout << "index = " << A->row[i] - A->split[rank] << ", max = " << maxPerRow[A->row[i] - A->split[rank]] << endl;
//                if(rank==1) cout << "A.val = " << -A->values[i] << ", max = " << maxPerRow[A->row[i] - A->split[rank]] << ", divide = " << -A->values[i] / maxPerRow[A->row[i] - A->split[rank]] << endl;
            }
    }

/*    if(rank==0)
        for (i=0; i<Si.size(); i++)
            cout << "val = " << Sval[i] << endl;*/




    // finding the arrays for redistributing S_transpose. We consider Si, as STj and Sj as STi.
    long tempIndex;
    int sendSizeArray[nprocs];
    fill(&sendSizeArray[0], &sendSizeArray[nprocs], 0);
    for (i=0; i<Si.size(); i++){
        tempIndex = lower_bound2(&A->split[0], &A->split[nprocs+1], Sj[i]); //redistribution based on Sj, which is rows of S transpose.
        sendSizeArray[tempIndex]++;
    }

/*    if (rank==0){
        cout << "sendSizeArray:" << endl;
        for(long i=0;i<nprocs;i++)
            cout << sendSizeArray[i] << endl;
    }*/

    int recvSizeArray[nprocs];
    MPI_Alltoall(sendSizeArray, 1, MPI_INT, recvSizeArray, 1, MPI_INT, MPI_COMM_WORLD);

/*    if (rank==0){
        cout << "recvSizeArray:" << endl;
        for(long i=0;i<nprocs;i++)
            cout << recvSizeArray[i] << endl;
    }*/

    int sOffset[nprocs];
    sOffset[0] = 0;
    for (i=1; i<nprocs; i++)
        sOffset[i] = sendSizeArray[i-1] + sOffset[i-1];

/*    if (rank==0){
        cout << "sOffset:" << endl;
        for(long i=0;i<nprocs;i++)
            cout << sOffset[i] << endl;
    }*/

    int rOffset[nprocs];
    rOffset[0] = 0;
    for (i=1; i<nprocs; i++)
        rOffset[i] = recvSizeArray[i-1] + rOffset[i-1];

/*    if (rank==0){
        cout << "rOffset:" << endl;
        for(long i=0;i<nprocs;i++)
            cout << rOffset[i] << endl;
    }*/

/*    long procOwner;
    unsigned int bufTemp;
    long sendBufI[initial_nnz_l];
    long sendBufJ[initial_nnz_l];
    long sendBufV[initial_nnz_l];
    unsigned int sIndex[nprocs];
    fill(&sIndex[0], &sIndex[nprocs], 0);

    for (i=0; i<Si.size(); i++){
        procOwner = lower_bound2(&A->split[0], &A->split[nprocs+1], data[3*i]);
        bufTemp = sOffset[procOwner]+sIndex[procOwner];
        sendBufI[bufTemp] = data[3*i];
        sendBufJ[bufTemp] = data[3*i+1];
        sendBufV[bufTemp] = data[3*i+2];
        sIndex[procOwner]++;
    }*/

    std::vector<long> STi(Si.size());
    std::vector<long> STj(Si.size());
    std::vector<double> STval(Si.size());

    long* STi_p = &(*(STi.begin()));
    long* STj_p = &(*(STj.begin()));
    double* STval_p = &(*(STval.begin()));

    long* Si_p = &(*(Si.begin()));
    long* Sj_p = &(*(Sj.begin()));
    double* Sval_p = &(*(Sval.begin()));

    MPI_Alltoallv(Si_p, sendSizeArray, sOffset, MPI_LONG, STj_p, recvSizeArray, rOffset, MPI_LONG, MPI_COMM_WORLD);
    MPI_Alltoallv(Sj_p, sendSizeArray, sOffset, MPI_LONG, STi_p, recvSizeArray, rOffset, MPI_LONG, MPI_COMM_WORLD);
    MPI_Alltoallv(Sval_p, sendSizeArray, sOffset, MPI_DOUBLE, STval_p, recvSizeArray, rOffset, MPI_DOUBLE, MPI_COMM_WORLD);

/*    i = 68;
    if(rank==0) cout << "S size  = " << Si.size() << endl;
    if(rank==0) cout << "S  = " << Si[i] << " " << Sj[i] << " " << Sval[i] << endl;
    if(rank==0) cout << "ST = " << STi[i] << " " << STj[i] << " " << STval[i] << endl;*/

    for(i=0; i<Si.size(); i++){
        Sval[i] += STval[i];
        Sval[i] *= 0.5;
    }


    return 0;
}


