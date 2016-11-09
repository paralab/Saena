#include <fstream>
#include <algorithm>
#include <sys/stat.h>
#include "stdlib.h"

#include "coomatrix.h"
#include "mpi.h"

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


COOMatrix::COOMatrix(const char* filePath, char* filePath2) {

    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    struct stat st;
    stat(filePath, &st);
    // 2*sizeof(long)+sizeof(double) = 24
    long nnz_g = st.st_size/24;

    MPI_Status status;
    MPI_File fh;
    MPI_Offset offset;

    MPI_File_open(MPI_COMM_WORLD, filePath2, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    long first_nnz_l = nnz_g / nprocs; //local nnz
    offset = rank * first_nnz_l * 24;
    long data[3*first_nnz_l];
    MPI_File_read_at(fh, offset, data, 3*first_nnz_l, MPI_UNSIGNED_LONG, &status);

    int count;
    MPI_Get_count(&status, MPI_UNSIGNED_LONG, &count);
    //printf("process %d read %d lines of triples\n", rank, count);
    MPI_File_close(&fh);

    //cout << "nnz_g = " << nnz_g << ", first_nnz_l = " << first_nnz_l << endl;

/*    if (rank==0){
        cout << "data:" << endl;
        for (int i=0; i<3*first_nnz_l;i++){
            cout << data[i] << endl;
        }
    }*/



    long m = 18;
    long n_buckets = nprocs*nprocs;
    // change this part to make it work with any number of processors, instead of just the factors of m
    long splitOffset = m / n_buckets;
    long firstSplit[n_buckets+1];

    for(long i=0; i<n_buckets+1; i++){
        firstSplit[i] = i*splitOffset;
    }

/*    if(rank==0){
        cout << "firstSplit:" << endl;
        for(int i=0; i<n_buckets+1; i++)
            cout << firstSplit[i] << endl;
    }*/

    // definition of buckets: bucket[i] = [ firstSplit[i] , firstSplit[i+1] ). Number of buckets = n_buckets

    long H_l[n_buckets];
    fill(&H_l[0], &H_l[n_buckets], 0);

    long* low2;
    long temp;
    for(long i=0; i<first_nnz_l; i++)
        H_l[lower_bound2(&firstSplit[0], &firstSplit[n_buckets], data[3*i])]++;

/*    if (rank==0){
        cout << "local histogram:" << endl;
        for(long i=0; i<n_buckets; i++)
            cout << H_l[i] << endl;
    }*/

    long H_g[n_buckets];
    MPI_Allreduce(H_l, H_g, n_buckets, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

/*    if (rank==0){
        cout << "global histogram:" << endl;
        for(long i=0; i<n_buckets; i++){
            cout << H_g[i] << endl;
        }
    }*/

    long H_g_scan[n_buckets];
    H_g_scan[0] = H_g[0];
    for (long i=1; i<n_buckets; i++)
        H_g_scan[i] = H_g[i] + H_g_scan[i-1];

/*
    if (rank==0){
        cout << "scan of global histogram:" << endl;
        for(long i=0; i<n_buckets; i++)
            cout << H_g_scan[i] << endl;
    }
*/

    long procNum=0;
    //long split[nprocs+1];
    split = (long*) malloc(sizeof(long) * nprocs+1);
    split[0]=0;
    for (long i=1; i<n_buckets; i++){
        if (H_g_scan[i] >= ((procNum+1)*nnz_g/nprocs)){
            //check the last element of split to be correct
            split[procNum+1] = splitOffset*(i+1);
            procNum++;
        }
    }

/*
    if (rank==0){
        cout << "split:" << endl;
        for(long i=0; i<nprocs+1; i++)
            cout << split[i] << endl;
    }
*/

/*    if(rank==3){
        procNum=2;
        long nnzTest = 0;
        for (long i=0; i<first_nnz_l; i++){
            if( data[3*i]>=split[procNum] && data[3*i]<split[procNum+1] )
                nnzTest++;
        }
        cout << "nnzTest = " << nnzTest << endl;
    }*/

    long tempIndex;
    int sendSizeArray[nprocs];
    fill(&sendSizeArray[0], &sendSizeArray[nprocs], 0);
    for (long i=0; i<first_nnz_l; i++){
        tempIndex = lower_bound2(&split[0], &split[nprocs+1], data[3*i]);
        sendSizeArray[tempIndex]++;
    }

/*    if (rank==0){
        cout << "sendSizeArray:" << endl;
        for(long i=0;i<nprocs;i++)
            cout << sendSizeArray[i] << endl;
    }*/

/*    cout << endl << "sendSizeArray:" << "rank=" << rank << endl;
    for(long j=0;j<nprocs;j++)
        cout  << sendSizeArray[j] << endl;*/

    int recvSizeArray[nprocs];
    MPI_Alltoall(sendSizeArray, 1, MPI_INT, recvSizeArray, 1, MPI_INT, MPI_COMM_WORLD);

/*    if (rank==0){
        cout << "recvSizeArray:" << endl;
        for(long i=0;i<nprocs;i++)
            cout << recvSizeArray[i] << endl;
    }*/

    int sOffset[nprocs];
    sOffset[0] = 0;
    for (long i=1; i<nprocs; i++)
        sOffset[i] = sendSizeArray[i-1] + sOffset[i-1];

/*    if (rank==0){
        cout << "sOffset:" << endl;
        for(long i=0;i<nprocs;i++)
            cout << sOffset[i] << endl;
    }*/

    int rOffset[nprocs];
    rOffset[0] = 0;
    for (long i=1; i<nprocs; i++)
        rOffset[i] = recvSizeArray[i-1] + rOffset[i-1];

/*    if (rank==0){
        cout << "rOffset:" << endl;
        for(long i=0;i<nprocs;i++)
            cout << rOffset[i] << endl;
    }*/

    long procOwner;
    long bufTemp;
    long sendBufI[first_nnz_l];
    long sendBufJ[first_nnz_l];
    long sendBufV[first_nnz_l];
    long sIndex[nprocs];
    fill(&sIndex[0], &sIndex[nprocs], 0);

    for (long i=0; i<first_nnz_l; i++){
        procOwner = lower_bound2(&split[0], &split[nprocs+1], data[3*i]);
        procOwner = lower_bound2(&split[0], &split[nprocs+1], data[3*i]);
        bufTemp = sOffset[procOwner]+sIndex[procOwner];
        sendBufI[bufTemp] = data[3*i];
        sendBufJ[bufTemp] = data[3*i+1];
        sendBufV[bufTemp] = data[3*i+2];
        sIndex[procOwner]++;
    }

/*    if (rank==0){
        cout << "sendBufI:" << endl;
        for (long i=0; i<first_nnz_l; i++)
            cout << sendBufI[i] << endl;
    }*/

    nnz_l = rOffset[nprocs-1] + recvSizeArray[nprocs-1];
    //long row[nnz_l];
    //long col[nnz_l];
    //double values[nnz_l];

    values = (double*) malloc(sizeof(double) * nnz_l);
    row = (long*) malloc(sizeof(long) * nnz_l);
    col = (long*) malloc(sizeof(long) * nnz_l);

    MPI_Alltoallv(sendBufI, sendSizeArray, sOffset, MPI_LONG, row, recvSizeArray, rOffset, MPI_LONG, MPI_COMM_WORLD);
    MPI_Alltoallv(sendBufJ, sendSizeArray, sOffset, MPI_LONG, col, recvSizeArray, rOffset, MPI_LONG, MPI_COMM_WORLD);
    MPI_Alltoallv(sendBufV, sendSizeArray, sOffset, MPI_DOUBLE, values, recvSizeArray, rOffset, MPI_DOUBLE, MPI_COMM_WORLD);

    // setting the number of rows for processor
    M = split[rank+1] - split[rank];

 /*   if (rank==0){
        cout << "M = " << M << endl;
        cout << "nnz_l = " << nnz_l << endl;
        valprint();
    }*/

    //M = m/nprocs;
    //N = Nbig;

    vElement = (long*) malloc(sizeof(long) * nnz_l);
    vElementRep = (long*) malloc(sizeof(long) * nnz_l);
    vElementSize = 0;
    recvCount = (int*)malloc(sizeof(long)*nprocs);
    std::fill(recvCount, recvCount + nprocs, 0.);
    procNum = 0;

    vElement[0] = col[0];
    vElementSize = 1;
    vElementRep[0] = 1;
    //recvCount[findProcess(col[0], procNum, nprocs)] = 1;
    recvCount[lower_bound2(&split[0], &split[n_buckets], col[0])] = 1;

    for (long i = 1; i < nnz_l; i++) {
        if(col[i] == col[i-1]){
            vElementRep[vElementSize-1] = vElementRep[vElementSize-1] + 1;
        }else{
            vElement[vElementSize] = col[i];
            vElementRep[vElementSize] = 1;
            vElementSize++;

            //procNum = findProcess(col[i], procNum, nprocs);
            procNum = lower_bound2(&split[0], &split[n_buckets], col[i]);
            recvCount[procNum]++;
            //cout << "recvCount[procNum] = " << recvCount[procNum] << endl;
        }
    }

/*    if (rank==0){
        cout << "recvCount:" << endl;
        for(int i=0; i<nprocs; i++)
            cout << recvCount[i] << endl;
    }*/

    vIndexCount = (int*)malloc(sizeof(int)*nprocs);

    MPI_Alltoall(recvCount, 1, MPI_INT, vIndexCount, 1, MPI_INT, MPI_COMM_WORLD);

/*    if (rank==1){
        cout << "vIndexCount:" << endl;
        for(int i=0; i<nprocs; i++)
            cout << vIndexCount[i] << endl;
    }*/

    int vIndexSize = 0;
    for (int i=0; i<nprocs; i++)
        vIndexSize += vIndexCount[i];

/*
    long* vBuf = (long*)malloc(sizeof(long)*nnz_l);
    for (long i=0; i<vElementSize; i++)
        vBuf[i] = vElement[i]%M;
*/

    int* vdispls = (int*)malloc(sizeof(int)*nprocs);
    int* rdispls = (int*)malloc(sizeof(int)*nprocs);
    vdispls[0] = 0;
    rdispls[0] = 0;

    for (int i=1; i<nprocs; i++){
        vdispls[i] = vdispls[i-1] + vIndexCount[i-1];
        rdispls[i] = rdispls[i-1] + recvCount[i-1];
    }

    vIndex = (long*)malloc(sizeof(long)*vIndexSize);
    MPI_Alltoallv(vElement, recvCount, rdispls, MPI_LONG, vIndex, vIndexCount, vdispls, MPI_LONG, MPI_COMM_WORLD);

    free(vdispls);
    free(rdispls);
    //free(vBuf);

}

COOMatrix::~COOMatrix() {
    free(values);
    free(row);
    free(col);
    free(vElement);
    free(vElementRep);
    free(recvCount);
    free(vIndex);
    free(vIndexCount);
}

void COOMatrix::matvec(double* v, double* w){
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

/*    long vIndexSize = 0;
    for (int i=0; i<nprocs; i++)
        vIndexSize += vIndexCount[i];*/

    int vdispls[nprocs];
    int rdispls[nprocs];
    vdispls[0] = 0;
    rdispls[0] = 0;
    for (int i=1; i<nprocs; i++){
        vdispls[i] = vdispls[i-1] + vIndexCount[i-1];
        rdispls[i] = rdispls[i-1] + recvCount[i-1];
    }

    long vIndexSize = vdispls[nprocs-1] + vIndexCount[nprocs-1];
    long recvSize = rdispls[nprocs-1] + recvCount[nprocs-1];

    double *vSend = (double*)malloc(sizeof(double) * vIndexSize);
    for(long i=0;i<vIndexSize;i++){
        //if (rank==1) cout << "vIndex[i]%M = " << vIndex[i]%M << endl;
        vSend[i] = v[vIndex[i]%M];
    }

    double* vValuesCompressed = (double*) malloc(sizeof(double) * recvSize);
    MPI_Alltoallv(vSend, vIndexCount, vdispls, MPI_DOUBLE, vValuesCompressed, recvCount, rdispls, MPI_DOUBLE, MPI_COMM_WORLD);

/*    if (rank==0){
        cout << "vValuesCompressed:" << endl;
        for(long int i=0;i<recvSize;i++) {
            cout << vValuesCompressed[i] << endl;
        }
    }*/

    long iter = 0;
    double *vValues = (double *) malloc(sizeof(double) * nnz_l);
    for (long i=0; i<vElementSize; i++){
        for (long j=0; j<vElementRep[i]; j++) {
            vValues[iter] = vValuesCompressed[i];
            iter++;
        }
    }

/*    if (rank==0){
        cout << "vValues:" << endl;
        for(long int i=0;i<nnz_l;i++) {
            cout << vValues[i] << endl;
        }
    }*/

    fill(&w[0], &w[M], 0);
    // the following and the above loops can be combined to have only one nested for loop.
    for(long i=0;i<nnz_l;i++) {
        w[row[i] - split[rank]] += values[i] * vValues[i];
        //w[row[i]] += values[i] * col[i];
        //w[i] += values[j] * v[row[j]];
    }

    free(vSend);
    free(vValuesCompressed);
    //free(vValues);
}

void COOMatrix::valprint(){
    cout << "val:" << endl;
    for(long int i=0;i<nnz_l;i++) {
        cout << values[i] << endl;
    }
}

void COOMatrix::rowprint(){
    cout << "row:" << endl;
    for(long int i=0;i<nnz_l;i++) {
        cout << row[i] << endl;
    }
}

void COOMatrix::colprint(){
    cout << endl << "col:" << endl;
    for(long int i=0;i<nnz_l;i++) {
        cout << col[i] << endl;
    }
}

void COOMatrix::vElementprint(){
    cout << endl << "vElement:" << endl;
    for(long int i=0;i<vElementSize;i++) {
        cout << vElement[i] << endl;
    }
}

void COOMatrix::vElementRepprint(){
    cout << endl << "vElementRep:" << endl;
    for(long int i=0;i<vElementSize;i++) {
        cout << vElementRep[i] << endl;
    }
}

void COOMatrix::print(){
    cout << endl << "triple:" << endl;
    for(long int i=0;i<nnz_l;i++) {
        cout << "(" << row[i] << " , " << col[i] << " , " << values[i] << ")" << endl;
    }
}

/*
int COOMatrix::findProcess(long a, int procNum, int p) {
    while(procNum < p){
        if (a >= procNum*M && a < (procNum+1)*M)
            return procNum;

        procNum++;
    }
    return procNum;
}*/
