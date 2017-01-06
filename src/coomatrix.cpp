#include <fstream>
#include <algorithm>
#include <sys/stat.h>
#include "coomatrix.h"
//#include <math.h>
#include "mpi.h"
#include <omp.h>

#define ITERATIONS 1000

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


COOMatrix::COOMatrix(char* Aname, long Mbig) {

    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    struct stat st;
    stat(Aname, &st);
    // 2*sizeof(long)+sizeof(double) = 24
    long nnz_g = st.st_size/24;

    MPI_Status status;
    MPI_File fh;
    MPI_Offset offset;

    MPI_File_open(MPI_COMM_WORLD, Aname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    long initial_nnz_l = long(floor(1.0*nnz_g/nprocs)); // initial local nnz
    if (rank==nprocs-1)
        initial_nnz_l = nnz_g - (nprocs-1)*initial_nnz_l;

    //offset = rank * initial_nnz_l * 24; // row index(long=8) + column index(long=8) + value(double=8) = 24
    // the offset for the last process will be wrong if you use the above formula, because initial_nnz_l of the last process will be used, instead of the initial_nnz_l of the other processes.
    offset = rank * long(floor(1.0*nnz_g/nprocs)) * 24; // row index(long=8) + column index(long=8) + value(double=8) = 24
    long data[3*initial_nnz_l];
    MPI_File_read_at(fh, offset, data, 3*initial_nnz_l, MPI_UNSIGNED_LONG, &status);

    int count;
    MPI_Get_count(&status, MPI_UNSIGNED_LONG, &count);
    //printf("process %d read %d lines of triples\n", rank, count);
    MPI_File_close(&fh);

    //cout << "nnz_g = " << nnz_g << ", initial_nnz_l = " << initial_nnz_l << endl;

/*    if (rank==0){
        cout << "data:" << endl;
        for (int i=0; i<3*initial_nnz_l;i++){
            cout << data[i] << endl;
        }
    }*/

    // definition of buckets: bucket[i] = [ firstSplit[i] , firstSplit[i+1] ). Number of buckets = n_buckets
    int n_buckets=0;

/*    if (Mbig > nprocs*nprocs){
        if (nprocs < 1000)
            n_buckets = nprocs*nprocs;
        else
            n_buckets = 1000*nprocs;
    }
    else
        n_buckets = Mbig;*/

    if (Mbig > nprocs*nprocs){
        if (nprocs < 1000)
            n_buckets = nprocs*nprocs;
        else
            n_buckets = 1000*nprocs;
    }
    else if(nprocs < Mbig){
        n_buckets = Mbig;
    } else{
        if(rank == 0)
            cout << "number of tasks cannot be greater than te number of rows of the matrix." << endl;
        MPI_Finalize();
    }

/*    if (rank==0)
        cout << "buck = " << n_buckets << ", Mbig = " << Mbig << endl;*/

    splitOffset.resize(n_buckets);
    int baseOffset = int(floor(1.0*Mbig/n_buckets));
    float offsetRes = float(1.0*Mbig/n_buckets) - baseOffset;
    //cout << "baseOffset = " << baseOffset << ", offsetRes = " << offsetRes << endl;
    float offsetResSum = 0;
    splitOffset[0] = 0;
    for(long i=1; i<n_buckets; i++){
        splitOffset[i] = baseOffset;
        offsetResSum += offsetRes;
        if (offsetResSum >= 1){
            splitOffset[i]++;
            offsetResSum -= 1;
        }
    }

/*    if (rank==0){
        cout << "splitOffset:" << endl;
        for(long i=0; i<n_buckets; i++)
            cout << splitOffset[i] << endl;
    }*/

    long firstSplit[n_buckets+1];
    firstSplit[0] = 0;
    for(long i=1; i<n_buckets; i++){
        firstSplit[i] = firstSplit[i-1] + splitOffset[i];
    }
    firstSplit[n_buckets] = Mbig;

/*    if (rank==0){
        cout << "firstSplit:" << endl;
        for(long i=0; i<n_buckets+1; i++)
            cout << firstSplit[i] << endl;
    }*/

    long H_l[n_buckets];
    fill(&H_l[0], &H_l[n_buckets], 0);

    for(long i=0; i<initial_nnz_l; i++)
        H_l[lower_bound2(&firstSplit[0], &firstSplit[n_buckets], data[3*i])]++;

/*    if (rank==0){
        cout << "initial_nnz_l = " << initial_nnz_l << endl;
        cout << "local histogram:" << endl;
        for(unsigned int i=0; i<n_buckets; i++)
            cout << H_l[i] << endl;
    }*/

    long H_g[n_buckets];
    MPI_Allreduce(H_l, H_g, n_buckets, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

/*    if (rank==0){
        cout << "global histogram:" << endl;
        for(unsigned int i=0; i<n_buckets; i++){
            cout << H_g[i] << endl;
        }
    }*/

    long H_g_scan[n_buckets];
    H_g_scan[0] = H_g[0];
    for (long i=1; i<n_buckets; i++)
        H_g_scan[i] = H_g[i] + H_g_scan[i-1];

/*    if (rank==0){
        cout << "scan of global histogram:" << endl;
        for(unsigned int i=0; i<n_buckets; i++)
            cout << H_g_scan[i] << endl;
    }*/

    unsigned int procNum=0;
    split.resize(nprocs+1);
    split[0]=0;
    for (unsigned int i=1; i<n_buckets; i++){
        //if (rank==0) cout << "here: " << (procNum+1)*nnz_g/nprocs << endl;
        if (H_g_scan[i] > ((procNum+1)*nnz_g/nprocs)){
            procNum++;
            split[procNum] = firstSplit[i];
        }
    }
    split[nprocs] = Mbig;

/*    if (rank==0){
        cout << "split:" << endl;
        for(unsigned int i=0; i<nprocs+1; i++)
            cout << split[i] << endl;
    }*/

    long tempIndex;
    int sendSizeArray[nprocs];
    fill(&sendSizeArray[0], &sendSizeArray[nprocs], 0);
    for (long i=0; i<initial_nnz_l; i++){
        tempIndex = lower_bound2(&split[0], &split[nprocs+1], data[3*i]);
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
    long sendBufI[initial_nnz_l];
    long sendBufJ[initial_nnz_l];
    long sendBufV[initial_nnz_l];
    long sIndex[nprocs];
    fill(&sIndex[0], &sIndex[nprocs], 0);

    for (long i=0; i<initial_nnz_l; i++){
        procOwner = lower_bound2(&split[0], &split[nprocs+1], data[3*i]);
        bufTemp = sOffset[procOwner]+sIndex[procOwner];
        sendBufI[bufTemp] = data[3*i];
        sendBufJ[bufTemp] = data[3*i+1];
        sendBufV[bufTemp] = data[3*i+2];
        sIndex[procOwner]++;
    }

/*    if (rank==1){
        cout << "sendBufJ:" << endl;
        for (long i=0; i<initial_nnz_l; i++)
            cout << sendBufJ[i] << endl;
    }*/

    nnz_l = rOffset[nprocs-1] + recvSizeArray[nprocs-1];
    //cout << "nnz_l = " << nnz_l << endl;

    values.resize(nnz_l);
    row.resize(nnz_l);
    col.resize(nnz_l);

    double* valuesP = &(*(values.begin()));
    long* rowP = &(*(row.begin()));
    long* colP = &(*(col.begin()));

    MPI_Alltoallv(sendBufI, sendSizeArray, sOffset, MPI_LONG, rowP, recvSizeArray, rOffset, MPI_LONG, MPI_COMM_WORLD);
    MPI_Alltoallv(sendBufJ, sendSizeArray, sOffset, MPI_LONG, colP, recvSizeArray, rOffset, MPI_LONG, MPI_COMM_WORLD);
    MPI_Alltoallv(sendBufV, sendSizeArray, sOffset, MPI_DOUBLE, valuesP, recvSizeArray, rOffset, MPI_DOUBLE, MPI_COMM_WORLD);

    // setting the number of rows for processor
    M = split[rank+1] - split[rank];

/*    if (rank==3){
        colprint();
        //cout << "M = " << M << endl;
        //cout << "nnz_l = " << nnz_l << endl;
        //valprint();
        //print();
    }*/

    vElement = (long*) malloc(sizeof(long) * nnz_l);
    vElementRep = (long*) malloc(sizeof(long) * nnz_l); //the size of this, which is nnz_l, will be less, exactly: vElementSize
    vElementSize = 0;
    recvCount = (int*)malloc(sizeof(long)*nprocs);
    std::fill(recvCount, recvCount + nprocs, 0);
    std::fill(vElementRep, vElementRep + nnz_l, 0);

    vElement[0] = col[0];
    vElementSize = 1;
    vElementRep[0] = 1;
    recvCount[lower_bound2(&split[0], &split[nprocs], col[0])] = 1;
    bool temp;
    for (long i = 1; i < nnz_l; i++) {

        temp = (col[i] != col[i-1]);

        vElementRep[vElementSize-1] += (1-temp);        // vElementRep[vElementSize-1]++;

        vElement[vElementSize] = col[i];
        vElementRep[vElementSize] += temp;
        vElementSize += temp;

        procNum = lower_bound2(&split[0], &split[nprocs], col[i]);
        recvCount[procNum] += temp;

        // following "if statement" is changed to the above part. Computing the temp value is the same as checking the if statement. Which one is better?

/*        if(col[i] == col[i-1]){
            vElementRep[vElementSize-1]++;
        }else{
            vElement[vElementSize] = col[i];
            vElementRep[vElementSize] = 1;
            vElementSize++;

            //procNum = findProcess(col[i], procNum, nprocs);
            procNum = lower_bound2(&split[0], &split[nprocs], col[i]);
            //cout << "rank = " << rank << ", nnz_l = " << nnz_l << ", col[i] = " << col[i] << ", procNum = " << procNum << endl;
            recvCount[procNum]++;
            //cout << "recvCount[procNum] = " << recvCount[procNum] << endl;
        }*/
    }

/*    if (rank==0){
        //vElementprint();
        //vElementRepprint();
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

/*    int vIndexSize = 0;
    for (int i=0; i<nprocs; i++)
        vIndexSize += vIndexCount[i];*/

    long* vBuf = (long*)malloc(sizeof(long)*nnz_l);
    for (long i=0; i<vElementSize; i++)
        vBuf[i] = vElement[i]%M;

    vdispls.resize(nprocs);
    rdispls.resize(nprocs);
    //int* vdispls = (int*)malloc(sizeof(int)*nprocs);
    //int* rdispls = (int*)malloc(sizeof(int)*nprocs);
    vdispls[0] = 0;
    rdispls[0] = 0;

    for (int i=1; i<nprocs; i++){
        vdispls[i] = vdispls[i-1] + vIndexCount[i-1];
        rdispls[i] = rdispls[i-1] + recvCount[i-1];
    }
    vIndexSize = vdispls[nprocs-1] + vIndexCount[nprocs-1];
    recvSize = rdispls[nprocs-1] + recvCount[nprocs-1];

    vIndex = (long*)malloc(sizeof(long)*vIndexSize);
    MPI_Alltoallv(vElement, recvCount, &*(rdispls.begin()), MPI_LONG, vIndex, vIndexCount, &*(vdispls.begin()), MPI_LONG, MPI_COMM_WORLD);

    vSend = (double*)malloc(sizeof(double) * vIndexSize);
    vecValues = (double*) malloc(sizeof(double) * recvSize);

    free(vBuf);
}

COOMatrix::~COOMatrix() {
    free(vElement);
    free(vElementRep);
    free(recvCount);
    free(vIndex);
    free(vIndexCount);
    free(vSend);
    free(vecValues);
}

void COOMatrix::matvec(double* v, double* w) {

    // put the values of the vector in vSend, for sending to other processors
    // to change the index from global to local: vIndex[i]-split[rank]
    for(long i=0;i<vIndexSize;i++)
        vSend[i] = v[( vIndex[i]-split[rank] )];

    // send out values of the vector
    MPI_Alltoallv(vSend, vIndexCount, &*(vdispls.begin()), MPI_DOUBLE, vecValues, recvCount, &*(rdispls.begin()), MPI_DOUBLE, MPI_COMM_WORLD);

    // compute matvec w = B * v
    // "values" is the array of value of the entries from matrix B
    // to change the index from global to local: row[iter] - split[rank]

    fill(&w[0], &w[M], 0);
    long iter = 0;
    for (long i=0; i<vElementSize; i++){
        for (long j=0; j<vElementRep[i]; j++) {
            w[row[iter] - split[rank]] += values[iter] * vecValues[i];
            iter++;
        }
    }
}

// *******************************************
// definition of the print functions
// *******************************************

void COOMatrix::valprint(){
    cout << "val:" << endl;
    for(long i=0;i<nnz_l;i++) {
        cout << values[i] << endl;
    }
}

void COOMatrix::rowprint(){
    cout << "row:" << endl;
    for(long i=0;i<nnz_l;i++) {
        cout << row[i] << endl;
    }
}

void COOMatrix::colprint(){
    cout << endl << "col:" << endl;
    for(long i=0;i<nnz_l;i++) {
        cout << col[i] << endl;
    }
}

void COOMatrix::vElementprint(){
    cout << endl << "vElement:" << endl;
    for(long i=0;i<vElementSize;i++) {
        cout << vElement[i] << endl;
    }
}

void COOMatrix::vElementRepprint(){
    cout << endl << "vElementRep:" << endl;
    for(long i=0;i<vElementSize;i++) {
        cout << vElementRep[i] << endl;
    }
}

void COOMatrix::print(){
    cout << endl << "triple:" << endl;
    for(long i=0;i<nnz_l;i++) {
        cout << "(" << row[i] << " , " << col[i] << " , " << values[i] << ")" << endl;
    }
}

/*int COOMatrix::findProcess(long a, int procNum, int p) {
    while(procNum < p){
        if (a >= procNum*M && a < (procNum+1)*M)
            return procNum;

        procNum++;
    }
    return procNum;
}*/
