#include <fstream>
#include <algorithm>
#include <sys/stat.h>
#include "coomatrix.h"
//#include <math.h>
#include "mpi.h"
#include <omp.h>

#define ITERATIONS 100

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

    int mpiopen = MPI_File_open(MPI_COMM_WORLD, Aname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if(mpiopen){
        if (rank==0) cout << "Unable to open the matrix file!" << endl;
        MPI_Finalize();
    }

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
    } else{ // nprocs > Mbig
        // it may be better to set nprocs=Mbig and work only with the first Mbig processors.
        if(rank == 0)
            cout << "number of tasks cannot be greater than te number of rows of the matrix." << endl;
        MPI_Finalize();
    }

/*    if (rank==0)
        cout << "n_buckets = " << n_buckets << ", Mbig = " << Mbig << endl;*/

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
    //cout << "rank=" << rank << ", nnz_l = " << nnz_l << endl;

    values.resize(nnz_l);
    row.resize(nnz_l);
    col.resize(nnz_l);

    double* valuesP = &(*(values.begin()));
    long* rowP = &(*(row.begin()));
    long* colP = &(*(col.begin()));

    MPI_Alltoallv(sendBufI, sendSizeArray, sOffset, MPI_LONG, rowP, recvSizeArray, rOffset, MPI_LONG, MPI_COMM_WORLD);
    MPI_Alltoallv(sendBufJ, sendSizeArray, sOffset, MPI_LONG, colP, recvSizeArray, rOffset, MPI_LONG, MPI_COMM_WORLD);
    MPI_Alltoallv(sendBufV, sendSizeArray, sOffset, MPI_DOUBLE, valuesP, recvSizeArray, rOffset, MPI_DOUBLE, MPI_COMM_WORLD);

    // setting the number of rows for each processor
    M = split[rank+1] - split[rank];

/*    if (rank==0){
        //colprint();
        cout << "M = " << M << endl;
        //cout << "nnz_l = " << nnz_l << endl;
        //valprint();
        //print();
    }*/

/*    //vElement = (long*) malloc(sizeof(long) * nnz_l); //the size of this, which is nnz_l, will be less, exactly: vElementSize
    vElement_local = (long*) malloc(sizeof(long) * nnz_l); //the size of this, which is nnz_l, will be less, exactly: vElementSize_local
    vElement_remote = (long*) malloc(sizeof(long) * nnz_l); //the size of this, which is nnz_l, will be less, exactly: vElementSize_remote
    //vElementRep = (long*) malloc(sizeof(long) * nnz_l); //the size of this, which is nnz_l, will be less, exactly: vElementSize
    vElementRep_local = (long*) malloc(sizeof(long) * nnz_l); //the size of this, which is nnz_l, will be less, exactly: vElementSize_local
    vElementRep_remote = (long*) malloc(sizeof(long) * nnz_l); //the size of this, which is nnz_l, will be less, exactly: vElementSize_remote
    //vElementSize = 0;
    //vElementSize_local = 0;
    //vElementSize_remote = 0;*/
    recvCount = (int*)malloc(sizeof(long)*nprocs);
    std::fill(recvCount, recvCount + nprocs, 0);
    //std::fill(vElementRep, vElementRep + nnz_l, 0);

    // take care of the first element here, since there is "col[i-1]" in the for loop below, so "i" cannot start from 0.
    if (col[0] >= split[rank] && col[0] < split[rank + 1]) {
        values_local.push_back(values[0]);
        row_local.push_back(row[0]);
        col_local.push_back(col[0]);

        vElement_local.push_back(col[0]);
        vElementRep_local.push_back(1);
    } else{
        values_remote.push_back(values[0]);
        row_remote.push_back(row[0]);
        col_remote.push_back(col[0]);

        vElement_remote.push_back(col[0]);
        vElementRep_remote.push_back(1);
        recvCount[lower_bound2(&split[0], &split[nprocs], col[0])] = 1;
    }

    //bool temp;
    for (long i = 1; i < nnz_l; i++) {

/*        temp = (col[i] != col[i-1]);

        vElementRep[vElementSize-1] += (1-temp);        // vElementRep[vElementSize-1]++;

        vElement[vElementSize] = col[i];
        vElementRep[vElementSize] += temp;
        vElementSize += temp;

        procNum = lower_bound2(&split[0], &split[nprocs], col[i]);
        recvCount[procNum] += temp;*/

        // The following "if statement" is equivalent to the above part. Computing the temp value is the same as checking the if statement. Which one is better?

        if (col[i] >= split[rank] && col[i] < split[rank+1]) {

            values_local.push_back(values[i]);
            row_local.push_back(row[i]);
            col_local.push_back(col[i]);

            if (col[i] == col[i - 1]) {
                (*(vElementRep_local.end()-1))++;
            } else {
                vElement_local.push_back(col[i]);
                vElementRep_local.push_back(1);
            }
        } else {
            values_remote.push_back(values[i]);
            row_remote.push_back(row[i]);
            col_remote.push_back(col[i]);

            if (col[i] == col[i - 1]) {
                (*(vElementRep_remote.end()-1))++;
            } else {
                vElement_remote.push_back(col[i]);
                vElementRep_remote.push_back(1);

                procNum = lower_bound2(&split[0], &split[nprocs], col[i]);
                recvCount[procNum]++;
            }
        }
    } // for i

/*    if (rank==0){
        cout << "values_local.size()=" << values_local.size() << ", rank=" << rank << endl;
        for(int i=0; i<values_local.size(); i++)
            cout << values_local[i] << endl;
    }*/

/*    if (rank==0){
        vElementRepprint_remote();
        cout << "values_local: rank=" << rank << endl;
        for(int i=0; i<values_local.size(); i++)
            cout << i << "\trow=" << row_local[i] <<", col=" << col_local[i] << ", val=" << values_local[i] << endl;
    }*/

    // don't receive anything from yourself
    recvCount[rank] = 0;

    // set vElementSize_local and vElementSize_remote
    //vElementSize_local = recvCount[rank];
    //vElementSize_remote = vElementSize - vElementSize_local;

/*    if (rank==0){
        cout << "rank=" << rank << ", vElement_local.size() = " << vElement_local.size() << endl;
        cout << "rank=" << rank << ", vElement_remote.size() = " << vElement_remote.size() << endl;

        vElementprint_local();
        vElementRepprint_local();
        vElementprint_remote();
        vElementRepprint_remote();

        cout << "recvCount: rank=" << rank  << endl;
        for(int i=0; i<nprocs; i++)
            cout << recvCount[i] << endl;
    }*/

    vIndexCount = (int*)malloc(sizeof(int)*nprocs);
    MPI_Alltoall(recvCount, 1, MPI_INT, vIndexCount, 1, MPI_INT, MPI_COMM_WORLD);

/*    if (rank==1){
        cout << "vIndexCount: rank=" << rank << endl;
        for(int i=0; i<nprocs; i++)
            cout << vIndexCount[i] << endl;
    }*/

    /*    int vIndexSize = 0;
    for (int i=0; i<nprocs; i++)
        vIndexSize += vIndexCount[i];*/

    numRecvProc = 0;
    numSendProc = 0;
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

/*    if (rank==0){
        cout << "rank=" << rank << ", numRecvProc=" << numRecvProc << ", numSendProc=" << numSendProc << endl;
    }*/

//    long* vBuf = (long*)malloc(sizeof(long)*nnz_l);
//    for (long i=0; i<vElementSize_remote; i++)
//        vBuf[i] = vElement[i]%M;

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
    MPI_Alltoallv(&(*(vElement_remote.begin())), recvCount, &*(rdispls.begin()), MPI_LONG, vIndex, vIndexCount, &(*(vdispls.begin())), MPI_LONG, MPI_COMM_WORLD);

/*    if (rank==0){
        cout << "vIndex: rank=" << rank  << endl;
        for(int i=0; i<vIndexSize; i++)
            cout << vIndex[i] << endl;
    }*/

    // change the indices from global to local
    for (unsigned int i=0; i<vIndexSize; i++)
        vIndex[i] -= split[rank];

    // change the indices from global to local
    for (unsigned int i=0; i<row_local.size(); i++)
        row_local[i] -= split[rank];
    for (unsigned int i=0; i<row_remote.size(); i++)
        row_remote[i] -= split[rank];

    // vSend = vector values to send to other procs
    // vecValues = vector values that received from other procs
    // These will be used in matvec and they are set here to reduce the time of matvec.
    vSend = (double*)malloc(sizeof(double) * vIndexSize);
    vecValues = (double*) malloc(sizeof(double) * recvSize);
    //free(vBuf);
}

COOMatrix::~COOMatrix() {
    //free(vElement);
    //free(vElementRep);
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
        vSend[i] = v[( vIndex[i] )];

/*    if (rank==0){
        cout << "vIndexSize=" << vIndexSize << ", vSend: rank=" << rank << endl;
        for(int i=0; i<vIndexSize; i++)
            cout << vSend[i] << endl;
    }*/

    // There are at least two ways to store the vector values that we need to to matvec:
    // 1. Have one vector, called vecValues, and use that for both local and remote multiplication.
    // 2. Create two vectors, vecValues_local and vecValues_remote.

    // vElementSize_local = M;

    // iSend your data, and iRecv from others
    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses = new MPI_Status[numSendProc+numRecvProc];

    //First place all recv requests. Do not recv from self.
    for(int i = 0; i < numRecvProc; i++) {
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, MPI_COMM_WORLD, &(requests[i]));
        //Mpi_Irecv( &(recvbuf[rdispls[i]]) , recvcnts[i], i, 1, comm, &(requests[i]) );
    }

    //Next send the messages. Do not send to self.
    for(int i = 0; i < numSendProc; i++) {
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, MPI_COMM_WORLD, &(requests[numRecvProc+i]));
        //par::Mpi_Issend<T>( &(sendbuf[sdispls[i]]), sendcnts[i], i, 1, comm, &(requests[numRecvProc+i]) );
    }

/*    if (rank==0){
        cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << endl;
        for(int i=0; i<recvSize; i++)
            cout << vecValues[i] << endl;
    }*/

/*    if (rank==0){
        cout << "vElement_local.size()=" << vElement_local.size() << ", M=" << M << ", vElement_local: rank=" << rank << endl;
        for(int i=0; i<M; i++)
            cout << v[i] << endl;
    }*/

/*    if (rank==0){
        vElementRepprint_remote();
        cout << "values_local: rank=" << rank << endl;
        for(int i=0; i<values_local.size(); i++)
            cout << i << "\trow=" << row_local[i] <<", col=" << col_local[i] << ", val=" << values_local[i] << endl;
    }*/

    // delete vElement_local in the setup phase. it is just v, since v is stored locally on each proc.
    // Also, delete values_local. Instead, find the starting index of values which correspond to the vector values which are stored locally. Then, use values[starting index].

    fill(&w[0], &w[M], 0);
    unsigned int iter = 0;
    // local loop first
    for (unsigned int i=0; i<M; ++i) {
        for (unsigned int j=0; j<vElementRep_local[i]; ++j,++iter) {
            //if(rank==0) cout << "row_local[iter]=" << row_local[iter] << ", values_local[iter]=" << values_local[iter] << ", v[i]=" << v[i] << endl;
            w[row_local[iter]] += values_local[iter] * v[i];
        }
    }

    // Wait for comm to finish.
    MPI_Waitall(numSendProc+numRecvProc, requests, statuses);

//    cout << "rank=" << rank << ", values_remote.size()=" << values_remote.size() << endl;
/*    if(rank==0){
        cout << "rank=" << rank << ", values_remote.size()=" << values_remote.size() << endl;
        for (int i=0; i<vElementRep_remote.size(); i++){
            cout << vElementRep_remote[i] << endl;
        }
    }*/

    iter = 0;
    // remote loop
    for (unsigned int i=0; i<recvSize; ++i) {
        for (unsigned int j=0; j<vElementRep_remote[i]; ++j, iter++) {
            //if(rank==0) cout << "iter=" << iter << ", vElementRep_remote[i]=" << vElementRep_remote[i] << endl;
            w[row_remote[iter]] += values_remote[iter] * vecValues[i];
        }
    }




    // send out values of the vector
    //MPI_Alltoallv(vSend, vIndexCount, &*(vdispls.begin()), MPI_DOUBLE, vecValues, recvCount, &*(rdispls.begin()), MPI_DOUBLE, MPI_COMM_WORLD);

    // compute matvec w = B * v
    // "values" is the array of value of the entries from matrix B
    // to change the index from global to local: row[iter] - split[rank]

/*    fill(&w[0], &w[M], 0);
    long iter = 0;
    for (long i=0; i<vElementSize; i++){
        for (long j=0; j<vElementRep[i]; j++, iter++) {
            w[row[iter] - split[rank]] += values[iter] * vecValues[i];
            //iter++;
        }
    }*/
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

void COOMatrix::vElementprint_local(){
    cout << endl << "vElement_local:" << endl;
    for(std::vector<long>::iterator it = vElement_local.begin() ; it != vElement_local.end(); ++it) {
        cout << *it << endl;
    }
}
void COOMatrix::vElementprint_remote(){
    cout << endl << "vElement_remote:" << endl;
    for(std::vector<long>::iterator it = vElement_remote.begin() ; it != vElement_remote.end(); ++it) {
        cout << *it << endl;
    }
}

void COOMatrix::vElementRepprint_local(){
    cout << endl << "vElementRep_local:" << endl;
    for(std::vector<long>::iterator it = vElementRep_local.begin() ; it != vElementRep_local.end(); ++it) {
        cout << *it << endl;
    }
}
void COOMatrix::vElementRepprint_remote(){
    cout << endl << "vElementRep_remote:" << endl;
    for(std::vector<long>::iterator it = vElementRep_remote.begin() ; it != vElementRep_remote.end(); ++it) {
        cout << *it << endl;
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
