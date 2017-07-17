#include <fstream>
#include <algorithm>
#include <sys/stat.h>
#include <string.h>
#include "mpi.h"
#include <omp.h>
#include "coomatrix.h"
#include "auxFunctions.h"


COOMatrix::COOMatrix() {

}


COOMatrix::COOMatrix(char* Aname, unsigned int Mbig2, MPI_Comm comm) {
    // the following variables of coomatrix class will be set in this function:
    // Mbig", "nnz_g", "initial_nnz_l", "data"
    // "data" is only required for repartition function.

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // set Mbig in the class
    Mbig = Mbig2;

    // find number of general nonzeros of the input matrix
    struct stat st;
    stat(Aname, &st);
    // 2*sizeof(long)+sizeof(double) = 24
    nnz_g = st.st_size / 24;

    // find initial local nonzero
    initial_nnz_l = (unsigned int) (floor(1.0 * nnz_g / nprocs)); // initial local nnz
    if (rank == nprocs - 1)
        initial_nnz_l = nnz_g - (nprocs - 1) * initial_nnz_l;

    //    if(rank==0) cout << "nnz_g = " << nnz_g << ", initial_nnz_l = " << initial_nnz_l << endl;

    // todo: change data from vector to malloc. then free after repartitioning.
    data.resize(3 * initial_nnz_l); // 3 is for i and j and val
    unsigned long* datap = &(*(data.begin()));

    // *************************** read the matrix ****************************

    MPI_Status status;
    MPI_File fh;
    MPI_Offset offset;

    int mpiopen = MPI_File_open(comm, Aname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (mpiopen) {
        if (rank == 0) cout << "Unable to open the matrix file!" << endl;
        MPI_Finalize();
    }

    //offset = rank * initial_nnz_l * 24; // row index(long=8) + column index(long=8) + value(double=8) = 24
    // the offset for the last process will be wrong if you use the above formula,
    // because initial_nnz_l of the last process will be used, instead of the initial_nnz_l of the other processes.

    offset = rank * (unsigned int) (floor(1.0 * nnz_g / nprocs)) * 24; // row index(long=8) + column index(long=8) + value(double=8) = 24

    MPI_File_read_at(fh, offset, datap, 3 * initial_nnz_l, MPI_UNSIGNED_LONG, &status);

//    double val;
//    if(rank==0)
//        for(long i=0; i<initial_nnz_l; i++){
//            val = data[3*i+2];
//            cout << datap[3*i] << "\t" << datap[3*i+1] << "\t" << val << endl;
//        }

//    int count;
//    MPI_Get_count(&status, MPI_UNSIGNED_LONG, &count);
    //printf("process %d read %d lines of triples\n", rank, count);
    MPI_File_close(&fh);

} //COOMatrix::COOMatrix


COOMatrix::~COOMatrix() {
    if(freeBoolean){
        free(vIndex);
        free(vSend);
        free(vSendULong);
        free(vecValues);
        free(vecValuesULong);
        free(iter_local_array);
        free(iter_remote_array);
        free(indicesP_local);
        free(indicesP_remote);
//    free(vIndexCount);
//    free(vIndexCount);
//    free(indicesP);
//        printf("**********~COOMatrix!!!!!!! \n");
    }
}


int COOMatrix::repartition(MPI_Comm comm){
    // before using this function these variables of coomatrix should be set:
    // Mbig", "nnz_g", "initial_nnz_l", "data"

    // the following variables of coomatrix class will be set in this function:
    // "nnz_l", "M", "split", "entry"

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // *************************** find splitters ****************************
    // split the matrix row-wise by splitters, so each processor get almost equal number of nonzeros

    // definition of buckets: bucket[i] = [ firstSplit[i] , firstSplit[i+1] ). Number of buckets = n_buckets
    int n_buckets = 0;

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
            cout << "number of tasks cannot be greater than the number of rows of the matrix." << endl;
        MPI_Finalize();
    }

//    if (rank==0) cout << "n_buckets = " << n_buckets << ", Mbig = " << Mbig << endl;

    std::vector<int> splitOffset;
    splitOffset.resize(n_buckets);
    int baseOffset = int(floor(1.0*Mbig/n_buckets));
    float offsetRes = float(1.0*Mbig/n_buckets) - baseOffset;
//    if (rank==0) cout << "baseOffset = " << baseOffset << ", offsetRes = " << offsetRes << endl;
    float offsetResSum = 0;
    splitOffset[0] = 0;
    for(unsigned int i=1; i<n_buckets; i++){
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

    unsigned long* firstSplit = (unsigned long*)malloc(sizeof(unsigned long)*(n_buckets+1));
    firstSplit[0] = 0;
    for(unsigned int i=1; i<n_buckets; i++){
        firstSplit[i] = firstSplit[i-1] + splitOffset[i];
    }
    firstSplit[n_buckets] = Mbig;

    splitOffset.clear();

/*    if (rank==0){
        cout << "firstSplit:" << endl;
        for(long i=0; i<n_buckets+1; i++)
            cout << firstSplit[i] << endl;
    }*/

    long* H_l = (long*)malloc(sizeof(long)*n_buckets);
    fill(&H_l[0], &H_l[n_buckets], 0);

    for(unsigned int i=0; i<initial_nnz_l; i++)
        H_l[lower_bound2(&firstSplit[0], &firstSplit[n_buckets], data[3*i])]++;

/*    if (rank==0){
        cout << "initial_nnz_l = " << initial_nnz_l << endl;
        cout << "local histogram:" << endl;
        for(unsigned int i=0; i<n_buckets; i++)
            cout << H_l[i] << endl;
    }*/

    long* H_g = (long*)malloc(sizeof(long)*n_buckets);
    MPI_Allreduce(H_l, H_g, n_buckets, MPI_LONG, MPI_SUM, comm);

    free(H_l);

/*    if (rank==1){
        cout << "global histogram:" << endl;
        for(unsigned int i=0; i<n_buckets; i++){
            cout << H_g[i] << endl;
        }
    }*/

//    long H_g_scan[n_buckets];
    long* H_g_scan = (long*)malloc(sizeof(long)*n_buckets);
    H_g_scan[0] = H_g[0];
    for (unsigned int i=1; i<n_buckets; i++)
        H_g_scan[i] = H_g[i] + H_g_scan[i-1];

    free(H_g);

/*    if (rank==0){
        cout << "scan of global histogram:" << endl;
        for(unsigned int i=0; i<n_buckets; i++)
            cout << H_g_scan[i] << endl;
    }*/

    long procNum = 0;
    split.resize(nprocs+1);
    split[0]=0;
    for (unsigned int i=1; i<n_buckets; i++){
        //if (rank==0) cout << "(procNum+1)*nnz_g/nprocs = " << (procNum+1)*nnz_g/nprocs << endl;
        if (H_g_scan[i] > ((procNum+1)*nnz_g/nprocs)){
            procNum++;
            split[procNum] = firstSplit[i];
        }
    }
    split[nprocs] = Mbig;

    free(H_g_scan);
    free(firstSplit);

//    if (rank==0){
//        cout << endl << "split:" << endl;
//        for(unsigned int i=0; i<nprocs+1; i++)
//            cout << split[i] << endl;}

    // set the number of rows for each process
    M = split[rank+1] - split[rank];

    // *************************** exchange data ****************************

    long tempIndex;
//    int sendSizeArray[nprocs];
    int* sendSizeArray = (int*)malloc(sizeof(int)*nprocs);
    fill(&sendSizeArray[0], &sendSizeArray[nprocs], 0);
    for (unsigned int i=0; i<initial_nnz_l; i++){
        tempIndex = lower_bound2(&split[0], &split[nprocs], data[3*i]);
        sendSizeArray[tempIndex]++;
    }

/*    if (rank==0){
        cout << "sendSizeArray:" << endl;
        for(long i=0;i<nprocs;i++)
            cout << sendSizeArray[i] << endl;
    }*/

//    int recvSizeArray[nprocs];
    int* recvSizeArray = (int*)malloc(sizeof(int)*nprocs);
    MPI_Alltoall(sendSizeArray, 1, MPI_INT, recvSizeArray, 1, MPI_INT, comm);

/*    if (rank==0){
        cout << "recvSizeArray:" << endl;
        for(long i=0;i<nprocs;i++)
            cout << recvSizeArray[i] << endl;
    }*/

//    int sOffset[nprocs];
    int* sOffset = (int*)malloc(sizeof(int)*nprocs);
    sOffset[0] = 0;
    for (int i=1; i<nprocs; i++)
        sOffset[i] = sendSizeArray[i-1] + sOffset[i-1];

/*    if (rank==0){
        cout << "sOffset:" << endl;
        for(long i=0;i<nprocs;i++)
            cout << sOffset[i] << endl;
    }*/

//    int rOffset[nprocs];
    int* rOffset = (int*)malloc(sizeof(int)*nprocs);
    rOffset[0] = 0;
    for (int i=1; i<nprocs; i++)
        rOffset[i] = recvSizeArray[i-1] + rOffset[i-1];

/*    if (rank==0){
        cout << "rOffset:" << endl;
        for(long i=0;i<nprocs;i++)
            cout << rOffset[i] << endl;
    }*/

    long procOwner;
    unsigned int bufTemp;
    cooEntry* sendBuf = (cooEntry*)malloc(sizeof(cooEntry)*initial_nnz_l);
    unsigned int* sIndex = (unsigned int*)malloc(sizeof(unsigned int)*nprocs);
    fill(&sIndex[0], &sIndex[nprocs], 0);

    // memcpy(sendBuf, data.data(), initial_nnz_l*3*sizeof(unsigned long));

    // todo: try to avoid this for loop.
    for (long i=0; i<initial_nnz_l; i++){
        procOwner = lower_bound2(&split[0], &split[nprocs], data[3*i]);
        bufTemp = sOffset[procOwner]+sIndex[procOwner];
        memcpy(sendBuf+bufTemp, data.data() + 3*i, sizeof(cooEntry));
        // todo: the above line is better than the following three lines. think why it works.
//        sendBuf[bufTemp].row = data[3*i];
//        sendBuf[bufTemp].col = data[3*i+1];
//        sendBuf[bufTemp].val = data[3*i+2];
//        if(rank==1) cout << sendBuf[bufTemp].row << "\t" << sendBuf[bufTemp].col << "\t" << sendBuf[bufTemp].val << endl;
        sIndex[procOwner]++;
    }

    free(sIndex);

//    if (rank==1){
//        cout << "sendBufJ:" << endl;
//        for (long i=0; i<initial_nnz_l; i++)
//            cout << sendBufJ[i] << endl;
//    }

    nnz_l = rOffset[nprocs-1] + recvSizeArray[nprocs-1];
//    cout << "rank=" << rank << ", nnz_l = " << nnz_l << endl;

//    cooEntry* entry = (cooEntry*)malloc(sizeof(cooEntry)*nnz_l);
//    cooEntry* entryP = &entry[0];
    entry.resize(nnz_l);

    MPI_Alltoallv(sendBuf, sendSizeArray, sOffset, cooEntry::mpi_datatype(), &entry[0], recvSizeArray, rOffset, cooEntry::mpi_datatype(), comm);

    free(sendSizeArray);
    free(recvSizeArray);
    free(sOffset);
    free(rOffset);
    free(sendBuf);

//    if (rank==1){
//        cout << "nnz_l = " << nnz_l << endl;
//        for (int i=0; i<nnz_l; i++)
//            cout << "i=" << i << "\t" << entry[i].row << "\t" << entry[i].col << "\t" << entry[i].val << endl;}

    return 0;
}


int COOMatrix::matrixSetup(MPI_Comm comm){
    // before using this function these variables of coomatrix should be set:
    // "Mbig", "M", "nnz_g", "split", "entry",

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    freeBoolean = true; // use this parameter to know if deconstructor for COOMatrix class should free the variables or not.

    // *************************** set the inverse of diagonal of A (for smoothers) ****************************

    invDiag.resize(M);
    double* invDiag_p = &(*(invDiag.begin()));
    inverseDiag(invDiag_p, comm);

/*    if(rank==1){
        for(unsigned int i=0; i<M; i++)
            cout << i << ":\t" << invDiag[i] << endl;
    }*/

    // *************************** set and exchange local and remote elements ****************************
    // local elements are elements that correspond to vector elements which are local to this process,
    // and, remote elements correspond to vector elements which should be received from another processes

    col_remote_size = 0;
    nnz_l_local = 0;
    nnz_l_remote = 0;
    int* recvCount = (int*)malloc(sizeof(int)*nprocs);
    std::fill(recvCount, recvCount + nprocs, 0);
//    nnzPerRow.assign(M,0);
    nnzPerRow_local.assign(M,0);
//    nnzPerCol_local.assign(Mbig,0); // todo: Nbig = Mbig, assuming A is symmetric.
//    nnzPerCol_remote.assign(M,0);

    // take care of the first element here, since there is "col[i-1]" in the for loop below, so "i" cannot start from 0.
//    nnzPerRow[row[0]-split[rank]]++;
    long procNum;
    if (entry[0].col >= split[rank] && entry[0].col < split[rank + 1]) {
        nnzPerRow_local[entry[0].row-split[rank]]++;
//        nnzPerCol_local[col[0]]++;
        nnz_l_local++;

        values_local.push_back(entry[0].val);
        row_local.push_back(entry[0].row);
        col_local.push_back(entry[0].col);

        //vElement_local.push_back(col[0]);
        vElementRep_local.push_back(1);

    } else{
        nnz_l_remote++;
//        nnzPerRow_remote[row[0]-split[rank]]++;

        values_remote.push_back(entry[0].val);
        row_remote.push_back(entry[0].row);
        col_remote_size++;
        col_remote.push_back(col_remote_size-1);
        col_remote2.push_back(entry[0].col);
//        nnzPerCol_remote[col_remote_size]++;
        nnzPerCol_remote.push_back(1);

        vElement_remote.push_back(entry[0].col);
        vElementRep_remote.push_back(1);
        recvCount[lower_bound2(&split[0], &split[nprocs], entry[0].col)] = 1;
    }

    for (long i = 1; i < nnz_l; i++) {
//        nnzPerRow[row[i]-split[rank]]++;
        if (entry[i].col >= split[rank] && entry[i].col < split[rank+1]) {
//            nnzPerCol_local[col[i]]++;
            nnz_l_local++;
            nnzPerRow_local[entry[i].row-split[rank]]++;

            values_local.push_back(entry[i].val);
            row_local.push_back(entry[i].row);
            col_local.push_back(entry[i].col);

            if (entry[i].col != entry[i-1].col) {
                vElementRep_local.push_back(1);
            } else {
                (*(vElementRep_local.end()-1))++;
            }
        } else {
            nnz_l_remote++;
//            nnzPerRow_remote[row[i]-split[rank]]++;

            values_remote.push_back(entry[i].val);
            row_remote.push_back(entry[i].row);
            // col_remote2 is the original col value and will be used in making strength matrix. col_remote will be used for matevec.
            col_remote2.push_back(entry[i].col);

            if (entry[i].col != entry[i-1].col) {
                col_remote_size++;
                vElement_remote.push_back(entry[i].col);
                vElementRep_remote.push_back(1);
                procNum = lower_bound2(&split[0], &split[nprocs], entry[i].col);
                recvCount[procNum]++;
                nnzPerCol_remote.push_back(1);
            } else {
                (*(vElementRep_remote.end()-1))++;
                (*(nnzPerCol_remote.end()-1))++;
            }
            // the original col values are not being used. the ordering starts from 0, and goes up by 1.
            col_remote.push_back(col_remote_size-1);
//            nnzPerCol_remote[col_remote_size]++;
        }
    } // for i

    // don't receive anything from yourself
    recvCount[rank] = 0;

/*    MPI_Barrier(comm);
    if (rank==2){
        cout << "recvCount: rank=" << rank << endl;
        for(int i=0; i<nprocs; i++)
            cout << i << "= " << recvCount[i] << endl;
    }*/

    int* vIndexCount = (int*)malloc(sizeof(int)*nprocs);
    MPI_Alltoall(recvCount, 1, MPI_INT, vIndexCount, 1, MPI_INT, comm);

/*    MPI_Barrier(comm);
    if (rank==2){
        cout << "vIndexCount: rank=" << rank << endl;
        for(int i=0; i<nprocs; i++)
            cout << i << "= " << vIndexCount[i] << endl;
    }*/

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

//    if (rank==0) cout << "rank=" << rank << ", numRecvProc=" << numRecvProc << ", numSendProc=" << numSendProc << endl;

    vdispls.resize(nprocs);
    rdispls.resize(nprocs);
    vdispls[0] = 0;
    rdispls[0] = 0;

    for (int i=1; i<nprocs; i++){
        vdispls[i] = vdispls[i-1] + vIndexCount[i-1];
        rdispls[i] = rdispls[i-1] + recvCount[i-1];
    }
    vIndexSize = vdispls[nprocs-1] + vIndexCount[nprocs-1];
    recvSize   = rdispls[nprocs-1] + recvCount[nprocs-1];

    vIndex = (long*)malloc(sizeof(long)*vIndexSize);
    MPI_Alltoallv(&(*(vElement_remote.begin())), recvCount, &*(rdispls.begin()), MPI_LONG, vIndex, vIndexCount, &(*(vdispls.begin())), MPI_LONG, comm);

    free(recvCount);
    free(vIndexCount);

/*    if (rank==0){
        cout << "vIndex: rank=" << rank  << endl;
        for(int i=0; i<vIndexSize; i++)
            cout << vIndex[i] << endl;
    }*/

    // change the indices from global to local
    for (unsigned int i=0; i<vIndexSize; i++)
        vIndex[i] -= split[rank];
    for (unsigned int i=0; i<row_local.size(); i++)
        row_local[i] -= split[rank];
    for (unsigned int i=0; i<row_remote.size(); i++)
        row_remote[i] -= split[rank];

    // vSend = vector values to send to other procs
    // vecValues = vector values that received from other procs
    // These will be used in matvec and they are set here to reduce the time of matvec.
    vSend     = (double*)malloc(sizeof(double) * vIndexSize);
    vecValues = (double*)malloc(sizeof(double) * recvSize);

    vSendULong     = (unsigned long*)malloc(sizeof(unsigned long) * vIndexSize);
    vecValuesULong = (unsigned long*)malloc(sizeof(unsigned long) * recvSize);

    // *************************** find start and end of each thread for matvec ****************************
    // also, find nnz per row for local and remote matvec

#pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }

    iter_local_array = (unsigned int *)malloc(sizeof(unsigned int )*(num_threads+1));
    iter_remote_array = (unsigned int *)malloc(sizeof(unsigned int )*(num_threads+1));
#pragma omp parallel
    {

        const int thread_id = omp_get_thread_num();
//        if(rank==0 && thread_id==0) cout << "number of procs = " << nprocs << ", number of threads = " << num_threads << endl;
        unsigned int istart = 0;
        unsigned int iend = 0;
        unsigned int iter_local, iter_remote;

        // compute local iter to do matvec using openmp (it is done to make iter independent data on threads)
        int index=0;
#pragma omp for
        for (unsigned int i = 0; i < M; ++i) {
            if(index==0){
                istart = i;
                index++;
                iend = istart;
            }
            iend++;
        }

        iter_local = 0;
        for (unsigned int i = istart; i < iend; ++i)
            iter_local += nnzPerRow_local[i];

        iter_local_array[0] = 0;
        iter_local_array[thread_id+1] = iter_local;

        // compute remote iter to do matvec using openmp (it is done to make iter independent data on threads)
        index=0;
#pragma omp for
        for (unsigned int i = 0; i < col_remote_size; ++i) {
            if(index==0){
                istart = i;
                index++;
                iend = istart;
            }
            iend++;
        }

        iter_remote = 0;
        if(nnzPerCol_remote.size() != 0){
            for (unsigned int i = istart; i < iend; ++i)
                iter_remote += nnzPerCol_remote[i];
        }

        iter_remote_array[0] = 0;
        iter_remote_array[thread_id+1] = iter_remote;

/*        if (rank==1 && thread_id==0){
            cout << "M=" << M << endl;
            cout << "recvSize=" << recvSize << endl;
            cout << "istart=" << istart << endl;
            cout << "iend=" << iend << endl;
            cout  << "nnz_l=" << nnz_l << ", iter_remote=" << iter_remote << ", iter_local=" << iter_local << endl;
        }*/
    }

    //scan of iter_local_array
    for(int i=1; i<num_threads+1; i++)
        iter_local_array[i] += iter_local_array[i-1];

    //scan of iter_remote_array
    for(int i=1; i<num_threads+1; i++)
        iter_remote_array[i] += iter_remote_array[i-1];

/*    if (rank==0){
        cout << "iter_local_array:" << endl;
        for(int i=0; i<num_threads+1; i++)
            cout << iter_local_array[i] << endl;
    }*/

/*    if (rank==0){
        cout << "iter_remote_array:" << endl;
        for(int i=0; i<num_threads+1; i++)
            cout << iter_remote_array[i] << endl;
    }*/

    // *************************** find sortings ****************************
    //find the sorting on rows on both local and remote data to be used in matvec

    indicesP_local = (unsigned long*)malloc(sizeof(unsigned long)*nnz_l_local);
    for(unsigned long i=0; i<nnz_l_local; i++)
        indicesP_local[i] = i;
    unsigned long* row_localP = &(*(row_local.begin()));
    std::sort(indicesP_local, &indicesP_local[nnz_l_local], sort_indices(row_localP));

    indicesP_remote = (unsigned long*)malloc(sizeof(unsigned long)*nnz_l_remote);
    for(unsigned long i=0; i<nnz_l_remote; i++)
        indicesP_remote[i] = i;
    unsigned long* row_remoteP = &(*(row_remote.begin()));
    std::sort(indicesP_remote, &indicesP_remote[nnz_l_remote], sort_indices(row_remoteP));

//    indicesP = (int*)malloc(sizeof(int)*nnz_l);
//    for(int i=0; i<nnz_l; i++)
//        indicesP[i] = i;
//    std::sort(indicesP, &indicesP[nnz_l], sort_indices(rowP));

    return 0;
}


int COOMatrix::matvec(double* v, double* w, MPI_Comm comm) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    totalTime = 0;
//    double t10 = MPI_Wtime();

    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i=0;i<vIndexSize;i++)
        vSend[i] = v[( vIndex[i] )];
//    double t20 = MPI_Wtime();
//    time[0] += (t20-t10);

/*    if (rank==0){
        cout << "vIndexSize=" << vIndexSize << ", vSend: rank=" << rank << endl;
        for(int i=0; i<vIndexSize; i++)
            cout << vSend[i] << endl;
    }*/

//    double t13 = MPI_Wtime();
    // iSend your data, and iRecv from others
    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses = new MPI_Status[numSendProc+numRecvProc];

    //First place all recv requests. Do not recv from self.
    for(int i = 0; i < numRecvProc; i++) {
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));
    }

    //Next send the messages. Do not send to self.
    for(int i = 0; i < numSendProc; i++) {
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));
    }

/*    if (rank==0){
        cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << endl;
        for(int i=0; i<recvSize; i++)
            cout << vecValues[i] << endl;
    }*/

//    double t11 = MPI_Wtime();
    // local loop
    fill(&w[0], &w[M], 0);
#pragma omp parallel
    {
        long iter = iter_local_array[omp_get_thread_num()];
#pragma omp for
        for (unsigned int i = 0; i < M; ++i) {
            for (unsigned int j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
                w[i] += values_local[indicesP_local[iter]] * v[col_local[indicesP_local[iter]] - split[rank]];
            }
        }
    }

//    double t21 = MPI_Wtime();
//    time[1] += (t21-t11);

    // Wait for comm to finish.
    MPI_Waitall(numSendProc+numRecvProc, requests, statuses);

/*    if (rank==1){
        cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << endl;
        for(int i=0; i<recvSize; i++)
            cout << vecValues[i] << endl;
    }*/

    // remote loop
//    double t12 = MPI_Wtime();
#pragma omp parallel
    {
        unsigned int iter = iter_remote_array[omp_get_thread_num()];
#pragma omp for
        for (unsigned int i = 0; i < col_remote_size; ++i) {
            for (unsigned int j = 0; j < nnzPerCol_remote[i]; ++j, ++iter) {
                w[row_remote[indicesP_remote[iter]]] += values_remote[indicesP_remote[iter]] * vecValues[col_remote[indicesP_remote[iter]]];
            }
        }
    }

//    double t22 = MPI_Wtime();
//    time[2] += (t22-t12);
//    double t23 = MPI_Wtime();
//    time[3] += (t23-t13);

    return 0;
}


int COOMatrix::inverseDiag(double* x, MPI_Comm comm) {
    int nprocs, rank;
//    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    for(unsigned int i=0; i<nnz_l; i++){
        if(entry[i].row == entry[i].col)
            x[entry[i].row-split[rank]] = 1/entry[i].val;
    }
    return 0;
}


int COOMatrix::jacobi(double* x, double* b, MPI_Comm comm) {

// Ax = b
// x = x - (D^(-1))(Ax - b)
// 1. B.matvec(x, one) --> put the value of matvec in one.
// 2. two = one - b
// 3. three = inverseDiag * two * omega
// 4. four = x - three

    float omega = float(2.0/3);
    unsigned int i;
    // replace allocating and deallocating with a pre-allocated memory.
    double* temp = (double*)malloc(sizeof(double)*M);
    matvec(x, temp, comm);
    for(i=0; i<M; i++){
        temp[i] -= b[i];
        temp[i] *= invDiag[i] * omega;
        x[i] -= temp[i];
    }
    free(temp);
    return 0;
}


int COOMatrix::print(){
    cout << endl << "triple:" << endl;
    for(long i=0;i<nnz_l;i++) {
        cout << "(" << entry[i].row << " , " << entry[i].col << " , " << entry[i].val << ")" << endl;
    }
    return 0;
}


int COOMatrix::SaenaSetup(){
    return 0;
}


int COOMatrix::SaenaSolve(){
    return 0;
}
