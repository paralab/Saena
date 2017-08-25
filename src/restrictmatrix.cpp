#include <iostream>
#include <algorithm>
#include <mpi.h>
#include <prolongmatrix.h>
#include <restrictmatrix.h>
//#include <dtypes.h>

using namespace std;

restrictMatrix::restrictMatrix(){}


int restrictMatrix::transposeP(prolongMatrix* P, MPI_Comm comm) {

    // todo: this matrix is not sorted at the end.
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    unsigned long i, j;

    arrays_defined = true;
    Mbig = P->Nbig;
    Nbig = P->Mbig;
    split = P->split;
    splitNew = P->splitNew;

    // set the number of rows for each process
    M = splitNew[rank+1] - splitNew[rank];
//    if(rank==1) cout << "transposeP: " << Mbig << ", " << Nbig << ", " << M << endl;

    // *********************** check for shrinking ************************
    /*
    bool shrinkLocal;
    bool shrinkGlobal = false;

    double temp1 = 0;
    for(i=1; i<nprocs+1; i++){
        temp1 += ((double)initialNumberOfRows[i] / P->splitNew[i]);
    }
    temp1 /= nprocs;

//    cout << rank << "\t" << temp1 << endl;

    if(temp1 >= 3){ // todo: decide about this later.
        // todo: don't forget to update initialNumberOfRows after shrinking.
        MPI_Group currentGroup;
        MPI_Comm_group(comm, &currentGroup);

        int newGroupSize = nprocs/4;
        int* newGroupRanks = (int*)malloc(sizeof(int)*newGroupSize);
        for(int i=0; i<newGroupSize; i++)
            newGroupRanks[i] = 4*i;

        MPI_Group newGroup;
        MPI_Group_incl(currentGroup, newGroupSize, newGroupRanks, &newGroup);

        MPI_Comm newComm;
        MPI_Comm_create_group(comm, newGroup, 0, &newComm);

        int newRank = -1;
        int newSize = -1;
        if (newComm != MPI_COMM_NULL) {
            MPI_Comm_rank(newComm, &newRank);
            MPI_Comm_size(newComm, &newSize);
//            cout << "rank = " << newRank << ", size = " << newSize << endl;
        }

        MPI_Group_free(&currentGroup);
        MPI_Group_free(&newGroup);
        if (newComm != MPI_COMM_NULL) MPI_Comm_free(&newComm);
        free(newGroupRanks);
    }
*/

    // *********************** send remote part of restriction ************************

//    MPI_Barrier(comm);
//    if(rank==1) cout << "\nvSend_t:" << endl;
    for (i = 0; i < P->nnz_l_remote; i++){ // all remote entries should be sent.
        P->vSend_t[i] = P->entry_remote[i];
        P->vSend_t[i].row += split[rank];
//        if(rank==1) printf("%lu\t %lu\t %f\n", P->entry_remote[i].row, P->entry_remote[i].col, P->entry_remote[i].val);
    }

    MPI_Request *requests = new MPI_Request[P->numSendProc_t + P->numRecvProc_t];
    MPI_Status  *statuses = new MPI_Status[P->numSendProc_t + P->numRecvProc_t];

    for(i = 0; i < P->numRecvProc_t; i++)
        MPI_Irecv(&P->vecValues_t[P->rdispls_t[P->recvProcRank_t[i]]], P->recvProcCount_t[i], cooEntry::mpi_datatype(), P->recvProcRank_t[i], 1, comm, &(requests[i]));

    for(i = 0; i < P->numSendProc_t; i++)
        MPI_Isend(&P->vSend_t[P->vdispls_t[P->sendProcRank_t[i]]], P->sendProcCount_t[i], cooEntry::mpi_datatype(), P->sendProcRank_t[i], 1, comm, &(requests[P->numRecvProc_t+i]));

    // *********************** assign local part of restriction ************************

    unsigned long iter = 0;
    for (i = 0; i < P->M; ++i) {
        for (j = 0; j < P->nnzPerRow_local[i]; ++j, ++iter) {
//            if(rank==1) cout << P->entry_local[P->indicesP_local[iter]].col << "\t" << P->entry_local[P->indicesP_local[iter]].col - P->splitNew[rank]
//                             << "\t" << P->entry_local[P->indicesP_local[iter]].row << "\t" << P->entry_local[P->indicesP_local[iter]].row + P->split[rank]
//                             << "\t" << P->entry_local[P->indicesP_local[iter]].val << endl;
            entry.push_back(cooEntry(P->entry_local[P->indicesP_local[iter]].col - splitNew[rank], // make row index local
                                           P->entry_local[P->indicesP_local[iter]].row + split[rank],    // make col index global
                                           P->entry_local[P->indicesP_local[iter]].val));
        }
    }

//    MPI_Barrier(comm);
//    iter = 0;
//    if(rank==1){
//        cout << endl << "local:" << " rank=" << rank << endl;
//        for (i = 0; i < P->M; ++i)
//            for (j = 0; j < P->nnzPerRow_local[i]; ++j, ++iter)
//                cout << entry[iter].row << "\t" << entry[iter].col << "\t" << entry[iter].val << endl;}
//    MPI_Barrier(comm);

    // *********************** assign remote part of restriction ************************

    MPI_Waitall(P->numSendProc_t + P->numRecvProc_t, requests, statuses);

//    MPI_Barrier(comm);
//    if(rank==1) cout << "vecValues_t:" << endl;
    for(i=0; i<P->recvSize_t; i++){
//        if(rank==1) printf("%lu\t %lu\t %f\n", P->vecValues_t[i].row, P->vecValues_t[i].col, P->vecValues_t[i].val);
//        if(rank==1) printf("%lu\t %lu\t %f\n", P->vecValues_t[i].row, P->vecValues_t[i].col - splitNew[rank], P->vecValues_t[i].val);
        entry.push_back(cooEntry(P->vecValues_t[i].col - splitNew[rank], // make row index local
                                        P->vecValues_t[i].row,
                                        P->vecValues_t[i].val));
    }

    std::sort(entry.begin(), entry.end());

//    MPI_Barrier(comm);
//    if(rank==2){
//        cout << endl << "final after sorting:" << " rank = " << rank << "\tP->recvSize_t = " << P->recvSize_t << endl;
//        for(i=0; i<entry.size(); i++)
//            cout << i << "\t" << entry[i].row << "\t" << entry[i].col << "\t" << entry[i].val << endl;}
//    MPI_Barrier(comm);

    nnz_l = entry.size();
    MPI_Allreduce(&nnz_l, &nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
    // todo: check why is R so imbalanced for 289 size matrix on 8 processors. use the following print function.
//    printf("rank=%d \t R.nnz_l=%lu \t R.nnz_g=%lu \n", rank, nnz_l, nnz_g);

    // *********************** setup matvec ************************

    long procNum;
    col_remote_size = 0; // number of remote columns
    int* recvCount = (int*)malloc(sizeof(int)*nprocs);
    std::fill(recvCount, recvCount + nprocs, 0);
    nnzPerRow_local.assign(M,0);
    nnzPerRowScan_local.assign(M+1, 0);
    nnz_l_local = 0;
    nnz_l_remote = 0;

    // todo: sometimes nnz_l is 0. check if everything is fine.
    if(entry.size() != 0){

        // take care of the first element here, since there is "col[i-1]" in the for loop below, so "i" cannot start from 0.
        // local
        if (entry[0].col >= split[rank] && entry[0].col < split[rank + 1]) {
            nnzPerRow_local[entry[0].row]++;
            entry_local.push_back(entry[0]);
            row_local.push_back(entry[0].row); // only for sorting at the end of prolongMatrix::findLocalRemote. then clear the vector. // todo: clear does not free memory. find a solution.
//        col_local.push_back(entry[0].col);
//        values_local.push_back(entry[0].val);
            //vElement_local.push_back(col[0]);
            vElementRep_local.push_back(1);

            // remote
        } else{
            entry_remote.push_back(entry[0]);
            row_remote.push_back(entry[0].row); // only for sorting at the end of prolongMatrix::findLocalRemote. then clear the vector. // todo: clear does not free memory. find a solution.
//        col_remote2.push_back(entry[0].col);
//        values_remote.push_back(entry[0].val);
            col_remote_size++; // number of remote columns
            col_remote.push_back(col_remote_size-1);
//        nnzPerCol_remote[col_remote_size-1]++;
            nnzPerCol_remote.push_back(1);
            vElement_remote.push_back(entry[0].col);
            vElementRep_remote.push_back(1);
            recvCount[lower_bound2(&split[0], &split[nprocs], entry[0].col)] = 1;
        }

        for (i = 1; i < nnz_l; i++) {

            // local
            if (entry[i].col >= split[rank] && entry[i].col < split[rank+1]) {
                nnzPerRow_local[entry[i].row]++;
                entry_local.push_back(cooEntry(entry[i].row, entry[i].col, entry[i].val));
                row_local.push_back(entry[i].row); // only for sorting at the end of prolongMatrix::findLocalRemote. then erase. // todo: clear does not free memory. find a solution.
//            col_local.push_back(entry[i].col);
//            values_local.push_back(entry[i].val);
                if (entry[i].col != entry[i-1].col)
                    vElementRep_local.push_back(1);
                else
                    (*(vElementRep_local.end()-1))++;

                // remote
            } else {
                entry_remote.push_back(cooEntry(entry[i].row, entry[i].col, entry[i].val));
                row_remote.push_back(entry[i].row); // only for sorting at the end of prolongMatrix::findLocalRemote. then clear the vector. // todo: clear does not free memory. find a solution.
                // col_remote2 is the original col value. col_remote starts from 0.
//            col_remote2.push_back(entry[i].col);
//            values_remote.push_back(entry[i].val);

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
                // the original col values are not being used for matvec. the ordering starts from 0, and goes up by 1.
                col_remote.push_back(col_remote_size-1);
//            nnzPerCol_remote[col_remote_size-1]++;
            }
        } // for i

        nnz_l_local  = entry_local.size();
        nnz_l_remote = entry_remote.size();
//        MPI_Barrier(comm); printf("rank=%d, nnz_l=%lu, nnz_l_local=%lu, nnz_l_remote=%lu \n", rank, nnz_l, nnz_l_local, nnz_l_remote); MPI_Barrier(comm);

        for(i=0; i<M; i++){
            nnzPerRowScan_local[i+1] = nnzPerRowScan_local[i] + nnzPerRow_local[i];
//        if(rank==0) printf("nnzPerRowScan_local=%d, nnzPerRow_local=%d\n", nnzPerRowScan_local[i], nnzPerRow_local[i]);
        }

    } // end of if(entry.size()) != 0

    int* vIndexCount = (int*)malloc(sizeof(int)*nprocs);
    MPI_Alltoall(recvCount, 1, MPI_INT, vIndexCount, 1, MPI_INT, comm);

//    for(int i=0; i<nprocs; i++){
//        MPI_Barrier(comm);
//        if(rank==2) cout << "recieve from proc " << i << "\trecvCount   = " << recvCount[i] << endl;
//        MPI_Barrier(comm);
//        if(rank==2) cout << "send to proc      " << i << "\tvIndexCount = " << vIndexCount[i] << endl;
//    }

    numRecvProc = 0;
    numSendProc = 0;
    for(int i=0; i<nprocs; i++){
        if(recvCount[i]!=0){
            numRecvProc++;
            recvProcRank.push_back(i);
            recvProcCount.push_back(recvCount[i]);
//            sendProcCount_t.push_back(vIndexCount_t[i]); // use recvProcRank for it.
//            if(rank==0) cout << i << "\trecvCount[i] = " << recvCount[i] << "\tvIndexCount_t[i] = " << vIndexCount_t[i] << endl;
        }
        if(vIndexCount[i]!=0){
            numSendProc++;
            sendProcRank.push_back(i);
            sendProcCount.push_back(vIndexCount[i]);
//            recvProcCount_t.push_back(recvCount_t[i]); // use sendProcRank for it.
        }
    }

    //  if (rank==0) cout << "rank=" << rank << ", numRecvProc=" << numRecvProc << ", numSendProc=" << numSendProc << endl;

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

//    for (int i=0; i<nprocs; i++)
//        if(rank==0) cout << "vIndexCount[i] = " << vIndexCount[i] << "\tvdispls[i] = " << vdispls[i] << "\trecvCount[i] = " << recvCount[i] << "\trdispls[i] = " << rdispls[i] << endl;
//    MPI_Barrier(comm);
//    for (int i=0; i<nprocs; i++)
//        if(rank==0) cout << "vIndexCount[i] = " << vIndexCount[i] << "\tvdispls[i] = " << vdispls[i] << "\trecvCount[i] = " << recvCount[i] << "\trdispls[i] = " << rdispls[i] << endl;

    // vIndex is the set of indices of elements that should be sent.
    vIndex = (unsigned long*)malloc(sizeof(unsigned long)*vIndexSize);
    MPI_Alltoallv(&(*(vElement_remote.begin())), recvCount, &*(rdispls.begin()), MPI_UNSIGNED_LONG, vIndex, vIndexCount, &(*(vdispls.begin())), MPI_UNSIGNED_LONG, comm);

    free(vIndexCount);
    free(recvCount);

    for (i=0; i<vIndexSize; i++){
//        if(rank==1) cout << vIndex[i] << "\t" << vIndex[i]-P->split[rank] << endl;
        vIndex[i] -= split[rank];
    }

    // vSend = vector values to send to other procs
    // vecValues = vector values that received from other procs
    // These will be used in matvec and they are set here to reduce the time of matvec.
    vSend     = (double*)malloc(sizeof(double) * vIndexSize);
    vecValues = (double*)malloc(sizeof(double) * recvSize);

    indicesP_local = (unsigned long*)malloc(sizeof(unsigned long)*nnz_l_local);
    for(i=0; i<nnz_l_local; i++)
        indicesP_local[i] = i;
    unsigned long* row_localP = &*row_local.begin();
    std::sort(indicesP_local, &indicesP_local[nnz_l_local], sort_indices(row_localP)); // todo: is it ordered only row-wise?
//    row_local.clear();

//    long start;
//    for(i = 0; i < M; ++i) {
//        start = nnzPerRowScan_local[i];
//        for(long j=0; j < nnzPerRow_local[i]; j++){
//            if(rank==1) printf("%lu \t %lu \t %f \n", entry_local[indicesP_local[start + j]].row+split[rank], entry_local[indicesP_local[start + j]].col, entry_local[indicesP_local[start + j]].val);
//        }
//    }

    indicesP_remote = (unsigned long*)malloc(sizeof(unsigned long)*nnz_l_remote);
    for(i=0; i<nnz_l_remote; i++)
        indicesP_remote[i] = i;
    unsigned long* row_remoteP = &*row_remote.begin();
    std::sort(indicesP_remote, &indicesP_remote[nnz_l_remote], sort_indices(row_remoteP));

//    MPI_Barrier(comm);
//    if(rank==1) cout << "nnz_l_remote = " << nnz_l_remote << "\t\trecvSize_t = " << recvSize_t << "\t\tvIndexSize_t = " << vIndexSize_t << endl;
//    if(rank==0){
//        for(i=0; i<nnz_l_remote; i++)
//            cout << row_remote[i] << "\t" << col_remote2[i] << " =\t" << values_remote[i] << "\t\t\t" << vElement_remote_t[i] << endl;
//    }
//    if(rank==0) cout << endl;
//    MPI_Barrier(comm);

    return 0;
} //end of restrictMatrix::transposeP


restrictMatrix::~restrictMatrix(){
    if(arrays_defined){
        free(vIndex);
        free(vSend);
        free(vecValues);
        free(indicesP_local);
        free(indicesP_remote);
    }
}


int restrictMatrix::matvec(double* v, double* w, MPI_Comm comm) {

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

//    if (rank==1){
//        cout << "vIndexSize=" << vIndexSize << ", vSend: rank=" << rank << endl;
//        for(int i=0; i<vIndexSize; i++)
//            cout << vSend[i] << endl;}

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

//    double t11 = MPI_Wtime();
    // local loop
    fill(&w[0], &w[M], 0);
//#pragma omp parallel        todo: check this openmp part.
//    {
//        long iter = iter_local_array[omp_get_thread_num()];
    long iter = 0;
//#pragma omp for
    for (unsigned int i = 0; i < M; ++i) {
        for (unsigned int j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
//                if(rank==1) cout << entry_local[indicesP_local[iter]].col - splitNew[rank] << "\t" << v[entry_local[indicesP_local[iter]].col - splitNew[rank]] << endl;
//                w[i] += values_local[indicesP_local[iter]] * v[col_local[indicesP_local[iter]] - split[rank]];
            w[i] += entry_local[indicesP_local[iter]].val * v[entry_local[indicesP_local[iter]].col - split[rank]]; // todo: at the end, should it be split or splitNew?
        }
    }
//    }

//    double t21 = MPI_Wtime();
//    time[1] += (t21-t11);

    // Wait for comm to finish.
    MPI_Waitall(numSendProc+numRecvProc, requests, statuses);

//    if (rank==1){
//        cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << endl;
//        for(int i=0; i<recvSize; i++)
//            cout << vecValues[i] << endl;}

    // remote loop
//    double t12 = MPI_Wtime();
//#pragma omp parallel
//    {
//        unsigned int iter = iter_remote_array[omp_get_thread_num()];
    iter = 0;
//#pragma omp for
    for (unsigned int i = 0; i < col_remote_size; ++i) {
        for (unsigned int j = 0; j < nnzPerCol_remote[i]; ++j, ++iter) {
//            if(rank==0){
//                cout << "matvec remote" << endl;
//                cout << row_remote[indicesP_remote[iter]] << "\t" << entry_remote[indicesP_remote[iter]].val << "\t" << vecValues[col_remote[indicesP_remote[iter]]] << endl;
//            }
            w[row_remote[indicesP_remote[iter]]] += entry_remote[indicesP_remote[iter]].val * vecValues[col_remote[indicesP_remote[iter]]];
        }
    }
//    }

//    double t22 = MPI_Wtime();
//    time[2] += (t22-t12);
//    double t23 = MPI_Wtime();
//    time[3] += (t23-t13);

    return 0;
}
