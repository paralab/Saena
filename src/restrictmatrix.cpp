#include <iostream>
#include <algorithm>
#include <mpi.h>
#include <prolongmatrix.h>
#include <restrictmatrix.h>
#include <dtypes.h>

using namespace std;

restrictMatrix::restrictMatrix(){}


restrictMatrix::restrictMatrix(prolongMatrix* P, MPI_Comm comm) {

    // todo: this matrix is not sorted at the end.
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    unsigned long i, j;

    Mbig = P->Nbig;
    Nbig = P->Mbig;

    // set the number of rows for each process
    M = P->splitNew[rank+1] - P->splitNew[rank];

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
//    if(rank==1) cout << "vSend_t:" << endl;
    for (i = 0; i < P->nnz_l_remote; i++){ // all remote entries should be sent.
        P->vSend_t[i] = P->entry_remote[i];
        P->vSend_t[i].row += P->split[rank];
//        P->vSend_t[2*i]   = P->col_remote2[i]; // don't forget to subtract splitNew[rank] from vecValues.
//        P->vSend_t[2*i+1] = P->row_remote[i] + P->split[rank];
//        if(rank==1) cout << i << "\t" << P->vSend_t[2*i] << "\t" << P->vSend_t[2*i+1] << endl;
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
//            if(rank==4) cout << P->entry_local[P->indicesP_local[iter]].col << "\t" << P->entry_local[P->indicesP_local[iter]].col - P->splitNew[rank]
//                             << "\t" << P->entry_local[P->indicesP_local[iter]].row << "\t" << P->entry_local[P->indicesP_local[iter]].row + P->split[rank]
//                             << "\t" << P->entry_local[P->indicesP_local[iter]].val << endl;
            entry_local.push_back(cooEntry(P->entry_local[P->indicesP_local[iter]].col - P->splitNew[rank], // make row index local
                                           P->entry_local[P->indicesP_local[iter]].row + P->split[rank],    // make col index global
                                           P->entry_local[P->indicesP_local[iter]].val));
        }
    }

    std::sort(entry_local.begin(), entry_local.end());

//    MPI_Barrier(comm);
//    iter = 0;
//    if(rank==4){
//        cout << endl << "local:" << " rank=" << rank << endl;
//        for (i = 0; i < P->M; ++i)
//            for (j = 0; j < P->nnzPerRow_local[i]; ++j, ++iter)
//                cout << entry_local[iter].row << "\t" << entry_local[iter].col << "\t" << entry_local[iter].val << endl;}
//    MPI_Barrier(comm);

    // *********************** assign remote part of restriction ************************

    MPI_Waitall(P->numSendProc_t + P->numRecvProc_t, requests, statuses);

//    MPI_Barrier(comm);
//    if(rank==0)cout << "vecValues_t:" << "\t" << rank << endl;
    for(i=0; i<P->recvSize_t; i++){
//        if(rank==0) printf("%lu\t %lu\t %f\n", P->vecValues_t[i].row, P->vecValues_t[i].col, P->vecValues_t[i].val);
        entry_remote.push_back(cooEntry(P->vecValues_t[i].col - P->splitNew[rank], // make row index local
                                        P->vecValues_t[i].row,
                                        P->vecValues_t[i].val));
    }

    std::sort(entry_remote.begin(), entry_remote.end());

//    MPI_Barrier(comm);
//    if(rank==4){
//        cout << endl << "remote:" << " rank = " << rank << "\tP->recvSize_t = " << P->recvSize_t << endl;
//        for(i=0; i<P->recvSize_t; i++)
//            cout << i << "\t" << entry_remote[i].row << "\t" << entry_remote[i].col << "\t" << entry_remote[i].val << endl;}
//    MPI_Barrier(comm);

    nnz_l_local  = entry_local.size();
    nnz_l_remote = entry_remote.size();
    nnz_l = nnz_l_local + nnz_l_remote;
    MPI_Allreduce(&nnz_l, &nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
    // todo: check why is R so imbalanced for 289 size matrix on 8 processors. use the following print function.
//    printf("rank=%d \t nnz_l=%lu \t nnz_l_local=%lu   \t nnz_l_remote=%lu \t nnz_g=%lu \n", rank, nnz_l, nnz_l_local, nnz_l_remote, nnz_g);


    // *********************** setup matvec - local R ************************

    long procNum;
    col_remote_size = 0; // number of remote columns
    int* recvCount = (int*)malloc(sizeof(int)*nprocs);
    std::fill(recvCount, recvCount + nprocs, 0);
    nnzPerRow_local.assign(M,0);

    // take care of the first element here, since there is "col[i-1]" in the for loop below, so "i" cannot start from 0.
    // local
    if (entry_local[0].col >= P->split[rank] && entry_local[0].col < P->split[rank + 1]) {
        nnzPerRow_local[entry_local[0].row]++;
        vElementRep_local.push_back(1);
//        nnz_l_local++;
//        row_local.push_back(entry_local[0].row); // only for sorting at the end of prolongMatrix::findLocalRemote. then clear the vector. // todo: clear does not free memory. find a solution.

        // remote
    } else{
        col_remote_size++; // number of remote columns
        nnzPerCol_remote.push_back(1);
        vElement_remote.push_back(entry_local[0].col);
        vElementRep_remote.push_back(1);
        recvCount[lower_bound2(&P->split[0], &P->split[nprocs], entry_local[0].col)] = 1;

//        nnz_l_remote++;
//        row_remote.push_back(entry_local[0].row); // only for sorting at the end of prolongMatrix::findLocalRemote. then clear the vector. // todo: clear does not free memory. find a solution.
//        col_remote2.push_back(entry[0].col);
//        values_remote.push_back(entry[0].val);
//        nnzPerCol_remote[col_remote_size-1]++;
    }
    for (i = 1; i < entry_local.size(); i++) {

        // local
        if (entry_local[i].col >= P->split[rank] && entry_local[i].col < P->split[rank+1]) {
            nnzPerRow_local[entry_local[i].row]++;
//            row_local.push_back(entry_local[i].row); // only for sorting at the end of prolongMatrix::findLocalRemote. then erase. // todo: clear does not free memory. find a solution.
//            col_local.push_back(entry[i].col);
//            values_local.push_back(entry[i].val);
            if (entry_local[i].col != entry_local[i-1].col)
                vElementRep_local.push_back(1);
            else
                (*(vElementRep_local.end()-1))++;

            // remote
        } else {
//            if(rank==2) printf("entry[i].row = %lu\n", entry[i].row+split[rank]);
//            row_remote.push_back(entry_local[i].row); // only for sorting at the end of prolongMatrix::findLocalRemote. then clear the vector. // todo: clear does not free memory. find a solution.
            // col_remote2 is the original col value. col_remote starts from 0.
//            col_remote2.push_back(entry[i].col);
//            values_remote.push_back(entry[i].val);

            if (entry_local[i].col != entry_local[i-1].col) {
                col_remote_size++;
                vElement_remote.push_back(entry_local[i].col);
                vElementRep_remote.push_back(1);
                procNum = lower_bound2(&P->split[0], &P->split[nprocs], entry_local[i].col);
                recvCount[procNum]++;
                nnzPerCol_remote.push_back(1);
            } else {
                (*(vElementRep_remote.end()-1))++;
                (*(nnzPerCol_remote.end()-1))++;
            }
        }
    } // for i

    // *********************** setup matvec - remote R ************************

    // take care of the first element here, since there is "col[i-1]" in the for loop below, so "i" cannot start from 0.
    // local
    if (entry_remote[0].col >= P->split[rank] && entry_remote[0].col < P->split[rank + 1]) {
        nnzPerRow_local[entry_remote[0].row]++;
        vElementRep_local.push_back(1);
//        nnz_l_local++;
//        row_local.push_back(entry_local[0].row); // only for sorting at the end of prolongMatrix::findLocalRemote. then clear the vector. // todo: clear does not free memory. find a solution.

        // remote
    } else{
        col_remote_size++; // number of remote columns
        nnzPerCol_remote.push_back(1);

        vElement_remote.push_back(entry_remote[0].col);
        vElementRep_remote.push_back(1);
        recvCount[lower_bound2(&P->split[0], &P->split[nprocs], entry_remote[0].col)] = 1;

//        nnz_l_remote++;
//        row_remote.push_back(entry_local[0].row); // only for sorting at the end of prolongMatrix::findLocalRemote. then clear the vector. // todo: clear does not free memory. find a solution.
//        col_remote2.push_back(entry[0].col);
//        values_remote.push_back(entry[0].val);
//        nnzPerCol_remote[col_remote_size-1]++;
    }

    for (i = 1; i < entry_remote.size(); i++) {

        // local
        if (entry_remote[i].col >= P->split[rank] && entry_remote[i].col < P->split[rank+1]) {
            nnzPerRow_local[entry_remote[i].row]++;
//            row_local.push_back(entry_local[i].row); // only for sorting at the end of prolongMatrix::findLocalRemote. then erase. // todo: clear does not free memory. find a solution.
//            col_local.push_back(entry[i].col);
//            values_local.push_back(entry[i].val);

            // remote
        } else {
            procNum = lower_bound2(&P->split[0], &P->split[nprocs], entry_remote[i].col);
//            if(rank==2) printf("entry[i].row = %lu\n", entry[i].row+split[rank]);
//            row_remote.push_back(entry_local[i].row); // only for sorting at the end of prolongMatrix::findLocalRemote. then clear the vector. // todo: clear does not free memory. find a solution.
            // col_remote2 is the original col value. col_remote starts from 0.
//            col_remote2.push_back(entry[i].col);
//            values_remote.push_back(entry[i].val);
//            recvCount_t[procNum]++;
//            nnzPerCol_remote_t.push_back(1);

            if (entry_remote[i].col != entry_remote[i-1].col) {
                col_remote_size++;
                vElement_remote.push_back(entry_remote[i].col);
                vElementRep_remote.push_back(1);
                procNum = lower_bound2(&P->split[0], &P->split[nprocs], entry_remote[i].col);
                recvCount[procNum]++;
                nnzPerCol_remote.push_back(1);
            } else {
                (*(vElementRep_remote.end()-1))++;
                (*(nnzPerCol_remote.end()-1))++;
            }
        }
    } // for i

} //end of restrictMatrix::restrictMatrix


restrictMatrix::~restrictMatrix(){
//    free(vIndex);
//    free(vSend);
//    free(vecValues);
//    free(indicesP_local);
//    free(indicesP_remote);
}

/*
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

//    if (rank==0){
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

//    if (rank==0){
//        cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << endl;
//        for(int i=0; i<recvSize; i++)
//            cout << vecValues[i] << endl;}

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
            w[i] += entry_local[indicesP_local[iter]].val * v[entry_local[indicesP_local[iter]].col - splitNew[rank]]; // todo: at the end, should it be split or splitNew?
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
*/