#include <mpi.h>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include "prolongmatrix.h"
#include "auxFunctions.h"

using namespace std;

prolongMatrix::prolongMatrix(){}

int prolongMatrix::findLocalRemote(cooEntry* entry, MPI_Comm comm){

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    unsigned long i;

    arrays_defined = true;

//    unsigned long* indices_p = (unsigned long*)malloc(sizeof(unsigned long)*nnz_l);
//    for(i=0; i<nnz_l; i++)
//        indices_p[i] = i;
//    std::sort(indices_p, &indices_p[nnz_l], sort_indices(c));

//    for(unsigned int i=0; i<nnz_l; i++){
//        if(rank==0) cout << r[indices_p[i]] << "\t" << c[indices_p[i]] << "\t\t" << v[indices_p[i]] << endl;
//        if(rank==0) cout << r[i] << "\t" << c[i] << "\t\t" << v[i] << "\t\t\t" << r[indices_p[i]] << "\t" << c[indices_p[i]] << "\t\t" << v[indices_p[i]] << endl;
//    }

    long procNum;
    col_remote_size = 0; // number of remote columns
    nnz_l_local = 0;
    nnz_l_remote = 0;
//    int recvCount[nprocs];
    int* recvCount = (int*)malloc(sizeof(int)*nprocs);
    std::fill(recvCount, recvCount + nprocs, 0);
//    int* recvCount_t = (int*)malloc(sizeof(int)*nprocs);
//    std::fill(recvCount_t, recvCount_t + nprocs, 0);
    nnzPerRow_local.assign(M,0);

    int* vIndexCount_t = (int*)malloc(sizeof(int)*nprocs);
    std::fill(vIndexCount_t, vIndexCount_t + nprocs, 0);

    // take care of the first element here, since there is "col[i-1]" in the for loop below, so "i" cannot start from 0.
    // local
    if (entry[0].col >= splitNew[rank] && entry[0].col < splitNew[rank + 1]) {
        nnzPerRow_local[entry[0].row]++;
        nnz_l_local++;

        entry_local.push_back(cooEntry(entry[0].row, entry[0].col, entry[0].val));
        row_local.push_back(entry[0].row); // only for sorting at the end of prolongMatrix::findLocalRemote. then clear the vector.
//        col_local.push_back(entry[0].col);
//        values_local.push_back(entry[0].val);

        //vElement_local.push_back(col[0]);
        vElementRep_local.push_back(1);

    // remote
    } else{
        nnz_l_remote++;
        entry_remote.push_back(cooEntry(entry[0].row, entry[0].col, entry[0].val));
//        row_remote.push_back(entry[0].row);
//        col_remote2.push_back(entry[0].col);
//        values_remote.push_back(entry[0].val);
        col_remote_size++; // number of remote columns
        col_remote.push_back(col_remote_size-1);
//        nnzPerCol_remote[col_remote_size-1]++;
        nnzPerCol_remote.push_back(1);

        vElement_remote.push_back(entry[0].col);
        vElementRep_remote.push_back(1);
        recvCount[lower_bound2(&split[0], &split[nprocs], entry[0].col)] = 1;

//        nnzPerCol_remote_t.push_back(1);
        vElement_remote_t.push_back(nnz_l_remote-1);
        vIndexCount_t[lower_bound2(&splitNew[0], &splitNew[nprocs], entry[0].col)] = 1;
//        recvCount_t[lower_bound2(&splitNew[0], &splitNew[nprocs], entry[0].col)] = 1;
    }

    for (i = 1; i < nnz_l; i++) {

        // local
        if (entry[i].col >= splitNew[rank] && entry[i].col < splitNew[rank+1]) {
            nnzPerRow_local[entry[i].row]++;
            nnz_l_local++;

            entry_local.push_back(cooEntry(entry[i].row, entry[i].col, entry[i].val));
            row_local.push_back(entry[i].row); // only for sorting at the end of prolongMatrix::findLocalRemote. then erase.
//            col_local.push_back(entry[i].col);
//            values_local.push_back(entry[i].val);

        // remote
        } else {
            nnz_l_remote++;
//            if(rank==2) printf("entry[i].row = %lu\n", entry[i].row+split[rank]);
            entry_remote.push_back(cooEntry(entry[i].row, entry[i].col, entry[i].val));
//            row_remote.push_back(entry[i].row);
            // col_remote2 is the original col value. col_remote starts from 0.
//            col_remote2.push_back(entry[i].col);
//            values_remote.push_back(entry[i].val);
            procNum = lower_bound2(&splitNew[0], &splitNew[nprocs], entry[i].col);
            vIndexCount_t[procNum]++;
//            recvCount_t[procNum]++;
            vElement_remote_t.push_back((unsigned long)nnz_l_remote-1); // todo: is (unsigned long) required here?
//            nnzPerCol_remote_t.push_back(1);

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

//    free(indices_p);

    nnzPerRowScan_local.resize(M+1);
    nnzPerRowScan_local[0] = 0;
    for(i=0; i<M; i++){
        nnzPerRowScan_local[i+1] = nnzPerRowScan_local[i] + nnzPerRow_local[i];
//        if(rank==0) printf("nnzPerRowScan_local=%d, nnzPerRow_local=%d\n", nnzPerRowScan_local[i], nnzPerRow_local[i]);
    }

    int* vIndexCount = (int*)malloc(sizeof(int)*nprocs);
    MPI_Alltoall(recvCount, 1, MPI_INT, vIndexCount, 1, MPI_INT, comm);

    int* recvCount_t = (int*)malloc(sizeof(int)*nprocs);
    MPI_Alltoall(vIndexCount_t, 1, MPI_INT, recvCount_t, 1, MPI_INT, comm);

//    for(int i=0; i<nprocs; i++){
//        MPI_Barrier(comm);
//        if(rank==1) cout << "recieve from proc " << i << "\trecvCount   = " << recvCount[i] << "\t\trecvCount_t   = " << recvCount_t[i] << endl;
//        MPI_Barrier(comm);
//        if(rank==1) cout << "send to proc      " << i << "\tvIndexCount = " << vIndexCount[i] << "\t\tvIndexCount_t = " << vIndexCount_t[i] << endl;
//    }
//    MPI_Barrier(comm);

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

    MPI_Barrier(comm);
    numRecvProc_t = 0;
    numSendProc_t = 0;
    for(int i=0; i<nprocs; i++){
        if(recvCount_t[i]!=0){
            numRecvProc_t++;
            recvProcRank_t.push_back(i);
            recvProcCount_t.push_back(recvCount_t[i]);
//            if(rank==2) cout << i << "\trecvCount_t[i] = " << recvCount_t[i] << endl;
        }
        if(vIndexCount_t[i]!=0){
            numSendProc_t++;
            sendProcRank_t.push_back(i);
            sendProcCount_t.push_back(vIndexCount_t[i]);
//            if(rank==1) cout << i << "\tvIndexCount_t[i] = " << vIndexCount_t[i] << endl;
        }
    }

    vdispls_t.resize(nprocs);
    rdispls_t.resize(nprocs);
    vdispls_t[0] = 0;
    rdispls_t[0] = 0;

    for (int i=1; i<nprocs; i++){
//        if(rank==0) cout << "vIndexCount_t = " << vIndexCount_t[i-1] << endl;
        vdispls_t[i] = vdispls_t[i-1] + vIndexCount_t[i-1];
        rdispls_t[i] = rdispls_t[i-1] + recvCount_t[i-1];
    }
    vIndexSize_t = vdispls_t[nprocs-1] + vIndexCount_t[nprocs-1]; // the same as: vIndexSize_t = nnz_l_remote;
    recvSize_t   = rdispls_t[nprocs-1] + recvCount_t[nprocs-1];

//    for (i=1; i<nprocs; i++){
//        vdispls_t[i] = 2*vdispls_t[i];
//        rdispls_t[i] = 2*rdispls_t[i];
//    }

//    MPI_Barrier(comm);
//    printf("rank = %d\tvIndexSize_t = %u\trecvSize_t = %u \n", rank, vIndexSize_t, recvSize_t);

    // todo: is this part required?
    // vElement_remote_t is the set of indices of entries that should be sent.
    // recvIndex_t       is the set of indices of entries that should be received.
//    recvIndex_t = (unsigned long*)malloc(sizeof(unsigned long)*recvSize_t);
//    MPI_Alltoallv(&(*(vElement_remote_t.begin())), vIndexCount_t, &*(vdispls_t.begin()), MPI_UNSIGNED_LONG, recvIndex_t, recvCount_t, &(*(rdispls_t.begin())), MPI_UNSIGNED_LONG, comm);

    free(vIndexCount_t);
    free(recvCount_t);

//    if(rank==1) cout << endl << endl;
//    for (unsigned int i=0; i<vElement_remote.size(); i++)
//        if(rank==1) cout << vElement_remote[i] << endl;

    // change the indices from global to local
    for (i=0; i<vIndexSize; i++)
        vIndex[i] -= split[rank]; // todo: check if this should be splitNew or split.

    // vSend = vector values to send to other procs
    // vecValues = vector values that received from other procs
    // These will be used in matvec and they are set here to reduce the time of matvec.
    vSend     = (unsigned long*)malloc(sizeof(unsigned long) * vIndexSize);
    vecValues = (unsigned long*)malloc(sizeof(unsigned long) * recvSize);

    vSend_t     = (cooEntry*)malloc(sizeof(cooEntry) * vIndexSize_t); // todo: check datatype here.
    vecValues_t = (cooEntry*)malloc(sizeof(cooEntry) * recvSize_t);

    indicesP_local = (unsigned long*)malloc(sizeof(unsigned long)*nnz_l_local);
    for(i=0; i<nnz_l_local; i++)
        indicesP_local[i] = i;
    unsigned long* row_localP = &(*(row_local.begin()));
    std::sort(indicesP_local, &indicesP_local[nnz_l_local], sort_indices(row_localP));
    row_local.clear();

//    long start;
//    for(i = 0; i < M; ++i) {
//        start = nnzPerRowScan_local[i];
//        for(long j=0; j < nnzPerRow_local[i]; j++){
//            if(rank==1) printf("P.row=%lu, P.col=%lu, P.val=%f \n", entry_local[indicesP_local[start + j]].row+split[rank], entry_local[indicesP_local[start + j]].col, entry_local[indicesP_local[start + j]].val);
//        }
//    }

//    indicesP_remote = (unsigned long*)malloc(sizeof(unsigned long)*nnz_l_remote);
//    for(i=0; i<nnz_l_remote; i++)
//        indicesP_remote[i] = i;
//    unsigned long* row_remoteP = &(*(row_remote.begin()));
//    std::sort(indicesP_remote, &indicesP_remote[nnz_l_remote], sort_indices(row_remoteP));

//    MPI_Barrier(comm);
//    if(rank==1) cout << "nnz_l_remote = " << nnz_l_remote << "\t\trecvSize_t = " << recvSize_t << "\t\tvIndexSize_t = " << vIndexSize_t << endl;
//    if(rank==0){
//        for(i=0; i<nnz_l_remote; i++)
//            cout << row_remote[i] << "\t" << col_remote2[i] << " =\t" << values_remote[i] << "\t\t\t" << vElement_remote_t[i] << endl;
//    }
//    if(rank==0) cout << endl;
//    MPI_Barrier(comm);

    return 0;
}

prolongMatrix::~prolongMatrix(){
    if(arrays_defined){
        free(vIndex);
        free(vSend);
        free(vecValues);
        free(indicesP_local);
//       free(recvIndex_t); // recvIndex_t is equivalent of vIndex.
        free(vSend_t);
        free(vecValues_t);
    }
}
