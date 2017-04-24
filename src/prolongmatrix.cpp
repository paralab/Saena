#include <mpi.h>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include "prolongmatrix.h"
#include "auxFunctions.cpp"

prolongMatrix::prolongMatrix(){}

int prolongMatrix::findLocalRemote(unsigned long* r, unsigned long* c, double* v){

    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    unsigned long i;

    unsigned long* indices_p = (unsigned long*)malloc(sizeof(unsigned long)*nnz_l);
    for(i=0; i<nnz_l; i++)
        indices_p[i] = i;
    std::sort(indices_p, &indices_p[nnz_l], sort_indices(c));

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
    nnz_row_local.assign(M,0);

    int* vIndexCount_t = (int*)malloc(sizeof(int)*nprocs);
    std::fill(vIndexCount_t, vIndexCount_t + nprocs, 0);

    // take care of the first element here, since there is "col[i-1]" in the for loop below, so "i" cannot start from 0.
    if (c[indices_p[0]] >= split[rank] && c[indices_p[0]] < split[rank + 1]) {
        nnz_row_local[r[indices_p[0]]]++;
        nnz_l_local++;

        values_local.push_back(v[indices_p[0]]);
        row_local.push_back(r[indices_p[0]]);
        col_local.push_back(c[indices_p[0]]);

        //vElement_local.push_back(col[0]);
        vElementRep_local.push_back(1);

    } else{
        nnz_l_remote++;
        values_remote.push_back(v[indices_p[0]]);
        row_remote.push_back(r[indices_p[0]]);
        col_remote_size++; // number of remote columns
        col_remote.push_back(col_remote_size-1);
        col_remote2.push_back(c[indices_p[0]]);
//        nnz_col_remote[col_remote_size-1]++;
        nnz_col_remote.push_back(1);

        vElement_remote.push_back(c[indices_p[0]]);
        vElementRep_remote.push_back(1);
        recvCount[lower_bound2(&split[0], &split[nprocs], c[indices_p[0]])] = 1;

        nnz_col_remote_t.push_back(1);
        vElement_remote_t.push_back(nnz_l_remote-1);
        vIndexCount_t[lower_bound2(&split[0], &split[nprocs], c[indices_p[0]])] = 1;
    }

    for (i = 1; i < nnz_l; i++) {

        // local
        if (c[indices_p[i]] >= split[rank] && c[indices_p[i]] < split[rank+1]) {
            nnz_row_local[r[indices_p[i]]]++;
            nnz_l_local++;

            values_local.push_back(v[indices_p[i]]);
            row_local.push_back(r[indices_p[i]]);
            col_local.push_back(c[indices_p[i]]);

        // remote
        } else {
            nnz_l_remote++;
            values_remote.push_back(v[indices_p[i]]);
//            if(rank==0) cout << v[indices_p[i]] << endl;
            row_remote.push_back(r[indices_p[i]]);
            // col_remote2 is the original col value and will be used in making strength matrix. col_remote will be used for matevec.
            col_remote2.push_back(c[indices_p[i]]);

            procNum = lower_bound2(&split[0], &split[nprocs], c[indices_p[i]]);
            vIndexCount_t[procNum]++;
            vElement_remote_t.push_back((unsigned long)nnz_l_remote-1);
            nnz_col_remote_t.push_back(1);

            if (c[indices_p[i]] != c[indices_p[i-1]]) {
                col_remote_size++;
                vElement_remote.push_back(c[indices_p[i]]);
                vElementRep_remote.push_back(1);
                procNum = lower_bound2(&split[0], &split[nprocs], c[indices_p[i]]);
                recvCount[procNum]++;
                nnz_col_remote.push_back(1);
            } else {
                (*(vElementRep_remote.end()-1))++;
                (*(nnz_col_remote.end()-1))++;
            }
            // the original col values are not being used for matvec. the ordering starts from 0, and goes up by 1.
            col_remote.push_back(col_remote_size-1);
//            nnz_col_remote[col_remote_size-1]++;
        }
    } // for i

    free(indices_p);

    int* vIndexCount = (int*)malloc(sizeof(int)*nprocs);
    MPI_Alltoall(recvCount, 1, MPI_INT, vIndexCount, 1, MPI_INT, MPI_COMM_WORLD);

    int* recvCount_t = (int*)malloc(sizeof(int)*nprocs);
    MPI_Alltoall(vIndexCount_t, 1, MPI_INT, recvCount_t, 1, MPI_INT, MPI_COMM_WORLD);

//    for(int i=0; i<nprocs; i++){
//        MPI_Barrier(MPI_COMM_WORLD);
//        if(rank==1) cout << "recieve from proc " << i << "\trecvCount   = " << recvCount[i] << "\t\trecvCount_t   = " << recvCount_t[i] << endl;
//        MPI_Barrier(MPI_COMM_WORLD);
//        if(rank==1) cout << "send to proc      " << i << "\tvIndexCount = " << vIndexCount[i] << "\t\tvIndexCount_t = " << vIndexCount_t[i] << endl;
//    }
//    MPI_Barrier(MPI_COMM_WORLD);


    numRecvProc = 0;
    numSendProc = 0;
    for(int i=0; i<nprocs; i++){
        if(recvCount[i]!=0){
            numRecvProc++;
            recvProcRank.push_back(i);
            recvProcCount.push_back(recvCount[i]);
            sendProcCount_t.push_back(vIndexCount_t[i]); // use recvProcRank for it.
        }
        if(vIndexCount[i]!=0){
            numSendProc++;
            sendProcRank.push_back(i);
            sendProcCount.push_back(vIndexCount[i]);
            recvProcCount_t.push_back(recvCount_t[i]); // use sendProcRank for it.
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
    recvSize = rdispls[nprocs-1] + recvCount[nprocs-1];

    vIndex = (unsigned long*)malloc(sizeof(unsigned long)*vIndexSize);
    MPI_Alltoallv(&(*(vElement_remote.begin())), recvCount, &*(rdispls.begin()), MPI_UNSIGNED_LONG, vIndex, vIndexCount, &(*(vdispls.begin())), MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

    free(vIndexCount);
    free(recvCount);


    vdispls_t.resize(nprocs);
    rdispls_t.resize(nprocs);
    vdispls_t[0] = 0;
    rdispls_t[0] = 0;

    for (int i=1; i<nprocs; i++){
//        if(rank==0) cout << "vIndexCount_t = " << vIndexCount_t[i-1] << endl;
        vdispls_t[i] = vdispls_t[i-1] + vIndexCount_t[i-1];
        rdispls_t[i] = rdispls_t[i-1] + recvCount_t[i-1];
    }
    vIndexSize_t = vdispls_t[nprocs-1] + vIndexCount_t[nprocs-1];
    recvSize_t   = rdispls_t[nprocs-1] + recvCount_t[nprocs-1];
//    if(rank==0) cout << "vIndexCount_t = " << vIndexCount_t[nprocs-1] << endl;

    // todo: is this part required?
    recvIndex_t = (unsigned long*)malloc(sizeof(unsigned long)*recvSize_t);
    MPI_Alltoallv(&(*(vElement_remote_t.begin())), vIndexCount_t, &*(vdispls_t.begin()), MPI_UNSIGNED_LONG, recvIndex_t, recvCount_t, &(*(rdispls_t.begin())), MPI_UNSIGNED_LONG, MPI_COMM_WORLD);

    free(vIndexCount_t);
    free(recvCount_t);


//    if(rank==1) cout << endl << endl;
//    for (unsigned int i=0; i<vElement_remote.size(); i++)
//        if(rank==1) cout << vElement_remote[i] << endl;

    // change the indices from global to local
    for (unsigned int i=0; i<vIndexSize; i++){
        vIndex[i] -= split[rank];
    }

    // vSend = vector values to send to other procs
    // vecValues = vector values that received from other procs
    // These will be used in matvec and they are set here to reduce the time of matvec.
    vSend     = (unsigned long*)malloc(sizeof(unsigned long) * vIndexSize);
    vecValues = (unsigned long*)malloc(sizeof(unsigned long) * recvSize);

    vSend_t     = (double*)malloc(sizeof(double) * vIndexSize_t);
    vecValues_t = (double*)malloc(sizeof(double) * recvSize_t);

    indicesP_local = (unsigned long*)malloc(sizeof(unsigned long)*nnz_l_local);
    for(i=0; i<nnz_l_local; i++)
        indicesP_local[i] = i;
    unsigned long* row_localP = &(*(row_local.begin()));
    std::sort(indicesP_local, &indicesP_local[nnz_l_local], sort_indices(row_localP));

//    indicesP_remote = (unsigned long*)malloc(sizeof(unsigned long)*nnz_l_remote);
//    for(i=0; i<nnz_l_remote; i++)
//        indicesP_remote[i] = i;
//    unsigned long* row_remoteP = &(*(row_remote.begin()));
//    std::sort(indicesP_remote, &indicesP_remote[nnz_l_remote], sort_indices(row_remoteP));


//    MPI_Barrier(MPI_COMM_WORLD);
//    if(rank==1) cout << "nnz_l_remote = " << nnz_l_remote << "\t\trecvSize_t = " << recvSize_t << "\t\tvIndexSize_t = " << vIndexSize_t << endl;
//    if(rank==0){
//        for(i=0; i<nnz_l_remote; i++)
//            cout << row_remote[i] << "\t" << col_remote2[i] << " =\t" << values_remote[i] << "\t\t\t" << vElement_remote_t[i] << endl;
//    }
//    if(rank==0) cout << endl;
//    MPI_Barrier(MPI_COMM_WORLD);

    return 0;
}

prolongMatrix::~prolongMatrix(){
    free(vIndex);
    free(vSend);
    free(vecValues);
    free(indicesP_local);
    free(recvIndex_t);
    free(vSend_t);
    free(vecValues_t);
}
