#include <iostream>
#include <mpi.h>
#include <prolongmatrix.h>
#include <restrictmatrix.h>
#include <dtypes.h>

using namespace std;

restrictMatrix::restrictMatrix(){}

restrictMatrix::restrictMatrix(prolongMatrix* P, unsigned long* initialNumberOfRows, MPI_Comm comm) {

    // todo: this matrix is not sorted at the end.
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    unsigned long i, j;

    Mbig = P->Nbig;
    Nbig = P->Mbig;

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
//    if(rank==0) cout << "vSend_t:" << endl;
    for (i = 0; i < P->nnz_l_remote; i++){ // all remote entries should be sent.
//        P->vSend_t[3*i]   = P->col_remote2[i]; // todo: don't forget to subtract splitNew[rank] from vecValues.
//        P->vSend_t[3*i+1] = P->row_remote[i] + P->split[rank];
//        P->vSend_t[3*i+2] = P->values_remote[i];
//        if(rank==1) cout << P->vSend_t[3*i] << "\t" << P->vSend_t[3*i+1] << "\t" << P->vSend_t[3*i+2] << endl;

        P->vSend_t[2*i]   = P->col_remote2[i]; // todo: don't forget to subtract splitNew[rank] from vecValues.
        P->vSend_t[2*i+1] = P->row_remote[i] + P->split[rank];
//        if(rank==1) cout << i << "\t" << P->vSend_t[2*i] << "\t" << P->vSend_t[2*i+1] << endl;

//        P->vSend_t[i] = P->col_remote2[i]; // todo: don't forget to change this.
//        if(rank==0) cout << i << "\t" << P->vSend_t[i] << endl;
    }

//    MPI_Barrier(comm);
//    if(rank==1) cout << "vSend_t: rank=" << rank << endl;
//    for (i = 0; i < P->nnz_l_remote; i++){ // all remote entries should be sent.
//        if(rank==1) cout << i << "\t" << P->vSend_t[2*i] << "\t" << P->vSend_t[2*i+1] << endl;
//    }
//    MPI_Barrier(comm);
//    if(rank==2) cout << "vSend_t: rank=" << rank << endl;
//    for (i = 0; i < P->nnz_l_remote; i++){ // all remote entries should be sent.
//        if(rank==2) cout << i << "\t" << P->vSend_t[2*i] << "\t" << P->vSend_t[2*i+1] << endl;
//    }

    MPI_Request *requests = new MPI_Request[P->numSendProc_t + P->numRecvProc_t];
    MPI_Status  *statuses = new MPI_Status[P->numSendProc_t + P->numRecvProc_t];

    // todo: check datatype here.
//    for (i = 0; i < P->numSendProc; i++)
//        MPI_Irecv(&P->vecValues_t[P->rdispls_t[P->sendProcRank[i]]], P->recvProcCount_t[i], MPI_UNSIGNED_LONG,
//                  P->sendProcRank[i], 1, comm, &(requests[i]));
//
//    for (i = 0; i < P->numRecvProc; i++)
//        MPI_Isend(&P->vSend_t[P->vdispls_t[P->recvProcRank[i]]], P->sendProcCount_t[i], MPI_UNSIGNED_LONG,
//                  P->recvProcRank[i], 1, comm, &(requests[P->numSendProc + i]));

    for(i = 0; i < P->numRecvProc_t; i++)
        MPI_Irecv(&P->vecValues_t[P->rdispls_t[P->recvProcRank_t[i]]], P->recvProcCount_t[i], MPI_UNSIGNED_LONG, P->recvProcRank_t[i], 1, comm, &(requests[i]));

    for(i = 0; i < P->numSendProc_t; i++)
        MPI_Isend(&P->vSend_t[P->vdispls_t[P->sendProcRank_t[i]]], P->sendProcCount_t[i], MPI_UNSIGNED_LONG, P->sendProcRank_t[i], 1, comm, &(requests[P->numRecvProc_t+i]));

//    for(i = 0; i < P->numRecvProc_t; i++)
//        MPI_Irecv(&P->vecValues_t[P->rdispls_t[P->recvProcRank_t[i]]], P->recvProcCount_t[i], cooEntry::mpi_datatype(), P->recvProcRank_t[i], 1, comm, &(requests[i]));
//
//    for(i = 0; i < P->numSendProc_t; i++)
//        MPI_Isend(&P->vSend_t[P->vdispls_t[P->sendProcRank_t[i]]], P->sendProcCount_t[i], cooEntry::mpi_datatype(), P->sendProcRank_t[i], 1, comm, &(requests[P->numRecvProc_t+i]));

//    if(rank==0)
//        for(i=0; i<P->sendProcCount_t[0]; i++)
//            cout << i << "\t" << P->vSend_t[P->vdispls_t[P->recvProcRank[0]]] << endl;

    // *********************** assign local part of restriction ************************

    //local
    unsigned long iter = 0;
    for (i = 0; i < P->M; ++i) {
        for (j = 0; j < P->nnz_row_local[i]; ++j, ++iter) {
//            if(rank==1) cout << P->row_local[P->indicesP_local[iter]] << "\t" << P->col_local[P->indicesP_local[iter]] << "\t" << P->values_local[P->indicesP_local[iter]] << endl;
//            if(rank==1) cout << P->col_local[P->indicesP_local[iter]] - P->splitNew[rank] << "\t" << P->row_local[P->indicesP_local[iter]] + P->split[rank] << "\t" << P->values_local[P->indicesP_local[iter]] << endl << endl;
            row_local.push_back(P->col_local[P->indicesP_local[iter]] - P->splitNew[rank]);
            col_local.push_back(P->row_local[P->indicesP_local[iter]] + P->split[rank]);
            values_local.push_back(P->values_local[P->indicesP_local[iter]]);
//            if(rank==1) cout << row_local[iter] << "\t" << col_local[iter] << "\t" << values_local[iter] << endl;
//            if(rank==1) cout << P->row_local[iter] << "\t" << P->col_local[iter] << "\t" << P->values_local[iter] << endl;
//            if(rank==1) cout << P->row_local[P->indicesP_local[iter]] << "\t" << P->col_local[P->indicesP_local[iter]] << "\t" << P->values_local[P->indicesP_local[iter]] << endl;
        }
    }

    // *********************** assign remote part of restriction ************************

    MPI_Waitall(P->numSendProc_t + P->numRecvProc_t, requests, statuses);

//    MPI_Barrier(comm);
//    if(rank==0)cout << "vecValues_t:" << "\t" << rank << endl;
//    MPI_Barrier(comm);
    for(i=0; i<P->recvSize_t; i++){
        row_remote.push_back(P->vecValues_t[2*i] - P->splitNew[rank]);
        col_remote.push_back(P->vecValues_t[2*i+1]);
//        if(rank==1) cout << P->vecValues_t[3*i] - P->splitNew[rank] << "\t" << P->vecValues_t[3*i+1] << "\t" << P->vecValues_t[3*i+2] << endl;
//        if(rank==0) cout << i << "\t" << P->vecValues_t[2*i] - P->splitNew[rank] << "\t" << P->vecValues_t[2*i+1] << endl;
//        if(rank==1) cout << P->vecValues_t[i] - P->splitNew[rank] << endl;
    }

//    MPI_Barrier(comm);
//    if(rank==2)cout << "vecValues_t:" << "\t" << rank << endl;
//    for(i=0; i<P->recvSize_t; i++)
//        if(rank==2) cout << i << "\t" << P->vecValues_t[2*i] << "\t" << P->vecValues_t[2*i+1] << endl;
}

//restrictMatrix::restrictMatrix(unsigned long Mb, unsigned long Nb, unsigned long nz_g, unsigned long nz_l, unsigned long* r, unsigned long* c, double* v){
//    Mbig = Mb;
//    Nbig = Nb;
//    nnz_g = nz_g;
//    nnz_l = nz_l;
//
//    row.resize(nnz_l);
//    col.resize(nnz_l);
//    values.resize(nnz_l);
//
//    for(long i=0; i<nnz_l; i++){
//        row[i] = r[i];
//        col[i] = c[i];
//        values[i] = v[i];
//    }
//}

restrictMatrix::~restrictMatrix(){}
