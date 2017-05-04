#include <prolongmatrix.h>
#include <iostream>
#include <mpi.h>
#include "restrictmatrix.h"

using namespace std;

restrictMatrix::restrictMatrix(){}

restrictMatrix::restrictMatrix(prolongMatrix* P, unsigned long* initialNumberOfRows, MPI_Comm comm) {
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    unsigned long i, j;

    Mbig = P->Nbig;
    Nbig = P->Mbig;

    // *********************** check for shrinking ************************

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


    // *********************** assign local part of restriction ************************

    //local
    unsigned long iter = 0;
    for (i = 0; i < P->M; ++i) {
        for (j = 0; j < P->nnz_row_local[i]; ++j, ++iter) {
            row_local.push_back(P->col_local[P->indicesP_local[iter]] - P->split[rank]);
            col_local.push_back(P->row_local[P->indicesP_local[iter]] + P->split[rank]);
            values_local.push_back(P->values_local[P->indicesP_local[iter]]);
//            if(rank==1) cout << row_local[iter] << "\t" << col_local[iter] << "\t" << values_local[iter] << endl;
//            if(rank==1) cout << P->row_local[iter] << "\t" << P->col_local[iter] << "\t" << P->values_local[iter] << endl;
//            if(rank==1) cout << P->row_local[P->indicesP_local[iter]] << "\t" << P->col_local[P->indicesP_local[iter]] << "\t" << P->values_local[P->indicesP_local[iter]] << endl;
        }
    }



//    for(unsigned int i=0;i<vIndexSize;i++)
//        vSend[i] = v[( vIndex[i] )];
//        S->vSend[i] = weight[(S->vIndex[i])];

//    for (i = 0; i < P->vIndexSize_t; i++){ // todo: make the size of vSend and the related data structures triple.
//        P->vSend_t[3*i]   = P->row_remote[P->vElement_remote_t[i]];
//        P->vSend_t[3*i+1] = P->col_remote2[P->vElement_remote_t[i]];
//        P->vSend_t[3*i+2] = P->values_remote[P->vElement_remote_t[i]];

//        P->vSend_t[i] = P->values_remote[i];
//        if(rank==1) cout << P->vElement_remote_t[i] << "\t" << P->vSend_t[i] << endl;
//        if(rank==1) cout << P->recvIndex_t[i] << endl;
//    }

//    if(rank==1) cout << endl << "vIndexSize_t = " << P->vIndexSize_t << ", nnz_l_remote = " << P->nnz_l_remote << endl;

//    if(rank==0) cout << endl << "second for:" << endl;
//    for (i = 0; i < P->nnz_l_remote; i++){
//        if(rank==0) cout << P->values_remote[i] << endl;
//    }

//    for (i = 0; i < S->numRecvProc; i++)
//        MPI_Irecv(&S->vecValues[S->rdispls[S->sendProcRank[i]]], S->recvProcCount[i], MPI_UNSIGNED_LONG,
//                  S->recvProcRank[i], 1, comm, &(requests[i]));
//
//    for (i = 0; i < S->numSendProc; i++)
//        MPI_Isend(&S->vSend[S->vdispls[S->recvProcRank[i]]], S->sendProcCount[i], MPI_UNSIGNED_LONG,
//                  S->sendProcRank[i], 1, comm, &(requests[S->numRecvProc + i]));

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
