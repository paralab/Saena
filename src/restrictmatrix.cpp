#include <iostream>
#include <algorithm>
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

} //end of restrictMatrix::restrictMatrix


restrictMatrix::~restrictMatrix(){}
