#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <mpi.h>

#include "csrmatrix.h"

using namespace std;

// sort indices and store the ordering.
class sort_indices
{
private:
    long* mparr;
public:
    sort_indices(long* parr) : mparr(parr) {}
    bool operator()(long i, long j) const { return mparr[i]<mparr[j]; }
};

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



//CSRMatrix::CSRMatrix(){}

int CSRMatrix::CSRMatrixSet(long* r, long* c, double* v, long m1, long m2, long m3, long* spl){
    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    M = m1;
    Mbig = m2;
    nnz_l = m3;
    split = spl;

    /*
    rowIndex.resize(M+1);
    rowIndex.assign(M+1, 0);
    col.resize(nnz_l);
    values.resize(nnz_l);

//    long* row = r;
//    col = c;
//    values = v;

    // save row-wise sorting in indicesRow_p
    long* indicesRow_p = (long*)malloc(sizeof(long)*nnz_l);
    for(int i=0; i<nnz_l; i++)
        indicesRow_p[i] = i;
    std::sort(indicesRow_p, &indicesRow_p[nnz_l], sort_indices(r));

    unsigned int i;
    for(i=0; i<nnz_l; i++){
        rowIndex[r[i]+1 - split[rank]]++;
        col[i] = c[indicesRow_p[i]];
        values[i] = v[indicesRow_p[i]];
//        if (rank==1) cout << "[" << r[indicesRow_p[i]]+1 << "," << c[indicesRow_p[i]]+1 << "] = " << v[indicesRow_p[i]] << endl;
    }

    for(i=0; i<M; i++)
        rowIndex[i+1] += rowIndex[i];

    MPI_Allreduce(&nnz_l, &nnz_g, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    average_sparsity = (1.0*nnz_g)/Mbig;
*/
    long procNum;
    long i;
    col_remote_size = -1;
    nnz_l_local = 0;
    nnz_l_remote = 0;
    int recvCount[nprocs];
    std::fill(recvCount, recvCount + nprocs, 0);
    nnz_row_local.assign(M,0);


    // take care of the first element here, since there is "col[i-1]" in the for loop below, so "i" cannot start from 0.
    if (c[0] >= split[rank] && c[0] < split[rank + 1]) {
        nnz_row_local[r[0]-split[rank]]++;
        nnz_l_local++;

        values_local.push_back(v[0]);
        row_local.push_back(r[0]);
        col_local.push_back(c[0]);

        //vElement_local.push_back(col[0]);
        vElementRep_local.push_back(1);

    } else{
        nnz_l_remote++;

        values_remote.push_back(v[0]);
        row_remote.push_back(r[0]);
        col_remote_size++;
        col_remote.push_back(col_remote_size);
//        nnz_row_remote[col_remote_size]++;
        nnz_row_remote.push_back(1);

        vElement_remote.push_back(c[0]);
        vElementRep_remote.push_back(1);
        recvCount[lower_bound2(&split[0], &split[nprocs], c[0])] = 1;
    }

    for (i = 1; i < nnz_l; i++) {

        if (c[i] >= split[rank] && c[i] < split[rank+1]) {
            nnz_row_local[r[i]-split[rank]]++;
            nnz_l_local++;

            values_local.push_back(v[i]);
            row_local.push_back(r[i]);
            col_local.push_back(c[i]);

//            if (col[i] != col[i - 1]) {
//                vElementRep_local.push_back(1);
//            } else {
//                (*(vElementRep_local.end()-1))++;
//            }
        } else {
            nnz_l_remote++;
            values_remote.push_back(v[i]);
            row_remote.push_back(r[i]);
            // col_remote2 is the original col value and will be used in making strength matrix. col_remote will be used for matevec.
            col_remote2.push_back(c[i]);

            if (c[i] != c[i - 1]) {
                col_remote_size++;
                vElement_remote.push_back(c[i]);
                vElementRep_remote.push_back(1);
                procNum = lower_bound2(&split[0], &split[nprocs], c[i]);
                recvCount[procNum]++;
                nnz_row_remote.push_back(1);
            } else {
                (*(vElementRep_remote.end()-1))++;
                (*(nnz_row_remote.end()-1))++;
            }
            // the original col values are not being used. the ordering starts from 0, and goes up by 1.
            col_remote.push_back(col_remote_size);

//            nnz_row_remote[col_remote_size]++;
        }
    } // for i
    // since col_remote_size starts from -1
    col_remote_size++;
    recvCount[rank] = 0;

    int vIndexCount[nprocs];
    MPI_Alltoall(recvCount, 1, MPI_INT, vIndexCount, 1, MPI_INT, MPI_COMM_WORLD);

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
    for (unsigned int i=0; i<row_local.size(); i++){
        row_local[i] -= split[rank];
        col_local[i] -= split[rank];
    }
    for (unsigned int i=0; i<row_remote.size(); i++){
        row_remote[i] -= split[rank];
//        col_remote[i] -= split[rank];
    }

    // vSend = vector values to send to other procs
    // vecValues = vector values that received from other procs
    // These will be used in matvec and they are set here to reduce the time of matvec.
    vSend = (long*)malloc(sizeof(long) * vIndexSize);
    vecValues = (long*) malloc(sizeof(long) * recvSize);



    indicesP_local = (long*)malloc(sizeof(long)*nnz_l_local);
    for(i=0; i<nnz_l_local; i++)
        indicesP_local[i] = i;
    long* row_localP = &(*(row_local.begin()));
    std::sort(indicesP_local, &indicesP_local[nnz_l_local], sort_indices(row_localP));

    indicesP_remote = (long*)malloc(sizeof(long)*nnz_l_remote);
    for(i=0; i<nnz_l_remote; i++)
        indicesP_remote[i] = i;
    long* row_remoteP = &(*(row_remote.begin()));
    std::sort(indicesP_remote, &indicesP_remote[nnz_l_remote], sort_indices(row_remoteP));

    return 0;
}

CSRMatrix::~CSRMatrix(){
//    rowIndex.resize(0);
//    col.resize(0);
//    values.resize(0);
}

void CSRMatrix::print(int r){
//    int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//
//    if (rank==r)
//        for(unsigned int i=0; i<M; i++){
//            for(long j=rowIndex[i]; j<rowIndex[i+1]; j++)
//                cout << "[" << i+1 << ",\t" << col[j]+1 << "] = \t" << values[j] << endl;
//        }
}