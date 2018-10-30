#include "strength_matrix.h"
#include "aux_functions.h"

#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cstring>
#include "mpi.h"


/*
// sort indices and store the ordering.
class sort_indices
{
private:
    index_t *mparr;
public:
    sort_indices(index_t *parr) : mparr(parr) {}
    bool operator()(index_t i, index_t j) const { return mparr[i]<mparr[j]; }
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
*/


int strength_matrix::set_parameters(index_t m1, index_t m2, std::vector<index_t> &spl, MPI_Comm com){
    M = m1;
    Mbig = m2;
    split = spl;
    comm = com;
    return 0;
}


int strength_matrix::setup_matrix(float connStrength){

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // *************************** make S symmetric and apply the connection strength parameter ****************************

    nnz_l = 0;
    int val_coeff, valT_coeff;
    std::vector<index_t> r(entry.size()), c(entry.size());
    std::vector<value_t> v(entry.size());

    for(nnz_t i=0; i<entry.size(); i++){

        if (entry[i].val <= connStrength)
            val_coeff = 0;
        else
            val_coeff = 1;

        if (entryT[i].val <= connStrength)
            valT_coeff = 0;
        else
            valT_coeff = 1;

        if (!val_coeff && !valT_coeff) // if both zero, skip.
            continue;

        r[nnz_l] = entryT[i].row;
        c[nnz_l] = entryT[i].col;
        v[nnz_l] = 0.5*(val_coeff * entry[i].val + valT_coeff * entryT[i].val);
        nnz_l++;
    }

    r.resize(nnz_l);
    c.resize(nnz_l);
    v.resize(nnz_l);
    r.shrink_to_fit();
    c.shrink_to_fit();
    v.shrink_to_fit();

    MPI_Allreduce(&nnz_l, &nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
//    if(rank==0) printf("S.nnz_l = %lu, S.nnz_g = %lu \n", nnz_l, nnz_g);

//    print_vector(r, -1, "row vector", comm);
//    print_vector(c, -1, "col vector", comm);
//    print_vector(v, -1, "val vector", comm);
//    if(rank==1){
//        for(index_t i = 0; i < r.size(); i++){
//            printf("%u \t%u \t%u \t%lf \n", i, r[i], c[i], v[i]);
//        }
//    }

    // *************************** setup the matrix ****************************

    long procNum;
    unsigned long i;
    col_remote_size = 0; // number of remote columns
    nnz_l_local = 0;
    nnz_l_remote = 0;
    std::vector<int> recvCount(nprocs, 0);
    nnzPerRow.assign(M,0);
    nnzPerRow_local.assign(M,0);

    // take care of the first element here, since there is "col[i-1]" in the for loop below, so "i" cannot start from 0.
    nnzPerRow[r[0]]++;
    if (c[0] >= split[rank] && c[0] < split[rank + 1]) {
        nnzPerRow_local[r[0]]++;
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
        col_remote_size++; // number of remote columns
        col_remote.push_back(col_remote_size-1);
        col_remote2.push_back(c[0]);
        nnzPerCol_remote.push_back(1);

        vElement_remote.push_back(c[0]);
        vElementRep_remote.push_back(1);
        recvCount[lower_bound2(&split[0], &split[nprocs], c[0])] = 1;
    }

    for (i = 1; i < nnz_l; i++) {
        nnzPerRow[r[i]]++;

        if (c[i] >= split[rank] && c[i] < split[rank+1]) {
            nnzPerRow_local[r[i]]++;
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
                nnzPerCol_remote.push_back(1);
            } else {
                vElementRep_remote.back()++;
                nnzPerCol_remote.back()++;
            }
            // the original col values are not being used for matvec. the ordering starts from 0, and goes up by 1.
            col_remote.push_back(col_remote_size-1);
//            nnzPerCol_remote[col_remote_size-1]++;
        }
    } // for i

     if(nprocs !=1){
         std::vector<int> vIndexCount(nprocs);
         MPI_Alltoall(&recvCount[0], 1, MPI_INT, &vIndexCount[0], 1, MPI_INT, comm);

         numRecvProc = 0;
         numSendProc = 0;
         for(int i=0; i<nprocs; i++){
             if(recvCount[i]!=0){
                 numRecvProc++;
                 recvProcRank.push_back(i);
                 recvProcCount.push_back(2*recvCount[i]); // make them double size for prolongation the communication in the aggregation_2_dist function.
             }
             if(vIndexCount[i]!=0){
                 numSendProc++;
                 sendProcRank.push_back(i);
                 sendProcCount.push_back(2*vIndexCount[i]); // make them double size for prolongation the communication in the aggregation_2_dist function.
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

         vIndex.resize(vIndexSize);
         MPI_Alltoallv(&*vElement_remote.begin(), &recvCount[0], &*rdispls.begin(), MPI_UNSIGNED,
                       &vIndex[0], &vIndexCount[0], &*vdispls.begin(), MPI_UNSIGNED, comm);

         // vSend = vector values to send to other procs
         // vecValues = vector values that received from other procs
         // These will be used in matvec and they are set here to reduce the time of matvec.
         vSend.resize(2*vIndexSize); // make them double size for prolongation the communication in the aggregation_2_dist function.
         vecValues.resize(2*recvSize); // make them double size for prolongation the communication in the aggregation_2_dist function.

         // make them double size for prolongation the communication in the aggregation_2_dist function.
         for (int i=1; i<nprocs; i++){
             vdispls[i] = 2*vdispls[i];
             rdispls[i] = 2*rdispls[i];
         }

//         print_vector(vIndex, -1, "vIndex", comm);

         // change the indices from global to local
         #pragma omp parallel for
         for (unsigned int i=0; i<vIndexSize; i++)
             vIndex[i] -= split[rank];
    }

    indicesP_local.resize(nnz_l_local);
    for(i=0; i<nnz_l_local; i++)
        indicesP_local[i] = i;

    index_t *row_localP = &*row_local.begin();
    std::sort(&indicesP_local[0], &indicesP_local[nnz_l_local], sort_indices(row_localP));

//    indicesP_remote = (unsigned long*)malloc(sizeof(unsigned long)*nnz_l_remote);
//    for(i=0; i<nnz_l_remote; i++)
//        indicesP_remote[i] = i;
//    unsigned long* row_remoteP = &(*(row_remote.begin()));
//    std::sort(indicesP_remote, &indicesP_remote[nnz_l_remote], sort_indices(row_remoteP));

//    print_info(-1);

    return 0;
}


strength_matrix::~strength_matrix() = default;


int strength_matrix::erase(){
    M    = 0;
    Mbig = 0;
    nnz_l  = 0;
    nnz_l_local  = 0;
    nnz_l_remote = 0;
    col_remote_size = 0;
    vIndexSize = 0;
    recvSize = 0;
    numRecvProc = 0;
    numSendProc = 0;

    vIndex.clear();
    vSend.clear();
    vecValues.clear();
    split.clear();
    values_local.clear();
    values_remote.clear();
    row_local.clear();
    row_remote.clear();
    col_local.clear();
    col_remote.clear();
    col_remote2.clear();
    nnzPerRow.clear();
    nnzPerRow_local.clear();
    nnzPerCol_remote.clear();
    vElement_remote.clear();
    vElementRep_local.clear();
    vElementRep_remote.clear();
    indicesP_local.clear();
    indicesP_remote.clear();
    vdispls.clear();
    rdispls.clear();
    recvProcRank.clear();
    recvProcCount.clear();
    sendProcRank.clear();
    sendProcCount.clear();
    return 0;
}


int strength_matrix::erase_update(){
//    M    = 0;
//    Mbig = 0;
    nnz_l  = 0;
    nnz_l_local  = 0;
    nnz_l_remote = 0;
    col_remote_size = 0;
    vIndexSize = 0;
    recvSize = 0;
    numRecvProc = 0;
    numSendProc = 0;

//    split.clear();
    vIndex.clear();
    vSend.clear();
    vecValues.clear();
    values_local.clear();
    values_remote.clear();
    row_local.clear();
    row_remote.clear();
    col_local.clear();
    col_remote.clear();
    col_remote2.clear();
    nnzPerRow.clear();
    nnzPerRow_local.clear();
    nnzPerCol_remote.clear();
    vElement_remote.clear();
    vElementRep_local.clear();
    vElementRep_remote.clear();
    indicesP_local.clear();
    indicesP_remote.clear();
    vdispls.clear();
    rdispls.clear();
    recvProcRank.clear();
    recvProcCount.clear();
    sendProcRank.clear();
    sendProcCount.clear();
    return 0;
}


void strength_matrix::print_entry(int ran){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::vector<cooEntry> entry(nnz_l);
    for(index_t i = 0; i < nnz_l_local; i++){
        entry[i] = cooEntry(row_local[i]+split[rank], col_local[i], values_local[i]);
    }

    for(index_t i = 0; i < nnz_l_remote; i++){
        entry[nnz_l_local+i] = cooEntry(row_remote[i]+split[rank], col_remote2[i], values_remote[i]);
    }

    std::sort(entry.begin(), entry.end());

    index_t iter = 0;
    if(ran >= 0) {
        if (rank == ran) {
            printf("\nstrength matrix (diagonal_block) on proc = %d \n", ran);
            printf("nnz = %lu \n", nnz_l);
            for (index_t i = 0; i < nnz_l; i++) {
                std::cout << iter << "\t" << entry[i] << std::endl;
                iter++;
            }
        }
    } else{
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("\nstrength matrix (diagonal_block) on proc = %d \n", rank);
                printf("nnz = %lu \n", nnz_l);
                for (index_t i = 0; i < nnz_l; i++) {
                    std::cout << iter << "\t" << entry[i] << std::endl;
                    iter++;
                }
            }
            MPI_Barrier(comm);
        }
    }
}


void strength_matrix::print_diagonal_block(int ran){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    index_t iter = 0;
    if(ran >= 0) {
        if (rank == ran) {
            printf("\nstrength matrix (diagonal_block) on proc = %d \n", ran);
            printf("nnz = %lu \n", nnz_l_local);
            for (index_t i = 0; i < nnz_l_local; i++) {
                std::cout << iter << "\t" << row_local[i] + split[rank] << "\t" << col_local[i] << "\t" << values_local[i] << std::endl;
                iter++;
            }
        }
    } else{
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("\nstrength matrix (diagonal_block) on proc = %d \n", rank);
                printf("nnz = %lu \n", nnz_l_local);
                for (index_t i = 0; i < nnz_l_local; i++) {
                    std::cout << iter << "\t" << row_local[i] + split[rank] << "\t" << col_local[i] << "\t" << values_local[i] << std::endl;
                    iter++;
                }
            }
            MPI_Barrier(comm);
        }
    }
}


void strength_matrix::print_off_diagonal(int ran){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    index_t iter = 0;
    if(ran >= 0) {
        if (rank == ran) {
            printf("\nstrength matrix (off_diagonal) on proc = %d \n", ran);
            printf("nnz = %lu \n", nnz_l_remote);
            for (index_t i = 0; i < nnz_l_remote; i++) {
                std::cout << iter << "\t" << row_remote[i] + split[rank] << "\t" << col_remote2[i] << "\t" << values_remote[i] << std::endl;
                iter++;
            }
        }
    } else{
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("\nstrength matrix (off_diagonal) on proc = %d \n", rank);
                printf("nnz = %lu \n", nnz_l_remote);
                for (index_t i = 0; i < nnz_l_remote; i++) {
                    std::cout << iter << "\t" << row_remote[i] + split[rank] << "\t" << col_remote2[i] << "\t" << values_remote[i] << std::endl;
                    iter++;
                }
            }
            MPI_Barrier(comm);
        }
    }
}


int strength_matrix::print_info(int ran){

    // if ran >= 0 print the matrix entries on proc with rank = ran
    // otherwise print the matrix entries on all processors in order. (first on proc 0, then proc 1 and so on.)

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(ran >= 0) {
        if (rank == ran) {
            printf("\nmatrix S info on proc = %d \n", ran);
            printf("Mbig = %u, \tM = %u, \tnnz_g = %lu, \tnnz_l = %lu \n", Mbig, M, nnz_g, nnz_l);
        }
    } else{
        MPI_Barrier(comm);
        if(rank==0) printf("\nmatrix S info:      Mbig = %u, \tnnz_g = %lu \n", Mbig, nnz_g);
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("matrix S on rank %d: M    = %u, \tnnz_l = %lu \n", proc, M, nnz_l);
            }
            MPI_Barrier(comm);
        }
    }

    return 0;
}


int strength_matrix::save_to_disk(){

    for(nnz_t i = 0; i < nnz_l; i++){

    }

    return 0;
}