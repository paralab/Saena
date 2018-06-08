#include "prolong_matrix.h"
//#include "aux_functions.h"

#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <omp.h>
#include "mpi.h"


prolong_matrix::prolong_matrix(){}


prolong_matrix::prolong_matrix(MPI_Comm com){
    comm = com;
}


prolong_matrix::~prolong_matrix(){
//    if(arrays_defined){
//        free(vIndex);
//        free(vSend);
//        free(vecValues);
//        free(indicesP_local);
//        free(indicesP_remote);
//        free(vSend_t);
//        free(vecValues_t);
//       free(recvIndex_t); // recvIndex_t is equivalent of vIndex.
//    }
}


int prolong_matrix::findLocalRemote(){


    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    arrays_defined = true;

//    printf("rank=%d \t P.nnz_l=%lu \t P.nnz_g=%lu \n", rank, nnz_l, nnz_g);

//    print_vector(entry, 0, "entry", comm);

    long procNum;
    col_remote_size = 0; // number of remote columns
    nnz_l_local = 0;
    nnz_l_remote = 0;
    int* recvCount = (int*)malloc(sizeof(int)*nprocs);
    std::fill(recvCount, recvCount + nprocs, 0);
//    int* recvCount_t = (int*)malloc(sizeof(int)*nprocs);
//    std::fill(recvCount_t, recvCount_t + nprocs, 0);
    nnzPerRow_local.assign(M,0);

    entry_local.clear();
    entry_remote.clear();
    row_local.clear();
    row_remote.clear();
    col_remote.clear();
    vElementRep_local.clear();
    vElement_remote.clear();
    vElement_remote_t.clear();
    vElementRep_remote.clear();
    nnzPerCol_remote.clear();

    int* vIndexCount_t = (int*)malloc(sizeof(int)*nprocs);
    std::fill(vIndexCount_t, vIndexCount_t + nprocs, 0);

    //todo: here: change push_back
    // take care of the first element here, since there is "col[i-1]" in the for loop below, so "i" cannot start from 0.
    // local
    if (entry[0].col >= splitNew[rank] && entry[0].col < splitNew[rank + 1]) {
        nnzPerRow_local[entry[0].row]++;
        nnz_l_local++;
        entry_local.push_back(entry[0]);
        row_local.push_back(entry[0].row); // only for sorting at the end of prolongMatrix::findLocalRemote. then clear the vector.
//        col_local.push_back(entry[0].col);
//        values_local.push_back(entry[0].val);
        //vElement_local.push_back(col[0]);
        vElementRep_local.push_back(1);

    // remote
    } else{
        nnz_l_remote++;
        entry_remote.push_back(entry[0]);
        row_remote.push_back(entry[0].row); // only for sorting at the end of prolongMatrix::findLocalRemote. then clear the vector.
//        col_remote2.push_back(entry[0].col);
//        values_remote.push_back(entry[0].val);
        col_remote_size++; // number of remote columns
        col_remote.push_back(col_remote_size-1);
//        nnzPerCol_remote[col_remote_size-1]++;
        nnzPerCol_remote.push_back(1);

        vElement_remote.push_back(entry[0].col);
        vElementRep_remote.push_back(1);
        recvCount[lower_bound2(&splitNew[0], &splitNew[nprocs], entry[0].col)] = 1;

//        nnzPerCol_remote_t.push_back(1);
        vElement_remote_t.push_back(nnz_l_remote-1);
        vIndexCount_t[lower_bound2(&splitNew[0], &splitNew[nprocs], entry[0].col)] = 1;
//        recvCount_t[lower_bound2(&splitNew[0], &splitNew[nprocs], entry[0].col)] = 1;
    }

    for (nnz_t i = 1; i < nnz_l; i++) {

        // local
        if (entry[i].col >= splitNew[rank] && entry[i].col < splitNew[rank+1]) {
            nnzPerRow_local[entry[i].row]++;
            nnz_l_local++;
            entry_local.push_back(entry[i]);
            row_local.push_back(entry[i].row); // only for sorting at the end of prolongMatrix::findLocalRemote. then clear.
//            col_local.push_back(entry[i].col);
//            values_local.push_back(entry[i].val);
            if (entry[i].col != entry[i-1].col)
                vElementRep_local.push_back(1);
            else
                (*(vElementRep_local.end()-1))++;

        // remote
        } else {
            nnz_l_remote++;
//            if(rank==2) printf("entry[i].row = %lu\n", entry[i].row+split[rank]);
            entry_remote.push_back(entry[i]);
            row_remote.push_back(entry[i].row); // only for sorting at the end of prolongMatrix::findLocalRemote. then clear the vector.
            // col_remote2 is the original col value. col_remote starts from 0.
//            col_remote2.push_back(entry[i].col);
//            values_remote.push_back(entry[i].val);
            procNum = lower_bound2(&splitNew[0], &splitNew[nprocs], entry[i].col);
            vIndexCount_t[procNum]++;
//            recvCount_t[procNum]++;
            vElement_remote_t.push_back((index_t)nnz_l_remote-1); // todo: is (unsigned long) required here?
//            nnzPerCol_remote_t.push_back(1);

            if (entry[i].col != entry[i-1].col) {
                col_remote_size++;
                vElement_remote.push_back(entry[i].col);
                vElementRep_remote.push_back(1);
                procNum = lower_bound2(&splitNew[0], &splitNew[nprocs], entry[i].col);
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

//    MPI_Barrier(comm); printf("rank=%d, P.nnz_l=%lu, P.nnz_l_local=%u, P.nnz_l_remote=%u \n", rank, nnz_l, nnz_l_local, nnz_l_remote); MPI_Barrier(comm);

    nnzPerRowScan_local.assign(M+1, 0);
    for(index_t i=0; i<M; i++){
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

    recvProcRank.clear();
    recvProcCount.clear();
    sendProcRank.clear();
    sendProcCount.clear();

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
    vIndex.resize(vIndexSize);
    MPI_Alltoallv(&*vElement_remote.begin(), recvCount, &*rdispls.begin(), MPI_UNSIGNED,
                  &vIndex[0], vIndexCount, &*vdispls.begin(), MPI_UNSIGNED, comm);

    free(vIndexCount);
    free(recvCount);

    recvProcRank_t.clear();
    recvProcCount_t.clear();
    sendProcRank_t.clear();
    sendProcCount_t.clear();

    numRecvProc_t = 0;
    numSendProc_t = 0;
    for(int i = 0; i < nprocs; i++){
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
    for (index_t i=0; i<vIndexSize; i++){
        vIndex[i] -= splitNew[rank];
    }

    // vSend = vector values to send to other procs
    // vecValues = vector values that received from other procs
    // These will be used in matvec and they are set here to reduce the time of matvec.
    vSend.resize(vIndexSize);
    vecValues.resize(recvSize);

    vSend_t.resize(vIndexSize_t);
    vecValues_t.resize(recvSize_t);

    // todo: change the following two parts the same as indicesP for A in coarsen, which is using entry, instead of row_local and row_remote.
    indicesP_local.resize(nnz_l_local);
    for(nnz_t i=0; i<nnz_l_local; i++)
        indicesP_local[i] = i;
    index_t *row_localP = &*row_local.begin();
    std::sort(&indicesP_local[0], &indicesP_local[nnz_l_local], sort_indices(row_localP)); // todo: is it ordered only row-wise?
    row_local.clear();
    row_local.shrink_to_fit();

//    long start;
//    for(i = 0; i < M; ++i) {
//        start = nnzPerRowScan_local[i];
//        for(long j=0; j < nnzPerRow_local[i]; j++){
//            if(rank==1) printf("%lu \t %lu \t %f \n", entry_local[indicesP_local[start + j]].row+split[rank], entry_local[indicesP_local[start + j]].col, entry_local[indicesP_local[start + j]].val);
//        }
//    }

    indicesP_remote.resize(nnz_l_remote);
    for(nnz_t i=0; i<nnz_l_remote; i++)
        indicesP_remote[i] = i;
    index_t* row_remoteP = &*row_remote.begin();
    std::sort(&indicesP_remote[0], &indicesP_remote[nnz_l_remote], sort_indices(row_remoteP));
    // todo: is this required?
//    row_remote.clear();
//    row_remote.shrink_to_fit();

//    MPI_Barrier(comm);
//    if(rank==1) cout << "nnz_l_remote = " << nnz_l_remote << "\t\trecvSize_t = " << recvSize_t << "\t\tvIndexSize_t = " << vIndexSize_t << endl;
//    if(rank==0){
//        for(i=0; i<nnz_l_remote; i++)
//            cout << row_remote[i] << "\t" << col_remote2[i] << " =\t" << values_remote[i] << "\t\t\t" << vElement_remote_t[i] << endl;
//    }
//    MPI_Barrier(comm);

    openmp_setup();
    w_buff.resize(num_threads*M); // allocate for w_buff for matvec

    return 0;
}


int prolong_matrix::openmp_setup() {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(verbose_prolong_setup) {
        MPI_Barrier(comm);
        printf("matrix_setup: rank = %d, thread1 \n", rank);
        MPI_Barrier(comm);
    }

//    printf("matrix_setup: rank = %d, Mbig = %u, M = %u, nnz_g = %u, nnz_l = %u, nnz_l_local = %u, nnz_l_remote = %u \n", rank, Mbig, M, nnz_g, nnz_l, nnz_l_local, nnz_l_remote);

#pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }

    iter_local_array.resize(num_threads+1);
    iter_remote_array.resize(num_threads+1);

#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
//        if(rank==0 && thread_id==0) std::cout << "number of procs = " << nprocs << ", number of threads = " << num_threads << std::endl;
        index_t istart = 0; // starting row index for each thread
        index_t iend = 0;   // last row index for each thread
        index_t iter_local, iter_remote;

        // compute local iter to do matvec using openmp (it is done to make iter independent data on threads)
        bool first_one = true;
#pragma omp for
        for (index_t i = 0; i < M; ++i) {
            if (first_one) {
                istart = i;
                first_one = false;
                iend = istart;
            }
            iend++;
        }
//        if(rank==1) printf("thread id = %d, istart = %u, iend = %u \n", thread_id, istart, iend);

        iter_local = 0;
        for (index_t i = istart; i < iend; ++i)
            iter_local += nnzPerRow_local[i];

        iter_local_array[0] = 0;
        iter_local_array[thread_id + 1] = iter_local;

        // compute remote iter to do matvec using openmp (it is done to make iter independent data on threads)
        first_one = true;
#pragma omp for
        for (index_t i = 0; i < col_remote_size; ++i) {
            if (first_one) {
                istart = i;
                first_one = false;
                iend = istart;
            }
            iend++;
        }

        iter_remote = 0;
        if (!nnzPerCol_remote.empty()) {
            for (index_t i = istart; i < iend; ++i)
                iter_remote += nnzPerCol_remote[i];
        }

        iter_remote_array[0] = 0;
        iter_remote_array[thread_id + 1] = iter_remote;

//        if (rank==1 && thread_id==0){
//            std::cout << "M=" << M << std::endl;
//            std::cout << "recvSize=" << recvSize << std::endl;
//            std::cout << "istart=" << istart << std::endl;
//            std::cout << "iend=" << iend << std::endl;
//            std::cout  << "nnz_l=" << nnz_l << ", iter_remote=" << iter_remote << ", iter_local=" << iter_local << std::endl;}

    } // end of omp parallel

    if(verbose_prolong_setup) {
        MPI_Barrier(comm);
        printf("matrix_setup: rank = %d, thread2 \n", rank);
        MPI_Barrier(comm);
    }

    //scan of iter_local_array
    for (int i = 1; i < num_threads + 1; i++)
        iter_local_array[i] += iter_local_array[i - 1];

    //scan of iter_remote_array
    for (int i = 1; i < num_threads + 1; i++)
        iter_remote_array[i] += iter_remote_array[i - 1];

//    print_vector(iter_local_array, 0, "iter_local_array", comm);
//    print_vector(iter_remote_array, 0, "iter_remote_array", comm);

    return 0;
}


int prolong_matrix::matvec(std::vector<value_t>& v, std::vector<value_t>& w) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    totalTime = 0;
//    double t10 = MPI_Wtime();

    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(index_t i = 0;i < vIndexSize;i++)
        vSend[i] = v[(vIndex[i])];
//    double t20 = MPI_Wtime();
//    time[0] += (t20-t10);

//    print_vector(vSend, 0, "vSend", comm);

//    double t13 = MPI_Wtime();
    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses = new MPI_Status[numSendProc+numRecvProc];

    //First place all recv requests. Do not recv from self.
    for(int i = 0; i < numRecvProc; i++)
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));

    //Next send the messages. Do not send to self.
    for(int i = 0; i < numSendProc; i++)
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));

    // local loop
    // ----------
//    double t11 = MPI_Wtime();
    value_t* v_p = &v[0] - splitNew[rank];

    #pragma omp parallel
    {
        nnz_t iter = iter_local_array[omp_get_thread_num()];
//        nnz_t iter = 0;
        #pragma omp for
            for (index_t i = 0; i < M; ++i) {
                w[i] = 0;
                for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
    //                if(rank==1) cout << entry_local[indicesP_local[iter]].col - splitNew[rank] << "\t" << v[entry_local[indicesP_local[iter]].col - splitNew[rank]] << endl;
    //                w[i] += entry_local[indicesP_local[iter]].val * v[entry_local[indicesP_local[iter]].col - splitNew[rank]];
                    w[i] += entry_local[indicesP_local[iter]].val * v_p[entry_local[indicesP_local[iter]].col];
                }
            }
    }

//    double t21 = MPI_Wtime();
//    time[1] += (t21-t11);

    // Wait for comm to finish.
    MPI_Waitall(numRecvProc, requests, statuses);

    // remote loop
    // -----------

/*
//    double t12 = MPI_Wtime();
//#pragma omp parallel
//    {
//        nnz_t iter = iter_remote_array[omp_get_thread_num()];
        nnz_t iter = 0;
//#pragma omp for
        for (index_t i = 0; i < col_remote_size; ++i) {
            for (index_t j = 0; j < nnzPerCol_remote[i]; ++j, ++iter) {
                w[row_remote[indicesP_remote[iter]]] += entry_remote[indicesP_remote[iter]].val * vecValues[col_remote[indicesP_remote[iter]]];
            }
        }
//    }
*/

#pragma omp parallel
    {
        unsigned int i, l;
        int thread_id = omp_get_thread_num();
        value_t *w_local = &w_buff[0] + (thread_id*M);
        if(thread_id==0)
            w_local = &*w.begin();
        else
            std::fill(&w_local[0], &w_local[M], 0);

        nnz_t iter = iter_remote_array[thread_id];
#pragma omp for
        for (index_t j = 0; j < col_remote_size; ++j) {
            for (i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
                w_local[row_remote[iter]] += entry_remote[iter].val * vecValues[j];

//                if(rank==0 && thread_id==0){
//                    printf("thread = %d\n", thread_id);
//                    printf("%u \t%u \tind_rem = %lu, row = %lu \tcol = %lu \tvecVal = %f \n",
//                           i, j, indicesP_remote[iter], row_remote[indicesP_remote[iter]],
//                           col_remote[indicesP_remote[iter]], vecValues[col_remote[indicesP_remote[iter]]]);}
            }
        }

        int thread_partner;
        int levels = (int)ceil(log2(num_threads));
        for (l = 0; l < levels; l++) {
            if (thread_id % int(pow(2, l+1)) == 0) {
                thread_partner = thread_id + int(pow(2, l));
//                printf("l = %d, levels = %d, thread_id = %d, thread_partner = %d \n", l, levels, thread_id, thread_partner);
                if(thread_partner < num_threads){
                    for (i = 0; i < M; i++)
                        w_local[i] += w_buff[thread_partner * M + i];
                }
            }
#pragma omp barrier
        }
    }

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    delete [] requests;
    delete [] statuses;

//    double t22 = MPI_Wtime();
//    time[2] += (t22-t12);
//    double t23 = MPI_Wtime();
//    time[3] += (t23-t13);

    return 0;
}


int prolong_matrix::print(int ran){

    // if ran >= 0 print the matrix entries on proc with rank = ran
    // otherwise print the matrix entries on all processors in order. (first on proc 0, then proc 1 and so on.)

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(ran >= 0) {
        if (rank == ran) {
            printf("\nmatrix on proc = %d \n", ran);
            printf("nnz = %lu \n", nnz_l);
            for (auto i:entry)
                std::cout << i << std::endl;
        }
    } else{
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("\nmatrix on proc = %d \n", proc);
                printf("nnz = %lu \n", nnz_l);
                for (auto i:entry)
                    std::cout << i << std::endl;
            }
            MPI_Barrier(comm);
        }
    }

    return 0;
}