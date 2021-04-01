#include "saena_matrix.h"

// the following defines the reduction operation on a vector in OpenMP.
// this is used in one of the matvec implementations (matvec_timing1).
//#pragma omp declare reduction(vec_double_plus : std::vector<value_t> : \
//                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<value_t>())) \
//                    initializer(omp_priv = omp_orig)

void saena_matrix::matvec_sparse(std::vector<value_t>& v, std::vector<value_t>& w) {
    // combination of openmp and waitany

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    print_info(-1);
//    print_vector(v, -1, "v", comm);

    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
#pragma omp parallel for
    for(index_t i = 0; i < vIndexSize; ++i)
        vSend[i] = v[vIndex[i]];

//    print_vector(vSend, 1, "vSend", comm);

    // receive and put the remote parts of v in vecValues.
    // they are received in order: first put the values from the lowest rank matrix, and so on.
    for(int i = 0; i < numRecvProc; ++i){
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], par::Mpi_datatype<value_t>::value(), recvProcRank[i], 1, comm, &requests[i]);
//        MPI_Test(&requests[i], &MPI_flag, &statuses[i]);
    }

    MPI_Request *requests_p = &requests[numRecvProc];
    for(int i = 0; i < numSendProc; ++i){
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], par::Mpi_datatype<value_t>::value(), sendProcRank[i], 1, comm, &requests_p[i]);
        MPI_Test(&requests_p[i], &MPI_flag, MPI_STATUSES_IGNORE);
    }

    // initialize w to 0
    fill(w.begin(), w.end(), 0);

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

#pragma omp parallel
    {
        value_t  tmp            = 0;
        value_t* v_p            = &v[0] - split[rank];
        index_t* col_local_p    = nullptr;
        value_t* values_local_p = nullptr;
        nnz_t    iter           = iter_local_array[omp_get_thread_num()];
#pragma omp for
        for (index_t i = 0; i < M; ++i) {
            col_local_p    = &col_local[iter];
            values_local_p = &values_local[iter];
            const index_t jend = nnzPerRow_local[i];
            tmp = 0;
            for (index_t j = 0; j < jend; ++j) {
//                if(rank==0) printf("%u \t%u \t%f \t%f \t%f \n", row_local[indicesP_local[iter]], col_local[indicesP_local[iter]], values_local[indicesP_local[iter]], v_p[col_local[indicesP_local[iter]]], values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]]);
                tmp += values_local_p[j] * v_p[col_local_p[j]];
            }
            w[i] += tmp;
            iter += jend;
        }
    }

    index_t* row_remote_p = nullptr;
    value_t* val_remote_p = nullptr;
    nnz_t iter = 0;
    int recv_proc_idx = 0;
    for(int np = 0; np < numRecvProc; ++np){
        MPI_Waitany(numRecvProc, &requests[0], &recv_proc_idx, MPI_STATUS_IGNORE);
        const int recv_proc = recvProcRank[recv_proc_idx];
//        if(rank==1) printf("recv_proc_idx = %d, recv_proc = %d, np = %d, numRecvProc = %d, recvCount[recv_proc] = %d\n",
//                              recv_proc_idx, recv_proc, np, numRecvProc, recvCount[recv_proc]);

        iter = nnzPerProcScan[recv_proc];
        value_t *vecValues_p        = &vecValues[rdispls[recv_proc]];
        auto    *nnzPerCol_remote_p = &nnzPerCol_remote[rdispls[recv_proc]];
        for (index_t j = 0; j < recvCount[recv_proc]; ++j) {
//            if(rank==1) printf("%u\n", nnzPerCol_remote_p[j]);
            row_remote_p = &row_remote[iter];
            val_remote_p = &values_remote[iter];
            const index_t iend = nnzPerCol_remote_p[j];
            const value_t vrem = vecValues_p[j];
#pragma omp simd
            for (index_t i = 0; i < iend; ++i) {
//                if(rank==1) printf("%ld \t%u \t%u \t%f \t%f\n",
//                iter, row_remote[iter], col_remote2[iter], values_remote[iter], vecValues[rdispls[recv_proc] + j]);
                w[row_remote_p[i]] += val_remote_p[i] * vrem;
            }
            iter += iend;
        }
    }

    MPI_Waitall(numSendProc, &requests[numRecvProc], MPI_STATUSES_IGNORE);
}

void saena_matrix::matvec_sparse2(std::vector<value_t>& v, std::vector<value_t>& w) {
    // with waitany, no openmp

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");
//    print_info(-1);
//    print_vector(v, -1, "v", comm);

//    if(nprocs > 1){
    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
    for(index_t i = 0; i < vIndexSize; ++i)
        vSend[i] = v[vIndex[i]];

//    print_vector(vSend, 1, "vSend", comm);

    // receive and put the remote parts of v in vecValues.
    // they are received in order: first put the values from the lowest rank matrix, and so on.
    for(int i = 0; i < numRecvProc; ++i){
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], par::Mpi_datatype<value_t>::value(), recvProcRank[i], 1, comm, &requests[i]);
//        MPI_Test(&requests[i], &MPI_flag, &statuses[i]);
    }

    MPI_Request *requests_p = &requests[numRecvProc];
    for(int i = 0; i < numSendProc; ++i){
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], par::Mpi_datatype<value_t>::value(), sendProcRank[i], 1, comm, &requests_p[i]);
//        MPI_Test(&requests[numRecvProc + i], &MPI_flag, &statuses[numRecvProc + i]);
        MPI_Test(&requests_p[i], &MPI_flag, MPI_STATUS_IGNORE);
    }

//    }

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

    value_t* v_p  = &v[0] - split[rank];
    nnz_t    iter = 0;
    for (index_t i = 0; i < M; ++i) { // rows
        w[i] = 0;
        for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) { // columns
//            if(rank==0) printf("%u \t%u \t%f \t%f \t%f \n", row_local[indicesP_local[iter]], col_local[indicesP_local[iter]], values_local[indicesP_local[iter]], v_p[col_local[indicesP_local[iter]]], values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]]);
            w[i] += values_local[iter] * v_p[col_local[iter]];
        }
    }

    int np = 0;
    int recv_proc = 0, recv_proc_idx = 0;
    while(np < numRecvProc){
        MPI_Waitany(numRecvProc, &requests[0], &recv_proc_idx, MPI_STATUS_IGNORE);
        ++np;

        recv_proc = recvProcRank[recv_proc_idx];
//        if(rank==1) printf("recv_proc_idx = %d, recv_proc = %d, np = %d, numRecvProc = %d, recvCount[recv_proc] = %d\n",
//                              recv_proc_idx, recv_proc, np, numRecvProc, recvCount[recv_proc]);

        iter = nnzPerProcScan[recv_proc];
        value_t *vecValues_p        = &vecValues[rdispls[recv_proc]];
        auto    *nnzPerCol_remote_p = &nnzPerCol_remote[rdispls[recv_proc]];
        for (index_t j = 0; j < recvCount[recv_proc]; ++j) {
//            if(rank==1) printf("%u\n", nnzPerCol_remote_p[j]);
            for (index_t i = 0; i < nnzPerCol_remote_p[j]; ++i, ++iter) {
//                if(rank==1) printf("%ld \t%u \t%u \t%f \t%f\n",
//                iter, row_remote[iter], col_remote2[iter], values_remote[iter], vecValues[rdispls[recv_proc] + j]);
                w[row_remote[iter]] += values_remote[iter] * vecValues_p[j];
            }
        }
    }

    MPI_Waitall(numSendProc, &requests[numRecvProc], MPI_STATUSES_IGNORE);
//    }
}

void saena_matrix::matvec_sparse3(std::vector<value_t>& v, std::vector<value_t>& w) {
    // with openmp, no waitany

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");
//    print_info(-1);
//    print_vector(v, -1, "v", comm);

//    if(nprocs > 1){
    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
#pragma omp parallel for
    for(index_t i = 0; i < vIndexSize; ++i)
        vSend[i] = v[vIndex[i]];

//    print_vector(vSend, 1, "vSend", comm);

    // receive and put the remote parts of v in vecValues.
    // they are received in order: first put the values from the lowest rank matrix, and so on.
    for(int i = 0; i < numRecvProc; ++i){
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], par::Mpi_datatype<value_t>::value(), recvProcRank[i], 1, comm, &requests[i]);
        MPI_Test(&requests[i], &MPI_flag, MPI_STATUS_IGNORE);
    }

   MPI_Request *requests_p = &requests[numRecvProc];
   for(int i = 0; i < numSendProc; ++i){
      MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], par::Mpi_datatype<value_t>::value(), sendProcRank[i], 1, comm, &requests_p[i]);
      MPI_Test(&requests_p[i], &MPI_flag, MPI_STATUS_IGNORE);
   }

//    }

   // initialize w to 0
   fill(w.begin(), w.end(), 0);

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

#pragma omp parallel
   {
      value_t  tmp            = 0;
      value_t* v_p            = &v[0] - split[rank];
      index_t* col_local_p    = nullptr;
      value_t* values_local_p = nullptr;
      nnz_t    iter           = iter_local_array[omp_get_thread_num()];
#pragma omp for
      for (index_t i = 0; i < M; ++i) {
         col_local_p    = &col_local[iter];
         values_local_p = &values_local[iter];
         const index_t jend = nnzPerRow_local[i];
         tmp = 0;
         for (index_t j = 0; j < jend; ++j) {
//                if(rank==0) printf("%u \t%u \t%f \t%f \t%f \n", row_local[indicesP_local[iter]], col_local[indicesP_local[iter]], values_local[indicesP_local[iter]], v_p[col_local[indicesP_local[iter]]], values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]]);
            tmp += values_local_p[j] * v_p[col_local_p[j]];
         }
         w[i] += tmp;
         iter += jend;
      }
   }

    MPI_Waitall(numRecvProc, &requests[0], MPI_STATUSES_IGNORE);

//    print_vector(vecValues, 1, "vecValues", comm);

    // remote loop
    // -----------
    // the col_index of the matrix entry does not matter. do the matvec on the first non-zero column (j=0).
    // the corresponding vector element is saved in vecValues[0]. and so on.

#pragma omp parallel
    {
        index_t i = 0, l = 0;
        int thread_id = omp_get_thread_num();
        value_t *w_local = &w_buff[0] + (thread_id*M);
        if(thread_id==0)
            w_local = &*w.begin();
        else
            std::fill(&w_local[0], &w_local[M], 0);

        nnz_t iter = iter_remote_array[thread_id];
        const index_t jend = col_remote_size;
        index_t *row_remote_p = nullptr;
        value_t *val_remote_p = nullptr;

#pragma omp for
        for (index_t j = 0; j < jend; ++j) {
           row_remote_p = &row_remote[iter];
           val_remote_p = &values_remote[iter];
           const index_t iend = nnzPerCol_remote[j];
           const value_t vrem = vecValues[j];
           for (i = 0; i < iend; ++i) {
                w_local[row_remote_p[i]] += val_remote_p[i] * vrem;
            }
           iter += iend;
        }

        const index_t iend = M;
        int thread_partner = 0;
        int levels = (int)ceil(log2(num_threads));
        for (l = 0; l < levels; l++) {
            if (thread_id % int(pow(2, l+1)) == 0) {
                thread_partner = thread_id + int(pow(2, l));
//                printf("l = %d, levels = %d, thread_id = %d, thread_partner = %d \n", l, levels, thread_id, thread_partner);
                if(thread_partner < num_threads){
                    value_t *w_buff_p = &w_buff[thread_partner * M];
                    for (i = 0; i < iend; ++i){
                        w_local[i] += w_buff_p[i];
                    }
                }
            }
#pragma omp barrier
        }
    }

    MPI_Waitall(numSendProc, &requests[numRecvProc], MPI_STATUSES_IGNORE);
//    }
}

void saena_matrix::matvec_sparse_float(std::vector<value_t>& v, std::vector<value_t>& w) {
    // combination of openmp and waitany

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    print_info(-1);
//    print_vector(v, -1, "v", comm);

    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
#pragma omp parallel for
    for(index_t i = 0; i < vIndexSize; ++i)
        vSend_f[i] = v[vIndex[i]];

//    print_vector(vSend, 1, "vSend", comm);

    // receive and put the remote parts of v in vecValues.
    // they are received in order: first put the values from the lowest rank matrix, and so on.
    for(int i = 0; i < numRecvProc; ++i){
        MPI_Irecv(&vecValues_f[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_FLOAT, recvProcRank[i], 1, comm, &requests[i]);
//        MPI_Test(&requests[i], &MPI_flag, &statuses[i]);
    }

    for(int i = 0; i < numSendProc; ++i){
        MPI_Isend(&vSend_f[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_FLOAT, sendProcRank[i], 1, comm, &requests[numRecvProc+i]);
        MPI_Test(&requests[numRecvProc + i], &MPI_flag, MPI_STATUSES_IGNORE);
    }

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

    // initialize w to 0
    fill(w.begin(), w.end(), 0);

#pragma omp parallel
    {
        value_t  tmp            = 0;
        value_t* v_p            = &v[0] - split[rank];
        index_t* col_local_p    = nullptr;
        value_t* values_local_p = nullptr;
        nnz_t    iter           = iter_local_array[omp_get_thread_num()];
#pragma omp for
        for (index_t i = 0; i < M; ++i) {
            col_local_p    = &col_local[iter];
            values_local_p = &values_local[iter];
            const index_t jend = nnzPerRow_local[i];
            tmp = 0;
            for (index_t j = 0; j < jend; ++j) {
//                if(rank==0) printf("%u \t%u \t%f \t%f \t%f \n", row_local[indicesP_local[iter]], col_local[indicesP_local[iter]], values_local[indicesP_local[iter]], v_p[col_local[indicesP_local[iter]]], values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]]);
                tmp += values_local_p[j] * v_p[col_local_p[j]];
            }
            w[i] += tmp;
            iter += jend;
        }
    }

    nnz_t iter = 0;
    int recv_proc = 0, recv_proc_idx = 0;
    for(int np = 0; np < numRecvProc; ++np){
        MPI_Waitany(numRecvProc, &requests[0], &recv_proc_idx, MPI_STATUS_IGNORE);
        recv_proc = recvProcRank[recv_proc_idx];
//        if(rank==1) printf("recv_proc_idx = %d, recv_proc = %d, np = %d, numRecvProc = %d, recvCount[recv_proc] = %d\n",
//                              recv_proc_idx, recv_proc, np, numRecvProc, recvCount[recv_proc]);

        iter = nnzPerProcScan[recv_proc];
        float *vecValues_p        = &vecValues_f[rdispls[recv_proc]];
        auto  *nnzPerCol_remote_p = &nnzPerCol_remote[rdispls[recv_proc]];
        for (index_t j = 0; j < recvCount[recv_proc]; ++j) {
//            if(rank==1) printf("%u\n", nnzPerCol_remote_p[j]);
            for (index_t i = 0; i < nnzPerCol_remote_p[j]; ++i, ++iter) {
//                if(rank==1) printf("%ld \t%u \t%u \t%f \t%f\n",
//                iter, row_remote[iter], col_remote2[iter], values_remote[iter], vecValues[rdispls[recv_proc] + j]);
                w[row_remote[iter]] += values_remote[iter] * vecValues_p[j];
            }
        }
    }

    MPI_Waitall(numSendProc, &requests[numRecvProc], MPI_STATUSES_IGNORE);
}

void saena_matrix::matvec_sparse_array(value_t *v, value_t *w) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");
//    print_info(-1);
//    print_vector(v, -1, "v", comm);

//    if(nprocs > 1){
    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
    for(index_t i = 0; i < vIndexSize; ++i)
        vSend[i] = v[vIndex[i]];

//    print_vector(vSend, 1, "vSend", comm);

    // receive and put the remote parts of v in vecValues.
    // they are received in order: first put the values from the lowest rank matrix, and so on.
    for(int i = 0; i < numRecvProc; ++i){
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &requests[i]);
//        MPI_Test(&requests[i], &MPI_flag, &statuses[i]);
    }

    for(int i = 0; i < numSendProc; ++i){
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &requests[numRecvProc+i]);
        MPI_Test(&requests[numRecvProc + i], &MPI_flag, &statuses[numRecvProc + i]);
    }

//    }

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

    value_t* v_p  = &v[0] - split[rank];
    nnz_t    iter = 0;
    for (index_t i = 0; i < M; ++i) { // rows
        w[i] = 0;
        for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) { // columns
//            if(rank==0) printf("%u \t%u \t%f \t%f \t%f \n", row_local[indicesP_local[iter]], col_local[indicesP_local[iter]], values_local[indicesP_local[iter]], v_p[col_local[indicesP_local[iter]]], values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]]);
            w[i] += values_local[iter] * v_p[col_local[iter]];
        }
    }

    int np = 0;
    int recv_proc = 0, recv_proc_idx = 0;
    while(np < numRecvProc){
        MPI_Waitany(numRecvProc, &requests[0], &recv_proc_idx, MPI_STATUS_IGNORE);
        ++np;

        recv_proc = recvProcRank[recv_proc_idx];
//        if(rank==1) printf("recv_proc_idx = %d, recv_proc = %d, np = %d, numRecvProc = %d, recvCount[recv_proc] = %d\n",
//                              recv_proc_idx, recv_proc, np, numRecvProc, recvCount[recv_proc]);

        iter = nnzPerProcScan[recv_proc];
        value_t *vecValues_p        = &vecValues[rdispls[recv_proc]];
        auto    *nnzPerCol_remote_p = &nnzPerCol_remote[rdispls[recv_proc]];
        for (index_t j = 0; j < recvCount[recv_proc]; ++j) {
//            if(rank==1) printf("%u\n", nnzPerCol_remote_p[j]);
            for (index_t i = 0; i < nnzPerCol_remote_p[j]; ++i, ++iter) {
//                if(rank==1) printf("%ld \t%u \t%u \t%f \t%f\n",
//                iter, row_remote[iter], col_remote2[iter], values_remote[iter], vecValues[rdispls[recv_proc] + j]);
                w[row_remote[iter]] += values_remote[iter] * vecValues_p[j];
            }
        }
    }

    MPI_Waitall(numSendProc, &requests[numRecvProc], MPI_STATUSES_IGNORE);
//    }
}

void saena_matrix::matvec_sparse_array2(value_t *v, value_t *w) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    int flag = 0;

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");
//    print_info(-1);
//    print_vector(v, -1, "v", comm);

//    if(nprocs > 1){
//    double t = MPI_Wtime();
    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
#pragma omp parallel for
    for(index_t i = 0; i < vIndexSize; ++i)
        vSend[i] = v[vIndex[i]];

//    print_vector(vSend, 0, "vSend", comm);

//    t = MPI_Wtime() - t;
//    part1 += t;
//    double tcomm = MPI_Wtime();

    // receive and put the remote parts of v in vecValues.
    // they are received in order: first put the values from the lowest rank matrix, and so on.
    for(int i = 0; i < numRecvProc; ++i){
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], par::Mpi_datatype<value_t>::value(), recvProcRank[i], 1, comm, &requests[i]);
        MPI_Test(&requests[i], &flag, &statuses[i]);
    }

    for(int i = 0; i < numSendProc; ++i){
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], par::Mpi_datatype<value_t>::value(), sendProcRank[i], 1, comm, &requests[numRecvProc+i]);
        MPI_Test(&requests[numRecvProc + i], &flag, &statuses[numRecvProc + i]);
    }
//    }

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

//    t = MPI_Wtime();

    value_t* v_p = &v[0] - split[rank];
#pragma omp parallel
    {
        nnz_t iter = iter_local_array[omp_get_thread_num()];
#pragma omp for
        for (index_t i = 0; i < M; ++i) {
            w[i] = 0;
            for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
//                if(rank==0) printf("%u \t%u \t%f \t%f \t%f \n", row_local[indicesP_local[iter]], col_local[indicesP_local[iter]], values_local[indicesP_local[iter]], v_p[col_local[indicesP_local[iter]]], values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]]);
                w[i] += values_local[iter] * v_p[col_local[iter]];
            }
        }
    }

//    t = MPI_Wtime() - t;
//    part4 += t;

//    if(nprocs > 1){
    // Wait for the receive communication to finish.
    MPI_Waitall(numRecvProc, &requests[0], MPI_STATUSES_IGNORE);

//        print_vector(vecValues, 1, "vecValues", comm);

    // remote loop
    // -----------
    // the col_index of the matrix entry does not matter. do the matvec on the first non-zero column (j=0).
    // the corresponding vector element is saved in vecValues[0]. and so on.

//    t = MPI_Wtime();
#if 0
    #pragma omp parallel
    {
        index_t i, l;
        int thread_id = omp_get_thread_num();
        value_t *w_local = &w_buff[0] + (thread_id * M);
        if(thread_id==0)
            w_local = &w[0];
        else
            std::fill(&w_local[0], &w_local[M], 0);

        nnz_t iter = iter_remote_array[thread_id];
#pragma omp for
        for (index_t j = 0; j < col_remote_size; ++j) {
            for (i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
                w_local[row_remote[iter]] += values_remote[iter] * vecValues[j];

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
//                    printf("l = %d, levels = %d, thread_id = %d, thread_partner = %d \n", l, levels, thread_id, thread_partner);
                if(thread_partner < num_threads){
                    for (i = 0; i < M; i++)
                        w_local[i] += w_buff[thread_partner * M + i];
                }
            }
#pragma omp barrier
        }
    }
#endif
//    t = MPI_Wtime() - t;
//    part6 += t;

    // basic remote loop without openmp
    nnz_t iter = 0;
    for (index_t j = 0; j < col_remote_size; ++j) {
        for (index_t i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
//                if(rank) printf("%u \t%u \t%f \t%f \t%f \n", row_remote[iter], col_remote2[iter], values_remote[iter], vecValues[j], values_remote[iter] * vecValues[j]);
            w[row_remote[iter]] += values_remote[iter] * vecValues[j];
        }
    }

    MPI_Waitall(numSendProc, &requests[numRecvProc], MPI_STATUSES_IGNORE);
//    }

//    tcomm = MPI_Wtime() - tcomm;
//    part3 += tcomm;
}

void saena_matrix::matvec_time_init(){
    matvec_iter = 0;
    part1 = 0;
    part2 = 0;
    part3 = 0;
    part4 = 0;
    part5 = 0;
    part6 = 0;
}

void saena_matrix::matvec_time_print(const int &opt /*= 1*/) const{
    // opt: pass 2 for the zfp version (it includes compresssion and decompression times)

    int rank;
    MPI_Comm_rank(comm, &rank);

    double tmp = 1;
    if(matvec_iter != 0){
        tmp = static_cast<double>(matvec_iter);
    }

    double p1ave = print_time_ave(part1 / tmp, "", comm); // send buff
    double p2ave = print_time_ave(part2 / tmp, "", comm); // compress
    double p3ave = 0;                                     // comm
    if(opt == 2){
        p3ave = print_time_ave((part3 - part4 - part5 - part6) / tmp, "", comm);
    }else { // (opt == 1)
        p3ave = print_time_ave((part3 - part5 - part6) / tmp, "", comm);
    }
    double p4ave = print_time_ave(part4 / tmp, "", comm); // decompress
    double p5ave = print_time_ave(part5 / tmp, "", comm); // remote
    double p6ave = print_time_ave(part6 / tmp, "", comm); // local
    if(!rank){
//        printf("matvec iteration: %ld", matvec_iter);
        if(opt == 1) {
            printf("\naverage time:\nsend_buff\nlocal\nremote\ncomm\n\n"
                   "%.10f\n%.10f\n%.10f\n%.10f\n", p1ave, p6ave, p5ave, p3ave);
        }else{
            printf("\naverage time:\nsend_buff\ncompressndecompress\nlocal\nremote\ncomm\n\n"
                   "%.10f\n%.10f\n%.10f\n%.10f\n%.10f\n%.10f\n", p1ave, p2ave, p4ave, p6ave, p5ave, p3ave);
        }
    }
}

void saena_matrix::matvec_time_print2() const{
    int rank;
    MPI_Comm_rank(comm, &rank);

    double tmp = 1;
    if(matvec_iter != 0){
        tmp = static_cast<double>(matvec_iter);
    }

    double p1ave = print_time_ave(part1 / tmp, "", comm);                   // send buff
    double p6ave = print_time_ave(part6 / tmp, "", comm);                   // local
    double p5ave = print_time_ave(part5 / tmp, "", comm);                   // remote
    double p3ave = print_time_ave((part3 - part5 - part6) / tmp, "", comm); // comm

    double tot = p1ave + p3ave + p5ave + p6ave;
    if(!rank){
        printf("\naverage time:\nsend_buff\nlocal\nremote\ncomm\ntot\n\n"
               "%.10f\n%.10f\n%.10f\n%.10f\n%.10f\n", p1ave, p6ave, p5ave, p3ave, tot);
    }
}

void saena_matrix::matvec_time_print3() const{
    int rank;
    MPI_Comm_rank(comm, &rank);

    double tmp = 1;
    if(matvec_iter != 0){
        tmp = static_cast<double>(matvec_iter);
    }

    print_time_all(part1 / tmp, "send_buff", comm);              // send buff
    print_time_all(part6 / tmp, "local", comm);                  // local
    print_time_all(part5 / tmp, "remote", comm);                 // remote
    print_time_all((part3 - part5 - part6) / tmp, "comm", comm); // comm
    print_time_all((part1 + part3) / tmp, "total", comm);        // total

    if(!rank) print_sep();
}

void saena_matrix::matvec_sparse_test_orig(std::vector<value_t>& v, std::vector<value_t>& w) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");
//    print_info(-1);
//    print_vector(v, -1, "v", comm);

    double t = 0, tcomm = 0;
    ++matvec_iter;

//    if(nprocs > 1){
    t = omp_get_wtime();
    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
    for(index_t i = 0; i < vIndexSize; ++i)
        vSend[i] = v[vIndex[i]];

    t = omp_get_wtime() - t;
    part1 += t;

//    print_vector(vSend, 1, "vSend", comm);

    tcomm = omp_get_wtime();

    // receive and put the remote parts of v in vecValues.
    // they are received in order: first put the values from the lowest rank matrix, and so on.
    for(int i = 0; i < numRecvProc; ++i){
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &requests[i]);
//        MPI_Test(&requests[i], &MPI_flag, &statuses[i]);
    }

    for(int i = 0; i < numSendProc; ++i){
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &requests[numRecvProc+i]);
        MPI_Test(&requests[numRecvProc + i], &MPI_flag, &statuses[numRecvProc + i]);
    }

//    }

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

    t = omp_get_wtime();

    value_t* v_p  = &v[0] - split[rank];
    nnz_t    iter = 0;
    for (index_t i = 0; i < M; ++i) { // rows
        w[i] = 0;
        for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) { // columns
//            if(rank==0) printf("%u \t%u \t%f \t%f \t%f \n", row_local[indicesP_local[iter]], col_local[indicesP_local[iter]], values_local[indicesP_local[iter]], v_p[col_local[indicesP_local[iter]]], values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]]);
            w[i] += values_local[iter] * v_p[col_local[iter]];
        }
    }

    t = omp_get_wtime() - t;
    part6 += t;

    int np = 0;
    int recv_proc = 0, recv_proc_idx = 0;
    while(np < numRecvProc){
        MPI_Waitany(numRecvProc, &requests[0], &recv_proc_idx, MPI_STATUS_IGNORE);
        ++np;

        recv_proc = recvProcRank[recv_proc_idx];
//        if(rank==1) printf("recv_proc_idx = %d, recv_proc = %d, np = %d, numRecvProc = %d, recvCount[recv_proc] = %d\n",
//                              recv_proc_idx, recv_proc, np, numRecvProc, recvCount[recv_proc]);
        t = omp_get_wtime();
        iter = nnzPerProcScan[recv_proc];
        value_t *vecValues_p        = &vecValues[rdispls[recv_proc]];
        auto    *nnzPerCol_remote_p = &nnzPerCol_remote[rdispls[recv_proc]];
        for (index_t j = 0; j < recvCount[recv_proc]; ++j) {
//            if(rank==1) printf("%u\n", nnzPerCol_remote_p[j]);
            for (index_t i = 0; i < nnzPerCol_remote_p[j]; ++i, ++iter) {
//                if(rank==1) printf("%ld \t%u \t%u \t%f \t%f\n",
//                iter, row_remote[iter], col_remote2[iter], values_remote[iter], vecValues[rdispls[recv_proc] + j]);
                w[row_remote[iter]] += values_remote[iter] * vecValues_p[j];
            }
        }
        t = omp_get_wtime() - t;
        part5 += t;
    }

    MPI_Waitall(numSendProc, &requests[numRecvProc], MPI_STATUSES_IGNORE);
//    }

    tcomm = omp_get_wtime() - tcomm;
    part3 += tcomm;
}

void saena_matrix::matvec_sparse_test1(std::vector<value_t>& v, std::vector<value_t>& w) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");

    double t = 0, tcomm = 0;

    ++matvec_iter;

//    print_info(-1);
//    print_vector(v, -1, "v", comm);

//    if(nprocs > 1){
    t = omp_get_wtime();
    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
    for(index_t i = 0; i < vIndexSize; ++i)
        vSend[i] = v[vIndex[i]];

    t = omp_get_wtime() - t;
    part1 += t;

//    print_vector(vSend, 0, "vSend", comm);

    tcomm = omp_get_wtime();

    // receive and put the remote parts of v in vecValues.
    // they are received in order: first put the values from the lowest rank matrix, and so on.
    for(int i = 0; i < numRecvProc; ++i){
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &requests[i]);
        MPI_Test(&requests[i], &MPI_flag, &statuses[i]);
    }

    for(int i = 0; i < numSendProc; ++i){
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &requests[numRecvProc+i]);
        MPI_Test(&requests[numRecvProc + i], &MPI_flag, &statuses[numRecvProc + i]);
    }

//    }

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

    t = omp_get_wtime();

    value_t* v_p  = &v[0] - split[rank];
    nnz_t    iter = 0;
    for (index_t i = 0; i < M; ++i) { // rows
        w[i] = 0;
        for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) { // columns
//            if(rank==0) printf("%u \t%u \t%f \t%f \t%f \n", row_local[indicesP_local[iter]], col_local[indicesP_local[iter]], values_local[indicesP_local[iter]], v_p[col_local[indicesP_local[iter]]], values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]]);
            w[i] += values_local[iter] * v_p[col_local[iter]];
        }
    }

    t = omp_get_wtime() - t;
    part6 += t;

//    if(nprocs > 1){
    MPI_Waitall(numRecvProc, &requests[0], &statuses[0]);
//    MPI_Waitall(numRecvProc, &requests[0], MPI_STATUSES_IGNORE);

//    print_vector(vecValues, 1, "vecValues", comm);

    // remote loop
    // -----------
    // the col_index of the matrix entry does not matter. do the matvec on the first non-zero column (j=0).
    // the corresponding vector element is saved in vecValues[0]. and so on.

    t = omp_get_wtime();

    iter = 0;
    for (index_t j = 0; j < col_remote_size; ++j) {
        for (index_t i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
//            if(rank==0) printf("%u \t%u \t%f \t%f \t%f \n", row_remote[iter], col_remote2[iter], values_remote[iter], vecValues[j], values_remote[iter] * vecValues[j]);
            w[row_remote[iter]] += values_remote[iter] * vecValues[j];
        }
    }

    t = omp_get_wtime() - t;
    part5 += t;

    MPI_Waitall(numSendProc, &requests[numRecvProc], &statuses[numRecvProc]);
//    MPI_Waitall(numSendProc, &requests[numRecvProc], MPI_STATUSES_IGNORE);
//    }

    tcomm = omp_get_wtime() - tcomm;
    part3 += tcomm;
}

void saena_matrix::matvec_sparse_test2(std::vector<value_t>& v, std::vector<value_t>& w) {
    // the size of vSend and vecValues are set too big for this function.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(!rank) printf("To call matvec_sparse_test3(), uncomment allocation for vecValues2 in set_off_on_diagonal()\n");
#if 0
//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");
//    print_info(-1);
//    print_vector(v, -1, "v", comm);

    ++matvec_iter;

    MPI_Request* requests = new MPI_Request[2];
    MPI_Status*  statuses = new MPI_Status[2];

    double t = 0;
    double tcomm = 0;
    int send_proc = 0, recv_proc = 0, recv_proc_prev = 0;

    if(nprocs > 1){
        // send to right, receive from left
        send_proc = (rank + 1) % nprocs;
        recv_proc = rank - 1;
        if(recv_proc < 0)
            recv_proc += nprocs;
        recv_proc_prev = 0; // the processor that we received data in the previous round

        t = omp_get_wtime();

        // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
        // put the values of thoss indices in vSend to send to other procs.
        index_t iter = 0;
        for(index_t i = sendCountScan[send_proc]; i < sendCountScan[send_proc + 1]; ++i){
            vSend[iter] = v[vIndex[i]];
            ++iter;
        }

        t = omp_get_wtime() - t;
        part1 += t;

//        print_vector(vSend, 0, "vSend", comm);

        tcomm = omp_get_wtime();

        if(recvCount[recv_proc] != 0){
//            if(rank==rankv) std::cout << "recv_proc: " << recv_proc << ", recvCount[recv_proc]: " << recvCount[recv_proc] << std::endl;
            MPI_Irecv(&vecValues[0], recvCount[recv_proc], par::Mpi_datatype<value_t>::value(), recv_proc, recv_proc, comm, &requests[0]);
            MPI_Test(&requests[0], &MPI_flag, &statuses[0]);
        }

        if(sendCount[send_proc] != 0){
//            if(rank==rankv) std::cout << "send_proc: " << send_proc << ", sendCount[send_proc]: " << sendCount[send_proc] << std::endl;
            MPI_Isend(&vSend[0], sendCount[send_proc], par::Mpi_datatype<value_t>::value(), send_proc, rank, comm, &requests[1]);
            MPI_Test(&requests[1], &MPI_flag, &statuses[1]);
        }
    }

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

    t = omp_get_wtime();

    value_t* v_p  = &v[0] - split[rank];
    nnz_t    iter = 0;
    for (index_t i = 0; i < M; ++i) { // rows
        w[i] = 0;
        for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) { // columns
//            if(rank==0) printf("%u \t%u \t%f \t%f \t%f \n",
//            row_local[indicesP_local[iter]], col_local[indicesP_local[iter]], values_local[indicesP_local[iter]], v_p[col_local[indicesP_local[iter]]], values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]]);
            w[i] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
        }
    }

    t = omp_get_wtime() - t;
    part6 += t;

    if(nprocs > 1) {
        t = omp_get_wtime();

        // Wait for the first receive communication to finish.
        if (recvCount[recv_proc] != 0) {
            MPI_Wait(&requests[0], &statuses[0]);
        }
        if (sendCount[send_proc] != 0) {
            MPI_Wait(&requests[1], &statuses[1]);
        }

        t = omp_get_wtime() - t;
        part7 += t;
//        waitt.emplace_back(t);

        tcomm = omp_get_wtime() - tcomm;
        part3 += tcomm;

        int k = 1;
        while (k < nprocs + 1) {

            send_proc = (send_proc + 1) % nprocs;
            recv_proc_prev = recv_proc--;
            if (recv_proc < 0)
                recv_proc += nprocs;

            int rankv = 0;
//            print_vector(split, rankv, "split", comm);
//            print_vector(recvCountScan, rankv, "recvCountScan", comm);
//            if (rank == rankv) std::cout << "k: " << k << ", send_proc:" << send_proc << ", recv_proc: " << recv_proc
//                          << ", recv_proc_prev: " << recv_proc_prev << "\nrecvCount[recv_proc]: "
//                          << recvCount[recv_proc] << ", recvCount[recv_proc_prev]: " << recvCount[recv_proc_prev]
//                          << ", sendCount[send_proc]: " << sendCount[send_proc] << std::endl;

            t = omp_get_wtime();

            iter = 0;
            for(index_t i = sendCountScan[send_proc]; i < sendCountScan[send_proc + 1]; ++i){
                vSend[iter++] = v[vIndex[i]];
            }

            t = omp_get_wtime() - t;
            part1 += t;

            tcomm = omp_get_wtime();

            if (recvCount[recv_proc] != 0) {
                MPI_Irecv(&vecValues2[0], recvCount[recv_proc], par::Mpi_datatype<value_t>::value(), recv_proc, recv_proc, comm, &requests[0]);
                MPI_Test(&requests[0], &MPI_flag, &statuses[0]);
            }

            if (sendCount[send_proc] != 0) {
                MPI_Isend(&vSend[0], sendCount[send_proc], par::Mpi_datatype<value_t>::value(), send_proc, rank, comm, &requests[1]);
                MPI_Test(&requests[1], &MPI_flag, &statuses[1]);
            }

            if (recvCount[recv_proc_prev] != 0) {

//                print_vector(vecValues, rankv, "vecValues", comm);

                // perform matvec for recv_proc_prev's data
                // ----------
                // the col_index of the matrix entry does not matter. do the matvec on the first non-zero column (j=0).
                // the corresponding vector element is saved in vecValues[0]. and so on.

                t = omp_get_wtime();

                auto *nnzPerCol_remote_p = &nnzPerCol_remote[recvCountScan[recv_proc_prev]];
                iter = nnzPerProcScan[recv_proc_prev];
                for (index_t j = 0; j < recvCount[recv_proc_prev]; ++j) {
                    for (index_t i = 0; i < nnzPerCol_remote_p[j]; ++i, ++iter) {
//                        if(rank==rankv) printf("%ld \t%u \t%u \t%f \t%f\n",
//                        iter, row_remote[iter], col_remote2[iter], values_remote[iter], vecValues[j]);
                        w[row_remote[iter]] += values_remote[iter] * vecValues[j];
                    }
                }

                t = omp_get_wtime() - t;
                part5 += t;
            }

            t = omp_get_wtime();

            if (recvCount[recv_proc] != 0) {
                MPI_Wait(&requests[0], &statuses[0]);
            }
            if (sendCount[send_proc] != 0) {
                MPI_Wait(&requests[1], &statuses[1]);
            }

            t = omp_get_wtime() - t;
            part7 += t;
//            waitt.emplace_back(t);

            tcomm = omp_get_wtime() - tcomm;
            part3 += tcomm;

            t = omp_get_wtime();

            vecValues.swap(vecValues2);

            t = omp_get_wtime() - t;
            part6 += t;

            ++k;
        }

        delete[] requests;
        delete[] statuses;
    }
//    print_vector(waitt, -1, "wait time", comm);
#endif
}

void saena_matrix::matvec_sparse_test3(std::vector<value_t>& v, std::vector<value_t>& w) {
    // the size of vSend and vecValues are set too big for this function.
    // move "setting the send buffer" part into the overlapped communication

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(!rank) printf("To call matvec_sparse_test3(), uncomment allocation for vSend2 and vecValues2 in set_off_on_diagonal()\n");
#if 0
    assert(v.size() == M);

    double t = 0;
    double tcomm = 0;

    ++matvec_iter;

    MPI_Request* requests = new MPI_Request[2];
    MPI_Status*  statuses = new MPI_Status[2];

    fill(w.begin(), w.end(), 0);

//    print_info(-1);
//    print_vector(v, -1, "v", comm);

    nnz_t iter = 0;
    int send_proc = 0, recv_proc = 0, send_proc_prev = 0, recv_proc_prev = 0;
    int k = 0; // counter for number of communications

    if(nprocs > 1){
        // send to right, receive from left
        ++k;
        send_proc = (rank + 1) % nprocs;
        recv_proc = rank - 1;
        if(recv_proc < 0)
            recv_proc += nprocs;
        recv_proc_prev = 0; // the processor that we received data in the previous round

        t = omp_get_wtime();

        // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
        // put the values of thoss indices in vSend to send to other procs.
        iter = 0;
        for(index_t i = sendCountScan[send_proc]; i < sendCountScan[send_proc + 1]; ++i){
            vSend[iter++] = v[vIndex[i]];
        }

        t = omp_get_wtime() - t;
        part1 += t;

//        print_vector(vSend, 0, "vSend", comm);

        tcomm = omp_get_wtime();

        if(recvCount[recv_proc] != 0){
//            if(rank==rankv) std::cout << "recv_proc: " << recv_proc << ", recvCount[recv_proc]: " << recvCount[recv_proc] << std::endl;
            MPI_Irecv(&vecValues[0], recvCount[recv_proc], par::Mpi_datatype<value_t>::value(), recv_proc, recv_proc, comm, &requests[0]);
            MPI_Test(&requests[0], &MPI_flag, &statuses[0]);
        }

        if(sendCount[send_proc] != 0){
//            if(rank==rankv) std::cout << "send_proc: " << send_proc << ", sendCount[send_proc]: " << sendCount[send_proc] << std::endl;
            MPI_Isend(&vSend[0], sendCount[send_proc], par::Mpi_datatype<value_t>::value(), send_proc, rank, comm, &requests[1]);
            MPI_Test(&requests[1], &MPI_flag, &statuses[1]);
        }

    }

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

    t = omp_get_wtime();

    value_t* v_p  = &v[0] - split[rank];
    iter = 0;
    for (index_t i = 0; i < M; ++i) { // rows
        for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) { // columns
//            if(rank==0) printf("%u \t%u \t%f \t%f \t%f \n",
//            row_local[indicesP_local[iter]], col_local[indicesP_local[iter]], values_local[indicesP_local[iter]], v_p[col_local[indicesP_local[iter]]], values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]]);
            w[i] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
        }
    }

    t = omp_get_wtime() - t;
    part6 += t;

    if(nprocs > 1) {
        ++k;
        send_proc_prev = send_proc;
        send_proc = (send_proc + 1) % nprocs;
        recv_proc_prev = recv_proc--;
        if (recv_proc < 0)
            recv_proc += nprocs;

        t = omp_get_wtime();

        if(k < nprocs + 1){
            iter = 0;
            for(index_t i = sendCountScan[send_proc]; i < sendCountScan[send_proc + 1]; ++i){
                vSend2[iter++] = v[vIndex[i]];
            }
        }

        t = omp_get_wtime() - t;
        part1 += t;

        t = omp_get_wtime();

        // Wait for the first receive communication to finish.
        if (recvCount[recv_proc_prev] != 0) {
            MPI_Wait(&requests[0], &statuses[0]);
        }
        if (sendCount[send_proc_prev] != 0) {
            MPI_Wait(&requests[1], &statuses[1]);
        }

        t = omp_get_wtime() - t;
        part7 += t;
//        waitt.emplace_back(t);

        tcomm = omp_get_wtime() - tcomm;
        part3 += tcomm;

        while (k < nprocs + 1) {

//            int rankv = 0;
//            print_vector(split, rankv, "split", comm);
//            print_vector(recvCountScan, rankv, "recvCountScan", comm);
//            if (rank == rankv) std::cout << "k: " << k << ", send_proc:" << send_proc << ", recv_proc: " << recv_proc
//                          << ", recv_proc_prev: " << recv_proc_prev << "\nrecvCount[recv_proc]: "
//                          << recvCount[recv_proc] << ", recvCount[recv_proc_prev]: " << recvCount[recv_proc_prev]
//                          << ", sendCount[send_proc]: " << sendCount[send_proc] << std::endl;

            tcomm = omp_get_wtime();

            if (recvCount[recv_proc] != 0) {
                MPI_Irecv(&vecValues2[0], recvCount[recv_proc], par::Mpi_datatype<value_t>::value(), recv_proc, recv_proc, comm, &requests[0]);
                MPI_Test(&requests[0], &MPI_flag, &statuses[0]);
            }

            if (sendCount[send_proc] != 0) {
                MPI_Isend(&vSend2[0], sendCount[send_proc], par::Mpi_datatype<value_t>::value(), send_proc, rank, comm, &requests[1]);
                MPI_Test(&requests[1], &MPI_flag, &statuses[1]);
            }

            if (recvCount[recv_proc_prev] != 0) {

//                print_vector(vecValues, rankv, "vecValues", comm);

                // perform matvec for recv_proc_prev's data
                // ----------
                // the col_index of the matrix entry does not matter. do the matvec on the first non-zero column (j=0).
                // the corresponding vector element is saved in vecValues[0]. and so on.

                t = omp_get_wtime();

                auto *nnzPerCol_remote_p = &nnzPerCol_remote[recvCountScan[recv_proc_prev]];
                iter = nnzPerProcScan[recv_proc_prev];
                for (index_t j = 0; j < recvCount[recv_proc_prev]; ++j) {
                    for (index_t i = 0; i < nnzPerCol_remote_p[j]; ++i, ++iter) {
//                        if(rank==rankv) printf("%ld \t%u \t%u \t%f \t%f\n",
//                        iter, row_remote[iter], col_remote2[iter], values_remote[iter], vecValues[j]);
                        w[row_remote[iter]] += values_remote[iter] * vecValues[j];
                    }
                }

                t = omp_get_wtime() - t;
                part5 += t;
            }

            ++k;
            send_proc_prev = send_proc;
            send_proc = (send_proc + 1) % nprocs;
            recv_proc_prev = recv_proc--;
            if (recv_proc < 0)
                recv_proc += nprocs;

            if(k < nprocs + 1){
                t = omp_get_wtime();

                iter = 0;
                for(index_t i = sendCountScan[send_proc]; i < sendCountScan[send_proc + 1]; ++i){
                    vSend[iter++] = v[vIndex[i]];
                }

                t = omp_get_wtime() - t;
                part1 += t;
            }

            t = omp_get_wtime();

            // wait to finish the comm.
            if (recvCount[recv_proc_prev] != 0) {
                MPI_Wait(&requests[0], &statuses[0]);
            }
            if (sendCount[send_proc_prev] != 0) {
                MPI_Wait(&requests[1], &statuses[1]);
            }

            t = omp_get_wtime() - t;
            part7 += t;
//            waitt.emplace_back(t);

            tcomm = omp_get_wtime() - tcomm;
            part3 += tcomm;

            t = omp_get_wtime();

            vSend.swap(vSend2);
            vecValues.swap(vecValues2);

            t = omp_get_wtime() - t;
            part6 += t;
        }

        delete[] requests;
        delete[] statuses;
    }
//    print_vector(waitt, -1, "wait time", comm);
#endif
}

void saena_matrix::matvec_sparse_test4(std::vector<value_t>& v, std::vector<value_t>& w) {

    int rank;
    MPI_Comm_rank(comm, &rank);

//    int nprocs;
//    MPI_Comm_size(comm, &nprocs);
//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");
//    print_info(-1);
//    print_vector(v, -1, "v", comm);

    ++matvec_iter;
    double t = 0, tcomm = 0;

//    if(nprocs > 1){
    t = omp_get_wtime();

    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
#pragma omp parallel for
    for(index_t i = 0; i < vIndexSize; ++i)
        vSend[i] = v[vIndex[i]];

    t = omp_get_wtime() - t;
    part1 += t;

//    print_vector(vSend, 0, "vSend", comm);

    tcomm = omp_get_wtime();

    // receive and put the remote parts of v in vecValues.
    // they are received in order: first put the values from the lowest rank matrix, and so on.
    for(int i = 0; i < numRecvProc; ++i){
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], par::Mpi_datatype<value_t>::value(), recvProcRank[i], 1, comm, &requests[i]);
        MPI_Test(&requests[i], &MPI_flag, &statuses[i]);
    }

    for(int i = 0; i < numSendProc; ++i){
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], par::Mpi_datatype<value_t>::value(), sendProcRank[i], 1, comm, &requests[numRecvProc+i]);
        MPI_Test(&requests[numRecvProc + i], &MPI_flag, &statuses[numRecvProc + i]);
    }
//    }

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

    t = omp_get_wtime();

    value_t* v_p = &v[0] - split[rank];
#pragma omp parallel
    {
        nnz_t iter = iter_local_array[omp_get_thread_num()];
#pragma omp for
        for (index_t i = 0; i < M; ++i) {
            w[i] = 0;
            for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
//                if(rank==0) printf("%u \t%u \t%f \t%f \t%f \n", row_local[indicesP_local[iter]], col_local[indicesP_local[iter]], values_local[indicesP_local[iter]], v_p[col_local[indicesP_local[iter]]], values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]]);
                w[i] += values_local[iter] * v_p[col_local[iter]];
            }
        }
    }

    t = omp_get_wtime() - t;
    part6 += t;

//    if(nprocs > 1){
    MPI_Waitall(numRecvProc, &requests[0], &statuses[0]);

//    print_vector(vecValues, 1, "vecValues", comm);

    // remote loop
    // -----------
    // the col_index of the matrix entry does not matter. do the matvec on the first non-zero column (j=0).
    // the corresponding vector element is saved in vecValues[0]. and so on.

    t = omp_get_wtime();

#pragma omp parallel
    {
        index_t i, l;
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
                w_local[row_remote[iter]] += values_remote[iter] * vecValues[j];

//                if(rank==0 && thread_id==0){
//                    printf("thread = %d\n", thread_id);
//                    printf("%u \t%u \tind_rem = %lu, row = %lu \tcol = %lu \tvecVal = %f \n",
//                           i, j, indicesP_remote[iter], row_remote[indicesP_remote[iter]],
//                           col_remote[indicesP_remote[iter]], vecValues[col_remote[indicesP_remote[iter]]]);}
            }
        }

        int thread_partner;
        for (l = 0; l < matvec_levels; l++) {
            if (thread_id % int(pow(2, l+1)) == 0) {
                thread_partner = thread_id + int(pow(2, l));
//                    printf("l = %d, levels = %d, thread_id = %d, thread_partner = %d \n", l, levels, thread_id, thread_partner);
                if(thread_partner < num_threads){
                    for (i = 0; i < M; i++)
                        w_local[i] += w_buff[thread_partner * M + i];
                }
            }
#pragma omp barrier
        }
    }

    t = omp_get_wtime() - t;
    part5 += t;

    // basic remote loop without openmp
//        nnz_t iter = 0;
//        for (index_t j = 0; j < col_remote_size; ++j) {
//            for (index_t i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
//                if(rank) printf("%u \t%u \t%f \t%f \t%f \n", row_remote[iter], col_remote2[iter], values_remote[iter], vecValues[j], values_remote[iter] * vecValues[j]);
//                w[row_remote[iter]] += values_remote[iter] * vecValues[j];
//            }
//        }

    MPI_Waitall(numSendProc, &requests[numRecvProc], &statuses[numRecvProc]);
//    }

    tcomm = omp_get_wtime() - tcomm;
    part3 += tcomm;
}

void saena_matrix::matvec_sparse_test_omp(std::vector<value_t>& v, std::vector<value_t>& w) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");

    double t, tcomm;

//    print_info(-1);
//    print_vector(v, -1, "v", comm);

//    if(nprocs > 1){
    t = omp_get_wtime();
    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
#pragma omp parallel for
    for(index_t i=0;i<vIndexSize;i++)
        vSend[i] = v[vIndex[i]];

    t = omp_get_wtime() - t;
    part1 += t;

//    print_vector(vSend, 0, "vSend", comm);

    tcomm = omp_get_wtime();

    // receive and put the remote parts of v in vecValues.
    // they are received in order: first put the values from the lowest rank matrix, and so on.
    for(int i = 0; i < numRecvProc; i++){
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));
        MPI_Test(&requests[i], &MPI_flag, &statuses[i]);
    }

    for(int i = 0; i < numSendProc; i++){
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));
        MPI_Test(&requests[numRecvProc + i], &MPI_flag, &statuses[numRecvProc + i]);
    }
//    }

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

    t = omp_get_wtime();

    value_t* v_p = &v[0] - split[rank];
#pragma omp parallel
    {
        nnz_t iter = iter_local_array[omp_get_thread_num()];
#pragma omp for
        for (index_t i = 0; i < M; ++i) {
            w[i] = 0;
            for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
//                if(rank==0) printf("%u \t%u \t%f \t%f \t%f \n", row_local[indicesP_local[iter]], col_local[indicesP_local[iter]], values_local[indicesP_local[iter]], v_p[col_local[indicesP_local[iter]]], values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]]);
                w[i] += values_local[iter] * v_p[col_local[iter]];
            }
        }
    }

    t = omp_get_wtime() - t;
    part4 += t;

//    if(nprocs > 1){
    // Wait for the receive communication to finish.
    MPI_Waitall(numRecvProc, &requests[0], &statuses[0]);

//    print_vector(vecValues, 1, "vecValues", comm);

    // remote loop
    // -----------
    // the col_index of the matrix entry does not matter. do the matvec on the first non-zero column (j=0).
    // the corresponding vector element is saved in vecValues[0]. and so on.

    t = omp_get_wtime();

#pragma omp parallel
    {
        index_t i = 0, l = 0;
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
                w_local[row_remote[iter]] += values_remote[iter] * vecValues[j];

//                if(rank==0 && thread_id==0){
//                    printf("thread = %d\n", thread_id);
//                    printf("%u \t%u \tind_rem = %lu, row = %lu \tcol = %lu \tvecVal = %f \n",
//                           i, j, indicesP_remote[iter], row_remote[indicesP_remote[iter]],
//                           col_remote[indicesP_remote[iter]], vecValues[col_remote[indicesP_remote[iter]]]);}
            }
        }

        int thread_partner = 0;
        int levels = (int)ceil(log2(num_threads));
        for (l = 0; l < levels; l++) {
            if (thread_id % int(pow(2, l+1)) == 0) {
                thread_partner = thread_id + int(pow(2, l));
//                printf("l = %d, levels = %d, thread_id = %d, thread_partner = %d \n", l, levels, thread_id, thread_partner);
                if(thread_partner < num_threads){
                    for (i = 0; i < M; i++){
                        w_local[i] += w_buff[thread_partner * M + i];
                    }
                }
            }
#pragma omp barrier
        }
    }

    t = omp_get_wtime() - t;
    part6 += t;

    // basic remote loop without openmp
//        nnz_t iter = 0;
//        for (index_t j = 0; j < col_remote_size; ++j) {
//            for (index_t i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
//                if(rank==0) printf("%u \t%u \t%f \t%f \t%f \n", row_remote[iter], col_remote2[iter], values_remote[iter], vecValues[j], values_remote[iter] * vecValues[j]);
//                w[row_remote[iter]] += values_remote[iter] * vecValues[j];
//            }
//        }

    MPI_Waitall(numSendProc, &requests[numRecvProc], &statuses[numRecvProc]);
//    }

    tcomm = omp_get_wtime() - tcomm;
    part3 += tcomm;
}