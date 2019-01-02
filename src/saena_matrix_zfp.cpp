#include "saena_matrix.h"
#include "zfparray1.h"


// zfp functions:

int saena_matrix::allocate_zfp(){
/*
    zfp_send_bufsize = rate / 2 * (unsigned)ceil(vIndexSize/4.0); // rate/8 * 4 * ceil(size/4). This is in bytes.
    zfp_recv_bufsize = rate / 2 * (unsigned)ceil(recvSize/4.0);
    zfp_send_buffer = (double*)malloc(zfp_send_bufsize);
    zfp_recv_buffer = (double*)malloc(zfp_recv_bufsize);
    free_zfp_buff = true;
*/
    return 0;
}

int saena_matrix::deallocate_zfp(){
/*
    if(free_zfp_buff){
        free(zfp_send_buffer);
        free(zfp_recv_buffer);
        free_zfp_buff = false;
    }
*/
    return 0;
}


int saena_matrix::matvec_sparse_zfp(std::vector<value_t>& v, std::vector<value_t>& w) {
/*
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");

    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
#pragma omp parallel for
    for(index_t i=0;i<vIndexSize;i++)
        vSend[i] = v[(vIndex[i])];

//    if(rank==1)
//        for(index_t i=0;i<vIndexSize;i++)
//            vSend[i] = double(i+1);

//    print_vector(vSend, -1, "vSend", comm);

//    int rem = vIndexSize % 4;
//    for(index_t i = 0; i < rem; i++)
//        vSend.push_back(0);

    unsigned minbits;      // min bits per block
    unsigned maxbits;      // max bits per block
    unsigned maxprec;      // max precision
    int minexp;        // min bit plane encoded
//    double rate_return;
    unsigned long size1;
    if(vIndexSize || recvSize){
        field = zfp_field_1d(&vSend[0], zfp_type_double, vIndexSize);
        stream = stream_open(zfp_send_buffer, zfp_send_bufsize);
        zfp = zfp_stream_open(stream);
        zfp_stream_set_rate(zfp, rate, zfp_type_double, 1, 0);
//        zfp_stream_set_bit_stream(zfp, stream);
//        zfp_stream_params(zfp, &minbits, &maxbits, &maxprec, &minexp);
//        if(rank==1) printf("minbits = %u, maxbits = %u \n", minbits, maxbits);
//        maxbits = send_bufsize;
//        minbits = maxbits;
//        zfp_stream_set_params(zfp, 4*rate, 4*rate, maxprec, minexp);
//        zfp_stream_flush(zfp);
        zfp_stream_rewind(zfp);
        if(vIndexSize)
            size1 = zfp_compress(zfp, field);
//        size1 = zfp_stream_compressed_size(zfp);
//        if(rank==0) fprintf(stderr, "%u compressed bytes (%.2f bps)\n", (uint)size1, (double)size1 * CHAR_BIT / M);
//        printf("rank %d: passed rate = %u, rate_return = %f \n", rank, rate, rate_return);
    }

//    printf("here111\n");
//    print_vector(rdispls, -1, "rdispls", comm);
//    for(index_t i =0; i < recvSize; i++)
//        if(rank==0) std::cout << recv_buffer[i] << std::endl;

    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses  = new MPI_Status[numSendProc+numRecvProc];

    // receive and put the remote parts of v in vecValues.
    // they are received in order: first put the values from the lowest rank matrix, and so on.
    int size4k;
    for(int i = 0; i < numRecvProc; i++) {
        size4k = 4 * (int)ceil(recvProcCount[i]/4.0);
        MPI_Irecv(&zfp_recv_buffer[rdispls[recvProcRank[i]]], size4k, MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));
//        MPI_Irecv(&recv_buffer[(rate/CHAR_BIT)*rdispls[recvProcRank[i]]], (rate/CHAR_BIT)*recvProcCount[i], MPI_UNSIGNED_CHAR, recvProcRank[i], 1, comm, &(requests[i]));
//        if(rank==0) printf("(rate/CHAR_BIT)*rdispls[recvProcRank[i]] = %d, (rate/CHAR_BIT)*recvProcCount[i] = %d, recvProcRank[i] = %d \n",
//                           (rate/CHAR_BIT)*rdispls[recvProcRank[i]], (rate/CHAR_BIT)*recvProcCount[i], recvProcRank[i]);
    }

    for(int i = 0; i < numSendProc; i++){
        size4k = 4 * (int)ceil(sendProcCount[i]/4.0);
        MPI_Isend(&zfp_send_buffer[vdispls[sendProcRank[i]]], size4k, MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));
//        MPI_Isend(&send_buffer[(rate/CHAR_BIT)*vdispls[sendProcRank[i]]], (rate/CHAR_BIT), MPI_UNSIGNED_CHAR, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));
//        if(rank==1) printf("(rate/CHAR_BIT)*vdispls[sendProcRank[i]] = %d, (rate/CHAR_BIT)*sendProcCount[i] = %d, sendProcRank[i] = %d \n",
//                           (rate/CHAR_BIT)*vdispls[sendProcRank[i]], (rate/CHAR_BIT)*sendProcCount[i], sendProcRank[i]);
    }

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

    value_t* v_p = &v[0] - split[rank];
#pragma omp parallel
    {
        nnz_t iter = iter_local_array[omp_get_thread_num()];
#pragma omp for
        for (index_t i = 0; i < M; ++i) {
            w[i] = 0;
            for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
                w[i] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
            }
        }
    }

    // Wait for the receive communication to finish.
    MPI_Waitall(numRecvProc, requests, statuses);

//    if(recvSize){
//        stream = stream_open(recv_buffer, recv_bufsize);
//        zfp_stream_set_bit_stream(zfp, stream);
//        zfp_stream_set_params(zfp, 4*send_bufsize, 4*send_bufsize, maxprec, minexp);
//        zfp_stream_rewind(zfp);
//        zfp_field_set_pointer(field, &vecValues[0]);
//        zfp_field_set_size_1d(field, recvSize);
//        zfp_decompress(zfp, field);
//    }

    if(recvSize){
        field2 = zfp_field_1d(&vecValues[0], zfp_type_double, recvSize);
        stream2 = stream_open(zfp_recv_buffer, zfp_recv_bufsize);
        zfp2 = zfp_stream_open(stream2);
        zfp_stream_set_rate(zfp2, rate, zfp_type_double, 1, 0);
//        zfp_stream_params(zfp2, &minbits, &maxbits, &maxprec, &minexp);
//        zfp_stream_set_params(zfp2, 4*rate, 4*rate, maxprec, minexp);
        zfp_stream_rewind(zfp2);
        zfp_decompress(zfp2, field2);
    }

//    print_vector(vecValues, -1, "vecValues", comm);

    // remote loop
    // -----------
    // the col_index of the matrix entry does not matter. do the matvec on the first non-zero column (j=0).
    // the corresponding vector element is saved in vecValues[0]. and so on.

    unsigned int i;
    nnz_t iter = 0;
    for (index_t j = 0; j < col_remote_size; ++j) {
        for (i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
            w[row_remote[iter]] += values_remote[iter] * vecValues[j];
//            if(rank==1) std::cout << vecValues[j] << std::endl;
        }
    }

    if(vIndexSize || recvSize){
        zfp_field_free(field);
        zfp_stream_close(zfp);
        stream_close(stream);

        zfp_field_free(field2);
        zfp_stream_close(zfp2);
        stream_close(stream2);
    }

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    delete [] requests;
    delete [] statuses;
    */
    return 0;
}
