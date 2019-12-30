#include "saena_matrix.h"
#include "zfparray1.h"
#include <omp.h>

// The zfp_field parameter object holds information about the uncompressed array. To specify the compressed array,
//a zfp_stream object must be allocated.

int saena_matrix::allocate_zfp(){

    free_zfp_buff = true;

    zfp_send_buff_sz = zfp_rate / 2 * (unsigned)ceil(vIndexSize/4.0); // rate/8 * 4 * ceil(size/4). This is in bytes.
    zfp_recv_buff_sz = zfp_rate / 2 * (unsigned)ceil(recvSize/4.0);
//    zfp_send_buff = (double*)malloc(zfp_send_buff_sz);
//    zfp_recv_buff = (double*)malloc(zfp_recv_buff_sz);
    zfp_send_buff    = new uchar[zfp_send_buff_sz];
    zfp_recv_buff    = new uchar[zfp_recv_buff_sz];

    send_field = zfp_field_1d(&vSend[0], zfptype, vIndexSize);

    send_zfp   = zfp_stream_open(nullptr);
    zfp_stream_set_rate(send_zfp, zfp_rate, zfptype, 1, 0);
//    zfp_stream_set_precision(send_zfp, zfp_precision);

//    zfp_send_buff_sz = zfp_stream_maximum_size(send_zfp, send_field);
//    zfp_send_buff    = new uchar[zfp_send_buff_sz];
    send_stream      = stream_open(zfp_send_buff, zfp_send_buff_sz);
    zfp_stream_set_bit_stream(send_zfp, send_stream);

//    printf("M = %u, \tvIndexSize = %u, \tzfp_send_buff_sz = %u\n", M, vIndexSize, zfp_send_buff_sz);

//    recv_field = zfp_field_1d(&vecValues[0], zfptype, recvSize);
//    recv_zfp   = zfp_stream_open(nullptr);
//    zfp_stream_set_rate(recv_zfp, zfp_rate, zfptype, 1, 0);
//    recv_stream      = stream_open(zfp_recv_buff, zfp_recv_buff_sz);
//    zfp_stream_set_bit_stream(recv_zfp, recv_stream);

    recv_field = zfp_field_1d(&vecValues[0], zfptype, recvSize);
    recv_stream = stream_open(zfp_recv_buff, zfp_recv_buff_sz);
    recv_zfp = zfp_stream_open(recv_stream);
    zfp_stream_set_rate(recv_zfp, zfp_rate, zfptype, 1, 0);

    return 0;
}

int saena_matrix::deallocate_zfp(){

    if(free_zfp_buff){

        zfp_field_free(send_field);
        zfp_stream_close(send_zfp);
        stream_close(send_stream);

        zfp_field_free(recv_field);
        zfp_stream_close(recv_zfp);
        stream_close(recv_stream);

        delete []zfp_send_buff;
        delete []zfp_recv_buff;
        free_zfp_buff = false;

    }

    return 0;
}

void saena_matrix::matvec_print_time(){
    int rank;
    MPI_Comm_rank(comm, &rank);

    double tmp = 1;
    if(matvec_iter != 0){
        tmp = static_cast<double>(matvec_iter);
    }

//    print_time(part1 / tmp, "send vector", comm);
//    print_time(part2 / tmp, "compress", comm);
//    print_time((part3-part4-part5-part6) / tmp, "comm", comm);
//    print_time(part4 / tmp, "local", comm);
//    print_time(part5 / tmp, "decompress", comm);
//    print_time(part6 / tmp, "remote", comm);

    double p1ave = print_time_ave_consecutive(part1 / tmp, comm);
    double p2ave = print_time_ave_consecutive(part2 / tmp, comm);
    double p3ave = print_time_ave_consecutive((part3-part4-part5-part6) / tmp, comm);
    double p4ave = print_time_ave_consecutive(part4 / tmp, comm);
    double p5ave = print_time_ave_consecutive(part5 / tmp, comm);
    double p6ave = print_time_ave_consecutive(part6 / tmp, comm);
    if(!rank){
//        printf("matvec iteration: %ld", matvec_iter);
        printf("average time:\nsend buff\ncompress\ncomm\nlocal\ndecompress\nremote\n\n"
               "%f\n%f\n%f\n%f\n%f\n%f\n", p1ave, p2ave, p3ave, p4ave, p5ave, p6ave);
    }

}


int saena_matrix::matvec_sparse_zfp(std::vector<value_t>& v, std::vector<value_t>& w) {
    // todo: add back the openmp parts

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    ++matvec_iter;

    double t = MPI_Wtime();
    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
//#pragma omp parallel for // todo: add back the openmp parts
    for(index_t i = 0;i < vIndexSize; ++i){
        vSend[i] = v[(vIndex[i])];
    }

    t = MPI_Wtime() - t;
    part1 += t;

    t = MPI_Wtime();
    if(vIndexSize || recvSize){
        zfp_stream_rewind(send_zfp);
        if(vIndexSize){
            zfp_send_comp_sz = zfp_compress(send_zfp, send_field);
//            printf("rank %d: vIndexSize = %u, zfp_send_buff_sz = %u, \tzfp_send_comp_sz = %u\n", rank, vIndexSize, zfp_send_buff_sz, zfp_send_comp_sz);
//            if(zfp_send_buff_sz != zfp_send_comp_sz){
//                printf("ERROR: rank %d: vIndexSize = %u, zfp_send_buff_sz = %u, \tzfp_send_comp_sz = %u\n", rank, vIndexSize, zfp_send_buff_sz, zfp_send_comp_sz);
//            }
        }
    }
    t = MPI_Wtime() - t;
    part2 += t;

    double tcomm = MPI_Wtime();
    auto requests = new MPI_Request[numSendProc+numRecvProc];
    auto statuses = new MPI_Status[numSendProc+numRecvProc];

    // receive and put the remote parts of v in vecValues.
    // they are received in order: first put the values from the lowest rank matrix, and so on.
    for(int i = 0; i < numRecvProc; ++i) {
        MPI_Irecv(&zfp_recv_buff[(zfp_rate/CHAR_BIT)*rdispls[recvProcRank[i]]], (zfp_rate/CHAR_BIT)*recvProcCount[i], MPI_UNSIGNED_CHAR, recvProcRank[i], 1, comm, &(requests[i]));
    }

    for(int i = 0; i < numSendProc; ++i){
        MPI_Isend(&zfp_send_buff[(zfp_rate/CHAR_BIT)*vdispls[sendProcRank[i]]], (zfp_rate/CHAR_BIT)*sendProcCount[i], MPI_UNSIGNED_CHAR, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));
    }

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

    t = MPI_Wtime();
    value_t* v_p  = &v[0] - split[rank];
    nnz_t    iter = 0;
    for (index_t i = 0; i < M; ++i) {
        w[i] = 0;
        for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
            w[i] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
        }
    }
    t = MPI_Wtime() - t;
    part4 += t;

    // the openmp version
    // todo: add back the openmp parts
#if 0
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
#endif
    // Wait for the receive communication to finish.
    MPI_Waitall(numRecvProc, requests, statuses);

    t = MPI_Wtime();
    if(recvSize){
        zfp_stream_rewind(recv_zfp);
        zfp_decompress(recv_zfp, recv_field);
    }
    t = MPI_Wtime() - t;
    part5 += t;

//    print_vector(vecValues, -1, "vecValues", comm);

    // remote loop
    // -----------
    // the col_index of the matrix entry does not matter. do the matvec on the first non-zero column (j=0).
    // the corresponding vector element is saved in vecValues[0]. and so on.

    t = MPI_Wtime();
    unsigned int i;
    iter = 0;
    for (index_t j = 0; j < col_remote_size; ++j) {
        for (i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
            w[row_remote[iter]] += values_remote[iter] * vecValues[j];
        }
    }
    t = MPI_Wtime() - t;
    part6 += t;

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    delete [] requests;
    delete [] statuses;
    tcomm = MPI_Wtime() - tcomm;
    part3 += tcomm;

    return 0;
}

// this version has the commented out parts
// int saena_matrix::matvec_sparse_zfp(std::vector<value_t>& v, std::vector<value_t>& w)
/*
int saena_matrix::matvec_sparse_zfp(std::vector<value_t>& v, std::vector<value_t>& w) {
    // todo: add back the openmp parts

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");

    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
//#pragma omp parallel for // todo: add back the openmp parts
    for(index_t i = 0;i < vIndexSize; ++i){
        vSend[i] = v[(vIndex[i])];
    }

//    if(rank==1)
//        for(index_t i=0;i<vIndexSize;i++)
//            vSend[i] = double(i+1);

//    print_vector(vSend, -1, "vSend", comm);

//    int rem = vIndexSize % 4;
//    for(index_t i = 0; i < rem; i++)
//        vSend.push_back(0);

//    unsigned minbits;      // min bits per block
//    unsigned maxbits;      // max bits per block
//    unsigned maxprec;      // max precision
//    int      minexp;       // min bit plane encoded
//    double rate_return;
//    unsigned long size1;

    if(vIndexSize || recvSize){

//        send_field = zfp_field_1d(&vSend[0], zfptype, vIndexSize);

//        send_zfp   = zfp_stream_open(nullptr);
//        zfp_stream_set_rate(send_zfp, zfp_rate, zfptype, 1, 0);
//        zfp_stream_set_precision(send_zfp, zfp_precision);

//        zfp_send_buff_sz = zfp_stream_maximum_size(send_zfp, send_field);
//        zfp_send_buff    = new uchar[zfp_send_buff_sz];
//        send_stream      = stream_open(zfp_send_buff, zfp_send_buff_sz);
//        zfp_stream_set_bit_stream(send_zfp, send_stream);

        zfp_stream_rewind(send_zfp);
        if(vIndexSize){
            zfp_send_comp_sz = zfp_compress(send_zfp, send_field);
//            printf("rank %d: vIndexSize = %u, zfp_send_buff_sz = %u, \tzfp_send_comp_sz = %u\n", rank, vIndexSize, zfp_send_buff_sz, zfp_send_comp_sz);
            if(zfp_send_buff_sz != zfp_send_comp_sz){
                printf("ERROR: rank %d: vIndexSize = %u, zfp_send_buff_sz = %u, \tzfp_send_comp_sz = %u\n", rank, vIndexSize, zfp_send_buff_sz, zfp_send_comp_sz);
            }
        }

//        size1 = zfp_stream_compressed_size(send_zfp);
//        if(rank==0) fprintf(stderr, "%u compressed bytes (%.2f bps)\n", (uint)size1, (double)size1 * CHAR_BIT / M);
//        printf("rank %d: passed rate = %u, rate_return = %f \n", rank, rate, rate_return);
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
//        size4k = 4 * (int)ceil(recvProcCount[i]/4.0);
//        MPI_Irecv(&zfp_recv_buff[rdispls[recvProcRank[i]]], size4k, MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));
//        MPI_Irecv(&zfp_recv_buff[(zfp_rate/CHAR_BIT)*rdispls[recvProcRank[i]]], (zfp_rate/CHAR_BIT)*recvProcCount[i], MPI_UNSIGNED_CHAR, recvProcRank[i], 1, comm, &(requests[i]));

        MPI_Irecv(&zfp_recv_buff[(zfp_rate/CHAR_BIT)*rdispls[recvProcRank[i]]], (zfp_rate/CHAR_BIT)*recvProcCount[i], MPI_UNSIGNED_CHAR, recvProcRank[i], 1, comm, &(requests[i]));
//        if(rank==0) printf("(rate/CHAR_BIT)*rdispls[recvProcRank[i]] = %d, (rate/CHAR_BIT)*recvProcCount[i] = %d, recvProcRank[i] = %d, \tzfp_rate/CHAR_BIT = %u \n",
//                           (zfp_rate/CHAR_BIT)*rdispls[recvProcRank[i]], (zfp_rate/CHAR_BIT)*recvProcCount[i], recvProcRank[i], zfp_rate/CHAR_BIT);
    }

    for(int i = 0; i < numSendProc; i++){
//        size4k = 4 * (int)ceil(sendProcCount[i]/4.0);
//        MPI_Isend(&zfp_send_buff[vdispls[sendProcRank[i]]], size4k, MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc + i]));
//        MPI_Isend(&zfp_send_buff[(zfp_rate/CHAR_BIT)*vdispls[sendProcRank[i]]], (zfp_rate/CHAR_BIT), MPI_UNSIGNED_CHAR, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));

        MPI_Isend(&zfp_send_buff[(zfp_rate/CHAR_BIT)*vdispls[sendProcRank[i]]], (zfp_rate/CHAR_BIT)*sendProcCount[i], MPI_UNSIGNED_CHAR, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));
//        if(rank==1) printf("(rate/CHAR_BIT)*vdispls[sendProcRank[i]] = %d, (rate/CHAR_BIT)*sendProcCount[i] = %d, sendProcRank[i] = %d \n",
//                           (zfp_rate/CHAR_BIT)*vdispls[sendProcRank[i]], (zfp_rate/CHAR_BIT)*sendProcCount[i], sendProcRank[i]);
    }

//    MPI_Waitall(numRecvProc, requests, statuses);
//    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
//    delete [] requests;
//    delete [] statuses;

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

    value_t* v_p = &v[0] - split[rank];
    nnz_t iter = 0;
    for (index_t i = 0; i < M; ++i) {
        w[i] = 0;
        for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
            w[i] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
        }
    }

    // the openmp version
    // todo: add back the openmp parts
#if 0
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
#endif
    // Wait for the receive communication to finish.
    MPI_Waitall(numRecvProc, requests, statuses);

//    if(recvSize){
//        send_stream = stream_open(recv_buffer, recv_bufsize);
//        zfp_stream_set_bit_stream(send_zfp, send_stream);
//        zfp_stream_set_params(send_zfp, 4*send_bufsize, 4*send_bufsize, maxprec, minexp);
//        zfp_stream_rewind(send_zfp);
//        zfp_field_set_pointer(send_field, &vecValues[0]);
//        zfp_field_set_size_1d(send_field, recvSize);
//        zfp_decompress(send_zfp, send_field);
//    }

    if(recvSize){
//        recv_field = zfp_field_1d(&vecValues[0], zfptype, recvSize);
//        recv_stream = stream_open(zfp_recv_buff, zfp_recv_buff_sz);
//        recv_zfp = zfp_stream_open(recv_stream);
//        zfp_stream_set_rate(recv_zfp, zfp_rate, zfptype, 1, 0);

//        zfp_stream_params(recv_zfp, &minbits, &maxbits, &maxprec, &minexp);
//        zfp_stream_set_params(recv_zfp, 4*rate, 4*rate, maxprec, minexp);

        zfp_stream_rewind(recv_zfp);
        zfp_decompress(recv_zfp, recv_field);
    }

//    print_vector(vecValues, -1, "vecValues", comm);

    // remote loop
    // -----------
    // the col_index of the matrix entry does not matter. do the matvec on the first non-zero column (j=0).
    // the corresponding vector element is saved in vecValues[0]. and so on.

    unsigned int i;
    iter = 0;
    for (index_t j = 0; j < col_remote_size; ++j) {
        for (i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
            w[row_remote[iter]] += values_remote[iter] * vecValues[j];
//            if(rank==1) std::cout << vecValues[j] << std::endl;
        }
    }

//    if(vIndexSize || recvSize){
//        zfp_field_free(send_field);
//        zfp_stream_close(send_zfp);
//        stream_close(send_stream);
//
//        zfp_field_free(recv_field);
//        zfp_stream_close(recv_zfp);
//        stream_close(recv_stream);
//    }

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    delete [] requests;
    delete [] statuses;

    return 0;
}
*/

// older version
// int saena_matrix::matvec_sparse_zfp(std::vector<value_t>& v, std::vector<value_t>& w)
/*
int saena_matrix::matvec_sparse_zfp(std::vector<value_t>& v, std::vector<value_t>& w) {
    // todo: add back the openmp parts

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");

    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
//#pragma omp parallel for // todo: add back the openmp parts
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
        send_field  = zfp_field_1d(&vSend[0], zfptype, vIndexSize);
        send_stream = stream_open(zfp_send_buff, zfp_send_buff_sz);
        send_zfp    = zfp_stream_open(send_stream);
        zfp_stream_set_rate(send_zfp, rate, zfptype, 1, 0);
//        zfp_stream_set_bit_stream(send_zfp, send_stream);
//        zfp_stream_params(send_zfp, &minbits, &maxbits, &maxprec, &minexp);
//        if(rank==1) printf("minbits = %u, maxbits = %u \n", minbits, maxbits);
//        maxbits = send_bufsize;
//        minbits = maxbits;
//        zfp_stream_set_params(send_zfp, 4*rate, 4*rate, maxprec, minexp);
//        zfp_stream_flush(send_zfp);
        zfp_stream_rewind(send_zfp);
        if(vIndexSize)
            size1 = zfp_compress(send_zfp, send_field);
//        size1 = zfp_stream_compressed_size(send_zfp);
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
        MPI_Irecv(&zfp_recv_buff[rdispls[recvProcRank[i]]], size4k, MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));
//        MPI_Irecv(&recv_buffer[(rate/CHAR_BIT)*rdispls[recvProcRank[i]]], (rate/CHAR_BIT)*recvProcCount[i], MPI_UNSIGNED_CHAR, recvProcRank[i], 1, comm, &(requests[i]));
//        if(rank==0) printf("(rate/CHAR_BIT)*rdispls[recvProcRank[i]] = %d, (rate/CHAR_BIT)*recvProcCount[i] = %d, recvProcRank[i] = %d \n",
//                           (rate/CHAR_BIT)*rdispls[recvProcRank[i]], (rate/CHAR_BIT)*recvProcCount[i], recvProcRank[i]);
    }

    for(int i = 0; i < numSendProc; i++){
        size4k = 4 * (int)ceil(sendProcCount[i]/4.0);
        MPI_Isend(&zfp_send_buff[vdispls[sendProcRank[i]]], size4k, MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));
//        MPI_Isend(&send_buffer[(rate/CHAR_BIT)*vdispls[sendProcRank[i]]], (rate/CHAR_BIT), MPI_UNSIGNED_CHAR, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));
//        if(rank==1) printf("(rate/CHAR_BIT)*vdispls[sendProcRank[i]] = %d, (rate/CHAR_BIT)*sendProcCount[i] = %d, sendProcRank[i] = %d \n",
//                           (rate/CHAR_BIT)*vdispls[sendProcRank[i]], (rate/CHAR_BIT)*sendProcCount[i], sendProcRank[i]);
    }

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

    value_t* v_p = &v[0] - split[rank];
    nnz_t iter = 0;
    for (index_t i = 0; i < M; ++i) {
        w[i] = 0;
        for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
            w[i] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
        }
    }

    // the openmp version
    // todo: add back the openmp parts
#if 0
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
#endif

    // Wait for the receive communication to finish.
    MPI_Waitall(numRecvProc, requests, statuses);

//    if(recvSize){
//        send_stream = stream_open(recv_buffer, recv_bufsize);
//        zfp_stream_set_bit_stream(send_zfp, send_stream);
//        zfp_stream_set_params(send_zfp, 4*send_bufsize, 4*send_bufsize, maxprec, minexp);
//        zfp_stream_rewind(send_zfp);
//        zfp_field_set_pointer(send_field, &vecValues[0]);
//        zfp_field_set_size_1d(send_field, recvSize);
//        zfp_decompress(send_zfp, send_field);
//    }

    if(recvSize){
        recv_field = zfp_field_1d(&vecValues[0], zfptype, recvSize);
        recv_stream = stream_open(zfp_recv_buff, zfp_recv_buff_sz);
        recv_zfp = zfp_stream_open(recv_stream);
        zfp_stream_set_rate(recv_zfp, rate, zfptype, 1, 0);
//        zfp_stream_params(recv_zfp, &minbits, &maxbits, &maxprec, &minexp);
//        zfp_stream_set_params(recv_zfp, 4*rate, 4*rate, maxprec, minexp);
        zfp_stream_rewind(recv_zfp);
        zfp_decompress(recv_zfp, recv_field);
    }

//    print_vector(vecValues, -1, "vecValues", comm);

    // remote loop
    // -----------
    // the col_index of the matrix entry does not matter. do the matvec on the first non-zero column (j=0).
    // the corresponding vector element is saved in vecValues[0]. and so on.

    unsigned int i;
    iter = 0;
    for (index_t j = 0; j < col_remote_size; ++j) {
        for (i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
            w[row_remote[iter]] += values_remote[iter] * vecValues[j];
//            if(rank==1) std::cout << vecValues[j] << std::endl;
        }
    }

    if(vIndexSize || recvSize){
        zfp_field_free(send_field);
        zfp_stream_close(send_zfp);
        stream_close(send_stream);

        zfp_field_free(recv_field);
        zfp_stream_close(recv_zfp);
        stream_close(recv_stream);
    }

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    delete [] requests;
    delete [] statuses;

    return 0;
}
*/