#include "saena_matrix.h"
#include "zfparray1.h"
#include <omp.h>
#include <cassert>

// The zfp_field parameter object holds information about the uncompressed array. To specify the compressed array,
//a zfp_stream object must be allocated.

int saena_matrix::allocate_zfp(){

    free_zfp_buff = true;

    // compute zfp_send_buff_sz in bytes:
    // rate / 8 * 4 * ceil(size / 4).
    // divide by 8 to convert bits to bytes
    // 4 * ceil(size / 4): because zfp compresses blocks of size 4.
    zfp_send_buff_sz = zfp_rate / 2 * (index_t)ceil(vIndexSize / 4.0);
    zfp_recv_buff_sz = zfp_rate / 2 * (index_t)ceil(recvSize / 4.0);
    zfp_send_buff  = new uchar[zfp_send_buff_sz];
    zfp_recv_buff  = new uchar[zfp_recv_buff_sz];
    zfp_recv_buff2 = new uchar[zfp_recv_buff_sz];

    send_field  = zfp_field_1d(&vSend[0], zfptype, vIndexSize);
    send_stream = stream_open(zfp_send_buff, zfp_send_buff_sz);
    send_zfp    = zfp_stream_open(send_stream);
    zfp_stream_set_rate(send_zfp, zfp_rate, zfptype, 1, 0);
//    zfp_stream_set_precision(send_zfp, zfp_precision);

//    printf("M = %u, \tvIndexSize = %u, \tzfp_send_buff_sz = %u\n", M, vIndexSize, zfp_send_buff_sz);

    recv_field  = zfp_field_1d(&vecValues[0], zfptype, recvSize);
    recv_stream = stream_open(zfp_recv_buff, zfp_recv_buff_sz);
    recv_zfp    = zfp_stream_open(recv_stream);
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
        delete []zfp_recv_buff2;
        free_zfp_buff = false;

    }

    return 0;
}


void saena_matrix::matvec_time_init(){
    matvec_iter = 0;
    part1 = 0;
    part2 = 0;
    part3 = 0;
    part4 = 0;
    part5 = 0;
    part6 = 0;
    part7 = 0;
}

void saena_matrix::matvec_time_print(const int &opt /*= 1*/) const{
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

    double p1ave = print_time_ave(part1 / tmp, "", comm); // send buff
    double p2ave = print_time_ave(part2 / tmp, "", comm); // compress
    double p3ave = 0;                                     // comm
    if(opt == 1){
        p3ave = print_time_ave((part3 - part4 - part5) / tmp, "", comm);
    }else { // (opt == 2)
        p3ave = print_time_ave((part3 - part5) / tmp, "", comm);
    }
    double p4ave = print_time_ave(part4 / tmp, "", comm); // decompress
    double p5ave = print_time_ave(part5 / tmp, "", comm); // compute
    double p6ave = print_time_ave(part6 / tmp, "", comm); // swap
    double p7ave = print_time_ave(part7 / tmp, "", comm); // wait
    if(!rank){
//        printf("matvec iteration: %ld", matvec_iter);
        printf("average time:\nsend buff\ncompress\ncomm\ndecompress\ncompute\nswap\nwait\n\n"
               "%f\n%f\n%f\n%f\n%f\n%f\n%f\n", p1ave, p2ave, p3ave, p4ave, p5ave, p6ave, p7ave);
    }

}


int saena_matrix::matvec_sparse_test(std::vector<value_t>& v, std::vector<value_t>& w) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");

    double t = 0, tcomm = 0;
    MPI_Request* requests = nullptr;
    MPI_Status*  statuses = nullptr;

    ++matvec_iter;

//    print_info(-1);
//    print_vector(v, -1, "v", comm);

//    if(nprocs > 1){
    t = omp_get_wtime();
    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
    for(index_t i=0;i<vIndexSize;i++)
        vSend[i] = v[(vIndex[i])];

    t = omp_get_wtime() - t;
    part1 += t;

//    print_vector(vSend, 0, "vSend", comm);

    tcomm = omp_get_wtime();
    requests = new MPI_Request[numSendProc+numRecvProc];
    statuses = new MPI_Status[numSendProc+numRecvProc];

    // for MPI_Test
    int flag = 0;

    // receive and put the remote parts of v in vecValues.
    // they are received in order: first put the values from the lowest rank matrix, and so on.
    for(int i = 0; i < numRecvProc; i++){
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &requests[i]);
        MPI_Test(&requests[i], &flag, &statuses[i]);
    }

    for(int i = 0; i < numSendProc; i++){
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &requests[numRecvProc+i]);
        MPI_Test(&requests[numRecvProc + i], &flag, &statuses[numRecvProc + i]);
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
            w[i] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
        }
    }

    t = omp_get_wtime() - t;
    part4 += t;

//    if(nprocs > 1){
    // Wait for the receive communication to finish.
    MPI_Waitall(numRecvProc, requests, statuses);

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
    part6 += t;

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    delete [] requests;
    delete [] statuses;
//    }

    tcomm = omp_get_wtime() - tcomm;
    part3 += tcomm;

    return 0;
}

int saena_matrix::matvec_sparse_test2(std::vector<value_t>& v, std::vector<value_t>& w) {

    // the size of vSend and vecValues are set too big for this function.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");

    double t = 0;
    double tcomm = 0;
    MPI_Request* requests = nullptr;
    MPI_Status*  statuses = nullptr;

    ++matvec_iter;

//    print_info(-1);
//    print_vector(v, -1, "v", comm);

    requests = new MPI_Request[2];
    statuses = new MPI_Status[2];

    int send_proc = 0, recv_proc = 0, recv_proc_prev = 0;
    // for MPI_Test
    int flag = 0;

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
            vSend[iter] = v[(vIndex[i])];
            ++iter;
        }

        t = omp_get_wtime() - t;
        part1 += t;

//        print_vector(vSend, 0, "vSend", comm);

        tcomm = omp_get_wtime();

        if(recvCount[recv_proc] != 0){
//            if(rank==rankv) std::cout << "recv_proc: " << recv_proc << ", recvCount[recv_proc]: " << recvCount[recv_proc] << std::endl;
            MPI_Irecv(&vecValues[0], recvCount[recv_proc], par::Mpi_datatype<value_t>::value(), recv_proc, recv_proc, comm, &requests[0]);
            MPI_Test(&requests[0], &flag, &statuses[0]);
        }

        if(sendCount[send_proc] != 0){
//            if(rank==rankv) std::cout << "send_proc: " << send_proc << ", sendCount[send_proc]: " << sendCount[send_proc] << std::endl;
            MPI_Isend(&vSend[0], sendCount[send_proc], par::Mpi_datatype<value_t>::value(), send_proc, rank, comm, &requests[1]);
            MPI_Test(&requests[1], &flag, &statuses[1]);
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
    part5 += t;

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
                vSend[iter++] = v[(vIndex[i])];
            }

            t = omp_get_wtime() - t;
            part1 += t;

            tcomm = omp_get_wtime();

            if (recvCount[recv_proc] != 0) {
                MPI_Irecv(&vecValues2[0], recvCount[recv_proc], par::Mpi_datatype<value_t>::value(), recv_proc, recv_proc, comm, &requests[0]);
                MPI_Test(&requests[0], &flag, &statuses[0]);
            }

            if (sendCount[send_proc] != 0) {
                MPI_Isend(&vSend[0], sendCount[send_proc], par::Mpi_datatype<value_t>::value(), send_proc, rank, comm, &requests[1]);
                MPI_Test(&requests[1], &flag, &statuses[1]);
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

            // wait to finish the comm.
            if (recvCount[recv_proc] != 0) {
                MPI_Wait(&requests[0], &statuses[0]);
            }
            if (sendCount[send_proc] != 0) {
                MPI_Wait(&requests[1], &statuses[1]);
            }

            t = omp_get_wtime() - t;
            part7 += t;

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

    return 0;
}

int saena_matrix::matvec_sparse_test3(std::vector<value_t>& v, std::vector<value_t>& w) {
    // this version is similar to matvec_sparse_test2, but vSend is set once at the beginning.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");

    double t = 0;
    double tcomm = 0;
    MPI_Request* requests = nullptr;
    MPI_Status*  statuses = nullptr;

    ++matvec_iter;

//    print_info(-1);
//    print_vector(v, -1, "v", comm);

    requests = new MPI_Request[2];
    statuses = new MPI_Status[2];

    int send_proc = 0, recv_proc = 0, recv_proc_prev = 0;
    // for MPI_Test
    int flag = 0;

    if(nprocs > 1){
        t = omp_get_wtime();

        // send to right, receive from left
        send_proc = (rank + 1) % nprocs;
        recv_proc = rank-1;
        if(recv_proc < 0)
            recv_proc += nprocs;
        recv_proc_prev = 0; // the processor that we received data in the previous round

        // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
        // put the values of thoss indices in vSend to send to other procs.
        for(index_t i = 0;i < vIndexSize; ++i){
            vSend[i] = v[(vIndex[i])];
        }

        t = omp_get_wtime() - t;
        part1 += t;

//        print_vector(vSend, 0, "vSend", comm);

        tcomm = omp_get_wtime();

        if(recvCount[recv_proc] != 0){
//            if(rank==rankv) std::cout << "recv_proc: " << recv_proc << ", recvCount[recv_proc]: " << recvCount[recv_proc] << std::endl;
            MPI_Irecv(&vecValues[rdispls[recv_proc]], recvCount[recv_proc], par::Mpi_datatype<value_t>::value(), recv_proc, 1, comm, &requests[0]);
            MPI_Test(&requests[0], &flag, &statuses[0]);
        }

        if(sendCount[send_proc] != 0){
//            if(rank==rankv) std::cout << "send_proc: " << send_proc << ", sendCount[send_proc]: " << sendCount[send_proc] << std::endl;
            MPI_Isend(&vSend[vdispls[send_proc]], sendCount[send_proc], par::Mpi_datatype<value_t>::value(), send_proc, 1, comm, &requests[1]);
            MPI_Test(&requests[1], &flag, &statuses[1]);
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
    part4 += t;

    if(nprocs > 1) {
        // Wait for the first receive communication to finish.
        if (recvCount[recv_proc] != 0) {
            MPI_Wait(&requests[0], &statuses[0]);
        }
        if (sendCount[send_proc] != 0) {
            MPI_Wait(&requests[1], &statuses[1]);
        }

        int k = 0;
        while (k < nprocs) {

            ++k;
            send_proc = (send_proc + 1) % nprocs;
            recv_proc_prev = recv_proc--;
            if (recv_proc < 0)
                recv_proc += nprocs;

//            int rankv = 0;
//            if (rank == rankv) std::cout << "k: " << k << ", send_proc:" << send_proc << ", recv_proc: " << recv_proc
//                          << ", recv_proc_prev: " << recv_proc_prev << std::endl;

            if (recvCount[recv_proc] != 0) {
                MPI_Irecv(&vecValues[rdispls[recv_proc]], recvCount[recv_proc], par::Mpi_datatype<value_t>::value(), recv_proc, 1, comm, &requests[0]);
                MPI_Test(&requests[0], &flag, &statuses[0]);
            }

            if (sendCount[send_proc] != 0) {
                MPI_Isend(&vSend[vdispls[send_proc]], sendCount[send_proc], par::Mpi_datatype<value_t>::value(), send_proc, 1, comm, &requests[1]);
                MPI_Test(&requests[1], &flag, &statuses[1]);
            }

            if (recvCount[recv_proc_prev]) {

//            print_vector(vecValues, 1, "vecValues", comm);

                // perform matvec for recv_proc_prev's data
                // ----------
                // the col_index of the matrix entry does not matter. do the matvec on the first non-zero column (j=0).
                // the corresponding vector element is saved in vecValues[0]. and so on.

                t = omp_get_wtime();

                iter = recvCountScan[recv_proc_prev];
                for (index_t j = recvCountScan[recv_proc_prev]; j < recvCountScan[recv_proc_prev + 1]; ++j) {
                    for (index_t i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
//                        if(rank==rankv) printf("%ld \t%u \t%u \t%f \t%f \t%f \n",
//                        iter, row_remote[iter], col_remote2[iter], values_remote[iter], vecValues[j], values_remote[iter] * vecValues[j]);
                        w[row_remote[iter]] += values_remote[iter] * vecValues[j];
                    }
                }

                t = omp_get_wtime() - t;
                part6 += t;

            }

            // wait to finish the comm.
            if (recvCount[recv_proc] != 0) {
                MPI_Wait(&requests[0], &statuses[0]);
            }
            if (sendCount[send_proc] != 0) {
                MPI_Wait(&requests[1], &statuses[1]);
            }
        }

        delete[] requests;
        delete[] statuses;
    }

    tcomm = omp_get_wtime() - tcomm;
    part3 += tcomm;

    return 0;
}

int saena_matrix::matvec_sparse_comp(std::vector<value_t>& v, std::vector<value_t>& w) {
    // This compressed version works only when the size of send buffer to each proc. is a multiple of 4.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    double t = 0, tcomm = 0;
    MPI_Request* requests = nullptr;
    MPI_Status*  statuses = nullptr;

    ++matvec_iter;

    t = omp_get_wtime();

    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
    for(index_t i = 0;i < vIndexSize; ++i){
        vSend[i] = v[(vIndex[i])];
    }

    t = omp_get_wtime() - t;
    part1 += t;

    t = omp_get_wtime();

    if(vIndexSize || recvSize){ // todo : is this "if" required?
        zfp_stream_rewind(send_zfp);
        if(vIndexSize){
            zfp_send_comp_sz = zfp_compress(send_zfp, send_field);
            assert(zfp_send_buff_sz == zfp_send_comp_sz);
//            printf("rank %d: vIndexSize = %u, zfp_send_buff_sz = %u, \tzfp_send_comp_sz = %u\n", rank, vIndexSize, zfp_send_buff_sz, zfp_send_comp_sz);
//            if(zfp_send_buff_sz != zfp_send_comp_sz){
//                printf("ERROR: rank %d: vIndexSize = %u, zfp_send_buff_sz = %u, \tzfp_send_comp_sz = %u\n", rank, vIndexSize, zfp_send_buff_sz, zfp_send_comp_sz);
//            }
        }
    }

    t = omp_get_wtime() - t;
    part2 += t;

    tcomm = omp_get_wtime();
    requests = new MPI_Request[numSendProc+numRecvProc];
    statuses = new MPI_Status[numSendProc+numRecvProc];

    // for MPI_Test
    int flag = 0;

    // receive and put the remote parts of v in vecValues.
    // they are received in order: first put the values from the lowest rank matrix, and so on.
    for(int i = 0; i < numRecvProc; ++i) {
        MPI_Irecv(&zfp_recv_buff[(zfp_rate/CHAR_BIT)*rdispls[recvProcRank[i]]], (zfp_rate/CHAR_BIT)*recvProcCount[i], MPI_UNSIGNED_CHAR, recvProcRank[i], 1, comm, &requests[i]);
        MPI_Test(&requests[i], &flag, statuses);
    }

    for(int i = 0; i < numSendProc; ++i){
        MPI_Isend(&zfp_send_buff[(zfp_rate/CHAR_BIT)*vdispls[sendProcRank[i]]], (zfp_rate/CHAR_BIT)*sendProcCount[i], MPI_UNSIGNED_CHAR, sendProcRank[i], 1, comm, &requests[numRecvProc+i]);
        MPI_Test(&requests[numRecvProc + i], &flag, statuses);
    }

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

    t = omp_get_wtime();

    value_t* v_p  = &v[0] - split[rank];
    nnz_t    iter = 0;
    for (index_t i = 0; i < M; ++i) {
        w[i] = 0;
        for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
            w[i] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
        }
    }

    t = omp_get_wtime() - t;
    part4 += t;

    // Wait for the receive communication to finish.
    MPI_Waitall(numRecvProc, requests, statuses);

    t = omp_get_wtime();
    // decompress
    if(recvSize){
        zfp_stream_rewind(recv_zfp);
        zfp_decompress(recv_zfp, recv_field);
    }
    t = omp_get_wtime() - t;
    part5 += t;

//    print_vector(vecValues, -1, "vecValues", comm);

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
    part6 += t;

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    delete [] requests;
    delete [] statuses;

    tcomm = omp_get_wtime() - tcomm;
    part3 += tcomm;

    return 0;
}

int saena_matrix::matvec_sparse_comp2(std::vector<value_t>& v, std::vector<value_t>& w) {

    // the size of vSend and vecValues are set too big for this function.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    assert(v.size() == M);

    double t = 0;
    double tcomm = 0;
    MPI_Request* requests = nullptr;
    MPI_Status*  statuses = nullptr;

    ++matvec_iter;

//    print_info(-1);
//    print_vector(v, -1, "v", comm);

    requests = new MPI_Request[2];
    statuses = new MPI_Status[2];

    size_t desize = 0;
    nnz_t iter = 0;
    index_t recv_size_comp = 0, send_size_comp = 0;
    int send_proc = 0, recv_proc = 0, recv_proc_prev = 0;
    int flag = 0; // for MPI_Test

    if(nprocs > 1){
        // send to right, receive from left
        send_proc = (rank + 1) % nprocs;
        recv_proc = rank - 1;
        if(recv_proc < 0)
            recv_proc += nprocs;
        recv_proc_prev = 0; // the processor that we received data in the previous round

        // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
        // put the values of thoss indices in vSend to send to other procs.
        if (sendCount[send_proc] != 0) {
            // set send buff
            t = omp_get_wtime();

            iter = 0;
            for(index_t i = sendCountScan[send_proc]; i < sendCountScan[send_proc + 1]; ++i){
                vSend[iter++] = v[(vIndex[i])];
            }

            t = omp_get_wtime() - t;
            part1 += t;

            // compress
            t = omp_get_wtime();

            send_size_comp = zfp_rate / 2 * (index_t)ceil(sendCount[send_proc] / 4.0);
            zfp_send_buff_sz = send_size_comp;
            send_field = zfp_field_1d(&vSend[0], zfptype, sendCount[send_proc]);
            send_stream = stream_open(zfp_send_buff, zfp_send_buff_sz);
            send_zfp = zfp_stream_open(send_stream);
            zfp_stream_set_rate(send_zfp, zfp_rate, zfptype, 1, 0);
            zfp_stream_rewind(send_zfp);
            zfp_send_comp_sz = zfp_compress(send_zfp, send_field);
            ASSERT(zfp_send_comp_sz == send_size_comp, "zfp_send_comp_sz: " << zfp_send_comp_sz << ", send_size_comp: " << send_size_comp);

            t = omp_get_wtime() - t;
            part2 += t;
        }

        tcomm = omp_get_wtime();

        if(recvCount[recv_proc] != 0){
            recv_size_comp = zfp_rate / 2 * (index_t)ceil(recvCount[recv_proc] / 4.0);
//            if(rank==rankv) std::cout << "recv_proc: " << recv_proc << ", recvCount[recv_proc]: " << recvCount[recv_proc] << std::endl;
//            MPI_Irecv(&vecValues[0], recvCount[recv_proc], par::Mpi_datatype<value_t>::value(), recv_proc, recv_proc, comm, &requests[0]);
            MPI_Irecv(&zfp_recv_buff[0], recv_size_comp, MPI_UNSIGNED_CHAR, recv_proc, recv_proc, comm, &requests[0]);
            MPI_Test(&requests[0], &flag, &statuses[0]);
        }

        if(sendCount[send_proc] != 0){
//            if(rank==2) std::cout << "send_proc: " << send_proc << ", sendCount[send_proc]: " << sendCount[send_proc]
//                        << ", send_size_comp: " << send_size_comp << ", zfp_send_comp_sz: " << zfp_send_comp_sz << std::endl;
//            MPI_Isend(&vSend[0], sendCount[send_proc], par::Mpi_datatype<value_t>::value(), send_proc, rank, comm, &requests[1]);
            MPI_Isend(&zfp_send_buff[0], send_size_comp, MPI_UNSIGNED_CHAR, send_proc, rank, comm, &requests[1]);
            MPI_Test(&requests[1], &flag, &statuses[1]);
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
        w[i] = 0;
        for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) { // columns
//            if(rank==0) printf("%u \t%u \t%f \t%f \t%f \n",
//            row_local[indicesP_local[iter]], col_local[indicesP_local[iter]], values_local[indicesP_local[iter]], v_p[col_local[indicesP_local[iter]]], values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]]);
            w[i] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
        }
    }

    t = omp_get_wtime() - t;
    part5 += t;

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

        tcomm = omp_get_wtime() - tcomm;
        part3 += tcomm;

        std::swap(zfp_recv_buff, zfp_recv_buff2);

        int k = 1;
        while (k < nprocs + 1) {

            send_proc = (send_proc + 1) % nprocs;
            recv_proc_prev = recv_proc--;
            if (recv_proc < 0)
                recv_proc += nprocs;

//            int rankv = 0;
//            print_vector(split, rankv, "split", comm);
//            print_vector(recvCountScan, rankv, "recvCountScan", comm);
//            if(rank==rankv) std::cout << "k: " << k << ", send_proc:" << send_proc << ", recv_proc: " << recv_proc
//                          << ", recv_proc_prev: " << recv_proc_prev << "\nrecvCount[recv_proc]: "
//                          << recvCount[recv_proc] << ", recvCount[recv_proc_prev]: " << recvCount[recv_proc_prev]
//                          << ", sendCount[send_proc]: " << sendCount[send_proc] << std::endl;

            t = omp_get_wtime();

            iter = 0;
            for(index_t i = sendCountScan[send_proc]; i < sendCountScan[send_proc + 1]; ++i){
                vSend[iter++] = v[(vIndex[i])];
            }

            t = omp_get_wtime() - t;
            part1 += t;

            t = omp_get_wtime();

            if (sendCount[send_proc] != 0) {
                send_size_comp = zfp_rate / 2 * (index_t)ceil(sendCount[send_proc] / 4.0);
                zfp_send_buff_sz = send_size_comp;
                send_field = zfp_field_1d(&vSend[0], zfptype, sendCount[send_proc]);
                send_stream = stream_open(zfp_send_buff, zfp_send_buff_sz);
                send_zfp = zfp_stream_open(send_stream);
                zfp_stream_set_rate(send_zfp, zfp_rate, zfptype, 1, 0);
                zfp_stream_rewind(send_zfp);
                zfp_send_comp_sz = zfp_compress(send_zfp, send_field);
                ASSERT(zfp_send_comp_sz == send_size_comp, "zfp_send_comp_sz: " << zfp_send_comp_sz << ", send_size_comp: " << send_size_comp);
            }

            t = omp_get_wtime() - t;
            part2 += t;

            tcomm = omp_get_wtime();

            if (recvCount[recv_proc] != 0) {
                recv_size_comp = zfp_rate / 2 * (index_t)ceil(recvCount[recv_proc] / 4.0);
//                MPI_Irecv(&vecValues2[0], recvCount[recv_proc], par::Mpi_datatype<value_t>::value(), recv_proc, recv_proc, comm, &requests[0]);
                MPI_Irecv(&zfp_recv_buff[0], recv_size_comp, MPI_UNSIGNED_CHAR, recv_proc, recv_proc, comm, &requests[0]);
                MPI_Test(&requests[0], &flag, &statuses[0]);
            }

            if (sendCount[send_proc] != 0) {
//                MPI_Isend(&vSend[0], sendCount[send_proc], par::Mpi_datatype<value_t>::value(), send_proc, rank, comm, &requests[1]);
                MPI_Isend(&zfp_send_buff[0], send_size_comp, MPI_UNSIGNED_CHAR, send_proc, rank, comm, &requests[1]);
                MPI_Test(&requests[1], &flag, &statuses[1]);
            }

            if (recvCount[recv_proc_prev] != 0) {

                t = omp_get_wtime();

                zfp_recv_buff_sz = zfp_rate / 2 * (index_t)ceil(recvCount[recv_proc_prev] / 4.0);
                recv_field  = zfp_field_1d(&vecValues[0], zfptype, recvCount[recv_proc_prev]);
                recv_stream = stream_open(zfp_recv_buff2, zfp_recv_buff_sz);
                recv_zfp    = zfp_stream_open(recv_stream);
                zfp_stream_set_rate(recv_zfp, zfp_rate, zfptype, 1, 0);
                zfp_stream_rewind(recv_zfp);
                desize = zfp_decompress(recv_zfp, recv_field);

//                if (!desize) {
//                    fprintf(stderr, "decompression failed\n");
//                }

                t = omp_get_wtime() - t;
                part4 += t;

//                if(rank==rankv) printf("recv_proc_prev = %d, recvCount[recv_proc_prev] = %d, zfp_recv_buff_sz = %d\n",
//                                        recv_proc_prev, recvCount[recv_proc_prev], zfp_recv_buff_sz);
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

            // wait to finish the comm.
            if (recvCount[recv_proc] != 0) {
                MPI_Wait(&requests[0], &statuses[0]);
            }
            if (sendCount[send_proc] != 0) {
                MPI_Wait(&requests[1], &statuses[1]);
            }

            t = omp_get_wtime() - t;
            part7 += t;

            tcomm = omp_get_wtime() - tcomm;
            part3 += tcomm;

            t = omp_get_wtime();

            vecValues.swap(vecValues2);
            std::swap(zfp_recv_buff, zfp_recv_buff2);

            t = omp_get_wtime() - t;
            part6 += t;

            ++k;
        }

        delete[] requests;
        delete[] statuses;
    }

    return 0;
}

int saena_matrix::matvec_sparse_test_omp(std::vector<value_t>& v, std::vector<value_t>& w) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");

    double t, tcomm;
    MPI_Request* requests = nullptr;
    MPI_Status*  statuses = nullptr;

//    print_info(-1);
//    print_vector(v, -1, "v", comm);

//    if(nprocs > 1){
    t = omp_get_wtime();
    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
#pragma omp parallel for
    for(index_t i=0;i<vIndexSize;i++)
        vSend[i] = v[(vIndex[i])];

    t = omp_get_wtime() - t;
    part1 += t;

//    print_vector(vSend, 0, "vSend", comm);

    tcomm = omp_get_wtime();
    requests = new MPI_Request[numSendProc+numRecvProc];
    statuses = new MPI_Status[numSendProc+numRecvProc];

    // receive and put the remote parts of v in vecValues.
    // they are received in order: first put the values from the lowest rank matrix, and so on.
    for(int i = 0; i < numRecvProc; i++)
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));

    for(int i = 0; i < numSendProc; i++)
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));
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
                w[i] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
            }
        }
    }

    t = omp_get_wtime() - t;
    part4 += t;

//    if(nprocs > 1){
    // Wait for the receive communication to finish.
    MPI_Waitall(numRecvProc, requests, statuses);

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

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    delete [] requests;
    delete [] statuses;
//    }

    tcomm = omp_get_wtime() - tcomm;
    part3 += tcomm;

    return 0;
}

int saena_matrix::matvec_sparse_comp_omp(std::vector<value_t>& v, std::vector<value_t>& w) {
    // todo: add back the openmp parts

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    ++matvec_iter;

    double t = omp_get_wtime();
    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
#pragma omp parallel for
    for(index_t i = 0;i < vIndexSize; ++i){
        vSend[i] = v[(vIndex[i])];
    }

    t = omp_get_wtime() - t;
    part1 += t;

    t = omp_get_wtime();
    if(vIndexSize || recvSize){
        zfp_stream_rewind(send_zfp);
        if(vIndexSize){
            zfp_send_comp_sz = zfp_compress(send_zfp, send_field);
            assert(zfp_send_buff_sz == zfp_send_comp_sz);
//            printf("rank %d: vIndexSize = %u, zfp_send_buff_sz = %u, \tzfp_send_comp_sz = %u\n", rank, vIndexSize, zfp_send_buff_sz, zfp_send_comp_sz);
//            if(zfp_send_buff_sz != zfp_send_comp_sz){
//                printf("ERROR: rank %d: vIndexSize = %u, zfp_send_buff_sz = %u, \tzfp_send_comp_sz = %u\n", rank, vIndexSize, zfp_send_buff_sz, zfp_send_comp_sz);
//            }
        }
    }
    t = omp_get_wtime() - t;
    part2 += t;

    double tcomm = omp_get_wtime();
    auto *requests = new MPI_Request[numSendProc+numRecvProc];
    auto *statuses = new MPI_Status[numSendProc+numRecvProc];

    // receive and put the remote parts of v in vecValues.
    // they are received in order: first put the values from the lowest rank matrix, and so on.
    for(int i = 0; i < numRecvProc; ++i) {
        MPI_Irecv(&zfp_recv_buff[(zfp_rate/CHAR_BIT)*rdispls[recvProcRank[i]]], (zfp_rate/CHAR_BIT)*recvProcCount[i], MPI_UNSIGNED_CHAR, recvProcRank[i], 1, comm, &(requests[i]));
    }

    for(int i = 0; i < numSendProc; ++i){
        MPI_Isend(&zfp_send_buff[(zfp_rate/CHAR_BIT)*vdispls[sendProcRank[i]]], (zfp_rate/CHAR_BIT)*sendProcCount[i], MPI_UNSIGNED_CHAR, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));
    }

//    int flag = 0;
//    MPI_Test(requests, &flag, statuses);

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

    t = omp_get_wtime();
    value_t* v_p  = &v[0] - split[rank];
    nnz_t    iter = 0;
    for (index_t i = 0; i < M; ++i) {
        w[i] = 0;
        for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
            w[i] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
        }
    }
    t = omp_get_wtime() - t;
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

    t = omp_get_wtime();
    if(recvSize){
        zfp_stream_rewind(recv_zfp);
        zfp_decompress(recv_zfp, recv_field);
    }
    t = omp_get_wtime() - t;
    part5 += t;

//    print_vector(vecValues, -1, "vecValues", comm);

    // remote loop
    // -----------
    // the col_index of the matrix entry does not matter. do the matvec on the first non-zero column (j=0).
    // the corresponding vector element is saved in vecValues[0]. and so on.

    t = omp_get_wtime();
    index_t i;
    iter = 0;
    for (index_t j = 0; j < col_remote_size; ++j) {
        for (i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
            w[row_remote[iter]] += values_remote[iter] * vecValues[j];
        }
    }
    t = omp_get_wtime() - t;
    part6 += t;

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    delete [] requests;
    delete [] statuses;
    tcomm = omp_get_wtime() - tcomm;
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