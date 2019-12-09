#include "saena_matrix.h"
#include "parUtils.h"
#include "dollar.hpp"

#include <fstream>
#include <cstring>
#include <algorithm>
//#include <sys/stat.h>
#include <omp.h>
//#include <printf.h>
#include "mpi.h"

// the following defines the reduction operation on a vector in OpenMP.
// this is used in one of the matvec implementations (matvec_timing1).
//#pragma omp declare reduction(vec_double_plus : std::vector<value_t> : \
//                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<value_t>())) \
//                    initializer(omp_priv = omp_orig)


int saena_matrix::matvec(std::vector<value_t>& v, std::vector<value_t>& w){

//    int rank;
//    MPI_Comm_rank(comm, &rank);
//    if(rank==0) printf("matvec! \n");

    if(switch_to_dense && density >= dense_threshold){
        std::cout << "dense matvec is commented out!" << std::endl;
        // uncomment to enable DENSE
//        if(!dense_matrix_generated){
//            generate_dense_matrix();
//        }
//        dense_matrix.matvec(v, w);

    }else{
        matvec_sparse(v,w);
//        matvec_sparse_zfp(v,w);
    }

    return 0;
}


int saena_matrix::matvec_sparse(std::vector<value_t>& v, std::vector<value_t>& w) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");

    MPI_Request* requests;
    MPI_Status* statuses;

//    print_info(-1);
//    print_vector(v, -1, "v", comm);

    if(nprocs > 1){
        // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
        // put the values of thoss indices in vSend to send to other procs.
#pragma omp parallel for
        for(index_t i=0;i<vIndexSize;i++)
            vSend[i] = v[(vIndex[i])];

//        print_vector(vSend, 0, "vSend", comm);

        requests = new MPI_Request[numSendProc+numRecvProc];
        statuses = new MPI_Status[numSendProc+numRecvProc];

        // receive and put the remote parts of v in vecValues.
        // they are received in order: first put the values from the lowest rank matrix, and so on.
        for(int i = 0; i < numRecvProc; i++)
            MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));

        for(int i = 0; i < numSendProc; i++)
            MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));
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
//                if(rank==0 && i==8) printf("%u \t%u \t%f \t%f \t%f \n", row_local[indicesP_local[iter]], col_local[indicesP_local[iter]], values_local[indicesP_local[iter]], v_p[col_local[indicesP_local[iter]]], values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]]);
                w[i] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
            }
        }
    }

    if(nprocs > 1){
        // Wait for the receive communication to finish.
        MPI_Waitall(numRecvProc, requests, statuses);

//        print_vector(vecValues, -1, "vecValues", comm);

        // remote loop
        // -----------
        // the col_index of the matrix entry does not matter. do the matvec on the first non-zero column (j=0).
        // the corresponding vector element is saved in vecValues[0]. and so on.

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

        // basic remote loop without openmp
//        nnz_t iter = 0;
//        for (index_t j = 0; j < col_remote_size; ++j) {
//            for (index_t i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
//                if(rank==0 && row_remote[iter]==8) printf("%u \t%u \t%f \t%f \t%f \n", row_remote[iter], col_remote2[iter], values_remote[iter], vecValues[j], values_remote[iter] * vecValues[j]);
//                w[row_remote[iter]] += values_remote[iter] * vecValues[j];
//            }
//        }

        MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
        delete [] requests;
        delete [] statuses;
    }

    return 0;
}

// different implementations of matvec
/*
int saena_matrix::matvec_timing1(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ){
//        printf("A.M != v.size() in matvec!!!\n");}

//    MPI_Barrier(comm);
    double t0_start = omp_get_wtime();
    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i=0;i<vIndexSize;i++)
        vSend[i] = v[( vIndex[i] )];
    double t0_end = omp_get_wtime();// try this: rdtsc for timing

//    print_vector(vSend, 0, "vSend", comm);

//    MPI_Barrier(comm);
    double t3_start = omp_get_wtime();
    // iSend your data, and iRecv from others
    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses  = new MPI_Status[numSendProc+numRecvProc];

    for(int i = 0; i < numRecvProc; i++)
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));

    for(int i = 0; i < numSendProc; i++)
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));

    // local loop
    // ----------
    double t1_start = omp_get_wtime();

    std::fill(w.begin(), w.end(), 0);
    value_t* v_p = &v[0] - split[rank];
#pragma omp parallel reduction(vec_double_plus:w)
    {
#pragma omp for
        for (unsigned int i = 0; i < nnz_l_local; ++i)
            w[row_local[i]] += values_local[i] * v_p[col_local[i]];
    }

    double t1_end = omp_get_wtime();

    MPI_Waitall(numRecvProc, requests, statuses);

    // remote loop
    double t2_start = omp_get_wtime();

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
//                printf("l = %d, levels = %d, thread_id = %d, thread_partner = %d \n", l, levels, thread_id, thread_partner);
                if(thread_partner < num_threads){
                    for (i = 0; i < M; i++)
                        w_local[i] += w_buff[thread_partner * M + i];
                }
            }
#pragma omp barrier
        }
    }

    double t2_end = omp_get_wtime();

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    double t3_end = omp_get_wtime();

    delete [] requests;
    delete [] statuses;

    // set vsend
    double time0_local = t0_end-t0_start;
//    double time0;
//    MPI_Allreduce(&time0_local, &time0, 1, MPI_DOUBLE, MPI_SUM, comm);
//    time[0] += time0/nprocs;

    // local loop
    double time1_local = t1_end-t1_start;
    double time1;
    MPI_Allreduce(&time1_local, &time1, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[1] += time1/nprocs;

    // remote loop
    double time2_local = t2_end-t2_start;
    double time2;
    MPI_Allreduce(&time2_local, &time2, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[2] += time2/nprocs;

    // communication = t3 + t0 - t1 - t2
    double time3_local = t3_end-t3_start + time0_local - time1_local - time2_local;
    double time3;
    MPI_Allreduce(&time3_local, &time3, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[3] += time3/nprocs;

//    time[0] += time0_local;
//    time[1] += time1_local;
//    time[2] += time2_local;
//    time[3] += time3_local;
//    time[4] += time4_local;

    return 0;
}


int saena_matrix::matvec_timing2(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ){
//        printf("A.M != v.size() in matvec!!!\n");}

//    MPI_Barrier(comm);
    double t0_start = omp_get_wtime();
    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i=0;i<vIndexSize;i++)
        vSend[i] = v[( vIndex[i] )];
    double t0_end = omp_get_wtime();// try this: rdtsc for timing

//    print_vector(vSend, 0, "vSend", comm);

//    MPI_Barrier(comm);
    double t3_start = omp_get_wtime();
    // iSend your data, and iRecv from others
    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses = new MPI_Status[numSendProc+numRecvProc];

    for(int i = 0; i < numRecvProc; i++)
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));

    for(int i = 0; i < numSendProc; i++)
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));

    // local loop
    // ----------
    double t1_start = omp_get_wtime();

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

    double t1_end = omp_get_wtime();

    MPI_Waitall(numRecvProc, requests, statuses);

//    print_vector(vecValues, 0, "vecValues", comm);

    // remote loop
    double t2_start = omp_get_wtime();

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
//                printf("l = %d, levels = %d, thread_id = %d, thread_partner = %d \n", l, levels, thread_id, thread_partner);
                if(thread_partner < num_threads){
                    for (i = 0; i < M; i++)
                        w_local[i] += w_buff[thread_partner * M + i];
                }
            }
#pragma omp barrier
        }
    }

    double t2_end = omp_get_wtime();

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    double t3_end = omp_get_wtime();

    delete [] requests;
    delete [] statuses;

    // set vsend
    double time0_local = t0_end-t0_start;
//    double time0;
//    MPI_Allreduce(&time0_local, &time0, 1, MPI_DOUBLE, MPI_SUM, comm);
//    time[0] += time0/nprocs;

    // local loop
    double time1_local = t1_end-t1_start;
    double time1;
    MPI_Allreduce(&time1_local, &time1, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[1] += time1/nprocs;

    // remote loop
    double time2_local = t2_end-t2_start;
    double time2;
    MPI_Allreduce(&time2_local, &time2, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[2] += time2/nprocs;

    // communication = t3 + t0 - t1 - t2
    double time3_local = t3_end-t3_start + time0_local - time1_local - time2_local;
    double time3;
    MPI_Allreduce(&time3_local, &time3, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[3] += time3/nprocs;

//    time[0] += time0_local;
//    time[1] += time1_local;
//    time[2] += time2_local;
//    time[3] += time3_local;
//    time[4] += time4_local;

    return 0;
}


int saena_matrix::matvec_timing3(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ){
//        printf("A.M != v.size() in matvec!!!\n");}

//    MPI_Barrier(comm);
    double t0_start = omp_get_wtime();
    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i=0;i<vIndexSize;i++)
        vSend[i] = v[( vIndex[i] )];
    double t0_end = omp_get_wtime();// try this: rdtsc for timing

//    print_vector(vSend, 0, "vSend", comm);

//    MPI_Barrier(comm);
    double t3_start = omp_get_wtime();
    // iSend your data, and iRecv from others
    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses = new MPI_Status[numSendProc+numRecvProc];

    for(int i = 0; i < numRecvProc; i++)
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));

    for(int i = 0; i < numSendProc; i++)
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));

//    if (rank==0){
//        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
//        for(int i=0; i<recvSize; i++)
//            std::cout << vecValues[i] << std::endl;}

    // local loop
    // ----------
    double t1_start = omp_get_wtime();

    value_t* v_p = &v[0] - split[rank];
    // local loop
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        long iter = iter_local_array2[thread_id];
#pragma omp for
        for (unsigned int i = 0; i < M; ++i) {
            w[i] = 0;
            for (unsigned int j = 0; j < nnzPerRow_local2[i]; ++j, ++iter) {
                w[i] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
            }
        }

        for (iter = iter_local_array[thread_id]; iter < iter_local_array2[thread_id]; ++iter)
            w[row_local[indicesP_local[iter]]] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
    }

    double t1_end = omp_get_wtime();

    MPI_Waitall(numRecvProc, requests, statuses);

//    print_vector(vecValues, 0, "vecValues", comm);

    // remote loop
    double t2_start = omp_get_wtime();

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
//                printf("l = %d, levels = %d, thread_id = %d, thread_partner = %d \n", l, levels, thread_id, thread_partner);
                if(thread_partner < num_threads){
                    for (i = 0; i < M; i++)
                        w_local[i] += w_buff[thread_partner * M + i];
                }
            }
#pragma omp barrier
        }
    }

    double t2_end = omp_get_wtime();

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    double t3_end = omp_get_wtime();

    delete [] requests;
    delete [] statuses;

    // set vsend
    double time0_local = t0_end-t0_start;
//    double time0;
//    MPI_Allreduce(&time0_local, &time0, 1, MPI_DOUBLE, MPI_SUM, comm);
//    time[0] += time0/nprocs;

    // local loop
    double time1_local = t1_end-t1_start;
    double time1;
    MPI_Allreduce(&time1_local, &time1, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[1] += time1/nprocs;

    // remote loop
    double time2_local = t2_end-t2_start;
    double time2;
    MPI_Allreduce(&time2_local, &time2, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[2] += time2/nprocs;

    // communication = t3 + t0 - t1 - t2
    double time3_local = t3_end-t3_start + time0_local - time1_local - time2_local;
    double time3;
    MPI_Allreduce(&time3_local, &time3, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[3] += time3/nprocs;

//    time[0] += time0_local;
//    time[1] += time1_local;
//    time[2] += time2_local;
//    time[3] += time3_local;
//    time[4] += time4_local;

    return 0;
}


int saena_matrix::matvec_timing4(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ){
//        printf("A.M != v.size() in matvec!!!\n");}

//    MPI_Barrier(comm);
    double t0_start = omp_get_wtime();
    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i=0;i<vIndexSize;i++)
        vSend[i] = v[( vIndex[i] )];
    double t0_end = omp_get_wtime();// try this: rdtsc for timing

//    print_vector(vSend, 0, "vSend", comm);

//    MPI_Barrier(comm);
    double t3_start = omp_get_wtime();
    // iSend your data, and iRecv from others
    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses = new MPI_Status[numSendProc+numRecvProc];

    for(int i = 0; i < numRecvProc; i++)
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));

    for(int i = 0; i < numSendProc; i++)
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));

//    if (rank==0){
//        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
//        for(int i=0; i<recvSize; i++)
//            std::cout << vecValues[i] << std::endl;}

    // local loop
    // ----------
    double t1_start = omp_get_wtime();

    // by doing this you will have a local index for v[col_local[i]].
    value_t* v_p = &v[0] - split[rank];
#pragma omp parallel
    {
        unsigned int i, l;
        int thread_id = omp_get_thread_num();
        value_t *w_local = &w_buff[0] + (thread_id*M);
        if(thread_id==0)
            w_local = &*w.begin();

        std::fill(&w_local[0], &w_local[M], 0);

#pragma omp for
        for (i = 0; i < nnz_l_local; ++i)
            w_local[row_local[i]] += values_local[i] * v_p[col_local[i]];

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

    double t1_end = omp_get_wtime();

    MPI_Waitall(numRecvProc, requests, statuses);

//    print_vector(vecValues, 0, "vecValues", comm);

    // remote loop
    double t2_start = omp_get_wtime();

    // this version is not parallel with OpenMP.
#if 0
    nnz_t iter = 0;
    for (index_t j = 0; j < col_remote_size; ++j) {
        for (index_t i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
            w[row_remote[iter]] += values_remote[iter] * vecValues[j];

//                if(rank==0 && omp_get_thread_num()==0){
//                    printf("thread = %d\n", omp_get_thread_num());
//                    printf("%u \t%u \tind_rem = %lu, row = %lu \tcol = %lu \tvecVal = %f \n",
//                           i, j, indicesP_remote[iter], row_remote[indicesP_remote[iter]],
//                           col_remote[indicesP_remote[iter]], vecValues[col_remote[indicesP_remote[iter]]]);}
        }
    }
#endif

    // the previous part is parallelized with OpenMP.
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
//                printf("l = %d, levels = %d, thread_id = %d, thread_partner = %d \n", l, levels, thread_id, thread_partner);
                if(thread_partner < num_threads){
                    for (i = 0; i < M; i++)
                        w_local[i] += w_buff[thread_partner * M + i];
                }
            }
#pragma omp barrier
        }
    }

    double t2_end = omp_get_wtime();

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    double t3_end = omp_get_wtime();

    delete [] requests;
    delete [] statuses;

    // set vsend
    double time0_local = t0_end-t0_start;
//    double time0;
//    MPI_Allreduce(&time0_local, &time0, 1, MPI_DOUBLE, MPI_SUM, comm);
//    time[0] += time0/nprocs;

    // local loop
    double time1_local = t1_end-t1_start;
    double time1;
    MPI_Allreduce(&time1_local, &time1, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[1] += time1/nprocs;

    // remote loop
    double time2_local = t2_end-t2_start;
    double time2;
    MPI_Allreduce(&time2_local, &time2, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[2] += time2/nprocs;

    // communication = t3 + t0 - t1 - t2
    double time3_local = t3_end-t3_start + time0_local - time1_local - time2_local;
    double time3;
    MPI_Allreduce(&time3_local, &time3, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[3] += time3/nprocs;

//    time[0] += time0_local;
//    time[1] += time1_local;
//    time[2] += time2_local;
//    time[3] += time3_local;
//    time[4] += time4_local;

    return 0;
}


int saena_matrix::matvec_timing4_alltoall(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time) {
// todo: to reduce the communication during matvec, consider reducing number of columns during coarsening,
// todo: instead of reducing general non-zeros, since that is what is communicated for matvec.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ){
//        printf("A.M != v.size() in matvec!!!\n");}

    double t0_start = omp_get_wtime();
    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i = 0; i < vIndexSize; i++)
        vSend[i] = v[( vIndex[i] )];
    double t0_end = omp_get_wtime();// try this: rdtsc for timing

//    print_vector(vSend, 0, "vSend", comm);

//    double t3_start = omp_get_wtime();

    double t4_start = omp_get_wtime();
    MPI_Alltoallv(&vSend[0], &sendCount[0], &sendCountScan[0], MPI_INT, &vecValues[0], &recvCount[0], &recvCountScan[0], MPI_INT, comm);
    double t4_end = omp_get_wtime();

//    print_vector(vecValues, 0, "vecValues", comm);

    double t1_start = omp_get_wtime();

    // local loop
    value_t* v_p = &v[0] - split[rank];
#pragma omp parallel
    {
        unsigned int i, l, idx;
        int thread_id = omp_get_thread_num();
        value_t *w_local = &w_buff[0] + (thread_id*M);
        if(thread_id==0)
            w_local = &*w.begin();

        std::fill(&w_local[0], &w_local[M], 0);

#pragma omp for
        for (i = 0; i < nnz_l_local; ++i)
            w_local[row_local[i]] += values_local[i] * v_p[col_local[i]];

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

    double t1_end = omp_get_wtime();

    // Wait for the communication to finish.
//    double t4_start = omp_get_wtime();
//    MPI_Waitall(numSendProc+numRecvProc, requests, statuses);
//    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
//    double t4_end = omp_get_wtime();

    // remote loop
    // todo: data race happens during "omp for" here, since the "for" loop splits based on the remote columns, but
    // todo: w[row] are being computed in every iteration , which means different threads may access the same w[row].

    double t2_start = omp_get_wtime();

    nnz_t iter = iter_remote_array[omp_get_thread_num()];
//#pragma omp for
    for (index_t j = 0; j < col_remote_size; ++j) {
        for (index_t i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
            w[row_remote[iter]] += values_remote[iter] * vecValues[j];

//                if(rank==0 && omp_get_thread_num()==0){
//                    printf("thread = %d\n", omp_get_thread_num());
//                    printf("%u \t%u \tind_rem = %lu, row = %lu \tcol = %lu \tvecVal = %f \n",
//                           i, j, indicesP_remote[iter], row_remote[indicesP_remote[iter]],
//                           col_remote[indicesP_remote[iter]], vecValues[col_remote[indicesP_remote[iter]]]);}
        }
    }

    double t2_end = omp_get_wtime();
//    double t3_end = omp_get_wtime();

#if 0
    double time0_local = t0_end-t0_start;
    double time0;
    MPI_Allreduce(&time0_local, &time0, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time1_local = t1_end-t1_start;
    double time1;
    MPI_Allreduce(&time1_local, &time1, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time2_local = t2_end-t2_start;
    double time2;
    MPI_Allreduce(&time2_local, &time2, 1, MPI_DOUBLE, MPI_SUM, comm);

//    double time3_local = t3_end-t3_start;
//    double time3;
//    MPI_Allreduce(&time3_local, &time3, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time4_local = t4_end-t4_start;
    double time4;
    MPI_Allreduce(&time4_local, &time4, 1, MPI_DOUBLE, MPI_SUM, comm);

    time[0] += time0/nprocs;
    time[1] += time1/nprocs;
    time[2] += time2/nprocs;
//    time[3] += time3/nprocs;
    time[4] += time4/nprocs;
#endif

//    time[0] += time0_local;
//    time[1] += time1_local;
//    time[2] += time2_local;
//    time[3] += time3_local;
//    time[4] += time4_local;

    return 0;
}


int saena_matrix::matvec_timing5(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time) {
    // old remote loop is used here.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ){
//        printf("A.M != v.size() in matvec!!!\n");}

    if(rank==0) printf("\n\nWARNING: indicesP_remote is not set. uncomment it in find_sorting() function to use matvec_timing5().\n\n");

    double t0_start = omp_get_wtime();
    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i=0;i<vIndexSize;i++)
        vSend[i] = v[( vIndex[i] )];
    double t0_end = omp_get_wtime();// try this: rdtsc for timing

//    print_vector(vSend, 0, "vSend", comm);

    double t3_start = omp_get_wtime();
    // iSend your data, and iRecv from others
    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses = new MPI_Status[numSendProc+numRecvProc];

    for(int i = 0; i < numRecvProc; i++)
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));

    for(int i = 0; i < numSendProc; i++)
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));

    // local loop
    // ----------
    double t1_start = omp_get_wtime();
    // by doing this you will have a local index for v[col_local[i]].
    value_t* v_p = &v[0] - split[rank];
#pragma omp parallel
    {
        unsigned int i, l, idx;
        int thread_id = omp_get_thread_num();
        value_t *w_local = &w_buff[0] + (thread_id*M);
        if(thread_id==0)
            w_local = &*w.begin();

        std::fill(&w_local[0], &w_local[M], 0);

#pragma omp for
        for (i = 0; i < nnz_l_local; ++i)
            w_local[row_local[i]] += values_local[i] * v_p[col_local[i]];

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

    double t1_end = omp_get_wtime();

    // Wait for the communication to finish.
    double t4_start = omp_get_wtime();
//    MPI_Waitall(numSendProc+numRecvProc, requests, statuses);
    MPI_Waitall(numRecvProc, requests, statuses);
    double t4_end = omp_get_wtime();

//    print_vector(vecValues, 0, "vecValues", comm);

    // remote loop
    // todo: data race happens during "omp for" here, since the "for" loop splits based on the remote columns, but
    // todo: w[row] are being computed in every iteration , which means different threads may access the same w[row].

    double t2_start = omp_get_wtime();
//#pragma omp parallel
//    {
    unsigned int iter = iter_remote_array[omp_get_thread_num()];
//#pragma omp for
    for (unsigned int i = 0; i < col_remote_size; ++i) {
        for (unsigned int j = 0; j < nnzPerCol_remote[i]; ++j, ++iter) {
            w[row_remote[indicesP_remote[iter]]] += values_remote[indicesP_remote[iter]] * vecValues[col_remote[indicesP_remote[iter]]];

//                if(rank==0 && omp_get_thread_num()==0){
//                    printf("thread = %d\n", omp_get_thread_num());
//                    printf("%u \t%u \tind_rem = %lu, row = %lu \tcol = %lu \tvecVal = %f \n",
//                           i, j, indicesP_remote[iter], row_remote[indicesP_remote[iter]],
//                           col_remote[indicesP_remote[iter]], vecValues[col_remote[indicesP_remote[iter]]]);}
        }
    }
//    }

    double t2_end = omp_get_wtime();

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    double t3_end = omp_get_wtime();

    delete [] requests;
    delete [] statuses;

#if 0
    double time0_local = t0_end-t0_start;
    double time0;
    MPI_Allreduce(&time0_local, &time0, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time1_local = t1_end-t1_start;
    double time1;
    MPI_Allreduce(&time1_local, &time1, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time2_local = t2_end-t2_start;
    double time2;
    MPI_Allreduce(&time2_local, &time2, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time3_local = t3_end-t3_start;
    double time3;
    MPI_Allreduce(&time3_local, &time3, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time4_local = t4_end-t4_start;
    double time4;
    MPI_Allreduce(&time4_local, &time4, 1, MPI_DOUBLE, MPI_SUM, comm);

    time[0] += time0/nprocs;
    time[1] += time1/nprocs;
    time[2] += time2/nprocs;
    time[3] += time3/nprocs;
    time[4] += time4/nprocs;
#endif

//    time[0] += time0_local;
//    time[1] += time1_local;
//    time[2] += time2_local;
//    time[3] += time3_local;
//    time[4] += time4_local;

    return 0;
}


int saena_matrix::matvec_timing5_alltoall(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time) {
// todo: to reduce the communication during matvec, consider reducing number of columns during coarsening,
// todo: instead of reducing general non-zeros, since that is what is communicated for matvec.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if( v.size() != M ){
        printf("A.M != v.size() in matvec!!!\n");}

    if(rank==0) printf("\n\nWARNING: indicesP_remote is not set. uncomment it in find_sorting() function to use matvec_timing5_alltoall().\n\n");

    double t0_start = omp_get_wtime();
    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i = 0; i < vIndexSize; i++)
        vSend[i] = v[( vIndex[i] )];
    double t0_end = omp_get_wtime();// try this: rdtsc for timing

//    print_vector(vSend, 0, "vSend", comm);

//    double t3_start = omp_get_wtime();

    double t4_start = omp_get_wtime();
    MPI_Alltoallv(&vSend[0], &sendCount[0], &sendCountScan[0], MPI_INT, &vecValues[0], &recvCount[0], &recvCountScan[0], MPI_INT, comm);
    double t4_end = omp_get_wtime();

//    if (rank==0){
//        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
//        for(int i=0; i<recvSize; i++)
//            std::cout << vecValues[i] << std::endl;
//    }

    double t1_start = omp_get_wtime();
    value_t* v_p = &v[0] - split[rank];
    // local loop
//    std::fill(&*w.begin(), &*w.end(), 0);
#pragma omp parallel
    {
        long iter = iter_local_array[omp_get_thread_num()];
#pragma omp for
        for (unsigned int i = 0; i < M; ++i) {
            w[i] = 0;
            for (unsigned int j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
                w[i] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
            }
        }
    }

    double t1_end = omp_get_wtime();

    // Wait for the communication to finish.
//    double t4_start = omp_get_wtime();
//    MPI_Waitall(numSendProc+numRecvProc, requests, statuses);
//    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
//    double t4_end = omp_get_wtime();

//    print_vector(vecValues, 0, "vecValues", comm);

    // remote loop
    // todo: data race happens during "omp for" here, since the "for" loop splits based on the remote columns, but
    // todo: w[row] are being computed in every iteration , which means different threads may access the same w[row].

    double t2_start = omp_get_wtime();
//#pragma omp parallel
//    {
    unsigned int iter = iter_remote_array[omp_get_thread_num()];
//#pragma omp for
    for (unsigned int i = 0; i < col_remote_size; ++i) {
        for (unsigned int j = 0; j < nnzPerCol_remote[i]; ++j, ++iter) {

//                if(rank==0 && omp_get_thread_num()==0){
//                    printf("thread = %d\n", omp_get_thread_num());
//                    printf("%u \t%u \tind_rem = %lu, row = %lu \tcol = %lu \tvecVal = %f \n",
//                           i, j, indicesP_remote[iter], row_remote[indicesP_remote[iter]],
//                           col_remote[indicesP_remote[iter]], vecValues[col_remote[indicesP_remote[iter]]]);}

            w[row_remote[indicesP_remote[iter]]] += values_remote[indicesP_remote[iter]] * vecValues[col_remote[indicesP_remote[iter]]];
        }
    }
//    }
    double t2_end = omp_get_wtime();
//    double t3_end = omp_get_wtime();

#if 0
    double time0_local = t0_end-t0_start;
    double time0;
    MPI_Allreduce(&time0_local, &time0, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time1_local = t1_end-t1_start;
    double time1;
    MPI_Allreduce(&time1_local, &time1, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time2_local = t2_end-t2_start;
    double time2;
    MPI_Allreduce(&time2_local, &time2, 1, MPI_DOUBLE, MPI_SUM, comm);

    //    double time3_local = t3_end-t3_start;
    //    double time3;
    //    MPI_Allreduce(&time3_local, &time3, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time4_local = t4_end-t4_start;
    double time4;
    MPI_Allreduce(&time4_local, &time4, 1, MPI_DOUBLE, MPI_SUM, comm);

    time[0] += time0/nprocs;
    time[1] += time1/nprocs;
    time[2] += time2/nprocs;
    //    time[3] += time3/nprocs;
    time[4] += time4/nprocs;
#endif

//    time[0] += time0_local;
//    time[1] += time1_local;
//    time[2] += time2_local;
//    time[3] += time3_local;
//    time[4] += time4_local;

    return 0;
}
*/