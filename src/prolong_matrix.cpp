#include "prolong_matrix.h"
#include "data_struct.h"

#include "parUtils.h" // for writeMatrixToFile()


prolong_matrix::prolong_matrix() = default;


prolong_matrix::prolong_matrix(MPI_Comm com){
    comm = com;
}


prolong_matrix::~prolong_matrix() = default;


int prolong_matrix::findLocalRemote(){

    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    printf("rank=%d \t P.nnz_l=%lu \t P.nnz_g=%lu \n", rank, nnz_l, nnz_g);
//    print_vector(entry, -1, "entry", comm);

    index_t procNum = 0;
    nnz_l_local     = 0;
    nnz_l_remote    = 0;
    col_remote_size = 0; // number of remote columns
    std::vector<int> recvCount(nprocs, 0);
    nnzPerRow_local.assign(M, 0);
    nnzPerRowScan_local.assign(M + 1, 0);

    entry_remote.clear();
    row_local.clear();
    col_local.clear();
    val_local.clear();
    row_remote.clear();
//    col_remote.clear();
    val_remote.clear();
    vElementRep_local.clear();
    vElement_remote.clear();
    vElement_remote_t.clear();
    vElementRep_remote.clear();
    nnzPerCol_remote.clear();

    std::vector<int> vIndexCount_t(nprocs, 0);

    if(nnz_l != 0) {

        // take care of the first element here, since there is "col[i-1]" in the for loop below, so "i" cannot start from 0.
        if (entry[0].col >= splitNew[rank] && entry[0].col < splitNew[rank + 1]) { // local
            nnzPerRow_local[entry[0].row]++;
            row_local.emplace_back(entry[0].row);
            col_local.emplace_back(entry[0].col);
            val_local.emplace_back(entry[0].val);
            vElementRep_local.emplace_back(1);
        } else { // remote
            entry_remote.emplace_back(entry[0]);
            row_remote.emplace_back(entry[0].row);
//            col_remote.emplace_back(entry[i].col);
            val_remote.emplace_back(entry[0].val);
            nnzPerCol_remote.emplace_back(1);

            vElement_remote.emplace_back(entry[0].col);
            vElementRep_remote.emplace_back(1);
            recvCount[lower_bound3(&splitNew[0], &splitNew[nprocs], entry[0].col)] = 1;

            vElement_remote_t.emplace_back(nnz_l_remote - 1);
            vIndexCount_t[lower_bound3(&splitNew[0], &splitNew[nprocs], entry[0].col)] = 1;
        }

        for (nnz_t i = 1; i < nnz_l; i++) {

            // local
            if (entry[i].col >= splitNew[rank] && entry[i].col < splitNew[rank + 1]) {
                nnzPerRow_local[entry[i].row]++;
                row_local.emplace_back(entry[i].row);
                col_local.emplace_back(entry[i].col);
                val_local.emplace_back(entry[i].val);
                if (entry[i].col != entry[i - 1].col)
                    vElementRep_local.emplace_back(1);
                else
                    (*(vElementRep_local.end() - 1))++;

                // remote
            } else {
//                if(rank==2) printf("entry[i].row = %lu\n", entry[i].row+split[rank]);
                entry_remote.emplace_back(entry[i]);
                row_remote.emplace_back(entry[i].row);
//                col_remote.emplace_back(entry[i].col);
                val_remote.emplace_back(entry[i].val);
                procNum = lower_bound3(&splitNew[0], &splitNew[nprocs], entry[i].col);
                vIndexCount_t[procNum]++;
                vElement_remote_t.emplace_back((index_t) nnz_l_remote - 1);

                if (entry[i].col != entry[i - 1].col) {
                    vElement_remote.emplace_back(entry[i].col);
                    vElementRep_remote.emplace_back(1);
                    procNum = lower_bound3(&splitNew[0], &splitNew[nprocs], entry[i].col);
                    recvCount[procNum]++;
                    nnzPerCol_remote.emplace_back(1);
                } else {
                    (*(vElementRep_remote.end() - 1))++;
                    (*(nnzPerCol_remote.end() - 1))++;
                }
            }
        } // for i

        nnz_l_local     = row_local.size();
        nnz_l_remote    = row_remote.size();
        col_remote_size = vElement_remote.size(); // number of remote columns

//    MPI_Barrier(comm); printf("rank=%d, P.nnz_l=%lu, P.nnz_l_local=%u, P.nnz_l_remote=%u \n", rank, nnz_l, nnz_l_local, nnz_l_remote); MPI_Barrier(comm);

        for (index_t i = 0; i < M; ++i) {
            nnzPerRowScan_local[i + 1] = nnzPerRowScan_local[i] + nnzPerRow_local[i];
//        if(rank==0) printf("nnzPerRowScan_local=%d, nnzPerRow_local=%d\n", nnzPerRowScan_local[i], nnzPerRow_local[i]);
        }

    }// if(nnz_l != 0)

    if(nprocs >1) {

        std::vector<int> vIndexCount(nprocs);
        MPI_Alltoall(&recvCount[0], 1, MPI_INT, &vIndexCount[0], 1, MPI_INT, comm);

        std::vector<int> recvCount_t(nprocs);
        MPI_Alltoall(&vIndexCount_t[0], 1, MPI_INT, &recvCount_t[0], 1, MPI_INT, comm);

//        for(int i=0; i<nprocs; i++){
//            MPI_Barrier(comm);
//            if(rank==1) cout << "recieve from proc " << i << "\trecvCount   = " << recvCount[i] << "\t\trecvCount_t   = " << recvCount_t[i] << endl;
//            MPI_Barrier(comm);
//            if(rank==1) cout << "send to proc      " << i << "\tvIndexCount = " << vIndexCount[i] << "\t\tvIndexCount_t = " << vIndexCount_t[i] << endl;
//        }
//        MPI_Barrier(comm);

        recvProcRank.clear();
        recvProcCount.clear();
        sendProcRank.clear();
        sendProcCount.clear();

        numRecvProc = 0;
        numSendProc = 0;
        for (int i = 0; i < nprocs; i++) {
            if (recvCount[i] != 0) {
                numRecvProc++;
                recvProcRank.emplace_back(i);
                recvProcCount.emplace_back(recvCount[i]);
//            sendProcCount_t.emplace_back(vIndexCount_t[i]); // use recvProcRank for it.
//            if(rank==0) cout << i << "\trecvCount[i] = " << recvCount[i] << "\tvIndexCount_t[i] = " << vIndexCount_t[i] << endl;
            }
            if (vIndexCount[i] != 0) {
                numSendProc++;
                sendProcRank.emplace_back(i);
                sendProcCount.emplace_back(vIndexCount[i]);
//            recvProcCount_t.emplace_back(recvCount_t[i]); // use sendProcRank for it.
            }
        }

        //  if (rank==0) cout << "rank=" << rank << ", numRecvProc=" << numRecvProc << ", numSendProc=" << numSendProc << endl;

        vdispls.resize(nprocs);
        rdispls.resize(nprocs);
        vdispls[0] = 0;
        rdispls[0] = 0;

        for (int i = 1; i < nprocs; i++) {
            vdispls[i] = vdispls[i - 1] + vIndexCount[i - 1];
            rdispls[i] = rdispls[i - 1] + recvCount[i - 1];
        }

        vIndexSize = vdispls[nprocs - 1] + vIndexCount[nprocs - 1];
        recvSize   = rdispls[nprocs - 1] + recvCount[nprocs - 1];

//        for (int i=0; i<nprocs; i++)
//            if(rank==0) cout << "vIndexCount[i] = " << vIndexCount[i] << "\tvdispls[i] = " << vdispls[i] << "\trecvCount[i] = " << recvCount[i] << "\trdispls[i] = " << rdispls[i] << endl;
//        MPI_Barrier(comm);
//        for (int i=0; i<nprocs; i++)
//            if(rank==0) cout << "vIndexCount[i] = " << vIndexCount[i] << "\tvdispls[i] = " << vdispls[i] << "\trecvCount[i] = " << recvCount[i] << "\trdispls[i] = " << rdispls[i] << endl;

        // vIndex is the set of indices of elements that should be sent.
        vIndex.resize(vIndexSize);
        MPI_Alltoallv(&*vElement_remote.begin(), &recvCount[0], &*rdispls.begin(), par::Mpi_datatype<index_t>::value(),
                      &vIndex[0], &vIndexCount[0], &*vdispls.begin(), par::Mpi_datatype<index_t>::value(), comm);

        MPI_Reduce(&vIndexSize, &matvec_comm_sz, 1, par::Mpi_datatype<index_t>::value(), MPI_SUM, 0, comm);
        matvec_comm_sz /= nprocs;
//        if(!rank) printf("\nP: ave comm sz = %d\n", vIndexSizeAvg);

        vIndexCount.clear();
        vIndexCount.shrink_to_fit();
        recvCount.clear();
        recvCount.shrink_to_fit();

        recvProcRank_t.clear();
        recvProcCount_t.clear();
        sendProcRank_t.clear();
        sendProcCount_t.clear();

        numRecvProc_t = 0;
        numSendProc_t = 0;
        for (int i = 0; i < nprocs; i++) {
            if (recvCount_t[i] != 0) {
                numRecvProc_t++;
                recvProcRank_t.emplace_back(i);
                recvProcCount_t.emplace_back(recvCount_t[i]);
//                if(rank==2) cout << i << "\trecvCount_t[i] = " << recvCount_t[i] << endl;
            }
            if (vIndexCount_t[i] != 0) {
                numSendProc_t++;
                sendProcRank_t.emplace_back(i);
                sendProcCount_t.emplace_back(vIndexCount_t[i]);
//                if(rank==1) cout << i << "\tvIndexCount_t[i] = " << vIndexCount_t[i] << endl;
            }
        }

        vdispls_t.resize(nprocs);
        rdispls_t.resize(nprocs);
        vdispls_t[0] = 0;
        rdispls_t[0] = 0;

        for (int i = 1; i < nprocs; i++) {
//        if(rank==0) cout << "vIndexCount_t = " << vIndexCount_t[i-1] << endl;
            vdispls_t[i] = vdispls_t[i - 1] + vIndexCount_t[i - 1];
            rdispls_t[i] = rdispls_t[i - 1] + recvCount_t[i - 1];
        }

        vIndexSize_t = vdispls_t[nprocs - 1] + vIndexCount_t[nprocs - 1]; // the same as: vIndexSize_t = nnz_l_remote;
        recvSize_t   = rdispls_t[nprocs - 1] + recvCount_t[nprocs - 1];

//        MPI_Barrier(comm);
//        printf("rank = %d\tvIndexSize_t = %u\trecvSize_t = %u \n", rank, vIndexSize_t, recvSize_t);

        // todo: is this part required?
        // vElement_remote_t is the set of indices of entries that should be sent.
        // recvIndex_t       is the set of indices of entries that should be received.
//        recvIndex_t = (unsigned long*)malloc(sizeof(unsigned long)*recvSize_t);
//        MPI_Alltoallv(&(*(vElement_remote_t.begin())), vIndexCount_t, &*(vdispls_t.begin()), MPI_UNSIGNED_LONG, recvIndex_t, recvCount_t, &(*(rdispls_t.begin())), MPI_UNSIGNED_LONG, comm);
//
//        if(rank==1) cout << endl << endl;
//        for (unsigned int i=0; i<vElement_remote.size(); i++)
//        if(rank==1) cout << vElement_remote[i] << endl;

        // change the indices from global to local
        for (index_t i = 0; i < vIndexSize; i++) {
            vIndex[i] -= splitNew[rank];
        }

        // vSend = vector values to send to other procs
        // vecValues = vector values that received from other procs
        // These will be used in matvec and they are set here to reduce the time of matvec.
        vSend.resize(vIndexSize);
        vecValues.resize(recvSize);

        vSend_t.resize(vIndexSize_t);
        vecValues_t.resize(recvSize_t);
    }

    // todo: change the following two parts the same as indicesP for A in compute_coarsen, which is using entry, instead of row_local and row_remote.
    indicesP_local.resize(nnz_l_local);
    for(nnz_t i = 0; i < nnz_l_local; ++i){
        indicesP_local[i] = i;
    }

    index_t *row_localP = &*row_local.begin();
    std::sort(&indicesP_local[0], &indicesP_local[nnz_l_local], sort_indices(row_localP)); // todo: is it ordered only row-wise?
//    row_local.clear();
//    row_local.shrink_to_fit();

//    indicesP_remote.resize(nnz_l_remote);
//    for(nnz_t i=0; i<nnz_l_remote; i++)
//        indicesP_remote[i] = i;
//    index_t* row_remoteP = &*row_remote.begin();
//    std::sort(&indicesP_remote[0], &indicesP_remote[nnz_l_remote], sort_indices(row_remoteP));
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


void prolong_matrix::matvec(std::vector<value_t>& v, std::vector<value_t>& w) {

//    int nprocs;
//    MPI_Comm_size(comm, &nprocs);
    int rank;
    MPI_Comm_rank(comm, &rank);

//    print_vector(v, -1, "v in matvec", comm);

//    totalTime = 0;
//    double t10 = MPI_Wtime();

    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(index_t i = 0;i < vIndexSize; ++i)
        vSend[i] = v[vIndex[i]];

//    double t20 = MPI_Wtime();
//    time[0] += (t20-t10);

//    print_vector(vIndex, -1, "vIndex", comm);
//    print_vector(vSend, -1, "vSend", comm);

//    double t13 = MPI_Wtime();
    int flag = 0;
    auto *requests = new MPI_Request[numSendProc + numRecvProc];
    auto *statuses = new MPI_Status[numSendProc + numRecvProc];

    double t1comm = omp_get_wtime();

    for(int i = 0; i < numRecvProc; i++){
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], par::Mpi_datatype<value_t>::value(), recvProcRank[i], 1, comm, &requests[i]);
        MPI_Test(&requests[i], &flag, &statuses[i]);
    }

    for(int i = 0; i < numSendProc; i++){
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], par::Mpi_datatype<value_t>::value(), sendProcRank[i], 1, comm, &requests[numRecvProc+i]);
        MPI_Test(&requests[numRecvProc + i], &flag, &statuses[numRecvProc + i]);
    }

    // local loop
    // ----------
    double t1loc = omp_get_wtime();
    value_t* v_p = &v[0] - splitNew[rank];

    #pragma omp parallel
    {
        nnz_t iter = iter_local_array[omp_get_thread_num()];
//        nnz_t iter = 0;
        #pragma omp for
            for (index_t i = 0; i < M; ++i) {
                w[i] = 0;
                for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
//                    if(rank==0) printf("%u \t%.18f \t%.18f \t%.18f \n",
//                            entry_local[indicesP_local[iter]].col, entry_local[indicesP_local[iter]].val, v_p[entry_local[indicesP_local[iter]].col], entry_local[indicesP_local[iter]].val * v_p[entry_local[indicesP_local[iter]].col]);
//                    w[i] += entry_local[indicesP_local[iter]].val * v_p[entry_local[indicesP_local[iter]].col];
                    w[i] += val_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
                }
            }
    }

    double t2loc = omp_get_wtime();
    tloc += (t2loc - t1loc);

    MPI_Waitall(numRecvProc, requests, statuses);

//    print_vector(vecValues, 0, "vecValues", comm);

    // remote loop
    // -----------

    double t1rem = omp_get_wtime();

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

/*
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
*/

    // remote matvec without openmp part
    nnz_t iter = 0;
    for (index_t j = 0; j < col_remote_size; ++j) {
        for (index_t i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
//            if(rank==0 && entry_remote[iter].row==1) printf("%u \t%.18f \t%.18f \t%.18f \t%u \n",
//                                       entry_remote[iter].col, entry_remote[iter].val, vecValues[j], entry_remote[iter].val * vecValues[j], col_remote[j]);
//            w[entry_remote[iter].row] += entry_remote[iter].val * vecValues[j];
            w[row_remote[iter]] += val_remote[iter] * vecValues[j];
        }
    }

    double t2rem = omp_get_wtime();
    trem += (t2rem - t1rem);

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    delete [] requests;
    delete [] statuses;

    double t2comm = omp_get_wtime();
    ttot  += (t2comm - t1comm);
    tcomm += (t2comm - t1comm) - (t2loc - t1loc) - (t2rem - t1rem);
}


int prolong_matrix::print_entry(int ran){

    // if ran >= 0 print the matrix entries on proc with rank = ran
    // otherwise print the matrix entries on all processors in order. (first on proc 0, then proc 1 and so on.)

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(ran >= 0) {
        if (rank == ran) {
            printf("\nprolongation matrix on proc = %d \n", ran);
            printf("nnz = %lu \n", nnz_l);
            for (auto i:entry)
                std::cout << i << std::endl;
        }
    } else{
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("\nprolongation matrix on proc = %d \n", proc);
                printf("nnz = %lu \n", nnz_l);
                for (auto i:entry)
                    std::cout << i << std::endl;
            }
            MPI_Barrier(comm);
        }
    }

    return 0;
}


int prolong_matrix::print_info(int ran){

    // if ran >= 0 print the matrix info on proc with rank = ran
    // otherwise print the matrix info on all processors in order. (first on proc 0, then proc 1 and so on.)

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(ran >= 0) {
        if (rank == ran) {
            printf("\nmatrix P info on proc = %d \n", ran);
            printf("Mbig = %u, \tNbig = %u, \tM = %u, \tnnz_g = %lu, \tnnz_l = %lu \n", Mbig, Nbig, M, nnz_g, nnz_l);
        }
    } else{
        MPI_Barrier(comm);
        if(rank==0) printf("\nmatrix P info:      Mbig = %u, \tNbig = %u, \tnnz_g = %lu \n", Mbig, Nbig, nnz_g);
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("matrix P on rank %d: M    = %u, \tnnz_l = %lu \n", proc, M, nnz_l);
            }
            MPI_Barrier(comm);
        }
    }

    return 0;
}


int prolong_matrix::writeMatrixToFile(){
    // the matrix file will be written in the HOME directory.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(rank==0) printf("The matrix file will be written in the HOME directory. \n");
    writeMatrixToFile("");
}


int prolong_matrix::writeMatrixToFile(const char *folder_name){
    // Create txt files with name P-r0.txt for processor 0, P-r1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat P-r0.mtx P-r1.mtx > P.mtx
    // row and column indices of txt files should start from 1, not 0.
    // write the files inside ${HOME}/folder_name
    // this is the default case for the sorting which is column-major.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    const char* homeDir = getenv("HOME");

    std::ofstream outFileTxt;
    std::string outFileNameTxt = homeDir;
    outFileNameTxt += "/";
    outFileNameTxt += folder_name;
    outFileNameTxt += "/P-r";
    outFileNameTxt += std::to_string(rank);
    outFileNameTxt += ".mtx";
    outFileTxt.open(outFileNameTxt);

    if(rank==0) std::cout << "\nWriting the prolongation matrix in: " << outFileNameTxt << std::endl;

    std::vector<cooEntry> entry_temp1;

    for(nnz_t i = 0; i < entry.size(); i++){
        entry_temp1.emplace_back( entry[i].row + split[rank], entry[i].col, entry[i].val );
    }

    std::vector<cooEntry> entry_temp2;
    par::sampleSort(entry_temp1, entry_temp2, comm);

    // sort row-wise
//    std::vector<cooEntry_row> entry_temp1(entry.size());
//    std::memcpy(&*entry_temp1.begin(), &*entry.begin(), entry.size() * sizeof(cooEntry));
//    std::vector<cooEntry_row> entry_temp2;
//    par::sampleSort(entry_temp1, entry_temp2, comm);

    // first line of the file: row_size col_size nnz
    if(rank==0) {
        outFileTxt << Mbig << "\t" << Mbig << "\t" << nnz_g << std::endl;
    }

    for (nnz_t i = 0; i < entry_temp2.size(); i++) {
//        if(rank==0) std::cout  << A->entry[i].row + 1 << "\t" << A->entry[i].col + 1 << "\t" << A->entry[i].val << std::endl;
        outFileTxt << entry_temp2[i].row + 1 << "\t" << entry_temp2[i].col + 1 << "\t" << entry_temp2[i].val << std::endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

    return 0;
}
