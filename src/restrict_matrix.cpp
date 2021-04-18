#include "data_struct.h"
#include "prolong_matrix.h"
#include "restrict_matrix.h"
#include "parUtils.h" // for writeMatrixToFile()


restrict_matrix::restrict_matrix() = default;


int restrict_matrix::transposeP(prolong_matrix* P) {

    // splitNew is the row partition for restrict_matrix and split is column partition. it is the opposite of prolong_matrix and saena_matrix.

    comm = P->comm;
    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    Mbig     = P->Nbig;
    Nbig     = P->Mbig;
    split    = std::move(P->split);
    splitNew = P->splitNew;

    // set the number of rows for each process
    M = splitNew[rank+1] - splitNew[rank];

    // this is used in triple_mat_mult
    M_max = 0;
    for(index_t i = 0; i < nprocs; i++){
        if(splitNew[i+1] - splitNew[i] > M_max){
            M_max = splitNew[i+1] - splitNew[i];
        }
    }

    if(verbose_transposeP){
        MPI_Barrier(comm);
        printf("rank = %d, R.Mbig = %u, R.NBig = %u, M = %u \n", rank, Mbig, Nbig, M);
        MPI_Barrier(comm);
        printf("rank %d: transposeP part1\n", rank);
    }

#ifdef __DEBUG1__
//    P->print_info(-1);
#endif

    // *********************** send remote part of restriction ************************

    auto *requests = new MPI_Request[P->numSendProc_t + P->numRecvProc_t];
    auto *statuses  = new MPI_Status[P->numSendProc_t + P->numRecvProc_t];

    if(nprocs > 1) {

#ifdef __DEBUG1__
//    MPI_Barrier(comm);
//    if(rank==1) cout << "\nvSend_t: P->nnz_l_remote = " << P->nnz_l_remote << endl;
#endif

        // all remote entries should be sent.
        for (nnz_t i = 0; i < P->nnz_l_remote; i++) {
            P->vSend_t[i] = cooEntry(P->row_remote[i] + split[rank],
                                     P->col_remote[i],
                                     P->val_remote[i]);
        }

        // col_remote won't be used after this point
        P->col_remote.clear();
        P->col_remote.shrink_to_fit();

#ifdef __DEBUG1__
//        if(rank==0) printf("numRecvProc_t = %u \tnumSendProc_t = %u \n", P->numRecvProc_t, P->numSendProc_t);
//        print_vector(P->recvProcCount_t, 0, "recvProcCount_t", comm);
#endif

        auto dt = cooEntry::mpi_datatype();
        int flag = 1;
        for (nnz_t i = 0; i < P->numRecvProc_t; i++) {
            MPI_Irecv(&P->vecValues_t[P->rdispls_t[P->recvProcRank_t[i]]], P->recvProcCount_t[i], dt,
                      P->recvProcRank_t[i], 1, comm, &requests[i]);
            MPI_Test(&requests[i], &flag, MPI_STATUS_IGNORE);
        }

        for (nnz_t i = 0; i < P->numSendProc_t; i++) {
            MPI_Isend(&P->vSend_t[P->vdispls_t[P->sendProcRank_t[i]]], P->sendProcCount_t[i], dt,
                      P->sendProcRank_t[i], 1, comm, &requests[P->numRecvProc_t + i]);
            MPI_Test(&requests[P->numRecvProc_t + i], &flag, MPI_STATUS_IGNORE);
        }

        MPI_Type_free(&dt);
    }

    P->vdispls_t.clear();
    P->vdispls_t.shrink_to_fit();
    P->rdispls_t.clear();
    P->rdispls_t.shrink_to_fit();
    P->recvProcRank_t.clear();
    P->recvProcRank_t.shrink_to_fit();
    P->sendProcRank_t.clear();
    P->sendProcRank_t.shrink_to_fit();
    P->recvProcCount_t.clear();
    P->recvProcCount_t.shrink_to_fit();
    P->sendProcCount_t.clear();
    P->sendProcCount_t.shrink_to_fit();

    if(verbose_transposeP){
        MPI_Barrier(comm);
        printf("rank %d: transposeP part2\n", rank);
        MPI_Barrier(comm);
    }

    // *********************** assign local part of restriction ************************

//    print_vector(splitNew, 0, "splitNew", comm);

    entry.clear();

    const nnz_t nnzl = P->nnz_l_local;
    const index_t OFST = split[rank], OFSTNEW = splitNew[rank];
    for (index_t i = 0; i < nnzl; ++i) {
        entry.emplace_back(cooEntry(P->col_local[i] - OFSTNEW, // make row index local
                                    P->row_local[i] + OFST,    // make col index global
                                    P->val_local[i]));
#ifdef __DEBUG1__
        ASSERT(P->col_local[i] - OFSTNEW >= 0, "rank " << rank << ": row = " << P->col_local[i] << ", OFSTNEW = " << OFSTNEW);
        ASSERT(P->row_local[i] + OFST >= 0,    "rank " << rank << ": col = " << P->row_local[i] + OFST << ", OFST = " << OFST);
        ASSERT(P->row_local[i] + OFST < Nbig,  "rank " << rank << ": col = " << P->row_local[i] + OFST << ", OFST = " << OFST << ", Nbig = " << Nbig);
#endif
    }

#ifdef __DEBUG1__
    {
//    nnz_t iter = 0;
//    for (index_t i = 0; i < P->M; ++i) {
//        for (index_t j = 0; j < P->nnzPerRow_local[i]; ++j, ++iter) {
//            entry.emplace_back(cooEntry(P->col_local[P->indicesP_local[iter]] - OFSTNEW, // make row index local
//                                        P->row_local[P->indicesP_local[iter]] + OFST,    // make col index global
//                                        P->val_local[P->indicesP_local[iter]]));
//
//#ifdef __DEBUG1__
//            ASSERT(P->col_local[P->indicesP_local[iter]] - OFSTNEW >= 0, "rank " << rank << ": row = " << P->col_local[P->indicesP_local[iter]] << ", OFSTNEW = " << OFSTNEW);
//            ASSERT(P->row_local[P->indicesP_local[iter]] + OFST >= 0, "rank " << rank << ": col = " << P->row_local[P->indicesP_local[iter]] + OFST << ", OFST = " << OFST);
//            ASSERT(P->row_local[P->indicesP_local[iter]] + OFST < Nbig, "rank " << rank << ": col = " << P->row_local[P->indicesP_local[iter]] + OFST << ", OFST = " << OFST << ", Nbig = " << Nbig);
//#endif
//        }
//    }

//    MPI_Barrier(comm);
//    iter = 0;
//    if(rank==1){
//        cout << endl << "local:" << " rank=" << rank << endl;
//        for (i = 0; i < P->M; ++i)
//            for (j = 0; j < P->nnzPerRow_local[i]; ++j, ++iter)
//                cout << entry[iter].row << "\t" << entry[iter].col << "\t" << entry[iter].val << endl;}
//    MPI_Barrier(comm);
    }
#endif

    // clear() is being called on entry(), so shrink_to_fit() in case it was bigger before.
    entry.shrink_to_fit();

    if(verbose_transposeP){
        MPI_Barrier(comm);
        printf("rank %d: transposeP part3-1\n", rank);
        MPI_Barrier(comm);
    }

    // *********************** assign remote part of restriction ************************

    if(nprocs > 1) {

        MPI_Waitall(P->numRecvProc_t, requests, statuses);

#ifdef __DEBUG1__
//        MPI_Barrier(comm);
//        if(rank==1) cout << "vecValues_t:" << endl;
#endif

        const nnz_t pend = P->recvSize_t;
        for (nnz_t i = 0; i < pend; ++i) {
//            if(rank==2) printf("%u \t%u \t%f \n", P->vecValues_t[i].row, P->vecValues_t[i].col, P->vecValues_t[i].val);
            entry.emplace_back(cooEntry(P->vecValues_t[i].col - OFSTNEW, // make row index local
                                        P->vecValues_t[i].row,
                                        P->vecValues_t[i].val));
        }

#ifdef __DEBUG1__
        if (verbose_transposeP) {
            MPI_Barrier(comm);
            printf("rank %d: transposeP part3-2\n", rank);
            MPI_Barrier(comm);
        }
#endif

        MPI_Waitall(P->numSendProc_t, P->numRecvProc_t + requests, P->numRecvProc_t + statuses);
    }

    delete [] requests;
    delete [] statuses;

    P->vSend_t.clear();
    P->vSend_t.shrink_to_fit();
    P->vecValues_t.clear();
    P->vecValues_t.shrink_to_fit();

    std::sort(entry.begin(), entry.end());

#ifdef __DEBUG1__
//    print_vector(entry, -1, "entry", comm);
//    MPI_Barrier(comm);
//    if(rank==2){
//        cout << endl << "final after sorting:" << " rank = " << rank << "\tP->recvSize_t = " << P->recvSize_t << endl;
//        for(i=0; i<entry.size(); i++)
//            cout << i << "\t" << entry[i].row << "\t" << entry[i].col << "\t" << entry[i].val << endl;}
//    MPI_Barrier(comm);
#endif

    nnz_l = entry.size();
    MPI_Allreduce(&nnz_l, &nnz_g, 1, par::Mpi_datatype<nnz_t>::value(), MPI_SUM, comm);

#ifdef __DEBUG1__
    if(verbose_transposeP){
        MPI_Barrier(comm);
        printf("rank = %d, R.Mbig = %u, R.Nbig = %u, M = %u, R.nnz_l = %lu, R.nnz_g = %lu \n", rank, Mbig, Nbig, M, nnz_l, nnz_g);
        MPI_Barrier(comm);
        printf("rank %d: transposeP part4\n", rank);
        MPI_Barrier(comm);
    }
#endif

    // *********************** setup matvec ************************

//    nnzPerRow_local.clear();
    vElement_remote.clear();
//    vElementRep_local.clear();
//    vElementRep_remote.clear();
    nnzPerCol_remote.clear();
//    nnzPerRowScan_local.clear();
    vdispls.clear();
    rdispls.clear();
    recvProcRank.clear();
    recvProcCount.clear();
    sendProcRank.clear();
    sendProcCount.clear();
//    indicesP_local.clear();
//    indicesP_remote.clear();
//    entry_local.clear();
//    entry_remote.clear();
//    row_local.clear();
    col_local.clear();
    val_local.clear();
    row_remote.clear();
//    col_remote.clear();
    val_remote.clear();

    recvCount.resize(nprocs);
    nnzPerRow_local.assign(M, 0);
//    nnzPerRowScan_local.assign(M+1, 0);

    index_t procNum = 0, procNumTmp = 0;
    nnz_t tmp = 0, tmp2 = 0;
    nnzPerProcScan.assign(nprocs + 1, 0);
    auto *nnzProc_p = &nnzPerProcScan[1];

    assert(nnz_l == entry.size());

    // store local entries in this vector for sorting in row-major order.
    // then split it to row_loc, col_loc, val_loc.
    vector<cooEntry_row> ent_loc_row;

    nnz_t i = 0;
    while(i < nnz_l) {
        procNum = lower_bound3(&split[0], &split[nprocs], entry[i].col);
//        if(rank==0) printf("col = %u \tprocNum = %d \n", entry[i].col, procNum);

        if(procNum == rank){ // local
            while(i < nnz_l && entry[i].col < split[procNum + 1]) {
//                if(!rank) printf("entry[i].row = %d, split[rank] = %d, dif = %d\n", entry[i].row, split[rank], entry[i].row - split[rank]);
//                if(!rank) cout << entry[i] << endl;
                ++nnzPerRow_local[entry[i].row];
                ent_loc_row.emplace_back(entry[i].row, entry[i].col, entry[i].val);
//                row_local.emplace_back(entry[i].row);
//                col_local.emplace_back(entry[i].col);
//                val_local.emplace_back(entry[i].val);
                ++i;
            }

        }else{ // remote
            tmp = i;
            tmp2 = vElement_remote.size();
            while(i < nnz_l && entry[i].col < split[procNum + 1]) {
                vElement_remote.emplace_back(entry[i].col);
                nnzPerCol_remote.emplace_back(0);

                do{
                    // the original col values are not being used in matvec. the ordering starts from 0, and goes up by 1.
                    // col_remote2 is the original col value and will be used in making strength matrix.
//                    col_remote.emplace_back(vElement_remote.size() - 1);
//                    col_remote2.emplace_back(entry[i].col);
                    row_remote.emplace_back(entry[i].row);
                    val_remote.emplace_back(entry[i].val);
                    ++nnzPerCol_remote.back();
#ifdef _USE_PETSC_
//                    ++nnzPerRow_remote[entry[i].row - split[rank]];
#endif
                }while(++i < nnz_l && entry[i].col == entry[i - 1].col);
            }

            recvCount[procNum] = vElement_remote.size() - tmp2;
            nnzProc_p[procNum] = i - tmp;
        }
    } // for i

    nnz_l_local     = ent_loc_row.size();
    nnz_l_remote    = row_remote.size();
    col_remote_size = vElement_remote.size(); // number of remote columns

    recvCount[rank] = 0;

    // sort local entries in row-major order and remote entries in column-major order
    sort(ent_loc_row.begin(), ent_loc_row.end());

//    print_vector(ent_loc_row, -1, "ent_loc_row", comm);

//    row_local.resize(nnz_l_local);
    col_local.resize(nnz_l_local);
    val_local.resize(nnz_l_local);
    for(i = 0; i < nnz_l_local; ++i){
//        row_local[i] = ent_loc_row[i].row;
        col_local[i] = ent_loc_row[i].col;
        val_local[i] = ent_loc_row[i].val;
    }

    ent_loc_row.clear();
    ent_loc_row.shrink_to_fit();

    for (i = 1; i < nprocs + 1; ++i){
        nnzPerProcScan[i] += nnzPerProcScan[i - 1];
    }

#ifdef __DEBUG1__
    if(verbose_transposeP){
        MPI_Barrier(comm); printf("rank %d: transposeP part6\n", rank); MPI_Barrier(comm);
    }
#endif

    if(nprocs > 1) {
        std::vector<int> vIndexCount(nprocs);
        MPI_Alltoall(&recvCount[0], 1, MPI_INT, &vIndexCount[0], 1, MPI_INT, comm);

#ifdef __DEBUG1__
//    for(int i=0; i<nprocs; i++){
//        MPI_Barrier(comm);
//        if(rank==2) cout << "recieve from proc " << i << "\trecvCount   = " << recvCount[i] << endl;
//        MPI_Barrier(comm);
//        if(rank==2) cout << "send to proc      " << i << "\tvIndexCount = " << vIndexCount[i] << endl;
//    }
#endif

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
//                recvProcCount_t.emplace_back(recvCount_t[i]); // use sendProcRank for it.
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

        vIndexSize = (index_t)vdispls[nprocs - 1] + vIndexCount[nprocs - 1];
        recvSize   = (index_t)rdispls[nprocs - 1] + recvCount[nprocs - 1];

#ifdef __DEBUG1__
//        for (int i=0; i<nprocs; i++)
//            if(rank==0) cout << "vIndexCount[i] = " << vIndexCount[i] << "\tvdispls[i] = " << vdispls[i] << "\trecvCount[i] = " << recvCount[i] << "\trdispls[i] = " << rdispls[i] << endl;
//        MPI_Barrier(comm);
//        for (int i=0; i<nprocs; i++)
//            if(rank==0) cout << "vIndexCount[i] = " << vIndexCount[i] << "\tvdispls[i] = " << vdispls[i] << "\trecvCount[i] = " << recvCount[i] << "\trdispls[i] = " << rdispls[i] << endl;
#endif

        // vIndex is the set of indices of elements that should be sent.
        vIndex.resize(vIndexSize);
        MPI_Alltoallv(&vElement_remote[0], &recvCount[0],   &rdispls[0], par::Mpi_datatype<index_t>::value(),
                      &vIndex[0],          &vIndexCount[0], &vdispls[0], par::Mpi_datatype<index_t>::value(), comm);

        MPI_Reduce(&vIndexSize, &matvec_comm_sz, 1, par::Mpi_datatype<index_t>::value(), MPI_SUM, 0, comm);
        matvec_comm_sz /= nprocs;

        vElement_remote.clear();
        vElement_remote.shrink_to_fit();

#ifdef __DEBUG1__
//        if(!rank) printf("R: ave comm sz = %d\n", matvec_comm_sz);
        if (verbose_transposeP) {
            MPI_Barrier(comm);
            printf("rank %d: transposeP part7\n", rank);
            MPI_Barrier(comm);
        }
#endif

#pragma omp parallel for
        for (index_t i = 0; i < vIndexSize; ++i) {
//            if(rank==1) cout << vIndex[i] << "\t" << vIndex[i]-P->split[rank] << endl;
            vIndex[i] -= split[rank];
        }

        // vSend = vector values to send to other procs
        // vecValues = vector values that received from other procs
        // These will be used in matvec and they are set here to reduce the time of matvec.
        vSend.resize(vIndexSize);
        vecValues.resize(recvSize);

        vSend_f.resize(vIndexSize);
        vecValues_f.resize(recvSize);

        mv_req.resize(numSendProc + numRecvProc);
        mv_stat.resize(numSendProc + numRecvProc);
    }

//    indicesP_local.resize(nnz_l_local);
//    for(nnz_t i = 0; i < nnz_l_local; ++i)
//        indicesP_local[i] = i;
//    index_t *row_localP = &*row_local.begin();
//    std::sort(&indicesP_local[0], &indicesP_local[nnz_l_local], sort_indices(row_localP)); // NOTE: this is ordered only row-wise, not row-major.

#ifdef __DEBUG1__
    {
//    long start;
//    for(i = 0; i < M; ++i) {
//        start = nnzPerRowScan_local[i];
//        for(long j=0; j < nnzPerRow_local[i]; j++){
//            if(rank==1) printf("%lu \t %lu \t %f \n", entry_local[indicesP_local[start + j]].row+split[rank], entry_local[indicesP_local[start + j]].col, entry_local[indicesP_local[start + j]].val);
//        }
//    }

//    indicesP_remote.resize(nnz_l_remote);
//    #pragma omp parallel for
//    for(nnz_t i = 0; i < nnz_l_remote; i++)
//        indicesP_remote[i] = i;
//    index_t *row_remoteP = &*row_remote.begin();
//    std::sort(&indicesP_remote[0], &indicesP_remote[nnz_l_remote], sort_indices(row_remoteP));

//    MPI_Barrier(comm);
//    if(rank==1) cout << "nnz_l_remote = " << nnz_l_remote << "\t\trecvSize_t = " << recvSize_t << "\t\tvIndexSize_t = " << vIndexSize_t << endl;
//    if(rank==0){
//        for(i=0; i<nnz_l_remote; i++)
//            cout << row_remote[i] << "\t" << col_remote2[i] << " =\t" << val_remote[i] << "\t\t\t" << vElement_remote_t[i] << endl;
//    }
//    if(rank==0) cout << endl;
//    MPI_Barrier(comm);
    }
    if(verbose_transposeP){
        MPI_Barrier(comm);
        printf("rank %d: transposeP done!\n", rank);
        MPI_Barrier(comm);
    }
#endif

    openmp_setup();
    w_buff.resize(num_threads*M); // allocate for w_buff for matvec

    // compute nnz_max
    MPI_Allreduce(&nnz_l, &nnz_max, 1, par::Mpi_datatype<nnz_t>::value(), MPI_MAX, comm);

    // compute nnz_list
    nnz_list.resize(nprocs);
    MPI_Allgather(&nnz_l, 1, par::Mpi_datatype<nnz_t>::value(), &nnz_list[0], 1, par::Mpi_datatype<nnz_t>::value(), comm);

#ifdef __DEBUG1__
//    print_vector(nnz_list, 1, "nnz_list", comm);
#endif

    return 0;
} //end of restrictMatrix::transposeP


int restrict_matrix::openmp_setup() {

    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    if(verbose_restrict_setup) {
        MPI_Barrier(comm);
        printf("matrix_setup: rank = %d, thread1 \n", rank);
        MPI_Barrier(comm);
    }
#endif

//    printf("matrix_setup: rank = %d, Mbig = %u, M = %u, nnz_g = %u, nnz_l = %u, nnz_l_local = %u, nnz_l_remote = %u \n", rank, Mbig, M, nnz_g, nnz_l, nnz_l_local, nnz_l_remote);

#pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }

    matvec_levels = static_cast<int>( ceil( log2(num_threads) ) );

    iter_local_array.resize(num_threads+1);
    iter_remote_array.resize(num_threads+1);

#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
//        if(rank==0 && thread_id==0) std::cout << "number of procs = " << nprocs << ", number of threads = " << num_threads << std::endl;
        index_t istart = 0; // starting row index for each thread
        index_t iend = 0;   // last row index for each thread
        index_t iter_local = 0, iter_remote = 0;

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

#ifdef __DEBUG1__
    if(verbose_restrict_setup) {
        MPI_Barrier(comm);
        printf("matrix_setup: rank = %d, thread2 \n", rank);
        MPI_Barrier(comm);
    }
#endif

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


restrict_matrix::~restrict_matrix() = default;


void restrict_matrix::matvec_sparse(const value_t *v, value_t *w) {

    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    int flag = 0;
    const index_t sz = M;

    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(index_t i = 0; i < vIndexSize; ++i)
        vSend[i] = v[vIndex[i]];

//    double t20 = MPI_Wtime();
//    time[0] += (t20-t10);

//    double t13 = MPI_Wtime();
//    double t1comm = omp_get_wtime();

    for(int i = 0; i < numRecvProc; ++i){
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], par::Mpi_datatype<value_t>::value(), recvProcRank[i], 1, comm, &mv_req[i]);
//        MPI_Test(&requests[i], &flag, &statuses[i]);
    }

    MPI_Request *req_p = &mv_req[numRecvProc];
    for(int i = 0; i < numSendProc; ++i){
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], par::Mpi_datatype<value_t>::value(), sendProcRank[i], 1, comm, &req_p[i]);
        MPI_Test(&req_p[i], &flag, MPI_STATUS_IGNORE);
    }

    // initialize w to 0
    fill(&w[0], &w[sz], 0.0);

    // local loop
    // ----------
//    double t1loc = omp_get_wtime();

//    nnz_t iter = 0;
//    for (index_t i = 0; i < M; ++i) {
//        for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
//            w[i] += val_local[iter] * v_p[col_local[iter]];
//        }
//    }

#pragma omp parallel
    {
              value_t  tmp         = 0.0;
        const value_t* v_p         = &v[0] - split[rank];
        const index_t* col_local_p = nullptr;
        const value_t* val_local_p = nullptr;
              nnz_t    iter        = iter_local_array[omp_get_thread_num()];
//        nnz_t iter = 0;
#pragma omp for
        for (index_t i = 0; i < sz; ++i) {
            col_local_p = &col_local[iter];
            val_local_p = &val_local[iter];
            const index_t jend = nnzPerRow_local[i];
            tmp = 0.0;
            for (index_t j = 0; j < jend; ++j) {
//                    if(rank==0) printf("%u \t%.18f \t%.18f \t%.18f \n",
//                            entry_local[indicesP_local[iter]].col, entry_local[indicesP_local[iter]].val, v_p[entry_local[indicesP_local[iter]].col], entry_local[indicesP_local[iter]].val * v_p[entry_local[indicesP_local[iter]].col]);
//                w[i] += val_local[iter] * v_p[col_local[iter]];
                tmp += val_local_p[j] * v_p[col_local_p[j]];
            }
            w[i] += tmp;
            iter += jend;
        }
    }

//    double t2loc = omp_get_wtime();
//    tloc += (t2loc - t1loc);

    // remote loop
    // -----------
//    double t1rem = omp_get_wtime();

    const index_t* row_remote_p = nullptr;
    const value_t* val_remote_p = nullptr;
    nnz_t iter = 0;
    int recv_proc_idx = 0;
    for(int np = 0; np < numRecvProc; ++np){
        MPI_Waitany(numRecvProc, &mv_req[0], &recv_proc_idx, MPI_STATUS_IGNORE);
        const int recv_proc = recvProcRank[recv_proc_idx];
//        if(rank==1) printf("recv_proc_idx = %d, recv_proc = %d, np = %d, numRecvProc = %d, recvCount[recv_proc] = %d\n",
//                              recv_proc_idx, recv_proc, np, numRecvProc, recvCount[recv_proc]);

        iter = nnzPerProcScan[recv_proc];
        const value_t *vecValues_p        = &vecValues[rdispls[recv_proc]];
        const auto    *nnzPerCol_remote_p = &nnzPerCol_remote[rdispls[recv_proc]];
        for (index_t j = 0; j < recvCount[recv_proc]; ++j) {
//            if(rank==1) printf("%u\n", nnzPerCol_remote_p[j]);
            row_remote_p = &row_remote[iter];
            val_remote_p = &val_remote[iter];
            const index_t iend = nnzPerCol_remote_p[j];
            const value_t vrem = vecValues_p[j];
//#pragma omp simd
            for (index_t i = 0; i < iend; ++i) {
//                if(rank==1) printf("%ld \t%u \t%u \t%f \t%f\n",
//                iter, row_remote[iter], col_remote2[iter], val_remote[iter], vecValues[rdispls[recv_proc] + j]);
                w[row_remote_p[i]] += val_remote_p[i] * vrem;
            }
            iter += iend;
        }
    }

/*
    MPI_Waitall(numRecvProc, requests, statuses);
    iter = 0;
    for (index_t j = 0; j < col_remote_size; ++j) {
        for (index_t i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
            w[row_remote[iter]] += val_remote[iter] * vecValues[j];
        }
    }
*/

//    double t2rem = omp_get_wtime();
//    trem += (t2rem - t1rem);

    MPI_Waitall(numSendProc, &mv_req[numRecvProc], MPI_STATUSES_IGNORE);

//    double t2comm = omp_get_wtime();
//    ttot  += (t2comm - t1comm);
//    tcomm += (t2comm - t1comm) - (t2loc - t1loc) - (t2rem - t1rem);
}

void restrict_matrix::matvec_sparse_float(const value_t *v, value_t *w) {

    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    int flag = 0;
    const index_t sz = M;

    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(index_t i = 0; i < vIndexSize; ++i)
        vSend_f[i] = v[vIndex[i]];

//    double t20 = MPI_Wtime();
//    time[0] += (t20-t10);

//    double t13 = MPI_Wtime();
//    double t1comm = omp_get_wtime();

    for(int i = 0; i < numRecvProc; ++i){
        MPI_Irecv(&vecValues_f[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_FLOAT, recvProcRank[i], 1, comm, &mv_req[i]);
//        MPI_Test(&requests[i], &flag, &statuses[i]);
    }

    MPI_Request *req_p = &mv_req[numRecvProc];
    for(int i = 0; i < numSendProc; ++i){
        MPI_Isend(&vSend_f[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_FLOAT, sendProcRank[i], 1, comm, &req_p[i]);
        MPI_Test(&req_p[i], &flag, MPI_STATUS_IGNORE);
    }

    // initialize w to 0
    fill(&w[0], &w[sz], 0.0);

    // local loop
    // ----------
//    double t1loc = omp_get_wtime();

#pragma omp parallel
    {
              value_t  tmp         = 0.0;
        const value_t* v_p         = &v[0] - split[rank];
        const index_t* col_local_p = nullptr;
        const value_t* val_local_p = nullptr;
              nnz_t    iter        = iter_local_array[omp_get_thread_num()];
//        nnz_t iter = 0;
#pragma omp for
        for (index_t i = 0; i < sz; ++i) {
            col_local_p = &col_local[iter];
            val_local_p = &val_local[iter];
            const index_t jend = nnzPerRow_local[i];
            tmp = 0.0;
            for (index_t j = 0; j < jend; ++j) {
//                    if(rank==0) printf("%u \t%.18f \t%.18f \t%.18f \n",
//                            entry_local[indicesP_local[iter]].col, entry_local[indicesP_local[iter]].val, v_p[entry_local[indicesP_local[iter]].col], entry_local[indicesP_local[iter]].val * v_p[entry_local[indicesP_local[iter]].col]);
//                w[i] += val_local[iter] * v_p[col_local[iter]];
                tmp += val_local_p[j] * v_p[col_local_p[j]];
            }
            w[i] += tmp;
            iter += jend;
        }
    }

//    double t2loc = omp_get_wtime();
//    tloc += (t2loc - t1loc);

    // remote loop
    // -----------
//    double t1rem = omp_get_wtime();

    const index_t* row_remote_p = nullptr;
    const value_t* val_remote_p = nullptr;
    nnz_t iter = 0;
    int recv_proc_idx = 0;
    for(int np = 0; np < numRecvProc; ++np){
        MPI_Waitany(numRecvProc, &mv_req[0], &recv_proc_idx, MPI_STATUS_IGNORE);
        const int recv_proc = recvProcRank[recv_proc_idx];
//        if(rank==1) printf("recv_proc_idx = %d, recv_proc = %d, np = %d, numRecvProc = %d, recvCount[recv_proc] = %d\n",
//                              recv_proc_idx, recv_proc, np, numRecvProc, recvCount[recv_proc]);

        iter = nnzPerProcScan[recv_proc];
        const float   *vecValues_p        = &vecValues_f[rdispls[recv_proc]];
        const auto    *nnzPerCol_remote_p = &nnzPerCol_remote[rdispls[recv_proc]];
        for (index_t j = 0; j < recvCount[recv_proc]; ++j) {
//            if(rank==1) printf("%u\n", nnzPerCol_remote_p[j]);
            row_remote_p = &row_remote[iter];
            val_remote_p = &val_remote[iter];
            const index_t iend = nnzPerCol_remote_p[j];
            const value_t vrem = vecValues_p[j];
//#pragma omp simd
            for (index_t i = 0; i < iend; ++i) {
//                if(rank==1) printf("%ld \t%u \t%u \t%f \t%f\n",
//                iter, row_remote[iter], col_remote2[iter], val_remote[iter], vecValues[rdispls[recv_proc] + j]);
                w[row_remote_p[i]] += val_remote_p[i] * vrem;
            }
            iter += iend;
        }
    }

/*
    MPI_Waitall(numRecvProc, requests, statuses);
    iter = 0;
    for (index_t j = 0; j < col_remote_size; ++j) {
        for (index_t i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
            w[row_remote[iter]] += val_remote[iter] * vecValues[j];
        }
    }
*/

//    double t2rem = omp_get_wtime();
//    trem += (t2rem - t1rem);

    MPI_Waitall(numSendProc, &mv_req[numRecvProc], MPI_STATUSES_IGNORE);

//    double t2comm = omp_get_wtime();
//    ttot  += (t2comm - t1comm);
//    tcomm += (t2comm - t1comm) - (t2loc - t1loc) - (t2rem - t1rem);
}

void restrict_matrix::matvec2(std::vector<value_t>& v, std::vector<value_t>& w) {

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // put the values of the vector in vSend, for sending to other processors
    for(index_t i = 0; i < vIndexSize; ++i)
        vSend[i] = v[vIndex[i]];

//    double t20 = MPI_Wtime();
//    time[0] += (t20-t10);

//    double t13 = MPI_Wtime();
//    int flag = 0;
    auto *requests = new MPI_Request[numSendProc + numRecvProc];
    auto *statuses = new MPI_Status[numSendProc + numRecvProc];

//    double t1comm = omp_get_wtime();

    for(int i = 0; i < numRecvProc; i++){
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], par::Mpi_datatype<value_t>::value(), recvProcRank[i], 1, comm, &requests[i]);
//        MPI_Test(&requests[i], &flag, &statuses[i]);
    }

    for(int i = 0; i < numSendProc; i++){
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], par::Mpi_datatype<value_t>::value(), sendProcRank[i], 1, comm, &requests[numRecvProc+i]);
//        MPI_Test(&requests[numRecvProc + i], &flag, &statuses[numRecvProc + i]);
    }

    // local loop
    // ----------
//    double t1loc = omp_get_wtime();
    value_t* v_p = &v[0] - split[rank];
    std::fill(w.begin(), w.end(), 0);

    nnz_t iter = 0;
    for (index_t i = 0; i < M; ++i) {
        for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
//                if(rank==1) cout << entry_local[indicesP_local[iter]].col - split[rank] << "\t" << v[entry_local[indicesP_local[iter]].col - split[rank]] << endl;
            w[i] += val_local[iter] * v_p[col_local[iter]];
        }
    }

//    double t2loc = omp_get_wtime();
//    tloc += (t2loc - t1loc);

    // Wait for comm to finish.
    MPI_Waitall(numRecvProc, requests, statuses);

    // remote loop
    // -----------

//    double t1rem = omp_get_wtime();

    // remote matvec without openmp part
    iter = 0;
    for (index_t j = 0; j < col_remote_size; ++j) {
        for (index_t i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
            w[row_remote[iter]] += val_remote[iter] * vecValues[j];
        }
    }

//    double t2rem = omp_get_wtime();
//    trem += (t2rem - t1rem);

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    delete [] requests;
    delete [] statuses;

//    double t2comm = omp_get_wtime();
//    ttot  += (t2comm - t1comm);
//    tcomm += (t2comm - t1comm) - (t2loc - t1loc) - (t2rem - t1rem);
}

void restrict_matrix::matvec_omp(std::vector<value_t>& v, std::vector<value_t>& w) {

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(index_t i = 0; i < vIndexSize; ++i)
        vSend[i] = v[vIndex[i]];

//    double t20 = MPI_Wtime();
//    time[0] += (t20-t10);

//    double t13 = MPI_Wtime();
//    int flag = 0;
    auto *requests = new MPI_Request[numSendProc + numRecvProc];
    auto *statuses = new MPI_Status[numSendProc + numRecvProc];

//    double t1comm = omp_get_wtime();

    for(int i = 0; i < numRecvProc; i++){
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], par::Mpi_datatype<value_t>::value(), recvProcRank[i], 1, comm, &requests[i]);
//        MPI_Test(&requests[i], &flag, &statuses[i]);
    }

    for(int i = 0; i < numSendProc; i++){
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], par::Mpi_datatype<value_t>::value(), sendProcRank[i], 1, comm, &requests[numRecvProc+i]);
//        MPI_Test(&requests[numRecvProc + i], &flag, &statuses[numRecvProc + i]);
    }

    // local loop
    // ----------
//    double t1loc = omp_get_wtime();
    value_t* v_p = &v[0] - split[rank];
    std::fill(w.begin(), w.end(), 0.0);
#pragma omp parallel
    {
        nnz_t iter = iter_local_array[omp_get_thread_num()];
#pragma omp for
        for (index_t i = 0; i < M; ++i) {
            for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
//                if(rank==1) cout << entry_local[indicesP_local[iter]].col - split[rank] << "\t" << v[entry_local[indicesP_local[iter]].col - split[rank]] << endl;
                w[i] += val_local[iter] * v_p[col_local[iter]];
            }
        }
    }

//    double t2loc = omp_get_wtime();
//    tloc += (t2loc - t1loc);

    // Wait for comm to finish.
    MPI_Waitall(numRecvProc, requests, statuses);

    // remote loop
    // -----------

//    double t1rem = omp_get_wtime();

    #pragma omp parallel
    {
        index_t i = 0, l = 0;
        int thread_id = omp_get_thread_num();
        value_t *w_local = &w_buff[0] + (thread_id*M);
        if(thread_id==0)
            w_local = &*w.begin();
        else
            std::fill(&w_local[0], &w_local[M], 0.0);

        nnz_t iter = iter_remote_array[thread_id];
        #pragma omp for
        for (index_t j = 0; j < col_remote_size; ++j) {
            for (i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
                w_local[row_remote[iter]] += val_remote[iter] * vecValues[j];

//                if(rank==0 && thread_id==0){
//                    printf("thread = %d\n", thread_id);
//                    printf("%u \t%u \tind_rem = %lu, row = %lu \tcol = %lu \tvecVal = %f \n",
//                           i, j, indicesP_remote[iter], row_remote[indicesP_remote[iter]],
//                           col_remote[indicesP_remote[iter]], vecValues[col_remote[indicesP_remote[iter]]]);}
            }
        }

        int thread_partner = 0;
        for (l = 0; l < matvec_levels; l++) {
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

//    double t2rem = omp_get_wtime();
//    trem += (t2rem - t1rem);

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    delete [] requests;
    delete [] statuses;

//    double t2comm = omp_get_wtime();
//    ttot  += (t2comm - t1comm);
//    tcomm += (t2comm - t1comm) - (t2loc - t1loc) - (t2rem - t1rem);
}


int restrict_matrix::print_entry(int ran) const{

    // if ran >= 0 print the matrix entries on proc with rank = ran
    // otherwise print the matrix entries on all processors in order. (first on proc 0, then proc 1 and so on.)

    int rank = 0, nprocs = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(ran >= 0) {
        if (rank == ran) {
            printf("\nrestriction matrix on proc = %d \n", ran);
            printf("nnz = %lu \n", nnz_l);
            for (auto i:entry)
                std::cout << i << std::endl;
        }
    } else{
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("\nrestriction matrix on proc = %d \n", proc);
                printf("nnz = %lu \n", nnz_l);
                for (auto i:entry)
                    std::cout << i << std::endl;
            }
            MPI_Barrier(comm);
        }
    }

    return 0;
}


int restrict_matrix::print_info(int ran) const{

    // if ran >= 0 print the matrix info on proc with rank = ran
    // otherwise print the matrix info on all processors in order. (first on proc 0, then proc 1 and so on.)

    int rank = 0, nprocs = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(ran >= 0) {
        if (rank == ran) {
            printf("\nmatrix R info on proc = %d \n", ran);
            printf("Mbig = %u, \tNbig = %u, \tM = %u, \tnnz_g = %lu, \tnnz_l = %lu \n", Mbig, Nbig, M, nnz_g, nnz_l);
        }
    } else{
        MPI_Barrier(comm);
        if(rank==0) printf("\nmatrix R info:      Mbig = %u, \tNbig = %u, \tnnz_g = %lu \n", Mbig, Nbig, nnz_g);
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("matrix R on rank %d: M = %u, \tnnz_l = %lu \n", proc, M, nnz_l);
            }
            MPI_Barrier(comm);
        }
    }

    return 0;
}


int restrict_matrix::writeMatrixToFile(const std::string &name) const{
    // Create txt files with name R-r0.txt for processor 0, R-r1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat R-r0.mtx R-r1.mtx > R.mtx
    // row and column indices of txt files should start from 1, not 0.
    // write the files inside ${HOME}/folder_name
    // this is the default case for the sorting which is column-major.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(!rank) printf("estrict_matrix::writeMatrixToFile: R.splitNew is cleared. This func. needs to be updated!\n");
    return 0;

    std::string outFileNameTxt = name + "R-r" + std::to_string(rank) + ".mtx";
    std::ofstream outFileTxt(outFileNameTxt);

    if(rank==0) std::cout << "\nWriting the restriction matrix in: " << outFileNameTxt << std::endl;

    std::vector<cooEntry> entry_temp1;

    for(nnz_t i = 0; i < entry.size(); i++){
        entry_temp1.emplace_back( entry[i].row + splitNew[rank], entry[i].col, entry[i].val );
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
        outFileTxt << Mbig << "\t" << Nbig << "\t" << nnz_g << std::endl;
    }

    for (nnz_t i = 0; i < entry_temp2.size(); i++) {
//        if(rank==0) std::cout  << A->entry[i].row + 1 << "\t" << A->entry[i].col + 1 << "\t" << A->entry[i].val << std::endl;
        outFileTxt << entry_temp2[i].row + 1 << "\t" << entry_temp2[i].col + 1 << "\t" << std::setprecision(12) << entry_temp2[i].val << std::endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

    return 0;
}