#include "saena_vector.h"
#include "aux_functions.h"
#include "parUtils.h"

saena_vector::saena_vector() = default;

saena_vector::saena_vector(MPI_Comm com) {
    comm = com;
}

void saena_vector::set_comm(MPI_Comm com){
    comm = com;
}

saena_vector::~saena_vector() = default;


int saena_vector::set_idx_offset(const index_t offset){
    idx_offset = offset;
    return 0;
}


int saena_vector::set_dup_flag(bool add){
    add_duplicates = add;
    return 0;
}


void saena_vector::set(const index_t idx_, const value_t val){

//    if(fabs(val) > 1e-14){
//        entry.emplace_back(row, val);
//    }

    index_t idx = idx_ + idx_offset;
    orig_order.emplace_back(idx);

    if(add_duplicates){
        vecEntry temp_old;
        vecEntry temp_new = vecEntry(idx, val);
        std::pair<std::set<vecEntry>::iterator, bool> p = data_set.insert(temp_new);
        if (!p.second){
            temp_old = *(p.first);
            temp_new.val += temp_old.val;
            auto hint = p.first;
            hint++;
            data_set.erase(p.first);
            data_set.insert(hint, temp_new);
        }
    } else {
        vecEntry temp_new = vecEntry(idx, val);
        std::pair<std::set<vecEntry>::iterator, bool> p = data_set.insert(temp_new);
        if (!p.second){
            auto hint = p.first; // hint is std::set<cooEntry>::iterator
            hint++;
            data_set.erase(p.first);
//            if(!almost_zero(val))
            data_set.insert(hint, temp_new);
        }
        // if the entry is zero and it was not a duplicate, just erase it.
//        if(p.second && almost_zero(val))
//            data_set.erase(p.first);
    }
}

void saena_vector::set(const index_t* idx, const value_t* val, const index_t size){
    for(index_t i = 0; i < size; i++){
        set(idx[i], val[i]);
    }
}

void saena_vector::set(const value_t* val, const index_t size, const index_t offset /* = 0*/){
//    assert(val);
    for(index_t i = 0; i < size; i++){
        set(i + offset, val[i]);
    }
}


void saena_vector::remove_duplicates() {
    // parameters needed for this function:
    // parameters being set in this function:

    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::vector<vecEntry> data_unsorted(data_set.begin(), data_set.end());

//    print_vector(data_unsorted, -1, "data_unsorted", comm);

    std::vector<vecEntry> data_sorted_dup;
    par::sampleSort(data_unsorted, data_sorted_dup, comm);

//    print_vector(data_sorted_dup, -1, "data_sorted_dup", comm);

    // clear data_unsorted and free memory.
    data_unsorted.clear();
    data_unsorted.shrink_to_fit();

#ifdef __DEBUG1__
    for(int i = 0; i < data_sorted_dup.size() - 1; ++i){
        assert((data_sorted_dup[i].row == data_sorted_dup[i+1].row) ||
               (data_sorted_dup[i].row == data_sorted_dup[i+1].row - 1) );
    }
#endif

    if(data_sorted_dup.empty()) {
        printf("error: data_sorted_dup of the vector has no element on process %d! \n", rank);
        MPI_Finalize();
        return;
    }

/*
    // size of data may be smaller because of duplicates. In that case its size will be reduced after finding the exact size.
    data.resize(data_sorted_dup.size());
    nnz_t data_size = 0;
    if(!data_sorted_dup.empty()){
        data[0] = data_sorted_dup[0];
        data_size++;
    }
    if(add_duplicates){
        for(nnz_t i = 1; i < data_sorted_dup.size(); i++) {
            if (data_sorted_dup[i] == data_sorted_dup[i - 1]) {
                data[data_size - 1].val += data_sorted_dup[i].val;
            } else {
                data[data_size] = data_sorted_dup[i];
                data_size++;
            }
        }
    } else {
        for(nnz_t i = 1; i < data_sorted_dup.size(); i++){
            if(data_sorted_dup[i] == data_sorted_dup[i - 1]){
                data[data_size - 1] = data_sorted_dup[i];
            }else{
                data[data_size] = data_sorted_dup[i];
                data_size++;
            }
        }
    }
    data.resize(data_size);
    data.shrink_to_fit();
*/

    // remove duplicates
    // -----------------------
    nnz_t data_size_minus1 = data_sorted_dup.size()-1;
    if(add_duplicates){
        for(nnz_t i = 0; i < data_sorted_dup.size(); i++){
            data.emplace_back(data_sorted_dup[i]);
            while(i < data_size_minus1 && data_sorted_dup[i] == data_sorted_dup[i+1]){ // values of entries with the same row should be added.
    //            std::cout << data_sorted_dup[i] << "\t" << data_sorted_dup[i+1] << std::endl;
                data.back().val += data_sorted_dup[++i].val;
            }
        }
    } else {
        for(nnz_t i = 0; i < data_sorted_dup.size(); i++){
            data.emplace_back(data_sorted_dup[i]);
            while(i < data_size_minus1 && data_sorted_dup[i] == data_sorted_dup[i+1]){ // values of entries with the same row should be added.
                ++i;
            }
        }
    }

//    print_vector(data, -1, "data", comm);

    // check for dupliactes on boundary points of the processors
    // ---------------------------------------------------------
    // receive first element of your left neighbor and check if it is equal to your last element.
    auto dt = vecEntry::mpi_datatype();
    vecEntry first_element_neighbor;
    if(rank != nprocs-1)
        MPI_Recv(&first_element_neighbor, 1, dt, rank+1, 0, comm, MPI_STATUS_IGNORE);

    if(rank!= 0)
        MPI_Send(&data[0], 1, dt, rank-1, 0, comm);

    MPI_Type_free(&dt);

    vecEntry last_element = data.back();
    if(rank != nprocs-1){
        if(last_element == first_element_neighbor) {
            data.pop_back();
        }
    }

    // if duplicates should be added together:
    // then for ALL processors send my_last_element_val to the right neighbor and add it to its first element's value.
    // if(last_element == first_element_neighbor) then send last_element.val, otherwise just send 0.
    // this has the reverse communication of the previous part.
    value_t my_last_element_val = 0, left_neighbor_last_val = 0;

    if(add_duplicates){
        if(last_element == first_element_neighbor)
            my_last_element_val = last_element.val;

        if(rank != 0)
            MPI_Recv(&left_neighbor_last_val, 1, MPI_DOUBLE, rank-1, 0, comm, MPI_STATUS_IGNORE);

        if(rank!= nprocs-1)
            MPI_Send(&my_last_element_val, 1, MPI_DOUBLE, rank+1, 0, comm);

        data[0].val += left_neighbor_last_val;
    }

#ifdef __DEBUG1__
    int rhs_loc_sz = data.size(), rhs_tot_sz = 0;
    MPI_Reduce(&rhs_loc_sz, &rhs_tot_sz, 1, MPI_INT, MPI_SUM, nprocs - 1, comm);
    if(rank == nprocs - 1)
        assert(rhs_tot_sz - 1 == data_sorted_dup.back().row);
#endif

//    print_vector(data, -1, "final data", comm);
}


void saena_vector::assemble(){
    remove_duplicates();
}


index_t saena_vector::get_size() const{
    return data.size();
}


void saena_vector::get_vec(value_t *&v){
    const index_t sz = get_size();
    v = saena_aligned_alloc<value_t>(sz);
    for(index_t i = 0; i < sz; ++i){
        v[i] = data[i].val;
//        std::cout << data[i].val << std::endl;
    }
}


int saena_vector::return_vec(const value_t *u1, value_t *&u2){
    // input:  u1
    // output: u2
    // if it is run in serial, only do the permutation, otherwise communication is needed for the duplicates.

    // todo: check where is the best to compute some of these variables, especially if this function is being called multiple times.
    // todo: check which variables here are not required later and can be freed at the end of this function.

    int rank = 0, nprocs = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    if (verbose_return_vec) {
        MPI_Barrier(comm);
        printf("return_vec: rank = %d, step 1 \n", rank);
        MPI_Barrier(comm);
    }
//    print_vector(u1, -1, "u1", comm);
//    print_vector(split, 0, "rhs split", comm);
//    print_vector(orig_order, -1, "orig_order", comm);
#endif

    const index_t sz = get_size();
    const index_t orig_sz = orig_order.size();
//    u2.resize(orig_order.size());
    if(u2 != nullptr)
        saena_free(u2);
    u2 = saena_aligned_alloc<value_t>(orig_sz);

    if(nprocs == 1){

        for(index_t i = 0; i < orig_sz; ++i){
            u2[i] = u1[orig_order[i]];
        }

    } else {

        // for the local indices, put the values in the correct place,
        // for the remote ones, save info for the communication
        // ----------------------------------

        if(!return_vec_prep) {
            return_vec_prep = true;
            long procNum = 0;
            recvCount.assign(nprocs, 0);
            std::fill(&u2[0], &u2[orig_sz], 0.0);

            for (index_t i = 0; i < orig_order.size(); i++) {
//            if(rank==1) printf("%u \t%u\n", i, orig_order[i]);
                if (orig_order[i] >= split[rank] && orig_order[i] < split[rank + 1]) {
//                if(rank==1) printf("%u \t%u \t%f\n", i, orig_order[i], u1[orig_order[i]]);
                    u2[i] = u1[orig_order[i] - split[rank]];
                } else { // elements that should be received from other procs
                    remote_idx_tuple.emplace_back(i, orig_order[i]);
//                    recv_idx.emplace_back(orig_order[i]);
//                    remote_idx.emplace_back(i);
                    procNum = lower_bound2(&split[0], &split[nprocs], orig_order[i]);
                    ++recvCount[procNum];
                }
            }

            std::sort(remote_idx_tuple.begin(), remote_idx_tuple.end());
//            print_vector(remote_idx_tuple, -1, "remote_idx_tuple", comm);

            recv_idx.resize(remote_idx_tuple.size());
            for (index_t i = 0; i < remote_idx_tuple.size(); i++) {
                recv_idx[i] = remote_idx_tuple[i].idx2;
            }

#ifdef __DEBUG1__
            {
                // after sorting recv_idx, the order of remote_idx should be changed the same way. The new order is saved
                // in idx_order, so remote_idx[idx_order[i]] has the new order.
//            std::vector<index_t> idx_order(remote_idx.size());
//            #pragma omp parallel for
//            for (index_t i = 0; i < remote_idx.size(); i++)
//                idx_order[i] = i;
//            std::sort(idx_order.begin(), idx_order.end(), sort_indices(&recv_idx[0]));
//            std::sort(remote_idx.begin(), remote_idx.end(), sort_indices(&recv_idx[0]));
//
//            std::sort(remote_idx.begin(), remote_idx.end(), sort_indices(&recv_idx[0]));
//            std::sort(recv_idx.begin(), recv_idx.end());
//
//            if(rank==1){
//                for(int i = 0; i < remote_idx.size(); i++)
//                    std::cout << remote_idx[i] << "\t"<< remote_idx[idx_order[i]] << "\t" << recv_idx[i] << std::endl;
//            }

//            print_vector(u2, -1, "u2 local", comm);
//            print_vector(recvCount, -1, "recvCount", comm);
//            print_vector(remote_idx, -1, "remote_idx", comm);
//            print_vector(recv_idx, -1, "recv_idx", comm);
            }
            if (verbose_return_vec) {
                MPI_Barrier(comm);
                printf("return_vec: rank = %d, step 2 \n", rank);
                MPI_Barrier(comm);
            }
#endif

            // compute the variables for communication
            // ---------------------------------------

            sendCount.resize(nprocs);
            MPI_Alltoall(&recvCount[0], 1, MPI_INT, &sendCount[0], 1, MPI_INT, comm);

#ifdef __DEBUG1__
//            print_vector(sendCount, 0, "sendCount", comm);
#endif

            recvCountScan.resize(nprocs);
            sendCountScan.resize(nprocs);
            recvCountScan[0] = 0;
            sendCountScan[0] = 0;
            for (index_t i = 1; i < nprocs; i++) {
                recvCountScan[i] = recvCountScan[i - 1] + recvCount[i - 1];
                sendCountScan[i] = sendCountScan[i - 1] + sendCount[i - 1];
            }

            numRecvProc = 0;
            numSendProc = 0;
            for (int i = 0; i < nprocs; i++) {
                if (recvCount[i] != 0) {
                    numRecvProc++;
                    recvProcRank.emplace_back(i);
                    recvProcCount.emplace_back(recvCount[i]);
                }
                if (sendCount[i] != 0) {
                    numSendProc++;
                    sendProcRank.emplace_back(i);
                    sendProcCount.emplace_back(sendCount[i]);
                }
            }

#ifdef __DEBUG1__
            if (verbose_return_vec) {
//            if (rank==0) std::cout << "rank=" << rank << ", numRecvProc=" << numRecvProc <<
//                                      ", numSendProc=" << numSendProc << std::endl;
                MPI_Barrier(comm);
                printf("return_vec: rank = %d, step 3 \n", rank);
                MPI_Barrier(comm);
            }
#endif

            vdispls.resize(nprocs);
            rdispls.resize(nprocs);
            vdispls[0] = 0;
            rdispls[0] = 0;

            for (int i = 1; i < nprocs; i++) {
                vdispls[i] = vdispls[i - 1] + sendCount[i - 1];
                rdispls[i] = rdispls[i - 1] + recvCount[i - 1];
            }
            send_sz = vdispls[nprocs - 1] + sendCount[nprocs - 1];
            recv_sz = rdispls[nprocs - 1] + recvCount[nprocs - 1];

            // send_idx: elements that should be sent to other procs.
            send_idx.resize(send_sz);
            MPI_Alltoallv(&recv_idx[0], &recvCount[0], &rdispls[0], par::Mpi_datatype<index_t>::value(),
                          &send_idx[0], &sendCount[0], &vdispls[0], par::Mpi_datatype<index_t>::value(), comm);

#ifdef __DEBUG1__
            if (verbose_return_vec) {
//            print_vector(send_idx, -1, "send_idx", comm);
                MPI_Barrier(comm);
                printf("return_vec: rank = %d, step 4 \n", rank);
                MPI_Barrier(comm);
            }
#endif

            // change the indices from global to local
#pragma omp parallel for
            for (index_t i = 0; i < send_sz; i++) {
                send_idx[i] -= split[rank];
            }

            // send_vals = vector values to be sent to other procs
            // recv_vals = vector values to be received from other procs
            // These will be used in return_vec and they are set here to reduce the time of return_vec.
            send_vals.resize(send_sz);
            recv_vals.resize(recv_sz);

#ifdef __DEBUG1__
            if (verbose_return_vec) {
//                print_vector(send_idx, 1, "send_idx", comm);
                MPI_Barrier(comm);
                printf("return_vec: rank = %d, step 5 \n", rank);
                MPI_Barrier(comm);
            }
#endif
        }

        // perform the communication
        // ----------------------------------
        // the indices of the v on this proc that should be sent to other procs are saved in send_idx.
        // put the values of thoss indices in send_vals to send to other procs.

//#pragma omp parallel for
        for (index_t i = 0; i < send_sz; i++)
            send_vals[i] = u1[send_idx[i]];

//        print_vector(send_vals, -1, "send_vals", comm);

        auto *requests = new MPI_Request[numSendProc + numRecvProc];
        auto *statuses = new MPI_Status[numSendProc + numRecvProc];

        // receive and put the remote parts of v in recv_vals.
        // they are received in order: first put the values from the lowest rank matrix, and so on.
        for (int i = 0; i < numRecvProc; i++)
            MPI_Irecv(&recv_vals[rdispls[recvProcRank[i]]], recvProcCount[i], par::Mpi_datatype<value_t>::value(),
                      recvProcRank[i], 1, comm, &requests[i]);

        for (int i = 0; i < numSendProc; i++)
            MPI_Isend(&send_vals[vdispls[sendProcRank[i]]], sendProcCount[i], par::Mpi_datatype<value_t>::value(),
                      sendProcRank[i], 1, comm, &requests[numRecvProc + i]);

        MPI_Waitall(numRecvProc, requests, statuses);

#ifdef __DEBUG1__
        if (verbose_return_vec) {
            MPI_Barrier(comm);
            printf("return_vec: rank = %d, step 6 \n", rank);
            MPI_Barrier(comm);
        }
#endif

        // put the remote values in the output vector
        // ------------------------------------------

//#pragma omp parallel for
        for (index_t i = 0; i < remote_idx_tuple.size(); i++) {
            u2[remote_idx_tuple[i].idx1] = recv_vals[i];
        }

        MPI_Waitall(numSendProc, numRecvProc + requests, numRecvProc + statuses);
        delete[] requests;
        delete[] statuses;

#ifdef __DEBUG1__
        if (verbose_return_vec) {
            MPI_Barrier(comm);
            printf("return_vec: rank = %d, end \n", rank);
            MPI_Barrier(comm);
        }
#endif
    }

    return 0;
}

int saena_vector::return_vec(value_t *&u2){
    // input:  u2
    // output: u2

    // copy u2 to u1
    const index_t sz = get_size();
    auto *u1 = saena_aligned_alloc<value_t>(sz);
    return_vec(u1, u2);
    saena_free(u1);
    return 0;
}


int saena_vector::print_entry(int ran){

    // if ran >= 0 print_entry the vector entries on proc with rank = ran
    // otherwise print the vector entries on all processors in order. (first on proc 0, then proc 1 and so on.)

    int rank = 0, nprocs = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    index_t iter = 0;
    if(ran >= 0) {
        if (rank == ran) {
            printf("\nvector on proc = %d \n", ran);
            printf("nnz = %lu \n", data.size());
            for (const auto &i : data) {
                std::cout << iter << "\t" << std::setprecision(16) << i << std::endl;
                iter++;
            }
        }
    } else{
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("\nvector on proc = %d \n", proc);
                printf("nnz = %lu \n", data.size());
                for (const auto &i : data) {
                    std::cout << iter << "\t" << std::setprecision(16) << i << std::endl;
                    iter++;
                }
            }
            MPI_Barrier(comm);
        }
    }

    return 0;
}
