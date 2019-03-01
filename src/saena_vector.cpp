#include <set>
#include "saena_vector.h"
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


int saena_vector::set(index_t row, value_t val){

//    if(fabs(val) > 1e-14){
//        entry.emplace_back(row, val);
//    }

    row += idx_offset;
    orig_order.emplace_back(row);

    if(add_duplicates){

        vecEntry temp_old;
        vecEntry temp_new = vecEntry(row, val);
        std::pair<std::set<vecEntry>::iterator, bool> p = data_set.insert(temp_new);

        if (!p.second){
            temp_old = *(p.first);
            temp_new.val += temp_old.val;

//            std::set<cooEntry_row>::iterator hint = p.first;
            auto hint = p.first;
            hint++;
            data_set.erase(p.first);
            data_set.insert(hint, temp_new);
        }

    } else {

        vecEntry temp_new = vecEntry(row, val);
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

    return 0;
}


int saena_vector::set(value_t* val, index_t size, index_t offset){

    for(index_t i = 0; i < size; i++){
        set(i + offset, val[i]);
    }

    return 0;
}

int saena_vector::set(value_t* val, index_t size){

    set(val, size, 0);

    return 0;
}

int saena_vector::remove_duplicates() {
    // parameters needed for this function:

    // parameters being set in this function:

    int nprocs, rank;
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

    if(data_sorted_dup.empty()) {
        printf("error: data_sorted_dup of the vector has no element on process %d! \n", rank);
        MPI_Finalize();
        return -1;}

    // size of data may be smaller because of duplicates. In that case its size will be reduced after finding the exact size.
//    data.resize(data_sorted_dup.size());
//    nnz_t data_size = 0;
//    if(!data_sorted_dup.empty()){
//        data[0] = data_sorted_dup[0];
//        data_size++;
//    }
//    if(add_duplicates){
//        for(nnz_t i = 1; i < data_sorted_dup.size(); i++) {
//            if (data_sorted_dup[i] == data_sorted_dup[i - 1]) {
//                data[data_size - 1].val += data_sorted_dup[i].val;
//            } else {
//                data[data_size] = data_sorted_dup[i];
//                data_size++;
//            }
//        }
//    } else {
//        for(nnz_t i = 1; i < data_sorted_dup.size(); i++){
//            if(data_sorted_dup[i] == data_sorted_dup[i - 1]){
//                data[data_size - 1] = data_sorted_dup[i];
//            }else{
//                data[data_size] = data_sorted_dup[i];
//                data_size++;
//            }
//        }
//    }
//    data.resize(data_size);
//    data.shrink_to_fit();

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
    vecEntry first_element_neighbor;
    if(rank != nprocs-1)
        MPI_Recv(&first_element_neighbor, 1, vecEntry::mpi_datatype(), rank+1, 0, comm, MPI_STATUS_IGNORE);

    if(rank!= 0)
        MPI_Send(&data[0], 1, vecEntry::mpi_datatype(), rank-1, 0, comm);

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

//    print_vector(data, -1, "final data", comm);

    return 0;
}

int saena_vector::assemble(){

    remove_duplicates();

    return 0;
}


int saena_vector::get_vec(std::vector<double> &vec){

    vec.resize(data.size());
    for(index_t i = 0; i < data.size(); i++){
        vec[i] = data[i].val;
//        std::cout << data[i].val << std::endl;
    }

    return 0;
}


int saena_vector::return_vec(std::vector<double> &u1, std::vector<double> &u2){
    // input:  u1
    // output: u2
    // if it is run in serial, only do the permutation, otherwise communication is needed for the duplicates.

    // todo: check where is the best to compute some of these variables, especially if this function is being called multiple times.
    // todo: check which variables here are not required later and can be freed at the end of this function.

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if (verbose_return_vec) {
        MPI_Barrier(comm);
        printf("return_vec: rank = %d, step 1 \n", rank);
        MPI_Barrier(comm);
//        print_vector(u1, -1, "u1", comm);
    }
//    print_vector(split, 0, "split", comm);
//    print_vector(orig_order, -1, "orig_order", comm);

    u2.resize(orig_order.size());

    if(nprocs == 1){

        for(index_t i = 0; i < orig_order.size(); i++){
            u2[i] = u1[orig_order[i]];
        }

    } else {

        // for the local indices, put the values in the correct place,
        // for the remote ones, save info for the communication
        // ----------------------------------

        long procNum;
        recvCount.assign(nprocs, 0);
        std::fill(u2.begin(), u2.end(), 0);
        for (index_t i = 0; i < orig_order.size(); i++) {
//            if(rank==1) printf("%u \t%u\n", i, orig_order[i]);
            if (orig_order[i] >= split[rank] && orig_order[i] < split[rank + 1]) {
//                if(rank==1) printf("%u \t%u \t%f\n", i, orig_order[i], u1[orig_order[i]]);
                u2[i] = u1[orig_order[i] - split[rank]];
            } else { // elements that should be received from other procs
                remote_idx.emplace_back(i);
                vElement_remote.emplace_back(orig_order[i]);
                procNum = lower_bound2(&split[0], &split[nprocs], orig_order[i]);
                recvCount[procNum]++;
            }
        }

        std::sort(vElement_remote.begin(), vElement_remote.end());
//        recvCount[rank] = 0; // don't receive from yourself.

        if (verbose_return_vec) {
            MPI_Barrier(comm);
            printf("return_vec: rank = %d, step 2 \n", rank);
            MPI_Barrier(comm);
        }
//        print_vector(u2, -1, "u2 local", comm);
//        print_vector(recvCount, -1, "recvCount", comm);
//        print_vector(remote_idx, -1, "remote_idx", comm);
//        print_vector(vElement_remote, -1, "vElement_remote", comm);

        // compute the variables for communication
        // ---------------------------------------

        sendCount.resize(nprocs);
        MPI_Alltoall(&recvCount[0], 1, MPI_INT, &sendCount[0], 1, MPI_INT, comm);

//        print_vector(sendCount, 0, "sendCount", comm);

        recvCountScan.resize(nprocs);
        sendCountScan.resize(nprocs);
        recvCountScan[0] = 0;
        sendCountScan[0] = 0;
        for (unsigned int i = 1; i < nprocs; i++) {
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

        if (verbose_return_vec) {
//            if (rank==0) std::cout << "rank=" << rank << ", numRecvProc=" << numRecvProc <<
//                                      ", numSendProc=" << numSendProc << std::endl;
            MPI_Barrier(comm);
            printf("return_vec: rank = %d, step 3 \n", rank);
            MPI_Barrier(comm);
        }

        vdispls.resize(nprocs);
        rdispls.resize(nprocs);
        vdispls[0] = 0;
        rdispls[0] = 0;

        for (int i = 1; i < nprocs; i++) {
            vdispls[i] = vdispls[i - 1] + sendCount[i - 1];
            rdispls[i] = rdispls[i - 1] + recvCount[i - 1];
        }
        vIndexSize = vdispls[nprocs - 1] + sendCount[nprocs - 1];
        recvSize   = rdispls[nprocs - 1] + recvCount[nprocs - 1];

        // vIndex: elements that should be sent to other procs.
        vIndex.resize(vIndexSize);
        MPI_Alltoallv(&vElement_remote[0], &recvCount[0], &rdispls[0], MPI_UNSIGNED,
                      &vIndex[0],          &sendCount[0], &vdispls[0], MPI_UNSIGNED, comm);

        if (verbose_return_vec) {
//            print_vector(vIndex, -1, "vIndex", comm);
            MPI_Barrier(comm);
            printf("return_vec: rank = %d, step 4 \n", rank);
            MPI_Barrier(comm);
        }

        // change the indices from global to local
#pragma omp parallel for
        for (index_t i = 0; i < vIndexSize; i++) {
            vIndex[i] -= split[rank];
        }

        // vSend     = vector values to send to other procs
        // vecValues = vector values to be received from other procs
        // These will be used in return_vec and they are set here to reduce the time of return_vec.
        vSend.resize(vIndexSize);
        vecValues.resize(recvSize);

        if (verbose_return_vec) {
//            print_vector(vIndex, 1, "vIndex", comm);
            MPI_Barrier(comm);
            printf("return_vec: rank = %d, step 5 \n", rank);
            MPI_Barrier(comm);
        }

        // perform the communication
        // ----------------------------------
        // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
        // put the values of thoss indices in vSend to send to other procs.

#pragma omp parallel for
        for (index_t i = 0; i < vIndexSize; i++)
            vSend[i] = u1[(vIndex[i])];

//        print_vector(vSend, 0, "vSend", comm);

        auto requests = new MPI_Request[numSendProc + numRecvProc];
        auto statuses = new MPI_Status[numSendProc + numRecvProc];

        // receive and put the remote parts of v in vecValues.
        // they are received in order: first put the values from the lowest rank matrix, and so on.
        for (int i = 0; i < numRecvProc; i++)
            MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm,
                      &(requests[i]));

        for (int i = 0; i < numSendProc; i++)
            MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm,
                      &(requests[numRecvProc + i]));

        MPI_Waitall(numRecvProc, requests, statuses);

        if (verbose_return_vec) {
            MPI_Barrier(comm);
            printf("return_vec: rank = %d, step 6 \n", rank);
            MPI_Barrier(comm);
        }

        // put the remote values in the output vector
        // ------------------------------------------

#pragma omp parallel for
        for (index_t i = 0; i < remote_idx.size(); i++) {
            u2[remote_idx[i]] = vecValues[i];
        }

        MPI_Waitall(numSendProc, numRecvProc + requests, numRecvProc + statuses);
        delete[] requests;
        delete[] statuses;

        if (verbose_return_vec) {
            MPI_Barrier(comm);
            printf("return_vec: rank = %d, end \n", rank);
            MPI_Barrier(comm);
        }

    }

    return 0;
}

int saena_vector::return_vec(std::vector<double> &u2){
    // input:  u2
    // output: u2

    // copy u2 to u1
    std::vector<double> u1 = u2;
    return_vec(u1, u2);

    return 0;
}


int saena_vector::print_entry(int ran){

    // if ran >= 0 print_entry the vector entries on proc with rank = ran
    // otherwise print the vector entries on all processors in order. (first on proc 0, then proc 1 and so on.)

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    index_t iter = 0;
    if(ran >= 0) {
        if (rank == ran) {
            printf("\nvector on proc = %d \n", ran);
            printf("nnz = %lu \n", data.size());
            for (auto i:data) {
                std::cout << iter << "\t" << i << std::endl;
                iter++;
            }
        }
    } else{
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("\nvector on proc = %d \n", proc);
                printf("nnz = %lu \n", data.size());
                for (auto i:data) {
                    std::cout << iter << "\t" << i << std::endl;
                    iter++;
                }
            }
            MPI_Barrier(comm);
        }
    }

    return 0;
}