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


int saena_matrix::assemble() {

    if(!assembled){
        repartition_nnz_initial();
        matrix_setup();
        if(enable_shrink) compute_matvec_dummy_time();
    }else{
        repartition_nnz_update();
        matrix_setup_update();
    }

    return 0;
}


int saena_matrix::setup_initial_data(){
    // parameters needed for this function:
    // comm, data_coo

    // parameters being set in this function:
    // Mbig, initial_nnz_l, nnz_g, data

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    std::cout << rank << " : " << __func__ << initial_nnz_l << std::endl;

    std::set<cooEntry_row>::iterator it;
    cooEntry_row temp;
    nnz_t iter = 0;
    index_t Mbig_local = 0;

    // read this: https://stackoverflow.com/questions/5034211/c-copy-set-to-vector
    data_unsorted.resize(data_coo.size());
    for(it = data_coo.begin(); it != data_coo.end(); ++it){
        data_unsorted[iter] = *it;
        ++iter;

        temp = *it;
        if(temp.col > Mbig_local)
            Mbig_local = temp.col;
    }

    // Mbig is the size of the matrix, which is the maximum of rows and columns.
    // Up to here Mbig_local is the maximum of cols.
    // last element's row is the maximum of rows, since data_coo is sorted row-major.

    if(data_unsorted.back().row > Mbig_local)
        Mbig_local = data_unsorted[iter].row;

    MPI_Allreduce(&Mbig_local, &Mbig, 1, MPI_UNSIGNED, MPI_MAX, comm);
    Mbig++; // since indices start from 0, not 1.
//    std::cout << "Mbig = " << Mbig << std::endl;

    remove_duplicates();

    initial_nnz_l = data.size();
    MPI_Allreduce(&initial_nnz_l, &nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
//    MPI_Barrier(comm); printf("rank = %d, Mbig = %u, nnz_g = %u, initial_nnz_l = %u \n", rank, Mbig, nnz_g, initial_nnz_l); MPI_Barrier(comm);

    return 0;
}


int saena_matrix::setup_initial_data2(){
    // parameters needed for this function:
    // comm, data_coo

    // parameters being set in this function:
    // Mbig, initial_nnz_l, nnz_g, data

    int rank;
    MPI_Comm_rank(comm, &rank);

//    std::cout << rank << " : " << __func__ << initial_nnz_l << std::endl;

    std::set<cooEntry_row>::iterator it;
    nnz_t iter = 0;

    data_unsorted.resize(data_coo.size());
    for(it = data_coo.begin(); it != data_coo.end(); ++it){
        data_unsorted[iter] = *it;
        ++iter;
    }

    remove_duplicates();

    initial_nnz_l = data.size();
    nnz_t nnz_g_temp = nnz_g;
    MPI_Allreduce(&initial_nnz_l, &nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
    if((nnz_g_temp != nnz_g) && rank == 0){
        printf("error: number of global nonzeros is changed during the matrix update:\nbefore: %lu \tafter: %lu", nnz_g, nnz_g_temp);
        MPI_Finalize();
        return -1;
    }

    return 0;
}


int saena_matrix::remove_duplicates() {
    // parameters needed for this function:
    // comm, data_unsorted

    // parameters being set in this function:
    // data

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // todo: free memory for data_coo. consider move semantics. check if the following idea is correct.
    // clear data_coo and free memory.
    // Using move semantics, the address of data_coo is swapped with data_temp. So data_coo will be empty
    // and data_temp will be deleted when this function is finished.
    std::set<cooEntry_row> data_temp = std::move(data_coo);

    // since shrink_to_fit() does not work for std::set, set.erase() is being used, not sure if it frees memory.
//    data_coo.erase(data_coo.begin(), data_coo.end());
//    data_coo.clear();

//    printf("rank = %d \t\t\t before sort: data_unsorted size = %lu\n", rank, data_unsorted.size());

//    for(int i=0; i<data_unsorted.size(); i++)
//        if(rank==0) std::cout << data_unsorted[i] << std::endl;

//    par::sampleSort(data_unsorted, comm);

    std::vector<cooEntry_row> data_sorted_row;
    par::sampleSort(data_unsorted, data_sorted_row, comm);

    // clear data_unsorted and free memory.
    data_unsorted.clear();
    data_unsorted.shrink_to_fit();

    if(data_sorted_row.empty()) {
        printf("error: data has no element on process %d! \n", rank);
        MPI_Finalize();
        return -1;}

    // switch from cooEntry_row to cooEntry.
    std::vector<cooEntry> data_sorted(data_sorted_row.size());
    memcpy(&data_sorted[0], &data_sorted_row[0], data_sorted_row.size() * sizeof(cooEntry));

//    printf("rank = %d \t\t\t after  sort: data_sorted size = %lu\n", rank, data_sorted.size());
//    print_vector(data_sorted, -1, "data_sorted", comm);

    // size of data may be smaller because of duplicates. In that case its size will be reduced after finding the exact size.
    data.resize(data_sorted.size());

    // put the first element of data_unsorted to data.
    nnz_t data_size = 0;
    if(!data_sorted.empty()){
        data[0] = data_sorted[0];
        data_size++;
    }

    for(nnz_t i=1; i<data_sorted.size(); i++){
        if(data_sorted[i] == data_sorted[i-1]){
            if(add_duplicates){
                data[data_size-1].val += data_sorted[i].val;
            }else{
                data[data_size-1] = data_sorted[i];
            }
        }else{
            data[data_size] = data_sorted[i];
            data_size++;
        }
    }

    data.resize(data_size);
    data.shrink_to_fit();

    // receive first element of your left neighbor and check if it is equal to your last element.
    cooEntry first_element_neighbor;
    if(rank != nprocs-1)
        MPI_Recv(&first_element_neighbor, 1, cooEntry::mpi_datatype(), rank+1, 0, comm, MPI_STATUS_IGNORE);

    if(rank!= 0)
        MPI_Send(&data[0], 1, cooEntry::mpi_datatype(), rank-1, 0, comm);

    cooEntry last_element = data.back();
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

    return 0;
}


int saena_matrix::matrix_setup() {
    // before using this function the following parameters of saena_matrix should be set:
    // "Mbig", "M", "nnz_g", "split", "entry",

    if(active) {
        int nprocs, rank;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu \n", rank, Mbig, M, nnz_g, nnz_l);
            MPI_Barrier(comm);
        }

//        printf("matrix_setup: rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu \n", rank, Mbig, M, nnz_g, nnz_l);
//        print_vector(entry, -1, "entry", comm);

        assembled = true;
        freeBoolean = true; // use this parameter to know if destructor for saena_matrix class should free the variables or not.
        total_active_procs = nprocs;

        // *************************** set the inverse of diagonal of A (for smoothers) ****************************

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, inv_diag \n", rank);
            MPI_Barrier(comm);
        }

        inv_diag.resize(M);
        inverse_diag();

//        print_vector(inv_diag, -1, "inv_diag", comm);

        // *************************** set rho ****************************

//        set_rho();

        // *************************** set and exchange on-diagonal and off-diagonal elements ****************************

        set_off_on_diagonal();

        // *************************** find sortings ****************************

        find_sortings();

        // *************************** find start and end of each thread for matvec ****************************
        // also, find nnz per row for local and remote matvec

        openmp_setup();
        w_buff.resize(num_threads*M); // allocate for w_buff for matvec3()

        // *************************** scale ****************************
        // scale the matrix to have its diagonal entries all equal to 1.

//        print_vector(entry, 0, "entry", comm);
        scale_matrix();
//        print_vector(entry, 0, "entry", comm);

        // *************************** print_entry info ****************************

/*
        nnz_t total_nnz_l_local;
        nnz_t total_nnz_l_remote;
        MPI_Allreduce(&nnz_l_local,  &total_nnz_l_local,  1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
        MPI_Allreduce(&nnz_l_remote, &total_nnz_l_remote, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
        int local_percent  = int(100*(float)total_nnz_l_local/nnz_g);
        int remote_percent = int(100*(float)total_nnz_l_remote/nnz_g);
        if(rank==0) printf("\nMbig = %u, nnz_g = %lu, total_nnz_l_local = %lu (%%%d), total_nnz_l_remote = %lu (%%%d) \n",
                           Mbig, nnz_g, total_nnz_l_local, local_percent, total_nnz_l_remote, remote_percent);

//        printf("rank %d: col_remote_size = %u \n", rank, col_remote_size);
        index_t col_remote_size_min, col_remote_size_ave, col_remote_size_max;
        MPI_Allreduce(&col_remote_size, &col_remote_size_min, 1, MPI_UNSIGNED, MPI_MIN, comm);
        MPI_Allreduce(&col_remote_size, &col_remote_size_ave, 1, MPI_UNSIGNED, MPI_SUM, comm);
        MPI_Allreduce(&col_remote_size, &col_remote_size_max, 1, MPI_UNSIGNED, MPI_MAX, comm);
        if(rank==0) printf("\nremote_min = %u, remote_ave = %u, remote_max = %u \n",
                           col_remote_size_min, (col_remote_size_ave/nprocs), col_remote_size_max);
*/

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, done \n", rank);
            MPI_Barrier(comm);
        }

    } // if(active)
    return 0;
}


int saena_matrix::matrix_setup_no_scale(){
    // before using this function the following parameters of saena_matrix should be set:
    // "Mbig", "M", "nnz_g", "split", "entry",

    // todo: here: check if there is another if(active) before calling this function.
    if(active) {
        int nprocs, rank;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);

#pragma omp parallel
        if(rank==0 && omp_get_thread_num()==0) printf("\nnumber of processes = %d, number of threads = %d\n\n", nprocs, omp_get_num_threads());

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu \n", rank, Mbig, M, nnz_g, nnz_l);
            MPI_Barrier(comm);}

//        print_vector(entry, -1, "entry", comm);

        assembled = true;
        freeBoolean = true; // use this parameter to know if destructor for saena_matrix class should free the variables or not.
        total_active_procs = nprocs;

        // *************************** set the inverse of diagonal of A (for smoothers) ****************************

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, inv_diag \n", rank);
            MPI_Barrier(comm);
        }

        inv_diag.resize(M);
        inverse_diag();

//        print_vector(inv_diag, -1, "inv_diag", comm);

        // *************************** set rho ****************************

//        set_rho();

        // *************************** set and exchange on-diagonal and off-diagonal elements ****************************

        set_off_on_diagonal();

        // *************************** find sortings ****************************

        find_sortings();

        // *************************** find start and end of each thread for matvec ****************************
        // also, find nnz per row for local and remote matvec

        openmp_setup();
        w_buff.resize(num_threads*M); // allocate for w_buff for matvec3()

        // *************************** scale ****************************
        // scale the matrix to have its diagonal entries all equal to 1.

//        scale_matrix();

        // *************************** print_entry info ****************************

/*
        nnz_t total_nnz_l_local;
        nnz_t total_nnz_l_remote;
        MPI_Allreduce(&nnz_l_local,  &total_nnz_l_local,  1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
        MPI_Allreduce(&nnz_l_remote, &total_nnz_l_remote, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
        int local_percent  = int(100*(float)total_nnz_l_local/nnz_g);
        int remote_percent = int(100*(float)total_nnz_l_remote/nnz_g);
        if(rank==0) printf("\nMbig = %u, nnz_g = %lu, total_nnz_l_local = %lu (%%%d), total_nnz_l_remote = %lu (%%%d) \n",
                           Mbig, nnz_g, total_nnz_l_local, local_percent, total_nnz_l_remote, remote_percent);

//        printf("rank %d: col_remote_size = %u \n", rank, col_remote_size);
        index_t col_remote_size_min, col_remote_size_ave, col_remote_size_max;
        MPI_Allreduce(&col_remote_size, &col_remote_size_min, 1, MPI_UNSIGNED, MPI_MIN, comm);
        MPI_Allreduce(&col_remote_size, &col_remote_size_ave, 1, MPI_UNSIGNED, MPI_SUM, comm);
        MPI_Allreduce(&col_remote_size, &col_remote_size_max, 1, MPI_UNSIGNED, MPI_MAX, comm);
        if(rank==0) printf("\nremote_min = %u, remote_ave = %u, remote_max = %u \n",
                           col_remote_size_min, (col_remote_size_ave/nprocs), col_remote_size_max);
*/

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, done \n", rank);
            MPI_Barrier(comm);
        }

    } // if(active)
    return 0;
}


int saena_matrix::matrix_setup_update() {
    // update values_local, values_remote and inv_diag.

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

//    assembled = true;

    // todo: check if instead of clearing and pushing back, it is possible to only update the values.
    values_local.clear();
    values_remote.clear();

    if(!entry.empty()) {
        for (nnz_t i = 0; i < nnz_l; i++) {
            if (entry[i].col >= split[rank] && entry[i].col < split[rank + 1]) {
                values_local.emplace_back(entry[i].val);
            } else {
                values_remote.emplace_back(entry[i].val);
            }
        }
    }

//    inv_diag.resize(M);
    inverse_diag();

    scale_matrix();

    return 0;
}


int saena_matrix::matrix_setup_update_no_scale() {
    // update values_local, values_remote and inv_diag.

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

//    assembled = true;

    // todo: check if instead of clearing and pushing back, it is possible to only update the values.
    values_local.clear();
    values_remote.clear();

    if(!entry.empty()) {
        for (nnz_t i = 0; i < nnz_l; i++) {
            if (entry[i].col >= split[rank] && entry[i].col < split[rank + 1]) {
                values_local.emplace_back(entry[i].val);
            } else {
                values_remote.emplace_back(entry[i].val);
            }
        }
    }

    inv_diag.resize(M);
    inverse_diag();

    scale_matrix();

    return 0;
}


/*
int saena_matrix::set_rho(){

    // computing rhoDA for the prolongation matrix: P = (I - 4/(3*rhoDA) * DA) * P_t
    // rhoDA = min( norm(DA , 1) , norm(DA , inf) )

        double norm1_local = 0;
        for(unsigned long i=0; i<M; i++)
            norm1_local += abs(inv_diag[i]);
        MPI_Allreduce(&norm1_local, &norm1, 1, MPI_DOUBLE, MPI_SUM, comm);

        double normInf_local = inv_diag[0];
        for(unsigned long i=1; i<M; i++)
            if( abs(inv_diag[i]) > normInf_local )
                normInf_local = abs(inv_diag[i]);
        MPI_Allreduce(&normInf_local, &normInf, 1, MPI_DOUBLE, MPI_MAX, comm);

        if(normInf < norm1)
            rhoDA = normInf;
        else
            rhoDA = norm1;

    return 0;
}
*/

int saena_matrix::set_off_on_diagonal(){
    // set and exchange on-diagonal and off-diagonal elements
    // on-diagonal (local) elements are elements that correspond to vector elements which are local to this process.
    // off-diagonal (remote) elements correspond to vector elements which should be received from another processes.

    if(active){
        int nprocs, rank;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, local remote1 \n", rank);
            MPI_Barrier(comm);
        }

//        print_entry(-1);

        col_remote_size = 0;
        nnz_l_local = 0;
        nnz_l_remote = 0;
        recvCount.assign(nprocs, 0);
        nnzPerRow_local.assign(M, 0);
        if(nprocs > 1){
            nnzPerRow_remote.assign(M, 0);
        }
//        nnzPerRow.assign(M,0);
//        nnzPerCol_local.assign(Mbig,0); // Nbig = Mbig, assuming A is symmetric.
//        nnzPerCol_remote.assign(M,0);

        // take care of the first element here, since there is "col[i-1]" in the for loop below, so "i" cannot start from 0.
//        nnzPerRow[row[0]-split[rank]]++;
        long procNum;
        if(!entry.empty()){
            if (entry[0].col >= split[rank] && entry[0].col < split[rank + 1]) {
                nnzPerRow_local[entry[0].row - split[rank]]++;
//                nnzPerCol_local[col[0]]++;
                nnz_l_local++;
                values_local.push_back(entry[0].val);
                row_local.push_back(entry[0].row - split[rank]);
                col_local.push_back(entry[0].col);
                //vElement_local.push_back(col[0]);
//                vElementRep_local.push_back(1);

            } else {
                nnz_l_remote++;
                nnzPerRow_remote[entry[0].row - split[rank]]++;
                values_remote.push_back(entry[0].val);
                row_remote.push_back(entry[0].row - split[rank]);
                col_remote_size++;
                col_remote.push_back(col_remote_size - 1);
                col_remote2.push_back(entry[0].col);
                nnzPerCol_remote.push_back(1);
                vElement_remote.push_back(entry[0].col);
                vElementRep_remote.push_back(1);
//                if(rank==1) printf("col = %u \tprocNum = %ld \n", entry[0].col, lower_bound3(&split[0], &split[nprocs], entry[0].col));
                recvCount[lower_bound2(&split[0], &split[nprocs], entry[0].col)] = 1;
            }
        }

        if(entry.size() >= 2){
            for (nnz_t i = 1; i < nnz_l; i++) {
//                nnzPerRow[row[i]-split[rank]]++;
//                if(rank==0) std::cout << entry[i] << std::endl;
                if (entry[i].col >= split[rank] && entry[i].col < split[rank + 1]) {
                    nnz_l_local++;
//                    nnzPerCol_local[col[i]]++;
                    nnzPerRow_local[entry[i].row - split[rank]]++;
                    values_local.push_back(entry[i].val);
                    row_local.push_back(entry[i].row - split[rank]);
                    col_local.push_back(entry[i].col);

//                    if (entry[i].col != entry[i - 1].col)
//                        vElementRep_local.push_back(1);
//                    else
//                        vElementRep_local.back()++;
                } else {
                    nnz_l_remote++;
                    nnzPerRow_remote[entry[i].row - split[rank]]++;
                    values_remote.push_back(entry[i].val);
                    row_remote.push_back(entry[i].row - split[rank]);
                    // col_remote2 is the original col value and will be used in making strength matrix. col_remote will be used for matevec.
                    col_remote2.push_back(entry[i].col);

                    if (entry[i].col != entry[i - 1].col) {
                        col_remote_size++;
                        vElement_remote.push_back(entry[i].col);
                        vElementRep_remote.push_back(1);
                        procNum = lower_bound2(&split[0], &split[nprocs], entry[i].col);
//                        if(rank==1) printf("col = %u \tprocNum = %ld \n", entry[i].col, procNum);
                        recvCount[procNum]++;
                        nnzPerCol_remote.push_back(1);
                    } else {
                        vElementRep_remote.back()++;
                        nnzPerCol_remote.back()++;
                    }
                    // the original col values are not being used. the ordering starts from 0, and goes up by 1.
                    col_remote.push_back(col_remote_size - 1);
                }
            } // for i
        }

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, local remote2 \n", rank);
            MPI_Barrier(comm);
        }

        // don't receive anything from yourself
        recvCount[rank] = 0;

//        print_vector(recvCount, 0, "recvCount", comm);

        if(nprocs != 1){
            sendCount.resize(nprocs);
            MPI_Alltoall(&recvCount[0], 1, MPI_INT, &sendCount[0], 1, MPI_INT, comm);

//        print_vector(sendCount, 0, "sendCount", comm);

            recvCountScan.resize(nprocs);
            sendCountScan.resize(nprocs);
            recvCountScan[0] = 0;
            sendCountScan[0] = 0;
            for (unsigned int i = 1; i < nprocs; i++){
                recvCountScan[i] = recvCountScan[i-1] + recvCount[i-1];
                sendCountScan[i] = sendCountScan[i-1] + sendCount[i-1];
            }

            numRecvProc = 0;
            numSendProc = 0;
            for (int i = 0; i < nprocs; i++) {
                if (recvCount[i] != 0) {
                    numRecvProc++;
                    recvProcRank.push_back(i);
                    recvProcCount.push_back(recvCount[i]);
                }
                if (sendCount[i] != 0) {
                    numSendProc++;
                    sendProcRank.push_back(i);
                    sendProcCount.push_back(sendCount[i]);
                }
            }
//            if (rank==0) std::cout << "rank=" << rank << ", numRecvProc=" << numRecvProc << ", numSendProc=" << numSendProc << std::endl;

            if(verbose_matrix_setup) {
                MPI_Barrier(comm);
                printf("matrix_setup: rank = %d, local remote3 \n", rank);
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
            recvSize = rdispls[nprocs - 1] + recvCount[nprocs - 1];

            vIndex.resize(vIndexSize);
            MPI_Alltoallv(&vElement_remote[0], &recvCount[0], &rdispls[0], MPI_UNSIGNED,
                          &vIndex[0],          &sendCount[0], &vdispls[0], MPI_UNSIGNED, comm);

//    print_vector(vIndex, -1, "vIndex", comm);

            if(verbose_matrix_setup) {
                MPI_Barrier(comm);
                printf("matrix_setup: rank = %d, local remote4 \n", rank);
                MPI_Barrier(comm);
            }

            // change the indices from global to local
#pragma omp parallel for
            for (index_t i = 0; i < vIndexSize; i++)
                vIndex[i] -= split[rank];


            // vSend = vector values to send to other procs
            // vecValues = vector values that received from other procs
            // These will be used in matvec and they are set here to reduce the time of matvec.
            vSend.resize(vIndexSize);
            vecValues.resize(recvSize);

            vSendULong.resize(vIndexSize);
            vecValuesULong.resize(recvSize);

//            send_bufsize = rate / 2 * (unsigned)ceil(vIndexSize/4.0); // rate/8 * 4 * ceil(size/4). This is in bytes.
//            recv_bufsize = rate / 2 * (unsigned)ceil(recvSize/4.0);
//            send_buffer = (unsigned char*)malloc(8*send_bufsize);
//            recv_buffer = (unsigned char*)malloc(8*recv_bufsize);
            zfp_send_bufsize = rate / 2 * (unsigned)ceil(vIndexSize/4.0); // rate/8 * 4 * ceil(size/4). This is in bytes.
            zfp_recv_bufsize = rate / 2 * (unsigned)ceil(recvSize/4.0);
            zfp_send_buffer = (double*)malloc(zfp_send_bufsize);
            zfp_recv_buffer = (double*)malloc(zfp_recv_bufsize);
            free_zfp_buff = true;
//            printf("rank %d: vIndexSize = %d, recvSize = %d, send_bufsize = %d, recv_bufsize = %d \n",
//               rank, vIndexSize, recvSize, send_bufsize, recv_bufsize);
        }
    }

    return 0;
}


int saena_matrix::find_sortings(){
    //find the sorting on rows on both local and remote data, which will be used in matvec

    if(active){
        int rank;
        MPI_Comm_rank(comm, &rank);
        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, find_sortings \n", rank);
            MPI_Barrier(comm);
        }

        indicesP_local.resize(nnz_l_local);
#pragma omp parallel for
        for (nnz_t i = 0; i < nnz_l_local; i++)
            indicesP_local[i] = i;

        index_t *row_localP = &*row_local.begin();
        std::sort(&indicesP_local[0], &indicesP_local[nnz_l_local], sort_indices(row_localP));

//    if(rank==0)
//        for(index_t i=0; i<nnz_l_local; i++)
//            std::cout << row_local[indicesP_local[i]] << "\t" << col_local[indicesP_local[i]]
//                      << "\t" << values_local[indicesP_local[i]] << std::endl;

//        indicesP_remote.resize(nnz_l_remote);
//        for (nnz_t i = 0; i < nnz_l_remote; i++)
//            indicesP_remote[i] = i;
//        index_t *row_remoteP = &*row_remote.begin();
//        std::sort(&indicesP_remote[0], &indicesP_remote[nnz_l_remote], sort_indices(row_remoteP));
    }

    return 0;
}


int saena_matrix::openmp_setup() {

    // *************************** find start and end of each thread for matvec ****************************
    // also, find nnz per row for local and remote matvec

    if(active){
        int nprocs, rank;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, thread1 \n", rank);
            MPI_Barrier(comm);
        }

//        printf("matrix_setup: rank = %d, Mbig = %u, M = %u, nnz_g = %u, nnz_l = %u, nnz_l_local = %u, nnz_l_remote = %u \n", rank, Mbig, M, nnz_g, nnz_l, nnz_l_local, nnz_l_remote);

#pragma omp parallel
        {
            num_threads = omp_get_num_threads();
        }

        iter_local_array.resize(num_threads+1);
        iter_remote_array.resize(num_threads+1);

#pragma omp parallel
        {
            const int thread_id = omp_get_thread_num();
//            if(rank==0 && thread_id==0) std::cout << "number of procs = " << nprocs << ", number of threads = " << num_threads << std::endl;
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
//            if(rank==1) printf("thread id = %d, istart = %u, iend = %u \n", thread_id, istart, iend);

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

        if(verbose_matrix_setup) {
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

//        print_vector(iter_local_array, 0, "iter_local_array", comm);
//        print_vector(iter_remote_array, 0, "iter_remote_array", comm);

        // setup variables for another matvec implementation
        // -------------------------------------------------
        iter_local_array2.resize(num_threads+1);
        iter_local_array2[0] = 0;
        iter_local_array2[num_threads] = iter_local_array[num_threads];
        nnzPerRow_local2.resize(M);

#pragma omp parallel
        {
            const int thread_id = omp_get_thread_num();
            iter_local_array2[thread_id] = iter_local_array[thread_id]; // the default value

#pragma omp for
            for(index_t i = 0; i < M; i++)
                nnzPerRow_local2[i] = nnzPerRow_local[i];

            index_t iter = iter_local_array[thread_id];
//            if(rank==1) printf("rank %d thread %d \titer = %d \tindices = %lu \trow = %lu \n", rank, thread_id, iter, indicesP_local[iter], row_local[indicesP_local[iter]]);
            index_t starting_row;
            if(thread_id != 0 && iter < nnz_l_local){
                starting_row = row_local[indicesP_local[iter]];
//                if(rank==1) printf("rank %d thread %d: starting_row = %lu \tstarting nnz = %u \n", rank, thread_id, starting_row, iter);
                unsigned int left_over = 0;
                while(left_over < nnzPerRow_local[starting_row]){
//                    if(rank == 1) printf("%lu \t%lu \t%lu  \n",row_local[indicesP_local[iter]], col_local[indicesP_local[iter]] - split[rank], starting_row);
                    if(col_local[indicesP_local[iter]] - split[rank] >= starting_row){
                        iter_local_array2[thread_id] = iter;
                        nnzPerRow_local2[row_local[indicesP_local[iter]]] -= left_over;
                        break;
                    }
                    iter++;
                    left_over++;
                }
            }

        } // end of omp parallel

//        if(rank==0){
//            printf("\niter_local_array and iter_local_array2: \n");
//            for(int i = 0; i < num_threads+1; i++)
//                printf("%u \t%u \n", iter_local_array[i], iter_local_array2[i]);}
    } //if(active)

    return 0;
}


int saena_matrix::scale_matrix(){

    // scale matrix: A = D^{-1/2} * A * D^{-1/2}
    // values_local, values_remote and entry are being updated.
    // A[i] *= D^{-1/2}[row[i]] * D^{-1/2}[col[i]]

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");

//    MPI_Barrier(comm); if(rank==1) printf("start of saena_matrix::scale()\n"); MPI_Barrier(comm);

//    print_vector(inv_diag, -1, "inv_diag", comm);
    std::fill(inv_diag.begin(), inv_diag.end(), 1);

    MPI_Request* requests = nullptr;
    MPI_Status* statuses  = nullptr;

    if(nprocs > 1){
        // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
        // put the values of thoss indices in vSend to send to other procs.
#pragma omp parallel for
        for(index_t i=0;i<vIndexSize;i++)
            vSend[i] = inv_sq_diag[(vIndex[i])];

//        print_vector(vSend, -1, "vSend", comm);

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

//    index_t* col_p = &col_local[0] - split[rank];
#pragma omp parallel for
    for(nnz_t i = 0; i < nnz_l_local; i++) {
        values_local[i] *= inv_sq_diag[row_local[i]] * inv_sq_diag[col_local[i] - split[rank]];//D^{-1/2} * A * D^{-1/2}
    }

//    print_vector(values_local, -1, "values_local", comm);

    if(nprocs > 1){
        // Wait for the receive communication to finish.
        MPI_Waitall(numRecvProc, requests, statuses);

//        print_vector(vecValues, -1, "vecValues", comm);

        // remote loop
        // -----------
        // the col_index of the matrix entry does not matter. do the matvec on the first non-zero col// D^{-1/2} * A * D^{-1/2}umn (j=0).
        // the corresponding vector element is saved in vecValues[0]. and so on.

#pragma omp parallel
        {
            unsigned int i, l;
            int thread_id = omp_get_thread_num();
            nnz_t iter = iter_remote_array[thread_id];
#pragma omp for
            for (index_t j = 0; j < col_remote_size; ++j) {
                for (i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
                    values_remote[iter] *= inv_sq_diag[row_remote[iter]] * vecValues[j]; // D^{-1/2} * A * D^{-1/2}
                }
            }
        }
    }

    // update the entry vector
    entry.clear();
    entry.resize(nnz_l);

    // todo: change the local and remote parameters to cooEntry class to be able to use memcpy here.
//    memcpy(&*entry.begin(), );

    // copy local entries
#pragma omp parallel for
    for(nnz_t i = 0; i < nnz_l_local; i++)
        entry[i] = cooEntry(row_local[i]+split[rank], col_local[i], values_local[i]);

    if(nprocs > 1){
        // copy remote entries
#pragma omp parallel for
        for(nnz_t i = 0; i < nnz_l_remote; i++)
            entry[nnz_l_local + i] = cooEntry(row_remote[i]+split[rank], col_remote2[i], values_remote[i]);
    }

    std::sort(entry.begin(), entry.end());

    if(nprocs > 1){
        MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
        delete [] requests;
        delete [] statuses;
    }

//    MPI_Barrier(comm); if(rank==0) printf("end of saena_matrix::scale()\n"); MPI_Barrier(comm);

    return 0;
}


int saena_matrix::scale_back_matrix(){

    // scale back matrix: A = D^{1/2} * A * D^{1/2}
    // values_local, values_remote and entry are being updated.
    // A[i] /= D^{-1/2}[row[i]] * D^{-1/2}[col[i]]

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");

//    MPI_Barrier(comm); if(rank==1) printf("start of saena_matrix::scale()\n"); MPI_Barrier(comm);

//    print_vector(inv_diag, -1, "inv_diag", comm);
    std::fill(inv_diag.begin(), inv_diag.end(), 1);

    MPI_Request* requests = nullptr;
    MPI_Status* statuses  = nullptr;

    if(nprocs > 1){
        // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
        // put the values of thoss indices in vSend to send to other procs.
#pragma omp parallel for
        for(index_t i=0;i<vIndexSize;i++)
            vSend[i] = inv_sq_diag[(vIndex[i])];

//        print_vector(vSend, -1, "vSend", comm);

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

//    index_t* col_p = &col_local[0] - split[rank];
#pragma omp parallel for
    for(nnz_t i = 0; i < nnz_l_local; i++) {
        values_local[i] /= inv_sq_diag[row_local[i]] * inv_sq_diag[col_local[i] - split[rank]];//D^{-1/2} * A * D^{-1/2}
    }

//    print_vector(values_local, -1, "values_local", comm);

    if(nprocs > 1){
        // Wait for the receive communication to finish.
        MPI_Waitall(numRecvProc, requests, statuses);

//        print_vector(vecValues, -1, "vecValues", comm);

        // remote loop
        // -----------
        // the col_index of the matrix entry does not matter. do the matvec on the first non-zero col// D^{-1/2} * A * D^{-1/2}umn (j=0).
        // the corresponding vector element is saved in vecValues[0]. and so on.

#pragma omp parallel
        {
            unsigned int i, l;
            int thread_id = omp_get_thread_num();
            nnz_t iter = iter_remote_array[thread_id];
#pragma omp for
            for (index_t j = 0; j < col_remote_size; ++j) {
                for (i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
                    values_remote[iter] /= inv_sq_diag[row_remote[iter]] * vecValues[j]; // D^{-1/2} * A * D^{-1/2}
                }
            }
        }
    }

    // update the entry vector
    entry.clear();
    entry.resize(nnz_l);

    // todo: change the local and remote parameters to cooEntry class to be able to use memcpy here.
//    memcpy(&*entry.begin(), );

    // copy local entries
#pragma omp parallel for
    for(nnz_t i = 0; i < nnz_l_local; i++)
        entry[i] = cooEntry(row_local[i]+split[rank], col_local[i], values_local[i]);

    if(nprocs > 1){
        // copy remote entries
#pragma omp parallel for
        for(nnz_t i = 0; i < nnz_l_remote; i++)
            entry[nnz_l_local + i] = cooEntry(row_remote[i]+split[rank], col_remote2[i], values_remote[i]);
    }

    std::sort(entry.begin(), entry.end());

    if(nprocs > 1){
        MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
        delete [] requests;
        delete [] statuses;
    }

//    MPI_Barrier(comm); if(rank==0) printf("end of saena_matrix::scale()\n"); MPI_Barrier(comm);

    return 0;
}


int saena_matrix::inverse_diag() {
    int rank;
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
//    MPI_Barrier(comm);
//    if(rank==0) printf("inverse_diag!!!\n");
//    print_vector(split, 0, "split", comm);
//    print_entry(-1);
//    print_vector(inv_diag, -1, "inv diag", comm);
//    print_vector(inv_sq_diag, -1, "inv_sq_diag diag", comm);
//    MPI_Barrier(comm);
#endif

    double temp;
    inv_diag.assign(M, 0);
    inv_sq_diag.assign(M, 0);

    if(!entry.empty()) {
        for (nnz_t i = 0; i < nnz_l; i++) {

            if (entry[i].row == entry[i].col) {
//                if(rank==0) std::cout << i << "\t" << entry[i] << std::endl;
                if ( !almost_zero(entry[i].val) ) {
                    temp = 1.0 / entry[i].val;
                    inv_diag[entry[i].row - split[rank]] = temp;
                    inv_sq_diag[entry[i].row - split[rank]] = sqrt(temp);
                    if (fabs(temp) > highest_diag_val) {
                        highest_diag_val = fabs(temp);
                    }
                } else {
                    // there is no zero entry in the matrix (sparse), but just to be sure, this part is added.
                    if (rank == 0)
                        printf("Error: there is a zero diagonal element (at row index = %u)\n", entry[i].row);
                    MPI_Finalize();
                    return -1;
                }
            }
        }
    }

#ifdef __DEBUG1__
//    MPI_Barrier(comm);
//    print_vector(inv_diag, -1, "inv diag", comm);
//    print_vector(inv_sq_diag, -1, "inv_sq_diag diag", comm);
//    MPI_Barrier(comm);
#endif

    for(auto i:inv_diag) {
        if (i == 0){
            printf("rank %d: inverse_diag: At least one diagonal entry is 0.\n", rank);
        }
    }

    temp = highest_diag_val;
    MPI_Allreduce(&temp, &highest_diag_val, 1, MPI_DOUBLE, MPI_MAX, comm);
//    if(rank==0) printf("\ninverse_diag: highest_diag_val = %f \n", highest_diag_val);

    return 0;
}


int saena_matrix::generate_dense_matrix() {
    dense_matrix.convert_saena_matrix(this);
    dense_matrix_generated = true;
    return 0;
}