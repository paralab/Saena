#include "saena_matrix.h"
#include "parUtils.h"
#include "dollar.hpp"

#include <fstream>
#include <cstring>
#include <algorithm>
//#include <sys/stat.h>
#include <omp.h>
//#include <printf.


int saena_matrix::shrink_cpu(){

    // if number of rows on Ac < threshold*number of rows on A, then shrink.
    // redistribute Ac from processes 4k+1, 4k+2 and 4k+3 to process 4k.
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    bool verbose_shrink = false;

//    MPI_Barrier(comm);
//    if(rank==0) printf("\n****************************\n");
//    if(rank==0) printf("***********SHRINK***********\n");
//    if(rank==0) printf("****************************\n\n");
//    MPI_Barrier(comm);

//    MPI_Barrier(comm); printf("rank = %d \tnnz_l = %u \n", rank, nnz_l); MPI_Barrier(comm);

//    print_vector(entry, -1, "entry", comm);

    // assume cpu_shrink_thre2 is 4 (it is simpler to explain)
    // 1 - create a new comm, consisting only of processes 4k, 4k+1, 4k+2 and 4k+3 (with new ranks 0,1,2,3)
    int color = rank / cpu_shrink_thre2;
    MPI_Comm_split(comm, color, rank, &comm_horizontal);

    int rank_new, nprocs_new;
    MPI_Comm_size(comm_horizontal, &nprocs_new);
    MPI_Comm_rank(comm_horizontal, &rank_new);

//    MPI_Barrier(comm_horizontal);
//    printf("rank = %d, rank_new = %d on Ac->comm_horizontal \n", rank, rank_new);
//    MPI_Barrier(comm_horizontal);

/*
    // 2 - update the number of rows on process 4k, and resize "entry".
    unsigned int Ac_M_neighbors_total = 0;
    unsigned int Ac_nnz_neighbors_total = 0;
    MPI_Reduce(&M, &Ac_M_neighbors_total, 1, MPI_UNSIGNED, MPI_SUM, 0, comm_horizontal);
    MPI_Reduce(&nnz_l, &Ac_nnz_neighbors_total, 1, MPI_UNSIGNED, MPI_SUM, 0, comm_horizontal);

    if(rank_new == 0){
        M = Ac_M_neighbors_total;
        entry.resize(Ac_nnz_neighbors_total);
//        printf("rank = %d, Ac_M_neighbors = %d \n", rank, Ac_M_neighbors_total);
//        printf("rank = %d, Ac_nnz_neighbors = %d \n", rank, Ac_nnz_neighbors_total);
    }

    // last cpu that its right neighbors are going be shrinked to.
    auto last_root_cpu = (unsigned int)floor(nprocs/cpu_shrink_thre2) * cpu_shrink_thre2;
//    printf("last_root_cpu = %u\n", last_root_cpu);

    int neigbor_rank;
    unsigned int A_recv_nnz = 0; // set to 0 just to avoid "not initialized" warning
    unsigned long offset = nnz_l; // put the data on root from its neighbors at the end of entry[] which is of size nnz_l
    if(nprocs_new > 1) { // if there is no neighbor, skip.
        for (neigbor_rank = 1; neigbor_rank < cpu_shrink_thre2; neigbor_rank++) {

//            if( rank == last_root_cpu && (rank + neigbor_rank >= nprocs) )
//                stop_forloop = true;
            // last row of cpus should stop to avoid passing the last neighbor cpu.
//            if( rank >= last_root_cpu){
//                MPI_Bcast(&stop_forloop, 1, MPI_CXX_BOOL, 0, Ac->comm_horizontal);
//                printf("rank = %d, neigbor_rank = %d, stop_forloop = %d \n", rank, neigbor_rank, stop_forloop);
//                if (stop_forloop)
//                    break;}

            if( rank == last_root_cpu && (rank + neigbor_rank >= nprocs) )
                break;

            // 3 - send and receive size of Ac.
            if (rank_new == 0)
                MPI_Recv(&A_recv_nnz, 1, MPI_UNSIGNED, neigbor_rank, 0, comm_horizontal, MPI_STATUS_IGNORE);

            if (rank_new == neigbor_rank)
                MPI_Send(&nnz_l, 1, MPI_UNSIGNED, 0, 0, comm_horizontal);

            // 4 - send and receive Ac.
            if (rank_new == 0) {
//                printf("rank = %d, neigbor_rank = %d, recv size = %u, offset = %lu \n", rank, neigbor_rank, A_recv_nnz, offset);
                MPI_Recv(&*(entry.begin() + offset), A_recv_nnz, cooEntry::mpi_datatype(), neigbor_rank, 1,
                         comm_horizontal, MPI_STATUS_IGNORE);
                offset += A_recv_nnz; // set offset for the next iteration
            }

            if (rank_new == neigbor_rank) {
//                printf("rank = %d, neigbor_rank = %d, send size = %u, offset = %lu \n", rank, neigbor_rank, Ac->nnz_l, offset);
                MPI_Send(&*entry.begin(), nnz_l, cooEntry::mpi_datatype(), 0, 1, comm_horizontal);
            }

            // update local index for rows
//        if(rank_new == 0)
//            for(i=0; i < A_recv_nnz; i++)
//                Ac->entry[offset + i].row += Ac->split[rank + neigbor_rank] - Ac->split[rank];
        }
    }

    // even though the entries are sorted before shrinking, after shrinking they still need to be sorted locally,
    // because remote elements damage sorting after shrinking.
    std::sort(entry.begin(), entry.end());
*/
//    MPI_Barrier(comm); MPI_Barrier(Ac->comm_horizontal);
//    if(rank == 2){
//        std::cout << "\nafter shrinking: rank = " << rank << ", size = " << Ac->entry.size() <<std::endl;
//        for(i=0; i < Ac->entry.size(); i++)
//            std::cout << i << "\t" << Ac->entry[i]  <<std::endl;}
//    MPI_Barrier(comm); MPI_Barrier(Ac->comm_horizontal);

//    Ac->active_old_comm = true; // this is used for prolong and post-smooth
    active = false;
    if(rank_new == 0){
        active = true;
//        printf("active: rank = %d, rank_new = %d \n", rank, rank_new);
    }
/*
    // 5 - update 4k.nnz_l and split. nnz_g stays the same, so no need to update.
    if(active){
        nnz_l = entry.size();
        split_old = split; // save the old split for shrinking rhs and u
        split.clear();
        for(i = 0; i < nprocs+1; i++){
//            if(rank==0) printf("P->splitNew[i] = %lu\n", P_splitNew[i]);
            if( i % cpu_shrink_thre2 == 0){
                split.push_back( P_splitNew[i] );
            }
        }
        split.push_back( P_splitNew[nprocs] );
        // assert M == split[rank+1] - split[rank]

//        if(rank==0) {
//            printf("Ac split after shrinking: \n");
//            for (int i = 0; i < Ac->split.size(); i++)
//                printf("%lu \n", Ac->split[i]);}
    }
*/
    // 6 - create a new comm including only processes with 4k rank.
    MPI_Group bigger_group;
    MPI_Comm_group(comm, &bigger_group);
    total_active_procs = (unsigned int)ceil((double)nprocs / cpu_shrink_thre2); // note: this is ceiling, not floor.
    std::vector<int> ranks(total_active_procs);
    for(unsigned int i = 0; i < total_active_procs; i++)
        ranks[i] = cpu_shrink_thre2 * i;
//        ranks.push_back(Ac->cpu_shrink_thre2 * i);

//    MPI_Barrier(comm);
//    printf("total_active_procs = %u \n", total_active_procs);
//    for(i=0; i<ranks.size(); i++)
//        if(rank==0) std::cout << ranks[i] << std::endl;

//    MPI_Comm new_comm;
    MPI_Group group_new;
    MPI_Group_incl(bigger_group, total_active_procs, &*ranks.begin(), &group_new);
    MPI_Comm_create_group(comm, group_new, 0, &comm);
//    comm = new_comm;

//    if(Ac->active) {
//        int rankkk;
//        MPI_Comm_rank(Ac->comm, &rankkk);
//        MPI_Barrier(Ac->comm);
//        if (rankkk == 4) {
//                std::cout << "\ninside cpu_shrink, after shrinking" << std::endl;
//            std::cout << "\nrank = " << rank << ", size = " << Ac->entry.size() << std::endl;
//            for (i = 0; i < Ac->entry.size(); i++)
//                std::cout << i << "\t" << Ac->entry[i] << std::endl;}
//        MPI_Barrier(Ac->comm);
//    }

    std::vector<index_t> split_temp = split;
    split.clear();
    if(active){
        split.resize(total_active_procs+1);
        split.shrink_to_fit();
        split[0] = 0;
        split[total_active_procs] = Mbig;
        for(unsigned int i = 1; i < total_active_procs; i++){
//            if(rank==0) printf("%u \t%lu \n", i, split_old[ranks[i]]);
            split[i] = split_temp[ranks[i]];
        }
//        print_vector(split, -1, comm);
    }

//    print_vector(split, -1, "split", comm);

    // 7 - update 4k.nnz_g
//    if(Ac->active)
//        MPI_Allreduce(&Ac->nnz_l, &Ac->nnz_g, 1, MPI_UNSIGNED, MPI_SUM, Ac->comm);

//    if(Ac->active){
//        MPI_Comm_size(Ac->comm, &nprocs);
//        MPI_Comm_rank(Ac->comm, &rank);
//        printf("\n\nrank = %d, nprocs = %d, M = %u, nnz_l = %u, nnz_g = %u, Ac->split[rank+1] = %lu, Ac->split[rank] = %lu \n",
//               rank, nprocs, Ac->M, Ac->nnz_l, Ac->nnz_g, Ac->split[rank+1], Ac->split[rank]);
//    }

//    free(&bigger_group);
//    free(&group_new);
//    free(&comm_new2);
//    if(active)
//        MPI_Comm_free(&new_comm);

//    last_M_shrink = Mbig;
//    shrinked = true;
    return 0;
}


int saena_matrix::shrink_cpu_minor(){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    bool verbose_shrink = false;

//    MPI_Barrier(comm);
//    if(rank==0) printf("\n****************************\n");
//    if(rank==0) printf("********MINOR SHRINK********\n");
//    if(rank==0) printf("****************************\n\n");
//    MPI_Barrier(comm);

    shrinked_minor = true;

    active_minor = false;
    if(split[rank+1] - split[rank] != 0){
        active_minor = true;
//        printf("active: rank = %d \n", rank);
    }
    active = active_minor;

    MPI_Group bigger_group;
    MPI_Comm_group(comm, &bigger_group);

    total_active_procs = 0;
    std::vector<int> ranks(nprocs);
    for(unsigned int i = 0; i < nprocs; i++){
        if(split[i+1] - split[i] != 0){
            ranks[total_active_procs] = i;
            total_active_procs++;
        }
    }

    ranks.resize(total_active_procs);
    ranks.shrink_to_fit();

//    print_vector(split, 0, "split before shrinking", comm);

//    comm_old_minor = comm;
    MPI_Group group_new;
    MPI_Group_incl(bigger_group, total_active_procs, &*ranks.begin(), &group_new);
    MPI_Comm_create_group(comm, group_new, 1, &comm);

//    std::vector<index_t> split_temp = split;
    split_old_minor = split;
    split.clear();
    if(active_minor){
        split.resize(total_active_procs+1);
        split.shrink_to_fit();
        split[0] = 0;
        split[total_active_procs] = Mbig;
        for(unsigned int i = 1; i < total_active_procs; i++){
//            if(rank==0) printf("%u \t%lu \n", i, split_old[ranks[i]]);
            split[i] = split_old_minor[ranks[i]];
        }
//        print_vector(split, 0, "split after shrinking", comm);
    }

    return 0;
}


int saena_matrix::compute_matvec_dummy_time(){

    int rank;
    MPI_Comm_rank(comm, &rank);

    int matvec_iter_warmup = 5;
    int matvec_iter        = 10;
    std::vector<double> v_dummy(M, 1);
    std::vector<double> w_dummy(M);
//    std::vector<double> time_matvec(4, 0);

    MPI_Barrier(comm); // todo: should I keep this barrier?
//    double t1 = omp_get_wtime();

    // warm-up
    for (int i = 0; i < matvec_iter_warmup; i++) {
        matvec_dummy(v_dummy, w_dummy);
        v_dummy.swap(w_dummy);
    }

    std::fill(matvec_dummy_time.begin(), matvec_dummy_time.end(), 0);
    for (int i = 0; i < matvec_iter; i++) {
        matvec_dummy(v_dummy, w_dummy);
        v_dummy.swap(w_dummy);
    }

//    double t2 = omp_get_wtime();

    matvec_dummy_time[3] += matvec_dummy_time[0]; // total matvec time
    matvec_dummy_time[0] = matvec_dummy_time[3] - matvec_dummy_time[1] - matvec_dummy_time[2]; // communication including vSet

//    if (rank == 0) {
//        std::cout << std::endl << "decide_shrinking:" << std::endl;
//        std::cout << "comm:   " << matvec_dummy_time[0] / matvec_iter << std::endl; // comm including "set vSend"
//        std::cout << "local:  " << matvec_dummy_time[1] / matvec_iter << std::endl; // local loop
//        std::cout << "remote: " << matvec_dummy_time[2] / matvec_iter << std::endl; // remote loop
//        std::cout << "total:  " << matvec_dummy_time[3] / matvec_iter << std::endl; // total time
//    }

    return 0;
}


int saena_matrix::decide_shrinking(std::vector<double> &prev_time){

    // matvec_dummy_time[0]: communication (including "set vSend")
    // matvec_dummy_time[1]: local loop
    // matvec_dummy_time[2]: remote loop
    // matvec_dummy_time[3]: total time

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    int thre_loc, thre_comm;

//    if(rank==0)
//        printf("\nlocal  = %e \nremote = %e \ncomm   = %e \ntotal division = %f \nlocal division = %f \ncomm division = %f \n",
//               matvec_dummy_time[0], matvec_dummy_time[1], matvec_dummy_time[2],
//               matvec_dummy_time[3]/prev_time[3], prev_time[1]/matvec_dummy_time[1],
//               matvec_dummy_time[0]/prev_time[0]);

    if( (matvec_dummy_time[3] > shrink_total_thre * prev_time[3])
        && (shrink_local_thre * matvec_dummy_time[1] < prev_time[1])
        && (matvec_dummy_time[0] > shrink_communic_thre * prev_time[0]) ){

        do_shrink = true;

        thre_loc  = (int) floor(prev_time[1] / matvec_dummy_time[1]);
        thre_comm = (int) ceil(matvec_dummy_time[0] / (4 * prev_time[0]));
        if(thre_comm >= nprocs) thre_comm = nprocs; // todo: cpu_shrink_thre2 = nprocs was causing issue. solve that, then change nprocs-1 to nprocs.
        if(rank==0) printf("thre_loc = %d, thre_comm = %d \n", thre_loc, thre_comm);

        cpu_shrink_thre2 = std::max(thre_loc, thre_comm);
        if(cpu_shrink_thre2 == 1) cpu_shrink_thre2 = 2;
        if(cpu_shrink_thre2 > 5) cpu_shrink_thre2 = 5;
        if(rank==0) printf("SHRINK: cpu_shrink_thre2 = %d \n", cpu_shrink_thre2);

    } else if( (matvec_dummy_time[3] > 2 * prev_time[3])
               && (matvec_dummy_time[0] > 3 * prev_time[0]) ){

        do_shrink = true;

        cpu_shrink_thre2 = (int) ceil(matvec_dummy_time[0] / (4 * prev_time[0]));
        if(rank==0) printf("cpu_shrink_thre2 = %d \n", cpu_shrink_thre2);
        if(cpu_shrink_thre2 == 1) cpu_shrink_thre2 = 2;
        if(rank==0) printf("SHRINK: cpu_shrink_thre2 = %d \n", cpu_shrink_thre2);
    }

    return 0;
}


int saena_matrix::matrix_setup_dummy(){
    set_off_on_diagonal_dummy();
    return 0;
}


int saena_matrix::set_off_on_diagonal_dummy(){
    // set and exchange on-diagonal and off-diagonal elements
    // on-diagonal (local) elements are elements that correspond to vector elements which are local to this process.
    // off-diagonal (remote) elements correspond to vector elements which should be received from another processes.

    if(active){
        int nprocs, rank;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup_dummy: rank = %d, local remote1 \n", rank);
            MPI_Barrier(comm);
        }

        col_remote_size = 0;
        nnz_l_local = 0;
        nnz_l_remote = 0;
        recvCount.assign(nprocs, 0);
//        nnzPerRow_local.assign(M, 0);
//        nnzPerRow_remote.assign(M, 0);

        // take care of the first element here, since there is "col[i-1]" in the for loop below, so "i" cannot start from 0.
        long procNum;
        if(!entry.empty()){
            if (entry[0].col >= split[rank] && entry[0].col < split[rank + 1]) {
//                nnzPerRow_local[entry[0].row - split[rank]]++;
//                nnzPerCol_local[col[0]]++;
                nnz_l_local++;
//                values_local.push_back(entry[0].val);
//                row_local.push_back(entry[0].row - split[rank]);
                col_local.push_back(entry[0].col);
                //vElement_local.push_back(col[0]);
//                vElementRep_local.push_back(1);

            } else {
                nnz_l_remote++;
//                nnzPerRow_remote[entry[0].row - split[rank]]++;
//                values_remote.push_back(entry[0].val);
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
//                    nnzPerRow_local[entry[i].row - split[rank]]++;
//                    values_local.push_back(entry[i].val);
//                    row_local.push_back(entry[i].row - split[rank]);
                    col_local.push_back(entry[i].col);

//                    if (entry[i].col != entry[i - 1].col)
//                        vElementRep_local.push_back(1);
//                    else
//                        vElementRep_local.back()++;
                } else {
                    nnz_l_remote++;
//                    nnzPerRow_remote[entry[i].row - split[rank]]++;
//                    values_remote.push_back(entry[i].val);
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

        // dummy values
        values_local.assign(nnz_l_local, 1);
        values_remote.assign(nnz_l_remote, 1);

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup_dummy: rank = %d, local remote2 \n", rank);
            MPI_Barrier(comm);
        }

        // don't receive anything from yourself
        recvCount[rank] = 0;

//        print_vector(recvCount, 0, "recvCount", comm);

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

//        if (rank==0) std::cout << "rank=" << rank << ", numRecvProc=" << numRecvProc << ", numSendProc=" << numSendProc << std::endl;

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup_dummy: rank = %d, local remote3 \n", rank);
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

        vIndex.resize(vIndexSize);
        MPI_Alltoallv(&vElement_remote[0], &recvCount[0], &rdispls[0], MPI_UNSIGNED,
                      &vIndex[0],          &sendCount[0], &vdispls[0], MPI_UNSIGNED, comm);

//    print_vector(vIndex, -1, "vIndex", comm);

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup_dummy: rank = %d, local remote4 \n", rank);
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

//        vSendULong.resize(vIndexSize);
//        vecValuesULong.resize(recvSize);
    }

    return 0;
}


// int saena_matrix::find_sortings_dummy()
/*
int saena_matrix::find_sortings_dummy(){
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
//
//        index_t *row_remoteP = &*row_remote.begin();
//        std::sort(&indicesP_remote[0], &indicesP_remote[nnz_l_remote], sort_indices(row_remoteP));
    }

    return 0;
}
*/


int saena_matrix::matvec_dummy(std::vector<value_t>& v, std::vector<value_t>& w) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");

    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
    double t0_start = omp_get_wtime();

#pragma omp parallel for
    for(index_t i=0;i<vIndexSize;i++)
        vSend[i] = v[(vIndex[i])];

    double t0_end = omp_get_wtime();

//    if (rank==1) std::cout << "\nvIndexSize=" << vIndexSize << std::endl;
//    print_vector(vSend, 0, "vSend", comm);

    double t3_start = omp_get_wtime();

    MPI_Request* requests;
    MPI_Status* statuses;

    if(nprocs > 1){
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

    double t1_start = omp_get_wtime();

    value_t* v_p = &v[0] - split[rank];
    long iter = 0;
    for (index_t i = 0; i < M; ++i) {
        w[i] = 0;
        for (index_t j = 0; j < nnz_l_local/M; ++j, ++iter)
            w[i] += values_local[iter] * v_p[col_local[iter]];
    }

//    for (index_t i = 0; i < M; ++i) {
//        w[i] = 0;
//        for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
//            w[i] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
//        }
//    }

    double t1_end = omp_get_wtime();

    double t2_start = 0, t2_end = 0;

    if(nprocs > 1){
        // Wait for the receive communication to finish.
        MPI_Waitall(numRecvProc, requests, statuses);

//    print_vector(vecValues, 0, "vecValues", comm);

        // remote loop
        // -----------
        // the col_index of the matrix entry does not matter. do the matvec on the first non-zero column (j=0).
        // the corresponding vector element is saved in vecValues[0]. and so on.

        t2_start = omp_get_wtime();

        iter = 0;
        for (index_t j = 0; j < col_remote_size; ++j) {
            for (index_t i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
                w[row_remote[iter]] += values_remote[iter] * vecValues[j];

//                if(rank==0 && thread_id==0){
//                    printf("thread = %d\n", thread_id);
//                    printf("%u \t%u \tind_rem = %lu, row = %lu \tcol = %lu \tvecVal = %f \n",
//                           i, j, indicesP_remote[iter], row_remote[indicesP_remote[iter]],
//                           col_remote[indicesP_remote[iter]], vecValues[col_remote[indicesP_remote[iter]]]);}
            }
        }

        t2_end = omp_get_wtime();
        MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    }

    double t3_end = omp_get_wtime();

    delete [] requests;
    delete [] statuses;

    matvec_dummy_time.assign(4, 0);

    // set vsend
    double time0_local = t0_end-t0_start;
    double time0;
    MPI_Allreduce(&time0_local, &time0, 1, MPI_DOUBLE, MPI_SUM, comm);
    matvec_dummy_time[0] += time0/nprocs;

    // local loop
    double time1_local = t1_end-t1_start;
    double time1;
    MPI_Allreduce(&time1_local, &time1, 1, MPI_DOUBLE, MPI_SUM, comm);
    matvec_dummy_time[1] += time1/nprocs;

    // remote loop
    double time2_local = t2_end-t2_start;
    double time2;
    MPI_Allreduce(&time2_local, &time2, 1, MPI_DOUBLE, MPI_SUM, comm);
    matvec_dummy_time[2] += time2/nprocs;

    // communication = t3 - t1 - t2
    double time3_local = t3_end-t3_start;
    double time3;
    MPI_Allreduce(&time3_local, &time3, 1, MPI_DOUBLE, MPI_SUM, comm);
    matvec_dummy_time[3] += time3/nprocs;

    return 0;
}

