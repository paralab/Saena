#include "saena_matrix.h"
#include "parUtils.h"

int saena_matrix::decide_shrinking(std::vector<double> &prev_time){
    // set cpu_shrink_thre2 and do_shrink

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
//        if(rank==0) printf("thre_loc = %d, thre_comm = %d \n", thre_loc, thre_comm);

        cpu_shrink_thre2 = std::max(thre_loc, thre_comm);
        if(cpu_shrink_thre2 == 1) cpu_shrink_thre2 = 2;
        if(cpu_shrink_thre2 > 5) cpu_shrink_thre2 = 5;
//        if(rank==0) printf("SHRINK: cpu_shrink_thre2 = %d \n", cpu_shrink_thre2);

    } else if( (matvec_dummy_time[3] > 2 * prev_time[3])
               && (matvec_dummy_time[0] > 3 * prev_time[0]) ){

        do_shrink = true;

        cpu_shrink_thre2 = (int) ceil(matvec_dummy_time[0] / (4 * prev_time[0]));
//        if(rank==0) printf("cpu_shrink_thre2 = %d \n", cpu_shrink_thre2);
        if(cpu_shrink_thre2 == 1) cpu_shrink_thre2 = 2;
//        if(rank==0) printf("SHRINK: cpu_shrink_thre2 = %d \n", cpu_shrink_thre2);
    }

    return 0;
}

int saena_matrix::decide_shrinking_c(){
    // set cpu_shrink_thre2 and do_shrink

    int rank = -1, nprocs = -1;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(nprocs == 1){
        return 0;
    }

    do_shrink = true;

#if 0
    // nprocs_new = nprocs / cpu_shrink_thre2   =>   cpu_shrink_thre2 = nprocs / nprocs_new
    if(nprocs > 1000){
        cpu_shrink_thre2 = nprocs / 40; // nprocs_new = 40
    }else if(nprocs > 500){
        cpu_shrink_thre2 = nprocs / 20;
    }else if(nprocs > 200){
        cpu_shrink_thre2 = nprocs / 10;
    }else{
        cpu_shrink_thre2 = nprocs;
    }
#endif

    cpu_shrink_thre2 = nprocs;

    return 0;
}


int saena_matrix::shrink_set_params(std::vector<int> &send_size_array){
    int nprocs = -1, rank = -1;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    if(verbose_shrink){
        MPI_Barrier(comm);
        if(!rank) print_sep();
        if(!rank) printf("\nshrink_set_params\n");
        print_vector(split, 0, "split before shrinking", comm);
        MPI_Barrier(comm);
    }
//    print_vector(send_size_array, -1, "send_size_array", comm);
#endif

    shrinked = true;
    last_M_shrink = Mbig;
//    last_nnz_shrink = nnz_g;
    last_density_shrink = density;
    double remainder = 0.0;
    int root_cpu = nprocs;
    for(int proc = nprocs - 1; proc > 0; --proc){
        remainder = proc % cpu_shrink_thre2;
//        if(rank==0) printf("proc = %ld, remainder = %f\n", proc, remainder);
        if(remainder == 0)
            root_cpu = proc;
        else{
            split[proc] = split[root_cpu];
        }
    }

    M = split[rank+1] - split[rank];

#ifdef __DEBUG1__
    if(verbose_shrink){
        MPI_Barrier(comm);
        print_vector(split, 0, "split after shrinking", comm);
//        if(!rank) printf("M = %d\n", M);
        MPI_Barrier(comm);
    }
#endif

    root_cpu = 0;
    for(int proc = 0; proc < nprocs; ++proc){
        remainder = proc % cpu_shrink_thre2;
//        if(rank==0) printf("proc = %ld, remainder = %f\n", proc, remainder);
        if(remainder == 0)
            root_cpu = proc;
        else{
            send_size_array[root_cpu] += send_size_array[proc];
            send_size_array[proc] = 0;
        }
    }

#ifdef __DEBUG1__
    if(verbose_shrink){
        MPI_Barrier(comm);
        if(!rank) printf("\nshrink_set_params: done\n");
        if(!rank) print_sep();
        MPI_Barrier(comm);
    }
//    print_vector(send_size_array, -1, "send_size_array", comm);
#endif

    return 0;
}

int saena_matrix::shrink_cpu(){

    // if number of rows on Ac < threshold * number of rows on A, then shrink.
    // redistribute Ac from processes 4k+1, 4k+2 and 4k+3 to process 4k.
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    if(verbose_shrink){
        MPI_Barrier(comm);
        if(!rank) print_sep();
        if(!rank) printf("\nshrink_cpu\n");
        MPI_Barrier(comm);
    }
//    MPI_Barrier(comm); printf("rank = %d \tnnz_l = %u \n", rank, nnz_l); MPI_Barrier(comm);
//    print_vector(entry, -1, "entry", comm);
#endif

    // For simplicity, assume cpu_shrink_thre2 is 4 (it is simpler to explain this way)
    // create a new comm, consisting only of processes 4k, 4k+1, 4k+2 and 4k+3 (with new ranks 0,1,2,3)
    int color = rank / cpu_shrink_thre2;
    MPI_Comm_split(comm, color, rank, &comm_horizontal);

    int rank_new, nprocs_new;
    MPI_Comm_size(comm_horizontal, &nprocs_new);
    MPI_Comm_rank(comm_horizontal, &rank_new);

#ifdef __DEBUG1__
    if(verbose_shrink){
        MPI_Barrier(comm_horizontal);
        printf("rank = %d, rank_new = %d on Ac->comm_horizontal \n", rank, rank_new);
        MPI_Barrier(comm_horizontal);
    }
#endif

//    MPI_Barrier(comm_horizontal);
//    printf("rank = %d, rank_new = %d on Ac->comm_horizontal \n", rank, rank_new);
//    MPI_Barrier(comm_horizontal);

    active = false;
    if(rank_new == 0){
        active = true;
//        printf("active: rank = %d, rank_new = %d \n", rank, rank_new);
    }

    // create a new comm including only processes with 4k rank.
    MPI_Group bigger_group;
    MPI_Comm_group(comm, &bigger_group);
    total_active_procs = (index_t)ceil((double)nprocs / cpu_shrink_thre2); // note: ceiling should be used, not floor.
    std::vector<int> ranks(total_active_procs);
    for(index_t i = 0; i < total_active_procs; i++)
        ranks[i] = cpu_shrink_thre2 * i;

#ifdef __DEBUG1__
    if(verbose_shrink){
        MPI_Barrier(comm);
        print_vector(ranks, 0, "ranks", comm);
        MPI_Barrier(comm);
    }
#endif

    MPI_Group group_new;
    MPI_Group_incl(bigger_group, total_active_procs, &*ranks.begin(), &group_new);
    MPI_Comm_create_group(comm, group_new, 0, &comm);

    MPI_Group_free(&bigger_group);
    MPI_Group_free(&group_new);

    std::vector<index_t> split_temp = split;
    split.clear();
    if(active){
        split.resize(total_active_procs + 1);
        split.shrink_to_fit();
        split[0] = 0;
        split[total_active_procs] = Mbig;
        for(index_t i = 1; i < total_active_procs; ++i){
//            if(rank==0) printf("%u \t%lu \n", i, split_old[ranks[i]]);
            split[i] = split_temp[ranks[i]];
        }
    }

#ifdef __DEBUG1__
    if(active && verbose_shrink) {
        MPI_Barrier(comm);
        print_vector(split, 0, "split after shrinking", comm);
        if(!rank) printf("\nshrink_cpu: done\n");
        if(!rank) print_sep();
        MPI_Barrier(comm);
    }
#endif

    return 0;
}

int saena_matrix::shrink_cpu_minor(){

    int rank = -1, nprocs = -1;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    if(verbose_shrink){
        MPI_Barrier(comm);
        if(!rank) print_sep();
        if(!rank) printf("\nshrink_cpu_minor\n");
        MPI_Barrier(comm);
    }
#endif

//    shrinked_minor = true;

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
    for(index_t i = 0; i < nprocs; i++){
        if(split[i+1] - split[i] != 0){
            ranks[total_active_procs++] = i;
        }
    }

    ranks.resize(total_active_procs);
//    ranks.shrink_to_fit();

    MPI_Group group_new;
    MPI_Group_incl(bigger_group, total_active_procs, &*ranks.begin(), &group_new);
    MPI_Comm_create_group(comm, group_new, 1, &comm);

    MPI_Group_free(&bigger_group);
    MPI_Group_free(&group_new);

#ifdef __DEBUG1__
    if(active_minor && verbose_shrink){
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);
        MPI_Barrier(comm); print_vector(split, 0, "split before shrinking", comm); MPI_Barrier(comm);
    }
#endif

    comm_old = comm;

    std::vector<index_t> split_old_minor = split;
    split.clear();
    if(active_minor){
        split.resize(total_active_procs + 1);
        split.shrink_to_fit();
        split[0] = 0;
        split[total_active_procs] = Mbig;
        for(index_t i = 1; i < total_active_procs; ++i){
//            if(rank==0) printf("%u \t%lu \n", i, split_old[ranks[i]]);
            split[i] = split_old_minor[ranks[i]];
        }
    }

#ifdef __DEBUG1__
    if(active_minor && verbose_shrink) {
//        MPI_Barrier(comm);
        print_vector(split, 0, "split after shrinking", comm);
        if(!rank) printf("\nshrink_cpu_minor: done\n");
        if(!rank) print_sep();
//        MPI_Barrier(comm);
    }
#endif

    return 0;
}

int saena_matrix::shrink_cpu_c(){
    // to make this enable:
    // set enable_shrink_c in saena_matrix.h to true
    // set dynamic_levels in saena_object.h to false
    // find a good number of levels for the multigrid hierarchy, then set that in solver.set_multigrid_max_level
    // decide_shrinking_c() should be called before this funcion to set cpu_shrink_thre2 and do_shrink.

    // For simplicity, assume cpu_shrink_thre2 is 4 (it is simpler to explain this way)
    // if number of rows on Ac < threshold * number of rows on A, then shrink.
    // redistribute Ac from processes 4k+1, 4k+2 and 4k+3 to process 4k.
    int rank = -1, nprocs = -1;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    if(verbose_shrink){
        MPI_Barrier(comm);
        if(!rank) print_sep();
        if(!rank) printf("\nshrink_cpu_coarsest\n");
        MPI_Barrier(comm);
    }
//    MPI_Barrier(comm); printf("rank = %d \tnnz_l = %u \n", rank, nnz_l); MPI_Barrier(comm);
//    print_vector(entry, -1, "entry", comm);
#endif

    comm_old = comm;

    // create a new comm, consisting only of processes 4k, 4k+1, 4k+2 and 4k+3 (with new ranks 0,1,2,3)
    int color = rank / cpu_shrink_thre2;
    MPI_Comm_split(comm, color, rank, &comm_horizontal);

    int rank_new = -1, nprocs_new = -1;
    MPI_Comm_size(comm_horizontal, &nprocs_new);
    MPI_Comm_rank(comm_horizontal, &rank_new);

#ifdef __DEBUG1__
    if(verbose_shrink){
        MPI_Barrier(comm_horizontal);
        if(!rank) printf("rank = %d, rank_new = %d on Ac->comm_horizontal \n", rank, rank_new);
        MPI_Barrier(comm_horizontal);
    }
//    MPI_Barrier(comm); printf("rank = %d \tnnz_l = %u \n", rank, nnz_l); MPI_Barrier(comm);
//    print_vector(entry, -1, "entry", comm);
#endif

    active = false;
    if(rank_new == 0){
        active = true;
    }

#ifdef __DEBUG1__
    if(verbose_shrink && !rank_new){
        printf("active: rank = %d, rank_new = %d \n", rank, rank_new);
    }
#endif

    // create a new comm including only processes with 4k rank.
    MPI_Group bigger_group;
    MPI_Comm_group(comm, &bigger_group);
    total_active_procs = (index_t) ceil( (double)nprocs / cpu_shrink_thre2 ); // note: ceiling should be used, not floor.
    std::vector<int> ranks(total_active_procs);
    for(index_t i = 0; i < total_active_procs; ++i)
        ranks[i] = cpu_shrink_thre2 * i;

#ifdef __DEBUG1__
    if(verbose_shrink){
        if(!rank_new) printf("total_active_procs = %u \n", total_active_procs);
        print_vector(ranks, 0, "ranks", comm);
        print_vector(split, 0, "split before shrinking", comm);
    }
#endif

    MPI_Group group_new;
    MPI_Group_incl(bigger_group, total_active_procs, &*ranks.begin(), &group_new);
    MPI_Comm_create_group(comm, group_new, 0, &comm);

    MPI_Group_free(&bigger_group);
    MPI_Group_free(&group_new);

    std::vector<index_t> split_temp = split;
    split.clear();
    if(active){
        split.resize(total_active_procs + 1);
        split.shrink_to_fit();
        split[0] = 0;
        split[total_active_procs] = Mbig;
        for(index_t i = 1; i < total_active_procs; ++i){
//            if(rank==0) printf("%u \t%lu \n", i, split_old[ranks[i]]);
            split[i] = split_temp[ranks[i]];
        }
    }

#ifdef __DEBUG1__
    if(active && verbose_shrink){
        print_vector(split, 0, "split after shrinking", comm);
        if(!rank) printf("\nshrink_cpu_c: done\n");
        if(!rank) print_sep();
    }
#endif

    return 0;
}


int saena_matrix::compute_matvec_dummy_time(){

    int rank = -1, nprocs = -1;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    int matvec_iter_warmup = 5;
    int matvec_iter_dummy  = 5;
    std::vector<double> v_dummy(M, 1);
    std::vector<double> w_dummy(M);
//    std::vector<double> time_matvec(4, 0);

#ifdef __DEBUG1__
    if(verbose_matvec_dummy) {
        printf("rank %d: decide shrinking: step1\n", rank);
        MPI_Barrier(comm);
    }
#endif

    matvec_dummy_time.resize(4);

    // warm-up
    for (int i = 0; i < matvec_iter_warmup; ++i) {
        matvec_dummy(v_dummy, w_dummy);
        v_dummy.swap(w_dummy);
    }

#ifdef __DEBUG1__
    if(verbose_matvec_dummy) {
        MPI_Barrier(comm);
        printf("rank %d: decide shrinking: step2\n", rank);
        MPI_Barrier(comm);
    }
#endif

    std::fill(matvec_dummy_time.begin(), matvec_dummy_time.end(), 0);

    MPI_Barrier(comm); // todo: should I keep this barrier?
    for (int i = 0; i < matvec_iter_dummy; i++) {
        matvec_dummy(v_dummy, w_dummy);
        v_dummy.swap(w_dummy);
    }

#ifdef __DEBUG1__
    if(verbose_matvec_dummy) {
        MPI_Barrier(comm);
        printf("rank %d: decide shrinking: step3\n", rank);
        MPI_Barrier(comm);
    }
#endif

//    double t2 = omp_get_wtime();

    matvec_dummy_time[3] += matvec_dummy_time[0]; // total matvec time
    matvec_dummy_time[0] = matvec_dummy_time[3] - matvec_dummy_time[1] - matvec_dummy_time[2]; // communication including vSet

    std::vector<double> tempt(4);
    tempt[0] = matvec_dummy_time[0] / nprocs;
    tempt[1] = matvec_dummy_time[1] / nprocs;
    tempt[2] = matvec_dummy_time[2] / nprocs;
    tempt[3] = matvec_dummy_time[3] / nprocs;

    MPI_Allreduce(&tempt[0], &matvec_dummy_time[0], matvec_dummy_time.size(), MPI_DOUBLE, MPI_SUM, comm);

//    print_vector(matvec_dummy_time, 0, "matvec_dummy_time final", comm);

//    if (rank == 0) {
//        std::cout << std::endl << "decide_shrinking:" << std::endl;
//        std::cout << "comm:   " << matvec_dummy_time[0] / matvec_iter_dummy << std::endl; // comm including "set vSend"
//        std::cout << "local:  " << matvec_dummy_time[1] / matvec_iter_dummy << std::endl; // local loop
//        std::cout << "remote: " << matvec_dummy_time[2] / matvec_iter_dummy << std::endl; // remote loop
//        std::cout << "total:  " << matvec_dummy_time[3] / matvec_iter_dummy << std::endl; // total time
//    }

//    if (!rank) {
//        printf("next level matvec time: %f\n", matvec_dummy_time[3] / matvec_iter_dummy);
//    }

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
//                values_local.emplace_back(entry[0].val);
//                row_local.emplace_back(entry[0].row - split[rank]);
                col_local.emplace_back(entry[0].col);
                //vElement_local.emplace_back(col[0]);
//                vElementRep_local.emplace_back(1);

            } else {
                nnz_l_remote++;
//                nnzPerRow_remote[entry[0].row - split[rank]]++;
//                values_remote.emplace_back(entry[0].val);
                row_remote.emplace_back(entry[0].row - split[rank]);
                col_remote_size++;
                col_remote.emplace_back(col_remote_size - 1);
                col_remote2.emplace_back(entry[0].col);
                nnzPerCol_remote.emplace_back(1);
                vElement_remote.emplace_back(entry[0].col);
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
//                    values_local.emplace_back(entry[i].val);
//                    row_local.emplace_back(entry[i].row - split[rank]);
                    col_local.emplace_back(entry[i].col);
                } else {
                    nnz_l_remote++;
//                    nnzPerRow_remote[entry[i].row - split[rank]]++;
//                    values_remote.emplace_back(entry[i].val);
                    row_remote.emplace_back(entry[i].row - split[rank]);
                    // col_remote2 is the original col value and will be used in making strength matrix. col_remote will be used for matevec.
                    col_remote2.emplace_back(entry[i].col);

                    if (entry[i].col != entry[i - 1].col) {
                        col_remote_size++;
                        vElement_remote.emplace_back(entry[i].col);
                        procNum = lower_bound2(&split[0], &split[nprocs], entry[i].col);
//                        if(rank==1) printf("col = %u \tprocNum = %ld \n", entry[i].col, procNum);
                        recvCount[procNum]++;
                        nnzPerCol_remote.emplace_back(1);
                    } else {
                        nnzPerCol_remote.back()++;
                    }
                    // the original col values are not being used. the ordering starts from 0, and goes up by 1.
                    col_remote.emplace_back(col_remote_size - 1);
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
        for (index_t i = 1; i < nprocs; i++){
            recvCountScan[i] = recvCountScan[i-1] + recvCount[i-1];
            sendCountScan[i] = sendCountScan[i-1] + sendCount[i-1];
        }

        for (int i = 0; i < nprocs; i++) {
            if (recvCount[i] != 0) {
                recvProcRank.emplace_back(i);
                recvProcCount.emplace_back(recvCount[i]);
            }
            if (sendCount[i] != 0) {
                sendProcRank.emplace_back(i);
                sendProcCount.emplace_back(sendCount[i]);
            }
        }

        numRecvProc = recvProcRank.size();
        numSendProc = sendProcRank.size();

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
        MPI_Alltoallv(&vElement_remote[0], &recvCount[0], &rdispls[0], par::Mpi_datatype<index_t>::value(),
                      &vIndex[0],          &sendCount[0], &vdispls[0], par::Mpi_datatype<index_t>::value(), comm);

//        print_vector(vIndex, -1, "vIndex", comm);

        vElement_remote.clear();
        vElement_remote.shrink_to_fit();

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

int saena_matrix::matvec_dummy(std::vector<value_t>& v, std::vector<value_t>& w) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");

#ifdef __DEBUG1__
    if(verbose_matvec_dummy){
        MPI_Barrier(comm);
        printf("rank %d: matvec_dummy: step1\n", rank);
        MPI_Barrier(comm);
    }
#endif

    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
    double t0_start = omp_get_wtime();

#pragma omp parallel for
    for(index_t i=0;i<vIndexSize;i++)
        vSend[i] = v[(vIndex[i])];

    double t0_end = omp_get_wtime();

//    if (rank==1) std::cout << "\nvIndexSize=" << vIndexSize << std::endl;
//    print_vector(vSend, 0, "vSend", comm);

#ifdef __DEBUG1__
    if(verbose_matvec_dummy) {
        MPI_Barrier(comm);
        printf("rank %d: matvec_dummy: step2\n", rank);
        MPI_Barrier(comm);
    }
#endif

    double t3_start = omp_get_wtime();

    MPI_Request* requests = nullptr;
    MPI_Status*  statuses = nullptr;

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

#ifdef __DEBUG1__
    if(verbose_matvec_dummy) {
        MPI_Barrier(comm);
        printf("rank %d: matvec_dummy: step3\n", rank);
        MPI_Barrier(comm);
    }
#endif

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

#ifdef __DEBUG1__
    if(verbose_matvec_dummy) {
        MPI_Barrier(comm);
        printf("rank %d: matvec_dummy: step4\n", rank);
        MPI_Barrier(comm);
    }
#endif

    double t1_end = omp_get_wtime();

    double t2_start = 0, t2_end = 0;

    if(nprocs > 1){
        // Wait for the receive communication to finish.
        MPI_Waitall(numRecvProc, requests, statuses);

//        printf("rank %d: col_remote_size = %u, \tnumSendProc = %u, \tnumRecvProc = %u\n", rank, col_remote_size, numSendProc, numRecvProc);
//        print_vector(vecValues, 1, "vecValues", comm);

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
        delete [] requests;
        delete [] statuses;
    }

    double t3_end = omp_get_wtime();

    matvec_dummy_time[0] += t0_end - t0_start;  // set vsend
    matvec_dummy_time[1] += t1_end - t1_start;  // local loop
    matvec_dummy_time[2] += t2_end - t2_start;  // remote loop
    matvec_dummy_time[3] += t3_end - t3_start;  // communication + local loop + remote loop

#if 0
    // set vsend
    double time0_local = t0_end - t0_start;
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
#endif

#ifdef __DEBUG1__
//    print_vector(matvec_dummy_time, 0, "matvec_dummy_time", comm);
    if(verbose_matvec_dummy) {
        MPI_Barrier(comm);
        printf("rank %d: matvec_dummy: done\n", rank);
        MPI_Barrier(comm);
    }
#endif

    return 0;
}