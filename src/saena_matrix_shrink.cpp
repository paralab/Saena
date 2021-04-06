#include "saena_matrix.h"

int saena_matrix::decide_shrinking(std::vector<double> &prev_time){
    // set cpu_shrink_thre2 and do_shrink
#if 0
    int rank = 0, nprocs = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // matvec_dummy_time[0]: communication (including "set vSend")
    // matvec_dummy_time[1]: local loop
    // matvec_dummy_time[2]: remote loop
    // matvec_dummy_time[3]: total time

    int thre_loc = 0, thre_comm = 0;

//    if(rank==0)
//        printf("\nlocal  = %e \nremote = %e \ncomm   = %e \ntotal division = %f \nlocal division = %f \ncomm division = %f \n",
//               matvec_dummy_time[1], matvec_dummy_time[2], matvec_dummy_time[0],
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
#endif

    // matvec_dummy_time[0]: send_buf + local + remote
    // matvec_dummy_time[3]: comm
    if(matvec_dummy_time[3] > 2 * matvec_dummy_time[0]){
        do_shrink = true;
        cpu_shrink_thre2 = floor(matvec_dummy_time[3] / matvec_dummy_time[0] / 4);
        if(cpu_shrink_thre2 <= 1) cpu_shrink_thre2 = 2;

//        int rank = 0, nprocs = 0;
//        MPI_Comm_size(comm, &nprocs);
//        MPI_Comm_rank(comm, &rank);
//        if(!rank) printf("cpu_shrink_thre2 = %d\n", cpu_shrink_thre2);
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
    MPI_Comm comm_horizontal = MPI_COMM_NULL;
    MPI_Comm_split(comm, color, rank, &comm_horizontal);

    int rank_new = 0, nprocs_new = 0;
    MPI_Comm_size(comm_horizontal, &nprocs_new);
    MPI_Comm_rank(comm_horizontal, &rank_new);

    MPI_Comm_free(&comm_horizontal);

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

    comm_old = comm;

    MPI_Group group_new;
    MPI_Group_incl(bigger_group, total_active_procs, &*ranks.begin(), &group_new);
    MPI_Comm_create_group(comm_old, group_new, 0, &comm);

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
    MPI_Comm comm_horizontal = MPI_COMM_NULL;
    MPI_Comm_split(comm_old, color, rank, &comm_horizontal);

    int rank_new = -1, nprocs_new = -1;
    MPI_Comm_size(comm_horizontal, &nprocs_new);
    MPI_Comm_rank(comm_horizontal, &rank_new);

    MPI_Comm_free(&comm_horizontal);

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


void saena_matrix::compute_matvec_dummy_time(){

    int rank = -1, nprocs = -1;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::vector<double> v_dummy(M, 1);
    std::vector<double> w_dummy(M);
//    std::vector<double> time_matvec(4, 0);

#ifdef __DEBUG1__
    if(verbose_matvec_dummy) {
        printf("rank %d: decide shrinking: step1\n", rank);
        MPI_Barrier(comm);
    }
#endif

    matvec_dummy_time.assign(4, 0);

    // warm-up
    for (int i = 0; i < matvec_iter_dummy; ++i) {
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

    matvec_dummy_time[0] += matvec_dummy_time[1] + matvec_dummy_time[2]; // send_buf + loc + remote
    matvec_dummy_time[3] -= matvec_dummy_time[1] + matvec_dummy_time[2]; // communication

    // take average between processors
    std::vector<double> tempt(4);
    tempt[0] = matvec_dummy_time[0] / nprocs / matvec_iter_dummy;
    tempt[1] = matvec_dummy_time[1] / nprocs / matvec_iter_dummy;
    tempt[2] = matvec_dummy_time[2] / nprocs / matvec_iter_dummy;
    tempt[3] = matvec_dummy_time[3] / nprocs / matvec_iter_dummy;
    MPI_Allreduce(&tempt[0], &matvec_dummy_time[0], matvec_dummy_time.size(), MPI_DOUBLE, MPI_SUM, comm);

//    print_vector(matvec_dummy_time, 0, "matvec_dummy_time", comm);

//    if (rank == 0) {
//        std::cout << std::endl << "decide_shrinking:" << std::endl;
//        std::cout << "comm:   " << matvec_dummy_time[0] / matvec_iter_dummy << std::endl; // send_buf + loc + remote
//        std::cout << "local:  " << matvec_dummy_time[1] / matvec_iter_dummy << std::endl; // local
//        std::cout << "remote: " << matvec_dummy_time[2] / matvec_iter_dummy << std::endl; // remote
//        std::cout << "total:  " << matvec_dummy_time[3] / matvec_iter_dummy << std::endl; // comm
//    }

//    if (!rank) {
//        printf("next level matvec time: %f\n", matvec_dummy_time[3] / matvec_iter_dummy);
//    }
}

void saena_matrix::matrix_setup_dummy(){
    set_off_on_diagonal_dummy();
//    find_sortings_dummy();
}

void saena_matrix::set_off_on_diagonal_dummy(){
    // set and exchange on-diagonal and off-diagonal elements
    // on-diagonal (local) elements are elements that correspond to vector elements which are local to this process.
    // off-diagonal (remote) elements correspond to vector elements which should be received from another processes.

    if(active){
        int nprocs = 0, rank = 0;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
        if(verbose_matrix_setup_sh) {
            MPI_Barrier(comm);
            printf("matrix_setup_shrink: rank = %d, local remote1 \n", rank);
            MPI_Barrier(comm);
//            print_vector(split, 0, "split", comm);
//            print_vector(entry, -1, "entry", comm);
        }
#endif

        nnz_l_local     = 0;
        nnz_l_remote    = 0;
        col_remote_size = 0;
        recvCount.assign(nprocs, 0);
        nnzPerRow_local.assign(M, 0);

#ifdef _USE_PETSC_
        if(nprocs > 1){
            nnzPerRow_remote.assign(M, 0);
        }
#endif

        index_t procNum = 0, procNumTmp = 0;
        nnz_t tmp = 0;
        nnzPerProcScan.assign(nprocs + 1, 0);
        auto *nnzProc_p = &nnzPerProcScan[1];

        assert(nnz_l == entry.size());

        // store local entries in this vector for sorting in row-major order.
        // then split it to row_loc, col_loc, val_loc.
        vector<cooEntry_row> ent_loc_row;

        nnz_t i = 0;
        while(i < nnz_l) {
            procNum = lower_bound2(&split[0], &split[nprocs], entry[i].col);
//            if(rank==0) printf("col = %u \tprocNum = %d \n", entry[i].col, procNum);

            if(procNum == rank){ // local
                while(i < nnz_l && entry[i].col < split[procNum + 1]) {
//                    if(rank == 1) printf("entry[i].row = %d, split[rank] = %d, dif = %d\n", entry[i].row, split[rank], entry[i].row - split[rank]);
//                    if(!rank) cout << entry[i] << endl;
                    ++nnzPerRow_local[entry[i].row - split[rank]];
                    ent_loc_row.emplace_back(entry[i].row - split[rank], entry[i].col, entry[i].val);
                    ++i;
                }

            }else{ // remote
                tmp = i;
                while(i < nnz_l && entry[i].col < split[procNum + 1]) {

                    vElement_remote.emplace_back(entry[i].col);
                    ++recvCount[procNum];
                    nnzPerCol_remote.emplace_back(0);

                    do{
//                        col_remote.emplace_back(vElement_remote.size() - 1);
//                        col_remote2.emplace_back(entry[i].col);
                        row_remote.emplace_back(entry[i].row - split[rank]);
                        values_remote.emplace_back(entry[i].val);
                        ++nnzPerCol_remote.back();
#ifdef _USE_PETSC_
                        ++nnzPerRow_remote[entry[i].row - split[rank]];
#endif
                    }while(++i < nnz_l && entry[i].col == entry[i - 1].col);
                }
                nnzProc_p[procNum] = i - tmp;
            }
        } // for i

        nnz_l_local     = ent_loc_row.size();
        nnz_l_remote    = row_remote.size();
        col_remote_size = vElement_remote.size();

#ifdef __DEBUG1__
//        print_vector(nnzPerRow_local, 0, "nnzPerRow_local", comm);
        if(verbose_matrix_setup_sh) {
            MPI_Barrier(comm);
            printf("matrix_setup_shrink: rank = %d, local remote2 \n", rank);
            MPI_Barrier(comm);
        }
#endif

        // don't receive anything from yourself
        recvCount[rank] = 0;

        // sort local entries in row-major order and remote entries in column-major order
        sort(ent_loc_row.begin(), ent_loc_row.end());

        for(const auto &a : ent_loc_row){
            row_local.emplace_back(a.row);
            col_local.emplace_back(a.col);
            val_local.emplace_back(a.val);
        }

        ent_loc_row.clear();
        ent_loc_row.shrink_to_fit();

        if(nprocs != 1){

            for (i = 1; i < nprocs + 1; ++i){
                nnzPerProcScan[i] += nnzPerProcScan[i - 1];
            }

            sendCount.resize(nprocs);
            MPI_Alltoall(&recvCount[0], 1, MPI_INT, &sendCount[0], 1, MPI_INT, comm);

#ifdef __DEBUG1__
//            print_vector(nnzPerProcScan, 0, "nnzPerProcScan", comm);
//            print_vector(sendCount, 0, "sendCount", comm);
#endif

            for (i = 0; i < nprocs; ++i) {
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

            requests.resize(numSendProc+numRecvProc);
            statuses.resize(numSendProc+numRecvProc);

#ifdef __DEBUG1__
//            if (rank==0) std::cout << "rank=" << rank << ", numRecvProc=" << numRecvProc
//                                   << ", numSendProc=" << numSendProc << std::endl;
            if(verbose_matrix_setup_sh) {
                MPI_Barrier(comm);
                printf("matrix_setup_shrink: rank = %d, local remote3 \n", rank);
                MPI_Barrier(comm);
            }
#endif

            vdispls.resize(nprocs);
            rdispls.resize(nprocs);
            vdispls[0] = 0;
            rdispls[0] = 0;

            for (i = 1; i < nprocs; ++i) {
                vdispls[i] = vdispls[i - 1] + sendCount[i - 1];
                rdispls[i] = rdispls[i - 1] + recvCount[i - 1];
            }

            // total number of elements that each proc. sends and receives during matvec:
            vIndexSize = vdispls[nprocs - 1] + sendCount[nprocs - 1];
            recvSize   = rdispls[nprocs - 1] + recvCount[nprocs - 1];

            vIndex.resize(vIndexSize);
            MPI_Alltoallv(&vElement_remote[0], &recvCount[0], &rdispls[0], par::Mpi_datatype<index_t>::value(),
                          &vIndex[0],          &sendCount[0], &vdispls[0], par::Mpi_datatype<index_t>::value(), comm);

            vElement_remote.clear();
            vElement_remote.shrink_to_fit();

#ifdef __DEBUG1__
//            print_vector(vIndex, -1, "vIndex", comm);
            if(verbose_matrix_setup_sh) {
                MPI_Barrier(comm);
                printf("matrix_setup_shrink: rank = %d, local remote4 \n", rank);
                MPI_Barrier(comm);
            }
#endif

            // change the indices from global to local
#pragma omp parallel for
            for (i = 0; i < vIndexSize; i++){
                vIndex[i] -= split[rank];
            }

            // vSend     = vector values to send to other procs
            // vecValues = vector values to be received from other procs
            // These will be used in matvec and they are set here to reduce the time of matvec.
            vSend.resize(vIndexSize);
            vecValues.resize(recvSize);

//            vSend2.resize(vIndexSize);
//            vecValues2.resize(recvSize);

#ifdef SAENA_USE_ZFP
            if(use_zfp){
                allocate_zfp();
            }
#endif
        }

        // compute M_max
//        MPI_Allreduce(&M, &M_max, 1, MPI_UNSIGNED, MPI_MAX, comm);
//        M_max = 0;
//        for(i = 0; i < nprocs; ++i){
//            M_max = max(M_max, split[i+1] - split[i]);
//        }

        // compute nnz_max
//        MPI_Allreduce(&nnz_l, &nnz_max, 1, par::Mpi_datatype<nnz_t>::value(), MPI_MAX, comm);

        // compute nnz_list
//        nnz_list.resize(nprocs);
//        MPI_Allgather(&nnz_l, 1, par::Mpi_datatype<nnz_t>::value(), &nnz_list[0], 1, par::Mpi_datatype<nnz_t>::value(), comm);

#ifdef __DEBUG1__
//        print_vector(nnz_list, 1, "nnz_list", comm);
#endif

        // to be used in smoothers
//        temp1.resize(M);
//        temp2.resize(M);
    }
}

void saena_matrix::find_sortings_dummy(){
//    if(active) {
//        indicesP_local.resize(nnz_l_local);
//#pragma omp parallel for
//        for (nnz_t i = 0; i < nnz_l_local; i++)
//            indicesP_local[i] = i;
//
//        index_t *row_localP = &*row_local.begin();
//        std::sort(&indicesP_local[0], &indicesP_local[nnz_l_local], sort_indices(row_localP));
//    }
}

void saena_matrix::matvec_dummy(std::vector<value_t>& v, std::vector<value_t>& w) {

    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");
//    print_info(-1);
//    print_vector(v, -1, "v", comm);

    double t = 0, tcomm = 0;
//    ++matvec_iter;

    t = omp_get_wtime();
    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
    for(index_t i = 0; i < vIndexSize; ++i)
        vSend[i] = v[vIndex[i]];

    t = omp_get_wtime() - t;
    matvec_dummy_time[0] += t;

//    print_vector(vSend, 1, "vSend", comm);

    tcomm = omp_get_wtime();

    // receive and put the remote parts of v in vecValues.
    // they are received in order: first put the values from the lowest rank matrix, and so on.
    for(int i = 0; i < numRecvProc; ++i){
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], par::Mpi_datatype<value_t>::value(), recvProcRank[i], 1, comm, &requests[i]);
//        MPI_Test(&requests[i], &MPI_flag, &statuses[i]);
    }

    for(int i = 0; i < numSendProc; ++i){
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], par::Mpi_datatype<value_t>::value(), sendProcRank[i], 1, comm, &requests[numRecvProc+i]);
        MPI_Test(&requests[numRecvProc + i], &MPI_flag, &statuses[numRecvProc + i]);
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
            w[i] += val_local[iter] * v_p[col_local[iter]];
        }
    }

    t = omp_get_wtime() - t;
    matvec_dummy_time[1] += t;

    int np = 0;
    int recv_proc = 0, recv_proc_idx = 0;
    value_t *vecValues_p        = nullptr;
    index_t *nnzPerCol_remote_p = nullptr;
    while(np < numRecvProc){
        MPI_Waitany(numRecvProc, &requests[0], &recv_proc_idx, MPI_STATUS_IGNORE);
        ++np;

        recv_proc = recvProcRank[recv_proc_idx];
//        if(rank==1) printf("recv_proc_idx = %d, recv_proc = %d, np = %d, numRecvProc = %d, recvCount[recv_proc] = %d\n",
//                              recv_proc_idx, recv_proc, np, numRecvProc, recvCount[recv_proc]);

        t = omp_get_wtime();
        iter = nnzPerProcScan[recv_proc];
        vecValues_p        = &vecValues[rdispls[recv_proc]];
        nnzPerCol_remote_p = &nnzPerCol_remote[rdispls[recv_proc]];
        for (index_t j = 0; j < recvCount[recv_proc]; ++j) {
            for (index_t i = 0; i < nnzPerCol_remote_p[j]; ++i, ++iter) {
                w[row_remote[iter]] += values_remote[iter] * vecValues_p[j];
            }
        }
        t = omp_get_wtime() - t;
        matvec_dummy_time[2] += t;
    }

    MPI_Waitall(numSendProc, &requests[numRecvProc], MPI_STATUSES_IGNORE);

    tcomm = omp_get_wtime() - tcomm;
    matvec_dummy_time[3] += tcomm;
}