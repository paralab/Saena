#include "saena_matrix.h"
#include <unordered_map>

int saena_matrix::assemble(bool scale /*= false*/, bool use_den /*= false*/) {

    if(!assembled){
        repartition_nnz_initial();
        this->use_dense = use_den;
        matrix_setup(scale);
//        if(enable_shrink) compute_matvec_dummy_time(); // compute the matvec time for the coarsest level,
                                                       // which will be used when deciding about shrinking for level 1.
    }else{
        repartition_nnz_update();
        matrix_setup_update(scale);
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

//    printf("rank %d %s: data_coo.size = %ld\n", rank, __func__, data_coo.size());

    std::set<cooEntry_row>::iterator it;
    cooEntry_row temp;
    nnz_t iter = 0;

    // read this: https://stackoverflow.com/questions/5034211/c-copy-set-to-vector
    data_unsorted.resize(data_coo.size());
    for (it = data_coo.begin(); it != data_coo.end(); ++it) {
        data_unsorted[iter++] = *it;
    }

    // clear data_coo
    // since there is no function similar to shrink_to_fit for std::set, I am not sure if clear() would free memory,
    // so I use std::move on a tmp variable that gets deleted after this function returns.
    std::set<cooEntry_row> tmp(std::move(data_coo));

    remove_duplicates();

//    print_vector(data_with_bound, -1, "data_with_bound", comm);

    if(remove_boundary){
        remove_boundary_nodes();

        index_t Mbig_local = 0;
        if(!data.empty())
            Mbig_local = data.back().row;
        MPI_Allreduce(&Mbig_local, &Mbig, 1, par::Mpi_datatype<index_t>::value(), MPI_MAX, comm);
        Mbig++; // since indices start from 0, not 1.
    }else{
        data = std::move(data_with_bound);
    }

//    print_vector(data, -1, "data", comm);

    Nbig = Mbig; // the matrix is implemented as square

    initial_nnz_l = data.size();
    MPI_Allreduce(&initial_nnz_l, &nnz_g, 1, par::Mpi_datatype<nnz_t>::value(), MPI_SUM, comm);

//    printf("rank = %d, Mbig = %u, nnz_g = %ld, initial_nnz_l = %ld \n", rank, Mbig, nnz_g, initial_nnz_l);

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

    if(!rank) cout << "The matrix assemble function is being called for the second time."
                      "If the update part is required, enable it before calling the matrix update functions." << endl;
    MPI_Abort(comm, 1);

#if 0
    std::set<cooEntry_row>::iterator it;
    nnz_t iter = 0;

    data_unsorted.resize(data_coo.size());
    for(it = data_coo.begin(); it != data_coo.end(); ++it){
        data_unsorted[iter++] = *it;
    }

    remove_duplicates();

    initial_nnz_l = data.size();
    nnz_t nnz_g_temp = nnz_g;
    MPI_Allreduce(&initial_nnz_l, &nnz_g, 1, par::Mpi_datatype<nnz_t>::value(), MPI_SUM, comm);
    if((nnz_g_temp != nnz_g) && rank == 0){
        printf("error: number of global nonzeros is changed during the matrix update:\nbefore: %lu \tafter: %lu", nnz_g, nnz_g_temp);
        MPI_Finalize();
        return -1;
    }
#endif
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
//    std::set<cooEntry_row> data_temp = std::move(data_coo);

    // since shrink_to_fit() does not work for std::set, set.erase() is being used, not sure if it frees memory.
//    data_coo.erase(data_coo.begin(), data_coo.end());
//    data_coo.clear();

//    printf("rank = %d \t\t\t before sort: data_unsorted size = %lu\n", rank, data_unsorted.size());

//    for(int i=0; i<data_unsorted.size(); i++)
//        if(rank==0) std::cout << data_unsorted[i] << std::endl;

    // initial Mbig. it will get updated later, if boundary nodes get removed
    index_t Mbig_l = 0;
    if(!data_unsorted.empty())
        Mbig_l = data_unsorted.back().row;
    MPI_Allreduce(&Mbig_l, &Mbig, 1, par::Mpi_datatype<index_t>::value(), MPI_MAX, comm);
    Mbig++; // since indices start from 0, not 1.

//    printf("rank %d: Mbig_l = %d, Mbig = %d\n", rank, Mbig_l, Mbig);

    index_t ofst = Mbig / nprocs;

    // initial split. it will get updated later
    split.resize(nprocs + 1);
    for(int i = 0; i < nprocs; ++i){
        split[i] = i * ofst;
    }
    split[nprocs] = Mbig;

//    print_vector(split, 0, "split", comm);

    std::vector<cooEntry_row> data_sorted_row;
//    par::sampleSort(data_unsorted, data_sorted_row, comm);
    par::sampleSort(data_unsorted, data_sorted_row, split, comm);

//    print_vector(data_sorted_row, -1, "data_sorted_row", comm);

    // clear data_unsorted and free memory.
    data_unsorted.clear();
    data_unsorted.shrink_to_fit();

    if(data_sorted_row.empty()) {
        printf("Error: data has no element on process %d! \n", rank);
        exit(EXIT_FAILURE);
    }

    // switch from cooEntry_row to cooEntry.
//    std::vector<cooEntry> data_sorted(data_sorted_row.size());
//    memcpy(&data_sorted[0], &data_sorted_row[0], data_sorted_row.size() * sizeof(cooEntry));

//    printf("rank = %d \t\t\t after  sort: data_sorted size = %lu\n", rank, data_sorted.size());

    // remove duplicates
    // -----------------------

    // put the first element of data_unsorted to data.
    value_t tmp     = 0.0;
    nnz_t data_size = 0;
    const nnz_t sz  = data_sorted_row.size();

    if(add_duplicates){
        for(nnz_t i = 0; i < sz; ++i) {
//            if(rank==1) cout << data_sorted_row[i] << endl;
            tmp = data_sorted_row[i].val;
            while(i + 1 < sz && data_sorted_row[i + 1] == data_sorted_row[i]){
                tmp += data_sorted_row[++i].val;
            }

            if (fabs(tmp) > ALMOST_ZERO) {
                data_with_bound.emplace_back( cooEntry(data_sorted_row[i].row, data_sorted_row[i].col, tmp) );
            }
        }
    } else {
        for(nnz_t i = 0; i < sz; ++i) {
            while(i + 1 < sz && data_sorted_row[i + 1] == data_sorted_row[i]){
                ++i;
            }
            if (fabs(data_sorted_row[i].val) > ALMOST_ZERO) {
                data_with_bound.emplace_back(data_sorted_row[i].row, data_sorted_row[i].col, data_sorted_row[i].val);
            }
        }
    }

    // todo: replace the previous part with this
#if 0
    nnz_t data_sorted_size_minus1 = data_sorted.size()-1;
    if(add_duplicates){
        for(nnz_t i = 0; i < data_sorted.size(); i++){
            data.emplace_back(data_sorted[i]);
            while(i < data_sorted_size_minus1 && data_sorted[i] == data_sorted[i+1]){ // values of entries with the same row should be added.
                std::cout << data_sorted[i] << "\t" << data_sorted[i+1] << std::endl;
                data.back().val += data_sorted[++i].val;
            }
        }
    } else {
        for(nnz_t i = 0; i < data_sorted.size(); i++){
            data.emplace_back(data_sorted[i]);
            while(i < data_sorted_size_minus1 && data_sorted[i] == data_sorted[i+1]){ // values of entries with the same row should be added.
                ++i;
            }
        }
    }
#endif

#if 0
    // check for dupliactes on boundary points of the processors
    // ---------------------------------------------------------
    // receive first element of your left neighbor and check if it is equal to your last element.
    cooEntry first_element_neighbor = cooEntry(0, 0, 0.0);
    if(rank != nprocs-1)
        MPI_Recv(&first_element_neighbor, 1, cooEntry::mpi_datatype(), rank+1, 0, comm, MPI_STATUS_IGNORE);

    if(rank!= 0)
        MPI_Send(&data_with_bound[0], 1, cooEntry::mpi_datatype(), rank-1, 0, comm);

    cooEntry last_element = data_with_bound.back();
    if(rank != nprocs-1){
        if(last_element == first_element_neighbor) {
            data_with_bound.pop_back();
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

        data_with_bound[0].val += left_neighbor_last_val;
    }
#endif

//    print_vector(data_with_bound, -1, "data_with_bound", comm);

    return 0;
}


int saena_matrix::remove_boundary_nodes() {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    int rank_v = 1;
//    print_vector(split, rank_v, "split", comm);
//    print_vector(data_with_bound, rank_v, "data_with_bound", comm);
#endif

    // update column indices after removing the boundary nodes using this map.
    // m[old_index] = new_index
    // only the interior nodes need to be mapped
//    unordered_map<index_t, index_t> m;

    // M will be updated later
    M = split[rank + 1] - split[rank];
    auto *new_idx = new index_t[M];
    assert(new_idx);
    for(int i = 0; i < M; ++i){
        new_idx[i] = -1;
    }
//    std::fill(&new_idx[0], &new_idx[M], -1);
//    vector<index_t> new_idx(M, -1);
//    print_array(new_idx, M, rank_v, "send_idx", comm);

    // save M and Mbig before removing boundary nodes
    M_orig = M;

    const index_t ofst = split[rank];
    const nnz_t   SZ   = data_with_bound.size();

    index_t i = 0;
    for(; i < SZ - 1; ++i){
        if(i + 1 < SZ && data_with_bound[i].row != data_with_bound[i + 1].row){ // boundary
#ifdef __DEBUG1__
//            if(rank == rank_v) std::cout << "boundry: " << data_with_bound[i] << std::endl;
//            if(rank == rank_v) std::cout << "new row idx: " << data_with_bound[i].row - ofst - bound_row.size() << std::endl;
            assert(data_with_bound[i].row - ofst >= 0);
#endif
            bound_row.emplace_back(data_with_bound[i].row - ofst);
            bound_val.emplace_back(data_with_bound[i].val);
            ASSERT(data_with_bound[i].row == data_with_bound[i].col, data_with_bound[i].row << " != " << data_with_bound[i].col);
        }else{
#ifdef __DEBUG1__
//            if(rank == rank_v) std::cout << "interior: " << data_with_bound[i] << std::endl;
//            if(rank == rank_v) std::cout << data_with_bound[i].row << " -> " << data_with_bound[i].row - bound_row.size() << std::endl;
//            if(rank == rank_v) std::cout << "new row idx: " << data_with_bound[i].row - ofst - bound_row.size() << std::endl;
#endif
            new_idx[data_with_bound[i].row - ofst] = data_with_bound[i].row - bound_row.size() - ofst;
            data.emplace_back(data_with_bound[i].row - bound_row.size() - ofst, data_with_bound[i].col, data_with_bound[i].val);
            while(i + 1 < SZ && data_with_bound[i].row == data_with_bound[i + 1].row){
                ++i;
                data.emplace_back(data_with_bound[i].row - bound_row.size() - ofst,
                                  data_with_bound[i].col,
                                  data_with_bound[i].val);
            }
        }
    }

    // if the last nonzero is left, it means the last row of the matrix had only one nozero, otherwise the whole row
    // would have been added in the previous while loop
    if(i == SZ - 1){
//        if(rank == rank_v) std::cout << "bound: " << data_with_bound[i] << std::endl;
        bound_row.emplace_back(data_with_bound[i].row - ofst);
        bound_val.emplace_back(data_with_bound[i].val);
    }

//    printf("rank %d: total points = %d, boundary points = %ld, interior points = %ld\n",
//           rank, M_orig, bound_row.size(), M_orig - bound_row.size());

//    print_array(new_idx, M, rank_v, "new_idx", comm);
//    print_vector(data, -1, "data before update", comm);
//    print_vector(bound_row, -1, "bound_row", comm);
//    print_vector(bound_val, -1, "bound_val", comm);

    // check if there is any diagonal boundary point on all processes
    bool bnd_l = !bound_row.empty(), bnd = true; // set true if there is boundary
    MPI_Allreduce(&bnd_l, &bnd, 1, MPI_CXX_BOOL, MPI_LOR, comm);

    if(!bnd){
        remove_boundary = false;
        data = move(data_with_bound);
    }else {
        // update the column indices to the new indices after removing the boundary nodes
        if (nprocs == 1) {
            for (auto &d : data) {
//                if(rank == rank_v) cout << d.col << "\t" << new_idx[d.col] << endl;
                d.col = new_idx[d.col];
            }
        } else {
            index_t max_sz = 0;
            for (i = 1; i < split.size(); ++i) {
                max_sz = max(max_sz, split[i] - split[i - 1]);
            }

//            printf("rank %d: max_sz = %d\n", rank, max_sz);
//            if(rank == rank_v) cout << "max size = " << max_sz << endl;

            // M will be updated later
            const index_t M_tmp = M - bound_row.size();
            vector<index_t> split_tmp(nprocs + 1);
            MPI_Allgather(&M_tmp, 1, par::Mpi_datatype<index_t>::value(), &split_tmp[1], 1,
                          par::Mpi_datatype<index_t>::value(), comm);

            split_tmp[0] = 0;
            for (i = 1; i < nprocs + 1; ++i) {
                split_tmp[i] += split_tmp[i - 1];
            }

//            print_vector(split_tmp, 1, "split_tmp", comm);

            for (i = 0; i < M; ++i) {
                if (new_idx[i] != -1)
                    new_idx[i] += split_tmp[rank];
            }

//            print_array(new_idx, M, rank_v, "new_idx", comm);

            for (auto &d : data) {
                d.row += split_tmp[rank];
            }

            sort(data.begin(), data.end());

//            print_vector(data, rank_v, "data", comm);

            int right_neighbor = (rank + 1) % nprocs;
            int left_neighbor = rank - 1;
            if (left_neighbor < 0) {
                left_neighbor += nprocs;
            }

            int owner = 0, next_owner = 0;
            nnz_t send_sz = M;
            nnz_t recv_sz = 0;

            index_t *send_idx = new index_t[max_sz];
            memcpy(send_idx, new_idx, M * sizeof(index_t));
            index_t *send_idx_p = nullptr;

            index_t *recv_idx = new index_t[max_sz];
            memcpy(recv_idx, send_idx, send_sz * sizeof(index_t));

//            print_array(send_idx, send_sz, rank_v, "send_idx", comm);

            int flag = 0;
            MPI_Request reqs[2];
//            MPI_Status  statuses[2];

            nnz_t it = 0;
            if (!data.empty()) {
                while (data[it].col < split[rank] && it < data.size()) {
                    ++it;
                }
            }

            nnz_t it2 = 0;
            const nnz_t SZ2 = data.size();

            for (int k = rank; k < rank + nprocs; ++k) {
//                MPI_Barrier(comm);
//                if(rank==rank_v) printf("rank %d: k = %d, it2 = %ld, SZ2 = %ld\n", rank, k, it2, SZ2);
//                print_array(send_idx, send_sz, rank_v, "send_idx", comm);

                owner = k % nprocs;
                next_owner = (k + 1) % nprocs;
                recv_sz = split[next_owner + 1] - split[next_owner];

                MPI_Irecv(recv_idx, recv_sz, par::Mpi_datatype<index_t>::value(), right_neighbor, 0, comm, reqs);
                MPI_Isend(send_idx, send_sz, par::Mpi_datatype<index_t>::value(), left_neighbor, 0, comm, reqs + 1);

//                MPI_Test(reqs,   &flag, statuses);
//                MPI_Test(reqs+1, &flag, statuses+1);

                // update column indices
                send_idx_p = &send_idx[0] - split[owner];
//                if(rank==rank_v) cout << endl;
                if (it2 < SZ2 && split[owner] <= data[it].col) {
                    while (it2 < SZ2 && data[it].col < split[owner + 1]) {
//                        if (rank == rank_v) cout << it << "\t" << data[it] << "\t" << split[owner] << "\t" <<
//                                 data[it].col - split[owner] << "\t" << send_idx[data[it].col - split[owner]] << endl;

                        ASSERT(send_idx_p[data[it].col] >= 0,
                               "rank " << rank << ": " << send_idx_p[data[it].col] << ", owner: " << owner);
                        data[it].col = send_idx_p[data[it].col];
                        ++it;
                        ++it2;
                        if (it >= SZ2) {
                            it = 0;
                            break;
                        }
                    }
                }

                MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
                send_sz = recv_sz;
                std::swap(send_idx, recv_idx);

//                print_vector(data, rank_v, "data", comm);
//                print_array(send_idx, send_sz, rank_v, "send_idx", comm);
            }

            delete[] send_idx;
            delete[] recv_idx;

            sort(data.begin(), data.end(), row_major);
        }
    }

#ifdef __DEBUG1__
//    printf("rank %d: done removing boundary\n", rank);
//    print_vector(data, -1, "data after removing boundary nodes", comm);
//    print_vector(bound_row, rank_v, "bound_row", comm);
//    print_vector(bound_val, rank_v, "bound_val", comm);
#endif

    delete [] new_idx;
    data_with_bound.clear();
    data_with_bound.shrink_to_fit();

    return 0;
}

int saena_matrix::matrix_setup(bool scale /*= false*/) {
    // before using this function the following parameters of saena_matrix should be set:
    // "Mbig", "M", "nnz_g", "split", "entry",

    if(active) {
        int nprocs, rank;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);

        if (verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu \n", rank, Mbig, M, nnz_g,
                   nnz_l);
            MPI_Barrier(comm);
        }

        assert(nnz_g != 0);

//        printf("matrix_setup: rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu \n", rank, Mbig, M, nnz_g, nnz_l);
//        print_vector(entry, -1, "entry", comm);

        assembled = true;
        total_active_procs = nprocs;

        // *************************** set the inverse of diagonal of A (for smoothers) ****************************

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, inv_diag \n", rank);
            MPI_Barrier(comm);
        }

        inverse_diag();

//        print_vector(inv_diag, -1, "inv_diag", comm);

        // *************************** set rho ****************************

//        set_rho();

        // *************************** set and exchange on-diagonal and off-diagonal elements ****************************

        set_off_on_diagonal();

        // *************************** find sortings ****************************

//        find_sortings();

        // *************************** find start and end of each thread for matvec ****************************
        // also, find nnz per row for local and remote matvec

        openmp_setup();
        w_buff.resize(num_threads * M); // allocate for w_buff for matvec3()

        // *************************** scale ****************************
        // scale the matrix to have its diagonal entries all equal to 1.

//        for(nnz_t i = 0; i < nnz_l_local; i++) {
//            if (rank == 0)
//                printf("%u \t%u \t%f\n", row_local[i], col_local[i], val_local[i]);
//        }

//        print_vector(entry, 0, "entry", comm);
        if(scale)
            scale_matrix();
//        print_vector(entry, 0, "entry", comm);

//        for(nnz_t i = 0; i < nnz_l_local; i++) {
//            if (rank == 0)
//                printf("%u \t%u \t%f\n", row_local[i], col_local[i], val_local[i]);
//        }

        // *************************** dense data structure ****************************

        if(use_dense)
            generate_dense_matrix();

        // *************************** print_entry info ****************************

#ifdef __DEBUG1__
#if 0
        nnz_t total_nnz_l_local;
        nnz_t total_nnz_l_remote;
        MPI_Allreduce(&nnz_l_local,  &total_nnz_l_local,  1, par::Mpi_datatype<nnz_t>::value(), MPI_SUM, comm);
        MPI_Allreduce(&nnz_l_remote, &total_nnz_l_remote, 1, par::Mpi_datatype<nnz_t>::value(), MPI_SUM, comm);
        int local_percent  = int(100*(float)total_nnz_l_local/nnz_g);
        int remote_percent = int(100*(float)total_nnz_l_remote/nnz_g);
        if(rank==0) printf("\nMbig = %u, nnz_g = %lu, total_nnz_l_local = %lu (%%%d), total_nnz_l_remote = %lu (%%%d) \n",
                           Mbig, nnz_g, total_nnz_l_local, local_percent, total_nnz_l_remote, remote_percent);

//        printf("rank %d: col_remote_size = %u \n", rank, col_remote_size);
        index_t col_remote_size_min, col_remote_size_ave, col_remote_size_max;
        MPI_Allreduce(&col_remote_size, &col_remote_size_min, 1, par::Mpi_datatype<index_t>::value(), MPI_MIN, comm);
        MPI_Allreduce(&col_remote_size, &col_remote_size_ave, 1, par::Mpi_datatype<index_t>::value(), MPI_SUM, comm);
        MPI_Allreduce(&col_remote_size, &col_remote_size_max, 1, par::Mpi_datatype<index_t>::value(), MPI_MAX, comm);
        if(rank==0) printf("\nremote_min = %u, remote_ave = %u, remote_max = %u \n",
                           col_remote_size_min, (col_remote_size_ave/nprocs), col_remote_size_max);
#endif
#endif

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, done \n", rank);
            MPI_Barrier(comm);
        }

    } // if(active)

    return 0;
}


int saena_matrix::matrix_setup_update(bool scale /*= false*/) {
    // update val_local, val_remote and inv_diag.

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

//    assembled = true;

    // todo: check if instead of clearing and pushing back, it is possible to only update the values.
//    saena_free(val_local);
//    saena_free(val_remote);

    if(!entry.empty()) {
        for (nnz_t i = 0; i < nnz_l; i++) {
            if (entry[i].col >= split[rank] && entry[i].col < split[rank + 1]) {
//                val_local.emplace_back(entry[i].val);
                val_local[i] = entry[i].val;
            } else {
//                val_remote.emplace_back(entry[i].val);
                val_remote[i] = entry[i].val;
            }
        }
    }

//    inv_diag.resize(M);
    inverse_diag();

    if(scale){
        scale_matrix();
    }

    return 0;
}


int saena_matrix::matrix_setup_lazy_update() {
    // update val_local, val_remote and inv_diag.

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

//    assembled = true;

    if(entry.size() != nnz_l){
        printf("nnz_l has changed in the update process!\n");
    }

    // todo: check if instead of clearing and pushing back, it is possible to only update the values.
//    saena_free(val_local);
//    saena_free(val_remote);

    if(!entry.empty()) {
        for (nnz_t i = 0; i < nnz_l; i++) {
            if (entry[i].col >= split[rank] && entry[i].col < split[rank + 1]) {
//                val_local.emplace_back(entry[i].val);
                val_local[i] = entry[i].val;
            } else {
//                val_remote.emplace_back(entry[i].val);
                val_remote[i] = entry[i].val;
            }
        }
    }

//    inv_diag.resize(M);
//    inverse_diag();

//    scale_matrix();

    return 0;
}


int saena_matrix::update_diag_lazy(){

    int rank;
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
//    MPI_Barrier(comm);
//    if(rank==0) printf("update_diag_lazy!!!\n");
//    print_vector(split, 0, "split", comm);
//    print_entry(-1);
//    print_vector(inv_diag, -1, "inv diag", comm);
//    print_vector(inv_sq_diag, -1, "inv_sq_diag diag", comm);
//    MPI_Barrier(comm);
#endif

    std::vector<value_t> inv_sq_diag_old = inv_sq_diag;

    double temp;
//    inv_diag.assign(M, 0);
//    inv_sq_diag.assign(M, 0);

    if(!entry.empty()) {
        for (nnz_t i = 0; i < nnz_l; i++) {

            if (entry[i].row == entry[i].col) {
//                if(rank==0) std::cout << i << "\t" << entry[i] << std::endl;
                if ( !almost_zero(entry[i].val) ) {
                    temp = 1.0 / entry[i].val;
//                    inv_diag_original[entry[i].row - split[rank]] = temp; // this line is different from inverse_diag()
                    inv_sq_diag[entry[i].row - split[rank]] = sqrt(temp);
//                    if (fabs(temp) > highest_diag_val) {
//                        highest_diag_val = fabs(temp);
//                    }
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

//    if(rank==0){
//        printf("inv_sq_diag_old and inv_sq_diag\n");
//        for(index_t i = 0; i < M; i++){
//            std::cout << inv_sq_diag_old[i] << "\t" << inv_sq_diag[i] << std::endl;
//        }
//    }

#ifdef __DEBUG1__
//    MPI_Barrier(comm);
//    print_vector(inv_diag, -1, "inv diag", comm);
//    print_vector(inv_sq_diag, -1, "inv_sq_diag diag", comm);
//    MPI_Barrier(comm);
#endif

//    for(auto i:inv_diag) {
//        if (i == 0){
//            printf("rank %d: inverse_diag: At least one diagonal entry is 0.\n", rank);
//        }
//    }

//    temp = highest_diag_val;
//    MPI_Allreduce(&temp, &highest_diag_val, 1, MPI_DOUBLE, MPI_MAX, comm);
//    if(rank==0) printf("\ninverse_diag: highest_diag_val = %f \n", highest_diag_val);

    return 0;
}


// int saena_matrix::set_rho()
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

    // todo: check which variables here is not required later and can be freed at the end of this function.

    if(active){
        int nprocs, rank;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
        int rank_v = 1;
        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, local remote1 \n", rank);
            MPI_Barrier(comm);
//            print_entry(-1);
//            print_vector(split, 0, "split", comm);
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

        // store remote entries in the following vectors. These are not memory-aligned. So then move them into aligned
        // arrays.
        vector<index_t> row_rem_unaligned;
        vector<index_t> col_rem_unaligned;
        vector<index_t> col_rem2_unaligned;
        vector<value_t> val_rem_unaligned;

        nnz_t i = 0;
        while(i < nnz_l) {
//            if(rank==rank_v) cout << endl << entry[i] << endl;
            procNum = lower_bound2(&split[0], &split[nprocs], entry[i].col);
//            if(rank==rank_v) printf("col = %u \tprocNum = %d \n", entry[i].col, procNum);
            if(procNum == rank){ // local
                while(i < nnz_l && entry[i].col < split[procNum + 1]) {
//                    if(rank==rank_v) printf("entry[i].row = %d, split[rank] = %d, dif = %d - local\n", entry[i].row, split[rank], entry[i].row - split[rank]);
//                    if(rank==rank_v) cout << entry[i] << endl;
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
//                        if(rank==rank_v) cout << entry[i] << endl;

                        // the original col values are not being used in matvec. the ordering starts from 0, and goes up by 1.
                        // col_remote2 is the original col value and will be used in making strength matrix.
                        col_rem_unaligned.emplace_back(vElement_remote.size() - 1);
                        col_rem2_unaligned.emplace_back(entry[i].col);
                        row_rem_unaligned.emplace_back(entry[i].row - split[rank]);
                        val_rem_unaligned.emplace_back(entry[i].val);
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
        nnz_l_remote    = row_rem_unaligned.size();
        col_remote_size = vElement_remote.size();

#ifdef __DEBUG1__
        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, local remote2 \n", rank);
            MPI_Barrier(comm);
        }
#endif

        // don't receive anything from yourself
        recvCount[rank] = 0;

//        print_vector(recvCount, 0, "recvCount", comm);

        // sort local entries in row-major order and remote entries in column-major order
        sort(ent_loc_row.begin(), ent_loc_row.end());

//        print_vector(ent_loc_row, -1, "ent_loc_row", comm);

        const nnz_t nnzl = nnz_l_local;
        row_local = saena_aligned_alloc<index_t>(nnzl);
        assert(row_local);
        col_local = saena_aligned_alloc<index_t>(nnzl);
        assert(col_local);
        val_local = saena_aligned_alloc<value_t>(nnzl);
        assert(val_local);

        for(i = 0; i < nnzl; ++i){
            row_local[i] = ent_loc_row[i].row;
            col_local[i] = ent_loc_row[i].col;
            val_local[i] = ent_loc_row[i].val;
        }

//        print_array(row_local, nnzl, -1, "row_local", comm);
//        print_array(col_local, nnzl, -1, "col_local", comm);
//        print_array(val_local, nnzl, -1, "val_local", comm);

        ent_loc_row.clear();
        ent_loc_row.shrink_to_fit();

        // copy remote entries to the aligned arrays
        const nnz_t nnzr = nnz_l_remote;
        row_remote = saena_aligned_alloc<index_t>(nnzr);
        assert(row_remote);
        col_remote = saena_aligned_alloc<index_t>(nnzr);
        assert(col_remote);
        col_remote2 = saena_aligned_alloc<index_t>(nnzr);
        assert(col_remote2);
        val_remote = saena_aligned_alloc<value_t>(nnzr);
        assert(val_remote);

        std::copy(&row_rem_unaligned[0], &row_rem_unaligned[nnzr], &row_remote[0]);
        std::copy(&col_rem_unaligned[0], &col_rem_unaligned[nnzr], &col_remote[0]);
        std::copy(&col_rem2_unaligned[0], &col_rem2_unaligned[nnzr], &col_remote2[0]);
        std::copy(&val_rem_unaligned[0], &val_rem_unaligned[nnzr], &val_remote[0]);

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
            if(verbose_matrix_setup) {
                MPI_Barrier(comm);
                printf("matrix_setup: rank = %d, local remote3 \n", rank);
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

#ifdef __DEBUG1__
            {
//                print_vector(vdispls, -1, "vdispls", comm);
//                print_vector(rdispls, -1, "rdispls", comm);
//                printf("rank %d: vIndexSize = %d, recvSize = %d\n", rank, vIndexSize, recvSize);
//                printf("rank %d: vIndexSize = %d, recvSize = %d, send_bufsize = %d, recv_bufsize = %d \n",
//                   rank, vIndexSize, recvSize, send_bufsize, recv_bufsize);
/*
                // compute min, average and max send size during matvec
                int vIndexSize_ave = 0, vIndexSize_min = 0, vIndexSize_max = 0;
                MPI_Allreduce(&vIndexSize, &vIndexSize_min, 1, MPI_INT, MPI_MIN, comm);
                MPI_Allreduce(&vIndexSize, &vIndexSize_max, 1, MPI_INT, MPI_MAX, comm);
                MPI_Allreduce(&vIndexSize, &vIndexSize_ave, 1, MPI_INT, MPI_SUM, comm);
                vIndexSize_ave /= nprocs;

                // compute min, average and max receive size during matvec
                int recvSize_ave = 0, recvSize_min = 0, recvSize_max = 0;
                MPI_Allreduce(&recvSize, &recvSize_min, 1, MPI_INT, MPI_MIN, comm);
                MPI_Allreduce(&recvSize, &recvSize_max, 1, MPI_INT, MPI_MAX, comm);
                MPI_Allreduce(&recvSize, &recvSize_ave, 1, MPI_INT, MPI_SUM, comm);
                recvSize_ave /= nprocs;
                if(!rank) printf("\nsend_sz = (%d, %d, %d), recv_sz = (%d, %d, %d) (min, ave, max)\n",
                       vIndexSize_min, vIndexSize_ave, vIndexSize_max, recvSize_min, recvSize_ave, recvSize_max);

*/
            }
#endif

            vIndex.resize(vIndexSize);
            MPI_Alltoallv(&vElement_remote[0], &recvCount[0], &rdispls[0], par::Mpi_datatype<index_t>::value(),
                          &vIndex[0],          &sendCount[0], &vdispls[0], par::Mpi_datatype<index_t>::value(), comm);

            vElement_remote.clear();
            vElement_remote.shrink_to_fit();
            sendCount.clear();
            sendCount.shrink_to_fit();

#ifdef __DEBUG1__
//            print_vector(vIndex, -1, "vIndex", comm);
            if(verbose_matrix_setup) {
                MPI_Barrier(comm);
                printf("matrix_setup: rank = %d, local remote4 \n", rank);
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

            vSend_f.resize(vIndexSize);
            vecValues_f.resize(recvSize);

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
        M_max = 0;
        for(i = 0; i < nprocs; ++i){
            M_max = max(M_max, split[i+1] - split[i]);
        }

        // compute nnz_max
        MPI_Allreduce(&nnz_l, &nnz_max, 1, par::Mpi_datatype<nnz_t>::value(), MPI_MAX, comm);

        // compute nnz_list
        nnz_list.resize(nprocs);
        MPI_Allgather(&nnz_l, 1, par::Mpi_datatype<nnz_t>::value(), &nnz_list[0], 1, par::Mpi_datatype<nnz_t>::value(), comm);

#ifdef __DEBUG1__
//        print_vector(nnz_list, 1, "nnz_list", comm);
#endif

        // to be used in smoothers
        temp1 = saena_aligned_alloc<value_t>(M);
        temp2 = saena_aligned_alloc<value_t>(M);
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
/*
        indicesP_local.resize(nnz_l_local);
#pragma omp parallel for
        for (nnz_t i = 0; i < nnz_l_local; i++)
            indicesP_local[i] = i;

        index_t *row_localP = &*row_local.begin();
        std::sort(&indicesP_local[0], &indicesP_local[nnz_l_local], sort_indices(row_localP));
*/

//    if(rank==0)
//        for(index_t i=0; i<nnz_l_local; i++)
//            std::cout << row_local[indicesP_local[i]] << "\t" << col_local[indicesP_local[i]]
//                      << "\t" << val_local[indicesP_local[i]] << std::endl;

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

        matvec_levels = static_cast<int>( ceil( log2(num_threads) ) );

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
/*
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
*/

//        if(rank==0){
//            printf("\niter_local_array and iter_local_array2: \n");
//            for(int i = 0; i < num_threads+1; i++)
//                printf("%u \t%u \n", iter_local_array[i], iter_local_array2[i]);}
    } //if(active)

    return 0;
}


int saena_matrix::scale_matrix(bool full_scale/* = false*/){

    // scale matrix: A = D^{-1/2} * A * D^{-1/2}
    // val_local, val_remote and entry are being updated.
    // A[i] *= D^{-1/2}[row[i]] * D^{-1/2}[col[i]]

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");

//    MPI_Barrier(comm); if(rank==1) printf("start of saena_matrix::scale()\n"); MPI_Barrier(comm);

    MPI_Request* requests = nullptr;
    MPI_Status*  statuses = nullptr;

    if(nprocs > 1){
        // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
        // put the values of thoss indices in vSend to send to other procs.
#pragma omp parallel for
        for(index_t i = 0; i < vIndexSize; ++i)
            vSend[i] = inv_sq_diag[vIndex[i]];

//        print_vector(vSend, -1, "vSend", comm);

        int flag = 0; // used for MPI_Test
        requests = new MPI_Request[numSendProc + numRecvProc];
        statuses = new MPI_Status[numSendProc + numRecvProc];

        // receive and put the remote parts of v in vecValues.
        // they are received in order: first put the values from the lowest rank matrix, and so on.
        for(int i = 0; i < numRecvProc; ++i){
            MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], par::Mpi_datatype<value_t>::value(), recvProcRank[i], 1, comm, &requests[i]);
            MPI_Test(&requests[i], &flag, &statuses[i]);
        }

        for(int i = 0; i < numSendProc; ++i){
            MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], par::Mpi_datatype<value_t>::value(), sendProcRank[i], 1, comm, &requests[numRecvProc+i]);
            MPI_Test(&requests[numRecvProc + i], &flag, &statuses[numRecvProc + i]);
        }
    }

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

    // use these to avoid subtracting split[rank] from each case
//    auto *inv_sq_diag_p = &inv_sq_diag[0] - split[rank];

#pragma omp parallel for
    for(nnz_t i = 0; i < nnz_l_local; i++) {
//        cout << i << "\t" << std::setprecision(16) << val_local[i] << "\t" << std::setprecision(16) << inv_sq_diag[row_local[i]] * inv_sq_diag[col_local[i] - split[rank]] << endl;
//        val_local[i] *= inv_sq_diag[row_local[i]] * inv_sq_diag_p[col_local[i]]; //D^{-1/2} * A * D^{-1/2}
        val_local[i] *= inv_sq_diag[row_local[i]] * inv_sq_diag[col_local[i] - split[rank]]; //D^{-1/2} * A * D^{-1/2}
    }

//    print_vector(val_local, -1, "val_local", comm);

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
            index_t i;
            int thread_id = omp_get_thread_num();
            nnz_t iter = iter_remote_array[thread_id];
#pragma omp for
            for (index_t j = 0; j < col_remote_size; ++j) {
                for (i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
                    val_remote[iter] *= inv_sq_diag[row_remote[iter]] * vecValues[j]; // D^{-1/2} * A * D^{-1/2}
                }
            }
        }
    }

    if(full_scale) {
        // update the entry vector
        entry.clear();
//        entry.resize(nnz_l);

        // copy local entries
#pragma omp parallel for
        for (nnz_t i = 0; i < nnz_l_local; ++i) {
//            entry[i] = cooEntry(row_local[i]+split[rank], col_local[i], val_local[i]);
            entry.emplace_back(cooEntry(row_local[i] + split[rank], col_local[i], val_local[i]));
        }

        if (nprocs > 1) {
            // copy remote entries
#pragma omp parallel for
            for (nnz_t i = 0; i < nnz_l_remote; ++i) {
//                entry[nnz_l_local + i] = cooEntry(row_remote[i]+split[rank], col_remote2[i], val_remote[i]);
                entry.emplace_back(cooEntry(row_remote[i] + split[rank], col_remote2[i], val_remote[i]));
            }
        }

        std::sort(entry.begin(), entry.end());

//        print_vector(inv_diag, -1, "inv_diag", comm);

        swap(inv_diag, inv_diag_orig);
//        inv_diag_orig    = std::move(inv_diag);
        inv_sq_diag_orig = std::move(inv_sq_diag);

        inv_diag = saena_aligned_alloc<value_t>(M);
        fill(&inv_diag[0], &inv_diag[M], 1.0);
        inv_sq_diag.assign(inv_sq_diag_orig.size(), 1);
//        inv_sq_diag = inv_sq_diag_orig;
    }

    if(nprocs > 1){
        MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
        delete [] requests;
        delete [] statuses;
    }


//    MPI_Barrier(comm); if(rank==0) printf("end of saena_matrix::scale()\n"); MPI_Barrier(comm);

    return 0;
}


int saena_matrix::scale_back_matrix(bool full_scale/* = false*/){

    // scale back matrix: A = D^{1/2} * A * D^{1/2}
    // val_local, val_remote and entry are being updated.
    // A[i] /= D^{-1/2}[row[i]] * D^{-1/2}[col[i]]

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ) printf("A.M != v.size() in matvec!\n");
//    MPI_Barrier(comm); if(rank==1) printf("start of saena_matrix::scale()\n"); MPI_Barrier(comm);

    if(full_scale) {
        inv_diag = std::move(inv_diag_orig);
        inv_sq_diag = std::move(inv_sq_diag_orig);
    }

    MPI_Request* requests = nullptr;
    MPI_Status* statuses  = nullptr;

    if(nprocs > 1){
        // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
        // put the values of thoss indices in vSend to send to other procs.
#pragma omp parallel for
        for(index_t i = 0; i < vIndexSize; ++i)
            vSend[i] = inv_sq_diag[vIndex[i]];

//        print_vector(vSend, -1, "vSend", comm);

        int flag = 0;
        requests = new MPI_Request[numSendProc + numRecvProc];
        statuses = new MPI_Status[numSendProc + numRecvProc];

        // receive and put the remote parts of v in vecValues.
        // they are received in order: first put the values from the lowest rank matrix, and so on.
        for(int i = 0; i < numRecvProc; ++i){
            MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], par::Mpi_datatype<value_t>::value(), recvProcRank[i], 1, comm, &requests[i]);
            MPI_Test(&requests[i], &flag, &statuses[i]);
        }

        for(int i = 0; i < numSendProc; ++i){
            MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], par::Mpi_datatype<value_t>::value(), sendProcRank[i], 1, comm, &requests[numRecvProc+i]);
            MPI_Test(&requests[numRecvProc + i], &flag, &statuses[numRecvProc + i]);
        }
    }

    // local loop
    // ----------
    // D^{-1/2} * A * D^{-1/2}
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

//    index_t* col_p = &col_local[0] - split[rank];
#pragma omp parallel for
    for(nnz_t i = 0; i < nnz_l_local; i++) {
        val_local[i] /= inv_sq_diag[row_local[i]] * inv_sq_diag[col_local[i] - split[rank]];
//        cout << i << "\t"  << std::setprecision(16) << val_local[i] << "\t"  << std::setprecision(16) << inv_sq_diag[row_local[i]] * inv_sq_diag[col_local[i] - split[rank]] << endl;
    }

//    print_vector(val_local, -1, "val_local", comm);

    if(nprocs > 1){
        // Wait for the receive communication to finish.
        MPI_Waitall(numRecvProc, requests, statuses);

//        print_vector(vecValues, -1, "vecValues", comm);

        // remote loop
        // -----------
        // D^{-1/2} * A * D^{-1/2}
        // the col_index of the matrix entry does not matter. do the matvec on the first non-zero col// D^{-1/2} * A * D^{-1/2}umn (j=0).
        // the corresponding vector element is saved in vecValues[0]. and so on.

#pragma omp parallel
        {
            index_t i;
            int thread_id = omp_get_thread_num();
            nnz_t iter = iter_remote_array[thread_id];
#pragma omp for
            for (index_t j = 0; j < col_remote_size; ++j) {
                for (i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
                    val_remote[iter] /= inv_sq_diag[row_remote[iter]] * vecValues[j];
                }
            }
        }
    }

    if(full_scale) {
        // update the entry vector
        entry.clear();
//        entry.resize(nnz_l);

        // copy local entries
#pragma omp parallel for
        for (nnz_t i = 0; i < nnz_l_local; ++i) {
            entry.emplace_back(cooEntry(row_local[i] + split[rank], col_local[i], val_local[i]));
        }

        if (nprocs > 1) {
            // copy remote entries
#pragma omp parallel for
            for (nnz_t i = 0; i < nnz_l_remote; ++i) {
                entry.emplace_back(cooEntry(row_remote[i] + split[rank], col_remote2[i], val_remote[i]));
            }
        }

        std::sort(entry.begin(), entry.end());
    }

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
    inv_diag = saena_aligned_alloc<value_t>(M);
    fill(&inv_diag[0], &inv_diag[M], 0.0);
    inv_sq_diag.assign(M, 0.0);     // D^{-1/2}

    value_t *inv_diag_p    = &inv_diag[0] - split[rank];
    value_t *inv_sq_diag_p = &inv_sq_diag[0] - split[rank];

    if(!entry.empty()) {
        for (nnz_t i = 0; i < nnz_l; i++) {

            if (entry[i].row == entry[i].col) {
//                if(rank==0) std::cout << i << "\t" << entry[i] << std::endl;
                if ( !almost_zero(entry[i].val) ) {
                    temp = 1.0 / entry[i].val;
                    inv_diag_p[entry[i].row] = temp;
                    inv_sq_diag_p[entry[i].row] = sqrt(fabs(temp)); // TODO: should fabs be used here?
//                    if (fabs(temp) > highest_diag_val) {
//                        highest_diag_val = fabs(temp);
//                    }
                } else {
                    printf("Error on rank %d: there is a zero diagonal element at row index = %d\n", rank, entry[i].row);
                    MPI_Finalize();
                    exit(EXIT_FAILURE);
                }
            }
        }
    }

    for(int i = 0; i < M; ++i){
        if(almost_zero(inv_diag[i])){
            printf("rank %d: zero diagonal at row %d: %f\n", rank, i, inv_diag[i]);
            exit(EXIT_FAILURE);
        }
//        ASSERT(inv_diag[i] != 0, "rank " << rank << ": " << i << "\t" << inv_diag[i]);
    }

#ifdef __DEBUG1__
//    MPI_Barrier(comm);
//    print_vector(inv_diag, -1, "inv diag", comm);
//    print_vector(inv_sq_diag, -1, "inv_sq_diag diag", comm);
//    MPI_Barrier(comm);
#endif

//    for(auto i:inv_diag) {
//        if (i == 0){
//            printf("rank %d: inverse_diag: At least one diagonal entry is 0.\n", rank);
//        }
//    }

//    temp = highest_diag_val;
//    MPI_Allreduce(&temp, &highest_diag_val, 1, MPI_DOUBLE, MPI_MAX, comm);
//    if(rank==0) printf("\ninverse_diag: highest_diag_val = %f \n", highest_diag_val);

    return 0;
}

vector<index_t> saena_matrix::get_orig_split(){
    return split_b;
}

int saena_matrix::generate_dense_matrix() {
//    cout << "generate dense" << endl;
    use_dense = true;
    dense_matrix = new saena_matrix_dense;
    assert(dense_matrix);
    dense_matrix->convert_saena_matrix(this);
//    erase();
    return 0;
}