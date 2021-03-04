#include "saena_object.h"
#include "saena_matrix.h"
#include "saena_vector.h"


//int saena_object::set_repartition_rhs(std::vector<value_t> rhs0)
/*
int saena_object::set_repartition_rhs(std::vector<value_t> rhs0){

    MPI_Comm comm = grids[0].A->comm;
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    // ************** set variables **************

//    grids[0].rhs_orig = rhs1; // use this for returning solution to the original order, based on the input rhs.
//    rhs1->split = &grids[0].A->split[0];
//    std::vector<double> rhs0;
//    rhs1->get_vec(rhs0);

//    print_vector(grids[0].A->split, 0, "split", comm);
//    print_vector(rhs0, -1, "rhs0", comm);

    // ************** check rhs size **************

    index_t rhs_size_local = (index_t)rhs0.size(), rhs_size_total;
    MPI_Allreduce(&rhs_size_local, &rhs_size_total, 1, MPI_UNSIGNED, MPI_SUM, comm);
    if(grids[0].A->Mbig != rhs_size_total){
        if(rank==0) printf("Error: size of LHS (=%u) and RHS (=%u) are not equal!\n", grids[0].A->Mbig,rhs_size_total);
        MPI_Finalize();
        return -1;
    }

    // ************** repartition rhs, based on A.split **************

    std::vector<index_t> rhs_init_partition(nprocs);
    rhs_init_partition[rank] = (index_t)rhs0.size();
    auto temp = (index_t)rhs0.size();

    MPI_Allgather(&temp, 1, MPI_UNSIGNED, &*rhs_init_partition.begin(), 1, MPI_UNSIGNED, comm);
//    MPI_Alltoall(&*grids[0].rhs_init_partition.begin(), 1, MPI_INT, &*grids[0].rhs_init_partition.begin(), 1, MPI_INT, grids[0].comm);

//    print_vector(rhs_init_partition, 1, "rhs_init_partition", comm);

    std::vector<index_t> init_partition_scan(nprocs+1);
    init_partition_scan[0] = 0;
    for(int i = 1; i < nprocs+1; i++)
        init_partition_scan[i] = init_partition_scan[i-1] + rhs_init_partition[i-1];

//    print_vector(init_partition_scan, 1, "init_partition_scan", comm);

    index_t start, end, start_proc, end_proc;
    start = grids[0].A->split[rank];
    end   = grids[0].A->split[rank+1];
    start_proc = lower_bound2(&*init_partition_scan.begin(), &*init_partition_scan.end(), start);
    end_proc   = lower_bound2(&*init_partition_scan.begin(), &*init_partition_scan.end(), end);
    if(init_partition_scan[rank+1] == grids[0].A->split[rank+1])
        end_proc--;
//    if(rank == 1) printf("\nstart_proc = %u, end_proc = %u \n", start_proc, end_proc);

    grids[0].rcount.assign(nprocs, 0);
    if(start_proc < end_proc){
//        if(rank==1) printf("start_proc = %u, end_proc = %u\n", start_proc, end_proc);
//        if(rank==1) printf("init_partition_scan[start_proc+1] = %u, grids[0].A->split[rank] = %u\n", init_partition_scan[start_proc+1], grids[0].A->split[rank]);
        grids[0].rcount[start_proc] = init_partition_scan[start_proc+1] - grids[0].A->split[rank];
        grids[0].rcount[end_proc] = grids[0].A->split[rank+1] - init_partition_scan[end_proc];

        for(int i = start_proc+1; i < end_proc; i++){
//            if(rank==ran) printf("init_partition_scan[i+1] = %lu, init_partition_scan[i] = %lu\n", init_partition_scan[i+1], init_partition_scan[i]);
            grids[0].rcount[i] = init_partition_scan[i+1] - init_partition_scan[i];
        }
    }else if(start_proc == end_proc){
//        grids[0].rcount[start_proc] = grids[0].A->split[start_proc + 1] - grids[0].A->split[start_proc];
        grids[0].rcount[start_proc] = grids[0].A->split[rank + 1] - grids[0].A->split[rank];
    }else{
        printf("error in set_repartition_rhs function: start_proc > end_proc\n");
        MPI_Finalize();
        return -1;
    }

//    print_vector(grids[0].rcount, -1, "grids[0].rcount", comm);

    start = init_partition_scan[rank];
    end   = init_partition_scan[rank+1];
    start_proc = lower_bound2(&*grids[0].A->split.begin(), &*grids[0].A->split.end(), start);
    end_proc   = lower_bound2(&*grids[0].A->split.begin(), &*grids[0].A->split.end(), end);
    if(init_partition_scan[rank+1] == grids[0].A->split[rank+1])
        end_proc--;
//    if(rank == ran) printf("\nstart_proc = %lu, end_proc = %lu \n", start_proc, end_proc);

    grids[0].scount.assign(nprocs, 0);
    if(end_proc > start_proc){
//        if(rank==1) printf("start_proc = %u, end_proc = %u\n", start_proc, end_proc);
//        if(rank==1) printf("init_partition_scan[rank+1] = %u, grids[0].A->split[end_proc] = %u\n", init_partition_scan[rank+1], grids[0].A->split[end_proc]);
        grids[0].scount[start_proc] = grids[0].A->split[start_proc+1] - init_partition_scan[rank];
        grids[0].scount[end_proc] = init_partition_scan[rank+1] - grids[0].A->split[end_proc];

        for(int i = start_proc+1; i < end_proc; i++){
            grids[0].scount[i] = grids[0].A->split[i+1] - grids[0].A->split[i];
        }
    } else if(start_proc == end_proc)
        grids[0].scount[start_proc] = init_partition_scan[rank+1] - init_partition_scan[rank];
    else{
        printf("error in set_repartition_rhs function: start_proc > end_proc\n");
        MPI_Finalize();
        return -1;
    }

//    print_vector(grids[0].scount, -1, "grids[0].scount", comm);

//    std::vector<int> rdispls(nprocs);
    grids[0].rdispls.resize(nprocs);
    grids[0].rdispls[0] = 0;
    for(int i = 1; i < nprocs; i++)
        grids[0].rdispls[i] = grids[0].rcount[i-1] + grids[0].rdispls[i-1];

//    print_vector(grids[0].rdispls, -1, "grids[0].rdispls", comm);

//    std::vector<int> sdispls(nprocs);
    grids[0].sdispls.resize(nprocs);
    grids[0].sdispls[0] = 0;
    for(int i = 1; i < nprocs; i++)
        grids[0].sdispls[i] = grids[0].sdispls[i-1] + grids[0].scount[i-1];

//    print_vector(grids[0].sdispls, -1, "grids[0].sdispls", comm);

    // check if repartition is required. it is not required if the number of rows on all processors does not change.
    bool repartition_local = true;
    if(start_proc == end_proc)
        repartition_local = false;
    MPI_Allreduce(&repartition_local, &repartition, 1, MPI_CXX_BOOL, MPI_LOR, comm);
//    printf("rank = %d, repartition_local = %d, repartition = %d \n", rank, repartition_local, repartition);

    // todo: replace Alltoall with a for loop of send and recv.
    if(repartition){
        grids[0].rhs.resize(grids[0].A->split[rank+1] - grids[0].A->split[rank]);
        MPI_Alltoallv(&rhs0[0],         &grids[0].scount[0], &grids[0].sdispls[0], MPI_DOUBLE,
                      &grids[0].rhs[0], &grids[0].rcount[0], &grids[0].rdispls[0], MPI_DOUBLE, comm);
    } else{
        grids[0].rhs = rhs0;
    }

//    print_vector(grids[0].rhs, -1, "rhs after repartition", comm);

    // scale rhs
    // ---------
    scale_vector(grids[0].rhs, grids[0].A->inv_sq_diag);

    return 0;
}
*/

int saena_object::set_repartition_rhs(saena_vector *rhs1){

    saena_matrix    *A   = grids[0].A;
    vector<value_t> &rhs = grids[0].rhs;

    MPI_Comm comm = A->comm;
    int rank = 0, nprocs = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    // ************** set variables **************

#ifdef __DEBUG1__
    if(verbose_set_rhs){MPI_Barrier(A->comm); if(!rank) printf("set_repartition_rhs: start\n"); MPI_Barrier(A->comm);}
#endif

    grids[0].rhs_orig = rhs1; // use this for returning solution to the original order, based on the input rhs.
//    rhs1->split = &grids[0].A->split[0];
    rhs1->split = std::move(A->split_b);

    rhs1->get_vec(rhs);

//    print_vector(rhs, -1, "rhs", comm);

    // ************** check rhs size **************

    index_t Mbig_l = rhs.size(), Mbig = 0;

    if(remove_boundary){
        std::vector<value_t> rhs_large;
        rhs_large.swap(rhs);

        // compute split for rhs_large to pass to repart_vector()
        // initial split. have equal number of rows on each proc (except probably at the last proc)
        // it will get updated later
        MPI_Allreduce(&Mbig_l, &Mbig, 1, par::Mpi_datatype<index_t>::value(), MPI_SUM, comm);
        index_t ofst = Mbig / nprocs;
        vector<index_t> split(nprocs + 1);
        for (int i = 0; i < nprocs; ++i) {
            split[i] = i * ofst;
        }
        split[nprocs] = Mbig;

//        print_vector(split, 0, "split", comm);
//        printf("rhs_large.size before repart = %ld\n", rhs_large.size());

        repart_vector(rhs_large, split, comm);

//        printf("rhs_large.size after repart = %ld\n", rhs_large.size());
//        print_vector(rhs_large, -1, "rhs_large after repart", comm);

        remove_boundary_rhs(rhs_large, rhs, rhs1->comm);

//        print_vector(rhs, -1, "rhs after remove_boundary_rhs", comm);
    }

    Mbig_l = rhs.size();
    MPI_Allreduce(&Mbig_l, &Mbig, 1, par::Mpi_datatype<index_t>::value(), MPI_SUM, comm);
    if(A->Mbig != Mbig){
        if(rank==0) printf("Error: size of LHS (=%u) and RHS (=%u) are not equal!\n", A->Mbig, Mbig);
        MPI_Abort(comm, 1);
    }

    repart_vector(rhs, A->split, comm);

//    print_vector(rhs, -1, "rhs after final repart", comm);

#if 0
    // ************** repartition rhs, based on A.split **************

#ifdef __DEBUG1__
    if(verbose_set_rhs){MPI_Barrier(A->comm); if(!rank) printf("set_repartition_rhs: step 1\n"); MPI_Barrier(A->comm);}
#endif

    std::vector<index_t> split_init(nprocs + 1);
    auto temp = (index_t) rhs0.size();
    MPI_Allgather(&temp, 1, par::Mpi_datatype<index_t>::value(), &split_init[1], 1, par::Mpi_datatype<index_t>::value(), comm);
//    MPI_Alltoall(&*grids[0].rhs_init_partition.begin(), 1, MPI_INT, &*grids[0].rhs_init_partition.begin(), 1, MPI_INT, grids[0].comm);

    split_init[0] = 0;
    for(int i = 1; i < nprocs+1; i++)
        split_init[i] += split_init[i - 1];

#ifdef __DEBUG1__
//    print_vector(split_init, 1, "split_init", comm);
    if(verbose_set_rhs){MPI_Barrier(A->comm); if(!rank) printf("set_repartition_rhs: step 2\n"); MPI_Barrier(A->comm);}
#endif

    index_t start = 0, end = 0, start_proc = 0, end_proc = 0;
    start = A->split[rank];
    end   = A->split[rank+1];
    start_proc = lower_bound2(&*split_init.begin(), &*split_init.end(), start);
    end_proc   = lower_bound2(&*split_init.begin(), &*split_init.end(), end);
    if(split_init[rank + 1] == A->split[rank + 1])
        end_proc--;
//    if(rank == 1) printf("\nstart_proc = %u, end_proc = %u \n", start_proc, end_proc);

    grids[0].rcount.assign(nprocs, 0);
    if(start_proc < end_proc){
//        if(rank==1) printf("start_proc = %u, end_proc = %u\n", start_proc, end_proc);
//        if(rank==1) printf("split_init[start_proc+1] = %u, A->split[rank] = %u\n", split_init[start_proc+1], A->split[rank]);
        grids[0].rcount[start_proc] = split_init[start_proc + 1] - A->split[rank];
        grids[0].rcount[end_proc] = A->split[rank+1] - split_init[end_proc];

        for(int i = start_proc+1; i < end_proc; i++){
//            if(rank==ran) printf("split_init[i+1] = %lu, split_init[i] = %lu\n",
//                                  split_init[i+1], split_init[i]);
            grids[0].rcount[i] = split_init[i + 1] - split_init[i];
        }

    }else if(start_proc == end_proc){
//        grids[0].rcount[start_proc] = A->split[start_proc + 1] - A->split[start_proc];
        grids[0].rcount[start_proc] = A->split[rank + 1] - A->split[rank];
    }else{
        printf("error in set_repartition_rhs function: start_proc > end_proc\n");
        MPI_Finalize();
        return -1;
    }

#ifdef __DEBUG1__
//    print_vector(grids[0].rcount, -1, "grids[0].rcount", comm);
    if(verbose_set_rhs){MPI_Barrier(A->comm); if(!rank) printf("set_repartition_rhs: step 3\n"); MPI_Barrier(A->comm);}
#endif

    start = split_init[rank];
    end   = split_init[rank + 1];
    start_proc = lower_bound2(&*A->split.begin(), &*A->split.end(), start);
    end_proc   = lower_bound2(&*A->split.begin(), &*A->split.end(), end);
    if(split_init[rank + 1] == A->split[rank + 1])
        end_proc--;

//    if(rank == ran) printf("\nstart_proc = %lu, end_proc = %lu \n", start_proc, end_proc);

    grids[0].scount.assign(nprocs, 0);
    if(end_proc > start_proc){
//        if(rank==1) printf("start_proc = %u, end_proc = %u\n", start_proc, end_proc);
//        if(rank==1) printf("split_init[rank+1] = %u, A->split[end_proc] = %u\n", split_init[rank+1], A->split[end_proc]);
        grids[0].scount[start_proc] = A->split[start_proc+1] - split_init[rank];
        grids[0].scount[end_proc] = split_init[rank + 1] - A->split[end_proc];

        for(int i = start_proc+1; i < end_proc; i++){
            grids[0].scount[i] = A->split[i+1] - A->split[i];
        }
    } else if(start_proc == end_proc)
        grids[0].scount[start_proc] = split_init[rank + 1] - split_init[rank];
    else{
        printf("error in set_repartition_rhs function: start_proc > end_proc\n");
        MPI_Finalize();
        return -1;
    }

#ifdef __DEBUG1__
//    print_vector(grids[0].scount, -1, "grids[0].scount", comm);
    if(verbose_set_rhs){MPI_Barrier(A->comm); if(!rank) printf("set_repartition_rhs: step 4\n"); MPI_Barrier(A->comm);}
#endif

    grids[0].rdispls.resize(nprocs);
    grids[0].rdispls[0] = 0;
    for(int i = 1; i < nprocs; i++)
        grids[0].rdispls[i] = grids[0].rcount[i-1] + grids[0].rdispls[i-1];

    grids[0].sdispls.resize(nprocs);
    grids[0].sdispls[0] = 0;
    for(int i = 1; i < nprocs; i++)
        grids[0].sdispls[i] = grids[0].sdispls[i-1] + grids[0].scount[i-1];

#ifdef __DEBUG1__
//    print_vector(grids[0].rdispls, -1, "grids[0].rdispls", comm);
//    print_vector(grids[0].sdispls, -1, "grids[0].sdispls", comm);
    if(verbose_set_rhs){MPI_Barrier(A->comm); if(!rank) printf("set_repartition_rhs: step 5\n"); MPI_Barrier(A->comm);}
#endif

    // check if repartition is required. it is not required if the number of rows on all processors does not change.
    bool repartition_local = true;
    if(start_proc == end_proc)
        repartition_local = false;
    MPI_Allreduce(&repartition_local, &repartition, 1, MPI_CXX_BOOL, MPI_LOR, comm);
//    printf("rank = %d, repartition_local = %d, repartition = %d \n", rank, repartition_local, repartition);

    // todo: replace Alltoall with a for loop of send and recv.
    if(repartition){
        rhs.resize(A->split[rank+1] - A->split[rank]);
        MPI_Alltoallv(&rhs0[0], &grids[0].scount[0], &grids[0].sdispls[0], par::Mpi_datatype<value_t>::value(),
                      &rhs[0],  &grids[0].rcount[0], &grids[0].rdispls[0], par::Mpi_datatype<value_t>::value(), comm);
    } else{
        std::swap(rhs, rhs0);
    }

#ifdef __DEBUG1__
//    print_vector(grids[0].rhs, -1, "rhs after repartition", comm);
    if(verbose_set_rhs){MPI_Barrier(A->comm); if(!rank) printf("set_repartition_rhs: step 6\n"); MPI_Barrier(A->comm);}
#endif
#endif

    // scale rhs
    // ---------
    if(scale){
        scale_vector(rhs, A->inv_sq_diag_orig);
    }

    // write rhs to a file
//    writeVectorToFile(rhs, "rhs", comm, true, A->split[rank]);

#ifdef __DEBUG1__
//    print_vector(grids[0].rhs, -1, "rhs after repartition", comm);
    if(verbose_set_rhs){MPI_Barrier(A->comm); if(!rank) printf("set_repartition_rhs: end\n"); MPI_Barrier(A->comm);}
#endif

    return 0;
}


int saena_object::repart_vector(vector<value_t> &v, vector<index_t> &split, MPI_Comm comm){
    // v: the vector to be repartitioned
    // split: the desired repartition

    int rank = 0, nprocs = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    if(nprocs == 1){
        return 0;
    }

    // ************** set variables **************

#ifdef __DEBUG1__
    if(verbose_repart_vec){MPI_Barrier(comm); if(!rank) printf("repart_vec: start\n"); MPI_Barrier(comm);}
#endif

//    print_vector(split, 0, "split", comm);
//    print_vector(v, -1, "v", comm);

    // ************** repartition v, based on A.split **************

#ifdef __DEBUG1__
    if(verbose_repart_vec){MPI_Barrier(comm); if(!rank) printf("repart_vec: step 1\n"); MPI_Barrier(comm);}
#endif

    std::vector<index_t> split_init(nprocs + 1);
    auto temp = (index_t) v.size();
    MPI_Allgather(&temp, 1, par::Mpi_datatype<index_t>::value(), &split_init[1], 1, par::Mpi_datatype<index_t>::value(), comm);

    split_init[0] = 0;
    for(int i = 1; i < nprocs+1; i++)
        split_init[i] += split_init[i - 1];

#ifdef __DEBUG1__
//    print_vector(split_init, 1, "split_init", comm);
    if(verbose_repart_vec){MPI_Barrier(comm); if(!rank) printf("repart_vec: step 2\n"); MPI_Barrier(comm);}
#endif

    index_t start = 0, end = 0, start_proc = 0, end_proc = 0;
    start = split[rank];
    end   = split[rank+1];
    start_proc = lower_bound2(&*split_init.begin(), &*split_init.end(), start);
    end_proc   = lower_bound2(&*split_init.begin(), &*split_init.end(), end);
    if(split_init[rank + 1] == split[rank + 1])
        end_proc--;
//    if(rank == 3) printf("\nstart_proc = %u, end_proc = %u \n", start_proc, end_proc);

    grids[0].rcount.assign(nprocs, 0);
    if(start_proc < end_proc){
//        if(rank==1) printf("start_proc = %u, end_proc = %u\n", start_proc, end_proc);
//        if(rank==1) printf("split_init[start_proc+1] = %u, split[rank] = %u\n", split_init[start_proc+1], split[rank]);
        grids[0].rcount[start_proc] = split_init[start_proc + 1] - split[rank];
        grids[0].rcount[end_proc]   = split[rank+1] - split_init[end_proc];

        for(int i = start_proc+1; i < end_proc; i++){
//            if(rank==ran) printf("split_init[i+1] = %lu, split_init[i] = %lu\n", split_init[i+1], split_init[i]);
            grids[0].rcount[i] = split_init[i + 1] - split_init[i];
        }

    }else if(start_proc == end_proc){
//        grids[0].rcount[start_proc] = split[start_proc + 1] - split[start_proc];
        grids[0].rcount[start_proc] = split[rank + 1] - split[rank];
    }else{
        printf("error in repart_vec function: start_proc > end_proc\n");
        MPI_Finalize();
        return -1;
    }

#ifdef __DEBUG1__
//    print_vector(grids[0].rcount, -1, "grids[0].rcount", comm);
//    print_vector(split_init, rank_v, "split_init", comm);
//    print_vector(split, rank_v, "split", comm);
    if(verbose_repart_vec){MPI_Barrier(comm); if(!rank) printf("repart_vec: step 3\n"); MPI_Barrier(comm);}
#endif

    start = split_init[rank];
    end   = split_init[rank + 1];
    start_proc = lower_bound2(&*split.begin(), &*split.end(), start);
    end_proc   = lower_bound2(&*split.begin(), &*split.end(), end);
//    if( (split_init[rank] != split_init[rank + 1]) && (split_init[rank + 1] == split[rank + 1]) )
    if( (split_init[rank] != split_init[rank + 1]) && (split_init[rank + 1] == split[rank + 1]) )
        end_proc--;

//    if(rank == rank_v) printf("\nstart_proc = %d, end_proc = %d, start = %d, end = %d\n", start_proc, end_proc, start, end);

    grids[0].scount.assign(nprocs, 0);
    if(end_proc > start_proc){
//        if(rank==1) printf("start_proc = %u, end_proc = %u\n", start_proc, end_proc);
//        if(rank==1) printf("split_init[rank+1] = %u, split[end_proc] = %u\n", split_init[rank+1], split[end_proc]);
        grids[0].scount[start_proc] = split[start_proc+1] - split_init[rank];
        grids[0].scount[end_proc] = split_init[rank + 1] - split[end_proc];

        for(int i = start_proc+1; i < end_proc; i++){
            grids[0].scount[i] = split[i+1] - split[i];
        }
    } else if(start_proc == end_proc)
        grids[0].scount[start_proc] = split_init[rank + 1] - split_init[rank];
    else{
        printf("error in repart_vec function: start_proc > end_proc\n");
        MPI_Abort(comm, 1);
        return -1;
    }

#ifdef __DEBUG1__
//    print_vector(grids[0].scount, -1, "grids[0].scount", comm);
    if(verbose_repart_vec){MPI_Barrier(comm); if(!rank) printf("repart_vec: step 4\n"); MPI_Barrier(comm);}
#endif

    grids[0].rdispls.resize(nprocs);
    grids[0].rdispls[0] = 0;
    for(int i = 1; i < nprocs; i++)
        grids[0].rdispls[i] = grids[0].rcount[i-1] + grids[0].rdispls[i-1];

    grids[0].sdispls.resize(nprocs);
    grids[0].sdispls[0] = 0;
    for(int i = 1; i < nprocs; i++)
        grids[0].sdispls[i] = grids[0].sdispls[i-1] + grids[0].scount[i-1];

#ifdef __DEBUG1__
//    print_vector(grids[0].rdispls, -1, "grids[0].rdispls", comm);
//    print_vector(grids[0].sdispls, -1, "grids[0].sdispls", comm);
    if(verbose_repart_vec){MPI_Barrier(comm); if(!rank) printf("repart_vec: step 5\n"); MPI_Barrier(comm);}
#endif

    // check if repartition is required. it is not required if the number of rows on all processors does not change.
    bool repartition_local = true;
    if(start_proc == end_proc)
        repartition_local = false;
    MPI_Allreduce(&repartition_local, &repartition, 1, MPI_CXX_BOOL, MPI_LOR, comm);
//    printf("rank = %d, repartition_local = %d, repartition = %d \n", rank, repartition_local, repartition);

//    print_vector(v, -1, "v inside repartition", comm);

    // todo: replace Alltoall with a for loop of send and recv.
    if(repartition){
        vector<value_t> v_tmp;
        v_tmp.swap(v);
        if(split[rank + 1] - split[rank] > 0)
            v.resize(split[rank + 1] - split[rank]);
        MPI_Alltoallv(&v_tmp[0], &grids[0].scount[0], &grids[0].sdispls[0], par::Mpi_datatype<value_t>::value(),
                      &v[0],     &grids[0].rcount[0], &grids[0].rdispls[0], par::Mpi_datatype<value_t>::value(), comm);
    }

#ifdef __DEBUG1__
//    print_vector(v, -1, "v after repartition", comm);
    if(verbose_repart_vec){MPI_Barrier(comm); if(!rank) printf("repart_vec: end\n"); MPI_Barrier(comm);}
#endif
    return 0;
}


bool saena_object::active(int l){
    return grids[l].A->active;
}

int saena_object::set_shrink_levels(std::vector<bool> sh_lev_vec) {
    shrink_level_vector = std::move(sh_lev_vec);
    return 0;
}

int saena_object::set_shrink_values(std::vector<int> sh_val_vec) {
    shrink_values_vector = std::move(sh_val_vec);
    return 0;
}

int saena_object::set_repart_thre(float thre) {
    repart_thre = thre;
    return 0;
}


int saena_object::repartition_u(std::vector<value_t>& u0){

    int rank = 0;
    MPI_Comm_rank(grids[0].A->comm, &rank);
//    MPI_Comm_size(grids[0].A->comm, &nprocs);

    // make a copy of u0 to be used in Alltoallv as sendbuf. u0 itself will be recvbuf there.
    std::vector<value_t> u_temp = u0;

    // ************** repartition u, based on A.split **************

    // todo: replace Alltoall with a for loop of send and recv.
    u0.resize(grids[0].A->split[rank+1] - grids[0].A->split[rank]);
    MPI_Alltoallv(&u_temp[0], &grids[0].scount[0], &grids[0].sdispls[0], par::Mpi_datatype<value_t>::value(),
                  &u0[0],     &grids[0].rcount[0], &grids[0].rdispls[0], par::Mpi_datatype<value_t>::value(), grids[0].A->comm);

    return 0;
}

int saena_object::repartition_back_u(std::vector<value_t>& u0){

    MPI_Comm comm = grids[0].A->comm;
    int rank = 0, nprocs = 0;
    MPI_Comm_rank(grids[0].A->comm, &rank);
    MPI_Comm_size(grids[0].A->comm, &nprocs);

//    print_vector(grids[0].A->split, 0, "split", comm);

    // rdispls should be the opposite of the initial repartition function. So, rdispls should be the scan of scount.
    // the same for sdispls.
    std::vector<int> rdispls(nprocs);
    rdispls[0] = 0;
    for(int i = 1; i < nprocs; i++)
        rdispls[i] = rdispls[i-1] + grids[0].scount[i-1];

//    print_vector(grids[0].scount, -1, "rec size", comm);
//    print_vector(rdispls, -1, "rdispls", grids[0].A->comm);

    std::vector<int> sdispls(nprocs);
    sdispls[0] = 0;
    for(int i = 1; i < nprocs; i++)
        sdispls[i] = sdispls[i-1] + grids[0].rcount[i-1];

//    print_vector(grids[0].rcount, -1, "send size", comm);
//    print_vector(sdispls, -1, "sdispls", grids[0].A->comm);

    index_t rhs_init_size = rdispls[nprocs-1] + grids[0].scount[nprocs-1]; // this is the summation over all rcount values on each proc.
//    printf("rank = %d, rhs_init_size = %lu \n", rank, rhs_init_size);

    // make a copy of u0 to be used in Alltoall as sendbuf. u0 itself will be recvbuf there.
    std::vector<value_t> u_temp = u0;
//    u0.clear();
    u0.resize(rhs_init_size);
//    std::fill(u0.begin(), u0.end(), -111);
//    print_vector(u_temp, 2, "u_temp", grids[0].A->comm);

    // todo: replace Alltoall with a for loop of send and recv.
    MPI_Alltoallv(&u_temp[0], &grids[0].rcount[0], &sdispls[0], par::Mpi_datatype<value_t>::value(),
                  &u0[0],     &grids[0].scount[0], &rdispls[0], par::Mpi_datatype<value_t>::value(), comm);

//    MPI_Barrier(grids[0].A->comm);
//    print_vector(u0, -1, "u after repartition_back_u", grids[0].A->comm);

    return 0;
}