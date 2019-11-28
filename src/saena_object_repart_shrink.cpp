#include "saena_object.h"
#include "saena_matrix.h"
#include "grid.h"
#include "aux_functions.h"
#include <parUtils.h>
#include "dollar.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mpi.h>
#include <saena_vector.h>


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

    MPI_Comm comm = grids[0].A->comm;
    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    // ************** set variables **************

    grids[0].rhs_orig = rhs1; // use this for returning solution to the original order, based on the input rhs.
//    rhs1->split = &grids[0].A->split[0];
    rhs1->split = grids[0].A->split;
    std::vector<double> rhs0;
    rhs1->get_vec(rhs0);

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


int saena_object::repartition_u(std::vector<value_t>& u0){

    int rank;
    MPI_Comm_rank(grids[0].A->comm, &rank);
//    MPI_Comm_size(grids[0].A->comm, &nprocs);

    // make a copy of u0 to be used in Alltoallv as sendbuf. u0 itself will be recvbuf there.
    std::vector<value_t> u_temp = u0;

    // ************** repartition u, based on A.split **************

    // todo: replace Alltoall with a for loop of send and recv.
    u0.resize(grids[0].A->split[rank+1] - grids[0].A->split[rank]);
    MPI_Alltoallv(&u_temp[0], &grids[0].scount[0], &grids[0].sdispls[0], MPI_DOUBLE,
                  &u0[0],     &grids[0].rcount[0], &grids[0].rdispls[0], MPI_DOUBLE, grids[0].A->comm);

    return 0;
}


int saena_object::repartition_back_u(std::vector<value_t>& u0){

    MPI_Comm comm = grids[0].A->comm;
    int rank, nprocs;
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
    MPI_Alltoallv(&u_temp[0], &grids[0].rcount[0], &sdispls[0], MPI_DOUBLE,
                  &u0[0],     &grids[0].scount[0], &rdispls[0], MPI_DOUBLE, comm);

//    MPI_Barrier(grids[0].A->comm);
//    print_vector(u0, -1, "u after repartition_back_u", grids[0].A->comm);

    return 0;
}


int saena_object::shrink_cpu_A(saena_matrix* Ac, std::vector<index_t>& P_splitNew){

    // if number of rows on Ac < threshold*number of rows on A, then shrink.
    // redistribute Ac from processes 4k+1, 4k+2 and 4k+3 to process 4k.
    MPI_Comm comm = Ac->comm;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    index_t i;
    bool verbose_shrink = false;

//    MPI_Barrier(comm);
//    if(rank==0) printf("\n****************************\n");
//    if(rank==0) printf("***********SHRINK***********\n");
//    if(rank==0) printf("****************************\n\n");
//    MPI_Barrier(comm);

//    MPI_Barrier(comm); printf("rank = %d \tnnz_l = %u \n", rank, Ac->nnz_l); MPI_Barrier(comm);

//    MPI_Barrier(comm);
//    if(rank == 0){
//        std::cout << "\nbefore shrinking!!!" <<std::endl;
//        std::cout << "\nrank = " << rank << ", size = " << Ac->entry.size() <<std::endl;
//        for(i=0; i < Ac->entry.size(); i++)
//            std::cout << i << "\t" << Ac->entry[i]  <<std::endl;
//    }

    // assume cpu_shrink_thre2 is 4 (it is simpler to explain)
    // 1 - create a new comm, consisting only of processes 4k, 4k+1, 4k+2 and 4k+3 (with new ranks 0,1,2,3)
    int color = rank / Ac->cpu_shrink_thre2;
    MPI_Comm_split(comm, color, rank, &Ac->comm_horizontal);

    int rank_new, nprocs_new;
    MPI_Comm_size(Ac->comm_horizontal, &nprocs_new);
    MPI_Comm_rank(Ac->comm_horizontal, &rank_new);

//    MPI_Barrier(Ac->comm_horizontal);
//    printf("rank = %d, rank_new = %d on Ac->comm_horizontal \n", rank, rank_new);

    // 2 - update the number of rows on process 4k, and resize "entry".
    index_t Ac_M_neighbors_total = 0;
    unsigned int Ac_nnz_neighbors_total = 0;
    MPI_Reduce(&Ac->M, &Ac_M_neighbors_total, 1, MPI_UNSIGNED, MPI_SUM, 0, Ac->comm_horizontal);
    MPI_Reduce(&Ac->nnz_l, &Ac_nnz_neighbors_total, 1, MPI_UNSIGNED, MPI_SUM, 0, Ac->comm_horizontal);

    if(rank_new == 0){
        Ac->M = Ac_M_neighbors_total;
        Ac->entry.resize(Ac_nnz_neighbors_total);
//        printf("rank = %d, Ac_M_neighbors = %d \n", rank, Ac_M_neighbors_total);
//        printf("rank = %d, Ac_nnz_neighbors = %d \n", rank, Ac_nnz_neighbors_total);
    }

    // last cpu that its right neighbors are going be shrinked to.
    auto last_root_cpu = (unsigned int)floor(nprocs/Ac->cpu_shrink_thre2) * Ac->cpu_shrink_thre2;
//    printf("last_root_cpu = %u\n", last_root_cpu);

    int neigbor_rank;
    nnz_t A_recv_nnz = 0; // set to 0 just to avoid "not initialized" warning
    unsigned long offset = Ac->nnz_l; // put the data on root from its neighbors at the end of entry[] which is of size nnz_l
    if(nprocs_new > 1) { // if there is no neighbor, skip.
        for (neigbor_rank = 1; neigbor_rank < Ac->cpu_shrink_thre2; neigbor_rank++) {

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
                MPI_Recv(&A_recv_nnz, 1, MPI_UNSIGNED_LONG, neigbor_rank, 0, Ac->comm_horizontal, MPI_STATUS_IGNORE);

            if (rank_new == neigbor_rank)
                MPI_Send(&Ac->nnz_l, 1, MPI_UNSIGNED_LONG, 0, 0, Ac->comm_horizontal);

            // 4 - send and receive Ac.
            if (rank_new == 0) {
//                printf("rank = %d, neigbor_rank = %d, recv size = %u, offset = %lu \n", rank, neigbor_rank, A_recv_nnz, offset);
                MPI_Recv(&*(Ac->entry.begin() + offset), A_recv_nnz, cooEntry::mpi_datatype(), neigbor_rank, 1,
                         Ac->comm_horizontal, MPI_STATUS_IGNORE);
                offset += A_recv_nnz; // set offset for the next iteration
            }

            if (rank_new == neigbor_rank) {
//                printf("rank = %d, neigbor_rank = %d, send size = %u, offset = %lu \n", rank, neigbor_rank, Ac->nnz_l, offset);
                MPI_Send(&*Ac->entry.begin(), Ac->nnz_l, cooEntry::mpi_datatype(), 0, 1, Ac->comm_horizontal);
            }

            // update local index for rows
//        if(rank_new == 0)
//            for(i=0; i < A_recv_nnz; i++)
//                Ac->entry[offset + i].row += Ac->split[rank + neigbor_rank] - Ac->split[rank];
        }
    }

    // even though the entries are sorted before shrinking, after shrinking they still need to be sorted locally,
    // because remote elements damage sorting after shrinking.
    std::sort(Ac->entry.begin(), Ac->entry.end());

//    MPI_Barrier(comm); MPI_Barrier(Ac->comm_horizontal);
//    if(rank == 2){
//        std::cout << "\nafter shrinking: rank = " << rank << ", size = " << Ac->entry.size() <<std::endl;
//        for(i=0; i < Ac->entry.size(); i++)
//            std::cout << i << "\t" << Ac->entry[i]  <<std::endl;}
//    MPI_Barrier(comm); MPI_Barrier(Ac->comm_horizontal);

    Ac->active = false;
//    Ac->active_old_comm = true; // this is used for prolong and post-smooth
    if(rank_new == 0){
        Ac->active = true;
//        printf("active: rank = %d, rank_new = %d \n", rank, rank_new);
    }

    // 5 - update 4k.nnz_l and split. nnz_g stays the same, so no need to update.
/*
    if(Ac->active){
        Ac->nnz_l = Ac->entry.size();
        Ac->split_old = Ac->split; // save the old split for shrinking rhs and u
        Ac->split.clear();
        for(i = 0; i < nprocs+1; i++){
//            if(rank==0) printf("P->splitNew[i] = %lu\n", P_splitNew[i]);
            if( i % Ac->cpu_shrink_thre2 == 0){
                Ac->split.emplace_back( P_splitNew[i] );
            }
        }
        Ac->split.emplace_back( P_splitNew[nprocs] );
        // assert M == split[rank+1] - split[rank]

        print_vector(Ac->split, 0, "Ac->split after shrinking:", Ac->comm);

//        if(rank==0) {
//            printf("Ac split after shrinking: \n");
//            for (int i = 0; i < Ac->split.size(); i++)
//                printf("%lu \n", Ac->split[i]);}
    }
*/

    // 6 - create a new comm including only processes with 4k rank.
    MPI_Group bigger_group;
    MPI_Comm_group(comm, &bigger_group);
    auto total_active_procs = (unsigned int)ceil((double)nprocs / Ac->cpu_shrink_thre2); // note: this is ceiling, not floor.
    std::vector<int> ranks(total_active_procs);
    for(i = 0; i < total_active_procs; i++)
        ranks[i] = Ac->cpu_shrink_thre2 * i;

//    printf("total_active_procs = %u \n", total_active_procs);
//    print_vector(ranks, 0, "ranks", Ac->comm);

    MPI_Group group_new;
    MPI_Group_incl(bigger_group, total_active_procs, &*ranks.begin(), &group_new);
    MPI_Comm_create_group(comm, group_new, 0, &Ac->comm);

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

//    Ac->split_old = Ac->split; // save the old split for shrinking rhs and u
    std::vector<index_t> split_temp = Ac->split;
    Ac->split.clear();
    if(Ac->active){
        Ac->nnz_l = Ac->entry.size();

        Ac->split.resize(total_active_procs+1);
        Ac->split.shrink_to_fit();
        Ac->split[0] = 0;
        Ac->split[total_active_procs] = Ac->Mbig;
        for(unsigned int i = 1; i < total_active_procs; i++){
//            if(rank==0) printf("%u \t%lu \n", i, split_old[ranks[i]]);
            Ac->split[i] = split_temp[ranks[i]];
        }
//        print_vector(Ac->split, 0, "Ac->split after shrinking:", comm);
    }

    // 7 - update 4k.nnz_g
//    if(Ac->active)
//        MPI_Allreduce(&Ac->nnz_l, &Ac->nnz_g, 1, MPI_UNSIGNED, MPI_SUM, Ac->comm);

//    if(Ac->active){
//        MPI_Comm_size(Ac->comm, &nprocs);
//        MPI_Comm_rank(Ac->comm, &rank);
//        printf("\n\nrank = %d, nprocs = %d, M = %u, nnz_l = %u, nnz_g = %u, Ac->split[rank+1] = %lu, Ac->split[rank] = %lu \n",
//               rank, nprocs, Ac->M, Ac->nnz_l, Ac->nnz_g, Ac->split[rank+1], Ac->split[rank]);
//    }

    // todo: how should these be freed?
//    free(&bigger_group);
//    free(&group_new);
//    free(&comm_new2);

    Ac->last_M_shrink = Ac->Mbig;
    Ac->shrinked = true;

//    MPI_Barrier(comm); if(rank==0) printf("shrinking done!\n"); MPI_Barrier(comm);

    return 0;
}


// int saena_object::repartition_u_shrink_prepare
/*
int saena_object::repartition_u_shrink_prepare(std::vector<value_t> &u, Grid &grid){

    MPI_Comm comm = grid.A->comm;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    MPI_Barrier(comm);
//    if(rank == 0) printf("\nsplit_old: \n");
//    print_vector(grid.A->split_old, 0, comm);
//    MPI_Barrier(comm);
//    if(rank == 0) printf("\nsplit: \n");
//    print_vector(grid.A->split, 0, comm);
//    MPI_Barrier(comm);

    std::vector<int> scount(nprocs, 0);

    long least_proc;
    least_proc = lower_bound2(&grid.A->split[0], &grid.A->split[nprocs], 0 + grid.A->split_old[rank]);
    scount[least_proc]++;
//    printf("rank %d: 0 + A.split_old[rank] = %u, least_proc = %ld \n", rank, 0 + A.split_old[rank], least_proc);

    long curr_proc = least_proc;
    for(index_t i = 1; i < u.size(); i++){
        if(i + grid.A->split_old[rank] >= grid.A->split[curr_proc+1])
            curr_proc++; //todo: if shrinked==true then curr_proc += A.shrink_thre2
        scount[curr_proc]++;
//        if(rank==0) printf("i + A.split_old[rank] = %u, curr_proc = %ld \n", i + A.split_old[rank], curr_proc);
    }

//    print_vector(send_size_array, -1, comm);

    std::vector<int> rcount(nprocs);
    MPI_Alltoall(&scount[0], 1, MPI_INT, &rcount[0], 1, MPI_INT, comm);
//    print_vector(recv_size_array, -1, comm);

    std::vector<int> send_offset(nprocs);
    send_offset[0] = 0;
    for (int i=1; i<nprocs; i++)
        send_offset[i] = scount[i-1] + send_offset[i-1];

//    print_vector(send_offset, -1, comm);

    std::vector<int> recv_offset(nprocs);
    recv_offset[0] = 0;
    for (int i=1; i<nprocs; i++)
        recv_offset[i] = rcount[i-1] + recv_offset[i-1];

//    print_vector(recv_offset, 2, comm);

    std::vector<value_t> u_old = u;
    u.resize(grid.A->M);
    MPI_Alltoallv(&u_old[0], &scount[0], &send_offset[0], MPI_DOUBLE,
                  &u[0],     &rcount[0], &recv_offset[0], MPI_DOUBLE, comm);

    return 0;
}
*/


// new version of int saena_object::repartition_u_shrink_prepare(Grid *grid)
// it only consider repartition when shrinking has happened.
/*
int saena_object::repartition_u_shrink_prepare(Grid *grid){

    MPI_Comm comm = grid->Ac.comm;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // Note: A is grid->Ac!
    // --------------------
    saena_matrix *A = &grid->Ac;

//    print_vector(A->split_old, 0, "split_old", comm);
//    print_vector(A->split, 0, "split", comm);
//    printf("rank %d: A->M = %u, A->M_old = %u \n", rank, A->M, A->M_old);
//    MPI_Barrier(comm);

    grid->scount2.assign(nprocs, 0);

    long least_proc;
    least_proc = lower_bound3(&A->split[0], &A->split[nprocs], 0 + A->split_old[rank]);
    grid->scount2[least_proc]++;
//    printf("rank %d: 0 + A.split_old[rank] = %u, least_proc = %ld \n", rank, 0 + A->split_old[rank], least_proc);

    long curr_proc = least_proc;
    for(index_t i = 1; i < A->M_old; i++){
        if(i + A->split_old[rank] >= A->split[curr_proc+1]){
//            if(A->shrinked)
                curr_proc += A->cpu_shrink_thre2;
//            else
//                curr_proc++;
        }
        grid->scount2[curr_proc]++;
//        if(rank==2) printf("i + A.split_old[rank] = %u, curr_proc = %ld \n", i + A->split_old[rank], curr_proc);
    }

//    print_vector(grid->scount2, -1, comm);

    // instead of only resizing rcount2, it is put equal to scount2 in case of nprocs = 1.
    grid->rcount2 = grid->scount2;
    if(nprocs > 1)
        MPI_Alltoall(&grid->scount2[0], 1, MPI_INT, &grid->rcount2[0], 1, MPI_INT, comm);

//    print_vector(grid->rcount2, -1, comm);

//    std::vector<int> sdispls2(nprocs);
    grid->sdispls2.resize(nprocs);
    grid->sdispls2[0] = 0;
    for (int i=1; i<nprocs; i++)
        grid->sdispls2[i] = grid->scount2[i-1] + grid->sdispls2[i-1];

//    print_vector(grid->sdispls2, -1, comm);

//    std::vector<int> rdispls2(nprocs);
    grid->rdispls2.resize(nprocs);
    grid->rdispls2[0] = 0;
    for (int i=1; i<nprocs; i++)
        grid->rdispls2[i] = grid->rcount2[i-1] + grid->rdispls2[i-1];

//    print_vector(grid->rdispls2, -1, comm);

    return 0;
}
*/


int saena_object::repartition_u_shrink_prepare(Grid *grid){

//    MPI_Comm comm = grid->A->comm;
    MPI_Comm comm = grid->Ac.comm;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // Note: A is grid->Ac!
    // --------------------
    saena_matrix *A = &grid->Ac;

//    print_vector(A->split_old, 0, "split_old", comm);
//    print_vector(A->split, 0, "split", comm);
//    MPI_Barrier(comm); printf("rank %d: A->M = %u, A->M_old = %u \n", rank, A->M, A->M_old); MPI_Barrier(comm);

    grid->scount2.assign(nprocs, 0);

    long least_proc = 0, curr_proc;
    if(A->M_old != 0){
        least_proc = lower_bound3(&A->split[0], &A->split[nprocs], 0 + A->split_old[rank]);
        grid->scount2[least_proc]++;
//    printf("rank %d: 0 + A.split_old[rank] = %u, least_proc = %ld \n", rank, 0 + A->split_old[rank], least_proc);

        curr_proc = least_proc;
        for(index_t i = 1; i < A->M_old; i++){
            if(i + A->split_old[rank] >= A->split[curr_proc+1]){
                if(A->shrinked)
                    curr_proc += A->cpu_shrink_thre2;
                else
                    curr_proc++;
            }
            grid->scount2[curr_proc]++;
            //        if(rank==2) printf("i + A.split_old[rank] = %u, curr_proc = %ld \n", i + A->split_old[rank], curr_proc);
        }
    }

//    print_vector(grid->scount2, -1, "scount2", comm);

    // instead of only resizing rcount2, it is put equal to scount2 in case of nprocs = 1.
    grid->rcount2 = grid->scount2;
    if(nprocs > 1)
        MPI_Alltoall(&grid->scount2[0], 1, MPI_INT, &grid->rcount2[0], 1, MPI_INT, comm);

//    print_vector(grid->rcount2, -1, "rcount2", comm);

//    std::vector<int> sdispls2(nprocs);
    grid->sdispls2.resize(nprocs);
    grid->sdispls2[0] = 0;
    for (int i=1; i<nprocs; i++)
        grid->sdispls2[i] = grid->scount2[i-1] + grid->sdispls2[i-1];

//    print_vector(grid->sdispls2, "sdispls2, -1, comm);

//    std::vector<int> rdispls2(nprocs);
    grid->rdispls2.resize(nprocs);
    grid->rdispls2[0] = 0;
    for (int i=1; i<nprocs; i++)
        grid->rdispls2[i] = grid->rcount2[i-1] + grid->rdispls2[i-1];

//    print_vector(grid->rdispls2, -1, "rdispls2, comm);

    return 0;
}


int saena_object::repartition_u_shrink_coarsest_prepare(Grid *grid){
    // compute: scount2, rcount2, sdispls2, rdispls2

//    MPI_Comm comm = grid->A->comm;
    MPI_Comm comm = grid->Ac.comm;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // Note: A is grid->Ac!
    // --------------------
    saena_matrix *A = &grid->Ac;

//    print_vector(A->split_old, 0, "split_old", comm);
//    print_vector(A->split, 0, "split", comm);
//    MPI_Barrier(comm); printf("rank %d: A->M = %u, A->M_old = %u \n", rank, A->M, A->M_old); MPI_Barrier(comm);

    grid->scount2.assign(nprocs, 0);

    long least_proc = 0, curr_proc;
    if(A->M_old != 0){
        least_proc = lower_bound3(&A->split[0], &A->split[nprocs], 0 + A->split_old[rank]);
        grid->scount2[least_proc]++;
//        printf("rank %d: 0 + A.split_old[rank] = %u, least_proc = %ld \n", rank, 0 + A->split_old[rank], least_proc);

        curr_proc = least_proc;
        for(index_t i = 1; i < A->M_old; i++){
            if(i + A->split_old[rank] >= A->split[curr_proc+1]){
                if(A->shrinked)
                    curr_proc += A->cpu_shrink_thre2;
                else
                    curr_proc++;
            }
            grid->scount2[curr_proc]++;
//            if(rank==2) printf("i + A.split_old[rank] = %u, curr_proc = %ld \n", i + A->split_old[rank], curr_proc);
        }
    }

//    print_vector(grid->scount2, -1, "scount2", comm);

    // instead of only resizing rcount2, it is put equal to scount2 in case of nprocs = 1.
    grid->rcount2 = grid->scount2;
    if(nprocs > 1)
        MPI_Alltoall(&grid->scount2[0], 1, MPI_INT, &grid->rcount2[0], 1, MPI_INT, comm);

//    print_vector(grid->rcount2, -1, "rcount2", comm);

//    std::vector<int> sdispls2(nprocs);
    grid->sdispls2.resize(nprocs);
    grid->sdispls2[0] = 0;
    for (int i=1; i<nprocs; i++)
        grid->sdispls2[i] = grid->scount2[i-1] + grid->sdispls2[i-1];

//    print_vector(grid->sdispls2, "sdispls2, -1, comm);

//    std::vector<int> rdispls2(nprocs);
    grid->rdispls2.resize(nprocs);
    grid->rdispls2[0] = 0;
    for (int i=1; i<nprocs; i++)
        grid->rdispls2[i] = grid->rcount2[i-1] + grid->rdispls2[i-1];

//    print_vector(grid->rdispls2, -1, "rdispls2, comm);

    return 0;
}


int saena_object::repartition_u_shrink(std::vector<value_t> &u, Grid &grid){

    MPI_Comm comm = grid.Ac.comm;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    MPI_Barrier(grid.A->comm);
//    printf("rank %d: A->M = %u, A->M_old = %u \n", rank, grid.Ac.M, grid.Ac.M_old);
//    MPI_Barrier(grid.A->comm);
//    print_vector(u, 1, "u inside repartition_u_shrink", comm);

    std::vector<value_t> u_old = u;
    u.resize(grid.Ac.M);
    MPI_Alltoallv(&u_old[0], &grid.scount2[0], &grid.sdispls2[0], MPI_DOUBLE,
                  &u[0],     &grid.rcount2[0], &grid.rdispls2[0], MPI_DOUBLE, comm);

    return 0;
}


int saena_object::repartition_back_u_shrink(std::vector<value_t> &u, Grid &grid){

    MPI_Comm comm = grid.Ac.comm;
//    MPI_Comm comm = grid.A->comm;
//    int rank, nprocs;
//    MPI_Comm_size(comm, &nprocs);
//    MPI_Comm_rank(comm, &rank);

    std::vector<value_t> u_old = u;
    u.resize(grid.Ac.M_old);
    MPI_Alltoallv(&u_old[0], &grid.rcount2[0], &grid.rdispls2[0], MPI_DOUBLE,
                  &u[0],     &grid.scount2[0], &grid.sdispls2[0], MPI_DOUBLE, comm);

    return 0;
}


//int saena_object::repartition_u_shrink_minor_prepare(Grid *grid)
/*
int saena_object::repartition_u_shrink_minor_prepare(Grid *grid){

    MPI_Comm comm = grid->Ac.comm;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // Note: A is grid->Ac!
    // --------------------
    saena_matrix *A = &grid->Ac;

//    print_vector(A->split_old, 0, "split_old", comm);
//    print_vector(A->split, 0, "split", comm);
//    printf("rank %d: A->M = %u, A->M_old = %u \n", rank, A->M, A->M_old);
//    MPI_Barrier(comm);

    grid->scount3.assign(nprocs, 0);

    long least_proc;
    least_proc = lower_bound3(&A->split[0], &A->split[nprocs], 0 + A->split_old_minor[rank]);
    grid->scount3[least_proc]++;
//    printf("rank %d: 0 + A.split_old[rank] = %u, least_proc = %ld \n", rank, 0 + A->split_old[rank], least_proc);

    long curr_proc = least_proc;
    for(index_t i = 1; i < A->M; i++){
        while(i + A->split_old_minor[rank] >= A->split[curr_proc+1]){
             curr_proc++;
        }
        grid->scount3[curr_proc]++;
//        if(rank==2) printf("i + A.split_old[rank] = %u, curr_proc = %ld \n", i + A->split_old[rank], curr_proc);
    }

//    print_vector(grid->scount3, -1, comm);

    // instead of only resizing rcount2, it is put equal to scount2 in case of nprocs = 1.
    grid->rcount3 = grid->scount3;
    if(nprocs > 1)
        MPI_Alltoall(&grid->scount3[0], 1, MPI_INT, &grid->rcount3[0], 1, MPI_INT, comm);

//    print_vector(grid->rcount3, -1, comm);

//    std::vector<int> sdispls3(nprocs);
    grid->sdispls3.resize(nprocs);
    grid->sdispls3[0] = 0;
    for (int i=1; i<nprocs; i++)
        grid->sdispls3[i] = grid->scount3[i-1] + grid->sdispls3[i-1];

//    print_vector(grid->sdispls3, -1, comm);

//    std::vector<int> rdispls2(nprocs);
    grid->rdispls3.resize(nprocs);
    grid->rdispls3[0] = 0;
    for (int i=1; i<nprocs; i++)
        grid->rdispls3[i] = grid->rcount3[i-1] + grid->rdispls3[i-1];

//    print_vector(grid->rdispls3, -1, comm);

    return 0;
}
*/


//int saena_object::repartition_u_shrink_minor(std::vector<value_t> &u, Grid &grid)
/*
int saena_object::repartition_u_shrink_minor(std::vector<value_t> &u, Grid &grid){

    MPI_Comm comm = grid.Ac.comm;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    MPI_Barrier(grid.A->comm);
//    printf("rank %d: A->M = %u, A->M_old = %u \n", rank, grid.Ac.M, grid.Ac.M_old);
//    MPI_Barrier(grid.A->comm);

    std::vector<value_t> u_old = u;
    u.resize(grid.Ac.M);
    MPI_Alltoallv(&u_old[0], &grid.scount3[0], &grid.sdispls3[0], MPI_DOUBLE,
                  &u[0],     &grid.rcount3[0], &grid.rdispls3[0], MPI_DOUBLE, comm);

    return 0;
}
*/


// int saena_object::repartition_back_u_shrink
/*
int saena_object::repartition_back_u_shrink(std::vector<value_t> &u, saena_matrix &A){

    MPI_Comm comm = A.comm; //todo: after shrinking check if it is still true
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    MPI_Barrier(comm);
//    if(rank == 0) printf("\nsplit_old: \n");
//    print_vector(A.split_old, 0, comm);
//    MPI_Barrier(comm);
//    if(rank == 0) printf("\nsplit: \n");
//    print_vector(A.split, 0, comm);
//    MPI_Barrier(comm);

    std::vector<int> send_size_array(nprocs, 0);

    long least_proc;
    least_proc = lower_bound2(&A.split_old[0], &A.split_old[nprocs], 0 + A.split[rank]);
    send_size_array[least_proc]++;
//    printf("rank %d: 0 + A.split[rank] = %u, least_proc = %ld \n", rank, 0 + A.split[rank], least_proc);

    long curr_proc = least_proc;
    for(index_t i = 1; i < u.size(); i++){
        if(i + A.split[rank] >= A.split_old[curr_proc+1])
            curr_proc++; //todo: if shrinked==true then curr_proc += A.shrink_thre2
        send_size_array[curr_proc]++;
//        if(rank==0) printf("i + A.split[rank] = %u, curr_proc = %ld \n", i + A.split[rank], curr_proc);
    }

//    print_vector(send_size_array, -1, comm);

    std::vector<int> recv_size_array(nprocs);
    MPI_Alltoall(&send_size_array[0], 1, MPI_INT, &recv_size_array[0], 1, MPI_INT, comm);

//    print_vector(recv_size_array, -1, comm);

    std::vector<int> send_offset(nprocs);
    send_offset[0] = 0;
    for (int i=1; i<nprocs; i++)
        send_offset[i] = send_size_array[i-1] + send_offset[i-1];

//    print_vector(send_offset, -1, comm);

    std::vector<int> recv_offset(nprocs);
    recv_offset[0] = 0;
    for (int i=1; i<nprocs; i++)
        recv_offset[i] = recv_size_array[i-1] + recv_offset[i-1];

//    print_vector(recv_offset, 2, comm);

    std::vector<value_t> u_old = u;
    u.resize(A.M_old);
    MPI_Alltoallv(&u_old[0], &send_size_array[0], &send_offset[0], MPI_DOUBLE,
                  &u[0],     &recv_size_array[0], &recv_offset[0], MPI_DOUBLE, comm);

    return 0;
}
*/


int saena_object::shrink_u_rhs(Grid* grid, std::vector<value_t>& u, std::vector<value_t>& rhs){

    int rank, nprocs;
    MPI_Comm_size(grid->A->comm, &nprocs);
    MPI_Comm_rank(grid->A->comm, &rank);

    int rank_horizontal = -1;
    int nprocs_horizontal = -1;
//    if(grid->A->active){
    MPI_Comm_size(grid->Ac.comm_horizontal, &nprocs_horizontal);
    MPI_Comm_rank(grid->Ac.comm_horizontal, &rank_horizontal);
//    }

//    if(rank==0) {
//        printf("\nbefore shrinking: rank = %d, level = %d, rhs.size = %lu \n", rank, grid->currentLevel,
//               rhs.size());
//        for(unsigned long i=0; i<rhs.size(); i++)
//            printf("rhs[%lu] = %f \n", i, rhs[i]);}
//    MPI_Barrier(grid->A->comm);

    unsigned long offset = 0;
    if(grid->Ac.active){
        offset = grid->Ac.split_old[rank + 1] - grid->Ac.split_old[rank];
        u.assign(grid->Ac.M,0); // u is just zero, so no communication is required.
        rhs.resize(grid->Ac.M); // it is already of size grid->A->M. this is redundant.
    }

    // last cpu that its right neighbors are going be shrinked to.
    auto last_root_cpu = (unsigned int)floor(nprocs/grid->A->cpu_shrink_thre2) * grid->A->cpu_shrink_thre2;
//    printf("last_root_cpu = %u\n", last_root_cpu);

    int neigbor_rank;
    index_t recv_size = 0;
//    index_t send_size = rhs.size();
    for(neigbor_rank = 1; neigbor_rank < grid->A->cpu_shrink_thre2; neigbor_rank++){

        if( rank == last_root_cpu && (rank + neigbor_rank >= nprocs) )
            break;

//        if(rank_horizontal == 0 && (rank + neigbor_rank >= nprocs) )
//            break;

        // send and receive size of rhs.
//        if(rank_horizontal == 0)
//            MPI_Recv(&recv_size, 1, MPI_UNSIGNED, neigbor_rank, 0, comm_new, MPI_STATUS_IGNORE);
//        if(rank_horizontal == neigbor_rank)
//            MPI_Send(&local_size, 1, MPI_UNSIGNED, 0, 0, comm_new);

        // send and receive rhs and u.

        if(rank_horizontal == 0){
//            printf("rank = %d, grid->A->split_old[rank + neigbor_rank + 1] = %lu, grid->A->split_old[rank + neigbor_rank] = %lu \n",
//                     rank, grid->A->split_old[rank + neigbor_rank + 1], grid->A->split_old[rank + neigbor_rank]);

            recv_size = grid->Ac.split_old[rank + neigbor_rank + 1] - grid->Ac.split_old[rank + neigbor_rank];
//            printf("rank = %d, neigbor_rank = %d, recv_size = %u, offset = %lu \n", rank, neigbor_rank, recv_size, offset);
//            MPI_Recv(&*(u.begin() + offset),   recv_size, MPI_DOUBLE, neigbor_rank, 0, grid->Ac.comm_horizontal, MPI_STATUS_IGNORE);
            MPI_Recv(&*(rhs.begin() + offset), recv_size, MPI_DOUBLE, neigbor_rank, 1, grid->Ac.comm_horizontal, MPI_STATUS_IGNORE);
            offset += recv_size; // set offset for the next iteration
        }

        if(rank_horizontal == neigbor_rank){
//            printf("rank = %d, neigbor_rank = %d, local_size = %lu \n", rank, neigbor_rank, rhs.size());
//            MPI_Send(&*u.begin(),   send_size, MPI_DOUBLE, 0, 0, grid->Ac.comm_horizontal);
            MPI_Send(&*rhs.begin(), rhs.size(), MPI_DOUBLE, 0, 1, grid->Ac.comm_horizontal);
        }
    }

//    MPI_Barrier(grid->Ac.comm_horizontal);
//    if(rank==0){
//        printf("\nafter shrinking: rank = %d, level = %d, rhs.size = %lu \n", rank, grid->currentLevel, rhs.size());
//        for(unsigned long i=0; i<rhs.size(); i++)
//            printf("rhs[%lu] = %f \n", i, rhs[i]);}
//    MPI_Barrier(grid->Ac.comm_horizontal);
//    if(rank==1){
//        printf("\nafter shrinking: rank = %d, level = %d, rhs.size = %lu \n", rank, grid->currentLevel, rhs.size());
//        for(unsigned long i=0; i<rhs.size(); i++)
//            printf("rhs[%lu] = %f \n", i, rhs[i]);}
//    MPI_Barrier(grid->Ac.comm_horizontal);

    return 0;
}


int saena_object::unshrink_u(Grid* grid, std::vector<value_t>& u) {

    int rank, nprocs;
    MPI_Comm_size(grid->A->comm, &nprocs);
    MPI_Comm_rank(grid->A->comm, &rank);

//    int nprocs_horizontal = -1;
//    MPI_Comm_size(grid->Ac.comm_horizontal, &nprocs_horizontal);
    int rank_horizontal = -1;
    MPI_Comm_rank(grid->Ac.comm_horizontal, &rank_horizontal);

//    MPI_Barrier(grid->A->comm);
//    if(rank==0) {
//        printf("\nbefore un-shrinking: rank = %d, level = %d, u.size = %lu \n", rank, grid->currentLevel,
//               u.size());
//        for(unsigned long i=0; i<u.size(); i++)
//            printf("u[%lu] = %f \n", i, u[i]);}
//    MPI_Barrier(grid->A->comm);

//    if(rank==1){
//        printf("\nbefore un-shrinking: rank_new = %d, level = %d, u.size = %lu \n", rank, grid->currentLevel,
//               u.size());
//        for(unsigned long i=0; i<u.size(); i++)
//            printf("u[%lu] = %f \n", i, u[i]);}

    unsigned long offset = 0;
    // initialize offset to the size of u on the sender processor
    if(grid->Ac.active)
        offset = grid->Ac.split_old[rank + 1] - grid->Ac.split_old[rank];

    // last cpu that its right neighbors are going be shrinked to.
    auto last_root_cpu = (unsigned int)floor(nprocs/grid->A->cpu_shrink_thre2) * grid->A->cpu_shrink_thre2;
//    printf("last_root_cpu = %u\n", last_root_cpu);


    int neighbor_size = (nprocs % grid->A->cpu_shrink_thre2) - 1;
    int requests_size;
    if(rank_horizontal != 0)
        requests_size = 1;
    else
        requests_size = grid->A->cpu_shrink_thre2 - 1;

    if(rank == last_root_cpu)
        requests_size = neighbor_size;

//    std::vector<MPI_Request> reqs;
//    MPI_Request req;
    MPI_Request* requests = new MPI_Request[requests_size]; // 2 is for send and also for receive
    MPI_Status *statuses  = new MPI_Status[requests_size];

    int neigbor_rank;
    index_t send_size, recv_size;
    for(neigbor_rank = 1; neigbor_rank < grid->A->cpu_shrink_thre2; neigbor_rank++){
//        printf("rank = %d, rank_horizontal = %d, nprocs = %d, neigbor_rank = %d, recv_size = %lu, offset = %lu \n",
//               rank, rank_horizontal, nprocs, neigbor_rank, u.size(), offset);

        if(rank == last_root_cpu && (rank + neigbor_rank >= nprocs) )
            break;

        // send and receive size of u.
//        if(nprocs_horizontal == 0)
//            MPI_Recv(&recv_size, 1, MPI_UNSIGNED, neigbor_rank, 0, comm_new, MPI_STATUS_IGNORE);
//        if(nprocs_horizontal == neigbor_rank)
//            MPI_Send(&local_size, 1, MPI_UNSIGNED, 0, 0, comm_new);

        // send and receive u.
        // ------------------
        if(rank_horizontal == neigbor_rank){
            recv_size = grid->A->split[rank + 1] - grid->A->split[rank];
            u.resize(recv_size);
//            printf("un-shrink: rank = %d, neigbor_rank = %d, recv_size = %lu, offset = %lu \n", rank, neigbor_rank, u.size(), offset);
            MPI_Recv(&*u.begin(), recv_size, MPI_DOUBLE, 0, 0, grid->Ac.comm_horizontal, MPI_STATUS_IGNORE);
//            MPI_Recv(&*u.begin(), u.size(), MPI_DOUBLE, 0, 0, grid->Ac.comm_horizontal, MPI_STATUS_IGNORE);
//            MPI_Irecv(&*u.begin(), u.size(), MPI_DOUBLE, 0, 0, grid->Ac.comm_horizontal, &requests[0]);
//            reqs.emplace_back(req);
        }

        if(rank_horizontal == 0){
            send_size = grid->Ac.split_old[rank + neigbor_rank + 1] - grid->Ac.split_old[rank + neigbor_rank];
//            printf("un-shrink: rank = %d, neigbor_rank = %d, send_size = %u, offset = %lu \n", rank, neigbor_rank, send_size, offset);
            MPI_Send(&*(u.begin() + offset), send_size, MPI_DOUBLE, neigbor_rank, 0, grid->Ac.comm_horizontal);
//            MPI_Isend(&*u.begin() + offset, send_size, MPI_DOUBLE, neigbor_rank, 0, grid->Ac.comm_horizontal, &requests[neigbor_rank - 1]);
//            reqs.emplace_back(req);
            offset += send_size; // set offset for the next iteration
        }
    }

//    MPI_Barrier(grid->A->comm);
//    printf("done un-shrink: rank = %d\n", rank);
//    MPI_Barrier(grid->A->comm);

//    MPI_Waitall(requests_size, requests, statuses);

    if(rank_horizontal == 0){
        u.resize(grid->Ac.split_old[rank + 1] - grid->Ac.split_old[rank]);
        // todo: is shrink_to_fit required, or it's better to keep the memory for u to be used for the next vcycle iterations?
//        u.shrink_to_fit();
    }

//    MPI_Barrier(grid->A->comm);
//    if(rank_horizontal==0){
//        printf("\nafter un-shrinking: rank = %d, level = %d, u.size = %lu \n", rank, grid->currentLevel, u.size());
//        for(unsigned long i=0; i<u.size(); i++)
//            printf("u[%lu] = %f \n", i, u[i]);}
//    MPI_Barrier(grid->A->comm);
//    if(rank==1){
//        printf("\nafter un-shrinking: rank_horizontal = %d, level = %d, u.size = %lu \n", rank_horizontal, grid->currentLevel, u.size());
//        for(unsigned long i=0; i<u.size(); i++)
//            printf("u[%lu] = %f \n", i, u[i]);}
//    MPI_Barrier(grid->A->comm);
//    if(rank_horizontal==2){
//        printf("\nafter un-shrinking: rank = %d, level = %d, u.size = %lu \n", rank, grid->currentLevel, u.size());
//        for(unsigned long i=0; i<u.size(); i++)
//            printf("u[%lu] = %f \n", i, u[i]);}
//    MPI_Barrier(grid->A->comm);
//    if(rank_horizontal==3){
//        printf("\nafter un-shrinking: rank = %d, level = %d, u.size = %lu \n", rank, grid->currentLevel, u.size());
//        for(unsigned long i=0; i<u.size(); i++)
//            printf("u[%lu] = %f \n", i, u[i]);}
//    MPI_Barrier(grid->A->comm);

    delete [] requests;
    delete [] statuses;

    return 0;
}


bool saena_object::active(int l){
    return grids[l].A->active;
}


int saena_object::set_shrink_levels(std::vector<bool> sh_lev_vec) {
    shrink_level_vector = sh_lev_vec;
    return 0;
}


int saena_object::set_shrink_values(std::vector<int> sh_val_vec) {
    shrink_values_vector = sh_val_vec;
    return 0;
}


int saena_object::set_repartition_threshold(float thre) {
    repartition_threshold = thre;
    return 0;
}
