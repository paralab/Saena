#include "grid.h"

void Grid::repart_u_prepare(){
    // this function should only be called in parallel
    // this function sets the following parameters of grid:
    // scount3, sdispls2, rcount3, rdispls2
    // allocates memory for:
    // u_old, requests

//    MPI_Comm comm = grid->A->comm;
    MPI_Comm comm = Ac.comm;
    int rank = 0, nprocs = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    assert(nprocs > 0);
#endif

//    print_vector(Ac.split_old, 0, "split_old", comm);
//    print_vector(split, 0, "split", comm);
//    MPI_Barrier(comm); printf("rank %d: Ac.M = %u, Ac.M_old = %u \n", rank, Ac.M, Ac.M_old); MPI_Barrier(comm);

    scount2.assign(nprocs, 0);

    long least_proc = 0, curr_proc = 0;
    if(Ac.M_old != 0){
        least_proc = lower_bound3(&Ac.split[0], &Ac.split[nprocs], 0 + Ac.split_old[rank]);
        scount2[least_proc]++;
//        printf("rank %d: 0 + A.split_old[rank] = %u, least_proc = %ld \n", rank, 0 + Ac.split_old[rank], least_proc);

        curr_proc = least_proc;
        for(index_t i = 1; i < Ac.M_old; i++){
            if(i + Ac.split_old[rank] >= Ac.split[curr_proc+1]){
                if(Ac.shrinked)
                    curr_proc += Ac.cpu_shrink_thre2;
                else
                    curr_proc++;
            }
            scount2[curr_proc]++;
//            if(rank==0) printf("i + A.split_old[rank] = %u, curr_proc = %ld \n", i + Ac.split_old[rank], curr_proc);
        }
    }

//    print_vector(scount2, -1, "scount2", comm);

    rcount2.resize(nprocs);
    MPI_Alltoall(&scount2[0], 1, MPI_INT, &rcount2[0], 1, MPI_INT, comm);

//    print_vector(rcount2, -1, "rcount2", comm);

    sdispls2.resize(nprocs);
    sdispls2[0] = 0;
    for (int i = 0; i < nprocs - 1; ++i){
        sdispls2[i + 1] = scount2[i] + sdispls2[i];
        if(scount2[i]){
            scount3.emplace_back(scount2[i]);
            sproc_id.emplace_back(i);
        }
    }

    if(scount2[nprocs - 1]){
        scount3.emplace_back(scount2[nprocs - 1]);
        sproc_id.emplace_back(nprocs - 1);
    }

//    print_vector(scount3, -1, "scount3", comm);
//    print_vector(sproc_id, -1, "sproc_id", comm);
//    print_vector(sdispls2, -1, "sdispls2", comm);

    rdispls2.resize(nprocs);
    rdispls2[0] = 0;
    for (int i = 0; i < nprocs - 1; ++i){
        rdispls2[i + 1] = rcount2[i] + rdispls2[i];
        if(rcount2[i]){
            rcount3.emplace_back(rcount2[i]);
            rproc_id.emplace_back(i);
        }
    }

    if(rcount2[nprocs - 1]){
        rcount3.emplace_back(rcount2[nprocs - 1]);
        rproc_id.emplace_back(nprocs - 1);
    }

//    print_vector(rcount3, -1, "rcount3", comm);
//    print_vector(rproc_id, -1, "rproc_id", comm);
//    print_vector(rdispls2, -1, "rdispls2", comm);

    rcount2.clear();
    rcount2.shrink_to_fit();
    scount2.clear();
    scount2.shrink_to_fit();

    requests.resize(rcount3.size() + scount3.size());
    u_old = saena_aligned_alloc<value_t>(Ac.M);
}

void Grid::repart_u(value_t *&u){
    // this is used in vcycle

    MPI_Comm comm = Ac.comm_old;

//    MPI_Comm comm = grid.Ac.comm;
//    MPI_Comm comm = grid.A->comm;
//    int nprocs = 0;
//    MPI_Comm_size(comm, &nprocs);
//    int rank = 0;
//    MPI_Comm_rank(comm, &rank);

    swap(u, u_old);

    int flag = 0;
    int reqs = 0;
    for(int i = 0; i < rcount3.size(); ++i){
        MPI_Irecv(&u[rdispls2[rproc_id[i]]], rcount3[i], par::Mpi_datatype<value_t>::value(), rproc_id[i], 0, comm, &requests[reqs]);
        MPI_Test(&requests[reqs], &flag, MPI_STATUSES_IGNORE);
        ++reqs;
    }

    for(int i = 0; i < scount3.size(); ++i){
        MPI_Isend(&u_old[sdispls2[sproc_id[i]]], scount3[i], par::Mpi_datatype<value_t>::value(), sproc_id[i], 0, comm, &requests[reqs]);
        MPI_Test(&requests[reqs], &flag, MPI_STATUSES_IGNORE);
        ++reqs;
    }

    MPI_Waitall(reqs, &requests[0], MPI_STATUSES_IGNORE);

//    print_vector(u, -1, "u", comm);
}

void Grid::repart_back_u(value_t *&u){
    // this is used in vcycle

    MPI_Comm comm = Ac.comm_old;

//    MPI_Comm comm = grid.Ac.comm;
//    MPI_Comm comm = grid.A->comm;
//    int nprocs;
//    MPI_Comm_size(comm, &nprocs);
//    int rank;
//    MPI_Comm_rank(comm, &rank);

    swap(u, u_old);

    int flag = 0;
    int reqs = 0;
    for(int i = 0; i < scount3.size(); ++i){
        MPI_Irecv(&u[sdispls2[sproc_id[i]]], scount3[i], par::Mpi_datatype<value_t>::value(), sproc_id[i], 0, comm, &requests[reqs]);
        MPI_Test(&requests[reqs], &flag, MPI_STATUSES_IGNORE);
        ++reqs;
    }

    for(int i = 0; i < rcount3.size(); ++i){
        MPI_Isend(&u_old[rdispls2[rproc_id[i]]], rcount3[i], par::Mpi_datatype<value_t>::value(), rproc_id[i], 0, comm, &requests[reqs]);
        MPI_Test(&requests[reqs], &flag, MPI_STATUSES_IGNORE);
        ++reqs;
    }

    MPI_Waitall(reqs, &requests[0], MPI_STATUSES_IGNORE);

//    print_vector(u, -1, "u", comm);
}

void Grid::allocate_mem(){
    if(active){
        res         = saena_aligned_alloc<value_t>(A->M);
        uCorr       = saena_aligned_alloc<value_t>(A->M);
        res_coarse  = saena_aligned_alloc<value_t>(Ac.M_old);
        uCorrCoarse = saena_aligned_alloc<value_t>(Ac.M);
    }
}

void Grid::free_mem(){
    if(active){
        saena_free(res);
        saena_free(uCorr);
        saena_free(res_coarse);
        saena_free(uCorrCoarse);
    }
}