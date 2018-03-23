#include "iostream"

#include <saena_matrix_dense.h>
#include <aux_functions.h>

saena_matrix_dense::saena_matrix_dense()=default;

saena_matrix_dense::saena_matrix_dense(index_t M1, index_t Nbig1){
    M = M1;
    Nbig = Nbig1;

    entry = new value_t*[M];
    for(index_t i = 0; i < M; i++)
        entry[i] = new value_t[Nbig];
}

saena_matrix_dense::saena_matrix_dense(index_t M1, index_t Nbig1, MPI_Comm comm1){
    M = M1;
    Nbig = Nbig1;
    comm = comm1;

    entry = new value_t*[M];
    for(index_t i = 0; i < M; i++)
        entry[i] = new value_t[Nbig];
}


saena_matrix_dense::~saena_matrix_dense() {
    for(index_t i = 0; i < M; i++)
        delete [] entry[i];
    delete [] entry;
}


int saena_matrix_dense::set(index_t row, index_t col, value_t val){
    entry[row][col] = val;
    return 0;
}


int saena_matrix_dense::print(int ran){

    // if ran >= 0 print the matrix entries on proc with rank = ran
    // otherwise print the matrix entries on all processors in order. (first on proc 0, then proc 1 and so on.)

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(split.empty()){
        if(rank==0) printf("Error: split for the dense matrix is not set!\n");
        MPI_Finalize();
        return -1;
    }

    if(ran >= 0) {
        if (rank == ran) {
            printf("\nmatrix on proc = %d \n", ran);
            for(index_t i = 0; i < M; i++){
                printf("\n");
                for(index_t j = 0; j < Nbig; j++)
                    printf("A[%u][%u] = %f \n", i+split[rank], j, entry[i][j]);
            }
        }
    } else{
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("\nmatrix on proc = %d \n", proc);
                for(index_t i = 0; i < M; i++){
                    printf("\n");
                    for(index_t j = 0; j < Nbig; j++)
                        printf("A[%u][%u] = %f \n", i+split[rank], j, entry[i][j]);
                }
            }
            MPI_Barrier(comm);
        }
    }
    return 0;
}


int saena_matrix_dense::matvec(std::vector<value_t>& v, std::vector<value_t>& w){

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(split.empty()){
        if(rank==0) printf("Error: split for the dense matrix is not set!\n");
        MPI_Finalize();
        return -1;
    }

    if(M != v.size()){
        if(rank==0) printf("Error: vector v does not have the same size as the local number of the rows of the matrix!\n");
        MPI_Finalize();
        return -1;
    }

    // local loop
    // ----------
    w.assign(M, 0);
    for(index_t i = 0; i < M; i++) {
        for (index_t j = split[rank]; j < split[rank + 1]; j++) {
            w[i] += entry[i][j] * v[j - split[rank]];
//            if(rank==2) printf("A[%u][%u] = %f \t%f \n", i, j, entry[i][j], v[j - split[rank]]);
        }
    }

    std::vector<value_t> v_recv;
    std::vector<value_t> v_send = v;

    // remote loop
    // -----------
    MPI_Status sendRecvStatus;
    int right_neighbor = (rank + 1)%nprocs;
    int left_neighbor = rank - 1;
    if (left_neighbor < 0)
        left_neighbor += nprocs;
//    if(rank==0) printf("%d, %d\n", left_neighbor, right_neighbor);

//    int right_owner = right_neighbor;
    int owner = left_neighbor;
    unsigned int recv_size, send_size = M;

    for(index_t k = 1; k < nprocs; k++){
        // Send v to the right_neighbor processor, receive v from the left_neighbor processor.
        // In the next step: send v that was received in the previous step to the right_neighbor processor,
        // receive v from the left_neighbor processor. And so on.
        // --------------------------------------------------------------------

        // tag = sender rank
        MPI_Sendrecv(&send_size, 1, MPI_UNSIGNED, right_neighbor, right_neighbor,
                     &recv_size, 1, MPI_UNSIGNED, left_neighbor,  rank, comm, &sendRecvStatus);

        v_recv.resize(recv_size);
//        if(rank==0) printf("rank %d recv_size = %d from %d, send_size = %d to %d \n",
//                           rank, recv_size, left_neighbor, send_size, right_neighbor);

        // tag = sender rank
        MPI_Sendrecv(&v_send[0], send_size, MPI_DOUBLE, right_neighbor, right_neighbor,
                     &v_recv[0], recv_size, MPI_DOUBLE, left_neighbor,  rank, comm, &sendRecvStatus);

//        print_vector(v_recv, 0, comm);

        for(index_t i = 0; i < M; i++)
            for(index_t j = split[owner]; j < split[owner+1]; j++) {
                w[i] += entry[i][j] * v_recv[j - split[owner]];
//                if(rank==2) printf("A[%u][%u] = %f \t%f \n", i, j, entry[i][j], v_recv[j - split[owner]]);
            }

        v_send = v_recv;
        send_size = recv_size;

        owner--;
        if(owner<0) owner += nprocs;
//        right_owner = (right_owner + 1)%nprocs;
    }

    return 0;
}