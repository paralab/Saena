#include <saena_matrix_dense.h>
#include <saena_matrix.h>
#include <dtypes.h>


saena_matrix_dense::saena_matrix_dense() = default;

saena_matrix_dense::saena_matrix_dense(index_t M1, index_t Nbig1){
    M    = M1;
    Nbig = Nbig1;
    entry.resize(M);
    for(index_t i = 0; i < M; i++){
        entry[i].resize(Nbig);
    }
}

saena_matrix_dense::saena_matrix_dense(index_t M1, index_t Nbig1, MPI_Comm comm1){
    M    = M1;
    Nbig = Nbig1;
    comm = comm1;
    entry.resize(M);
    for(index_t i = 0; i < M; i++){
        entry[i].resize(Nbig);
    }
}

// copy constructor
saena_matrix_dense::saena_matrix_dense(const saena_matrix_dense &B){
    comm = B.comm;
    M    = B.M;
    Nbig = B.Nbig;
    entry.resize(M);
    for(index_t i = 0; i < M; i++){
        entry[i].resize(Nbig);
    }
    split = B.split;
}

saena_matrix_dense::~saena_matrix_dense() {
    erase();
}

saena_matrix_dense& saena_matrix_dense::operator=(const saena_matrix_dense &B){
    if(this != &B) {
        comm = B.comm;
        M = B.M;
        Nbig = B.Nbig;
        entry.resize(M);
        for (index_t i = 0; i < M; i++) {
            entry[i].resize(Nbig);
        }
        split = B.split;
    }
    return *this;
}


int saena_matrix_dense::assemble() {
    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    split.resize(nprocs + 1);
    if(nprocs > 1){
        MPI_Allgather(&M, 1, par::Mpi_datatype<index_t>::value(), &split[1], 1, par::Mpi_datatype<index_t>::value(), comm);
    }

    for(int i = 1; i < nprocs + 1; ++i){
        split[i] += split[i-1];
    }

#ifdef SAENA_USE_ZFP
    if(use_zfp){
        allocate_zfp();
    }
#endif

//    print_vector(split, -1, "split", comm);

    return 0;
}


int saena_matrix_dense::erase(){
    if(!entry.empty()){
        for(index_t i = 0; i < M; ++i){
            entry[i].clear();
            entry[i].shrink_to_fit();
        }
        entry.clear();
        entry.shrink_to_fit();
    }
    split.clear();
    Nbig = 0;
    M = 0;
    return 0;
}


int saena_matrix_dense::print(int ran){

    // if ran >= 0 print_entry the matrix entries on proc with rank = ran
    // otherwise print_entry the matrix entries on all processors in order. (first on proc 0, then proc 1 and so on.)

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

    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if(rank==0) printf("dense matvec! \n");

    assert(!split.empty());
    assert(M == v.size());

//    MPI_Status sendRecvStatus;
    int right_neighbor = (rank + 1)%nprocs;
    int left_neighbor = rank - 1;
    if (left_neighbor < 0)
        left_neighbor += nprocs;
//    if(rank==0) printf("%d, %d\n", left_neighbor, right_neighbor);

    int owner = 0, next_owner = 0;
    index_t recv_size = 0, send_size = M;

    std::vector<value_t> v_recv = v;
    std::vector<value_t> v_send = v;

    fill(w.begin(), w.end(), 0);

    auto *requests = new MPI_Request[2];
    auto *statuses = new MPI_Status[2];

    for(index_t k = rank; k < rank+nprocs; k++){
        // Both local and remote loops are done here. The first iteration is the local loop. The rest are remote.
        // Send v to the left_neighbor processor, receive v from the right_neighbor processor.
        // In the next step: send v that was received in the previous step to the left_neighbor processor,
        // receive v from the right_neighbor processor. And so on.
        // --------------------------------------------------------------------

        // compute recv_size
        owner      = k % nprocs;
        next_owner = (k+1) % nprocs;
        recv_size  = split[next_owner + 1] - split[next_owner];
        v_recv.resize(recv_size); // TODO

        MPI_Irecv(&v_recv[0], recv_size, par::Mpi_datatype<value_t>::value(), right_neighbor, right_neighbor, comm, &requests[0]);
        MPI_Isend(&v_send[0], send_size, par::Mpi_datatype<value_t>::value(), left_neighbor,  rank,           comm, &requests[1]);
//        MPI_Test(); TODO

#pragma omp parallel for
        for(index_t i = 0; i < M; i++) {
            for (index_t j = split[owner]; j < split[owner + 1]; j++) {
//                if(rank==0) printf("A[%u][%u] = %f \t%f \n", i, j, entry[i][j], v_send[j - split[owner]]);
                w[i] += entry[i][j] * v_send[j - split[owner]]; // TODO
            }
        }

        MPI_Waitall(2, requests, statuses);
        std::swap(v_recv, v_send);
        send_size = recv_size;
    }

    delete [] requests;
    delete [] statuses;

    return 0;
}

int saena_matrix_dense::matvec_test(std::vector<value_t>& v, std::vector<value_t>& w){

    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if(rank==0) printf("dense matvec! \n");

    assert(!split.empty());
    assert(M == v.size());

    double t = 0, tcomm = 0;
    ++matvec_iter;

    int flag = 0;
//    MPI_Status sendRecvStatus;
    int right_neighbor = (rank + 1)%nprocs;
    int left_neighbor = rank - 1;
    if (left_neighbor < 0)
        left_neighbor += nprocs;
//    if(rank==0) printf("%d, %d\n", left_neighbor, right_neighbor);

    int owner = 0, next_owner = 0;
    index_t recv_size = 0, send_size = M;

    t = MPI_Wtime();

    std::vector<value_t> v_recv = v;
    std::vector<value_t> v_send = v;

    t = MPI_Wtime() - t;
    part1 += t;

    auto *requests = new MPI_Request[2];
    auto *statuses = new MPI_Status[2];

    for(index_t k = rank; k < rank+nprocs; k++){
        // Both local and remote loops are done here. The first iteration is the local loop. The rest are remote.
        // Send v to the left_neighbor processor, receive v from the right_neighbor processor.
        // In the next step: send v that was received in the previous step to the left_neighbor processor,
        // receive v from the right_neighbor processor. And so on.
        // --------------------------------------------------------------------

        // compute recv_size
        next_owner = (k+1) % nprocs;
        recv_size = split[next_owner+1] - split[next_owner];
        v_recv.resize(recv_size);

        tcomm = MPI_Wtime();

        MPI_Irecv(&v_recv[0], recv_size, par::Mpi_datatype<value_t>::value(), right_neighbor, right_neighbor, comm, &requests[0]);
        MPI_Isend(&v_send[0], send_size, par::Mpi_datatype<value_t>::value(), left_neighbor,  rank,           comm, &requests[1]);

        MPI_Test(requests,   &flag, statuses);
        MPI_Test(requests+1, &flag, statuses+1);

        t = MPI_Wtime();

        owner = k%nprocs;
#pragma omp parallel for
        for(index_t i = 0; i < M; i++) {
            for (index_t j = split[owner]; j < split[owner + 1]; j++) {
                w[i] += entry[i][j] * v_send[j - split[owner]];
//                if(rank==2) printf("A[%u][%u] = %f \t%f \n", i, j, entry[i][j], v_recv[j - split[owner]]);
            }
        }

        t = MPI_Wtime() - t;
        part5 += t;

        MPI_Waitall(2, requests, statuses);

        tcomm = MPI_Wtime() - tcomm;
        part3 += tcomm;

        t = MPI_Wtime();

        std::swap(v_recv, v_send);
        send_size = recv_size;

        t = MPI_Wtime() - t;
        part6 += t;
    }

    delete [] requests;
    delete [] statuses;

    return 0;
}

#ifdef SAENA_USE_ZFP
int saena_matrix_dense::matvec_comp(std::vector<value_t>& v, std::vector<value_t>& w){

    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if(rank==0) printf("dense compressed matvec! \n");

    assert(!split.empty());
    assert(M == v.size());

    double t = 0, tcomm = 0;
    ++matvec_iter;

//    MPI_Status sendRecvStatus;
    int right_neighbor = (rank + 1)%nprocs;
    int left_neighbor  = rank - 1;
    if (left_neighbor < 0)
        left_neighbor += nprocs;
//    if(rank==0) printf("%d, %d\n", left_neighbor, right_neighbor);

    int flag = 0; // for MPI_Test
    int owner = 0, next_owner = 0;
    index_t recv_size = 0;
    index_t send_size = M;
    index_t recv_size_comp = 0;
    index_t send_size_comp = zfp_rate / 2 * (index_t)ceil(send_size / 4.0);

//    std::vector<value_t> v_recv = v;
//    std::vector<value_t> v_send = v;

    t = MPI_Wtime();

    memcpy(&vecValues[0], &v[0], M * sizeof(value_t));
    memcpy(&vSend[0],     &v[0], M * sizeof(value_t));

    t = MPI_Wtime() - t;
    part1 += t;

    t = MPI_Wtime();

    zfp_stream_rewind(send_zfp);
    zfp_send_comp_sz = zfp_compress(send_zfp, send_field);

    t = MPI_Wtime() - t;
    part2 += t;

//    assert(zfp_send_buff_sz == zfp_send_comp_sz);
//    printf("rank %d: vIndexSize = %u, zfp_send_buff_sz = %u, \tzfp_send_comp_sz = %u\n", rank, vIndexSize, zfp_send_buff_sz, zfp_send_comp_sz);
//    if(zfp_send_buff_sz != zfp_send_comp_sz){
//        printf("ERROR: rank %d: vIndexSize = %u, zfp_send_buff_sz = %u, \tzfp_send_comp_sz = %u\n", rank, vIndexSize, zfp_send_buff_sz, zfp_send_comp_sz);
//    }

    auto *requests = new MPI_Request[2];
    auto *statuses = new MPI_Status[2];

    for(index_t k = rank; k < rank+nprocs; k++){
        // Both local and remote loops are done here. The first iteration is the local loop. The rest are remote.
        // Send v to the left_neighbor processor, receive v from the right_neighbor processor.
        // In the next step: send v that was received in the previous step to the left_neighbor processor,
        // receive v from the right_neighbor processor. And so on.
        // --------------------------------------------------------------------

        // compute recv_size
        next_owner = (k + 1) % nprocs;
        recv_size  = split[next_owner+1] - split[next_owner];
        recv_size_comp = zfp_rate / 2 * (index_t)ceil(recv_size / 4.0);
//        v_recv.resize(recv_size);

//        MPI_Irecv(&vecValues[0], recv_size, par::Mpi_datatype<value_t>::value(), right_neighbor, right_neighbor, comm, &requests[0]);
//        MPI_Isend(&vSend[0], send_size, par::Mpi_datatype<value_t>::value(), left_neighbor,  rank,           comm, &requests[1]);

//        MPI_Irecv(&zfp_recv_buff[0], recv_size, par::Mpi_datatype<value_t>::value(), right_neighbor, right_neighbor, comm, &requests[0]);
//        MPI_Isend(&zfp_send_buff[0], send_size, par::Mpi_datatype<value_t>::value(), left_neighbor,  rank,           comm, &requests[1]);

        tcomm = MPI_Wtime();

        MPI_Irecv(&zfp_recv_buff[0], recv_size_comp, MPI_UNSIGNED_CHAR, right_neighbor, right_neighbor, comm, &requests[0]);
        MPI_Test(requests,   &flag, statuses);

        MPI_Isend(&zfp_send_buff[0], send_size_comp, MPI_UNSIGNED_CHAR, left_neighbor,  rank,           comm, &requests[1]);
        MPI_Test(requests+1, &flag, statuses+1);

        t = MPI_Wtime();

        zfp_stream_rewind(send_zfp);
        zfp_decompress(send_zfp, send_field);

        t = MPI_Wtime() - t;
        part4 += t;

//        if(rank == 1) cout << "===========================\n\n" << "owner: " << owner;
//        print_vector(vSend, 1, "vSend", comm);

        t = MPI_Wtime();

        // TODO: change vSend to use pointer shiftted by split[owner]
        owner = k % nprocs;
#pragma omp parallel for
        for(index_t i = 0; i < M; ++i) {
            for (index_t j = split[owner]; j < split[owner + 1]; ++j) {
                w[i] += entry[i][j] * vSend[j - split[owner]];
//                if(rank==2) printf("A[%u][%u] = %f \t%f \n", i, j, entry[i][j], v_recv[j - split[owner]]);
            }
        }

        t = MPI_Wtime() - t;
        part5 += t;

        send_size = recv_size;
        MPI_Waitall(2, requests, statuses);

        tcomm = MPI_Wtime() - tcomm;
        part3 += tcomm;

//        zfp_stream_rewind(recv_zfp);
//        zfp_decompress(recv_zfp, recv_field);

//        if(rank == 0) cout << "next owner: " << next_owner;
//        print_vector(vecValues, 0, "vecValues", comm);

        t = MPI_Wtime();

        std::swap(vecValues, vSend);
        std::swap(zfp_recv_buff, zfp_send_buff);

        send_field  = zfp_field_1d(&vSend[0], zfptype, M);
        send_stream = stream_open(zfp_send_buff, zfp_send_buff_sz);
//        send_zfp    = zfp_stream_open(send_stream);
        zfp_stream_set_bit_stream(send_zfp, send_stream);

        // TODO: change M in zfp_field_1d to split[nex_owner + 1] - split[nex_owner]

        recv_field  = zfp_field_1d(&vecValues[0], zfptype, M);
        recv_stream = stream_open(zfp_recv_buff, zfp_recv_buff_sz);
//        recv_zfp    = zfp_stream_open(recv_stream);
        zfp_stream_set_bit_stream(recv_zfp, recv_stream);

        t = MPI_Wtime() - t;
        part6 += t;
    }

    delete [] requests;
    delete [] statuses;

    return 0;
}

int saena_matrix_dense::allocate_zfp(){

    free_zfp_buff = true;

    // TODO: update the sizes for general cases.
    vSend.resize(M);
    vecValues.resize(M);

    // compute zfp_send_buff_sz in bytes:
    // rate / 8 * 4 * ceil(size / 4).
    // divide by 8 to convert bits to bytes
    // 4 * ceil(size / 4): because zfp compresses blocks of size 4.
    zfp_send_buff_sz = zfp_rate / 2 * (unsigned)ceil(M / 4.0);
    zfp_recv_buff_sz = zfp_rate / 2 * (unsigned)ceil(M / 4.0);
    zfp_send_buff    = new uchar[zfp_send_buff_sz];
    zfp_recv_buff    = new uchar[zfp_recv_buff_sz];

    send_field  = zfp_field_1d(&vSend[0], zfptype, M);
    send_stream = stream_open(zfp_send_buff, zfp_send_buff_sz);
    send_zfp    = zfp_stream_open(send_stream);
    zfp_stream_set_rate(send_zfp, zfp_rate, zfptype, 1, 0);
//    zfp_stream_set_precision(send_zfp, zfp_precision);

//    printf("M = %u, \tvIndexSize = %u, \tzfp_send_buff_sz = %u\n", M, vIndexSize, zfp_send_buff_sz);

    recv_field  = zfp_field_1d(&vecValues[0], zfptype, M);
    recv_stream = stream_open(zfp_recv_buff, zfp_recv_buff_sz);
    recv_zfp    = zfp_stream_open(recv_stream);
    zfp_stream_set_rate(recv_zfp, zfp_rate, zfptype, 1, 0);

    return 0;
}

int saena_matrix_dense::deallocate_zfp(){

    if(free_zfp_buff){

        zfp_field_free(send_field);
        zfp_stream_close(send_zfp);
        stream_close(send_stream);

        zfp_field_free(recv_field);
        zfp_stream_close(recv_zfp);
        stream_close(recv_stream);

        delete []zfp_send_buff;
        delete []zfp_recv_buff;
        free_zfp_buff = false;

    }

    return 0;
}
#endif

void saena_matrix_dense::matvec_time_init(){
    matvec_iter = 0;
    part1 = 0;
    part2 = 0;
    part3 = 0;
    part4 = 0;
    part5 = 0;
    part6 = 0;
}

void saena_matrix_dense::matvec_time_print() const{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    double tmp = 1;
    if(matvec_iter != 0){
        tmp = static_cast<double>(matvec_iter);
    }

//    print_time(part1 / tmp, "send vector", comm);
//    print_time(part2 / tmp, "compress", comm);
//    print_time((part3-part4-part5-part6) / tmp, "comm", comm);
//    print_time(part4 / tmp, "local", comm);
//    print_time(part5 / tmp, "decompress", comm);
//    print_time(part6 / tmp, "remote", comm);

    double p1ave = print_time_ave(part1 / tmp, "", comm);               // send buff
    double p2ave = print_time_ave(part2 / tmp, "", comm);               // compress
    double p3ave = print_time_ave((part3-part4-part5) / tmp, "", comm); // comm
    double p4ave = print_time_ave(part4 / tmp, "", comm);               // decompress
    double p5ave = print_time_ave(part5 / tmp, "", comm);               // compute
    double p6ave = print_time_ave(part6 / tmp, "", comm);               // swap
    if(!rank){
//        printf("matvec iteration: %ld", matvec_iter);
        printf("average time:\nsend buff\ncompress\ncomm\ndecompress\ncompute\nswap\n\n"
               "%f\n%f\n%f\n%f\n%f\n%f\n", p1ave, p2ave, p3ave, p4ave, p5ave, p6ave);
    }

}


// another dense matvec implementation, but without overlapping.
/*
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
*/


int saena_matrix_dense::convert_saena_matrix(saena_matrix *A){

    comm = A->comm;
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    M     = A->M;
    Nbig  = A->Mbig;
    split = A->split;

//    printf("rank %d: step 1, M = %d, Nbig = %d, A->entry.size = %ld, A->nnz_l = %ld\n",
//            rank, M, Nbig, A->entry.size(), A->nnz_l);
//    print_vector(split, -1, "split", comm);

    if(entry.empty()){
        entry.resize(M);
        for(index_t i = 0; i < M; i++){
            entry[i].assign(Nbig, 0);
        }
    }else{
        assert(entry.size() == M);
    }

#pragma omp parallel for
    for(nnz_t i = 0; i < A->nnz_l; i++){
        entry[A->entry[i].row - split[rank]][A->entry[i].col] = A->entry[i].val;
    }

    return 0;
}