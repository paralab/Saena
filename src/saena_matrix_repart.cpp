#include "saena_matrix.h"

int saena_matrix::repartition_nnz_initial(){
    // before using this function these variables of saena_matrix should be set:
    // "Mbig", "nnz_g", "initial_nnz_l", "data"

    // the following variables of saena_matrix class will be set in this function:
    // "nnz_l", "M", "split", "entry"

    // summary: number of buckets are computed based of the number fo rows and number of processors.
    // firstSplit[] is of size n_buckets+1 and is a row partition of the matrix with almost equal number of rows.
    // then the buckets (firsSplit) are combined to have almost the same number of nonzeros. This is split[].

    // if set functions are used the following function should be used.
    if(!read_from_file)
        setup_initial_data();

    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    if(verbose_repart && rank == 0) printf("repartition - step 1!\n");
#endif

    if(Mbig == 0){
        std::cout << "Number of rows of the matrix is 0 in " << __func__ << std::endl;
        exit(EXIT_FAILURE);
    }

    density = (nnz_g / double(Mbig)) / (Mbig);
    last_density_shrink = density;
    last_M_shrink = Mbig;
//    last_nnz_shrink = nnz_g;

    split_b = split;
    split.resize(nprocs + 1);

    if(nprocs > 1) {
        // *************************** find splitters ****************************
        // split the matrix row-wise by splitters, so each processor get almost equal number of nonzeros

        // definition of buckets: bucket[i] = [ firstSplit[i] , firstSplit[i+1] ). Number of buckets = n_buckets
        int n_buckets = 0;

        if (Mbig > nprocs * nprocs) {
            if (nprocs < 1000)
                n_buckets = nprocs * nprocs;
            else
                n_buckets = 1000 * nprocs;
        } else if (nprocs <= Mbig) {
            n_buckets = Mbig;
        } else { // nprocs > Mbig
            // it may be better to set nprocs=Mbig and work only with the first Mbig processors.
            if (rank == 0)
                std::cout << "number of tasks cannot be greater than the number of rows of the matrix." << std::endl;
            MPI_Finalize();
        }

//        if (rank==0) std::cout << "n_buckets = " << n_buckets << ", Mbig = " << Mbig << std::endl;

        std::vector<index_t> splitOffset(n_buckets);
        auto baseOffset = index_t(floor(1.0 * Mbig / n_buckets));
        float offsetRes = float(1.0 * Mbig / n_buckets) - baseOffset;
//        if (rank==0) std::cout << "baseOffset = " << baseOffset << ", offsetRes = " << offsetRes << std::endl;
        float offsetResSum = 0;
        splitOffset[0] = 0;
        for (index_t i = 1; i < n_buckets; ++i) {
            splitOffset[i] = baseOffset;
            offsetResSum += offsetRes;
            if (offsetResSum >= 1) {
                splitOffset[i]++;
                offsetResSum -= 1;
            }
        }

#ifdef __DEBUG1__
//        print_vector(splitOffset, 0, "splitOffset", comm);
        if (verbose_repart && rank == 0) printf("repartition - step 2!\n");
#endif

        std::vector<index_t> firstSplit(n_buckets + 1);
        firstSplit[0] = 0;
        for (index_t i = 1; i < n_buckets; ++i) {
            firstSplit[i] = firstSplit[i - 1] + splitOffset[i];
        }
        firstSplit[n_buckets] = Mbig;

        splitOffset.clear();
        splitOffset.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(firstSplit, -1, "firstSplit", comm);
#endif

        index_t least_bucket = 0, last_bucket = n_buckets - 1;
        if(!data.empty()){
            least_bucket = lower_bound2(&firstSplit[0], &firstSplit[n_buckets], data[0].row);
            last_bucket  = lower_bound2(&firstSplit[0], &firstSplit[n_buckets], data.back().row);
            last_bucket++;
        }

#ifdef __DEBUG1__
//    if (rank==0) std::cout << "least_bucket:" << least_bucket << ", last_bucket = " << last_bucket << std::endl;
#endif

        // H_l is the histogram of (local) nnz of buckets
        std::vector<nnz_t> H_l(n_buckets, 0);
//        H_l[least_bucket]++; // add the first element to local histogram (H_l) here.

        for (nnz_t i = 0; i < initial_nnz_l; i++) {
            if (data[i].row >= firstSplit[least_bucket + 1]) {
                least_bucket += lower_bound2(&firstSplit[least_bucket], &firstSplit[last_bucket], data[i].row);
//                if (rank==0) std::cout << "row = " << data[i].row << ", least_bucket = " << least_bucket << std::endl;
            }
            H_l[least_bucket]++;
        }

#ifdef __DEBUG1__
//    print_vector(H_l, 0, "H_l", comm);
#endif

        // H_g is the histogram of (global) nnz per bucket
        std::vector<nnz_t> H_g(n_buckets);
        MPI_Allreduce(&H_l[0], &H_g[0], n_buckets, par::Mpi_datatype<nnz_t>::value(), MPI_SUM, comm);

        H_l.clear();
        H_l.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(H_g, 0, "H_g", comm);
#endif

        std::vector<nnz_t> H_g_scan(n_buckets);
        H_g_scan[0] = H_g[0];
        for (nnz_t i = 1; i < n_buckets; i++)
            H_g_scan[i] = H_g[i] + H_g_scan[i - 1];

        H_g.clear();
        H_g.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(H_g_scan, 0, "H_g_scan", comm);
        if (verbose_repart && rank == 0) printf("repartition - step 3!\n");
#endif

        // -------------------------------------------
        // determine number of rows on each proc based on having almost the same number of nonzeros per proc.

        const nnz_t NNZ_PROC = nnz_g / nprocs;

        index_t procNum = 0;
        split[0] = 0;
        for (nnz_t i = 1; i < n_buckets; ++i) {
            // allocate at least 1 row to the remaining processors
            if(Mbig - firstSplit[i + 1] < nprocs - (procNum + 1)){
                for(; i < n_buckets; ++i){
                    procNum++;
                    split[procNum] = firstSplit[i];
                }
                break;
            }

//            if (rank==0) std::cout << "(procNum+1)*NNZ_PROC = " << (procNum+1)*NNZ_PROC << ", procNum = " << procNum << std::endl;
            if (H_g_scan[i] > ((procNum + 1) * NNZ_PROC)) {
                ++procNum;
                split[procNum] = firstSplit[i];
            }
        }
        split[nprocs] = Mbig;

//        if(!rank) printf("procNum = %d, n_buckets = %d\n", procNum, n_buckets);
        assert((procNum+1 == nprocs) || (procNum == nprocs));

        // if there are dense rows in the matrix, the above loop may not allocate rows for the matrices (which is done in
        // split). If that happens then assign one row to the matrices without any row.
//        int it = 0;
//        for (int i = nprocs; i >= procNum; --i) {
//            split[i] = Mbig - it;
//            ++it;
//        }

        H_g_scan.clear();
        H_g_scan.shrink_to_fit();
        firstSplit.clear();
        firstSplit.shrink_to_fit();

#ifdef __DEBUG1__
        {
//        MPI_Barrier(comm);
//        print_vector(split, 0, "split before", comm);
//        MPI_Barrier(comm);

        // todo: return split to its original form
        // update split to make zfp work, since it only works on arrays of size 4k,
        // if Mbig is 4k, then each split[i] should be a multiple of 4.
//        if(Mbig / nprocs > 4){
//            for(index_t i = 1; i < nprocs; ++i){
//                split[i] += (4 - (split[i] % 4)) % 4;
//            }
//        }
//
//        MPI_Barrier(comm);
//        print_vector(split, 0, "split", comm);
//        MPI_Barrier(comm);
        }
#endif

        // set the number of rows for each process
        M = split[rank + 1] - split[rank];
        assert(M != 0);

#ifdef __DEBUG1__
//    M_old = M;

//    if(M % 4){
//        printf("\nM is not a multiple of 4 on proc %d: M = %u \nzfp will no work!\n", rank, M);
//        exit(EXIT_FAILURE);
//    }

        if (verbose_repart && rank == 0) printf("repartition - step 4!\n");

//    unsigned int M_min_global;
//    MPI_Allreduce(&M, &M_min_global, 1, MPI_UNSIGNED, MPI_MIN, comm);
#endif

        // *************************** exchange data ****************************

        index_t least_proc = 0, last_proc = nprocs - 1;
        if(!data.empty()){
            least_proc = (index_t) lower_bound2(&split[0], &split[nprocs], data[0].row);
            last_proc  = (index_t) lower_bound2(&split[0], &split[nprocs], data.back().row);
            last_proc++;
        }

#ifdef __DEBUG1__
//        if (rank==1) std::cout << "\nleast_proc:" << least_proc << ", last_proc = " << last_proc << std::endl;
//        print_vector(split, 1, "split", comm);
#endif

        // todo: check if data is sorted row-major, then remove lower_bound and add if statement.
        std::vector<int> send_size_array(nprocs, 0);
        // add the first element to send_size_array
//        send_size_array[least_proc]++;

        for (nnz_t i = 0; i < initial_nnz_l; i++){
            if(data[i].row >= split[least_proc+1]){
                least_proc += lower_bound2(&split[least_proc], &split[last_proc], data[i].row);
            }
//            if(rank==2) printf("row = %u, %u \n", data[i].row, least_proc);
            ++send_size_array[least_proc];
        }

#ifdef __DEBUG1__
//        print_vector(send_size_array, -1, "send_size_array", comm);
#endif

        std::vector<int> recv_size_array(nprocs);
        MPI_Alltoall(&send_size_array[0], 1, MPI_INT, &recv_size_array[0], 1, MPI_INT, comm);

#ifdef __DEBUG1__
//        print_vector(recv_size_array, 0, "recv_size_array", comm);
#endif

        std::vector<int> send_offset(nprocs);
        send_offset[0] = 0;
        for (int i=1; i<nprocs; i++)
            send_offset[i] = send_size_array[i-1] + send_offset[i-1];

#ifdef __DEBUG1__
//        print_vector(send_offset, 0, "send_offset", comm);
#endif

        std::vector<int> recv_offset(nprocs);
        recv_offset[0] = 0;
        for (int i = 1; i < nprocs; i++)
            recv_offset[i] = recv_size_array[i-1] + recv_offset[i-1];

#ifdef __DEBUG1__
//        print_vector(recv_offset, 0, "recv_offset", comm);
        if(verbose_repart && rank == 0) printf("repartition - step 5!\n");
#endif

        nnz_l = recv_offset[nprocs-1] + recv_size_array[nprocs-1];

#ifdef __DEBUG1__
//        printf("rank=%d \t A.nnz_l=%lu \t A.nnz_g=%lu \n", rank, nnz_l, nnz_g);
        if(verbose_repart && rank == 0) printf("repartition - step 6!\n");
#endif

        auto dt = cooEntry::mpi_datatype();
        entry.resize(nnz_l);
        MPI_Alltoallv(&data[0],  &send_size_array[0], &send_offset[0], dt,
                      &entry[0], &recv_size_array[0], &recv_offset[0], dt, comm);

        MPI_Type_free(&dt);
        data.clear();
        data.shrink_to_fit();
    }else{
        M = Mbig;
        split[0] = 0;
        split[1] = Mbig;
        nnz_l = initial_nnz_l;
        entry.swap(data);
    }

    split_old = split;

    std::sort(entry.begin(), entry.end());

#ifdef __DEBUG1__
//    print_vector(entry, -1, "entry", comm);
//    print_entry(0);
//    MPI_Barrier(comm); printf("repartition: rank = %d, Mbig = %u, M = %u, nnz_g = %u, nnz_l = %u \n", rank, Mbig, M, nnz_g, nnz_l); MPI_Barrier(comm);
#endif

    active = true;
    active_minor = true;

#ifdef __DEBUG1__
    if(verbose_repart && rank == 0) printf("repartition - end\n");
#endif

    return 0;
}


int saena_matrix::repartition_nnz_update(){
    // before using this function these variables of saena_matrix should be set:
    // Mbig", "nnz_g", "initial_nnz_l", "data"

    // the following variables of saena_matrix class will be set in this function:
    // "nnz_l", "M", "split", "entry"

    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(verbose_repart_update && rank == 0) printf("repartition - step 1!\n");

    // *************************** setup_initial_data2 ****************************

    if(!rank) cout << "The matrix assemble function is being called for the second time."
                      "If the update part is required, enable it before calling the matrix update functions." << endl;
    MPI_Abort(comm, 1);

#if 0
    setup_initial_data2();

    // *************************** exchange data ****************************

    density = (nnz_g / double(Mbig)) / (Mbig);

//    print_vector(data, -1, "data", comm);
    std::sort(data.begin(), data.end(), row_major);
//    print_vector(data, -1, "data", comm);

    long least_proc, last_proc;
    least_proc = lower_bound2(&split[0], &split[nprocs], data[0].row);
    last_proc  = lower_bound2(&split[0], &split[nprocs], data.back().row);
    last_proc++;

//    if (rank==1) std::cout << "\nleast_proc:" << least_proc << ", last_proc = " << last_proc << std::endl;

    std::vector<int> send_size_array(nprocs, 0);
    // add the first element to send_size_array
    send_size_array[least_proc]++;

    for (nnz_t i = 1; i < initial_nnz_l; i++){
        if(data[i].row >= split[least_proc+1]) {
            least_proc += lower_bound2(&split[least_proc], &split[last_proc], data[i].row);
        }
        send_size_array[least_proc]++;
    }

//    print_vector(send_size_array, 0, "send_size_array", comm);

    std::vector<int> recv_size_array(nprocs);
    MPI_Alltoall(&send_size_array[0], 1, MPI_INT, &recv_size_array[0], 1, MPI_INT, comm);

//    print_vector(recv_size_array, 0, "recv_size_array", comm);

    std::vector<int> send_offset(nprocs);
    send_offset[0] = 0;
    for (int i=1; i<nprocs; i++)
        send_offset[i] = send_size_array[i-1] + send_offset[i-1];

//    print_vector(send_offset, 0, "send_offset", comm);

    std::vector<int> recv_offset(nprocs);
    recv_offset[0] = 0;
    for (int i = 1; i < nprocs; i++)
        recv_offset[i] = recv_size_array[i-1] + recv_offset[i-1];

//    print_vector(recv_offset, 0, "recv_offset", comm);

    if(verbose_repart_update && rank==0) printf("repartition - step 2!\n");

    unsigned long nnz_l_temp = recv_offset[nprocs-1] + recv_size_array[nprocs-1];
//    printf("rank=%d \t A.nnz_l=%u \t A.nnz_g=%u \n", rank, nnz_l, nnz_g);
    if(nnz_l_temp != nnz_l) printf("error: number of local nonzeros is changed on processor %d during the matrix update!\n", rank);

    if(verbose_repart_update && rank==0) printf("repartition - step 3!\n");

//    entry.clear();
    entry.resize(nnz_l_temp);
    entry.shrink_to_fit();

    MPI_Alltoallv(&data[0],  &send_size_array[0], &send_offset[0], cooEntry::mpi_datatype(),
                  &entry[0], &recv_size_array[0], &recv_offset[0], cooEntry::mpi_datatype(), comm);

    std::sort(entry.begin(), entry.end());

    // clear data and free memory.
    data.clear();
    data.shrink_to_fit();

//    print_entry(0);
//    MPI_Barrier(comm); printf("repartition: rank = %d, Mbig = %u, M = %u, nnz_g = %u, nnz_l = %u \n", rank, Mbig, M, nnz_g, nnz_l); MPI_Barrier(comm);

    if(verbose_repart_update && rank==0) printf("repartition - step 4!\n");
#endif
    return 0;
}


// this version of repartition_nnz() is WITHOUT cpu shrinking. OLD VERSION.
/*
int saena_matrix::repartition_nnz(){

    // summary: number of buckets are computed based of the number fo rows and number of processors.
    // firstSplit[] is of size n_buckets+1 and is a row partition of the matrix with almost equal number of rows.
    // then the buckets (firsSplit) are combined to have almost the same number of nonzeros. This is split[].

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    bool verbose_repart = false;

    if(verbose_repart && rank==0) printf("repartition - step 1!\n");

//    last_M_shrink = Mbig;

    // *************************** find splitters ****************************
    // split the matrix row-wise by splitters, so each processor get almost equal number of nonzeros

    // definition of buckets: bucket[i] = [ firstSplit[i] , firstSplit[i+1] ). Number of buckets = n_buckets
    int n_buckets = 0;

//    if (Mbig > nprocs*nprocs){
//        if (nprocs < 1000)
//            n_buckets = nprocs*nprocs;
//        else
//            n_buckets = 1000*nprocs;
//    }
//    else
//        n_buckets = Mbig;

    if (Mbig > nprocs*nprocs){
        if (nprocs < 1000)
            n_buckets = nprocs*nprocs;
        else
            n_buckets = 1000*nprocs;
    }
    else if(nprocs < Mbig){
        n_buckets = Mbig;
    } else{ // nprocs > Mbig
        // it may be better to set nprocs=Mbig and work only with the first Mbig processors.
        if(rank == 0)
            std::cout << "number of MPI tasks cannot be greater than the number of rows of the matrix." << std::endl;
        MPI_Finalize();
    }

//    if (rank==0) std::cout << "n_buckets = " << n_buckets << ", Mbig = " << Mbig << std::endl;

    std::vector<int> splitOffset(n_buckets);
    auto baseOffset = int(floor(1.0*Mbig/n_buckets));
    float offsetRes = float(1.0*Mbig/n_buckets) - baseOffset;
//    if (rank==0) std::cout << "baseOffset = " << baseOffset << ", offsetRes = " << offsetRes << std::endl;
    float offsetResSum = 0;
    splitOffset[0] = 0;
    for(unsigned int i=1; i<n_buckets; i++){
        splitOffset[i] = baseOffset;
        offsetResSum += offsetRes;
        if (offsetResSum >= 1){
            splitOffset[i]++;
            offsetResSum -= 1;
        }
    }

//    if (rank==0){
//        std::cout << "splitOffset:" << std::endl;
//        for(long i=0; i<n_buckets; i++)
//            std::cout << splitOffset[i] << std::endl;}

    if(verbose_repart && rank==0) printf("repartition - step 2!\n");

    std::vector<unsigned long> firstSplit(n_buckets+1);
    firstSplit[0] = 0;
    for(unsigned int i=1; i<n_buckets; i++){
        firstSplit[i] = firstSplit[i-1] + splitOffset[i];
    }
    firstSplit[n_buckets] = Mbig;

    splitOffset.clear();
    splitOffset.shrink_to_fit();

//    if (rank==0){
//        std::cout << "\nfirstSplit:" << std::endl;
//        for(long i=0; i<n_buckets+1; i++)
//            std::cout << firstSplit[i] << std::endl;
//    }

    initial_nnz_l = nnz_l;
    // H_l is the histogram of (local) nnz per bucket
    std::vector<long> H_l(n_buckets, 0);
    for(unsigned int i=0; i<initial_nnz_l; i++)
        H_l[lower_bound2(&firstSplit[0], &firstSplit[n_buckets], entry[i].row)]++;

//    if (rank==0){
//        std::cout << "\ninitial_nnz_l = " << initial_nnz_l << std::endl;
//        std::cout << "local histogram:" << std::endl;
//        for(unsigned int i=0; i<n_buckets; i++)
//            std::cout << H_l[i] << std::endl;
//    }

    // H_g is the histogram of (global) nnz per bucket
    std::vector<long> H_g(n_buckets);
    MPI_Allreduce(&H_l[0], &H_g[0], n_buckets, MPI_LONG, MPI_SUM, comm);

    H_l.clear();
    H_l.shrink_to_fit();

//    if (rank==1){
//        std::cout << "global histogram:" << std::endl;
//        for(unsigned int i=0; i<n_buckets; i++){
//            std::cout << H_g[i] << std::endl;
//        }
//    }

    std::vector<long> H_g_scan(n_buckets);
    H_g_scan[0] = H_g[0];
    for (unsigned int i=1; i<n_buckets; i++)
        H_g_scan[i] = H_g[i] + H_g_scan[i-1];

    H_g.clear();
    H_g.shrink_to_fit();

//    if (rank==0){
//        std::cout << "scan of global histogram:" << std::endl;
//        for(unsigned int i=0; i<n_buckets; i++)
//            std::cout << H_g_scan[i] << std::endl;}

    if(verbose_repart && rank==0) printf("repartition - step 3!\n");

//    if (rank==0){
//        std::cout << std::endl << "split old:" << std::endl;
//        for(unsigned int i=0; i<nprocs+1; i++)
//            std::cout << split[i] << std::endl;
//        std::cout << std::endl;}

    long procNum = 0;
    // determine number of rows on each proc based on having almost the same number of nonzeros per proc.
    // -------------------------------------------
    for (unsigned int i=1; i<n_buckets; i++){
        if (H_g_scan[i] > (procNum+1)*nnz_g/nprocs){
            procNum++;
            split[procNum] = firstSplit[i];
        }
    }
    split[nprocs] = Mbig;
    split_old = split;

    H_g_scan.clear();
    H_g_scan.shrink_to_fit();
    firstSplit.clear();
    firstSplit.shrink_to_fit();

//    if (rank==0){
//        std::cout << std::endl << "split:" << std::endl;
//        for(unsigned int i=0; i<nprocs+1; i++)
//            std::cout << split[i] << std::endl;
//        std::cout << std::endl;}

    // set the number of rows for each process
    M = split[rank+1] - split[rank];

    if(verbose_repart && rank==0) printf("repartition - step 4!\n");

//    unsigned int M_min_global;
//    MPI_Allreduce(&M, &M_min_global, 1, MPI_UNSIGNED, MPI_MIN, comm);

    // *************************** exchange data ****************************

    long tempIndex;
    int* sendSizeArray = (int*)malloc(sizeof(int)*nprocs);
    std::fill(&sendSizeArray[0], &sendSizeArray[nprocs], 0);
    for (unsigned int i=0; i<initial_nnz_l; i++){
        tempIndex = lower_bound2(&split[0], &split[nprocs], entry[i].row);
        sendSizeArray[tempIndex]++;
    }

//    if (rank==0){
//        std::cout << "sendSizeArray:" << std::endl;
//        for(long i=0;i<nprocs;i++)
//            std::cout << sendSizeArray[i] << std::endl;
//    }

//    int recvSizeArray[nprocs];
    int* recvSizeArray = (int*)malloc(sizeof(int)*nprocs);
    MPI_Alltoall(sendSizeArray, 1, MPI_INT, recvSizeArray, 1, MPI_INT, comm);

//    if (rank==0){
//        std::cout << "recvSizeArray:" << std::endl;
//        for(long i=0;i<nprocs;i++)
//            std::cout << recvSizeArray[i] << std::endl;
//    }

//    int sOffset[nprocs];
    int* sOffset = (int*)malloc(sizeof(int)*nprocs);
    sOffset[0] = 0;
    for (int i=1; i<nprocs; i++)
        sOffset[i] = sendSizeArray[i-1] + sOffset[i-1];

//    if (rank==0){
//        std::cout << "sOffset:" << std::endl;
//        for(long i=0;i<nprocs;i++)
//            std::cout << sOffset[i] << std::endl;}

//    int rOffset[nprocs];
    int* rOffset = (int*)malloc(sizeof(int)*nprocs);
    rOffset[0] = 0;
    for (int i=1; i<nprocs; i++)
        rOffset[i] = recvSizeArray[i-1] + rOffset[i-1];

//    if (rank==0){
//        std::cout << "rOffset:" << std::endl;
//        for(long i=0;i<nprocs;i++)
//            std::cout << rOffset[i] << std::endl;}

    if(verbose_repart && rank==0) printf("repartition - step 5!\n");

    long procOwner;
    unsigned int bufTemp;
    cooEntry* sendBuf = (cooEntry*)malloc(sizeof(cooEntry)*initial_nnz_l);
    unsigned int* sIndex = (unsigned int*)malloc(sizeof(unsigned int)*nprocs);
    std::fill(&sIndex[0], &sIndex[nprocs], 0);

    // memcpy(sendBuf, data.data(), initial_nnz_l*3*sizeof(unsigned long));

    // todo: try to avoid this for loop.
    for (long i=0; i<initial_nnz_l; i++){
        procOwner = lower_bound2(&split[0], &split[nprocs], entry[i].row);
        bufTemp = sOffset[procOwner]+sIndex[procOwner];
//        memcpy(sendBuf+bufTemp, data.data() + 3*i, sizeof(cooEntry));
        memcpy(sendBuf+bufTemp, entry.data() + i, sizeof(cooEntry));
        // todo: the above line is better than the following three lines. think why it works.
//        sendBuf[bufTemp].row = data[3*i];
//        sendBuf[bufTemp].col = data[3*i+1];
//        sendBuf[bufTemp].val = data[3*i+2];
//        if(rank==1) std::cout << sendBuf[bufTemp].row << "\t" << sendBuf[bufTemp].col << "\t" << sendBuf[bufTemp].val << std::endl;
        sIndex[procOwner]++;
    }

    free(sIndex);

    // clear data and free memory.
//    data.clear();
//    data.shrink_to_fit();

//    if (rank==1){
//        std::cout << "sendBuf:" << std::endl;
//        for (long i=0; i<initial_nnz_l; i++)
//            std::cout << sendBuf[i] << "\t" << entry[i] << std::endl;
//    }

//    MPI_Barrier(comm);
//    if (rank==2){
//        std::cout << "\nrank = " << rank << ", nnz_l = " << nnz_l << std::endl;
//        for (int i=0; i<nnz_l; i++)
//            std::cout << i << "\t" << entry[i] << std::endl;}
//    MPI_Barrier(comm);

    nnz_l = rOffset[nprocs-1] + recvSizeArray[nprocs-1];
//    printf("rank=%d \t A.nnz_l=%u \t A.nnz_g=%u \n", rank, nnz_l, nnz_g);

//    cooEntry* entry = (cooEntry*)malloc(sizeof(cooEntry)*nnz_l);
//    cooEntry* entryP = &entry[0];

    if(verbose_repart && rank==0) printf("repartition - step 6!\n");

    entry.clear();
    entry.resize(nnz_l);
    entry.shrink_to_fit();

    MPI_Alltoallv(sendBuf, sendSizeArray, sOffset, cooEntry::mpi_datatype(), &entry[0], recvSizeArray, rOffset, cooEntry::mpi_datatype(), comm);

    free(sendSizeArray);
    free(recvSizeArray);
    free(sOffset);
    free(rOffset);
    free(sendBuf);

    std::sort(entry.begin(), entry.end());

//    MPI_Barrier(comm);
//    if (rank==2){
//        std::cout << "\nrank = " << rank << ", nnz_l = " << nnz_l << std::endl;
//        for (int i=0; i<nnz_l; i++)
//            std::cout << i << "\t" << entry[i] << std::endl;}
//    MPI_Barrier(comm);
//    if (rank==1){
//        std::cout << "\nrank = " << rank << ", nnz_l = " << nnz_l << std::endl;
//        for (int i=0; i<nnz_l; i++)
//            std::cout << "i=" << i << "\t" << entry[i].row << "\t" << entry[i].col << "\t" << entry[i].val << std::endl;}
//    MPI_Barrier(comm);

//    MPI_Barrier(comm); printf("repartition: rank = %d, Mbig = %u, M = %u, nnz_g = %u, nnz_l = %u \n", rank, Mbig, M, nnz_g, nnz_l); MPI_Barrier(comm);

    if(verbose_repart && rank==0) printf("repartition - step 7!\n");

    return 0;
}
*/


int saena_matrix::repart(){
    // repart_row: if true repartition based on number of rows, otherwise based on the number of nonzeros

    // summary: number of buckets are computed based of the number of <<rows>> and number of processors.
    // firstSplit[] is of size n_buckets+1 and is a row partition of the matrix with almost equal number of rows.
    // then the buckets (firsSplit) are combined to have almost the same number of nonzeros. This is split[].
    // note: this function includes cpu shrinking.

    int nprocs = -1, rank = -1;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    assert(!entry.empty());

    if(verbose_repart){
        MPI_Barrier(comm);
        printf("repart - step1! rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu \n",
               rank, Mbig, M, nnz_g, nnz_l);
        MPI_Barrier(comm);
    }

    // *************************** find splitters ****************************
    // split the matrix row-wise by splitters, so each processor get almost equal number of nonzeros
    // definition of buckets: bucket[i] = [ firstSplit[i] , firstSplit[i+1] ). Number of buckets = n_buckets
    // ***********************************************************************

    int n_buckets = nprocs;

    if(!repart_row) {
        if (Mbig > nprocs * nprocs) {
            if (nprocs < 1000)
                n_buckets = nprocs * nprocs;
            else
                n_buckets = 1000 * nprocs;
        } else if (nprocs <= Mbig) {
            n_buckets = Mbig;
        } else { // nprocs > Mbig
            // todo: it may be better to call shrink_cpu_minor, so then nprocs=Mbig and work only with the first Mbig processors.
            if (rank == 0)
                printf("number of MPI tasks (%d) cannot be greater than the number of rows of the matrix (%d).\n",
                       nprocs, Mbig);
            MPI_Finalize();
        }
    }

//    if (rank==0) std::cout << "n_buckets = " << n_buckets << ", Mbig = " << Mbig << std::endl;

    std::vector<index_t> splitOffset(n_buckets);
    auto baseOffset = int(floor(1.0 * Mbig / n_buckets));
    float offsetRes = float(1.0 * Mbig / n_buckets) - baseOffset;
//    if (rank==0) std::cout << "baseOffset = " << baseOffset << ", offsetRes = " << offsetRes << std::endl;
    float offsetResSum = 0;
    splitOffset[0] = 0;
    for(index_t i = 1; i < n_buckets; ++i){
        splitOffset[i] = baseOffset;
        offsetResSum += offsetRes;
        if (offsetResSum >= 1){
            splitOffset[i]++;
            offsetResSum -= 1;
        }
    }

//    print_vector(splitOffset, 0, "splitOffset", comm);

    if(!repart_row) {

        if (verbose_repart && rank == 0) printf("repart - step 2!\n");

        std::vector<index_t> firstSplit(n_buckets + 1);
        firstSplit[0] = 0;
        for (index_t i = 1; i < n_buckets; i++)
            firstSplit[i] = firstSplit[i - 1] + splitOffset[i];
        firstSplit[n_buckets] = Mbig;

#ifdef __DEBUG1__
//    print_vector(firstSplit, 0, "firstSplit", comm);;
#endif

        splitOffset.clear();
        splitOffset.shrink_to_fit();

        std::sort(entry.begin(), entry.end(), row_major);

        index_t least_bucket = 0, last_bucket = 0;
        least_bucket = lower_bound2(&firstSplit[0], &firstSplit[n_buckets], entry[0].row);
        last_bucket  = lower_bound2(&firstSplit[0], &firstSplit[n_buckets], entry.back().row);
        last_bucket++;

#ifdef __DEBUG1__
//    if (rank==0) std::cout << "least_bucket:" << least_bucket << ", last_bucket = " << last_bucket << std::endl;
#endif

        // H_l is the histogram of (local) nnz of buckets
        std::vector<nnz_t> H_l(n_buckets, 0);

        initial_nnz_l = nnz_l;
        // H_l is the histogram of (local) nnz per bucket
        for (nnz_t i = 0; i < nnz_l; i++) {
            least_bucket += lower_bound2(&firstSplit[least_bucket], &firstSplit[last_bucket], entry[i].row);
            H_l[least_bucket]++;
        }

//    print_vector(H_l, 0, "H_l", comm);

        // H_g is the histogram of (global) nnz per bucket
        std::vector<nnz_t> H_g(n_buckets);
        MPI_Allreduce(&H_l[0], &H_g[0], n_buckets, par::Mpi_datatype<nnz_t>::value(), MPI_SUM, comm);

        H_l.clear();
        H_l.shrink_to_fit();

//    print_vector(H_g, 0, "H_g", comm);

        std::vector<nnz_t> H_g_scan(n_buckets);
        H_g_scan[0] = H_g[0];
        for (nnz_t i = 1; i < n_buckets; i++)
            H_g_scan[i] = H_g[i] + H_g_scan[i - 1];

        H_g.clear();
        H_g.shrink_to_fit();

//    print_vector(H_g_scan, 0, "H_g_scan", comm);

        if (verbose_repart && rank == 0) printf("repart - step 3!\n");

        // -------------------------------------------
        // determine number of rows on each proc based on having almost the same number of nonzeros per proc.

        split_old = split;
        int procNum = 0;
        for (index_t i = 1; i < n_buckets; ++i) {
            // allocate at least 1 row to the remaining processors
            if (Mbig - firstSplit[i + 1] < nprocs - (procNum + 1)) {
                for (; i < n_buckets; ++i) {
                    procNum++;
                    split[procNum] = firstSplit[i];
                }
                break;
            }

            if (H_g_scan[i] > (procNum + 1) * nnz_g / nprocs) {
//            if(!rank) printf("%d, %d, %ld, %d, %d\n", i, H_g_scan[i], (procNum+1)*nnz_g/nprocs, procNum, firstSplit[i]);
                procNum++;
                split[procNum] = firstSplit[i];
            }
        }
        split[nprocs] = Mbig;
    } else {
        split_old = split;
        split[0] = 0;
        for(index_t i = 1; i < nprocs; ++i){
            split[i] = split[i-1] + splitOffset[i];
        }
        split[nprocs] = Mbig;
    }

    splitOffset.clear();
    splitOffset.shrink_to_fit();

//    print_vector(split, 0, "split", comm);

    // set the number of rows for each process
    M = split[rank+1] - split[rank];

    assert(M != 0);

    if(verbose_repart && rank == 0) printf("repart - step 4!\n");

    // *************************** exchange data ****************************

    // TODO: check if this is needed.
    if(repart_row){
        std::sort(entry.begin(), entry.end(), row_major);
    }

    if(nprocs > 1){
        index_t least_proc = 0, last_proc = 0;
        least_proc = lower_bound2(&split[0], &split[nprocs], entry[0].row);
        last_proc  = lower_bound2(&split[0], &split[nprocs], entry.back().row);
        last_proc++;

//        if (rank==1) std::cout << "\nleast_proc:" << least_proc << ", last_proc = " << last_proc << std::endl;

        std::vector<int> send_size_array(nprocs, 0);
        for (nnz_t i = 0; i < nnz_l; ++i){
            least_proc += lower_bound2(&split[least_proc], &split[last_proc], entry[i].row);
            send_size_array[least_proc]++;
        }

//        print_vector(send_size_array, 0, "send_size_array", comm);

        // this part is for cpu shrinking. assign all the rows on non-root procs to their roots.
        // ---------------------------------
//        if(enable_shrink && nprocs >= cpu_shrink_thre2 && (last_M_shrink >= (Mbig * cpu_shrink_thre1)) ){
//        if(rank==0) printf("last_density_shrink = %f, density = %f, inequality = %d \n", last_density_shrink, density, (density >= (last_density_shrink * cpu_shrink_thre1)));
        if( (enable_shrink || enable_shrink_c) && (nprocs >= cpu_shrink_thre2) && do_shrink){
            shrink_set_params(send_size_array);
        }

        std::vector<int> recv_size_array(nprocs);
        MPI_Alltoall(&send_size_array[0], 1, MPI_INT, &recv_size_array[0], 1, MPI_INT, comm);

//        print_vector(recv_size_array, -1, "recv_size_array", comm);

        std::vector<int> send_offset(nprocs);
        send_offset[0] = 0;
        for (int i = 1; i < nprocs; i++)
            send_offset[i] = send_size_array[i - 1] + send_offset[i - 1];

//        print_vector(send_offset, -1, "send_offset", comm);

        std::vector<int> recv_offset(nprocs);
        recv_offset[0] = 0;
        for (int i = 1; i < nprocs; i++)
            recv_offset[i] = recv_size_array[i - 1] + recv_offset[i - 1];

//        print_vector(recv_offset, 0, "recv_offset", comm);

        if (verbose_repart && rank == 0) printf("repart - step 5!\n");

        nnz_l = recv_offset[nprocs - 1] + recv_size_array[nprocs - 1];
//        printf("rank=%d \t A.nnz_l=%u \t A.nnz_g=%u \n", rank, nnz_l, nnz_g);

        if (verbose_repart && rank == 0) printf("repart - step 6!\n");

        std::vector<cooEntry> entry_old(std::move(entry));
        entry.resize(nnz_l);

        auto dt = cooEntry::mpi_datatype();

        MPI_Alltoallv(&entry_old[0], &send_size_array[0], &send_offset[0], dt,
                      &entry[0],     &recv_size_array[0], &recv_offset[0], dt, comm);

        MPI_Type_free(&dt);
    }

    std::sort(entry.begin(), entry.end());
//    print_vector(entry, -1, "entry", comm);

    if(verbose_repart) {
        MPI_Barrier(comm);
        printf("repart - end! rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu \n",
               rank, Mbig, M, nnz_g, nnz_l);
        MPI_Barrier(comm);
    }

    return 0;
}

// int saena_matrix::repartition_nnz()
#if 0
int saena_matrix::repartition_nnz(){

    // summary: number of buckets are computed based of the number of <<rows>> and number of processors.
    // firstSplit[] is of size n_buckets+1 and is a row partition of the matrix with almost equal number of rows.
    // then the buckets (firsSplit) are combined to have almost the same number of nonzeros. This is split[].
    // note: this function includes cpu shrinking.

    int nprocs = -1, rank = -1;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    assert(!entry.empty());

    if(verbose_repart){
        MPI_Barrier(comm);
        printf("repartition_nnz - step1! rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu \n",
               rank, Mbig, M, nnz_g, nnz_l);
        MPI_Barrier(comm);
    }

    // *************************** find splitters ****************************
    // split the matrix row-wise by splitters, so each processor get almost equal number of nonzeros
    // definition of buckets: bucket[i] = [ firstSplit[i] , firstSplit[i+1] ). Number of buckets = n_buckets
    // ***********************************************************************

    int n_buckets = 0;
    if (Mbig > nprocs*nprocs){
        if (nprocs < 1000)
            n_buckets = nprocs*nprocs;
        else
            n_buckets = 1000*nprocs;
    }
    else if(nprocs <= Mbig){
        n_buckets = Mbig;
    } else{ // nprocs > Mbig
        // todo: it may be better to call shrink_cpu_minor, so then nprocs=Mbig and work only with the first Mbig processors.
        if(rank == 0)
            printf("number of MPI tasks (%d) cannot be greater than the number of rows of the matrix (%d).\n", nprocs, Mbig);
        MPI_Finalize();
    }

//    if (rank==0) std::cout << "n_buckets = " << n_buckets << ", Mbig = " << Mbig << std::endl;

    std::vector<index_t> splitOffset(n_buckets);
    auto baseOffset = int(floor(1.0 * Mbig / n_buckets));
    float offsetRes = float(1.0 * Mbig / n_buckets) - baseOffset;
//    if (rank==0) std::cout << "baseOffset = " << baseOffset << ", offsetRes = " << offsetRes << std::endl;
    float offsetResSum = 0;
    splitOffset[0] = 0;
    for(index_t i=1; i<n_buckets; i++){
        splitOffset[i] = baseOffset;
        offsetResSum += offsetRes;
        if (offsetResSum >= 1){
            splitOffset[i]++;
            offsetResSum -= 1;
        }
    }

//    print_vector(splitOffset, 0, "splitOffset", comm);

    if(verbose_repart && rank == 0) printf("repartition_nnz - step 2!\n");

    std::vector<index_t> firstSplit(n_buckets + 1);
    firstSplit[0] = 0;
    for(index_t i=1; i<n_buckets; i++)
        firstSplit[i] = firstSplit[i - 1] + splitOffset[i];
    firstSplit[n_buckets] = Mbig;

#ifdef __DEBUG1__
//    print_vector(firstSplit, 0, "firstSplit", comm);;
#endif

    splitOffset.clear();
    splitOffset.shrink_to_fit();

    std::sort(entry.begin(), entry.end(), row_major);

    index_t least_bucket = 0, last_bucket = 0;
    least_bucket = lower_bound2(&firstSplit[0], &firstSplit[n_buckets], entry[0].row);
    last_bucket  = lower_bound2(&firstSplit[0], &firstSplit[n_buckets], entry.back().row);
    last_bucket++;

#ifdef __DEBUG1__
//    if (rank==0) std::cout << "least_bucket:" << least_bucket << ", last_bucket = " << last_bucket << std::endl;
#endif

    // H_l is the histogram of (local) nnz of buckets
    std::vector<index_t > H_l(n_buckets, 0);

    initial_nnz_l = nnz_l;
    // H_l is the histogram of (local) nnz per bucket
    for(nnz_t i=0; i<initial_nnz_l; i++){
        least_bucket += lower_bound2(&firstSplit[least_bucket], &firstSplit[last_bucket], entry[i].row);
        H_l[least_bucket]++;
    }

//    print_vector(H_l, 0, "H_l", comm);

    // H_g is the histogram of (global) nnz per bucket
    std::vector<index_t> H_g(n_buckets);
    MPI_Allreduce(&H_l[0], &H_g[0], n_buckets, par::Mpi_datatype<index_t>::value(), MPI_SUM, comm);

    H_l.clear();
    H_l.shrink_to_fit();

//    print_vector(H_g, 0, "H_g", comm);

    std::vector<index_t > H_g_scan(n_buckets);
    H_g_scan[0] = H_g[0];
    for (index_t i=1; i<n_buckets; i++)
        H_g_scan[i] = H_g[i] + H_g_scan[i-1];

    H_g.clear();
    H_g.shrink_to_fit();

//    print_vector(H_g_scan, 0, "H_g_scan", comm);

    if(verbose_repart && rank == 0) printf("repartition_nnz - step 3!\n");

    // -------------------------------------------
    // determine number of rows on each proc based on having almost the same number of nonzeros per proc.

    split_old = split;
    int procNum = 0;
    for (index_t i = 1; i < n_buckets; ++i){
        // allocate at least 1 row to the remaining processors
        if(Mbig - firstSplit[i + 1] < nprocs - (procNum + 1)){
            for(; i < n_buckets; ++i){
                procNum++;
                split[procNum] = firstSplit[i];
            }
            break;
        }

        if (H_g_scan[i] > (procNum+1)*nnz_g/nprocs){
//            if(!rank) printf("%d, %d, %ld, %d, %d\n", i, H_g_scan[i], (procNum+1)*nnz_g/nprocs, procNum, firstSplit[i]);
            procNum++;
            split[procNum] = firstSplit[i];
        }
    }
    split[nprocs] = Mbig;

//    print_vector(split, 0, "split", comm);

    H_g_scan.clear();
    H_g_scan.shrink_to_fit();
    firstSplit.clear();
    firstSplit.shrink_to_fit();

    // set the number of rows for each process
    M = split[rank+1] - split[rank];

    assert(M != 0);

    if(verbose_repart && rank == 0) printf("repartition_nnz - step 4!\n");

    // *************************** exchange data ****************************

    if(nprocs > 1){
        std::vector<int> send_size_array(nprocs, 0);
//        for (unsigned int i=0; i<initial_nnz_l; i++){
//            tempIndex = lower_bound2(&split[0], &split[nprocs], entry[i].row);
//            sendSizeArray[tempIndex]++;
//        }

        long least_proc, last_proc;
        least_proc = lower_bound2(&split[0], &split[nprocs], entry[0].row);
        last_proc  = lower_bound2(&split[0], &split[nprocs], entry.back().row);
        last_proc++;

//        if (rank==1) std::cout << "\nleast_proc:" << least_proc << ", last_proc = " << last_proc << std::endl;

        for (nnz_t i = 0; i < initial_nnz_l; ++i){
            least_proc += lower_bound2(&split[least_proc], &split[last_proc], entry[i].row);
            send_size_array[least_proc]++;
        }

//        print_vector(send_size_array, 0, "send_size_array", comm);

        // this part is for cpu shrinking. assign all the rows on non-root procs to their roots.
        // ---------------------------------
//        if(enable_shrink && nprocs >= cpu_shrink_thre2 && (last_M_shrink >= (Mbig * cpu_shrink_thre1)) ){
//        if(rank==0) printf("last_density_shrink = %f, density = %f, inequality = %d \n", last_density_shrink, density, (density >= (last_density_shrink * cpu_shrink_thre1)));
        if( (enable_shrink || enable_shrink_c) && (nprocs >= cpu_shrink_thre2) && do_shrink){
            shrink_set_params(send_size_array);
        }

        std::vector<int> recv_size_array(nprocs);
        MPI_Alltoall(&send_size_array[0], 1, MPI_INT, &recv_size_array[0], 1, MPI_INT, comm);

//        print_vector(recv_size_array, -1, "recv_size_array", comm);

        std::vector<int> send_offset(nprocs);
        send_offset[0] = 0;
        for (int i = 1; i < nprocs; i++)
            send_offset[i] = send_size_array[i - 1] + send_offset[i - 1];

//        print_vector(send_offset, -1, "send_offset", comm);

        std::vector<int> recv_offset(nprocs);
        recv_offset[0] = 0;
        for (int i = 1; i < nprocs; i++)
            recv_offset[i] = recv_size_array[i - 1] + recv_offset[i - 1];

//        print_vector(recv_offset, 0, "recv_offset", comm);

        if (verbose_repart && rank == 0) printf("repartition_nnz - step 5!\n");

        nnz_l = recv_offset[nprocs - 1] + recv_size_array[nprocs - 1];
//        printf("rank=%d \t A.nnz_l=%u \t A.nnz_g=%u \n", rank, nnz_l, nnz_g);

        if (verbose_repart && rank == 0) printf("repartition_nnz - step 6!\n");

        std::vector<cooEntry> entry_old(std::move(entry));
        entry.resize(nnz_l);

        MPI_Alltoallv(&entry_old[0], &send_size_array[0], &send_offset[0], cooEntry::mpi_datatype(),
                      &entry[0],     &recv_size_array[0], &recv_offset[0], cooEntry::mpi_datatype(), comm);

    }

    std::sort(entry.begin(), entry.end());
//    print_vector(entry, -1, "entry", comm);

    if(verbose_repart) {
        MPI_Barrier(comm);
        printf("repartition_nnz - end! rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu \n",
               rank, Mbig, M, nnz_g, nnz_l);
        MPI_Barrier(comm);
    }

    return 0;
}
#endif

// int saena_matrix::repartition_row()
#if 0
int saena_matrix::repartition_row(){

    // summary: number of buckets are computed based of the number fo rows and number of processors.
    // firstSplit[] is of size n_buckets+1 and is a row partition of the matrix with almost equal number of rows.
    // then the buckets (firsSplit) are combined to have almost the same number of nonzeros. This is split[].
    // note: this version of repartition4() is WITH cpu shrinking.

    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if(rank==0) printf("\nuse repartition based on the number of rows for the next level!\n");
    if(verbose_repart && rank == 0) printf("repartition4 - step 1!\n");

//    MPI_Barrier(comm);
//    printf("repartition4 - start! rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu \n",
//           rank, Mbig, M, nnz_g, nnz_l);
//    MPI_Barrier(comm);

    // *************************** find splitters ****************************
    // split the matrix row-wise by splitters, so each processor gets almost equal number of rows

    std::vector<index_t> splitOffset(nprocs);
    auto baseOffset = int(floor(1.0 * Mbig / nprocs));
    float offsetRes = float(1.0 * Mbig / nprocs) - baseOffset;
//    if (rank==0) std::cout << "baseOffset = " << baseOffset << ", offsetRes = " << offsetRes << std::endl;
    float offsetResSum = 0;
    splitOffset[0] = 0;
    for(index_t i=1; i<nprocs; i++){
        splitOffset[i] = baseOffset;
        offsetResSum += offsetRes;
        if (offsetResSum >= 1){
            splitOffset[i]++;
            offsetResSum -= 1;
        }
    }

//    print_vector(splitOffset, 0, "splitOffset", comm);

    if(verbose_repart && rank == 0) printf("repartition4 - step 2!\n");

    split_old = split;
    split[0] = 0;
    for(index_t i=1; i<nprocs; i++){
        split[i] = split[i-1] + splitOffset[i];
    }
    split[nprocs] = Mbig;

    splitOffset.clear();
    splitOffset.shrink_to_fit();

//    print_vector(split, 0, "split", comm);

    // set the number of rows for each process
    M = split[rank+1] - split[rank];
//    M_old = M;

    if(verbose_repart && rank == 0) printf("repartition4 - step 4!\n");

//    unsigned int M_min_global;
//    MPI_Allreduce(&M, &M_min_global, 1, MPI_UNSIGNED, MPI_MIN, comm);

    // *************************** exchange data ****************************

    std::sort(entry.begin(), entry.end(), row_major);

    long least_proc = 0, last_proc = 0;
    least_proc = lower_bound2(&split[0], &split[nprocs], entry[0].row);
    last_proc  = lower_bound2(&split[0], &split[nprocs], entry.back().row);
    last_proc++;

//    if (rank==1) std::cout << "\nleast_proc:" << least_proc << ", last_proc = " << last_proc << std::endl;

    std::vector<int> send_size_array(nprocs, 0);
    for (nnz_t i=0; i<nnz_l; i++){
        least_proc += lower_bound2(&split[least_proc], &split[last_proc], entry[i].row);
        send_size_array[least_proc]++;
    }

//    print_vector(send_size_array, 0, "send_size_array", comm);

    // this part is for cpu shrinking. assign all the rows on non-root procs to their roots.
    // ---------------------------------
//    if(enable_shrink && nprocs >= cpu_shrink_thre2 && (last_M_shrink >= (Mbig * cpu_shrink_thre1)) ){
//    if(rank==0) printf("last_density_shrink = %f, density = %f, inequality = %d \n", last_density_shrink, density, (density >= (last_density_shrink * cpu_shrink_thre1)));
    if((enable_shrink || enable_shrink_c) && (nprocs >= cpu_shrink_thre2) && do_shrink){
        shrink_set_params(send_size_array);
    }

    std::vector<int> recv_size_array(nprocs);
    MPI_Alltoall(&send_size_array[0], 1, MPI_INT, &recv_size_array[0], 1, MPI_INT, comm);

//    print_vector(recv_size_array, 0, "recv_size_array", comm);

    std::vector<int> send_offset(nprocs);
    send_offset[0] = 0;
    for (int i=1; i<nprocs; i++)
        send_offset[i] = send_size_array[i-1] + send_offset[i-1];

//    print_vector(send_offset, 0, "send_offset", comm);

    std::vector<int> recv_offset(nprocs);
    recv_offset[0] = 0;
    for (int i=1; i<nprocs; i++)
        recv_offset[i] = recv_size_array[i-1] + recv_offset[i-1];

//    print_vector(recv_offset, 0, "recv_offset", comm);

    if(verbose_repart && rank == 0) printf("repartition4 - step 5!\n");

    nnz_l = recv_offset[nprocs-1] + recv_size_array[nprocs-1];
//    printf("rank=%d \t A.nnz_l=%lu \t A.nnz_g=%lu \n", rank, nnz_l, nnz_g);

    if(verbose_repart && rank == 0) printf("repartition4 - step 6!\n");

    std::vector<cooEntry> entry_old(std::move(entry));
    entry.resize(nnz_l);

    MPI_Alltoallv(&entry_old[0], &send_size_array[0], &send_offset[0], cooEntry::mpi_datatype(),
                  &entry[0],     &recv_size_array[0], &recv_offset[0], cooEntry::mpi_datatype(), comm);

    std::sort(entry.begin(), entry.end());

//    print_vector(entry, -1, "entry", comm);

    if(verbose_repart) {
        MPI_Barrier(comm);
        printf("repartition4 - step 7! rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu \n",
               rank, Mbig, M, nnz_g, nnz_l);
        MPI_Barrier(comm);
    }

    return 0;
}
#endif


int saena_matrix::repartition_nnz_update_Ac(){

    // summary: number of buckets are computed based of the number fo rows and number of processors.
    // firstSplit[] is of size n_buckets+1 and is a row partition of the matrix with almost equal number of rows.
    // then the buckets (firsSplit) are combined to have almost the same number of nonzeros. This is split[].
    // note: this version of repartition3() is WITH cpu shrinking.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(verbose_repart && rank == 0) printf("repartition5 - step 1!\n");

    density = (nnz_g / double(Mbig)) / (Mbig);

    if(!rank) printf("entry_temp should be set before calling repartition_nnz_update_Ac()\n");
    MPI_Abort(comm, 1);

#ifdef __DEBUG1__
//    MPI_Barrier(comm);
//    printf("repartition5 - start! rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu, entry_temp.size before repart = %lu \n",
//           rank, Mbig, M, nnz_g, nnz_l, entry_temp.size());
//    MPI_Barrier(comm);
//    print_vector(split, 0, "split", comm);
//    if(Mbig == 48) print_vector(entry, -1, "entry", comm);
//    if(Mbig == 48) print_vector(entry_temp, -1, "entry_temp", comm);
#endif

    // *************************** exchange data ****************************
#if 0
    std::sort(entry_temp.begin(), entry_temp.end(), row_major);

    long least_proc = 0, last_proc = nprocs-1;
    if(!entry_temp.empty()){
        least_proc = lower_bound2(&split[0], &split[nprocs], entry_temp[0].row);
        last_proc  = lower_bound2(&split[0], &split[nprocs], entry_temp.back().row);
        last_proc++;
    }

//    if (rank==0) std::cout << "\nleast_proc:" << least_proc << ", last_proc = " << last_proc << std::endl;

    std::vector<int> send_size_array(nprocs, 0);
    // add the first element to send_size_array
    send_size_array[least_proc]++;
    for (nnz_t i = 1; i < entry_temp.size(); i++){
//        least_proc += lower_bound2(&split[least_proc], &split[last_proc], entry_temp[i].row);
        if(entry_temp[i].row >= split[least_proc+1]){
            least_proc += lower_bound2(&split[least_proc], &split[last_proc], entry_temp[i].row);
        }
        send_size_array[least_proc]++;
    }

    if(verbose_repart && rank==0) printf("repartition5 - step 2!\n");

//    print_vector(send_size_array, -1, "send_size_array", comm);

    std::vector<int> recv_size_array(nprocs);
    MPI_Alltoall(&send_size_array[0], 1, MPI_INT, &recv_size_array[0], 1, MPI_INT, comm);

//    print_vector(recv_size_array, -1, "recv_size_array", comm);

//    int* sOffset = (int*)malloc(sizeof(int)*nprocs);
    std::vector<int> send_offset(nprocs);
    send_offset[0] = 0;
    for (int i=1; i<nprocs; i++)
        send_offset[i] = send_size_array[i-1] + send_offset[i-1];

//    print_vector(send_offset, -1, "send_offset", comm);

//    int* rOffset = (int*)malloc(sizeof(int)*nprocs);
    std::vector<int> recv_offset(nprocs);
    recv_offset[0] = 0;
    for (int i=1; i<nprocs; i++)
        recv_offset[i] = recv_size_array[i-1] + recv_offset[i-1];

//    print_vector(recv_offset, -1, "recv_offset", comm);

    if(verbose_repart && rank==0) printf("repartition5 - step 3!\n");

    nnz_t recv_size = recv_offset[nprocs-1] + recv_size_array[nprocs-1];
//    printf("rank=%d \t recv_size=%lu \t A.nnz_g=%lu \tremote size = %lu \n", rank, recv_size, nnz_g, row_remote.size());

    std::vector<cooEntry> entry_old = entry_temp;
    entry_temp.resize(recv_size);
//    entry.shrink_to_fit();

    MPI_Alltoallv(&entry_old[0],  &send_size_array[0], &send_offset[0], cooEntry::mpi_datatype(),
                  &entry_temp[0], &recv_size_array[0], &recv_offset[0], cooEntry::mpi_datatype(), comm);

    std::sort(entry_temp.begin(), entry_temp.end());

//    print_vector(entry_temp, -1, "entry_temp", comm);
    if(verbose_repart && rank==0) printf("repartition5 - step 4!\n");

    // copy the entries into a std::set to have O(logn) (?) for finding elements, since it will be sorted.
    std::set<cooEntry> entry_set(entry.begin(), entry.end());
//    print_vector(entry, -1, "entry before update", comm);

    // update the new entries.
    // entry_set: the current entries
    // entry_temp: new entries
    std::pair<std::set<cooEntry>::iterator, bool> p;
    cooEntry temp_old, temp_new;
    for(nnz_t i = 0; i < entry_temp.size(); i++){
        if(!almost_zero(entry_temp[i].val)){
            p = entry_set.insert(entry_temp[i]);

            if (!p.second){
                temp_old = *(p.first);
                temp_new = entry_temp[i];
                temp_new.val += temp_old.val;

                auto hint = p.first; // std::set<cooEntry>::iterator
                hint++;
                entry_set.erase(p.first);
                entry_set.insert(hint, temp_new);
            } else {
                std::string msg = "\nerror on rank "; msg += std::to_string(rank);
                msg += ": entry to update is not available in repartition_nnz_update_Ac(): ";
                msg += std::to_string(entry_temp[i].row); msg += " "; msg += std::to_string(entry_temp[i].col);
                msg += " "; msg += std::to_string(entry_temp[i].val);
                std::cout << std::setprecision(12) << msg << std::endl;
//                std::cout << "error on rank " << rank << ": entry to update is not available in repartition_nnz_update_Ac(): \n"
//                          << "this entry: " << entry_temp[i] << std::endl << std::endl;
            }
        }
    }

    // this part replaces the current entry with the new entry.
//    for(nnz_t i = 0; i < entry_temp.size(); i++) {
//        p = entry_set.insert(entry_temp[i]);
    // in the case of duplicate, if the new value is zero, remove the older one and don't insert the zero.
//        if (!p.second) {
//            auto hint = p.first; // hint is std::set<cooEntry>::iterator
//            hint++;
//            entry_set.erase(p.first);
//            if (!almost_zero(entry_temp[i].val))
//                entry_set.insert(hint, entry_temp[i]);
//        }
    // if the entry is zero and it was not a duplicate, just erase it.
//        if (p.second && almost_zero(entry_temp[i].val))
//            entry_set.erase(p.first);
//    }

    if(verbose_repart && rank==0) printf("repartition5 - step 6!\n");

//    printf("rank %d: entry.size = %lu, entry_set.size = %lu \n", rank, entry.size(), entry_set.size());

//    entry.resize(entry_set.size());
//    std::copy(entry_set.begin(), entry_set.end(), entry.begin());

    nnz_t it2 = 0;
    std::set<cooEntry>::iterator it;
    for(it=entry_set.begin(); it!=entry_set.end(); ++it){
//        std::cout << *it << std::endl;
        entry[it2] = *it;
        it2++;
    }

//    if(Mbig == 48) print_vector(entry, -1, "entry", comm);
//    print_vector(entry, -1, "entry", comm);

//    entry_temp.clear();
//    entry_temp.shrink_to_fit();

    if(verbose_repart) {
        MPI_Barrier(comm);
        printf("repartition5 - end! rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu \n",
               rank, Mbig, M, nnz_g, nnz_l);
        MPI_Barrier(comm);
    }
//    MPI_Barrier(comm);
//    printf("repartition5 - end! rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu \n",
//           rank, Mbig, M, nnz_g, nnz_l);
//    MPI_Barrier(comm);
#endif
    return 0;
}