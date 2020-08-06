#include "strength_matrix.h"
#include "data_struct.h"
#include "aux_functions.h"
#include "parUtils.h"


/*
// sort indices and store the ordering.
class sort_indices
{
private:
    index_t *mparr;
public:
    sort_indices(index_t *parr) : mparr(parr) {}
    bool operator()(index_t i, index_t j) const { return mparr[i]<mparr[j]; }
};

// binary search tree using the lower bound
template <class T>
T lower_bound2(T *left, T *right, T val) {
    T* first = left;
    while (left < right) {
        T *middle = left + (right - left) / 2;
        if (*middle < val){
            left = middle + 1;
        }
        else{
            right = middle;
        }
    }
    if(val == *left){
        return distance(first, left);
    }
    else
        return distance(first, left-1);
}
*/


int strength_matrix::set_parameters(index_t m1, index_t m2, std::vector<index_t> &spl, MPI_Comm com){
    M = m1;
    Mbig = m2;
    split = spl;
    comm = com;
    return 0;
}


int strength_matrix::setup_matrix(float connStrength){

    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // *************************** make S symmetric and apply the connection strength parameter ****************************

//    print_vector(entry, -1, "entry", comm);
//    print_vector(entryT, -1, "entryT", comm);

//    int val_coeff = 0, valT_coeff = 0;
    std::vector<index_t> r;
    std::vector<index_t> c;
    std::vector<value_t> v;

    //TODO: check this part
#if 0
    for(nnz_t i = 0; i < entry.size(); i++){

        if (entry[i].val <= connStrength)
            val_coeff = 0;
        else
            val_coeff = 1;

        if (entryT[i].val <= connStrength)
            valT_coeff = 0;
        else
            valT_coeff = 1;

        if (!val_coeff && !valT_coeff) // if both zero, skip.
            continue;

        r[nnz_l] = entryT[i].row;
        c[nnz_l] = entryT[i].col;
        v[nnz_l] = 0.5*(val_coeff * entry[i].val + valT_coeff * entryT[i].val);
        nnz_l++;
    }
#endif

    for(nnz_t i = 0; i < entry.size(); i++){
//        if(!rank) printf("S(%d, %d) \t= %f, %f\n", entryT[i].row, entryT[i].col, entryT[i].val, entry[i].val);
        if (entry[i].val > connStrength){
            r.emplace_back(entryT[i].row);
            c.emplace_back(entryT[i].col);
            v.emplace_back( 0.5 * (entry[i].val + entryT[i].val) );
        }
    }

    nnz_l = r.size();
    MPI_Allreduce(&nnz_l, &nnz_g, 1, par::Mpi_datatype<nnz_t>::value(), MPI_SUM, comm);
//    if(rank==0) printf("S.nnz_l = %lu, S.nnz_g = %lu \n", nnz_l, nnz_g);

//    print_vector(r, -1, "row vector", comm);
//    print_vector(c, -1, "col vector", comm);
//    print_vector(v, -1, "val vector", comm);
//    if(rank==1){
//        for(index_t i = 0; i < r.size(); i++){
//            printf("%u \t%u \t%u \t%lf \n", i, r[i], c[i], v[i]);
//        }
//    }

    // *************************** setup the matrix ****************************

    long procNum = 0;
    nnz_t i = 0;
    col_remote_size = 0; // number of remote columns
    nnz_l_local = 0;
    nnz_l_remote = 0;
    std::vector<int> recvCount(nprocs, 0);
    nnzPerRow.assign(M,0);
    nnzPerRow_local.assign(M,0);

    // take care of the first element here, since there is "col[i-1]" in the for loop below, so "i" cannot start from 0.
    nnzPerRow[r[0]]++;
    if (c[0] >= split[rank] && c[0] < split[rank + 1]) {
        nnzPerRow_local[r[0]]++;
        nnz_l_local++;

        values_local.emplace_back(v[0]);
        row_local.emplace_back(r[0]);
        col_local.emplace_back(c[0]);

        //vElement_local.emplace_back(col[0]);
        vElementRep_local.emplace_back(1);
    } else{
        ++nnz_l_remote;

        values_remote.emplace_back(v[0]);
        row_remote.emplace_back(r[0]);
        col_remote_size++; // number of remote columns
        col_remote.emplace_back(col_remote_size-1);
        col_remote2.emplace_back(c[0]);
        nnzPerCol_remote.emplace_back(1);

        vElement_remote.emplace_back(c[0]);
        vElementRep_remote.emplace_back(1);
        recvCount[lower_bound2(&split[0], &split[nprocs], c[0])] = 1;
    }

    for (i = 1; i < nnz_l; ++i) {
        ++nnzPerRow[r[i]];

        if (c[i] >= split[rank] && c[i] < split[rank+1]) {
            ++nnzPerRow_local[r[i]];
            ++nnz_l_local;

            values_local.emplace_back(v[i]);
            row_local.emplace_back(r[i]);
            col_local.emplace_back(c[i]);

        } else {
            ++nnz_l_remote;
            values_remote.emplace_back(v[i]);
            row_remote.emplace_back(r[i]);
            // col_remote2 is the original col value and will be used in making strength matrix. col_remote will be used for matevec.
            col_remote2.emplace_back(c[i]);

            if (c[i] != c[i - 1]) {
                ++col_remote_size;
                vElement_remote.emplace_back(c[i]);
                vElementRep_remote.emplace_back(1);
                procNum = lower_bound2(&split[0], &split[nprocs], c[i]);
                ++recvCount[procNum];
                nnzPerCol_remote.emplace_back(1);
            } else {
                ++vElementRep_remote.back();
                ++nnzPerCol_remote.back();
            }
            // the original col values are not being used for matvec. the ordering starts from 0, and goes up by 1.
            col_remote.emplace_back(col_remote_size-1);
//            nnzPerCol_remote[col_remote_size-1]++;
        }
    } // for i

     if(nprocs > 1){
         std::vector<int> vIndexCount(nprocs);
         MPI_Alltoall(&recvCount[0], 1, par::Mpi_datatype<index_t>::value(), &vIndexCount[0], 1, par::Mpi_datatype<index_t>::value(), comm);

         numRecvProc = 0;
         numSendProc = 0;
         for(int j = 0; j < nprocs; j++){
             if(recvCount[j] != 0){
                 numRecvProc++;
                 recvProcRank.emplace_back(j);
                 recvProcCount.emplace_back(2*recvCount[j]); // make them double size for prolongation the communication in the aggregation_2_dist function.
             }
             if(vIndexCount[j] != 0){
                 numSendProc++;
                 sendProcRank.emplace_back(j);
                 sendProcCount.emplace_back(2*vIndexCount[j]); // make them double size for prolongation the communication in the aggregation_2_dist function.
             }
         }

//        if (rank==0) cout << "rank=" << rank << ", numRecvProc=" << numRecvProc << ", numSendProc=" << numSendProc << endl;

         vdispls.resize(nprocs);
         rdispls.resize(nprocs);
         vdispls[0] = 0;
         rdispls[0] = 0;

         for (int j = 1; j < nprocs; j++){
             vdispls[j] = vdispls[j-1] + vIndexCount[j-1];
             rdispls[j] = rdispls[j-1] + recvCount[j-1];
         }

         vIndexSize = vdispls[nprocs-1] + vIndexCount[nprocs-1];
         recvSize   = rdispls[nprocs-1] + recvCount[nprocs-1];

         vIndex.resize(vIndexSize);
         MPI_Alltoallv(&vElement_remote[0], &recvCount[0],   &rdispls[0], par::Mpi_datatype<index_t>::value(),
                       &vIndex[0],          &vIndexCount[0], &vdispls[0], par::Mpi_datatype<index_t>::value(), comm);

         // vSend = vector values to send to other procs
         // vecValues = vector values that received from other procs
         // These will be used in matvec and they are set here to reduce the time of matvec.
//         vSend.resize(2*vIndexSize); // make them double size for prolongation the communication in the aggregation_2_dist function.
//         vecValues.resize(2*recvSize); // make them double size for prolongation the communication in the aggregation_2_dist function.

         // make them double size for prolongation the communication in the aggregation_1_dist function.
         for (int j = 1; j < nprocs; j++){
             vdispls[j] = 2 * vdispls[j];
             rdispls[j] = 2 * rdispls[j];
         }

//         print_vector(vIndex, -1, "vIndex", comm);

         // change the indices from global to local
         #pragma omp parallel for
         for (index_t j = 0; j < vIndexSize; j++)
             vIndex[j] -= split[rank];
    }

    indicesP_local.resize(nnz_l_local);
    for(nnz_t j = 0; j < nnz_l_local; j++)
        indicesP_local[j] = j;

    index_t *row_localP = &*row_local.begin();
    std::sort(&indicesP_local[0], &indicesP_local[nnz_l_local], sort_indices(row_localP));

//    indicesP_remote = (unsigned long*)malloc(sizeof(unsigned long)*nnz_l_remote);
//    for(i=0; i<nnz_l_remote; i++)
//        indicesP_remote[i] = i;
//    unsigned long* row_remoteP = &(*(row_remote.begin()));
//    std::sort(indicesP_remote, &indicesP_remote[nnz_l_remote], sort_indices(row_remoteP));

//    print_info(-1);
//    print_entry(-1);

    return 0;
}


strength_matrix::~strength_matrix() = default;


int strength_matrix::erase(){
    M    = 0;
    Mbig = 0;
    nnz_l  = 0;
    nnz_l_local  = 0;
    nnz_l_remote = 0;
    col_remote_size = 0;
    vIndexSize = 0;
    recvSize = 0;
    numRecvProc = 0;
    numSendProc = 0;

    vIndex.clear();
    split.clear();
    values_local.clear();
    values_remote.clear();
    row_local.clear();
    row_remote.clear();
    col_local.clear();
    col_remote.clear();
    col_remote2.clear();
    nnzPerRow.clear();
    nnzPerRow_local.clear();
    nnzPerCol_remote.clear();
    vElement_remote.clear();
    vElementRep_local.clear();
    vElementRep_remote.clear();
    indicesP_local.clear();
    indicesP_remote.clear();
    vdispls.clear();
    rdispls.clear();
    recvProcRank.clear();
    recvProcCount.clear();
    sendProcRank.clear();
    sendProcCount.clear();
    return 0;
}


int strength_matrix::erase_update(){
//    M    = 0;
//    Mbig = 0;
    nnz_l  = 0;
    nnz_l_local  = 0;
    nnz_l_remote = 0;
    col_remote_size = 0;
    vIndexSize = 0;
    recvSize = 0;
    numRecvProc = 0;
    numSendProc = 0;

//    split.clear();
    vIndex.clear();
    values_local.clear();
    values_remote.clear();
    row_local.clear();
    row_remote.clear();
    col_local.clear();
    col_remote.clear();
    col_remote2.clear();
    nnzPerRow.clear();
    nnzPerRow_local.clear();
    nnzPerCol_remote.clear();
    vElement_remote.clear();
    vElementRep_local.clear();
    vElementRep_remote.clear();
    indicesP_local.clear();
    indicesP_remote.clear();
    vdispls.clear();
    rdispls.clear();
    recvProcRank.clear();
    recvProcCount.clear();
    sendProcRank.clear();
    sendProcCount.clear();
    return 0;
}


void strength_matrix::print_entry(int ran){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::vector<cooEntry> entry(nnz_l);
    for(index_t i = 0; i < nnz_l_local; i++){
        entry[i] = cooEntry(row_local[i]+split[rank], col_local[i], values_local[i]);
    }

    for(index_t i = 0; i < nnz_l_remote; i++){
        entry[nnz_l_local+i] = cooEntry(row_remote[i]+split[rank], col_remote2[i], values_remote[i]);
    }

    std::sort(entry.begin(), entry.end());

    index_t iter = 0;
    if(ran >= 0) {
        if (rank == ran) {
            printf("\nstrength matrix on proc = %d \n", ran);
            printf("nnz = %lu \n", nnz_l);
            for (index_t i = 0; i < nnz_l; i++) {
                std::cout << iter << "\t" << entry[i] << std::endl;
                iter++;
            }
        }
    } else{
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("\nstrength matrix on proc = %d \n", rank);
                printf("nnz = %lu \n", nnz_l);
                for (index_t i = 0; i < nnz_l; i++) {
                    std::cout << iter << "\t" << entry[i] << std::endl;
                    iter++;
                }
            }
            MPI_Barrier(comm);
        }
    }
}


void strength_matrix::print_diagonal_block(int ran){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    index_t iter = 0;
    if(ran >= 0) {
        if (rank == ran) {
            printf("\nstrength matrix (diagonal_block) on proc = %d \n", ran);
            printf("nnz = %lu \n", nnz_l_local);
            for (index_t i = 0; i < nnz_l_local; i++) {
                std::cout << iter << "\t" << row_local[i] + split[rank] << "\t" << col_local[i] << "\t" << values_local[i] << std::endl;
                iter++;
            }
        }
    } else{
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("\nstrength matrix (diagonal_block) on proc = %d \n", rank);
                printf("nnz = %lu \n", nnz_l_local);
                for (index_t i = 0; i < nnz_l_local; i++) {
                    std::cout << iter << "\t" << row_local[i] + split[rank] << "\t" << col_local[i] << "\t" << values_local[i] << std::endl;
                    iter++;
                }
            }
            MPI_Barrier(comm);
        }
    }
}


void strength_matrix::print_off_diagonal(int ran){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    index_t iter = 0;
    if(ran >= 0) {
        if (rank == ran) {
            printf("\nstrength matrix (off_diagonal) on proc = %d \n", ran);
            printf("nnz = %lu \n", nnz_l_remote);
            for (index_t i = 0; i < nnz_l_remote; i++) {
                std::cout << iter << "\t" << row_remote[i] + split[rank] << "\t" << col_remote2[i] << "\t" << values_remote[i] << std::endl;
                iter++;
            }
        }
    } else{
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("\nstrength matrix (off_diagonal) on proc = %d \n", rank);
                printf("nnz = %lu \n", nnz_l_remote);
                for (index_t i = 0; i < nnz_l_remote; i++) {
                    std::cout << iter << "\t" << row_remote[i] + split[rank] << "\t" << col_remote2[i] << "\t" << values_remote[i] << std::endl;
                    iter++;
                }
            }
            MPI_Barrier(comm);
        }
    }
}


int strength_matrix::print_info(int ran){

    // if ran >= 0 print the matrix entries on proc with rank = ran
    // otherwise print the matrix entries on all processors in order. (first on proc 0, then proc 1 and so on.)

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(ran >= 0) {
        if (rank == ran) {
            printf("\nmatrix S info on proc = %d \n", ran);
            printf("Mbig = %u, \tM = %u, \tnnz_g = %lu, \tnnz_l = %lu \n", Mbig, M, nnz_g, nnz_l);
        }
    } else{
        MPI_Barrier(comm);
        if(rank==0) printf("\nmatrix S info:      Mbig = %u, \tnnz_g = %lu \n", Mbig, nnz_g);
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("matrix S on rank %d: M    = %u, \tnnz_l = %lu \n", proc, M, nnz_l);
            }
            MPI_Barrier(comm);
        }
    }

    return 0;
}


int strength_matrix::save_to_disk(){

    for(nnz_t i = 0; i < nnz_l; i++){

    }

    return 0;
}


int strength_matrix::randomVector(std::vector<unsigned long>& V, long size, MPI_Comm comm) {

    int rank;
    MPI_Comm_rank(comm, &rank);

//    for (unsigned long i=0; i<V.size(); i++){
//        srand (i);
//        V[i] = rand()%(V.size());
//    }

    unsigned long max_weight = ( (1UL<<63) - 1);

    nnz_t i = 0;
    index_t max_degree_local = 0;
    for(i = 0; i < M; ++i){
        if(nnzPerRow[i] > max_degree_local)
            max_degree_local = nnzPerRow[i];
    }

    index_t max_degree = 0;
    MPI_Allreduce(&max_degree_local, &max_degree, 1, par::Mpi_datatype<index_t>::value(), MPI_MAX, comm);
    max_degree++;
//    printf("rank = %d, max degree local = %lu, max degree = %lu \n", rank, max_degree_local, max_degree);

    //Type of random number distribution
//    std::uniform_real_distribution<float> dist(-1.0,1.0); //(min, max)
    unsigned int max_rand = ( (1UL<<32) - 1);
    std::uniform_int_distribution<unsigned int> dist(0,max_rand); //(min, max)

    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng;

    //Initialize with non-deterministic seeds
    rng.seed(std::random_device{}());

    std::vector<double> rand(M);
    for (i = 0; i < V.size(); i++){
        V[i] = ((unsigned long)(max_degree - nnzPerRow[i])<<32) + dist(rng);
//        V[i] = V[i] << 32;
//        V[i] += dist(rng);
//        if(rank==0) cout << i << "\tnnzPerRow = " << nnzPerRow[i] << "\t weight = " << V[i] << endl;
    }

    // to have one node with the highest weight possible, so that node will be a root and consequently P and R won't be zero matrices.
    // the median index is being chosen here.
    if (V.size() != 0)
        V[ floor(V.size()/2) ] = max_weight;

    return 0;
}

int strength_matrix::randomVector2(std::vector<double>& V){

//    int rank;
//    MPI_Comm_rank(comm, &rank);

//    for (unsigned long i=0; i<V.size(); i++){
//        srand (i);
//        V[i] = rand()%(V.size());
//    }

    //Type of random number distribution
    std::uniform_real_distribution<double> dist(0.0,1.0); //(min, max)

    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng;

    //Initialize with non-deterministic seeds
    rng.seed(std::random_device{}());

    for (unsigned long i=0; i<V.size(); i++){
        V[i] = dist(rng);
    }

    return 0;
}

int strength_matrix::set_weight(std::vector<unsigned long>& V) {
    // This function DOES NOT generate a random vector. It computes the maximum degree of all the nodes.
    // (degree of node i = number of nonzeros on row i)
    // Then assign to a higher degree, a lower weight ( weight(node i) = max_degree - degree(node i) )
    // This method is similar to Yavneh's paper, in which nodes with lower degrees become coarse nodes first,
    // then nodes with higher degrees.
    // Yavneh's paper: Non-Galerkin Multigrid Based on Sparsified Smoothed Aggregation - pages: A51-A52

    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    nnz_t i = 0;
    index_t max_degree_local = 0;
    for(i = 0; i < M; ++i){
        if(nnzPerRow[i] > max_degree_local)
            max_degree_local = nnzPerRow[i];
    }

    index_t max_degree = 0;
    MPI_Allreduce(&max_degree_local, &max_degree, 1, par::Mpi_datatype<index_t>::value(), MPI_MAX, comm);
    max_degree++;

//    printf("rank = %d, max degree local = %lu, max degree = %lu \n", rank, max_degree_local, max_degree);

    std::vector<double> rand(M);
    for (i = 0; i < V.size(); ++i){
        V[i] = max_degree - nnzPerRow[i];
//        if(rank==1) std::cout << i << "\tnnzPerRow = " << nnzPerRow[i] << "\t weight = " << V[i] << std::endl;
    }

    return 0;
}

int strength_matrix::randomVector4(std::vector<unsigned long>& V, long size) {

//    int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//    for (unsigned long i=0; i<V.size(); i++){
//        srand (i);
//        V[i] = rand()%(V.size());
//    }

    //Type of random number distribution
    std::uniform_int_distribution<unsigned long> dist(1, size); //(min, max)

    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng;

    //Initialize with non-deterministic seeds
    rng.seed(std::random_device{}());

    for (unsigned long i = 0; i < V.size(); i++)
        V[i] = dist(rng);

    // to have one node with the highest weight possible, so that node will be a root and consequently P and R won't be zero matrices.
    // the median index is being chosen here.
    if (V.size() != 0)
        V[ floor(V.size()/2) ] = size + 1;

    return 0;
}