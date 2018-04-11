#include <fstream>
#include <algorithm>
#include <sys/stat.h>
#include <cstring>
#include "mpi.h"
#include <omp.h>
#include "saena_matrix.h"
#include "parUtils.h"

//#pragma omp declare reduction(vec_double_plus : std::vector<value_t> : \
//                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<value_t>())) \
//                    initializer(omp_priv = omp_orig)


saena_matrix::saena_matrix(){}


saena_matrix::saena_matrix(MPI_Comm com) {
    comm = com;
    comm_old = com;
}


saena_matrix::saena_matrix(char* Aname, MPI_Comm com) {
    // the following variables of saena_matrix class will be set in this function:
    // Mbig", "nnz_g", "initial_nnz_l", "data"
    // "data" is only required for repartition function.

    read_from_file = true;
    comm = com;
    comm_old = com;

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // find number of general nonzeros of the input matrix
    struct stat st;
    stat(Aname, &st);
    nnz_g = st.st_size / (2*sizeof(index_t) + sizeof(value_t));

    // find initial local nonzero
    initial_nnz_l = nnz_t(floor(1.0 * nnz_g / nprocs)); // initial local nnz
    if (rank == nprocs - 1)
        initial_nnz_l = nnz_g - (nprocs - 1) * initial_nnz_l;

    if(verbose_saena_matrix){
        MPI_Barrier(comm);
        printf("saena_matrix: part 1. rank = %d, nnz_g = %lu, initial_nnz_l = %lu \n", rank, nnz_g, initial_nnz_l);
        MPI_Barrier(comm);}

//    printf("\nrank = %d, nnz_g = %lu, initial_nnz_l = %lu \n", rank, nnz_g, initial_nnz_l);

    // todo: change data from vector to malloc. then free after repartitioning.
    data_unsorted.resize(initial_nnz_l);
    cooEntry* datap = &(*(data_unsorted.begin()));

    // *************************** read the matrix ****************************

    MPI_Status status;
    MPI_File fh;
    MPI_Offset offset;

    int mpiopen = MPI_File_open(comm, Aname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (mpiopen) {
        if (rank == 0) std::cout << "Unable to open the matrix file!" << std::endl;
        MPI_Finalize();
    }

    //offset = rank * initial_nnz_l * 24; // row index(long=8) + column index(long=8) + value(double=8) = 24
    // the offset for the last process will be wrong if you use the above formula,
    // because initial_nnz_l of the last process will be used, instead of the initial_nnz_l of the other processes.

    offset = rank * nnz_t(floor(1.0 * nnz_g / nprocs)) * (2*sizeof(index_t) + sizeof(value_t));

    MPI_File_read_at(fh, offset, datap, initial_nnz_l, cooEntry::mpi_datatype(), &status);

//    double val;
//    if(rank==0)
//        for(long i=0; i<initial_nnz_l; i++){
//            val = data_unsorted[3*i+2];
//            std::cout << datap[3*i] << "\t" << datap[3*i+1] << "\t" << val << std::endl;
//        }

//    int count;
//    MPI_Get_count(&status, MPI_UNSIGNED_LONG, &count);
    //printf("process %d read %d lines of triples\n", rank, count);
    MPI_File_close(&fh);

//    for(int i=0; i<data_unsorted.size(); i++)
//        if(rank==ran) std::cout << data_unsorted[i] << std::endl;

//    printf("rank = %d \t\t\t before sort: data_unsorted size = %lu\n", rank, data_unsorted.size());

//    MPI_Barrier(comm); printf("rank = %d\t setup_initial 22222222222222222222\n", rank);  MPI_Barrier(comm);
//    par::sampleSort(data_unsorted, comm);

    std::vector<cooEntry> data_sorted;
    par::sampleSort(data_unsorted, data_sorted, comm);

//    printf("rank = %d \t\t\t after  sort: data_sorted size = %lu\n", rank, data_sorted.size());

    // clear data_unsorted and free memory.
    data_unsorted.clear();
    data_unsorted.shrink_to_fit();

//    for(int i=0; i<data_unsorted.size(); i++)
//        if(rank==ran) std::cout << data_sorted[i] << std::endl;

    // todo: "data" vector can completely be avoided. function repartition should be changed to use a vector of cooEntry
    // todo: (which is "data_unsorted" here), instead of "data" (which is a vector of unsigned long of size 3*nnz).

    // size of data may be smaller because of duplicates. In that case its size will be reduced after finding the exact size.
    data.resize(data_sorted.size());

    // put the first element of data_unsorted to data.
    nnz_t data_size = 0;
    if(!data_sorted.empty()){
        data[0] = data_sorted[0];
//        data.push_back(data_sorted[0].row);
//        data.push_back(data_sorted[0].col);
//        data.push_back(reinterpret_cast<unsigned long&>(data_sorted[0].val));
        data_size++;
    }

//    double val_temp;
    for(nnz_t i=1; i<data_sorted.size(); i++){
        if(data_sorted[i] == data_sorted[i-1]){
            if(add_duplicates){
                data[data_size-1].val += data_sorted[i].val;
//                data.pop_back();
//                val_temp = data_sorted[i-1].val + data_sorted[i].val;
//                data.push_back(reinterpret_cast<unsigned long&>(val_temp));
            }else{
                data[data_size-1] = data_sorted[i];
            }
        }else{
            data[data_size] = data_sorted[i];
//            data.push_back(data_sorted[i].row);
//            data.push_back(data_sorted[i].col);
//            data.push_back(reinterpret_cast<unsigned long&>(data_sorted[i].val));
            data_size++;
        }
    }

    data.resize(data_size);
    data.shrink_to_fit();

    if(data.empty()) {
        printf("error: data has no element on process %d! \n", rank);
        MPI_Finalize();}

//    cooEntry first_element = data[0];
    cooEntry first_element_neighbor;

    // send last element to the left neighbor and check if it is equal to the last element of the left neighbor.
    if(rank != nprocs-1)
        MPI_Recv(&first_element_neighbor, 1, cooEntry::mpi_datatype(), rank+1, 0, comm, MPI_STATUS_IGNORE);

    if(rank!= 0)
        MPI_Send(&data[0], 1, cooEntry::mpi_datatype(), rank-1, 0, comm);

    cooEntry last_element = data.back();
    if(rank != nprocs-1){
        if(last_element == first_element_neighbor) {
//            if(rank==0) std::cout << "remove!" << std::endl;
            data.pop_back();
        }
    }

    // if duplicates should be added together and last_element == first_element_neighbor
    // then send only the value of last_element to the right neighbor and add it to the last element's value.
    // this has the reverse communication of the previous part.
    value_t left_neighbor_last_val;
    if((last_element == first_element_neighbor) && add_duplicates ){
        if(rank != 0)
            MPI_Recv(&left_neighbor_last_val, 1, MPI_DOUBLE, rank-1, 0, comm, MPI_STATUS_IGNORE);

        if(rank!= nprocs-1)
            MPI_Send(&last_element.val, 1, MPI_DOUBLE, rank+1, 0, comm);

        data[0].val += left_neighbor_last_val;
    }

//    if(rank==ran) std::cout << "after  sorting\n" << "data size = " << data.size() << std::endl;
//    for(int i=0; i<data.size(); i++)
//        if(rank==ran) std::cout << data[3*i] << "\t" << data[3*i+1] << "\t" << data[3*i+2] << std::endl;

    initial_nnz_l = data.size();
    // todo: here: nnz_g was set at the beginning of this function. why would we need this line?
    MPI_Allreduce(&initial_nnz_l, &nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

    // *************************** find Mbig (global number of rows) ****************************
    // First find the maximum of rows. Then, compare it with the maximum of columns.
    // The one that is bigger is the size of the matrix.

    index_t Mbig_local = 0;
    for(nnz_t i=0; i<initial_nnz_l; i++){
        if(data[i].row > Mbig_local)
            Mbig_local = data[i].row;
    }

    last_element = data.back();
    if(last_element.col > Mbig_local)
        Mbig_local = last_element.col;

    MPI_Allreduce(&Mbig_local, &Mbig, 1, MPI_UNSIGNED, MPI_MAX, comm);
    Mbig++; // since indices start from 0, not 1.

    if(verbose_saena_matrix){
        MPI_Barrier(comm);
        printf("saena_matrix: part 2. rank = %d, nnz_g = %lu, initial_nnz_l = %lu, Mbig = %u \n", rank, nnz_g, initial_nnz_l, Mbig);
        MPI_Barrier(comm);}

//    if(rank==0){
//        for(unsigned long i = 0; i < initial_nnz_l; i++)
//            std::cout << data[i] << std::endl;}

}


saena_matrix::~saena_matrix() {
    if(freeBoolean){
//        free(vIndex);
//        free(vSend);
//        free(vSendULong);
//        free(vecValues);
//        free(vecValuesULong);
//        free(indicesP_local);
//        free(indicesP_remote);
//        free(iter_local_array);
//        free(iter_remote_array);
//        iter_local_array.clear();
//        iter_local_array.shrink_to_fit();
//        iter_remote_array.clear();
//        iter_remote_array.shrink_to_fit();
//        delete w_buff;
    }
}


int saena_matrix::set(index_t row, index_t col, value_t val){

    cooEntry temp_new = cooEntry(row, col, val);
    std::pair<std::set<cooEntry>::iterator, bool> p = data_coo.insert(temp_new);

    if (!p.second){
        auto hint = p.first; // hint is std::set<cooEntry>::iterator
        hint++;
        data_coo.erase(p.first);
        // in the case of duplicate, if the new value is zero, remove the older one and don't insert the zero.
        if(!almost_zero(val))
            data_coo.insert(hint, temp_new);
    }

    // if the entry is zero and it was not a duplicate, just erase it.
    if(p.second && almost_zero(val))
        data_coo.erase(p.first);

    return 0;
}


int saena_matrix::set(index_t* row, index_t* col, value_t* val, nnz_t nnz_local){

    if(nnz_local <= 0){
        printf("size in the set function is either zero or negative!");
        return 0;
    }

    cooEntry temp_new;
    std::pair<std::set<cooEntry>::iterator, bool> p;

    // todo: isn't it faster to allocate memory for nnz_local, then assign, instead of inserting one by one.
    for(unsigned int i=0; i<nnz_local; i++){

        temp_new = cooEntry(row[i], col[i], val[i]);
        p = data_coo.insert(temp_new);

        if (!p.second){
            auto hint = p.first; // hint is std::set<cooEntry>::iterator
            hint++;
            data_coo.erase(p.first);
            // if the entry is zero and it was not a duplicate, just erase it.
            if(!almost_zero(val[i]))
                data_coo.insert(hint, temp_new);
        }

        // if the entry is zero, erase it.
        if(p.second && almost_zero(val[i]))
            data_coo.erase(p.first);
    }

    return 0;
}


int saena_matrix::set2(index_t row, index_t col, value_t val){

    // todo: if there are duplicates with different values on two different processors, what should happen?
    // todo: which one should be removed? Hari said "do it randomly".

    cooEntry temp_old;
    cooEntry temp_new = cooEntry(row, col, val);

    std::pair<std::set<cooEntry>::iterator, bool> p = data_coo.insert(temp_new);

    if (!p.second){
        temp_old = *(p.first);
        temp_new.val += temp_old.val;

        std::set<cooEntry>::iterator hint = p.first;
        hint++;
        data_coo.erase(p.first);
        data_coo.insert(hint, temp_new);
    }

    return 0;
}


int saena_matrix::set2(index_t* row, index_t* col, value_t* val, nnz_t nnz_local){

    if(nnz_local <= 0){
        printf("size in the set function is either zero or negative!");
        return 0;
    }

    cooEntry temp_old, temp_new;
    std::pair<std::set<cooEntry>::iterator, bool> p;

    for(unsigned int i=0; i<nnz_local; i++){
        if(!almost_zero(val[i])){
            temp_new = cooEntry(row[i], col[i], val[i]);
            p = data_coo.insert(temp_new);

            if (!p.second){
                temp_old = *(p.first);
                temp_new.val += temp_old.val;

                std::set<cooEntry>::iterator hint = p.first;
                hint++;
                data_coo.erase(p.first);
                data_coo.insert(hint, temp_new);
            }
        }
    }

    return 0;
}

/*
int saena_matrix::set3(unsigned int row, unsigned int col, double val){

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // update the matrix size if required.
//    if(row >= Mbig)
//        Mbig = row + 1; // "+ 1" is there since row starts from 0, not 1.
//    if(col >= Mbig)
//        Mbig = col + 1;

//    auto proc_num = lower_bound2(&*split.begin(), &*split.end(), (unsigned long)row);
//    printf("proc_num = %ld\n", proc_num);

    cooEntry recv_buf;
    cooEntry send_buf(row, col, val);

//    if(rank == proc_num)
//        MPI_Recv(&send_buf, 1, cooEntry::mpi_datatype(), , 0, comm, NULL);
//    if(rank != )
//        MPI_Send(&recv_buf, 1, cooEntry::mpi_datatype(), proc_num, 0, comm);

    //todo: change send_buf to recv_buf after completing the communication for the parallel version.
    auto position = lower_bound2(&*entry.begin(), &*entry.end(), send_buf);
//    printf("position = %lu \n", position);
//    printf("%lu \t%lu \t%f \n", entry[position].row, entry[position].col, entry[position].val);

    if(send_buf == entry[position]){
        if(add_duplicates){
            entry[position].val += send_buf.val;
        }else{
            entry[position].val = send_buf.val;
        }
    }else{
        printf("\nAttention: the structure of the matrix is being changed, so matrix.assemble() is required to call after being done calling matrix.set()!\n\n");
        entry.push_back(send_buf);
        std::sort(&*entry.begin(), &*entry.end());
        nnz_g++;
        nnz_l++;
    }

//    printf("\nentry:\n");
//    for(long i = 0; i < nnz_l; i++)
//        std::cout << entry[i] << std::endl;

    return 0;
}

int saena_matrix::set3(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local){

    if(nnz_local <= 0){
        printf("size in the set function is either zero or negative!");
        return 0;
    }

    cooEntry temp;
    long position;
    for(unsigned int i = 0; i < nnz_local; i++){
        temp = cooEntry(row[i], col[i], val[i]);
        position = lower_bound2(&*entry.begin(), &*entry.end(), temp);
        if(temp == entry[position]){
            if(add_duplicates){
                entry[position].val += temp.val;
            }else{
                entry[position].val  = temp.val;
            }
        }else{
            printf("\nAttention: the structure of the matrix is being changed, so matrix.assemble() is required to call after being done calling matrix.set()!\n\n");
            entry.push_back(temp);
            std::sort(&*entry.begin(), &*entry.end());
            nnz_g++;
            nnz_l++;
        }
    }

//    printf("\nentry:\n");
//    for(long i = 0; i < nnz_l; i++)
//        std::cout << entry[i] << std::endl;

    return 0;
}
*/

void saena_matrix::set_comm(MPI_Comm com){
    comm = com;
    comm_old = com;
}


int saena_matrix::setup_initial_data(){

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    std::cout << rank << " : " << __func__ << initial_nnz_l << std::endl;

    std::set<cooEntry>::iterator it;
    nnz_t iter = 0;
    index_t Mbig_local = 0;
    cooEntry temp;

    data_unsorted.resize(data_coo.size());
    for(it=data_coo.begin(); it!=data_coo.end(); ++it){
        data_unsorted[iter] = *it;
        ++iter;

        temp = *it;
        if(temp.row > Mbig_local)
            Mbig_local = temp.row;
    }

    // todo: free memory for data_coo. consider move semantics. check if the following idea is correct.
    // clear data_coo and free memory.
    // Using move semantics, the address of data_coo is swapped with data_temp. So data_coo will be empty
    // and data_temp will be deleted when this function is finished.
    std::set<cooEntry> data_temp = std::move(data_coo);

    // since shrink_to_fit() does not work for std::set, set.erase() is being used, not sure if it frees memory.
//    data_coo.erase(data_coo.begin(), data_coo.end());
//    data_coo.clear();

    // Mbig is the size of the matrix, which is the maximum of rows and columns.
    // up to here Mbig_local is the maximum of rows.
    // data[3*iter+1] is the maximum of columns, since it is sorted based on columns.

    iter--;
    if(data_unsorted[iter].col > Mbig_local)
        Mbig_local = data_unsorted[iter].col;

    MPI_Allreduce(&Mbig_local, &Mbig, 1, MPI_UNSIGNED, MPI_MAX, comm);
    Mbig++; // since indices start from 0, not 1.
//    std::cout << "Mbig = " << Mbig << std::endl;

//    printf("rank = %d \t\t\t before sort: data_unsorted size = %lu\n", rank, data_unsorted.size());

//    for(int i=0; i<data_unsorted.size(); i++)
//        if(rank==0) std::cout << data_unsorted[i] << std::endl;

//    MPI_Barrier(comm); printf("rank = %d\t setup_initial 11111111111111111111\n", rank);  MPI_Barrier(comm);

    std::vector<cooEntry> data_sorted;
    par::sampleSort(data_unsorted, data_sorted, comm);

//    MPI_Barrier(comm); printf("rank = %d\t setup_initial 22222222222222222222\n", rank);  MPI_Barrier(comm);

    // clear data_unsorted and free memory.
    data_unsorted.clear();
    data_unsorted.shrink_to_fit();

//    MPI_Barrier(comm); std::cout << std::endl; MPI_Barrier(comm);
//    printf("rank = %d \t\t\t after  sort: data_sorted size = %lu\n", rank, data_sorted.size());

//    par::sampleSort(data_unsorted, comm);

    // todo: "data" vector can completely be avoided. function repartition should be changed to use a vector of cooEntry
    // todo: (which is "data_sorted" here), instead of "data" (which is a vector of unsigned long of size 3*nnz).

    // size of data may be smaller because of duplicates. In that case its size will be reduced after finding the exact size.
    data.resize(data_sorted.size());

    // put the first element of data_unsorted to data.
    nnz_t data_size = 0;
    if(!data_sorted.empty()){
        data[0] = data_sorted[0];
//        data.push_back(data_sorted[0].row);
//        data.push_back(data_sorted[0].col);
//        data.push_back(reinterpret_cast<unsigned long&>(data_sorted[0].val));
        data_size++;
    }

//    double val_temp;
    for(nnz_t i=1; i<data_sorted.size(); i++){
        if(data_sorted[i] == data_sorted[i-1]){
            if(add_duplicates){
                data[data_size-1].val += data_sorted[i].val;
//                data.pop_back();
//                val_temp = data_sorted[i-1].val + data_sorted[i].val;
//                data.push_back(reinterpret_cast<unsigned long&>(val_temp));
            }else{
                data[data_size-1] = data_sorted[i];
            }
        }else{
            data[data_size] = data_sorted[i];
//            data.push_back(data_sorted[i].row);
//            data.push_back(data_sorted[i].col);
//            data.push_back(reinterpret_cast<unsigned long&>(data_sorted[i].val));
            data_size++;
        }
    }

    data.resize(data_size);
    data.shrink_to_fit();

//    MPI_Barrier(comm); printf("rank = %d, data_size = %u, size of data = %lu\n", rank, data_size, data.size());

//    std::vector<cooEntry>::iterator iterator;
//    for(iterator = data_unsorted.begin(); iterator != data_unsorted.end(); iterator++){
//        if(*iterator == *(iterator+1))
//            data_unsorted.erase(iterator);
//    }
//
//    initial_nnz_l = data_unsorted.size();
//    data.resize(3 * initial_nnz_l);
//    unsigned int iter1 = 0;
//    for(auto &i:data_unsorted){
//        data[3*iter1]   = i.row;
//        data[3*iter1+1] = i.col;
//        data[3*iter1+2] = reinterpret_cast<unsigned long&>(i.val);
//        iter1++;
//    }

    if(data.empty()) {
        printf("error: data has no element on process %d! \n", rank);
        MPI_Finalize();
        return -1;}

//    cooEntry first_element = data[0];
    cooEntry first_element_neighbor;

    // send last element to the left neighbor and check if it is equal to the last element of the left neighbor.
    if(rank != nprocs-1)
        MPI_Recv(&first_element_neighbor, 1, cooEntry::mpi_datatype(), rank+1, 0, comm, MPI_STATUS_IGNORE);

    if(rank!= 0)
        MPI_Send(&data[0], 1, cooEntry::mpi_datatype(), rank-1, 0, comm);

//    MPI_Barrier(comm); printf("rank = %d\t setup_initial 5555555555555555555\n", rank);  MPI_Barrier(comm);
//    MPI_Barrier(comm); printf("rank = %d\t data.size() = %lu, data_size = %u, last = %u \n", rank, data.size(), data_size, 3*(data_size-1)+2);  MPI_Barrier(comm);

//    cooEntry last_element = cooEntry(data[3*(data_size-1)], data[3*(data_size-1)+1], data[3*(data_size-1)+2]);
//    MPI_Barrier(comm); if(rank==0) std::cout << "rank = " << rank << "\t first_element = " << first_element_neighbor << ", last element = " << last_element << std::endl;  MPI_Barrier(comm);

    cooEntry last_element = data.back();
    if(rank != nprocs-1){
        if(last_element == first_element_neighbor) {
//            if(rank==0) std::cout << "remove!" << std::endl;
            data.pop_back();
        }
    }

    // if duplicates should be added together and last_element == first_element_neighbor
    // then send only the value of last_element to the right neighbor and add it to the last element's value.
    // this has the reverse communication of the previous part.
    value_t left_neighbor_last_val;
    if((last_element == first_element_neighbor) && add_duplicates ){
        if(rank != 0)
            MPI_Recv(&left_neighbor_last_val, 1, MPI_DOUBLE, rank-1, 0, comm, MPI_STATUS_IGNORE);

        if(rank!= nprocs-1)
            MPI_Send(&last_element.val, 1, MPI_DOUBLE, rank+1, 0, comm);

        data[0].val += left_neighbor_last_val;
    }

//    MPI_Barrier(comm); printf("rank = %d\t setup_initial 6666666666666666666\n", rank);  MPI_Barrier(comm);

    initial_nnz_l = data.size();
    MPI_Allreduce(&initial_nnz_l, &nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
//    MPI_Barrier(comm); printf("rank = %d, Mbig = %u, nnz_g = %u, initial_nnz_l = %u \n", rank, Mbig, nnz_g, initial_nnz_l); MPI_Barrier(comm);

    return 0;
}


int saena_matrix::setup_initial_data2(){

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    std::cout << rank << " : " << __func__ << initial_nnz_l << std::endl;

    std::set<cooEntry>::iterator it;
    nnz_t iter = 0;

    data_unsorted.resize(data_coo.size());
    for(it=data_coo.begin(); it!=data_coo.end(); ++it){
        data_unsorted[iter] = *it;
        ++iter;
    }

    // todo: free memory for data_coo. consider move semantics. check if the following idea is correct.
    // clear data_coo and free memory.
    // Using move semantics, the address of data_coo is swapped with data_temp. So data_coo will be empty
    // and data_temp will be deleted when this function is finished.
    std::set<cooEntry> data_temp = std::move(data_coo);

    std::vector<cooEntry> data_sorted;
    par::sampleSort(data_unsorted, data_sorted, comm);

    // clear data_unsorted and free memory.
    data_unsorted.clear();
    data_unsorted.shrink_to_fit();

//    MPI_Barrier(comm); std::cout << std::endl; MPI_Barrier(comm);
//    printf("rank = %d \t\t\t after  sort: data_sorted size = %lu\n", rank, data_sorted.size());

    // todo: "data" vector can completely be avoided. function repartition should be changed to use a vector of cooEntry
    // todo: (which is "data_sorted" here), instead of "data" (which is a vector of unsigned long of size 3*nnz).

    data.resize(data_sorted.size());

    // put the first element of data_unsorted to data.
    nnz_t data_size = 0;
    if(!data_sorted.empty()){
        data[0] = data_sorted[0];
//        data.push_back(data_sorted[0].row);
//        data.push_back(data_sorted[0].col);
//        data.push_back(reinterpret_cast<unsigned long&>(data_sorted[0].val));
        data_size++;
    }

//    double val_temp;
    for(nnz_t i=1; i<data_sorted.size(); i++){
        if(data_sorted[i] == data_sorted[i-1]){
            if(add_duplicates){
                data[data_size-1].val += data_sorted[i].val;
//                data.pop_back();
//                val_temp = data_sorted[i-1].val + data_sorted[i].val;
//                data.push_back(reinterpret_cast<unsigned long&>(val_temp));
            }else{
                data[data_size-1] = data_sorted[i];
            }
        }else{
            data[data_size] = data_sorted[i];
//            data.push_back(data_sorted[i].row);
//            data.push_back(data_sorted[i].col);
//            data.push_back(reinterpret_cast<unsigned long&>(data_sorted[i].val));
            data_size++;
        }
    }

    data.resize(data_size);
    data.shrink_to_fit();

//    MPI_Barrier(comm); printf("rank = %d, data_size = %u, size of data = %lu\n", rank, data_size, data.size());

//    std::vector<cooEntry>::iterator iterator;
//    for(iterator = data_unsorted.begin(); iterator != data_unsorted.end(); iterator++){
//        if(*iterator == *(iterator+1))
//            data_unsorted.erase(iterator);
//    }
//
//    initial_nnz_l = data_unsorted.size();
//    data.resize(3 * initial_nnz_l);
//    unsigned int iter1 = 0;
//    for(auto &i:data_unsorted){
//        data[3*iter1]   = i.row;
//        data[3*iter1+1] = i.col;
//        data[3*iter1+2] = reinterpret_cast<unsigned long&>(i.val);
//        iter1++;
//    }

    if(data.empty()) {
        printf("error: data has no element on process %d! \n", rank);
        MPI_Finalize();
        return -1;}

//    cooEntry first_element = data[0];
    cooEntry first_element_neighbor;

    // send last element to the left neighbor and check if it is equal to the last element of the left neighbor.
    if(rank != nprocs-1)
        MPI_Recv(&first_element_neighbor, 1, cooEntry::mpi_datatype(), rank+1, 0, comm, MPI_STATUS_IGNORE);

    if(rank!= 0)
        MPI_Send(&data[0], 1, cooEntry::mpi_datatype(), rank-1, 0, comm);

//    MPI_Barrier(comm); printf("rank = %d\t setup_initial 5555555555555555555\n", rank);  MPI_Barrier(comm);
//    MPI_Barrier(comm); printf("rank = %d\t data.size() = %lu, data_size = %u, last = %u \n", rank, data.size(), data_size, 3*(data_size-1)+2);  MPI_Barrier(comm);

//    cooEntry last_element = cooEntry(data[3*(data_size-1)], data[3*(data_size-1)+1], data[3*(data_size-1)+2]);
//    MPI_Barrier(comm); if(rank==0) std::cout << "rank = " << rank << "\t first_element = " << first_element_neighbor << ", last element = " << last_element << std::endl;  MPI_Barrier(comm);

    cooEntry last_element = data.back();
    if(rank != nprocs-1){
        if(last_element == first_element_neighbor) {
//            if(rank==0) std::cout << "remove!" << std::endl;
            data.pop_back();
        }
    }

    // if duplicates should be added together and last_element == first_element_neighbor
    // then send only the value of last_element to the right neighbor and add it to the last element's value.
    // this has the reverse communication of the previous part.
    value_t left_neighbor_last_val = 0;
    if((last_element == first_element_neighbor) && add_duplicates ){
        if(rank != 0)
            MPI_Recv(&left_neighbor_last_val, 1, MPI_DOUBLE, rank-1, 0, comm, MPI_STATUS_IGNORE);

        if(rank!= nprocs-1)
            MPI_Send(&last_element.val, 1, MPI_DOUBLE, rank+1, 0, comm);

        data[0].val += left_neighbor_last_val;
    }

//    MPI_Barrier(comm); printf("rank = %d\t setup_initial 6666666666666666666\n", rank);  MPI_Barrier(comm);

    initial_nnz_l = data.size();
    nnz_t nnz_g_temp = nnz_g;
    MPI_Allreduce(&initial_nnz_l, &nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
    if(rank == 0 && (nnz_g_temp != nnz_g) ) printf("error: number of global nonzeros is changed during the matrix update!\n");

    return 0;
}


int saena_matrix::destroy(){
    return 0;
}


int saena_matrix::erase(){
//    data.clear();
//    data.shrink_to_fit();

    entry.clear();
    split.clear();
    split_old.clear();
    values_local.clear();
    row_local.clear();
    values_remote.clear();
    row_remote.clear();
    col_local.clear();
    col_remote.clear();
    col_remote2.clear();
    nnzPerRow_local.clear();
    nnzPerCol_remote.clear();
    invDiag.clear();
    vdispls.clear();
    rdispls.clear();
    recvProcRank.clear();
    recvProcCount.clear();
    sendProcRank.clear();
    sendProcCount.clear();
    sendProcCount.clear();
    vElementRep_local.clear();
    vElementRep_remote.clear();

    entry.shrink_to_fit();
    split.shrink_to_fit();
    split_old.shrink_to_fit();
    values_local.shrink_to_fit();
    values_remote.shrink_to_fit();
    row_local.shrink_to_fit();
    row_remote.shrink_to_fit();
    col_local.shrink_to_fit();
    col_remote.shrink_to_fit();
    col_remote2.shrink_to_fit();
    nnzPerRow_local.shrink_to_fit();
    nnzPerCol_remote.shrink_to_fit();
    invDiag.shrink_to_fit();
    vdispls.shrink_to_fit();
    rdispls.shrink_to_fit();
    recvProcRank.shrink_to_fit();
    recvProcCount.shrink_to_fit();
    sendProcRank.shrink_to_fit();
    sendProcCount.shrink_to_fit();
    sendProcCount.shrink_to_fit();
    vElementRep_local.shrink_to_fit();
    vElementRep_remote.shrink_to_fit();

    M = 0;
    Mbig = 0;
    nnz_g = 0;
    nnz_l = 0;
    nnz_l_local = 0;
    nnz_l_remote = 0;
    col_remote_size = 0;
    recvSize = 0;
    numRecvProc = 0;
    numSendProc = 0;
    assembled = false;

    return 0;
}


int saena_matrix::erase2(){
//    data.clear();
//    data.shrink_to_fit();

    entry.clear();
    split.clear();
    split_old.clear();
    values_local.clear();
    row_local.clear();
    values_remote.clear();
    row_remote.clear();
    col_local.clear();
    col_remote.clear();
    col_remote2.clear();
    nnzPerRow_local.clear();
    nnzPerCol_remote.clear();
    invDiag.clear();
    vdispls.clear();
    rdispls.clear();
    recvProcRank.clear();
    recvProcCount.clear();
    sendProcRank.clear();
    sendProcCount.clear();
    vElementRep_local.clear();
    vElementRep_remote.clear();
    vIndex.clear();
    vSend.clear();
    vecValues.clear();
    vSendULong.clear();
    vecValuesULong.clear();
    indicesP_local.clear();
    indicesP_remote.clear();
    recvCount.clear();
    recvCountScan.clear();
    sendCount.clear();
    sendCountScan.clear();
    iter_local_array.clear();
    iter_remote_array.clear();
    iter_local_array2.clear();
    iter_remote_array2.clear();
    vElement_remote.clear();
    w_buff.clear();

    entry.shrink_to_fit();
    split.shrink_to_fit();
    split_old.shrink_to_fit();
    values_local.shrink_to_fit();
    values_remote.shrink_to_fit();
    row_local.shrink_to_fit();
    row_remote.shrink_to_fit();
    col_local.shrink_to_fit();
    col_remote.shrink_to_fit();
    col_remote2.shrink_to_fit();
    nnzPerRow_local.shrink_to_fit();
    nnzPerCol_remote.shrink_to_fit();
    invDiag.shrink_to_fit();
    vdispls.shrink_to_fit();
    rdispls.shrink_to_fit();
    recvProcRank.shrink_to_fit();
    recvProcCount.shrink_to_fit();
    sendProcRank.shrink_to_fit();
    sendProcCount.shrink_to_fit();
    vElementRep_local.shrink_to_fit();
    vElementRep_remote.shrink_to_fit();
    vIndex.shrink_to_fit();
    vSend.shrink_to_fit();
    vecValues.shrink_to_fit();
    vSendULong.shrink_to_fit();
    vecValuesULong.shrink_to_fit();
    indicesP_local.shrink_to_fit();
    indicesP_remote.shrink_to_fit();
    recvCount.shrink_to_fit();
    recvCountScan.shrink_to_fit();
    sendCount.shrink_to_fit();
    sendCountScan.shrink_to_fit();
    iter_local_array.shrink_to_fit();
    iter_remote_array.shrink_to_fit();
    iter_local_array2.shrink_to_fit();
    iter_remote_array2.shrink_to_fit();
    vElement_remote.shrink_to_fit();
    w_buff.shrink_to_fit();

//    M = 0;
//    Mbig = 0;
//    nnz_g = 0;
//    nnz_l = 0;
//    nnz_l_local = 0;
//    nnz_l_remote = 0;
//    col_remote_size = 0;
//    recvSize = 0;
//    numRecvProc = 0;
//    numSendProc = 0;
//    vIndexSize = 0;
//    shrinked = false;
//    active = true;
    assembled = false;
    freeBoolean = false;

    return 0;
}


int saena_matrix::erase_keep_remote(){

    entry.clear();

    // push back the remote part
    for(unsigned long i = 0; i < row_remote.size(); i++)
        entry.emplace_back(cooEntry(row_remote[i], col_remote2[i], values_remote[i]));

    split.clear();
    split_old.clear();
    values_local.clear();
    values_remote.clear();
    row_local.clear();
    row_remote.clear();
    col_local.clear();
    col_remote.clear();
    col_remote2.clear();
    nnzPerRow_local.clear();
    nnzPerCol_remote.clear();
    invDiag.clear();
    vdispls.clear();
    rdispls.clear();
    recvProcRank.clear();
    recvProcCount.clear();
    sendProcRank.clear();
    sendProcCount.clear();
    sendProcCount.clear();
    vElementRep_local.clear();
    vElementRep_remote.clear();

    // erase_keep_remote() is used in coarsen2(), so keep the memory reserved for performance.
/*
    entry.shrink_to_fit();
    split.shrink_to_fit();
    split_old.shrink_to_fit();
    values_local.shrink_to_fit();
    values_remote.shrink_to_fit();
    row_local.shrink_to_fit();
    row_remote.shrink_to_fit();
    col_local.shrink_to_fit();
    col_remote.shrink_to_fit();
    col_remote2.shrink_to_fit();
    nnzPerRow_local.shrink_to_fit();
    nnzPerCol_remote.shrink_to_fit();
    invDiag.shrink_to_fit();
    vdispls.shrink_to_fit();
    rdispls.shrink_to_fit();
    recvProcRank.shrink_to_fit();
    recvProcCount.shrink_to_fit();
    sendProcRank.shrink_to_fit();
    sendProcCount.shrink_to_fit();
    sendProcCount.shrink_to_fit();
    vElementRep_local.shrink_to_fit();
    vElementRep_remote.shrink_to_fit();
*/

    M = 0;
    Mbig = 0;
    nnz_g = 0;
    nnz_l = 0;
    nnz_l_local = 0;
    nnz_l_remote = 0;
    col_remote_size = 0;
    recvSize = 0;
    numRecvProc = 0;
    numSendProc = 0;
    assembled = false;

    return 0;
}


int saena_matrix::erase_keep_remote2(){

    entry.clear();

    // push back the remote part
    for(unsigned long i = 0; i < row_remote.size(); i++)
        entry.emplace_back(cooEntry(row_remote[i], col_remote2[i], values_remote[i]));

    split.clear();
    split_old.clear();
    values_local.clear();
    row_local.clear();
    values_remote.clear();
    row_remote.clear();
    col_local.clear();
    col_remote.clear();
    col_remote2.clear();
    nnzPerRow_local.clear();
    nnzPerCol_remote.clear();
    invDiag.clear();
    vdispls.clear();
    rdispls.clear();
    recvProcRank.clear();
    recvProcCount.clear();
    sendProcRank.clear();
    sendProcCount.clear();
    vElementRep_local.clear();
    vElementRep_remote.clear();
    vIndex.clear();
    vSend.clear();
    vecValues.clear();
    vSendULong.clear();
    vecValuesULong.clear();
    indicesP_local.clear();
    indicesP_remote.clear();
    recvCount.clear();
    recvCountScan.clear();
    sendCount.clear();
    sendCountScan.clear();
    iter_local_array.clear();
    iter_remote_array.clear();
    iter_local_array2.clear();
    iter_remote_array2.clear();
    vElement_remote.clear();
    w_buff.clear();

    // erase_keep_remote() is used in coarsen2(), so keep the memory reserved for performance.
    // so don't use shrink_to_fit() on these vectors.

    M = 0;
    Mbig = 0;
    nnz_g = 0;
    nnz_l = 0;
    nnz_l_local = 0;
    nnz_l_remote = 0;
    col_remote_size = 0;
    recvSize = 0;
    numRecvProc = 0;
    numSendProc = 0;
    vIndexSize = 0;
//    assembled = false;
//    shrinked = false;
//    active = true;
    freeBoolean = false;

    return 0;
}


int saena_matrix::set_zero(){

#pragma omp parallel for
    for(nnz_t i = 0; i < nnz_l; i++)
        entry[i].val = 0;

    values_local.clear();
    values_remote.clear();

    return 0;
}


int saena_matrix::repartition(){
    // before using this function these variables of SaenaMatrix should be set:
    // Mbig", "nnz_g", "initial_nnz_l", "data"

    // the following variables of saena_matrix class will be set in this function:
    // "nnz_l", "M", "split", "entry"

    // summary: number of buckets are computed based of the number fo rows and number of processors.
    // firstSplit[] is of size n_buckets+1 and is a row partition of the matrix with almost equal number of rows.
    // then the buckets (firsSplit) are combined to have almost the same number of nonzeros. This is split[].

    // if set functions are used the following function should be used.
    if(!read_from_file)
        setup_initial_data();

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(repartition_verbose && rank==0) printf("repartition - step 1!\n");

    density = (nnz_g / double(Mbig)) / (Mbig);
    last_density_shrink = density;
    last_M_shrink = Mbig;
//    last_nnz_shrink = nnz_g;

//    if(rank==0) printf("\n", nnz_g, Mbig);

    // *************************** find splitters ****************************
    // split the matrix row-wise by splitters, so each processor get almost equal number of nonzeros

    // definition of buckets: bucket[i] = [ firstSplit[i] , firstSplit[i+1] ). Number of buckets = n_buckets
    int n_buckets = 0;

/*    if (Mbig > nprocs*nprocs){
        if (nprocs < 1000)
            n_buckets = nprocs*nprocs;
        else
            n_buckets = 1000*nprocs;
    }
    else
        n_buckets = Mbig;*/

    if (Mbig > nprocs*nprocs){
        if (nprocs < 1000)
            n_buckets = nprocs*nprocs;
        else
            n_buckets = 1000*nprocs;
    }
    else if(nprocs <= Mbig){
        n_buckets = Mbig;
    } else{ // nprocs > Mbig
        // it may be better to set nprocs=Mbig and work only with the first Mbig processors.
        if(rank == 0)
            std::cout << "number of tasks cannot be greater than the number of rows of the matrix." << std::endl;
        MPI_Finalize();
    }

//    if (rank==0) std::cout << "n_buckets = " << n_buckets << ", Mbig = " << Mbig << std::endl;

    std::vector<index_t > splitOffset(n_buckets);
    auto baseOffset = index_t(floor(1.0*Mbig/n_buckets));
    float offsetRes = float(1.0*Mbig/n_buckets) - baseOffset;
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

/*    if (rank==0){
        std::cout << "splitOffset:" << std::endl;
        for(long i=0; i<n_buckets; i++)
            std::cout << splitOffset[i] << std::endl;
    }*/

    if(repartition_verbose && rank==0) printf("repartition - step 2!\n");

    std::vector<index_t > firstSplit(n_buckets+1);
    firstSplit[0] = 0;
    for(index_t i=1; i<n_buckets; i++){
        firstSplit[i] = firstSplit[i-1] + splitOffset[i];
    }
    firstSplit[n_buckets] = Mbig;

    splitOffset.clear();
    splitOffset.shrink_to_fit();

//    if (rank==0){
//        std::cout << "firstSplit:" << std::endl;
//        for(long i=0; i<n_buckets+1; i++)
//            std::cout << firstSplit[i] << std::endl;}

    std::sort(data.begin(), data.end(), row_major);

//    if (rank==0){
//        std::cout << "entry row-major: nnz = " << initial_nnz_l << std::endl;
//        for(unsigned int i=0; i<data.size(); i++)
//            std::cout << data[i] << std::endl;}

    index_t least_bucket, last_bucket;
    least_bucket = lower_bound2(&firstSplit[0], &firstSplit[n_buckets], data[0].row);
    last_bucket  = lower_bound2(&firstSplit[0], &firstSplit[n_buckets], data.back().row);
    last_bucket++;

//    if (rank==0)
//        std::cout << "least_bucket:" << least_bucket << ", last_bucket = " << last_bucket << std::endl;

    // H_l is the histogram of (local) nnz of buckets
    std::vector<index_t> H_l(n_buckets, 0);

    for(nnz_t i=0; i<initial_nnz_l; i++){
//        H_l[lower_bound2(&firstSplit[0], &firstSplit[n_buckets], data[i].row)]++;
        least_bucket += lower_bound2(&firstSplit[least_bucket], &firstSplit[last_bucket], data[i].row);
//        if (rank==0) std::cout << "row = " << data[i].row << ", least_bucket = " << least_bucket << std::endl;
        H_l[least_bucket]++;
    }

//    unsigned long bucket_idx = lower_bound2(&firstSplit[0], &firstSplit[n_buckets], data[0].row);
//    H_l[bucket_idx]++;

//    for(unsigned int i=1; i<initial_nnz_l; i++) {
//        if (data[i].row >= firstSplit[bucket_idx+1])
//            bucket_idx++;
//        H_l[bucket_idx]++;
//    }

//    if (rank==1){
//        std::cout << "initial_nnz_l = " << initial_nnz_l << std::endl;
//        std::cout << "local histogram:" << std::endl;
//        for(unsigned int i=0; i<n_buckets; i++)
//            std::cout << H_l[i] << std::endl;}

    // H_g is the histogram of (global) nnz per bucket
    std::vector<index_t> H_g(n_buckets);
    MPI_Allreduce(&H_l[0], &H_g[0], n_buckets, MPI_UNSIGNED, MPI_SUM, comm);

    H_l.clear();
    H_l.shrink_to_fit();

//    if (rank==1){
//        std::cout << "global histogram:" << std::endl;
//        for(unsigned int i=0; i<n_buckets; i++){
//            std::cout << H_g[i] << std::endl;}}

    std::vector<index_t> H_g_scan(n_buckets);
    H_g_scan[0] = H_g[0];
    for (index_t i=1; i<n_buckets; i++)
        H_g_scan[i] = H_g[i] + H_g_scan[i-1];

    H_g.clear();
    H_g.shrink_to_fit();

//    if (rank==0){
//        std::cout << "scan of global histogram:" << std::endl;
//        for(unsigned int i=0; i<n_buckets; i++)
//            std::cout << H_g_scan[i] << std::endl;}

    if(repartition_verbose && rank==0) printf("repartition - step 3!\n");

    index_t procNum = 0;
    split.resize(nprocs+1);
    split[0]=0;
    for (index_t i=1; i<n_buckets; i++){
        //if (rank==0) std::cout << "(procNum+1)*nnz_g/nprocs = " << (procNum+1)*nnz_g/nprocs << std::endl;
        if (H_g_scan[i] > ((procNum+1)*nnz_g/nprocs)){
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
//    M_old = M;

    if(repartition_verbose && rank==0) printf("repartition - step 4!\n");

//    unsigned int M_min_global;
//    MPI_Allreduce(&M, &M_min_global, 1, MPI_UNSIGNED, MPI_MIN, comm);

    // *************************** exchange data ****************************

//    std::vector<sendrecv_t> sendSizeArray(nprocs, 0);
    int* sendSizeArray = (int*)malloc(sizeof(int)*nprocs);
    std::fill(&sendSizeArray[0], &sendSizeArray[nprocs], 0);

    index_t least_proc, last_proc;
    least_proc = lower_bound2(&split[0], &split[nprocs], data[0].row);
    last_proc  = lower_bound2(&split[0], &split[nprocs], data.back().row);
    last_proc++;

//    if (rank==1)
//        std::cout << "\nleast_proc:" << least_proc << ", last_proc = " << last_proc << std::endl;

    for (nnz_t i=0; i<initial_nnz_l; i++){
//        least_proc = lower_bound2(&split[0], &split[nprocs], data[i].row);
        least_proc += lower_bound2(&split[least_proc], &split[last_proc], data[i].row);
//        if (rank==1) std::cout << "least_proc:" << least_proc << std::endl;
        sendSizeArray[least_proc]++;
    }

//    if (rank==0){
//        std::cout << "sendSizeArray:" << std::endl;
//        for(long i=0;i<nprocs;i++)
//            std::cout << sendSizeArray[i] << std::endl;}

//    std::vector<sendrecv_t> recvSizeArray(nprocs);
    int* recvSizeArray = (int*)malloc(sizeof(int)*nprocs);
    MPI_Alltoall(&sendSizeArray[0], 1, MPI_INT, &recvSizeArray[0], 1, MPI_INT, comm);

/*    if (rank==0){
        std::cout << "recvSizeArray:" << std::endl;
        for(long i=0;i<nprocs;i++)
            std::cout << recvSizeArray[i] << std::endl;
    }*/

    int* sOffset = (int*)malloc(sizeof(int)*nprocs);
//    std::vector<index_t > sOffset(nprocs);
    sOffset[0] = 0;
    for (int i=1; i<nprocs; i++)
        sOffset[i] = sendSizeArray[i-1] + sOffset[i-1];

/*    if (rank==0){
        std::cout << "sOffset:" << std::endl;
        for(long i=0;i<nprocs;i++)
            std::cout << sOffset[i] << std::endl;
    }*/

    int* rOffset = (int*)malloc(sizeof(int)*nprocs);
//    std::vector<index_t > rOffset(nprocs);
    rOffset[0] = 0;
    for (int i=1; i<nprocs; i++)
        rOffset[i] = recvSizeArray[i-1] + rOffset[i-1];

/*    if (rank==0){
        std::cout << "rOffset:" << std::endl;
        for(long i=0;i<nprocs;i++)
            std::cout << rOffset[i] << std::endl;
    }*/

    if(repartition_verbose && rank==0) printf("repartition - step 5!\n");

    nnz_l = rOffset[nprocs-1] + recvSizeArray[nprocs-1];
//    printf("rank=%d \t A.nnz_l=%lu \t A.nnz_g=%lu \n", rank, nnz_l, nnz_g);

    if(repartition_verbose && rank==0) printf("repartition - step 6!\n");

    entry.resize(nnz_l);
//    MPI_Alltoallv(sendBuf, sendSizeArray, sOffset, cooEntry::mpi_datatype(), &entry[0], recvSizeArray, rOffset, cooEntry::mpi_datatype(), comm);
    MPI_Alltoallv(&data[0],  sendSizeArray, sOffset, cooEntry::mpi_datatype(),
                  &entry[0], recvSizeArray, rOffset, cooEntry::mpi_datatype(), comm);

    data.clear();
    data.shrink_to_fit();

    std::sort(entry.begin(), entry.end());

    free(sendSizeArray);
    free(recvSizeArray);
    free(sOffset);
    free(rOffset);
//    free(sendBuf);

//    MPI_Barrier(comm);
//    if (rank==0){
//        std::cout << "\nrank = " << rank << ", nnz_l = " << nnz_l << std::endl;
//        for (int i=0; i<nnz_l; i++)
//            std::cout << "i=" << i << "\t" << entry[i].row << "\t" << entry[i].col << "\t" << entry[i].val << std::endl;}
//    MPI_Barrier(comm);
//    if (rank==1){
//        std::cout << "\nrank = " << rank << ", nnz_l = " << nnz_l << std::endl;
//        for (int i=0; i<nnz_l; i++)
//            std::cout << "i=" << i << "\t" << entry[i].row << "\t" << entry[i].col << "\t" << entry[i].val << std::endl;}
//    MPI_Barrier(comm);

//    MPI_Barrier(comm); printf("repartition: rank = %d, Mbig = %u, M = %u, nnz_g = %u, nnz_l = %u \n", rank, Mbig, M, nnz_g, nnz_l); MPI_Barrier(comm);

    if(repartition_verbose && rank==0) printf("repartition - step 7!\n");


    return 0;
}


int saena_matrix::repartition2(){
    // before using this function these variables of SaenaMatrix should be set:
    // Mbig", "nnz_g", "initial_nnz_l", "data"

    // the following variables of SaenaMatrix class will be set in this function:
    // "nnz_l", "M", "split", "entry"

    bool repartition_verbose = false;

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(repartition_verbose && rank==0) printf("repartition - step 1!\n");

    density = (nnz_g / double(Mbig)) / (Mbig);

    // *************************** exchange data ****************************

    std::sort(data.begin(), data.end(), row_major);

    long tempIndex;
//    int sendSizeArray[nprocs];
    int* sendSizeArray = (int*)malloc(sizeof(int)*nprocs);
    std::fill(&sendSizeArray[0], &sendSizeArray[nprocs], 0);
//    for (unsigned int i=0; i<initial_nnz_l; i++){
//        tempIndex = lower_bound2(&split[0], &split[nprocs], data[i].row);
//        sendSizeArray[tempIndex]++;
//    }

    long least_proc, last_proc;
    least_proc = lower_bound2(&split[0], &split[nprocs], data[0].row);
    last_proc  = lower_bound2(&split[0], &split[nprocs], data.back().row);
    last_proc++;

//    if (rank==1)
//        std::cout << "\nleast_proc:" << least_proc << ", last_proc = " << last_proc << std::endl;

    for (nnz_t i=0; i<initial_nnz_l; i++){
        least_proc += lower_bound2(&split[least_proc], &split[last_proc], data[i].row);
        sendSizeArray[least_proc]++;
    }

/*    if (rank==0){
        std::cout << "sendSizeArray:" << std::endl;
        for(long i=0;i<nprocs;i++)
            std::cout << sendSizeArray[i] << std::endl;
    }*/

//    int recvSizeArray[nprocs];
    int* recvSizeArray = (int*)malloc(sizeof(int)*nprocs);
    MPI_Alltoall(sendSizeArray, 1, MPI_INT, recvSizeArray, 1, MPI_INT, comm);

/*    if (rank==0){
        std::cout << "recvSizeArray:" << std::endl;
        for(long i=0;i<nprocs;i++)
            std::cout << recvSizeArray[i] << std::endl;
    }*/

//    int sOffset[nprocs];
    int* sOffset = (int*)malloc(sizeof(int)*nprocs);
    sOffset[0] = 0;
    for (int i=1; i<nprocs; i++)
        sOffset[i] = sendSizeArray[i-1] + sOffset[i-1];

/*    if (rank==0){
        std::cout << "sOffset:" << std::endl;
        for(long i=0;i<nprocs;i++)
            std::cout << sOffset[i] << std::endl;
    }*/

//    int rOffset[nprocs];
    int* rOffset = (int*)malloc(sizeof(int)*nprocs);
    rOffset[0] = 0;
    for (int i=1; i<nprocs; i++)
        rOffset[i] = recvSizeArray[i-1] + rOffset[i-1];

/*    if (rank==0){
        std::cout << "rOffset:" << std::endl;
        for(long i=0;i<nprocs;i++)
            std::cout << rOffset[i] << std::endl;
    }*/

    if(repartition_verbose && rank==0) printf("repartition - step 2!\n");

    unsigned long nnz_l_temp = rOffset[nprocs-1] + recvSizeArray[nprocs-1];
//    printf("rank=%d \t A.nnz_l=%u \t A.nnz_g=%u \n", rank, nnz_l, nnz_g);
    if(nnz_l_temp != nnz_l) printf("error: number of local nonzeros is changed on processor %d during the matrix update!\n", rank);

    if(repartition_verbose && rank==0) printf("repartition - step 3!\n");

    entry.clear();
    entry.resize(nnz_l_temp);
    entry.shrink_to_fit();

    MPI_Alltoallv(&data[0], sendSizeArray, sOffset, cooEntry::mpi_datatype(), &entry[0], recvSizeArray, rOffset, cooEntry::mpi_datatype(), comm);

    std::sort(entry.begin(), entry.end());

    // clear data and free memory.
    data.clear();
    data.shrink_to_fit();

    free(sendSizeArray);
    free(recvSizeArray);
    free(sOffset);
    free(rOffset);

//    MPI_Barrier(comm);
//    if (rank==0){
//        std::cout << "\nrank = " << rank << ", nnz_l = " << nnz_l << std::endl;
//        for (int i=0; i<nnz_l; i++)
//            std::cout << "i=" << i << "\t" << entry[i].row << "\t" << entry[i].col << "\t" << entry[i].val << std::endl;}
//    MPI_Barrier(comm);
//    if (rank==1){
//        std::cout << "\nrank = " << rank << ", nnz_l = " << nnz_l << std::endl;
//        for (int i=0; i<nnz_l; i++)
//            std::cout << "i=" << i << "\t" << entry[i].row << "\t" << entry[i].col << "\t" << entry[i].val << std::endl;}
//    MPI_Barrier(comm);

//    MPI_Barrier(comm); printf("repartition: rank = %d, Mbig = %u, M = %u, nnz_g = %u, nnz_l = %u \n", rank, Mbig, M, nnz_g, nnz_l); MPI_Barrier(comm);

    if(repartition_verbose && rank==0) printf("repartition - step 4!\n");

    return 0;
}

// this version of repartition3() is WITHOUT cpu shrinking.
/*
int saena_matrix::repartition3(){

    // summary: number of buckets are computed based of the number fo rows and number of processors.
    // firstSplit[] is of size n_buckets+1 and is a row partition of the matrix with almost equal number of rows.
    // then the buckets (firsSplit) are combined to have almost the same number of nonzeros. This is split[].

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    bool repartition_verbose = false;

    if(repartition_verbose && rank==0) printf("repartition - step 1!\n");

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

    if(repartition_verbose && rank==0) printf("repartition - step 2!\n");

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

    if(repartition_verbose && rank==0) printf("repartition - step 3!\n");

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

    if(repartition_verbose && rank==0) printf("repartition - step 4!\n");

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

    if(repartition_verbose && rank==0) printf("repartition - step 5!\n");

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

    if(repartition_verbose && rank==0) printf("repartition - step 6!\n");

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

    if(repartition_verbose && rank==0) printf("repartition - step 7!\n");

    return 0;
}
*/


int saena_matrix::repartition3(){

    // summary: number of buckets are computed based of the number fo rows and number of processors.
    // firstSplit[] is of size n_buckets+1 and is a row partition of the matrix with almost equal number of rows.
    // then the buckets (firsSplit) are combined to have almost the same number of nonzeros. This is split[].
    // note: this version of repartition3() is WITH cpu shrinking.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    bool repartition_verbose = false;

    if(repartition_verbose && rank==0) printf("repartition3 - step 1!\n");

    density = (nnz_g / double(Mbig)) / (Mbig);

//    MPI_Barrier(comm);
//    printf("repartition3 - start! rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu \n",
//           rank, Mbig, M, nnz_g, nnz_l);
//    MPI_Barrier(comm);

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
        // todo: it may be better to set nprocs=Mbig and work with only the first Mbig processors.
        if(rank == 0)
            std::cout << "number of MPI tasks cannot be greater than the number of rows of the matrix." << std::endl;
        MPI_Finalize();
    }

//    if (rank==0) std::cout << "n_buckets = " << n_buckets << ", Mbig = " << Mbig << std::endl;

    std::vector<index_t> splitOffset(n_buckets);
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

    if(repartition_verbose && rank==0) printf("repartition3 - step 2!\n");

    std::vector<index_t > firstSplit(n_buckets+1);
    firstSplit[0] = 0;
    for(index_t i=1; i<n_buckets; i++)
        firstSplit[i] = firstSplit[i-1] + splitOffset[i];
    firstSplit[n_buckets] = Mbig;

    splitOffset.clear();
    splitOffset.shrink_to_fit();

//    if (rank==0){
//        std::cout << "\nfirstSplit:" << std::endl;
//        for(long i=0; i<n_buckets+1; i++)
//            std::cout << firstSplit[i] << std::endl;}

    std::sort(entry.begin(), entry.end(), row_major);

//    if (rank==0){
//        std::cout << "entry row-major: nnz = " << initial_nnz_l << std::endl;
//        for(unsigned int i=0; i<data.size(); i++)
//            std::cout << data[i] << std::endl;}

    long least_bucket, last_bucket;
    least_bucket = lower_bound2(&firstSplit[0], &firstSplit[n_buckets], entry[0].row);
    last_bucket  = lower_bound2(&firstSplit[0], &firstSplit[n_buckets], entry.back().row);
    last_bucket++;

//    if (rank==0)
//        std::cout << "least_bucket:" << least_bucket << ", last_bucket = " << last_bucket << std::endl;

    // H_l is the histogram of (local) nnz of buckets
    std::vector<index_t > H_l(n_buckets, 0);

    initial_nnz_l = nnz_l;
    // H_l is the histogram of (local) nnz per bucket
    for(nnz_t i=0; i<initial_nnz_l; i++){
        least_bucket += lower_bound2(&firstSplit[least_bucket], &firstSplit[last_bucket], entry[i].row);
        H_l[least_bucket]++;
    }

//    if (rank==0){
//        std::cout << "\ninitial_nnz_l = " << initial_nnz_l << std::endl;
//        std::cout << "local histogram:" << std::endl;
//        for(unsigned int i=0; i<n_buckets; i++)
//            std::cout << H_l[i] << std::endl;
//    }

    // H_g is the histogram of (global) nnz per bucket
    std::vector<index_t> H_g(n_buckets);
    MPI_Allreduce(&H_l[0], &H_g[0], n_buckets, MPI_UNSIGNED, MPI_SUM, comm);

    H_l.clear();
    H_l.shrink_to_fit();

//    if (rank==1){
//        std::cout << "global histogram:" << std::endl;
//        for(unsigned int i=0; i<n_buckets; i++){
//            std::cout << H_g[i] << std::endl;
//        }
//    }

    std::vector<index_t > H_g_scan(n_buckets);
    H_g_scan[0] = H_g[0];
    for (index_t i=1; i<n_buckets; i++)
        H_g_scan[i] = H_g[i] + H_g_scan[i-1];

    H_g.clear();
    H_g.shrink_to_fit();

//    if (rank==0){
//        std::cout << "scan of global histogram:" << std::endl;
//        for(unsigned int i=0; i<n_buckets; i++)
//            std::cout << H_g_scan[i] << std::endl;}

    if(repartition_verbose && rank==0) printf("repartition3 - step 3!\n");

//    if (rank==0){
//        std::cout << std::endl << "split old:" << std::endl;
//        for(unsigned int i=0; i<nprocs+1; i++)
//            std::cout << split[i] << std::endl;
//        std::cout << std::endl;}

    // determine number of rows on each proc based on having almost the same number of nonzeros per proc.
    // -------------------------------------------

    split_old = split;
    long procNum = 0;
    for (index_t i = 1; i < n_buckets; i++){
        if (H_g_scan[i] > (procNum+1)*nnz_g/nprocs){
            procNum++;
            split[procNum] = firstSplit[i];
        }
    }
    split[nprocs] = Mbig;

//    print_vector(split, -1, comm);

//    if (rank==0){
//        std::cout << std::endl << "split:" << std::endl;
//        for(unsigned int i=0; i<nprocs+1; i++)
//            std::cout << split[i] << std::endl;
//        std::cout << std::endl;}

    H_g_scan.clear();
    H_g_scan.shrink_to_fit();
    firstSplit.clear();
    firstSplit.shrink_to_fit();

    // set the number of rows for each process
    M = split[rank+1] - split[rank];
//    M_old = M;

    if(repartition_verbose && rank==0) printf("repartition3 - step 4!\n");

    // *************************** exchange data ****************************

    std::vector<int> send_size_array(nprocs, 0);
//    for (unsigned int i=0; i<initial_nnz_l; i++){
//        tempIndex = lower_bound2(&split[0], &split[nprocs], entry[i].row);
//        sendSizeArray[tempIndex]++;
//    }

    long least_proc, last_proc;
    least_proc = lower_bound2(&split[0], &split[nprocs], entry[0].row);
    last_proc  = lower_bound2(&split[0], &split[nprocs], entry.back().row);
    last_proc++;

//    if (rank==1)
//        std::cout << "\nleast_proc:" << least_proc << ", last_proc = " << last_proc << std::endl;

    for (nnz_t i=0; i<initial_nnz_l; i++){
        least_proc += lower_bound2(&split[least_proc], &split[last_proc], entry[i].row);
        send_size_array[least_proc]++;
    }

//    if (rank==5){
//        std::cout << "sendSizeArray: " << rank << std::endl;
//        for(long i=0;i<nprocs;i++)
//            std::cout << sendSizeArray[i] << std::endl;}

    // this part is for cpu shrinking. assign all the rows on non-root procs to their roots.
    // ---------------------------------
//    if(enable_shrink && nprocs >= cpu_shrink_thre2 && (last_M_shrink >= (Mbig * cpu_shrink_thre1)) ){
//    if(rank==0) printf("last_density_shrink = %f, density = %f, inequality = %d \n", last_density_shrink, density, (density >= (last_density_shrink * cpu_shrink_thre1)));
    if(enable_shrink && (nprocs >= cpu_shrink_thre2) && (density >= (last_density_shrink * cpu_shrink_thre1)) ){
        shrinked = true;
        last_M_shrink = Mbig;
//        last_nnz_shrink = nnz_g;
        last_density_shrink = density;
        double remainder;
        int root_cpu = nprocs;
        for(int proc = nprocs-1; proc > 0; proc--){
            remainder = proc % cpu_shrink_thre2;
//        if(rank==0) printf("proc = %ld, remainder = %f\n", proc, remainder);
            if(remainder == 0)
                root_cpu = proc;
            else{
                split[proc] = split[root_cpu];
//                sendSizeArray[root_cpu] += sendSizeArray[proc];
//                sendSizeArray[proc] = 0;
            }
        }

//        M_old = M;
        M = split[rank+1] - split[rank];

//        if (rank==0){
//            std::cout << std::endl << "split:" << std::endl;
//            for(unsigned int i=0; i<nprocs+1; i++)
//                std::cout << split[i] << std::endl;
//            std::cout << std::endl;}

        root_cpu = 0;
        for(int proc = 0; proc < nprocs; proc++){
            remainder = proc % cpu_shrink_thre2;
//        if(rank==0) printf("proc = %ld, remainder = %f\n", proc, remainder);
            if(remainder == 0)
                root_cpu = proc;
            else{
                send_size_array[root_cpu] += send_size_array[proc];
                send_size_array[proc] = 0;
            }
        }

//        if (rank==5){
//            std::cout << "sendSizeArray: " << rank << std::endl;
//            for(long i=0;i<nprocs;i++)
//                std::cout << sendSizeArray[i] << std::endl;}
    }

    std::vector<int> recv_size_array(nprocs);
    MPI_Alltoall(&send_size_array[0], 1, MPI_INT, &recv_size_array[0], 1, MPI_INT, comm);

//    if (rank==1){
//        std::cout << "recvSizeArray:" << std::endl;
//        for(long i=0;i<nprocs;i++)
//            std::cout << recvSizeArray[i] << std::endl;}

//    int* sOffset = (int*)malloc(sizeof(int)*nprocs);
    std::vector<int> send_offset(nprocs);
    send_offset[0] = 0;
    for (int i=1; i<nprocs; i++)
        send_offset[i] = send_size_array[i-1] + send_offset[i-1];

//    if (rank==0){
//        std::cout << "sOffset:" << std::endl;
//        for(long i=0;i<nprocs;i++)
//            std::cout << sOffset[i] << std::endl;}

//    int* rOffset = (int*)malloc(sizeof(int)*nprocs);
    std::vector<int> recv_offset(nprocs);
    recv_offset[0] = 0;
    for (int i=1; i<nprocs; i++)
        recv_offset[i] = recv_size_array[i-1] + recv_offset[i-1];

//    if (rank==1){
//        std::cout << "rOffset:" << std::endl;
//        for(long i=0;i<nprocs;i++)
//            std::cout << rOffset[i] << std::endl;}

    if(repartition_verbose && rank==0) printf("repartition3 - step 5!\n");

    nnz_l = recv_offset[nprocs-1] + recv_size_array[nprocs-1];
//    printf("rank=%d \t A.nnz_l=%u \t A.nnz_g=%u \n", rank, nnz_l, nnz_g);

    if(repartition_verbose && rank==0) printf("repartition3 - step 6!\n");

    std::vector<cooEntry> entry_old = entry;
    entry.resize(nnz_l);
    entry.shrink_to_fit();

    MPI_Alltoallv(&entry_old[0], &send_size_array[0], &send_offset[0], cooEntry::mpi_datatype(),
                  &entry[0],     &recv_size_array[0], &recv_offset[0], cooEntry::mpi_datatype(), comm);

    std::sort(entry.begin(), entry.end());

//    MPI_Barrier(comm);
//    if (rank==0){
//        std::cout << "\nrank = " << rank << ", nnz_l = " << nnz_l << std::endl;
//        for (int i=0; i<nnz_l; i++)
//            std::cout << i << "\t" << entry[i] << std::endl;}
//    MPI_Barrier(comm);
//    if (rank==2){
//        std::cout << "\nrank = " << rank << ", nnz_l = " << nnz_l << std::endl;
//        for (int i=0; i<nnz_l; i++)
//            std::cout << i << "\t" << entry[i] << std::endl;}
//    MPI_Barrier(comm);

    if(repartition_verbose) {
        MPI_Barrier(comm);
        printf("repartition3 - end! rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu \n",
               rank, Mbig, M, nnz_g, nnz_l);
        MPI_Barrier(comm);
    }

    return 0;
}


int saena_matrix::repartition4(){

    // summary: number of buckets are computed based of the number fo rows and number of processors.
    // firstSplit[] is of size n_buckets+1 and is a row partition of the matrix with almost equal number of rows.
    // then the buckets (firsSplit) are combined to have almost the same number of nonzeros. This is split[].
    // note: this version of repartition4() is WITH cpu shrinking.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    bool repartition_verbose = false;

    if(rank==0) printf("\nuse repartition based on the number of rows for the next level!\n");
    if(repartition_verbose && rank==0) printf("repartition4 - step 1!\n");

    density = (nnz_g / double(Mbig)) / (Mbig);

//    MPI_Barrier(comm);
//    printf("repartition4 - start! rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu \n",
//           rank, Mbig, M, nnz_g, nnz_l);
//    MPI_Barrier(comm);

    // *************************** find splitters ****************************
    // split the matrix row-wise by splitters, so each processor gets almost equal number of rows

    std::vector<index_t> splitOffset(nprocs);
    auto baseOffset = int(floor(1.0*Mbig/nprocs));
    float offsetRes = float(1.0*Mbig/nprocs) - baseOffset;
//    if (rank==0) std::cout << "baseOffset = " << baseOffset << ", offsetRes = " << offsetRes << std::endl;
    float offsetResSum = 0;
    splitOffset[0] = 0;
    for(unsigned int i=1; i<nprocs; i++){
        splitOffset[i] = baseOffset;
        offsetResSum += offsetRes;
        if (offsetResSum >= 1){
            splitOffset[i]++;
            offsetResSum -= 1;
        }
    }

//    if (rank==0){
//        std::cout << "splitOffset:" << std::endl;
//        for(long i=0; i<nprocs; i++)
//            std::cout << splitOffset[i] << std::endl;}

    if(repartition_verbose && rank==0) printf("repartition4 - step 2!\n");

    split_old = split;
    split[0] = 0;
    for(index_t i=1; i<nprocs; i++)
        split[i] = split[i-1] + splitOffset[i];
    split[nprocs] = Mbig;

    splitOffset.clear();
    splitOffset.shrink_to_fit();

//    if (rank==0){
//        std::cout << std::endl << "split:" << std::endl;
//        for(unsigned int i=0; i<nprocs+1; i++)
//            std::cout << split[i] << std::endl;
//        std::cout << std::endl;}

    // set the number of rows for each process
    M = split[rank+1] - split[rank];
//    M_old = M;

    if(repartition_verbose && rank==0) printf("repartition4 - step 4!\n");

//    unsigned int M_min_global;
//    MPI_Allreduce(&M, &M_min_global, 1, MPI_UNSIGNED, MPI_MIN, comm);

    // *************************** exchange data ****************************

    int* sendSizeArray = (int*)malloc(sizeof(int)*nprocs);
    std::fill(&sendSizeArray[0], &sendSizeArray[nprocs], 0);
//    for (unsigned int i=0; i<initial_nnz_l; i++){
//        tempIndex = lower_bound2(&split[0], &split[nprocs], entry[i].row);
//        sendSizeArray[tempIndex]++;
//    }

    std::sort(entry.begin(), entry.end(), row_major);

    long least_proc, last_proc;
    least_proc = lower_bound2(&split[0], &split[nprocs], entry[0].row);
    last_proc  = lower_bound2(&split[0], &split[nprocs], entry.back().row);
    last_proc++;

//    if (rank==1)
//        std::cout << "\nleast_proc:" << least_proc << ", last_proc = " << last_proc << std::endl;

    for (nnz_t i=0; i<nnz_l; i++){
        least_proc += lower_bound2(&split[least_proc], &split[last_proc], entry[i].row);
        sendSizeArray[least_proc]++;
    }

//    if (rank==5){
//        std::cout << "sendSizeArray: " << rank << std::endl;
//        for(long i=0;i<nprocs;i++)
//            std::cout << sendSizeArray[i] << std::endl;}

    // this part is for cpu shrinking. assign all the rows on non-root procs to their roots.
    // ---------------------------------
//    if(enable_shrink && nprocs >= cpu_shrink_thre2 && (last_M_shrink >= (Mbig * cpu_shrink_thre1)) ){
//    if(rank==0) printf("last_density_shrink = %f, density = %f, inequality = %d \n", last_density_shrink, density, (density >= (last_density_shrink * cpu_shrink_thre1)));
    if(enable_shrink && (nprocs >= cpu_shrink_thre2) && (density >= (last_density_shrink * cpu_shrink_thre1)) ){
        shrinked = true;
        last_M_shrink = Mbig;
//        last_nnz_shrink = nnz_g;
        last_density_shrink = density;
        double remainder;
        int root_cpu = nprocs;
        for(int proc = nprocs-1; proc > 0; proc--){
            remainder = proc % cpu_shrink_thre2;
//        if(rank==0) printf("proc = %ld, remainder = %f\n", proc, remainder);
            if(remainder == 0)
                root_cpu = proc;
            else{
                split[proc] = split[root_cpu];
//                sendSizeArray[root_cpu] += sendSizeArray[proc];
//                sendSizeArray[proc] = 0;
            }
        }

//        M_old = M;
        M = split[rank+1] - split[rank];

//        if (rank==0){
//            std::cout << std::endl << "split:" << std::endl;
//            for(unsigned int i=0; i<nprocs+1; i++)
//                std::cout << split[i] << std::endl;
//            std::cout << std::endl;}

        root_cpu = 0;
        for(int proc = 0; proc < nprocs; proc++){
            remainder = proc % cpu_shrink_thre2;
//        if(rank==0) printf("proc = %ld, remainder = %f\n", proc, remainder);
            if(remainder == 0)
                root_cpu = proc;
            else{
                sendSizeArray[root_cpu] += sendSizeArray[proc];
                sendSizeArray[proc] = 0;
            }
        }

//        if (rank==5){
//            std::cout << "sendSizeArray: " << rank << std::endl;
//            for(long i=0;i<nprocs;i++)
//                std::cout << sendSizeArray[i] << std::endl;}
    }

    int* recvSizeArray = (int*)malloc(sizeof(int)*nprocs);
    MPI_Alltoall(sendSizeArray, 1, MPI_INT, recvSizeArray, 1, MPI_INT, comm);

//    if (rank==1){
//        std::cout << "recvSizeArray:" << std::endl;
//        for(long i=0;i<nprocs;i++)
//            std::cout << recvSizeArray[i] << std::endl;}

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

//    if (rank==1){
//        std::cout << "rOffset:" << std::endl;
//        for(long i=0;i<nprocs;i++)
//            std::cout << rOffset[i] << std::endl;}

    if(repartition_verbose && rank==0) printf("repartition4 - step 5!\n");

    nnz_l = rOffset[nprocs-1] + recvSizeArray[nprocs-1];
//    printf("rank=%d \t A.nnz_l=%lu \t A.nnz_g=%lu \n", rank, nnz_l, nnz_g);

    if(repartition_verbose && rank==0) printf("repartition4 - step 6!\n");

    std::vector<cooEntry> entry_old = entry;
    entry.clear(); // todo: check if it is fine to remove this.
    entry.resize(nnz_l);
    entry.shrink_to_fit();

    MPI_Alltoallv(&entry_old[0], sendSizeArray, sOffset, cooEntry::mpi_datatype(),
                  &entry[0],     recvSizeArray, rOffset, cooEntry::mpi_datatype(), comm);

    free(sendSizeArray);
    free(recvSizeArray);
    free(sOffset);
    free(rOffset);

    std::sort(entry.begin(), entry.end());

//    MPI_Barrier(comm);
//    if (rank==0){
//        std::cout << "\nrank = " << rank << ", nnz_l = " << nnz_l << std::endl;
//        for (int i=0; i<nnz_l; i++)
//            std::cout << i << "\t" << entry[i] << std::endl;}
//    MPI_Barrier(comm);
//    if (rank==2){
//        std::cout << "\nrank = " << rank << ", nnz_l = " << nnz_l << std::endl;
//        for (int i=0; i<nnz_l; i++)
//            std::cout << i << "\t" << entry[i] << std::endl;}
//    MPI_Barrier(comm);

    if(repartition_verbose) {
        MPI_Barrier(comm);
        printf("repartition4 - step 7! rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu \n",
               rank, Mbig, M, nnz_g, nnz_l);
        MPI_Barrier(comm);
    }

    return 0;
}


int saena_matrix::shrink_cpu(){

    // if number of rows on Ac < threshold*number of rows on A, then shrink.
    // redistribute Ac from processes 4k+1, 4k+2 and 4k+3 to process 4k.
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    bool verbose_shrink = false;

//    MPI_Barrier(comm);
//    if(rank==0) printf("\n****************************\n");
//    if(rank==0) printf("***********SHRINK***********\n");
//    if(rank==0) printf("****************************\n\n");
//    MPI_Barrier(comm);

//    MPI_Barrier(comm); printf("rank = %d \tnnz_l = %u \n", rank, nnz_l); MPI_Barrier(comm);

//    MPI_Barrier(comm);
//    if(rank == 0){
//        std::cout << "\nbefore shrinking!!!" <<std::endl;
//        std::cout << "\nrank = " << rank << ", size = " << entry.size() <<std::endl;
//        for(i=0; i < entry.size(); i++)
//            std::cout << i << "\t" <<entry[i]  <<std::endl;
//    }
//    MPI_Barrier(comm);
//    if(rank == 1){
//        std::cout << "\nrank = " << rank << ", size = " << entry.size() <<std::endl;
//        for(i=0; i < entry.size(); i++)
//            std::cout << i << "\t" << entry[i]  <<std::endl;
//    }
//    MPI_Barrier(comm);

    // assume cpu_shrink_thre2 is 4 (it is simpler to explain)
    // 1 - create a new comm, consisting only of processes 4k, 4k+1, 4k+2 and 4k+3 (with new ranks 0,1,2,3)
    int color = rank / cpu_shrink_thre2;
    MPI_Comm_split(comm, color, rank, &comm_horizontal);

    int rank_new, nprocs_new;
    MPI_Comm_size(comm_horizontal, &nprocs_new);
    MPI_Comm_rank(comm_horizontal, &rank_new);

//    MPI_Barrier(comm_horizontal);
//    printf("rank = %d, rank_new = %d on Ac->comm_horizontal \n", rank, rank_new);
//    MPI_Barrier(comm_horizontal);

/*
    // 2 - update the number of rows on process 4k, and resize "entry".
    unsigned int Ac_M_neighbors_total = 0;
    unsigned int Ac_nnz_neighbors_total = 0;
    MPI_Reduce(&M, &Ac_M_neighbors_total, 1, MPI_UNSIGNED, MPI_SUM, 0, comm_horizontal);
    MPI_Reduce(&nnz_l, &Ac_nnz_neighbors_total, 1, MPI_UNSIGNED, MPI_SUM, 0, comm_horizontal);

    if(rank_new == 0){
        M = Ac_M_neighbors_total;
        entry.resize(Ac_nnz_neighbors_total);
//        printf("rank = %d, Ac_M_neighbors = %d \n", rank, Ac_M_neighbors_total);
//        printf("rank = %d, Ac_nnz_neighbors = %d \n", rank, Ac_nnz_neighbors_total);
    }

    // last cpu that its right neighbors are going be shrinked to.
    auto last_root_cpu = (unsigned int)floor(nprocs/cpu_shrink_thre2) * cpu_shrink_thre2;
//    printf("last_root_cpu = %u\n", last_root_cpu);

    int neigbor_rank;
    unsigned int A_recv_nnz = 0; // set to 0 just to avoid "not initialized" warning
    unsigned long offset = nnz_l; // put the data on root from its neighbors at the end of entry[] which is of size nnz_l
    if(nprocs_new > 1) { // if there is no neighbor, skip.
        for (neigbor_rank = 1; neigbor_rank < cpu_shrink_thre2; neigbor_rank++) {

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
                MPI_Recv(&A_recv_nnz, 1, MPI_UNSIGNED, neigbor_rank, 0, comm_horizontal, MPI_STATUS_IGNORE);

            if (rank_new == neigbor_rank)
                MPI_Send(&nnz_l, 1, MPI_UNSIGNED, 0, 0, comm_horizontal);

            // 4 - send and receive Ac.
            if (rank_new == 0) {
//                printf("rank = %d, neigbor_rank = %d, recv size = %u, offset = %lu \n", rank, neigbor_rank, A_recv_nnz, offset);
                MPI_Recv(&*(entry.begin() + offset), A_recv_nnz, cooEntry::mpi_datatype(), neigbor_rank, 1,
                         comm_horizontal, MPI_STATUS_IGNORE);
                offset += A_recv_nnz; // set offset for the next iteration
            }

            if (rank_new == neigbor_rank) {
//                printf("rank = %d, neigbor_rank = %d, send size = %u, offset = %lu \n", rank, neigbor_rank, Ac->nnz_l, offset);
                MPI_Send(&*entry.begin(), nnz_l, cooEntry::mpi_datatype(), 0, 1, comm_horizontal);
            }

            // update local index for rows
//        if(rank_new == 0)
//            for(i=0; i < A_recv_nnz; i++)
//                Ac->entry[offset + i].row += Ac->split[rank + neigbor_rank] - Ac->split[rank];
        }
    }

    // even though the entries are sorted before shrinking, after shrinking they still need to be sorted locally,
    // because remote elements damage sorting after shrinking.
    std::sort(entry.begin(), entry.end());
*/
//    MPI_Barrier(comm); MPI_Barrier(Ac->comm_horizontal);
//    if(rank == 2){
//        std::cout << "\nafter shrinking: rank = " << rank << ", size = " << Ac->entry.size() <<std::endl;
//        for(i=0; i < Ac->entry.size(); i++)
//            std::cout << i << "\t" << Ac->entry[i]  <<std::endl;}
//    MPI_Barrier(comm); MPI_Barrier(Ac->comm_horizontal);

//    Ac->active_old_comm = true; // this is used for prolong and post-smooth
    active = false;
    if(rank_new == 0){
        active = true;
//        printf("active: rank = %d, rank_new = %d \n", rank, rank_new);
    }
/*
    // 5 - update 4k.nnz_l and split. nnz_g stays the same, so no need to update.
    if(active){
        nnz_l = entry.size();
        split_old = split; // save the old split for shrinking rhs and u
        split.clear();
        for(i = 0; i < nprocs+1; i++){
//            if(rank==0) printf("P->splitNew[i] = %lu\n", P_splitNew[i]);
            if( i % cpu_shrink_thre2 == 0){
                split.push_back( P_splitNew[i] );
            }
        }
        split.push_back( P_splitNew[nprocs] );
        // assert M == split[rank+1] - split[rank]

//        if(rank==0) {
//            printf("Ac split after shrinking: \n");
//            for (int i = 0; i < Ac->split.size(); i++)
//                printf("%lu \n", Ac->split[i]);}
    }
*/
    // 6 - create a new comm including only processes with 4k rank.
    MPI_Group bigger_group;
    MPI_Comm_group(comm, &bigger_group);
    auto total_active_procs = (unsigned int)ceil((double)nprocs / cpu_shrink_thre2); // note: this is ceiling, not floor.
    std::vector<int> ranks(total_active_procs);
    for(unsigned int i = 0; i < total_active_procs; i++)
        ranks[i] = cpu_shrink_thre2 * i;
//        ranks.push_back(Ac->cpu_shrink_thre2 * i);

//    MPI_Barrier(comm);
//    printf("total_active_procs = %u \n", total_active_procs);
//    for(i=0; i<ranks.size(); i++)
//        if(rank==0) std::cout << ranks[i] << std::endl;

//    MPI_Comm new_comm;
    MPI_Group group_new;
    MPI_Group_incl(bigger_group, total_active_procs, &*ranks.begin(), &group_new);
    MPI_Comm_create_group(comm, group_new, 0, &comm);
//    comm = new_comm;

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

    std::vector<index_t> split_temp = split;
    split.clear();
    if(active){
        split.resize(total_active_procs+1);
        split.shrink_to_fit();
        split[0] = 0;
        split[total_active_procs] = Mbig;
        for(unsigned int i = 1; i < total_active_procs; i++){
//            if(rank==0) printf("%u \t%lu \n", i, split_old[ranks[i]]);
            split[i] = split_temp[ranks[i]];
        }
//        print_vector(split, -1, comm);
    }

//    MPI_Barrier(comm);
//    if(rank==0) {
//        printf("\nsplit:\n");
//        for (long i = 0; i < total_active_procs + 1; i++)
//            printf("%lu \n", split[i]);}

    // 7 - update 4k.nnz_g
//    if(Ac->active)
//        MPI_Allreduce(&Ac->nnz_l, &Ac->nnz_g, 1, MPI_UNSIGNED, MPI_SUM, Ac->comm);

//    if(Ac->active){
//        MPI_Comm_size(Ac->comm, &nprocs);
//        MPI_Comm_rank(Ac->comm, &rank);
//        printf("\n\nrank = %d, nprocs = %d, M = %u, nnz_l = %u, nnz_g = %u, Ac->split[rank+1] = %lu, Ac->split[rank] = %lu \n",
//               rank, nprocs, Ac->M, Ac->nnz_l, Ac->nnz_g, Ac->split[rank+1], Ac->split[rank]);
//    }

//    free(&bigger_group);
//    free(&group_new);
//    free(&comm_new2);
//    if(active)
//        MPI_Comm_free(&new_comm);

//    last_M_shrink = Mbig;
//    shrinked = true;
    return 0;
}


int saena_matrix::matrix_setup() {
    // before using this function these variables of saena_matrix should be set:
    // "Mbig", "M", "nnz_g", "split", "entry",

    // todo: here: check if you can remove here. check if there is another if(active) before calling this function.
    if(active) {
        int nprocs, rank;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);

        assembled = true;
        freeBoolean = true; // use this parameter to know if destructor for SaenaMatrix class should free the variables or not.

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu \n", rank, Mbig, M, nnz_g, nnz_l);
            MPI_Barrier(comm);}

//        MPI_Barrier(comm);
//        printf("matrix_setup: rank = %d, Mbig = %u, M = %u, nnz_g = %lu, nnz_l = %lu \n", rank, Mbig, M, nnz_g, nnz_l);
//        MPI_Barrier(comm);
//        if(rank==0){
//            printf("\nrank = %d\n", rank);
//            for(unsigned int i=0; i<nnz_l; i++)
//                printf("%u \t%lu \t%lu \t%f \n", i, entry[i].row, entry[i].col, entry[i].val);}
//        MPI_Barrier(comm);

//        MPI_Barrier(comm);
//        if (rank==0){
//            std::cout << std::endl << "split:" << std::endl;
//            for(unsigned int i=0; i<nprocs+1; i++)
//                std::cout << split[i] << std::endl;
//            std::cout << std::endl;}
//        MPI_Barrier(comm);

        // *************************** set the inverse of diagonal of A (for smoothers) ****************************

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, invDiag \n", rank);
            MPI_Barrier(comm);
        }

        invDiag.resize(M);
        inverse_diag(invDiag);

//        if(rank==1){
//            for(unsigned int i=0; i<M; i++)
//                std::cout << i << ":\t" << invDiag[i] << std::endl;}

        // computing rhoDA for the prolongation matrix: P = (I - 4/(3*rhoDA) * DA) * P_t
        // rhoDA = min( norm(DA , 1) , norm(DA , inf) )
        /*
            double norm1_local = 0;
            for(unsigned long i=0; i<M; i++)
                norm1_local += abs(invDiag[i]);
            MPI_Allreduce(&norm1_local, &norm1, 1, MPI_DOUBLE, MPI_SUM, comm);

            double normInf_local = invDiag[0];
            for(unsigned long i=1; i<M; i++)
                if( abs(invDiag[i]) > normInf_local )
                    normInf_local = abs(invDiag[i]);
            MPI_Allreduce(&normInf_local, &normInf, 1, MPI_DOUBLE, MPI_MAX, comm);

            if(normInf < norm1)
                rhoDA = normInf;
            else
                rhoDA = norm1;
        */

        // *************************** set and exchange local and remote elements ****************************
        // local elements are elements that correspond to vector elements which are local to this process,
        // and, remote elements correspond to vector elements which should be received from another processes

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, local remote1 \n", rank);
            MPI_Barrier(comm);
        }

        col_remote_size = 0;
        nnz_l_local = 0;
        nnz_l_remote = 0;
        recvCount.assign(nprocs, 0);
        nnzPerRow_local.assign(M, 0);
        nnzPerRow_remote.assign(M, 0);
//        nnzPerRow.assign(M,0);
//        nnzPerCol_local.assign(Mbig,0); // todo: Nbig = Mbig, assuming A is symmetric.
//        nnzPerCol_remote.assign(M,0);

        // take care of the first element here, since there is "col[i-1]" in the for loop below, so "i" cannot start from 0.
//        nnzPerRow[row[0]-split[rank]]++;
        long procNum;
        if(!entry.empty()){
            if (entry[0].col >= split[rank] && entry[0].col < split[rank + 1]) {
                nnzPerRow_local[entry[0].row - split[rank]]++;
//                nnzPerCol_local[col[0]]++;
                nnz_l_local++;
                values_local.push_back(entry[0].val);
                row_local.push_back(entry[0].row);
                col_local.push_back(entry[0].col);
                //vElement_local.push_back(col[0]);
                vElementRep_local.push_back(1);

            } else {
                nnz_l_remote++;
                nnzPerRow_remote[entry[0].row - split[rank]]++;
                values_remote.push_back(entry[0].val);
                row_remote.push_back(entry[0].row);
                col_remote_size++;
                col_remote.push_back(col_remote_size - 1);
                col_remote2.push_back(entry[0].col);
                nnzPerCol_remote.push_back(1);
                vElement_remote.push_back(entry[0].col);
                vElementRep_remote.push_back(1);
//                if(rank==1) printf("col = %u \tprocNum = %ld \n", entry[0].col, lower_bound3(&split[0], &split[nprocs], entry[0].col));
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
                    nnzPerRow_local[entry[i].row - split[rank]]++;
                    values_local.push_back(entry[i].val);
                    row_local.push_back(entry[i].row);
                    col_local.push_back(entry[i].col);

                    if (entry[i].col != entry[i - 1].col) {
                        vElementRep_local.push_back(1);
                    } else {
                        (*(vElementRep_local.end() - 1))++;
                    }
                } else {
                    nnz_l_remote++;
                    nnzPerRow_remote[entry[i].row - split[rank]]++;
                    values_remote.push_back(entry[i].val);
                    row_remote.push_back(entry[i].row);
                    // col_remote2 is the original col value and will be used in making strength matrix. col_remote will be used for matevec.
                    col_remote2.push_back(entry[i].col);

                    if (entry[i].col != entry[i - 1].col) {
                        col_remote_size++;
                        vElement_remote.push_back(entry[i].col);
                        vElementRep_remote.push_back(1);
                        procNum = lower_bound2(&split[0], &split[nprocs], entry[i].col);
//                        if(rank==1) printf("col = %u \tprocNum = %ld \n", entry[i].col, procNum);
                        recvCount[procNum]++;
                        nnzPerCol_remote.push_back(1);
                    } else {
                        (*(vElementRep_remote.end() - 1))++;
                        (*(nnzPerCol_remote.end() - 1))++;
                    }
                    // the original col values are not being used. the ordering starts from 0, and goes up by 1.
                    col_remote.push_back(col_remote_size - 1);
                }
            } // for i
        }

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, local remote2 \n", rank);
            MPI_Barrier(comm);
        }

        // don't receive anything from yourself
        recvCount[rank] = 0;

//        MPI_Barrier(comm);
//        if (rank==0){
//            std::cout << "\nrecvCount: rank = " << rank << std::endl;
//            for(int i=0; i<nprocs; i++)
//                std::cout << "from proc " << i << ": " << recvCount[i] << std::endl;}
//        MPI_Barrier(comm);
//        if (rank==1){
//            std::cout << "\nrecvCount: rank = " << rank << std::endl;
//            for(int i=0; i<nprocs; i++)
//                std::cout << "from proc " << i << ": " << recvCount[i]  << std::endl;}
//        MPI_Barrier(comm);

//        sendCount.assign(M, 0);
        sendCount.resize(nprocs);
        MPI_Alltoall(&recvCount[0], 1, MPI_INT, &sendCount[0], 1, MPI_INT, comm);

//        MPI_Barrier(comm);
//        if (rank==0){
//            std::cout << "\nsendCount: rank=" << rank << std::endl;
//            for(int i=0; i<nprocs; i++)
//                std::cout << i << "= " << sendCount[i] << std::endl;}
//        MPI_Barrier(comm);
//        if (rank==1){
//            std::cout << "\nsendCount: rank=" << rank << std::endl;
//            for(int i=0; i<nprocs; i++)
//                std::cout << i << "= " << sendCount[i] << std::endl;}

        recvCountScan.resize(nprocs);
        sendCountScan.resize(nprocs);
        recvCountScan[0] = 0;
        sendCountScan[0] = 0;
        for (unsigned int i = 1; i < nprocs; i++){
            recvCountScan[i] = recvCountScan[i-1] + recvCount[i-1];
            sendCountScan[i] = sendCountScan[i-1] + sendCount[i-1];
        }

//        MPI_Barrier(comm);
//        if(rank==1)
//            for (unsigned int i = 0; i < nprocs; i++)
//                printf("%u \t%d \t%d \n", i, recvCount[i], recvCountScan[i]);
//        MPI_Barrier(comm);

        numRecvProc = 0;
        numSendProc = 0;
        for (int i = 0; i < nprocs; i++) {
            if (recvCount[i] != 0) {
                numRecvProc++;
                recvProcRank.push_back(i);
                recvProcCount.push_back(recvCount[i]);
            }
            if (sendCount[i] != 0) {
                numSendProc++;
                sendProcRank.push_back(i);
                sendProcCount.push_back(sendCount[i]);
            }
        }

//    if (rank==0) std::cout << "rank=" << rank << ", numRecvProc=" << numRecvProc << ", numSendProc=" << numSendProc << std::endl;

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, local remote3 \n", rank);
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
        recvSize = rdispls[nprocs - 1] + recvCount[nprocs - 1];

//        vIndex = (long *) malloc(sizeof(long) * vIndexSize);
        vIndex.resize(vIndexSize);
        MPI_Alltoallv(&vElement_remote[0], &recvCount[0], &rdispls[0], MPI_UNSIGNED,
                      &vIndex[0],          &sendCount[0], &vdispls[0], MPI_UNSIGNED, comm);

//        if (rank==1){
//            std::cout << "vIndex: rank=" << rank  << std::endl;
//            for(int i=0; i<vIndexSize; i++)
//                std::cout << vIndex[i] << std::endl;}

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, local remote4 \n", rank);
            MPI_Barrier(comm);
        }

        // change the indices from global to local
#pragma omp parallel for
        for (index_t i = 0; i < vIndexSize; i++)
            vIndex[i] -= split[rank];

#pragma omp parallel for
        for (index_t i = 0; i < row_local.size(); i++)
            row_local[i] -= split[rank];

#pragma omp parallel for
        for (index_t i = 0; i < row_remote.size(); i++)
            row_remote[i] -= split[rank];

        // vSend = vector values to send to other procs
        // vecValues = vector values that received from other procs
        // These will be used in matvec and they are set here to reduce the time of matvec.
        vSend.resize(vIndexSize);
        vecValues.resize(recvSize);

        vSendULong.resize(vIndexSize);
        vecValuesULong.resize(recvSize);

        // *************************** find sortings ****************************
        //find the sorting on rows on both local and remote data to be used in matvec

//        indicesP_local = (unsigned long *) malloc(sizeof(unsigned long) * nnz_l_local);
        indicesP_local.resize(nnz_l_local);
        for (nnz_t i = 0; i < nnz_l_local; i++)
            indicesP_local[i] = i;

        index_t *row_localP = &*row_local.begin();
        std::sort(&indicesP_local[0], &indicesP_local[nnz_l_local], sort_indices(row_localP));

//        indicesP_remote = (unsigned long *) malloc(sizeof(unsigned long) * nnz_l_remote);
        indicesP_remote.resize(nnz_l_remote);
        for (nnz_t i = 0; i < nnz_l_remote; i++)
            indicesP_remote[i] = i;

        index_t *row_remoteP = &*row_remote.begin();
        std::sort(&indicesP_remote[0], &indicesP_remote[nnz_l_remote], sort_indices(row_remoteP));

        // *************************** find start and end of each thread for matvec ****************************
        // also, find nnz per row for local and remote matvec

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

//        iter_local_array = (unsigned int *) malloc(sizeof(unsigned int) * (num_threads + 1));
//        iter_remote_array = (unsigned int *) malloc(sizeof(unsigned int) * (num_threads + 1));
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

//        if(rank==0){
//            std::cout << "iter_local_array:" << std::endl;
//            for(int i=0; i<num_threads+1; i++)
//                std::cout << iter_local_array[i] << std::endl;}

//        if (rank==0){
//            std::cout << "iter_remote_array:" << std::endl;
//            for(int i=0; i<num_threads+1; i++)
//                std::cout << iter_remote_array[i] << std::endl;}

        // setup variables for another matvec implementation
        // -------------------------------------------------
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

//        if(rank==0){
//            printf("\niter_local_array and iter_local_array2: \n");
//            for(int i = 0; i < num_threads+1; i++)
//                printf("%u \t%u \n", iter_local_array[i], iter_local_array2[i]);}

        // *************************** allocate for w_buff for matvec3() ****************************

        w_buff.resize(num_threads*M);

/*
        nnz_t total_nnz_l_local;
        nnz_t total_nnz_l_remote;
        MPI_Allreduce(&nnz_l_local,  &total_nnz_l_local,  1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
        MPI_Allreduce(&nnz_l_remote, &total_nnz_l_remote, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
        int local_percent  = int(100*(float)total_nnz_l_local/nnz_g);
        int remote_percent = int(100*(float)total_nnz_l_remote/nnz_g);
        if(rank==0) printf("\nMbig = %u, nnz_g = %lu, total_nnz_l_local = %lu (%%%d), total_nnz_l_remote = %lu (%%%d) \n",
                           Mbig, nnz_g, total_nnz_l_local, local_percent, total_nnz_l_remote, remote_percent);

//        printf("rank %d: col_remote_size = %u \n", rank, col_remote_size);
        index_t col_remote_size_min, col_remote_size_ave, col_remote_size_max;
        MPI_Allreduce(&col_remote_size, &col_remote_size_min, 1, MPI_UNSIGNED, MPI_MIN, comm);
        MPI_Allreduce(&col_remote_size, &col_remote_size_ave, 1, MPI_UNSIGNED, MPI_SUM, comm);
        MPI_Allreduce(&col_remote_size, &col_remote_size_max, 1, MPI_UNSIGNED, MPI_MAX, comm);
        if(rank==0) printf("\nremote_min = %u, remote_ave = %u, remote_max = %u \n",
                           col_remote_size_min, (col_remote_size_ave/nprocs), col_remote_size_max);
*/

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, done \n", rank);
            MPI_Barrier(comm);
        }

    } // end of if(active)
    return 0;
}


int saena_matrix::matrix_setup2() {
// update values_local, values_remote and invDiag.

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

//    assembled = true;

    values_local.clear();
    values_remote.clear();

    if(!entry.empty()){
        if (entry[0].col >= split[rank] && entry[0].col < split[rank + 1]) {
            values_local.push_back(entry[0].val);
//            row_local.push_back(entry[0].row);
//            col_local.push_back(entry[0].col);
        } else {
            values_remote.push_back(entry[0].val);
//            row_remote.push_back(entry[0].row);
//            col_remote.push_back(col_remote_size - 1);
//            col_remote2.push_back(entry[0].col);
        }
    }

    if(entry.size() >= 2){
        for (nnz_t i = 1; i < nnz_l; i++) {
            if (entry[i].col >= split[rank] && entry[i].col < split[rank + 1]) {
                values_local.push_back(entry[i].val);
//                row_local.push_back(entry[i].row);
//                col_local.push_back(entry[i].col);
            } else {
                values_remote.push_back(entry[i].val);
//                row_remote.push_back(entry[i].row);
                // col_remote2 is the original col value and will be used in making strength matrix. col_remote will be used for matevec.
//                col_remote2.push_back(entry[i].col);
                // the original col values are not being used. the ordering starts from 0, and goes up by 1.
//                col_remote.push_back(col_remote_size - 1);
            }
        }
    }

    invDiag.resize(M);
    inverse_diag(invDiag);

    return 0;
}


int saena_matrix::matvec(std::vector<value_t>& v, std::vector<value_t>& w) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ){
//        printf("A.M != v.size() in matvec!!!\n");}

    // the indices of the v on this proc that should be sent to other procs are saved in vIndex.
    // put the values of thoss indices in vSend to send to other procs.
#pragma omp parallel for
    for(index_t i=0;i<vIndexSize;i++)
        vSend[i] = v[(vIndex[i])];

//    printf("hereeeeee\n");

//    if (rank==1) std::cout << "\nvIndexSize=" << vIndexSize << std::endl;
//    print_vector(vSend, 0, "vSend", comm);

    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses  = new MPI_Status[numSendProc+numRecvProc];

    // receive and put the remote parts of v in vecValues.
    // they are received in order: first put the values from the lowest rank matrix, and so on.
    for(int i = 0; i < numRecvProc; i++)
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));

    for(int i = 0; i < numSendProc; i++)
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));

    // local loop
    // ----------
    // compute the on-diagonal part of matvec on each thread and save it in w_local.
    // then, do a reduction on w_local on all threads, based on a binary tree.

    value_t* v_p = &v[0] - split[rank];
    // local loop
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        long iter = iter_local_array2[thread_id];
#pragma omp for
        for (unsigned int i = 0; i < M; ++i) {
            w[i] = 0;
            for (unsigned int j = 0; j < nnzPerRow_local2[i]; ++j, ++iter) {
                w[i] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
            }
        }

        for (iter = iter_local_array[thread_id]; iter < iter_local_array2[thread_id]; ++iter)
            w[row_local[indicesP_local[iter]]] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
    }

    // Wait for the receive communication to finish.
    MPI_Waitall(numRecvProc, requests, statuses);

//    if (rank==0){
//        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
//        for(int i=0; i<recvSize; i++)
//            std::cout << vecValues[i] << std::endl;}

    // remote loop
    // -----------
    // the col_index of the matrix entry does not matter. do the matvec on the first non-zero column (j=0).
    // the corresponding vector element is saved in vecValues[0]. and so on.

#pragma omp parallel
    {
        unsigned int i, l;
        int thread_id = omp_get_thread_num();
        value_t *w_local = &w_buff[0] + (thread_id*M);
        if(thread_id==0)
            w_local = &*w.begin();
        else
            std::fill(&w_local[0], &w_local[M], 0);

        nnz_t iter = iter_remote_array[thread_id];
#pragma omp for
        for (index_t j = 0; j < col_remote_size; ++j) {
            for (i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
                w_local[row_remote[iter]] += values_remote[iter] * vecValues[j];

//                if(rank==0 && thread_id==0){
//                    printf("thread = %d\n", thread_id);
//                    printf("%u \t%u \tind_rem = %lu, row = %lu \tcol = %lu \tvecVal = %f \n",
//                           i, j, indicesP_remote[iter], row_remote[indicesP_remote[iter]],
//                           col_remote[indicesP_remote[iter]], vecValues[col_remote[indicesP_remote[iter]]]);}
            }
        }

        int thread_partner;
        int levels = (int)ceil(log2(num_threads));
        for (l = 0; l < levels; l++) {
            if (thread_id % int(pow(2, l+1)) == 0) {
                thread_partner = thread_id + int(pow(2, l));
//                printf("l = %d, levels = %d, thread_id = %d, thread_partner = %d \n", l, levels, thread_id, thread_partner);
                if(thread_partner < num_threads){
                    for (i = 0; i < M; i++)
                        w_local[i] += w_buff[thread_partner * M + i];
                }
            }
#pragma omp barrier
        }
    }

    // todo: remove indicesP_remote. it is not required in the remote matvec anymore.

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    delete [] requests;
    delete [] statuses;
    return 0;
}


int saena_matrix::matvec_timing1(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time) {
/*
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ){
//        printf("A.M != v.size() in matvec!!!\n");}

//    MPI_Barrier(comm);
    double t0_start = omp_get_wtime();
    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i=0;i<vIndexSize;i++)
        vSend[i] = v[( vIndex[i] )];
    double t0_end = omp_get_wtime();// try this: rdtsc for timing

//    if (rank==0){
//        std::cout << "vIndexSize=" << vIndexSize << ", vSend: rank=" << rank << std::endl;
//        for(int i=0; i<vIndexSize; i++)
//            std::cout << vSend[i] << std::endl;}

//    MPI_Barrier(comm);
    double t3_start = omp_get_wtime();
    // iSend your data, and iRecv from others
    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses = new MPI_Status[numSendProc+numRecvProc];

    for(int i = 0; i < numRecvProc; i++)
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));

    for(int i = 0; i < numSendProc; i++)
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));

//    if (rank==0){
//        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
//        for(int i=0; i<recvSize; i++)
//            std::cout << vecValues[i] << std::endl;}

    // local loop
    // ----------
    double t1_start = omp_get_wtime();

    w.assign(w.size(), 0);
    value_t* v_p = &v[0] - split[rank];
    unsigned int i;
#pragma omp parallel for default(shared) private(i) reduction(vec_double_plus:w)
    for (i = 0; i < nnz_l_local; ++i)
        w[row_local[i]] += values_local[i] * v_p[col_local[i]];

    double t1_end = omp_get_wtime();

    MPI_Waitall(numRecvProc, requests, statuses);

//    if (rank==1){
//        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
//        for(int i=0; i<recvSize; i++)
//            std::cout << vecValues[i] << std::endl;}

    // remote loop
    double t2_start = omp_get_wtime();

#pragma omp parallel
    {
        unsigned int i, l;
        int thread_id = omp_get_thread_num();
        value_t *w_local = w_buff + (thread_id*M);
        if(thread_id==0)
            w_local = &*w.begin();
        else
            std::fill(&w_local[0], &w_local[M], 0)
        nnz_t iter = iter_remote_array[thread_id];
#pragma omp for
        for (index_t j = 0; j < col_remote_size; ++j) {
            for (i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
                w_local[row_remote[iter]] += values_remote[iter] * vecValues[j];

//                if(rank==0 && thread_id==0){
//                    printf("thread = %d\n", thread_id);
//                    printf("%u \t%u \tind_rem = %lu, row = %lu \tcol = %lu \tvecVal = %f \n",
//                           i, j, indicesP_remote[iter], row_remote[indicesP_remote[iter]],
//                           col_remote[indicesP_remote[iter]], vecValues[col_remote[indicesP_remote[iter]]]);}
            }
        }

        int thread_partner;
        int levels = (int)ceil(log2(num_threads));
        for (l = 0; l < levels; l++) {
            if (thread_id % int(pow(2, l+1)) == 0) {
                thread_partner = thread_id + int(pow(2, l));
//                printf("l = %d, levels = %d, thread_id = %d, thread_partner = %d \n", l, levels, thread_id, thread_partner);
                if(thread_partner < num_threads){
                    for (i = 0; i < M; i++)
                        w_local[i] += w_buff[thread_partner * M + i];
                }
            }
#pragma omp barrier
        }
    }

    double t2_end = omp_get_wtime();

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    double t3_end = omp_get_wtime();

    delete [] requests;
    delete [] statuses;

    // set vsend
    double time0_local = t0_end-t0_start;
    double time0;
    MPI_Allreduce(&time0_local, &time0, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[0] += time0/nprocs;

    // local loop
    double time1_local = t1_end-t1_start;
    double time1;
    MPI_Allreduce(&time1_local, &time1, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[1] += time1/nprocs;

    // remote loop
    double time2_local = t2_end-t2_start;
    double time2;
    MPI_Allreduce(&time2_local, &time2, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[2] += time2/nprocs;

    // communication = t3 - t1 - t2
    double time3_local = t3_end-t3_start;
    double time3;
    MPI_Allreduce(&time3_local, &time3, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[3] += time3/nprocs;

//    time[0] += time0_local;
//    time[1] += time1_local;
//    time[2] += time2_local;
//    time[3] += time3_local;
//    time[4] += time4_local;
*/
    return 0;
}


int saena_matrix::matvec_timing2(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ){
//        printf("A.M != v.size() in matvec!!!\n");}

//    MPI_Barrier(comm);
    double t0_start = omp_get_wtime();
    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i=0;i<vIndexSize;i++)
        vSend[i] = v[( vIndex[i] )];
    double t0_end = omp_get_wtime();// try this: rdtsc for timing

//    if (rank==0){
//        std::cout << "vIndexSize=" << vIndexSize << ", vSend: rank=" << rank << std::endl;
//        for(int i=0; i<vIndexSize; i++)
//            std::cout << vSend[i] << std::endl;}

//    MPI_Barrier(comm);
    double t3_start = omp_get_wtime();
    // iSend your data, and iRecv from others
    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses = new MPI_Status[numSendProc+numRecvProc];

    for(int i = 0; i < numRecvProc; i++)
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));

    for(int i = 0; i < numSendProc; i++)
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));

//    if (rank==0){
//        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
//        for(int i=0; i<recvSize; i++)
//            std::cout << vecValues[i] << std::endl;}

    // local loop
    // ----------
    double t1_start = omp_get_wtime();

    value_t* v_p = &v[0] - split[rank];
    // local loop
//    std::fill(&*w.begin(), &*w.end(), 0);
#pragma omp parallel
    {
        nnz_t iter = iter_local_array[omp_get_thread_num()];
#pragma omp for
        for (index_t i = 0; i < M; ++i) {
            w[i] = 0;
            for (index_t j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
                w[i] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
            }
        }
    }

    double t1_end = omp_get_wtime();

    MPI_Waitall(numRecvProc, requests, statuses);

//    if (rank==1){
//        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
//        for(int i=0; i<recvSize; i++)
//            std::cout << vecValues[i] << std::endl;}

    // remote loop
    double t2_start = omp_get_wtime();

#pragma omp parallel
    {
        unsigned int i, l;
        int thread_id = omp_get_thread_num();
        value_t *w_local = &w_buff[0] + (thread_id*M);
        if(thread_id==0)
            w_local = &*w.begin();
        else
            std::fill(&w_local[0], &w_local[M], 0);

        nnz_t iter = iter_remote_array[thread_id];
#pragma omp for
        for (index_t j = 0; j < col_remote_size; ++j) {
            for (i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
                w_local[row_remote[iter]] += values_remote[iter] * vecValues[j];

//                if(rank==0 && thread_id==0){
//                    printf("thread = %d\n", thread_id);
//                    printf("%u \t%u \tind_rem = %lu, row = %lu \tcol = %lu \tvecVal = %f \n",
//                           i, j, indicesP_remote[iter], row_remote[indicesP_remote[iter]],
//                           col_remote[indicesP_remote[iter]], vecValues[col_remote[indicesP_remote[iter]]]);}
            }
        }

        int thread_partner;
        int levels = (int)ceil(log2(num_threads));
        for (l = 0; l < levels; l++) {
            if (thread_id % int(pow(2, l+1)) == 0) {
                thread_partner = thread_id + int(pow(2, l));
//                printf("l = %d, levels = %d, thread_id = %d, thread_partner = %d \n", l, levels, thread_id, thread_partner);
                if(thread_partner < num_threads){
                    for (i = 0; i < M; i++)
                        w_local[i] += w_buff[thread_partner * M + i];
                }
            }
#pragma omp barrier
        }
    }

    double t2_end = omp_get_wtime();

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    double t3_end = omp_get_wtime();

    delete [] requests;
    delete [] statuses;

    // set vsend
    double time0_local = t0_end-t0_start;
    double time0;
    MPI_Allreduce(&time0_local, &time0, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[0] += time0/nprocs;

    // local loop
    double time1_local = t1_end-t1_start;
    double time1;
    MPI_Allreduce(&time1_local, &time1, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[1] += time1/nprocs;

    // remote loop
    double time2_local = t2_end-t2_start;
    double time2;
    MPI_Allreduce(&time2_local, &time2, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[2] += time2/nprocs;

    // communication = t3 - t1 - t2
    double time3_local = t3_end-t3_start;
    double time3;
    MPI_Allreduce(&time3_local, &time3, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[3] += time3/nprocs;

//    time[0] += time0_local;
//    time[1] += time1_local;
//    time[2] += time2_local;
//    time[3] += time3_local;
//    time[4] += time4_local;

    return 0;
}


int saena_matrix::matvec_timing3(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ){
//        printf("A.M != v.size() in matvec!!!\n");}

//    MPI_Barrier(comm);
    double t0_start = omp_get_wtime();
    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i=0;i<vIndexSize;i++)
        vSend[i] = v[( vIndex[i] )];
    double t0_end = omp_get_wtime();// try this: rdtsc for timing

//    if (rank==0){
//        std::cout << "vIndexSize=" << vIndexSize << ", vSend: rank=" << rank << std::endl;
//        for(int i=0; i<vIndexSize; i++)
//            std::cout << vSend[i] << std::endl;}

//    MPI_Barrier(comm);
    double t3_start = omp_get_wtime();
    // iSend your data, and iRecv from others
    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses = new MPI_Status[numSendProc+numRecvProc];

    for(int i = 0; i < numRecvProc; i++)
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));

    for(int i = 0; i < numSendProc; i++)
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));

//    if (rank==0){
//        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
//        for(int i=0; i<recvSize; i++)
//            std::cout << vecValues[i] << std::endl;}

    // local loop
    // ----------
    double t1_start = omp_get_wtime();

    value_t* v_p = &v[0] - split[rank];
    // local loop
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        long iter = iter_local_array2[thread_id];
#pragma omp for
        for (unsigned int i = 0; i < M; ++i) {
            w[i] = 0;
            for (unsigned int j = 0; j < nnzPerRow_local2[i]; ++j, ++iter) {
                w[i] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
            }
        }

        for (iter = iter_local_array[thread_id]; iter < iter_local_array2[thread_id]; ++iter)
            w[row_local[indicesP_local[iter]]] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
    }

    double t1_end = omp_get_wtime();

    MPI_Waitall(numRecvProc, requests, statuses);

//    if (rank==1){
//        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
//        for(int i=0; i<recvSize; i++)
//            std::cout << vecValues[i] << std::endl;}

    // remote loop
    double t2_start = omp_get_wtime();

#pragma omp parallel
    {
        unsigned int i, l;
        int thread_id = omp_get_thread_num();
        value_t *w_local = &w_buff[0] + (thread_id*M);
        if(thread_id==0)
            w_local = &*w.begin();
        else
            std::fill(&w_local[0], &w_local[M], 0);

        nnz_t iter = iter_remote_array[thread_id];
#pragma omp for
        for (index_t j = 0; j < col_remote_size; ++j) {
            for (i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
                w_local[row_remote[iter]] += values_remote[iter] * vecValues[j];

//                if(rank==0 && thread_id==0){
//                    printf("thread = %d\n", thread_id);
//                    printf("%u \t%u \tind_rem = %lu, row = %lu \tcol = %lu \tvecVal = %f \n",
//                           i, j, indicesP_remote[iter], row_remote[indicesP_remote[iter]],
//                           col_remote[indicesP_remote[iter]], vecValues[col_remote[indicesP_remote[iter]]]);}
            }
        }

        int thread_partner;
        int levels = (int)ceil(log2(num_threads));
        for (l = 0; l < levels; l++) {
            if (thread_id % int(pow(2, l+1)) == 0) {
                thread_partner = thread_id + int(pow(2, l));
//                printf("l = %d, levels = %d, thread_id = %d, thread_partner = %d \n", l, levels, thread_id, thread_partner);
                if(thread_partner < num_threads){
                    for (i = 0; i < M; i++)
                        w_local[i] += w_buff[thread_partner * M + i];
                }
            }
#pragma omp barrier
        }
    }

    double t2_end = omp_get_wtime();

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    double t3_end = omp_get_wtime();

    delete [] requests;
    delete [] statuses;

    // set vsend
    double time0_local = t0_end-t0_start;
    double time0;
    MPI_Allreduce(&time0_local, &time0, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[0] += time0/nprocs;

    // local loop
    double time1_local = t1_end-t1_start;
    double time1;
    MPI_Allreduce(&time1_local, &time1, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[1] += time1/nprocs;

    // remote loop
    double time2_local = t2_end-t2_start;
    double time2;
    MPI_Allreduce(&time2_local, &time2, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[2] += time2/nprocs;

    // communication = t3 - t1 - t2
    double time3_local = t3_end-t3_start;
    double time3;
    MPI_Allreduce(&time3_local, &time3, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[3] += time3/nprocs;

//    time[0] += time0_local;
//    time[1] += time1_local;
//    time[2] += time2_local;
//    time[3] += time3_local;
//    time[4] += time4_local;

    return 0;
}


int saena_matrix::matvec_timing4(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time) {

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ){
//        printf("A.M != v.size() in matvec!!!\n");}

//    MPI_Barrier(comm);
    double t0_start = omp_get_wtime();
    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i=0;i<vIndexSize;i++)
        vSend[i] = v[( vIndex[i] )];
    double t0_end = omp_get_wtime();// try this: rdtsc for timing

//    if (rank==0){
//        std::cout << "vIndexSize=" << vIndexSize << ", vSend: rank=" << rank << std::endl;
//        for(int i=0; i<vIndexSize; i++)
//            std::cout << vSend[i] << std::endl;}

//    MPI_Barrier(comm);
    double t3_start = omp_get_wtime();
    // iSend your data, and iRecv from others
    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses = new MPI_Status[numSendProc+numRecvProc];

    for(int i = 0; i < numRecvProc; i++)
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));

    for(int i = 0; i < numSendProc; i++)
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));

//    if (rank==0){
//        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
//        for(int i=0; i<recvSize; i++)
//            std::cout << vecValues[i] << std::endl;}

    // local loop
    // ----------
    double t1_start = omp_get_wtime();

    // by doing this you will have a local index for v[col_local[i]].
    value_t* v_p = &v[0] - split[rank];
#pragma omp parallel
    {
        unsigned int i, l, idx;
        int thread_id = omp_get_thread_num();
        value_t *w_local = &w_buff[0] + (thread_id*M);
        if(thread_id==0)
            w_local = &*w.begin();

        std::fill(&w_local[0], &w_local[M], 0);

#pragma omp for
        for (i = 0; i < nnz_l_local; ++i)
            w_local[row_local[i]] += values_local[i] * v_p[col_local[i]];

        int thread_partner;
        int levels = (int)ceil(log2(num_threads));
        for (l = 0; l < levels; l++) {
            if (thread_id % int(pow(2, l+1)) == 0) {
                thread_partner = thread_id + int(pow(2, l));
//                printf("l = %d, levels = %d, thread_id = %d, thread_partner = %d \n", l, levels, thread_id, thread_partner);
                if(thread_partner < num_threads){
                    for (i = 0; i < M; i++)
                        w_local[i] += w_buff[thread_partner * M + i];
                }
            }
#pragma omp barrier
        }
    }

    double t1_end = omp_get_wtime();

    MPI_Waitall(numRecvProc, requests, statuses);

//    if (rank==1){
//        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
//        for(int i=0; i<recvSize; i++)
//            std::cout << vecValues[i] << std::endl;}

    // remote loop
    double t2_start = omp_get_wtime();

    // this version is not parallel with OpenMP.
    /*
    nnz_t iter = 0;
    for (index_t j = 0; j < col_remote_size; ++j) {
        for (index_t i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
            w[row_remote[iter]] += values_remote[iter] * vecValues[j];

//                if(rank==0 && omp_get_thread_num()==0){
//                    printf("thread = %d\n", omp_get_thread_num());
//                    printf("%u \t%u \tind_rem = %lu, row = %lu \tcol = %lu \tvecVal = %f \n",
//                           i, j, indicesP_remote[iter], row_remote[indicesP_remote[iter]],
//                           col_remote[indicesP_remote[iter]], vecValues[col_remote[indicesP_remote[iter]]]);}
        }
    }
*/
    // the previous part is parallelized with OpenMP.
#pragma omp parallel
    {
        unsigned int i, l;
        int thread_id = omp_get_thread_num();
        value_t *w_local = &w_buff[0] + (thread_id*M);
        if(thread_id==0)
            w_local = &*w.begin();
        else
            std::fill(&w_local[0], &w_local[M], 0);

        nnz_t iter = iter_remote_array[thread_id];
#pragma omp for
        for (index_t j = 0; j < col_remote_size; ++j) {
            for (i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
                w_local[row_remote[iter]] += values_remote[iter] * vecValues[j];

//                if(rank==0 && thread_id==0){
//                    printf("thread = %d\n", thread_id);
//                    printf("%u \t%u \tind_rem = %lu, row = %lu \tcol = %lu \tvecVal = %f \n",
//                           i, j, indicesP_remote[iter], row_remote[indicesP_remote[iter]],
//                           col_remote[indicesP_remote[iter]], vecValues[col_remote[indicesP_remote[iter]]]);}
            }
        }

        int thread_partner;
        int levels = (int)ceil(log2(num_threads));
        for (l = 0; l < levels; l++) {
            if (thread_id % int(pow(2, l+1)) == 0) {
                thread_partner = thread_id + int(pow(2, l));
//                printf("l = %d, levels = %d, thread_id = %d, thread_partner = %d \n", l, levels, thread_id, thread_partner);
                if(thread_partner < num_threads){
                    for (i = 0; i < M; i++)
                        w_local[i] += w_buff[thread_partner * M + i];
                }
            }
#pragma omp barrier
        }
    }

    double t2_end = omp_get_wtime();

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    double t3_end = omp_get_wtime();

    delete [] requests;
    delete [] statuses;

    // set vsend
    double time0_local = t0_end-t0_start;
    double time0;
    MPI_Allreduce(&time0_local, &time0, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[0] += time0/nprocs;

    // local loop
    double time1_local = t1_end-t1_start;
    double time1;
    MPI_Allreduce(&time1_local, &time1, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[1] += time1/nprocs;

    // remote loop
    double time2_local = t2_end-t2_start;
    double time2;
    MPI_Allreduce(&time2_local, &time2, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[2] += time2/nprocs;

    // communication = t3 - t1 - t2
    double time3_local = t3_end-t3_start;
    double time3;
    MPI_Allreduce(&time3_local, &time3, 1, MPI_DOUBLE, MPI_SUM, comm);
    time[3] += time3/nprocs;

//    time[0] += time0_local;
//    time[1] += time1_local;
//    time[2] += time2_local;
//    time[3] += time3_local;
//    time[4] += time4_local;

    return 0;
}


int saena_matrix::matvec_timing4_alltoall(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time) {
// todo: to reduce the communication during matvec, consider reducing number of columns during coarsening,
// todo: instead of reducing general non-zeros, since that is what is communicated for matvec.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ){
//        printf("A.M != v.size() in matvec!!!\n");}

    double t0_start = omp_get_wtime();
    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i = 0; i < vIndexSize; i++)
        vSend[i] = v[( vIndex[i] )];
    double t0_end = omp_get_wtime();// try this: rdtsc for timing

/*    if (rank==0){
        std::cout << "vIndexSize=" << vIndexSize << ", vSend: rank=" << rank << std::endl;
        for(int i=0; i<vIndexSize; i++)
            std::cout << vSend[i] << std::endl;
    }*/

//    double t3_start = omp_get_wtime();

    double t4_start = omp_get_wtime();
    MPI_Alltoallv(&vSend[0], &sendCount[0], &sendCountScan[0], MPI_INT, &vecValues[0], &recvCount[0], &recvCountScan[0], MPI_INT, comm);
    double t4_end = omp_get_wtime();

/*    if (rank==0){
        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
        for(int i=0; i<recvSize; i++)
            std::cout << vecValues[i] << std::endl;
    }*/

    double t1_start = omp_get_wtime();

    // local loop
    value_t* v_p = &v[0] - split[rank];
#pragma omp parallel
    {
        unsigned int i, l, idx;
        int thread_id = omp_get_thread_num();
        value_t *w_local = &w_buff[0] + (thread_id*M);
        if(thread_id==0)
            w_local = &*w.begin();

        std::fill(&w_local[0], &w_local[M], 0);

#pragma omp for
        for (i = 0; i < nnz_l_local; ++i)
            w_local[row_local[i]] += values_local[i] * v_p[col_local[i]];

        int thread_partner;
        int levels = (int)ceil(log2(num_threads));
        for (l = 0; l < levels; l++) {
            if (thread_id % int(pow(2, l+1)) == 0) {
                thread_partner = thread_id + int(pow(2, l));
//                printf("l = %d, levels = %d, thread_id = %d, thread_partner = %d \n", l, levels, thread_id, thread_partner);
                if(thread_partner < num_threads){
                    for (i = 0; i < M; i++)
                        w_local[i] += w_buff[thread_partner * M + i];
                }
            }
#pragma omp barrier
        }
    }

    double t1_end = omp_get_wtime();

    // Wait for the communication to finish.
//    double t4_start = omp_get_wtime();
//    MPI_Waitall(numSendProc+numRecvProc, requests, statuses);
//    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
//    double t4_end = omp_get_wtime();

/*    if (rank==1){
        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
        for(int i=0; i<recvSize; i++)
            std::cout << vecValues[i] << std::endl;
    }*/

    // remote loop
    // todo: data race happens during "omp for" here, since the "for" loop splits based on the remote columns, but
    // todo: w[row] are being computed in every iteration , which means different threads may access the same w[row].

    double t2_start = omp_get_wtime();

    nnz_t iter = iter_remote_array[omp_get_thread_num()];
//#pragma omp for
    for (index_t j = 0; j < col_remote_size; ++j) {
        for (index_t i = 0; i < nnzPerCol_remote[j]; ++i, ++iter) {
            w[row_remote[iter]] += values_remote[iter] * vecValues[j];

//                if(rank==0 && omp_get_thread_num()==0){
//                    printf("thread = %d\n", omp_get_thread_num());
//                    printf("%u \t%u \tind_rem = %lu, row = %lu \tcol = %lu \tvecVal = %f \n",
//                           i, j, indicesP_remote[iter], row_remote[indicesP_remote[iter]],
//                           col_remote[indicesP_remote[iter]], vecValues[col_remote[indicesP_remote[iter]]]);}
        }
    }

    double t2_end = omp_get_wtime();
//    double t3_end = omp_get_wtime();

    double time0_local = t0_end-t0_start;
    double time0;
    MPI_Allreduce(&time0_local, &time0, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time1_local = t1_end-t1_start;
    double time1;
    MPI_Allreduce(&time1_local, &time1, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time2_local = t2_end-t2_start;
    double time2;
    MPI_Allreduce(&time2_local, &time2, 1, MPI_DOUBLE, MPI_SUM, comm);

//    double time3_local = t3_end-t3_start;
//    double time3;
//    MPI_Allreduce(&time3_local, &time3, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time4_local = t4_end-t4_start;
    double time4;
    MPI_Allreduce(&time4_local, &time4, 1, MPI_DOUBLE, MPI_SUM, comm);

    time[0] += time0/nprocs;
    time[1] += time1/nprocs;
    time[2] += time2/nprocs;
//    time[3] += time3/nprocs;
    time[4] += time4/nprocs;

//    time[0] += time0_local;
//    time[1] += time1_local;
//    time[2] += time2_local;
//    time[3] += time3_local;
//    time[4] += time4_local;

    return 0;
}


int saena_matrix::matvec_timing5(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time) {
    // old remote loop is used here.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ){
//        printf("A.M != v.size() in matvec!!!\n");}

    double t0_start = omp_get_wtime();
    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i=0;i<vIndexSize;i++)
        vSend[i] = v[( vIndex[i] )];
    double t0_end = omp_get_wtime();// try this: rdtsc for timing

/*    if (rank==0){
        std::cout << "vIndexSize=" << vIndexSize << ", vSend: rank=" << rank << std::endl;
        for(int i=0; i<vIndexSize; i++)
            std::cout << vSend[i] << std::endl;
    }*/

    double t3_start = omp_get_wtime();
    // iSend your data, and iRecv from others
    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses = new MPI_Status[numSendProc+numRecvProc];

    for(int i = 0; i < numRecvProc; i++)
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));

    for(int i = 0; i < numSendProc; i++)
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));

/*    if (rank==0){
        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
        for(int i=0; i<recvSize; i++)
            std::cout << vecValues[i] << std::endl;
    }*/

    // local loop
    // ----------
    double t1_start = omp_get_wtime();
    // by doing this you will have a local index for v[col_local[i]].
    value_t* v_p = &v[0] - split[rank];
#pragma omp parallel
    {
        unsigned int i, l, idx;
        int thread_id = omp_get_thread_num();
        value_t *w_local = &w_buff[0] + (thread_id*M);
        if(thread_id==0)
            w_local = &*w.begin();

        std::fill(&w_local[0], &w_local[M], 0);

#pragma omp for
        for (i = 0; i < nnz_l_local; ++i)
            w_local[row_local[i]] += values_local[i] * v_p[col_local[i]];

        int thread_partner;
        int levels = (int)ceil(log2(num_threads));
        for (l = 0; l < levels; l++) {
            if (thread_id % int(pow(2, l+1)) == 0) {
                thread_partner = thread_id + int(pow(2, l));
//                printf("l = %d, levels = %d, thread_id = %d, thread_partner = %d \n", l, levels, thread_id, thread_partner);
                if(thread_partner < num_threads){
                    for (i = 0; i < M; i++)
                        w_local[i] += w_buff[thread_partner * M + i];
                }
            }
#pragma omp barrier
        }
    }

    double t1_end = omp_get_wtime();

    // Wait for the communication to finish.
    double t4_start = omp_get_wtime();
//    MPI_Waitall(numSendProc+numRecvProc, requests, statuses);
    MPI_Waitall(numRecvProc, requests, statuses);
    double t4_end = omp_get_wtime();

/*    if (rank==1){
        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
        for(int i=0; i<recvSize; i++)
            std::cout << vecValues[i] << std::endl;
    }*/

    // remote loop
    // todo: data race happens during "omp for" here, since the "for" loop splits based on the remote columns, but
    // todo: w[row] are being computed in every iteration , which means different threads may access the same w[row].

    double t2_start = omp_get_wtime();
//#pragma omp parallel
//    {
    unsigned int iter = iter_remote_array[omp_get_thread_num()];
//#pragma omp for
    for (unsigned int i = 0; i < col_remote_size; ++i) {
        for (unsigned int j = 0; j < nnzPerCol_remote[i]; ++j, ++iter) {
            w[row_remote[indicesP_remote[iter]]] += values_remote[indicesP_remote[iter]] * vecValues[col_remote[indicesP_remote[iter]]];

//                if(rank==0 && omp_get_thread_num()==0){
//                    printf("thread = %d\n", omp_get_thread_num());
//                    printf("%u \t%u \tind_rem = %lu, row = %lu \tcol = %lu \tvecVal = %f \n",
//                           i, j, indicesP_remote[iter], row_remote[indicesP_remote[iter]],
//                           col_remote[indicesP_remote[iter]], vecValues[col_remote[indicesP_remote[iter]]]);}
        }
    }
//    }

    double t2_end = omp_get_wtime();

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    double t3_end = omp_get_wtime();

    delete [] requests;
    delete [] statuses;

    double time0_local = t0_end-t0_start;
    double time0;
    MPI_Allreduce(&time0_local, &time0, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time1_local = t1_end-t1_start;
    double time1;
    MPI_Allreduce(&time1_local, &time1, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time2_local = t2_end-t2_start;
    double time2;
    MPI_Allreduce(&time2_local, &time2, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time3_local = t3_end-t3_start;
    double time3;
    MPI_Allreduce(&time3_local, &time3, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time4_local = t4_end-t4_start;
    double time4;
    MPI_Allreduce(&time4_local, &time4, 1, MPI_DOUBLE, MPI_SUM, comm);

    time[0] += time0/nprocs;
    time[1] += time1/nprocs;
    time[2] += time2/nprocs;
    time[3] += time3/nprocs;
    time[4] += time4/nprocs;

//    time[0] += time0_local;
//    time[1] += time1_local;
//    time[2] += time2_local;
//    time[3] += time3_local;
//    time[4] += time4_local;

    return 0;
}


int saena_matrix::matvec_timing5_alltoall(std::vector<value_t>& v, std::vector<value_t>& w, std::vector<double>& time) {
// todo: to reduce the communication during matvec, consider reducing number of columns during coarsening,
// todo: instead of reducing general non-zeros, since that is what is communicated for matvec.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if( v.size() != M ){
        printf("A.M != v.size() in matvec!!!\n");}

    double t0_start = omp_get_wtime();
    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i = 0; i < vIndexSize; i++)
        vSend[i] = v[( vIndex[i] )];
    double t0_end = omp_get_wtime();// try this: rdtsc for timing

/*    if (rank==0){
        std::cout << "vIndexSize=" << vIndexSize << ", vSend: rank=" << rank << std::endl;
        for(int i=0; i<vIndexSize; i++)
            std::cout << vSend[i] << std::endl;
    }*/

//    double t3_start = omp_get_wtime();

    double t4_start = omp_get_wtime();
    MPI_Alltoallv(&vSend[0], &sendCount[0], &sendCountScan[0], MPI_INT, &vecValues[0], &recvCount[0], &recvCountScan[0], MPI_INT, comm);
    double t4_end = omp_get_wtime();

/*    if (rank==0){
        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
        for(int i=0; i<recvSize; i++)
            std::cout << vecValues[i] << std::endl;
    }*/

    double t1_start = omp_get_wtime();
    value_t* v_p = &v[0] - split[rank];
    // local loop
//    std::fill(&*w.begin(), &*w.end(), 0);
#pragma omp parallel
    {
        long iter = iter_local_array[omp_get_thread_num()];
#pragma omp for
        for (unsigned int i = 0; i < M; ++i) {
            w[i] = 0;
            for (unsigned int j = 0; j < nnzPerRow_local[i]; ++j, ++iter) {
                w[i] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
            }
        }
    }

    double t1_end = omp_get_wtime();

    // Wait for the communication to finish.
//    double t4_start = omp_get_wtime();
//    MPI_Waitall(numSendProc+numRecvProc, requests, statuses);
//    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
//    double t4_end = omp_get_wtime();

/*    if (rank==1){
        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
        for(int i=0; i<recvSize; i++)
            std::cout << vecValues[i] << std::endl;
    }*/

    // remote loop
    // todo: data race happens during "omp for" here, since the "for" loop splits based on the remote columns, but
    // todo: w[row] are being computed in every iteration , which means different threads may access the same w[row].

    double t2_start = omp_get_wtime();
//#pragma omp parallel
//    {
    unsigned int iter = iter_remote_array[omp_get_thread_num()];
//#pragma omp for
    for (unsigned int i = 0; i < col_remote_size; ++i) {
        for (unsigned int j = 0; j < nnzPerCol_remote[i]; ++j, ++iter) {

//                if(rank==0 && omp_get_thread_num()==0){
//                    printf("thread = %d\n", omp_get_thread_num());
//                    printf("%u \t%u \tind_rem = %lu, row = %lu \tcol = %lu \tvecVal = %f \n",
//                           i, j, indicesP_remote[iter], row_remote[indicesP_remote[iter]],
//                           col_remote[indicesP_remote[iter]], vecValues[col_remote[indicesP_remote[iter]]]);}

            w[row_remote[indicesP_remote[iter]]] += values_remote[indicesP_remote[iter]] * vecValues[col_remote[indicesP_remote[iter]]];
        }
    }
//    }
    double t2_end = omp_get_wtime();
//    double t3_end = omp_get_wtime();

    double time0_local = t0_end-t0_start;
    double time0;
    MPI_Allreduce(&time0_local, &time0, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time1_local = t1_end-t1_start;
    double time1;
    MPI_Allreduce(&time1_local, &time1, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time2_local = t2_end-t2_start;
    double time2;
    MPI_Allreduce(&time2_local, &time2, 1, MPI_DOUBLE, MPI_SUM, comm);

//    double time3_local = t3_end-t3_start;
//    double time3;
//    MPI_Allreduce(&time3_local, &time3, 1, MPI_DOUBLE, MPI_SUM, comm);

    double time4_local = t4_end-t4_start;
    double time4;
    MPI_Allreduce(&time4_local, &time4, 1, MPI_DOUBLE, MPI_SUM, comm);

    time[0] += time0/nprocs;
    time[1] += time1/nprocs;
    time[2] += time2/nprocs;
//    time[3] += time3/nprocs;
    time[4] += time4/nprocs;

//    time[0] += time0_local;
//    time[1] += time1_local;
//    time[2] += time2_local;
//    time[3] += time3_local;
//    time[4] += time4_local;

    return 0;
}



int saena_matrix::residual(std::vector<value_t>& u, std::vector<value_t>& rhs, std::vector<value_t>& res){
    // Vector res = A*u - rhs;

//    int nprocs, rank;
//    MPI_Comm_size(comm, &nprocs);
//    MPI_Comm_rank(comm, &rank);

//    std::vector<value_t> matvecTemp(M);
    matvec(u, res);
//    if(rank==1)
//        for(long i=0; i<matvecTemp.size(); i++)
//            std::cout << matvecTemp[i] << std::endl;

#pragma omp parallel for
    for(index_t i = 0; i < M; i++)
        res[i] -= rhs[i];
//    if(rank==1)
//        for(long i=0; i<res.size(); i++)
//            std::cout << res[i] << std::endl;

    return 0;
}


int saena_matrix::inverse_diag(std::vector<value_t>& x) {
    int nprocs, rank;
//    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    for(nnz_t i=0; i<nnz_l; i++){
//        if(rank==4) printf("%u \t%lu \t%lu \t%f \n", i, entry[i].row, entry[i].col, entry[i].val);

        if(entry[i].row == entry[i].col){
            if(entry[i].val != 0)
                x[entry[i].row-split[rank]] = 1.0/entry[i].val;
            else{
                // there is no zero entry in the matrix (sparse), but just to be sure this part is added.
                if(rank==0) printf("Error: there is a zero diagonal element (at row index = %u)\n", entry[i].row);
                MPI_Finalize();
                return -1;
            }
        }
    }
    return 0;
}


int saena_matrix::jacobi(int iter, std::vector<value_t>& u, std::vector<value_t>& rhs, std::vector<value_t>& temp) {

// Ax = rhs
// u = u - (D^(-1))(Au - rhs)
// 1. B.matvec(u, one) --> put the value of matvec in one.
// 2. two = one - rhs
// 3. three = inverseDiag * two * omega
// 4. four = u - three

//    int rank;
//    MPI_Comm_rank(comm, &rank);

    for(int j = 0; j < iter; j++){
        matvec(u, temp);

#pragma omp parallel for
        for(index_t i = 0; i < M; i++){
            temp[i] -= rhs[i];
            temp[i] *= invDiag[i] * jacobi_omega;
            u[i]    -= temp[i];
        }
    }

    return 0;
}


int saena_matrix::chebyshev(int iter, std::vector<value_t>& u, std::vector<value_t>& rhs, std::vector<value_t>& res, std::vector<value_t>& d){

//    int rank;
//    MPI_Comm_rank(comm, &rank);

//    eig_max_of_invdiagXA *= 10;

    double alpha = 0.25 * eig_max_of_invdiagXA;
    double beta = eig_max_of_invdiagXA;
    double delta = (beta - alpha)/2;
    double theta = (beta + alpha)/2;
    double s1 = theta/delta;
    double rhok = 1/s1;
    double rhokp1, d1, d2;

    // first loop
    residual(u, rhs, res);
#pragma omp parallel for
    for(index_t i = 0; i < u.size(); i++){
        d[i] = (-res[i] * invDiag[i]) / theta;
        u[i] += d[i];
//        if(rank==0) printf("invDiag[%lu] = %f, \tres[%lu] = %f, \td[%lu] = %f, \tu[%lu] = %f \n",
//                           i, invDiag[i], i, res[i], i, d[i], i, u[i]);
    }

    for(int i = 1; i < iter; i++){
        rhokp1 = 1 / (2*s1 - rhok);
        d1     = rhokp1 * rhok;
        d2     = 2*rhokp1 / delta;
        rhok   = rhokp1;
        residual(u, rhs, res);

#pragma omp parallel for
        for(index_t j = 0; j < u.size(); j++){
            d[j] = ( d1 * d[j] ) + ( d2 * (-res[j]) * invDiag[j] );
            u[j] += d[j];
//        if(rank==0) printf("u[%lu] = %f \n", j, u[j]);
        }
    }

    return 0;
}

int saena_matrix::print(int ran){

    // if ran >= 0 print the matrix entries on proc with rank = ran
    // otherwise print the matrix entries on all processors in order. (first on proc 0, then proc 1 and so on.)

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(ran >= 0) {
        if (rank == ran) {
            printf("\nmatrix on proc = %d \n", ran);
            for (auto i:entry)
                std::cout << i << std::endl;
        }
    } else{
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("\nmatrix on proc = %d \n", proc);
                for (auto i:entry)
                    std::cout << i << std::endl;
            }
            MPI_Barrier(comm);
        }
    }

    return 0;
}