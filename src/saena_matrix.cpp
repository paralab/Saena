#include <fstream>
#include <algorithm>
#include <sys/stat.h>
#include <cstring>
#include "mpi.h"
#include <omp.h>
#include "saena_matrix.h"
#include "parUtils.h"
//#include "El.hpp"


#pragma omp declare reduction(vec_double_plus : std::vector<double> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
                    initializer(omp_priv = omp_orig)


saena_matrix::saena_matrix(){}


saena_matrix::saena_matrix(MPI_Comm com) {
    comm = com;
    comm_old = com;
}


saena_matrix::saena_matrix(char* Aname, MPI_Comm com) {
    // the following variables of SaenaMatrix class will be set in this function:
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
    // 2*sizeof(long)+sizeof(double) = 24
    nnz_g = st.st_size / 24;

    // find initial local nonzero
    initial_nnz_l = (unsigned int) (floor(1.0 * nnz_g / nprocs)); // initial local nnz
    if (rank == nprocs - 1)
        initial_nnz_l = nnz_g - (nprocs - 1) * initial_nnz_l;

//    printf("\nrank = %d, nnz_g = %u, initial_nnz_l = %u\n", rank, nnz_g, initial_nnz_l);

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

    offset = rank * (unsigned int) (floor(1.0 * nnz_g / nprocs)) * 24; // row index(long=8) + column index(long=8) + value(double=8) = 24

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
    unsigned int data_size = 0;
    if(!data_sorted.empty()){
        data[0] = data_sorted[0];
//        data.push_back(data_sorted[0].row);
//        data.push_back(data_sorted[0].col);
//        data.push_back(reinterpret_cast<unsigned long&>(data_sorted[0].val));
        data_size++;
    }

//    double val_temp;
    for(unsigned int i=1; i<data_sorted.size(); i++){
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
    double left_neighbor_last_val;
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
    MPI_Allreduce(&initial_nnz_l, &nnz_g, 1, MPI_UNSIGNED, MPI_SUM, comm);

    // *************************** find Mbig (global number of rows) ****************************
    // First find the maximum of rows. Then, compare it with the maximum of columns.
    // The one that is bigger is the size of the matrix.

    unsigned int Mbig_local = 0;
    for(unsigned long i=0; i<initial_nnz_l; i++){
        if(data[i].row > Mbig_local)
            Mbig_local = data[i].row;
    }

    last_element = data.back();
    if(last_element.col > Mbig_local)
        Mbig_local = last_element.col;

    MPI_Allreduce(&Mbig_local, &Mbig, 1, MPI_UNSIGNED, MPI_MAX, comm);
    Mbig++; // since indices start from 0, not 1.

//    printf("rank = %d, Mbig = %u, nnz_g = %u, initial_nnz_l = %u \n", rank, Mbig, nnz_g, initial_nnz_l);

//    if(rank==0){
//        for(unsigned long i = 0; i < initial_nnz_l; i++)
//            std::cout << data[i] << std::endl;}
}


saena_matrix::~saena_matrix() {
    if(freeBoolean){
        free(vIndex);
        free(vSend);
        free(vSendULong);
        free(vecValues);
        free(vecValuesULong);
        free(indicesP_local);
        free(indicesP_remote);
//        free(iter_local_array);
//        free(iter_remote_array);
        iter_local_array.clear();
        iter_local_array.shrink_to_fit();
        iter_remote_array.clear();
        iter_remote_array.shrink_to_fit();
        delete w_buff;
    }
}


int saena_matrix::set(unsigned int row, unsigned int col, double val){

    cooEntry temp_new = cooEntry(row, col, val);
    std::pair<std::set<cooEntry>::iterator, bool> p = data_coo.insert(temp_new);

    if (!p.second){
        auto hint = p.first; // hint is std::set<cooEntry>::iterator
        hint++;
        data_coo.erase(p.first);
        // in the case of duplicate, if the new value is zero, remove the older one and don't insert the zero.
        if(val != 0)
            data_coo.insert(hint, temp_new);
    }

    // if the entry is zero and it was not a duplicate, just erase it.
    if(p.second && val == 0)
        data_coo.erase(p.first);

    return 0;
}


int saena_matrix::set(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local){

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
            if(val[i] != 0)
                data_coo.insert(hint, temp_new);
        }

        // if the entry is zero, erase it.
        if(p.second && val[i] == 0)
            data_coo.erase(p.first);
    }

    return 0;
}


int saena_matrix::set2(unsigned int row, unsigned int col, double val){

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


int saena_matrix::set2(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local){

    if(nnz_local <= 0){
        printf("size in the set function is either zero or negative!");
        return 0;
    }

    cooEntry temp_old, temp_new;
    std::pair<std::set<cooEntry>::iterator, bool> p;

    for(unsigned int i=0; i<nnz_local; i++){
        if(val[i] != 0){
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
    unsigned int iter = 0;
    unsigned int Mbig_local = 0;
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
    unsigned int data_size = 0;
    if(!data_sorted.empty()){
        data[0] = data_sorted[0];
//        data.push_back(data_sorted[0].row);
//        data.push_back(data_sorted[0].col);
//        data.push_back(reinterpret_cast<unsigned long&>(data_sorted[0].val));
        data_size++;
    }

//    double val_temp;
    for(unsigned int i=1; i<data_sorted.size(); i++){
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
    double left_neighbor_last_val;
    if((last_element == first_element_neighbor) && add_duplicates ){
        if(rank != 0)
            MPI_Recv(&left_neighbor_last_val, 1, MPI_DOUBLE, rank-1, 0, comm, MPI_STATUS_IGNORE);

        if(rank!= nprocs-1)
            MPI_Send(&last_element.val, 1, MPI_DOUBLE, rank+1, 0, comm);

        data[0].val += left_neighbor_last_val;
    }

//    MPI_Barrier(comm); printf("rank = %d\t setup_initial 6666666666666666666\n", rank);  MPI_Barrier(comm);

    initial_nnz_l = data.size();
    MPI_Allreduce(&initial_nnz_l, &nnz_g, 1, MPI_UNSIGNED, MPI_SUM, comm);
//    MPI_Barrier(comm); printf("rank = %d, Mbig = %u, nnz_g = %u, initial_nnz_l = %u \n", rank, Mbig, nnz_g, initial_nnz_l); MPI_Barrier(comm);

    return 0;
}


int saena_matrix::setup_initial_data2(){

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    std::cout << rank << " : " << __func__ << initial_nnz_l << std::endl;

    std::set<cooEntry>::iterator it;
    unsigned int iter = 0;

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

    // put the first element of data_unsorted to data.
    unsigned int data_size = 0;
    if(!data_sorted.empty()){
        data[0] = data_sorted[0];
//        data.push_back(data_sorted[0].row);
//        data.push_back(data_sorted[0].col);
//        data.push_back(reinterpret_cast<unsigned long&>(data_sorted[0].val));
        data_size++;
    }

//    double val_temp;
    for(unsigned int i=1; i<data_sorted.size(); i++){
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
    double left_neighbor_last_val;
    if((last_element == first_element_neighbor) && add_duplicates ){
        if(rank != 0)
            MPI_Recv(&left_neighbor_last_val, 1, MPI_DOUBLE, rank-1, 0, comm, MPI_STATUS_IGNORE);

        if(rank!= nprocs-1)
            MPI_Send(&last_element.val, 1, MPI_DOUBLE, rank+1, 0, comm);

        data[0].val += left_neighbor_last_val;
    }

//    MPI_Barrier(comm); printf("rank = %d\t setup_initial 6666666666666666666\n", rank);  MPI_Barrier(comm);

    initial_nnz_l = data.size();
    unsigned long nnz_g_temp = nnz_g;
    MPI_Allreduce(&initial_nnz_l, &nnz_g, 1, MPI_UNSIGNED, MPI_SUM, comm);
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

    // todo: is it better to free the memory or it is good to keep the memory saved for performance?
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


int saena_matrix::set_zero(){

    // todo: add openmp
    for(unsigned long i = 0; i < nnz_l; i++)
        entry[i].val = 0;

    values_local.clear();
    values_remote.clear();

    return 0;
}


int saena_matrix::repartition(){
    // before using this function these variables of SaenaMatrix should be set:
    // Mbig", "nnz_g", "initial_nnz_l", "data"

    // the following variables of SaenaMatrix class will be set in this function:
    // "nnz_l", "M", "split", "entry"

    // summary: number of buckets are computed based of the number fo rows and number of processors.
    // firstSplit[] is of size n_buckets+1 and is a row partition of the matrix with almost equal number of rows.
    // then the buckets (firsSplit) are combined to have almost the same number of nonzeros. This is split[].

    bool repartition_verbose = false;

    // if set functions are used the following function should be used.
    if(!read_from_file)
        setup_initial_data();

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(repartition_verbose && rank==0) printf("repartition - step 1!\n");

    last_M_shrink = Mbig;

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
    else if(nprocs < Mbig){
        n_buckets = Mbig;
    } else{ // nprocs > Mbig
        // it may be better to set nprocs=Mbig and work only with the first Mbig processors.
        if(rank == 0)
            std::cout << "number of tasks cannot be greater than the number of rows of the matrix." << std::endl;
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

/*    if (rank==0){
        std::cout << "splitOffset:" << std::endl;
        for(long i=0; i<n_buckets; i++)
            std::cout << splitOffset[i] << std::endl;
    }*/

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
//        std::cout << "firstSplit:" << std::endl;
//        for(long i=0; i<n_buckets+1; i++)
//            std::cout << firstSplit[i] << std::endl;}

//    std::sort(entry.begin(), entry.end(), row_major);

//    if (rank==0){
//        std::cout << "entry row-major: nnz = " << initial_nnz_l << std::endl;
//        for(unsigned int i=0; i<n_buckets; i++)
//            std::cout << entry[i] << std::endl;}

    // H_l is the histogram of (local) nnz of buckets
    std::vector<long> H_l(n_buckets, 0);
    for(unsigned int i=0; i<initial_nnz_l; i++)
        H_l[lower_bound2(&firstSplit[0], &firstSplit[n_buckets], data[i].row)]++;

//    if (rank==1){
//        std::cout << "initial_nnz_l = " << initial_nnz_l << std::endl;
//        std::cout << "local histogram:" << std::endl;
//        for(unsigned int i=0; i<n_buckets; i++)
//            std::cout << H_l[i] << std::endl;}

    // H_g is the histogram of (global) nnz per bucket
    std::vector<long> H_g(n_buckets);
    MPI_Allreduce(&H_l[0], &H_g[0], n_buckets, MPI_LONG, MPI_SUM, comm);

    H_l.clear();
    H_l.shrink_to_fit();

//    if (rank==1){
//        std::cout << "global histogram:" << std::endl;
//        for(unsigned int i=0; i<n_buckets; i++){
//            std::cout << H_g[i] << std::endl;}}

    std::vector<long> H_g_scan(n_buckets);
    H_g_scan[0] = H_g[0];
    for (unsigned int i=1; i<n_buckets; i++)
        H_g_scan[i] = H_g[i] + H_g_scan[i-1];

    H_g.clear();
    H_g.shrink_to_fit();

/*    if (rank==0){
        std::cout << "scan of global histogram:" << std::endl;
        for(unsigned int i=0; i<n_buckets; i++)
            std::cout << H_g_scan[i] << std::endl;
    }*/

    if(repartition_verbose && rank==0) printf("repartition - step 3!\n");

    long procNum = 0;
    split.resize(nprocs+1);
    split[0]=0;
    for (unsigned int i=1; i<n_buckets; i++){
        //if (rank==0) std::cout << "(procNum+1)*nnz_g/nprocs = " << (procNum+1)*nnz_g/nprocs << std::endl;
        if (H_g_scan[i] > ((procNum+1)*nnz_g/nprocs)){
            procNum++;
            split[procNum] = firstSplit[i];
        }
    }
    split[nprocs] = Mbig;

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
        tempIndex = lower_bound2(&split[0], &split[nprocs], data[i].row);
        sendSizeArray[tempIndex]++;
    }

//    if (rank==0){
//        std::cout << "sendSizeArray:" << std::endl;
//        for(long i=0;i<nprocs;i++)
//            std::cout << sendSizeArray[i] << std::endl;}

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

    if(repartition_verbose && rank==0) printf("repartition - step 5!\n");

    long procOwner;
    unsigned int bufTemp;
    cooEntry* sendBuf = (cooEntry*)malloc(sizeof(cooEntry)*initial_nnz_l);
    unsigned int* sIndex = (unsigned int*)malloc(sizeof(unsigned int)*nprocs);
    std::fill(&sIndex[0], &sIndex[nprocs], 0);

    // memcpy(sendBuf, data.data(), initial_nnz_l*3*sizeof(unsigned long));

    // todo: try to avoid this for loop.
    for (long i=0; i<initial_nnz_l; i++){
        procOwner = lower_bound2(&split[0], &split[nprocs], data[i].row);
        bufTemp = sOffset[procOwner]+sIndex[procOwner];
        memcpy(sendBuf+bufTemp, data.data() + i, sizeof(cooEntry));
        // todo: the above line is better than the following three lines. think why it works.
//        sendBuf[bufTemp].row = data[3*i];
//        sendBuf[bufTemp].col = data[3*i+1];
//        sendBuf[bufTemp].val = data[3*i+2];
//        if(rank==1) std::cout << sendBuf[bufTemp].row << "\t" << sendBuf[bufTemp].col << "\t" << sendBuf[bufTemp].val << std::endl;
        sIndex[procOwner]++;
    }

    // clear data and free memory.
    data.clear();
    data.shrink_to_fit();

    free(sIndex);

//    if (rank==1){
//        std::cout << "sendBufJ:" << std::endl;
//        for (long i=0; i<initial_nnz_l; i++)
//            std::cout << sendBufJ[i] << std::endl;
//    }

    nnz_l = rOffset[nprocs-1] + recvSizeArray[nprocs-1];
//    printf("rank=%d \t A.nnz_l=%u \t A.nnz_g=%u \n", rank, nnz_l, nnz_g);

//    cooEntry* entry = (cooEntry*)malloc(sizeof(cooEntry)*nnz_l);
//    cooEntry* entryP = &entry[0];

    if(repartition_verbose && rank==0) printf("repartition - step 6!\n");

    entry.resize(nnz_l);
    MPI_Alltoallv(sendBuf, sendSizeArray, sOffset, cooEntry::mpi_datatype(), &entry[0], recvSizeArray, rOffset, cooEntry::mpi_datatype(), comm);

    free(sendSizeArray);
    free(recvSizeArray);
    free(sOffset);
    free(rOffset);
    free(sendBuf);

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

    // *************************** exchange data ****************************

    long tempIndex;
//    int sendSizeArray[nprocs];
    int* sendSizeArray = (int*)malloc(sizeof(int)*nprocs);
    std::fill(&sendSizeArray[0], &sendSizeArray[nprocs], 0);
    for (unsigned int i=0; i<initial_nnz_l; i++){
        tempIndex = lower_bound2(&split[0], &split[nprocs], data[i].row);
        sendSizeArray[tempIndex]++;
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

    long procOwner;
    unsigned int bufTemp;
    cooEntry* sendBuf = (cooEntry*)malloc(sizeof(cooEntry)*initial_nnz_l);
    unsigned int* sIndex = (unsigned int*)malloc(sizeof(unsigned int)*nprocs);
    std::fill(&sIndex[0], &sIndex[nprocs], 0);

    // memcpy(sendBuf, data.data(), initial_nnz_l*3*sizeof(unsigned long));

    // todo: try to avoid this for loop.
    for (long i=0; i<initial_nnz_l; i++){
        procOwner = lower_bound2(&split[0], &split[nprocs], data[i].row);
        bufTemp = sOffset[procOwner]+sIndex[procOwner];
        memcpy(sendBuf+bufTemp, data.data() + i, sizeof(cooEntry));
        // todo: the above line is better than the following three lines. think why it works.
//        sendBuf[bufTemp].row = data[3*i];
//        sendBuf[bufTemp].col = data[3*i+1];
//        sendBuf[bufTemp].val = data[3*i+2];
//        if(rank==1) std::cout << sendBuf[bufTemp].row << "\t" << sendBuf[bufTemp].col << "\t" << sendBuf[bufTemp].val << std::endl;
        sIndex[procOwner]++;
    }

    // clear data and free memory.
    data.clear();
    data.shrink_to_fit();

    free(sIndex);

//    if (rank==1){
//        std::cout << "sendBufJ:" << std::endl;
//        for (long i=0; i<initial_nnz_l; i++)
//            std::cout << sendBufJ[i] << std::endl;
//    }

    unsigned long nnz_l_temp;
    nnz_l_temp = rOffset[nprocs-1] + recvSizeArray[nprocs-1];
//    printf("rank=%d \t A.nnz_l=%u \t A.nnz_g=%u \n", rank, nnz_l, nnz_g);

    if(nnz_l_temp != nnz_l) printf("error: number of local nonzeros is changed on processor %d during the matrix update!\n", rank);

    if(repartition_verbose && rank==0) printf("repartition - step 3!\n");

    entry.clear();
    entry.resize(nnz_l_temp);
    MPI_Alltoallv(sendBuf, sendSizeArray, sOffset, cooEntry::mpi_datatype(), &entry[0], recvSizeArray, rOffset, cooEntry::mpi_datatype(), comm);
    entry.shrink_to_fit();

    free(sendSizeArray);
    free(recvSizeArray);
    free(sOffset);
    free(rOffset);
    free(sendBuf);

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

    MPI_Barrier(comm);
    printf("repartition3 - start! rank = %d, Mbig = %u, M = %u, nnz_g = %u, nnz_l = %u \n",
           rank, Mbig, M, nnz_g, nnz_l);
    MPI_Barrier(comm);

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

    if(repartition_verbose && rank==0) printf("repartition3 - step 2!\n");

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

    if(repartition_verbose && rank==0) printf("repartition3 - step 3!\n");

//    if (rank==0){
//        std::cout << std::endl << "split old:" << std::endl;
//        for(unsigned int i=0; i<nprocs+1; i++)
//            std::cout << split[i] << std::endl;
//        std::cout << std::endl;}

    long procNum = 0;
    // determine number of rows on each proc based on having almost the same number of nonzeros per proc.
    // -------------------------------------------
    for (unsigned int i = 1; i < n_buckets; i++){
        if (H_g_scan[i] > (procNum+1)*nnz_g/nprocs){
            procNum++;
            split[procNum] = firstSplit[i];
        }
    }
    split[nprocs] = Mbig;
    split_old = split;

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

    if(repartition_verbose && rank==0) printf("repartition3 - step 4!\n");

//    unsigned int M_min_global;
//    MPI_Allreduce(&M, &M_min_global, 1, MPI_UNSIGNED, MPI_MIN, comm);

    // *************************** exchange data ****************************

    long tempIndex;
    int* sendSizeArray = (int*)malloc(sizeof(int)*nprocs);
    std::fill(&sendSizeArray[0], &sendSizeArray[nprocs], 0);
    for (unsigned int i=0; i<initial_nnz_l; i++){
        tempIndex = lower_bound2(&split[0], &split[nprocs], entry[i].row);
        sendSizeArray[tempIndex]++;
//        if(rank==5)
//            printf("%ld \t%ld \n", entry[i].row, tempIndex);
    }

//    if (rank==4){
//        std::cout << "sendSizeArray:" << std::endl;
//        for(long i=0;i<nprocs;i++)
//            std::cout << sendSizeArray[i] << std::endl;}

    int* recvSizeArray = (int*)malloc(sizeof(int)*nprocs);
    MPI_Alltoall(sendSizeArray, 1, MPI_INT, recvSizeArray, 1, MPI_INT, comm);

//    if (rank==0){
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

//    if (rank==0){
//        std::cout << "rOffset:" << std::endl;
//        for(long i=0;i<nprocs;i++)
//            std::cout << rOffset[i] << std::endl;}

    if(repartition_verbose && rank==0) printf("repartition3 - step 5!\n");

    long procOwner;
    unsigned int bufTemp;
    cooEntry* sendBuf = (cooEntry*)malloc(sizeof(cooEntry)*initial_nnz_l);
    unsigned int* sIndex = (unsigned int*)malloc(sizeof(unsigned int)*nprocs);
    std::fill(&sIndex[0], &sIndex[nprocs], 0);

    // todo: try to avoid this for loop.
    // todo: if the order is changed to row-major order, then sendBuf is already ready and also
    // set up sendBuf for alltoallv
    for (long i=0; i<initial_nnz_l; i++){
        procOwner = lower_bound2(&split[0], &split[nprocs], entry[i].row);
        bufTemp = sOffset[procOwner]+sIndex[procOwner];
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

    if(repartition_verbose && rank==0) printf("repartition3 - step 6!\n");

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

    if(repartition_verbose) {
        MPI_Barrier(comm);
        printf("repartition3 - step 7! rank = %d, Mbig = %u, M = %u, nnz_g = %u, nnz_l = %u \n",
               rank, Mbig, M, nnz_g, nnz_l);
        MPI_Barrier(comm);
    }

    MPI_Barrier(comm);
    printf("repartition3 - step 7! rank = %d, Mbig = %u, M = %u, nnz_g = %u, nnz_l = %u \n",
           rank, Mbig, M, nnz_g, nnz_l);
    MPI_Barrier(comm);

//    if(shrinked)
//        shrink_cpu();

    if(repartition_verbose && rank==0) printf("repartition3 - done!\n");

    return 0;
}


/*
int saena_matrix::shrink_cpu(){

    // if number of rows on Ac < threshold*number of rows on A, then shrink.
    // redistribute Ac from processes 4k+1, 4k+2 and 4k+3 to process 4k.
//    MPI_Comm comm = comm;
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    unsigned long i;
    bool verbose_shrink = true;

    MPI_Barrier(comm);
    if(rank==0) printf("\n****************************\n");
    if(rank==0) printf("***********SHRINK***********\n");
    if(rank==0) printf("****************************\n\n");
    MPI_Barrier(comm);

//    MPI_Barrier(comm); printf("rank = %d \tnnz_l = %u \n", rank, Ac->nnz_l); MPI_Barrier(comm);

//    MPI_Barrier(comm);
//    if(rank == 0){
//        std::cout << "\nbefore shrinking!!!" <<std::endl;
//        std::cout << "\nrank = " << rank << ", size = " << Ac->entry.size() <<std::endl;
//        for(i=0; i < Ac->entry.size(); i++)
//            std::cout << i << "\t" << Ac->entry[i]  <<std::endl;
//    }
//    MPI_Barrier(comm);
//    if(rank == 1){
//        std::cout << "\nrank = " << rank << ", size = " << Ac->entry.size() <<std::endl;
//        for(i=0; i < Ac->entry.size(); i++)
//            std::cout << i << "\t" << Ac->entry[i]  <<std::endl;
//    }
//    MPI_Barrier(comm);
//    if(rank == 2){
//        std::cout << "\nbefore shrinking!!!" <<std::endl;
//        std::cout << "\nrank = " << rank << ", size = " << Ac->entry.size() <<std::endl;
//        for(i=0; i < Ac->entry.size(); i++)
//            std::cout << i << "\t" << Ac->entry[i]  <<std::endl;
//    }
//    MPI_Barrier(comm);

    // assume cpu_shrink_thre2 is 4 (it is simpler to explain)
    // 1 - create a new comm, consisting only of processes 4k, 4k+1, 4k+2 and 4k+3 (with new ranks 0,1,2,3)
    int color = rank / cpu_shrink_thre2;
    MPI_Comm_split(comm, color, rank, &comm_horizontal);

    int rank_new, nprocs_new;
    MPI_Comm_size( comm_horizontal, &nprocs_new);
    MPI_Comm_rank( comm_horizontal, &rank_new);

//    MPI_Barrier(Ac->comm_horizontal);
//    printf("rank = %d, rank_new = %d on Ac->comm_horizontal \n", rank, rank_new);

    // 2 - update the number of rows on process 4k, and resize "entry".
    unsigned int Ac_M_neighbors_total = 0;
    unsigned int Ac_nnz_neighbors_total = 0;
    MPI_Reduce(& M, &Ac_M_neighbors_total, 1, MPI_UNSIGNED, MPI_SUM, 0,  comm_horizontal);
    MPI_Reduce(&nnz_l, &Ac_nnz_neighbors_total, 1, MPI_UNSIGNED, MPI_SUM, 0,  comm_horizontal);

    if(rank_new == 0){
         M = Ac_M_neighbors_total;
         entry.resize(Ac_nnz_neighbors_total);
        printf("rank = %d, Ac_M_neighbors = %d \n", rank, Ac_M_neighbors_total);
        printf("rank = %d, Ac_nnz_neighbors = %d \n", rank, Ac_nnz_neighbors_total);
    }

    // last cpu that its right neighbors are going be shrinked to.
    auto last_root_cpu = (unsigned int)floor(nprocs/ cpu_shrink_thre2) *  cpu_shrink_thre2;
    printf("last_root_cpu = %u\n", last_root_cpu);
    MPI_Barrier(comm);


    int neigbor_rank;
    unsigned int A_recv_nnz = 0; // set to 0 just to avoid "not initialized" warning
    unsigned long offset =  nnz_l; // put the data on root from its neighbors at the end of entry[] which is of size nnz_l
    if(nprocs_new > 1) { // if there is no neighbor, skip.
        for (neigbor_rank = 1; neigbor_rank <  cpu_shrink_thre2; neigbor_rank++) {

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
                MPI_Recv(&A_recv_nnz, 1, MPI_UNSIGNED, neigbor_rank, 0,  comm_horizontal, MPI_STATUS_IGNORE);

            if (rank_new == neigbor_rank)
                MPI_Send(&nnz_l, 1, MPI_UNSIGNED, 0, 0,  comm_horizontal);

            // 4 - send and receive Ac.
            if (rank_new == 0) {
//                printf("rank = %d, neigbor_rank = %d, recv size = %u, offset = %lu \n", rank, neigbor_rank, A_recv_nnz, offset);
                MPI_Recv(&*( entry.begin() + offset), A_recv_nnz, cooEntry::mpi_datatype(), neigbor_rank, 1,
                          comm_horizontal, MPI_STATUS_IGNORE);
                offset += A_recv_nnz; // set offset for the next iteration
            }

            if (rank_new == neigbor_rank) {
//                printf("rank = %d, neigbor_rank = %d, send size = %u, offset = %lu \n", rank, neigbor_rank, Ac->nnz_l, offset);
                MPI_Send(&*entry.begin(),  nnz_l, cooEntry::mpi_datatype(), 0, 1,  comm_horizontal);
            }

            // update local index for rows
//        if(rank_new == 0)
//            for(i=0; i < A_recv_nnz; i++)
//                Ac->entry[offset + i].row += Ac->split[rank + neigbor_rank] - Ac->split[rank];
        }
    }

    // even though the entries are sorted before shrinking, after shrinking they still need to be sorted locally,
    // because remote elements damage sorting after shrinking.
    std::sort( entry.begin(),  entry.end());

//    MPI_Barrier(comm); MPI_Barrier(Ac->comm_horizontal);
//    if(rank == 2){
//        std::cout << "\nafter shrinking: rank = " << rank << ", size = " << Ac->entry.size() <<std::endl;
//        for(i=0; i < Ac->entry.size(); i++)
//            std::cout << i << "\t" << Ac->entry[i]  <<std::endl;}
//    MPI_Barrier(comm); MPI_Barrier(Ac->comm_horizontal);

     active = false;
//    Ac->active_old_comm = true; // this is used for prolong and post-smooth
    if(rank_new == 0){
         active = true;
//        printf("active: rank = %d, rank_new = %d \n", rank, rank_new);
    }

    // 5 - create a new comm including only processes with 4k rank.
    MPI_Group bigger_group;
    MPI_Comm_group(comm, &bigger_group);
    auto total_active_procs = (unsigned int)ceil((double)nprocs /  cpu_shrink_thre2); // note: this is ceiling, not floor.
    std::vector<int> ranks(total_active_procs);
    for(unsigned int i = 0; i < total_active_procs; i++)
        ranks[i] =  cpu_shrink_thre2 * i;
//        ranks.push_back(Ac->cpu_shrink_thre2 * i);

//    printf("total_active_procs = %u \n", total_active_procs);
//    for(i=0; i<ranks.size(); i++)
//        if(rank==0) std::cout << ranks[i] << std::endl;

    MPI_Group group_new;
    MPI_Group_incl(bigger_group, total_active_procs, &*ranks.begin(), &group_new);
    MPI_Comm_create_group(comm, group_new, 0, & comm);

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

    // 6 - update 4k.nnz_l and split. nnz_g stays the same, so no need to update.
    split.clear();
    if(active){
        nnz_l =  entry.size();
        split_old =  split; // save the old split for shrinking rhs and u
        split.resize(total_active_procs+1);
        split.shrink_to_fit();
        split[0] = 0;
        split[total_active_procs] = Mbig;
        for(unsigned int i = 1; i < total_active_procs; i++){
//            if(rank==0) printf("%u \t%lu \n", i, split_old[ranks[i]]);
            split[i] = split_old[ranks[i]];
        }
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

     last_M_shrink =  Mbig;
     shrinked = true;
    return 0;
}
*/


int saena_matrix::shrink_cpu(){

    // if number of rows on Ac < threshold*number of rows on A, then shrink.
    // redistribute Ac from processes 4k+1, 4k+2 and 4k+3 to process 4k.
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
//    unsigned long i;
    bool verbose_shrink = false;

    MPI_Barrier(comm);
    if(rank==0) printf("\n****************************\n");
    if(rank==0) printf("***********SHRINK***********\n");
    if(rank==0) printf("****************************\n\n");
    MPI_Barrier(comm);

//    MPI_Barrier(comm); printf("rank = %d \tnnz_l = %u \n", rank, nnz_l); MPI_Barrier(comm);

//    MPI_Barrier(comm);
//    if(rank == 0){
//        std::cout << "\nbefore shrinking!!!" <<std::endl;
//        std::cout << "\nrank = " << rank << ", size = " << Ac->entry.size() <<std::endl;
//        for(i=0; i < Ac->entry.size(); i++)
//            std::cout << i << "\t" << Ac->entry[i]  <<std::endl;
//    }
//    MPI_Barrier(comm);
//    if(rank == 1){
//        std::cout << "\nrank = " << rank << ", size = " << Ac->entry.size() <<std::endl;
//        for(i=0; i < Ac->entry.size(); i++)
//            std::cout << i << "\t" << Ac->entry[i]  <<std::endl;
//    }
//    MPI_Barrier(comm);
//    if(rank == 2){
//        std::cout << "\nbefore shrinking!!!" <<std::endl;
//        std::cout << "\nrank = " << rank << ", size = " << Ac->entry.size() <<std::endl;
//        for(i=0; i < Ac->entry.size(); i++)
//            std::cout << i << "\t" << Ac->entry[i]  <<std::endl;
//    }
//    MPI_Barrier(comm);

    // assume cpu_shrink_thre2 is 4 (it is simpler to explain)
    // 1 - create a new comm, consisting only of processes 4k, 4k+1, 4k+2 and 4k+3 (with new ranks 0,1,2,3)
    int color = rank / cpu_shrink_thre2;
    MPI_Comm_split(comm, color, rank, &comm_horizontal);

    int rank_new, nprocs_new;
    MPI_Comm_size(comm_horizontal, &nprocs_new);
    MPI_Comm_rank(comm_horizontal, &rank_new);

    MPI_Barrier(comm_horizontal);
    printf("rank = %d, rank_new = %d on Ac->comm_horizontal \n", rank, rank_new);
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

    active = false;
//    Ac->active_old_comm = true; // this is used for prolong and post-smooth
    if(rank_new == 0){
        active = true;
        printf("active: rank = %d, rank_new = %d \n", rank, rank_new);
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

    MPI_Group group_new;
    MPI_Group_incl(bigger_group, total_active_procs, &*ranks.begin(), &group_new);
    MPI_Comm_create_group(comm, group_new, 0, &comm);

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

    split.clear();
    if(active){
        split.resize(total_active_procs+1);
        split.shrink_to_fit();
        split[0] = 0;
        split[total_active_procs] = Mbig;
        for(unsigned int i = 1; i < total_active_procs; i++){
            if(rank==0) printf("%u \t%lu \n", i, split_old[ranks[i]]);
            split[i] = split_old[ranks[i]];
        }
    }

    MPI_Barrier(comm);
    if(rank==0) {
        printf("\nsplit:\n");
        for (long i = 0; i < total_active_procs + 1; i++)
            printf("%lu \n", split[i]);
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

    // todo: how should I free these?
//    free(&bigger_group);
//    free(&group_new);
//    free(&comm_new2);

    last_M_shrink = Mbig;
//    shrinked = true;
    return 0;
}


int saena_matrix::matrix_setup() {
    // before using this function these variables of SaenaMatrix should be set:
    // "Mbig", "M", "nnz_g", "split", "entry",

    if(active) {
        int nprocs, rank;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);
        bool verbose_matrix_setup = false;

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, Mbig = %u, M = %u, nnz_g = %u, nnz_l = %u \n", rank, Mbig, M, nnz_g, nnz_l);
            MPI_Barrier(comm);
        }
        MPI_Barrier(comm); printf("matrix_setup: rank = %d, Mbig = %u, M = %u, nnz_g = %u, nnz_l = %u \n", rank, Mbig, M, nnz_g, nnz_l);

        assembled = true;
        freeBoolean = true; // use this parameter to know if destructor for SaenaMatrix class should free the variables or not.

//        MPI_Barrier(comm);
//        printf("matrix_setup: rank = %d, Mbig = %u, M = %u, nnz_g = %u, nnz_l = %u \n", rank, Mbig, M, nnz_g, nnz_l);
//        MPI_Barrier(comm);
//        if(rank==0){
//            printf("\nrank = %d\n", rank);
//            for(unsigned int i=0; i<nnz_l; i++)
//                printf("%u \t%lu \t%lu \t%f \n", i, entry[i].row, entry[i].col, entry[i].val);}
//        MPI_Barrier(comm);

//        if (rank==0){
//            std::cout << std::endl << "split:" << std::endl;
//            for(unsigned int i=0; i<nprocs+1; i++)
//                std::cout << split[i] << std::endl;
//            std::cout << std::endl;
//        }

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
//        recvCount.resize(nprocs);
        recvCount.assign(nprocs, 0);
        nnzPerRow_local.assign(M, 0);
//    nnzPerRow.assign(M,0);
//    nnzPerCol_local.assign(Mbig,0); // todo: Nbig = Mbig, assuming A is symmetric.
//    nnzPerCol_remote.assign(M,0);

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
//            nnzPerRow_remote[row[0]-split[rank]]++;

                values_remote.push_back(entry[0].val);
                row_remote.push_back(entry[0].row);
                col_remote_size++;
                col_remote.push_back(col_remote_size - 1);
                col_remote2.push_back(entry[0].col);
//            nnzPerCol_remote[col_remote_size]++;
                nnzPerCol_remote.push_back(1);

                vElement_remote.push_back(entry[0].col);
                vElementRep_remote.push_back(1);
                recvCount[lower_bound2(&split[0], &split[nprocs], entry[0].col)] = 1;
            }
        }

        if(entry.size() >= 2){
            for (long i = 1; i < nnz_l; i++) {
//                nnzPerRow[row[i]-split[rank]]++;
//                if(rank==1) std::cout << entry[i] << std::endl;
                if (entry[i].col >= split[rank] && entry[i].col < split[rank + 1]) {
//                    nnzPerCol_local[col[i]]++;
                    nnz_l_local++;
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
//                nnzPerRow_remote[row[i]-split[rank]]++;

                    values_remote.push_back(entry[i].val);
                    row_remote.push_back(entry[i].row);
                    // col_remote2 is the original col value and will be used in making strength matrix. col_remote will be used for matevec.
                    col_remote2.push_back(entry[i].col);

                    if (entry[i].col != entry[i - 1].col) {
                        col_remote_size++;
                        vElement_remote.push_back(entry[i].col);
                        vElementRep_remote.push_back(1);
                        procNum = lower_bound2(&split[0], &split[nprocs], entry[i].col);
                        recvCount[procNum]++;
                        nnzPerCol_remote.push_back(1);
                    } else {
                        (*(vElementRep_remote.end() - 1))++;
                        (*(nnzPerCol_remote.end() - 1))++;
                    }
                    // the original col values are not being used. the ordering starts from 0, and goes up by 1.
                    col_remote.push_back(col_remote_size - 1);
//                nnzPerCol_remote[col_remote_size]++;
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
//            std::cout << "recvCount: rank = " << rank << std::endl;
//            for(int i=0; i<nprocs; i++)
//                std::cout << "from proc " << i << ": " << recvCount[i] << std::endl;}

//        sendCount.assign(M, 0);
        sendCount.resize(nprocs);
        MPI_Alltoall(&recvCount[0], 1, MPI_INT, &sendCount[0], 1, MPI_INT, comm);

//        MPI_Barrier(comm);
//        if (rank==1){
//            std::cout << "sendCount: rank=" << rank << std::endl;
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

//        for (unsigned int i = 0; i < nprocs; i++)
//            if(rank==1) printf("%u \t%d \t%d \n", i, recvCount[i], recvCountScan[i]);

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

        vIndex = (long *) malloc(sizeof(long) * vIndexSize);
        MPI_Alltoallv(&(*(vElement_remote.begin())), &recvCount[0], &*(rdispls.begin()), MPI_LONG, vIndex, &sendCount[0],
                      &(*(vdispls.begin())), MPI_LONG, comm);

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
        for (unsigned int i = 0; i < vIndexSize; i++)
            vIndex[i] -= split[rank];
        for (unsigned int i = 0; i < row_local.size(); i++)
            row_local[i] -= split[rank];
        for (unsigned int i = 0; i < row_remote.size(); i++)
            row_remote[i] -= split[rank];

        // vSend = vector values to send to other procs
        // vecValues = vector values that received from other procs
        // These will be used in matvec and they are set here to reduce the time of matvec.
        vSend = (double *) malloc(sizeof(double) * vIndexSize);
        vecValues = (double *) malloc(sizeof(double) * recvSize);

        vSendULong = (unsigned long *) malloc(sizeof(unsigned long) * vIndexSize);
        vecValuesULong = (unsigned long *) malloc(sizeof(unsigned long) * recvSize);

        //    printf("rank = %d\t 11111111111111111111111111\n", rank);

        // *************************** find sortings ****************************
        //find the sorting on rows on both local and remote data to be used in matvec

        indicesP_local = (unsigned long *) malloc(sizeof(unsigned long) * nnz_l_local);
        for (unsigned long i = 0; i < nnz_l_local; i++)
            indicesP_local[i] = i;
        unsigned long *row_localP = &(*(row_local.begin()));
        std::sort(indicesP_local, &indicesP_local[nnz_l_local], sort_indices(row_localP));

        indicesP_remote = (unsigned long *) malloc(sizeof(unsigned long) * nnz_l_remote);
        for (unsigned long i = 0; i < nnz_l_remote; i++)
            indicesP_remote[i] = i;
        unsigned long *row_remoteP = &(*(row_remote.begin()));
        std::sort(indicesP_remote, &indicesP_remote[nnz_l_remote], sort_indices(row_remoteP));

        //    printf("rank = %d\t 333333333333333333333\n", rank);

        //    indicesP = (unsigned long*)malloc(sizeof(unsigned long)*nnz_l);
        //    for(unsigned long i=0; i<nnz_l; i++)
        //        indicesP[i] = i;
        //    std::sort(indicesP, &indicesP[nnz_l], sort_indices2(&*entry.begin()));

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
            unsigned int istart = 0; // starting row index for each thread
            unsigned int iend = 0;   // last row index for each thread
            unsigned int iter_local, iter_remote;

            // compute local iter to do matvec using openmp (it is done to make iter independent data on threads)
            bool first_one = true;
#pragma omp for
            for (unsigned int i = 0; i < M; ++i) {
                if (first_one) {
                    istart = i;
                    first_one = false;
                    iend = istart;
                }
                iend++;
            }
//            if(rank==1) printf("thread id = %d, istart = %u, iend = %u \n", thread_id, istart, iend);

            iter_local = 0;
            for (unsigned int i = istart; i < iend; ++i)
                iter_local += nnzPerRow_local[i];

            iter_local_array[0] = 0;
            iter_local_array[thread_id + 1] = iter_local;

            // compute remote iter to do matvec using openmp (it is done to make iter independent data on threads)
            first_one = true;
#pragma omp for
            for (unsigned int i = 0; i < col_remote_size; ++i) {
                if (first_one) {
                    istart = i;
                    first_one = false;
                    iend = istart;
                }
                iend++;
            }

            iter_remote = 0;
            if (nnzPerCol_remote.size() != 0) {
                for (unsigned int i = istart; i < iend; ++i)
                    iter_remote += nnzPerCol_remote[i];
            }

            iter_remote_array[0] = 0;
            iter_remote_array[thread_id + 1] = iter_remote;

            /*        if (rank==1 && thread_id==0){
                        std::cout << "M=" << M << std::endl;
                        std::cout << "recvSize=" << recvSize << std::endl;
                        std::cout << "istart=" << istart << std::endl;
                        std::cout << "iend=" << iend << std::endl;
                        std::cout  << "nnz_l=" << nnz_l << ", iter_remote=" << iter_remote << ", iter_local=" << iter_local << std::endl;
                    }*/
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
            for(unsigned int i = 0; i < M; i++)
                nnzPerRow_local2[i] = nnzPerRow_local[i];

            unsigned int iter = iter_local_array[thread_id];
//            if(rank==1) printf("rank %d thread %d \titer = %d \tindices = %lu \trow = %lu \n", rank, thread_id, iter, indicesP_local[iter], row_local[indicesP_local[iter]]);
            unsigned long starting_row;
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

        w_buff = new double[num_threads*M];

        // *************************** find the greatest eigenvalue ****************************

        if(verbose_matrix_setup) {
            MPI_Barrier(comm);
            printf("matrix_setup: rank = %d, find_eig \n", rank);
            MPI_Barrier(comm);
        }

        // todo: execute this line only if the smoother is set to chebyshev.
        // set eig_max here
//        find_eig();

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
        for (long i = 1; i < nnz_l; i++) {
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

    inverse_diag(invDiag);

    // update eig_max here
    //todo: is this line required?
//    find_eig();

    return 0;
}


int saena_matrix::matvec(std::vector<double>& v, std::vector<double>& w) {
// todo: to reduce the communication during matvec, consider reducing number of columns during coarsening,
// todo: instead of reducing general non-zeros, since that is what is communicated for matvec.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ){
//        printf("A.M != v.size() in matvec!!!\n");}

    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i=0;i<vIndexSize;i++)
        vSend[i] = v[( vIndex[i] )];

/*    if (rank==0){
        std::cout << "vIndexSize=" << vIndexSize << ", vSend: rank=" << rank << std::endl;
        for(int i=0; i<vIndexSize; i++)
            std::cout << vSend[i] << std::endl;
    }*/

    // iSend your data, and iRecv from others
    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses = new MPI_Status[numSendProc+numRecvProc];

    //First place all recv requests. Do not recv from self.
    for(int i = 0; i < numRecvProc; i++) {
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));
    }

    //Next send the messages. Do not send to self.
    for(int i = 0; i < numSendProc; i++) {
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));
    }

/*    if (rank==0){
        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
        for(int i=0; i<recvSize; i++)
            std::cout << vecValues[i] << std::endl;
    }*/

    double* v_p = &v[0] - split[rank];
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

    // Wait for the communication to finish.
    MPI_Waitall(numRecvProc, requests, statuses);

/*    if (rank==1){
        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
        for(int i=0; i<recvSize; i++)
            std::cout << vecValues[i] << std::endl;
    }*/

    // remote loop
    // todo: data race happens during "omp for" here, since the "for" loop splits based on the remote columns, but
    // todo: w[row] are being computed in every iteration , which means different threads may access the same w[row].

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

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    delete [] requests;
    delete [] statuses;
    return 0;
}


int saena_matrix::matvec2(std::vector<double>& v, std::vector<double>& w) {
// todo: to reduce the communication during matvec, consider reducing number of columns during coarsening,
// todo: instead of reducing general non-zeros, since that is what is communicated for matvec.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ){
//        printf("A.M != v.size() in matvec!!!\n");}

//    totalTime = 0;
//    double t10 = MPI_Wtime();

    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i = 0; i < vIndexSize; i++)
        vSend[i] = v[( vIndex[i] )];
//    double t20 = MPI_Wtime();
//    time[0] += (t20-t10);

/*    if (rank==0){
        std::cout << "vIndexSize=" << vIndexSize << ", vSend: rank=" << rank << std::endl;
        for(int i=0; i<vIndexSize; i++)
            std::cout << vSend[i] << std::endl;
    }*/

//    double t13 = MPI_Wtime();
    // iSend your data, and iRecv from others
    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses = new MPI_Status[numSendProc+numRecvProc];

    //First place all recv requests. Do not recv from self.
    for(int i = 0; i < numRecvProc; i++) {
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));
    }

    //Next send the messages. Do not send to self.
    for(int i = 0; i < numSendProc; i++) {
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));
    }

/*    if (rank==0){
        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
        for(int i=0; i<recvSize; i++)
            std::cout << vecValues[i] << std::endl;
    }*/


//    double t11 = MPI_Wtime();
    // local loop
    // ----------
//    std::fill(&*w.begin(), &*w.end(), 0);
    double* v_p = &v[0] - split[rank];
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

        int left_over = iter_local_array2[thread_id] - iter_local_array[thread_id];
        if(left_over!=0){
            for (unsigned int iter = iter_local_array[thread_id]; iter < iter_local_array2[thread_id]; ++iter) {
                w[row_local[indicesP_local[iter]]] += values_local[indicesP_local[iter]] * v_p[col_local[indicesP_local[iter]]];
            }
        }
    }

//    double t21 = MPI_Wtime();
//    time[1] += (t21-t11);

    // Wait for comm to finish
    // -----------------------
    MPI_Waitall(numRecvProc, requests, statuses);

/*    if (rank==1){
        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
        for(int i=0; i<recvSize; i++)
            std::cout << vecValues[i] << std::endl;
    }*/

    // remote loop
    // -----------
    // todo: data race happens during "omp for" here, since the "for" loop splits based on the remote columns, but
    // todo: w[row] are being computed in every iteration , which means different threads may access the same w[row].

//    double t12 = MPI_Wtime();
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

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    delete [] requests;
    delete [] statuses;

//    double t22 = MPI_Wtime();
//    time[2] += (t22-t12);
//    double t23 = MPI_Wtime();
//    time[3] += (t23-t13);

    return 0;
}


int saena_matrix::matvec3(std::vector<double>& v, std::vector<double>& w) {
// todo: to reduce the communication during matvec, consider reducing number of columns during coarsening,
// todo: instead of reducing general non-zeros, since that is what is communicated for matvec.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ){
//        printf("A.M != v.size() in matvec!!!\n");}

//    totalTime = 0;
//    double t10 = MPI_Wtime();

    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i = 0; i < vIndexSize; i++)
        vSend[i] = v[( vIndex[i] )];
//    double t20 = MPI_Wtime();
//    time[0] += (t20-t10);

/*    if (rank==0){
        std::cout << "vIndexSize=" << vIndexSize << ", vSend: rank=" << rank << std::endl;
        for(int i=0; i<vIndexSize; i++)
            std::cout << vSend[i] << std::endl;
    }*/

//    double t13 = MPI_Wtime();
    // iSend your data, and iRecv from others
    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses = new MPI_Status[numSendProc+numRecvProc];

    //First place all recv requests. Do not recv from self.
    for(int i = 0; i < numRecvProc; i++) {
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));
    }

    //Next send the messages. Do not send to self.
    for(int i = 0; i < numSendProc; i++) {
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));
    }

/*    if (rank==0){
        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
        for(int i=0; i<recvSize; i++)
            std::cout << vecValues[i] << std::endl;
    }*/

//    double t11 = MPI_Wtime();
    // local loop
    // ----------
//    for (unsigned int i = 0; i < nnz_l_local; ++i)
//        w[row_local[i]] += values_local[i] * v[col_local[i] - split[rank]];
//    std::vector< std::vector<double> > w_local( w.size(), std::vector<double>(w.size()) );

    // local loop
    // ----------
//    w.assign(M, 0);
    double* v_p = &v[0] - split[rank];
#pragma omp parallel
    {
        unsigned int i, l, idx;
        int thread_id = omp_get_thread_num();
        double *w_local = w_buff + (thread_id * M);
        if (thread_id == 0)
            w_local = &*w.begin();

        std::fill(&w_local[0], &w_local[M], 0);

#pragma omp for
        for (i = 0; i < nnz_l_local; ++i)
            w_local[row_local[i]] += values_local[i] * v_p[col_local[i]];

        int thread_partner;
        int levels = (int) ceil(log2(num_threads));
        for (l = 0; l < levels; l++) {
            if (thread_id % int(pow(2, l + 1)) == 0) {
                thread_partner = thread_id + int(pow(2, l));
//                printf("l = %d, levels = %d, thread_id = %d, thread_partner = %d \n", l, levels, thread_id, thread_partner);
                if (thread_partner < num_threads) {
                    for (i = 0; i < M; i++)
                        w_local[i] += w_buff[thread_partner * M + i];
                }
            }
#pragma omp barrier
        }

/*
#pragma omp for
        for (i = 0; i < M; ++i)
            for(l = 0; l < num_threads; l++)
                w[i] += w_buff[l*M+i];
*/
    }

// #pragma omp for
//    for (i = 0; i < M; ++i)
//        for(j = 0; j < num_threads; j++)
//            w[i] += w_buff[j*M+i];

//    double t21 = MPI_Wtime();
//    time[1] += (t21-t11);

    // Wait for comm to finish.
    MPI_Waitall(numRecvProc, requests, statuses);

/*    if (rank==1){
        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
        for(int i=0; i<recvSize; i++)
            std::cout << vecValues[i] << std::endl;
    }*/

    // remote loop
    // todo: data race happens during "omp for" here, since the "for" loop splits based on the remote columns, but
    // todo: w[row] are being computed in every iteration , which means different threads may access the same w[row].

//    double t12 = MPI_Wtime();
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

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    delete [] requests;
    delete [] statuses;

//    double t22 = MPI_Wtime();
//    time[2] += (t22-t12);
//    double t23 = MPI_Wtime();
//    time[3] += (t23-t13);

    return 0;
}


int saena_matrix::matvec4(std::vector<double>& v, std::vector<double>& w) {
// todo: to reduce the communication during matvec, consider reducing number of columns during coarsening,
// todo: instead of reducing general non-zeros, since that is what is communicated for matvec.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ){
//        printf("A.M != v.size() in matvec!!!\n");}

//    totalTime = 0;
//    double t10 = MPI_Wtime();

    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i = 0; i < vIndexSize; i++)
        vSend[i] = v[( vIndex[i] )];
//    double t20 = MPI_Wtime();
//    time[0] += (t20-t10);

/*    if (rank==0){
        std::cout << "vIndexSize=" << vIndexSize << ", vSend: rank=" << rank << std::endl;
        for(int i=0; i<vIndexSize; i++)
            std::cout << vSend[i] << std::endl;
    }*/

//    double t13 = MPI_Wtime();
    // iSend your data, and iRecv from others
    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses = new MPI_Status[numSendProc+numRecvProc];

    //First place all recv requests. Do not recv from self.
    for(int i = 0; i < numRecvProc; i++) {
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));
    }

    //Next send the messages. Do not send to self.
    for(int i = 0; i < numSendProc; i++) {
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));
    }

/*    if (rank==0){
        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
        for(int i=0; i<recvSize; i++)
            std::cout << vecValues[i] << std::endl;
    }*/

//    double t11 = MPI_Wtime();
    // local loop
//    std::fill(&*w.begin(), &*w.end(), 0);

    w.assign(w.size(), 0);
//    for (unsigned int i = 0; i < nnz_l_local; ++i)
//        w[row_local[i]] += values_local[i] * v[col_local[i] - split[rank]];

    double* v_p = &v[0] - split[rank];
    unsigned int i;
#pragma omp parallel for default(shared) private(i) reduction(vec_double_plus:w)
    for (i = 0; i < nnz_l_local; ++i)
        w[row_local[i]] += values_local[i] * v_p[col_local[i]];

//    double t21 = MPI_Wtime();
//    time[1] += (t21-t11);

    // Wait for comm to finish.
    MPI_Waitall(numRecvProc, requests, statuses);

/*    if (rank==1){
        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
        for(int i=0; i<recvSize; i++)
            std::cout << vecValues[i] << std::endl;
    }*/

    // remote loop
    // todo: data race happens during "omp for" here, since the "for" loop splits based on the remote columns, but
    // todo: w[row] are being computed in every iteration , which means different threads may access the same w[row].

//    double t12 = MPI_Wtime();
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

    MPI_Waitall(numSendProc, numRecvProc+requests, numRecvProc+statuses);
    delete [] requests;
    delete [] statuses;

//    double t22 = MPI_Wtime();
//    time[2] += (t22-t12);
//    double t23 = MPI_Wtime();
//    time[3] += (t23-t13);

    return 0;
}


int saena_matrix::matvec_timing(std::vector<double>& v, std::vector<double>& w, std::vector<double>& time) {
// todo: to reduce the communication during matvec, consider reducing number of columns during coarsening,
// todo: instead of reducing general non-zeros, since that is what is communicated for matvec.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    if( v.size() != M ){
//        printf("A.M != v.size() in matvec!!!\n");
//        MPI_Finalize();
//        return -1;}

    MPI_Barrier(comm);
    printf("rank %d: matvec step0! vIndexSize = %d \n", rank, vIndexSize);

    double t0_start = omp_get_wtime();
    // put the values of the vector in vSend, for sending to other processors
#pragma omp parallel for
    for(unsigned int i=0;i<vIndexSize;i++){
//        if (rank == 0) printf("%u \t%lu  \t%f\n", i, vIndex[i], v[( vIndex[i] )]);
        vSend[i] = v[( vIndex[i] )];
    }
    double t0_end = omp_get_wtime();// try this: rdtsc for timing

//    printf("rank %d: matvec step1! \n", rank);

/*    if (rank==0){
        std::cout << "vIndexSize=" << vIndexSize << ", vSend: rank=" << rank << std::endl;
        for(int i=0; i<vIndexSize; i++)
            std::cout << vSend[i] << std::endl;
    }*/

    double t3_start = omp_get_wtime();
    // iSend your data, and iRecv from others
    MPI_Request* requests = new MPI_Request[numSendProc+numRecvProc];
    MPI_Status* statuses = new MPI_Status[numSendProc+numRecvProc];

    //First place all recv requests. Do not recv from self.
    for(int i = 0; i < numRecvProc; i++) {
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));
    }

    //Next send the messages. Do not send to self.
    for(int i = 0; i < numSendProc; i++) {
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));
    }

//    printf("rank %d: matvec step2! \n", rank);

/*    if (rank==0){
        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
        for(int i=0; i<recvSize; i++)
            std::cout << vecValues[i] << std::endl;
    }*/

    double t1_start = omp_get_wtime();
    double* v_p = &v[0] - split[rank];
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

//    printf("rank %d: matvec step3! \n", rank);

    // Wait for the communication to finish.
//    double t4_start = omp_get_wtime();
//    MPI_Waitall(numSendProc+numRecvProc, requests, statuses);
    MPI_Waitall(numRecvProc, requests, statuses);
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

//    printf("rank %d: matvec step4! \n", rank);

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

//    double time4_local = t4_end-t4_start;
//    double time4;
//    MPI_Allreduce(&time4_local, &time4, 1, MPI_DOUBLE, MPI_SUM, comm);

    time[0] += time0/nprocs;
    time[1] += time1/nprocs;
    time[2] += time2/nprocs;
    time[3] += time3/nprocs;
//    time[4] += time4/nprocs;

//    time[0] += time0_local;
//    time[1] += time1_local;
//    time[2] += time2_local;
//    time[3] += time3_local;
//    time[4] += time4_local;

    return 0;
}


int saena_matrix::matvec_timing_alltoall(std::vector<double>& v, std::vector<double>& w, std::vector<double>& time) {
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
    double* v_p = &v[0] - split[rank];
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


int saena_matrix::matvec_timing2(std::vector<double>& v, std::vector<double>& w, std::vector<double>& time) {
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

    //First place all recv requests. Do not recv from self.
    for(int i = 0; i < numRecvProc; i++) {
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));
    }

    //Next send the messages. Do not send to self.
    for(int i = 0; i < numSendProc; i++) {
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));
    }

/*    if (rank==0){
        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
        for(int i=0; i<recvSize; i++)
            std::cout << vecValues[i] << std::endl;
    }*/

    double t1_start = omp_get_wtime();
    double* v_p = &v[0] - split[rank];
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

    // Wait for the communication to finish.
    double t4_start = omp_get_wtime();
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


int saena_matrix::matvec_timing3(std::vector<double>& v, std::vector<double>& w, std::vector<double>& time) {
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

    //First place all recv requests. Do not recv from self.
    for(int i = 0; i < numRecvProc; i++) {
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));
    }

    //Next send the messages. Do not send to self.
    for(int i = 0; i < numSendProc; i++) {
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));
    }

/*    if (rank==0){
        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
        for(int i=0; i<recvSize; i++)
            std::cout << vecValues[i] << std::endl;
    }*/

    // by doing this you will have a local index for v[col_local[i]].

    double t1_start = omp_get_wtime();

    // local loop
    // ----------
//    w.assign(M, 0);
    double* v_p = &v[0] - split[rank];
//    unsigned int i, l, idx;
#pragma omp parallel
    {
        unsigned int i, l, idx;
        int thread_id = omp_get_thread_num();
        double *w_local = w_buff + (thread_id*M);
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

/*
#pragma omp for
        for (i = 0; i < M; ++i)
            for(l = 0; l < num_threads; l++)
                w[i] += w_buff[l*M+i];
*/
        }

/*
        idx = 0;
#pragma omp for
        for(k = 0; k < num_threads; k++) {
            for (i = 0; i < M; ++i)
                w[i] += w_buff[idx++];
        }
*/
/*
        idx = 0;
        for(k = 0; k < num_threads; k++) {
#pragma omp for
            for (i = 0; i < M; ++i)
                w[i] += w_buff[idx++];
        }
*/
    } // end of openmp

/*
    unsigned idx1, idx2, thread_per_level;
    int levels = (int)ceil(log2(num_threads));
//    printf("levels = %d \n", (int)log2(num_threads));
    for (l = 0; l < levels; l++){
        thread_per_level = (unsigned)ceil(num_threads/pow(2, l+1));
        for(k = 0; k < thread_per_level; k++){
            idx1 = (k*int(pow(2, l+1))) * M;
            idx2 = idx1 + int(pow(2, l)) * M;
            if(idx2 >= M*num_threads)
                break;
//            printf("l = %u, k = %u, t = %d, t_partner = %d \n", l, k, idx1/M, idx2/M);
#pragma omp parallel for
            for (i = 0; i < M; ++i)
                w_buff[ idx1 + i ] += w_buff[ idx2 + i ];
        }
    }

    //todo: improve this part
#pragma omp parallel for
    for (i = 0; i < M; ++i)
        w[i] = w_buff[i];
*/

/*
    unsigned idx = 0;
    for(k = 0; k < num_threads; k++) {
        for (i = 0; i < M; ++i)
            w[i] += w_buff[idx++];
    }
*/
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


int saena_matrix::matvec_timing4(std::vector<double>& v, std::vector<double>& w, std::vector<double>& time) {
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

    //First place all recv requests. Do not recv from self.
    for(int i = 0; i < numRecvProc; i++) {
        MPI_Irecv(&vecValues[rdispls[recvProcRank[i]]], recvProcCount[i], MPI_DOUBLE, recvProcRank[i], 1, comm, &(requests[i]));
    }

    //Next send the messages. Do not send to self.
    for(int i = 0; i < numSendProc; i++) {
        MPI_Isend(&vSend[vdispls[sendProcRank[i]]], sendProcCount[i], MPI_DOUBLE, sendProcRank[i], 1, comm, &(requests[numRecvProc+i]));
    }

/*    if (rank==0){
        std::cout << "recvSize=" << recvSize << ", vecValues: rank=" << rank << std::endl;
        for(int i=0; i<recvSize; i++)
            std::cout << vecValues[i] << std::endl;
    }*/

    double t1_start = omp_get_wtime();

    // local loop
    // ----------
    w.assign(w.size(), 0);
//    for (unsigned int i = 0; i < nnz_l_local; ++i)
//        w[row_local[i]] += values_local[i] * v[col_local[i] - split[rank]];

    double* v_p = &v[0] - split[rank];
    unsigned int i;
#pragma omp parallel for default(shared) private(i) reduction(vec_double_plus:w)
    for (i = 0; i < nnz_l_local; ++i)
        w[row_local[i]] += values_local[i] * v_p[col_local[i]];

    double t1_end = omp_get_wtime();

    // Wait for the communication to finish.
    double t4_start = omp_get_wtime();
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


int saena_matrix::residual(std::vector<double>& u, std::vector<double>& rhs, std::vector<double>& res){
    // Vector res = A*u - rhs;

//    int nprocs, rank;
//    MPI_Comm_size(comm, &nprocs);
//    MPI_Comm_rank(comm, &rank);

    std::vector<double> matvecTemp(M);
    matvec(u, matvecTemp);
//    if(rank==1)
//        for(long i=0; i<matvecTemp.size(); i++)
//            std::cout << matvecTemp[i] << std::endl;

    for(long i=0; i<M; i++)
        res[i] = matvecTemp[i] - rhs[i];
//    if(rank==1)
//        for(long i=0; i<res.size(); i++)
//            std::cout << res[i] << std::endl;

    return 0;
}


int saena_matrix::inverse_diag(std::vector<double>& x) {
    int nprocs, rank;
//    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    for(unsigned int i=0; i<nnz_l; i++){
//        if(rank==4) printf("%u \t%lu \t%lu \t%f \n", i, entry[i].row, entry[i].col, entry[i].val);

        if(entry[i].row == entry[i].col){
            if(entry[i].val != 0)
                x[entry[i].row-split[rank]] = 1/entry[i].val;
            else{
                // there is no zero entry in the matrix (sparse), but just to be sure this part is added.
                if(rank==0) printf("Error: there is a zero diagonal element (at row index = %lu)\n", entry[i].row);
                MPI_Finalize();
                return -1;
            }
        }
    }
    return 0;
}


int saena_matrix::jacobi(int iter, std::vector<double>& u, std::vector<double>& rhs, std::vector<double>& temp) {

// Ax = rhs
// u = u - (D^(-1))(Ax - rhs)
// 1. B.matvec(u, one) --> put the value of matvec in one.
// 2. two = one - rhs
// 3. three = inverseDiag * two * omega
// 4. four = u - three

    unsigned int i, j;

//    int rank;
//    MPI_Comm_rank(comm, &rank);

    for(j = 0; j < iter; j++){
//        printf("jacobi iter = %u \n", j);
//    MPI_Barrier(comm);
//    double t1 = MPI_Wtime();

        matvec(u, temp);

//    double t2 = MPI_Wtime();
//    print_time(t1, t2, "jacobi matvec time:", comm);

//    MPI_Barrier(comm);
//    t1 = MPI_Wtime();

#pragma omp parallel for
        for(i=0; i<M; i++){
            temp[i] -= rhs[i];
            temp[i] *= invDiag[i] * jacobi_omega;
            u[i] -= temp[i];
        }

//    t2 = MPI_Wtime();
//    print_time(t1, t2, "jacobi forloop time:", comm);
    }

    return 0;
}


int saena_matrix::find_eig() {

/*
    int argc = 0;
    char** argv = {NULL};
//    El::Environment env( argc, argv );
    El::Initialize( argc, argv );

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    const El::Int n = Mbig;

    // *************************** serial ***************************

//    El::Matrix<double> A(n,n);
//    El::Zero( A );
//    for(unsigned long i = 0; i<nnz_l; i++)
//        A(entry[i].row, entry[i].col) = entry[i].val * invDiag[entry[i].row];

//    El::Print( A, "\nGlobal Elemental matrix (serial):\n" );

//    El::Matrix<El::Complex<double>> w(n,1);

    // *************************** parallel ***************************

    El::DistMatrix<double> A(n,n);
    El::Zero( A );
    A.Reserve(nnz_l);
    for(unsigned long i = 0; i < nnz_l; i++){
//        if(rank==1) printf("%lu \t%lu \t%f \t%lu \t%f \n", entry[i].row, entry[i].col, entry[i].val, entry[i].row - split[rank], invDiag[entry[i].row - split[rank]]);
        A.QueueUpdate(entry[i].row, entry[i].col, entry[i].val * invDiag[entry[i].row - split[rank]]);
    }
    A.ProcessQueues();
//    El::Print( A, "\nGlobal Elemental matrix:\n" );

    El::DistMatrix<El::Complex<double>> w(n,1);

    // *************************** common part between serial and parallel ***************************

    El::SchurCtrl<double> schurCtrl;
    schurCtrl.time = false;
//    schurCtrl.hessSchurCtrl.progress = true;
//    El::Schur( A, w, V, schurCtrl ); //  eigenvectors will be saved in V.

//    printf("before Schur!\n");
    El::Schur( A, w, schurCtrl ); // eigenvalues will be saved in w.
//    printf("after Schur!\n");
//    MPI_Barrier(comm); El::Print( w, "eigenvalues:" ); MPI_Barrier(comm);

    eig_max_diagxA = w.Get(0,0).real();
    for(unsigned long i = 1; i < n; i++)
        if(w.Get(i,0).real() > eig_max_diagxA)
            eig_max_diagxA = w.Get(i,0).real();
*/



//    if(rank==0) printf("eig_max = %f \n", eig_max_diagxA);

/*
    // parallel (draft)
    const El::Grid grid(comm, nprocs);
//    const El::Grid grid(comm, nprocs, El::ROW_MAJOR);
//    printf("rank = %d, Row = %d, Col = %d \n", rank, grid.Row(), grid.Col());


//    El::DistMatrix<double> A(n, n, grid);
//    El::Zero( A );
    El::SetDefaultBlockHeight(M);
    El::SetDefaultBlockWidth(Mbig);
    El::DistMatrix<double,El::VC, El::STAR, El::BLOCK> B(n, n, grid);

//    printf("rank = %d, BlockHeight = %d, BlockWidth = %d, ColCut = %d, RowCut = %d \n", rank, B.BlockHeight(), B.BlockWidth(), B.ColCut(), B.RowCut());
//    printf("rank = %d, LocalRowOffset[1] = %d, GlobalRow[1] = %d, GlobalCol[40] = %d, DefaultBlockHeight = %d \n", rank, B.LocalRowOffset(1), B.GlobalRow(1), B.GlobalCol(40), El::DefaultBlockHeight());

//    bool colMajor = true;
//    const El::GridOrder order = ( colMajor ? El::COLUMN_MAJOR : ROW_MAJOR );

    auto& C = B.Matrix();
    C.Resize( M, Mbig ); // change the submatrices' sizes to Saena's sizes.
    El::Zero( C );
//    printf("rank = %d, C.Height() = %d, C.Width() = %d, B.LocalRowOffset(5) = %d \n", rank, C.Height(), C.Width(), B.LocalRowOffset(5));
//    El::Matrix<double> C(M, Mbig);

//    const El::Int localHeight = A.LocalHeight();
//    const El::Int localWidth  = A.LocalWidth();
//    printf("rank = %d, localHeight = %d, localWidth = %d \n", rank, localHeight, localWidth);

//    long iter = 0;
//    for( El::Int jLoc=0; jLoc<localWidth; ++jLoc )
//        for( El::Int iLoc=0; iLoc<localHeight; ++iLoc ){
//            if(rank==1) ALoc(iLoc,jLoc) = rank*1000 + iter;
//            A.Set(iLoc, jLoc, rank*1000 + iter);
//            iter++;
//            ALoc(iLoc,jLoc) = iLoc+split[rank] + jLoc * localHeight;
//        }

//    for(unsigned long i = 0; i<nnz_l; i++){
//        if(rank==1) std::cout << entry[i].row - split[rank] << "\t" << entry[i].col << "\t" << entry[i].val << std::endl;
//        C(entry[i].row - split[rank], entry[i].col) = entry[i].val;
//    }

//    MPI_Barrier(comm);
//    if(rank==0) El::Print( C, "\nLocal Elemental matrix:\n" );
//    MPI_Barrier(comm);
//    if(rank==1) El::Print( C, "\nLocal Elemental matrix:\n" );

//    El::DistMatrix<double> E(B);
//    MPI_Barrier(comm);
//    El::Print( E, "\nGlobal Elemental matrix:\n" );
*/

//    El::Finalize();
    return 0;
}


int saena_matrix::chebyshev(int iter, std::vector<double>& u, std::vector<double>& rhs, std::vector<double>& res, std::vector<double>& d){

    int rank;
    MPI_Comm_rank(comm, &rank);

    unsigned long i;
    double alpha = 0.25 * eig_max_diagxA;
    double beta = eig_max_diagxA;
    double delta = (beta - alpha)/2;
    double theta = (beta + alpha)/2;
    double s1 = theta/delta;
    double rhok = 1/s1;
    double rhokp1, d1, d2;

    // first loop
    residual(u, rhs, res);
#pragma omp parallel for
    for(i = 0; i < u.size(); i++){
        d[i] = (-res[i] * invDiag[i]) / theta;
        u[i] += d[i];
//        if(rank==0) printf("invDiag[%lu] = %f, \tres[%lu] = %f, \td[%lu] = %f, \tu[%lu] = %f \n",
//                           i, invDiag[i], i, res[i], i, d[i], i, u[i]);
    }

    for( i = 1; i < iter; i++){
        rhokp1 = 1 / (2*s1 - rhok);
        d1     = rhokp1 * rhok;
        d2     = 2*rhokp1 / delta;
        rhok   = rhokp1;
        residual(u, rhs, res);
#pragma omp parallel for
        for(unsigned long j = 0; j < u.size(); j++){
            d[j] = ( d1 * d[j] ) + ( d2 * (-res[j]) * invDiag[j] );
            u[j] += d[j];
//        if(rank==0) printf("u[%lu] = %f \n", j, u[j]);
        }
    }

    return 0;
}
