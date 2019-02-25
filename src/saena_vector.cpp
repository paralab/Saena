#include <set>
#include "saena_vector.h"
#include "parUtils.h"


saena_vector::saena_vector() = default;

saena_vector::saena_vector(MPI_Comm com) {
    comm = com;
}

void saena_vector::set_comm(MPI_Comm com){
    comm = com;
}

saena_vector::~saena_vector() = default;


int saena_vector::set_rep_dup(index_t row, value_t val){

//    if(fabs(val) > 1e-14){
//        entry.emplace_back(row, val);
//    }

    vecEntry temp_new = vecEntry(row, val);
    std::pair<std::set<vecEntry>::iterator, bool> p = data_set.insert(temp_new);

    if (!p.second){
        auto hint = p.first; // hint is std::set<cooEntry>::iterator
        hint++;
        data_set.erase(p.first);
        // in the case of duplicate, if the new value is zero, remove the older one and don't insert the zero.
        if(!almost_zero(val))
            data_set.insert(hint, temp_new);
    }

    // if the entry is zero and it was not a duplicate, just erase it.
    if(p.second && almost_zero(val))
        data_set.erase(p.first);

    return 0;
}

int saena_vector::set_add_dup(index_t row, value_t val){

    // if there are duplicates with different values on two different processors, what should happen?
    // which one should be removed? We do it randomly.

    vecEntry temp_old;
    vecEntry temp_new = vecEntry(row, val);

    std::pair<std::set<vecEntry>::iterator, bool> p = data_set.insert(temp_new);

    if (!p.second){
        temp_old = *(p.first);
        temp_new.val += temp_old.val;

//        std::set<cooEntry_row>::iterator hint = p.first;
        auto hint = p.first;
        hint++;
        data_set.erase(p.first);
        data_set.insert(hint, temp_new);
    }

    return 0;
}


int saena_vector::remove_duplicates() {
    // parameters needed for this function:

    // parameters being set in this function:

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::vector<vecEntry> data_unsorted(data_set.begin(), data_set.end());

    std::vector<vecEntry> data_sorted_dup;
    par::sampleSort(data_unsorted, data_sorted_dup, comm);

//    print_vector(data_sorted_dup, -1, "data_sorted_dup", comm);

    // clear data_unsorted and free memory.
    data_unsorted.clear();
    data_unsorted.shrink_to_fit();

    if(data_sorted_dup.empty()) {
        printf("error: data_sorted_dup of the vector has no element on process %d! \n", rank);
        MPI_Finalize();
        return -1;}

    // size of data may be smaller because of duplicates. In that case its size will be reduced after finding the exact size.
    data.resize(data_sorted_dup.size());

    // put the first element of data_unsorted to data.
    nnz_t data_size = 0;
    if(!data_sorted_dup.empty()){
        data[0] = data_sorted_dup[0];
        data_size++;
    }

    if(add_duplicates){
        for(nnz_t i = 1; i < data_sorted_dup.size(); i++) {
            if (data_sorted_dup[i] == data_sorted_dup[i - 1]) {
                data[data_size - 1].val += data_sorted_dup[i].val;
            } else {
                data[data_size] = data_sorted_dup[i];
                data_size++;
            }
        }
    } else {
        for(nnz_t i = 1; i < data_sorted_dup.size(); i++){
            if(data_sorted_dup[i] == data_sorted_dup[i - 1]){
                data[data_size - 1] = data_sorted_dup[i];
            }else{
                data[data_size] = data_sorted_dup[i];
                data_size++;
            }
        }
    }

    data.resize(data_size);
    data.shrink_to_fit();

//    print_vector(data, -1, "data", comm);

    // receive first element of your left neighbor and check if it is equal to your last element.
    vecEntry first_element_neighbor;
    if(rank != nprocs-1)
        MPI_Recv(&first_element_neighbor, 1, vecEntry::mpi_datatype(), rank+1, 0, comm, MPI_STATUS_IGNORE);

    if(rank!= 0)
        MPI_Send(&data[0], 1, vecEntry::mpi_datatype(), rank-1, 0, comm);

    vecEntry last_element = data.back();
    if(rank != nprocs-1){
        if(last_element == first_element_neighbor) {
            data.pop_back();
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

        data[0].val += left_neighbor_last_val;
    }

//    print_vector(data, -1, "final data", comm);

    return 0;
}

int saena_vector::assemble(){

    remove_duplicates();

    return 0;
}


int saena_vector::get_vec(std::vector<double> &vec){

    vec.resize(data.size());
    for(index_t i = 0; i < data.size(); i++){
        vec[i] = data[i].val;
    }

    return 0;
}
