#ifndef SAENA_AUXFUNCTIONS_H
#define SAENA_AUXFUNCTIONS_H

#include <data_struct.h>

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <mpi.h>


class strength_matrix;
class saena_matrix;


// returns true if val is less than the machine precision for data type T, which means it is almost zero.
template<class T>
bool almost_zero(T val){
//    return (fabs(val) < std::numeric_limits<T>::min());
    return (fabs(val) < 1e-12);
}


// sort indices and store the ordering.
class sort_indices
{
private:
    index_t *mparr;
public:
    explicit sort_indices(index_t *parr) : mparr(parr) {}
    bool operator()(index_t i, index_t j) const { return mparr[i]<mparr[j]; }
};

class sort_indices2
{
private:
    cooEntry* mparr;
public:
    explicit sort_indices2(cooEntry* parr) : mparr(parr) {}
    bool operator()(index_t i, index_t j) const { return mparr[i].row < mparr[j].row; }
};


// binary search tree using the lower bound
template <class T>
long lower_bound2(T *left, T *right, T val){
    T* first = left;
    while (left < right) {
        T *middle = left + (right - left) / 2;

        if (*middle < val)
            left = middle + 1;
        else
            right = middle;
    }

    if(val == *left){
        return std::distance(first, left);
    }
    else
        return std::distance(first, left-1);
}

// binary search tree using the lower bound
// difference with lower_bound2: in case of a val equal to one of vector entries:
// lower_bound2 returns the most left one
// lower_bound3 returns the most right one
// todo: is this std::upper_bound? change std::upper_bound just like how it is done for lower_bound2.
template <class T>
long lower_bound3(T *left, T *right, T val){
    T* first = left;
    while (left < right) {
        T *middle = right - (right - left) / 2;

        if (*middle < val)
            left = middle;
        else{
            if(*middle == val)
                left = middle;
            else
                right = middle-1;
        }
//        std::cout << "left = " << *left << ", middle = " << *(right - (right - left) / 2) << ", right = " << *right << ", val = " << val << std::endl;
    }

    if(val >= *right)
        return std::distance(first, left);
    else
        return std::distance(first, left-1);
}

// binary search tree using the upper bound
/*
template <class T>
long upper_bound2(T *left, T *right, T val){
    T* first = left;
    while (left < right) {
        T *middle = right - (right - left) / 2;
        std::cout << "left = " << *left << ", middle = " << *middle << ", right = " << *right << ", val = " << val << std::endl;
        if (*middle <= val)
            left = middle;
        else
            right = middle-1;
    }

    if(val == *right){
        // when using on split, some procs have equal split value (M=0), so go to the next proc until M != 0.
//        while(*left == *(left+1))
//            left++;

        return std::distance(first, right);
    }
    else
        return std::distance(first, right+1);
}
*/


void setIJV(char* file_name, index_t* I,index_t* J, value_t* V, nnz_t nnz_g, nnz_t initial_nnz_l, MPI_Comm comm);


int dotProduct(std::vector<value_t>& r, std::vector<value_t>& s, double* dot, MPI_Comm comm);

int pnorm(std::vector<value_t>& r, value_t &norm, MPI_Comm comm);
value_t pnorm(std::vector<value_t>& r, MPI_Comm comm);

double print_time(double t_start, double t_end, std::string function_name, MPI_Comm comm);

double print_time(double t_diff, std::string function_name, MPI_Comm comm);

double print_time_ave(double t_diff, std::string function_name, MPI_Comm comm);

double print_time_ave_consecutive(double t_diff, MPI_Comm comm);

template<class T>
int print_vector(const std::vector<T> &v, const int ran, const std::string &name, MPI_Comm comm){
    // if ran >= 0 print the vector elements on proc with rank = ran
    // otherwise print the vector elements on all processors in order. (first on proc 0, then proc 1 and so on.)

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    index_t iter = 0;
    if(ran >= 0) {
        if (rank == ran) {
            printf("\n%s on proc = %d, size = %ld: \n", name.c_str(), ran, v.size());
            for (auto i:v) {
                std::cout << iter << "\t" << i << std::endl;
                iter++;
            }
            printf("\n");
        }
    } else{
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("\n%s on proc = %d, size = %ld: \n", name.c_str(), proc, v.size());
                for (auto i:v) {
                    std::cout << iter << "\t" << i << std::endl;
                    iter++;
                }
                printf("\n");
            }
            MPI_Barrier(comm);
        }
    }

    return 0;
}

template<class T>
int print_agg(const std::vector<T> &v, const int ran, const std::string &name, MPI_Comm comm){
    // if ran >= 0 print the vector elements on proc with rank = ran
    // otherwise print the vector elements on all processors in order. (first on proc 0, then proc 1 and so on.)

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    index_t iter = 0;
    if(ran >= 0) {
        if (rank == ran) {
            printf("\n%s on proc = %d, size = %ld: \n", name.c_str(), ran, v.size());
            for (auto i:v) {
                std::cout << i << std::endl;
                iter++;
            }
            printf("\n");
        }
    } else{
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("\n%s on proc = %d, size = %ld: \n", name.c_str(), proc, v.size());
                for (auto i:v) {
                    std::cout << i << std::endl;
                    iter++;
                }
                printf("\n");
            }
            MPI_Barrier(comm);
        }
    }

    return 0;
}


template<class T>
int print_array(const T &v, const nnz_t sz, const int ran, const std::string &name, MPI_Comm comm){
    // if ran >= 0 print the array elements on proc with rank = ran
    // otherwise print the array elements on all processors in order. (first on proc 0, then proc 1 and so on.)

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    index_t iter = 0;
    if(ran >= 0) {
        if (rank == ran) {
            printf("\n%s on proc = %d, size = %ld: \n", name.c_str(), ran, sz);
            for (index_t i = 0; i < sz; i++) {
                std::cout << iter << "\t" << v[i] << std::endl;
                iter++;
            }
            printf("\n");
        }
    } else{
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("\n%s on proc = %d, size = %ld: \n", name.c_str(), proc, sz);
                for (index_t i = 0; i < sz; i++) {
                    std::cout << iter << "\t" << v[i] << std::endl;
                    iter++;
                }
                printf("\n");
            }
            MPI_Barrier(comm);
        }
    }

    return 0;
}


int read_vector_file(std::vector<value_t>& v, saena_matrix *A, char *file, MPI_Comm comm);

int write_vector_file_d(std::vector<value_t>& v, index_t vSize, std::string name, MPI_Comm comm);

int write_agg(std::vector<unsigned long>& v, std::string name, int level, MPI_Comm comm);


int generate_rhs(std::vector<value_t>& rhs, index_t mx, index_t my, index_t mz, MPI_Comm comm);

int generate_rhs_old(std::vector<value_t>& rhs);


#endif //SAENA_AUXFUNCTIONS_H