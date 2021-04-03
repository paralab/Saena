#ifndef SAENA_AUXFUNCTIONS_H
#define SAENA_AUXFUNCTIONS_H

#include "data_struct.h"

class strength_matrix;
class saena_matrix;


// returns true if val is less than the threshold for data type T, which means it is almost zero.
template<class T>
bool inline almost_zero(const T &val){
//    return (fabs(val) < std::numeric_limits<T>::min());
    return (fabs(val) < ALMOST_ZERO);
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
//        std::cout << "left = " << *left << ", middle = " << *middle << ", right = " << *right << ", val = " << val << std::endl;
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


inline void dotProduct(const value_t *r, const value_t *s, const index_t sz, value_t* dot, MPI_Comm comm){
    value_t dot_l = 0;
    for(index_t i = 0; i < sz; ++i)
        dot_l += r[i] * s[i];
    MPI_Allreduce(&dot_l, dot, 1, MPI_DOUBLE, MPI_SUM, comm);
//    MPI_Allreduce(&dot_l, dot, 1, par::Mpi_datatype<value_t>::value(), MPI_SUM, comm);
}


int pnorm(std::vector<value_t>& r, value_t &norm, MPI_Comm comm);
value_t pnorm(std::vector<value_t>& r, MPI_Comm comm);

double print_time(double t_start, double t_end, const std::string &function_name, MPI_Comm comm);
double print_time(double t_diff,      const std::string &function_name, MPI_Comm comm, bool print_time = false, bool print_name = true, int optype = 0);
double print_time_all(double t_diff,  const std::string &function_name, MPI_Comm comm);
double print_time_ave(double t_diff,  const std::string &function_name, MPI_Comm comm, bool print_time = false, bool print_name = true);
double print_time_ave2(double t_diff, const std::string &function_name, MPI_Comm comm, bool print_time = false, bool print_name = true);
double print_time_ave_consecutive(double t_diff, MPI_Comm comm);

double average_iter(index_t iter, MPI_Comm comm);

template<class T>
int print_vector(const std::vector<T> &v, const int ran, const std::string &name, MPI_Comm comm){
    // if ran >= 0 print the vector elements on proc with rank = ran
    // otherwise print the vector elements on all processors in order. (first on proc 0, then proc 1 and so on.)

    int rank = -1, nprocs = -1;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::stringstream buffer;
    index_t iter = 0;
    if(ran >= 0) {
        if (rank == ran) {
            printf("\n%s on proc = %d, size = %ld: \n", name.c_str(), ran, v.size());
            for (const auto &i : v) {
                buffer << iter << "\t" << std::setprecision(16) << i;
                std::cout << buffer.str() << std::endl;
                buffer.str("");
                iter++;
            }
            printf("\n");
        }
    } else{
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("\n%s on proc = %d, size = %ld: \n", name.c_str(), proc, v.size());
                for (const auto &i : v) {
                    buffer << iter << "\t" << std::setprecision(16) << i;
                    std::cout << buffer.str() << std::endl;
                    buffer.str("");
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


int write_agg(std::vector<unsigned long>& v, std::string name, int level, MPI_Comm comm);


int generate_rhs(std::vector<value_t>& rhs, index_t mx, index_t my, index_t mz, MPI_Comm comm);

int generate_rhs_old(std::vector<value_t>& rhs);


class decrement
{
private:
    index_t num;
public:
    decrement(index_t n) : num(n) {  }

    // This operator overloading enables calling
    // operator function () on objects of decrement
    index_t operator () (index_t arr_num) const {
        return num - arr_num;
    }
};

class increment
{
private:
    index_t num;
public:
    increment(index_t n) : num(n) {  }

    // This operator overloading enables calling
    // operator function () on objects of increment
    index_t operator () (index_t arr_num) const {
        return num + arr_num;
    }
};


int read_from_file_rhs(std::vector<value_t>& v, const std::vector<index_t>& split, char *file, MPI_Comm comm);

template <class T>
int write_to_file_vec(std::vector<T>& v, const std::string &name, MPI_Comm comm) {

    // Create txt files with name name0.txt for processor 0, name1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat name0.txt name1.txt > output.txt
    // The file will be saved in the same folder as the executive file.

    int rank;
    MPI_Comm_rank(comm, &rank);

    std::string outFileNameTxt = "./";
    outFileNameTxt += name;
    outFileNameTxt += std::to_string(rank);
    outFileNameTxt += ".txt";

    std::ofstream outFileTxt(outFileNameTxt);

    // write the size
    if (!rank){
        outFileTxt << v.size() << std::endl;
    }

    outFileTxt << std::setprecision(std::numeric_limits<long double>::digits10 + 1);

    // write the vector values
    for (auto i:v) {
        outFileTxt << i << std::endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

    return 0;
}


#endif //SAENA_AUXFUNCTIONS_H