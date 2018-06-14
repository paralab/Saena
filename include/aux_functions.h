#ifndef SAENA_AUXFUNCTIONS_H
#define SAENA_AUXFUNCTIONS_H

#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <cmath>
#include <mpi.h>


typedef unsigned int index_t;
typedef unsigned long nnz_t;
typedef double value_t;

class strength_matrix;
class saena_matrix;

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
        // when using on split, some procs have equal split value (M=0), so go to the next proc until M != 0.
//        while(*left == *(left+1))
//            left++;

        return std::distance(first, left);
    }
    else
        return std::distance(first, left-1);
}

// binary search tree using the lower bound
// difference with lower_bound2: in case of a val equal to one of vector entries:
// lower_bound2 returns the most left one
// lower_bound3 returns the most right one
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

/*
// binary search tree using the upper bound
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


int randomVector(std::vector<unsigned long>& V, long size, strength_matrix* S, MPI_Comm comm);

int randomVector2(std::vector<double>& V);

int randomVector3(std::vector<unsigned long>& V, long size, strength_matrix* S, MPI_Comm comm);

int randomVector4(std::vector<unsigned long>& V, long size);


// the order of this class is called "Column-major order"
class cooEntry{
public:
    index_t row;
    index_t col;
    value_t val;

    cooEntry(){}

    cooEntry(index_t i, index_t j, value_t v){
        row = i;
        col = j;
        val = v;
    }

    bool operator == (const cooEntry& node2) const
    {
        return (row == node2.row && col == node2.col);
    }

    bool operator < (const cooEntry& node2) const
    {
        if(col < node2.col)
            return (true);
        else if(col == node2.col)
            return(row < node2.row);
        else
            return false;
    }

    bool operator <= (const cooEntry& node2) const
    {
        if(col < node2.col)
            return (true);
        else if(col == node2.col)
            return(row <= node2.row);
        else
            return false;
    }

    bool operator > (const cooEntry& node2) const
    {
        if(col > node2.col)
            return (true);
        else if(col == node2.col)
            return(row > node2.row);
        else
            return false;
    }

    bool operator >= (const cooEntry& node2) const
    {
        if(col > node2.col)
            return (true);
        else if(col == node2.col)
            return(row >= node2.row);
        else
            return false;
    }

    static MPI_Datatype mpi_datatype()
    {
        static bool         first = true;
        static MPI_Datatype datatype;

        if (first)
        {
            first = false;
            MPI_Type_contiguous(sizeof(cooEntry), MPI_BYTE, &datatype);
            MPI_Type_commit(&datatype);
        }

        return datatype;
    }
};


std::ostream & operator<<(std::ostream & stream, const cooEntry & item);


bool row_major (const cooEntry& node1, const cooEntry& node2);


// the order of this class is called "Row-major order"
class cooEntry_row{
public:
    index_t row;
    index_t col;
    value_t val;

    cooEntry_row(){}

    cooEntry_row(index_t i, index_t j, value_t v){
        row = i;
        col = j;
        val = v;
    }

    bool operator == (const cooEntry& node2) const
    {
        return (row == node2.row && col == node2.col);
    }

    bool operator < (const cooEntry& node2) const
    {
        if(row < node2.row)
            return (true);
        else if(row == node2.row)
            return( col < node2.col);
        else
            return false;
    }

    bool operator <= (const cooEntry& node2) const
    {
        if(row < node2.row)
            return (true);
        else if(row == node2.row)
            return( col <= node2.col);
        else
            return false;
    }

    bool operator > (const cooEntry& node2) const
    {
        if(row > node2.row)
            return (true);
        else if(row == node2.row)
            return( col > node2.col);
        else
            return false;
    }

    bool operator >= (const cooEntry& node2) const
    {
        if(  row > node2.row)
            return (true);
        else if(row == node2.row)
            return( col >= node2.col);
        else
            return false;
    }

    static MPI_Datatype mpi_datatype()
    {
        static bool         first = true;
        static MPI_Datatype datatype;

        if (first)
        {
            first = false;
            MPI_Type_contiguous(sizeof(cooEntry), MPI_BYTE, &datatype);
            MPI_Type_commit(&datatype);
        }

        return datatype;
    }
};

//template <class T>
//float myNorm(std::vector<T>& v);

//double myNorm(std::vector<double>& v);


/*
//template <typename cooEntry>
vector<cooEntry> sort_indices3(const vector<cooEntry>& v){

//     initialize original index locations
        vector<cooEntry> idx(v.size());
        for(unsigned long i=0; i<v.size(); i++)
            idx[i].row = i;
//    iota(idx.begin(), idx.end(), 0);

//     sort indexes based on comparing values in v
        std::sort(idx.begin(), idx.end(), [&v] (size_t i, size_t j) -> bool {return v[i].row < v[j].row;});

        return idx;
}
*/

class sort_indices2
{
private:
    cooEntry* mparr;
public:
    sort_indices2(cooEntry* parr) : mparr(parr) {}
    bool operator()(index_t i, index_t j) const { return mparr[i].row < mparr[j].row; }
};


/*
template <typename T>
vector<size_t> sort_indices5(const vector<T> &v) {

    // initialize original index locations
    vector<size_t> idx(v.size());
//    iota(idx.begin(), idx.end(), 0);
    for(unsigned long i=0; i<v.size(); i++)
        idx[i] = i;

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) ->bool {return v[i1] < v[i2];});

    return idx;
}
*/


void setIJV(char* file_name, index_t* I,index_t* J, value_t* V, nnz_t nnz_g, nnz_t initial_nnz_l, MPI_Comm comm);


int dotProduct(std::vector<value_t>& r, std::vector<value_t>& s, double* dot, MPI_Comm comm);


double print_time(double t1, double t2, std::string function_name, MPI_Comm comm);


int print_time_average(double t1, double t2, std::string function_name, int iter, MPI_Comm comm);


int writeVectorToFiled(std::vector<value_t>& v, index_t vSize, std::string name, MPI_Comm comm);


int generate_rhs(std::vector<value_t>& rhs, index_t mx, index_t my, index_t mz, MPI_Comm comm);


int generate_rhs_old(std::vector<value_t>& rhs);


// returns true if val is less than the machine precision for data type T, which means it is almost zero.
template<class T>
bool almost_zero(T val){
//    return (fabs(val) < std::numeric_limits<T>::min());
    return (fabs(val) < 1e-12);
}

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
            }
            MPI_Barrier(comm);
        }
    }

    return 0;
}

#endif //SAENA_AUXFUNCTIONS_H
