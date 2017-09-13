//
// Created by abaris on 4/28/17.
//

#ifndef SAENA_AUXFUNCTIONS_H
#define SAENA_AUXFUNCTIONS_H

#include <algorithm>
#include <mpi.h>
#include "strength_matrix.h"


// sort indices and store the ordering.
class sort_indices
{
private:
    unsigned long* mparr;
public:
    sort_indices(unsigned long* parr) : mparr(parr) {}
    bool operator()(unsigned long i, unsigned long j) const { return mparr[i]<mparr[j]; }
};


// binary search tree using the lower bound
template <class T>
T lower_bound2(T *left, T *right, T val);


int randomVector(std::vector<unsigned long>& V, long size, strength_matrix* S, MPI_Comm comm);

int randomVector2(std::vector<double>& V);

int randomVector3(std::vector<unsigned long>& V, long size, strength_matrix* S, MPI_Comm comm);

int randomVector4(std::vector<unsigned long>& V, long size);


class cooEntry{
public:
    unsigned long row;
    unsigned long col;
    double val;

    cooEntry(){}

    cooEntry(unsigned long i, unsigned long j, double v){
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
    bool operator()(unsigned long i, unsigned long j) const { return mparr[i].row < mparr[j].row; }
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


void setIJV(char* file_name, unsigned int* I,unsigned int* J, double* V, unsigned int nnz_g, unsigned int initial_nnz_l, MPI_Comm comm);


int dotProduct(std::vector<double>& r, std::vector<double>& s, double* dot, MPI_Comm comm);




#endif //SAENA_AUXFUNCTIONS_H
