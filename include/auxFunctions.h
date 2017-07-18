//
// Created by abaris on 4/28/17.
//

#ifndef SAENA_AUXFUNCTIONS_H
#define SAENA_AUXFUNCTIONS_H

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


int randomVector(unsigned long size, unsigned long* V);


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


//template <class T>
//float myNorm(std::vector<T>& v);

//double myNorm(std::vector<double>& v);


#endif //SAENA_AUXFUNCTIONS_H
