#ifndef SAENA_DATA_STRUCT_H
#define SAENA_DATA_STRUCT_H

#include <iostream>
#include <vector>
#include "mpi.h"
#include <cmath>

typedef unsigned int index_t;
typedef unsigned long nnz_t;
typedef double value_t;


inline index_t rem_sz(index_t sz, unsigned int k){
    return static_cast<index_t>( ceil(sz * ((k+1) / 8.0) ) );
}

inline index_t tot_sz(index_t sz, unsigned int k, short q){
//    printf("r_sz: %u, \tq: %d, \tsizeof(q): %ld, tot: %ld\n", static_cast<index_t>( ceil(sz * (k+2) / (double)8) ), q,
//            sizeof(q), static_cast<index_t>( ceil(sz * (k+2) / (double)8) ) + q * sizeof(q));
    return static_cast<index_t>( ceil(sz * ((k+1) / 8.0)) ) + q * sizeof(q);
}


// the order of this class is "column-major order"
class cooEntry{
public:
    index_t row;
    index_t col;
    value_t val;

    cooEntry() = default;

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

    cooEntry operator + (const cooEntry& node2) const
    {
        if (row != node2.row || col != node2.col){
            printf("ERROR: adding two entries without the same indices!");
        }
        return (cooEntry(row, col, val+node2.val));
    }

//    cooEntry operator++ () const
//    {
//        return (cooEntry(row+1, col, val));
//    }

    value_t get_val() const
    {
        return val;
    }

//    value_t get_val_sq() const
//    {
//        return val * val;
//    }

    value_t get_val_sq() const
    {
        if(row == col){
            return 10000000;
        } else{
            return val * val;
        }
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


// the order of this class is "row-major order".
class cooEntry_row{
public:
    index_t row;
    index_t col;
    value_t val;

    cooEntry_row() = default;

    cooEntry_row(index_t i, index_t j, value_t v){
        row = i;
        col = j;
        val = v;
    }

    bool operator == (const cooEntry_row& node2) const
    {
        return (row == node2.row && col == node2.col);
    }

    bool operator < (const cooEntry_row& node2) const
    {
        if(row < node2.row)
            return (true);
        else if(row == node2.row)
            return( col < node2.col);
        else
            return false;
    }

    bool operator <= (const cooEntry_row& node2) const
    {
        if(row < node2.row)
            return (true);
        else if(row == node2.row)
            return( col <= node2.col);
        else
            return false;
    }

    bool operator > (const cooEntry_row& node2) const
    {
        if(row > node2.row)
            return (true);
        else if(row == node2.row)
            return( col > node2.col);
        else
            return false;
    }

    bool operator >= (const cooEntry_row& node2) const
    {
        if(  row > node2.row)
            return (true);
        else if(row == node2.row)
            return( col >= node2.col);
        else
            return false;
    }

    cooEntry_row operator + (const cooEntry_row& node2) const
    {
        if (row != node2.row || col != node2.col){
            printf("ERROR: adding two entries without the same indices!");
        }
        return (cooEntry_row(row, col, val+node2.val));
    }

    cooEntry_row operator++ (int) const
    {
        return (cooEntry_row(row, col+1, val));
    }

    static MPI_Datatype mpi_datatype()
    {
        static bool         first = true;
        static MPI_Datatype datatype;

        if (first)
        {
            first = false;
            MPI_Type_contiguous(sizeof(cooEntry_row), MPI_BYTE, &datatype);
            MPI_Type_commit(&datatype);
        }

        return datatype;
    }
};

std::ostream & operator<<(std::ostream & stream, const cooEntry_row & item);


class vecEntry {
public:
    index_t row;
    value_t val;

    vecEntry() = default;

    vecEntry(index_t i, value_t v){
        row = i;
        val = v;
    }

    bool operator == (const vecEntry& node2) const
    {
        return (row == node2.row);
    }

    bool operator < (const vecEntry& node2) const
    {
        return(row < node2.row);
    }

    bool operator <= (const vecEntry& node2) const
    {
        return(row <= node2.row);
    }

    bool operator > (const vecEntry& node2) const
    {
        return(row > node2.row);
    }

    bool operator >= (const vecEntry& node2) const
    {
        return(row >= node2.row);
    }

    vecEntry operator + (const vecEntry& node2) const
    {
        if (row != node2.row){
            printf("ERROR: adding two entries without the same indices!");
        }
        return (vecEntry(row, val + node2.val));
    }

    value_t get_val() const
    {
        return val;
    }

    static MPI_Datatype mpi_datatype()
    {
        static bool         first = true;
        static MPI_Datatype datatype;

        if (first)
        {
            first = false;
            MPI_Type_contiguous(sizeof(vecEntry), MPI_BYTE, &datatype);
            MPI_Type_commit(&datatype);
        }

        return datatype;
    }
};

std::ostream & operator<<(std::ostream & stream, const vecEntry & item);


// this class is used in saena_vector class, in return_vec() function.
class tuple1{
public:
    index_t idx1;
    index_t idx2;

    tuple1() = default;

    tuple1(index_t i, index_t j){
        idx1 = i;
        idx2 = j;
    }

    bool operator == (const tuple1& node2) const
    {
        return (idx2 == node2.idx2);
    }

    bool operator < (const tuple1& node2) const
    {
        return(idx2 < node2.idx2);
    }

    bool operator <= (const tuple1& node2) const
    {
        return(idx2 <= node2.idx2);
    }

    bool operator > (const tuple1& node2) const
    {
        return(idx2 > node2.idx2);
    }

    bool operator >= (const tuple1& node2) const
    {
        return(idx2 >= node2.idx2);
    }

    static MPI_Datatype mpi_datatype()
    {
        static bool         first = true;
        static MPI_Datatype datatype;

        if (first)
        {
            first = false;
            MPI_Type_contiguous(sizeof(tuple1), MPI_BYTE, &datatype);
            MPI_Type_commit(&datatype);
        }

        return datatype;
    }
};

std::ostream & operator<<(std::ostream & stream, const tuple1 & item);


class vecCol{
public:
    vecEntry *rv;
    index_t  *c;
//    nnz_t sz;

    vecCol() = default;
    vecCol(vecEntry *_rv, index_t *_c){
        rv = _rv;
        c  = _c;
//        sz = _sz;
    }

    bool operator == (const vecCol& node2) const
    {
        return (rv->row == node2.rv->row && c == node2.c);
    }

    bool operator < (const vecCol& node2) const
    {
        if(c < node2.c)
            return (true);
        else if(c == node2.c)
            return(rv->row < node2.rv->row);
        else
            return false;
    }

    bool operator <= (const vecCol& node2) const
    {
        if(c < node2.c)
            return (true);
        else if(c == node2.c)
            return(rv->row <= node2.rv->row);
        else
            return false;
    }

    bool operator > (const vecCol& node2) const
    {
        if(c > node2.c)
            return (true);
        else if(c == node2.c)
            return(rv->row > node2.rv->row);
        else
            return false;
    }

    bool operator >= (const vecCol& node2) const
    {
        if(c > node2.c)
            return (true);
        else if(c == node2.c)
            return(rv->row >= node2.rv->row);
        else
            return false;
    }
};

std::ostream & operator<<(std::ostream & stream, const vecCol & item);

bool vecCol_col_major (const vecCol& node1, const vecCol& node2);


class GR_sz {
public:
    unsigned int k   = 0; // Golomb-Rice parameter (M = 2^k)
    unsigned int r   = 0; // remainder size in bytes
    int          q   = 0; // quotient size in number of short numbers
    unsigned int tot = 0; // total size in bytes

    unsigned long max_tot = 0; // in bytes (char)

    std::vector<int> ks;
    std::vector<int> qs;

    GR_sz() = default;
};

class CSCMat{
private:

public:

    MPI_Comm comm = MPI_COMM_WORLD;

    index_t *row      = nullptr;
    value_t *val      = nullptr;
    index_t *col_scan = nullptr;

    index_t col_sz  = 0;
    nnz_t   nnz     = 0;
    nnz_t   max_nnz = 0;
    index_t max_M   = 0;

    std::vector<index_t> split;
    std::vector<nnz_t>   nnz_list;

    // compresseion parameters
    // =======================

    bool verbose_prep_compute = false;
    bool verbose_prep         = false;
    unsigned long max_comp_sz = 0; // in bytes (char)

    GR_sz comp_row;
    GR_sz comp_col;

    // =======================

    CSCMat() = default;
    int compress_prep_compute(const index_t *v, index_t v_sz, GR_sz &comp_sz);
    int compress_prep_compute2(const index_t *v, index_t v_sz, GR_sz &comp_sz);
    int compress_prep();
};


class CSCMat_mm{
private:

public:

    index_t row_sz, row_offset, col_sz, col_offset;
    nnz_t   nnz;

    index_t *r, *col_scan;
    value_t *v;

    bool free_r = false, free_c = false, free_v = false;

    CSCMat_mm(): row_sz(0), row_offset(0), col_sz(0), col_offset(0), nnz(0), r(nullptr), v(nullptr), col_scan(nullptr) {}

    CSCMat_mm(index_t _row_sz, index_t _row_offset, index_t _col_sz, index_t _col_offset, nnz_t _nnz):
              row_sz(_row_sz), row_offset(_row_offset), col_sz(_col_sz), col_offset(_col_offset), nnz(_nnz),
              r(nullptr), v(nullptr), col_scan(nullptr) {}

    CSCMat_mm(index_t _row_sz, index_t _row_offset, index_t _col_sz, index_t _col_offset, nnz_t _nnz,
              index_t *_r, value_t *_v, index_t *_col_scan):
              row_sz(_row_sz), row_offset(_row_offset), col_sz(_col_sz), col_offset(_col_offset), nnz(_nnz),
              r(_r), v(_v), col_scan(_col_scan) {}

    ~CSCMat_mm(){
        if(free_r){
            delete []r;
            free_r = false;
        }
        if(free_c){
            delete []col_scan;
            free_c = false;
        }
        if(free_v){
            delete []v;
            free_v = false;
        }
    }

    void set_params(index_t _row_sz, index_t _row_offset, index_t _col_sz, index_t _col_offset, nnz_t _nnz){
        row_sz     = _row_sz;
        row_offset = _row_offset;
        col_sz     = _col_sz;
        col_offset = _col_offset;
        nnz        = _nnz;
    }

    void set_params(index_t _row_sz, index_t _row_offset, index_t _col_sz, index_t _col_offset, nnz_t _nnz,
                    index_t *_r, value_t *_v, index_t *_col_scan){
        row_sz     = _row_sz;
        row_offset = _row_offset;
        col_sz     = _col_sz;
        col_offset = _col_offset;
        nnz        = _nnz;
        r          = _r;
        v          = _v;
        col_scan   = _col_scan;
    }
};


class CSRMat{
private:

public:
    index_t *col      = nullptr;
    value_t *val      = nullptr;
    index_t *row_scan = nullptr;

    index_t row_sz  = 0;
    nnz_t   nnz     = 0;
    nnz_t   max_nnz = 0;
    index_t max_M   = 0;
    std::vector<index_t> split;
    std::vector<nnz_t>   nnz_list;

    CSRMat() = default;
};


#endif //SAENA_DATA_STRUCT_H
