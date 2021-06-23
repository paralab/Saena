#ifndef SAENA_DATA_STRUCT_H
#define SAENA_DATA_STRUCT_H

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <sys/stat.h>
#include <iomanip>

#include <string>
#include <vector>
#include <set>
#include <cmath>
#include <random>

#ifdef SANEA_USE_PSTL
#include "pstl/execution"
#include "pstl/algorithm"
#else
#include <algorithm>
#endif

#include <mpi.h>
#include <omp.h>

//#include "dollar.hpp"
#include "combblas_functions.h"

using namespace std;

typedef int           index_t; // Saena index type
typedef long          nnz_t;   // Saena nonzero type
typedef double        value_t; // Saena value type
typedef unsigned char uchar;

#define ALMOST_ZERO 1e-14

#define SAENA_PI 3.1415926535897932384626433832795029

// alignment size: use 64 for AVX512
#define ALIGN_SZ 64

// uncomment to disable compression in matmat
//#define MATMAT_NO_COMPRESS

// from usort
//the following are UBUNTU/LINUX, and MacOS ONLY terminal color codes.
#define COLORRESET  "\033[0m"
#define BLACK       "\033[30m"          /* Black */
#define RED         "\033[31m"          /* Red */
#define GREEN       "\033[32m"          /* Green */
#define YELLOW      "\033[33m"          /* Yellow */
#define BLUE        "\033[34m"          /* Blue */
#define MAGENTA     "\033[35m"          /* Magenta */
#define CYAN        "\033[36m"          /* Cyan */
#define WHITE       "\033[37m"          /* White */
#define BOLDBLACK   "\033[1m\033[30m"   /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"   /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"   /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"   /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"   /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"   /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"   /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"   /* Bold White */

// customize assert to print message
#ifndef NDEBUG
#   define ASSERT(condition, message) \
    do { \
        if (! (condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)
#else
#   define ASSERT(condition, message) do { } while (false)
#endif

// print a line to separate messages
void inline print_sep(){
    std::stringstream buf;
    buf << "\n******************************************************\n";
    std::cout << buf.str();
}

// print 3 lines to separate messages
void inline print_3sep(){
    std::stringstream buf;
    buf << "\n******************************************************\n"
        << "******************************************************\n"
        << "******************************************************\n";
    std::cout << buf.str();
}

// remainder size in compression
inline nnz_t rem_sz(const nnz_t sz, const unsigned int &k){
    return static_cast<index_t>( sz * ((k+1) / 8.0) );
}

// total size in compression
inline nnz_t tot_sz(const nnz_t sz, const unsigned int &k, const int &q){
//    printf("r_sz: %u, \tq: %d, \tsizeof(short): %ld, tot: %ld\n", rem_sz(sz, k), q, sizeof(short), rem_sz(sz, k) + q * sizeof(short));
//    return rem_sz(sz, k) + q * sizeof(short);
    return (k > 0) ? ( rem_sz(sz, k) + q * sizeof(short) ) : ( sz * sizeof(index_t) );
}


// matrix entries in COO format: row, col, val
// the order of this class is "column-major order"
class cooEntry{
public:
    index_t row;
    index_t col;
    value_t val;

    cooEntry() = default;

    cooEntry(index_t i, index_t j, value_t v) : row(i), col(j), val(v){}

    bool operator == (const cooEntry& node2) const {
        return (row == node2.row && col == node2.col);
    }

    bool operator != (const cooEntry& node2) const {
        return !(*this == node2);
    }

    bool operator < (const cooEntry& node2) const {
        if(col == node2.col){
            return row < node2.row;
        } else {
            return col < node2.col;
        }
    }

    bool operator <= (const cooEntry& node2) const {
        if(col == node2.col){
            return row <= node2.row;
        } else {
            return col < node2.col;
        }
    }

    bool operator > (const cooEntry& node2) const {
        if(col == node2.col){
            return row > node2.row;
        } else {
            return col > node2.col;
        }
    }

    bool operator >= (const cooEntry& node2) const {
        if(col == node2.col){
            return row <= node2.row;
        } else {
            return col < node2.col;
        }
    }

    cooEntry operator + (const cooEntry& node2) const
    {
        if (row != node2.row || col != node2.col){
            printf("ERROR: adding two entries without the same indices!");
        }
        return (cooEntry(row, col, val+node2.val));
    }

    // Define prefix increment operator.
    cooEntry& operator ++ ()
    {
        ++row;
        ++col;
        return *this;
    }

    // Define postfix increment operator.
    cooEntry operator ++ (int)
    {
        cooEntry tmp = *this;
        ++*this;
        return tmp;
    }

    // Define prefix decrement operator.
    cooEntry& operator -- () {
        --row;
        --col;
        return *this;
    }

    // Define postfix decrement operator.
    cooEntry operator -- (int) {
        cooEntry tmp = *this;
        --*this;
        return tmp;
    }

    value_t get_val() const {
        return val;
    }

//    value_t get_val_sq() const
//    {
//        return val * val;
//    }

    value_t get_val_sq() const {
        if(row == col){
            return 10000000;
        } else{
            return val * val;
        }
    }

    static MPI_Datatype mpi_datatype() {
        static MPI_Datatype datatype;
        MPI_Type_contiguous(sizeof(cooEntry), MPI_BYTE, &datatype);
        MPI_Type_commit(&datatype);
        return datatype;
    }

/*
    static MPI_Datatype mpi_datatype() {
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
*/
};


// overload << for cooEntry class
std::ostream & operator<<(std::ostream & stream, const cooEntry & item);


bool row_major (const cooEntry& node1, const cooEntry& node2);


// matrix entries in COO format: row, col, val
// the order of this class is "row-major order".
class cooEntry_row{
public:
    index_t row;
    index_t col;
    value_t val;

    cooEntry_row() = default;

    cooEntry_row(index_t i, index_t j, value_t v) : row(i), col(j), val(v){}

    bool operator == (const cooEntry_row& node2) const {
        return (row == node2.row && col == node2.col);
    }

    bool operator < (const cooEntry_row& node2) const {
        if(row == node2.row){
            return col < node2.col;
        } else {
            return row < node2.row;
        }
    }

    bool operator <= (const cooEntry_row& node2) const {
        if(row == node2.row){
            return col <= node2.col;
        } else {
            return row < node2.row;
        }
    }

    bool operator > (const cooEntry_row& node2) const {
        if(row == node2.row){
            return col > node2.col;
        } else {
            return row > node2.row;
        }
    }

    bool operator >= (const cooEntry_row& node2) const {
        if(row == node2.row){
            return col >= node2.col;
        } else {
            return row > node2.row;
        }
    }

    cooEntry_row operator + (const cooEntry_row& node2) const
    {
        if (row != node2.row || col != node2.col){
            printf("ERROR: adding two entries without the same indices!");
        }
        return (cooEntry_row(row, col, val+node2.val));
    }

    // Define prefix increment operator.
    cooEntry_row& operator ++ () {
        ++row;
        ++col;
        return *this;
    }

    // Define postfix increment operator.
    cooEntry_row operator ++ (int) {
        cooEntry_row tmp = *this;
        ++*this;
        return tmp;
    }

    // Define prefix decrement operator.
    cooEntry_row& operator -- () {
        --row;
        --col;
        return *this;
    }

    // Define postfix decrement operator.
    cooEntry_row operator -- (int) {
        cooEntry_row tmp = *this;
        --*this;
        return tmp;
    }

    static MPI_Datatype mpi_datatype() {
        static MPI_Datatype datatype;
        MPI_Type_contiguous(sizeof(cooEntry_row), MPI_BYTE, &datatype);
        MPI_Type_commit(&datatype);
        return datatype;
    }
};

// overload << for cooEntry_row class
std::ostream & operator<<(std::ostream & stream, const cooEntry_row & item);


// a class to store vector entries
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

    static MPI_Datatype mpi_datatype() {
        static MPI_Datatype datatype;
        MPI_Type_contiguous(sizeof(vecEntry), MPI_BYTE, &datatype);
        MPI_Type_commit(&datatype);
        return datatype;
    }
};

// overload << for vecEntry class
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

//    static MPI_Datatype mpi_datatype() {
//        static MPI_Datatype datatype;
//        MPI_Type_contiguous(sizeof(tuple1), MPI_BYTE, &datatype);
//        MPI_Type_commit(&datatype);
//        return datatype;
//    }
};

// overload << for tuple1 class
std::ostream & operator<<(std::ostream & stream, const tuple1 & item);

// a class to store parameters for Golomb-Rice method
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

// A class to store matrix in CSC (Compressed Sparse Column) format
// this is used in the matmat compression part
class CSCMat{
public:

    MPI_Comm comm = MPI_COMM_WORLD;

    index_t *row      = nullptr;
    value_t *val      = nullptr;
    index_t *col_scan = nullptr;

    index_t col_sz  = 0;
    nnz_t   nnz     = 0;
    nnz_t   max_nnz = 0;
    index_t max_M   = 0;

    bool use_trans = true;

    std::vector<index_t> split;
    std::vector<nnz_t>   nnz_list;

    // compresseion parameters
    // =======================

    unsigned long max_comp_sz = 0; // in bytes (char)

    GR_sz comp_row;
    GR_sz comp_col;

    bool verbose_prep_compute = false;
    bool verbose_prep         = false;

    // =======================

    CSCMat() = default;
    void compress_prep_compute(const index_t *v, index_t v_sz, GR_sz &comp_sz) const;
    void compress_prep_compute2(const index_t *v, index_t v_sz, GR_sz &comp_sz);
    void compress_prep();
};


// a class to facilitate calling the recusive function fast_mm (used in matmat)
class CSCMat_mm{
public:
    index_t row_sz, row_offset, col_sz, col_offset;
    nnz_t   nnz;

    index_t *r, *col_scan;
    value_t *v;

    bool free_r = false, free_c = false, free_v = false;

    CSCMat_mm(): row_sz(0), row_offset(0), col_sz(0), col_offset(0), nnz(0), r(nullptr), col_scan(nullptr), v(nullptr) {}

    CSCMat_mm(index_t _row_sz, index_t _row_offset, index_t _col_sz, index_t _col_offset, nnz_t _nnz):
              row_sz(_row_sz), row_offset(_row_offset), col_sz(_col_sz), col_offset(_col_offset), nnz(_nnz),
              r(nullptr), col_scan(nullptr), v(nullptr) {}

    CSCMat_mm(index_t _row_sz, index_t _row_offset, index_t _col_sz, index_t _col_offset, nnz_t _nnz,
              index_t *_r, value_t *_v, index_t *_col_scan):
              row_sz(_row_sz), row_offset(_row_offset), col_sz(_col_sz), col_offset(_col_offset), nnz(_nnz),
              r(_r), col_scan(_col_scan), v(_v) {}

    ~CSCMat_mm(){
        if(free_r){
            free(r);
            r = nullptr;
            free_r = false;
        }
        if(free_c){
            free(col_scan);
            col_scan = nullptr;
            free_c = false;
        }
        if(free_v){
            free(v);
            v = nullptr;
            free_v = false;
        }
    }

    inline void set_params(const index_t &_row_sz, const index_t &_row_offset, const index_t &_col_sz,
                    const index_t &_col_offset, const nnz_t &_nnz){
        row_sz     = _row_sz;
        row_offset = _row_offset;
        col_sz     = _col_sz;
        col_offset = _col_offset;
        nnz        = _nnz;
    }

    inline void set_params(const index_t &_row_sz, const index_t &_row_offset, const index_t &_col_sz,
                    const index_t &_col_offset, const nnz_t &_nnz, index_t *_r, value_t *_v, index_t *_col_scan){
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

#endif //SAENA_DATA_STRUCT_H