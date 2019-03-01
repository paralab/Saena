#ifndef SAENA_SAENA_VECTOR_H
#define SAENA_SAENA_VECTOR_H

#include "aux_functions.h"

#include <set>
#include "mpi.h"
#include <boost/numeric/ublas/vector.hpp>

/**
 * @author Majid Rasouli
 * @breif Contains the basic structure for the Saena vector class (saena_vector).
 *
 * */

typedef unsigned int  index_t;
typedef unsigned long nnz_t;
typedef double        value_t;

class saena_vector {

    // Steps of creating a vector of this class:
//    parameter	        type	                    reason
//    -----------------------------------------------------------------------------------
//    data_set	        std::set<vecEntry>		    add entries by set()
//    data_unsorted	    std::vector<vecEntry>	    switch from std::set to std::vector
//    data_sorted_dup   std::vector<vecEntry>		parallel sort
//    data		        std::vector<vecEntry>		remove duplicates

private:

public:

    MPI_Comm comm = MPI_COMM_WORLD;
    index_t M;
    index_t Mbig;

    bool add_duplicates = false;
    int set_dup_flag(bool add);

    std::set<vecEntry>    data_set;
    std::vector<vecEntry> data;
//    std::vector<double>   val;

    index_t idx_offset = 0;
    std::vector<index_t> orig_order; // save the input order
    std::vector<index_t> remote_idx; // indices that should receive their value from other procs
    std::vector<index_t> split;
//    index_t *split; // point to the split of input matrix. size of it is (size of comm + 1).

    index_t vIndexSize = 0;
    index_t recvSize   = 0;
    std::vector<index_t> vIndex;
    std::vector<value_t> vSend;
    std::vector<value_t> vecValues;

    int numRecvProc = 0;
    int numSendProc = 0;
    std::vector<int> vdispls;
    std::vector<int> rdispls;
    std::vector<int> recvCount;
    std::vector<int> recvCountScan;
    std::vector<int> sendCount;
    std::vector<int> sendCountScan;
    std::vector<int> recvProcRank;
    std::vector<int> recvProcCount;
    std::vector<int> sendProcRank;
    std::vector<int> sendProcCount;
    std::vector<index_t> vElement_remote;

    bool verbose_return_vec = false;

    saena_vector();
    explicit saena_vector(MPI_Comm com);
    ~saena_vector();

    void set_comm(MPI_Comm com);
    int set_idx_offset(index_t offset);
    int set(index_t row, value_t val); // replace duplicates
//    int set_rep_dup(index_t row, value_t val); // replace duplicates
//    int set_add_dup(index_t row, value_t val); // add duplicates
//    int set(index_t *row, value_t *val, index_t size);
    int set(value_t* val, index_t size, index_t offset);
    int set(value_t* val, index_t size);
    int remove_duplicates();
    int assemble();

    int get_vec(std::vector<double> &vec);
    int print_entry(int ran);

    int return_vec(std::vector<double> &u);
    int return_vec(std::vector<double> &u1, std::vector<double> &u2);
};

#endif //SAENA_SAENA_VECTOR_H
