#ifndef SAENA_SAENA_VECTOR_H
#define SAENA_SAENA_VECTOR_H

#include "mpi.h"
#include "aux_functions.h"
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

    MPI_Comm comm;
    index_t M;
    index_t Mbig;

    bool add_duplicates = false;
    std::set<vecEntry>    data_set;
    std::vector<vecEntry> data;
//    std::vector<double>   val;


    saena_vector();
    explicit saena_vector(MPI_Comm com);
    ~saena_vector();

    void set_comm(MPI_Comm com);
    int set_rep_dup(index_t row, value_t val); // replace duplicates
    int set_add_dup(index_t row, value_t val); // add duplicates
//    int set(index_t *row, value_t *val, index_t size);
    int remove_duplicates();
    int assemble();

    int get_vec(std::vector<double> &vec);
//    int print_entry(int ran);

};

#endif //SAENA_SAENA_VECTOR_H
