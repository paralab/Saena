#include <saena_object.h>
#include "saena_matrix.h"
#include "strength_matrix.h"
#include "prolong_matrix.h"
#include "restrict_matrix.h"
#include "grid.h"
#include "aux_functions.h"

int saena_object::pcoarsen(Grid *grid){

    // A parameters:
    // A.entry[i]: to access entry i of A, which is local to this processor.
    //             each entry is in COO format: i, j, val
    // A.nnz_l: local number of nonzeros on this processor.
    // A.nnz_g: total number of nonzeros on all the processors.
    // A.split: it is a vector that stores the range of row indices on each processor.
    //          split[rank]: starting row index
    //          split[rank+1]: last row index (not including)
    //          example: if split[rank] is 5 and split[rank+1] is 9, it shows rows 4, 5, 6, 7, 8 are stored on this processor.
    // A.M: row size which equals to split[rank+1] - split[rank].
    // A.Mbig: the row size of the whole matrix.
    // A.print_info: print information of A, which are nnz_l, nnz_g, M, Mbig.
    // A.print_entry: print entries of A.

    saena_matrix    *A  = grid->A;
    prolong_matrix  *P  = &grid->P;
    restrict_matrix *R  = &grid->R;
    saena_matrix    *Ac = &grid->Ac;

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    return 0;
}
