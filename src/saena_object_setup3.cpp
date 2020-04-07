#include <saena_object.h>
#include "saena_matrix.h"
#include "strength_matrix.h"
#include "prolong_matrix.h"
#include "restrict_matrix.h"
#include "grid.h"
#include "aux_functions.h"

int saena_object::pcoarsen(Grid *grid){

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
