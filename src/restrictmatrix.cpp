#include <prolongmatrix.h>
#include <iostream>
#include <mpi.h>
#include "restrictmatrix.h"

using namespace std;

restrictMatrix::restrictMatrix(){}

restrictMatrix::restrictMatrix(prolongMatrix* P) {
    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // todo: fist figure out the epartitioning of R.
    
    Mbig = P->Nbig;
    Nbig = P->Mbig;

    unsigned long i, j;

    unsigned long iter = 0;
    for (i = 0; i < P->M; ++i) {
        for (j = 0; j < P->nnz_row_local[i]; ++j, ++iter) {
            if(rank==1) cout << P->row_local[iter] << "\t" << P->col_local[iter] << "\t" << P->values_local[iter] << endl;
//            if(rank==1) cout << P->row_local[P->indicesP_local[iter]] << "\t" << P->col_local[P->indicesP_local[iter]] << "\t" << P->values_local[P->indicesP_local[iter]] << endl;
        }
    }
    cout << endl;

    //local
//    unsigned long iter = 0;
    iter = 0;
    for (i = 0; i < P->M; ++i) {
        for (j = 0; j < P->nnz_row_local[i]; ++j, ++iter) {
            row_local.push_back(P->col_local[P->indicesP_local[iter]] - P->split[rank]);
            col_local.push_back(P->row_local[P->indicesP_local[iter]] + P->split[rank]);
            values_local.push_back(P->values_local[P->indicesP_local[iter]]);
            if(rank==1) cout << row_local[iter] << "\t" << col_local[iter] << "\t" << values_local[iter] << endl;
//            if(rank==1) cout << P->row_local[iter] << "\t" << P->col_local[iter] << "\t" << P->values_local[iter] << endl;
//            if(rank==1) cout << P->row_local[P->indicesP_local[iter]] << "\t" << P->col_local[P->indicesP_local[iter]] << "\t" << P->values_local[P->indicesP_local[iter]] << endl;
        }
    }

}

//restrictMatrix::restrictMatrix(unsigned long Mb, unsigned long Nb, unsigned long nz_g, unsigned long nz_l, unsigned long* r, unsigned long* c, double* v){
//    Mbig = Mb;
//    Nbig = Nb;
//    nnz_g = nz_g;
//    nnz_l = nz_l;
//
//    row.resize(nnz_l);
//    col.resize(nnz_l);
//    values.resize(nnz_l);
//
//    for(long i=0; i<nnz_l; i++){
//        row[i] = r[i];
//        col[i] = c[i];
//        values[i] = v[i];
//    }
//}

restrictMatrix::~restrictMatrix(){}
