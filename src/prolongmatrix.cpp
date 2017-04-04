#include "prolongmatrix.h"

prolongMatrix::prolongMatrix(){

}

prolongMatrix::prolongMatrix(long Mb, long Nb, long nz_g, long nz_l, long* r, long* c, double* v){
    Mbig = Mb;
    Nbig = Nb;
    nnz_g = nz_g;
    nnz_l = nz_l;

    row.resize(nnz_l);
    col.resize(nnz_l);
    values.resize(nnz_l);

    for(long i=0; i<nnz_l; i++){
        row[i] = r[i];
        col[i] = c[i];
        values[i] = v[i];
    }
}

prolongMatrix::~prolongMatrix(){

}
