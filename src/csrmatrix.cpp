#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <mpi.h>

#include "csrmatrix.h"

using namespace std;

// sort indices and store the ordering.
class sort_indices
{
private:
    long* mparr;
public:
    sort_indices(long* parr) : mparr(parr) {}
    bool operator()(long i, long j) const { return mparr[i]<mparr[j]; }
};

//CSRMatrix::CSRMatrix(){}

int CSRMatrix::CSRMatrixSet(long* r, long* c, double* v, long m1, long m2, long m3, long* spl){
    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    M = m1;
    Mbig = m2;
    nnz_l = m3;
    split = spl;

    rowIndex.resize(M+1);
    rowIndex.assign(M+1, 0);
    col.resize(nnz_l);
    values.resize(nnz_l);

//    long* row = r;
//    col = c;
//    values = v;

    // save row-wise sorting in indicesRow_p
    long* indicesRow_p = (long*)malloc(sizeof(long)*nnz_l);
    for(int i=0; i<nnz_l; i++)
        indicesRow_p[i] = i;
    std::sort(indicesRow_p, &indicesRow_p[nnz_l], sort_indices(r));

    unsigned int i;
    for(i=0; i<nnz_l; i++){
        rowIndex[r[i]+1 - split[rank]]++;
        col[i] = c[indicesRow_p[i]];
        values[i] = v[indicesRow_p[i]];
//        if (rank==1) cout << "[" << r[indicesRow_p[i]]+1 << "," << c[indicesRow_p[i]]+1 << "] = " << v[indicesRow_p[i]] << endl;
    }

    for(i=0; i<M; i++)
        rowIndex[i+1] += rowIndex[i];

    MPI_Allreduce(&nnz_l, &nnz_g, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    average_sparsity = (1.0*nnz_g)/Mbig;

    return 0;
}

CSRMatrix::~CSRMatrix(){
    rowIndex.resize(0);
    col.resize(0);
    values.resize(0);
}

void CSRMatrix::print(int r){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank==r)
        for(unsigned int i=0; i<M; i++){
            for(long j=rowIndex[i]; j<rowIndex[i+1]; j++)
                cout << "[" << i+1 << ",\t" << col[j]+1 << "] = \t" << values[j] << endl;
        }
}