#ifndef SAENA_RESTRICTMATRIX_H
#define SAENA_RESTRICTMATRIX_H

#include <vector>

class restrictMatrix {
// A matrix of this class is ordered first column-wise, then row-wise, using std:sort with cooEntry class "< operator".
// It is sorted in restrictMatrix constructor function in restrictmatrix.cpp.
private:

public:
    unsigned long Mbig;
    unsigned long Nbig;
    unsigned long nnz_g;
    unsigned long nnz_l;
    unsigned long nnz_l_local;
    unsigned long nnz_l_remote;

    std::vector<cooEntry> entry_local;
    std::vector<cooEntry> entry_remote;

    // split is P.splitNew.

//    std::vector<unsigned long> row;
//    std::vector<unsigned long> col;
//    std::vector<double> values;
//    std::vector<unsigned long> row_local;
//    std::vector<unsigned long> row_remote;
//    std::vector<unsigned long> col_local;
//    std::vector<unsigned long> col_remote;
//    std::vector<double> values_local;
//    std::vector<double> values_remote;

    restrictMatrix();
    restrictMatrix(prolongMatrix* P, MPI_Comm comm);
//    restrictMatrix(unsigned long Mbig, unsigned long Nbig, unsigned long nnz_g, unsigned long nnz_l, unsigned long* row, unsigned long* col, double* values);
    ~restrictMatrix();
};

#endif //SAENA_RESTRICTMATRIX_H
