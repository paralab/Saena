#ifndef SAENA_RESTRICTMATRIX_H
#define SAENA_RESTRICTMATRIX_H

#include <vector>

class restrictMatrix {
private:

public:
    unsigned long Mbig;
    unsigned long Nbig;
    unsigned long nnz_g;
    unsigned long nnz_l;

//    std::vector<unsigned long> row;
//    std::vector<unsigned long> col;
//    std::vector<double> values;

    std::vector<unsigned long> row_local;
    std::vector<unsigned long> row_remote;
    std::vector<unsigned long> col_local;
    std::vector<unsigned long> col_remote;
    std::vector<double> values_local;
    std::vector<double> values_remote;

    restrictMatrix();
    restrictMatrix(prolongMatrix* P, unsigned long* initialNumberOfRows, MPI_Comm comm);
//    restrictMatrix(unsigned long Mbig, unsigned long Nbig, unsigned long nnz_g, unsigned long nnz_l, unsigned long* row, unsigned long* col, double* values);
    ~restrictMatrix();
};


#endif //SAENA_RESTRICTMATRIX_H
