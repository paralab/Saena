#include <iostream>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include "mpi.h"

#include "grid.h"
#include "saena.hpp"
#include <El.hpp>

using namespace std;

int main(int argc, char* argv[]){
    El::Environment env( argc, argv );

    const El::Int n = 10;
    El::Matrix<El::Complex<double>> A(n,n);
    El::Gaussian(A, n, n);
//    El::Print( A, "\nElemental matrix made for finding eigenvalues:\n" );

    El::Matrix<El::Complex<double>> w(n,1), V(n,n);
    El::SchurCtrl<double> schurCtrl;
    schurCtrl.time = true;
    //schurCtrl.hessSchurCtrl.progress = true;
//    El::Schur( A, w, V, schurCtrl );
    El::Schur( A, w, schurCtrl );
    Print( w, "eigenvalues:" );

    return 0;
}
