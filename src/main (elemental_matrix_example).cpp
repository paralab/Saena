/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>

int
main( int argc, char* argv[] )
{
    El::Environment env( argc, argv );
    El::mpi::Comm comm = El::mpi::COMM_WORLD;

    const El::Int n = 10;
    const El::Grid grid(comm);
    El::DistMatrix<double> B(grid);

    // 1- Create a zero matrix
    B.Resize( n, n );
    El::Zero( B );

    // 2- Create Legendre matrix
//    El::Legendre( B, n );

    El::Print( B, "Elemental matrix:" );

    return 0;
}
