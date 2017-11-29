#include <iostream>
#include <vector>
#include <saena_matrix.h>
#include<fstream>
#include "mpi.h"
#include "saena.hpp"
using namespace std;
int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
//    MPI_Comm comm = MPI_COMM_WORLD;
//    int nprocs, rank;
//    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // ********************* Copy From Here *********************

    // ********************* Set Matrix By forloops *********************

    int iter = 0;
    unsigned int size = 0;
    std::vector<unsigned int> I;
    std::vector<unsigned int> J;
    std::vector<double> V;
    unsigned int x; 
    ifstream inFile;
    inFile.open("/home/boss/Dropbox/Projects/Saena/build/rowi.txt");
    while (inFile >> x) {
      I.push_back(x);
      size++;
    }
    inFile.close();
    inFile.open("/home/boss/Dropbox/Projects/Saena/build/colj.txt");
    while (inFile >> x) {
      J.push_back(x);
     
    }
    inFile.close();
    double xx;
    inFile.open("/home/boss/Dropbox/Projects/Saena/build/valij.txt");
    while (inFile >> xx) {
      V.push_back(xx);
     
    }
    inFile.close();
    
    // for(unsigned int i=0; i<size; i++){
    //     for(unsigned int j=0; j<size; j++) {
    //         I[iter] = i;
    //         J[iter] = j;
    //         V[iter] = i + j + 1;
    //         if(i == j) // make the matrix strictly diagonally dominant
    //             V[iter] *= 4;
    //         iter++;
    //     }
    // }


    saena::matrix A(MPI_COMM_WORLD); // comm: MPI_Communicator

    A.add_duplicates(true); // in case of duplicates add the values. otherwise ignore this line.
//    cout<<"\n calling set!!!\n here size="<<size;
    A.set(&*I.begin(), &*J.begin(), &*V.begin(), size); // size: size of I (or J or V, they obviously should have the same size)
//     cout<<"\n calling assemble!**\n here!";
    A.assemble();

//    print values
//    for(auto i:A.get_internal_matrix()->entry)
//        std::cout << i << std::endl;

    // ********************* Set Matrix By Laplacian *********************

    /*    unsigned int size = 16; // set a square number
    saena::matrix A(MPI_COMM_WORLD);
    laplacian2D(&A, size, MPI_COMM_WORLD); // second argument is dof on each processor
    A.assemble();

    //print values
   for(auto i:A.get_internal_matrix()->entry)
       std::cout << i << std::endl;
    */
    // ********************* Set rhs and u *********************

    std::vector<double> rhs;
    std::vector<double> u_Nekpp;
    
    // for(int i=0; i<size; i++)
    //     rhs[i] = i+1;
    double r; unsigned int sizerhs = 0;
    inFile.open("/home/boss/Dropbox/Projects/Saena/build/Nektarpprhs.txt");
    while (inFile >> r) {
      rhs.push_back(r);
      sizerhs++;
    }
    inFile.close();

    inFile.open("/home/boss/Dropbox/Projects/Saena/build/Nektarppout.txt");
    while (inFile >> r) {
      u_Nekpp.push_back(r);
    }
    inFile.close();

    std::vector<double> u(sizerhs,0);




    string outFileName = "nektar001.bin";
    ofstream outFile;
    outFile.open(outFileName.c_str(), ios::out | ios::binary);

    saena_matrix* B = A.get_internal_matrix();
    for(int i = 0; i<B->nnz_g; i++){
//        cout << B->entry[i].row << "\t" << B->entry[i].col << "\t" << B->entry[i].val << endl;
        outFile.write((char*)&B->entry[i].row, sizeof(long));
        outFile.write((char*)&B->entry[i].col, sizeof(long));
        outFile.write((char*)&B->entry[i].val, sizeof(double));
    }
    outFile.close();




    // *************************** 1 - Saena Test ****************************

    int vcycle_num            = 40;
    double relative_tolerance = 1e-8;
    std::string smoother      = "chebyshev";
    int preSmooth             = 3;
    int postSmooth            = 3;

    saena::options opts(vcycle_num, relative_tolerance, smoother, preSmooth, postSmooth);
//    saena::options opts; // use the default options.

    saena::amg solver;

    solver.set_verbose(false);
//    cout<<"\n calling setmatrix!\n";
    solver.set_matrix(&A);

    solver.set_rhs(rhs); // rhs should be std::vector double
//    cout<<"\n calling solve!\n";
    solver.solve(u, &opts);

    //vidhi: (testing saena's u vs Nektapp's u
    int flag = 0;
    for ( int i = 0; i< sizerhs; i++ )
      { if(abs(u[i]-u_Nekpp[i]) > 1e-5)
    	  {
    	    cout<<"\n mismatch at index"<<i<<" value = "<<u[i]<<" expected value = "<<u_Nekpp[i];
    	    flag = 1;
    	    //break;
    	  }
      }
    if(flag == 0)
      cout<<"\n All good ! ";
    
    

    solver.destroy();

    // *************************** 2 - Matvec Test ****************************
    // note: this only works for np = 1.

/*
    std::vector<double> w1(size);
    saena_matrix* B = A.get_internal_matrix();
    B->matvec(rhs, w1);

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0){
        std::cout << "\nSaena matvec: rank = " << rank << std::endl;
        for(auto i:w1)
            std::cout << i << std::endl;}

    std::vector<double> w2(size,0);
    for(int i=0; i<B->nnz_l; i++)
        w2[B->entry[i].row] += B->entry[i].val * rhs[B->entry[i].col];

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0) {
        std::cout << "\nmanual matvec: rank = " << rank << std::endl;
        for (auto i:w2)
            std::cout << i << std::endl;}
*/

 // *************************** 3 - Jacobi Test ****************************

/*
    double dot;
    std::vector<double> temp(size);
    std::vector<double> res(size);
    saena_matrix* B = A.get_internal_matrix();
    u.assign(size, 0);

    for(int i=0; i<10; i++){
//        std::cout << "\nfirst jacobi:" << std::endl;
        B->jacobi(u, rhs, temp);
//        for(auto j:u)
//            std::cout << j << std::endl;

        B->residual(u, rhs, res);
        dotProduct(res, res, &dot, MPI_COMM_WORLD);
        std::cout << "\niter: " << i+1 << ", res_norm = " << sqrt(dot) << std::endl;
    }
    std::cout << std::endl;
*/

// *************************** Destroy ****************************

    A.destroy();

// *************************** Copy Up To Here ****************************

    return 0;
}
