#include <iostream>
#include <algorithm>
#include <sys/stat.h>
#include <assert.h>
#include "mpi.h"
#include "coomatrix.h"
#include "AMGClass.h"
#include "grid.h"
//#include "auxFunctions.h"
//#include "strengthmatrix.h"

#define ITERATIONS 1

using namespace std;

int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    int assert1, assert2, assert3;

    if(argc < 3)
    {
        if(rank == 0)
        {
            cout << "Usage: ./Saena <MatrixA> <vecX>" << endl;
            cout << "Files should be in triples format." << endl;
        }
        MPI_Finalize();
        return -1;
    }
    // *************************** get number of rows ****************************

    char* Vname(argv[2]);
    struct stat vst;
    stat(Vname, &vst);
    unsigned int Mbig = vst.st_size/8;  // sizeof(double) = 8

    // *************************** Setup Phase: Initialize the matrix ****************************

    char* Aname(argv[1]);

    // timing the setup phase
//    MPI_Barrier(comm);
//    double t1 = MPI_Wtime();

    COOMatrix A (Aname, Mbig, comm);
    assert1 = A.repartition(comm);
    assert2 = A.matrixSetup(comm);

//    MPI_Barrier(comm);
//    double t2 = MPI_Wtime();

//    if (rank==0)
//        cout << "\nMatrix setup in Saena took " << t2 - t1 << " seconds!" << endl << endl;

    // *************************** read the vector and set rhs ****************************

    MPI_Status status;
    MPI_File fh;
    MPI_Offset offset;

    int mpiopen = MPI_File_open(comm, Vname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if(mpiopen){
        if (rank==0) cout << "Unable to open the vector file!" << endl;
        MPI_Finalize();
        return -1;
    }

    // define the size of v as the local number of rows on each process
    std::vector <double> v(A.M);
    double* vp = &(*(v.begin()));

    // vector should have the following format: first line shows the value in row 0, second line shows the value in row 1
    offset = A.split[rank] * 8; // value(double=8)
    MPI_File_read_at(fh, offset, vp, A.M, MPI_UNSIGNED_LONG, &status);

//    int count;
//    MPI_Get_count(&status, MPI_UNSIGNED_LONG, &count);
    //printf("process %d read %d lines of triples\n", rank, count);
    MPI_File_close(&fh);

    // set rhs
    std::vector<double> rhs(A.M);
    A.matvec(&*v.begin(), &*rhs.begin(), comm);
//    if(rank==0)
//        for(long i = 0; i < rhs.size(); i++)
//            cout << rhs[i] << endl;

    // *************************** AMG ****************************

    int maxLevel       = 2; // not including fine level. fine level is 0.
    int vcycle_num     = 10 ;
    double relTol      = 1e-6;
    string relaxType   = "jacobi";
    int preSmooth      = 3;
    int postSmooth     = 3;
    float connStrength = 0.5; // connection strength parameter
    float tau          = 3; // is used during making aggregates.
    bool doSparsify    = 0;

    AMGClass amgClass (maxLevel, vcycle_num, relTol, relaxType, preSmooth, postSmooth, connStrength, tau, doSparsify);
    Grid grids[maxLevel+1];
    amgClass.AMGSetup(grids, &A, comm);
//    MPI_Barrier(comm); printf("----------AMGSetup----------\n"); MPI_Barrier(comm);

//    MPI_Barrier(comm);
//    for(int i=0; i<maxLevel; i++)
//        if(rank==0) cout << "size = " << maxLevel << ", current level = " << grids[i].currentLevel << ", coarse level = " << grids[i].coarseGrid->currentLevel
//                         << ", A.Mbig = " << grids[i].A->Mbig << ", A.M = " << grids[i].A->M << ", Ac.Mbig = " << grids[i].Ac.Mbig << ", Ac.M = " << grids[i].Ac.M << endl;
//    MPI_Barrier(comm);

    std::vector<double> u(A.M);
    u.assign(A.M, 0); // initial guess
//    randomVector2(u);
//    if(rank==1) cout << "\ninitial guess u" << endl;
//    if(rank==1)
//        for(auto i:u)
//            cout << i << endl;

    amgClass.AMGSolve(grids, u, rhs, comm);

//    amgClass.writeMatrixToFile(&grids[1].Ac, comm);

//    int max = 10;
//    double tol = 1e-10;
//    amgClass.solveCoarsest(&A, u, rhs, max, tol, comm);

//    MPI_Barrier(comm); printf("----------main----------\n"); MPI_Barrier(comm);

    // *************************** finalize ****************************

//    MPI_Barrier(comm); cout << rank << "\t*******end*******" << endl;
    MPI_Finalize();
    return 0;
}
