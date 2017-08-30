#include <iostream>
#include <algorithm>
//#include <sys/stat.h>
//#include <assert.h>
#include <mpich/mpi.h>
//#include "mpi.h"

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

    unsigned long i;
//    int assert1, assert2, assert3;

    if(argc < 4)
    {
        if(rank == 0)
        {
//            cout << "Usage: ./Saena <MatrixA> <rhs_vec> <u_vec>" << endl;
            cout << "Usage: ./Saena <MatrixA> <rhs_vec> <number of columns of Matrix>" << endl;
            cout << "Matrix file should be in triples format." << endl;
        }
        MPI_Finalize();
        return -1;
    }
    // *************************** get number of rows ****************************

//    char* Vname(argv[2]);
//    struct stat vst;
//    stat(Vname, &vst);
//    unsigned int Mbig = vst.st_size/8;  // sizeof(double) = 8
    unsigned int Mbig = stoul(argv[3]);

    // *************************** Setup Phase: Initialize the matrix ****************************

    char* Aname(argv[1]);

    // timing the setup phase
//    MPI_Barrier(comm);
//    double t1 = MPI_Wtime();

    COOMatrix A (Aname, Mbig, comm);
    A.repartition(comm);
    A.matrixSetup(comm);

//    MPI_Barrier(comm);
//    double t2 = MPI_Wtime();

//    if (rank==0)
//        cout << "\nMatrix setup in Saena took " << t2 - t1 << " seconds!" << endl << endl;

    // *************************** read the vector and set rhs ****************************

    MPI_Status status;
    MPI_File fh;
    MPI_Offset offset;

    char* Vname(argv[2]);
    int mpiopen = MPI_File_open(comm, Vname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if(mpiopen){
        if (rank==0) cout << "Unable to open the rhs vector file!" << endl;
        MPI_Finalize();
        return -1;
    }

    // define the size of v as the local number of rows on each process
    std::vector <double> v(A.M);
    double* vp = &(*(v.begin()));

    // vector should have the following format: first line shows the value in row 0, second line shows the value in row 1
    offset = A.split[rank] * 8; // value(double=8)
    MPI_File_read_at(fh, offset, vp, A.M, MPI_DOUBLE, &status);

//    int count;
//    MPI_Get_count(&status, MPI_UNSIGNED_LONG, &count);
    //printf("process %d read %d lines of triples\n", rank, count);
    MPI_File_close(&fh);

    // set rhs
    std::vector<double> rhs(A.M);
    A.matvec(&*v.begin(), &*rhs.begin(), comm);
//    rhs.assign(A.M, 0);
//    if(rank==0)
//        for(i = 0; i < rhs.size(); i++)
//            cout << rhs[i] << endl;

//    for(i=0; i<A.M; i++)
//        rhs[i] = i + 1 + A.split[rank];

    // *************************** read the vector and set u0 ****************************
    // there are 3 options for u0:
    // 1- zero
    // 2- random
    // 3- flat from homg

    std::vector<double> u(A.M);
    u.assign(A.M, 0); // initial guess = 0
    //    randomVector2(u); // initial guess = random
    //    if(rank==1) cout << "\ninitial guess u" << endl;
    //    if(rank==1)
    //        for(auto i:u)
    //            cout << i << endl;

/*
    // u0 is generated as flat from homg: u0 = eigenvalues*ones. check homg010_u0flat.m file
    MPI_Status status3;
    MPI_File fh3;
    MPI_Offset offset3;

    char* Uname(argv[3]);
    int mpiopen3 = MPI_File_open(comm, Uname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh3);
    if(mpiopen3){
        if (rank==0) cout << "Unable to open the U vector file!" << endl;
        MPI_Finalize();
        return -1;
    }

    // vector should have the following format: first line shows the value in row 0, second line shows the value in row 1
    offset3 = A.split[rank] * 8; // value(double=8)
    MPI_File_read_at(fh3, offset3, &*u.begin(), A.M, MPI_DOUBLE, &status3);
    MPI_File_close(&fh3);
*/

//    if(rank==0)
//        for(i = 0; i < u.size(); i++)
//            cout << u[i] << endl;

    // *************************** AMG - Setup ****************************

    int maxLevel       = 2;
    int vcycle_num     = 10;
    double relTol      = 1e-6;
    string relaxType   = "jacobi";
    int preSmooth      = 3;
    int postSmooth     = 3;

    AMGClass amgClass (maxLevel, vcycle_num, relTol, relaxType, preSmooth, postSmooth);
    Grid grids[maxLevel+1];
    amgClass.AMGSetup(grids, &A, comm);

//    MPI_Barrier(comm);
//    for(int i=0; i<maxLevel; i++)
//        if(rank==0) cout << "size = " << maxLevel << ", current level = " << grids[i].currentLevel << ", coarse level = " << grids[i].coarseGrid->currentLevel
//                         << ", A.Mbig = " << grids[i].A->Mbig << ", A.M = " << grids[i].A->M << ", Ac.Mbig = " << grids[i].Ac.Mbig << ", Ac.M = " << grids[i].Ac.M << endl;
//    MPI_Barrier(comm);

    // *************************** AMG - Solve ****************************

    amgClass.AMGSolve(grids, u, rhs, comm);

//    amgClass.writeMatrixToFileA(grids[1].A, "Ac", comm);
//    amgClass.writeMatrixToFileP(&grids[0].P, "P", comm);
//    amgClass.writeMatrixToFileR(&grids[0].R, "R", comm);

    // *************************** Residual ****************************

/*
    std::vector<double> res(A.M);
    amgClass.residual(&A, u, rhs, res, comm);
    double dot;
    amgClass.dotProduct(res, res, &dot, comm);
    dot = sqrt(dot);

    double rhsNorm;
    amgClass.dotProduct(rhs, rhs, &rhsNorm, comm);
    rhsNorm = sqrt(rhsNorm);

    double relativeResidual = dot / rhsNorm;
    if(rank==0) cout << "relativeResidual = " << relativeResidual << endl;
*/

    // *************************** tests ****************************

//    std::vector<double> res(A.M);
//    amgClass.residual(&A, u, rhs, res, comm);
//    amgClass.writeVectorToFile(v, A.Mbig, "V", comm);

    // write the solution of overall multigrid to file
//    amgClass.writeVectorToFile(u, A.Mbig, "V", comm);
//    int AMGClass::writeVectorToFile(std::vector<T>& v, unsigned long vSize, string name, MPI_Comm comm) {

    // write the solution of only jacobi to file
//    std::vector<double> uu;
//    uu.assign(A.M, 0);
//    for(i=0; i<10; i++)
//        A.jacobi(uu, rhs, comm);
//    amgClass.writeVectorToFile(uu, A.Mbig, "U", comm);

//    std::vector<double> resCoarse(grids[1].A->M);
//    grids[0].R.matvec(&*rhs.begin(), &*resCoarse.begin(), comm);
//    for(i=0; i<resCoarse.size(); i++)
//        resCoarse[i] = -resCoarse[i];
//    if(rank==3)
//        for(auto i:resCoarse)
//            cout << -i << endl;

//    if(rank==0){
//        cout << "nnz_l = " << grids[0].R.nnz_l << ", nnz_l_local = " << grids[0].R.nnz_l_local << ", nnz_l_remote = " << grids[0].R.nnz_l_remote << endl;
//        for(i=0; i<grids[0].R.entry.size(); i++)
//            cout << grids[0].R.entry[i].row << "\t" << grids[0].R.entry[i].col << "\t" << grids[0].R.entry[i].val << endl;
//    }

//    grids[0].P.matvec(&*resCoarse.begin(), &*u.begin(), comm);
//    if(rank==0)
//        for(i=0; i<u.size(); i++)
//            cout << u[i] << endl;

/*
    // write norm of residual for mutile solve iterations
//    std::vector <long> v1;
//    std::vector <float> v2;
//    std::vector <double> v3;
    double dot;
    std::vector<double> res(A.M);
    std::vector<double> res_norm;
//    string name;
    for(i=0; i<10; i++){
//        name = "V";
//        name += std::to_string(i);
//        name += "_";
        amgClass.AMGSolve(grids, u, rhs, comm);
        amgClass.residual(&A, u, rhs, res, comm);
        amgClass.dotProduct(res, res, &dot, comm);
        res_norm.push_back(sqrt(dot));
//        amgClass.writeVectorToFiled(u, A.Mbig, name, comm);
//        amgClass.test(v1);
    }
    amgClass.writeVectorToFiled(res_norm, res_norm.size(), "res_norm", comm);
*/

    // *************************** write residual or the solution to a file ****************************

//    double dot;
//    std::vector<double> res(A.M);
//    amgClass.residual(&A, u, rhs, res, comm);
//    amgClass.dotProduct(res, res, &dot, comm);
//    if(rank==0) cout << "initial residual = " << sqrt(dot) << endl;

//    A.jacobi(u, rhs, comm);
//    int max = 20;
//    double tol = 1e-12;
//    amgClass.solveCoarsest(&A, u, rhs, max, tol, comm);
//    A.jacobi(u, rhs, comm);

//    amgClass.residual(&A, u, rhs, res, comm);
//    amgClass.dotProduct(res, res, &dot, comm);
//    if(rank==0) cout << "final residual = " << sqrt(dot) << endl;

/*
    char* outFileNameTxt = "u.bin";
    MPI_Status status2;
    MPI_File fh2;
    MPI_Offset offset2;
    MPI_File_open(comm, outFileNameTxt, MPI_MODE_CREATE| MPI_MODE_WRONLY, MPI_INFO_NULL, &fh2);
    offset2 = A.split[rank] * 8; // value(double=8)
    // write the solution
    MPI_File_write_at(fh2, offset2, &*u.begin(), A.M, MPI_DOUBLE, &status2);
    // write the residual
//    MPI_File_write_at(fh2, offset2, &*res.begin(), A.M, MPI_DOUBLE, &status2);
    MPI_File_close(&fh2);
*/

    // *************************** use jacobi to find the answer x ****************************

/*
//    for(unsigned int i=0; i<A.M; i++)
//        v[i] = i + 1 + A.split[rank];

    std::vector<double> res(A.M);
    amgClass.residual(&A, u, rhs, res, comm);
    double dot;
    amgClass.dotProduct(res, res, &dot, comm);
    double initialNorm = sqrt(dot);
    if(rank==0) cout << "\ninitial norm(res) = " << initialNorm << endl;

    // initial x for Ax=b
//    std::vector <double> x(A.M);
//    double* xp = &(*(x.begin()));
//    x.assign(A.M, 0);
    // u first points to the initial guess, after doing jacobi it is the approximate answer for the system
    // vp points to the right-hand side
    int vv = 20;
    for(int i=0; i<vv; i++){
        A.jacobi(u, rhs, comm);
        amgClass.residual(&A, u, rhs, res, comm);
        amgClass.dotProduct(res, res, &dot, comm);
//        if(rank==0) cout << sqrt(dot) << endl;
        if(rank==0) cout << sqrt(dot)/initialNorm << endl;
    }

//    for(auto i:u)
//        cout << i << endl;

//    amgClass.residual(&A, u, rhs, res, comm);
//    if(rank==0)
//        for(auto i:res)
//            cout << i << endl;
//    amgClass.dotProduct(res, res, &dot, comm);
//    if(rank==0) cout << sqrt(dot)/initialNorm << endl;
*/

    // *************************** write the result of jacobi (or its residual) to file ****************************

/*
    char* outFileNameTxt = "jacobi_saena.bin";
    MPI_Status status2;
    MPI_File fh2;
    MPI_Offset offset2;
    MPI_File_open(comm, outFileNameTxt, MPI_MODE_CREATE| MPI_MODE_WRONLY, MPI_INFO_NULL, &fh2);
    offset2 = A.split[rank] * 8; // value(double=8)
    // write the solution
    MPI_File_write_at(fh2, offset2, &*u.begin(), A.M, MPI_DOUBLE, &status2);
    // write the residual
//    MPI_File_write_at(fh2, offset2, &*res.begin(), A.M, MPI_DOUBLE, &status2);
//    int count2;
//    MPI_Get_count(&status2, MPI_UNSIGNED_LONG, &count2);
    //printf("process %d wrote %d lines of triples\n", rank, count2);
    MPI_File_close(&fh2);
*/

    // *************************** matvec ****************************

/*
    std::vector <double> w(A.M);
    double* wp = &(*(w.begin()));
    int time_num = 4; // 4 of them are used to time 3 phases in matvec. check the print section to see how they work.
    double time[time_num]; // array for timing matvec
    fill(&time[0], &time[time_num], 0);
    // warming up
    for(int i=0; i<ITERATIONS; i++){
        A.matvec(vp, wp, time);
        v = w;
    }
    fill(&time[0], &time[time_num], 0);
    MPI_Barrier(comm);
    t1 = MPI_Wtime();
    for(int i=0; i<ITERATIONS; i++){
        A.matvec(vp, wp, time);
        v = w;
//        for(int j=0; j<time_num; j++)
//            time[j] += time[j]/ITERATIONS;
    }
    MPI_Barrier(comm);
    t2 = MPI_Wtime();
    //end of timing matvec
    if (rank==0){
        cout << "Saena matvec total time: " << (time[0]+time[3])/ITERATIONS << endl;
        cout << "phase 0: " << time[0]/ITERATIONS << endl;
        cout << "phase 1: " << (time[3]-time[1]-time[2])/ITERATIONS << endl;
        cout << "phase 2: " << (time[1]+time[2])/ITERATIONS << endl;
    }
    if (rank==0)
        cout << "Matvec in Saena took " << (t2 - t1)/ITERATIONS << " seconds!" << endl;
*/

    // *************************** write the result of matvec to file ****************************

/*
    char* outFileNameTxt = "matvec_result_saena.bin";
    MPI_Status status2;
    MPI_File fh2;
    MPI_Offset offset2;
    MPI_File_open(comm, outFileNameTxt, MPI_MODE_CREATE| MPI_MODE_WRONLY, MPI_INFO_NULL, &fh2);
    offset2 = A.split[rank] * 8; // value(double=8)
    MPI_File_write_at(fh2, offset2, vp, A.M, MPI_UNSIGNED_LONG, &status2);
    int count2;
    MPI_Get_count(&status2, MPI_UNSIGNED_LONG, &count2);
    //printf("process %d wrote %d lines of triples\n", rank, count2);
    MPI_File_close(&fh2);
*/

    // *************************** finalize ****************************

//    MPI_Barrier(comm); cout << rank << "\t*******end*******" << endl;
    MPI_Finalize();
    return 0;
}
