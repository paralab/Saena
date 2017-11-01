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

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    int assert1, assert2, assert3;
//    unsigned long i;
    bool verbose = false;

    if(verbose) if(rank==0) std::cout << "\nnumber of processes = " << nprocs << std::endl;

    /*
    if(argc != 3)
    {
        if(rank == 0)
        {
            cout << "Usage: ./Saena <2 or 3 for 2d or 3d laplacian> <dof on each proc>" << endl;
//            cout << "Usage: ./Saena <2 or 3 for 2d or 3d laplacian> <dof on each proc> <rhs_vector>" << endl;
        }
        MPI_Finalize();
        return -1;
    }
*/

    if(argc != 3)
    {
        if(rank == 0)
        {
            cout << "Usage: ./Saena <MatrixA> <rhs_vec>" << endl;
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
//    unsigned int num_rows_global = stoul(argv[3]);

    // *************************** initialize the matrix ****************************

    // ******** 1 - initialize the matrix: read from file *************

    char* file_name(argv[1]);
    // timing the matrix setup phase
    double t1 = MPI_Wtime();

    saena::matrix A (file_name, comm);
    A.assemble();

    double t2 = MPI_Wtime();
    if(verbose) print_time(t1, t2, "Matrix Assemble:", comm);

    // ******** 2 - initialize the matrix: use setIJV *************

/*
    char* file_name(argv[1]);

    // set nnz_g for every example.
    unsigned int nnz_g = 65;

    auto initial_nnz_l = (unsigned int) (floor(1.0 * nnz_g / nprocs)); // initial local nnz
    if (rank == nprocs - 1)
        initial_nnz_l = nnz_g - (nprocs - 1) * initial_nnz_l;

    auto* I = (unsigned int*) malloc(sizeof(unsigned int) * initial_nnz_l);
    auto* J = (unsigned int*) malloc(sizeof(unsigned int) * initial_nnz_l);
    auto* V = (double*) malloc(sizeof(double) * initial_nnz_l);
    setIJV(file_name, I, J, V, nnz_g, initial_nnz_l, comm);

    // timing the matrix setup phase
    double t1 = MPI_Wtime();

    saena::matrix A(comm);
    A.add_duplicates(false);
    A.set(I, J, V, initial_nnz_l);


    if(rank==0) A.set(13, 7, -1);
    if(rank==1) A.set(7, 13, -1);


    A.assemble();

    double t2 = MPI_Wtime();
    if(verbose) print_time(t1, t2, "Matrix Assemble:", comm);

    free(I); free(J); free(V);
*/

    // ******** 3 - initialize the matrix: laplacian *************

/*
    int dimension( stoi(argv[1]) );
    unsigned int matrix_size( stoi(argv[2]) );
    MPI_Barrier(comm);
    if(verbose) if(rank==0) printf("%dD Laplacian: dof on each proc = %u \n", dimension, matrix_size);

    // timing the matrix setup phase
    double t1 = MPI_Wtime();

    saena::matrix A(comm);

    if(dimension == 2)
        laplacian2D(&A, matrix_size, comm); // second argument is dof on each processor
    else if(dimension == 3)
        laplacian3D(&A, matrix_size, comm); // second argument is dof on each processor
    else{
        if(rank==0)
            printf("Error: Enter 2 or 3 for the dimension!\n");
        MPI_Finalize();
        return -1;
    }

    A.assemble();

    double t2 = MPI_Wtime();
    if(verbose) print_time(t1, t2, "Matrix Assemble:", comm);
*/

    // ******** write the matrix to file *************

/*
    saena_matrix* B = A.get_internal_matrix();
    std::ofstream outfile;
    std::string file_name = "laplacian";
    file_name += "3D";
    file_name += std::to_string(B->Mbig);
    file_name += "_";
    file_name += std::to_string(rank);
    file_name += ".txt";
    outfile.open(file_name);
    if(rank==0)
        outfile << B->Mbig << "\t" << B->Mbig << "\t" << B->nnz_g << std::endl;
    for(unsigned int i = 0; i < B->nnz_l; i++)
        outfile << B->entry[i].row+1 << "\t" << B->entry[i].col+1 << "\t" << B->entry[i].val << std::endl;
    outfile.close();
*/

    // *************************** set rhs ****************************

    unsigned int num_local_row = A.get_num_local_rows();
    std::vector<double> rhs(num_local_row);

    // ********** 1 - set rhs: use generate_rhs **********

/*
    std::vector<double> v(num_local_row);
    generate_rhs(v, num_local_row);
    A.get_internal_matrix()->matvec(v, rhs);
*/

//    if(rank==0)
//        for(unsigned long i=0; i<rhs.size(); i++)
//            std::cout << rhs[i] << std::endl;

    // ********** 2 - set rhs: read from file **********

    char* Vname(argv[2]);
//    char* Vname(argv[3]);

    // check if the size of rhs match the number of rows of A
    struct stat st;
    stat(Vname, &st);
    unsigned int rhs_size = st.st_size / sizeof(double);
    if(rhs_size != A.get_internal_matrix()->Mbig){
        if(rank==0) printf("Error: Size of RHS does not match the number of rows of the LHS matrix!\n");
        if(rank==0) printf("Number of rows of LHS = %d\n", A.get_internal_matrix()->Mbig);
        if(rank==0) printf("Size of RHS = %d\n", rhs_size);
        MPI_Finalize();
        return -1;
    }

    MPI_Status status;
    MPI_File fh;
    MPI_Offset offset;

    int mpiopen = MPI_File_open(comm, Vname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if(mpiopen){
        if (rank==0) cout << "Unable to open the rhs vector file!" << endl;
        MPI_Finalize();
        return -1;
    }

    // define the size of v as the local number of rows on each process
    std::vector <double> v(num_local_row);
    double* vp = &(*(v.begin()));

    // vector should have the following format: first line shows the value in row 0, second line shows the value in row 1
    offset = A.get_internal_matrix()->split[rank] * 8; // value(double=8)
    MPI_File_read_at(fh, offset, vp, num_local_row, MPI_DOUBLE, &status);

//    int count;
//    MPI_Get_count(&status, MPI_UNSIGNED_LONG, &count);
    //printf("process %d read %d lines of triples\n", rank, count);
    MPI_File_close(&fh);

    // set rhs
    A.get_internal_matrix()->matvec(v, rhs);
//    rhs = v;

    // ********** repartition checking part **********

    // this part is for testing repartition functionality of set_rhs and also set_u and repartition_back_u functions.
/*
//    std::vector<double> rhs;
 //    std::vector<double> u(num_local_row, 0);
//    printf("num_loc_row = %d \n", num_local_row);
    if(rank==0){
        for(i = 0; i < v.size()-3; i++)
            rhs.push_back(v[i]);
    }
    if(rank==1){
        rhs.push_back( (double)(-144.135) );
        rhs.push_back( (double)7862.14 );
        rhs.push_back( (double)45087.3 );
        for(i = 0; i < v.size(); i++)
            rhs.push_back(v[i]);
        rhs.push_back( (double)74109.6 );
        rhs.push_back( (double)(-8738.59) );
        rhs.push_back( (double)29545.4 );
    }
    if(rank==2){
        for(i = 3; i < v.size(); i++)
            rhs.push_back(v[i]);
    }
    printf("rank = %d, rhs = %lu \n", rank, rhs.size());

    std::vector<double> u;
    if(rank==0)
        u.assign(5,0);
    if(rank==1)
        u.assign(11,0);
    if(rank==2)
        u.assign(9,0);
*/

    // ********** 3 - set rhs: use the assign function **********

//    rhs.assign(num_local_row, 1);

    // ********** 4 - set rhs: set one by one **********

/*
    saena_matrix* B = A.get_internal_matrix();
    for(long i=0; i<num_local_row; i++)
        rhs[i] = i + 1 + B->split[rank];
*/

//    for(i=0; i<num_local_row; i++)
//        rhs[i] = 0;

    // ********** print rhs **********

//    if(rank==0)
//        for(i = 0; i < rhs.size(); i++)
//            cout << rhs[i] << endl;

    // *************************** set u0 ****************************
    // there are 3 options for u0:
    // 1- zero
    // 2- random
    // 3- flat from homg

    std::vector<double> u(num_local_row);

    // ********* 1- set u0: use eigenvalues *********

    u.assign(num_local_row, 0); // initial guess = 0

    // ********* 2- set u0: random *********

//    randomVector2(u); // initial guess = random
//    if(rank==1) cout << "\ninitial guess u" << endl;

    // ********* 3- set u0: use eigenvalues *********

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
    MPI_File_read_at(fh3, offset3, &*u.begin(), num_local_row, MPI_DOUBLE, &status3);
    MPI_File_close(&fh3);
*/

    // ********** print u **********

//    if(rank==1){
//        printf("rank = %d \tu.size() = %lu\n", rank, u.size());
//        for(i = 0; i < u.size(); i++)
//            cout << u[i] << endl;
//    }

    // *************************** AMG - Setup ****************************

    t1 = MPI_Wtime();

//    int max_level             = 2; // this is moved to saena_object.
    int vcycle_num            = 20;
    double relative_tolerance = 1e-8;
    std::string smoother      = "chebyshev"; // choices: "jacobi", "chebyshev"
    int preSmooth             = 3;
    int postSmooth            = 3;

    saena::options opts(vcycle_num, relative_tolerance, smoother, preSmooth, postSmooth);
//    saena::options opts((char*)"options001.xml");
//    saena::options opts;
    saena::amg solver;
    solver.set_verbose(verbose); // set verbose at the beginning of the main function.
    solver.set_matrix(&A);
    solver.set_rhs(rhs);

    t2 = MPI_Wtime();
    if(solver.verbose) print_time(t1, t2, "Setup:", comm);

//    MPI_Barrier(comm);
//    for(int i=0; i<maxLevel; i++)
//        if(rank==0) cout << "size = " << maxLevel << ", current level = " << grids[i].currentLevel << ", coarse level = " << grids[i].coarseGrid->currentLevel
//                         << ", num_local_rowbig = " << grids[i].A->Mbig << ", num_local_row = " << grids[i].A->M << ", Ac.Mbig = " << grids[i].Ac.Mbig << ", Ac.M = " << grids[i].Ac.M << endl;
//    MPI_Barrier(comm);

    // *************************** AMG - Solve ****************************

    t1 = MPI_Wtime();

    solver.solve(u, &opts);

    t2 = MPI_Wtime();
    if(solver.verbose) print_time(t1, t2, "Solve:", comm);

    // print A*u
//    std::vector<double> temp1(num_local_row);
//    A.get_internal_matrix()->matvec(u, temp1);
//    MPI_Barrier(comm);
//    if(rank==0){
//        printf("\nrank = %d \ttemp1.size() = %lu \n", rank, temp1.size());
//        for(i = 0; i < temp1.size(); i++)
//            cout << i << "\t" << u[i] << endl;}
//    MPI_Barrier(comm);



//    MPI_Barrier(comm);
//    if(rank==0){
//        printf("\nrank = %d \tu.size() = %lu \n", rank, u.size());
//        for(i = 0; i < u.size(); i++)
//            cout << i << "\t" << u[i] << endl;}
//    MPI_Barrier(comm);
//    if(rank==1){
//        printf("\nrank = %d \tu.size() = %lu \n", rank, u.size());
//        for(i = 0; i < u.size(); i++)
//            cout << i << "\t" << u[i] << endl;}
//    MPI_Barrier(comm);

    // *************************** Residual ****************************

/*
    std::vector<double> res(num_local_row);
    Saena1.residual(&A, u, rhs, res, comm);
    double dot;
    Saena1.dotProduct(res, res, &dot, comm);
    dot = sqrt(dot);

    double rhsNorm;
    Saena1.dotProduct(rhs, rhs, &rhsNorm, comm);
    rhsNorm = sqrt(rhsNorm);

    double relativeResidual = dot / rhsNorm;
    if(rank==0) cout << "relativeResidual = " << relativeResidual << endl;
*/

    // *************************** tests ****************************

//    std::vector<double> res(num_local_row);
//    Saena1.residual(&A, u, rhs, res, comm);
//    Saena1.writeVectorToFile(v, num_local_rowbig, "V", comm);

    // write the solution of overall multigrid to file
//    Saena1.writeVectorToFile(u, num_local_rowbig, "V", comm);
//    int Saena1::writeVectorToFile(std::vector<T>& v, unsigned long vSize, std::string name, MPI_Comm comm) {

    // write the solution of only jacobi to file
//    std::vector<double> uu;
//    uu.assign(num_local_row, 0);
//    for(i=0; i<10; i++)
//        A.jacobi(uu, rhs, comm);
//    Saena1.writeVectorToFile(uu, num_local_rowbig, "U", comm);

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

    //jacobi
//    std::vector<double> temp(num_local_row);
//    saena_matrix* B = A.get_internal_matrix();
//    for (int i = 0; i < postSmooth; i++)
//        B->jacobi(postSmooth, u, rhs, temp);

/*
    //chebyshev
    saena_matrix* B = A.get_internal_matrix();
    std::vector<double> uu(num_local_row, 0);
    std::vector<double> temp1(num_local_row);
    std::vector<double> temp2(num_local_row);
    B->chebyshev(1, uu, rhs, temp1, temp2);

//    if(rank==0)
//        for(long i=0; i<uu.size(); i++)
//            cout << i << "\t" << uu[i] << endl;
*/

    // *********** write norm of residual for mutiple solve iterations ***********
    /*
    double dot;
    std::vector<double> res(num_local_row);
    std::vector<double> res_norm;
    saena_matrix* B = A.get_internal_matrix();

    // add initial res
    B->residual(u, rhs, res);
    dotProduct(res, res, &dot, comm);
    res_norm.push_back(sqrt(dot));

    for(i=0; i<20; i++){
        solver.solve(u, &opts);
        B->residual(u, rhs, res);
        dotProduct(res, res, &dot, comm);
        res_norm.push_back(sqrt(dot));
    }
    if(rank==0) writeVectorToFiled(res_norm, res_norm.size(), "res_norm", comm);
*/

//    Saena1.writeMatrixToFileA(grids[1].A, "Ac", comm);
//    Saena1.writeMatrixToFileP(&grids[0].P, "P", comm);
//    Saena1.writeMatrixToFileR(&grids[0].R, "R", comm);

    // *************************** write residual or the solution to a file ****************************

//    double dot;
//    std::vector<double> res(num_local_row);
//    Saena1.residual(&A, u, rhs, res, comm);
//    Saena1.dotProduct(res, res, &dot, comm);
//    if(rank==0) cout << "initial residual = " << sqrt(dot) << endl;

//    A.jacobi(u, rhs, comm);
//    int max = 20;
//    double tol = 1e-12;
//    Saena1.solveCoarsest(&A, u, rhs, max, tol, comm);
//    A.jacobi(u, rhs, comm);

//    Saena1.residual(&A, u, rhs, res, comm);
//    Saena1.dotProduct(res, res, &dot, comm);
//    if(rank==0) cout << "final residual = " << sqrt(dot) << endl;

/*
    char* outFileNameTxt = "u.bin";
    MPI_Status status2;
    MPI_File fh2;
    MPI_Offset offset2;
    MPI_File_open(comm, outFileNameTxt, MPI_MODE_CREATE| MPI_MODE_WRONLY, MPI_INFO_NULL, &fh2);
    offset2 = A.split[rank] * 8; // value(double=8)
    // write the solution
    MPI_File_write_at(fh2, offset2, &*u.begin(), num_local_row, MPI_DOUBLE, &status2);
    // write the residual
//    MPI_File_write_at(fh2, offset2, &*res.begin(), num_local_row, MPI_DOUBLE, &status2);
    MPI_File_close(&fh2);
*/

    // *************************** use jacobi to find the answer x ****************************

/*
//    for(unsigned int i=0; i<num_local_row; i++)
//        v[i] = i + 1 + A.split[rank];

    std::vector<double> res(num_local_row);
    Saena1.residual(&A, u, rhs, res, comm);
    double dot;
    Saena1.dotProduct(res, res, &dot, comm);
    double initialNorm = sqrt(dot);
    if(rank==0) cout << "\ninitial norm(res) = " << initialNorm << endl;

    // initial x for Ax=b
//    std::vector <double> x(num_local_row);
//    double* xp = &(*(x.begin()));
//    x.assign(num_local_row, 0);
    // u first points to the initial guess, after doing jacobi it is the approximate answer for the system
    // vp points to the right-hand side
    int vv = 20;
    for(int i=0; i<vv; i++){
        A.jacobi(u, rhs, comm);
        Saena1.residual(&A, u, rhs, res, comm);
        Saena1.dotProduct(res, res, &dot, comm);
//        if(rank==0) cout << sqrt(dot) << endl;
        if(rank==0) cout << sqrt(dot)/initialNorm << endl;
    }

//    for(auto i:u)
//        cout << i << endl;

//    Saena1.residual(&A, u, rhs, res, comm);
//    if(rank==0)
//        for(auto i:res)
//            cout << i << endl;
//    Saena1.dotProduct(res, res, &dot, comm);
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
    MPI_File_write_at(fh2, offset2, &*u.begin(), num_local_row, MPI_DOUBLE, &status2);
    // write the residual
//    MPI_File_write_at(fh2, offset2, &*res.begin(), num_local_row, MPI_DOUBLE, &status2);
//    int count2;
//    MPI_Get_count(&status2, MPI_UNSIGNED_LONG, &count2);
    //printf("process %d wrote %d lines of triples\n", rank, count2);
    MPI_File_close(&fh2);
*/

    // *************************** matvec ****************************

/*
    std::vector <double> w(num_local_row);
    double* wp = &(*(w.begin()));
    int time_num = 4; // 4 of them are used to time 3 phases in matvec. check the print section to see how they work.
    double time[time_num]; // array for timing matvec
    fill(&time[0], &time[time_num], 0);
    // warming up
    for(int i=0; i<ITERATIONS; i++){
        num_local_rowatvec(vp, wp, time);
        v = w;
    }
    fill(&time[0], &time[time_num], 0);
    MPI_Barrier(comm);
    t1 = MPI_Wtime();
    for(int i=0; i<ITERATIONS; i++){
        num_local_rowatvec(vp, wp, time);
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
    MPI_File_write_at(fh2, offset2, vp, num_local_row, MPI_UNSIGNED_LONG, &status2);
    int count2;
    MPI_Get_count(&status2, MPI_UNSIGNED_LONG, &count2);
    //printf("process %d wrote %d lines of triples\n", rank, count2);
    MPI_File_close(&fh2);
*/

    // *************************** finalize ****************************

    A.destroy();
    solver.destroy();
    MPI_Finalize();
    return 0;
}
