#include <iostream>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include "mpi.h"
#include <vector>

#include "grid.h"
#include "saena.hpp"
#include <El.hpp>

//using namespace std;

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
    if(argc != 4)
    {
        if(rank == 0)
            std::cout << "Usage to make a 3DLaplacian: ./Saena <x grid size> <y grid size> <z grid size>" << std::endl;
        MPI_Finalize();
        return -1;
    }
*/
    if(argc != 3)
    {
        if(rank == 0)
        {
            std::cout << "Usage: ./Saena <MatrixA> <rhs_vec>" << std::endl;
            std::cout << "Matrix file should be in triples format." << std::endl;
        }
        MPI_Finalize();
        return -1;
    }
/*
    if(argc != 4)
    {
        if(rank == 0)
        {
            std::cout << "Usage: ./Saena <MatrixA> <rhs_vec> <MatrixA_new>" << std::endl;
            std::cout << "Matrix file should be in triples format." << std::endl;
        }
        MPI_Finalize();
        return -1;
    }
*/

    // *************************** get number of rows ****************************

//    char* Vname(argv[2]);
//    struct stat vst;
//    stat(Vname, &vst);
//    unsigned int Mbig = vst.st_size/8;  // sizeof(double) = 8
//    unsigned int num_rows_global = stoul(argv[3]);

    // *************************** initialize the matrix ****************************

    // ******** 1 - initialize the matrix: laplacian *************
/*
    unsigned int mx(std::stoi(argv[1]));
    unsigned int my(std::stoi(argv[2]));
    unsigned int mz(std::stoi(argv[3]));

    if(verbose){
        MPI_Barrier(comm);
        if(rank==0) printf("3D Laplacian: grid size: x = %d, y = %d, z = %d \n", mx, my, mz);
        MPI_Barrier(comm);}

    // timing the matrix setup phase
    double t1 = MPI_Wtime();

    saena::matrix A(comm);
//    saena::laplacian2D(&A, 4, comm);
    saena::laplacian3D(&A, mx, my, mz, comm);

    double t2 = MPI_Wtime();
    if(verbose) print_time(t1, t2, "Matrix Assemble:", comm);
*/

    // ******** 2 - initialize the matrix: read from file *************

    char* file_name(argv[1]);
    // timing the matrix setup phase
    double t1 = MPI_Wtime();

    saena::matrix A (file_name, comm);
    A.assemble();

    double t2 = MPI_Wtime();
    if(verbose) print_time(t1, t2, "Matrix Assemble:", comm);

    // ******** 3 - initialize the matrix: use setIJV *************

/*
    char* file_name(argv[1]);

    // set nnz_g for every example.
    unsigned int nnz_g = 36216;

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

//    if(rank==0) A.set(13, 7, -1);
//    if(rank==1) A.set(7, 13, -1);

    A.assemble();

    double t2 = MPI_Wtime();
    if(verbose) print_time(t1, t2, "Matrix Assemble:", comm);

    free(I); free(J); free(V);
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

//    MPI_Barrier(comm);
//    saena_matrix* B = A.get_internal_matrix();
//    if(rank==0){
//        printf("split_old:\n");
//        for(int i=0; i<nprocs+1; i++)
//            printf("%lu \n", B->split[i]);
//    }
//    MPI_Barrier(comm);

//    std::vector<double> rhs;

    // ********** 1 - set rhs: use generate_rhs **********
/*
    std::vector<double> v(num_local_row);
    generate_rhs_old(v);
    A.get_internal_matrix()->matvec(v, rhs);
*/
//    generate_rhs(rhs, mx, my, mz, comm);

//    if(rank==0)
//        for(unsigned long i=0; i<rhs.size(); i++)
//            std::cout << i << "\t" << rhs[i] << std::endl;

    // repartition rhs to have the same partition as the matrix A
    // ----------------------------------------------------------
/*
    saena_matrix *B = A.get_internal_matrix();

    std::vector<double> rhs_temp = rhs;
//    rhs.swap(rhs_temp);

//    if(rank==0)
//        for(unsigned long i=0; i<rhs.size(); i++)
//            std::cout << i << "\t" << rhs_temp[i] << std::endl;

//    int recv_size = B->split[rank+1] - B->split[rank];
//    MPI_Barrier(comm); printf("\nrank = %d, recv_size = %u\n", rank, recv_size); MPI_Barrier(comm);

    std::vector<int> recv_size_array(nprocs);
    for(unsigned int i = 0; i < nprocs; i++)
        recv_size_array[i] = B->split[i+1] - B->split[i];

    if(rank==0){
        printf("\nrecv_size_array:\n");
        for(int i = 0; i < nprocs; i++)
            printf("%d \t %u\n", i, recv_size_array[i]);}

    // rdispls is just B->split
    std::vector<int> rdispls(nprocs);
    rdispls[0] = 0;
    for(unsigned int i = 1; i < nprocs; i++)
        rdispls[i] = rdispls[i-1] + recv_size_array[i-1];

    if(rank==0){
        printf("\nrdispls:\n");
        for(int i = 0; i < nprocs; i++)
            printf("%d \t %d\n", i, rdispls[i]);}

    int send_size = rhs.size();
    MPI_Barrier(comm); printf("\nrank = %d, send_size = %d\n", rank, send_size); MPI_Barrier(comm);

    std::vector<int> send_size_array(nprocs);
    MPI_Allgather(&send_size, 1, MPI_INT, &*send_size_array.begin(), 1, MPI_INT, comm);

    if(rank==0){
        printf("\nsend_size_array:\n");
        for(int i = 0; i < nprocs; i++)
            printf("%d \t %u\n", i, send_size_array[i]);}

    std::vector<int> sdispls(nprocs);
    sdispls[0] = 0;
    for(unsigned int i = 1; i < nprocs; i++)
        sdispls[i] = sdispls[i-1] + send_size_array[i-1];

    if(rank==0){
        printf("\nsdispls:\n");
        for(int i = 0; i < nprocs; i++)
            printf("%d \t %d\n", i, sdispls[i]);}



    MPI_Barrier(comm); printf("\nrank = %d, rhs.resize = %lu\n", rank, B->split[rank+1] - B->split[rank]); MPI_Barrier(comm);



    rhs.clear();
    rhs.resize(B->split[rank+1] - B->split[rank]);

    MPI_Alltoallv(&*rhs_temp.begin(), &send_size_array[0], &sdispls[0], MPI_DOUBLE,
                  &*rhs.begin(), &recv_size_array[0], &rdispls[0], MPI_DOUBLE, comm);

    std::cout << rhs[0] << std::endl;

    return 0;

    if(rhs.size() != num_local_row){
        printf("error: rhs on process %d has not the correct length!\n", rank);
        printf("rhs size on process %d = %lu, but it should be of size %d  \n", rank, rhs.size(), num_local_row);
        A.destroy();
        MPI_Finalize();
        return -1;
    }

    if(rank==0)
        for(unsigned long i=0; i<rhs.size(); i++)
            std::cout << i << "\t" << rhs[i] << std::endl;
*/

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
        if (rank==0) std::cout << "Unable to open the rhs vector file!" << std::endl;
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
//    A.get_internal_matrix()->matvec(v, rhs);W
    rhs = v;

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
    // 1- default is zero
    // 2- random
    // 3- flat from homg

    std::vector<double> u(num_local_row, 0);

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
    int vcycle_num            = 100;
    double relative_tolerance = 1e-12;
    std::string smoother      = "chebyshev"; // choices: "jacobi", "chebyshev"
    int preSmooth             = 3;
    int postSmooth            = 3;

    saena::options opts(vcycle_num, relative_tolerance, smoother, preSmooth, postSmooth);
//    saena::options opts((char*)"options001.xml");
//    saena::options opts;
    saena::amg solver;
    solver.set_verbose(verbose); // set verbose at the beginning of the main function.
//    solver.set_multigrid_max_level(0); // 0 means only use direct solver, so no multigrid will be used.

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

//    solver.solve(u, &opts);
    solver.solve_pcg(u, &opts);

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

    // *************************** test set3() ****************************

//    saena_matrix *B = A.get_internal_matrix();
//    for(long i = 0; i < B->nnz_l; i++)
//        std::cout << B->entry[i] << std::endl;

//    A.add_duplicates(true);
//    A.set(8, 7, 100);
//    A.set(2, 1, 100);

//    for(long i = 0; i < B->nnz_l; i++)
//        std::cout << B->entry[i] << std::endl;
/*
    unsigned int row[3] = {9,   8,   20};
    unsigned int col[3] = {8,   7,   18};
    double val[3]       = {100.111, 200.222, 300.333};
    A.set(row, col, val, 3);
    A.assemble();
*/
//    saena_matrix *B = A.get_internal_matrix();
//    for(long i = 0; i < B->nnz_l; i++)
//        std::cout << B->entry[i] << std::endl;

    // *************************** LHS update Experiment ****************************

    // try this: ./Saena ./data/25o1s4.bin ./data/vectors/v25.bin ./data/25o1s4_2.bin
    // or:       ./Saena ./data/81s4x8o1mu1.bin ./data/vectors/v81.bin ./data/81s4x8o1mu1_2.bin
    // or:       ./Saena ./data/2DMed_sorted.bin ./data/vectors/v961.bin ./data/2DMed_sorted_2.bin
/*
    char* file_name2(argv[3]);
    saena::matrix A_new (file_name2, comm);
    A_new.assemble();
*/

/*
    saena_matrix *B = A_new.get_internal_matrix();
//    printf("A_new:\n%u \t%u \t%u \t%u \n", B->Mbig, B->M, B->nnz_g, B->nnz_l);
//    printf("A before: rank %d \n%u \t%u \t%u \t%u \n", rank, B->Mbig, B->M, B->nnz_g, B->nnz_l);
    for (long i = 0; i < B->nnz_l; i++)
        A.set(B->entry[i].row, B->entry[i].col, B->entry[i].val);
    A.assemble();
//    printf("A after: rank %d \n%u \t%u \t%u \t%u \n", rank, B->Mbig, B->M, B->nnz_g, B->nnz_l);
*/

//    if(rank==0) printf("A after nnz: \n");
//    saena_matrix *C = A.get_internal_matrix();
//    if(rank==0) printf("nnz_l_local = %d \tnnz_l_remote = %d \tvalues_local = %lu \tvalues_remote = %lu \n",
//                       C->nnz_l_local, C->nnz_l_remote, C->values_local.size(), C->values_remote.size());

//    printf("A_new:\n");
//    for(long i = 0; i < B->nnz_l_remote; i++)
//        printf("%lu \t%lu \t%f \n", B->row_local[i], B->col_local[i], B->values_local[i]);

//    printf("\nA:\n");
//    for(long i = 0; i < C->nnz_l_remote; i++)
//        printf("%lu \t%lu \t%f \n", C->row_local[i], C->col_local[i], C->values_local[i]);

//    u.assign(num_local_row, 0);
//    solver.solve_pcg_update(u, &opts, &A_new);
//    solver.solve_pcg_update2(u, &opts, &A_new);
//    solver.solve_pcg_update3(u, &opts, &A_new);
//    solver.solve_pcg_update4(u, &opts, &A_new);

//    printf("\nprint u:\n");
//    for(long i = 0; i < num_local_row; i++)
//        printf("%f\n", u[i]);


    // *************************** Matvec Expermient ****************************

    // Saena Matvec
    // ------------
/*
    int matvec_iter = 3;
    std::vector<double> Av(num_local_row);

    saena_matrix* B = A.get_internal_matrix();
    t1 = MPI_Wtime();

    for(int i = 0; i < matvec_iter ; i++){
        B->matvec(rhs, Av);
        for(int j = 0; j < num_local_row ; j++)
            rhs[j] = Av[j];
    }

    t2 = MPI_Wtime();
    print_time_average(t1, t2, "Saena Matvec:", matvec_iter, comm);

    if(rank==2){
        std::cout << "Saena matvec:" << std::endl;
        for(unsigned long i = 0; i < num_local_row; i++)
            std::cout << i + B->split[rank] << "\t" << Av[i] << std::endl;}
*/

    // Elemental Matvec
    // ----------------
/*
    El::Initialize( argc, argv );

    const El::Int n = B->Mbig;

    El::DistMatrix<double> C(n,n);
    El::Zero( C );
    C.Reserve(B->nnz_l);
    for(unsigned long i = 0; i < B->nnz_l; i++){
//        if(rank==1) std::cout << entry[i].row << "\t" << entry[i].col << "\t" << entry[i].val << std::endl;
        C.QueueUpdate(B->entry[i].row, B->entry[i].col, B->entry[i].val);
    }
    C.ProcessQueues();

    El::DistMatrix<double> w(n,1), y(n,1);
    w.Reserve(num_local_row);
    for(unsigned long i = 0; i < num_local_row; i++){
//        if(rank==0) std::cout << i+B->split[rank] << "\t" << 1 << "\t" << v[i] << std::endl;
        w.QueueUpdate(i+B->split[rank], 0, v[i]);
    }
    w.ProcessQueues();

    const char uploChar = 'L';
    const El::UpperOrLower uplo = El::CharToUpperOrLower( uploChar );
    double alpha = 1;
    double beta = 0;

    t1 = MPI_Wtime();

    for(int i = 0; i < matvec_iter ; i++)
        El::Symv( uplo, alpha, C, w, beta, y);

    t2 = MPI_Wtime();
    print_time_average(t1, t2, "Elemental Matvec:", matvec_iter, comm);

//    El::Print(y, "Elemental matvec: ");

    El::Finalize();
*/
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
//    A_new.destroy();
    solver.destroy();
    MPI_Finalize();
    return 0;
}
