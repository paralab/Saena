#include <iostream>
#include <algorithm>
#include <sys/stat.h>
#include "mpi.h"
#include "coomatrix.h"
#include "AMGClass.h"
#include "strengthmatrix.h"

#define ITERATIONS 1

using namespace std;

int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    COOMatrix B (Aname, Mbig);
    B.MatrixSetup();

    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    if (rank==0)
        cout << "Matrix setup in Saena took " << t2 - t1 << " seconds!" << endl;

    // *************************** AMG ****************************

    int levels          = 2;
    int vcycle_num      = 1;
    double relTol       = 1e-6;
    string relaxType    = "jacobi";
    int preSmooth       = 2;
    int postSmooth      = 2;
    float connStrength  = 0.5; // connection strength parameter
    float tau           = 3; // is used during making aggregates.
    bool doSparsify     = 0;

    AMGClass amgClass (levels, vcycle_num, relTol, relaxType, preSmooth, postSmooth, connStrength, tau);
    amgClass.AMGsetup(&B, doSparsify);







    // *************************** read the vector ****************************

/*    MPI_Status status;
    MPI_File fh;
    MPI_Offset offset;

    int mpiopen = MPI_File_open(MPI_COMM_WORLD, Vname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if(mpiopen){
        if (rank==0) cout << "Unable to open the vector file!" << endl;
        MPI_Finalize();
        return -1;
    }

    // define the size of v as the local number of rows on each process
    std::vector <double> v(B.M);
    double* vp = &(*(v.begin()));

    // vector should have the following format: first line shows the value in row 0, second line shows the value in row 1
    offset = B.split[rank] * 8; // value(double=8)
    MPI_File_read_at(fh, offset, vp, B.M, MPI_UNSIGNED_LONG, &status);

    int count;
    MPI_Get_count(&status, MPI_UNSIGNED_LONG, &count);
    //printf("process %d read %d lines of triples\n", rank, count);
    MPI_File_close(&fh);*/

    // *************************** use jacobi to find the answer x ****************************

/*    for(unsigned int i=0; i<B.M; i++){
        v[i] = i + 1 + B.split[rank];
    }

    // initial x for Ax=b
    std::vector <double> x(B.M);
    double* xp = &(*(x.begin()));
    x.assign(B.M, 0);

    // xp first points to the initial guess, after doing jacobi it is the approximate answer for the system
    // vp points to the right-hand side
    int vv = 40;
    for(int i=0; i<vv; i++)
        B.jacobi(xp, vp);*/

    // *************************** write the result of jacobi to file ****************************

/*    char* outFileNameTxt = "jacobi_saena.bin";

    MPI_Status status2;
    MPI_File fh2;
    MPI_Offset offset2;
    MPI_File_open(MPI_COMM_WORLD, outFileNameTxt, MPI_MODE_CREATE| MPI_MODE_WRONLY, MPI_INFO_NULL, &fh2);

    offset2 = B.split[rank] * 8; // value(double=8)
    MPI_File_write_at(fh2, offset2, xp, B.M, MPI_UNSIGNED_LONG, &status2);

    int count2;
    MPI_Get_count(&status2, MPI_UNSIGNED_LONG, &count2);
    //printf("process %d wrote %d lines of triples\n", rank, count2);
    MPI_File_close(&fh2);*/

    // *************************** matvec ****************************

/*    std::vector <double> w(B.M);
    double* wp = &(*(w.begin()));

    int time_num = 4; // 4 of them are used to time 3 phases in matvec. check the print section to see how they work.
    double time[time_num]; // array for timing matvec
    fill(&time[0], &time[time_num], 0);

    // warming up
    for(int i=0; i<ITERATIONS; i++){
        B.matvec(vp, wp, time);
        v = w;
    }

    fill(&time[0], &time[time_num], 0);
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    for(int i=0; i<ITERATIONS; i++){
        B.matvec(vp, wp, time);
        v = w;

//        for(int j=0; j<time_num; j++)
//            time[j] += time[j]/ITERATIONS;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    //end of timing matvec

    if (rank==0){
        cout << "Saena matvec total time: " << (time[0]+time[3])/ITERATIONS << endl;
        cout << "phase 0: " << time[0]/ITERATIONS << endl;
        cout << "phase 1: " << (time[3]-time[1]-time[2])/ITERATIONS << endl;
        cout << "phase 2: " << (time[1]+time[2])/ITERATIONS << endl;
    }

    if (rank==0)
        cout << "Matvec in Saena took " << (t2 - t1)/ITERATIONS << " seconds!" << endl;*/

    // *************************** write the result of matvec to file ****************************

/*    char* outFileNameTxt = "matvec_result_saena.bin";

    MPI_Status status2;
    MPI_File fh2;
    MPI_Offset offset2;
    MPI_File_open(MPI_COMM_WORLD, outFileNameTxt, MPI_MODE_CREATE| MPI_MODE_WRONLY, MPI_INFO_NULL, &fh2);

    offset2 = B.split[rank] * 8; // value(double=8)
    MPI_File_write_at(fh2, offset2, vp, B.M, MPI_UNSIGNED_LONG, &status2);

    int count2;
    MPI_Get_count(&status2, MPI_UNSIGNED_LONG, &count2);
    //printf("process %d wrote %d lines of triples\n", rank, count2);
    MPI_File_close(&fh2);*/

    MPI_Finalize();
    return 0;
}
