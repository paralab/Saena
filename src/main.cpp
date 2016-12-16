#include <iostream>
#include <fstream>
#include <algorithm>
#include "coomatrix.h"
//#include "cscmatrix.h"
//#include "csrmatrix.h"
#include <sys/time.h>
#include "mpi.h"

#define ITERATIONS 100

using namespace std;

int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //const char* Aname = "/home/abaris/Dropbox/Projects/Saena/data/ch3-3-b1/ch3-3-b1.bin";
    //char* Aname2 = "/home/abaris/Dropbox/Projects/Saena/data/ch3-3-b1/ch3-3-b1.bin";

    if(argc < 4)
    {
        if(rank == 0)
        {
            cout << "Usage: ./Saena <MatrixA> <vecX> <#rows of A>" << endl;
            cout << "<MatrixA> is absolute address, and files should be in triples format" << endl;
        }
        MPI_Finalize();
        return -1;
    }

    char* Aname(argv[1]);
    long Mbig = stol(argv[3]);

/*    if (Mbig%(nprocs*nprocs) != 0){
        if (rank==0)
            cout << "This code only works when the number of rows are divisible by (number of prcessors)^2" << endl;

        MPI_Finalize();
        return -1;
    }*/


    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    COOMatrix B (Aname, Mbig);
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();

    if (rank==0)
        cout << "Setup in Saena took " << t2 - t1 << " seconds!" << endl;

    char* Vname(argv[2]);
    MPI_Status status;
    MPI_File fh;
    MPI_Offset offset;

    MPI_File_open(MPI_COMM_WORLD, Vname, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    double* v = (double*) malloc(sizeof(double) * B.M);

    // vector should have the following format: first line shows the value in row 0, second line shows the value in row 1
    offset = B.split[rank] * 8; // value(double=8)
    MPI_File_read_at(fh, offset, v, B.M, MPI_UNSIGNED_LONG, &status);

    int count;
    MPI_Get_count(&status, MPI_UNSIGNED_LONG, &count);
    //printf("process %d read %d lines of triples\n", rank, count);
    MPI_File_close(&fh);

/*    srand (time(NULL));
    for (long i=0; i<B.M; i++){
        v[i] = rand();
    }*/

    double* w = (double*) malloc(sizeof(double) * B.M);

    // warming up
    for(int i=0; i<ITERATIONS; i++){
        B.matvec(v, w);
        v = w;
    }

    // timing matvec
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    for(int i=0; i<ITERATIONS; i++){
        B.matvec(v, w);
        v = w;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    //end of timing matvec

    if (rank==0)
        cout << "Matvec in Saena took " << (t2 - t1)/ITERATIONS << " seconds!" << endl;


/*
    if (rank == 0) cout << endl << "result of matvec: " << endl;
    for(unsigned int i=0;i<M;i++)
        cout << w[i] << endl;
*/

    // write the result of the matvec
    // txt file
    char* outFileNameTxt = "matvec_result_saena.bin";

    MPI_Status status2;
    MPI_File fh2;
    MPI_Offset offset2;

    MPI_File_open(MPI_COMM_WORLD, outFileNameTxt, MPI_MODE_CREATE| MPI_MODE_WRONLY, MPI_INFO_NULL, &fh2);

    offset2 = B.split[rank] * 8; // value(double=8)
    MPI_File_write_at(fh2, offset2, v, B.M, MPI_UNSIGNED_LONG, &status2);

    int count2;
    MPI_Get_count(&status2, MPI_UNSIGNED_LONG, &count2);
    //printf("process %d wrote %d lines of triples\n", rank, count2);
    MPI_File_close(&fh2);

    free(v);
    //free(w);
    //free(Aname);
    //free(Aname2);

    MPI_Finalize();
    return 0;
}
