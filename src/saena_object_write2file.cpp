#include "saena_object.h"
#include "saena_matrix.h"
#include "strength_matrix.h"
#include "prolong_matrix.h"
#include "restrict_matrix.h"
#include "grid.h"
#include "aux_functions.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mpi.h>


int saena_object::writeMatrixToFileA(saena_matrix* A, std::string name){
    // Create txt files with name Ac-r0.txt for processor 0, Ac-r1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat Ac-r0.txt Ac-r1.txt > Ac.txt
    // row and column indices of txt files should start from 1, not 0.

    // todo: check global or local index and see if A->split[rank] is required for rows.

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::ofstream outFileTxt;
    std::string outFileNameTxt = "/home/boss/Dropbox/Projects/Saena_base/build/writeMatrix/";
    outFileNameTxt += name;
    outFileNameTxt += "-r";
    outFileNameTxt += std::to_string(rank);
    outFileNameTxt += ".txt";
    outFileTxt.open(outFileNameTxt);

//    if(rank==0)
//        outFileTxt << A->Mbig << "\t" << A->Mbig << "\t" << A->nnz_g << std::endl;

    for (long i = 0; i < A->nnz_l; i++) {
        if(A->entry[i].row == A->entry[i].col)
            continue;

//        std::cout       << A->entry[i].row + 1 << "\t" << A->entry[i].col + 1 << "\t" << A->entry[i].val << std::endl;
        outFileTxt << A->entry[i].row + 1 << "\t" << A->entry[i].col + 1 << "\t" << A->entry[i].val << std::endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

/*
    // this is the code for writing the result of jacobi to a file.
    char* outFileNameTxt = "jacobi_saena.bin";
    MPI_Status status2;
    MPI_File fh2;
    MPI_Offset offset2;
    MPI_File_open(comm, outFileNameTxt, MPI_MODE_CREATE| MPI_MODE_WRONLY, MPI_INFO_NULL, &fh2);
    offset2 = A.split[rank] * 8; // value(double=8)
    MPI_File_write_at(fh2, offset2, xp, A.M, MPI_UNSIGNED_LONG, &status2);
    int count2;
    MPI_Get_count(&status2, MPI_UNSIGNED_LONG, &count2);
    //printf("process %d wrote %d lines of triples\n", rank, count2);
    MPI_File_close(&fh2);
*/

/*
    // failed try to write this part.
    MPI_Status status;
    MPI_File fh;
    MPI_Offset offset;

//    const char* fileName = "/home/abaris/Dropbox/Projects/Saena/build/Ac.txt";
//    const char* fileName = (const char*)malloc(sizeof(const char)*49);
//    fileName = "/home/abaris/Dropbox/Projects/Saena/build/Ac" + "1.txt";
    std::string fileName = "/home/abaris/Acoarse/Ac";
    fileName += std::to_string(7);
    fileName += ".txt";

    int mpierror = MPI_File_open(comm, fileName.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if (mpierror) {
        if (rank == 0) std::cout << "Unable to open the matrix file!" << std::endl;
        MPI_Finalize();
    }

//    std::vector<unsigned int> nnzScan(nprocs);
//    mpierror = MPI_Allgather(&A->nnz_l, 1, MPI_UNSIGNED, &nnzScan[1], 1, MPI_UNSIGNED, comm);
//    if (mpierror) {
//        if (rank == 0) std::cout << "Unable to gather!" << std::endl;
//        MPI_Finalize();
//    }

//    nnzScan[0] = 0;
//    for(unsigned long i=0; i<nprocs; i++){
//        nnzScan[i+1] = nnzScan[i] + nnzScan[i+1];
//        if(rank==1) std::cout << nnzScan[i] << std::endl;
//    }
//    offset = nnzScan[rank];
//    MPI_File_write_at_all(fh, rank, &nnzScan[rank])
//    unsigned int a = 1;
//    double b = 3;
//    MPI_File_write_at(fh, rank, &rank, 1, MPI_INT, &status);
//    MPI_File_write_at_all(fh, offset, &A->entry[0], A->nnz_l, cooEntry::mpi_datatype(), &status);
//    MPI_File_write_at_all(fh, &A->entry[0], A->nnz_l, cooEntry::mpi_datatype(), &status);
//    if (mpierror) {
//        if (rank == 0) std::cout << "Unable to write to the matrix file!" << std::endl;
//        MPI_Finalize();
//    }

//    int count;
//    MPI_Get_count(&status, MPI_UNSIGNED_LONG, &count);
    //printf("process %d read %d lines of triples\n", rank, count);
    MPI_File_close(&fh);
*/

    return 0;
}


int saena_object::writeMatrixToFileP(prolong_matrix* P, std::string name) {
    // Create txt files with name P0.txt for processor 0, P1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat P0.txt P1.txt > P.txt
    // row and column indices of txt files should start from 1, not 0.

    MPI_Comm comm = P->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::ofstream outFileTxt;
    std::string outFileNameTxt = "/home/abaris/Dropbox/Projects/Saena/build/writeMatrix/";
    outFileNameTxt += name;
    outFileNameTxt += std::to_string(rank);
    outFileNameTxt += ".txt";
    outFileTxt.open(outFileNameTxt);

    if (rank == 0)
        outFileTxt << P->Mbig << "\t" << P->Mbig << "\t" << P->nnz_g << std::endl;
    for (long i = 0; i < P->nnz_l; i++) {
//        std::cout       << P->entry[i].row + 1 + P->split[rank] << "\t" << P->entry[i].col + 1 << "\t" << P->entry[i].val << std::endl;
        outFileTxt << P->entry[i].row + 1 + P->split[rank] << "\t" << P->entry[i].col + 1 << "\t" << P->entry[i].val << std::endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

    return 0;
}


int saena_object::writeMatrixToFileR(restrict_matrix* R, std::string name) {
    // Create txt files with name R0.txt for processor 0, R1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat R0.txt R1.txt > R.txt
    // row and column indices of txt files should start from 1, not 0.

    MPI_Comm comm = R->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::ofstream outFileTxt;
    std::string outFileNameTxt = "/home/abaris/Dropbox/Projects/Saena/build/writeMatrix/";
    outFileNameTxt += name;
    outFileNameTxt += std::to_string(rank);
    outFileNameTxt += ".txt";
    outFileTxt.open(outFileNameTxt);

    if (rank == 0)
        outFileTxt << R->Mbig << "\t" << R->Mbig << "\t" << R->nnz_g << std::endl;
    for (long i = 0; i < R->nnz_l; i++) {
//        std::cout       << R->entry[i].row + 1 + R->splitNew[rank] << "\t" << R->entry[i].col + 1 << "\t" << R->entry[i].val << std::endl;
        outFileTxt << R->entry[i].row + 1 +  R->splitNew[rank] << "\t" << R->entry[i].col + 1 << "\t" << R->entry[i].val << std::endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

    return 0;
}


int saena_object::writeVectorToFileul(std::vector<unsigned long>& v, std::string name, MPI_Comm comm) {

    // Create txt files with name name-r0.txt for processor 0, name-r1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat name-r0.txt name-r1.txt > V.txt

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::ofstream outFileTxt;
    std::string outFileNameTxt = "/home/boss/Dropbox/Projects/Saena_base/build/writeMatrix/";
    outFileNameTxt += name;
    outFileNameTxt += "-r";
    outFileNameTxt += std::to_string(rank);
    outFileNameTxt += ".txt";
    outFileTxt.open(outFileNameTxt);

    if (rank == 0)
        outFileTxt << v.size() << std::endl;
    for (long i = 0; i < v.size(); i++) {
//        std::cout       << R->entry[i].row + 1 + R->splitNew[rank] << "\t" << R->entry[i].col + 1 << "\t" << R->entry[i].val << std::endl;
        outFileTxt << v[i]+1 << std::endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

    return 0;
}


int saena_object::writeVectorToFileul2(std::vector<unsigned long>& v, std::string name, MPI_Comm comm) {

    // Create txt files with name name-r0.txt for processor 0, name-r1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat name-r0.txt name-r1.txt > V.txt
    // This version also writes the index number, so it has two columns, instead of 1.

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::ofstream outFileTxt;
    std::string outFileNameTxt = "/home/boss/Dropbox/Projects/Saena_base/build/writeMatrix/";
    outFileNameTxt += name;
    outFileNameTxt += "-r";
    outFileNameTxt += std::to_string(rank);
    outFileNameTxt += ".txt";
    outFileTxt.open(outFileNameTxt);

//    if (rank == 0)
//        outFileTxt << v.size() << std::endl;
    for (long i = 0; i < v.size(); i++) {
        outFileTxt << i+1 << "\t" << v[i]+1 << std::endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

    return 0;
}


int saena_object::writeVectorToFileui(std::vector<unsigned int>& v, std::string name, MPI_Comm comm) {

    // Create txt files with name name-r0.txt for processor 0, name-r1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat name-r0.txt name-r1.txt > V.txt

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::ofstream outFileTxt;
    std::string outFileNameTxt = "/home/boss/Dropbox/Projects/Saena_base/build/writeMatrix/";
    outFileNameTxt += name;
    outFileNameTxt += "-r";
    outFileNameTxt += std::to_string(rank);
    outFileNameTxt += ".txt";
    outFileTxt.open(outFileNameTxt);

    if (rank == 0)
        outFileTxt << v.size() << std::endl;
    for (long i = 0; i < v.size(); i++) {
//        std::cout << v[i] + 1 + split[rank] << std::endl;
        outFileTxt << v[i]+1 << std::endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

    return 0;
}