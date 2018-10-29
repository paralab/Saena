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


// to write saena matrix to a file use related function from saena_matrix class.

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