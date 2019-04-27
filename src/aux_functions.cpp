#include <aux_functions.h>
#include "saena_matrix.h"

#include <iostream>
#include <random>
#include <fstream>
#include <math.h>
#include <sys/stat.h>

class saena_matrix;


void setIJV(char* file_name, index_t *I, index_t *J, value_t *V, nnz_t nnz_g, nnz_t initial_nnz_l, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    if(rank==0) printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    if(rank==0) printf("ERROR: change datatypes for function setIJV!!!");
    if(rank==0) printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");

    std::vector<unsigned long> data(2*initial_nnz_l); // 2 is for i and j and val, i and j are uint so both of them are like 1 ulong.
    unsigned long* datap;
    if(initial_nnz_l != 0)
        datap = &data[0];
    else
        printf("Error: initial_nnz_l is 0 in setIJV().\n");

    // *************************** read the matrix ****************************

    MPI_Status status;
    MPI_File fh;
    MPI_Offset offset;

    int mpiopen = MPI_File_open(comm, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (mpiopen) {
        if (rank == 0) std::cout << "Unable to open the matrix file!" << std::endl;
        MPI_Finalize();
    }

    offset = rank * (unsigned int) (floor(1.0 * nnz_g / nprocs)) * 24; // row index(long=8) + column index(long=8) + value(double=8) = 24
    MPI_File_read_at(fh, offset, datap, 3 * initial_nnz_l, MPI_UNSIGNED_LONG, &status);
    MPI_File_close(&fh);

    for(unsigned int i = 0; i<initial_nnz_l; i++){
        I[i] = data[3*i];
        J[i] = data[3*i+1];
        V[i] = reinterpret_cast<double&>(data[3*i+2]);
    }
}


int dotProduct(std::vector<value_t>& r, std::vector<value_t>& s, value_t* dot, MPI_Comm comm){

    double dot_l = 0;
    for(index_t i=0; i<r.size(); i++)
        dot_l += r[i] * s[i];
    MPI_Allreduce(&dot_l, dot, 1, MPI_DOUBLE, MPI_SUM, comm);

    return 0;
}


double print_time(double t_start, double t_end, std::string function_name, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    double min, max, average;
    double t_dif = t_end - t_start;

    MPI_Reduce(&t_dif, &min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&t_dif, &max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&t_dif, &average, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    average /= nprocs;

    if (rank==0)
        std::cout << std::endl << function_name << "\nmin: " << min << "\nave: " << average << "\nmax: " << max << std::endl << std::endl;

    return average;
}


double print_time(double t_dif, std::string function_name, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    double min, max, average;
//    double t_dif = t2 - t1;

    MPI_Reduce(&t_dif, &min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&t_dif, &max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&t_dif, &average, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    average /= nprocs;

    if (rank==0)
        std::cout << std::endl << function_name << "\nmin: " << min << "\nave: " << average << "\nmax: " << max << std::endl << std::endl;

    return average;
}


double print_time_ave(double t_dif, std::string function_name, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    double average;
    MPI_Reduce(&t_dif, &average, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    average /= nprocs;

    if (rank==0)
        std::cout << std::endl << function_name << "\n" << average << std::endl;

    return average;
}


double print_time_ave_consecutive(double t_dif, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    double average;
    MPI_Reduce(&t_dif, &average, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    average /= nprocs;

//    if (rank==0)
//        std::cout << average << "+";

    return average;
}


int read_vector_file(std::vector<value_t>& v, saena_matrix *A, char *file, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // check if the size of rhs match the number of rows of A
    struct stat st;
    stat(file, &st);
    unsigned int rhs_size = st.st_size / sizeof(double);
    if(rhs_size != A->Mbig){
        if(rank==0) printf("Error: Size of RHS does not match the number of rows of the LHS matrix!\n");
        if(rank==0) printf("Number of rows of LHS = %d\n", A->Mbig);
        if(rank==0) printf("Size of RHS = %d\n", rhs_size);
        MPI_Finalize();
        return -1;
    }

    MPI_Status status;
    MPI_File fh;
    MPI_Offset offset;

    int mpiopen = MPI_File_open(comm, file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if(mpiopen){
        if (rank==0) std::cout << "Unable to open the rhs vector file!" << std::endl;
        MPI_Finalize();
        return -1;
    }

    // define the size of v as the local number of rows on each process
//    std::vector <double> v(A.M);
    v.resize(A->M);
    double* vp = &(*(v.begin()));

    // vector should have the following format: first line shows the value in row 0, second line shows the value in row 1
    offset = A->split[rank] * 8; // value(double=8)
    MPI_File_read_at(fh, offset, vp, A->M, MPI_DOUBLE, &status);

//    int count;
//    MPI_Get_count(&status, MPI_UNSIGNED_LONG, &count);
    //printf("process %d read %d lines of triples\n", rank, count);
    MPI_File_close(&fh);

//    print_vector(v, -1, "v", comm);

    return 0;
}


//template <class T>
int write_vector_file_d(std::vector<value_t>& v, index_t vSize, std::string name, MPI_Comm comm) {

    // Create txt files with name name0.txt for processor 0, name1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat name0.txt name1.txt > V.txt

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
        outFileTxt << vSize << std::endl;
    for (long i = 0; i < v.size(); i++) {
//        std::cout       << R->entry[i].row + 1 + R->splitNew[rank] << "\t" << R->entry[i].col + 1 << "\t" << R->entry[i].val << std::endl;
        outFileTxt << v[i] << std::endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

    return 0;
}


int generate_rhs(std::vector<value_t>& rhs, index_t mx, index_t my, index_t mz, MPI_Comm comm) {

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    bool verbose_gen_rhs = true;

    if(rank==0) printf("change datatypes for function generate_rhs!!!");

    if(verbose_gen_rhs){
        MPI_Barrier(comm);
        printf("rank = %d, generate_rhs: start \n", rank);
        MPI_Barrier(comm);}

    int       i,j,k,xm,ym,zm,xs,ys,zs;
    double    Hx,Hy,Hz;
//    double    ***array;
    unsigned int node;

#define PETSC_PI 3.14159265358979323846

    Hx   = 1.0 / (double)(mx);
    Hy   = 1.0 / (double)(my);
    Hz   = 1.0 / (double)(mz);

    // split the 3D grid by only the z axis. So put the whole x and y grids on processors, but split z by the number of processors.
    xs = 0;
    xm = mx;
    ys = 0;
    ym = my;
    zm = (int)floor(mz / nprocs);
    zs = rank * zm;
    if(rank == nprocs - 1)
        zm = mz - ( (nprocs - 1) * zm);

    if(verbose_gen_rhs){
        MPI_Barrier(comm);
        printf("rank = %d, corners: \nxs = %d, ys = %d, zs = %d, xm = %d, ym = %d, zm = %d\n", rank, xs, ys, zs, xm, ym, zm);
        MPI_Barrier(comm);}

    double val;
    for (k=zs; k<zs+zm; k++) {
        for (j=ys; j<ys+ym; j++) {
            for (i=xs; i<xs+xm; i++) {
                node = mx * my * k + mx * j + i; // for 2D it should be = mx * j + i
                if(rank==0) printf("node = %u\n", node);
                val = 12 * PETSC_PI * PETSC_PI
                                 * cos(2*PETSC_PI*(((double)i+0.5)*Hx))
                                 * cos(2*PETSC_PI*(((double)j+0.5)*Hy))
                                 * cos(2*PETSC_PI*(((double)k+0.5)*Hz))
                                 * Hx * Hy * Hz;
                rhs.emplace_back(val);
            }
        }
    }

    if(verbose_gen_rhs){
        MPI_Barrier(comm);
        printf("rank = %d, generate_rhs: end \n", rank);
        MPI_Barrier(comm);}

    return 0;
}


int generate_rhs_old(std::vector<value_t>& rhs){

    index_t size = rhs.size();

    //Type of random number distribution
    std::uniform_real_distribution<value_t> dist(0, 1); //(min, max)
    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng;
    //Initialize with non-deterministic seeds
    rng.seed(std::random_device{}());

    for (index_t i=0; i<size; i++){
        rhs[i] = dist(rng);
//        rhs[i] = (value_t)(i+1) / 100;
//        std::cout << i << "\t" << rhs[i] << std::endl;
    }

    return 0;
}
