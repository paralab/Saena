#include <iostream>
#include <random>
#include <fstream>
#include <aux_functions.h>
#include "saena_matrix.h"
#include "strength_matrix.h"
#include <math.h>

class saena_matrix;

int randomVector(std::vector<unsigned long>& V, long size, strength_matrix* S, MPI_Comm comm) {

    int rank;
    MPI_Comm_rank(comm, &rank);

//    for (unsigned long i=0; i<V.size(); i++){
//        srand (i);
//        V[i] = rand()%(V.size());
//    }

    unsigned long max_weight = ( (1UL<<63) - 1);

    unsigned long i;
    unsigned int max_degree_local = 0;
    for(i=0; i<S->M; i++){
        if(S->nnzPerRow[i] > max_degree_local)
            max_degree_local = S->nnzPerRow[i];
    }

    unsigned int max_degree;
    MPI_Allreduce(&max_degree_local, &max_degree, 1, MPI_UNSIGNED, MPI_MAX, comm);
    max_degree++;
//    printf("rank = %d, max degree local = %lu, max degree = %lu \n", rank, max_degree_local, max_degree);

    //Type of random number distribution
//    std::uniform_real_distribution<float> dist(-1.0,1.0); //(min, max)
    unsigned int max_rand = ( (1UL<<32) - 1);
    std::uniform_int_distribution<unsigned int> dist(0,max_rand); //(min, max)

    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng;

    //Initialize with non-deterministic seeds
    rng.seed(std::random_device{}());

    std::vector<double> rand(S->M);
    for (i = 0; i < V.size(); i++){
        V[i] = ((unsigned long)(max_degree - S->nnzPerRow[i])<<32) + dist(rng);
//        V[i] = V[i] << 32;
//        V[i] += dist(rng);
//        if(rank==0) cout << i << "\tnnzPerRow = " << S->nnzPerRow[i] << "\t weight = " << V[i] << endl;
    }

    // to have one node with the highest weight possible, so that node will be a root and consequently P and R won't be zero matrices.
    // the median index is being chosen here.
    if (V.size() != 0)
        V[ floor(V.size()/2) ] = max_weight;

    return 0;
}

int randomVector2(std::vector<double>& V){

//    int rank;
//    MPI_Comm_rank(comm, &rank);

//    for (unsigned long i=0; i<V.size(); i++){
//        srand (i);
//        V[i] = rand()%(V.size());
//    }

    //Type of random number distribution
    std::uniform_real_distribution<double> dist(0.0,1.0); //(min, max)

    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng;

    //Initialize with non-deterministic seeds
    rng.seed(std::random_device{}());

    for (unsigned long i=0; i<V.size(); i++){
        V[i] = dist(rng);
    }

    return 0;
}

int randomVector3(std::vector<unsigned long>& V, long size, strength_matrix* S, MPI_Comm comm) {
    // This function DOES NOT generate a random vector. It computes the maximum degree of all the nodes.
    // (degree of node i = number of nonzeros on row i)
    // Then assign to a higher degree, a lower weight ( weghit(node i) = max_degree - degree(node i) )
    // This method is similar to Yavneh's paper, in which nodes with lower degrees become coarse nodes first,
    // then nodes with higher degrees.

    int rank;
    MPI_Comm_rank(comm, &rank);

//    for (unsigned long i=0; i<V.size(); i++){
//        srand (i);
//        V[i] = rand()%(V.size());
//    }

    unsigned long i;
    unsigned long max_degree_local = 0;
    for(i=0; i<S->M; i++){
        if(S->nnzPerRow[i] > max_degree_local)
            max_degree_local = S->nnzPerRow[i];
    }

    unsigned long max_degree;
    MPI_Allreduce(&max_degree_local, &max_degree, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);
    max_degree++;
//    printf("rank = %d, max degree local = %lu, max degree = %lu \n", rank, max_degree_local, max_degree);

    //Type of random number distribution
//    std::uniform_real_distribution<double> dist(-1.0,1.0); //(min, max)

    //Mersenne Twister: Good quality random number generator
//    std::mt19937 rng;

    //Initialize with non-deterministic seeds
//    rng.seed(std::random_device{}());

    std::vector<double> rand(S->M);
    for (i = 0; i < V.size(); i++){
        V[i] = max_degree - S->nnzPerRow[i];
//        if(rank==0) cout << i << "\tnnzPerRow = " << S->nnzPerRow[i] << "\t weight = " << V[i] << endl;
    }
//        rand[i] = dist(rng);

    // to have one node with the highest weight possible, so that node will be a root and
    // consequently P and R won't be zero matrices. the median index is being chosen here.
    // todo: fix this later. doing this as follows will affect the aggregation in a bad way.
//    if (V.size() >= 2)
//        V[ floor(V.size()/2) ] = size + 1;
//    else if(V.size() == 1)
//        V[0] = size + 1;

    return 0;
}

int randomVector4(std::vector<unsigned long>& V, long size) {

//    int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//    for (unsigned long i=0; i<V.size(); i++){
//        srand (i);
//        V[i] = rand()%(V.size());
//    }

    //Type of random number distribution
    std::uniform_int_distribution<unsigned long> dist(1, size); //(min, max)

    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng;

    //Initialize with non-deterministic seeds
    rng.seed(std::random_device{}());

    for (unsigned long i = 0; i < V.size(); i++)
        V[i] = dist(rng);

    // to have one node with the highest weight possible, so that node will be a root and consequently P and R won't be zero matrices.
    // the median index is being chosen here.
    if (V.size() != 0)
        V[ floor(V.size()/2) ] = size + 1;

    return 0;
}


//template <class T>
//float myNorm(std::vector<T>& v){
//    float norm = 0;
//    for(long i=0; i<v.size(); i++)
//        norm += v[i] * v[i];
//
//    norm = sqrt(norm);
//    return  norm;
//}

/*
double myNorm(std::vector<double>& v){
    double norm = 0;
    for(long i=0; i<v.size(); i++)
        norm += v[i] * v[i];

    norm = sqrt(norm);
    return  norm;
}
*/


std::ostream & operator<<(std::ostream & stream, const cooEntry & item) {
    stream << item.row << "\t" << item.col << "\t" << item.val;
    return stream;
}


void setIJV(char* file_name, index_t *I, index_t *J, value_t *V, nnz_t nnz_g, nnz_t initial_nnz_l, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    if(rank==0) printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    if(rank==0) printf("ERROR: change datatypes for function setIJV!!!");
    if(rank==0) printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");

    std::vector<unsigned long> data;
    data.resize(3 * initial_nnz_l); // 3 is for i and j and val
    unsigned long* datap = &(*(data.begin()));

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


int dotProduct(std::vector<value_t>& r, std::vector<value_t>& s, double* dot, MPI_Comm comm){

    double dot_l = 0;
    for(index_t i=0; i<r.size(); i++)
        dot_l += r[i] * s[i];
    MPI_Allreduce(&dot_l, dot, 1, MPI_DOUBLE, MPI_SUM, comm);

    return 0;
}


double print_time(double t1, double t2, std::string function_name, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    double min, max, average;
    double t_dif = t2 - t1;

    MPI_Reduce(&t_dif, &min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&t_dif, &max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&t_dif, &average, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    average /= nprocs;

    if (rank==0)
        std::cout << std::endl << function_name << "\nmin: " << min << "\nave: " << average << "\nmax: " << max << std::endl << std::endl;

    return average;
}


int print_time_average(double t1, double t2, std::string function_name, int iter, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);

    double min, max, average;
    double t_dif = t2 - t1;

    MPI_Reduce(&t_dif, &min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    MPI_Reduce(&t_dif, &max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    MPI_Reduce(&t_dif, &average, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    average /= nprocs;

    if (rank==0)
        std::cout << std::endl << function_name << "\nmin: " << min/iter << "\nave: " << average/iter << "\nmax: " << max/iter << std::endl << std::endl;

    return 0;
}


//template <class T>
//int SaenaObject::writeVectorToFile(std::vector<T>& v, unsigned long vSize, std::string name, MPI_Comm comm) {
int writeVectorToFiled(std::vector<value_t>& v, index_t vSize, std::string name, MPI_Comm comm) {

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
                rhs.push_back(val);
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
//        rhs[i] = dist(rng);
        rhs[i] = (value_t)(i+1) /100;
//        std::cout << i << "\t" << rhs[i] << std::endl;
    }

    return 0;
}


bool row_major (const cooEntry& node1, const cooEntry& node2)
{
    if(node1.row < node2.row)
        return (true);
    else if(node1.row == node2.row)
        return(node1.col <= node2.col);
    else
        return false;
}