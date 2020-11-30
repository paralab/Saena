#pragma once

#include <saena_object.h>

void inline saena_object::smooth(Grid *grid, std::vector<value_t> &u, std::vector<value_t> &rhs, int iter) const{
    if(smoother == "jacobi"){
        grid->A->jacobi(iter, u, rhs);
    }else if(smoother == "chebyshev"){
        grid->A->chebyshev(iter, u, rhs);
    }
//        else{
//            printf("Error: Unknown smoother");
//            MPI_Finalize();
//            exit(EXIT_FAILURE);
//        }
}

template <class T>
int saena_object::writeVectorToFile(std::vector<T>& v, const std::string &name, MPI_Comm comm /*= MPI_COMM_WORLD*/,
        bool mat_market /*= false*/, index_t OFST /*= 0*/) {

    // name: pass the name of the file. The file will be saved in the working directory. To save the file in another
    //       directory, pass that path.
    // Create txt files with name name-r0.txt for processor 0, name-r1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat name-r0.txt name-r1.txt > V.txt
    // This version also writes the index number, so it has two columns, instead of 1.

    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::string outFileNameTxt = name + "-r" + std::to_string(rank) + ".txt";
    std::ofstream outFileTxt(outFileNameTxt);

    index_t sz_loc = v.size();
    index_t sz = 0;
    MPI_Reduce(&sz_loc, &sz, 1, par::Mpi_datatype<index_t>::value(), MPI_SUM, 0, comm);

    outFileTxt << std::setprecision(std::numeric_limits<long double>::digits10 + 1);

    if(mat_market){
        if(!rank)
            outFileTxt << sz << "\t" << 1 << "\t" << sz << std::endl;

        for (long i = 0; i < sz_loc; i++) {
            outFileTxt << OFST + i + 1 << "\t" << v[i] << std::endl;
        }
    }else{
        for (long i = 0; i < sz_loc; i++) {
            outFileTxt << v[i] << std::endl;
        }
    }

    outFileTxt.clear();
    outFileTxt.close();

    return 0;
}

template <class T>
int saena_object::scale_vector_scalar(std::vector<T> &v, T a, std::vector<T> &w, bool add /*= false*/){
    // if(add)
    //   w += a * v
    // else
    //   w = a * v
    // ************

//        MPI_Comm comm = MPI_COMM_WORLD;
//        int nprocs, rank;
//        MPI_Comm_size(comm, &nprocs);
//        MPI_Comm_rank(comm, &rank);

//        MPI_Barrier(comm);
//        if(!rank) std::cout << __func__ << ", scalar: " << a << std::endl;
//        MPI_Barrier(comm);

    if(v == w){
#pragma omp parallel for
        for(index_t i = 0; i < v.size(); i++){
            v[i] *= a;
        }
    }else{

        if(add){
#pragma omp parallel for
            for(index_t i = 0; i < v.size(); i++){
                w[i] += v[i] * a;
            }
        }else{
#pragma omp parallel for
            for(index_t i = 0; i < v.size(); i++){
                w[i] = v[i] * a;
            }
        }

    }

//        MPI_Barrier(comm);
//        if(!rank) std::cout << __func__ << ": end" << std::endl;
//        MPI_Barrier(comm);

    return 0;
}