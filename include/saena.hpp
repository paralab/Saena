#pragma once

#include <vector>
#include <mpi.h>

class saena_matrix;
class saena_object;

namespace saena {

    class matrix {
    public:
        matrix(MPI_Comm comm);
        matrix(char* name, MPI_Comm comm); // read from file

        // The difference between set and set2 is that if there is a repetition, set will erase the previous one
        // and insert the new one, but in set2, the values of those entries will be added together.
        int set(unsigned int i, unsigned int j, double val); // set individual value
        int set(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local); // set multiple values
        int set(unsigned int i, unsigned int j, unsigned int size_x, unsigned int size_y, double* val); // set contiguous block
        int set(unsigned int i, unsigned int j, unsigned int* di, unsigned int* dj, double* val, unsigned int nnz_local); // set generic block

        int set2(unsigned int i, unsigned int j, double val); // set individual value
        int set2(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local); // set multiple values
        int set2(unsigned int i, unsigned int j, unsigned int size_x, unsigned int size_y, double* val); // set contiguous block
        int set2(unsigned int i, unsigned int j, unsigned int* di, unsigned int* dj, double* val, unsigned int nnz_local); // set generic block

        int assemble();
        unsigned int get_num_local_rows();
        saena_matrix* get_internal_matrix();
        void destroy();

    protected:
        saena_matrix* m_pImpl;
    };

    class options {
    private:
        int vcycle_num            = 50;
        double relative_tolerance = 1e-10;
        std::string smoother      = "jacobi";
        int preSmooth             = 3;
        int postSmooth            = 3;

    public:
        options();
        options(int vcycle_num, double relative_tolerance, std::string smoother, int preSmooth, int postSmooth);
        options(char* name); // to set parameters from an xml file
        ~options();

        void set(int vcycle_num, double relative_tolerance, std::string smoother, int preSmooth, int postSmooth);
        void set_vcycle_num(int vcycle_num);
        void set_relative_tolerance(double relative_tolerance);
        void set_smoother(std::string smoother);
        void set_preSmooth(int preSmooth);
        void set_postSmooth(int postSmooth);

        int         get_vcycle_num();
        double      get_relative_tolerance();
        std::string get_smoother();
        int         get_preSmooth();
        int         get_postSmooth();
    };

    class amg {
    public:
        amg();
        int set_matrix(saena::matrix* A);
        int set_rhs(std::vector<double> rhs);
        void save_to_file(char* name, unsigned int* agg); // to save aggregates to a file.
        unsigned int* load_from_file(char* name); // to load aggregates from a file.
        // before calling solve function, vector "u" is the initial guess.
        // After calling solve, it will be the solution.
        int solve(std::vector<double>& u, saena::options* opts);
        void destroy();

        bool verbose = false;
        int set_verbose(bool verb);

    protected:
        saena_object* m_pImpl;
    };

    // second argument is dof on each processor
    int laplacian2D(saena::matrix* A, unsigned int dof_local, MPI_Comm comm);
    int laplacian3D(saena::matrix* A, unsigned int dof_local, MPI_Comm comm);
}
