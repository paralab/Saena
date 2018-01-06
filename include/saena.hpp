#pragma once

#include <vector>
#include <mpi.h>

class saena_matrix;
class saena_object;

namespace saena {

    class matrix {
    public:
        matrix(MPI_Comm comm);
        matrix();
        matrix(char* name, MPI_Comm comm); // read from file
        ~matrix();

        void set_comm(MPI_Comm comm);
        int set(unsigned int i, unsigned int j, double val); // set individual value
        int set(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local); // set multiple values
        int set(unsigned int i, unsigned int j, unsigned int size_x, unsigned int size_y, double* val); // set contiguous block
        int set(unsigned int i, unsigned int j, unsigned int* di, unsigned int* dj, double* val, unsigned int nnz_local); // set generic block

        bool add_dup = false; // if false replace the duplicate, otherwise add the values together.
        int add_duplicates(bool add);
        int assemble();
        unsigned int get_num_local_rows();
        saena_matrix* get_internal_matrix();
        int erase();
        void destroy();

    protected:
        saena_matrix* m_pImpl;
    };

    class options {
    private:
        int vcycle_num            = 200;
        double relative_tolerance = 1e-10;
        std::string smoother      = "chebyshev";
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
        int solve_pcg(std::vector<double>& u, saena::options* opts);
        // if solver is made based of a matrix, let's call it A, and there is an updated version of A, let's call it B,
        // and one wants to solve B*x = rhs instead of A*x = rhs, then solve_pcg_update can be used and B can be passed as the third argument.
        int solve_pcg_update(std::vector<double>& u, saena::options* opts, saena::matrix* A_new);
        // similar to solve_pcg_update, but updates the LHS with A_new.
        int solve_pcg_update2(std::vector<double>& u, saena::options* opts, saena::matrix* A_new);
        // similar to solve_pcg_update, but updates grids[i].A for all levels, wusing the previously made grids[i].P and R.
        int solve_pcg_update3(std::vector<double>& u, saena::options* opts, saena::matrix* A_new);
        void destroy();

        bool verbose = false;
        int set_verbose(bool verb);

        int set_multigrid_max_level(int max); // 0 means only use direct solver, so no multigrid will be used.

    protected:
        saena_object* m_pImpl;
    };

    // second argument is dof on each processor
    int laplacian2D(saena::matrix* A, unsigned int dof_local, MPI_Comm comm);
    int laplacian3D(saena::matrix* A, unsigned int mx, unsigned int my, unsigned int mz, MPI_Comm comm);
    int laplacian3D_old(saena::matrix* A, unsigned int dof_local, MPI_Comm comm);
}
