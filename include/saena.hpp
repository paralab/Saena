#pragma once

#include <vector>
#include <mpi.h>

class saena_matrix;
class saena_matrix_dense;
class saena_object;

typedef unsigned int index_t;
typedef unsigned long nnz_t;
typedef double value_t;


namespace saena {

    class matrix {
    public:
        matrix(MPI_Comm comm);
        matrix();
        matrix(char* name, MPI_Comm comm); // read from file
        ~matrix();

        void set_comm(MPI_Comm comm);
        MPI_Comm get_comm();

        int set(index_t i, index_t j, value_t val); // set individual value
        int set(index_t* row, index_t* col, value_t* val, nnz_t nnz_local); // set multiple values
        int set(index_t i, index_t j, unsigned int size_x, unsigned int size_y, value_t* val); // set contiguous block
        int set(index_t i, index_t j, unsigned int* di, unsigned int* dj, value_t* val, nnz_t nnz_local); // set generic block

        bool add_dup = false; // if false replace the duplicate, otherwise add the values together.
        int add_duplicates(bool add);
        int assemble();
        int assemble_band_matrix();
        saena_matrix* get_internal_matrix();
        index_t get_num_rows();
        index_t get_num_local_rows();
        nnz_t get_nnz();
        nnz_t get_local_nnz();

        int print(int ran);

        int enable_shrink(bool val);

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
        int set_matrix(saena::matrix* A, saena::options* opts);
        int set_rhs(std::vector<value_t> rhs);
        saena_object* get_object();
        int set_shrink_levels(std::vector<bool> sh_lev_vec);
        int set_shrink_values(std::vector<int> sh_val_vec);
        int switch_repartition(bool val);
        int set_repartition_threshold(float thre);
        int switch_to_dense(bool val);
        int set_dense_threshold(float thre);
        double get_dense_threshold();

        // before calling solve function, vector "u" is the initial guess.
        // After calling solve, it will be the solution.
        int solve(std::vector<value_t>& u, saena::options* opts);
        int solve_pcg(std::vector<value_t>& u, saena::options* opts);
        // if solver is made based of a matrix, let's call it A, and there is an updated version of A, let's call it B,
        // and one wants to solve B*x = rhs instead of A*x = rhs, then solve_pcg_update can be used and B can be passed as the third argument.
//        int solve_pcg_update(std::vector<value_t>& u, saena::options* opts, saena::matrix* A_new);
        // similar to solve_pcg_update, but updates the LHS with A_new.
        int solve_pcg_update1(std::vector<value_t>& u, saena::options* opts, saena::matrix* A_new);
        // similar to solve_pcg_update, but updates grids[i].A for all levels, using the previously made grids[i].P and R.
        int solve_pcg_update2(std::vector<value_t>& u, saena::options* opts, saena::matrix* A_new);
        // similar to solve_pcg_update3, but does R*A*P only for the local (diagonal blocks).
        int solve_pcg_update3(std::vector<value_t>& u, saena::options* opts, saena::matrix* A_new);

        void save_to_file(char* name, unsigned long* agg); // to save aggregates to a file.
        unsigned long* load_from_file(char* name); // to load aggregates from a file.

        void destroy();

        bool verbose = false;
        int set_verbose(bool verb);

        int set_multigrid_max_level(int max); // 0 means only use direct solver, so no multigrid will be used.

    protected:
        saena_object* m_pImpl;
    };


    // second argument is dof on each processor
    int laplacian2D_old(saena::matrix* A, unsigned int dof_local, MPI_Comm comm);
    int laplacian3D(saena::matrix* A, unsigned int mx, unsigned int my, unsigned int mz, MPI_Comm comm);
    int laplacian3D_old(saena::matrix* A, unsigned int dof_local, MPI_Comm comm);
    int band_matrix(saena::matrix &A, index_t M, unsigned int bandwidth);
}
