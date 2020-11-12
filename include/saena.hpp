#pragma once

#include "data_struct.h"

class saena_matrix;
class saena_vector;
class saena_matrix_dense;
class saena_object;


namespace saena {

    class matrix {
    public:
        matrix();
        explicit matrix(MPI_Comm comm);
        matrix(const matrix &B); // copy constructor
        ~matrix();

        matrix& operator=(const matrix &B);

        int read_file(const char *name);
        int read_file(const char *name, const std::string &input_type);

        void set_comm(MPI_Comm comm);

        int set(index_t i, index_t j, value_t val); // set individual value
        int set(index_t* row, index_t* col, value_t* val, nnz_t nnz_local); // set multiple values
        int set(index_t i, index_t j, unsigned int size_x, unsigned int size_y, value_t* val); // set contiguous block
        int set(index_t i, index_t j, unsigned int* di, unsigned int* dj, value_t* val, nnz_t nnz_local); // set generic block

        void set_p_order(int _p_order);
        void set_prodim(int _prodim);

        void set_num_threads(const int &nth);

        bool add_dup = true; // if false replace the duplicate, otherwise add the values together.
        int add_duplicates(bool add);

        int assemble(bool scale = true);
        int assemble_band_matrix();

        int assemble_writeToFile(const char *folder_name = "");
        int writeMatrixToFile(const std::string &name = "") const;

        saena_matrix* get_internal_matrix();
        MPI_Comm get_comm();
        index_t get_num_rows();
        index_t get_num_local_rows();
        nnz_t get_nnz();
        nnz_t get_local_nnz();

        int print(int ran, std::string name = "");

        int set_shrink(bool val);

        int erase();
        int erase_lazy_update();
        int erase_no_shrink_to_fit();
        void destroy();

    protected:
        saena_matrix* m_pImpl;
    };

    class vector {
    public:
        vector();
        explicit vector(MPI_Comm comm);
        vector(const vector &B); // copy constructor
        ~vector();

        vector& operator=(const vector &B);

//        int read_file(const char *name);
//        int read_file(const char *name, const std::string &input_type);

        void set_comm(MPI_Comm comm);
        int set_idx_offset(index_t offset);

        int set(index_t i, value_t val); // set individual value
        int set(const index_t* idx, const value_t* val, index_t size); // set multiple values
        int set(const value_t* val, index_t size, index_t offset); // set multiple values
        int set(const value_t* val, index_t size); // set multiple values
//        int set(index_t* row, index_t* col, value_t* val, nnz_t nnz_local); // set multiple values
//        int set(index_t i, index_t j, unsigned int size_x, unsigned int size_y, value_t* val); // set contiguous block
//        int set(index_t i, index_t j, unsigned int* di, unsigned int* dj, value_t* val, nnz_t nnz_local); // set generic block

//        bool add_dup = false; // if false replace the duplicate, otherwise add the values together.
        int set_dup_flag(bool add);

        int assemble();
//        int assemble_writeToFile();
//        int assemble_writeToFile(const char *folder_name);

        int get_vec(std::vector<double> &vec);
        int return_vec(std::vector<double> &u1, std::vector<double> &u2);
        saena_vector* get_internal_vector();
        MPI_Comm get_comm();
//        index_t get_num_rows();
//        index_t get_num_local_rows();
//        nnz_t get_nnz();
//        nnz_t get_local_nnz();

        int print_entry(int ran);

//        int enable_shrink(bool val);

//        int erase();
//        int erase_lazy_update();
//        int erase_no_shrink_to_fit();
//        void destroy();

    protected:
        saena_vector* m_pImpl;
    };

    class options {
    private:
        int    solver_max_iter    = 500;
        double relative_tolerance = 1e-12;
        std::string smoother      = "chebyshev";
        int    preSmooth          = 3;
        int    postSmooth         = 3;

    public:
        options();
        options(int max_iter, double relative_tolerance, std::string smoother, int preSmooth, int postSmooth);
        explicit options(char* name); // to set parameters from an xml file
        ~options();

        void set(int max_iter, double relative_tolerance, std::string smoother, int preSmooth, int postSmooth);
        void set_max_iter(int max_iter);
        void set_relative_tolerance(double relative_tolerance);
        void set_smoother(std::string smoother);
        void set_preSmooth(int preSmooth);
        void set_postSmooth(int postSmooth);

        int         get_max_iter();
        double      get_relative_tolerance();
        std::string get_smoother();
        int         get_preSmooth();
        int         get_postSmooth();
    };

    class amg {
    public:
        amg();
        ~amg();

        void set_dynamic_levels(const bool &dl = true);
        int set_matrix(saena::matrix* A, saena::options* opts);
        int set_matrix(saena::matrix* A, saena::options* opts, std::vector<std::vector<int>> &l2g, std::vector<int> &g2u, int m_bdydof, std::vector<int> &order_dif); // for Nektar++
//        int set_rhs(std::vector<value_t> rhs); // note: this function copies the rhs.
        int set_rhs(saena::vector &rhs); // note: this function copies the rhs.

        void set_num_threads(const int &nth);

        saena_object* get_object();
        int set_shrink_levels(std::vector<bool> sh_lev_vec);
        int set_shrink_values(std::vector<int> sh_val_vec);
        int switch_repartition(bool val);
        int set_repartition_threshold(float thre);
        int switch_to_dense(bool val);
        int set_dense_threshold(float thre);
        double get_dense_threshold();
        MPI_Comm get_orig_comm();

        // before calling solve function, vector "u" is the initial guess.
        // After calling solve, it will be the solution.
        int solve(std::vector<value_t>& u, saena::options* opts);
        int solve_smoother(std::vector<value_t>& u, saena::options* opts);
        int solve_CG(std::vector<value_t>& u, saena::options* opts);
        int solve_pCG(std::vector<value_t>& u, saena::options* opts);
        // if solver is made based of a matrix, let's call it A, and there is an updated version of A, let's call it B,
        // and one wants to solve B*x = rhs instead of A*x = rhs, then solve_pcg_update can be used and B can be passed as the third argument.
//        int solve_pcg_update(std::vector<value_t>& u, saena::options* opts, saena::matrix* A_new);
        // similar to solve_pcg_update, but updates the LHS with A_new.

        int solve_GMRES(std::vector<value_t>& u, saena::options* opts);
        int solve_pGMRES(std::vector<value_t>& u, saena::options* opts);

        int update1(saena::matrix* A_ne); // only update the finest level A, which is the input matrix.
        int update2(saena::matrix* A_ne); // updates grids[i].A for all levels, using the previously made grids[i].P and R.
        int update3(saena::matrix* A_ne); // like update 2, but only update local parts of As.
//        int solve_pcg_update1(std::vector<value_t>& u, saena::options* opts);
//        int solve_pcg_update2(std::vector<value_t>& u, saena::options* opts);
//        int solve_pcg_update3(std::vector<value_t>& u, saena::options* opts);

        void destroy();

        bool verbose = false;
        int set_verbose(bool verb);

        int matrix_diff(saena::matrix &A, saena::matrix &B);

        int set_multigrid_max_level(int max); // 0 means only use direct solver, so no multigrid will be used.

        int set_scale(bool sc); // 0 means only use direct solver, so no multigrid will be used.

        int set_sample_sz_percent(double s_sz_prcnt);

        int lazy_update_counter = 0; // note: for lazy update project. delete it when done.

        void matmat(saena::matrix *A, saena::matrix *B, saena::matrix *C, bool assemble = true, bool print_timing = false);

        void profile_matvecs();

    protected:
        saena_object* m_pImpl;
    };

    // ==========================
    // Matrix Generator Functions
    // ==========================

    // second argument is degree-of-freedom on each processor
    int laplacian2D_old(saena::matrix* A, index_t dof_local);

    int laplacian3D(saena::matrix* A, index_t mx, index_t my, index_t mz, bool scale = true);
    int laplacian3D_old(saena::matrix* A, index_t dof_local);
    int laplacian3D_old2(saena::matrix* A, index_t mx, index_t my, index_t mz, bool scale = true);

    int laplacian3D_set_rhs(std::vector<double> &rhs, index_t mx, index_t my, index_t mz, MPI_Comm comm);
    int laplacian3D_set_rhs_old2(std::vector<double> &rhs, index_t mx, index_t my, index_t mz, MPI_Comm comm);
    int laplacian3D_set_rhs_zero(std::vector<double> &rhs, unsigned int mx, unsigned int my, unsigned int mz, MPI_Comm comm);

    int band_matrix(saena::matrix &A, index_t M, unsigned int bandwidth);

    int random_symm_matrix(saena::matrix &A, index_t M, float density);

    // ==========================

    int read_vector_file(std::vector<value_t>& v, saena::matrix &A, char *file, MPI_Comm comm);

    index_t find_split(index_t loc_size, index_t &my_split, MPI_Comm comm);
}