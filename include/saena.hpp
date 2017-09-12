#pragma once

class saena_matrix;
class saena_object;

namespace saena {

    class matrix {
    public:
        matrix(unsigned int num_rows_global, MPI_Comm comm);
        matrix(char* name, unsigned int global_rows, MPI_Comm comm); // read from file

        // difference between set and set2 is that if there is a repetition, set will erase the previous one
        // and add the new one, but in set2, the values of those entries will be added.
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
        int vcycle_num;
        double relative_tolerance;
        std::string smoother;
        int preSmooth;
        int postSmooth;

    public:
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
        amg(saena::matrix* A);
        void save_to_file(char* name, unsigned int* agg); // to save aggregates to a file.
        unsigned int* load_from_file(char* name); // to load aggregates from a file.
        // before calling solve function, vector "u" is the initial guess.
        // After calling solve, it will be the solution.
        void solve(std::vector<double>& u, std::vector<double>& rhs, saena::options* opts);
        void destroy();

    protected:
        saena_object* m_pImpl;
    };
}
