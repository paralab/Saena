#pragma once

class SaenaMatrix;
class SaenaObject;

namespace saena {

    class matrix {
    public:
        matrix(unsigned int num_rows_global);
        matrix(char* name, unsigned int global_rows, MPI_Comm comm); // read from file
//        int reserve(unsigned int nnz_local);
        int set(unsigned int row, unsigned int col, double val);    // set individual value
        int set(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local); // set multiple values
        int set(unsigned int row_offset, unsigned int col_offset, unsigned int block_size, double* values); // set contiguous block
        int set(unsigned int global_row_offset, unsigned int global_col_offset, unsigned int* local_row_offset,
                 unsigned int* local_col_offset, double* values); // set generic block
        int assembly(MPI_Comm comm);
        unsigned int get_num_local_rows();
        SaenaMatrix* get_internal_matrix();
        void destroy();

    protected:
        SaenaMatrix* m_pImpl;
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
        amg(saena::matrix* A, int max_level);
        void save_to_file(char* name, unsigned int* agg); // to save aggregates to a file.
        unsigned int* load_from_file(char* name); // to load aggregates from a file.
        // before calling solve function, vector "u" is the initial guess.
        // After calling solve, it will be the solution.
        void solve(std::vector<double>& u, std::vector<double>& rhs, saena::options* opts);
        void destroy();

    protected:
        SaenaObject* m_pImpl;
    };
}
