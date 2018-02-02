#ifndef SAENA_SAENA_OBJECT_H
#define SAENA_SAENA_OBJECT_H

class strength_matrix;
class saena_matrix;
class prolong_matrix;
class restrict_matrix;
class Grid;

class saena_object {
public:
    int max_level = 6; // fine grid is level 0.
    // coarsening will stop if the number of rows on one processor goes below 10.
    unsigned int least_row_threshold = 10;
    // coarsening will stop if the number of rows of last level divided by previous level is lower this value.
    double row_reduction_threshold = 0.90;
    int vcycle_num = 200;
    double relative_tolerance = 1e-8;
    std::string smoother = "chebyshev";
    int preSmooth  = 3;
    int postSmooth = 3;
    float connStrength = 0.5; // connection strength parameter
    bool doSparsify = false;
    std::vector<Grid> grids;
    int CG_max_iter = 200;
    double CG_tol = 1e-10;
    bool verbose = false;
    bool repartition = true;

    saena_object();
    ~saena_object();
    int destroy();
    void set_parameters(int vcycle_num, double relative_tolerance, std::string smoother, int preSmooth, int postSmooth);
    int level_setup(Grid* grid);
    int setup(saena_matrix* A);
    int find_aggregation(saena_matrix* A, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew);
    int create_strength_matrix(saena_matrix* A, strength_matrix* S);
    int aggregation(strength_matrix* S, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew);
    int create_prolongation(saena_matrix* A, std::vector<unsigned long>& aggregate, prolong_matrix* P);
    int coarsen(saena_matrix* A, prolong_matrix* P, restrict_matrix* R, saena_matrix* Ac);
    // this function is similar to the coarsen(), but does R*A*P for only local (diagonal) blocks.
    int coarsen2(saena_matrix* A, prolong_matrix* P, restrict_matrix* R, saena_matrix* Ac);
    int solve_coarsest_CG(saena_matrix* A, std::vector<double>& u, std::vector<double>& rhs);
    int solve_coarsest_Elemental(saena_matrix* A, std::vector<double>& u, std::vector<double>& rhs);
    int smooth(Grid* grid, std::string smoother, std::vector<double>& u, std::vector<double>& rhs, int iter);
    int vcycle(Grid* grid, std::vector<double>& u, std::vector<double>& rhs);
    int solve(std::vector<double>& u);
    int solve_pcg(std::vector<double>& u);
    int solve_pcg_update(std::vector<double>& u, saena_matrix* A_new);
    int solve_pcg_update2(std::vector<double>& u, saena_matrix* A_new);
    int solve_pcg_update3(std::vector<double>& u, saena_matrix* A_new);
    int solve_pcg_update4(std::vector<double>& u, saena_matrix* A_new);
    int set_repartition_rhs(std::vector<double>& rhs);
    int repartition_u(std::vector<double>& u);
    int repartition_back_u(std::vector<double>& u);
    int shrink_cpu_A(saena_matrix* Ac, std::vector<unsigned long>& P_splitNew);
    int shrink_rhs_u(Grid* grid, std::vector<double>& u, std::vector<double>& rhs);
    int unshrink_u(Grid* grid, std::vector<double>& u);

    int writeMatrixToFileA(saena_matrix* A, std::string name);
    int writeMatrixToFileP(prolong_matrix* P, std::string name);
    int writeMatrixToFileR(restrict_matrix* R, std::string name);
    int writeVectorToFileul(std::vector<unsigned long>& v, unsigned long vSize, std::string name, MPI_Comm comm);
//    template <class T>
//    int writeVectorToFile(std::vector<T>& v, unsigned long vSize, std::string name, MPI_Comm comm);
    int change_aggregation(saena_matrix* A, std::vector<unsigned long>& aggregate, std::vector<unsigned long>& splitNew);
};

#endif //SAENA_SAENA_OBJECT_H
