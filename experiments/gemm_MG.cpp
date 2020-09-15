#include "grid.h"
#include "saena_object.h"
#include "saena_matrix.h"
#include "saena.hpp"

#ifdef _USE_PETSC_
#include "petsc_functions.h"
#endif

using namespace std;

int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    bool verbose = false;

    if(argc != 2){
        if(!rank){
            std::cout << "This is how you can read a matrix from a file: ./Saena <MatrixA>" << std::endl;
        }
        MPI_Finalize();
        return -1;
    }

    omp_set_num_threads(1);

    // *************************** initialize the matrix ****************************

    char *Aname(argv[1]);

    {

        saena::matrix A(comm);
        A.read_file(Aname);
        A.assemble();

        // *************************** print info ****************************

        saena::amg solver;

        // *************************** checking the correctness of matrix-matrix product ****************************
#if 0
        {
//            saena::amg solver;
            saena::matrix C(comm);
            solver.matmat(&A, &B, &C);

            // check the correctness with PETSc
            petsc_check_matmat(A.get_internal_matrix(), B.get_internal_matrix(), C.get_internal_matrix());

//            C.get_internal_matrix()->print_info(0);
//            C.print(-1);

            // view A, B and C
//            petsc_viewer(A.get_internal_matrix());
//            petsc_viewer(B.get_internal_matrix());
//            petsc_viewer(C.get_internal_matrix());

//            if(!rank) printf("=====================\n\n");
        }
#endif
        // *************************** Saena matmat experiment ****************************

        int vcycle_num            = 300;
        double relative_tolerance = 1e-12;
        std::string smoother      = "jacobi";
        int preSmooth             = 3;
        int postSmooth            = 3;

        saena::options opts(vcycle_num, relative_tolerance, smoother, preSmooth, postSmooth);
        solver.set_multigrid_max_level(4);
        solver.set_matrix(&A, &opts);
        saena_object* amg = solver.get_object();
        int levels = amg->max_level;

        // matmat comparison for finest level
        saena::matrix B(A);
        saena::matrix C(comm);
        solver.matmat(&A, &B, &C, false, true);
        petsc_matmat_ave2(A.get_internal_matrix(), B.get_internal_matrix(), 5);

        // matmat comparison for the coarser levels
        for(int l = 0; l < levels; l++) {
            auto *B1 = &amg->grids[l].Ac;
            if(B1->active){
                auto *B2 = new saena_matrix(*B1);
//                B2->set_comm(B1->comm);
                saena_matrix C2(B1->comm);
                amg->matmat(B1, B2, &C2, false, true);
                petsc_matmat_ave2(B1, B2, 5);
                delete B2;
            }
        }

        solver.destroy();
    }

    // *************************** finalize ****************************

    MPI_Finalize();
    return 0;
}