#include "superlu_ddefs.h"
#include "saena_object.h"
#include "saena_matrix.h"
#include "strength_matrix.h"
#include "grid.h"
#include "aux_functions.h"
#include "ietl_saena.h"

#include <random>

#ifdef _USE_PETSC_
#include "petsc_functions.h"
#endif

void saena_object::set_parameters(int max_iter, double tol, std::string sm, int preSm, int postSm){
//    maxLevel = l-1; // maxLevel does not include fine level. fine level is 0.
    solver_max_iter = max_iter;
    solver_tol      = tol;
    smoother        = sm;
    preSmooth       = preSm;
    postSmooth      = postSm;
}


MPI_Comm saena_object::get_orig_comm(){
    return grids[0].A->comm;
}


int saena_object::setup(saena_matrix* A) {
    int nprocs = -1, rank = -1;
    MPI_Comm_size(A->comm, &nprocs);
    MPI_Comm_rank(A->comm, &rank);

#pragma omp parallel default(none) shared(rank, nprocs)
    if(!rank && omp_get_thread_num()==0)
        printf("\nnumber of processes = %d\nnumber of threads   = %d\n\n", nprocs, omp_get_num_threads());

    if(verbose_setup){
        MPI_Barrier(A->comm);
        if(!rank){
            printf("_____________________________\n\n");
            printf("level = 0 \nnumber of procs = %d \nmatrix size \t= %d \nnonzero \t= %lu \ndensity \t= %.6f \n",
                   nprocs, A->Mbig, A->nnz_g, A->density);
        }
        MPI_Barrier(A->comm);
    }

#ifdef __DEBUG1__
    if(verbose_setup_steps){
        MPI_Barrier(A->comm);

        if(!rank){
#ifdef SPLIT_NNZ
            printf("\nsplit based on nnz\n");
#endif
#ifdef SPLIT_SIZE
            printf("\nsplit based on matrix size\n");
#endif
            std::cout << "coarsen_method: " << coarsen_method << std::endl;
            printf("\nsetup: start: find_eig()\n");
        }

        MPI_Barrier(A->comm);
    }
#endif

    if(smoother=="chebyshev"){
        find_eig(*A);
    }

    if(fabs(sample_sz_percent - 1) < 1e-4)
        doSparsify = false;

#ifdef __DEBUG1__
    if(verbose_setup_steps){
        MPI_Barrier(A->comm);
        if(!rank) printf("setup: generate_dense_matrix()\n");
        MPI_Barrier(A->comm);
    }
#endif

    A->switch_to_dense = switch_to_dense;
    A->dense_threshold = dense_threshold;
    if(switch_to_dense && A->density > dense_threshold) {
        A->generate_dense_matrix();
    }

    std::vector< std::vector< std::vector<int> > > map_all;
    std::vector< std::vector<int> > g2u_all;
    std::vector<int> order_dif;

#ifdef __DEBUG1__
    if(verbose_setup_steps){
        MPI_Barrier(A->comm);
        if(!rank) printf("setup: level 0\n");
        MPI_Barrier(A->comm);
    }
#endif

    grids.resize(max_level + 1);
    grids[0] = Grid(A, 0);
    grids[0].active = true;

    int res = 0;
    int rank_new = rank;
    for(int i = 0; i < max_level; ++i){

#ifdef __DEBUG1__
        if(verbose_setup_steps){
            MPI_Barrier(grids[i].A->comm);
            if(!rank_new) printf("\nsetup: level %d\n", i+1);
            MPI_Barrier(grids[i].A->comm);
        }
#endif

        if (shrink_level_vector.size() > i + 1 && shrink_level_vector[i+1])
            grids[i].A->enable_shrink_next_level = true;
        if (shrink_values_vector.size() > i + 1)
            grids[i].A->cpu_shrink_thre2_next_level = shrink_values_vector[i+1];

        res = coarsen(&grids[i], map_all, g2u_all, order_dif); // create P, R and Ac for grid[i]

        if(res != 0){
            if(res == 1){
                max_level = i + 1;
            }else if(res == 2){
                max_level = i;
                break;
            }else{
                printf("Invalid return value in saena_object::setup()");
                exit(EXIT_FAILURE);
            }
        }

        grids[i + 1] = Grid(&grids[i].Ac, i + 1);   // Pass A to grids[i+1] (created as Ac in grids[i])
        grids[i].coarseGrid = &grids[i + 1];        // connect grids[i+1] to grids[i]

        grids[i + 1].active = grids[i].Ac.active;

        if(grids[i].Ac.active) {
            MPI_Comm_rank(grids[i].Ac.comm, &rank_new);

            if (smoother == "chebyshev") {
#ifdef __DEBUG1__
                if(verbose_setup_steps){
                    MPI_Barrier(grids[i].Ac.comm);
                    if(!rank) printf("setup: find_eig()\n");
                    MPI_Barrier(grids[i].Ac.comm);
                }
#endif
                find_eig(grids[i].Ac);
            }

            if (verbose_setup) {

                if (!rank_new) {
                    printf("_____________________________\n\n");
                    printf("level = %d \nnumber of procs = %d \nmatrix size \t= %d \nnonzero \t= %lu"
                           "\ndensity \t= %.6f \ncoarsen method \t= %s\n",
                           grids[i + 1].level, grids[i + 1].A->total_active_procs, grids[i + 1].A->Mbig, grids[i + 1].A->nnz_g,
                           grids[i + 1].A->density, (grids[i].A->p_order == 1 ? "h-coarsen" : "p-coarsen"));
                }
            }

            // write matrix to file
//            if(i == 2){
//                grids[i + 1].A->writeMatrixToFile("saena");
//            }

        }else{
#ifdef __DEBUG1__
            if(verbose_setup_steps){printf("rank %d is not active for grids[%d].Ac.\n", rank, i);}
#endif
            break;
        }
    }

    // max_level is the lowest on the active processors in the last grid. So MPI_MIN is used in the following MPI_Allreduce.
    int max_level_send = max_level;
    MPI_Allreduce(&max_level_send, &max_level, 1, MPI_INT, MPI_MIN, grids[0].A->comm);
    grids.resize(max_level + 1);

#ifdef __DEBUG1__
    {
        if(verbose_setup_steps){
            MPI_Barrier(A->comm);
            if(!rank) printf("setup: finished creating the hierarchy. max_level = %u \n", max_level);
            MPI_Barrier(A->comm);
        }

        // print active procs
        // ==================
//        for(int l = 0; l < max_level; ++l){
//            MPI_Barrier(A->comm);
//            if(grids[l].active) {
//                printf("level %d: rank %d is active\n", l + 1, rank);
//            }
//        }
    }
#endif

    if(verbose_setup){
        if(!rank){
            std::stringstream buf;
            buf << "_____________________________\n\n"
                << "number of levels = << " << max_level << " >> (the finest level is 0)\n";
            std::cout << buf.str();
            if(doSparsify) printf("final sample size percent = %f\n", 1.0 * sample_prcnt_numer / sample_prcnt_denom);
            print_sep();
        }
    }

#ifdef __DEBUG1__
    if(verbose_setup_steps){
        MPI_Barrier(A->comm);
        if(!rank) printf("setup: setup_SuperLU()\n");
        MPI_Barrier(A->comm);
    }
#endif

    if(grids.back().active) {
        A_coarsest = grids.back().A;
        setup_SuperLU();
    }

#ifdef __DEBUG1__
    if(verbose_setup_steps){
//        A_coarsest->print_info(-1);
        MPI_Barrier(A->comm);
        if(!rank) printf("setup done!\n");
        MPI_Barrier(A->comm);
    }
#endif

    return 0;
}


int saena_object::setup(saena_matrix* A, std::vector<std::vector<int>> &m_l2g, std::vector<int> &m_g2u, int m_bdydof, std::vector<int> &order_dif) {
    int nprocs = -1, rank = -1;
    MPI_Comm_size(A->comm, &nprocs);
    MPI_Comm_rank(A->comm, &rank);

    #pragma omp parallel default(none) shared(rank, nprocs)
    if(!rank && omp_get_thread_num()==0)
        printf("\nnumber of processes = %d\nnumber of threads   = %d\n\n", nprocs, omp_get_num_threads());

    if(verbose_setup){
        MPI_Barrier(A->comm);
        if(!rank){
            printf("_____________________________\n\n");
            printf("level = 0 \nnumber of procs = %d \nmatrix size \t= %d \nnonzero \t= %lu \ndensity \t= %.6f \n",
                   nprocs, A->Mbig, A->nnz_g, A->density);
        }
        MPI_Barrier(A->comm);
    }

#ifdef __DEBUG1__
    if(verbose_setup_steps){
        MPI_Barrier(A->comm);

        if(!rank){
            #ifdef SPLIT_NNZ
            printf("\nsplit based on nnz\n");
            #endif
            #ifdef SPLIT_SIZE
            printf("\nsplit based on matrix size\n");
            #endif
            std::cout << "coarsen_method: " << coarsen_method << std::endl;
            printf("\nsetup: start: find_eig()\n");
        }

        MPI_Barrier(A->comm);
    }
#endif

    if(smoother=="chebyshev"){
        find_eig(*A);
    }

    if(fabs(sample_sz_percent - 1) < 1e-4)
        doSparsify = false;

#ifdef __DEBUG1__
    if(verbose_setup_steps){
        MPI_Barrier(A->comm);
        if(!rank) printf("setup: generate_dense_matrix()\n");
        MPI_Barrier(A->comm);
    }
#endif

    A->switch_to_dense = switch_to_dense;
    A->dense_threshold = dense_threshold;
    if(switch_to_dense && A->density > dense_threshold) {
        A->generate_dense_matrix();
    }

#ifdef __DEBUG1__
    if(verbose_setup_steps){
        MPI_Barrier(A->comm);
        if(!rank) printf("setup: mesh info\n");
        MPI_Barrier(A->comm);
    }
#endif

    std::vector< std::vector< std::vector<int> > > map_all;
    if(!m_l2g.empty()){
        map_all.emplace_back(std::move(m_l2g));
    }

    std::vector< std::vector<int> > g2u_all;
    if(!m_g2u.empty()){
//        g2u_all.emplace_back(std::move(m_g2u));
        g2u_all.emplace_back(m_g2u);
    }

    bdydof = m_bdydof;

#ifdef __DEBUG1__
    if(verbose_setup_steps){
        MPI_Barrier(A->comm);
        if(!rank) printf("setup: level 0\n");
        MPI_Barrier(A->comm);
    }
#endif

    grids.resize(max_level + 1);
    grids[0] = Grid(A, 0);
    grids[0].active = true;

    int res = 0;
    int rank_new = rank;
    for(int i = 0; i < max_level; ++i){

#ifdef __DEBUG1__
        if(verbose_setup_steps){
            MPI_Barrier(A->comm);
            if(!rank_new) printf("\nsetup: level %d\n", i+1);
            MPI_Barrier(A->comm);
        }
#endif

        if (shrink_level_vector.size() > i + 1 && shrink_level_vector[i+1])
            grids[i].A->enable_shrink_next_level = true;
        if (shrink_values_vector.size() > i + 1)
            grids[i].A->cpu_shrink_thre2_next_level = shrink_values_vector[i+1];

        res = coarsen(&grids[i], map_all, g2u_all, order_dif); // create P, R and Ac for grid[i]

        if(res != 0){
            if(res == 1){
                max_level = i + 1;
            }else if(res == 2){
                max_level = i;
                break;
            }else{
                printf("Invalid return value in saena_object::setup()");
                exit(EXIT_FAILURE);
            }
        }

        grids[i + 1] = Grid(&grids[i].Ac, i + 1);   // Pass A to grids[i+1] (created as Ac in grids[i])
        grids[i].coarseGrid = &grids[i + 1];        // connect grids[i+1] to grids[i]

        grids[i + 1].active = grids[i].Ac.active;

        if(grids[i].Ac.active) {
            MPI_Comm_rank(grids[i].Ac.comm, &rank_new);

            if (smoother == "chebyshev") {
#ifdef __DEBUG1__
                if(verbose_setup_steps){
                    MPI_Barrier(grids[i].Ac.comm);
                    if(!rank) printf("setup: find_eig()\n");
                    MPI_Barrier(grids[i].Ac.comm);
                }
#endif
                find_eig(grids[i].Ac);
            }

            if (verbose_setup) {

                if (!rank_new) {
                    printf("_____________________________\n\n");
                    printf("level = %d \nnumber of procs = %d \nmatrix size \t= %d \nnonzero \t= %lu"
                           "\ndensity \t= %.6f \ncoarsen method \t= %s\n",
                           grids[i + 1].level, grids[i + 1].A->total_active_procs, grids[i + 1].A->Mbig, grids[i + 1].A->nnz_g,
                           grids[i + 1].A->density, (grids[i].A->p_order == 1 ? "h-coarsen" : "p-coarsen"));
                }
            }

            // write matrix to file
//            if(i == 2){
//                grids[i + 1].A->writeMatrixToFile("saena");
//            }
        }else{
#ifdef __DEBUG1__
            if(verbose_setup_steps){printf("rank %d is not active for grids[%d].Ac.\n", rank, i);}
#endif
            break;
        }
    }

    // max_level is the lowest on the active processors in the last grid. So MPI_MIN is used in the following MPI_Allreduce.
    int max_level_send = max_level;
    MPI_Allreduce(&max_level_send, &max_level, 1, MPI_INT, MPI_MIN, grids[0].A->comm);
    grids.resize(max_level + 1);

#ifdef __DEBUG1__
    {
        if(verbose_setup_steps){
            MPI_Barrier(A->comm);
            if(!rank) printf("setup: finished creating the hierarchy. max_level = %u \n", max_level);
            MPI_Barrier(A->comm);
        }

        // print active procs
        // ==================
//        for(int l = 0; l < max_level; ++l){
//            MPI_Barrier(A->comm);
//            if(grids[l].active) {
//                printf("level %d: rank %d is active\n", l + 1, rank);
//            }
//        }
    }
#endif

    if(verbose_setup){
        if(!rank){
            std::stringstream buf;
            buf << "_____________________________\n\n"
                << "number of levels = << " << BLUE << max_level << COLORRESET << " >> (the finest level is 0)\n";
            std::cout << buf.str();
            if(doSparsify) printf("final sample size percent = %f\n", 1.0 * sample_prcnt_numer / sample_prcnt_denom);
            print_sep();
        }
    }

#ifdef __DEBUG1__
    if(verbose_setup_steps){
        MPI_Barrier(A->comm);
        if(!rank) printf("setup: setup_SuperLU()\n");
        MPI_Barrier(A->comm);
    }
#endif

    if(grids.back().active) {
        A_coarsest = grids.back().A;
        setup_SuperLU();
    }

#ifdef __DEBUG1__
    if(verbose_setup_steps){
//        A_coarsest->print_info(-1);
        MPI_Barrier(A->comm);
        if(!rank) printf("setup done!\n");
        MPI_Barrier(A->comm);
    }
#endif

    return 0;
}


int saena_object::coarsen(Grid *grid, std::vector< std::vector< std::vector<int> > > &map_all, std::vector< std::vector<int> > &g2u_all, std::vector<int> &order_dif){

#ifdef __DEBUG1__
    int nprocs = -1, rank = -1;
    MPI_Comm_size(grid->A->comm, &nprocs);
    MPI_Comm_rank(grid->A->comm, &rank);

    if(verbose_setup_steps){
        MPI_Barrier(grid->A->comm);
        if(!rank) printf("coarsen: start. level = %d\ncreate_prolongation\n", grid->level);
        MPI_Barrier(grid->A->comm);
    }

//    grid->A->print_info(-1);
    double t1 = 0, t2 = 0;
#endif

    // **************************** create_prolongation ****************************

#ifdef __DEBUG1__
    t1 = omp_get_wtime();
#endif

    int ret_val = create_prolongation(grid, map_all, g2u_all, order_dif);

    if(ret_val == 2){
        return ret_val;
    }

#ifdef __DEBUG1__
    t2 = omp_get_wtime();
    if(verbose_coarsen) print_time(t1, t2, "Prolongation: level "+std::to_string(grid->level), grid->A->comm);

//    MPI_Barrier(grid->A->comm); printf("rank %d: here after create_prolongation!!! \n", rank); MPI_Barrier(grid->A->comm);
//    print_vector(grid->P.split, 0, "grid->P.split", grid->A->comm);
//    print_vector(grid->P.splitNew, 0, "grid->P.splitNew", grid->A->comm);
//    grid->P.print_info(-1);
//    grid->P.print_entry(-1);

    if(verbose_setup_steps){
        MPI_Barrier(grid->A->comm); if(!rank) printf("coarsen: R.transposeP\n"); MPI_Barrier(grid->A->comm);
    }
#endif

    // **************************** restriction ****************************

#ifdef __DEBUG1__
    t1 = omp_get_wtime();
#endif

    grid->R.transposeP(&grid->P);

#ifdef __DEBUG1__
    t2 = omp_get_wtime();
    if(verbose_coarsen) print_time(t1, t2, "Restriction: level "+std::to_string(grid->level), grid->A->comm);

//    MPI_Barrier(grid->A->comm); printf("rank %d: here after transposeP!!! \n", rank); MPI_Barrier(grid->A->comm);
//    print_vector(grid->R.entry_local, -1, "grid->R.entry_local", grid->A->comm);
//    print_vector(grid->R.entry_remote, -1, "grid->R.entry_remote", grid->A->comm);
//    grid->R.print_info(-1);
//    grid->R.print_entry(-1);

    if(verbose_setup_steps){
        MPI_Barrier(grid->A->comm); if(!rank) printf("coarsen: compute_coarsen\n"); MPI_Barrier(grid->A->comm);
    }
#endif

    // **************************** compute_coarsen ****************************

#ifdef __DEBUG1__
//    MPI_Barrier(grid->A->comm);
//    double t11 = MPI_Wtime();
    t1 = omp_get_wtime();
#endif

    compute_coarsen(grid);

#ifdef __DEBUG1__
    t2 = omp_get_wtime();
//    double t22 = MPI_Wtime();
    if(verbose_coarsen) print_time(t1, t2, "compute_coarsen: level "+std::to_string(grid->level), grid->A->comm);
//    print_time_ave(t22-t11, "compute_coarsen: level "+std::to_string(grid->level), grid->A->comm);

//    MPI_Barrier(grid->A->comm); printf("rank %d: here after compute_coarsen!!! \n", rank); MPI_Barrier(grid->A->comm);
//    if(grid->Ac.active) print_vector(grid->Ac.split, 1, "grid->Ac.split", grid->Ac.comm);
//    if(grid->Ac.active) print_vector(grid->Ac.entry, 1, "grid->Ac.entry", grid->A->comm);

//    printf("rank = %d, M = %u, nnz_l = %lu, nnz_g = %lu, Ac.M = %u, Ac.nnz_l = %lu, Ac.nnz_g = %lu\n",
//           rank, grid->A->M, grid->A->nnz_l, grid->A->nnz_g, grid->Ac.M, grid->Ac.nnz_l, grid->Ac.nnz_g);

//    int rank1;
//    MPI_Comm_rank(grid->A->comm, &rank1);
//    printf("Mbig = %u, M = %u, nnz_l = %lu, nnz_g = %lu \n", grid->Ac.Mbig, grid->Ac.M, grid->Ac.nnz_l, grid->Ac.nnz_g);
//    print_vector(grid->Ac.entry, 0, "grid->Ac.entry", grid->Ac.comm);

    if(verbose_setup_steps){
        MPI_Barrier(grid->A->comm); if(!rank) printf("coarsen: end\n"); MPI_Barrier(grid->A->comm);
    }
#endif

    // **************************** Visualize using PETSc ****************************

//    petsc_viewer(grid->A);
//    petsc_viewer(&grid->P);
//    petsc_viewer(&grid->R);
//    petsc_viewer(&grid->Ac);

    // **************************** compute_coarsen in PETSc ****************************

    // this part is only for experiments.
//    petsc_coarsen(&grid->R, grid->A, &grid->P);
//    petsc_coarsen_PtAP(&grid->R, grid->A, &grid->P);
//    petsc_coarsen_2matmult(&grid->R, grid->A, &grid->P);
//    petsc_check_matmatmat(&grid->R, grid->A, &grid->P, &grid->Ac);

//    map_matmat.clear();

    return ret_val;
}


int saena_object::create_prolongation(Grid *grid, std::vector< std::vector< std::vector<int> > > &map_all, std::vector< std::vector<int> > &g2u_all, std::vector<int> &order_dif) {

    int ret_val = 0;
    if(grid->A->p_order == 1){
        ret_val = SA(grid);
    }else{
        pcoarsen(grid, map_all, g2u_all, order_dif);
    }

    return ret_val;
}


int saena_object::scale_vector(std::vector<value_t>& v, std::vector<value_t>& w) {

#pragma omp parallel for
    for(index_t i = 0; i < v.size(); i++)
        v[i] *= w[i];

    return 0;
}


int saena_object::find_eig(saena_matrix& A){

    // for preconditioned Chebyshev (smoother) we need to estimate the max eigenvalue of D^{-1} * A,
    // A.values_local and A.values_remote are being used in A.matvec which is used in ietl to compute the max
    // eigenvalue, so update A values, compute the eigenvalue, then return A values to their original ones.
    for(index_t i = 0; i < A.values_local.size(); ++i){
        A.values_local[i] *= A.inv_diag[A.row_local[i]];
    }

    for(index_t i = 0; i < A.values_remote.size(); ++i){
        A.values_remote[i] *= A.inv_diag[A.row_remote[i]];
    }

//    find_eig_Elemental(A);
    find_eig_ietl(A);

    for(index_t i = 0; i < A.values_local.size(); ++i){
        A.values_local[i] /= A.inv_diag[A.row_local[i]];
    }

    for(index_t i = 0; i < A.values_remote.size(); ++i){
        A.values_remote[i] /= A.inv_diag[A.row_remote[i]];
    }

    return 0;
}
