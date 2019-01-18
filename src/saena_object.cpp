#include "superlu_ddefs.h"
#include "saena_object.h"
#include "saena_matrix.h"
#include "strength_matrix.h"
#include "prolong_matrix.h"
#include "restrict_matrix.h"
#include "grid.h"
#include "aux_functions.h"
#include "ietl_saena.h"
#include "dollar.hpp"

#include "petsc_functions.h"

#include <sys/stat.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <set>
#include <random>
#include <mpi.h>


saena_object::saena_object() = default;


saena_object::~saena_object() = default;


int saena_object::destroy(){
    return 0;
}


void saena_object::set_parameters(int vcycle_n, double relT, std::string sm, int preSm, int postSm){
//    maxLevel = l-1; // maxLevel does not include fine level. fine level is 0.
    vcycle_num = vcycle_n;
    relative_tolerance  = relT;
    smoother = sm;
    preSmooth = preSm;
    postSmooth = postSm;
}


MPI_Comm saena_object::get_orig_comm(){
    return grids[0].A->comm;
}


int saena_object::setup(saena_matrix* A) {
    int nprocs, rank, rank_new;
    MPI_Comm_size(A->comm, &nprocs);
    MPI_Comm_rank(A->comm, &rank);
    A->active_old_comm = true;

    #pragma omp parallel
    if(rank==0 && omp_get_thread_num()==0)
        printf("\nnumber of processes = %d, number of threads = %d\n\n", nprocs, omp_get_num_threads());

//    float row_reduction_min;
//    float total_row_reduction;
//    index_t M_current;

    if(verbose_setup)
        if(rank==0){
            printf("_____________________________\n\n");
            printf("level = 0 \nnumber of procs = %d \nmatrix size \t= %d \nnonzero \t= %lu \ndensity \t= %.6f \n",
                   nprocs, A->Mbig, A->nnz_g, A->density);}

    if(fabs(sample_sz_percent - 1) < 1e-4) doSparsify = false;

    if(verbose_setup_steps && rank==0) printf("setup: find_eig()\n");
    if(smoother=="chebyshev"){
//        double t1 = omp_get_wtime();
        find_eig(*A);
//        double t2 = omp_get_wtime();
//        if(verbose_level_setup) print_time(t1, t2, "find_eig() level 0: ", A->comm);
    }

    if(verbose_setup_steps && rank==0) printf("setup: generate_dense_matrix()\n");
    A->switch_to_dense = switch_to_dense;
    A->dense_threshold = dense_threshold;
    if(switch_to_dense && A->density > dense_threshold) {
        A->generate_dense_matrix();
    }

    mempool1 = new value_t[matmat_size_thre];
    mempool2 = new index_t[A->Mbig * 4];

    if(verbose_setup_steps && rank==0) printf("setup: first Grid\n");
    grids.resize(max_level+1);
    grids[0] = Grid(A, max_level, 0); // pass A to grids[0]

    if(verbose_setup_steps && rank==0) printf("setup: other Grids\n");
    for(int i = 0; i < max_level; i++){
//        MPI_Barrier(grids[0].A->comm); printf("rank = %d, level setup; before if\n", rank); MPI_Barrier(grids[0].A->comm);
//        if(grids[i].A->active) {
            if (shrink_level_vector.size()>i+1) if(shrink_level_vector[i+1]) grids[i].A->enable_shrink_next_level = true;
            if (shrink_values_vector.size()>i+1) grids[i].A->cpu_shrink_thre2_next_level = shrink_values_vector[i+1];
            coarsen(&grids[i]); // create P, R and Ac for grid[i]
            grids[i + 1] = Grid(&grids[i].Ac, max_level, i + 1); // Pass A to grids[i+1] (created as Ac in grids[i]) // todo: use emplace_back for grids.
            grids[i].coarseGrid = &grids[i + 1]; // connect grids[i+1] to grids[i]

            if(grids[i].Ac.active) {
                if (smoother == "chebyshev") {
//                    double t1 = omp_get_wtime();
                    find_eig(grids[i].Ac);
//                    double t2 = omp_get_wtime();
//                    if(verbose_level_setup) print_time(t1, t2, "find_eig(): ", A->comm);
                }

                if (verbose_setup) {
                    MPI_Comm_rank(grids[i].Ac.comm, &rank_new);
                    if (rank_new == 0) {
//                    MPI_Comm_size(grids[i].Ac.comm, &nprocs);
                        printf("_____________________________\n\n");
                        printf("level = %d \nnumber of procs = %d \nmatrix size \t= %d \nnonzero \t= %lu \ndensity \t= %.6f \n",
                               grids[i + 1].currentLevel, grids[i + 1].A->total_active_procs, grids[i + 1].A->Mbig, grids[i + 1].A->nnz_g,
                               grids[i + 1].A->density);
                    }
                }
            }

            // decide if next level for multigrid is required or not.
            // threshold to set maximum multigrid level
            if(dynamic_levels){
//                MPI_Allreduce(&grids[i].Ac.M, &M_current, 1, MPI_UNSIGNED, MPI_MIN, grids[i].Ac.comm);
//                total_row_reduction = (float) grids[0].A->Mbig / grids[i].Ac.Mbig;
                grids[i+1].row_reduction_min = (float) grids[i].Ac.Mbig / grids[i].A->Mbig;

//                if(grids[i].Ac.active){ MPI_Barrier(grids[i].Ac.comm); printf("row_reduction_min = %f, row_reduction_up_thrshld = %f, least_row_threshold = %u \n", grids[i+1].row_reduction_min, row_reduction_up_thrshld, least_row_threshold); MPI_Barrier(grids[i].Ac.comm);}
//                if(rank==0) if(row_reduction_min < 0.1) printf("\nWarning: Coarsening is too aggressive! Increase connStrength in saena_object.h\n");
//                row_reduction_local = (float) grids[i].Ac.M / grids[i].A->M;
//                MPI_Allreduce(&row_reduction_local, &row_reduction_min, 1, MPI_FLOAT, MPI_MIN, grids[i].Ac.comm);
//                if(rank==0) printf("row_reduction_min = %f, row_reduction_up_thrshld = %f \n", row_reduction_min, row_reduction_up_thrshld);
//                if(rank==0) printf("grids[i].Ac.Mbig = %d, grids[0].A->Mbig = %d, inequality = %d \n", grids[i].Ac.Mbig, grids[0].A->Mbig, (grids[i].Ac.Mbig*1000 < grids[0].A->Mbig));

                if ( (grids[i].Ac.Mbig < least_row_threshold) ||
                     (grids[i+1].row_reduction_min > row_reduction_up_thrshld) ||
                     (grids[i+1].row_reduction_min < row_reduction_down_thrshld) ) {

                    if(grids[i].Ac.Mbig < least_row_threshold) {
                        max_level = grids[i].currentLevel + 1;
                    }

                    if(grids[i+1].row_reduction_min > row_reduction_up_thrshld ||
                       grids[i+1].row_reduction_min < row_reduction_down_thrshld){
                        max_level = grids[i].currentLevel;
//                        grids.resize(max_level);
//                        grids.pop_back();
                        // todo: when destroy() for P and R is written, delete them by that.
                    }
//                    if(grids[i].Ac.active){ MPI_Barrier(grids[i].Ac.comm); printf("rank %d: max_level is set to %d \n", rank, max_level); MPI_Barrier(grids[i].Ac.comm);}
                }
            }
//        }
        if(!grids[i].Ac.active)
            break;
    }

    // max_level is the lowest on the active processors in the last grid. So MPI_MIN is used in the following MPI_Allreduce.
    int max_level_send = max_level;
    MPI_Allreduce(&max_level_send, &max_level, 1, MPI_INT, MPI_MIN, grids[0].A->comm);
    grids.resize(max_level);

//    printf("rank = %d, max_level = %d\n", rank, max_level);
//    printf("i = %u, max_level = %u \n", i, max_level);

    // set the "active" flag to false to next levels after the first not active processor.
//    for(int l = 0; l < max_level; ++l){
//        if(!grids[l].A->active) {
//            for (int k = l+1; k < max_level; ++k) {
//                printf("rank = %d level %d\n", rank, k);
//                grids[k].Ac.active = false;
//            }
//            break;
//        }
//    }

//    MPI_Barrier(A->comm);
//    for(int l = 0; l < max_level; ++l){
//        printf("\nlevel = %d\n", l);
//        if(grids[l].Ac.active) {
//            printf("rank = %d active\n", rank);
//        } else {
//            printf("rank = %d not active\n", rank);
//        }
//        MPI_Barrier(A->comm);
//    }

/*
    // grids[i+1].row_reduction_min is 0 by default. for the active processors in the last grid, it will be non-zero.
    // that's why MPI_MAX is used in the following MPI_Allreduce.
    float row_reduction_min_send = grids[i].row_reduction_min;
    MPI_Allreduce(&row_reduction_min_send, &grids[i].row_reduction_min, 1, MPI_FLOAT, MPI_MAX, grids[0].A->comm);
    // delete the coarsest level, if the size is not reduced much.
    if (grids[i].row_reduction_min > row_reduction_up_thrshld) {
        grids.pop_back();
        max_level--;
    }
*/

    delete[] mempool1;
    delete[] mempool2;

    if(verbose_setup){
        MPI_Barrier(A->comm);
        if(!rank){
            printf("_____________________________\n\n");
            printf("number of levels = << %d >> (the finest level is 0)\n", max_level);
            if(doSparsify) printf("final sample size percent = %f\n", 1.0 * sample_prcnt_numer / sample_prcnt_denom);
            printf("\n******************************************************\n");
        }
        MPI_Barrier(A->comm);
    }

//    MPI_Barrier(grids[0].A->comm); printf("rank %d: setup done!\n", rank); MPI_Barrier(grids[0].A->comm);

//    if(rank==0) dollar::text(std::cout);

    return 0;
}


int saena_object::coarsen(Grid *grid){

#ifdef __DEBUG1__
    int nprocs, rank;
    MPI_Comm_size(grid->A->comm, &nprocs);
    MPI_Comm_rank(grid->A->comm, &rank);

//    if(verbose_level_setup){
//        MPI_Barrier(grid->A->comm);
//        printf("rank = %d, start of coarsen: level = %d \n", rank, grid->currentLevel);
//        MPI_Barrier(grid->A->comm);
//    }

//    grid->A->print_info(-1);
#endif

    // **************************** find_aggregation ****************************

#ifdef __DEBUG1__
    double t1 = omp_get_wtime();
#endif

    std::vector<unsigned long> aggregate(grid->A->M);
    find_aggregation(grid->A, aggregate, grid->P.splitNew);

#ifdef __DEBUG1__
    double t2 = omp_get_wtime();
    if(verbose_level_setup) print_time(t1, t2, "Aggregation: level "+std::to_string(grid->currentLevel), grid->A->comm);

//    MPI_Barrier(grid->A->comm); printf("rank %d: here after find_aggregation!!! \n", rank); MPI_Barrier(grid->A->comm);
//    print_vector(aggregate, -1, "aggregate", grid->A->comm);
#endif

    // **************************** changeAggregation ****************************

    // use this to read aggregation from file and replace the aggregation_2_dist computed here.
//    changeAggregation(grid->A, aggregate, grid->P.splitNew, grid->A->comm);

    // **************************** create_prolongation ****************************

#ifdef __DEBUG1__
    t1 = omp_get_wtime();
#endif

    create_prolongation(grid->A, aggregate, &grid->P);

#ifdef __DEBUG1__
    t2 = omp_get_wtime();
    if(verbose_level_setup) print_time(t1, t2, "Prolongation: level "+std::to_string(grid->currentLevel), grid->A->comm);

//    MPI_Barrier(grid->A->comm); printf("rank %d: here after create_prolongation!!! \n", rank); MPI_Barrier(grid->A->comm);
//    print_vector(grid->P.split, 0, "grid->P.split", grid->A->comm);
//    print_vector(grid->P.splitNew, 0, "grid->P.splitNew", grid->A->comm);
//    grid->P.print_info(-1);
//    grid->P.print_entry(-1);
#endif

    // **************************** restriction ****************************

#ifdef __DEBUG1__
    t1 = omp_get_wtime();
#endif

    grid->R.transposeP(&grid->P);

#ifdef __DEBUG1__
    t2 = omp_get_wtime();
    if(verbose_level_setup) print_time(t1, t2, "Restriction: level "+std::to_string(grid->currentLevel), grid->A->comm);

//    MPI_Barrier(grid->A->comm); printf("rank %d: here after transposeP!!! \n", rank); MPI_Barrier(grid->A->comm);
//    grid->R.print_info(-1);
//    grid->R.print_entry(-1);
//    print_vector(grid->R.entry_local, -1, "grid->R.entry_local", grid->A->comm);
//    print_vector(grid->R.entry_remote, -1, "grid->R.entry_remote", grid->A->comm);
#endif

    // **************************** compute_coarsen ****************************

#ifdef __DEBUG1__
//    MPI_Barrier(grid->A->comm);
//    double t11 = MPI_Wtime();
    t1 = omp_get_wtime();
#endif

    compute_coarsen(grid);
//    compute_coarsen_old(grid);

#ifdef __DEBUG1__
    t2 = omp_get_wtime();
//    double t22 = MPI_Wtime();
    if(verbose_level_setup) print_time(t1, t2, "compute_coarsen: level "+std::to_string(grid->currentLevel), grid->A->comm);
//    print_time_ave(t22-t11, "compute_coarsen: level "+std::to_string(grid->currentLevel), grid->A->comm);

//    MPI_Barrier(grid->A->comm); printf("rank %d: here after compute_coarsen!!! \n", rank); MPI_Barrier(grid->A->comm);
//    if(grid->Ac.active) print_vector(grid->Ac.split, 1, "grid->Ac.split", grid->Ac.comm);
//    if(grid->Ac.active) print_vector(grid->Ac.entry, 1, "grid->Ac.entry", grid->A->comm);

//    printf("rank = %d, M = %u, nnz_l = %lu, nnz_g = %lu, Ac.M = %u, Ac.nnz_l = %lu \n",
//           rank, grid->A->M, grid->A->nnz_l, grid->A->nnz_g, grid->Ac.M, grid->Ac.nnz_l);
#endif

    if(!rank)
        printf("rank = %d, Mbig = %u, M = %u, nnz_l = %lu, nnz_g = %lu \n",
               rank, grid->Ac.Mbig, grid->Ac.M, grid->Ac.nnz_l, grid->Ac.nnz_g);
//    print_vector(grid->Ac.entry, 0, "grid->Ac.entry", grid->Ac.comm);

    // **************************** compute_coarsen in PETSc ****************************
//    petsc_viewer(&grid->Ac);

    // this part is only for experiments.
//    petsc_coarsen(&grid->R, grid->A, &grid->P);
    petsc_coarsen_PtAP(&grid->R, grid->A, &grid->P);
//    petsc_coarsen_2matmult(&grid->R, grid->A, &grid->P);
//    petsc_check_matmatmat(&grid->R, grid->A, &grid->P, &grid->Ac);

    return 0;
}


int saena_object::scale_vector(std::vector<value_t>& v, std::vector<value_t>& w) {

#pragma omp parallel for
    for(index_t i = 0; i < v.size(); i++)
        v[i] *= w[i];

    return 0;
}


int saena_object::find_eig(saena_matrix& A){

//    find_eig_Elemental(A);
    find_eig_ietl(A);

    return 0;
}
