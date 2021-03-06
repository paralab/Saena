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

void saena_object::set_parameters(int max_iter, double tol, std::string sm, int preSm, int postSm, bool dynamic_lev,
                                  int max_lev, int float_lev, double fil_thr, double fil_max, int fil_st, int fil_rate){
    solver_max_iter = max_iter;
    solver_tol      = tol;
    smoother        = std::move(sm);
    preSmooth       = preSm;
    postSmooth      = postSm;
    dynamic_levels  = dynamic_lev;
    max_level       = max_lev;
    float_level     = float_lev;
    filter_thre     = fil_thr;
    filter_max      = fil_max;
    filter_start    = fil_st;
    filter_rate     = fil_rate;
}

void saena_object::set_solve_params(int max_iter, double tol, std::string sm, int preSm, int postSm){
    solver_max_iter = max_iter;
    solver_tol      = tol;

    smoother        = std::move(sm);
    assert(smoother == "jacobi" || smoother == "chebyshev");

    preSmooth       = preSm;
    assert(preSmooth >= 0);

    postSmooth      = postSm;
    assert(postSmooth >= 0);
}

void saena_object::print_parameters(saena_matrix *A) const{
    int nprocs = 0, rank = 0;
    MPI_Comm_size(A->comm, &nprocs);
    MPI_Comm_rank(A->comm, &rank);

    MPI_Barrier(A->comm);
    if(!rank){
#pragma omp parallel default(none) shared(rank, nprocs)
        if(omp_get_thread_num()==0)
            printf("\nnumber of processes: %d\nnumber of threads:   %d\n", nprocs, omp_get_num_threads());

        printf("Smoother:            %s (%d, %d)\n", smoother.c_str(), preSmooth, postSmooth);
        printf("Operator Smoother:   %s (%.2f)\n", PSmoother.c_str(), connStrength);
        printf("Dynamic Levels:      %s (%d)\n", dynamic_levels ? "True" : "False", max_level);
        printf("Remove Boundary:     %s\n", remove_boundary ? "True" : "False");
        printf("Max iter = %d, rel tol = %.0e, float matvec lev = %d\n",
               solver_max_iter, solver_tol, float_level);
        printf("Filter: thre = %.0e, max = %.0e, start = %d, rate = %d\n",
               filter_thre, filter_max, filter_start, filter_rate);
        printf("_____________________________\n\n");
        printf("level = 0 \nnumber of procs = %d \nmatrix size \t= %d \nnonzero \t= %lu \ndensity \t= %.6f \n",
               nprocs, A->Mbig, A->nnz_g, A->density);
        if(A->use_dense){
            printf("dense structure = True\n");
        }
    }
    MPI_Barrier(A->comm);
}

void saena_object::destroy_mpi_comms(){
    for(int l = max_level - 1; l >= 0; --l){
        if(grids[l].active && grids[l].A->shrinked) {
            if(grids[l].A->comm != MPI_COMM_NULL && grids[l].A->comm != MPI_COMM_WORLD)
                MPI_Comm_free(&grids[l].A->comm);
        }
    }

    // last level is accessed differently, through Ac
    if(max_level > 0){
        int l = max_level - 1;
        if(grids[l + 1].active && grids[l].Ac.shrinked) {
            if(grids[l].Ac.comm != MPI_COMM_NULL && grids[l].Ac.comm != MPI_COMM_WORLD)
                MPI_Comm_free(&grids[l].Ac.comm);
        }
    }
}

MPI_Comm saena_object::get_orig_comm(){
    return grids[0].A->comm;
}

void saena_object::set_dynamic_levels(const bool &dl /*= true*/){
    dynamic_levels = dl;
}


int saena_object::setup(saena_matrix* A) {
    std::vector<std::vector<int>> m_l2g;
    std::vector<int> m_g2u;
    std::vector<int> order_dif;
    setup(A, m_l2g, m_g2u, 0, order_dif);
    return 0;
}


int saena_object::setup(saena_matrix* A, std::vector<std::vector<int>> &m_l2g, std::vector<int> &m_g2u, int m_bdydof, std::vector<int> &order_dif) {

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

    int nprocs = -1, rank = -1;
    MPI_Comm_size(A->comm, &nprocs);
    MPI_Comm_rank(A->comm, &rank);

    if(verbose_setup) {
        print_parameters(A);
    }

    if(smoother=="chebyshev"){
        find_eig(*A);
    }

    if(A->remove_boundary){
        remove_boundary = A->remove_boundary;
        std::swap(bound_row, A->bound_row);
        std::swap(bound_val, A->bound_val);
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

    if(fabs(sample_sz_percent - 1) < 1e-4)
        doSparsify = false;

#ifdef __DEBUG1__
    if(verbose_setup_steps){
        MPI_Barrier(A->comm);
        if(!rank) printf("setup: generate_dense_matrix()\n");
        MPI_Barrier(A->comm);
    }
#endif

    if(switch_to_dense && A->density > density_thre && A->Mbig <= dense_sz_thre) {
        A->generate_dense_matrix();
    }

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

        if(i >= float_level){
            grids[i].P.use_double = false; // use single-precision matvec for this matrix
            grids[i].R.use_double = false; // use single-precision matvec for this matrix
        }

        if(i + 1 >= float_level){
            grids[i].Ac.use_double = false; // use single-precision matvec for this matrix
        }

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
                    if(grids[i + 1].A->use_dense){
                        printf("dense structure = True\n");
                    }
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
    if(verbose_coarsen) print_time(t2 - t1, "Prolongation: level "+std::to_string(grid->level), grid->A->comm);

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
    if(verbose_coarsen) print_time(t2 - t1, "Restriction: level "+std::to_string(grid->level), grid->A->comm);

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
    if(verbose_coarsen) print_time(t2 - t1, "compute_coarsen: level "+std::to_string(grid->level), grid->A->comm);
//    print_time_ave(t22-t11, "compute_coarsen: level "+std::to_string(grid->level), grid->A->comm);
    {

//        MPI_Barrier(grid->A->comm); printf("rank %d: here after compute_coarsen!!! \n", rank); MPI_Barrier(grid->A->comm);
//        if(grid->Ac.active) print_vector(grid->Ac.split, 1, "grid->Ac.split", grid->Ac.comm);
//        if(grid->Ac.active) print_vector(grid->Ac.entry, 1, "grid->Ac.entry", grid->A->comm);
//
//        printf("rank = %d, M = %u, nnz_l = %lu, nnz_g = %lu, Ac.M = %u, Ac.nnz_l = %lu, Ac.nnz_g = %lu\n",
//               rank, grid->A->M, grid->A->nnz_l, grid->A->nnz_g, grid->Ac.M, grid->Ac.nnz_l, grid->Ac.nnz_g);
//
//        int rank1;
//        MPI_Comm_rank(grid->A->comm, &rank1);
//        printf("Mbig = %u, M = %u, nnz_l = %lu, nnz_g = %lu \n", grid->Ac.Mbig, grid->Ac.M, grid->Ac.nnz_l, grid->Ac.nnz_g);
//        if(grid->Ac.active) print_vector(grid->Ac.entry, -1, "grid->Ac.entry", grid->Ac.comm);

        if (verbose_setup_steps) {
            MPI_Barrier(grid->A->comm);
            if (!rank) printf("coarsen: end\n");
            MPI_Barrier(grid->A->comm);
        }

        // **************************** write matrices to files ****************************

//        grid->A->writeMatrixToFile("A0");
//        grid->P.writeMatrixToFile();
//        grid->R.writeMatrixToFile();
//        grid->Ac.writeMatrixToFile("A1");

        // **************************** Visualize using PETSc ****************************

//        petsc_viewer(grid->A);
//        petsc_viewer(&grid->P);
//        petsc_viewer(&grid->R);
//        petsc_viewer(&grid->Ac);

        // **************************** compute_coarsen in PETSc ****************************

        // this part is only for experiments.
//        petsc_coarsen(&grid->R, grid->A, &grid->P);
//        petsc_coarsen_PtAP(&grid->R, grid->A, &grid->P);
//        petsc_coarsen_2matmult(&grid->R, grid->A, &grid->P);
//        petsc_check_matmatmat(&grid->R, grid->A, &grid->P, &grid->Ac);

//        map_matmat.clear();
    }
#endif

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


int saena_object::find_eig(saena_matrix& A) const{

    // if the linear system is not scaled, scale the matrix only for computing the eigenvalue that is
    // being used in chebyshev, since chebyshev uses the preconditioned matrix.

    if(!scale)
        A.scale_matrix(false);

//    find_eig_Elemental(A);
    find_eig_ietl(A);

    if(!scale)
        A.scale_back_matrix(false);

//    A.print_entry(-1, "A");

    return 0;
}


void saena_object::profile_matvecs(){
    const int iter = 5;
    double t = 0, t1 = 0, t2 = 0;

    for(int l = 0; l <= max_level; ++l){
        if(grids[l].active) {
            t = 0;
            vector<value_t> v(grids[l].A->M, 1);
            vector<value_t> w(grids[l].A->M);
            MPI_Barrier(grids[l].A->comm);
            for(int i = 0; i < iter; ++i){
                t1 = omp_get_wtime();
                grids[l].A->matvec(v, w);
                t2 = omp_get_wtime();
                t += t2 - t1;
                swap(v, w);
            }
            print_time_all(t / iter, "matvec level " + to_string(l), grids[l].A->comm);
        }
    }
}

void saena_object::profile_matvecs_breakdown(){
    const int iter = 5;
    double t = 0, t1 = 0, t2 = 0;

#ifdef __DEBUG1__
    // for testing the correctness of different matvec implementations
/*
    if(grids[0].active) {
        int rank = 0;
        MPI_Comm_rank(grids[0].A->comm, &rank);
        vector<value_t> v(grids[0].A->M);
        for(int i = 0; i < grids[0].A->M; ++i){
            v[i] = i + grids[0].A->split[rank];
        }
        vector<value_t> w1(grids[0].A->M), w2(grids[0].A->M);
        grids[0].A->matvec_sparse_test_orig(v, w1);
        grids[0].A->matvec_sparse_test4(v, w2);
        for(int i = 0; i < grids[0].A->M; ++i){
            ASSERT(w1[i] == w2[i], "rank " << rank << ": i = " << i << ": " << w1[i] << " != " << w2[i]);
        }
    }
*/
#endif

    for(int l = 0; l <= max_level; ++l){
        if(grids[l].active) {
            int rank = 0;
            MPI_Comm_rank(grids[l].A->comm, &rank);
            if(!rank) printf("\nlevel %d:", l);
            vector<value_t> v(grids[l].A->M, 0.123);
            vector<value_t> w(grids[l].A->M);

            // warm up
            for(int i = 0; i < iter; ++i){
                t1 = omp_get_wtime();
                grids[l].A->matvec_sparse_test_orig(v, w);
                t2 = omp_get_wtime();
                t += t2 - t1;
                swap(v, w);
            }

            t = 0;
            grids[l].A->matvec_time_init();
            MPI_Barrier(grids[l].A->comm);
            for(int i = 0; i < iter; ++i){
                t1 = omp_get_wtime();
                grids[l].A->matvec_sparse_test_orig(v, w);
                t2 = omp_get_wtime();
                t += t2 - t1;
                swap(v, w);
            }
//            print_time_all(t / iter, "matvec level " + to_string(l), grids[l].A->comm);
//            grids[l].A->matvec_time_print(); // for matvec test2 and test3
            grids[l].A->matvec_time_print2(); // average
//            grids[l].A->matvec_time_print3(); // min, average and max for all parts of matvec
        }
    }
}

void saena_object::remove_boundary_rhs(std::vector<value_t> &rhs_large, std::vector<value_t> &rhs0, MPI_Comm comm){
    int rank = 0, nprocs = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
    rank_v = 1;

//    print_vector(rhs_large, -1, "rhs_large", comm);

    index_t Mbig_l = rhs_large.size(), Mbig = 0;
    MPI_Allreduce(&Mbig_l, &Mbig, 1, par::Mpi_datatype<index_t>::value(), MPI_SUM, comm);
    index_t ofst = Mbig / nprocs;

    // initial split. it will get updated later
    vector<index_t> split(nprocs + 1);
    for(int i = 0; i < nprocs; ++i){
        split[i] = i * ofst;
    }
    split[nprocs] = Mbig;

//    vector<value_t> rhs_large_s; // sorted
//    par::sampleSort(rhs_large, rhs_large_s, split, comm);

//    if(rank == rank_v) cout << "bound_row.size(): " << bound_row.size() << ", rhs_large.size(): " << rhs_large.size() << endl;
//    print_vector(bound_row, rank_v, "bound_row", comm);
//    print_vector(bound_val, rank_v, "bound_val", comm);

    rhs0.resize(rhs_large.size() - bound_row.size());
    bound_sol.resize(bound_row.size());

    index_t it1 = 0, it2 = 0;
    for(index_t i = 0; i < rhs_large.size(); ++i){
//        if(rank==rank_v) cout << i << "\t" << it1 << "\t" << bound_row[it1] << endl;
        if(bound_row[it1] == i){
            bound_sol[it1] = rhs_large[i] / bound_val[it1];
            it1++;
        }else{
            rhs0[it2++] = rhs_large[i];
        }
    }

//    print_vector(bound_sol, rank_v, "bound_sol", comm);
//    print_vector(rhs0, rank_v, "rhs after removing boundary", comm);
}

void saena_object::add_boundary_sol(std::vector<value_t> &u){
    std::vector<value_t> u_small;
    std::swap(u, u_small);
    const index_t SZ = u_small.size() + bound_sol.size();
    u.resize(SZ);

    index_t it1 = 0, it2 = 0;
    for(int i = 0; i < SZ; ++i){
        if(i != bound_row[it1]){
            u[i] = u_small[it2++];
        }else{
            u[i] = bound_sol[it1++];
        }
    }
//    ASSERT(it1 - 1 < bound_row.size(), it1 - 1 << "\t" << bound_row.size());
//    ASSERT(it2 - 1 < u_small.size(),   it2 - 1 << "\t" << u_small.size());
}
