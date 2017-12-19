#include <vector>
#include <string>
#include <cstring>
#include <mpi.h>

#include "saena.hpp"
#include "saena_matrix.h"
#include "saena_object.h"
#include "pugixml.hpp"

// ******************************* matrix *******************************

saena::matrix::matrix(MPI_Comm comm) {
    m_pImpl = new saena_matrix(comm);
}

saena::matrix::matrix() {
    m_pImpl = new saena_matrix();
}

void saena::matrix::set_comm(MPI_Comm comm) {
    m_pImpl->set_comm(comm);
}

saena::matrix::matrix(char *name, MPI_Comm comm) {
    m_pImpl = new saena_matrix(name, comm);
}


saena::matrix::~matrix(){
    m_pImpl->erase();
//    delete m_pImpl;
}


int saena::matrix::set(unsigned int i, unsigned int j, double val){

    if( val != 0) {
        if (!m_pImpl->assembled) {
            if (!add_dup)
                m_pImpl->set(i, j, val);
            else
                m_pImpl->set2(i, j, val);
        } else {
            m_pImpl->set3(i, j, val);
        }
    }

    return 0;
}

int saena::matrix::set(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local){

    if (!m_pImpl->assembled) {
        if (!add_dup) {
            m_pImpl->set(row, col, val, nnz_local);
        } else {
            m_pImpl->set2(row, col, val, nnz_local);
        }
    }else{
        m_pImpl->set3(row, col, val, nnz_local);
    }

    return 0;
}

int saena::matrix::set(unsigned int i, unsigned int j, unsigned int size_x, unsigned int size_y, double* val){
// ordering of val should be first columns, then rows.
    unsigned int ii, jj, iter;
    iter = 0;

    for(jj = 0; jj < size_y; jj++) {
        for(ii = 0; ii < size_x; ii++) {
            if( val[iter] != 0){
                if(!add_dup)
                    m_pImpl->set(i+ii, j+jj, val[iter]);
                else
                    m_pImpl->set2(i+ii, j+jj, val[iter]);

                iter++;
            }
        }
    }

    return 0;
}

int saena::matrix::set(unsigned int i, unsigned int j, unsigned int* di, unsigned int* dj, double* val, unsigned int nnz_local){
    unsigned int ii;

    for(ii = 0; ii < nnz_local; ii++) {
        if(val[ii] != 0){
            if(!add_dup)
                m_pImpl->set(i+di[ii], j+dj[ii], val[ii]);
            else
                m_pImpl->set2(i+di[ii], j+dj[ii], val[ii]);
        }
    }

    return 0;
}


int saena::matrix::assemble() {
    m_pImpl->repartition();
    m_pImpl->matrix_setup();
    return 0;
}

unsigned int saena::matrix::get_num_local_rows() {
    return m_pImpl->M;
}

saena_matrix* saena::matrix::get_internal_matrix(){
    return m_pImpl;
}

int saena::matrix::erase(){
    m_pImpl->erase();
    return 0;
}

void saena::matrix::destroy(){
    m_pImpl->erase();
//    m_pImpl->~matrix();
}

int saena::matrix::add_duplicates(bool add) {
    if(add){
        add_dup = true;
        m_pImpl->add_duplicates = true;
    }
    else{
        add_dup = false;
        m_pImpl->add_duplicates = false;
    }
    return 0;
}


// ******************************* options *******************************

saena::options::options(){}

saena::options::options(int vcycle_n, double relT, std::string sm, int preSm, int postSm){
    vcycle_num         = vcycle_n;
    relative_tolerance = relT;
    smoother           = sm;
    preSmooth          = preSm;
    postSmooth         = postSm;
}

saena::options::options(char* name){
    pugi::xml_document doc;
    if (!doc.load_file(name))
        std::cout << "Could not find the xml file!" << std::endl;

    pugi::xml_node opts = doc.child("SAENA").first_child();

    pugi::xml_attribute attr = opts.first_attribute();
    vcycle_num = std::stoi(attr.value());

    attr = attr.next_attribute();
    relative_tolerance = std::stod(attr.value());

    attr = attr.next_attribute();
    smoother = attr.value();

    attr = attr.next_attribute();
    preSmooth = std::stoi(attr.value());

    attr = attr.next_attribute();
    postSmooth = std::stoi(attr.value());

//    for (pugi::xml_attribute attr2 = opts.first_attribute(); attr2; attr2 = attr2.next_attribute())
//        std::cout << attr2.name() << " = " << attr2.value() << std::endl;

//    std::cout << "vcycle_num = " << vcycle_num << ", relative_tolerance = " << relative_tolerance
//              << ", smoother = " << smoother << ", preSmooth = " << preSmooth << ", postSmooth = " << postSmooth << std::endl;

}

saena::options::~options(){
}

void saena::options::set(int vcycle_n, double relT, std::string sm, int preSm, int postSm){
    vcycle_num         = vcycle_n;
    relative_tolerance = relT;
    smoother           = sm;
    preSmooth          = preSm;
    postSmooth         = postSm;
}


void saena::options::set_vcycle_num(int v){
    vcycle_num = v;
}

void saena::options::set_relative_tolerance(double r){
    relative_tolerance = r;
}

void saena::options::set_smoother(std::string s){
    smoother = s;
}

void saena::options::set_preSmooth(int pr){
    preSmooth = pr;
}

void saena::options::set_postSmooth(int po){
    postSmooth = po;
}


int saena::options::get_vcycle_num(){
    return vcycle_num;
}

double saena::options::get_relative_tolerance(){
    return relative_tolerance;
}

std::string saena::options::get_smoother(){
    return smoother;
}

int saena::options::get_preSmooth(){
    return preSmooth;
}

int saena::options::get_postSmooth(){
    return postSmooth;
}


// ******************************* amg *******************************

saena::amg::amg(){
    m_pImpl = new saena_object();
}

int saena::amg::set_matrix(saena::matrix* A){
    m_pImpl->setup(A->get_internal_matrix());
    return 0;
}

int saena::amg::set_rhs(std::vector<double> rhs){
    m_pImpl->set_repartition_rhs(rhs);
    return 0;
}


int saena::amg::solve(std::vector<double>& u, saena::options* opts){
    m_pImpl->set_parameters(opts->get_vcycle_num(), opts->get_relative_tolerance(),
                            opts->get_smoother(), opts->get_preSmooth(), opts->get_postSmooth());
    m_pImpl->solve(u);
    return 0;
}


int saena::amg::solve_pcg(std::vector<double>& u, saena::options* opts){
    m_pImpl->set_parameters(opts->get_vcycle_num(), opts->get_relative_tolerance(),
                            opts->get_smoother(), opts->get_preSmooth(), opts->get_postSmooth());
    m_pImpl->solve_pcg(u);
    return 0;
}


int saena::amg::solve_pcg_update(std::vector<double>& u, saena::options* opts, saena::matrix* A_new){
    m_pImpl->set_parameters(opts->get_vcycle_num(), opts->get_relative_tolerance(),
                            opts->get_smoother(), opts->get_preSmooth(), opts->get_postSmooth());
    m_pImpl->solve_pcg_update(u, A_new->get_internal_matrix());
    return 0;
}


int saena::amg::solve_pcg_update2(std::vector<double>& u, saena::options* opts, saena::matrix* A_new){
    m_pImpl->set_parameters(opts->get_vcycle_num(), opts->get_relative_tolerance(),
                            opts->get_smoother(), opts->get_preSmooth(), opts->get_postSmooth());
    m_pImpl->solve_pcg_update2(u, A_new->get_internal_matrix());
    return 0;
}


int saena::amg::solve_pcg_update3(std::vector<double>& u, saena::options* opts, saena::matrix* A_new){
    m_pImpl->set_parameters(opts->get_vcycle_num(), opts->get_relative_tolerance(),
                            opts->get_smoother(), opts->get_preSmooth(), opts->get_postSmooth());
    m_pImpl->solve_pcg_update3(u, A_new->get_internal_matrix());
    return 0;
}


void saena::amg::save_to_file(char* name, unsigned int* agg){

}

unsigned int* saena::amg::load_from_file(char* name){
    return nullptr;
}

void saena::amg::destroy(){
    // will add later.
}

int saena::amg::set_verbose(bool verb) {
    m_pImpl->verbose = verb;
    verbose = verb;
    return 0;
}

int saena::amg::set_multigrid_max_level(int max){
    m_pImpl->max_level = max;
    return 0;
}


int saena::laplacian2D(saena::matrix* A, unsigned int n_matrix_local, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    unsigned int n_matrix = nprocs * n_matrix_local;
    unsigned int n_grid = floor(sqrt(n_matrix)); // number of rows (or columns) of the matrix
//    if(rank==0) std::cout << "n_matrix = " << n_matrix << ", n_grid = " << n_grid << ", != " << (n_matrix != n_grid * n_grid) << std::endl;

    if(n_matrix != n_grid * n_grid){
        if(rank==0) printf("\nerror: (dof_local * nprocs = %u) should be a squared number!\n\n", nprocs * n_matrix_local);
        MPI_Finalize();
        return -1;
    }

    unsigned int node, node_start, node_end; // node = global node index
//    auto offset = (unsigned int)floor(n_matrix / nprocs);

    node_start = rank * n_matrix_local;
    node_end   = node_start + n_matrix_local;
//    printf("rank = %d, node_start = %u, node_end = %u \n", rank, node_start, node_end);

//    if(rank == nprocs -1)
//        node_end = node_start + ( n_matrix - ((nprocs-1) * offset) );

    unsigned int modulo, division;
    for(node = node_start; node < node_end; node++) {
        modulo = node % n_grid;
        division = (unsigned int)floor(node / n_grid);

        if(modulo != 0 && modulo != (n_grid-1) && division != 0 && division != (n_grid-1) ){ // k is not a boundary node
            A->set(node, node, 4);

            modulo = (node+1) % n_grid;
            division = (unsigned int)floor( (node+1) / n_grid);
            if(modulo != 0 && modulo != (n_grid-1) && division != 0 && division != (n_grid-1) )// k+1 is not a boundary node
                A->set(node, node+1, -1);

            modulo = (node-1) % n_grid;
            division = (unsigned int)floor( (node-1) / n_grid);
            if(modulo != 0 && modulo != (n_grid-1) && division != 0 && division != (n_grid-1) )// k-1 is not a boundary node
                A->set(node, node-1, -1);

            modulo = (node-n_grid) % n_grid;
            division = (unsigned int)floor( (node-n_grid) / n_grid);
            if(modulo != 0 && modulo != (n_grid-1) && division != 0 && division != (n_grid-1) )// k-n_grid is not a boundary node
                A->set(node, node-n_grid, -1);

            modulo = (node+n_grid) % n_grid;
            division = (unsigned int)floor( (node+n_grid) / n_grid);
            if(modulo != 0 && modulo != (n_grid-1) && division != 0 && division != (n_grid-1) )// k+n_grid is not a boundary node
                A->set(node, node+n_grid, -1);
        }
    }

    // A.set overwrites the value in case of a duplicate.
    // boundary values are being overwritten by 1.
    for(node = node_start; node < node_end; node++) {
        modulo = node % n_grid;
        division = (unsigned int)floor(node / n_grid);

        // boundary node
        if(modulo == 0 || modulo == (n_grid-1) || division == 0 || division == (n_grid-1) )
            A->set(node, node, 1);
    }
    A->assemble();

    return 0;
}


int saena::laplacian3D(saena::matrix* A, unsigned int mx, unsigned int my, unsigned int mz, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    int       i,j,k,xm,ym,zm,xs,ys,zs,num, numi, numj, numk;
    double    v[7],Hx,Hy,Hz,HyHzdHx,HxHzdHy,HxHydHz;
    unsigned int col_index[7];
    unsigned int node;

    Hx      = 1.0 / (double)(mx);
    Hy      = 1.0 / (double)(my);
    Hz      = 1.0 / (double)(mz);
//    printf("\nrank %d: mx = %d, my = %d, mz = %d, Hx = %f, Hy = %f, Hz = %f\n", rank, mx, my, mz, Hx, Hy, Hz);

    HyHzdHx = Hy*Hz/Hx;
    HxHzdHy = Hx*Hz/Hy;
    HxHydHz = Hx*Hy/Hz;

    // split the 3D grid by only the z axis. So put the whole x and y grids on processors, but split z by the number of processors.
    xs = 0;
    xm = mx;
    ys = 0;
    ym = my;
    zm = (int)floor(mz / nprocs);
    zs = rank * zm;
    if(rank == nprocs - 1)
        zm = mz - ( (nprocs - 1) * zm);
//    printf("rank %d: corners: \nxs = %d, ys = %d, zs = %d, xm = %d, ym = %d, zm = %d\n", rank, xs, ys, zs, xm, ym, zm);

    for (k=zs; k<zs+zm; k++) {
        for (j=ys; j<ys+ym; j++) {
            for (i=xs; i<xs+xm; i++) {
                node = mx * my * k + mx * j + i; // for 2D it should be = mx * j + i
//                if(rank==0) printf("node = %u\n", node);

                if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1) {
//                    if(rank==0) printf("boundary!\n");
                    num = 0; numi=0; numj=0; numk=0;
                    if (k!=0) {
                        v[num]     = -HxHydHz;
//                        col[num].i = i;
//                        col[num].j = j;
//                        col[num].k = k-1;
                        col_index[num] = node - (mx * my);
                        num++; numk++;
                    }
                    if (j!=0) {
                        v[num]     = -HxHzdHy;
//                        col[num].i = i;
//                        col[num].j = j-1;
//                        col[num].k = k;
                        col_index[num] = node - mx;
                        num++; numj++;
                    }
                    if (i!=0) {
                        v[num]     = -HyHzdHx;
//                        col[num].i = i-1;
//                        col[num].j = j;
//                        col[num].k = k;
                        col_index[num] = node - 1;
                        num++; numi++;
                    }
                    if (i!=mx-1) {
                        v[num]     = -HyHzdHx;
//                        col[num].i = i+1;
//                        col[num].j = j;
//                        col[num].k = k;
                        col_index[num] = node + 1;
                        num++; numi++;
                    }
                    if (j!=my-1) {
                        v[num]     = -HxHzdHy;
//                        col[num].i = i;
//                        col[num].j = j+1;
//                        col[num].k = k;
                        col_index[num] = node + mx;
                        num++; numj++;
                    }
                    if (k!=mz-1) {
                        v[num]     = -HxHydHz;
//                        col[num].i = i;
//                        col[num].j = j;
//                        col[num].k = k+1;
                        col_index[num] = node + (mx * my);
                        num++; numk++;
                    }
                    v[num]     = (double)(numk)*HxHydHz + (double)(numj)*HxHzdHy + (double)(numi)*HyHzdHx;
//                    col[num].i = i;   col[num].j = j;   col[num].k = k;
                    col_index[num] = node;
                    num++;
                    for(int l = 0; l < num; l++){
//                        printf("%d \t%u \t%u \t%f \n", l, node, col_index[l], v[l]);
                        A->set(node, col_index[l], v[l]);
                    }

                } else {
//                    if(rank==0) printf("not boundary!\n");

//                    col[0].i = i;   col[0].j = j;   col[0].k = k-1;
                    v[0] = -HxHydHz;
                    col_index[0] = node - (mx * my);
                    A->set(node, col_index[0], v[0]);

//                    col[1].i = i;   col[1].j = j-1; col[1].k = k;
                    v[1] = -HxHzdHy;
                    col_index[1] = node - mx;
                    A->set(node, col_index[1], v[1]);

//                    col[2].i = i-1; col[2].j = j;   col[2].k = k;
                    v[2] = -HyHzdHx;
                    col_index[2] = node - 1;
                    A->set(node, col_index[2], v[2]);

//                    col[3].i = i;   col[3].j = j;   col[3].k = k;
                    v[3] = 2.0*(HyHzdHx + HxHzdHy + HxHydHz);
                    col_index[3] = node;
                    A->set(node, col_index[3], v[3]);

//                    col[4].i = i+1; col[4].j = j;   col[4].k = k;
                    v[4] = -HyHzdHx;
                    col_index[4] = node + 1;
                    A->set(node, col_index[4], v[4]);

//                    col[5].i = i;   col[5].j = j+1; col[5].k = k;
                    v[5] = -HxHzdHy;
                    col_index[5] = node + mx;
                    A->set(node, col_index[5], v[5]);

//                    col[6].i = i;   col[6].j = j;   col[6].k = k+1;
                    v[6] = -HxHydHz;
                    col_index[6] = node + (mx * my);
                    A->set(node, col_index[6], v[6]);

//                    for(int l = 0; l < 7; l++)
//                        printf("%d \t%u \t%u \t%f \n", l, node, col_index[l], v[l]);
                }
            }
        }
    }
//    printf("here\n");
    A->assemble();
//    printf("after\n");

    return 0;
}


int saena::laplacian3D_old(saena::matrix* A, unsigned int n_matrix_local, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    unsigned int n_matrix = nprocs * n_matrix_local;
    unsigned int n_grid = cbrt(n_matrix); // number of rows (or columns) of the matrix
//    if(rank==0) std::cout << "n_matrix = " << n_matrix << ", n_grid = " << n_grid << std::endl;

    if(n_matrix != n_grid * n_grid * n_grid){
        if(rank==0) printf("\nerror: (dof_local * nprocs) should be a cubed number (of power 3!)\n\n");
        MPI_Finalize();
        return -1;
    }

    unsigned int node, node_start, node_end; // node = global node index
//    auto offset = (unsigned int)floor(n_matrix / nprocs);

    node_start = rank * n_matrix_local;
    node_end   = node_start + n_matrix_local;
//    printf("rank = %d, node_start = %u, node_end = %u \n", rank, node_start, node_end);

//    if(rank == nprocs -1)
//        node_end = node_start + ( n_matrix - ((nprocs-1) * offset) );

    unsigned int modulo, division, division_sq;
    for(node = node_start; node < node_end; node++) {
        modulo = node % n_grid;
        division = (unsigned int)floor(node / n_grid);
        division_sq = (unsigned int)floor(node / (n_grid*n_grid));

        if(modulo != 0 && modulo != (n_grid-1) && division != 0 && division != (n_grid-1) && division_sq != 0 && division_sq != (n_grid-1) ){ // k is not a boundary node
            A->set(node, node, 6);

//            A->set(node, node+1, -1);
//            A->set(node, node-1, -1);
//            A->set(node, node-n_grid, -1);
//            A->set(node, node+n_grid, -1);
//            A->set(node, node - (n_grid * n_grid), -1);
//            A->set(node, node + (n_grid * n_grid), -1);

            modulo = (node+1) % n_grid;
            division = (unsigned int)floor( (node+1) / n_grid);
            division_sq = (unsigned int)floor( (node+1) / (n_grid*n_grid));
            if(modulo != 0 && modulo != (n_grid-1) && division != 0 && division != (n_grid-1) && division_sq != 0 && division_sq != (n_grid-1) )// k+1 is not a boundary node
                A->set(node, node+1, -1);

            modulo = (node-1) % n_grid;
            division = (unsigned int)floor( (node-1) / n_grid);
            division_sq = (unsigned int)floor( (node-1) / (n_grid*n_grid));
            if(modulo != 0 && modulo != (n_grid-1) && division != 0 && division != (n_grid-1) && division_sq != 0 && division_sq != (n_grid-1) )// k-1 is not a boundary node
                A->set(node, node-1, -1);

            modulo = (node-n_grid) % n_grid;
            division = (unsigned int)floor( (node-n_grid) / n_grid);
            division_sq = (unsigned int)floor( (node-n_grid) / (n_grid*n_grid));
            if(modulo != 0 && modulo != (n_grid-1) && division != 0 && division != (n_grid-1) && division_sq != 0 && division_sq != (n_grid-1) )// k-n_grid is not a boundary node
                A->set(node, node-n_grid, -1);

            modulo = (node+n_grid) % n_grid;
            division = (unsigned int)floor( (node+n_grid) / n_grid);
            division_sq = (unsigned int)floor( (node+n_grid) / (n_grid*n_grid));
            if(modulo != 0 && modulo != (n_grid-1) && division != 0 && division != (n_grid-1) && division_sq != 0 && division_sq != (n_grid-1) )// k+n_grid is not a boundary node
                A->set(node, node+n_grid, -1);

            modulo = (node - (n_grid * n_grid)) % n_grid;
            division = (unsigned int)floor( (node - (n_grid * n_grid)) / n_grid);
            division_sq = (unsigned int)floor( (node - (n_grid * n_grid)) / (n_grid*n_grid));
            if(modulo != 0 && modulo != (n_grid-1) && division != 0 && division != (n_grid-1) && division_sq != 0 && division_sq != (n_grid-1) )// k - (n_grid * n_grid) is not a boundary node
                A->set(node, node - (n_grid * n_grid), -1);

            modulo = (node + (n_grid * n_grid)) % n_grid;
            division = (unsigned int)floor( (node + (n_grid * n_grid)) / n_grid);
            division_sq = (unsigned int)floor( (node + (n_grid * n_grid)) / (n_grid*n_grid));
            if(modulo != 0 && modulo != (n_grid-1) && division != 0 && division != (n_grid-1) && division_sq != 0 && division_sq != (n_grid-1) )// k + (n_grid * n_grid) is not a boundary node
                A->set(node, node + (n_grid * n_grid), -1);
        }
    }

    // A.set overwrites the value in case of a duplicate.
    // boundary values are being overwritten by 1.
    for(node = node_start; node < node_end; node++) {
        modulo = node % n_grid;
        division = (unsigned int)floor(node / n_grid);
        division_sq = (unsigned int)floor(node / (n_grid*n_grid));

        // boundary node
        if(modulo == 0 || modulo == (n_grid-1) || division == 0 || division == (n_grid-1) || division_sq == 0 || division_sq == (n_grid-1)  )
            A->set(node, node, 1);
    }

    return 0;
}
