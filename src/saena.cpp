#include "saena.hpp"
#include "saena_matrix.h"
#include "saena_vector.h"
#include "saena_object.h"
#include "grid.h"
#include "parUtils.h"
//#include "pugixml.hpp"

# define PETSC_PI 3.14159265358979323846

// ******************************* matrix *******************************

saena::matrix::matrix(MPI_Comm comm) {
    m_pImpl = new saena_matrix(comm);
}

saena::matrix::matrix() {
    m_pImpl = new saena_matrix();
}

// copy constructor
saena::matrix::matrix(const saena::matrix &B){
    m_pImpl = new saena_matrix(*B.m_pImpl);
    add_dup = B.add_dup;
}

saena::matrix& saena::matrix::operator=(const saena::matrix &B){
    delete m_pImpl;
    m_pImpl = new saena_matrix(*B.m_pImpl);
    add_dup = B.add_dup;
    return *this;
}

void saena::matrix::set_comm(MPI_Comm comm) {
    m_pImpl->set_comm(comm);
}

MPI_Comm saena::matrix::get_comm(){
    return m_pImpl->comm;
}

saena::matrix::~matrix(){
//    m_pImpl->erase();
    delete m_pImpl;
}


int saena::matrix::read_file(const char *name) {
    m_pImpl->read_file(name);
    return 0;
}

int saena::matrix::read_file(const char *name, const std::string &input_type) {
    m_pImpl->read_file(name, input_type);
    return 0;
}


int saena::matrix::set(index_t i, index_t j, value_t val){

        if (!add_dup) {
            m_pImpl->set(i, j, val);
        } else {
            m_pImpl->set2(i, j, val);
        }

    return 0;
}

int saena::matrix::set(index_t* row, index_t* col, value_t* val, nnz_t nnz_local){

    if (!add_dup)
        m_pImpl->set(row, col, val, nnz_local);
    else
        m_pImpl->set2(row, col, val, nnz_local);

    return 0;
}

int saena::matrix::set(index_t i, index_t j, unsigned int size_x, unsigned int size_y, value_t* val){
    // ordering of val should be column-major.
    unsigned int ii, jj;
    nnz_t iter = 0;
    //todo: add openmp
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

int saena::matrix::set(index_t i, index_t j, unsigned int* di, unsigned int* dj, value_t* val, nnz_t nnz_local){
    nnz_t ii;
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


void saena::matrix::set_p_order(int _p_order){
    m_pImpl->set_p_order(_p_order);
}

void saena::matrix::set_prodim(int _prodim){
    m_pImpl->set_prodim(_prodim);
}


void saena::matrix::set_num_threads(const int &nth){
    omp_set_num_threads(nth);
}


int saena::matrix::assemble(bool scale /*= true*/) {
    m_pImpl->assemble(scale);
    return 0;
}

int saena::matrix::assemble_band_matrix(){
    m_pImpl->matrix_setup();
    return 0;
}


int saena::matrix::assemble_writeToFile(const char *folder_name/* = ""*/){

    if(!m_pImpl->assembled){
        m_pImpl->repartition_nnz_initial();
        m_pImpl->matrix_setup(false);
        if(m_pImpl->enable_shrink) m_pImpl->compute_matvec_dummy_time();
        m_pImpl->writeMatrixToFile(folder_name);
    }else{
        m_pImpl->repartition_nnz_update();
        m_pImpl->matrix_setup_update(false);
        m_pImpl->writeMatrixToFile(folder_name);
    }

    return 0;
}

int saena::matrix::writeMatrixToFile(const std::string &name/* = ""*/) const{
    m_pImpl->writeMatrixToFile(name);
    return 0;
}


saena_matrix* saena::matrix::get_internal_matrix(){
    return m_pImpl;
}

index_t saena::matrix::get_num_rows(){
    return m_pImpl->Mbig;
}

index_t saena::matrix::get_num_local_rows() {
    return m_pImpl->M;
}

nnz_t saena::matrix::get_nnz(){
    return m_pImpl->nnz_g;
}

nnz_t saena::matrix::get_local_nnz(){
    return m_pImpl->nnz_l;
}

int saena::matrix::print(int ran, std::string name){
    m_pImpl->print_entry(ran, name);
    return 0;
}

int saena::matrix::set_shrink(bool val){
    m_pImpl->enable_shrink   = val;
    m_pImpl->enable_shrink_c = val;
    return 0;
}


int saena::matrix::erase(){
    m_pImpl->erase();
    return 0;
}

int saena::matrix::erase_lazy_update(){
    m_pImpl->erase_lazy_update();
    return 0;
}

int saena::matrix::erase_no_shrink_to_fit(){
    m_pImpl->erase_no_shrink_to_fit();
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
    } else{
        add_dup = false;
        m_pImpl->add_duplicates = false;
    }
    return 0;
}


// ******************************* vector *******************************

saena::vector::vector(MPI_Comm comm) {
    m_pImpl = new saena_vector(comm);
}

saena::vector::vector() {
    m_pImpl = new saena_vector();
}

// copy constructor
saena::vector::vector(const saena::vector &B) {
    m_pImpl = new saena_vector(*B.m_pImpl);
    m_pImpl->set_dup_flag(B.m_pImpl->add_duplicates);
//    add_dup = B.add_dup;
}

saena::vector& saena::vector::operator=(const saena::vector &B) {
    delete m_pImpl;
    m_pImpl = new saena_vector(*B.m_pImpl);
    m_pImpl->set_dup_flag(B.m_pImpl->add_duplicates);
//    add_dup = B.add_dup;
    return *this;
}

void saena::vector::set_comm(MPI_Comm comm) {
    m_pImpl->set_comm(comm);
}

MPI_Comm saena::vector::get_comm() {
    return m_pImpl->comm;
}

saena::vector::~vector() {
//    m_pImpl->erase();
    delete m_pImpl;
}


int saena::vector::set_idx_offset(const index_t offset){
    m_pImpl->set_idx_offset(offset);
    return 0;
}


//int saena::vector::read_file(const char *name) {
//    m_pImpl->read_file(name);
//    return 0;
//}

//int saena::vector::read_file(const char *name, const std::string &input_type) {
//    m_pImpl->read_file(name, input_type);
//    return 0;
//}


int saena::vector::set(const index_t i, const value_t val){
    m_pImpl->set(i, val);
    return 0;
}

int saena::vector::set(const index_t* idx, const value_t* val, const index_t size){
    m_pImpl->set(idx, val, size);
    return 0;
}

int saena::vector::set(const value_t* val, const index_t size, const index_t offset){
    m_pImpl->set(val, size, offset);
    return 0;
}

int saena::vector::set(const value_t* val, const index_t size){
    m_pImpl->set(val, size);
    return 0;
}

int saena::vector::set_dup_flag(bool add){
    m_pImpl->set_dup_flag(add);
    return 0;
}


int saena::vector::assemble() {
    m_pImpl->assemble();
    return 0;
}


int saena::vector::get_vec(std::vector<double> &vec){
    m_pImpl->get_vec(vec);
    return 0;
}


int saena::vector::return_vec(std::vector<double> &u1, std::vector<double> &u2){
    m_pImpl->return_vec(u1, u2);
    return 0;
}


saena_vector* saena::vector::get_internal_vector(){
    return m_pImpl;
}

int saena::vector::print_entry(int rank_){
    m_pImpl->print_entry(rank_);
    return 0;
}


// ******************************* options *******************************

saena::options::options() = default;

saena::options::options(int vcycle_n, double relT, std::string sm, int preSm, int postSm){
    solver_max_iter         = vcycle_n;
    relative_tolerance = relT;
    smoother           = sm;
    preSmooth          = preSm;
    postSmooth         = postSm;
}

saena::options::options(char* name){
    printf("Enable pugi to be able to use function options(char* name)!");
    exit(EXIT_FAILURE);
    
/*
    pugi::xml_document doc;
    if (!doc.load_file(name))
        std::cout << "Could not find the xml file!" << std::endl;

    pugi::xml_node opts = doc.child("SAENA").first_child();

    pugi::xml_attribute attr = opts.first_attribute();
    solver_max_iter = std::stoi(attr.value());

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

//    std::cout << "solver_max_iter = " << solver_max_iter << ", solver_tol = " << solver_tol
//              << ", smoother = " << smoother << ", preSmooth = " << preSmooth << ", postSmooth = " << postSmooth << std::endl;
*/
}

saena::options::~options() = default;

void saena::options::set(int max_it, double relT, std::string sm, int preSm, int postSm){
    solver_max_iter    = max_it;
    relative_tolerance = relT;
    smoother           = sm;
    preSmooth          = preSm;
    postSmooth         = postSm;
}


void saena::options::set_max_iter(int v){
    solver_max_iter = v;
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


int saena::options::get_max_iter(){
    return solver_max_iter;
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

saena::amg::~amg(){
    delete m_pImpl;
}

void saena::amg::set_dynamic_levels(const bool &dl) {
    m_pImpl->set_dynamic_levels(dl);
}

int saena::amg::set_matrix(saena::matrix* A, saena::options* opts){
    m_pImpl->set_parameters(opts->get_max_iter(), opts->get_relative_tolerance(),
                            opts->get_smoother(), opts->get_preSmooth(), opts->get_postSmooth());
    m_pImpl->setup(A->get_internal_matrix());
    return 0;
}

int saena::amg::set_matrix(saena::matrix* A, saena::options* opts, std::vector<std::vector<int>> &m_l2g, std::vector<int> &m_g2u, int m_bdydof, std::vector<int> &order_dif){
    m_pImpl->set_parameters(opts->get_max_iter(), opts->get_relative_tolerance(),
                            opts->get_smoother(), opts->get_preSmooth(), opts->get_postSmooth());
    m_pImpl->setup(A->get_internal_matrix(), m_l2g, m_g2u, m_bdydof, order_dif);
    return 0;
}

int saena::amg::set_rhs(saena::vector &rhs){
    m_pImpl->set_repartition_rhs(rhs.get_internal_vector());
    return 0;
}


void saena::amg::set_num_threads(const int &nth){
    omp_set_num_threads(nth);
}


saena_object* saena::amg::get_object() {
    return m_pImpl;
}


int saena::amg::set_shrink_levels(std::vector<bool> sh_lev_vec) {
    m_pImpl->set_shrink_levels(sh_lev_vec);
    return 0;
}

int saena::amg::set_shrink_values(std::vector<int> sh_val_vec) {
    m_pImpl->set_shrink_values(sh_val_vec);
    return 0;
}


int saena::amg::switch_repartition(bool val) {
    m_pImpl->switch_repartition = val;
    return 0;
}


int saena::amg::set_repartition_threshold(float thre){
    m_pImpl->set_repartition_threshold(thre);
    return 0;
}


int saena::amg::switch_to_dense(bool val) {
    m_pImpl->switch_to_dense = val;
    return 0;
}


int saena::amg::set_dense_threshold(float thre){
    m_pImpl->dense_threshold = thre;
    return 0;
}


double saena::amg::get_dense_threshold(){
    return m_pImpl->dense_threshold;
}


MPI_Comm saena::amg::get_orig_comm(){
    return m_pImpl->get_orig_comm();
}


int saena::amg::solve_smoother(std::vector<value_t>& u, saena::options* opts){
    m_pImpl->set_parameters(opts->get_max_iter(), opts->get_relative_tolerance(),
                            opts->get_smoother(), opts->get_preSmooth(), opts->get_postSmooth());
    m_pImpl->solve_smoother(u);
    Grid *g = &m_pImpl->grids[0];
    g->rhs_orig->return_vec(u);
    return 0;
}

int saena::amg::solve(std::vector<value_t>& u, saena::options* opts){
    m_pImpl->set_parameters(opts->get_max_iter(), opts->get_relative_tolerance(),
                            opts->get_smoother(), opts->get_preSmooth(), opts->get_postSmooth());
    m_pImpl->solve(u);
    Grid *g = &m_pImpl->grids[0];
    g->rhs_orig->return_vec(u);
    return 0;
}


int saena::amg::solve_CG(std::vector<value_t>& u, saena::options* opts){
    m_pImpl->set_parameters(opts->get_max_iter(), opts->get_relative_tolerance(),
                            opts->get_smoother(), opts->get_preSmooth(), opts->get_postSmooth());
    m_pImpl->solve_CG(u);
    Grid *g = &m_pImpl->grids[0];
    g->rhs_orig->return_vec(u);
    return 0;
}


int saena::amg::solve_pCG(std::vector<value_t>& u, saena::options* opts){
    m_pImpl->set_parameters(opts->get_max_iter(), opts->get_relative_tolerance(),
                            opts->get_smoother(), opts->get_preSmooth(), opts->get_postSmooth());
    m_pImpl->solve_pCG(u);
    Grid *g = &m_pImpl->grids[0];
    g->rhs_orig->return_vec(u);
    return 0;
}


int saena::amg::solve_GMRES(std::vector<value_t>& u, saena::options* opts){
    m_pImpl->set_parameters(opts->get_max_iter(), opts->get_relative_tolerance(),
                            opts->get_smoother(), opts->get_preSmooth(), opts->get_postSmooth());
    m_pImpl->GMRES(u);
    Grid *g = &m_pImpl->grids[0];
    g->rhs_orig->return_vec(u);
    return 0;
}


int saena::amg::solve_pGMRES(std::vector<value_t>& u, saena::options* opts){
    m_pImpl->set_parameters(opts->get_max_iter(), opts->get_relative_tolerance(),
                            opts->get_smoother(), opts->get_preSmooth(), opts->get_postSmooth());
    m_pImpl->pGMRES(u);
    Grid *g = &m_pImpl->grids[0];
    g->rhs_orig->return_vec(u);
    return 0;
}

int saena::amg::update1(saena::matrix* A_ne){
    m_pImpl->update1(A_ne->get_internal_matrix());
    return 0;
}


int saena::amg::update2(saena::matrix* A_ne){
    m_pImpl->update2(A_ne->get_internal_matrix());
    return 0;
}


int saena::amg::update3(saena::matrix* A_ne){
    m_pImpl->update3(A_ne->get_internal_matrix());
    return 0;
}


//int saena::amg::solve_pcg_update1 & 2 & 3
/*
int saena::amg::solve_pcg_update1(std::vector<value_t>& u, saena::options* opts){
    m_pImpl->set_parameters(opts->get_max_iter(), opts->get_relative_tolerance(),
                            opts->get_smoother(), opts->get_preSmooth(), opts->get_postSmooth());
    m_pImpl->solve_pcg_update1(u);
    return 0;
}


int saena::amg::solve_pcg_update2(std::vector<value_t>& u, saena::options* opts){
    m_pImpl->set_parameters(opts->get_max_iter(), opts->get_relative_tolerance(),
                            opts->get_smoother(), opts->get_preSmooth(), opts->get_postSmooth());
    m_pImpl->solve_pcg_update2(u);
    return 0;
}


int saena::amg::solve_pcg_update3(std::vector<value_t>& u, saena::options* opts){
    m_pImpl->set_parameters(opts->get_max_iter(), opts->get_relative_tolerance(),
                            opts->get_smoother(), opts->get_preSmooth(), opts->get_postSmooth());
    m_pImpl->solve_pcg_update3(u);
    return 0;
}
*/


void saena::amg::destroy(){
    m_pImpl->destroy();
}


int saena::amg::set_verbose(bool verb) {
    m_pImpl->verbose = verb;
//    m_pImpl->verbose_setup = verb;
    verbose = verb;
    return 0;
}


int saena::amg::set_multigrid_max_level(int max){
    m_pImpl->max_level = max;
    return 0;
}


int saena::amg::set_scale(bool sc){
    m_pImpl->scale = sc;
    return 0;
}


int saena::amg::set_sample_sz_percent(double s_sz_prcnt){
    m_pImpl->sample_sz_percent = s_sz_prcnt;
    return 0;
}


int saena::amg::matrix_diff(saena::matrix &A1, saena::matrix &B1){

    saena_matrix *A = A1.get_internal_matrix();
    saena_matrix *B = B1.get_internal_matrix();

//    if(A->active){
        MPI_Comm comm = A->comm;
        int nprocs, rank;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);

        if(A->nnz_g != B->nnz_g)
            if(rank==0) std::cout << "error: matrix_diff(): A.nnz_g != B.nnz_g" << std::endl;

//        A->print_entry(-1);
//        B->print_entry(-1);

        MPI_Barrier(comm);
        printf("\nmatrix_diff: \n");
        MPI_Barrier(comm);

//        if(rank==0){
            for(nnz_t i = 0; i < A->entry.size(); i++){
//                if(!almost_zero(A->entry[i].val - B->entry[i].val)){
                    std::cout << A->entry[i] << "\t" << B->entry[i] << "\t" << A->entry[i].val - B->entry[i].val << std::endl;
//                }
            }
//        }
//    }

    MPI_Barrier(comm);
    printf("A->entry.size() = %lu, B->entry.size() = %lu \n", A->entry.size(), B->entry.size());
    MPI_Barrier(comm);

    return 0;
}


void saena::amg::matmat(saena::matrix *A, saena::matrix *B, saena::matrix *C, bool assemble /*=true*/, const bool print_timing /*=false*/){
    m_pImpl->matmat(A->get_internal_matrix(), B->get_internal_matrix(), C->get_internal_matrix(), assemble, print_timing);
}


void saena::amg::profile_matvecs(){
    m_pImpl->profile_matvecs();
}


int saena::laplacian2D(saena::matrix* A, index_t mx, index_t my, bool scale /*= true*/){
// NOTE: this info should be updated!
// changed the function in: petsc-3.13.5/src/ksp/ksp/tutorials/ex45.c
/*
Laplacian in 2D. Modeled by the partial differential equation
   - Laplacian u = f,    0 < x,y < 1,

with Dirichlet boundary conditions f(x) defined on the boundary
   x = 0, x = 1, y = 0, y = 1.
*/

    MPI_Comm comm = A->get_comm();
    int rank = 0, nprocs = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    if(nprocs > 1){
        if(!rank) printf("laplacian2D works only in serial!\n");
        MPI_Abort(comm, 1);
    }

//    printf("rank %d: mx = %d, my = %d\n", rank, mx, my);

    int i, j, xm, ym, xs, ys;
    value_t v[5], Hx, Hy, HydHx, HxdHy;
    index_t col_index[5];
    index_t node = 0;

    Hx = 1.0 / (mx - 1);
    Hy = 1.0 / (my - 1);

//    printf("\nrank %d: mx = %d, my = %d, mz = %d, Hx = %f, Hy = %f, Hz = %f\n", rank, mx, my, mz, Hx, Hy, Hz);

    HydHx = Hy / Hx;
    HxdHy = Hx / Hy;

    // split the 3D grid by only the z axis. So put the whole x and y grids on processors, but split z by the number of processors.
    xs = 0;
    xm = mx;
    ys = 0;
    ym = my;

//    printf("rank %d: xs = %d, ys = %d, zs = %d, xm = %d, ym = %d, zm = %d\n", rank, xs, ys, zs, xm, ym, zm);

    const int XMAX = mx - 1;
    const int YMAX = my - 1;

    for (j=ys; j<ys+ym; j++) {
        for (i=xs; i<xs+xm; i++) {
            node = mx * j + i;

//            row.i = i; row.j = j;
            if (i==0 || j==0 || i==XMAX || j==YMAX) {
//                v[0] = 2.0*(HxHydHz + HxHzdHy);
//                MatSetValuesStencil(B,1,&row,1,&row,v,INSERT_VALUES);
                A->set(node, node, 2.0*(HxdHy + HydHx));
            } else {
//                v[0] = -HxHydHz;col[0].i = i; col[0].j = j; col[0].k = k-1;
//                v[1] = -HxHzdHy;col[1].i = i; col[1].j = j-1; col[1].k = k;
//                v[2] = -HyHzdHx;col[2].i = i-1; col[2].j = j; col[2].k = k;
//                v[3] = 2.0*(HxHydHz + HxHzdHy + HyHzdHx);col[3].i = row.i; col[3].j = row.j; col[3].k = row.k;
//                v[4] = -HyHzdHx;col[4].i = i+1; col[4].j = j; col[4].k = k;
//                v[5] = -HxHzdHy;col[5].i = i; col[5].j = j+1; col[5].k = k;
//                v[6] = -HxHydHz;col[6].i = i; col[6].j = j; col[6].k = k+1;
//                ierr = MatSetValuesStencil(B,1,&row,7,col,v,INSERT_VALUES);CHKERRQ(ierr);

                if(j - 1 != 0)
                    A->set(node, node - mx, -HxdHy);

                if(i - 1 != 0)
                    A->set(node, node - 1, -HydHx);

                if(i + 1 != XMAX)
                    A->set(node, node + 1, -HydHx);

                if(j + 1 != YMAX)
                    A->set(node, node + mx, -HxdHy);

                A->set(node, node, 2.0*(HxdHy + HydHx));
            }
        }
    }

    A->assemble(scale);

    return 0;
}

int saena::laplacian2D_set_rhs(std::vector<double> &rhs, index_t mx, index_t my, MPI_Comm comm){

//    u = sin(2 * PETSC_PI * x) * sin(2 * PETSC_PI * y)
//    rhs = 8 * PETSC_PI * PETSC_PI * sin(2 * PETSC_PI * x) * sin(2 * PETSC_PI * y)

    int rank = 0, nprocs = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    int i = 0, j = 0, k = 0, xm = 0, ym = 0, xs = 0, ys = 0;
    value_t Hx = 0.0, Hy = 0.0, HydHx = 0.0, HxdHy = 0.0;
    index_t node = 0;

    Hx = 1.0 / (mx - 1);
    Hy = 1.0 / (my - 1);

    printf("\nrank %d: mx = %d, my = %d, Hx = %f, Hy = %f\n", rank, mx, my, Hx, Hy);

//    HxdHy = Hx/Hy;
//    HydHx = Hy/Hx;

    // split the 3D grid by only the z axis. So put the whole x and y grids on processors, but split z by the number of processors.
    xs = 0;
    xm = mx;
    ys = 0;
    ym = my;

//    printf("rank %d: corners: \nxs = %d, ys = %d, zs = %d, xm = %d, ym = %d, zm = %d\n", rank, xs, ys, zs, xm, ym, zm);

    const int XMAX = mx - 1;
    const int YMAX = my - 1;

    rhs.resize(xm * ym);

    index_t iter = 0;
    for (j=ys; j<ys+ym; j++) {
        for (i=xs; i<xs+xm; i++) {
            rhs[iter++] = 8 * PETSC_PI * PETSC_PI * sin(2 * PETSC_PI * i * Hx) * sin(2 * PETSC_PI * j * Hy);
        }
    }

    return 0;
}

int saena::laplacian2D_check_solution(std::vector<double> &u, index_t mx, index_t my, MPI_Comm comm){

    int rank = 0, nprocs = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    int i = 0, j = 0, k = 0, xm = 0, ym = 0, xs = 0, ys = 0;
    value_t Hx = 0.0, Hy = 0.0, HydHx = 0.0, HxdHy = 0.0;
    index_t node = 0;

    Hx = 1.0 / (mx - 1);
    Hy = 1.0 / (my - 1);

//    printf("\nrank %d: mx = %d, my = %d, Hx = %f, Hy = %f\n", rank, mx, my, Hx, Hy);

//    HxdHy = Hx/Hy;
//    HydHx = Hy/Hx;

    // split the 3D grid by only the z axis. So put the whole x and y grids on processors, but split z by the number of processors.
    xs = 0;
    xm = mx;
    ys = 0;
    ym = my;

//    printf("rank %d: corners: \nxs = %d, ys = %d, zs = %d, xm = %d, ym = %d, zm = %d\n", rank, xs, ys, zs, xm, ym, zm);

    const int XMAX = mx - 1;
    const int YMAX = my - 1;
    const double TWOPI = 2 * PETSC_PI;

    double dif = 0, tmp = 0;
    index_t iter = 0;
    for (j=ys; j<ys+ym; j++) {
        for (i=xs; i<xs+xm; i++) {
            tmp = u[iter++] - sin(TWOPI * i * Hx) * sin(TWOPI * j * Hy);
            dif += tmp * tmp;
//            if(fabs(tmp) > 1e-5)
//                printf("%d, %d, %d, %12.8f, %12.8f, %12.8f\n", i, j, iter-1, tmp, u[iter-1], tmp - u[iter-1]);
        }
    }

    cout << "\nnorm of diff = " << sqrt(dif) << endl;

    return 0;
}


int saena::laplacian2D_old(saena::matrix* A, index_t n_matrix_local){

    MPI_Comm comm = A->get_comm();
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    index_t n_matrix = nprocs * n_matrix_local;
    index_t n_grid = floor(sqrt(n_matrix)); // number of rows (or columns) of the matrix
//    if(rank==0) std::cout << "n_matrix = " << n_matrix << ", n_grid = " << n_grid << ", != " << (n_matrix != n_grid * n_grid) << std::endl;

    if(n_matrix != n_grid * n_grid){
        if(rank==0) printf("\nerror: (dof_local * nprocs = %u) should be a squared number!\n\n", nprocs * n_matrix_local);
        MPI_Finalize();
        return -1;
    }

    index_t node, node_start, node_end; // node = global node index
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


int saena::laplacian3D(saena::matrix* A, index_t mx, index_t my, index_t mz, bool scale /*= true*/){
// NOTE: this info should be updated!
// changed the function in: petsc-3.13.5/src/ksp/ksp/tutorials/ex45.c
/*
Laplacian in 3D. Modeled by the partial differential equation
   - Laplacian u = f,    0 < x,y,z < 1,

with Dirichlet boundary conditions f(x) defined on the boundary
   x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.
*/

    MPI_Comm comm = A->get_comm();
    int rank = 0, nprocs = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    printf("rank %d: mx = %d, my = %d, mz = %d\n", rank, mx, my, mz);

    if(rank < mz) {

        int i, j, k, xm, ym, zm, xs, ys, zs, num, numi, numj, numk;
        value_t v[7], Hx, Hy, Hz, HyHzdHx, HxHzdHy, HxHydHz;
        index_t col_index[7];
        index_t node;

        Hx = 1.0 / (mx - 1);
        Hy = 1.0 / (my - 1);
        Hz = 1.0 / (mz - 1);

//        printf("\nrank %d: mx = %d, my = %d, mz = %d, Hx = %f, Hy = %f, Hz = %f\n", rank, mx, my, mz, Hx, Hy, Hz);

//        HyHzdHx = Hy * Hz / Hx;
//        HxHzdHy = Hx * Hz / Hy;
//        HxHydHz = Hx * Hy / Hz;

        HyHzdHx = 1;
        HxHzdHy = 1;
        HxHydHz = 1;

        // split the 3D grid by only the z axis. So put the whole x and y grids on processors, but split z by the number of processors.
        xs = 0;
        xm = mx;
        ys = 0;
        ym = my;
        if(mz > nprocs){
            zm = (int) floor(mz / nprocs);
            zs = rank * zm;
            if (rank == nprocs - 1)
                zm = mz - ((nprocs - 1) * zm);
        }else{ // the first mz processors each generates one 2D x,y-grid.
            zm = 1;
            zs = rank * zm;
        }

        printf("rank %d: xs = %d, ys = %d, zs = %d, xm = %d, ym = %d, zm = %d\n", rank, xs, ys, zs, xm, ym, zm);

        const int XMAX = mx - 1;
        const int YMAX = my - 1;
        const int ZMAX = mz - 1;

        for (k=zs; k<zs+zm; k++) {
            for (j=ys; j<ys+ym; j++) {
                for (i=xs; i<xs+xm; i++) {
                    node = mx * my * k + mx * j + i; // for 2D it should be = mx * j + i

//                    row.i = i; row.j = j; row.k = k;
                    if (i==0 || j==0 || k==0 || i==XMAX || j==YMAX || k==ZMAX) {
//                        v[0] = 2.0*(HxHydHz + HxHzdHy + HyHzdHx);
//                        MatSetValuesStencil(B,1,&row,1,&row,v,INSERT_VALUES);
                        A->set(node, node, 2.0*(HxHydHz + HxHzdHy + HyHzdHx));
//                        cout << node << "\t" << 2.0*(HxHydHz + HxHzdHy + HyHzdHx) << endl;
                    } else {
//                        v[0] = -HxHydHz;col[0].i = i; col[0].j = j; col[0].k = k-1;
//                        v[1] = -HxHzdHy;col[1].i = i; col[1].j = j-1; col[1].k = k;
//                        v[2] = -HyHzdHx;col[2].i = i-1; col[2].j = j; col[2].k = k;
//                        v[3] = 2.0*(HxHydHz + HxHzdHy + HyHzdHx);col[3].i = row.i; col[3].j = row.j; col[3].k = row.k;
//                        v[4] = -HyHzdHx;col[4].i = i+1; col[4].j = j; col[4].k = k;
//                        v[5] = -HxHzdHy;col[5].i = i; col[5].j = j+1; col[5].k = k;
//                        v[6] = -HxHydHz;col[6].i = i; col[6].j = j; col[6].k = k+1;
//                        ierr = MatSetValuesStencil(B,1,&row,7,col,v,INSERT_VALUES);CHKERRQ(ierr);

                        if(k - 1 != 0){
                            A->set(node, node - (mx * my), -HxHydHz);
                        }

                        if(j - 1 != 0) {
                            A->set(node, node - mx, -HxHzdHy);
                        }

                        if(i - 1 != 0) {
                            A->set(node, node - 1, -HyHzdHx);
                        }

                        A->set(node, node, 2.0*(HxHydHz + HxHzdHy + HyHzdHx));

                        if(i + 1 != XMAX) {
                            A->set(node, node + 1, -HyHzdHx);
                        }

                        if(j + 1 != YMAX) {
                            A->set(node, node + mx, -HxHzdHy);
                        }

                        if(k + 1 != ZMAX) {
                            A->set(node, node + (mx * my), -HxHydHz);
                        }
                    }
                }
            }
        }
    }

    A->assemble(scale);

    return 0;
}

int saena::laplacian3D_set_rhs(std::vector<double> &rhs, index_t mx, index_t my, index_t mz, MPI_Comm comm){
//    u = sin(2 * PETSC_PI * x) * sin(2 * PETSC_PI * y) * sin(2 * PETSC_PI * z)
//    rhs = 12 * PETSC_PI * PETSC_PI * sin(2 * PETSC_PI * x) * sin(2 * PETSC_PI * y) * sin(2 * PETSC_PI * z)

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(rank < mz) {

        int i, j, k, xm, ym, zm, xs, ys, zs;
        value_t Hx, Hy, Hz, HxHydHz, HyHzdHx, HxHzdHy;
        index_t node;

        Hx = 1.0 / (mx - 1);
        Hy = 1.0 / (my - 1);
        Hz = 1.0 / (mz - 1);

//        printf("\nrank %d: mx = %d, my = %d, mz = %d, Hx = %f, Hy = %f, Hz = %f\n", rank, mx, my, mz, Hx, Hy, Hz);

//        HxHydHz = Hx*Hy/Hz;
//        HxHzdHy = Hx*Hz/Hy;
//        HyHzdHx = Hy*Hz/Hx;

        // split the 3D grid by only the z axis. So put the whole x and y grids on processors, but split z by the number of processors.
        xs = 0;
        xm = mx;
        ys = 0;
        ym = my;
        if(mz > nprocs){
            zm = (int) floor(mz / nprocs);
            zs = rank * zm;
            if (rank == nprocs - 1)
                zm = mz - ((nprocs - 1) * zm);
        }else{ // the first mz processors generate one 2D x,y-grid on themselves.
            zm = 1;
            zs = rank * zm;
        }

//        printf("rank %d: corners: \nxs = %d, ys = %d, zs = %d, xm = %d, ym = %d, zm = %d\n", rank, xs, ys, zs, xm, ym, zm);

        const int XMAX = mx - 1;
        const int YMAX = my - 1;
        const int ZMAX = mz - 1;

        const double TWOPI       = 2 * PETSC_PI;
        const double TWOELVEPISQ = 12 * PETSC_PI * PETSC_PI;

        rhs.resize(xm * ym * zm);
//        printf("rank %d: rhs.sz = %lu\n", rank, rhs.size());

        index_t iter = 0;
        for (k=zs; k<zs+zm; k++) {
            for (j=ys; j<ys+ym; j++) {
                for (i=xs; i<xs+xm; i++) {
                    rhs[iter++] = TWOELVEPISQ * sin(TWOPI * i * Hx) * sin(TWOPI * j * Hy) * sin(TWOPI * k * Hz);
                }
            }
        }
    }

    return 0;
}

int saena::laplacian3D_check_solution(std::vector<double> &u, index_t mx, index_t my, index_t mz, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    double dif = 0, tmp = 0;

    if(rank < mz) {

        int i, j, k, xm, ym, zm, xs, ys, zs;
        value_t Hx, Hy, Hz, HxHydHz, HyHzdHx, HxHzdHy;
        index_t node;

        Hx = 1.0 / (mx - 1);
        Hy = 1.0 / (my - 1);
        Hz = 1.0 / (mz - 1);

//        printf("\nrank %d: mx = %d, my = %d, mz = %d, Hx = %f, Hy = %f, Hz = %f\n", rank, mx, my, mz, Hx, Hy, Hz);

//        HxHydHz = Hx*Hy/Hz;
//        HxHzdHy = Hx*Hz/Hy;
//        HyHzdHx = Hy*Hz/Hx;

        // split the 3D grid by only the z axis. So put the whole x and y grids on processors, but split z by the number of processors.
        xs = 0;
        xm = mx;
        ys = 0;
        ym = my;
        if(mz > nprocs){
            zm = (int) floor(mz / nprocs);
            zs = rank * zm;
            if (rank == nprocs - 1)
                zm = mz - ((nprocs - 1) * zm);
        }else{ // the first mz processors generate one 2D x,y-grid on themselves.
            zm = 1;
            zs = rank * zm;
        }

//        printf("rank %d: corners: \nxs = %d, ys = %d, zs = %d, xm = %d, ym = %d, zm = %d\n", rank, xs, ys, zs, xm, ym, zm);

        const int XMAX = mx - 1;
        const int YMAX = my - 1;
        const int ZMAX = mz - 1;

        index_t iter = 0;
        for (k=zs; k<zs+zm; k++) {
            for (j=ys; j<ys+ym; j++) {
                for (i=xs; i<xs+xm; i++) {
                    tmp = u[iter++] - sin(2 * PETSC_PI * i * Hx) * sin(2 * PETSC_PI * j * Hy)  * sin(2 * PETSC_PI * k * Hz);
                    dif += tmp * tmp;
                }
            }
        }
    }

    cout << "\nnorm of diff = " << sqrt(dif) << endl;

//    rhs[iter] = 12 * PETSC_PI * PETSC_PI
//                * sin(2 * PETSC_PI * (((value_t) i + 0.5) * Hx))
//                * sin(2 * PETSC_PI * (((value_t) j + 0.5) * Hy))
//                * sin(2 * PETSC_PI * (((value_t) k + 0.5) * Hz))
//                * Hx * Hy * Hz;
    // sin(2 * PETSC_PI * i) * sin(2 * PETSC_PI * j)  * sin(2 * PETSC_PI * k)

    return 0;
}


int saena::laplacian3D_old(saena::matrix* A, index_t n_matrix_local){

    MPI_Comm comm = A->get_comm();
    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    index_t n_matrix = nprocs * n_matrix_local;
    index_t n_grid = cbrt(n_matrix); // number of rows (or columns) of the matrix
//    if(rank==0) std::cout << "n_matrix = " << n_matrix << ", n_grid = " << n_grid << std::endl;

    if(n_matrix != n_grid * n_grid * n_grid){
        if(rank==0) printf("\nerror: (dof_local * nprocs) should be a cubed number (of power 3!)\n\n");
        MPI_Finalize();
        return -1;
    }

    index_t node, node_start, node_end; // node = global node index
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


int saena::laplacian3D_old2(saena::matrix* A, index_t mx, index_t my, index_t mz, bool scale /*= true*/){

    MPI_Comm comm = A->get_comm();
    int rank = 0, nprocs = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    printf("rank %d: mx = %d, my = %d, mz = %d\n", rank, mx, my, mz);

    if(rank < mz) {

        int i, j, k, xm, ym, zm, xs, ys, zs, num, numi, numj, numk;
        value_t v[7], Hx, Hy, Hz, HyHzdHx, HxHzdHy, HxHydHz;
        index_t col_index[7];
        index_t node;

        Hx = 1.0 / mx;
        Hy = 1.0 / my;
        Hz = 1.0 / mz;

//        printf("\nrank %d: mx = %d, my = %d, mz = %d, Hx = %f, Hy = %f, Hz = %f\n", rank, mx, my, mz, Hx, Hy, Hz);

        HyHzdHx = Hy * Hz / Hx;
        HxHzdHy = Hx * Hz / Hy;
        HxHydHz = Hx * Hy / Hz;

        // split the 3D grid by only the z axis. So put the whole x and y grids on processors, but split z by the number of processors.
        xs = 0;
        xm = mx;
        ys = 0;
        ym = my;
        if(mz > nprocs){
            zm = (int) floor(mz / nprocs);
            zs = rank * zm;
            if (rank == nprocs - 1)
                zm = mz - ((nprocs - 1) * zm);
        }else{ // the first mz processors generate one 2D x,y-grid on themselves.
            zm = 1;
            zs = rank * zm;
        }

//        printf("rank %d: xs = %d, ys = %d, zs = %d, xm = %d, ym = %d, zm = %d\n", rank, xs, ys, zs, xm, ym, zm);

        for (k = zs; k < zs + zm; k++) {
            for (j = ys; j < ys + ym; j++) {
                for (i = xs; i < xs + xm; i++) {
                    node = mx * my * k + mx * j + i; // for 2D it should be = mx * j + i
//                    if(rank==0) printf("node = %u\n", node);

                    if (i == 0 || j == 0 || k == 0 || i == mx - 1 || j == my - 1 || k == mz - 1) {
//                    if(rank==0) printf("boundary!\n");
                        num = 0;
                        numi = 0;
                        numj = 0;
                        numk = 0;
                        if (k != 0) {
                            v[num] = -HxHydHz;
//                        col[num].i = i;
//                        col[num].j = j;
//                        col[num].k = k-1;
                            col_index[num] = node - (mx * my);
                            num++;
                            numk++;
                        }
                        if (j != 0) {
                            v[num] = -HxHzdHy;
//                        col[num].i = i;
//                        col[num].j = j-1;
//                        col[num].k = k;
                            col_index[num] = node - mx;
                            num++;
                            numj++;
                        }
                        if (i != 0) {
                            v[num] = -HyHzdHx;
//                        col[num].i = i-1;
//                        col[num].j = j;
//                        col[num].k = k;
                            col_index[num] = node - 1;
                            num++;
                            numi++;
                        }
                        if (i != mx - 1) {
                            v[num] = -HyHzdHx;
//                        col[num].i = i+1;
//                        col[num].j = j;
//                        col[num].k = k;
                            col_index[num] = node + 1;
                            num++;
                            numi++;
                        }
                        if (j != my - 1) {
                            v[num] = -HxHzdHy;
//                        col[num].i = i;
//                        col[num].j = j+1;
//                        col[num].k = k;
                            col_index[num] = node + mx;
                            num++;
                            numj++;
                        }
                        if (k != mz - 1) {
                            v[num] = -HxHydHz;
//                        col[num].i = i;
//                        col[num].j = j;
//                        col[num].k = k+1;
                            col_index[num] = node + (mx * my);
                            num++;
                            numk++;
                        }
                        v[num] = (value_t) (numk) * HxHydHz + (value_t) (numj) * HxHzdHy + (value_t) (numi) * HyHzdHx;
//                    col[num].i = i;   col[num].j = j;   col[num].k = k;
                        col_index[num] = node;
                        num++;
                        for (int l = 0; l < num; l++) {
//                            if(!rank) printf("%d \t%u \t%u \t%f \n", l, node, col_index[l], v[l]);
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
                        v[3] = 2.0 * (HyHzdHx + HxHzdHy + HxHydHz);
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

//                        if(!rank)
//                            for(int l = 0; l < 7; l++)
//                                printf("%d \t%u \t%u \t%f \n", l, node, col_index[l], v[l]);
                    }
                }
            }
        }
    }

    A->assemble(scale);

    return 0;
}

int saena::laplacian3D_set_rhs_old2(std::vector<double> &rhs, index_t mx, index_t my, index_t mz, MPI_Comm comm){
    // set rhs entries using the cos() function.

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(rank < mz) {

        int i, j, k, xm, ym, zm, xs, ys, zs;
        value_t Hx, Hy, Hz;
        index_t node;

        Hx = 1.0 / mx;
        Hy = 1.0 / my;
        Hz = 1.0 / mz;
//        printf("\nrank %d: mx = %d, my = %d, mz = %d, Hx = %f, Hy = %f, Hz = %f\n", rank, mx, my, mz, Hx, Hy, Hz);

        // split the 3D grid by only the z axis. So put the whole x and y grids on processors, but split z by the number of processors.
        xs = 0;
        xm = mx;
        ys = 0;
        ym = my;
        if(mz > nprocs){
            zm = (int) floor(mz / nprocs);
            zs = rank * zm;
            if (rank == nprocs - 1)
                zm = mz - ((nprocs - 1) * zm);
        }else{ // the first mz processors generate one 2D x,y-grid on themselves.
            zm = 1;
            zs = rank * zm;
        }

//        printf("rank %d: corners: \nxs = %d, ys = %d, zs = %d, xm = %d, ym = %d, zm = %d\n", rank, xs, ys, zs, xm, ym, zm);

        rhs.resize(mx * my * zm);

        index_t iter = 0;
        for (k = zs; k < zs + zm; k++) {
            for (j = ys; j < ys + ym; j++) {
                for (i = xs; i < xs + xm; i++) {
                    node = mx * my * k + mx * j + i; // for 2D it should be = mx * j + i
                    rhs[iter] = 12 * PETSC_PI * PETSC_PI
                                * cos(2 * PETSC_PI * (((value_t) i + 0.5) * Hx))
                                * cos(2 * PETSC_PI * (((value_t) j + 0.5) * Hy))
                                * cos(2 * PETSC_PI * (((value_t) k + 0.5) * Hz))
                                * Hx * Hy * Hz;
//                    if(rank==1) printf("node = %d, rhs[node] = %f \n", node, rhs[node]);
                    iter++;
                }
            }
        }
    }

    return 0;
}


int saena::laplacian3D_old3(saena::matrix* A, index_t mx, index_t my, index_t mz, bool scale /*= true*/){
// from petsc-3.13.5/src/ksp/ksp/tutorials/ex45.c
/*
Laplacian in 3D. Modeled by the partial differential equation
   - Laplacian u = 1,0 < x,y,z < 1,

with boundary conditions
   u = 1 for x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.
*/

    MPI_Comm comm = A->get_comm();
    int rank = 0, nprocs = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

//    printf("rank %d: mx = %d, my = %d, mz = %d\n", rank, mx, my, mz);

    if(rank < mz) {

        int i, j, k, xm, ym, zm, xs, ys, zs, num, numi, numj, numk;
        value_t v[7], Hx, Hy, Hz, HyHzdHx, HxHzdHy, HxHydHz;
        index_t col_index[7];
        index_t node;

        Hx = 1.0 / (value_t)(mx - 1);
        Hy = 1.0 / (value_t)(my - 1);
        Hz = 1.0 / (value_t)(mz - 1);

//        printf("\nrank %d: mx = %d, my = %d, mz = %d, Hx = %f, Hy = %f, Hz = %f\n", rank, mx, my, mz, Hx, Hy, Hz);

        HyHzdHx = Hy * Hz / Hx;
        HxHzdHy = Hx * Hz / Hy;
        HxHydHz = Hx * Hy / Hz;

        // split the 3D grid by only the z axis. So put the whole x and y grids on processors, but split z by the number of processors.
        xs = 0;
        xm = mx;
        ys = 0;
        ym = my;
        if(mz > nprocs){
            zm = (int) floor(mz / nprocs);
            zs = rank * zm;
            if (rank == nprocs - 1)
                zm = mz - ((nprocs - 1) * zm);
        }else{ // the first mz processors generate one 2D x,y-grid on themselves.
            zm = 1;
            zs = rank * zm;
        }

//        printf("rank %d: xs = %d, ys = %d, zs = %d, xm = %d, ym = %d, zm = %d\n", rank, xs, ys, zs, xm, ym, zm);

        for (k=zs; k<zs+zm; k++) {
            for (j=ys; j<ys+ym; j++) {
                for (i=xs; i<xs+xm; i++) {
                    node = mx * my * k + mx * j + i; // for 2D it should be = mx * j + i

//                    row.i = i; row.j = j; row.k = k;
                    if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1) {
//                        v[0] = 2.0*(HxHydHz + HxHzdHy + HyHzdHx);
//                        MatSetValuesStencil(B,1,&row,1,&row,v,INSERT_VALUES);
                        A->set(node, node, 2.0*(HxHydHz + HxHzdHy + HyHzdHx));
                    } else {
//                        v[0] = -HxHydHz;col[0].i = i; col[0].j = j; col[0].k = k-1;
//                        v[1] = -HxHzdHy;col[1].i = i; col[1].j = j-1; col[1].k = k;
//                        v[2] = -HyHzdHx;col[2].i = i-1; col[2].j = j; col[2].k = k;
//                        v[3] = 2.0*(HxHydHz + HxHzdHy + HyHzdHx);col[3].i = row.i; col[3].j = row.j; col[3].k = row.k;
//                        v[4] = -HyHzdHx;col[4].i = i+1; col[4].j = j; col[4].k = k;
//                        v[5] = -HxHzdHy;col[5].i = i; col[5].j = j+1; col[5].k = k;
//                        v[6] = -HxHydHz;col[6].i = i; col[6].j = j; col[6].k = k+1;
//                        ierr = MatSetValuesStencil(B,1,&row,7,col,v,INSERT_VALUES);CHKERRQ(ierr);

                        A->set(node, node - (mx * my), -HxHydHz);
                        A->set(node, node - mx,        -HxHzdHy);
                        A->set(node, node - 1,         -HyHzdHx);
                        A->set(node, node,             2.0*(HxHydHz + HxHzdHy + HyHzdHx));
                        A->set(node, node + 1,         -HyHzdHx);
                        A->set(node, node + mx ,       -HxHzdHy);
                        A->set(node, node + (mx * my), -HxHydHz);
                    }
                }
            }
        }
    }

    A->assemble(scale);

    return 0;
}

int saena::laplacian3D_set_rhs_old3(std::vector<double> &rhs, index_t mx, index_t my, index_t mz, MPI_Comm comm){
    // set rhs entries using the cos() function.

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(rank < mz) {

        int i, j, k, xm, ym, zm, xs, ys, zs;
        value_t Hx, Hy, Hz, HxHydHz, HyHzdHx, HxHzdHy;
        index_t node;

        Hx = 1.0 / (value_t)(mx - 1);
        Hy = 1.0 / (value_t)(my - 1);
        Hz = 1.0 / (value_t)(mz - 1);

//        printf("\nrank %d: mx = %d, my = %d, mz = %d, Hx = %f, Hy = %f, Hz = %f\n", rank, mx, my, mz, Hx, Hy, Hz);

        HxHydHz = Hx*Hy/Hz;
        HxHzdHy = Hx*Hz/Hy;
        HyHzdHx = Hy*Hz/Hx;

        // split the 3D grid by only the z axis. So put the whole x and y grids on processors, but split z by the number of processors.
        xs = 0;
        xm = mx - 1;
        ys = 0;
        ym = my - 1;
        if((mz - 1) > nprocs){
            zm = (int) floor( (mz - 1) / nprocs);
            zs = rank * zm;
            if (rank == nprocs - 1)
                zm = (mz - 1) - ((nprocs - 1) * zm);
        }else{ // the first (mz - 1) processors generate one 2D x,y-grid on themselves.
            zm = 1;
            zs = rank * zm;
        }

//        printf("rank %d: corners: \nxs = %d, ys = %d, zs = %d, xm = %d, ym = %d, zm = %d\n", rank, xs, ys, zs, xm, ym, zm);

        rhs.resize((mx - 1) * (my - 1) * (zm - 1));

        index_t iter = 0;
        for (k=zs; k<zs+zm; k++) {
            for (j=ys; j<ys+ym; j++) {
                for (i=xs; i<xs+xm; i++) {
                    if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1) {
//                        barray[k][j][i] = 2.0*(HxHydHz + HxHzdHy + HyHzdHx);
                        rhs[iter] = 2.0*(HxHydHz + HxHzdHy + HyHzdHx);
                    } else {
//                        barray[k][j][i] = Hx*Hy*Hz;
                        rhs[iter] = Hx*Hy*Hz;
                    }
                }
            }
        }
    }

//    rhs[iter] = 12 * PETSC_PI * PETSC_PI
//                * sin(2 * PETSC_PI * (((value_t) i + 0.5) * Hx))
//                * sin(2 * PETSC_PI * (((value_t) j + 0.5) * Hy))
//                * sin(2 * PETSC_PI * (((value_t) k + 0.5) * Hz))
//                * Hx * Hy * Hz;
    // sin(2 * PETSC_PI * i) * sin(2 * PETSC_PI * j)  * sin(2 * PETSC_PI * k)

    return 0;
}


int saena::laplacian3D_set_rhs_zero(std::vector<double> &rhs, unsigned int mx, unsigned int my, unsigned int mz, MPI_Comm comm){
    // set rhs entries corresponding to boundary points to zero.

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    int     i,j,k,xm,ym,zm,xs,ys,zs,num, numi, numj, numk;
    value_t v[7],Hx,Hy,Hz,HyHzdHx,HxHzdHy,HxHydHz;
    index_t col_index[7];
    index_t node;

    Hx      = 1.0 / (value_t)(mx);
    Hy      = 1.0 / (value_t)(my);
    Hz      = 1.0 / (value_t)(mz);
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
                if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1) {
                    rhs[node] = 0;
                }
            }
        }
    }

    return 0;
}


int saena::band_matrix(saena::matrix &A, index_t M, unsigned int bandwidth){
    // generates a band matrix with bandwidth of size "bandwidth".
    // set bandwidth to 0 to have a diagonal matrix.
    // value of entry (i,j) = 1 / (i + j + 1)

    int rank, nprocs;
    MPI_Comm_size(A.get_comm(), &nprocs);
    MPI_Comm_rank(A.get_comm(), &rank);

    index_t Mbig = M * nprocs;
//    printf("rank %d: M = %u, Mbig = %u \n", rank, M, Mbig);

    if(bandwidth >= Mbig){
        printf("Error: bandwidth is greater than the size of the matrix\n");
        exit(EXIT_FAILURE);
    }

    //Type of random number distribution
//    std::uniform_real_distribution<value_t> dist(0, 1); //(min, max)
    //Mersenne Twister: Good quality random number generator
//    std::mt19937 rng;
    //Initialize with non-deterministic seeds
//    rng.seed(std::random_device{}());

    value_t val = 1;
    index_t d;
    for(index_t i = rank*M; i < (rank+1)*M; i++){
        d = 0;
        for(index_t j = i; j <= i+bandwidth; j++){
//            val = dist(rng); // comment out this to have all values equal to 1.
            val = 1.0 / (i + j + 1); // comment out this to have all values equal to 1.
            if(i==j){
//                printf("%u \t%u \t%f \n", i, j, val);
                A.set(i, j, val);
            }else{
                if(j < Mbig){
//                    printf("%u \t%u \t%f \n", i, j, val);
                    A.set(i, j, val);
                }
                if(j >= 2*d) { // equivalent to if(j-2*d >= 0)
//                    printf("%u \t%u \t%f \n", i, j - (2 * d), 1.0 / (i + j - (2 * d) + 1);
                    A.set(i, j - (2 * d), 1.0 / (i + j - (2 * d) + 1));
                }
            }
            d++;
        }
    }

    saena_matrix *B = A.get_internal_matrix();

//    B->print_entry(-1);
//    std::sort(B->data_coo.begin(), B->data_coo.end());
//    B->print_entry(-1);

    B->entry.resize(B->data_coo.size());
    nnz_t iter = 0;
    for(auto i:B->data_coo){
        B->entry[iter] = cooEntry(i.row, i.col, i.val);
        iter++;
    }
    std::sort(B->entry.begin(), B->entry.end());

    B->active = true;
    B->nnz_l = iter;
    MPI_Allreduce(&iter, &B->nnz_g, 1, par::Mpi_datatype<nnz_t>::value(), MPI_SUM, A.get_comm());
    B->Mbig = Mbig;
    B->M = M;
    B->density = ((double)B->nnz_g / B->Mbig) / B->Mbig;
    B->split.resize(nprocs+1);
    for(index_t i = 0; i < nprocs+1; i++){
        B->split[i] = i*M;
    }

//    B->print_entry(-1);

//    A.assemble();
    A.assemble_band_matrix();

//    printf("rank %d: M = %u, Mbig = %u, nnz_l = %lu, nnz_g = %lu \n",
//            rank, A.get_num_local_rows(), A.get_num_rows(), A.get_local_nnz(), A.get_nnz());
//    if(!rank) printf("Mbig = %u, nnz_g = %lu, density = %.8f \n", A.get_num_rows(), A.get_nnz(), A.get_internal_matrix()->density);

    return 0;
}


int saena::random_symm_matrix(saena::matrix &A, index_t M, float density){
    // generates a random matrix with density of size "density".

    int rank, nprocs;
    MPI_Comm_size(A.get_comm(), &nprocs);
    MPI_Comm_rank(A.get_comm(), &rank);

    if(density <= 0 || density > 1){
        if(!rank) printf("Error: density should be in the range (0,1].\n");
        exit(EXIT_FAILURE);
    }

    index_t       Mbig  = nprocs * M;
    unsigned long nnz_l = floor(density * M * Mbig);

//    if(nnz_l < M){
//        printf("\nrank %d: The diagonal entries should be nonzero, so the density is increased to satisfy that.\n"
//               "nnz_l = %ld, density = %f, M = %u, Mbig = %u\n", rank, nnz_l, density, M, Mbig);
//    }

    //Type of random number distribution
    std::uniform_real_distribution<value_t> dist(0, 1);       //(min, max). this one is for the value of the entries.
    std::uniform_int_distribution<index_t>  dist2(0, M-1);    //(min, max). this one is for the row indices.
    std::uniform_int_distribution<index_t>  dist3(0, Mbig-1); //(min, max). this one is for the column indices.

    //Mersenne Twister: Good quality random number generator
    std::mt19937 rng(std::random_device{}());
    std::mt19937 rng2(std::random_device{}());
    std::mt19937 rng3(std::random_device{}());

    index_t offset = M * rank;

//    MPI_Barrier(comm);
//    std::cout << "\nM: " << M << ", Mbig: " << Mbig << ", nnz_l: " << nnz_l << ", nnz_g: " << nnz_l * nprocs
//              << ", offset: " << offset << ", density: " << density << std::endl;
//    MPI_Barrier(comm);

    // add the diagonal
    index_t M_end = offset + M;
    if(rank == nprocs - 1){
        M_end = Mbig;
    }
    for(index_t i = offset; i < M_end; i++){
        A.set(i , i, dist(rng));
    }

    index_t ii, jj;
    value_t vv;

    // The diagonal entries are added. Also, to keep the matrix symmetric, for each generated entry (i, j, v),
    // the entry (j, i, v) is added.
    unsigned long nnz_l_updated = (nnz_l - M) / 2;

    if(nnz_l > M){
        while(nnz_l_updated) {
            vv = dist(rng);
            ii = dist2(rng2) + offset;
            jj = dist3(rng3);
            if(ii > jj){
                nnz_l_updated--;
                A.set(ii, jj, vv);
                A.set(jj, ii, vv);        // to keep the matrix symmetric

//                if(rank == 1) {
//                    std::cout << ii << "\t" << jj << "\t" << vv << std::endl;
//                    printf("nnz_l_updated: %ld \n\n", nnz_l_updated);
//                }
            }
        }
    }

    A.assemble();
//    A.print(0);

    return 0;
}


int saena::read_vector_file(std::vector<value_t>& v, saena::matrix &A, char *file, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // check if the size of rhs match the number of rows of A
//    struct stat st;
//    stat(file, &st);
//    unsigned int rhs_size = st.st_size / sizeof(double);
//    if(rhs_size != A->Mbig){
//        if(rank==0) printf("Error: Size of RHS does not match the number of rows of the LHS matrix!\n");
//        if(rank==0) printf("Number of rows of LHS = %d\n", A->Mbig);
//        if(rank==0) printf("Size of RHS = %d\n", rhs_size);
//        MPI_Finalize();
//        return -1;
//    }

    MPI_Status status;
    MPI_File fh;
    MPI_Offset offset;

    int mpiopen = MPI_File_open(comm, file, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if(mpiopen){
        if (rank==0) std::cout << "Unable to open the rhs vector file!" << std::endl;
        MPI_Finalize();
        return -1;
    }

    // define the size of v as the local number of rows on each process
//    std::vector <double> v(A.M);
    v.resize(A.get_num_local_rows());
    double* vp = &(*(v.begin()));

    // vector should have the following format: first line shows the value in row 0, second line shows the value in row 1
    offset = A.get_internal_matrix()->split[rank] * 8; // value(double=8)
    MPI_File_read_at(fh, offset, vp, A.get_num_local_rows(), MPI_DOUBLE, &status);

//    int count;
//    MPI_Get_count(&status, MPI_UNSIGNED_LONG, &count);
    //printf("process %d read %d lines of triples\n", rank, count);
    MPI_File_close(&fh);

//    print_vector(v, -1, "v", comm);

    return 0;
}


index_t saena::find_split(index_t loc_size, index_t &my_split, MPI_Comm comm){

    int rank, nprocs;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::vector<index_t> split_temp(nprocs);
    split_temp[rank] = loc_size;
    MPI_Allgather(&loc_size,      1, par::Mpi_datatype<index_t>::value(),
                  &split_temp[0], 1, par::Mpi_datatype<index_t>::value(), comm);

    std::vector<index_t> split_vec(nprocs+1);
    split_vec[0] = 0;
    for(index_t i = 1; i < nprocs + 1; i++){
        split_vec[i] = split_vec[i-1] + split_temp[i-1];
    }

//    print_vector(split_vec, 0, "split_vec", comm);

    my_split = split_vec[rank];
    return 0;
}
