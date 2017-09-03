#include <vector>
#include <string>
#include <mpi.h>

#include "saena.hpp"
#include "SaenaMatrix.h"
#include "SaenaObject.h"

// ******************************* matrix *******************************

saena::matrix::matrix(char *name, unsigned int global_rows, MPI_Comm comm) {
    m_pImpl = new SaenaMatrix(name, global_rows, comm);
}

void saena::matrix::set(unsigned int row, unsigned int col, double val){
    // will add later.
}

void saena::matrix::set(unsigned int* row, unsigned int* col, double* val){
    // will add later.
}

void saena::matrix::set(unsigned int row_offset, unsigned int col_offset, unsigned int block_size, double* values){

}
void saena::matrix::set(unsigned int global_row_offset, unsigned int global_col_offset, unsigned int* local_row_offset,
                        unsigned int* local_col_offset, double* values){

}

void saena::matrix::init(MPI_Comm comm) {
    m_pImpl->repartition(comm);
    m_pImpl->matrixSetup(comm);
}

unsigned int saena::matrix::get_num_local_rows() {
    return m_pImpl->M;
}


SaenaMatrix* saena::matrix::get_internal_matrix(){
    return m_pImpl;
}

void saena::matrix::destroy(){
    // will add later.
}

// ******************************* options *******************************

saena::options::options(int vcycle_n, double relT, std::string sm, int preSm, int postSm){
    vcycle_num         = vcycle_n;
    relative_tolerance = relT;
    smoother           = sm;
    preSmooth          = preSm;
    postSmooth         = postSm;
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


//int saena::options::get_max_level(){
//    return max_level;
//}

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

saena::amg::amg(saena::matrix* A, int max_level){
    m_pImpl = new SaenaObject(max_level);
    m_pImpl->Setup(A->get_internal_matrix());
}

void saena::amg::solve(std::vector<double>& u, std::vector<double>& rhs, saena::options* opts){
    m_pImpl->set_parameters(opts->get_vcycle_num(), opts->get_relative_tolerance(),
                            opts->get_smoother(), opts->get_preSmooth(), opts->get_postSmooth());
    m_pImpl->Solve(u, rhs);
}

void saena::amg::destroy(){
    // will add later.
}