#include <vector>
#include <string>
#include <cstring>
#include <mpi.h>

#include "saena.hpp"
#include "saena_matrix.h"
#include "saena_object.h"
#include "pugixml.hpp"

// ******************************* matrix *******************************

saena::matrix::matrix(unsigned int num_rows_global, MPI_Comm comm) {
    m_pImpl = new saena_matrix(num_rows_global, comm);
}

saena::matrix::matrix(char *name, unsigned int global_rows, MPI_Comm comm) {
    m_pImpl = new saena_matrix(name, global_rows, comm);
}


int saena::matrix::set(unsigned int i, unsigned int j, double val){
    if( val != 0)
        m_pImpl->set(i, j, val);
    return 0;
}

int saena::matrix::set(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local){
    m_pImpl->set(row, col, val, nnz_local);
    return 0;
}

int saena::matrix::set(unsigned int i, unsigned int j, unsigned int size_x, unsigned int size_y, double* val){
// ordering of val should be first columns, then rows.
    unsigned int ii, jj, iter;
    iter = 0;
    for(jj = 0; jj < size_y; jj++) {
        for(ii = 0; ii < size_x; ii++) {
            if( val[iter] != 0){
                m_pImpl->set(i+ii, j+jj, val[iter]);
                iter++;
            }
        }
    }
    return 0;
}

int saena::matrix::set(unsigned int i, unsigned int j, unsigned int* di, unsigned int* dj, double* val, unsigned int nnz_local){
    unsigned int ii;
    for(ii = 0; ii < nnz_local; ii++) {
        if(val[ii] != 0)
            m_pImpl->set(i+di[ii], j+dj[ii], val[ii]);
    }
    return 0;
}


int saena::matrix::set2(unsigned int i, unsigned int j, double val){
    if( val != 0)
        m_pImpl->set2(i, j, val);
    return 0;
}

int saena::matrix::set2(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local){
    m_pImpl->set2(row, col, val, nnz_local);
    return 0;
}

int saena::matrix::set2(unsigned int i, unsigned int j, unsigned int size_x, unsigned int size_y, double* val){
// ordering of val should be first columns, then rows.
    unsigned int ii, jj, iter;
    iter = 0;
    for(jj = 0; jj < size_y; jj++) {
        for(ii = 0; ii < size_x; ii++) {
            if( val[iter] != 0){
                m_pImpl->set2(i+ii, j+jj, val[iter]);
                iter++;
            }
        }
    }
    return 0;
}

int saena::matrix::set2(unsigned int i, unsigned int j, unsigned int* di, unsigned int* dj, double* val, unsigned int nnz_local){
    unsigned int ii;
    for(ii = 0; ii < nnz_local; ii++) {
        if(val[ii] != 0)
            m_pImpl->set2(i+di[ii], j+dj[ii], val[ii]);
    }
    return 0;
}


int saena::matrix::assemble() {
    m_pImpl->setup_initial_data();
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

saena::amg::amg(saena::matrix* A){
    m_pImpl = new saena_object();
    m_pImpl->setup(A->get_internal_matrix());
}

void saena::amg::solve(std::vector<double>& u, std::vector<double>& rhs, saena::options* opts){
    m_pImpl->set_parameters(opts->get_vcycle_num(), opts->get_relative_tolerance(),
                            opts->get_smoother(), opts->get_preSmooth(), opts->get_postSmooth());
    m_pImpl->solve(u, rhs);
}

void saena::amg::save_to_file(char* name, unsigned int* agg){

}

unsigned int* saena::amg::load_from_file(char* name){
    return nullptr;
}

void saena::amg::destroy(){
    // will add later.
}