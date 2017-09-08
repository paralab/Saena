#include <vector>
#include <string>
#include <mpi.h>

#include "saena.hpp"
#include "SaenaMatrix.h"
#include "SaenaObject.h"
#include "pugixml.hpp"

// ******************************* matrix *******************************

saena::matrix::matrix(unsigned int num_rows_global, MPI_Comm comm) {
    m_pImpl = new SaenaMatrix(num_rows_global, comm);
}

saena::matrix::matrix(char *name, unsigned int global_rows, MPI_Comm comm) {
    m_pImpl = new SaenaMatrix(name, global_rows, comm);
}

//int saena::matrix::reserve(unsigned int nnz_local){
//    return 0;
//}

int saena::matrix::set(unsigned int row, unsigned int col, double val){
    m_pImpl->set(row, col, val);
    return 0;

}

int saena::matrix::set(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local){
    m_pImpl->set(row, col, val, nnz_local);
    return 0;
}

int saena::matrix::set(unsigned int row_offset, unsigned int col_offset, unsigned int block_size, double* values){
    return 0;
}

int saena::matrix::set(unsigned int global_row_offset, unsigned int global_col_offset, unsigned int* local_row_offset,
                        unsigned int* local_col_offset, double* values){
    return 0;
}

int saena::matrix::assemble() {
    m_pImpl->setup_initial_data();
    m_pImpl->repartition();
    m_pImpl->matrixSetup();
    return 0;
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

saena::options::options(char* name){
    pugi::xml_document doc;
    if (!doc.load_file(name))
        std::cout << "Could not find the xml file!" << std::endl;

    pugi::xml_node opts = doc.child("SAENA").first_child();

    pugi::xml_attribute attr = opts.first_attribute();
    vcycle_num = stoi(attr.value());

    attr = attr.next_attribute();
    relative_tolerance = stod(attr.value());

    attr = attr.next_attribute();
    smoother = attr.value();

    attr = attr.next_attribute();
    preSmooth = stoi(attr.value());

    attr = attr.next_attribute();
    postSmooth = stoi(attr.value());

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
    return 0;
}

void saena::amg::destroy(){
    // will add later.
}