#include "saena_matrix.h"


saena_matrix::saena_matrix() = default;


saena_matrix::saena_matrix(MPI_Comm com) {
    comm = com;
}


void saena_matrix::set_comm(MPI_Comm com){
    comm = com;
}


int saena_matrix::read_file(const string &filename, const std::string &input_type /* = "" */) {
    // the following variables of saena_matrix class will be set in this function:
    // Mbig", "nnz_g", "initial_nnz_l", "data"
    // "data" is only required for repartition function.

    int rank = 0, nprocs = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    read_from_file = true;

    // check if file exists
    // ====================
    std::ifstream inFile_check(filename.c_str());
    if (!inFile_check.is_open()) {
        if (!rank) std::cout << "\nCould not open the matrix file <" << filename << ">" << std::endl;
        MPI_Finalize();
        exit(EXIT_FAILURE);
    }
    inFile_check.close();

    // find extension of the file
    // ==========================
    size_t extIndex = filename.find_last_of('.');
    if(extIndex == string::npos || extIndex == filename.size() - 1){
        if (!rank) cout << "The matrix file name does not have an extension!" << endl;
        MPI_Abort(comm, 1);
    }

    std::string file_extension = filename.substr(extIndex+1, string::npos); // string::npos -> the end of the string
    std::string outFileName = filename.substr(0, extIndex) + ".bin";

//    if(rank==0) std::cout << "file_extension: " << file_extension << std::endl;
//    if(rank==0) std::cout << "outFileName: " << outFileName << std::endl;

    // if the file is not binary, generate its binary file
    // ===================================================
    if(file_extension != "bin") {
        if (file_extension == "mtx") {

            std::ifstream inFile_check_bin(outFileName.c_str());

            if (inFile_check_bin.is_open()) {

//                if (rank == 0)
//                    std::cout <<"A binary file with the same name exists. Using that file instead of the mtx file.\n\n";
                inFile_check_bin.close();

            } else {

                // write the file in binary by proc 0.
                if (rank == 0) {

//                    std::cout << "First a binary file with name \"" << outFileName
//                              << "\" will be created in the same directory. \n\n";

                    std::ifstream inFile(filename.c_str());

                    // the fourth word in the Matrix Market format is the data type of the matrix
                    // example: %%MatrixMarket matrix coordinate pattern symmetric
                    // it can be real, complex, pattern, integer
                    // "pattern" means there is no value, so just consider 1 as the value of the entries.
                    // if it is not "pattern", then check the next word. If it is "symmetric",
                    // then set mat_type to "triangle".

                    string mat_type(input_type);
                    string tmp;

                    if(mat_type.empty()){
                        inFile >> tmp;
                        if(tmp == "%%MatrixMarket"){ // check if there is metadata at the first line of the file.
                            for(int i = 1; i < 4; ++i){
                                inFile >> tmp;
                            }

                            if(tmp != "pattern"){
                                inFile >> tmp;
                                if(tmp == "symmetric"){
                                    mat_type = "triangle";
                                }
//                                else{
//                                    std::cerr << "the input type is not valid!" << std::endl;
//                                    MPI_Finalize();
//                                    exit(EXIT_FAILURE);
//                                }
                            }else{
                                mat_type = tmp;
                            }
                        }
                        // reset input file position
                        inFile.seekg(0, ios::beg);
                    }

//                    cout << mat_type << endl;

                    // ignore comments
                    while (inFile.peek() == '%') inFile.ignore(2048, '\n');

                    // M and N are the size of the matrix with nnz nonzeros
                    nnz_t M_in = 0, N_in = 0, nnz = 0;
                    inFile >> M_in >> N_in >> nnz;

//                    printf("M = %ld, N = %ld, nnz = %ld \n", M_in, N_in, nnz);

                    std::ofstream outFile;
                    outFile.open(outFileName.c_str(), std::ios::out | std::ios::binary);

                    std::vector<cooEntry> entry_temp1;
//                    std::vector<cooEntry> entry;
                    // number of nonzeros is less than 2*nnz, considering the diagonal
                    // that's why there is a resize for entry when nnz is found.

                    index_t a = 0, b = 0, i = 0;
                    value_t c = 0.0;

                    if (mat_type.empty()) {

                        entry_temp1.resize(nnz);
                        while (inFile >> a >> b >> c) {
                            // for mtx format, rows and columns start from 1, instead of 0.
//                            std::cout << "a = " << a << ", b = " << b << ", value = " << c << std::endl;
                            entry_temp1[i++] = cooEntry(a - 1, b - 1, c);
//                            cout << entry_temp1[i] << endl;
                        }

                    } else if (mat_type == "triangle") {

                        entry_temp1.resize(2 * nnz);
                        while (inFile >> a >> b >> c) {
                            // for mtx format, rows and columns start from 1, instead of 0.
//                        std::cout << "a = " << a << ", b = " << b << ", value = " << c << std::endl;
                            entry_temp1[i++] = cooEntry(a - 1, b - 1, c);
//                        cout << entry_temp1[i] << endl;
                            // add the lower triangle, not any diagonal entry
                            if (a != b) {
                                entry_temp1[i++] = cooEntry(b - 1, a - 1, c);
                                ++nnz;
                            }
                        }
                        entry_temp1.resize(nnz);

                    } else if (mat_type == "pattern") { // add 1 for value for a pattern matrix

                        while (inFile >> a >> b) {
                            // for mtx format, rows and columns start from 1, instead of 0.
//                            std::cout << "a = " << a << ", b = " << b << std::endl;
                            entry_temp1.emplace_back(a - 1, b - 1, 1.0);
//                            cout << entry_temp1.back() << endl;
                            if (a != b) {
                                entry_temp1.emplace_back(b - 1, a - 1, 1.0);
//                                cout << entry_temp1.back() << endl;
                            }
                        }

                        nnz = entry_temp1.size();

                    } else if (mat_type == "tripattern") {

                        entry_temp1.resize(2 * nnz);
                        while (inFile >> a >> b) {
                            // for mtx format, rows and columns start from 1, instead of 0.
//                        std::cout << "a = " << a << ", b = " << b << std::endl;
                            entry_temp1[i++] = cooEntry(a - 1, b - 1, 1.0);
//                        std::cout << entry_temp1[i] << std::endl;

                            // add the lower triangle, not any diagonal entry
                            if (a != b) {
                                entry_temp1[i++] = cooEntry(b - 1, a - 1, 1.0);
                                nnz++;
                            }
                        }
                        entry_temp1.resize(nnz);

                    } else {
                        std::cerr << "the input type is not valid!" << std::endl;
                        MPI_Finalize();
                        exit(EXIT_FAILURE);
                    }

                    std::sort(entry_temp1.begin(), entry_temp1.end());

                    for (i = 0; i < nnz; ++i) {
//                        std::cout << entry_temp1[i] << std::endl;
                        outFile.write((char *) &entry_temp1[i].row, sizeof(index_t));
                        outFile.write((char *) &entry_temp1[i].col, sizeof(index_t));
                        outFile.write((char *) &entry_temp1[i].val, sizeof(value_t));
                    }

                    inFile.close();
                    outFile.close();
                }

            }

            // wait until the binary file being written by proc 0 is ready.
            MPI_Barrier(comm);

        } else if (file_extension == "dat") { // dense matrix

            std::ifstream inFile_check_bin(outFileName.c_str());

            if (inFile_check_bin.is_open()) {

//                if (rank == 0)
//                    std::cout << "\nA binary file with the same name exists. Using that file instead of the mtx"
//                                 " file.\n\n";
                inFile_check_bin.close();

            } else {

                if (rank == 0) {

//                    std::cout << "\nFirst a binary file with name \"" << outFileName
//                              << "\" will be created in the same directory. \n\n";

                    std::ifstream inFile(filename.c_str());

                    if (!inFile.is_open()) {
                        std::cout << "Could not open the file!" << std::endl;
                        return -1;
                    }

                    // ignore comments
                    while (inFile.peek() == '%') inFile.ignore(2048, '\n');

                    std::ofstream outFile;
                    outFile.open(outFileName.c_str(), std::ios::out | std::ios::binary);

                    std::vector<cooEntry> entry_temp1;

                    std::string line;
                    double temp = 0.0;
                    index_t row = 0, col = 0;

                    while (std::getline(inFile, line)) {
                        std::istringstream iss(line);

                        col = 0;
                        while (iss >> temp) {
                            if (temp != 0) {
                                entry_temp1.emplace_back(cooEntry(row, col, temp));
                            }
                            col++;
                        }
                        row++;
                    }

                    std::sort(entry_temp1.begin(), entry_temp1.end());

//                print_vector(entry_temp1, 0, "entry_temp1", comm);

                    for (auto const &num : entry_temp1) {
//                        std::cout << entry_temp1[i] << std::endl;
                        outFile.write((char *) &num.row, sizeof(index_t));
                        outFile.write((char *) &num.col, sizeof(index_t));
                        outFile.write((char *) &num.val, sizeof(value_t));
                    }

                    inFile.close();
                    outFile.close();
                }
            }

        } else {
            if (rank == 0)
                printf("The extension of file should be either mtx (matrix market)"
                       " or bin (binary) or dat (dense)! \n");
        }
    }

    // find number of general nonzeros of the input matrix
    struct stat st;
    if(stat(outFileName.c_str(), &st)){
        if (!rank) std::cout << "\nCould not open file <" << filename << ">" << std::endl;
        MPI_Barrier(comm);
        exit(EXIT_FAILURE);
    }

    nnz_g = st.st_size / (2*sizeof(index_t) + sizeof(value_t));

    if(nnz_g == 0){
        std::ostringstream errmsg;
        errmsg << "number of nonzeros is 0 on rank " << rank << " inside function " << __func__ << std::endl;
        std::cout << errmsg.str();
        exit(EXIT_FAILURE);
    }

    // find initial local nonzero
    initial_nnz_l = nnz_t(floor(1.0 * nnz_g / nprocs)); // initial local nnz
    if (rank == nprocs - 1) {
        initial_nnz_l = nnz_g - (nprocs - 1) * initial_nnz_l;
    }

    if(verbose_saena_matrix){
        MPI_Barrier(comm);
        printf("saena_matrix: part 1. rank = %d, nnz_g = %lu, initial_nnz_l = %lu \n", rank, nnz_g, initial_nnz_l);
        MPI_Barrier(comm);}

//    printf("\nrank = %d, nnz_g = %lu, initial_nnz_l = %lu \n", rank, nnz_g, initial_nnz_l);

    data_unsorted.resize(initial_nnz_l);
    cooEntry_row* datap = &data_unsorted[0];

    // *************************** read the matrix ****************************

    MPI_Status status;
    MPI_File fh;
    MPI_Offset offset;

    int mpiopen = MPI_File_open(comm, outFileName.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (mpiopen) {
        if (rank == 0) std::cout << "Unable to open the matrix file!" << std::endl;
        MPI_Finalize();
    }

    //offset = rank * initial_nnz_l * 24; // row index(long=8) + column index(long=8) + value(double=8) = 24
    // the offset for the last process will be wrong if you use the above formula,
    // because initial_nnz_l of the last process will be used, instead of the initial_nnz_l of the other processes.

    offset = rank * nnz_t(floor(1.0 * nnz_g / nprocs)) * (2*sizeof(index_t) + sizeof(value_t));

    MPI_File_read_at(fh, offset, datap, initial_nnz_l, cooEntry_row::mpi_datatype(), &status);

//    int count;
//    MPI_Get_count(&status, MPI_UNSIGNED_LONG, &count);
    //printf("process %d read %d lines of triples\n", rank, count);
    MPI_File_close(&fh);

//    print_vector(data_unsorted, -1, "data_unsorted", comm);
//    printf("rank = %d \t\t\t before sort: data_unsorted size = %lu\n", rank, data_unsorted.size());

    remove_duplicates();

    if(remove_boundary){
        remove_boundary_nodes();

        index_t Mbig_local = 0;
        if(!data.empty())
            Mbig_local = data.back().row;
        MPI_Allreduce(&Mbig_local, &Mbig, 1, par::Mpi_datatype<index_t>::value(), MPI_MAX, comm);
        Mbig++; // since indices start from 0, not 1.
    }else{
        data = std::move(data_with_bound);
    }

    // after removing duplicates, initial_nnz_l and nnz_g will be smaller, so update them.
    initial_nnz_l = data.size();
    MPI_Allreduce(&initial_nnz_l, &nnz_g, 1, par::Mpi_datatype<nnz_t>::value(), MPI_SUM, comm);

    // *************************** find Mbig (global number of rows) ****************************
    // Since data[] has row-major order, the last element on the last process is the number of rows.
    // Broadcast it from the last process to the other processes.

//    cooEntry last_element = data.back();
//    Mbig = last_element.row + 1; // since indices start from 0, not 1.
//    MPI_Bcast(&Mbig, 1, par::Mpi_datatype<index_t>::value(), nprocs-1, comm);

    if(verbose_saena_matrix){
        MPI_Barrier(comm);
        printf("saena_matrix: part 2. rank = %d, nnz_g = %lu, initial_nnz_l = %lu, Mbig = %u \n", rank, nnz_g, initial_nnz_l, Mbig);
        MPI_Barrier(comm);
    }

//    print_vector(data, -1, "data", comm);

    return 0;
}


saena_matrix::~saena_matrix(){
#ifdef SAENA_USE_ZFP
    if(free_zfp_buff){
        deallocate_zfp();
    }
#endif
};


int saena_matrix::set(index_t row, index_t col, value_t val){

    cooEntry_row temp_new = cooEntry_row(row, col, val);
    std::pair<std::set<cooEntry_row>::iterator, bool> p = data_coo.insert(temp_new);

    if (!p.second){
        auto hint = p.first; // hint is std::set<cooEntry>::iterator
        hint++;
        data_coo.erase(p.first);
        // in the case of duplicate, if the new value is zero, remove the older one and don't insert the zero.
        if(!almost_zero(val))
            data_coo.insert(hint, temp_new);
    }

    // if the entry is zero and it was not a duplicate, just erase it.
    if(p.second && almost_zero(val))
        data_coo.erase(p.first);

    return 0;
}


int saena_matrix::set(index_t* row, index_t* col, value_t* val, nnz_t nnz_local){

    if(nnz_local <= 0){
        printf("size in the set function is either zero or negative!");
        return 0;
    }

    cooEntry_row temp_new;
    std::pair<std::set<cooEntry_row>::iterator, bool> p;

    // todo: isn't it faster to allocate memory for nnz_local, then assign, instead of inserting one by one.
    for(nnz_t i=0; i<nnz_local; i++){

        temp_new = cooEntry_row(row[i], col[i], val[i]);
        p = data_coo.insert(temp_new);

        if (!p.second){
            auto hint = p.first; // hint is std::set<cooEntry>::iterator
            hint++;
            data_coo.erase(p.first);
            // if the entry is zero and it was not a duplicate, just erase it.
            if(!almost_zero(val[i]))
                data_coo.insert(hint, temp_new);
        }

        // if the entry is zero, erase it.
        if(p.second && almost_zero(val[i]))
            data_coo.erase(p.first);
    }

    return 0;
}


int saena_matrix::set2(index_t row, index_t col, value_t val){

    // if there are duplicates with different values on two different processors, what should happen?
    // which one should be removed? We do it randomly.

    cooEntry_row temp_old;
    cooEntry_row temp_new = cooEntry_row(row, col, val);

    std::pair<std::set<cooEntry_row>::iterator, bool> p = data_coo.insert(temp_new);

    if (!p.second){
        temp_old = *(p.first);
        temp_new.val += temp_old.val;

//        std::set<cooEntry_row>::iterator hint = p.first;
        auto hint = p.first;
        hint++;
        data_coo.erase(p.first);
        data_coo.insert(hint, temp_new);
    }

    return 0;
}


int saena_matrix::set2(index_t* row, index_t* col, value_t* val, nnz_t nnz_local){

    if(nnz_local <= 0){
        printf("size in the set function is either zero or negative!");
        return 0;
    }

    cooEntry_row temp_old, temp_new;
    std::pair<std::set<cooEntry_row>::iterator, bool> p;

    for(nnz_t i=0; i<nnz_local; ++i){
        if(!almost_zero(val[i])){
            temp_new = cooEntry_row(row[i], col[i], val[i]);
            p = data_coo.insert(temp_new);

            if (!p.second){
                temp_old = *(p.first);
                temp_new.val += temp_old.val;

                std::set<cooEntry_row>::iterator hint = p.first;
                hint++;
                data_coo.erase(p.first);
                data_coo.insert(hint, temp_new);
            }
        }
    }

    return 0;
}


void saena_matrix::set_p_order(int _p_order){
    p_order = _p_order;
}

void saena_matrix::set_prodim(int _prodim){
    prodim = _prodim;
}

// int saena_matrix::set3(unsigned int row, unsigned int col, double val)
/*
int saena_matrix::set3(unsigned int row, unsigned int col, double val){

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // update the matrix size if required.
//    if(row >= Mbig)
//        Mbig = row + 1; // "+ 1" is there since row starts from 0, not 1.
//    if(col >= Mbig)
//        Mbig = col + 1;

//    auto proc_num = lower_bound2(&*split.begin(), &*split.end(), (unsigned long)row);
//    printf("proc_num = %ld\n", proc_num);

    cooEntry recv_buf;
    cooEntry send_buf(row, col, val);

//    if(rank == proc_num)
//        MPI_Recv(&send_buf, 1, cooEntry::mpi_datatype(), , 0, comm, NULL);
//    if(rank != )
//        MPI_Send(&recv_buf, 1, cooEntry::mpi_datatype(), proc_num, 0, comm);

    //todo: change send_buf to recv_buf after completing the communication for the parallel version.
    auto position = lower_bound2(&*entry.begin(), &*entry.end(), send_buf);
//    printf("position = %lu \n", position);
//    printf("%lu \t%lu \t%f \n", entry[position].row, entry[position].col, entry[position].val);

    if(send_buf == entry[position]){
        if(add_duplicates){
            entry[position].val += send_buf.val;
        }else{
            entry[position].val = send_buf.val;
        }
    }else{
        printf("\nAttention: the structure of the matrix is being changed, so matrix.assemble() is required to call after being done calling matrix.set()!\n\n");
        entry.emplace_back(send_buf);
        std::sort(&*entry.begin(), &*entry.end());
        nnz_g++;
        nnz_l++;
    }

//    printf("\nentry:\n");
//    for(long i = 0; i < nnz_l; i++)
//        std::cout << entry[i] << std::endl;

    return 0;
}

int saena_matrix::set3(unsigned int* row, unsigned int* col, double* val, unsigned int nnz_local){

    if(nnz_local <= 0){
        printf("size in the set function is either zero or negative!");
        return 0;
    }

    cooEntry temp;
    long position;
    for(unsigned int i = 0; i < nnz_local; i++){
        temp = cooEntry(row[i], col[i], val[i]);
        position = lower_bound2(&*entry.begin(), &*entry.end(), temp);
        if(temp == entry[position]){
            if(add_duplicates){
                entry[position].val += temp.val;
            }else{
                entry[position].val  = temp.val;
            }
        }else{
            printf("\nAttention: the structure of the matrix is being changed, so matrix.assemble() is required to call after being done calling matrix.set()!\n\n");
            entry.emplace_back(temp);
            std::sort(&*entry.begin(), &*entry.end());
            nnz_g++;
            nnz_l++;
        }
    }

//    printf("\nentry:\n");
//    for(long i = 0; i < nnz_l; i++)
//        std::cout << entry[i] << std::endl;

    return 0;
}
*/


int saena_matrix::destroy(){
    return 0;
}


int saena_matrix::erase(){

    data_coo.clear();
//    data.clear();
//    data.shrink_to_fit();

    entry.clear();
    split.clear();
    split_old.clear();
    values_local.clear();
    row_local.clear();
    values_remote.clear();
    row_remote.clear();
    col_local.clear();
    col_remote.clear();
    col_remote2.clear();
    nnzPerRow_local.clear();
    nnzPerCol_remote.clear();
    inv_diag.clear();
    vdispls.clear();
    rdispls.clear();
    recvProcRank.clear();
    recvProcCount.clear();
    sendProcRank.clear();
    sendProcCount.clear();
    sendProcCount.clear();
    requests.clear();
    statuses.clear();

    entry.shrink_to_fit();
    split.shrink_to_fit();
    split_old.shrink_to_fit();
    values_local.shrink_to_fit();
    values_remote.shrink_to_fit();
    row_local.shrink_to_fit();
    row_remote.shrink_to_fit();
    col_local.shrink_to_fit();
    col_remote.shrink_to_fit();
    col_remote2.shrink_to_fit();
    nnzPerRow_local.shrink_to_fit();
    nnzPerCol_remote.shrink_to_fit();
    inv_diag.shrink_to_fit();
    vdispls.shrink_to_fit();
    rdispls.shrink_to_fit();
    recvProcRank.shrink_to_fit();
    recvProcCount.shrink_to_fit();
    sendProcRank.shrink_to_fit();
    sendProcCount.shrink_to_fit();
    sendProcCount.shrink_to_fit();
    requests.shrink_to_fit();
    statuses.shrink_to_fit();

#ifdef SAENA_USE_ZFP
    if(free_zfp_buff){
        deallocate_zfp();
    }
#endif

    M = 0;
    Mbig = 0;
    nnz_g = 0;
    nnz_l = 0;
    nnz_l_local = 0;
    nnz_l_remote = 0;
    col_remote_size = 0;
    recvSize = 0;
    numRecvProc = 0;
    numSendProc = 0;
    assembled = false;

    return 0;
}


int saena_matrix::erase2(){

    data_coo.clear();
//    data.clear();
//    data.shrink_to_fit();

    entry.clear();
    split.clear();
    split_old.clear();
    values_local.clear();
    row_local.clear();
    values_remote.clear();
    row_remote.clear();
    col_local.clear();
    col_remote.clear();
    col_remote2.clear();
    nnzPerRow_local.clear();
    nnzPerCol_remote.clear();
    inv_diag.clear();
    vdispls.clear();
    rdispls.clear();
    recvProcRank.clear();
    recvProcCount.clear();
    sendProcRank.clear();
    sendProcCount.clear();
    vIndex.clear();
    vSend.clear();
    vecValues.clear();
//    vSend2.clear();
//    vecValues2.clear();
    recvCount.clear();
    sendCount.clear();
    iter_local_array.clear();
    iter_remote_array.clear();
//    iter_local_array2.clear();
    vElement_remote.clear();
    w_buff.clear();

    entry.shrink_to_fit();
    split.shrink_to_fit();
    split_old.shrink_to_fit();
    values_local.shrink_to_fit();
    values_remote.shrink_to_fit();
    row_local.shrink_to_fit();
    row_remote.shrink_to_fit();
    col_local.shrink_to_fit();
    col_remote.shrink_to_fit();
    col_remote2.shrink_to_fit();
    nnzPerRow_local.shrink_to_fit();
    nnzPerCol_remote.shrink_to_fit();
    inv_diag.shrink_to_fit();
    vdispls.shrink_to_fit();
    rdispls.shrink_to_fit();
    recvProcRank.shrink_to_fit();
    recvProcCount.shrink_to_fit();
    sendProcRank.shrink_to_fit();
    sendProcCount.shrink_to_fit();
    vIndex.shrink_to_fit();
    vSend.shrink_to_fit();
    vecValues.shrink_to_fit();
//    vSend2.shrink_to_fit();
//    vecValues2.shrink_to_fit();
    recvCount.shrink_to_fit();
    sendCount.shrink_to_fit();
    iter_local_array.shrink_to_fit();
    iter_remote_array.shrink_to_fit();
//    iter_local_array2.shrink_to_fit();
    vElement_remote.shrink_to_fit();
    w_buff.shrink_to_fit();

#ifdef SAENA_USE_ZFP
    if(free_zfp_buff){
        deallocate_zfp();
    }
#endif

//    M = 0;
//    Mbig = 0;
//    nnz_g = 0;
//    nnz_l = 0;
//    nnz_l_local = 0;
//    nnz_l_remote = 0;
//    col_remote_size = 0;
//    recvSize = 0;
//    numRecvProc = 0;
//    numSendProc = 0;
//    vIndexSize = 0;
//    shrinked = false;
//    active = true;
    assembled = false;

    return 0;
}


int saena_matrix::erase_update_local(){

//    row_local_temp.clear();
//    col_local_temp.clear();
//    values_local_temp.clear();
//    row_local.swap(row_local_temp);
//    col_local.swap(col_local_temp);
//    values_local.swap(values_local_temp);

//    entry.clear();
    // push back the remote part
//    for(unsigned long i = 0; i < row_remote.size(); i++)
//        entry.emplace_back(cooEntry(row_remote[i], col_remote2[i], values_remote[i]));

//    split.clear();
//    split_old.clear();
    values_local.clear();
    row_local.clear();
    col_local.clear();
    row_remote.clear();
    col_remote.clear();
    col_remote2.clear();
    values_remote.clear();
    nnzPerRow_local.clear();
    nnzPerCol_remote.clear();
    inv_diag.clear();
    vdispls.clear();
    rdispls.clear();
    recvProcRank.clear();
    recvProcCount.clear();
    sendProcRank.clear();
    sendProcCount.clear();
    sendProcCount.clear();

//    M = 0;
//    Mbig = 0;
//    nnz_g = 0;
//    nnz_l = 0;
//    nnz_l_local = 0;
    nnz_l_remote = 0;
    col_remote_size = 0;
    recvSize = 0;
    numRecvProc = 0;
    numSendProc = 0;
    assembled = false;

    return 0;
}


int saena_matrix::erase_keep_remote2(){

    entry.clear();

    // push back the remote part
    for(unsigned long i = 0; i < row_remote.size(); i++)
        entry.emplace_back(cooEntry(row_remote[i], col_remote2[i], values_remote[i]));

    split.clear();
    split_old.clear();
    values_local.clear();
    row_local.clear();
    values_remote.clear();
    row_remote.clear();
    col_local.clear();
    col_remote.clear();
    col_remote2.clear();
    nnzPerRow_local.clear();
    nnzPerCol_remote.clear();
    inv_diag.clear();
    vdispls.clear();
    rdispls.clear();
    recvProcRank.clear();
    recvProcCount.clear();
    sendProcRank.clear();
    sendProcCount.clear();
    vIndex.clear();
    vSend.clear();
    vecValues.clear();
//    vSend2.clear();
//    vecValues2.clear();
    recvCount.clear();
    sendCount.clear();
    iter_local_array.clear();
    iter_remote_array.clear();
//    iter_local_array2.clear();
    vElement_remote.clear();
    w_buff.clear();

    // erase_keep_remote() is used in coarsen2(), so keep the memory reserved for performance.
    // so don't use shrink_to_fit() on these vectors.

    M = 0;
    Mbig = 0;
    nnz_g = 0;
    nnz_l = 0;
    nnz_l_local = 0;
    nnz_l_remote = 0;
    col_remote_size = 0;
    recvSize = 0;
    numRecvProc = 0;
    numSendProc = 0;
    vIndexSize = 0;
//    assembled = false;
//    shrinked = false;
//    active = true;

    return 0;
}


int saena_matrix::erase_after_shrink() {

    nnz_l_local = 0;
    nnz_l_remote = 0;
    numRecvProc = 0;
    numSendProc = 0;
    vIndexSize = 0;
    recvSize = 0;

    row_local.clear();
    col_local.clear();
    values_local.clear();
    row_remote.clear();
//    col_remote.clear();
//    col_remote2.clear();
    values_remote.clear();

    vElement_remote.clear();
    nnzPerRow_local.clear();
    nnzPerCol_remote.clear();
    nnzPerProcScan.clear();
    recvCount.clear();
    sendCount.clear();
    recvProcRank.clear();
    recvProcCount.clear();
    sendProcRank.clear();
    sendProcCount.clear();
    vdispls.clear();
    rdispls.clear();
    vIndex.clear();
    vSend.clear();
    vecValues.clear();
    return 0;
}


int saena_matrix::erase_lazy_update(){
    data_coo.clear();
    return 0;
}


int saena_matrix::erase_no_shrink_to_fit(){

    data_coo.clear();
    data.clear(); //todo: is this required?
    data_unsorted.clear(); //todo: is this required?

    entry.clear();
    split.clear();
    split_old.clear();
    values_local.clear();
    row_local.clear();
    values_remote.clear();
    row_remote.clear();
    col_local.clear();
    col_remote.clear();
    col_remote2.clear();
    nnzPerRow_local.clear();
    nnzPerCol_remote.clear();
    inv_diag.clear();
    vdispls.clear();
    rdispls.clear();
    recvProcRank.clear();
    recvProcCount.clear();
    sendProcRank.clear();
    sendProcCount.clear();
    sendProcCount.clear();

//    data.shrink_to_fit();
//    data_unsorted.shrink_to_fit();
//    entry.shrink_to_fit();
//    split.shrink_to_fit();
//    split_old.shrink_to_fit();
//    values_local.shrink_to_fit();
//    values_remote.shrink_to_fit();
//    row_local.shrink_to_fit();
//    row_remote.shrink_to_fit();
//    col_local.shrink_to_fit();
//    col_remote.shrink_to_fit();
//    col_remote2.shrink_to_fit();
//    nnzPerRow_local.shrink_to_fit();
//    nnzPerCol_remote.shrink_to_fit();
//    inv_diag.shrink_to_fit();
//    vdispls.shrink_to_fit();
//    rdispls.shrink_to_fit();
//    recvProcRank.shrink_to_fit();
//    recvProcCount.shrink_to_fit();
//    sendProcRank.shrink_to_fit();
//    sendProcCount.shrink_to_fit();
//    sendProcCount.shrink_to_fit();
//    vElementRep_local.shrink_to_fit();
//    vElementRep_remote.shrink_to_fit();

#ifdef SAENA_USE_ZFP
    if(free_zfp_buff){
        deallocate_zfp();
    }
#endif

    M = 0;
    Mbig = 0;
    nnz_g = 0;
    nnz_l = 0;
    nnz_l_local = 0;
    nnz_l_remote = 0;
    col_remote_size = 0;
    recvSize = 0;
    numRecvProc = 0;
    numSendProc = 0;
    assembled = false;

    return 0;
}


int saena_matrix::set_zero(){

#pragma omp parallel for
    for(nnz_t i = 0; i < nnz_l; i++)
        entry[i].val = 0;

    values_local.clear();
    values_remote.clear();

    return 0;
}


void saena_matrix::jacobi(int iter, std::vector<value_t>& u, std::vector<value_t>& rhs) {

// Ax = rhs
// u = u - (D^(-1))(Au - rhs)
// 1. B.matvec(u, one) --> put the value of matvec in one.
// 2. two = one - rhs
// 3. three = inverseDiag * two * omega
// 4. four = u - three

//    int rank;
//    MPI_Comm_rank(comm, &rank);

    for(int j = 0; j < iter; j++){
        matvec(u, temp1);

        #pragma omp parallel for
        for(index_t i = 0; i < M; i++){
            temp1[i] -= rhs[i];
            temp1[i] *= inv_diag[i] * jacobi_omega;
            u[i]    -= temp1[i];
        }
    }
}


void saena_matrix::chebyshev(const int &iter, std::vector<value_t>& u, std::vector<value_t>& rhs){

#ifdef __DEBUG1__
//    int rank;
//    MPI_Comm_rank(comm, &rank);
//    for(auto &i : inv_diag){
//        assert(i == 1);     // the matrix is scaled to have diagonal 1.
//    }
#endif

    const double alpha = 0.13 * eig_max_of_invdiagXA; // homg: 0.25 * eig_max
    const double beta  = eig_max_of_invdiagXA;
    const double delta = (beta - alpha) / 2;
    const double theta = (beta + alpha) / 2;
    const double s1    = theta / delta;
    const double twos1 = 2 * s1;     // to avoid the multiplication in the "for" loop.
          double rhok  = 1 / s1;
          double rhokp1 = 0.0, two_rhokp1 = 0.0, d1 = 0.0, d2 = 0.0;

    std::vector<value_t>& res = temp1;
    std::vector<value_t>& d   = temp2;

    // first loop
    residual_negative(u, rhs, res);

    #pragma omp parallel for
    for(index_t i = 0; i < u.size(); ++i){
        d[i] = (res[i] * inv_diag[i]) / theta;
        u[i] += d[i];
//        if(rank==0) printf("inv_diag[%u] = %f, \tres[%u] = %f, \td[%u] = %f, \tu[%u] = %f \n",
//                           i, inv_diag[i], i, res[i], i, d[i], i, u[i]);
    }

    for(int i = 1; i < iter; ++i){
        rhokp1 = 1 / (twos1 - rhok);
        two_rhokp1 = 2 * rhokp1;
        d1     = rhokp1 * rhok;
        d2     = two_rhokp1 / delta;
        rhok   = rhokp1;

        residual_negative(u, rhs, res);

        #pragma omp parallel for
        for(index_t j = 0; j < u.size(); ++j){
            d[j] = ( d1 * d[j] ) + ( d2 * res[j] * inv_diag[j]);
            u[j] += d[j];
//            if(rank==0) printf("inv_diag[%u] = %f, \tres[%u] = %f, \td[%u] = %f, \tu[%u] = %f \n",
//                               j, inv_diag[j], j, res[j], j, d[j], j, u[j]);
        }
    }
}


int saena_matrix::print_entry(int ran, const std::string &name) const{

    // if ran >= 0 print_entry the matrix entries on proc with rank = ran
    // otherwise print the matrix entries on all processors in order. (first on proc 0, then proc 1 and so on.)

    if(active) {
        int rank = 0, nprocs = 0;
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &rank);

        index_t iter = 0;
        if (ran >= 0) {
            if (rank == ran) {
                std::cout << "\nmatrix " << name << " on proc " << ran << std::endl;
//                printf("\nmatrix on proc = %d \n", ran);
                printf("nnz = %lu \n", nnz_l);
                for (auto i:entry) {
                    std::cout << iter << "\t" << i << std::endl;
                    iter++;
                }
            }
        } else {
            for (index_t proc = 0; proc < nprocs; proc++) {
                MPI_Barrier(comm);
                if (rank == proc) {
                    std::cout << "\nmatrix " << name << " on proc " << proc << std::endl;
//                    printf("\nmatrix on proc = %d \n", proc);
                    printf("nnz = %lu \n", nnz_l);
                    for (auto i:entry) {
                        std::cout << iter << "\t" << i << std::endl;
                        iter++;
                    }
                }
                MPI_Barrier(comm);
            }
        }
    }
    return 0;
}


int saena_matrix::print_info(int ran, const std::string &name) const{

    // if ran >= 0 print the matrix info on proc with rank = ran
    // otherwise print the matrix info on all processors in order. (first on proc 0, then proc 1 and so on.)

    int rank = 0, nprocs = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(ran >= 0) {
        if (rank == ran) {
            std::cout << "\nmatrix " << name << " info on proc " << ran << std::endl;
            printf("Mbig = %u, \tM = %u, \tnnz_g = %lu, \tnnz_l = %lu \n", Mbig, M, nnz_g, nnz_l);
        }
    } else{
        MPI_Barrier(comm);
        if(rank==0) printf("\nmatrix %s info:      Mbig = %u, \tnnz_g = %lu \n", name.c_str(), Mbig, nnz_g);
        for(index_t proc = 0; proc < nprocs; proc++){
            MPI_Barrier(comm);
            if (rank == proc) {
                printf("matrix %s on rank %d: M    = %u, \tnnz_l = %lu \n", name.c_str(), proc, M, nnz_l);
            }
            MPI_Barrier(comm);
        }
    }

    return 0;
}


int saena_matrix::writeMatrixToFile(const std::string &name) const{
    // name: pass the name of the file. The file will be saved in the working directory. To save the file in another
    //       directory, pass that path.
    // Create txt files with name mat-r0.txt for processor 0, mat-r1.txt for processor 1, etc.
    // Then, concatenate them in terminal: cat mat-r0.mtx mat-r1.mtx > mat.mtx
    // row and column indices of txt files should start from 1, not 0.
    // write the files inside ${HOME}/folder_name
    // this is the default case for the sorting which is column-major.

    int nprocs = 0, rank = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::string outFileNameTxt = name + "-r" + std::to_string(rank) + ".mtx";
    std::ofstream outFileTxt(outFileNameTxt);

    if(rank==0) std::cout << "\nWriting the matrix in: " << outFileNameTxt << std::endl;

//    std::vector<cooEntry> entry_temp1 = entry;
//    std::vector<cooEntry> entry_temp2;
//    par::sampleSort(entry_temp1, entry_temp2, comm);

    // sort row-wise
//    std::vector<cooEntry_row> entry_temp1(entry.size());
//    std::memcpy(&*entry_temp1.begin(), &*entry.begin(), entry.size() * sizeof(cooEntry));
//    std::vector<cooEntry_row> entry_temp2;
//    par::sampleSort(entry_temp1, entry_temp2, comm);

    // first line of the file: row_size col_size nnz
    if(rank==0) {
        outFileTxt << "%%MatrixMarket matrix coordinate real general" << std::endl;
        outFileTxt << Mbig << "\t" << Mbig << "\t" << nnz_g << std::endl;
    }

    for (nnz_t i = 0; i < entry.size(); i++) {
//        if(rank==0) std::cout  << A->entry[i].row + 1 << "\t" << A->entry[i].col + 1 << "\t" << A->entry[i].val << std::endl;
        outFileTxt << entry[i].row + 1 << "\t" << entry[i].col + 1 << "\t"
                   << std::setprecision(12) << entry[i].val << std::endl;
    }

    outFileTxt.clear();
    outFileTxt.close();

    return 0;
}