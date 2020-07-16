#include "saena_object.h"
#include "saena_matrix.h"
#include "strength_matrix.h"
#include "prolong_matrix.h"
//#include "restrict_matrix.h"
#include "grid.h"
#include "aux_functions.h"
#include "parUtils.h"

#include <numeric>

// Assume the mesh info is the connectivity
// using a 2d vector for now
int saena_object::pcoarsen(Grid *grid, vector< vector< vector<int> > > &map_all,std::vector< std::vector<int> > &g2u_all, std::vector<int> &order_dif){

    saena_matrix   *A  = grid->A;
    prolong_matrix *P  = &grid->P;
    saena_matrix   *Ac = &grid->Ac;

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
//    int rank_v = 0;
    if(verbose_pcoarsen) {
        MPI_Barrier(comm); if (!rank) printf("\ncoarsen: start: rank = %d\n", rank); MPI_Barrier(comm);
    }

//    if (!rank) {
//        cout << "\nmap_all.size = " << map_all.size() << endl;
//        cout << "map_all[0].size = " << map_all[0].size() << endl;
//        cout << "map_all[0][0].size = " << map_all[0][0].size() << endl;
//        cout << "g2u_all.size = " << g2u_all.size() << endl;
//        cout << "g2u_all[0].size = " << g2u_all[0].size() << endl;
//        cout << "order_dif.size = " << order_dif.size() << endl;
//        cout << "bdydof = " << bdydof << endl << endl;
//    }

//    print_vector(map_all[0][0], -1, "map_all[0][0]", comm);
//    print_vector(g2u_all[0], -1, "g2u_all[0]", comm);
//    print_vector(order_dif, -1, "order_dif", comm);
#endif

    int order = A->p_order;
    prodim    = A->prodim;

    // assume divided by 2
//    Ac->set_p_order(A->p_order / 2);

    next_order = A->p_order - order_dif[grid->currentLevel];
    if (next_order < 1)
        next_order = 1;

    Ac->set_p_order(next_order);

    vector< vector<int> > map = mesh_info(order, map_all, comm);

#ifdef __DEBUG1__
    if(verbose_pcoarsen) {
        MPI_Barrier(comm);
        if(!rank) printf("coarsen: step 1\n");
        MPI_Barrier(comm);
    }
#endif

    g2umap(order, g2u_all, map_all, comm);

#ifdef __DEBUG1__
    if(verbose_pcoarsen) {
        MPI_Barrier(comm);
        if (!rank) printf("coarsen: step 2\n");
        MPI_Barrier(comm);
    }
    /*int row_m = map.size();
    int col_m = map.at(0).size();
    if (rank == rank_v) {
        cout << "map row: " << row_m << ", and col: " << col_m << "\n";
        for (int i=0; i<row_m; i++)
        {
            for (int j=0; j<col_m; j++)
                cout << map[i][j] << "\t";

            cout << "\n";
        }
        cout << "g2u size: " << g2u.size() << "\n";
        for (int i=0; i<g2u.size(); i++)
            cout << g2u[i] << endl;
    }
    MPI_Barrier(comm);
    exit(0);*/
#endif

    vector<cooEntry_row> P_temp;
    set_P_from_mesh(order, map, P_temp, comm, g2u_all, map_all);//, Rp);

    bdydof = next_bdydof;

#ifdef __DEBUG1__
//    print_vector(P_temp, -1, "P_temp", comm);
    if(verbose_pcoarsen) {
        MPI_Barrier(comm); if (!rank) printf("coarsen: step 3\n"); MPI_Barrier(comm);
//        int row_p = Pp.size();
//        int col_p = Pp.at(0).size();

//        std::stringstream buffer;
//        buffer << "rank " << rank << ": Pp row: " << row_p << ", and col: " << col_p;
//        cout << buffer.str() << endl;

//        for(int i = 0; i > row_p; ++i){
//            cout << Pp[0].size() << endl;
//        }
    }
#endif

    P->comm  = A->comm;
    P->split = A->split;
    P->Mbig  = A->Mbig;
    P->M     = A->M;

#ifdef __DEBUG1__
    if(verbose_pcoarsen) {
        MPI_Barrier(comm); if (!rank) printf("coarsen: step 4\n"); MPI_Barrier(comm);
    }
//    std::sort(P_temp.begin(), P_temp.end(), row_major);
//    print_vector(P_temp, -1, "P_temp", comm);
#endif

    vector<cooEntry_row> Pent;
    par::sampleSort(P_temp, Pent, P->split, comm);

#ifdef __DEBUG1__
//    print_vector(Pent, -1, "Pent", comm);
#endif

    // remove duplicates
    Pent.erase( unique( Pent.begin(), Pent.end() ), Pent.end() );

    P->entry.clear();
    P->entry.resize(Pent.size());
    int row_ofst = P->split[rank];

#pragma omp parallel for default(none) shared(P, Pent, row_ofst)
    for(int i = 0; i < Pent.size(); ++i){
        P->entry[i] = cooEntry( (Pent[i].row - row_ofst), Pent[i].col, Pent[i].val);
    }

    std::sort(P->entry.begin(), P->entry.end());

#ifdef __DEBUG1__
//    std::sort(A->entry.begin(), A->entry.end(), row_major);
//    print_vector(A->entry, -1, "A", comm);
//    print_vector(P->entry, -1, "P", comm);
//    print_vector(P->split, -1, "P->split", comm);
    if(verbose_pcoarsen) {
        MPI_Barrier(comm); if (!rank) printf("coarsen: step 5\n"); MPI_Barrier(comm);
    }
#endif

    // set nnz_l and nnz_g
    P->nnz_l = P->entry.size();
    MPI_Allreduce(&P->nnz_l, &P->nnz_g, 1, par::Mpi_datatype<nnz_t>::value(), MPI_SUM, P->comm);

    // set Nbig
    index_t Nbig_loc = 0;
    if(P->nnz_l != 0){
        Nbig_loc = P->entry.back().col + 1;
    }
    MPI_Allreduce(&Nbig_loc, &P->Nbig, 1, par::Mpi_datatype<index_t>::value(), MPI_MAX, comm);

    // set splitNew
    P->splitNew.resize(nprocs + 1);
    P->splitNew[0]      = 0;
    P->splitNew[nprocs] = P->Nbig;

    int ofst = P->Nbig / nprocs;
    for(int i = 1; i < nprocs; ++i){
        P->splitNew[i] = i * ofst;
    }

#ifdef __DEBUG1__
    if(verbose_pcoarsen) {
        MPI_Barrier(comm);
        std::stringstream buffer;
        buffer << "rank " << rank << ": P: " << "M = " << P->M << ", Mbig = " << P->Mbig << ", Nbig = " << P->Nbig
               << ", nnz_l = " << P->nnz_l << ", nnz_g = " << P->nnz_g;
        cout << buffer.str() << endl;
//        print_vector(P->splitNew, 0, "P->splitNew", comm);
        MPI_Barrier(comm);
        if (!rank) printf("coarsen: step 6\n");
        MPI_Barrier(comm);
    }
#endif

    P->findLocalRemote();

#ifdef __DEBUG1__
    if(verbose_pcoarsen) {
        MPI_Barrier(comm); if (!rank) printf("coarsen: end\n\n"); MPI_Barrier(comm);
    }
#endif

    return 0;
}


vector<int> saena_object::next_p_level_random(vector<int> ind_fine, int order, int *type_){
    // assuming the elemental ordering is like
    // 7--8--9
    // |  |  |
    // 4--5--6
    // |  |  |
    // 1--2--3

    // check element type
    // 0: tri
    // 1: quad
    // 2: tet
    // 3: hex
    // 4: prism

//    int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int type = -1;
    int vert_size = ind_fine.size();

    if (vert_size == (order+1)*(order+1))
        type = 1;
    else if (vert_size == (order+1)*(order+1)*(order+1))
        type = 3;
    else if (vert_size == (order+1)*(order+2)/2)
        type = 0;
    else if (vert_size == (order*order*order+11*order)/6+order*order+1)
        type = 2;
    else if ( vert_size == 6+9*(order-1)+3*(order-1)*(order-1)+(order-1)*(order-2)+(order-1)*(order-1)*(order-2)/2 )
        type = 4;
    else
    {
//        if (rank == rank_v)
        std::cout << "element type is not implemented!" << std::endl;
    }

    //cout << "type: " << type << endl;
    //cout << 6+9*(order-1)+3*(order-1)*(order-1)+(order-1)*(order-2)+(order-1)*(order-1)*(order-2)/2 << " " << vert_size << endl;
    //cout << "element type = " << type << endl;
    vector<int> indices;
    if (type == 1)
    {
        for (int i=0; i<=next_order; i++)
        {
            for (int j=0; j<=next_order; j++)
            {
                indices.push_back(ind_fine[(order+1)*i+j]);
            }
        }
    }

    if (type == 3)
    {
        for (int i=0; i<=next_order; i++)
        {
            for (int j=0; j<=next_order; j++)
            {
                for (int k=0; k<=next_order; k++)
                {
                    indices.push_back(ind_fine[(order+1)*(order+1)*i+(order+1)*j+k]);
                }
            }
        }
    }

    //cout << "===============" << endl;
    if (type == 0)
    {
        for (int i=0; i<=next_order; i++)
        {
            for (int j=0; j<=next_order-i; j++)
            {
                indices.push_back(ind_fine[(2*order+3-i)*i/2+j]);
                //cout << (2*order+3-i)*i/2+j << endl;
            }
        }
        /*int counter = 0;
        for (int i=0; i<=order; i++)
        {
            for (int j=0; j<=order-i; j++)
            {
                if (i<=order/2 && j<=order/2-i)
                    cout << counter << endl;

                counter ++;
            }
        }*/
    }

    //cout << "==============" << endl;
    if (type == 2)
    {
        int counter = 0;
        for (int i=0; i<=order; i++)
        {
            for (int j=0; j<=order-i; j++)
            {
                for (int k=0; k <= order-i-j; k++)
                {
                    //cout << sum_i+sum_j+k << endl;
                    if (i <= next_order && j <= next_order-i && k <= next_order-i-j)
                        indices.push_back(ind_fine[counter]);

                    ++counter;
                }
            }
        }
    }

    if (type == 4)
    {
        int counter = 0;
        for (int i=0; i<=order; i++)
        {
            for (int j=0; j<=order; j++)
            {
                for (int k=0; k<=order-i; k++)
                {
                    //cout << sum_i+sum_j+k << endl;
                    if (i<=next_order && j<=next_order && k<= next_order-i)
                        indices.push_back(ind_fine[counter]);

                    ++counter;
                }
            }
        }
    }

    if (type_ != NULL)
        *type_ = type;

    return indices;
}


vector<int> saena_object::coarse_p_node_arr(vector< vector<int> > map, int order)
{
    int total_elem = map.size();

    vector<int> ind;
    for (int i = 0; i < total_elem; ++i) {
        //cout << i << endl;
        //vector<int> ind_coarse = next_p_level_new2(map.at(i), order);
        vector<int> ind_coarse = next_p_level_random(map.at(i), order);
        for (int j=0; j<ind_coarse.size(); j++) {
            if (!ismember(ind_coarse.at(j), ind))
                ind.emplace_back(ind_coarse.at(j));
        }

    }
    return ind;
}


void saena_object::set_P_from_mesh(int order, vector<vector<int>> map, vector<cooEntry_row> &P_temp, MPI_Comm comm, vector< vector<int> > &g2u_all, vector< vector< vector<int> > > &map_all){
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    vector<int> g2u_f = g2u_all.at(g2u_all.size()-2);
    vector<int> g2u_c = g2u_all.at(g2u_all.size()-1);
    // get universal number of dof
    // any other better way?
    int g2u_f_univ_size = 0;
    int g2u_f_size = g2u_f.size();

    MPI_Allreduce(&g2u_f_size, &g2u_f_univ_size, 1, MPI_INT, MPI_SUM, comm);

    vector<int> g2u_univ_map(g2u_f_univ_size);
    vector<int> count_arr(nprocs);
    MPI_Allgather(&g2u_f_size, 1, MPI_INT, count_arr.data(), 1, MPI_INT, comm);

    vector<int> displs(nprocs);
    displs[0] = 0;
    for (int i=1; i<nprocs;i++)
        displs[i] = displs[i-1]+count_arr[i-1];

    MPI_Allgatherv(g2u_f.data(), g2u_f_size, MPI_INT, g2u_univ_map.data(), count_arr.data(), displs.data(), MPI_INT, comm);

    // compute the universal fine matrix size
    // ======================================
    vector<int> g2u_univ_map_tmp = g2u_univ_map;

    // sort and remove the duplicates
    sort( g2u_univ_map_tmp.begin(), g2u_univ_map_tmp.end() );
    g2u_univ_map_tmp.erase( unique( g2u_univ_map_tmp.begin(), g2u_univ_map_tmp.end() ), g2u_univ_map_tmp.end() );

    int univ_nodeno_fine = g2u_univ_map_tmp.size();
    g2u_univ_map_tmp.clear();

    // ======================================

    // find bdydof next level
    vector<int> coarse_node_ind = coarse_p_node_arr(map, order); // get coarse level matrix (local)

    // remove boundary nodes
    // boundary nodes are sorted as the first entries of the matrix
    for (int i=0; i<coarse_node_ind.size(); i++){
        if (coarse_node_ind[i] < bdydof){
            coarse_node_ind.erase(coarse_node_ind.begin()+i);
            --nodeno_coarse;
            --i;
        }
    }

    // col is global
    // row is universal
    // nodeno_coarse is the local coarse level size without boundary nodes
    vector< vector<double> > Pp_loc(univ_nodeno_fine);
    for (int i = 0; i < univ_nodeno_fine; i++)
        Pp_loc.at(i).resize(nodeno_coarse, 0);

#ifdef __DEBUG1__
    if(verbose_pcoarsen) {
        MPI_Barrier(comm);
        if (rank == rank_v)
            std::cout << "Pp_loc has row (universal) = " << Pp_loc.size() << ", and col(global) = " << Pp_loc[0].size() << endl;
        MPI_Barrier(comm);
    }
#endif

    // next level g2u
    // index next level node index
    // value this level g2u value
    //vector<int> g2u_map_c(nodeno_coarse);
    // coarse_node_ind index is coraser mesh node index
    // coarse_node_ind value is finer mesh node index

    sort(coarse_node_ind.begin(), coarse_node_ind.end());
    //cout << map.size() << " " << map.at(0).size() << "\n";
    //skip bdy node
    //vector<int> skip;
    // loop over all elements
    // elemno is the local element number
    int max_col = 0;
    int max_row = 0;
    for (int i=0; i<elemno; i++){
        // for each element extract coarser element indices
        int elem_type;
        vector<int> ind_coarse = next_p_level_random(map.at(i), order, &elem_type);

        //cout << i << " " << flag << endl;
        //flag = true;
        for (int j=0; j<ind_coarse.size(); j++){
            // skip bdy node
            if (ind_coarse.at(j)-1 < bdydof)
                continue;
            //if (flag)
            //cout << "======================" << endl;
            // interpolate basis function of j at all other nodes
            // in this element as the values of corresponding P entries
            // TODO only for 2d now upto basis order 8
            //vector<double> val = get_interpolation(j + 1, order, prodim);

            // modal basis no need for interpolation
            //vector<double> val = get_interpolation_new2(j, order, elem_type, flag);
            //vector<double> val = get_interpolation(j, order);

            /*if (flag)
            {
                for (int i=0; i<val.size(); i++)
                    cout << val[i] << " ";

                cout << endl;
            }*/

            // find coarse_node_ind index (coarser mesh node index) that
            // has the same value (finer mesh node index) as ind_coarse value
            // TODO This is slow, may need smarter way
            //int P_col = findloc(coarse_node_ind, ind_coarse.at(j));
            int P_col = map_all.at(map_all.size()-1).at(i).at(j)-1-next_bdydof;

            //cout << ind_coarse.at(j)-1-bdydof << " " << P_col << endl;
            P_temp.emplace_back(g2u_f.at(ind_coarse.at(j)-1-bdydof), g2u_c.at(P_col), 1.0);
        }
    }
    //if (rank == rank_v)
    //cout << "max row and col = " << max_row << " " << max_col << endl;
    std::sort(P_temp.begin(), P_temp.end());
    P_temp.erase( unique( P_temp.begin(), P_temp.end() ), P_temp.end() );
}


//this is the function as mesh info for test for now
inline vector< std::vector<int> > saena_object::mesh_info(int order, vector< vector< vector<int> > > &map_all, MPI_Comm comm)
{
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    if(verbose_pcoarsen) {
        MPI_Barrier(comm); if (!rank) printf("\nmesh_info: start\n"); MPI_Barrier(comm);
    }
#endif

    assert(!map_all.empty());

    // TODO: avoid copying
    vector <vector<int> > map = map_all.at(map_all.size()-1);

    if (map_all.size() == 1) {
        elemno = map_all[0].size();
    }

    if (order > 1)
    {

#ifdef __DEBUG1__
        if(verbose_pcoarsen) {
            MPI_Barrier(comm); if (!rank) printf("mesh_info: step 1\n"); MPI_Barrier(comm);
        }
#endif

        // coarse_node_ind index is coarser mesh node index
        // coarse_node_ind value is finer mesh node index

        vector<int> coarse_node_ind = coarse_p_node_arr(map, order);
        sort(coarse_node_ind.begin(), coarse_node_ind.end());

#ifdef __DEBUG1__
        if(verbose_pcoarsen) {
            MPI_Barrier(comm); if (!rank) printf("mesh_info: step 2\n"); MPI_Barrier(comm);
        }
#endif

        vector <vector<int> > map_next(elemno);
        for (int i = 0; i < elemno; ++i){
            vector<int> aline = map.at(i);
            //cout << i << endl;
            //vector<int> ind_coarse = next_p_level_new2(aline, order);
            vector<int> ind_coarse = next_p_level_random(aline, order);
            for (int j = 0; j < ind_coarse.size(); ++j){
                int mapped_val = findloc(coarse_node_ind, ind_coarse.at(j));
                map_next.at(i).emplace_back(mapped_val+1);
            }
        }

#ifdef __DEBUG1__
        if(verbose_pcoarsen) {
            MPI_Barrier(comm); if (!rank) printf("mesh_info: step 3\n"); MPI_Barrier(comm);
        }
#endif

        map_all.emplace_back(vector< vector<int> >());
        for (int i = 0; i < elemno; ++i){
            map_all.at(map_all.size()-1).push_back(vector<int>());
            for (int j = 0; j < map_next.at(i).size(); ++j){
                map_all.at(map_all.size()-1).at(i).emplace_back(map_next.at(i).at(j));
            }
        }

        // get fine and corase number of nodes in this P level
        nodeno_coarse = coarse_p_node_arr(map, order).size();
        nodeno_fine = 0;
        for (int i = 0; i < map.size(); ++i){
            nodeno_fine = std::max(*max_element(map[i].begin(), map[i].end()), nodeno_fine);
        }

#ifdef __DEBUG1__
        if(verbose_pcoarsen) {
            MPI_Barrier(comm);
            if (rank == rank_v){
                cout << "order = " << order << endl;
                cout << "elem # = " << elemno << endl;
                cout << "bdydof # = " << bdydof << endl;
                cout << "current fine node # = " << nodeno_fine << ", and next coarse node # = " << nodeno_coarse << endl;
            }
            MPI_Barrier(comm);
        }
#endif

    }

#ifdef __DEBUG1__
    if(verbose_pcoarsen) {
        MPI_Barrier(comm); if (!rank) printf("mesh_info: end\n\n"); MPI_Barrier(comm);
    }
#endif

    return map;
}


//this is the function as mesh info for test for now
void saena_object::g2umap(int order, vector< vector<int> > &g2u_all, vector< vector< vector<int> > > &map_all, MPI_Comm comm)
{
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    if(verbose_pcoarsen) {
        MPI_Barrier(comm); if (!rank) printf("\ng2umap: start\n"); MPI_Barrier(comm);
    }
#endif

    assert(map_all.size() >= 2);

    // entry value is based on finer node
    vector <int> g2u_next_fine_node;

    // coarse_node_ind index is coraser mesh node index
    // coarse_node_ind value is finer mesh node index
    vector< vector<int> > map   = map_all.at(map_all.size()-2);
    vector< vector<int> > map_c = map_all.at(map_all.size()-1);

#ifdef __DEBUG1__
    if(verbose_pcoarsen) {
        MPI_Barrier(comm); if (!rank) printf("g2umap: step 1\n"); MPI_Barrier(comm);
    }
#endif

    vector<int> coarse_node_ind = coarse_p_node_arr(map, order);
    sort(coarse_node_ind.begin(), coarse_node_ind.end());

#ifdef __DEBUG1__
    if(verbose_pcoarsen) {
        MPI_Barrier(comm); if (!rank) printf("g2umap: step 2\n"); MPI_Barrier(comm);
    }
#endif

    next_bdydof = 0;
    for (int i = 0; i < coarse_node_ind.size(); ++i)
    {
        // get universal dof value
        // this value will go to next level
        if (coarse_node_ind.at(i)-1 < bdydof)
            ++next_bdydof;
        else
            g2u_next_fine_node.push_back(g2u_all.at(g2u_all.size()-1).at(coarse_node_ind.at(i)-1-bdydof));
    }

#ifdef __DEBUG1__
    if(verbose_pcoarsen) {
        MPI_Barrier(comm); if (!rank) printf("g2umap: step 3\n"); MPI_Barrier(comm);
    }
#endif

    // now fill mapping from global to universal in next level
    // need communication
    int g2u_univ_size;
    int glb_size = g2u_next_fine_node.size();
    MPI_Allreduce(&glb_size, &g2u_univ_size, 1, MPI_INT, MPI_SUM, comm);

    vector<int> g2u_univ(g2u_univ_size);
    vector<int> count_arr(nprocs);
    MPI_Allgather(&glb_size,1,MPI_INT,count_arr.data(),1, MPI_INT,comm);

    vector<int> displs(nprocs);
    displs[0] = 0;
    for (int i=1; i<nprocs;i++)
        displs[i] = displs[i-1]+count_arr[i-1];
    MPI_Allgatherv(g2u_next_fine_node.data(), g2u_next_fine_node.size(), MPI_INT, g2u_univ.data(), count_arr.data(), displs.data(), MPI_INT, comm);

    // sort the universal g2u map to make sure it is consistent with universal Ac = R*A*P dof ordering
    // since universal P column is also sorted in the same way
    // now universal g2u index becomes the map value for Ac
    sort(g2u_univ.begin(),g2u_univ.end());
    g2u_univ.erase( unique( g2u_univ.begin(), g2u_univ.end() ), g2u_univ.end() );
    // loop over global map to assign universal value to it

#ifdef __DEBUG1__
    if(verbose_pcoarsen) {
        MPI_Barrier(comm); if (!rank) printf("g2umap: step 4\n"); MPI_Barrier(comm);
    }
#endif

    vector <int> g2u_next_coarse_node(nodeno_coarse-next_bdydof);

    //cout << g2u_next_coarse_node.size() << endl;
    //cout << g2u_next_fine_node.size() << endl;
    //exit(0);
    for (int el = 0; el < elemno; el++)
    {
        for (int i=0; i< map_c[el].size(); i++)
        {
            int ind = map_c.at(el).at(i);
            //cout << ind << endl;
            int glb = coarse_node_ind.at(ind-1);
            //cout << glb << endl;
            if (glb -1 < bdydof)
                continue;

            int uni = g2u_all.at(g2u_all.size()-1).at(glb-1-bdydof);
            //cout << uni << endl;
            int loc = findloc(g2u_univ, uni);
            //cout << loc << endl;
            g2u_next_coarse_node.at(ind-1-next_bdydof) = loc;
            //if (!rank)
            //cout << ind-1-next_bdydof << " " << loc << endl;
        }

    }

#ifdef __DEBUG1__
    if(verbose_pcoarsen) {
        MPI_Barrier(comm); if (!rank) printf("g2umap: step 5\n"); MPI_Barrier(comm);
    }
#endif

    g2u_all.push_back( vector<int> ());
    for (int i = 0; i < g2u_next_coarse_node.size(); ++i) {
        g2u_all.at(g2u_all.size()-1).push_back(g2u_next_coarse_node.at(i));
    }
    //std::cout << map_all.size() << " " << map_all.at(map_all.size()-1).size() << " " << map_all.at(map_all.size()-1).at(0).size() << std::endl;

#ifdef __DEBUG1__
    if(verbose_pcoarsen) {
        MPI_Barrier(comm);
        if (rank == rank_v) printf("next bdy node # = %d\n", next_bdydof);
        MPI_Barrier(comm);
        if (!rank) printf("g2umap: end\n\n");
        MPI_Barrier(comm);
    }
#endif

}


inline bool saena_object::ismember(int a, vector<int> arr)
{
    for (int i=0; i<arr.size(); i++)
    {
        if (a == arr.at(i))
            return true;
    }
    return false;
}


inline int saena_object::findloc(vector<int> arr, int a)
{
    for (int i=0; i<arr.size(); i++)
    {
        if (a == arr.at(i))
            return i;
    }

    cout << "coarse column is not in the fine mesh!!!" << endl;
    return -1;
}
