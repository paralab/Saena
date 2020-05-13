#include "saena_object.h"
#include "saena_matrix.h"
#include "strength_matrix.h"
#include "prolong_matrix.h"
//#include "restrict_matrix.h"
#include "grid.h"
#include "aux_functions.h"

#include <vector>
#include <cmath>
#include <stdio.h>
#include <sstream>
#include <parUtils.h>

using namespace std;

// Assume the mesh info is the connectivity
// using a 2d vector for now
int saena_object::pcoarsen(Grid *grid, vector< vector< vector<int> > > &map_all,std::vector< std::vector<int> > &g2u_all){

    saena_matrix   *A  = grid->A;
    prolong_matrix *P  = &grid->P;
    saena_matrix   *Ac = &grid->Ac;
    Ac->set_p_order(A->p_order / 2);

    // A parameters:
    // A.entry[i]: to access entry i of A, which is local to this processor.
    //             each entry is in COO format: i, j, val
    // A.nnz_l: local number of nonzeros on this processor.
    // A.nnz_g: total number of nonzeros on all the processors.
    // A.split: it is a vector that stores the range of row indices on each processor.
    //          split[rank]: starting row index
    //          split[rank+1]: last row index (not including)
    //          example: if split[rank] is 5 and split[rank+1] is 9, it shows rows 4, 5, 6, 7, 8 are stored on this processor.
    // A.M: row size which equals to split[rank+1] - split[rank].
    // A.Mbig: the row size of the whole matrix.
    // A.print_info: print information of A, which are nnz_l, nnz_g, M, Mbig.
    // A.print_entry: print entries of A.
	// P.Nbig: the colume size of P
	// P->entry[i] = cooEntry(i, j, val);

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
    if(verbose_coarsen) {
        MPI_Barrier(comm); printf("coarsen: start: rank = %d\n", rank); MPI_Barrier(comm);
    }
#endif

	//cout << "set up P" << endl;
	int order  = A->p_order;
	int prodim = 2;
    
    std::string filename_map = "../data/nektar/nek_map_"+std::to_string(rank)+".txt";
	std::string filename_g2u = "../data/nektar/nek_g2u_"+std::to_string(rank)+".txt";

#ifdef __DEBUG1__
//    std::cout << "filename_map: " << filename_map << std::endl;
//    std::cout << "filename_g2u: " << filename_g2u << std::endl;
#endif

//    vector< vector<int> > map = connect(order, a_elemno, prodim);
    vector< vector<int> > map = mesh_info(order, filename_map, map_all, comm);

#ifdef __DEBUG1__
    if(verbose_coarsen) {
        MPI_Barrier(comm); printf("coarsen: step 1: rank = %d\n", rank); MPI_Barrier(comm);
    }
#endif

    vector<int> g2u;
    if (map_all.size() == 1){
        g2u = g2umap(order, filename_g2u, g2u_all, map, comm);
    }else{
        g2u = g2umap(order, filename_g2u, g2u_all, map_all.at(map_all.size()-2), comm);
    }

#ifdef __DEBUG1__
    if(verbose_coarsen) {
        MPI_Barrier(comm); printf("coarsen: step 2: rank = %d\n", rank); MPI_Barrier(comm);
    }
#endif

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

    vector<cooEntry_row> P_temp;
//    vector< vector<double> > Pp;//, Rp;
	//set_PR_from_p(order, map, prodim, Pp);//, Rp);
    set_P_from_mesh(order, map, prodim, P_temp, comm, g2u);//, Rp);

#ifdef __DEBUG1__
//    print_vector(P_temp, -1, "P_temp", comm);
    if(verbose_coarsen) {
        MPI_Barrier(comm); printf("coarsen: step 3: rank = %d\n", rank); MPI_Barrier(comm);
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

/*	FILE *filename;
	filename = fopen("P.txt", "w");
	for (int i=0; i<row; i++)
	{
		for (int j=0; j<col; j++)
		{
			//cout << Pp[i][j] << " ";
			fprintf(filename, "%.12f ", Pp[i][j]);
		}
		//cout << "\n";
		fprintf(filename, "\n");
	}
	fclose(filename);
*/

//    vector<cooEntry> P_temp;
//    for(int i = 0 ; i < Pp.size(); ++i){
//    	for(int j = 0; j < Pp.at(0).size(); ++j){
//			if (fabs(Pp.at(i).at(j)) > 1e-12){
//                P_temp.emplace_back( cooEntry(i, j, Pp.at(i).at(j)) );
//			}
//    	}
//	}

    P->comm  = A->comm;
    P->split = A->split;
    P->Mbig  = A->Mbig;
    P->M     = A->M;

#ifdef __DEBUG1__
    if(verbose_coarsen) {
        MPI_Barrier(comm); printf("coarsen: step 4: rank = %d\n", rank); MPI_Barrier(comm);
    }
//    std::sort(P_temp.begin(), P_temp.end(), row_major);
//    print_vector(P_temp, -1, "P_temp", comm);
#endif

//    std::sort(P_temp.begin(), P_temp.end(), row_major);

    vector<cooEntry_row> Pent;
//    P->entry.clear();
    par::sampleSort(P_temp, Pent, P->split, comm);

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
//    print_vector(Pent, -1, "Pent", comm);
//    print_vector(P->split, -1, "P->split", comm);
    if(verbose_coarsen) {
        MPI_Barrier(comm); printf("coarsen: step 5: rank = %d\n", rank); MPI_Barrier(comm);
    }
#endif

//    P->comm  = A->comm;
//    P->split = A->split;
//    P->Mbig  = A->Mbig;
//    P->M     = A->M;
//	P->Nbig  = Pp.at(0).size(); //TODO

    if(rank == nprocs - 1){
        P->Nbig = P->entry.back().col + 1;
    }

    MPI_Bcast(&P->Nbig, 1, MPI_UNSIGNED, nprocs-1, comm);

    P->splitNew.resize(nprocs+1);
    P->splitNew[0]      = 0;
    P->splitNew[nprocs] = P->Nbig;

    int ofst = P->Nbig / nprocs;
    for(int i = 1; i < nprocs; ++i){
        P->splitNew[i] = i * ofst;
    }

    P->nnz_l = P->entry.size();
    MPI_Allreduce(&P->nnz_l, &P->nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, P->comm);

#ifdef __DEBUG1__
    if(verbose_coarsen) {
        MPI_Barrier(comm);
        std::stringstream buffer;
        buffer << "rank " << rank << ": P: " << "M = " << P->M << ", Mbig = " << P->Mbig << ", Nbig = " << P->Nbig
               << ", nnz_l = " << P->nnz_l << ", nnz_g = " << P->nnz_g;
        cout << buffer.str() << endl;
//        print_vector(P->splitNew, 0, "P->splitNew", comm);
        MPI_Barrier(comm);
    }
#endif

#ifdef __DEBUG1__
    if(verbose_coarsen) {
        MPI_Barrier(comm); printf("coarsen: step 6: rank = %d\n", rank); MPI_Barrier(comm);
    }
#endif

    P->findLocalRemote();

#ifdef __DEBUG1__
    if(verbose_coarsen) {
        MPI_Barrier(comm); printf("end of coarsen: step 2: rank = %d\n", rank); MPI_Barrier(comm);
    }
#endif

    return 0;
}


vector<int> saena_object::next_p_level(vector<int> ind_fine, int order){
    // assuming the elemental ordering is like
    // 7--8--9
    // |  |  |
    // 4--5--6
    // |  |  |
    // 1--2--3

    vector<int> indices;
    for (int i=0; i<order/2+1; i++){
        for (int j=0; j<order/2+1; j++){
#ifdef __DEBUG1__
//            ASSERT((2*j+2*(order+1)*i >= 0) && (2*j+2*(order+1)*i < ind_fine.size()),
//                   i << "\t" << j << "\t" << order << "\t" << 2*j+2*(order+1)*i);
//            cout << i << "\t" << j << "\t" << order << "\t" << ind_fine.size() << "\t" << 2*j+2*(order+1)*i << "\t" << ind_fine[2*j+2*(order+1)*i] << endl;
#endif
            indices.push_back(ind_fine[2*j+2*(order+1)*i]);
        }
    }

    // 3--7--4
    // |  |  |
    // 8--9--6
    // |  |  |
    // 1--5--2
    // only for test from 2 -> 1
    /*vector<int> indices;
    for (int i=0; i<4; i++)
        indices.push_back(ind_fine[i]);*/

    return indices;
}

vector<int> saena_object::coarse_p_node_arr(vector< vector<int> > map, int order)
{
    int total_elem = map.size();
    vector<int> ind;
    for (int i=0; i<total_elem; i++)
    {
        vector<int> ind_coarse = next_p_level(map.at(i), order);
        for (int j=0; j<ind_coarse.size(); j++)
        {
            if (!ismember(ind_coarse.at(j), ind))
                ind.push_back(ind_coarse.at(j));
        }
       
    }
    return ind;
} 

void saena_object::set_P_from_mesh(int order, vector<vector<int>> map, int prodim, vector<cooEntry_row> &P_temp, MPI_Comm comm, vector<int> g2u){
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // get universal number of dof
    // any other better way?
	int g2u_univ_size = 0;
	int g2u_size = g2u.size();
   
	MPI_Allreduce(&g2u_size, &g2u_univ_size, 1, MPI_INT, MPI_SUM, comm);

	vector<int> g2u_univ_map(g2u_univ_size);
	vector<int> count_arr(nprocs);
	MPI_Allgather(&g2u_size, 1, MPI_INT, count_arr.data(), 1, MPI_INT, comm);

	vector<int> displs(nprocs);
	displs[0] = 0;
	for (int i=1; i<nprocs;i++)
		displs[i] = displs[i-1]+count_arr[i-1];

	MPI_Allgatherv(g2u.data(), g2u.size(), MPI_INT, g2u_univ_map.data(), count_arr.data(), displs.data(), MPI_INT, comm);

	// compute the universal fine matrix size
	// ======================================
    vector<int> g2u_univ_map_tmp = g2u_univ_map;

    // sort and remove the duplicates
    sort( g2u_univ_map_tmp.begin(), g2u_univ_map_tmp.end() );
    g2u_univ_map_tmp.erase( unique( g2u_univ_map_tmp.begin(), g2u_univ_map_tmp.end() ), g2u_univ_map_tmp.end() );
    
    int univ_nodeno_fine = g2u_univ_map_tmp.size();
    g2u_univ_map_tmp.clear();
//    g2u_univ_map.clear(); //TODO

    // ======================================

    /*cout << "total_dof = " << univ_nodeno_fine << "\n";
    MPI_Barrier(comm);
    exit(0);*/

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

    if (rank == rank_v)
        std::cout << "Pp_loc has row (universal) = " << Pp_loc.size() << ", and col(global) = " << Pp_loc[0].size() << endl;

    /*MPI_Barrier(comm);
    exit(0);*/
    // next level g2u
    // index next level node index
    // value this level g2u value
	vector<int> g2u_map(nodeno_coarse);
    // coarse_node_ind index is coraser mesh node index
    // coarse_node_ind value is finer mesh node index

    sort(coarse_node_ind.begin(), coarse_node_ind.end());

	//cout << map.size() << " " << map.at(0).size() << "\n";
    //skip bdy node
	//vector<int> skip;
    // loop over all elements
    // elemno is the local element number
    for (int i=0; i<elemno; i++){
        // for each element extract coraser element indices
        vector<int> ind_coarse = next_p_level(map.at(i), order);

		//std::cout << "ind_coarse size: " << ind_coarse.size() << std::endl;
		//std::cout << "map.at(i) size: " << map.at(i).size() << std::endl;
		/*for (int ii=0; ii<ind_coarse.size(); ii++)
			std::cout << ind_coarse.at(ii) << " ";
		std::cout << std::endl;*/
		//cout << ind_coarse.size() << "\n";

        for (int j=0; j<ind_coarse.size(); j++){
            // skip bdy node
            if (ind_coarse.at(j)-1 < bdydof)
                continue;

            // interpolate basis function of j at all other nodes
            // in this element as the values of corresponding P entries
            // TODO only for 2d now upto basis order 8
            vector<double> val = get_interpolation(j + 1, order, prodim);

			// cout << val.size() << "\n";

            // find coarse_node_ind index (coarser mesh node index) that
            // has the same value (finer mesh node index) as ind_coarse value
            // TODO This is slow, may need smarter way
            int P_col = findloc(coarse_node_ind, ind_coarse.at(j));

            /*if (rank == rank_v && ind_coarse[j] == 42)
                cout << "P_col = " << P_col << std::endl;*/

            // index corase level dof
            // value universal value at this level
			g2u_map.at(P_col) = g2u[ind_coarse.at(j)-1-bdydof];
            // assuming the map ordering (connectivity) is the same as ref element
            // shared nodes between elements should have the same values
            // when evaluated in each of the elememnts
            //cout << "val size " << val.size() << endl;

            // set P entries
            for (int k=0; k<val.size(); k++){
                if (map.at(i).at(k)-1 < bdydof)
                    continue;

                // row: g2u.at(map.at(i).at(k)-1-bdydof)
                // col: P_col
                // universal col id: g2u_map[P_col]
                // row is universal
                if(fabs(val[k]) > 1e-14){
                    P_temp.emplace_back(g2u.at(map.at(i).at(k)-1-bdydof), g2u_map[P_col], val[k]);
                }

//                Pp_loc.at(g2u.at(map.at(i).at(k)-1-bdydof)).at(P_col) = val.at(k);
                /*if (rank == rank_v && ind_coarse[j] == 42)
				    cout << "row = " << g2u.at(map.at(i).at(k)-1-bdydof) << ", and value = " << val[k] << "\n";*/
			}
            /*if (rank == rank_v && ind_coarse[j] == 42)
                cout << "\n";*/
            /*if (rank == rank_v)
                cout << endl;*/
			// For nektar which removes dirichlet bc in matrix
			//if (ind_coarse.at(j) < bdydof && !ismember(P_col, skip))
				//skip.push_back(P_col);
        }
    }

    std::sort(P_temp.begin(), P_temp.end());
    P_temp.erase( unique( P_temp.begin(), P_temp.end() ), P_temp.end() );

//    print_vector(P_temp, -1, "P_temp", comm);




#if 0
    // this is different as the allgather at begining of the function
    // g2u's index is for current level global
    // while g2u_map's index is for next corase level
	g2u_size = g2u_map.size();
	MPI_Allreduce(&g2u_size, &g2u_univ_size, 1, MPI_INT, MPI_SUM, comm);

	//g2u_univ_map.clear();
	g2u_univ_map.resize(g2u_univ_size);
    count_arr.clear();
    count_arr.resize(nprocs);
	MPI_Allgather(&g2u_size,1,MPI_INT,count_arr.data(),1, MPI_INT,comm);

	displs.clear();
    displs.resize(nprocs);
	displs[0] = 0;
	for (int i=1; i<nprocs;i++)
		displs[i] = displs[i-1]+count_arr[i-1];

	MPI_Allgatherv(g2u_map.data(), g2u_map.size(), MPI_INT, g2u_univ_map.data(), count_arr.data(), displs.data(), MPI_INT, comm);
    
	//cout << "skip size: " << skip.size() << "\n";
	/*for (int ii=0; ii<skip.size(); ii++)
            std::cout << skip.at(ii) << " ";
    std::cout << std::endl;*/
    /*if (rank == rank_v)
    {
	    int row_p = Pp_loc.size();
        int col_p = Pp_loc.at(0).size();
        cout << "inside set_PR_from_p, Pp_loc row: " << row_p << ", and col: " << col_p << "\n";
        for (int i=0; i<Pp_loc.size(); i++)
        {
            for (int j=0; j<Pp_loc[0].size(); j++)
                cout << Pp_loc[i][j] << " ";
            cout<<endl;
        }
    }*/
    /* MPI_Barrier(comm);
    exit(0); */
	// Just to match nektar++ petsc matrix
	// assume all the Dirichlet BC nodes are at the begining
	// remove them from the matrix
	// remove columns
	/*for (int i=0; i<Pp_loc.size(); i++)
	{
		int counter = 0;
		for (int j = 0; j<Pp_loc.at(i).size();)
		{
			if (ismember(counter,skip))
				Pp_loc.at(i).erase(Pp_loc.at(i).begin()+j);
			else
				j++;
			
			counter ++;
		}
	}*/
	// remove rows
	//Pp_loc.erase(Pp_loc.begin(), Pp_loc.begin()+bdydof);
    //Rp = transp(Pp_loc);

	// comunication of universal P
	// collect glb_map and P colums
	// recast 2d vector into 1d
	// TODO this may be big. 
	// Pp is better in a COO or other proper format
    // col leading
	vector<double> Pp_1d;
	for (int i=0; i<Pp_loc[0].size(); i++)
	{
		for (int j=0; j<Pp_loc.size(); j++)
        {
			Pp_1d.push_back(Pp_loc[j][i]);
            //if (rank == rank_v)
               //cout << Pp_1d[Pp_1d.size()-1] << std::endl;
        }
	}
    //Pp_loc.clear();
	
	int Pp_univ_size;
	int Pp_size = Pp_1d.size();
	MPI_Allreduce(&Pp_size, &Pp_univ_size, 1, MPI_INT, MPI_SUM, comm);

	vector<double> Pp_univ(Pp_univ_size);
    count_arr.clear();
    count_arr.resize(nprocs);
    displs.clear();
    displs.resize(nprocs);
	MPI_Allgather(&Pp_size,1,MPI_INT,count_arr.data(),1, MPI_INT,comm);

	displs[0] = 0;
	for (int i=1; i<nprocs;i++)
		displs[i] = displs[i-1]+count_arr[i-1];
    
	MPI_Allgatherv(Pp_1d.data(), Pp_1d.size(), MPI_DOUBLE, Pp_univ.data(), count_arr.data(), displs.data(), MPI_DOUBLE, comm);
    //Pp_1d.clear();
    // make sure the ordering of Pp is the same as g2u map
	// recast Pp to 2d vector for sorting
	vector < vector<double> > Pp_2d;
	for (int i=0; i<Pp_univ.size()/univ_nodeno_fine; i++){
        Pp_2d.emplace_back(vector<double>());
		for (int j=0; j<univ_nodeno_fine; j++){
            Pp_2d[i].push_back(Pp_univ[i*univ_nodeno_fine+j]);
		}
    }
    //Pp_univ.clear();
   
   /*  if (rank == rank_v)
	{
        for (int i=0; i<univ_nodeno_fine; i++)
	    {
		    for (int j=0; j<Pp_univ.size()/univ_nodeno_fine; j++)
			    cout << Pp_univ[univ_nodeno_fine*j+i] << " ";
            
            std::cout << "\n";
	    }
        
    } */
    /*MPI_Barrier(comm);
    exit(0);*/

    /*if (rank == rank_v)
    {
        cout << "before sort" << endl;
        for (int i=0; i<g2u_univ_map.size(); i++)
            cout << g2u_univ_map[i] << endl;
    }
    
    if (rank == rank_v)
    {
        cout << "before sort" << endl;
        for (int i=0; i<univ_nodeno_fine; i++)
        {
            for (int j=0; j<Pp_univ.size()/univ_nodeno_fine;j++)
                cout << Pp_2d[j][i]<< " ";
            cout << endl;
        }
    }*/

    //sort Pp according to g2u_univ_map
    vector<pair<int, int> > vp;  
    for (int i = 0; i < g2u_univ_size; ++i) { 
        vp.emplace_back(make_pair(g2u_univ_map[i], i));
    } 

    //g2u_univ_map.clear();
    // Sorting pair vector 
    sort(vp.begin(), vp.end()); 
    
    vector < vector<double> > Pp_2d_tmp(Pp_2d.size(), vector<double>(Pp_2d[0].size()));
    for (int i=0; i<Pp_2d.size();i++)
        Pp_2d_tmp[i] = Pp_2d[vp[i].second];

    //Pp_2d.clear();
    /*if (rank == rank_v)
    {
        cout << "after sort" << endl;
        for (int i=0; i<g2u_univ_map.size(); i++)
            cout << vp[i].first << endl;
        cout << "after sort" << endl;
        for (int i=0; i<g2u_univ_map.size(); i++)
            cout << vp[i].second << endl;
    }
    if (rank == rank_v)
    {
        cout << "after sort" << endl;
        for (int i=0; i<univ_nodeno_fine; i++)
        {
            for (int j=0; j<Pp_univ.size()/univ_nodeno_fine;j++)
                cout << Pp_2d_tmp[j][i]<< " ";
            cout << endl;
        }
    }*/

    /*MPI_Barrier(comm);
    exit(0);*/
	// combine P for shared col (some of the rows are in one processor while others are in another) and remove extra ones 
   
    // for testing
    //vector< vector<double> > Pp_2d_tmp1 = Pp_2d_tmp;

	for (int i=0; i<Pp_2d_tmp.size()-1; )
	{
		if ( vp[i].first == vp[i+1].first )
        {
		    //same P_col			
		    //merge that P_col;
            do
            {
				for (int j=0; j< Pp_2d_tmp[0].size(); j++)
				{
					if (Pp_2d_tmp[i][j] == 0 && Pp_2d_tmp[i+1][j] != 0)
						Pp_2d_tmp[i][j] = Pp_2d_tmp[i+1][j];
				}
                // remove
				Pp_2d_tmp.erase(Pp_2d_tmp.begin()+i+1);
                vp.erase(vp.begin()+i+1);
			} 
            while (vp[i].first == vp[i+1].first);
		}
        i++;
	}
	
    Pp.resize(Pp_2d_tmp[0].size());
    for (int i=0; i<Pp.size(); i++)
        Pp[i].resize(Pp_2d_tmp.size());

	for (int i=0; i<Pp_2d_tmp.size(); i++){
		for (int j=0; j<Pp_2d_tmp[0].size(); j++)
			Pp[j][i] = Pp_2d_tmp[i][j];
	}  

    /*if (rank == rank_v)
    {
        cout << "last" << endl;
        for (int i=0; i<last.size(); i++)
	    {
		    for (int j=0; j<last[0].size(); j++)
			    cout << last[i][j] << " ";
            cout << endl;
	    }  
    }*/

    /*if (rank == rank_v)
    {
        cout << "after sort" << endl;
        for (int j=0; j<univ_nodeno_fine;j++)
            cout << Pp_2d_tmp1[36][j]<< " " << Pp_2d_tmp1[37][j] << " " << Pp[j][36] << endl;
    }
    MPI_Barrier(comm);
    exit(0);*/
#endif

}


#if 0
void saena_object::set_P_from_mesh(int order, vector<vector<int>> map, int prodim, vector<vector<double>> &Pp, MPI_Comm comm, vector<int> g2u)//, vector< vector<double> > &Rp)
{
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // get universal number of dof
    // any other better way?
    int g2u_univ_size = 0;
    int g2u_size = g2u.size();

    MPI_Allreduce(&g2u_size, &g2u_univ_size, 1, MPI_INT, MPI_SUM, comm);

    vector<int> g2u_univ_map(g2u_univ_size);
    vector<int> count_arr(nprocs);
    MPI_Allgather(&g2u_size, 1, MPI_INT, count_arr.data(), 1, MPI_INT, comm);

    vector<int> displs(nprocs);
    displs[0] = 0;
    for (int i=1; i<nprocs;i++)
        displs[i] = displs[i-1]+count_arr[i-1];

    MPI_Allgatherv(g2u.data(), g2u.size(), MPI_INT, g2u_univ_map.data(), count_arr.data(), displs.data(), MPI_INT, comm);

    // compute the universal fine matrix size
    // ======================================
    vector<int> g2u_univ_map_tmp = g2u_univ_map;

    // sort and remove the duplicates
    sort( g2u_univ_map_tmp.begin(), g2u_univ_map_tmp.end() );
    g2u_univ_map_tmp.erase( unique( g2u_univ_map_tmp.begin(), g2u_univ_map_tmp.end() ), g2u_univ_map_tmp.end() );

    int univ_nodeno_fine = g2u_univ_map_tmp.size();
    g2u_univ_map_tmp.clear();
//    g2u_univ_map.clear(); //TODO

    // ======================================

    /*cout << "total_dof = " << univ_nodeno_fine << "\n";
    MPI_Barrier(comm);
    exit(0);*/

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

    if (rank == rank_v)
        std::cout << "Pp_loc has row (universal) = " << Pp_loc.size() << ", and col(global) = " << Pp_loc[0].size() << endl;

    /*MPI_Barrier(comm);
    exit(0);*/
    // next level g2u
    // index next level node index
    // value this level g2u value
    vector<int> g2u_map(nodeno_coarse);
    // coarse_node_ind index is coraser mesh node index
    // coarse_node_ind value is finer mesh node index

    sort(coarse_node_ind.begin(), coarse_node_ind.end());

    //cout << map.size() << " " << map.at(0).size() << "\n";
    //skip bdy node
    //vector<int> skip;
    // loop over all elements
    // elemno is the local element number
    for (int i=0; i<elemno; i++){
        // for each element extract coraser element indices
        vector<int> ind_coarse = next_p_level(map.at(i), order);

        //std::cout << "ind_coarse size: " << ind_coarse.size() << std::endl;
        //std::cout << "map.at(i) size: " << map.at(i).size() << std::endl;
        /*for (int ii=0; ii<ind_coarse.size(); ii++)
            std::cout << ind_coarse.at(ii) << " ";
        std::cout << std::endl;*/
        //cout << ind_coarse.size() << "\n";

        for (int j=0; j<ind_coarse.size(); j++){
            // skip bdy node
            if (ind_coarse.at(j)-1 < bdydof)
                continue;

            // interpolate basis function of j at all other nodes
            // in this element as the values of corresponding P entries
            // TODO only for 2d now upto basis order 8
            vector<double> val = get_interpolation(j + 1, order, prodim);

            // cout << val.size() << "\n";

            // find coarse_node_ind index (coarser mesh node index) that
            // has the same value (finer mesh node index) as ind_coarse value
            // TODO This is slow, may need smarter way
            int P_col = findloc(coarse_node_ind, ind_coarse.at(j));

            /*if (rank == rank_v && ind_coarse[j] == 42)
                cout << "P_col = " << P_col << std::endl;*/

            // index corase level dof
            // value universal value at this level
            g2u_map.at(P_col) = g2u[ind_coarse.at(j)-1-bdydof];
            // assuming the map ordering (connectivity) is the same as ref element
            // shared nodes between elements should have the same values
            // when evaluated in each of the elememnts
            //cout << "val size " << val.size() << endl;

            // set P entries
            for (int k=0; k<val.size(); k++){
                if (map.at(i).at(k)-1 < bdydof)
                    continue;

                // row: g2u.at(map.at(i).at(k)-1-bdydof)
                // col: P_col
                // universal col id: g2u_map[P_col]
                // row is universal
                Pp_loc.at(g2u.at(map.at(i).at(k)-1-bdydof)).at(P_col) = val.at(k);
                /*if (rank == rank_v && ind_coarse[j] == 42)
				    cout << "row = " << g2u.at(map.at(i).at(k)-1-bdydof) << ", and value = " << val[k] << "\n";*/
            }
            /*if (rank == rank_v && ind_coarse[j] == 42)
                cout << "\n";*/
            /*if (rank == rank_v)
                cout << endl;*/
            // For nektar which removes dirichlet bc in matrix
            //if (ind_coarse.at(j) < bdydof && !ismember(P_col, skip))
            //skip.push_back(P_col);
        }
    }

    // this is different as the allgather at begining of the function
    // g2u's index is for current level global
    // while g2u_map's index is for next corase level
    g2u_size = g2u_map.size();
    MPI_Allreduce(&g2u_size, &g2u_univ_size, 1, MPI_INT, MPI_SUM, comm);

    //g2u_univ_map.clear();
    g2u_univ_map.resize(g2u_univ_size);
    count_arr.clear();
    count_arr.resize(nprocs);
    MPI_Allgather(&g2u_size,1,MPI_INT,count_arr.data(),1, MPI_INT,comm);

    displs.clear();
    displs.resize(nprocs);
    displs[0] = 0;
    for (int i=1; i<nprocs;i++)
        displs[i] = displs[i-1]+count_arr[i-1];

    MPI_Allgatherv(g2u_map.data(), g2u_map.size(), MPI_INT, g2u_univ_map.data(), count_arr.data(), displs.data(), MPI_INT, comm);

    //cout << "skip size: " << skip.size() << "\n";
    /*for (int ii=0; ii<skip.size(); ii++)
            std::cout << skip.at(ii) << " ";
    std::cout << std::endl;*/
    /*if (rank == rank_v)
    {
	    int row_p = Pp_loc.size();
        int col_p = Pp_loc.at(0).size();
        cout << "inside set_PR_from_p, Pp_loc row: " << row_p << ", and col: " << col_p << "\n";
        for (int i=0; i<Pp_loc.size(); i++)
        {
            for (int j=0; j<Pp_loc[0].size(); j++)
                cout << Pp_loc[i][j] << " ";
            cout<<endl;
        }
    }*/
    /* MPI_Barrier(comm);
    exit(0); */
    // Just to match nektar++ petsc matrix
    // assume all the Dirichlet BC nodes are at the begining
    // remove them from the matrix
    // remove columns
    /*for (int i=0; i<Pp_loc.size(); i++)
    {
        int counter = 0;
        for (int j = 0; j<Pp_loc.at(i).size();)
        {
            if (ismember(counter,skip))
                Pp_loc.at(i).erase(Pp_loc.at(i).begin()+j);
            else
                j++;

            counter ++;
        }
    }*/
    // remove rows
    //Pp_loc.erase(Pp_loc.begin(), Pp_loc.begin()+bdydof);
    //Rp = transp(Pp_loc);

    // comunication of universal P
    // collect glb_map and P colums
    // recast 2d vector into 1d
    // TODO this may be big.
    // Pp is better in a COO or other proper format
    // col leading
    vector<double> Pp_1d;
    for (int i=0; i<Pp_loc[0].size(); i++)
    {
        for (int j=0; j<Pp_loc.size(); j++)
        {
            Pp_1d.push_back(Pp_loc[j][i]);
            //if (rank == rank_v)
            //cout << Pp_1d[Pp_1d.size()-1] << std::endl;
        }
    }
    //Pp_loc.clear();

    int Pp_univ_size;
    int Pp_size = Pp_1d.size();
    MPI_Allreduce(&Pp_size, &Pp_univ_size, 1, MPI_INT, MPI_SUM, comm);

    vector<double> Pp_univ(Pp_univ_size);
    count_arr.clear();
    count_arr.resize(nprocs);
    displs.clear();
    displs.resize(nprocs);
    MPI_Allgather(&Pp_size,1,MPI_INT,count_arr.data(),1, MPI_INT,comm);

    displs[0] = 0;
    for (int i=1; i<nprocs;i++)
        displs[i] = displs[i-1]+count_arr[i-1];

    MPI_Allgatherv(Pp_1d.data(), Pp_1d.size(), MPI_DOUBLE, Pp_univ.data(), count_arr.data(), displs.data(), MPI_DOUBLE, comm);
    //Pp_1d.clear();
    // make sure the ordering of Pp is the same as g2u map
    // recast Pp to 2d vector for sorting
    vector < vector<double> > Pp_2d;
    for (int i=0; i<Pp_univ.size()/univ_nodeno_fine; i++){
        Pp_2d.emplace_back(vector<double>());
        for (int j=0; j<univ_nodeno_fine; j++){
            Pp_2d[i].push_back(Pp_univ[i*univ_nodeno_fine+j]);
        }
    }
    //Pp_univ.clear();

    /*  if (rank == rank_v)
     {
         for (int i=0; i<univ_nodeno_fine; i++)
         {
             for (int j=0; j<Pp_univ.size()/univ_nodeno_fine; j++)
                 cout << Pp_univ[univ_nodeno_fine*j+i] << " ";

             std::cout << "\n";
         }

     } */
    /*MPI_Barrier(comm);
    exit(0);*/

    /*if (rank == rank_v)
    {
        cout << "before sort" << endl;
        for (int i=0; i<g2u_univ_map.size(); i++)
            cout << g2u_univ_map[i] << endl;
    }

    if (rank == rank_v)
    {
        cout << "before sort" << endl;
        for (int i=0; i<univ_nodeno_fine; i++)
        {
            for (int j=0; j<Pp_univ.size()/univ_nodeno_fine;j++)
                cout << Pp_2d[j][i]<< " ";
            cout << endl;
        }
    }*/

    //sort Pp according to g2u_univ_map
    vector<pair<int, int> > vp;
    for (int i = 0; i < g2u_univ_size; ++i) {
        vp.emplace_back(make_pair(g2u_univ_map[i], i));
    }

    //g2u_univ_map.clear();
    // Sorting pair vector
    sort(vp.begin(), vp.end());

    vector < vector<double> > Pp_2d_tmp(Pp_2d.size(), vector<double>(Pp_2d[0].size()));
    for (int i=0; i<Pp_2d.size();i++)
        Pp_2d_tmp[i] = Pp_2d[vp[i].second];

    //Pp_2d.clear();
    /*if (rank == rank_v)
    {
        cout << "after sort" << endl;
        for (int i=0; i<g2u_univ_map.size(); i++)
            cout << vp[i].first << endl;
        cout << "after sort" << endl;
        for (int i=0; i<g2u_univ_map.size(); i++)
            cout << vp[i].second << endl;
    }
    if (rank == rank_v)
    {
        cout << "after sort" << endl;
        for (int i=0; i<univ_nodeno_fine; i++)
        {
            for (int j=0; j<Pp_univ.size()/univ_nodeno_fine;j++)
                cout << Pp_2d_tmp[j][i]<< " ";
            cout << endl;
        }
    }*/

    /*MPI_Barrier(comm);
    exit(0);*/
    // combine P for shared col (some of the rows are in one processor while others are in another) and remove extra ones

    // for testing
    //vector< vector<double> > Pp_2d_tmp1 = Pp_2d_tmp;

    for (int i=0; i<Pp_2d_tmp.size()-1; )
    {
        if ( vp[i].first == vp[i+1].first )
        {
            //same P_col
            //merge that P_col;
            do
            {
                for (int j=0; j< Pp_2d_tmp[0].size(); j++)
                {
                    if (Pp_2d_tmp[i][j] == 0 && Pp_2d_tmp[i+1][j] != 0)
                        Pp_2d_tmp[i][j] = Pp_2d_tmp[i+1][j];
                }
                // remove
                Pp_2d_tmp.erase(Pp_2d_tmp.begin()+i+1);
                vp.erase(vp.begin()+i+1);
            }
            while (vp[i].first == vp[i+1].first);
        }
        i++;
    }

    Pp.resize(Pp_2d_tmp[0].size());
    for (int i=0; i<Pp.size(); i++)
        Pp[i].resize(Pp_2d_tmp.size());

    for (int i=0; i<Pp_2d_tmp.size(); i++){
        for (int j=0; j<Pp_2d_tmp[0].size(); j++)
            Pp[j][i] = Pp_2d_tmp[i][j];
    }

    /*if (rank == rank_v)
    {
        cout << "last" << endl;
        for (int i=0; i<last.size(); i++)
	    {
		    for (int j=0; j<last[0].size(); j++)
			    cout << last[i][j] << " ";
            cout << endl;
	    }
    }*/

    /*if (rank == rank_v)
    {
        cout << "after sort" << endl;
        for (int j=0; j<univ_nodeno_fine;j++)
            cout << Pp_2d_tmp1[36][j]<< " " << Pp_2d_tmp1[37][j] << " " << Pp[j][36] << endl;
    }
    MPI_Barrier(comm);
    exit(0);*/
}
#endif


void saena_object::set_PR_from_p(int order, vector< vector<int> > map, int prodim, vector< vector<double> > &Pp)//, vector< vector<double> > &Rp)
{
	/*int bnd;
	// hard coded ...
	if (order == 2)
		bnd = 80;
	else if (order == 4)
		bnd = 160;
	else
		bnd = 320;*/

    Pp.resize(nodeno_fine);

    for (int i = 0; i < nodeno_fine; i++)
        Pp.at(i).resize(nodeno_coarse);

    // coarse_node_ind index is coraser mesh node index
    // coarse_node_ind value is finer mesh node index
    vector<int> coarse_node_ind = coarse_p_node_arr(map, order);
    sort(coarse_node_ind.begin(), coarse_node_ind.end());

	//cout << map.size() << " " << map.at(0).size() << "\n";
    // loop over all elements
    int total_elem = map.size();
	vector<int> skip;

    for (int i=0; i<total_elem; ++i){
        // for each element extract coraser element indices
        vector<int> ind_coarse = next_p_level(map.at(i), order);
		//std::cout << "ind_coarse size: " << ind_coarse.size() << std::endl;
		//std::cout << "map.at(i) size: " << map.at(i).size() << std::endl;

//		for (int ii=0; ii<ind_coarse.size(); ii++)
//			std::cout << ind_coarse.at(ii) << " ";
//		std::cout << std::endl;

		//cout << ind_coarse.size() << "\n";
        for (int j=0; j<ind_coarse.size(); j++)
        {
            // interpolate basis function of j at all other nodes
            // in this element as the values of corresponding P entries
            // TODO only for 2d now upto basis order 8
            vector<double> val = get_interpolation(j+1,order,prodim);
			// cout << val.size() << "\n";
            // find coarse_node_ind index (coarser mesh node index) that
            // has the same value (finer mesh node index) as ind_coarse value
            // This is slow, may need smarter way
            int P_col = findloc(coarse_node_ind, ind_coarse.at(j));
            // assuming the map ordering (connectivity) is the same as ref element
            // shared nodes between elements should have the same values
            // when evaluated in each of the elememnts
            for (int k=0; k<val.size(); k++){
                Pp.at(map.at(i).at(k)-1).at(P_col) = val.at(k);
				//cout << val[k] << "\n";
			}

			// For nektar which removes dirichlet bc in matrix
			if (ind_coarse.at(j) < bdydof && !ismember(P_col, skip))
				skip.push_back(P_col);
        }
    }

	//cout << "skip size: " << skip.size() << "\n";
	/*for (int ii=0; ii<skip.size(); ii++)
            std::cout << skip.at(ii) << " ";
    std::cout << std::endl;*/

	int row_p = Pp.size();
    int col_p = Pp.at(0).size();
    //cout << "inside set_PR_from_p, Pp row: " << row_p << ", and col: " << col_p << "\n";

	// Just to match nektar++ petsc matrix
	// assume all the Dirichlet BC nodes are at the begining
	// remove them from the matrix
	// remove columns
	for (int i=0; i<Pp.size(); i++){
		int counter = 0;
		for (int j = 0; j<Pp.at(i).size();)
		{
			if (ismember(counter,skip))
				Pp.at(i).erase(Pp.at(i).begin()+j);
			else
				j++;
			
			counter ++;
		}
	}

	// remove rows
	Pp.erase(Pp.begin(), Pp.begin()+bdydof);
    //Rp = transp(Pp);
	int row_after = Pp.size();
    int col_after = Pp.at(0).size();
    //cout << "inside set_PR_from_p after remove, Pp row: " << row_after << ", and col: " << col_after << "\n";
}


/*inline vector< vector<double> > saena_object::transp(vector< vector<double> > M)
{
    int row = M.size();
    int col = M[0].size();
    vector< vector<double> > N(col, vector<double>(row));
    for (int i=0; i<col; i++)
        for (int j=0; j<row; j++)
            N[i][j] = M[j][i];

    return N;
}*/


//this is the function as mesh info for test for now
/*inline vector< vector<int> > saena_object::connect(int order, int a_elemno, int prodim)
{
    vector <vector<int> > map(pow(a_elemno, prodim));
    int a_nodeno = a_elemno*order+1;
    for (int i=0; i<a_elemno; i++)
    {
        for (int j=0; j<a_elemno; j++)
        {
            int k = i*a_elemno+j;
            int st = a_nodeno*order*i+order*j;
            //en = st + order;
            for (int m=0; m<order+1; m++)
                for (int n=0; n<order+1; n++)
                    map.at(k).push_back(st+m*a_nodeno+n+1);
        }
    }
    return map;
}*/


//this is the function as mesh info for test for now
inline vector< std::vector<int> > saena_object::mesh_info(int order, string filename, vector< vector< vector<int> > > &map_all, MPI_Comm comm)
{
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    vector <vector<int> > map;
    if (map_all.empty())
    {
        // assume pure quad elememt for now
        ifstream ifs;
        ifs.open(filename.c_str());
        //std::cout << filename << std::endl;
        istringstream iss;
        string aLine;
        getline(ifs, aLine);
        elemno = 0;
        while(!aLine.empty())
        {
            elemno++;
            map.emplace_back(vector<int>());
            iss.str(aLine);
            for (int j=0; j<(order+1)*(order+1); j++)
            {
                int val = 0;
                iss >> val;
                map.at(map.size()-1).emplace_back(val+1);
            }
            iss.clear();
            getline(ifs, aLine);
        }
        ifs.clear();
        ifs.close();
        iss.clear();

		/*for(int k=0; k<map.at(0).size();k++)
		{
			std::cout << map.at(0).at(k) << " ";
		}
		std::cout << "\n";*/
		//exit(0);
    }
    else
    { 
        vector< vector<int> > map_pre = map_all.at(map_all.size()-1);
        // coarse_node_ind index is coraser mesh node index
        // coarse_node_ind value is finer mesh node index
        vector<int> coarse_node_ind = coarse_p_node_arr(map_pre, order*2);
        sort(coarse_node_ind.begin(), coarse_node_ind.end());
        
        for (int i = 0; i < elemno; ++i){
            vector<int> aline = map_pre.at(i);
            vector<int> ind_coarse = next_p_level(aline, order*2);
            for (int j = 0; j < ind_coarse.size(); ++j){
                int mapped_val = findloc(coarse_node_ind, ind_coarse.at(j));
                map.at(i).emplace_back(mapped_val+1);
            }
        }
    }

	map_all.emplace_back(vector< vector<int> >());
	for (int i = 0; i < map.size(); ++i){
		map_all.at(map_all.size()-1).push_back(vector<int>());
		for (int j = 0; j < map.at(0).size(); ++j){
			map_all.at(map_all.size()-1).at(i).emplace_back(map.at(i).at(j));
		}
	}

    // get fine and corase number of nodes in this P level
    nodeno_coarse = coarse_p_node_arr(map, order).size();
    nodeno_fine = 0;
    for (int i = 0; i < map.size(); ++i){
        nodeno_fine = std::max(*max_element(map[i].begin(), map[i].end()), nodeno_fine);        
    }

    if (rank == rank_v){
        cout << "elem # = " << elemno << endl;
        cout << "fine node # = " << nodeno_fine << ", and coarse node # = " << nodeno_coarse << endl;
    }

	//std::cout << map_all.size() << " " << map_all.at(map_all.size()-1).size() << " " << map_all.at(map_all.size()-1).at(0).size() << std::endl;

    return map;
}

//this is the function as mesh info for test for now
inline std::vector<int> saena_object::g2umap(int order, string filename, vector< vector<int> > &g2u_all, vector< vector<int> > map, MPI_Comm comm)
{
    vector<int> g2u;
    if (g2u_all.empty()){
        // assume pure quad elememt for now
        ifstream ifs;
        ifs.open(filename.c_str());
        istringstream iss;
        string aLine;
        getline(ifs, aLine);
        iss.str(aLine);
		iss >> bdydof;
		iss.clear();
		getline(ifs, aLine);
        while (!aLine.empty())
        {
            iss.str(aLine);
            int uid;
            iss >> uid;
            g2u.push_back(uid);
            iss.clear();
        	getline(ifs, aLine);
        }
        ifs.clear();
        ifs.close();
        iss.clear();
		/*for(int k=0; k<g2u.size()/2;k++)
		{
			std::cout << g2u.at(2*k) << " " << g2u.at(2*k+1);
		}
		std::cout << "\n";
		exit(0);*/
    }
    else
    { 
		vector <int> next_level_g2u;
        // coarse_node_ind index is coraser mesh node index
        // coarse_node_ind value is finer mesh node index
        vector<int> coarse_node_ind = coarse_p_node_arr(map, order*2);
        int next_bdydof = 0;
        for (int i = 0; i < coarse_node_ind.size(); ++i)
        {
			// get universal dof value
			// this value will go to next level
            if (coarse_node_ind.at(i)-1 <bdydof)
                next_bdydof ++;
            else
                next_level_g2u.push_back(g2u_all.at(g2u_all.size()-1).at(coarse_node_ind.at(i)-1-bdydof));
        }
		// now fill mapping from global to universal in next level
		// need communication
    	int nprocs, rank;
    	MPI_Comm_size(comm, &nprocs);
    	MPI_Comm_rank(comm, &rank);

		int g2u_univ_size;
		int glb_size = next_level_g2u.size();
		MPI_Allreduce(&glb_size, &g2u_univ_size, 1, MPI_INT, MPI_SUM, comm);
		vector<int> g2u_univ(g2u_univ_size);
		vector<int> count_arr(nprocs);
		MPI_Allgather(&glb_size,1,MPI_INT,count_arr.data(),1, MPI_INT,comm);
		vector<int> displs(nprocs);
		displs[0] = 0;
		for (int i=1; i<nprocs;i++)
			displs[i] = displs[i-1]+count_arr[i-1];

		MPI_Allgatherv(next_level_g2u.data(), next_level_g2u.size(), MPI_INT, g2u_univ.data(), count_arr.data(), displs.data(), MPI_INT, comm);
		// sort the universal g2u map to make sure it is consistent with universal Ac = R*A*P dof ordering 
		// since universal P column is also sorted in the same way
		// now universal g2u index becomes the map value for Ac
		sort(g2u_univ.begin(),g2u_univ.end());
        g2u_univ.erase( unique( g2u_univ.begin(), g2u_univ.end() ), g2u_univ.end() );
		// loop over global map to assign universal value to it
		
		for (int i=0; i<coarse_node_ind.size(); i++)
		{
			// if it is boundary node, it has no conterpart in g2u
			// bdydof will be at the begining since coarse_node_ind is sorted
			if (coarse_node_ind[i] -1 < bdydof)
                continue;
			int next_g2u_val = findloc(g2u_univ, g2u_all.at(g2u_all.size()-1).at(coarse_node_ind[i]-1-bdydof));
			// relate i-bdydof (global dof in next level) and next_g2u_val (universal dof in next level)
			g2u.push_back(next_g2u_val);		
		}
		bdydof = next_bdydof;
    }

	g2u_all.push_back( vector<int> ());
	for (int i=0; i<g2u.size(); i++)
	{
		g2u_all.at(g2u_all.size()-1).push_back(g2u.at(i));
	}

	//std::cout << map_all.size() << " " << map_all.at(map_all.size()-1).size() << " " << map_all.at(map_all.size()-1).at(0).size() << std::endl;
    return g2u;
}

// TODO hard coded
// assuming 2d for now ...
vector<double> saena_object::get_interpolation(int ind, int order, int prodim){
    vector<double> val;//((order+1)*(order+1));
    vector< vector<double> > coord((order+1)*(order+1));
	// uniformly distributed
    /*for (int i=0; i<order+1; i++)
    {
        for (int j=0; j<order+1; j++)
        {
            double x = -1.0+2.0/order*j;
            double y = -1.0+2.0/order*i;
            coord[(order+1)*i+j].push_back(x);
            coord[(order+1)*i+j].push_back(y);
        }
    }*/

	// Guass Lobatto distributed
    for (int i=0; i<order+1; i++)
    {
        for (int j=0; j<order+1; j++)
        {
			if (order == 2)
			{
            	double x = -1.0+2.0/order*j;
            	double y = -1.0+2.0/order*i;
            	coord.at((order+1)*i+j).push_back(x);
            	coord.at((order+1)*i+j).push_back(y);
			}
			else if (order == 4)
			{
				const double gl[5] = {-1.0, -sqrt(3.0/7), 0, sqrt(3.0/7), 1.0};
            	double x = gl[j];
            	double y = gl[i];
            	coord.at((order+1)*i+j).push_back(x);
            	coord.at((order+1)*i+j).push_back(y);
			}
        }
    }
    // from 2->1
    if (order == 2)
    {
        // 1d lagrange basis (1-x)/2 and (1+x)/2
        switch (ind)
        {
            case 1:
                for (int i=0; i<order+1; i++)
                {
                    for (int j=0; j<order+1; j++)
                    {
                        int k = (order+1)*i+j;
                        double tmp = (1-coord.at(k).at(0))/2*(1-coord.at(k).at(1))/2;
                        val.push_back(tmp);
                    }
                }
                break;
            case 2:
                for (int i=0; i<order+1; i++)
                {
                    for (int j=0; j<order+1; j++)
                    {
                        int k = (order+1)*i+j;
                        double tmp = (1+coord.at(k).at(0))/2*(1-coord.at(k).at(1))/2;
                        val.push_back(tmp);
                    }
                }
                break;
            case 3:
                for (int i=0; i<order+1; i++)
                {
                    for (int j=0; j<order+1; j++)
                    {
                        int k = (order+1)*i+j;
                        double tmp = (1-coord.at(k).at(0))/2*(1+coord.at(k).at(1))/2;
                        val.push_back(tmp);
                    }
                }
                break;
            case 4:
                for (int i=0; i<order+1; i++)
                {
                    for (int j=0; j<order+1; j++)
                    {
                        int k = (order+1)*i+j;
                        double tmp = (1+coord.at(k).at(0))/2*(1+coord.at(k).at(1))/2;
                        val.push_back(tmp);
                    }
                }
                break;
        }  
                
    }
    else if (order == 4)
    {
        // 1d lagrange basis (x-1)*x/2 (1-x)*(x+1) and (1+x)*x/2
        switch(ind)
        {
            case 1:
                for (int i=0; i<order+1; i++)
                {
                    for (int j=0; j<order+1; j++)
                    {
                        int k = (order+1)*i+j;
                        double tmp = (coord.at(k).at(0)-1)*coord.at(k).at(0)/2*
                                     (coord.at(k).at(1)-1)*coord.at(k).at(1)/2;
                        val.push_back(tmp);
                    }
                }
                break;
            case 2:
                for (int i=0; i<order+1; i++)
                {
                    for (int j=0; j<order+1; j++)
                    {
                        int k = (order+1)*i+j;
                        double tmp = (1-coord.at(k).at(0))*(coord.at(k).at(0)+1)*
                                     (coord.at(k).at(1)-1)*coord.at(k).at(1)/2;
                        val.push_back(tmp);
                    }
                }
                break;
            case 3:
                for (int i=0; i<order+1; i++)
                {
                    for (int j=0; j<order+1; j++)
                    {
                        int k = (order+1)*i+j;
                        double tmp = (1+coord.at(k).at(0))*coord.at(k).at(0)/2*
                                     (coord.at(k).at(1)-1)*coord.at(k).at(1)/2;
                        val.push_back(tmp);
                    }
                }
                break;
            case 4:
                for (int i=0; i<order+1; i++)
                {
                    for (int j=0; j<order+1; j++)
                    {
                        int k = (order+1)*i+j;
                        double tmp = (coord.at(k).at(0)-1)*coord.at(k).at(0)/2*
                                     (1-coord.at(k).at(1))*(1+coord.at(k).at(1));
                        val.push_back(tmp);
                    }
                }
                break;
            case 5:
                for (int i=0; i<order+1; i++)
                {
                    for (int j=0; j<order+1; j++)
                    {
                        int k = (order+1)*i+j;
                        double tmp = (1-coord.at(k).at(0))*(1+coord.at(k).at(0))*
                                     (1-coord.at(k).at(1))*(1+coord.at(k).at(1));
                        val.push_back(tmp);
                    }
                }
                break;
            case 6:
                for (int i=0; i<order+1; i++)
                {
                    for (int j=0; j<order+1; j++)
                    {
                        int k = (order+1)*i+j;
                        double tmp = (coord.at(k).at(0)+1)*coord.at(k).at(0)/2*
                                     (1-coord.at(k).at(1))*(1+coord.at(k).at(1));
                        val.push_back(tmp);
                    }
                }
                break;
            case 7:
                for (int i=0; i<order+1; i++)
                {
                    for (int j=0; j<order+1; j++)
                    {
                        int k = (order+1)*i+j;
                        double tmp = (coord.at(k).at(0)-1)*coord.at(k).at(0)/2*
                                     (1+coord.at(k).at(1))*coord.at(k).at(1)/2;
                        val.push_back(tmp);
                    }
                }
                break;
            case 8:
                for (int i=0; i<order+1; i++)
                {
                    for (int j=0; j<order+1; j++)
                    {
                        int k = (order+1)*i+j;
                        double tmp = (1-coord.at(k).at(0))*(coord.at(k).at(0)+1)*
                                     (1+coord.at(k).at(1))*coord.at(k).at(1)/2;
                        val.push_back(tmp);
                    }
                }
                break;
            case 9:
                for (int i=0; i<order+1; i++)
                {
                    for (int j=0; j<order+1; j++)
                    {
                        int k = (order+1)*i+j;
                        double tmp = (coord.at(k).at(0)+1)*coord.at(k).at(0)/2*
                                     (1+coord.at(k).at(1))*coord.at(k).at(1)/2;
                        val.push_back(tmp);
                    }
                }
                break;
        }
    }
	// This is a more clean way
	// will change above to this later
	else if (order == 8)
	{
		vector< vector<double> > table = eighth_order(order);
		int x_dir = (ind-1) % (order/2+1);
		int y_dir = (ind-1) / (order/2+1);
		for (int i=0; i<order+1; i++)
		{
			for (int j=0; j<order+1; j++)
			{
				int k = (order+1)*i+j;
				double tmp = table.at(x_dir).at(j)*table.at(y_dir).at(i);
				val.push_back(tmp);
			}
		}

	}
	
	return val;
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
    //exit(0);
    return -1;
}

// may make this general in the future
inline vector< vector<double> > saena_object::eighth_order(int order)
{
	const vector<double> gl4{-1.0, -sqrt(3.0/7), 0, sqrt(3.0/7), 1.0};
	const vector<double> gl8{-1.0, -0.8997579954, -0.6771862795, -0.3631174638, 0, 0.3631174638, 0.6771862795, 0.8997579954, 1.0};
	vector< vector<double> > table;
	for (int i=0; i<order/2+1; i++)
	{
		table.push_back(vector<double>());
		for (int j=0; j<order+1; j++)
		{
			double val = 1.0;
			for (int k=0; k<order/2+1; k++)
			{
				if (i!=k)
					val *= (gl8.at(j)-gl4.at(k))/(gl4.at(i)-gl4.at(k));
			}
			table.at(i).push_back(val);
		}
	}
	return table;
}
