#include "saena_object.h"
#include "saena_matrix.h"
#include "strength_matrix.h"
#include "prolong_matrix.h"
//#include "restrict_matrix.h"
#include "grid.h"
#include "aux_functions.h"
#include "parUtils.h"

#include <numeric>

using namespace std;

// Assume the mesh info is the connectivity
// using a 2d vector for now
int saena_object::pcoarsen(Grid *grid, vector< vector< vector<int> > > &map_all,std::vector< std::vector<int> > &g2u_all){

    // test Jacobi
    /*int N = 10;
    double alpha = 7;
    double beta = 5;
    double x[1] = {0.578495};

	double *xx = jacobiP(N,alpha,beta,x);
	cout << xx[N] << endl;
	exit(0);*/

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
    int order = A->p_order;
    prodim    = A->prodim;

    //std::cout << "problem dim = " << prodim << endl;

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

    if (rank == rank_v)
        cout << "*******************************Finish mesh_info ********************************" << endl;

    g2umap(order, filename_g2u, g2u_all, map_all, comm);

    if (rank == rank_v)
        cout << "*******************************Finish g2u ********************************" << endl;

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
    set_P_from_mesh(order, map, P_temp, comm, g2u_all, map_all);//, Rp);

    bdydof = next_bdydof;

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

    vector<cooEntry_row> Pent;
    par::sampleSort(P_temp, Pent, P->split, comm);

//    std::sort(P_temp.begin(), P_temp.end(), row_major);
//    Pent = P_temp;

    //print_vector(Pent, -1, "Pent", comm);

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
    for(int i = 0; i < P->entry.size(); ++i){
        assert(P->entry[i].row < P->M);
        assert(P->entry[i].col < P->Nbig);
        assert(fabs(P->entry[i].val) > ALMOST_ZERO);
    }

    if(verbose_coarsen) {
        MPI_Barrier(comm);
        std::stringstream buffer;
        buffer << "rank " << rank << ": P: " << "M = " << P->M << ", Mbig = " << P->Mbig << ", Nbig = " << P->Nbig
               << ", nnz_l = " << P->nnz_l << ", nnz_g = " << P->nnz_g;
        cout << buffer.str() << endl;
        MPI_Barrier(comm);
        printf("coarsen: step 6: rank = %d\n", rank);
        MPI_Barrier(comm);
//        print_vector(P->splitNew, 0, "P->splitNew", comm);
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

vector<int> saena_object::next_p_level_new2(vector<int> ind_fine, int order, int *type_){
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
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int type;
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
        if (rank == rank_v)
            std::cout << "element type is not implemented!" << std::endl;
    }

	//cout << "type: " << type << endl;
    //cout << 6+9*(order-1)+3*(order-1)*(order-1)+(order-1)*(order-2)+(order-1)*(order-1)*(order-2)/2 << " " << vert_size << endl;
    //cout << "element type = " << type << endl;
    vector<int> indices;
    if (type == 1)
    {

        for (int i=0; i<=order/2; i++)
        {
            for (int j=0; j<=order/2; j++)
            {
                indices.push_back(ind_fine[(order+1)*i+j]);
            }
        }
    }

    if (type == 3)
    {
        for (int i=0; i<=order/2; i++)
        {
            for (int j=0; j<=order/2; j++)
            {
                for (int k=0; k<=order/2; k++)
                {
                    indices.push_back(ind_fine[(order+1)*(order+1)*i+(order+1)*j+k]);
                }
            }
        }
    }

	// hard coded for now ...
	//cout << "===============" << endl;
    if (type == 0)
    {
		for (int i=0; i<=order/2; i++)
		{
			for (int j=0; j<=order/2-i; j++)
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
                for (int k=0; k<=order-i-j; k++)
                {
                    //cout << sum_i+sum_j+k << endl;
                    if (i<=order/2 && j<=order/2-i && k<= order/2-i-j)
                        indices.push_back(ind_fine[counter]);

                    counter ++;
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
                    if (i<=order/2 && j<=order/2 && k<= order/2-i)
                        indices.push_back(ind_fine[counter]);

                    counter ++;
                }
            }
        }
    }

    if (type_ != NULL)
        *type_ = type;

    return indices;
}

vector<int> saena_object::next_p_level_new(vector<int> ind_fine, int order, int *type_){
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
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int type;
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
        if (rank == rank_v)
            std::cout << "element type is not implemented!" << std::endl;
    }

    //cout << 6+9*(order-1)+3*(order-1)*(order-1)+(order-1)*(order-2)+(order-1)*(order-1)*(order-2)/2 << " " << vert_size << endl;
    //cout << "element type = " << type << endl;
    vector<int> indices;
    comp_ordering_map(type, order);
    vector<int> ind;
    if (type == 1){
        /*for (int i=0; i<=order; i++){
            for (int j=0; j<=order; j++){
#ifdef __DEBUG1__
//            ASSERT((2*j+2*(order+1)*i >= 0) && (2*j+2*(order+1)*i < ind_fine.size()),
//                   i << "\t" << j << "\t" << order << "\t" << 2*j+2*(order+1)*i);
//            cout << i << "\t" << j << "\t" << order << "\t" << ind_fine.size() << "\t" << 2*j+2*(order+1)*i << "\t" << ind_fine[2*j+2*(order+1)*i] << endl;
#endif
				if (i==0 || i==1 || i==3 || i==5 || i==7)
				{
					if (j==0 || j==1 || j==3 || j==5 || j==7)
					{
                		//indices.push_back(ind_fine[j+(order+1)*i]);
						cout << j+(order+1)*i << endl;

					}
				}
            }
        }
		cout << "*******" << endl;*/

        for (int i=0; i<=order; i++)
        {
            for (int j=0; j<=order; j++)
            {
                if (i%2 == 0 && j%2 == 0)
                    ind.push_back(ordering_map[(order+1)*i+j]);
            }
        }
    }

    if (type == 3)
    {
        /*for (int i=0; i<=order; i++){
            for (int j=0; j<=order; j++){
                for (int k=0; k<=order; k++){
					if (i==0 || i==1 || i==3 || i==5 || i==7)
						if (j==0 || j==1 || j==3 || j==5 || j==7)
							if (k==0 || k==1 || k==3 || k==5 || k==7)
                    			//indices.push_back(ind_fine[k+(order+1)*j+(order+1)*(order+1)*i]);
								cout << k+(order+1)*j+(order+1)*(order+1)*i << " ";
                }
            }
        }
		cout << endl;
		cout << "*******" << endl;*/
        for (int i=0; i<=order; i++)
        {
            for (int j=0; j<=order; j++)
            {
                for (int k=0; k<=order; k++)
                {
                    if (i%2 == 0 && j%2 == 0 && k%2 == 0)
                        ind.push_back(ordering_map[(order+1)*(order+1)*i+(order+1)*j+k]);
                }
            }
        }
    }

    // hard coded for now ...
    if (type == 0)
    {
        /*if (order == 2)
        {
            indices.push_back(ind_fine[0]);
            indices.push_back(ind_fine[1]);
            indices.push_back(ind_fine[3]);
		}
        if (order == 4)
        {
            indices.push_back(ind_fine[0]);
            indices.push_back(ind_fine[1]);
            indices.push_back(ind_fine[3]);
            indices.push_back(ind_fine[5]);
            indices.push_back(ind_fine[7]);
            indices.push_back(ind_fine[12]);
		}
        if (order == 8)
        {
            indices.push_back(ind_fine[0]);
            indices.push_back(ind_fine[1]);
            indices.push_back(ind_fine[3]);
            indices.push_back(ind_fine[5]);
            indices.push_back(ind_fine[7]);
            indices.push_back(ind_fine[9]);
            indices.push_back(ind_fine[11]);
            indices.push_back(ind_fine[13]);
            indices.push_back(ind_fine[15]);
            indices.push_back(ind_fine[24]);
            indices.push_back(ind_fine[26]);
            indices.push_back(ind_fine[28]);
            indices.push_back(ind_fine[35]);
            indices.push_back(ind_fine[37]);
            indices.push_back(ind_fine[42]);
		}*/
        for (int i=0; i<=order; i++)
        {
            for (int j=0; j<=order-i; j++)
            {
                //cout << (2*order+3-i)*i/2+j << endl;
                if (i%2 == 0 && j%2 == 0)
                {
                    ind.push_back(ordering_map[(2*order+3-i)*i/2+j]);
                }
            }
        }
    }
    if (type == 2)
    {
        /*if (order == 2)
        {
            indices.push_back(ind_fine[0]);
            indices.push_back(ind_fine[1]);
            indices.push_back(ind_fine[3]);
            indices.push_back(ind_fine[6]);
		}
        if (order == 4)
        {
            indices.push_back(ind_fine[0]);
            indices.push_back(ind_fine[1]);
            indices.push_back(ind_fine[3]);
            indices.push_back(ind_fine[5]);
            indices.push_back(ind_fine[7]);
            indices.push_back(ind_fine[12]);
            indices.push_back(ind_fine[15]);
            indices.push_back(ind_fine[17]);
            indices.push_back(ind_fine[22]);
            indices.push_back(ind_fine[28]);
		}
		if (order == 8)
		{
		}*/
        vector<int> ii;
        ii.push_back(0);
        for (int i=0; i<=order; i++)
        {
            int sum_i = accumulate(ii.begin(), ii.end(), 0);
            vector<int> jj;
            jj.push_back(0);
            for (int j=0; j<=order-i; j++)
            {
                int sum_j = accumulate(jj.begin(), jj.end(), 0);
                for (int k=0; k<=order-i-j; k++)
                {
                    //cout << sum_i+sum_j+k << endl;
                    if (i%2 == 0 && j%2 == 0 && k%2 == 0)
                        ind.push_back(ordering_map[sum_i+sum_j+k]);
                }
                jj.push_back(order+1-i-j);
            }
            ii.push_back((order+2-i)*(order+1-i)/2);
        }

    }
    if (type == 4)
    {
        if (order == 2)
        {
            indices.push_back(ind_fine[0]);
            indices.push_back(ind_fine[1]);
            indices.push_back(ind_fine[3]);
            indices.push_back(ind_fine[4]);
            indices.push_back(ind_fine[9]);
            indices.push_back(ind_fine[11]);
        }
        if (order == 4)
        {
        }
    }
    sort(ind.begin(), ind.end());
    for (int i=0; i<ind.size();i++)
    {
        cout << ind[i] << endl;
        indices.push_back(ind_fine[ind[i]]);
    }
    //exit(0);

    if (type_ != NULL)
        *type_ = type;
    return indices;
}

vector<int> saena_object::coarse_p_node_arr(vector< vector<int> > map, int order)
{
    int total_elem = map.size();
    vector<int> ind;
    for (int i=0; i<total_elem; i++)
    {
		//cout << i << endl;
        vector<int> ind_coarse = next_p_level_new2(map.at(i), order);
        for (int j=0; j<ind_coarse.size(); j++)
        {
            if (!ismember(ind_coarse.at(j), ind))
                ind.push_back(ind_coarse.at(j));
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

    //cout << univ_nodeno_fine << endl;
    //MPI_Barrier(MPI_COMM_WORLD);
    //exit(0);
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

    if (rank == rank_v)
    {
        cout << "order = " << order << endl;
        cout << "bdy node # = " << bdydof << endl;
        cout << "next bdy node # = " << next_bdydof << endl;
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
        // for each element extract coraser element indices
        //vector<int> ind_coarse = next_p_level(map.at(i), order);
        int elem_type;
        vector<int> ind_coarse = next_p_level_new2(map.at(i), order, &elem_type);
        //vector<int> ind_coarse = next_p_level(map.at(i), order);

        //std::cout << "ind_coarse size: " << ind_coarse.size() << std::endl;
        //std::cout << "map.at(i) size: " << map.at(i).size() << std::endl;
        /*if (order == 2)
        {
            for (int ii=0; ii<ind_coarse.size(); ii++)
                std::cout << ind_coarse.at(ii) << " ";
            std::cout << std::endl;
        }*/
        //cout << ind_coarse.size() << "\n";

        /*bool flag = true;
        for (int j=0; j<ind_coarse.size(); j++){
            // skip bdy node
            if (ind_coarse.at(j)-1 < bdydof)
            {
                flag = false;
                break;
            }
        }*/
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
            /*if (rank == rank_v && ind_coarse[j] == 42)
                cout << "P_col = " << P_col << std::endl;*/

            //cout << P_col << " " << tmp_col-1-next_bdydof<< endl;
            // index corase level dof
            // value universal value at this level
            //g2u_map.at(P_col) = g2u[ind_coarse.at(j)-1-bdydof];
            //cout << P_col << " " << g2u_map.at(P_col) << endl;
            // assuming the map ordering (connectivity) is the same as ref element
            // shared nodes between elements should have the same values
            // when evaluated in each of the elememnts
            //cout << "val size " << val.size() << endl;


            // set P entries
            // This is for nodal basis interpolation
            /*  for (int k=0; k<val.size(); k++){
                  if (map.at(i).at(k)-1 < bdydof)
                      continue;

                  // row: g2u.at(map.at(i).at(k)-1-bdydof)
                  // col: P_col
                  // universal col id: g2u_map[P_col]
                  // row is universal
                  if(fabs(val[k]) > 1e-14){
                      //cout << "HERE--------------" << endl;
                      // need a map of k to convert default ordering to modal ordering in nektar
                      int k_new = ordering_map[k];
                      P_temp.emplace_back(g2u_f.at(map.at(i).at(k_new)-1-bdydof), g2u_c.at(P_col), val[k]);
                      //P_temp.emplace_back(g2u.at(map.at(i).at(k)-1-bdydof), P_col, val[k]);
                      max_row = max(max_row, map.at(i).at(k)-1-bdydof);
                      max_col = max(max_col, P_col);
                  }

  //                Pp_loc.at(g2u.at(map.at(i).at(k)-1-bdydof)).at(P_col) = val.at(k);
                  //if (rank == rank_v && ind_coarse[j] == 42)
                    //  cout << "row = " << g2u.at(map.at(i).at(k)-1-bdydof) << ", and value = " << val[k] << "\n";

              }*/

            //cout << ind_coarse.at(j)-1-bdydof << " " << P_col << endl;
            P_temp.emplace_back(g2u_f.at(ind_coarse.at(j)-1-bdydof), g2u_c.at(P_col), 1.0);
            /*if (rank == rank_v && ind_coarse[j] == 42)
                cout << "\n";*/
            /*if (rank == rank_v)
                cout << endl;*/
            // For nektar which removes dirichlet bc in matrix
            //if (ind_coarse.at(j) < bdydof && !ismember(P_col, skip))
            //skip.push_back(P_col);
        }
        //if (flag)
        //exit(0);
    }
    //if (rank == rank_v)
    //cout << "max row and col = " << max_row << " " << max_col << endl;
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


void saena_object::set_PR_from_p(int order, vector< vector<int> > map, vector< vector<double> > &Pp)//, vector< vector<double> > &Rp)
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
            vector<double> val = get_interpolation(j+1,order);
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
    if (!rank)
    {
        cout << "\n===============================" << endl;
        cout << "create mesh info for this level" << endl;
        cout << "===============================" << endl;
    }
    vector <vector<int> > map = map_all.at(map_all.size()-1);
    if (map_all.size() == 1) {
        //map = map_all[0];
        elemno = map_all[0].size();
#if 0
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
#endif
    }

    if (order > 1)
    {
        // coarse_node_ind index is coraser mesh node index
        // coarse_node_ind value is finer mesh node index

        vector<int> coarse_node_ind = coarse_p_node_arr(map, order);
        sort(coarse_node_ind.begin(), coarse_node_ind.end());

        vector <vector<int> > map_next(elemno);
        for (int i = 0; i < elemno; ++i){
            vector<int> aline = map.at(i);
			//cout << i << endl;
            vector<int> ind_coarse = next_p_level_new2(aline, order);
            for (int j = 0; j < ind_coarse.size(); ++j){
                int mapped_val = findloc(coarse_node_ind, ind_coarse.at(j));
                map_next.at(i).emplace_back(mapped_val+1);
            }
        }
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
        if (rank == rank_v){
            cout << "elem # = " << elemno << endl;
            cout << "bdydof = " << bdydof << endl;
            cout << "current fine node # = " << nodeno_fine << ", and next coarse node # = " << nodeno_coarse << endl;
        }
    }
    else {
        if (rank == rank_v){
            cout << "This is the p = 1 level, and NO next coarse p level is computed" << endl;
        }
    }
    //std::cout << map_all.size() << " " << map_all.at(map_all.size()-1).size() << " " << map_all.at(map_all.size()-1).at(0).size() << std::endl;

    return map;

}

//this is the function as mesh info for test for now
void saena_object::g2umap(int order, string filename, vector< vector<int> > &g2u_all, vector< vector< vector<int> > > &map_all, MPI_Comm comm)
{

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    if (!rank)
    {
        cout << "\n===============================" << endl;
        cout << "create g2u for this level" << endl;
        cout << "===============================" << endl;
    }
    //if (g2u_all.size() == 1) {

    // assume pure quad elememt for now
    /*ifstream ifs;
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
    iss.clear();*/
    /*for(int k=0; k<g2u.size()/2;k++)
    {
        std::cout << g2u.at(2*k) << " " << g2u.at(2*k+1);
    }
    std::cout << "\n";
    exit(0);*/
    // } else {

    // entry value is based on finer node
    vector <int> g2u_next_fine_node;
    // coarse_node_ind index is coraser mesh node index
    // coarse_node_ind value is finer mesh node index
    vector< vector<int> > map = map_all.at(map_all.size()-2);
    vector< vector<int> > map_c = map_all.at(map_all.size()-1);

    vector<int> coarse_node_ind = coarse_p_node_arr(map, order);
    sort(coarse_node_ind.begin(), coarse_node_ind.end());
    next_bdydof = 0;
    for (int i = 0; i < coarse_node_ind.size(); ++i)
    {
        // get universal dof value
        // this value will go to next level
        if (coarse_node_ind.at(i)-1 <bdydof)
            next_bdydof ++;
        else
            g2u_next_fine_node.push_back(g2u_all.at(g2u_all.size()-1).at(coarse_node_ind.at(i)-1-bdydof));
    }
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

    //cout << g2u_univ.size() << endl;
    //MPI_Barrier(MPI_COMM_WORLD);
    //exit(0);
    // entry value is based on coarser node

    /*vector <int> g2u_next_coarse_node;
    for (int i=0; i<coarse_node_ind.size(); i++)
    {
        // if it is boundary node, it has no conterpart in g2u
        // bdydof will be at the begining since coarse_node_ind is sorted
        if (coarse_node_ind[i] -1 < bdydof)
            continue;
        int g2u_next_val = findloc(g2u_univ, g2u_all.at(g2u_all.size()-1).at(coarse_node_ind[i]-1-bdydof));
        // relate i-bdydof (global dof in next level) and next_g2u_val (universal dof in next level)
        g2u_next_coarse_node.push_back(g2u_next_val);
    }*/

    //cout << g2u_next_coarse_node.size() << endl;
    //MPI_Barrier(MPI_COMM_WORLD);
    //exit(0);
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


    g2u_all.push_back( vector<int> ());
    for (int i=0; i<g2u_next_coarse_node.size(); i++)
    {
        g2u_all.at(g2u_all.size()-1).push_back(g2u_next_coarse_node.at(i));
    }
    //std::cout << map_all.size() << " " << map_all.at(map_all.size()-1).size() << " " << map_all.at(map_all.size()-1).at(0).size() << std::endl;

}

// TODO hard coded
// assuming 2d for now ...
vector<double> saena_object::get_interpolation(int ind, int order){

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
                double tmp = table.at(x_dir).at(j)*table.at(y_dir).at(i);
                val.push_back(tmp);
            }
        }

    }

    return val;
}

// generalized version of get_interpolation
vector<double> saena_object::get_interpolation_new(int ind, int order, int type){

    // tensor product from 1d interpolant
    vector<double> val;//((order+1)*(order+1));
    //quad and hex
    //use tensor product
    if (type == 1 || type == 3)
    {
        vector< vector<double> > table = one_interp_P(order);
        if (prodim == 2)
        {
            int x_dir = (ind-1) % (order/2+1);
            int y_dir = (ind-1) / (order/2+1);
            for (int i=0; i<order+1; i++)
            {
                for (int j=0; j<order+1; j++)
                {
                    double tmp = table.at(x_dir).at(j)*table.at(y_dir).at(i);
                    val.push_back(tmp);
                }
            }
        }
        else if (prodim == 3)
        {
            // for a 2->1 example
            // id: 0 1 2 3 4 5 6 7
            //  x: 0 1 0 1 0 1 0 1
            //  y: 0 0 1 1 0 0 1 1
            //  z: 0 0 0 0 1 1 1 1

            int x_dir = (ind-1) % (order/2+1);
            int y_dir = ((ind-1) % ((order/2+1)*(order/2+1))) / (order/2+1);
            int z_dir = (ind-1) / ((order/2+1)*(order/2+1));
            for (int i=0; i<order+1; i++)
            {
                for (int j=0; j<order+1; j++)
                {
                    for (int k=0; k<order+1; k++)
                    {
                        double tmp = table.at(x_dir).at(k)*table.at(y_dir).at(j)*table.at(z_dir).at(i);
                        val.push_back(tmp);
                    }
                }
            }
        }
    }
        // tri and tet
        // use natrual coordinate
    else if (type == 2 || type == 4)
    {
        if (prodim == 2)
        {
            if (order == 8)
            {
                //order 4 basis
                for (int i=0; i<(order+1)*(order+2)/2; i++)
                {
                    std::vector<double> L = comp_L(i, order);
                    double tmp ;
                    // vertex
                    if (ind < 4)
                    {
                        tmp = pow(order/2, order/2-1)/factorial(order/2-1)*L[ind-1]*(L[ind-1]-(1/4))*(L[ind-1]-(2/4))*(L[ind-1]-(3/4));
                    }
                        // edge
                    else if (ind < 4+3*(order-1))
                    {
                        if (ind == 4)
                            tmp = 1.0/((3./4)*(1./4)*(3./4-1./4)*(3./4-2./4))*L[0]*L[1]*(L[0]-(1./4))*(L[0]-(2./4));
                        if (ind == 5)
                            tmp = 1.0/((2./4)*(2./4)*(2./4-1./4)*(2./4-3./4))*L[0]*L[1]*(L[0]-(1./4))*(L[0]-(3./4));
                        if (ind == 6)
                            tmp = 1.0/((1./4)*(3./4)*(1./4-2./4)*(1./4-3./4))*L[0]*L[1]*(L[0]-(2./4))*(L[0]-(3./4));

                        if (ind == 7)
                            tmp = 1.0/((3./4)*(1./4)*(3./4-1./4)*(3./4-2./4))*L[1]*L[2]*(L[1]-(1./4))*(L[1]-(2./4));
                        if (ind == 8)
                            tmp = 1.0/((2./4)*(2./4)*(2./4-1./4)*(2./4-3./4))*L[1]*L[2]*(L[1]-(1./4))*(L[1]-(3./4));
                        if (ind == 9)
                            tmp = 1.0/((1./4)*(3./4)*(1./4-2./4)*(1./4-3./4))*L[1]*L[2]*(L[1]-(2./4))*(L[1]-(3./4));

                        if (ind == 10)
                            tmp = 1.0/((3./4)*(1./4)*(3./4-1./4)*(3./4-2./4))*L[2]*L[0]*(L[2]-(1./4))*(L[2]-(2./4));
                        if (ind == 11)
                            tmp = 1.0/((2./4)*(2./4)*(2./4-1./4)*(2./4-3./4))*L[2]*L[0]*(L[2]-(1./4))*(L[2]-(3./4));
                        if (ind == 12)
                            tmp = 1.0/((1./4)*(3./4)*(1./4-2./4)*(1./4-3./4))*L[2]*L[0]*(L[2]-(2./4))*(L[2]-(3./4));

                    }
                    else
                    {
                        // TODO
                    }
                    val.push_back(tmp);
                }
            }
            //order 2 basis
            if (order == 4)
            {
                for (int i=0; i<(order+1)*(order+2)/2; i++)
                {
                    std::vector<double> L = comp_L(i, order);
                    double tmp ;
                    // vertex
                    if (ind < 4)
                    {
                        tmp = pow(order/2, order/2-1)/factorial(order/2-1)*L[ind-1]*(L[ind-1]-(1/2));
                    }
                        // edge
                    else if (ind < 4+3*(order-1))
                    {
                        if (ind == 4)
                            tmp = 1.0/((1./2)*(1./2))*L[0]*L[1];
                        if (ind == 5)
                            tmp = 1.0/((1./2)*(1./2))*L[1]*L[2];
                        if (ind == 6)
                            tmp = 1.0/((1./2)*(1./2))*L[2]*L[0];
                    }
                    val.push_back(tmp);
                }
            }
            //order 1 basis
            if (order == 2)
            {
                for (int i=0; i<(order+1)*(order+2)/2; i++)
                {
                    std::vector<double> L = comp_L(i,order);
                    double tmp ;
                    // vertex
                    if (ind < 4)
                    {
                        tmp = pow(order/2, order/2-1)/factorial(order/2-1)*L[ind-1];
                    }
                    val.push_back(tmp);
                }
            }
        }
        if (prodim == 3)
        {

            // TODO tet
        }
    }

    return val;
}

inline std::vector<double> saena_object::comp_L(int i, int order)
{
    std::vector<double> L;
    if (prodim == 2)
    {
        L.resize(3,0);
        // vertex
        if (i < 4)
        {
            L[i-1] = 1.0;
        }
            // edge
        else if (i < 4+(order-1))
        {
            L[0] = 1./order*(4+(order-1)-i);
            L[1] = 1./order*(i-3);
            L[2] = 0.;
        }
        else if (i < 4+2*(order-1))
        {
            L[1] = 1./order*(4+(order-1)-i);
            L[2] = 1./order*(i-3);
            L[0] = 0.;
        }
        else if (i < 4+3*(order-1))
        {
            L[2] = 1./order*(4+(order-1)-i);
            L[0] = 1./order*(i-3);
            L[1] = 0.;
        }
            // interior
            // only for order 8
        else
        {
            // TODO
        }
    }
    else
    {
        L.resize(4,0);
        // vertex
        if (i < 5)
        {
            L[i-1] = 1.0;
        }
            // edge
        else if (i < 5+(order-1))
        {
            L[0] = 1./order*(4+(order-1)-i);
            L[1] = 1./order*(i-3);
            L[2] = 0.;
            L[3] = 0.;
        }
        else if (i < 5+2*(order-1))
        {
            L[1] = 1./order*(4+(order-1)-i);
            L[2] = 1./order*(i-3);
            L[0] = 0.;
            L[3] = 0.;
        }
        else if (i < 5+3*(order-1))
        {
            L[2] = 1./order*(4+(order-1)-i);
            L[0] = 1./order*(i-3);
            L[1] = 0.;
            L[3] = 0.;
        }
        else if (i < 5+4*(order-1))
        {
            L[3] = 1./order*(4+(order-1)-i);
            L[0] = 1./order*(i-3);
            L[1] = 0.;
            L[2] = 0.;
        }
        else if (i < 5+5*(order-1))
        {
            L[3] = 1./order*(4+(order-1)-i);
            L[1] = 1./order*(i-3);
            L[0] = 0.;
            L[2] = 0.;
        }
        else if (i < 5+6*(order-1))
        {
            L[3] = 1./order*(4+(order-1)-i);
            L[2] = 1./order*(i-3);
            L[0] = 0.;
            L[1] = 0.;
        }
            // face
        else if (i < 5+6*(order-1)+(order-2)*(order-1)/2)
        {
            L[0] = 1./order*(4+(order-1)-i);
            L[1] = 1./order*(i-3);
            L[2] = 0.;
            L[3] = 0.;
        }
    }

    return L;
}


inline int saena_object::factorial(int n)
{
    if (n < 0)
    {
        std::cout << "wrong value to perform factorial operation" << std::endl;
        return 0;
    }

    if (n == 0)
    {
        return 1;
    }

    int prod=n;
    while (n>1)
    {
        prod *= n-1;
        n --;
    }
    return prod;
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

// genralized version of eight_order
inline vector< vector<double> > saena_object::one_interp_P(int order)
{
    // this is 1d interpolation for later tensor product
    const vector<double> gl1{-1.0, 1.0};
    const vector<double> gl2{-1.0, 0, 1.0};
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
                {
                    if (order == 8)
                        val *= (gl8.at(j)-gl4.at(k))/(gl4.at(i)-gl4.at(k));
                    if (order == 4)
                        val *= (gl4.at(j)-gl2.at(k))/(gl2.at(i)-gl2.at(k));
                    if (order == 2)
                        val *= (gl2.at(j)-gl1.at(k))/(gl1.at(i)-gl1.at(k));
                }
            }
            table.at(i).push_back(val);
        }
    }
    return table;
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



inline double saena_object::phi_P(int type, int p, double z, int q, int r)
{
    double y;
    if (prodim == 2)
    {
        if (p == 0 && q == 1 && (type == 0))
            y = 1;
        else if (p == 0)
            y = (1-z)/2;
        else if (p == 1)
            y = (1+z)/2;
        else
        {
            double x[1]={z};
            double *jac = jacobiP(p-2,1,1,x);
            y = (1-z)/2*(1+z)/2*jac[p-2];
            delete [] jac;
        }
    }
    else
    {
        if (p == 0 && q == 0 && r == 1 && type == 2)
            y = 1;
        else if (p == 0 && q == 1 && type == 2)
            y = 1;
        else if (p == 0 && r == 1 &&  type == 4)
            y = 1;
        else if (p == 0)
            y = (1-z)/2;
        else if (p == 1)
            y = (1+z)/2;
        else
        {
            double x[1]={z};
            double *jac = jacobiP(p-2,1,1,x);
            y = (1-z)/2*(1+z)/2*jac[p-2];
            delete [] jac;
        }
    }
    return y;
}

inline double saena_object::phi_Pq(int type, int p, int q, double z, int r)
{
    double y;
    if (prodim == 3)
    {
        if (p == 0 && q==0 && r == 1 && type == 2)
            y = 1;
        else if (p == 0)
            y = phi_P(type, q,z);
        else if (1 <= p && q == 0)
            y = pow(((1-z)/2),p);
        else
        {
            double x[1]={z};
            double *jac = jacobiP(q-1,2*p-1,1,x);
            y = pow(((1-z)/2),p)*(1+z)/2*jac[q-1];
            delete [] jac;
        }
    }
    else
    {
        if (p == 0)
            y = phi_P(type, q,z);
        else if (1 <= p && q == 0)
            y = pow(((1-z)/2),p);
        else
        {
            double x[1]={z};
            double *jac = jacobiP(q-1,2*p-1,1,x);
            y = pow(((1-z)/2),p)*(1+z)/2*jac[q-1];
            delete [] jac;
        }
    }
    return y;
}

inline double saena_object::phi_Pqr(int type, int p, int q, int r, double z)
{
    double y;
    if (p == 0)
        y = phi_Pq(type, q,r,z);
    else if (q == 0)
        y = phi_Pq(type, p,r,z);
    else if (1 <= p && 1 <= q && r == 0)
        y = pow(((1-z)/2),(p+q));
    else
    {
        double x[1]={z};
        double *jac = jacobiP(r-1,2*p+2*q-1,1,x);
        y = pow(((1-z)/2),(p+q))*(1+z)/2*jac[r-1];
        delete [] jac;
    }
    return y;
}

inline double *saena_object::jacobiP(int n, double alpha, double beta, double x[], int m)
{
    double c1;
    double c2;
    double c3;
    double c4;
    int i;
    int j;
    double *v;

    if ( alpha <= -1.0 )
    {
        cerr << "\n";
        cerr << "J_POLYNOMIAL - Fatal error!\n";
        cerr << "  Illegal input value of ALPHA = " << alpha << "\n";
        cerr << "  But ALPHA must be greater than -1.\n";
        exit ( 1 );
    }

    if ( beta <= -1.0 )
    {
        cerr << "\n";
        cerr << "J_POLYNOMIAL - Fatal error!\n";
        cerr << "  Illegal input value of BETA = " << beta << "\n";
        cerr << "  But BETA must be greater than -1.\n";
        exit ( 1 );
    }

    if ( n < 0 )
    {
        return NULL;
    }

    v = new double[m*(n+1)];

    for ( i = 0; i < m; i++ )
    {
        v[i+0*m] = 1.0;
    }

    if ( n == 0 )
    {
        return v;
    }

    for ( i = 0; i < m; i++ )
    {
        v[i+1*m] = ( 1.0 + 0.5 * ( alpha + beta ) ) * x[i]
                   + 0.5 * ( alpha - beta );
    }

    for ( i = 0; i < m; i++ )
    {
        for ( j = 2; j <= n; j++ )
        {
            c1 = 2.0 * ( double ) ( j ) * ( ( double ) ( j ) + alpha + beta )
                 * ( ( double ) ( 2 * j - 2 ) + alpha + beta );

            c2 = ( ( double ) ( 2 * j - 1 ) + alpha + beta )
                 * ( ( double ) ( 2 * j ) + alpha + beta )
                 * ( ( double ) ( 2 * j - 2 ) + alpha + beta );

            c3 = ( ( double ) ( 2 * j - 1 ) + alpha + beta )
                 * ( alpha + beta ) * ( alpha - beta );

            c4 = - ( double ) ( 2 ) * ( ( double ) ( j - 1 ) + alpha )
                 * ( ( double ) ( j - 1 ) + beta )
                 * ( ( double ) ( 2 * j ) + alpha + beta );

            v[i+j*m] = ( ( c3 + c2 * x[i] ) * v[i+(j-1)*m] + c4 * v[i+(j-2)*m] ) / c1;
        }
    }

    return v;
}

std::vector<double> saena_object::gl_1d(int order)
{

    if (order == 2)
    {
        vector<double> gl2{-1.0, 0.0, 1.0};
        return gl2;
    }
    else if(order == 4)
    {
        vector<double> gl4{-1.0, -sqrt(3.0/7), 0.0, sqrt(3.0/7), 1.0};
        return gl4;
    }
    else if (order == 8)
    {
        vector<double> gl8{-1.0, -0.8997579954, -0.6771862795, -0.3631174638, 0.0, 0.3631174638, 0.6771862795, 0.8997579954, 1.0};
        return gl8;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cout << "invalid order, return empty vector" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    exit(0);
}


std::vector<double> saena_object::get_interpolation_new2(int ind, int order, int type, bool flag1)
{
    const vector<double> gl = gl_1d(order);
    vector<double> val;//((order+1)*(order+1));
    //tri=0,quad=1,tet=2,hex=3,prism=4
    //cout << "element type in interp = " << type << endl;
    switch (type)
    {
        case 0: // tri
        {
            bool flag = false;
            int p, q;
            int counter = 0;
            for (p=0; p<=order/2; p++)
            {
                for (q = 0 ; q<=order/2-p; q++)
                {
                    if (counter == ind)
                    {
                        flag = true;
                        break;
                    }
                    counter ++;
                }
                if (flag)
                    break;
            }
            for (int i=0; i<=order; i++)
            {
                for (int j=0; j<=order-i; j++)
                {
                    double endp = -gl[i];
                    double rr = gl[i];
                    double ss = (gl[j]-(-1))/2*(endp-(-1))+(-1);
                    double aa;
                    if (fabs(ss-1)<1e-14)
                        aa = -1;
                    else
                        aa  = 2*(1+rr)/(1-ss)-1;
                    double bb = ss;
                    double tmp = phi_P(type,p,aa,q)*phi_Pq(type,p,q,bb);
                    val.push_back(tmp);
                }
            }
            break;
        }
        case 1: // quad
        {
            bool flag = false;
            int p,q;
            int counter = 0;
            for (p=0; p<=order/2; p++)
            {
                for (q=0; q<=order/2; q++)
                {
                    if (counter == ind)
                    {
                        flag = true;
                        break;
                    }
                    counter ++;
                }
                if (flag)
                    break;
            }
            //cout << "order index: " << p << " " << q << endl;
            for (int i=0; i<=order; i++)
            {
                for (int j=0; j<=order; j++)
                {
                    double tmp = phi_P(type,p,gl[i])*phi_P(type,q,gl[j]);
                    val.push_back(tmp);
                }
            }
            break;
        }
        case 2: // tet
        {
            bool flag = false;
            int p, q, r;
            int counter = 0;
            for (p=0; p<=order/2; p++)
            {
                for (q=0; q<=order/2-p; q++)
                {
                    for (r=0; r<=order/2-p-q; r++)
                    {
                        if (counter == ind)
                        {
                            flag = true;
                            break;
                        }
                        counter ++;
                    }
                    if (flag)
                        break;
                }
                if (flag)
                    break;
            }

            for (int i=0; i<=order; i++)
            {
                for (int j=0; j<=order-i; j++)
                {
                    for (int k=0; k<=order-i-j; k++)
                    {
                        double endp1 = -gl[i];
                        double rr = gl[i];
                        double ss = (gl[j]-(-1))/2.0*(endp1-(-1))+(-1);
                        double endp2 = endp1-ss-1;
                        double tt = (gl[k]-(-1))/2.0*(endp2-(-1))+(-1);
                        double aa;
                        if (fabs(tt+ss)<1e-14)
                            aa = -1;
                        else
                            aa = 2*(1+rr)/(-ss-tt)-1;
                        double bb;
                        if (fabs(tt-1)<1e-14)
                            bb = -1;
                        else
                            bb = 2*(1+ss)/(1-tt)-1;
                        double cc = tt;
                        //cout << aa << " " << bb << " " << cc << endl;
                        double tmp = phi_P(type,p,aa,q,r)*phi_Pq(type,p,q,bb,r)*phi_Pqr(type,p,q,r,cc);
                        //if (flag1)
                        //cout << phi_P(type,p,aa,q,r) << " " << phi_P(type,p,aa,q,r) << " " << phi_Pqr(type,p,q,r,cc) << endl;
                        val.push_back(tmp);
                    }
                }
            }
            break;
        }
        case 3: // hex
        {
            bool flag = false;
            int p, q, r;
            int counter = 0;
            for (p=0; p<=order/2; p++)
            {
                for (q=0; q<=order/2; q++)
                {
                    for (r=0; r<=order/2; r++)
                    {
                        if (counter == ind)
                        {
                            flag = true;
                            break;
                        }
                        counter ++;
                    }
                    if (flag)
                        break;
                }
                if (flag)
                    break;
            }

            for (int i=0; i<=order; i++)
            {
                for (int j=0; j<=order; j++)
                {
                    for (int k=0; k<=order; k++)
                    {
                        double tmp = phi_P(type,p,gl[i])*phi_P(type,q,gl[j])*phi_P(type,r,gl[k]);
                        val.push_back(tmp);
                    }
                }
            }
            break;
        }
        case 4: // prism
        {
            bool flag = false;
            int p, q, r;
            int counter = 0;
            for (p=0; p<=order/2; p++)
            {
                for (q=0; q<=order/2; q++)
                {
                    for (r=0; r<=order/2-p; r++)
                    {
                        if (counter == ind)
                        {
                            flag = true;
                            break;
                        }
                        counter ++;
                    }
                    if (flag)
                        break;
                }
                if (flag)
                    break;
            }

            for (int i=0; i<=order; i++)
            {
                for (int j=0; j<=order; j++)
                {
                    for (int k=0; k<=order-i; k++)
                    {
                        double endp = gl[order-i];
                        double rr = gl[i];
                        double ss = gl[j];
                        double tt = (gl[k]-(-1))/2*(endp-(-1))+(-1);
                        double aa;
                        if (fabs(tt-1)<1e-14)
                            aa = -1;
                        else
                            aa  = 2*(1+rr)/(1-tt)-1;
                        double bb = ss;
                        double cc = tt;
                        double tmp = phi_P(type,p,aa,q,r)*phi_P(type,q,bb,r)*phi_Pq(type,p,r,cc);
                        val.push_back(tmp);
                    }
                }
            }
            break;
        }
        default:
        {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0)
            {
                std::cout << "invalid element type" << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            exit(0);
        }
    }
    return val;
}

void saena_object::comp_ordering_map(int type, int order)
{
    if (type == 0)
    {
        //vector<int> ind;
        int i, i_, j, j_, sum;
        for (i=0; i<=order; i++)
        {
            for (j=0; j<=order-i; j++)
            {
                if (i == 0)
                {

                    if (j==order)
                        j_ = 1;
                    else if (j!=0)
                        j_ = j+1;
                    else
                        j_ = j;

                    sum = j_;
                }
                else if (i < order)
                {
                    if (j < order -i)
                    {
                        sum = (2*order+3-i)*i/2+j+order+1-i;
                    }
                    else
                        sum = order+order+1-i;
                }
                else
                {
                    sum = order+order+1-i;
                }
                ordering_map.push_back(sum);
                //if (i%2 == 0 && j%2 == 0)
                //ind.push_back(sum);
            }
        }
        /*sort(ind.begin(), ind.end());
        for (int i=0; i<ind.size(); i++)
            cout << ind[i] << endl;
        */
    }
    if (type == 1)
    {
        int i, i_, j, j_, sum;
        for (i=0; i<=order; i++)
        {
            if (i==order)
                i_ = 1;
            else if (j!=0)
                i_ = i+1;
            else
                i_ = i;
            for (j=0; j<=order; j++)
            {
                if (j==order)
                    j_ = 1;
                else if (j!=0)
                    j_ = j+1;
                else
                    j_ = j;

                sum = (order+1)*i_+j_;
                ordering_map.push_back(sum);

            }
        }
        /*for (int i=0; i<ordering_map.size(); i++)
        {
            cout << ordering_map[i] << endl;
            cout << "********" << endl;
        }*/
    }
    if (type == 3)
    {
        int i, i_, j, j_, k,k_,sum;
        for (i=0; i<=order; i++)
        {
            if (i==order)
                i_ = 1;
            else if (j!=0)
                i_ = i+1;
            else
                i_ = i;
            for (j=0; j<=order; j++)
            {
                if (j==order)
                    j_ = 1;
                else if (j!=0)
                    j_ = j+1;
                else
                    j_ = j;

                for (k=0; k<=order; k++)
                {
                    if (k==order)
                        k_ = 1;
                    else if (k!=0)
                        k_ = k+1;
                    else
                        k_ = k;

                    sum = (order+1)*(order+1)*i_+(order+1)*j_+k_;
                    ordering_map.push_back(sum);

                }
            }
        }
    }
}
