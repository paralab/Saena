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

using namespace std;

// Assume the mesh info is the connectivity
// using a 2d vector for now
int saena_object::pcoarsen(Grid *grid, vector< vector< vector<int> > > &map_all){

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

	cout << "set up P" << endl;
	int order    = A->p_order;
	int a_elemno = 10;
	int elemno = 100;
	int prodim   = 2;
	string filename = "/home/songzhex/Desktop/Saena/data/nektar/nek_map_cont_4.txt";
	//vector< vector<int> > map = connect(order, a_elemno, prodim);
	vector< vector<int> > map = mesh_info(order, elemno, filename, map_all);
	
	int row_m = map.size();
	int col_m = map[0].size();
	cout << "map row: " << row_m << ", and col: " << col_m << "\n";
	/*for (int i=0; i<row; i++)
	{
		for (int j=0; j<col; j++)
			cout << map[i][j] << "\t";

		cout << "\n";
	}
	exit(0);*/

	vector< vector<double> > Pp;//, Rp;
	set_PR_from_p(order, a_elemno, map, prodim, Pp);//, Rp);

		

	int row_p = Pp.size();
	int col_p = Pp[0].size();
	cout << "Pp row: " << row_p << ", and col: " << col_p << "\n";
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

    P->comm  = A->comm;
    P->split = A->split;
	P->Mbig  = Pp.size();
	P->M     = P->Mbig;
	P->Nbig  = Pp[0].size();
	int iter = 0;

	//TODO: change for parallel
    P->splitNew.resize(nprocs+1);
    P->splitNew[0]      = 0;
    P->splitNew[nprocs] = P->Mbig;

    P->entry.clear();
    for(int i=0;i<Pp.size();i++)
	{
    	for(int j=0;j<Pp[0].size();j++)
		{
			if (fabs(Pp[i][j]) > 1e-12)
			{
        		P->entry.emplace_back(cooEntry(i, j, Pp[i][j]));
        		iter++;
			}
    	}
	}

    std::sort(P->entry.begin(), P->entry.end());

	P->nnz_l = iter;
    MPI_Allreduce(&P->nnz_l, &P->nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, P->comm);

    P->findLocalRemote();

    return 0;
}

vector<int> saena_object::next_p_level(vector<int> ind_fine, int order)
{
    // assuming the elemental ordering is like
    // 7--8--9
    // |  |  |
    // 4--5--6
    // |  |  |
    // 1--2--3  
    vector<int> indices;
    for (int i=0; i<order/2+1; i++)
        for (int j=0; j<order/2+1; j++)
            indices.push_back(ind_fine[2*j+2*(order+1)*i]);

	
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

void saena_object::set_PR_from_p(int order, int a_elemno, vector< vector<int> > map, int prodim, vector< vector<double> > &Pp)//, vector< vector<double> > &Rp)
{
	int bnd;
	// hard coded ...
	if (order == 2)
		bnd = 80;
	else if (order == 4)
		bnd = 160;
	else
		bnd = 320;

    int nodeno_fine = pow(a_elemno*order+1, prodim);
    int nodeno_coarse = pow(a_elemno*(order/2)+1, prodim);
    Pp.resize(nodeno_fine);
    for (int i = 0; i < nodeno_fine; i++)
        Pp[i].resize(nodeno_coarse);
    // coarse_node_ind index is coraser mesh node index
    // coarse_node_ind value is finer mesh node index
    vector<int> coarse_node_ind = coarse_p_node_arr(map, order);
    //sort(coarse_node_ind.begin(), coarse_node_ind.end());

	cout << map.size() << " " << map[0].size() << "\n";
    // loop over all elements
    int total_elem = map.size();

	vector<int> skip;
    for (int i=0; i<total_elem; i++)
    {
        // for each element extract coraser element indices
        vector<int> ind_coarse = next_p_level(map[i], order);
		for (int ii=0; ii<ind_coarse.size(); ii++)
			std::cout << ind_coarse[ii] << " ";
		std::cout << std::endl;
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
            int P_col = findloc(coarse_node_ind, ind_coarse[j]);
            // assuming the map ordering (connectivity) is the same as ref element
            // shared nodes between elements should have the same values
            // when evaluated in each of the elememnts
            for (int k=0; k<val.size(); k++)
			{
                Pp[map[i][k]-1][P_col] = val[k];
				//cout << val[k] << "\n";
			}

			// For nektar which removes dirichlet bc in matrix
			if (ind_coarse[j] < bnd && !ismember(P_col, skip))
				skip.push_back(P_col);
        }
    }

	cout << "skip size: " << skip.size() << "\n";
	for (int ii=0; ii<skip.size(); ii++)
            std::cout << skip[ii] << " ";
    std::cout << std::endl;

	int row_p = Pp.size();
    int col_p = Pp[0].size();
    cout << "inside set_PR_from_p, Pp row: " << row_p << ", and col: " << col_p << "\n";

	// Just to match nektar++ petsc matrix
	// assume all the Dirichlet BC nodes are at the begining
	// remove them from the matrix
	// remove columns
	for (int i=0; i<Pp.size(); i++)
	{
		int counter = 0;
		for (int j = 0; j<Pp[i].size();)
		{
			if (ismember(counter,skip))
				Pp[i].erase(Pp[i].begin()+j);
			else
				j++;
			
			counter ++;
		}
	}
	// remove rows
	Pp.erase(Pp.begin(), Pp.begin()+bnd);
    //Rp = transp(Pp);
	int row_after = Pp.size();
    int col_after = Pp[0].size();
    cout << "inside set_PR_from_p after remove, Pp row: " << row_after << ", and col: " << col_after << "\n";
}

inline vector< vector<double> > saena_object::transp(vector< vector<double> > M)
{
    int row = M.size();
    int col = M[0].size();
    vector< vector<double> > N(col, vector<double>(row));
    for (int i=0; i<col; i++)
        for (int j=0; j<row; j++)
            N[i][j] = M[j][i];

    return N;
}

//this is the function as mesh info for test for now
inline vector< vector<int> > saena_object::connect(int order, int a_elemno, int prodim)
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
                    map[k].push_back(st+m*a_nodeno+n+1);
        }
    }
    return map;
}

//this is the function as mesh info for test for now
inline vector< std::vector<int> > saena_object::mesh_info(int order, int elemno, string filename, vector< vector< vector<int> > > &map_all)
{
    vector <vector<int> > map(elemno);
    if (map_all.empty())
    {
        // assume pure quad elememt for now
        ifstream ifs;
        ifs.open(filename.c_str());
        istringstream iss;
        for (int i=0; i<elemno; i++)
        {
            string aLine;
            getline(ifs, aLine);
            iss.str(aLine);
            for (int j=0; j<(order+1)*(order+1); j++)
            {
                int val;
                iss >> val;
                map[i].push_back(val+1);
            }
            iss.clear();
        }
        ifs.close();
        iss.clear();
        ifs.clear();
    }
    else
    { 
        vector< vector<int> > map_pre = map_all[map_all.size()-1];
        // coarse_node_ind index is coraser mesh node index
        // coarse_node_ind value is finer mesh node index
        vector<int> coarse_node_ind = coarse_p_node_arr(map_pre, order*2);
        sort(coarse_node_ind.begin(), coarse_node_ind.end());
        
        for (int i=0; i<elemno; i++)
        {
            vector<int> aline = map_pre[i];
            vector<int> ind_coarse = next_p_level(aline, order*2);
            for (int j=0; j<ind_coarse.size(); j++)
            {
                int mapped_val = findloc(coarse_node_ind, ind_coarse[j]);
                map[i].push_back(mapped_val+1);
            }
        }
    }

	map_all.push_back(vector< vector<int> >());
	for (int i=0; i<map.size(); i++)
	{
		map_all[map_all.size()-1].push_back(vector<int>());
		for (int j=0; j<map[0].size(); j++)
		{
			map_all[map_all.size()-1][i].push_back(map[i][j]);
		}
	}

	std::cout << map_all.size() << " " << map_all[map_all.size()-1].size() << " " << map_all[map_all.size()-1][0].size() << std::endl;
    return map;
}

// TODO hard coded
// assuming 2d for now ...
vector<double> saena_object::get_interpolation(int ind, int order, int prodim)
{
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
            	coord[(order+1)*i+j].push_back(x);
            	coord[(order+1)*i+j].push_back(y);
			}
			else if (order == 4)
			{
				const double gl[5] = {-1.0, -sqrt(3.0/7), 0, sqrt(3.0/7), 1.0};
            	double x = gl[j];
            	double y = gl[i];
            	coord[(order+1)*i+j].push_back(x);
            	coord[(order+1)*i+j].push_back(y);
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
                        double tmp = (1-coord[k][0])/2*(1-coord[k][1])/2;
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
                        double tmp = (1+coord[k][0])/2*(1-coord[k][1])/2;
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
                        double tmp = (1-coord[k][0])/2*(1+coord[k][1])/2;
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
                        double tmp = (1+coord[k][0])/2*(1+coord[k][1])/2;
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
                        double tmp = (coord[k][0]-1)*coord[k][0]/2*
                                     (coord[k][1]-1)*coord[k][1]/2;
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
                        double tmp = (1-coord[k][0])*(coord[k][0]+1)*
                                     (coord[k][1]-1)*coord[k][1]/2;
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
                        double tmp = (1+coord[k][0])*coord[k][0]/2*
                                     (coord[k][1]-1)*coord[k][1]/2;
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
                        double tmp = (coord[k][0]-1)*coord[k][0]/2*
                                     (1-coord[k][1])*(1+coord[k][1]);
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
                        double tmp = (1-coord[k][0])*(1+coord[k][0])*
                                     (1-coord[k][1])*(1+coord[k][1]);
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
                        double tmp = (coord[k][0]+1)*coord[k][0]/2*
                                     (1-coord[k][1])*(1+coord[k][1]);
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
                        double tmp = (coord[k][0]-1)*coord[k][0]/2*
                                     (1+coord[k][1])*coord[k][1]/2;
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
                        double tmp = (1-coord[k][0])*(coord[k][0]+1)*
                                     (1+coord[k][1])*coord[k][1]/2;
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
                        double tmp = (coord[k][0]+1)*coord[k][0]/2*
                                     (1+coord[k][1])*coord[k][1]/2;
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
				double tmp = table[x_dir][j]*table[y_dir][i];
				val.push_back(tmp);
			}
		}

	}
	
	return val;
}

vector<int> saena_object::coarse_p_node_arr(vector< vector<int> > map, int order)
{
    int total_elem = map.size();
    vector<int> ind;
    for (int i=0; i<total_elem; i++)
    {
        vector<int> ind_coarse = next_p_level(map[i], order);
        for (int j=0; j<ind_coarse.size(); j++)
        {
            if (!ismember(ind_coarse[j], ind))
                ind.push_back(ind_coarse[j]);
        }
       
    }
    return ind;
} 

inline bool saena_object::ismember(int a, vector<int> arr)
{
    for (int i=0; i<arr.size(); i++)
    {
        if (a == arr[i])
            return true;
    }
    return false;
}

inline int saena_object::findloc(vector<int> arr, int a)
{
    for (int i=0; i<arr.size(); i++)
    {
        if (a == arr[i])
            return i;
    }
    
    cout << "coarse column is not in the fine mesh!!!" << endl;
    exit(0);
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
					val *= (gl8[j]-gl4[k])/(gl4[i]-gl4[k]);
			}
			table[i].push_back(val);
		}
	}
	return table;
}
