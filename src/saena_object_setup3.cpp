#include <saena_object.h>
#include "saena_matrix.h"
#include "strength_matrix.h"
#include "prolong_matrix.h"
#include "restrict_matrix.h"
#include "grid.h"
#include "aux_functions.h"
#include <vector>
#include <cmath>
#include <stdio.h>

using namespace std;
// Assume the mesh info is the connectivity
// using a 2d vector for now
int saena_object::pcoarsen(/*Grid *grid*/){

    //saena_matrix    *A  = grid->A;
    //prolong_matrix  *P  = &grid->P;
    //restrict_matrix *R  = &grid->R;
    //saena_matrix    *Ac = &grid->Ac;

    //MPI_Comm comm = A->comm;
    //int nprocs, rank;
    //MPI_Comm_size(comm, &nprocs);
    //MPI_Comm_rank(comm, &rank);

	int order = 8;
	int a_elemno = 4;
	int prodim = 2;
	vector< vector<int> > map = connect(order, a_elemno, prodim);

	/*int row = map.size();
	int col = map[0].size();
	cout << "row: " << row << ", and col: " << col << "\n";
	for (int i=0; i<row; i++)
	{
		for (int j=0; j<col; j++)
			cout << map[i][j] << " ";

		cout << "\n";
	}*/

	vector< vector<double> > Pp, Rp;
	set_PR_from_p(order, a_elemno, map, prodim, Pp, Rp);
	
	int row = Pp.size();
	int col = Pp[0].size();
	cout << "row: " << row << ", and col: " << col << "\n";
	FILE *filename;
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

    return indices;
}

void saena_object::set_PR_from_p(int order, int a_elemno, vector< vector<int> > map, int prodim, vector< vector<double> > &Pp, vector< vector<double> > &Rp)
{
    int nodeno_fine = pow(a_elemno*order+1, prodim);
    int nodeno_coarse = pow(a_elemno*(order/2)+1, prodim);
    Pp.resize(nodeno_fine);
    for (int i = 0; i < nodeno_fine; i++)
        Pp[i].resize(nodeno_coarse);
    // coarse_node_ind index is coraser mesh node index
    // coarse_node_ind value is finer mesh node index
    vector<int> coarse_node_ind = coarse_p_node_arr(map, order);
    sort(coarse_node_ind.begin(), coarse_node_ind.end());

	//cout << Pp.size() << "\n";
    // loop over all elements
    int total_elem = map.size();
    for (int i=0; i<total_elem; i++)
    {
        // for each element extract coraser element indices
        vector<int> ind_coarse = next_p_level(map[i], order);

		//cout << ind_coarse.size() << "\n";
        for (int j=0; j<ind_coarse.size(); j++)
        {
            // interpolate basis function of j at all other nodes
            // in this element as the values of corresponding P entries
            // TODO only for 2d now upto basis order 4
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
        }
    }
    //Rp = transp(Pp);
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
