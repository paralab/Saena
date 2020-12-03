#include "aux_functions2.h"

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

//    u = sin(2 * SAENA_PI * x) * sin(2 * SAENA_PI * y)
//    rhs = 8 * SAENA_PI * SAENA_PI * sin(2 * SAENA_PI * x) * sin(2 * SAENA_PI * y)

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
            rhs[iter++] = 8 * SAENA_PI * SAENA_PI * sin(2 * SAENA_PI * i * Hx) * sin(2 * SAENA_PI * j * Hy);
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
    const double TWOPI = 2 * SAENA_PI;

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

// NOTE: mx is the node number
// node index = 0,1,2,...,mx-1
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

//        HyHzdHx =  (Hy * Hz / Hx);
//        HxHzdHy =  (Hx * Hz / Hy);
//        HxHydHz =  (Hx * Hy / Hz);

        HyHzdHx = 1.0 / (Hx*Hx);
        HxHzdHy = 1.0 / (Hy*Hy);
        HxHydHz = 1.0 / (Hz*Hz);

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

//        printf("rank %d: xs = %d, ys = %d, zs = %d, xm = %d, ym = %d, zm = %d\n", rank, xs, ys, zs, xm, ym, zm);

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
                        A->set(node, node, 1.0);//2.0*(HxHydHz + HxHzdHy + HyHzdHx));
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
//    u = sin(2 * SAENA_PI * x) * sin(2 * SAENA_PI * y) * sin(2 * SAENA_PI * z)
//    rhs = 12 * SAENA_PI * SAENA_PI * sin(2 * SAENA_PI * x) * sin(2 * SAENA_PI * y) * sin(2 * SAENA_PI * z)

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

        const double TWOPI       = 2 * SAENA_PI;
        const double TWOELVEPISQ = 12 * SAENA_PI * SAENA_PI;

        rhs.resize(xm * ym * zm);
//        printf("rank %d: rhs.sz = %lu\n", rank, rhs.size());

        index_t iter = 0;
        for (k=zs; k<zs+zm; k++) {
            for (j=ys; j<ys+ym; j++) {
                for (i=xs; i<xs+xm; i++) {
                    rhs[iter++] = TWOELVEPISQ * sin(TWOPI * i * Hx) * sin(TWOPI * j * Hy) * sin(TWOPI * k * Hz);
                    //std::cout << "\n";
                    //std::cout << i << " " << j << " " << k << std::endl;
                    //std::cout << sin(TWOPI * i * Hx) << " " << sin(TWOPI * j * Hx) << " " << sin(TWOPI * k * Hx) << std::endl;
                    //std::cout << TWOELVEPISQ * sin(TWOPI * i * Hx) * sin(TWOPI * j * Hy) * sin(TWOPI * k * Hz) << " " << rhs[iter] << "\n";
                    //iter ++;
/*                    rhs[iter] = 12 * SAENA_PI * SAENA_PI
                                * sin(2 * SAENA_PI * (((value_t) i ) * Hx))
                                * sin(2 * SAENA_PI * (((value_t) j ) * Hy))
                                * sin(2 * SAENA_PI * (((value_t) k ) * Hz))
                                * Hx * Hy * Hz;
					iter ++;*/
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
                    tmp = u[iter++] - sin(2 * SAENA_PI * i * Hx) * sin(2 * SAENA_PI * j * Hy)  * sin(2 * SAENA_PI * k * Hz);
                    dif += tmp * tmp;
                }
            }
        }
    }

    double dif_tot = 0.0;
    MPI_Reduce(&dif, &dif_tot, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    if(!rank) cout << "\nnorm of diff = " << sqrt(dif / (mx*my*mz)) << endl;

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
                    rhs[iter] = 12 * SAENA_PI * SAENA_PI
                                * cos(2 * SAENA_PI * (((value_t) i + 0.5) * Hx))
                                * cos(2 * SAENA_PI * (((value_t) j + 0.5) * Hy))
                                * cos(2 * SAENA_PI * (((value_t) k + 0.5) * Hz))
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

//    rhs[iter] = 12 * SAENA_PI * SAENA_PI
//                * sin(2 * SAENA_PI * (((value_t) i + 0.5) * Hx))
//                * sin(2 * SAENA_PI * (((value_t) j + 0.5) * Hy))
//                * sin(2 * SAENA_PI * (((value_t) k + 0.5) * Hz))
//                * Hx * Hy * Hz;
    // sin(2 * SAENA_PI * i) * sin(2 * SAENA_PI * j)  * sin(2 * SAENA_PI * k)

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

#if 0
    saena_matrix *B = A.get_internal_matrix();

//    B->print_entry(-1);
//    std::sort(B->data_coo.begin(), B->data_coo.end());
//    B->print_entry(-1);

    B->entry.resize(B->data_coo.size());
    nnz_t iter = 0;
    for(const auto &i:B->data_coo){
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
#endif

//    A.assemble();
    A.assemble_band_matrix();

//    printf("rank %d: M = %u, Mbig = %u, nnz_l = %lu, nnz_g = %lu \n",
//            rank, A.get_num_local_rows(), A.get_num_rows(), A.get_local_nnz(), A.get_nnz());
//    if(!rank) printf("Mbig = %u, nnz_g = %lu, density = %.8f \n", A.get_num_rows(), A.get_nnz(), A.get_internal_matrix()->density);

    return 0;
}


int saena::random_symm_matrix(saena::matrix &A, index_t M, float density){
    // generates a random matrix with density of size "density".

    int rank = 0, nprocs = 0;
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

    index_t ii = 0, jj = 0;
    value_t vv = 0.0;

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

    int rank = 0, nprocs = 0;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    std::vector<index_t> split_temp(nprocs);
    split_temp[rank] = loc_size;
    MPI_Allgather(&loc_size,      1, par::Mpi_datatype<index_t>::value(),
                  &split_temp[0], 1, par::Mpi_datatype<index_t>::value(), comm);

    my_split = 0;
    for(index_t i = 0; i < rank; i++){
        my_split += split_temp[i];
    }

    return 0;
}
