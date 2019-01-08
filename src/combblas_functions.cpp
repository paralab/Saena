#include "combblas_functions.h"

int combblas_matmult_DoubleBuff(){

    int nprocs, myrank;
//    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    string Aname = "/home/majidrp/Dropbox/Projects/Saena/data/CombBLAS/rmat_scale16_A.mtx";
    string Bname = "/home/majidrp/Dropbox/Projects/Saena/data/CombBLAS/rmat_scale16_B.mtx";
    typedef PlusTimesSRing<ElementType, ElementType> PTDOUBLEDOUBLE;
    PSpMat<ElementType>::MPI_DCCols A, B;	// construct objects

    A.ReadDistribute(Aname, 0);
    A.PrintInfo();
    B.ReadDistribute(Bname, 0);
    B.PrintInfo();
    SpParHelper::Print("Data read\n");

    { // force the calling of C's destructor
        PSpMat<ElementType>::MPI_DCCols C = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE, ElementType, PSpMat<ElementType>::DCCols >(A, B);
        int64_t cnnz = C.getnnz();
        ostringstream tinfo;
        tinfo << "C has a total of " << cnnz << " nonzeros" << endl;
        SpParHelper::Print(tinfo.str());
        SpParHelper::Print("Warmed up for DoubleBuff\n");
        C.PrintInfo();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Pcontrol(1,"SpGEMM_DoubleBuff");
    double t1 = MPI_Wtime(); 	// initilize (wall-clock) timer
    for(int i=0; i<ITERATIONS; i++)
    {
        PSpMat<ElementType>::MPI_DCCols C = Mult_AnXBn_DoubleBuff<PTDOUBLEDOUBLE, ElementType, PSpMat<ElementType>::DCCols >(A, B);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = MPI_Wtime();
    MPI_Pcontrol(-1,"SpGEMM_DoubleBuff");
    if(myrank == 0)
    {
        cout<<"Double buffered multiplications finished"<<endl;
        printf("%.6lf seconds elapsed per iteration\n", (t2-t1)/(double)ITERATIONS);
    }

    return 0;
}


int combblas_matmult_Synch(){

    int nprocs, myrank;
//    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

    string Aname = "/home/majidrp/Dropbox/Projects/Saena/data/CombBLAS/rmat_scale16_A.mtx";
    string Bname = "/home/majidrp/Dropbox/Projects/Saena/data/CombBLAS/rmat_scale16_B.mtx";
    typedef PlusTimesSRing<ElementType, ElementType> PTDOUBLEDOUBLE;
    PSpMat<ElementType>::MPI_DCCols A, B;	// construct objects

    A.ReadDistribute(Aname, 0);
    A.PrintInfo();
    B.ReadDistribute(Bname, 0);
    B.PrintInfo();
    SpParHelper::Print("Data read\n");

    {// force the calling of C's destructor
        PSpMat<ElementType>::MPI_DCCols C = Mult_AnXBn_Synch<PTDOUBLEDOUBLE, ElementType, PSpMat<ElementType>::DCCols >(A, B);
        int64_t cnnz = C.getnnz();
        ostringstream tinfo;
        tinfo << "C has a total of " << cnnz << " nonzeros" << endl;
        SpParHelper::Print(tinfo.str());
        SpParHelper::Print("Warmed up for Synch\n");
        C.PrintInfo();
    }
    SpParHelper::Print("Warmed up for Synch\n");
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Pcontrol(1,"SpGEMM_Synch");
    double t1 = MPI_Wtime(); 	// initilize (wall-clock) timer
    for(int i=0; i<ITERATIONS; i++)
    {
        PSpMat<ElementType>::MPI_DCCols C = Mult_AnXBn_Synch<PTDOUBLEDOUBLE, ElementType, PSpMat<ElementType>::DCCols >(A, B);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Pcontrol(-1,"SpGEMM_Synch");
    double t2 = MPI_Wtime();
    if(myrank == 0)
    {
        cout<<"Synchronous multiplications finished"<<endl;
        printf("%.6lf seconds elapsed per iteration\n", (t2-t1)/(double)ITERATIONS);
    }

    return 0;
}
