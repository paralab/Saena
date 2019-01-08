#include "combblas_functions.h"

int combblas_matmult_DoubleBuff(){

    int nprocs, myrank;
//    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
/*
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
*/
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


int combblas_GalerkinNew(){

//    MPI_Init(&argc, &argv);
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

//    if(argc < 5)
//    {
//        if(myrank == 0)
//        {
//            cout << "Usage: ./GalerkinNew <Matrix> <OffDiagonal> <Diagonal> <T(right hand side restriction matrix)>" << endl;
//            cout << "<Matrix> <OffDiagonal> <Diagonal> <T> are absolute addresses, and files should be in triples format" << endl;
//            cout << "Example: ./GalerkinNew TESTDATA/grid3d_k5.txt TESTDATA/offdiag_grid3d_k5.txt TESTDATA/diag_grid3d_k5.txt TESTDATA/restrict_T_grid3d_k5.txt" << endl;
//        }
//        MPI_Finalize();
//        return -1;
//    }
    {
        string Aname = "/home/majidrp/Dropbox/Projects/Saena/data/CombBLAS/grid3d_k5.txt";
        string Aoffd = "/home/majidrp/Dropbox/Projects/Saena/data/CombBLAS/offdiag_grid3d_k5.txt";
        string Adiag = "/home/majidrp/Dropbox/Projects/Saena/data/CombBLAS/diag_grid3d_k5.txt";
        string Tname = "/home/majidrp/Dropbox/Projects/Saena/data/CombBLAS/restrict_T_grid3d_k5.txt";

        // A = L+D
        // A*T = L*T + D*T;
        // S*(A*T) = S*L*T + S*D*T;
        ifstream inputD(Adiag.c_str());

        MPI_Barrier(MPI_COMM_WORLD);
        typedef PlusTimesSRing<double, double> PTDD;
        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );

        PSpMat<double>::MPI_DCCols A(fullWorld); // construct objects
        PSpMat<double>::MPI_DCCols L(fullWorld);
        PSpMat<double>::MPI_DCCols T(fullWorld);
        FullyDistVec<int,double> dvec(fullWorld);

        // For matrices, passing the file names as opposed to fstream objects
        A.ReadDistribute(Aname, 0);
        L.ReadDistribute(Aoffd, 0);
        T.ReadDistribute(Tname, 0);
        dvec.ReadDistribute(inputD,0);
        SpParHelper::Print("Data read\n");

        PSpMat<double>::MPI_DCCols S = T;
        S.Transpose();

        // force the calling of C's destructor; warm up instruction cache - also check correctness
        {
            PSpMat<double>::MPI_DCCols AT = PSpGEMM<PTDD>(A, T);
            PSpMat<double>::MPI_DCCols SAT = PSpGEMM<PTDD>(S, AT);

            PSpMat<double>::MPI_DCCols LT = PSpGEMM<PTDD>(L, T);
            PSpMat<double>::MPI_DCCols SLT = PSpGEMM<PTDD>(S, LT);
            PSpMat<double>::MPI_DCCols SD = S;
            SD.DimApply(Column, dvec, multiplies<double>());	// scale columns of S to get SD
            PSpMat<double>::MPI_DCCols SDT = PSpGEMM<PTDD>(SD, T);
            SLT += SDT;	// now this is SAT

            if(SLT == SAT)
            {
                SpParHelper::Print("Splitting approach is correct\n");
            }
            else
            {
                SpParHelper::Print("Error in splitting, go fix it\n");
                SLT.PrintInfo();
                SAT.PrintInfo();
                //SLT.SaveGathered("SLT.txt");
                //SAT.SaveGathered("SAT.txt");
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime(); 	// initilize (wall-clock) timer
        for(int i=0; i<ITERATIONS; i++)
        {
            PSpMat<double>::MPI_DCCols AT = PSpGEMM<PTDD>(A, T);
            PSpMat<double>::MPI_DCCols SAT = PSpGEMM<PTDD>(S, AT);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double t2 = MPI_Wtime();
        if(myrank == 0)
        {
            cout<<"Full restriction (without splitting) finished"<<endl;
            printf("%.6lf seconds elapsed per iteration\n", (t2-t1)/(double)ITERATIONS);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        t1 = MPI_Wtime(); 	// initilize (wall-clock) timer
        for(int i=0; i<ITERATIONS; i++)
        {

            PSpMat<double>::MPI_DCCols LT = PSpGEMM<PTDD>(L, T);
            PSpMat<double>::MPI_DCCols SLT = PSpGEMM<PTDD>(S, LT);
            PSpMat<double>::MPI_DCCols SD = S;
            SD.DimApply(Column, dvec, multiplies<double>());	// scale columns of S to get SD
            PSpMat<double>::MPI_DCCols SDT = PSpGEMM<PTDD>(SD, T);
            SLT += SDT;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        t2 = MPI_Wtime();
        if(myrank == 0)
        {
            cout<<"Full restriction (with splitting) finished"<<endl;
            printf("%.6lf seconds elapsed per iteration\n", (t2-t1)/(double)ITERATIONS);
        }
        inputD.clear();inputD.close();
    }

    return 0;
}
