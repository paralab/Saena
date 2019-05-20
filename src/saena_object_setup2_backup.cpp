#include "saena_object.h"
#include "saena_matrix.h"
#include "strength_matrix.h"
#include "prolong_matrix.h"
#include "restrict_matrix.h"
#include "grid.h"
#include "aux_functions.h"
#include "parUtils.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <mpi.h>


int saena_object::compute_coarsen_old(Grid *grid){

    // todo: to improve the performance of this function, consider using the arrays used for RA also for RAP.
    // todo: this way allocating and freeing memory will be halved.

    saena_matrix *A    = grid->A;
    prolong_matrix *P  = &grid->P;
    restrict_matrix *R = &grid->R;
    saena_matrix *Ac   = &grid->Ac;

    MPI_Comm comm = A->comm;
//    Ac->active_old_comm = true;

//    int rank1, nprocs1;
//    MPI_Comm_size(comm, &nprocs1);
//    MPI_Comm_rank(comm, &rank1);
//    if(A->active_old_comm)
//        printf("rank = %d, nprocs = %d active\n", rank1, nprocs1);

    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(verbose_compute_coarsen){
        MPI_Barrier(comm);
        if(rank==0) printf("\nstart of compute_coarsen nprocs: %d \n", nprocs);
        MPI_Barrier(comm);
        printf("rank %d: A.Mbig = %u, A.M = %u, A.nnz_g = %lu, A.nnz_l = %lu \n", rank, A->Mbig, A->M, A->nnz_g, A->nnz_l);
        MPI_Barrier(comm);
        printf("rank %d: P.Mbig = %u, P.M = %u, P.nnz_g = %lu, P.nnz_l = %lu \n", rank, P->Mbig, P->M, P->nnz_g, P->nnz_l);
        MPI_Barrier(comm);
        printf("rank %d: R.Mbig = %u, R.M = %u, R.nnz_g = %lu, R.nnz_l = %lu \n", rank, R->Mbig, R->M, R->nnz_g, R->nnz_l);
        MPI_Barrier(comm);
    }

    prolong_matrix RA_temp(comm); // RA_temp is being used to remove duplicates while pushing back to RA.

    // ************************************* RA_temp - R on-diag and A local (on and off-diag) *************************************
    // Some local and remote elements of RA_temp are computed here using local R and local A.
    // Note: A local means whole entries of A on this process, not just the diagonal block.

    nnz_t AMaxNnz;
    index_t AMaxM;
    MPI_Allreduce(&A->nnz_l, &AMaxNnz, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);
    MPI_Allreduce(&A->M,     &AMaxM,   1, MPI_UNSIGNED,      MPI_MAX, comm);
//    MPI_Barrier(comm); printf("\nrank=%d, AMaxNnz=%d, AMaxM = %d \n", rank, AMaxNnz, AMaxM); MPI_Barrier(comm);

    // alloacted memory for AMaxM, instead of A.M to avoid reallocation of memory for when receiving data from other procs.
//    std::vector<unsigned int> AnnzPerRow(AMaxM, 0);
//    unsigned int *AnnzPerRow_p = &AnnzPerRow[0] - A->split[rank];
//    for(nnz_t i=0; i<A->nnz_l; i++)
//        AnnzPerRow_p[A->entry[i].row]++;

    std::vector<unsigned int> AnnzPerRow = A->nnzPerRow_local;
//    for(nnz_t i=0; i<A->nnz_l; i++)
//        AnnzPerRow[A->row_remote[i]]++;

    if(nprocs > 1) {
        for (nnz_t i = 0; i < A->M; i++)
            AnnzPerRow[i] += A->nnzPerRow_remote[i];
    }

//    print_vector(AnnzPerRow, -1, "AnnzPerRow", comm);

    // alloacted memory for AMaxM+1, instead of A.M+1 to avoid reallocation of memory for when receiving data from other procs.
    std::vector<unsigned long> AnnzPerRowScan(AMaxM+1);
    AnnzPerRowScan[0] = 0;
    for(index_t i=0; i<A->M; i++){
        AnnzPerRowScan[i+1] = AnnzPerRowScan[i] + AnnzPerRow[i];
//        if(rank==1) printf("i=%lu, AnnzPerRow=%d, AnnzPerRowScan = %d\n", i+A->split[rank], AnnzPerRow[i], AnnzPerRowScan[i+1]);
    }

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 1: rank = %d\n", rank); MPI_Barrier(comm);}

    // find row-wise ordering for A and save it in indicesP
    std::vector<nnz_t> indices_row_wise(A->nnz_l);
    for(nnz_t i=0; i<A->nnz_l; i++)
        indices_row_wise[i] = i;
    std::sort(&indices_row_wise[0], &indices_row_wise[A->nnz_l], sort_indices2(&A->entry[0]));

    nnz_t jstart, jend;
    if(!R->entry_local.empty()) {
        for (index_t i = 0; i < R->nnz_l_local; i++) {
            jstart = AnnzPerRowScan[R->entry_local[i].col - P->split[rank]];
            jend   = AnnzPerRowScan[R->entry_local[i].col - P->split[rank] + 1];
            if(jend - jstart == 0) continue;
            for (nnz_t j = jstart; j < jend; j++) {
//            if(rank==0) std::cout << A->entry[indicesP[j]].row << "\t" << A->entry[indicesP[j]].col << "\t" << A->entry[indicesP[j]].val
//                             << "\t" << R->entry_local[i].col << "\t" << R->entry_local[i].col - P->split[rank] << std::endl;
                RA_temp.entry.emplace_back(cooEntry(R->entry_local[i].row,
                                                    A->entry[indices_row_wise[j]].col,
                                                    R->entry_local[i].val * A->entry[indices_row_wise[j]].val));
            }
        }
    }

//    if(rank==0){
//        std::cout << "\nRA_temp.entry.size = " << RA_temp.entry.size() << std::endl;
//        for(i=0; i<RA_temp.entry.size(); i++)
//            std::cout << RA_temp.entry[i].row + R->splitNew[rank] << "\t" << RA_temp.entry[i].col << "\t" << RA_temp.entry[i].val << std::endl;}

//    printf("rank %d: RA_temp.entry.size_local = %lu \n", rank, RA_temp.entry.size());
//    MPI_Barrier(comm); printf("rank %d: local RA.size = %lu \n", rank, RA_temp.entry.size()); MPI_Barrier(comm);
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 2: rank = %d\n", rank); MPI_Barrier(comm);}

    // ************************************* RA_temp - R off-diag and A remote (on and off-diag) *************************************

    // find the start and end nnz iterator of each block of R.
    // use A.split for this part to find each block corresponding to each processor's A.
//    unsigned int* left_block_nnz = (unsigned int*)malloc(sizeof(unsigned int)*(nprocs));
//    std::fill(left_block_nnz, &left_block_nnz[nprocs], 0);
    std::vector<nnz_t> left_block_nnz(nprocs, 0);

//    MPI_Barrier(comm); printf("rank=%d entry = %ld \n", rank, R->entry_remote[0].col); MPI_Barrier(comm);

    // find the owner of the first R.remote element.
    long procNum = 0;
//    unsigned int nnzIter = 0;
    if(!R->entry_remote.empty()){
        for (nnz_t i = 0; i < R->entry_remote.size(); i++) {
            procNum = lower_bound2(&*A->split.begin(), &*A->split.end(), R->entry_remote[i].col);
            left_block_nnz[procNum]++;
//        if(rank==1) printf("rank=%d, col = %lu, procNum = %ld \n", rank, R->entry_remote[0].col, procNum);
//        if(rank==1) std::cout << "\nprocNum = " << procNum << "   \tcol = " << R->entry_remote[0].col
//                              << "  \tnnzIter = " << nnzIter << "\t first" << std::endl;
//        nnzIter++;
        }
    }

//    unsigned int* left_block_nnz_scan = (unsigned int*)malloc(sizeof(unsigned int)*(nprocs+1));
    std::vector<nnz_t> left_block_nnz_scan(nprocs+1);
    left_block_nnz_scan[0] = 0;
    for(int i = 0; i < nprocs; i++)
        left_block_nnz_scan[i+1] = left_block_nnz_scan[i] + left_block_nnz[i];

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 3: rank = %d\n", rank); MPI_Barrier(comm);}

//    print_vector(left_block_nnz_scan, -1, "left_block_nnz_scan", comm);

//    printf("rank=%d A.nnz=%u \n", rank, A->nnz_l);
    AnnzPerRow.resize(AMaxM);
    indices_row_wise.resize(AMaxNnz);
    std::vector<cooEntry> Arecv(AMaxNnz);
    int left, right;
    nnz_t nnzSend, nnzRecv;
    long ARecvM;
    MPI_Status sendRecvStatus;
    nnz_t R_block_nnz_own;
    bool send_data = true;
    bool recv_data;
    nnz_t k, kstart, kend;
//    MPI_Barrier(comm); printf("\n\n rank = %d, loop starts! \n", rank); MPI_Barrier(comm);

//    print_vector(R->entry_remote, -1, "R->entry_remote", comm);

    // todo: after adding "R_remote block size" part, the current idea seems more efficient than this idea:
    // todo: change the algorithm so every processor sends data only to the next one and receives from the previous one in each iteration.
//    long tag1 = 0;
    for(int i = 1; i < nprocs; i++) {
        // send A to the right processor, receive A from the left processor.
        // "left" decreases by one in each iteration. "right" increases by one.
        right = (rank + i) % nprocs;
        left = rank - i;
        if (left < 0)
            left += nprocs;

        // *************************** RA_temp - A remote - sendrecv(size RBlock) ****************************
        // if the part of R_remote corresponding to this neighbor doesn't have any nonzeros,
        // don't receive A on it and don't send A to it. so communicate #nnz of that R_remote block to decide that.
        // todo: change sendrecv to separate send and recv to skip some of them.

        nnzSend = A->nnz_l;
        jstart  = left_block_nnz_scan[left];
        jend    = left_block_nnz_scan[left + 1];
        recv_data = true;
        if(jend - jstart == 0)
            recv_data = false;
        MPI_Sendrecv(&recv_data, 1, MPI_CXX_BOOL, left, rank,
                     &send_data, 1, MPI_CXX_BOOL, right, right, comm, &sendRecvStatus);

        if(!send_data)
            nnzSend = 0;

        // *************************** RA_temp - A remote - sendrecv(size A) ****************************

        // use sender rank for send and receive tags.
        MPI_Sendrecv(&nnzSend, 1, MPI_UNSIGNED_LONG, right, rank,
                     &nnzRecv, 1, MPI_UNSIGNED_LONG, left,  left, comm, &sendRecvStatus);

//        printf("i=%d, rank=%d, left=%d, right=%d \n", i, rank, left, right);
//        printf("i=%d, rank = %d own A->nnz_l = %u    \tnnzRecv = %u \n", i, rank, A->nnz_l, nnzRecv);

//        if(!R_block_nnz_own)
//            nnzRecv = 0;

//        if(rank==0) printf("i=%d, rank=%d, left=%d, right=%d, nnzSend=%u, nnzRecv = %u, recv_data = %d, send_data = %d \n",
//                           i, rank, left, right, nnzSend, nnzRecv, recv_data, send_data);

        // *************************** RA_temp - A remote - sendrecv(A) ****************************

        // use sender rank for send and receive tags.
        MPI_Sendrecv(&A->entry[0], nnzSend, cooEntry::mpi_datatype(), right, rank,
                     &Arecv[0],    nnzRecv, cooEntry::mpi_datatype(), left,  left, comm, &sendRecvStatus);

//        for(unsigned int j=0; j<nnzRecv; j++)
//                        printf("rank = %d, j=%d \t %lu \t %lu \t %f \n", rank, j, Arecv[j].row , Arecv[j].col, Arecv[j].val);

        // *************************** RA_temp - A remote - multiplication ****************************

        // if #nnz of R_remote block is zero, then skip.
        R_block_nnz_own = jend - jstart;
        if(R_block_nnz_own == 0) continue;

        ARecvM = A->split[left+1] - A->split[left];
        std::fill(&AnnzPerRow[0], &AnnzPerRow[ARecvM], 0);
        unsigned int *AnnzPerRow_p = &AnnzPerRow[0] - A->split[left];
        for(index_t j=0; j<nnzRecv; j++){
            AnnzPerRow_p[Arecv[j].row]++;
//            if(rank==2)
//                printf("%lu \tArecv[j].row[i] = %lu, Arecv[j].row - A->split[left] = %lu \n", j, Arecv[j].row, Arecv[j].row - A->split[left]);
        }

//        print_vector(AnnzPerRow, -1, "AnnzPerRow", comm);

        AnnzPerRowScan[0] = 0;
        for(index_t j=0; j<ARecvM; j++){
            AnnzPerRowScan[j+1] = AnnzPerRowScan[j] + AnnzPerRow[j];
//            if(rank==2) printf("i=%d, AnnzPerRow=%d, AnnzPerRowScan = %d\n", i, AnnzPerRow[i], AnnzPerRowScan[i]);
        }

        // find row-wise ordering for Arecv and save it in indicesPRecv
        for(nnz_t i=0; i<nnzRecv; i++)
            indices_row_wise[i] = i;
        std::sort(&indices_row_wise[0], &indices_row_wise[nnzRecv], sort_indices2(&Arecv[0]));

//        if(rank==1) std::cout << "block start = " << RBlockStart[left] << "\tend = " << RBlockStart[left+1] << "\tleft rank = " << left << "\t i = " << i << std::endl;
        // jstart is the starting entry index of R corresponding to this neighbor.
        for (index_t j = jstart; j < jend; j++) {
//            if(rank==1) std::cout << "R = " << R->entry_remote[j] << std::endl;
//            if(rank==1) std::cout << "col = " << R->entry_remote[j].col << "\tcol-split = " << R->entry_remote[j].col - P->split[left] << "\tstart = " << AnnzPerRowScan[R->entry_remote[j].col - P->split[left]] << "\tend = " << AnnzPerRowScan[R->entry_remote[j].col - P->split[left] + 1] << std::endl;
            // kstart is the starting row entry index of A corresponding to the R entry column.
            kstart = AnnzPerRowScan[R->entry_remote[j].col - P->split[left]];
            kend   = AnnzPerRowScan[R->entry_remote[j].col - P->split[left] + 1];
            if(kend - kstart == 0) continue; // if there isno nonzero on this row of A, then skip.
            for (k = kstart; k < kend; k++) {
//                if(rank==1) std::cout << "R = " << R->entry_remote[j] << "\tA = " << Arecv[indicesPRecv[k]] << std::endl;
                RA_temp.entry.emplace_back(cooEntry(R->entry_remote[j].row,
                                                    Arecv[indices_row_wise[k]].col,
                                                    R->entry_remote[j].val * Arecv[indices_row_wise[k]].val));
            }
        }

    } //for i
//    MPI_Barrier(comm); printf("\n\n rank = %d, loop ends! \n", rank); MPI_Barrier(comm);

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 4-1: rank = %d\n", rank); MPI_Barrier(comm);}

    // todo: check this: since entries of RA_temp with these row indices only exist on this processor,
    // todo: duplicates happen only on this processor, so sorting should be done locally.
    std::sort(RA_temp.entry.begin(), RA_temp.entry.end());

//    printf("rank %d: RA_temp.entry.size_total = %lu \n", rank, RA_temp.entry.size());
//    print_vector(RA_temp.entry, -1, "RA_temp.entry", comm);

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 4-2: rank = %d\n", rank); MPI_Barrier(comm);}

    prolong_matrix RA(comm);
    RA.entry.resize(RA_temp.entry.size());

    // remove duplicates.
    unsigned long entry_size = 0;
    for(nnz_t i=0; i<RA_temp.entry.size(); i++){
        RA.entry[entry_size] = RA_temp.entry[i];
//        if(rank==1) std::cout << RA_temp.entry[i] << std::endl;
        while(i < RA_temp.entry.size()-1 && RA_temp.entry[i] == RA_temp.entry[i+1]){ // values of entries with the same row and col should be added.
            RA.entry[entry_size].val += RA_temp.entry[i+1].val;
            i++;
//            if(rank==1) std::cout << RA_temp.entry[i+1].val << std::endl;
        }
//        if(rank==1) std::cout << std::endl << "final: " << std::endl << RA.entry[RA.entry.size()-1].val << std::endl;
        entry_size++;
        // todo: pruning. don't hard code tol. does this make the matrix non-symmetric?
//        if( abs(RA.entry.back().val) < 1e-6)
//            RA.entry.pop_back();
//        if(rank==1) std::cout << "final: " << std::endl << RA.entry.back().val << std::endl;
    }

    RA_temp.entry.clear();
    RA_temp.entry.shrink_to_fit();
    RA.entry.resize(entry_size);
    RA.entry.shrink_to_fit();

//    print_vector(RA.entry, -1, "RA.entry", comm);

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 4-3: rank = %d\n", rank); MPI_Barrier(comm);}

    // ************************************* RAP_temp - P local *************************************
    // Some local and remote elements of RAP_temp are computed here.
    // Note: P local means whole entries of P on this process, not just the diagonal block.

    prolong_matrix RAP_temp(comm); // RAP_temp is being used to remove duplicates while pushing back to RAP.
    index_t P_max_M;
    MPI_Allreduce(&P->M, &P_max_M, 1, MPI_UNSIGNED, MPI_MAX, comm);
//    MPI_Barrier(comm); printf("rank=%d, PMaxNnz=%d \n", rank, PMaxNnz); MPI_Barrier(comm);

    std::vector<unsigned int> PnnzPerRow(P_max_M, 0);
    for(nnz_t i=0; i<P->nnz_l; i++){
        PnnzPerRow[P->entry[i].row]++;
    }

//    print_vector(PnnzPerRow, -1, "PnnzPerRow", comm);

//    unsigned int* PnnzPerRowScan = (unsigned int*)malloc(sizeof(unsigned int)*(P_max_M+1));
    std::vector<unsigned long> PnnzPerRowScan(P_max_M+1);
    PnnzPerRowScan[0] = 0;
    for(nnz_t i = 0; i < P->M; i++){
        PnnzPerRowScan[i+1] = PnnzPerRowScan[i] + PnnzPerRow[i];
//        if(rank==2) printf("i=%lu, PnnzPerRow=%d, PnnzPerRowScan = %d\n", i, PnnzPerRow[i], PnnzPerRowScan[i]);
    }

    std::fill(&left_block_nnz[0], &left_block_nnz[nprocs], 0);
    if(!RA.entry.empty()){
        for (nnz_t i = 0; i < RA.entry.size(); i++) {
            procNum = lower_bound2(&P->split[0], &P->split[nprocs], RA.entry[i].col);
            left_block_nnz[procNum]++;
//        if(rank==1) printf("rank=%d, col = %lu, procNum = %ld \n", rank, R->entry_remote[0].col, procNum);
        }
    }

    left_block_nnz_scan[0] = 0;
    for(int i = 0; i < nprocs; i++)
        left_block_nnz_scan[i+1] = left_block_nnz_scan[i] + left_block_nnz[i];

//    print_vector(left_block_nnz_scan, -1, "left_block_nnz_scan", comm);

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 4-4: rank = %d\n", rank); MPI_Barrier(comm);}

    // todo: combine indicesP_Prolong and indicesP_ProlongRecv together.
    // find row-wise ordering for A and save it in indicesP
    indices_row_wise.resize(P->nnz_l);
    for(nnz_t i=0; i<P->nnz_l; i++)
        indices_row_wise[i] = i;

    std::sort(&indices_row_wise[0], &indices_row_wise[P->nnz_l], sort_indices2(&*P->entry.begin()));

    for(nnz_t i=left_block_nnz_scan[rank]; i<left_block_nnz_scan[rank+1]; i++){
        for(nnz_t j = PnnzPerRowScan[RA.entry[i].col - P->split[rank]]; j < PnnzPerRowScan[RA.entry[i].col - P->split[rank] + 1]; j++){

//            if(rank==3) std::cout << RA.entry[i].row + P->splitNew[rank] << "\t" << P->entry[indicesP_Prolong[j]].col << "\t" << RA.entry[i].val * P->entry[indicesP_Prolong[j]].val << std::endl;

            RAP_temp.entry.emplace_back(cooEntry(RA.entry[i].row + P->splitNew[rank],  // Ac.entry should have global indices at the end.
                                                 P->entry[indices_row_wise[j]].col,
                                                 RA.entry[i].val * P->entry[indices_row_wise[j]].val));
        }
    }

//    print_vector(RAP_temp.entry, -1, "RAP_temp.entry", comm);

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 5: rank = %d\n", rank); MPI_Barrier(comm);}

    // ************************************* RAP_temp - P remote *************************************

    nnz_t PMaxNnz;
    MPI_Allreduce(&P->nnz_l, &PMaxNnz, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);

    indices_row_wise.resize(PMaxNnz);
    std::vector<cooEntry> Precv(PMaxNnz);
    nnz_t PrecvM;

    for(int i = 1; i < nprocs; i++) {
        // send P to the right processor, receive P from the left processor. "left" decreases by one in each iteration. "right" increases by one.
        right = (rank + i) % nprocs;
        left = rank - i;
        if (left < 0)
            left += nprocs;

        // *************************** RAP_temp - P remote - sendrecv(size) ****************************

        // use sender rank for send and receive tags.
        MPI_Sendrecv(&P->nnz_l, 1, MPI_UNSIGNED_LONG, right, rank, &nnzRecv, 1, MPI_UNSIGNED_LONG, left, left, comm, &sendRecvStatus);
//        int MPI_Sendrecv(const void *sendbuf, int sendcount, MPI_Datatype sendtype, int dest, int sendtag, void *recvbuf,
//                         int recvcount, MPI_Datatype recvtype, int source, int recvtag, MPI_Comm comm, MPI_Status *status)

//        printf("i=%d, rank=%d, left=%d, right=%d \n", i, rank, left, right);
//        printf("i=%d, rank = %d own P->nnz_l = %lu    \tnnzRecv = %u \n", i, rank, P->nnz_l, nnzRecv);

        // *************************** RAP_temp - P remote - sendrecv(P) ****************************

        // use sender rank for send and receive tags.
        MPI_Sendrecv(&P->entry[0], P->nnz_l, cooEntry::mpi_datatype(), right, rank,
                     &Precv[0],    nnzRecv,  cooEntry::mpi_datatype(), left,  left, comm, &sendRecvStatus);

//        if(rank==1) for(int j=0; j<P->nnz_l; j++)
//                        printf("j=%d \t %lu \t %lu \t %f \n", j, P->entry[j].row, P->entry[j].col, P->entry[j].val);
//        if(rank==1) for(int j=0; j<nnzRecv; j++)
//                        printf("j=%d \t %lu \t %lu \t %f \n", j, Precv[j].row, Precv[j].col, Precv[j].val);

        // *************************** RAP_temp - P remote - multiplication ****************************

        PrecvM = P->split[left+1] - P->split[left];
        std::fill(&PnnzPerRow[0], &PnnzPerRow[PrecvM], 0);
        for(nnz_t j=0; j<nnzRecv; j++)
            PnnzPerRow[Precv[j].row]++;

//        if(rank==1)
//            for(j=0; j<PrecvM; j++)
//                std::cout << PnnzPerRow[i] << std::endl;

        PnnzPerRowScan[0] = 0;
        for(nnz_t j=0; j<PrecvM; j++){
            PnnzPerRowScan[j+1] = PnnzPerRowScan[j] + PnnzPerRow[j];
//            if(rank==1) printf("j=%lu, PnnzPerRow=%d, PnnzPerRowScan = %d\n", j, PnnzPerRow[j], PnnzPerRowScan[j]);
        }

        // find row-wise ordering for Arecv and save it in indicesPRecv
        for(nnz_t i=0; i<nnzRecv; i++)
            indices_row_wise[i] = i;

        std::sort(&indices_row_wise[0], &indices_row_wise[nnzRecv], sort_indices2(&Precv[0]));

//        if(rank==1) std::cout << "block start = " << RBlockStart[left] << "\tend = " << RBlockStart[left+1] << "\tleft rank = " << left << "\t i = " << i << std::endl;
        if(!RA.entry.empty()) {
            for (nnz_t j = left_block_nnz_scan[left]; j < left_block_nnz_scan[left + 1]; j++) {
//            if(rank==1) std::cout << "col = " << R->entry_remote[j].col << "\tcol-split = " << R->entry_remote[j].col - P->split[left] << "\tstart = " << AnnzPerRowScan[R->entry_remote[j].col - P->split[left]] << "\tend = " << AnnzPerRowScan[R->entry_remote[j].col - P->split[left] + 1] << std::endl;
                for (nnz_t k = PnnzPerRowScan[RA.entry[j].col - P->split[left]];
                     k < PnnzPerRowScan[RA.entry[j].col - P->split[left] + 1]; k++) {
//                if(rank==0) std::cout << Precv[indicesP_ProlongRecv[k]].row << "\t" << Precv[indicesP_ProlongRecv[k]].col << "\t" << Precv[indicesP_ProlongRecv[k]].val << std::endl;
                    RAP_temp.entry.emplace_back(cooEntry(
                            RA.entry[j].row + P->splitNew[rank], // Ac.entry should have global indices at the end.
                            Precv[indices_row_wise[k]].col,
                            RA.entry[j].val * Precv[indices_row_wise[k]].val));
                }
            }
        }

    } //for i

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 6-1: rank = %d\n", rank); MPI_Barrier(comm);}

    std::sort(RAP_temp.entry.begin(), RAP_temp.entry.end());

//    print_vector(RAP_temp.entry, -1, "RAP_temp.entry", comm);

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 6-2: rank = %d\n", rank); MPI_Barrier(comm);}

    Ac->entry.resize(RAP_temp.entry.size());

    // remove duplicates.
//    std::vector<cooEntry> Ac_temp;
    entry_size = 0;
    for(nnz_t i=0; i<RAP_temp.entry.size(); i++){
//        Ac->entry.push_back(RAP_temp.entry[i]);
        Ac->entry[entry_size] = RAP_temp.entry[i];
        while(i<RAP_temp.entry.size()-1 && RAP_temp.entry[i] == RAP_temp.entry[i+1]){ // values of entries with the same row and col should be added.
//            Ac->entry.back().val += RAP_temp.entry[i+1].val;
            Ac->entry[entry_size].val += RAP_temp.entry[i+1].val;
            i++;
        }
        entry_size++;
//        if( abs(Ac->entry.back().val) < 1e-6)
//            Ac->entry.pop_back();
    }

//    if(rank==0) printf("rank %d: entry_size = %lu \n", rank, entry_size);

    RAP_temp.entry.clear();
    RAP_temp.entry.shrink_to_fit();
    Ac->entry.resize(entry_size);
    Ac->entry.shrink_to_fit();

//    MPI_Barrier(comm); printf("rank %d: %lu \n", rank, Ac->entry.size()); MPI_Barrier(comm);
//    print_vector(Ac->entry, -1, "Ac->entry", comm);

//    par::sampleSort(Ac_temp, Ac->entry, comm);

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 7: rank = %d\n", rank); MPI_Barrier(comm);}

    Ac->nnz_l = entry_size;
    MPI_Allreduce(&Ac->nnz_l, &Ac->nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
    Ac->Mbig = P->Nbig;
    Ac->M = P->splitNew[rank+1] - P->splitNew[rank];
    Ac->M_old = Ac->M;
    Ac->split = P->splitNew;
    Ac->last_M_shrink = A->last_M_shrink;
    Ac->last_density_shrink = A->last_density_shrink;
//    Ac->last_nnz_shrink = A->last_nnz_shrink;
//    Ac->enable_shrink = A->enable_shrink;
//    Ac->enable_shrink = A->enable_shrink_next_level;
    Ac->comm = A->comm;
    Ac->comm_old = A->comm;
    Ac->active_old_comm = true;
    Ac->density = float(Ac->nnz_g) / (Ac->Mbig * Ac->Mbig);
    Ac->switch_to_dense = switch_to_dense;
    Ac->dense_threshold = dense_threshold;

    Ac->cpu_shrink_thre1 = A->cpu_shrink_thre1; //todo: is this required?
    if(A->cpu_shrink_thre2_next_level != -1) // this is -1 by default.
        Ac->cpu_shrink_thre2 = A->cpu_shrink_thre2_next_level;

    //return these to default, since they have been used in the above part.
    A->cpu_shrink_thre2_next_level = -1;
    A->enable_shrink_next_level = false;

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 8: rank = %d\n", rank); MPI_Barrier(comm);}

//    print_vector(Ac->split, -1, "Ac->split", comm);
    for(index_t i = 0; i < Ac->split.size()-1; i++){
        if(Ac->split[i+1] - Ac->split[i] == 0){
//            printf("rank %d: shrink minor in compute_coarsen: i = %d, split[i] = %d, split[i+1] = %d\n", rank, i, Ac->split[i], Ac->split[i+1]);
            Ac->shrink_cpu_minor();
            break;
        }
    }

//    MPI_Barrier(comm);
//    printf("Ac: rank = %d \tMbig = %u \tM = %u \tnnz_g = %lu \tnnz_l = %lu \tdensity = %f\n",
//           rank, Ac->Mbig, Ac->M, Ac->nnz_g, Ac->nnz_l, Ac->density);
//    MPI_Barrier(comm);

//    if(verbose_compute_coarsen){
//        printf("\nrank = %d, Ac->Mbig = %u, Ac->M = %u, Ac->nnz_l = %lu, Ac->nnz_g = %lu \n", rank, Ac->Mbig, Ac->M, Ac->nnz_l, Ac->nnz_g);}

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 9: rank = %d\n", rank); MPI_Barrier(comm);}


    if(Ac->active_minor){
        comm = Ac->comm;
        int rank_new;
        MPI_Comm_rank(Ac->comm, &rank_new);
//        Ac->print_info(-1);

        // ********** decide about shrinking **********

        if(Ac->enable_shrink && Ac->enable_dummy_matvec && nprocs > 1){
//            MPI_Barrier(Ac->comm); if(rank_new==0) printf("start decide shrinking\n"); MPI_Barrier(Ac->comm);
            Ac->matrix_setup_dummy();
            Ac->compute_matvec_dummy_time();
            Ac->decide_shrinking(A->matvec_dummy_time);
            Ac->erase_after_decide_shrinking();
//            MPI_Barrier(Ac->comm); if(rank_new==0) printf("finish decide shrinking\n"); MPI_Barrier(Ac->comm);
        }

        if(verbose_compute_coarsen){
            MPI_Barrier(comm); printf("compute_coarsen: step 10: rank = %d\n", rank); MPI_Barrier(comm);}

        // ********** setup matrix **********
        // Shrinking gets decided inside repartition_nnz() or repartition_row() functions, then repartition happens.
        // Finally, shrink_cpu() and matrix_setup() are called. In this way, matrix_setup is called only once.

        // decide to partition based on number of rows or nonzeros.
//    if(switch_repartition && Ac->density >= repartition_threshold)
        if(switch_repartition && Ac->density >= repartition_threshold){
            if(rank==0) printf("equi-ROW partition for the next level: density = %f, repartition_threshold = %f \n", Ac->density, repartition_threshold);
            Ac->repartition_row(); // based on number of rows
        }else{
            Ac->repartition_nnz(); // based on number of nonzeros
        }

//        if(Ac->shrinked_minor){
//            repartition_u_shrink_minor_prepare(grid);
//        }

        if(verbose_compute_coarsen){
            MPI_Barrier(comm); printf("compute_coarsen: step 11: rank = %d\n", rank); MPI_Barrier(comm);}

        repartition_u_shrink_prepare(grid);

        if(Ac->shrinked){
            Ac->shrink_cpu();
        }

        if(verbose_compute_coarsen){
            MPI_Barrier(comm); printf("compute_coarsen: step 12: rank = %d\n", rank); MPI_Barrier(comm);}

        if(Ac->active){
            Ac->matrix_setup();

            if(Ac->shrinked && Ac->enable_dummy_matvec)
                Ac->compute_matvec_dummy_time();

            if(switch_to_dense && Ac->density > dense_threshold){
                if(rank==0) printf("Switch to dense: density = %f, dense_threshold = %f \n", Ac->density, dense_threshold);
                Ac->generate_dense_matrix();
            }
        }

//        Ac->print_info(-1);
//        Ac->print_entry(-1);
    }

    comm = grid->A->comm;
    if(verbose_compute_coarsen){MPI_Barrier(comm); printf("end of compute_coarsen: rank = %d\n", rank); MPI_Barrier(comm);}

    return 0;
} // compute_coarsen_old()

int saena_object::compute_coarsen_old2(Grid *grid) {
    // This version is based on the older mat-mat product which was based on COO.

    // Output: Ac = R * A * P
    // Steps:
    // 1- Compute AP = A * P. To do that use the transpose of R_i, instead of P. Pass all R_j's to all the processors,
    //    Then, multiply local A_i by R_j on each process.
    // 2- Compute RAP = R * AP. Use transpose of P_i instead of R. It is done locally. So multiply P_i * (AP)_i.
    // 3- Sort and remove local duplicates.
    // 4- Do a parallel sort based on row-major order. A modified version of par::sampleSort from usort is used here.
    //    Again, remove duplicates.
    // 5- Not complete yet: Sparsify Ac.

    saena_matrix *A    = grid->A;
    prolong_matrix *P  = &grid->P;
    restrict_matrix *R = &grid->R;
    saena_matrix *Ac   = &grid->Ac;

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

#ifdef __DEBUG1__
//    print_vector(A->entry, -1, "A->entry", comm);
//    print_vector(P->entry, -1, "P->entry", comm);
//    print_vector(R->entry, -1, "R->entry", comm);

//    Ac->active_old_comm = true;

//    int rank1, nprocs1;
//    MPI_Comm_size(comm, &nprocs1);
//    MPI_Comm_rank(comm, &rank1);
//    if(A->active_old_comm)
//        printf("rank = %d, nprocs = %d active\n", rank1, nprocs1);

//#ifdef SPLIT_NNZ
//    if(rank==0) printf("\nfast_mm: split based on nnz\n");
//#endif
//#ifdef SPLIT_SIZE
//    if(rank==0) printf("\nfast_mm: split based on matrix size\n");
//#endif

    if (verbose_compute_coarsen) {
        MPI_Barrier(comm);
        if (rank == 0) printf("start of compute_coarsen nprocs: %d \n", nprocs);
        MPI_Barrier(comm);
        printf("rank %d: A.Mbig = %u, \tA.M = %u, \tA.nnz_g = %lu, \tA.nnz_l = %lu \n", rank, A->Mbig, A->M, A->nnz_g,
               A->nnz_l);
        MPI_Barrier(comm);
        printf("rank %d: P.Mbig = %u, \tP.M = %u, \tP.nnz_g = %lu, \tP.nnz_l = %lu \n", rank, P->Mbig, P->M, P->nnz_g,
               P->nnz_l);
        MPI_Barrier(comm);
        printf("rank %d: R.Mbig = %u, \tR.M = %u, \tR.nnz_g = %lu, \tR.nnz_l = %lu \n", rank, R->Mbig, R->M, R->nnz_g,
               R->nnz_l);
        MPI_Barrier(comm);
    }
#endif

    // set some of Ac parameters
    // -------------------------
    Ac->Mbig = P->Nbig;
    Ac->split = P->splitNew;
    Ac->M = Ac->split[rank+1] - Ac->split[rank];
    Ac->M_old = Ac->M;

    Ac->comm = A->comm;
    Ac->comm_old = A->comm;
    Ac->active_old_comm = true;

    // set dense parameters
    Ac->density = float(Ac->nnz_g) / (Ac->Mbig * Ac->Mbig);
    Ac->switch_to_dense = switch_to_dense;
    Ac->dense_threshold = dense_threshold;

    // set shrink parameters
    Ac->last_M_shrink = A->last_M_shrink;
    Ac->last_density_shrink = A->last_density_shrink;
    Ac->cpu_shrink_thre1 = A->cpu_shrink_thre1; //todo: is this required?
    if(A->cpu_shrink_thre2_next_level != -1) // this is -1 by default.
        Ac->cpu_shrink_thre2 = A->cpu_shrink_thre2_next_level;
    //return these to default, since they have been used in the above part.
    A->cpu_shrink_thre2_next_level = -1;
    A->enable_shrink_next_level = false;

#ifdef __DEBUG1__
//    MPI_Barrier(comm);
//    printf("Ac: rank = %d \tMbig = %u \tM = %u \tnnz_g = %lu \tnnz_l = %lu \tdensity = %f\n",
//           rank, Ac->Mbig, Ac->M, Ac->nnz_g, Ac->nnz_l, Ac->density);
//    MPI_Barrier(comm);

//    if(verbose_compute_coarsen){
//        printf("\nrank = %d, Ac->Mbig = %u, Ac->M = %u, Ac->nnz_l = %lu, Ac->nnz_g = %lu \n", rank, Ac->Mbig, Ac->M, Ac->nnz_l, Ac->nnz_g);}

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 2: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // minor shrinking
    // -------------------------------------
    Ac->active = true;
    Ac->active_minor = true;
    for(index_t i = 0; i < Ac->split.size()-1; i++){
        if(Ac->split[i+1] - Ac->split[i] == 0){
//            printf("rank %d: shrink minor in compute_coarsen: i = %d, split[i] = %d, split[i+1] = %d\n",
//                    rank, i, Ac->split[i], Ac->split[i+1]);
            Ac->shrink_cpu_minor();
            break;
        }
    }

#ifdef __DEBUG1__
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 3: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // *******************************************************
    // perform the triple multiplication: R*A*P
    // *******************************************************

    // reserve memory for matmat_size_thre2 used in fast_mm case1
//    map_matmat.reserve(2*matmat_size_thre2);
//    for(nnz_t i = 0; i < matmat_size_thre2; i++){
//        map_matmat[i] = 0;
//    }

//    MPI_Barrier(grid->A->comm);
//    double t11 = MPI_Wtime();

    std::vector<cooEntry_row> RAP_row_sorted;

    if(coarsen_method == "recursive"){
        triple_mat_mult_old2(grid, RAP_row_sorted);
    }else if(coarsen_method == "basic"){
        triple_mat_mult_basic(grid, RAP_row_sorted);
    }else if(coarsen_method == "no_overlap"){
        triple_mat_mult_no_overlap(grid, RAP_row_sorted);
    }else{
        printf("wrong coarsen method!\n");
    }

    // *******************************************************
    // form Ac
    // *******************************************************

    nnz_t size_minus_1 = 0;
    if(!RAP_row_sorted.empty()){
        size_minus_1 = RAP_row_sorted.size() - 1;
    }

    if(!doSparsify){

        // *******************************************************
        // version 1: without sparsification
        // *******************************************************
        // since RAP_row_sorted is sorted in row-major order, Ac->entry will be the same.

        Ac->entry.reserve(RAP_row_sorted.size()/30);

        // remove duplicates.
        cooEntry temp;
        for(nnz_t i = 0; i < RAP_row_sorted.size(); i++){
            temp = cooEntry(RAP_row_sorted[i].row, RAP_row_sorted[i].col, RAP_row_sorted[i].val);
            while(i < size_minus_1 && RAP_row_sorted[i] == RAP_row_sorted[i+1]){ // values of entries with the same row and col should be added.
                ++i;
                temp.val += RAP_row_sorted[i].val;
            }
            Ac->entry.emplace_back( temp );
        }

//        double t22 = MPI_Wtime();
//        print_time_ave(t22-t11, "triple_mat_mult: level "+std::to_string(grid->currentLevel), grid->A->comm);

        RAP_row_sorted.clear();
        RAP_row_sorted.shrink_to_fit();

    }else{

        // *******************************************************
        // version 2: with sparsification
        // *******************************************************

        // remove duplicates.
        // compute Frobenius norm squared (norm_frob_sq).
        cooEntry_row temp;
        double max_val = 0;
        double norm_frob_sq_local = 0, norm_frob_sq = 0;
        std::vector<cooEntry_row> Ac_orig;
//        nnz_t no_sparse_size = 0;
        for(nnz_t i = 0; i < RAP_row_sorted.size(); i++){
            temp = cooEntry_row(RAP_row_sorted[i].row, RAP_row_sorted[i].col, RAP_row_sorted[i].val);
            while(i < size_minus_1 && RAP_row_sorted[i] == RAP_row_sorted[i+1]){ // values of entries with the same row and col should be added.
                ++i;
                temp.val += RAP_row_sorted[i].val;
            }

//            if( fabs(val_temp) > sparse_epsilon / 2 / Ac->Mbig)
//            if(temp.val * temp.val > sparse_epsilon * sparse_epsilon / (4 * Ac->Mbig * Ac->Mbig) ){
            Ac_orig.emplace_back( temp );
            norm_frob_sq_local += temp.val * temp.val;
            if( fabs(temp.val) > max_val){
                max_val = temp.val;
            }
//            }
//            no_sparse_size++; //todo: just for test. delete this later!
        }

        MPI_Allreduce(&norm_frob_sq_local, &norm_frob_sq, 1, MPI_DOUBLE, MPI_SUM, comm);

#ifdef __DEBUG1__
//        if(rank==0) printf("\noriginal size   = %lu\n", Ac_orig.size());
//        if(rank==0) printf("\noriginal size without sparsification   \t= %lu\n", no_sparse_size);
//        if(rank==0) printf("filtered Ac size before sparsification \t= %lu\n", Ac_orig.size());

//        std::sort(Ac_orig.begin(), Ac_orig.end());
//        print_vector(Ac_orig, -1, "Ac_orig", A->comm);
#endif

        RAP_row_sorted.clear();
        RAP_row_sorted.shrink_to_fit();

//        auto sample_size = Ac_orig.size();
        auto sample_size_local = nnz_t(sample_sz_percent * Ac_orig.size());
//        auto sample_size = nnz_t(Ac->Mbig * Ac->Mbig * A->density);
//        if(rank==0) printf("sample_size     = %lu \n", sample_size);
        nnz_t sample_size;
        MPI_Allreduce(&sample_size_local, &sample_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

//        if(sparsifier == "TRSL"){
//
//            sparsify_trsl1(Ac_orig, Ac->entry, norm_frob_sq, sample_size, comm);
//
//        }else if(sparsifier == "drineas"){
//
//            sparsify_drineas(Ac_orig, Ac->entry, norm_frob_sq, sample_size, comm);
//
//        }else if(sparsifier == "majid"){
//
//            sparsify_majid(Ac_orig, Ac->entry, norm_frob_sq, sample_size, max_val, comm);
//
//        }else{
//            printf("\nerror: wrong sparsifier!");
//        }

        if(Ac->active_minor) {
            if (sparsifier == "majid") {
                //todo: fix sparsify_majid()
//                sparsify_majid(Ac_orig, Ac->entry, norm_frob_sq, sample_size, max_val, Ac->comm);
            } else {
                printf("\nerror: wrong sparsifier!");
            }
        }

    }

#ifdef __DEBUG1__
//    print_vector(Ac->entry, -1, "Ac->entry", A->comm);
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 9: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // *******************************************************
    // setup matrix
    // *******************************************************
    // Update this description: Shrinking gets decided inside repartition_nnz() or repartition_row() functions,
    // then repartition happens.
    // Finally, shrink_cpu() and matrix_setup() are called. In this way, matrix_setup is called only once.

    Ac->nnz_l = Ac->entry.size();
    MPI_Allreduce(&Ac->nnz_l, &Ac->nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

#ifdef __DEBUG1__
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 10: rank = %d\n", rank); MPI_Barrier(comm);
    }
#endif

    if(Ac->active_minor){
        comm = Ac->comm;
        int rank_new;
        MPI_Comm_rank(Ac->comm, &rank_new);

#ifdef __DEBUG1__
//        Ac->print_info(-1);
//        Ac->print_entry(-1);
#endif

        // ********** decide about shrinking **********
        //---------------------------------------------
        if(Ac->enable_shrink && Ac->enable_dummy_matvec && nprocs > 1){
//            MPI_Barrier(Ac->comm); if(rank_new==0) printf("start decide shrinking\n"); MPI_Barrier(Ac->comm);
            Ac->matrix_setup_dummy();
            Ac->compute_matvec_dummy_time();
            Ac->decide_shrinking(A->matvec_dummy_time);
            Ac->erase_after_decide_shrinking();
        }

#ifdef __DEBUG1__
        if(verbose_compute_coarsen){
            MPI_Barrier(comm); printf("compute_coarsen: step 11: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

        // decide to partition based on number of rows or nonzeros.
//    if(switch_repartition && Ac->density >= repartition_threshold)
        if(switch_repartition && Ac->density >= repartition_threshold){
            if(rank==0) printf("equi-ROW partition for the next level: density = %f, repartition_threshold = %f \n", Ac->density, repartition_threshold);
            Ac->repartition_row(); // based on number of rows
        }else{
            Ac->repartition_nnz(); // based on number of nonzeros
        }

#ifdef __DEBUG1__
        if(verbose_compute_coarsen){
            MPI_Barrier(comm); printf("compute_coarsen: step 12: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

        repartition_u_shrink_prepare(grid);

        Ac->active = true;
//        Ac->active_minor = true;
        if(Ac->shrinked){
            Ac->shrink_cpu();
        }

#ifdef __DEBUG1__
        if(verbose_compute_coarsen){
            MPI_Barrier(comm); printf("compute_coarsen: step 13: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

        if(Ac->active){
            Ac->matrix_setup();
//            Ac->matrix_setup_no_scale();

            if(Ac->shrinked && Ac->enable_dummy_matvec)
                Ac->compute_matvec_dummy_time();

            if(switch_to_dense && Ac->density > dense_threshold){
                if(rank==0) printf("Switch to dense: density = %f, dense_threshold = %f \n", Ac->density, dense_threshold);
                Ac->generate_dense_matrix();
            }
        }

#ifdef __DEBUG1__
//        Ac->print_info(-1);
//        Ac->print_entry(-1);
#endif

    }
    comm = grid->A->comm;

#ifdef __DEBUG1__
    if(verbose_compute_coarsen){MPI_Barrier(comm); printf("end of compute_coarsen: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // view matrix Ac
    // --------------
//    petsc_viewer(Ac);

    return 0;
} // compute_coarsen()


int saena_object::matmat_COO(saena_matrix *A, saena_matrix *B, saena_matrix *C){
    printf("update saena_object::matmat_COO() before use.\n");
    /*
    // B1 should be symmetric. Because we need its transpose. Treat its row indices as column indices and vice versa.

//    saena_matrix *A = A1.get_internal_matrix();
//    saena_matrix *B = B1.get_internal_matrix();

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    bool verbose_fastmm = false;

    mempool1 = new value_t[matmat_size_thre2];
    mempool2 = new index_t[A->Mbig * 4];
    mempool3 = new cooEntry[B->nnz_max * 2];

    MPI_Barrier(comm);
    double t_AP = MPI_Wtime();

    MPI_Barrier(comm);
    double t1 = MPI_Wtime();

    unsigned long send_size     = B->entry.size();
    unsigned long send_size_max = B->nnz_max;
//    MPI_Allreduce(&send_size, &send_size_max, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);
//    printf("send_size = %lu, send_size_max = %lu\n", send_size, send_size_max);

    // local transpose of R is being used to compute A*P. So R is transposed locally here.
//    std::vector<cooEntry> mat_send(R->entry.size());
//    auto mat_send = new cooEntry[send_size_max];
    auto mat_send = &mempool3[0];
//    transpose_locally(&R->entry[0], R->entry.size(), R->splitNew[rank], &mat_send[0]);
//    memcpy(&mat_send[0], &B->entry[0], B->entry.size() * sizeof(cooEntry));

//    std::sort(B->entry.begin(), B->entry.end(), row_major);

    for(nnz_t i = 0; i < B->entry.size(); i++){
        mat_send[i] = cooEntry(B->entry[i].col, B->entry[i].row, B->entry[i].val);
//        if(rank==1) std::cout << mat_send[i] << std::endl;
    }

    std::sort(&mat_send[0], &mat_send[B->entry.size()]);

    t1 = MPI_Wtime() - t1;
    print_time_ave(t1, "mat_send:", comm);

#ifdef __DEBUG1__
//    print_vector(A->entry, 1, "A->entry", comm);
//    print_vector(A->nnzPerColScan, 0, "A->nnzPerColScan", comm);
#endif

    MPI_Barrier(comm);
    t1 = MPI_Wtime() - t1;

    index_t *nnzPerColScan_left = &A->nnzPerColScan[0];
    index_t mat_recv_M_max      = B->M_max;

    std::vector<index_t> nnzPerColScan_right(mat_recv_M_max + 1);
    index_t *nnzPerCol_right   = &nnzPerColScan_right[1];
    index_t *nnzPerCol_right_p = &nnzPerCol_right[0]; // use this to avoid subtracting a fixed number,

    std::vector<cooEntry> AB_temp;

//    printf("\n");
    if(nprocs > 1){

        int right_neighbor = (rank + 1)%nprocs;
        int left_neighbor  = rank - 1;
        if (left_neighbor < 0){
            left_neighbor += nprocs;
        }

        int owner;
        unsigned long recv_size;
//        std::vector<cooEntry> mat_recv;
//        auto mat_recv = new cooEntry[send_size_max];
        auto mat_recv = &mempool3[send_size_max];
        index_t mat_recv_M;

        auto *requests = new MPI_Request[4];
        auto *statuses = new MPI_Status[4];

        for(int k = rank; k < rank+nprocs; k++){
            // This is overlapped. Both local and remote loops are done here.
            // The first iteration is the local loop. The rest are remote.
            // Send R_tranpose to the left_neighbor processor, receive R_tranpose from the right_neighbor.
            // In the next step: send R_tranpose that was received in the previous step to the left_neighbor processor,
            // receive R_tranpose from the right_neighbor. And so on.
            // --------------------------------------------------------------------

            // communicate size
            MPI_Irecv(&recv_size, 1, MPI_UNSIGNED_LONG, right_neighbor, right_neighbor, comm, requests);
            MPI_Isend(&send_size, 1, MPI_UNSIGNED_LONG, left_neighbor,  rank,           comm, requests+1);
            MPI_Waitall(1, requests, statuses);
//          printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
//            mat_recv.resize(recv_size);

#ifdef __DEBUG1__
//          print_vector(mat_recv, -1, "mat_recv", A->comm);
//          print_vector(mat_send, -1, "mat_send", A->comm);
#endif
            // communicate data
            MPI_Irecv(&mat_recv[0], recv_size, cooEntry::mpi_datatype(), right_neighbor, right_neighbor, comm, requests+2);
            MPI_Isend(&mat_send[0], send_size, cooEntry::mpi_datatype(), left_neighbor,  rank,           comm, requests+3);

            owner = k%nprocs;
            mat_recv_M = B->split[owner + 1] - B->split[owner];
//          printf("rank %d: owner = %d, mat_recv_M = %d, B_col_offset = %u \n", rank, owner, mat_recv_M, P->splitNew[owner]);

            std::fill(&nnzPerCol_right[0], &nnzPerCol_right[mat_recv_M], 0);
            nnzPerCol_right_p = &nnzPerCol_right[0] - B->split[owner];
            for(nnz_t i = 0; i < send_size; i++){
                nnzPerCol_right_p[mat_send[i].col]++;
            }

//            nnzPerColScan_right[0] = 0;
            for(nnz_t i = 0; i < mat_recv_M; i++){
                nnzPerColScan_right[i+1] += nnzPerColScan_right[i];
            }

#ifdef __DEBUG1__
//          print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
//          print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

            if(A->entry.empty() || send_size==0){ // skip!
#ifdef __DEBUG1__
                if(verbose_fastmm){
                    if(A->entry.empty()){
                        printf("\nskip: A->entry.size() == 0\n\n");
                    } else {
                        printf("\nskip: mat_send == 0\n\n");
                    }
                }
#endif
            } else {

                fast_mm(&A->entry[0], &mat_send[0], AB_temp, A->entry.size(), send_size,
                        A->M, A->split[rank], A->Mbig, 0, mat_recv_M, B->split[owner],
                        &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
                        &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//                fast_mm(&A->entry[0], &mat_send[0], AB_temp, A->entry.size(), mat_send.size(),
//                        A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[owner],
//                        &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
//                        &nnzPerColScan_right[0], &nnzPerColScan_right[1], map_matmat, A->comm);

            }

            MPI_Waitall(3, requests+1, statuses+1);

//            mat_recv.swap(mat_send);
            std::swap(mat_send, mat_recv);
            send_size = recv_size;

#ifdef __DEBUG1__
//          print_vector(AB_temp, -1, "AB_temp", A->comm);
//          print_vector(mat_send, -1, "mat_send", A->comm);
//          print_vector(mat_recv, -1, "mat_recv", A->comm);
//          prev_owner = owner;
//          printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
#endif

        }

//        mat_recv.clear();
//        mat_recv.shrink_to_fit();
//        delete [] mat_recv;
        delete [] requests;
        delete [] statuses;

    } else { // nprocs == 1 -> serial

        index_t mat_recv_M = B->split[rank + 1] - B->split[rank];

        std::fill(&nnzPerCol_right[0], &nnzPerCol_right[mat_recv_M], 0);
        nnzPerCol_right_p = &nnzPerCol_right[0] - B->split[rank];
        for(nnz_t i = 0; i < send_size; i++){
            nnzPerCol_right_p[mat_send[i].col]++;
        }

//        nnzPerColScan_right[0] = 0;
        for(nnz_t i = 0; i < mat_recv_M; i++){
            nnzPerColScan_right[i+1] += nnzPerColScan_right[i];
        }

#ifdef __DEBUG1__
//          print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
//          print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

        if(A->entry.empty() || send_size == 0){ // skip!
#ifdef __DEBUG1__
            if(verbose_fastmm){
                if(A->entry.empty()){
                    printf("\nskip: A->entry.size() == 0\n\n");
                } else {
                    printf("\nskip: mat_send == 0\n\n");
                }
            }
#endif
        } else {

//            double t1 = MPI_Wtime();

            fast_mm(&A->entry[0], &mat_send[0], AB_temp, A->entry.size(), send_size,
                    A->M, A->split[rank], A->Mbig, 0, mat_recv_M, B->split[rank],
                    &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
                    &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//            double t2 = MPI_Wtime();
//            printf("\nfast_mm of AB_temp = %f\n", t2-t1);
        }
    }

    t1 = MPI_Wtime() - t1;
    print_time_ave(t1, "AB_temp:", comm);

//    print_vector(AB_temp, 0, "AB_temp", comm);

    MPI_Barrier(comm);
    t1 = MPI_Wtime();

    std::sort(AB_temp.begin(), AB_temp.end());
//    std::vector<cooEntry> AB;
//    nnz_t AP_temp_size_minus1 = AB_temp.size()-1;
//    for(nnz_t i = 0; i < AB_temp.size(); i++){
//        AB.emplace_back(AB_temp[i]);
//        while(i < AP_temp_size_minus1 && AB_temp[i] == AB_temp[i+1]){ // values of entries with the same row and col should be added.
//            AB.back().val += AB_temp[++i].val;
//        }
//    }

    nnz_t AP_temp_size_minus1 = AB_temp.size()-1;
    for(nnz_t i = 0; i < AB_temp.size(); i++){
        C->entry.emplace_back(AB_temp[i]);
        while(i < AP_temp_size_minus1 && AB_temp[i] == AB_temp[i+1]){ // values of entries with the same row and col should be added.
//            std::cout << AB_temp[i] << "\t" << AB_temp[i+1] << std::endl;
            C->entry.back().val += AB_temp[++i].val;
        }
    }

    t1 = MPI_Wtime() - t1;
    print_time_ave(t1, "AB:", comm);

//    mat_send.clear();
//    mat_send.shrink_to_fit();
    AB_temp.clear();
    AB_temp.shrink_to_fit();
//    delete [] mat_send;

//    unsigned long AP_size_loc = AB.size(), AP_size;
//    MPI_Reduce(&AP_size_loc, &AP_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);
//    if(!rank) printf("A_nnz_g = %lu, \tP_nnz_g = %lu, \tAP_size = %lu\n", A->nnz_g, P->nnz_g, AP_size);

//    t_AP = MPI_Wtime() - t_AP;
//    print_time_ave(t_AP, "AB:", comm);

    delete[] mempool1;
    delete[] mempool2;
    delete[] mempool3;

#ifdef __DEBUG1__
//    print_vector(C->entry, -1, "AB = C", A->comm);
    if(verbose_fastmm){
        MPI_Barrier(comm); printf("compute_coarsen: step 5: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

//    if(rank==0) printf("\nAB:\n");
//    if(rank==0) dollar::text(std::cout);
//    dollar::clear();

    // -------------------------
    // set some of C parameters
    // -------------------------

    C->nnz_l = C->entry.size();
    MPI_Allreduce(&C->nnz_l, &C->nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

    C->Mbig = A->Mbig;
    C->split = A->split;
//    C->M = C->split[rank+1] - C->split[rank];
    C->M = A->M;
    C->M_old = C->M;

    C->comm = A->comm;
    C->comm_old = A->comm;
    C->active_old_comm = true;

    // set dense parameters
    C->density = ((double)C->nnz_g / C->Mbig) / C->Mbig;
//    C->switch_to_dense = switch_to_dense;
//    C->dense_threshold = dense_threshold;

    // set shrink parameters
//    C->last_M_shrink = A->last_M_shrink;
//    C->last_density_shrink = A->last_density_shrink;
//    C->cpu_shrink_thre1 = A->cpu_shrink_thre1; //todo: is this required?
//    if(A->cpu_shrink_thre2_next_level != -1) // this is -1 by default.
//        C->cpu_shrink_thre2 = A->cpu_shrink_thre2_next_level;
    //return these to default, since they have been used in the above part.
//    A->cpu_shrink_thre2_next_level = -1;
//    A->enable_shrink_next_level = false;

    if(C->active_minor){
        comm = C->comm;
        int rank_new;
        MPI_Comm_rank(C->comm, &rank_new);

#ifdef __DEBUG1__
//        C->print_info(-1);
//        C->print_entry(-1);
//        if(verbose_compute_coarsen){
//            MPI_Barrier(comm); printf("compute_coarsen: step 10: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

        // ********** decide about shrinking **********
        //---------------------------------------------
//        if(C->enable_shrink && C->enable_dummy_matvec && nprocs > 1){
//            MPI_Barrier(C->comm); if(rank_new==0) printf("start decide shrinking\n"); MPI_Barrier(C->comm);
//            C->matrix_setup_dummy();
//            C->compute_matvec_dummy_time();
//            C->decide_shrinking(A->matvec_dummy_time);
//            C->erase_after_decide_shrinking();
//            MPI_Barrier(C->comm); if(rank_new==0) printf("finish decide shrinking\n"); MPI_Barrier(C->comm);
//        }

//#ifdef __DEBUG1__
//        if(verbose_compute_coarsen){
//            MPI_Barrier(comm); printf("compute_coarsen: step 11: rank = %d\n", rank); MPI_Barrier(comm);}
//#endif

        // decide to partition based on number of rows or nonzeros.
//    if(switch_repartition && C->density >= repartition_threshold)
        if(switch_repartition && C->density >= repartition_threshold){
            if(rank==0) printf("equi-ROW partition for the next level: density = %f, repartition_threshold = %f \n", C->density, repartition_threshold);
            C->repartition_row(); // based on number of rows
        }else{
            C->repartition_nnz(); // based on number of nonzeros
        }

//#ifdef __DEBUG1__
//        if(verbose_compute_coarsen){
//            MPI_Barrier(comm); printf("compute_coarsen: step 12: rank = %d\n", rank); MPI_Barrier(comm);}
//#endif

//        repartition_u_shrink_prepare(grid);

        if(C->shrinked){
            C->shrink_cpu();
        }

//#ifdef __DEBUG1__
//        if(verbose_compute_coarsen){
//            MPI_Barrier(comm); printf("compute_coarsen: step 13: rank = %d\n", rank); MPI_Barrier(comm);}
//#endif

        if(C->active){
            C->matrix_setup();
//            C->matrix_setup_no_scale();

//            if(C->shrinked && C->enable_dummy_matvec)
//                C->compute_matvec_dummy_time();

//            if(switch_to_dense && C->density > dense_threshold){
//                if(rank==0) printf("Switch to dense: density = %f, dense_threshold = %f \n", C->density, dense_threshold);
//                C->generate_dense_matrix();
//            }
        }

#ifdef __DEBUG1__
//        C->print_info(-1);
//        C->print_entry(-1);
#endif

    }
*/
    return 0;
}

int saena_object::matmat_ave_orig_B(saena_matrix *A, saena_matrix *B, double &matmat_time){
    // This version works on general matrices.
    // this version is only for experiments.

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(!rank) printf("\n\nmatmat_ave_orig_B is not complete!\n");

    // =======================================
    // Convert A to CSC
    // =======================================

    auto Arv = new vecEntry[A->nnz_l]; // row and val
    auto Ac  = new index_t[A->Mbig+1]; // col_idx

    // todo: change to smart pointers
//    auto Arv = std::make_unique<vecEntry[]>(A->nnz_l); // row and val
//    auto Ac  = std::make_unique<index_t[]>(A->Mbig+1); // col_idx

    for(nnz_t i = 0; i < A->entry.size(); i++){
        Arv[i] = vecEntry(A->entry[i].row, A->entry[i].val);
    }

    std::fill(&Ac[0], &Ac[A->Mbig+1], 0);
    index_t *Ac_tmp = &Ac[1];
    for(auto ent:A->entry){
        Ac_tmp[ent.col]++;
    }

    for(nnz_t i = 0; i < A->Mbig; i++){
        Ac[i+1] += Ac[i];
    }

#ifdef __DEBUG1__
//    A->print_entry(0);
//    printf("A: nnz_l: %ld\tnnz_g: %ld\tM: %d\tM_big: %d\n", A->nnz_l, A->nnz_g, A->M, A->Mbig);
//    print_array(Ac, A->Mbig+1, 0, "Ac", comm);

//    std::cout << "\nA: nnz: " << A->nnz_l << std::endl ;
//    for(index_t j = 0; j < A->Mbig; j++){
//        for(index_t i = Ac[j]; i < Ac[j+1]; i++){
//            std::cout << std::setprecision(4) << Arv[i].row << "\t" << j << "\t" << Arv[i].val << std::endl;
//        }
//    }
#endif

    // =======================================
    // Convert B to CSC
    // =======================================

    auto Brv = new vecEntry[B->nnz_l]; // row and val
    auto Bc  = new index_t[B->Mbig+1];    // col_idx

    // todo: change to smart pointers
//    auto Brv = std::make_unique<vecEntry[]>(B->nnz_l); // row (actually col to have the transpose) and val
//    auto Bc  = std::make_unique<index_t[]>(B->M+1); // col_idx

    for(nnz_t i = 0; i < B->entry.size(); i++){
        Brv[i] = vecEntry(B->entry[i].row, B->entry[i].val);
    }

    std::fill(&Bc[0], &Bc[B->Mbig+1], 0);
    index_t *Bc_p = &Bc[1];
    for(auto ent:B->entry){
        Bc_p[ent.col]++;
    }

    for(nnz_t i = 0; i < B->Mbig; i++){
        Bc[i+1] += Bc[i];
    }

#ifdef __DEBUG1__
//    B->print_entry(0);
//    printf("B: nnz_l: %ld\tnnz_g: %ld\tM: %d\tM_big: %d\n", B->nnz_l, B->nnz_g, B->M, B->Mbig);
//    print_array(Bc, B->M+1, 0, "Bc", comm);
//
//    std::cout << "\nB: nnz: " << B->nnz_l << std::endl ;
//    for(index_t j = 0; j < B->Mbig; j++){
//        for(index_t i = Bc[j]; i < Bc[j+1]; i++){
//            std::cout << std::setprecision(4) << Brv[i].row << "\t" << j << "\t" << Brv[i].val << std::endl;
//        }
//    }
#endif

    // =======================================
    // Preallocate Memory
    // =======================================

    // mempool1: used for the dense buffer
    // mempool2: usages: for A: 1- nnzPerRow_left and 2- orig_row_idx. for B: 3- B_new_col_idx and 4- orig_col_idx
    // mempool2 size: 2 * A_row_size + 2 * B_col_size
    // important: it depends if B or transpose of B is being used for the multiplciation. if originl B is used then
    // B_col_size is B->Mbig, but for the transpos it is B->M, which is scalable.
    // mempool3: is used to store the the received part of B.

    index_t A_row_size = A->M;
    index_t B_col_size = B->Mbig; // for original B
//    index_t B_col_size = B->M;  // for when tranpose of B is used to do the multiplication.

    mempool1 = new value_t[matmat_size_thre2];
    mempool2 = new index_t[2 * A_row_size + 2 * B_col_size];
//    mempool3 = new cooEntry[B->nnz_max * 2];

//    mempool1 = std::make_unique<value_t[]>(matmat_size_thre2);
//    mempool2 = std::make_unique<index_t[]>(A->Mbig * 4);

    // =======================================
    // perform the multiplication
    // =======================================

    // todo: this version of fast_mm() considers B_row_size <- A_col_size and B_row_offset < A_row_offset.

//    std::vector<cooEntry> AB_temp;
//    fast_mm(Arv, Brv, AB_temp,
//            A->M, A->split[rank], A->Mbig, 0, B->M, B->split[rank],
//            &Ac[0], &Bc[0], A->comm);

#ifdef __DEBUG1__
//    print_vector(AB_temp, -1, "AB_temp", comm);
#endif

    // =======================================
    // sort and remove duplicates
    // =======================================

//    MPI_Barrier(comm);
//    t1 = MPI_Wtime();
/*
    std::sort(AB_temp.begin(), AB_temp.end());

    nnz_t AP_temp_size_minus1 = AB_temp.size()-1;
    std::vector<cooEntry> AB; // this won't be used.
    for(nnz_t i = 0; i < AB_temp.size(); i++){
        AB.emplace_back(AB_temp[i]);
        while(i < AP_temp_size_minus1 && AB_temp[i] == AB_temp[i+1]){ // values of entries with the same row and col should be added.
//            std::cout << AB_temp[i] << "\t" << AB_temp[i+1] << std::endl;
            AB.back().val += AB_temp[++i].val;
        }
    }

//    t1 = MPI_Wtime() - t1;
//    print_time_ave(t1, "AB:", comm);
*/
#ifdef __DEBUG1__
//    print_vector(AB, -1, "AB", comm);
//    writeMatrixToFile(AB, "Matlab/Saena/result", comm);
#endif

    // =======================================
    // finalize
    // =======================================

//    mat_send.clear();
//    mat_send.shrink_to_fit();
//    AB_temp.clear();
//    AB_temp.shrink_to_fit();

    delete []Arv;
    delete []Ac;
    delete []Brv;
    delete []Bc;

    delete[] mempool1;
    delete[] mempool2;
//    delete[] mempool3;

    return 0;
}


int saena_object::triple_mat_mult_old2(Grid *grid, std::vector<cooEntry_row> &RAP_row_sorted){

    saena_matrix    *A = grid->A;
    prolong_matrix  *P = &grid->P;
    restrict_matrix *R = &grid->R;
//    saena_matrix  *Ac = &grid->Ac;

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    if(!rank) printf("\n\ntriple_mat_mult_old2: update this function based on the new fast_mm()\n");

/*
    // *******************************************************
    // part 1: multiply: AP_temp = A_i * P_j. in which P_j = R_j_tranpose and 0 <= j < nprocs.
    // *******************************************************
//    double t_AP = MPI_Wtime();

    unsigned long send_size     = R->entry.size();
    unsigned long send_size_max = R->nnz_max;
//    MPI_Allreduce(&send_size, &send_size_max, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);

    // local transpose of R is being used to compute A*P. So R is transposed locally here.
//    std::vector<cooEntry> mat_send(R->entry.size());
    auto mat_send = new cooEntry[send_size_max];
    transpose_locally(&R->entry[0], R->entry.size(), R->splitNew[rank], &mat_send[0]);

#ifdef __DEBUG1__
//    print_vector(R->entry, -1, "R->entry", comm);
//    print_vector(R_tranpose, -1, "mat_send", comm);
#endif

//    std::vector<index_t> nnzPerCol_left(A->Mbig, 0);
//    for(nnz_t i = 0; i < A->entry.size(); i++){
//        nnzPerCol_left[A->entry[i].col]++;
//    }

#ifdef __DEBUG1__
//    print_vector(A->entry, 1, "A->entry", comm);
//    print_vector(nnzPerCol_left, 1, "nnzPerCol_left", comm);
#endif

    index_t *nnzPerColScan_left = &A->nnzPerColScan[0];
//    std::vector<index_t> nnzPerColScan_left(A->Mbig+1);
//    nnzPerColScan_left[0] = 0;
//    for(nnz_t i = 0; i < A->Mbig; i++){
//        nnzPerColScan_left[i+1] = nnzPerColScan_left[i] + nnzPerCol_left[i];
//    }

//    nnzPerCol_left.clear();
//    nnzPerCol_left.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(nnzPerColScan_left, -1, "nnzPerColScan_left", comm);

    // this is done in the for loop for all R_i's, including the local one.
//    std::vector<unsigned int> nnzPerCol_right(R->M, 0); // range of rows of R is range of cols of R_transpose.
//    for(nnz_t i = 0; i < R_tranpose.size(); i++){
//        nnzPerCol_right[R_tranpose[i].col]++;
//    }
//    std::vector<nnz_t> nnzPerColScan_right(P->M+1);
//    nnzPerColScan_right[0] = 0;
//    for(nnz_t i = 0; i < P->M; i++){
//        nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
//    }

//    print_vector(P->splitNew, -1, "P->splitNew", comm);

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 4: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // compute the maximum size for nnzPerCol_right and nnzPerColScan_right
    index_t mat_recv_M_max = R->M_max;
//    for(index_t i = 0; i < nprocs; i++){
//        if(P->splitNew[i+1] - P->splitNew[i] > mat_recv_M_max){
//            mat_recv_M_max = P->splitNew[i+1] - P->splitNew[i];
//        }
//    }

    // use this for fast_mm case1
//    std::unordered_map<index_t, value_t> map_matmat;
//    map_matmat.reserve(matmat_size_thre2);

    std::vector<index_t> nnzPerColScan_right(mat_recv_M_max + 1);
//    std::vector<index_t> nnzPerCol_right(mat_recv_M_max); // range of rows of R is range of cols of R_transpose.
    index_t *nnzPerCol_right   = &nnzPerColScan_right[1];
    index_t *nnzPerCol_right_p = &nnzPerCol_right[0]; // use this to avoid subtracting a fixed number,

    std::vector<cooEntry> AP_temp;
    AP_temp.reserve(A->nnz_l + R->nnz_l); // an estimate to reserve memory

//    printf("\n");
    if(nprocs > 1){

        int right_neighbor = (rank + 1)%nprocs;
        int left_neighbor  = rank - 1;
        if (left_neighbor < 0){
            left_neighbor += nprocs;
        }

        int owner;
        unsigned long recv_size;
//        std::vector<cooEntry> mat_recv;
        auto mat_recv = new cooEntry[send_size_max];
        index_t mat_recv_M;

        auto *requests = new MPI_Request[4];
        auto *statuses = new MPI_Status[4];

        for(int k = rank; k < rank+nprocs; k++){
            // This is overlapped. Both local and remote loops are done here.
            // The first iteration is the local loop. The rest are remote.
            // Send R_tranpose to the left_neighbor processor, receive R_tranpose from the right_neighbor.
            // In the next step: send R_tranpose that was received in the previous step to the left_neighbor processor,
            // receive R_tranpose from the right_neighbor. And so on.
            // --------------------------------------------------------------------

            // communicate size
            MPI_Irecv(&recv_size, 1, MPI_UNSIGNED_LONG, right_neighbor, right_neighbor, comm, requests);
            MPI_Isend(&send_size, 1, MPI_UNSIGNED_LONG, left_neighbor,  rank,           comm, requests+1);
            MPI_Waitall(1, requests, statuses);
//          printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
//            mat_recv.resize(recv_size);

#ifdef __DEBUG1__
//          print_vector(mat_recv, -1, "mat_recv", A->comm);
//          print_vector(mat_send, -1, "mat_send", A->comm);
#endif
            // communicate data
            MPI_Irecv(&mat_recv[0], recv_size, cooEntry::mpi_datatype(), right_neighbor, right_neighbor, comm, requests+2);
            MPI_Isend(&mat_send[0], send_size, cooEntry::mpi_datatype(), left_neighbor,  rank,           comm, requests+3);

            owner = k%nprocs;
            mat_recv_M = P->splitNew[owner + 1] - P->splitNew[owner];
//          printf("rank %d: owner = %d, mat_recv_M = %d, B_col_offset = %u \n", rank, owner, mat_recv_M, P->splitNew[owner]);

            std::fill(&nnzPerCol_right[0], &nnzPerCol_right[mat_recv_M], 0);
            nnzPerCol_right_p = &nnzPerCol_right[0] - P->splitNew[owner];
            for(nnz_t i = 0; i < send_size; i++){
                nnzPerCol_right_p[mat_send[i].col]++;
            }

//            nnzPerColScan_right[0] = 0;
            for(nnz_t i = 0; i < mat_recv_M; i++){
                nnzPerColScan_right[i+1] += nnzPerColScan_right[i];
            }

#ifdef __DEBUG1__
//          print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
//          print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

            if(A->entry.empty() || send_size==0){ // skip!
#ifdef __DEBUG1__
                if(verbose_compute_coarsen){
                    if(A->entry.empty()){
                        printf("\nskip: A->entry.size() == 0\n\n");
                    } else {
                        printf("\nskip: mat_send == 0\n\n");
                    }
                }
#endif
            } else {

                fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), send_size,
                        A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[owner],
                        &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
                        &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);
            }

            MPI_Waitall(3, requests+1, statuses+1);

//            mat_recv.swap(mat_send);
            std::swap(mat_send, mat_recv);
            send_size = recv_size;

#ifdef __DEBUG1__
//          print_vector(AP_temp, -1, "AP_temp", A->comm);
//          print_vector(mat_send, -1, "mat_send", A->comm);
//          print_vector(mat_recv, -1, "mat_recv", A->comm);
//          prev_owner = owner;
//          printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
#endif

        }

//        mat_recv.clear();
//        mat_recv.shrink_to_fit();
        delete [] mat_recv;
        delete [] requests;
        delete [] statuses;

    } else { // nprocs == 1 -> serial

        index_t mat_recv_M = P->splitNew[rank + 1] - P->splitNew[rank];

        std::fill(&nnzPerCol_right[0], &nnzPerCol_right[mat_recv_M], 0);
        nnzPerCol_right_p = &nnzPerCol_right[0] - P->splitNew[rank];
        for(nnz_t i = 0; i < send_size; i++){
            nnzPerCol_right_p[mat_send[i].col]++;
        }

//        nnzPerColScan_right[0] = 0;
        for(nnz_t i = 0; i < mat_recv_M; i++){
            nnzPerColScan_right[i+1] += nnzPerColScan_right[i];
        }

#ifdef __DEBUG1__
//          print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
//          print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

        if(A->entry.empty() || send_size == 0){ // skip!
#ifdef __DEBUG1__
            if(verbose_compute_coarsen){
                if(A->entry.empty()){
                    printf("\nskip: A->entry.size() == 0\n\n");
                } else {
                    printf("\nskip: mat_send == 0\n\n");
                }
            }
#endif
        } else {

//            double t1 = MPI_Wtime();

            fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), send_size,
                    A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[rank],
                    &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
                    &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//            double t2 = MPI_Wtime();
//            printf("\nfast_mm of AP_temp = %f\n", t2-t1);
        }
    }

    // todo: delete this after computing estimates for AP and RAP nnz.
//    nnz_t AP_temp_nnz_g_loc = AP_temp.size();
//    nnz_t AP_temp_nnz_g;
//    MPI_Allreduce(&AP_temp_nnz_g_loc, &AP_temp_nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

    std::sort(AP_temp.begin(), AP_temp.end());
    std::vector<cooEntry> AP;
    AP.reserve(AP_temp.size()/25); // 25 is not accurate.
    nnz_t AP_temp_size_minus1 = AP_temp.size()-1;
    for(nnz_t i = 0; i < AP_temp.size(); i++){
        AP.emplace_back(AP_temp[i]);
        while(i < AP_temp_size_minus1 && AP_temp[i] == AP_temp[i+1]){ // values of entries with the same row and col should be added.
//            std::cout << AP_temp[i] << "\t" << AP_temp[i+1] << std::endl;
            AP.back().val += AP_temp[++i].val;
        }
    }

    delete [] mat_send;
//    mat_send.clear();
//    mat_send.shrink_to_fit();
    AP_temp.clear();
    AP_temp.shrink_to_fit();

//    unsigned long AP_size_loc = AP.size(), AP_size;
//    MPI_Reduce(&AP_size_loc, &AP_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);
//    if(!rank) printf("A_nnz_g = %lu, \tP_nnz_g = %lu, \tAP_size = %lu\n", A->nnz_g, P->nnz_g, AP_size);

//    t_AP = MPI_Wtime() - t_AP;
//    print_time_ave(t_AP, "AP:", grid->A->comm);

#ifdef __DEBUG1__
//    print_vector(AP_temp, -1, "AP_temp", A->comm);
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 5: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

//    std::ofstream file("chrome.json");
//    dollar::chrome(file);
//    if(rank==0) printf("\nRA:\n");
//    if(rank==0) dollar::text(std::cout);
//    dollar::clear();

    // *******************************************************
    // part 2: multiply: R_i * (AP_temp)_i. in which R_i = P_i_tranpose
    // *******************************************************

    // local transpose of P is being used to compute R*(AP_temp). So P is transposed locally here.
    std::vector<cooEntry> P_tranpose(P->entry.size());
//    transpose_locally(&P->entry[0], P->entry.size(), P->split[rank], &P_tranpose[0]);

    // convert the indices to global
//    for(nnz_t i = 0; i < P_tranpose.size(); i++){
//        P_tranpose[i].col += P->split[rank];
//    }

    // compute nnzPerColScan_left for P_tranpose
//    nnzPerCol_left.assign(P->M, 0);
    std::vector<index_t> nnzPerColScan_left2(P->M + 1, 0);
    index_t *nnzPerCol_left = &nnzPerColScan_left2[1];
    index_t *nnzPerCol_left_p = &nnzPerCol_left[0] - P->split[rank];
//    for(nnz_t i = 0; i < P_tranpose.size(); i++){
//        nnzPerCol_left_p[P_tranpose[i].col]++;
//        nnzPerCol_left[P_tranpose[i].col - P->split[rank]]++;
//    }

    for(nnz_t i = 0; i < P->entry.size(); i++){
        P_tranpose[i] = cooEntry(P->entry[i].col, P->entry[i].row + P->split[rank], P->entry[i].val);
        nnzPerCol_left_p[P_tranpose[i].col]++;
    }

    std::sort(P_tranpose.begin(), P_tranpose.end());

#ifdef __DEBUG1__
//    print_vector(P->entry, -1, "P->entry", comm);
//    print_vector(P_tranpose, -1, "P_tranpose", comm);
#endif

//    nnzPerColScan_left.resize(P->M + 1);
//    nnzPerColScan_left[0] = 0;
    for(nnz_t i = 1; i < P->M + 1; i++){
        nnzPerColScan_left2[i] += nnzPerColScan_left2[i-1];
    }

//    nnzPerCol_left.clear();
//    nnzPerCol_left.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(nnzPerColScan_left2, -1, "nnzPerColScan_left2", comm);
#endif

    // compute nnzPerColScan_left for AP_temp
    nnzPerColScan_right.assign(P->Nbig + 1, 0);
    nnzPerCol_right = &nnzPerColScan_right[1];
//    nnzPerCol_right.assign(P->Nbig, 0);
    for(nnz_t i = 0; i < AP.size(); i++){
        nnzPerCol_right[AP[i].col]++;
    }

#ifdef __DEBUG1__
//    print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
#endif

//    nnzPerColScan_right[0] = 0;
    for(nnz_t i = 0; i < P->Nbig; i++){
        nnzPerColScan_right[i+1] += nnzPerColScan_right[i];
    }

//    nnzPerCol_right.clear();
//    nnzPerCol_right.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

    // multiply: R_i * (AP_temp)_i. in which R_i = P_i_tranpose
    // ===================================================

    std::vector<cooEntry> RAP_temp;
    RAP_temp.reserve(A->nnz_l/20 + R->nnz_l/20); // an estimate to reserve memory

    if(P_tranpose.empty() || AP.empty()){ // skip!
#ifdef __DEBUG1__
        if(verbose_compute_coarsen){
            if(P_tranpose.empty()){
                printf("\nskip: P_tranpose.size() == 0\n\n");
            } else {
                printf("\nskip: AP == 0\n\n");
            }
        }
#endif
    } else {

//        double t1 = MPI_Wtime();

        fast_mm(&P_tranpose[0], &AP[0], RAP_temp, P_tranpose.size(), AP.size(),
                P->Nbig, 0, P->M, P->split[rank], P->Nbig, 0,
                &nnzPerColScan_left2[0], &nnzPerColScan_left2[1],
                &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//        double t2 = MPI_Wtime();
//        printf("\nfast_mm of R(AP_temp) = %f \n", t2-t1);

    }

    // todo: delete this after computing estimates for AP and RAP nnz.
//    nnz_t RAP_temp_nnz_g_loc = RAP_temp.size();
//    nnz_t RAP_temp_nnz_g;
//    MPI_Allreduce(&RAP_temp_nnz_g_loc, &RAP_temp_nnz_g, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

    // free memory
    // -----------
    AP.clear();
    AP.shrink_to_fit();
    P_tranpose.clear();
    P_tranpose.shrink_to_fit();
//    nnzPerColScan_left.clear();
//    nnzPerColScan_left.shrink_to_fit();
    nnzPerColScan_right.clear();
    nnzPerColScan_right.shrink_to_fit();

//    if(rank==0) printf("\nRAP:\n");
//    if(rank==0) dollar::text(std::cout);
//    dollar::clear();

#ifdef __DEBUG1__
//    print_vector(RAP_temp, -1, "RAP_temp", A->comm);
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 6: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // remove local duplicates.
    // Entries should be sorted in row-major order first, since the matrix should be partitioned based on rows.
    // So cooEntry_row is used here. Remove local duplicates and put them in RAP_temp_row.
    nnz_t size_minus_1 = 0;
    if(!RAP_temp.empty()){
        size_minus_1 = RAP_temp.size() - 1;
    }

    std::vector<cooEntry_row> RAP_temp_row;
    for(nnz_t i = 0; i < RAP_temp.size(); i++){
        RAP_temp_row.emplace_back(cooEntry_row( RAP_temp[i].row, RAP_temp[i].col, RAP_temp[i].val ));
        while(i < size_minus_1 && RAP_temp[i] == RAP_temp[i+1]){ // values of entries with the same row and col should be added.
            i++;
            RAP_temp_row.back().val += RAP_temp[i].val;
        }
    }

    RAP_temp.clear();
    RAP_temp.shrink_to_fit();

#ifdef __DEBUG1__
//    if(!rank) printf("\nave_nnz: R: %lu \tA: %lu \tP: %lu \tAP_temp: %lu \tRAP_temp = %lu \n", R->nnz_g/nprocs, A->nnz_g/nprocs, P->nnz_g/nprocs, AP_temp_nnz_g/nprocs, RAP_temp_nnz_g/nprocs);
//    MPI_Barrier(comm); printf("rank %d: RAP_temp_row.size = %lu \n", rank, RAP_temp_row.size()); MPI_Barrier(comm);
//    print_vector(RAP_temp_row, -1, "RAP_temp_row", comm);
//    print_vector(P->splitNew, -1, "P->splitNew", comm);
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 7: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // sort globally
    // -------------
    par::sampleSort(RAP_temp_row, RAP_row_sorted, P->splitNew, comm);

    RAP_temp_row.clear();
    RAP_temp_row.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(RAP_row_sorted, -1, "RAP_row_sorted", A->comm);
//    MPI_Barrier(comm); printf("rank %d: RAP_row_sorted.size = %lu \n", rank, RAP_row_sorted.size()); MPI_Barrier(comm);

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 8: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

//    std::vector<cooEntry> RAP_sorted(RAP_row_sorted.size());
//    memcpy(&RAP_sorted[0], &RAP_row_sorted[0], RAP_row_sorted.size() * sizeof(cooEntry));
//    RAP_row_sorted.clear();
//    RAP_row_sorted.shrink_to_fit();

    // clear map_matmat and free memory
//    map_matmat.clear();
//    std::unordered_map<index_t, value_t> map_temp;
//    std::swap(map_matmat, map_temp);

    // todo: nnzPerColScan is not required after this function. find the best place to clear it.
//    A->nnzPerColScan.clear();
//    A->nnzPerColScan.shrink_to_fit();
*/
    return 0;
}

int saena_object::triple_mat_mult_old_RAP(Grid *grid, std::vector<cooEntry_row> &RAP_row_sorted){
/*
    saena_matrix *A    = grid->A;
    prolong_matrix *P  = &grid->P;
    restrict_matrix *R = &grid->R;
//    saena_matrix *Ac   = &grid->Ac;

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // *******************************************************
    // part 1: multiply: AP_temp = A_i * P_j. in which P_j = R_j_tranpose and 0 <= j < nprocs.
    // *******************************************************

//    double t_AP = MPI_Wtime();

    unsigned long send_size     = R->entry.size();
    unsigned long send_size_max = R->nnz_max;
//    MPI_Allreduce(&send_size, &send_size_max, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);

    // local transpose of R is being used to compute A*P. So R is transposed locally here.
//    std::vector<cooEntry> mat_send(R->entry.size());
    auto mat_send = new cooEntry[send_size_max];
    transpose_locally(&R->entry[0], R->entry.size(), R->splitNew[rank], &mat_send[0]);

#ifdef __DEBUG1__
//    print_vector(R->entry, -1, "R->entry", comm);
//    print_vector(R_tranpose, -1, "mat_send", comm);
#endif

//    std::vector<index_t> nnzPerCol_left(A->Mbig, 0);
//    for(nnz_t i = 0; i < A->entry.size(); i++){
//        nnzPerCol_left[A->entry[i].col]++;
//    }

#ifdef __DEBUG1__
//    print_vector(A->entry, 1, "A->entry", comm);
//    print_vector(nnzPerCol_left, 1, "nnzPerCol_left", comm);
#endif

    index_t *nnzPerColScan_left = &A->nnzPerColScan[0];
//    std::vector<index_t> nnzPerColScan_left(A->Mbig+1);
//    nnzPerColScan_left[0] = 0;
//    for(nnz_t i = 0; i < A->Mbig; i++){
//        nnzPerColScan_left[i+1] = nnzPerColScan_left[i] + nnzPerCol_left[i];
//    }

//    nnzPerCol_left.clear();
//    nnzPerCol_left.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(nnzPerColScan_left, -1, "nnzPerColScan_left", comm);

    // this is done in the for loop for all R_i's, including the local one.
//    std::vector<unsigned int> nnzPerCol_right(R->M, 0); // range of rows of R is range of cols of R_transpose.
//    for(nnz_t i = 0; i < R_tranpose.size(); i++){
//        nnzPerCol_right[R_tranpose[i].col]++;
//    }
//    std::vector<nnz_t> nnzPerColScan_right(P->M+1);
//    nnzPerColScan_right[0] = 0;
//    for(nnz_t i = 0; i < P->M; i++){
//        nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
//    }

//    print_vector(P->splitNew, -1, "P->splitNew", comm);

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 4: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // compute the maximum size for nnzPerCol_right and nnzPerColScan_right
    index_t mat_recv_M_max = R->M_max;
//    for(index_t i = 0; i < nprocs; i++){
//        if(P->splitNew[i+1] - P->splitNew[i] > mat_recv_M_max){
//            mat_recv_M_max = P->splitNew[i+1] - P->splitNew[i];
//        }
//    }

    // use this for fast_mm case1
//    std::unordered_map<index_t, value_t> map_matmat;
//    map_matmat.reserve(matmat_size_thre2);

    std::vector<index_t> nnzPerColScan_right(mat_recv_M_max + 1);
//    std::vector<index_t> nnzPerCol_right(mat_recv_M_max); // range of rows of R is range of cols of R_transpose.
    index_t *nnzPerCol_right = &nnzPerColScan_right[1];
    index_t *nnzPerCol_right_p = &nnzPerCol_right[0]; // use this to avoid subtracting a fixed number,

    std::vector<cooEntry> AP_temp;

//    printf("\n");
    if(nprocs > 1){

        int right_neighbor = (rank + 1)%nprocs;
        int left_neighbor  = rank - 1;
        if (left_neighbor < 0){
            left_neighbor += nprocs;
        }

        int owner;
        unsigned long recv_size;
//        std::vector<cooEntry> mat_recv;
        auto mat_recv = new cooEntry[send_size_max];
        index_t mat_recv_M;

        auto *requests = new MPI_Request[4];
        auto *statuses = new MPI_Status[4];

        for(int k = rank; k < rank+nprocs; k++){
            // This is overlapped. Both local and remote loops are done here.
            // The first iteration is the local loop. The rest are remote.
            // Send R_tranpose to the left_neighbor processor, receive R_tranpose from the right_neighbor.
            // In the next step: send R_tranpose that was received in the previous step to the left_neighbor processor,
            // receive R_tranpose from the right_neighbor. And so on.
            // --------------------------------------------------------------------

            // communicate size
            MPI_Irecv(&recv_size, 1, MPI_UNSIGNED_LONG, right_neighbor, right_neighbor, comm, requests);
            MPI_Isend(&send_size, 1, MPI_UNSIGNED_LONG, left_neighbor,  rank,           comm, requests+1);
            MPI_Waitall(1, requests, statuses);
//          printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
//            mat_recv.resize(recv_size);

#ifdef __DEBUG1__
//          print_vector(mat_recv, -1, "mat_recv", A->comm);
//          print_vector(mat_send, -1, "mat_send", A->comm);
#endif
            // communicate data
            MPI_Irecv(&mat_recv[0], recv_size, cooEntry::mpi_datatype(), right_neighbor, right_neighbor, comm, requests+2);
            MPI_Isend(&mat_send[0], send_size, cooEntry::mpi_datatype(), left_neighbor,  rank,           comm, requests+3);

            owner = k%nprocs;
            mat_recv_M = P->splitNew[owner + 1] - P->splitNew[owner];
//          printf("rank %d: owner = %d, mat_recv_M = %d, B_col_offset = %u \n", rank, owner, mat_recv_M, P->splitNew[owner]);

            std::fill(&nnzPerCol_right[0], &nnzPerCol_right[mat_recv_M], 0);
            nnzPerCol_right_p = &nnzPerCol_right[0] - P->splitNew[owner];
            for(nnz_t i = 0; i < send_size; i++){
                nnzPerCol_right_p[mat_send[i].col]++;
            }

//            nnzPerColScan_right[0] = 0;
            for(nnz_t i = 0; i < mat_recv_M; i++){
                nnzPerColScan_right[i+1] += nnzPerColScan_right[i];
            }

#ifdef __DEBUG1__
//          print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
//          print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

            if(A->entry.empty() || send_size==0){ // skip!
#ifdef __DEBUG1__
                if(verbose_compute_coarsen){
                    if(A->entry.empty()){
                        printf("\nskip: A->entry.size() == 0\n\n");
                    } else {
                        printf("\nskip: mat_send == 0\n\n");
                    }
                }
#endif
            } else {

                fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), send_size,
                        A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[owner],
                        &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
                        &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//                fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), mat_send.size(),
//                        A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[owner],
//                        &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
//                        &nnzPerColScan_right[0], &nnzPerColScan_right[1], map_matmat, A->comm);

            }

            MPI_Waitall(3, requests+1, statuses+1);

//            mat_recv.swap(mat_send);
            std::swap(mat_send, mat_recv);
            send_size = recv_size;

#ifdef __DEBUG1__
//          print_vector(AP_temp, -1, "AP_temp", A->comm);
//          print_vector(mat_send, -1, "mat_send", A->comm);
//          print_vector(mat_recv, -1, "mat_recv", A->comm);
//          prev_owner = owner;
//          printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
#endif

        }

//        mat_recv.clear();
//        mat_recv.shrink_to_fit();
        delete [] mat_recv;
        delete [] requests;
        delete [] statuses;

    } else { // nprocs == 1 -> serial

        index_t mat_recv_M = P->splitNew[rank + 1] - P->splitNew[rank];

        std::fill(&nnzPerCol_right[0], &nnzPerCol_right[mat_recv_M], 0);
        nnzPerCol_right_p = &nnzPerCol_right[0] - P->splitNew[rank];
        for(nnz_t i = 0; i < send_size; i++){
            nnzPerCol_right_p[mat_send[i].col]++;
        }

//        nnzPerColScan_right[0] = 0;
        for(nnz_t i = 0; i < mat_recv_M; i++){
            nnzPerColScan_right[i+1] += nnzPerColScan_right[i];
        }

#ifdef __DEBUG1__
//          print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
//          print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

        if(A->entry.empty() || send_size == 0){ // skip!
#ifdef __DEBUG1__
            if(verbose_compute_coarsen){
                if(A->entry.empty()){
                    printf("\nskip: A->entry.size() == 0\n\n");
                } else {
                    printf("\nskip: mat_send == 0\n\n");
                }
            }
#endif
        } else {

//            double t1 = MPI_Wtime();

            fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), send_size,
                    A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[rank],
                    &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
                    &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//            fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), mat_send.size(),
//                    A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[rank],
//                    &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
//                    &nnzPerColScan_right[0], &nnzPerColScan_right[1], map_matmat, A->comm);

//            double t2 = MPI_Wtime();
//            printf("\nfast_mm of AP_temp = %f\n", t2-t1);
        }
    }

    std::sort(AP_temp.begin(), AP_temp.end());
    std::vector<cooEntry> AP;
    nnz_t AP_temp_size_minus1 = AP_temp.size()-1;
    for(nnz_t i = 0; i < AP_temp.size(); i++){
        AP.emplace_back(AP_temp[i]);
        while(i < AP_temp_size_minus1 && AP_temp[i] == AP_temp[i+1]){ // values of entries with the same row and col should be added.
//            std::cout << AP_temp[i] << "\t" << AP_temp[i+1] << std::endl;
            AP.back().val += AP_temp[++i].val;
        }
    }

    delete [] mat_send;
//    mat_send.clear();
//    mat_send.shrink_to_fit();
    AP_temp.clear();
    AP_temp.shrink_to_fit();

//    unsigned long AP_size_loc = AP.size(), AP_size;
//    MPI_Reduce(&AP_size_loc, &AP_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);
//    if(!rank) printf("A_nnz_g = %lu, \tP_nnz_g = %lu, \tAP_size = %lu\n", A->nnz_g, P->nnz_g, AP_size);

//    t_AP = MPI_Wtime() - t_AP;
//    print_time_ave(t_AP, "AP:\n", grid->A->comm);

#ifdef __DEBUG1__
//    print_vector(AP_temp, -1, "AP_temp", A->comm);
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 5: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

//    std::ofstream file("chrome.json");
//    dollar::chrome(file);
//    if(rank==0) printf("\nRA:\n");
//    if(rank==0) dollar::text(std::cout);
//    dollar::clear();

    // *******************************************************
    // part 2: multiply: R_i * (AP_temp)_i. in which R_i = P_i_tranpose
    // *******************************************************

    // local transpose of P is being used to compute R*(AP_temp). So P is transposed locally here.
    std::vector<cooEntry> P_tranpose(P->entry.size());
//    transpose_locally(&P->entry[0], P->entry.size(), P->split[rank], &P_tranpose[0]);

    // convert the indices to global
//    for(nnz_t i = 0; i < P_tranpose.size(); i++){
//        P_tranpose[i].col += P->split[rank];
//    }

#ifdef __DEBUG1__
//    print_vector(P->entry, -1, "P->entry", comm);
//    print_vector(P_tranpose, -1, "P_tranpose", comm);
#endif

    // compute nnzPerColScan_left for P_tranpose
//    nnzPerCol_left.assign(P->M, 0);
    std::vector<index_t> nnzPerColScan_left2(P->M + 1, 0);
    index_t *nnzPerCol_left = &nnzPerColScan_left2[1];
    index_t *nnzPerCol_left_p = &nnzPerCol_left[0] - P->split[rank];
//    for(nnz_t i = 0; i < P_tranpose.size(); i++){
//        nnzPerCol_left_p[P_tranpose[i].col]++;
//        nnzPerCol_left[P_tranpose[i].col - P->split[rank]]++;
//    }

    for(nnz_t i = 0; i < P->entry.size(); i++){
        P_tranpose[i] = cooEntry(P->entry[i].col, P->entry[i].row + P->split[rank], P->entry[i].val);
        nnzPerCol_left_p[P_tranpose[i].col]++;
    }

    std::sort(P_tranpose.begin(), P_tranpose.end());

//    nnzPerColScan_left.resize(P->M + 1);
//    nnzPerColScan_left[0] = 0;
    for(nnz_t i = 1; i < P->M + 1; i++){
        nnzPerColScan_left2[i] += nnzPerColScan_left2[i-1];
    }

//    nnzPerCol_left.clear();
//    nnzPerCol_left.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(nnzPerColScan_left, -1, "nnzPerColScan_left", comm);
#endif

    // compute nnzPerColScan_left for AP_temp
    nnzPerColScan_right.assign(P->Nbig + 1, 0);
    nnzPerCol_right = &nnzPerColScan_right[1];
//    nnzPerCol_right.assign(P->Nbig, 0);
    for(nnz_t i = 0; i < AP.size(); i++){
        nnzPerCol_right[AP[i].col]++;
    }

#ifdef __DEBUG1__
//    print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
#endif

//    nnzPerColScan_right[0] = 0;
    for(nnz_t i = 0; i < P->Nbig; i++){
        nnzPerColScan_right[i+1] += nnzPerColScan_right[i];
    }

//    nnzPerCol_right.clear();
//    nnzPerCol_right.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

    // multiply: R_i * (AP_temp)_i. in which R_i = P_i_tranpose
    // ===================================================

//    printf("\n");
    std::vector<cooEntry> RAP_temp;

    if(P_tranpose.empty() || AP.empty()){ // skip!
#ifdef __DEBUG1__
        if(verbose_compute_coarsen){
            if(P_tranpose.empty()){
                printf("\nskip: P_tranpose.size() == 0\n\n");
            } else {
                printf("\nskip: AP == 0\n\n");
            }
        }
#endif
    } else {

//        double t1 = MPI_Wtime();

        fast_mm(&P_tranpose[0], &AP[0], RAP_temp, P_tranpose.size(), AP.size(),
                P->Nbig, 0, P->M, P->split[rank], P->Nbig, 0,
                &nnzPerColScan_left2[0], &nnzPerColScan_left2[1],
                &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//        fast_mm(&P_tranpose[0], &AP[0], RAP_temp, P_tranpose.size(), AP.size(),
//                P->Nbig, 0, P->M, P->split[rank], P->Nbig, 0,
//                &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
//                &nnzPerColScan_right[0], &nnzPerColScan_right[1], map_matmat, A->comm);

//        double t2 = MPI_Wtime();
//        printf("\nfast_mm of R(AP_temp) = %f \n", t2-t1);

    }

    // free memory
    // -----------
    AP.clear();
    AP.shrink_to_fit();
    P_tranpose.clear();
    P_tranpose.shrink_to_fit();
//    nnzPerColScan_left.clear();
//    nnzPerColScan_left.shrink_to_fit();
    nnzPerColScan_right.clear();
    nnzPerColScan_right.shrink_to_fit();

//    if(rank==0) printf("\nRAP:\n");
//    if(rank==0) dollar::text(std::cout);
//    dollar::clear();

#ifdef __DEBUG1__
//    print_vector(RAP_temp, -1, "RAP_temp", A->comm);
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 6: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // remove local duplicates.
    // Entries should be sorted in row-major order first, since the matrix should be partitioned based on rows.
    // So cooEntry_row is used here. Remove local duplicates and put them in RAP_temp_row.
    nnz_t size_minus_1 = 0;
    if(!RAP_temp.empty()){
        size_minus_1 = RAP_temp.size() - 1;
    }

    std::vector<cooEntry_row> RAP_temp_row;
    for(nnz_t i = 0; i < RAP_temp.size(); i++){
        RAP_temp_row.emplace_back(cooEntry_row( RAP_temp[i].row, RAP_temp[i].col, RAP_temp[i].val ));
        while(i < size_minus_1 && RAP_temp[i] == RAP_temp[i+1]){ // values of entries with the same row and col should be added.
            i++;
            RAP_temp_row.back().val += RAP_temp[i].val;
        }
    }

    RAP_temp.clear();
    RAP_temp.shrink_to_fit();

#ifdef __DEBUG1__
//    MPI_Barrier(comm); printf("rank %d: RAP_temp_row.size = %lu \n", rank, RAP_temp_row.size()); MPI_Barrier(comm);
//    print_vector(RAP_temp_row, -1, "RAP_temp_row", comm);
//    print_vector(P->splitNew, -1, "P->splitNew", comm);
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 7: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // sort globally
    // -------------
    par::sampleSort(RAP_temp_row, RAP_row_sorted, P->splitNew, comm);

    RAP_temp_row.clear();
    RAP_temp_row.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(RAP_row_sorted, -1, "RAP_row_sorted", A->comm);
//    MPI_Barrier(comm); printf("rank %d: RAP_row_sorted.size = %lu \n", rank, RAP_row_sorted.size()); MPI_Barrier(comm);

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 8: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

//    std::vector<cooEntry> RAP_sorted(RAP_row_sorted.size());
//    memcpy(&RAP_sorted[0], &RAP_row_sorted[0], RAP_row_sorted.size() * sizeof(cooEntry));
//    RAP_row_sorted.clear();
//    RAP_row_sorted.shrink_to_fit();

    // clear map_matmat and free memory
//    map_matmat.clear();
//    std::unordered_map<index_t, value_t> map_temp;
//    std::swap(map_matmat, map_temp);

    A->nnzPerColScan.clear();
    A->nnzPerColScan.shrink_to_fit();

*/

    return 0;
}

int saena_object::triple_mat_mult_no_overlap(Grid *grid, std::vector<cooEntry_row> &RAP_row_sorted){
/*
    saena_matrix *A    = grid->A;
    prolong_matrix *P  = &grid->P;
    restrict_matrix *R = &grid->R;
//    saena_matrix *Ac   = &grid->Ac;

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    coarsen_time = 0;
    MPI_Barrier(comm); //todo: delete
    double t1 = MPI_Wtime(); //todo: delete

    // *******************************************************
    // part 1: multiply: AP_temp = A_i * P_j. in which P_j = R_j_tranpose and 0 <= j < nprocs.
    // *******************************************************

    unsigned long send_size_max;
    unsigned long send_size = R->entry.size();
    MPI_Allreduce(&send_size, &send_size_max, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);

    // local transpose of R is being used to compute A*P. So R is transposed locally here.
    std::vector<cooEntry> mat_send(R->entry.size());
    transpose_locally(&R->entry[0], R->entry.size(), R->splitNew[rank], &mat_send[0]);

#ifdef __DEBUG1__
//    print_vector(R->entry, -1, "R->entry", comm);
//    print_vector(R_tranpose, -1, "mat_send", comm);
#endif

    std::vector<index_t> nnzPerCol_left(A->Mbig, 0);
    for(nnz_t i = 0; i < A->entry.size(); i++){
        nnzPerCol_left[A->entry[i].col]++;
    }

#ifdef __DEBUG1__
//    print_vector(A->entry, 1, "A->entry", comm);
//    print_vector(nnzPerCol_left, 1, "nnzPerCol_left", comm);
#endif

    std::vector<index_t> nnzPerColScan_left(A->Mbig+1);
    nnzPerColScan_left[0] = 0;
    for(nnz_t i = 0; i < A->Mbig; i++){
        nnzPerColScan_left[i+1] = nnzPerColScan_left[i] + nnzPerCol_left[i];
    }

    nnzPerCol_left.clear();
//    nnzPerCol_left.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(nnzPerColScan_left, -1, "nnzPerColScan_left", comm);

    // this is done in the for loop for all R_i's, including the local one.
//    std::vector<unsigned int> nnzPerCol_right(R->M, 0); // range of rows of R is range of cols of R_transpose.
//    for(nnz_t i = 0; i < R_tranpose.size(); i++){
//        nnzPerCol_right[R_tranpose[i].col]++;
//    }
//    std::vector<nnz_t> nnzPerColScan_right(P->M+1);
//    nnzPerColScan_right[0] = 0;
//    for(nnz_t i = 0; i < P->M; i++){
//        nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
//    }

//    print_vector(P->splitNew, -1, "P->splitNew", comm);

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 4: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // compute the maximum size for nnzPerCol_right and nnzPerColScan_right
    index_t mat_recv_M_max = 0;
    for(index_t i = 0; i < nprocs; i++){
        if(P->splitNew[i+1] - P->splitNew[i] > mat_recv_M_max){
            mat_recv_M_max = P->splitNew[i+1] - P->splitNew[i];
        }
    }

    // use this for fast_mm case1
//    std::unordered_map<index_t, value_t> map_matmat;
//    map_matmat.reserve(matmat_size_thre2);

    std::vector<index_t> nnzPerCol_right(mat_recv_M_max); // range of rows of R is range of cols of R_transpose.
    index_t *nnzPerCol_right_p = &nnzPerCol_right[0]; // use this to avoid subtracting a fixed number,
    std::vector<index_t> nnzPerColScan_right(mat_recv_M_max + 1);
    std::vector<cooEntry> AP_temp;

    t1 = MPI_Wtime() - t1; //todo: delete
    coarsen_time += print_time_ave_consecutive(t1, A->comm); //todo: delete
//    MPI_Barrier(comm);
//    if(rank==0) printf("\nRA:\n");

    if(nprocs > 1){

        t1 = MPI_Wtime(); //todo: delete

        int right_neighbor = (rank + 1)%nprocs;
        int left_neighbor  = rank - 1;
        if (left_neighbor < 0){
            left_neighbor += nprocs;
        }

        int owner;
        unsigned long send_size = mat_send.size();
        unsigned long recv_size;
        std::vector<cooEntry> mat_recv;
        index_t mat_recv_M = P->splitNew[rank + 1] - P->splitNew[rank];

        std::fill(&nnzPerCol_right[0], &nnzPerCol_right[mat_recv_M], 0);
        nnzPerCol_right_p = &nnzPerCol_right[0] - P->splitNew[rank];
        for(nnz_t i = 0; i < mat_send.size(); i++){
            nnzPerCol_right_p[mat_send[i].col]++;
        }

        nnzPerColScan_right[0] = 0;
        for(nnz_t i = 0; i < mat_recv_M; i++){
            nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
        }

#ifdef __DEBUG1__
//          print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
//          print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

        if(A->entry.empty() || mat_send.empty()){ // skip!
#ifdef __DEBUG1__
            if(verbose_compute_coarsen){
                if(A->entry.empty()){
                    printf("\nskip: A->entry.size() == 0\n\n");
                } else {
                    printf("\nskip: mat_send == 0\n\n");
                }
            }
#endif
        } else {

//            double t1 = MPI_Wtime();

            fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), mat_send.size(),
                    A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[rank],
                    &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
                    &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//            fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), mat_send.size(),
//                    A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[rank],
//                    &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
//                    &nnzPerColScan_right[0], &nnzPerColScan_right[1], map_matmat, A->comm);

//            double t2 = MPI_Wtime();
//            printf("\nfast_mm of AP_temp = %f\n", t2-t1);
//            print_time_ave_consecutive(t2-t1, A->comm);
        }

        t1 = MPI_Wtime() - t1; //todo: delete
        coarsen_time += print_time_ave_consecutive(t1, A->comm); //todo: delete

        auto *requests = new MPI_Request[4];
        auto *statuses = new MPI_Status[4];

        for(int k = rank+1; k < rank+nprocs; k++){
            // This is overlapped. Both local and remote loops are done here.
            // The first iteration is the local loop. The rest are remote.
            // Send R_tranpose to the left_neighbor processor, receive R_tranpose from the right_neighbor.
            // In the next step: send R_tranpose that was received in the previous step to the left_neighbor processor,
            // receive R_tranpose from the right_neighbor. And so on.
            // --------------------------------------------------------------------

            // communicate size
            MPI_Irecv(&recv_size, 1, MPI_UNSIGNED_LONG, right_neighbor, right_neighbor, comm, requests);
            MPI_Isend(&send_size, 1, MPI_UNSIGNED_LONG, left_neighbor,  rank,           comm, requests+1);
            MPI_Waitall(2, requests, statuses);
//          printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
            mat_recv.resize(recv_size);

#ifdef __DEBUG1__
//          print_vector(mat_recv, -1, "mat_recv", A->comm);
//          print_vector(mat_send, -1, "mat_send", A->comm);
#endif

            // communicate data
            MPI_Irecv(&mat_recv[0], recv_size, cooEntry::mpi_datatype(), right_neighbor, right_neighbor, comm, requests+2);
            MPI_Isend(&mat_send[0], send_size, cooEntry::mpi_datatype(), left_neighbor,  rank,           comm, requests+3);
            MPI_Waitall(2, requests+2, statuses+2);

            t1 = MPI_Wtime(); //todo: delete
            owner = k%nprocs;
            mat_recv_M = P->splitNew[owner + 1] - P->splitNew[owner];
//          printf("rank %d: owner = %d, mat_recv_M = %d, B_col_offset = %u \n", rank, owner, mat_recv_M, P->splitNew[owner]);

            std::fill(&nnzPerCol_right[0], &nnzPerCol_right[mat_recv_M], 0);
            nnzPerCol_right_p = &nnzPerCol_right[0] - P->splitNew[owner];
            for(nnz_t i = 0; i < mat_recv.size(); i++){
                nnzPerCol_right_p[mat_recv[i].col]++;
            }

            nnzPerColScan_right[0] = 0;
            for(nnz_t i = 0; i < mat_recv_M; i++){
                nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
            }

#ifdef __DEBUG1__
//          print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
//          print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

            if(A->entry.empty() || mat_send.empty()){ // skip!
#ifdef __DEBUG1__
                if(verbose_compute_coarsen){
                    if(A->entry.empty()){
                        printf("\nskip: A->entry.size() == 0\n\n");
                    } else {
                        printf("\nskip: mat_send == 0\n\n");
                    }
                }
#endif
            } else {

                fast_mm(&A->entry[0], &mat_recv[0], AP_temp, A->entry.size(), recv_size,
                        A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[owner],
                        &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
                        &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//                fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), mat_send.size(),
//                        A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[owner],
//                        &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
//                        &nnzPerColScan_right[0], &nnzPerColScan_right[1], map_matmat, A->comm);

            }

            mat_recv.swap(mat_send);
            send_size = recv_size;

            t1 = MPI_Wtime() - t1; //todo: delete
            coarsen_time += print_time_ave_consecutive(t1, A->comm); //todo: delete

#ifdef __DEBUG1__
//          print_vector(AP_temp, -1, "AP_temp", A->comm);
//          print_vector(mat_send, -1, "mat_send", A->comm);
//          print_vector(mat_recv, -1, "mat_recv", A->comm);
//          prev_owner = owner;
//          printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
#endif

        }

        mat_recv.clear();
        mat_recv.shrink_to_fit();

        delete [] requests;
        delete [] statuses;

    } else { // nprocs == 1 -> serial

        t1 = MPI_Wtime(); //todo: delete
        index_t mat_recv_M = P->splitNew[rank + 1] - P->splitNew[rank];

        std::fill(&nnzPerCol_right[0], &nnzPerCol_right[mat_recv_M], 0);
        nnzPerCol_right_p = &nnzPerCol_right[0] - P->splitNew[rank];
        for(nnz_t i = 0; i < mat_send.size(); i++){
            nnzPerCol_right_p[mat_send[i].col]++;
        }

        nnzPerColScan_right[0] = 0;
        for(nnz_t i = 0; i < mat_recv_M; i++){
            nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
        }

#ifdef __DEBUG1__
//          print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
//          print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

        if(A->entry.empty() || mat_send.empty()){ // skip!
#ifdef __DEBUG1__
            if(verbose_compute_coarsen){
                if(A->entry.empty()){
                    printf("\nskip: A->entry.size() == 0\n\n");
                } else {
                    printf("\nskip: mat_send == 0\n\n");
                }
            }
#endif
        } else {

//            double t1 = MPI_Wtime();

            fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), mat_send.size(),
                    A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[rank],
                    &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
                    &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//            fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), mat_send.size(),
//                    A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[rank],
//                    &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
//                    &nnzPerColScan_right[0], &nnzPerColScan_right[1], map_matmat, A->comm);

//            double t2 = MPI_Wtime();
//            printf("\nfast_mm of AP_temp = %f\n", t2-t1);
        }
        t1 = MPI_Wtime() - t1; //todo: delete
        coarsen_time += print_time_ave_consecutive(t1, A->comm); //todo: delete
    }

    t1 = MPI_Wtime(); //todo: delete
    std::sort(AP_temp.begin(), AP_temp.end());
    std::vector<cooEntry> AP;
    nnz_t AP_temp_size_minus1 = AP_temp.size()-1;
    for(nnz_t i = 0; i < AP_temp.size(); i++){
        AP.emplace_back(AP_temp[i]);
        while(i < AP_temp_size_minus1 && AP_temp[i] == AP_temp[i+1]){ // values of entries with the same row and col should be added.
//            std::cout << AP_temp[i] << "\t" << AP_temp[i+1] << std::endl;
            AP.back().val += AP_temp[++i].val;
        }
    }

    mat_send.clear();
    mat_send.shrink_to_fit();
    AP_temp.clear();
    AP_temp.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(AP_temp, -1, "AP_temp", A->comm);
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 5: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

//    std::ofstream file("chrome.json");
//    dollar::chrome(file);
//    if(rank==0) printf("\nRA:\n");
//    if(rank==0) dollar::text(std::cout);
//    dollar::clear();

    // *******************************************************
    // part 2: multiply: R_i * (AP_temp)_i. in which R_i = P_i_tranpose
    // *******************************************************

    // local transpose of P is being used to compute R*(AP_temp). So P is transposed locally here.
    std::vector<cooEntry> P_tranpose(P->entry.size());
    transpose_locally(P->entry, P->entry.size(), P_tranpose);

    // convert the indices to global
    for(nnz_t i = 0; i < P_tranpose.size(); i++){
        P_tranpose[i].col += P->split[rank];
    }

#ifdef __DEBUG1__
//    print_vector(P->entry, -1, "P->entry", comm);
//    print_vector(P_tranpose, -1, "P_tranpose", comm);
#endif

    // compute nnzPerColScan_left for P_tranpose
    nnzPerCol_left.assign(P->M, 0);
    index_t *nnzPerCol_left_p = &nnzPerCol_left[0] - P->split[rank];
    for(nnz_t i = 0; i < P_tranpose.size(); i++){
        nnzPerCol_left_p[P_tranpose[i].col]++;
//        nnzPerCol_left[P_tranpose[i].col - P->split[rank]]++;
    }

    nnzPerColScan_left.resize(P->M + 1);
    nnzPerColScan_left[0] = 0;
    for(nnz_t i = 0; i < P->M; i++){
        nnzPerColScan_left[i+1] = nnzPerColScan_left[i] + nnzPerCol_left[i];
    }

    nnzPerCol_left.clear();
    nnzPerCol_left.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(nnzPerColScan_left, -1, "nnzPerColScan_left", comm);
#endif

    // compute nnzPerColScan_left for AP_temp
    nnzPerCol_right.assign(P->Nbig, 0);
    for(nnz_t i = 0; i < AP.size(); i++){
        nnzPerCol_right[AP[i].col]++;
    }

#ifdef __DEBUG1__
//    print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
#endif

    nnzPerColScan_right.resize(P->Nbig+1);
    nnzPerColScan_right[0] = 0;
    for(nnz_t i = 0; i < P->Nbig; i++){
        nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
    }

    nnzPerCol_right.clear();
    nnzPerCol_right.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

    // multiply: R_i * (AP_temp)_i. in which R_i = P_i_tranpose
    // ===================================================

    std::vector<cooEntry> RAP_temp;
//    if(rank==0) printf("\nRAP:\n");

    if(P_tranpose.empty() || AP.empty()){ // skip!
#ifdef __DEBUG1__
        if(verbose_compute_coarsen){
            if(P_tranpose.empty()){
                printf("\nskip: P_tranpose.size() == 0\n\n");
            } else {
                printf("\nskip: AP == 0\n\n");
            }
        }
#endif
    } else {

//        double t1 = MPI_Wtime();
        fast_mm(&P_tranpose[0], &AP[0], RAP_temp, P_tranpose.size(), AP.size(),
                P->Nbig, 0, P->M, P->split[rank], P->Nbig, 0,
                &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
                &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//        fast_mm(&P_tranpose[0], &AP[0], RAP_temp, P_tranpose.size(), AP.size(),
//                P->Nbig, 0, P->M, P->split[rank], P->Nbig, 0,
//                &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
//                &nnzPerColScan_right[0], &nnzPerColScan_right[1], map_matmat, A->comm);
//        double t2 = MPI_Wtime();
//        printf("\nfast_mm of R(AP_temp) = %f \n", t2-t1);

    }

    // free memory
    // -----------
    AP.clear();
    AP.shrink_to_fit();
    P_tranpose.clear();
    P_tranpose.shrink_to_fit();
    nnzPerColScan_left.clear();
    nnzPerColScan_left.shrink_to_fit();
    nnzPerColScan_right.clear();
    nnzPerColScan_right.shrink_to_fit();

//    if(rank==0) printf("\nRAP:\n");
//    if(rank==0) dollar::text(std::cout);
//    dollar::clear();

#ifdef __DEBUG1__
//    print_vector(RAP_temp, -1, "RAP_temp", A->comm);
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 6: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // remove local duplicates.
    // Entries should be sorted in row-major order first, since the matrix should be partitioned based on rows.
    // So cooEntry_row is used here. Remove local duplicates and put them in RAP_temp_row.
    nnz_t size_minus_1 = 0;
    if(!RAP_temp.empty()){
        size_minus_1 = RAP_temp.size() - 1;
    }

    std::vector<cooEntry_row> RAP_temp_row;
    for(nnz_t i = 0; i < RAP_temp.size(); i++){
        RAP_temp_row.emplace_back(cooEntry_row( RAP_temp[i].row, RAP_temp[i].col, RAP_temp[i].val ));
        while(i < size_minus_1 && RAP_temp[i] == RAP_temp[i+1]){ // values of entries with the same row and col should be added.
            i++;
            RAP_temp_row.back().val += RAP_temp[i].val;
        }
    }

    RAP_temp.clear();
    RAP_temp.shrink_to_fit();

#ifdef __DEBUG1__
//    MPI_Barrier(comm); printf("rank %d: RAP_temp_row.size = %lu \n", rank, RAP_temp_row.size()); MPI_Barrier(comm);
//    print_vector(RAP_temp_row, -1, "RAP_temp_row", comm);
//    print_vector(P->splitNew, -1, "P->splitNew", comm);
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 7: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // sort globally
    // -------------
    par::sampleSort(RAP_temp_row, RAP_row_sorted, P->splitNew, comm);

    RAP_temp_row.clear();
    RAP_temp_row.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(RAP_row_sorted, -1, "RAP_row_sorted", A->comm);
//    MPI_Barrier(comm); printf("rank %d: RAP_row_sorted.size = %lu \n", rank, RAP_row_sorted.size()); MPI_Barrier(comm);

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 8: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

//    std::vector<cooEntry> RAP_sorted(RAP_row_sorted.size());
//    memcpy(&RAP_sorted[0], &RAP_row_sorted[0], RAP_row_sorted.size() * sizeof(cooEntry));
//    RAP_row_sorted.clear();
//    RAP_row_sorted.shrink_to_fit();

    // clear map_matmat and free memory
//    map_matmat.clear();
//    std::unordered_map<index_t, value_t> map_temp;
//    std::swap(map_matmat, map_temp);

    // remove duplicates.
    cooEntry temp;
    std::vector<cooEntry> Ac_dummy;
    for(nnz_t i = 0; i < RAP_row_sorted.size(); i++){
        temp = cooEntry(RAP_row_sorted[i].row, RAP_row_sorted[i].col, RAP_row_sorted[i].val);
        while(i < size_minus_1 && RAP_row_sorted[i] == RAP_row_sorted[i+1]){ // values of entries with the same row and col should be added.
            ++i;
            temp.val += RAP_row_sorted[i].val;
        }
        Ac_dummy.emplace_back( temp );
    }

//    RAP_row_sorted.clear();
//    RAP_row_sorted.shrink_to_fit();

    t1 = MPI_Wtime() - t1; //todo: delete
    coarsen_time += print_time_ave_consecutive(t1, A->comm); //todo: delete
    if(!rank) printf("coarsen_time:\n%f\n", coarsen_time);
*/
    return 0;
}

int saena_object::triple_mat_mult_basic(Grid *grid, std::vector<cooEntry_row> &RAP_row_sorted){

    saena_matrix *A    = grid->A;
    prolong_matrix *P  = &grid->P;
    restrict_matrix *R = &grid->R;
//    saena_matrix *Ac   = &grid->Ac;

    MPI_Comm comm = A->comm;
    int nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // *******************************************************
    // part 1: multiply: AP_temp = A_i * P_j. in which P_j = R_j_tranpose and 0 <= j < nprocs.
    // *******************************************************

    // local transpose of R is being used to compute A*P. So R is transposed locally here.
    std::vector<cooEntry> mat_send(R->entry.size());
    transpose_locally(&R->entry[0], R->entry.size(), R->splitNew[rank], &mat_send[0]);

#ifdef __DEBUG1__
//    print_vector(R->entry, -1, "R->entry", comm);
//    print_vector(R_tranpose, -1, "mat_send", comm);
#endif

    std::vector<index_t> nnzPerCol_left(A->Mbig, 0);
    for(nnz_t i = 0; i < A->entry.size(); i++){
        nnzPerCol_left[A->entry[i].col]++;
    }

#ifdef __DEBUG1__
//    print_vector(A->entry, 1, "A->entry", comm);
//    print_vector(nnzPerCol_left, 1, "nnzPerCol_left", comm);
#endif

    std::vector<index_t> nnzPerColScan_left(A->Mbig+1);
    nnzPerColScan_left[0] = 0;
    for(nnz_t i = 0; i < A->Mbig; i++){
        nnzPerColScan_left[i+1] = nnzPerColScan_left[i] + nnzPerCol_left[i];
    }

    nnzPerCol_left.clear();
//    nnzPerCol_left.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(nnzPerColScan_left, -1, "nnzPerColScan_left", comm);

    // this is done in the for loop for all R_i's, including the local one.
//    std::vector<unsigned int> nnzPerCol_right(R->M, 0); // range of rows of R is range of cols of R_transpose.
//    for(nnz_t i = 0; i < R_tranpose.size(); i++){
//        nnzPerCol_right[R_tranpose[i].col]++;
//    }
//    std::vector<nnz_t> nnzPerColScan_right(P->M+1);
//    nnzPerColScan_right[0] = 0;
//    for(nnz_t i = 0; i < P->M; i++){
//        nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
//    }

//    print_vector(P->splitNew, -1, "P->splitNew", comm);

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 4: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // compute the maximum size for nnzPerCol_right and nnzPerColScan_right
    index_t mat_recv_M_max = 0;
    for(index_t i = 0; i < nprocs; i++){
        if(P->splitNew[i+1] - P->splitNew[i] > mat_recv_M_max){
            mat_recv_M_max = P->splitNew[i+1] - P->splitNew[i];
        }
    }

    // use this for fast_mm case1
//    std::unordered_map<index_t, value_t> map_matmat;
//    map_matmat.reserve(matmat_size_thre2);

    std::vector<index_t> nnzPerCol_right(mat_recv_M_max); // range of rows of R is range of cols of R_transpose.
    index_t *nnzPerCol_right_p = &nnzPerCol_right[0]; // use this to avoid subtracting a fixed number,
    std::vector<index_t> nnzPerColScan_right(mat_recv_M_max + 1);
    std::vector<cooEntry> AP_temp;

    if(nprocs > 1){
        int right_neighbor = (rank + 1)%nprocs;
        int left_neighbor  = rank - 1;
        if (left_neighbor < 0){
            left_neighbor += nprocs;
        }

        int owner;
        unsigned long send_size = mat_send.size();
        unsigned long recv_size;
        std::vector<cooEntry> mat_recv;
        index_t mat_recv_M;

        auto *requests = new MPI_Request[4];
        auto *statuses = new MPI_Status[4];

        for(int k = rank; k < rank+nprocs; k++){
            // This is overlapped. Both local and remote loops are done here.
            // The first iteration is the local loop. The rest are remote.
            // Send R_tranpose to the left_neighbor processor, receive R_tranpose from the right_neighbor.
            // In the next step: send R_tranpose that was received in the previous step to the left_neighbor processor,
            // receive R_tranpose from the right_neighbor. And so on.
            // --------------------------------------------------------------------

            // communicate size
            MPI_Irecv(&recv_size, 1, MPI_UNSIGNED_LONG, right_neighbor, right_neighbor, comm, requests);
            MPI_Isend(&send_size, 1, MPI_UNSIGNED_LONG, left_neighbor,  rank,           comm, requests+1);
            MPI_Waitall(1, requests, statuses);
//          printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
            mat_recv.resize(recv_size);

#ifdef __DEBUG1__
//          print_vector(mat_recv, -1, "mat_recv", A->comm);
//          print_vector(mat_send, -1, "mat_send", A->comm);
#endif
            // communicate data
            MPI_Irecv(&mat_recv[0], recv_size, cooEntry::mpi_datatype(), right_neighbor, right_neighbor, comm, requests+2);
            MPI_Isend(&mat_send[0], send_size, cooEntry::mpi_datatype(), left_neighbor,  rank,           comm, requests+3);

            owner = k%nprocs;
            mat_recv_M = P->splitNew[owner + 1] - P->splitNew[owner];
//          printf("rank %d: owner = %d, mat_recv_M = %d, B_col_offset = %u \n", rank, owner, mat_recv_M, P->splitNew[owner]);

            std::fill(&nnzPerCol_right[0], &nnzPerCol_right[mat_recv_M], 0);
            nnzPerCol_right_p = &nnzPerCol_right[0] - P->splitNew[owner];
            for(nnz_t i = 0; i < mat_send.size(); i++){
                nnzPerCol_right_p[mat_send[i].col]++;
            }

            nnzPerColScan_right[0] = 0;
            for(nnz_t i = 0; i < mat_recv_M; i++){
                nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
            }

#ifdef __DEBUG1__
//          print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
//          print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

            if(A->entry.empty() || mat_send.empty()){ // skip!
#ifdef __DEBUG1__
                if(verbose_compute_coarsen){
                    if(A->entry.empty()){
                        printf("\nskip: A->entry.size() == 0\n\n");
                    } else {
                        printf("\nskip: mat_send == 0\n\n");
                    }
                }
#endif
            } else {

                fast_mm_basic(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), mat_send.size(),
                              A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[owner],
                              &nnzPerColScan_left[0], &nnzPerColScan_left[1],
                              &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//                fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), mat_send.size(),
//                        A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[owner],
//                        &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
//                        &nnzPerColScan_right[0], &nnzPerColScan_right[1], map_matmat, A->comm);

            }

            MPI_Waitall(3, requests+1, statuses+1);

            mat_recv.swap(mat_send);
            send_size = recv_size;

#ifdef __DEBUG1__
//          print_vector(AP_temp, -1, "AP_temp", A->comm);
//          print_vector(mat_send, -1, "mat_send", A->comm);
//          print_vector(mat_recv, -1, "mat_recv", A->comm);
//          prev_owner = owner;
//          printf("rank %d: recv_size = %lu, send_size = %lu \n", rank, recv_size, send_size);
#endif

        }

        mat_recv.clear();
        mat_recv.shrink_to_fit();

        delete [] requests;
        delete [] statuses;

    } else { // nprocs == 1 -> serial

        index_t mat_recv_M = P->splitNew[rank + 1] - P->splitNew[rank];

        std::fill(&nnzPerCol_right[0], &nnzPerCol_right[mat_recv_M], 0);
        nnzPerCol_right_p = &nnzPerCol_right[0] - P->splitNew[rank];
        for(nnz_t i = 0; i < mat_send.size(); i++){
            nnzPerCol_right_p[mat_send[i].col]++;
        }

        nnzPerColScan_right[0] = 0;
        for(nnz_t i = 0; i < mat_recv_M; i++){
            nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
        }

#ifdef __DEBUG1__
//          print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
//          print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

        if(A->entry.empty() || mat_send.empty()){ // skip!
#ifdef __DEBUG1__
            if(verbose_compute_coarsen){
                if(A->entry.empty()){
                    printf("\nskip: A->entry.size() == 0\n\n");
                } else {
                    printf("\nskip: mat_send == 0\n\n");
                }
            }
#endif
        } else {

//            double t1 = MPI_Wtime();

            fast_mm_basic(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), mat_send.size(),
                          A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[rank],
                          &nnzPerColScan_left[0], &nnzPerColScan_left[1],
                          &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//            fast_mm(&A->entry[0], &mat_send[0], AP_temp, A->entry.size(), mat_send.size(),
//                    A->M, A->split[rank], A->Mbig, 0, mat_recv_M, P->splitNew[rank],
//                    &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
//                    &nnzPerColScan_right[0], &nnzPerColScan_right[1], map_matmat, A->comm);

//            double t2 = MPI_Wtime();
//            printf("\nfast_mm of AP_temp = %f\n", t2-t1);
        }
    }

    std::sort(AP_temp.begin(), AP_temp.end());
    std::vector<cooEntry> AP;
    nnz_t AP_temp_size_minus1 = AP_temp.size()-1;
    for(nnz_t i = 0; i < AP_temp.size(); i++){
        AP.emplace_back(AP_temp[i]);
        while(i < AP_temp_size_minus1 && AP_temp[i] == AP_temp[i+1]){ // values of entries with the same row and col should be added.
//            std::cout << AP_temp[i] << "\t" << AP_temp[i+1] << std::endl;
            AP.back().val += AP_temp[++i].val;
        }
    }

    mat_send.clear();
    mat_send.shrink_to_fit();
    AP_temp.clear();
    AP_temp.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(AP_temp, -1, "AP_temp", A->comm);
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 5: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

//    std::ofstream file("chrome.json");
//    dollar::chrome(file);
//    if(rank==0) printf("\nRA:\n");
//    if(rank==0) dollar::text(std::cout);
//    dollar::clear();

    // *******************************************************
    // part 2: multiply: R_i * (AP_temp)_i. in which R_i = P_i_tranpose
    // *******************************************************

    // local transpose of P is being used to compute R*(AP_temp). So P is transposed locally here.
    std::vector<cooEntry> P_tranpose(P->entry.size());
    transpose_locally(&P->entry[0], P->entry.size(), &P_tranpose[0]);

    // convert the indices to global
    for(nnz_t i = 0; i < P_tranpose.size(); i++){
        P_tranpose[i].col += P->split[rank];
    }

#ifdef __DEBUG1__
//    print_vector(P->entry, -1, "P->entry", comm);
//    print_vector(P_tranpose, -1, "P_tranpose", comm);
#endif

    // compute nnzPerColScan_left for P_tranpose
    nnzPerCol_left.assign(P->M, 0);
    index_t *nnzPerCol_left_p = &nnzPerCol_left[0] - P->split[rank];
    for(nnz_t i = 0; i < P_tranpose.size(); i++){
        nnzPerCol_left_p[P_tranpose[i].col]++;
//        nnzPerCol_left[P_tranpose[i].col - P->split[rank]]++;
    }

    nnzPerColScan_left.resize(P->M + 1);
    nnzPerColScan_left[0] = 0;
    for(nnz_t i = 0; i < P->M; i++){
        nnzPerColScan_left[i+1] = nnzPerColScan_left[i] + nnzPerCol_left[i];
    }

    nnzPerCol_left.clear();
    nnzPerCol_left.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(nnzPerColScan_left, -1, "nnzPerColScan_left", comm);
#endif

    // compute nnzPerColScan_left for AP_temp
    nnzPerCol_right.assign(P->Nbig, 0);
    for(nnz_t i = 0; i < AP.size(); i++){
        nnzPerCol_right[AP[i].col]++;
    }

#ifdef __DEBUG1__
//    print_vector(nnzPerCol_right, -1, "nnzPerCol_right", comm);
#endif

    nnzPerColScan_right.resize(P->Nbig+1);
    nnzPerColScan_right[0] = 0;
    for(nnz_t i = 0; i < P->Nbig; i++){
        nnzPerColScan_right[i+1] = nnzPerColScan_right[i] + nnzPerCol_right[i];
    }

    nnzPerCol_right.clear();
    nnzPerCol_right.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(nnzPerColScan_right, -1, "nnzPerColScan_right", comm);
#endif

    // multiply: R_i * (AP_temp)_i. in which R_i = P_i_tranpose
    // ===================================================

    std::vector<cooEntry> RAP_temp;

    if(P_tranpose.empty() || AP.empty()){ // skip!
#ifdef __DEBUG1__
        if(verbose_compute_coarsen){
            if(P_tranpose.empty()){
                printf("\nskip: P_tranpose.size() == 0\n\n");
            } else {
                printf("\nskip: AP == 0\n\n");
            }
        }
#endif
    } else {

//        double t1 = MPI_Wtime();
        fast_mm_basic(&P_tranpose[0], &AP[0], RAP_temp, P_tranpose.size(), AP.size(),
                      P->Nbig, 0, P->M, P->split[rank], P->Nbig, 0,
                      &nnzPerColScan_left[0], &nnzPerColScan_left[1],
                      &nnzPerColScan_right[0], &nnzPerColScan_right[1], A->comm);

//        fast_mm(&P_tranpose[0], &AP[0], RAP_temp, P_tranpose.size(), AP.size(),
//                P->Nbig, 0, P->M, P->split[rank], P->Nbig, 0,
//                &nnzPerColScan_left[0],  &nnzPerColScan_left[1],
//                &nnzPerColScan_right[0], &nnzPerColScan_right[1], map_matmat, A->comm);
//        double t2 = MPI_Wtime();
//        printf("\nfast_mm of R(AP_temp) = %f \n", t2-t1);

    }

    // free memory
    // -----------
    AP.clear();
    AP.shrink_to_fit();
    P_tranpose.clear();
    P_tranpose.shrink_to_fit();
    nnzPerColScan_left.clear();
    nnzPerColScan_left.shrink_to_fit();
    nnzPerColScan_right.clear();
    nnzPerColScan_right.shrink_to_fit();

//    if(rank==0) printf("\nRAP:\n");
//    if(rank==0) dollar::text(std::cout);
//    dollar::clear();

#ifdef __DEBUG1__
//    print_vector(RAP_temp, -1, "RAP_temp", A->comm);
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 6: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // remove local duplicates.
    // Entries should be sorted in row-major order first, since the matrix should be partitioned based on rows.
    // So cooEntry_row is used here. Remove local duplicates and put them in RAP_temp_row.
    nnz_t size_minus_1 = 0;
    if(!RAP_temp.empty()){
        size_minus_1 = RAP_temp.size() - 1;
    }

    std::vector<cooEntry_row> RAP_temp_row;
    for(nnz_t i = 0; i < RAP_temp.size(); i++){
        RAP_temp_row.emplace_back(cooEntry_row( RAP_temp[i].row, RAP_temp[i].col, RAP_temp[i].val ));
        while(i < size_minus_1 && RAP_temp[i] == RAP_temp[i+1]){ // values of entries with the same row and col should be added.
            i++;
            RAP_temp_row.back().val += RAP_temp[i].val;
        }
    }

    RAP_temp.clear();
    RAP_temp.shrink_to_fit();

#ifdef __DEBUG1__
//    MPI_Barrier(comm); printf("rank %d: RAP_temp_row.size = %lu \n", rank, RAP_temp_row.size()); MPI_Barrier(comm);
//    print_vector(RAP_temp_row, -1, "RAP_temp_row", comm);
//    print_vector(P->splitNew, -1, "P->splitNew", comm);
    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 7: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

    // sort globally
    // -------------
    par::sampleSort(RAP_temp_row, RAP_row_sorted, P->splitNew, comm);

    RAP_temp_row.clear();
    RAP_temp_row.shrink_to_fit();

#ifdef __DEBUG1__
//    print_vector(RAP_row_sorted, -1, "RAP_row_sorted", A->comm);
//    MPI_Barrier(comm); printf("rank %d: RAP_row_sorted.size = %lu \n", rank, RAP_row_sorted.size()); MPI_Barrier(comm);

    if(verbose_compute_coarsen){
        MPI_Barrier(comm); printf("compute_coarsen: step 8: rank = %d\n", rank); MPI_Barrier(comm);}
#endif

//    std::vector<cooEntry> RAP_sorted(RAP_row_sorted.size());
//    memcpy(&RAP_sorted[0], &RAP_row_sorted[0], RAP_row_sorted.size() * sizeof(cooEntry));
//    RAP_row_sorted.clear();
//    RAP_row_sorted.shrink_to_fit();

    // clear map_matmat and free memory
//    map_matmat.clear();
//    std::unordered_map<index_t, value_t> map_temp;
//    std::swap(map_matmat, map_temp);

    return 0;
}