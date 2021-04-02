
/**
  @file parUtils.C
  @author Rahul S. Sampath, rahul.sampath@gmail.com
  @author Hari Sundar, hsundar@gmail.com
  */

#include "mpi.h"
#include "binUtils.h"
#include "dtypes.h"
#include "parUtils.h"

#define __DEBUG_PAR__

#ifdef __DEBUG__
#ifndef __DEBUG_PAR__
#define __DEBUG_PAR__
#endif
#endif

namespace par {

    unsigned int splitCommBinary( MPI_Comm orig_comm, MPI_Comm *new_comm) {
    int npes, rank;

    MPI_Group  orig_group, new_group;

    MPI_Comm_size(orig_comm, &npes);
    MPI_Comm_rank(orig_comm, &rank);

    unsigned int splitterRank = binOp::getPrevHighestPowerOfTwo(npes);

    int *ranksAsc, *ranksDesc;
    //Determine sizes for the 2 groups
    ranksAsc = new int[splitterRank];
    ranksDesc = new int[( npes - splitterRank)];

    int numAsc = 0;
    int numDesc = ( npes - splitterRank - 1);

    //This is the main mapping between old ranks and new ranks.
    for(int i=0; i<npes; i++) {
      if( static_cast<unsigned int>(i) < splitterRank) {
        ranksAsc[numAsc] = i;
        numAsc++;
      }else {
        ranksDesc[numDesc] = i;
        numDesc--;
      }
    }//end for i

    MPI_Comm_group(orig_comm, &orig_group);

    /* Divide tasks into two distinct groups based upon rank */
    if (static_cast<unsigned int>(rank) < splitterRank) {
      MPI_Group_incl(orig_group, splitterRank, ranksAsc, &new_group);
    }else {
      MPI_Group_incl(orig_group, (npes-splitterRank), ranksDesc, &new_group);
    }

    MPI_Comm_create(orig_comm, new_group, new_comm);

    delete [] ranksAsc;
    ranksAsc = NULL;

    delete [] ranksDesc;
    ranksDesc = NULL;

    return splitterRank;
    }//end function

    unsigned int splitCommBinaryNoFlip( MPI_Comm orig_comm, MPI_Comm *new_comm) {
    int npes, rank;

    MPI_Group  orig_group, new_group;

    MPI_Comm_size(orig_comm, &npes);
    MPI_Comm_rank(orig_comm, &rank);

    unsigned int splitterRank =  binOp::getPrevHighestPowerOfTwo(npes);

    int *ranksAsc, *ranksDesc;
    //Determine sizes for the 2 groups 
    ranksAsc = new int[splitterRank];
    ranksDesc = new int[( npes - splitterRank)];

    int numAsc = 0;
    int numDesc = 0; //( npes - splitterRank - 1);

    //This is the main mapping between old ranks and new ranks.
    for(int i = 0; i < npes; i++) {
      if(static_cast<unsigned int>(i) < splitterRank) {
        ranksAsc[numAsc] = i;
        numAsc++;
      }else {
        ranksDesc[numDesc] = i;
        numDesc++;
      }
    }//end for i

    MPI_Comm_group(orig_comm, &orig_group);

    /* Divide tasks into two distinct groups based upon rank */
    if (static_cast<unsigned int>(rank) < splitterRank) {
      MPI_Group_incl(orig_group, splitterRank, ranksAsc, &new_group);
    }else {
      MPI_Group_incl(orig_group, (npes-splitterRank), ranksDesc, &new_group);
    }

    MPI_Comm_create(orig_comm, new_group, new_comm);

    delete [] ranksAsc;
    ranksAsc = NULL;
    
    delete [] ranksDesc;
    ranksDesc = NULL;

    return splitterRank;
  }//end function

    //create Comm groups and remove empty processors...
    void splitComm2way(bool iAmEmpty, MPI_Comm * new_comm, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif

      MPI_Group  orig_group, new_group;
    int size;
    MPI_Comm_size(comm, &size);

    bool* isEmptyList = new bool[size];
    par::Mpi_Allgather<bool>(&iAmEmpty, isEmptyList, 1, comm);

    int numActive=0, numIdle=0;
    for(int i = 0; i < size; i++) {
      if(isEmptyList[i]) {
        numIdle++;
      }else {
        numActive++;
      }
    }//end for i

    int* ranksActive = new int[numActive];
    int* ranksIdle = new int[numIdle];

    numActive=0;
    numIdle=0;
    for(int i = 0; i < size; i++) {
      if(isEmptyList[i]) {
        ranksIdle[numIdle] = i;
        numIdle++;
      }else {
        ranksActive[numActive] = i;
        numActive++;
      }
    }//end for i

    delete [] isEmptyList;	
    isEmptyList = NULL;

    /* Extract the original group handle */
    MPI_Comm_group(comm, &orig_group);

    /* Divide tasks into two distinct groups based upon rank */
    if (!iAmEmpty) {
      MPI_Group_incl(orig_group, numActive, ranksActive, &new_group);
    }else {
      MPI_Group_incl(orig_group, numIdle, ranksIdle, &new_group);
    }

    /* Create new communicator */
    MPI_Comm_create(comm, new_group, new_comm);

    delete [] ranksActive;
    ranksActive = NULL;
    
    delete [] ranksIdle;
    ranksIdle = NULL;

  }//end function

    void splitCommUsingSplittingRank(int splittingRank, MPI_Comm* new_comm, MPI_Comm comm) {
#ifdef __PROFILE_WITH_BARRIER__
    MPI_Barrier(comm);
#endif

      MPI_Group  orig_group, new_group;
    int size;
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int* ranksActive = new int[splittingRank];
    int* ranksIdle = new int[size - splittingRank];

    for(int i = 0; i < splittingRank; i++) {
      ranksActive[i] = i;
    }

    for(int i = splittingRank; i < size; i++) {
      ranksIdle[i - splittingRank] = i;
    }

    /* Extract the original group handle */
    MPI_Comm_group(comm, &orig_group);

    /* Divide tasks into two distinct groups based upon rank */
    if (rank < splittingRank) {
      MPI_Group_incl(orig_group, splittingRank, ranksActive, &new_group);
    }else {
      MPI_Group_incl(orig_group, (size - splittingRank), ranksIdle, &new_group);
    }

    /* Create new communicator */
    MPI_Comm_create(comm, new_group, new_comm);

    delete [] ranksActive;
    ranksActive = NULL;
    
    delete [] ranksIdle;
    ranksIdle = NULL;

  }//end function

    //create Comm groups and remove empty processors...
    int splitComm2way(const bool* isEmptyList, MPI_Comm * new_comm, MPI_Comm comm) {
      
    MPI_Group  orig_group, new_group;
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    int numActive=0, numIdle=0;
    for(int i = 0; i < size; i++) {
      if(isEmptyList[i]) {
        numIdle++;
      }else {
        numActive++;
      }
    }//end for i

    int* ranksActive = new int[numActive];
    int* ranksIdle = new int[numIdle];

    numActive=0;
    numIdle=0;
    for(int i = 0; i < size; i++) {
      if(isEmptyList[i]) {
        ranksIdle[numIdle] = i;
        numIdle++;
      }else {
        ranksActive[numActive] = i;
        numActive++;
      }
    }//end for i

    /* Extract the original group handle */
    MPI_Comm_group(comm, &orig_group);

    /* Divide tasks into two distinct groups based upon rank */
    if (!isEmptyList[rank]) {
      MPI_Group_incl(orig_group, numActive, ranksActive, &new_group);
    }else {
      MPI_Group_incl(orig_group, numIdle, ranksIdle, &new_group);
    }

    /* Create new communicator */
    MPI_Comm_create(comm, new_group, new_comm);

    delete [] ranksActive;
    ranksActive = NULL;
    
    delete [] ranksIdle;
    ranksIdle = NULL;

    return 0;
  }//end function

	int AdjustCommunicationPattern(std::vector<int>& send_sizes, std::vector<int>& send_partners,
	                               std::vector<int>& recv_sizes, std::vector<int>& recv_partners, MPI_Comm comm)
	{
    int npes;
    int rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &npes);
		
		unsigned int k = send_sizes.size();
		
		// do scans ...
		DendroIntL lsz[k];
		DendroIntL gsz[k],  gscan[k];
		
		for(size_t i = 0; i < send_sizes.size(); ++i) {
			lsz[i] = send_sizes[i];
		}
		par::Mpi_Scan<DendroIntL>( lsz, gscan, k, MPI_SUM, comm);
		
		if (rank == npes-1) {
			for(size_t i = 0; i < k; ++i) {
				gsz[i] = gscan[i];
			}
		}		
		// broadcast from last proc to get total counts, per segment ...
		par::Mpi_Bcast<DendroIntL>( gsz, k, npes-1, comm);
		
		DendroIntL segment_p0[k];
		for(size_t i = 0; i < k; ++i) {
			segment_p0[i] = (i*npes)/k;
		}
		
		/*
		 * -- Dividing into k segments, so each segment will have npes/k procs.
		 * -- Each proc will have gsz[i]/(npes/k) elements.
		 * -- rank of proc which will get i-th send_buff is,
		 *        -- segment_p0[i] + gscan[i]    
		 */
				
		// figure out send_partners for k sends
		// send_partners.clear();
		for(size_t i = 0; i < k; ++i) {
			int new_part;
			int seg_npes  =   ( (i == k-1) ? npes - segment_p0[i] : segment_p0[i+1]-segment_p0[i] );
			int overhang  =   gsz[i] % seg_npes;
			DendroIntL rank_mid = gscan[i] - lsz[i]/2;
			if ( rank_mid < overhang*(gsz[i]/seg_npes + 1)) {
				new_part = segment_p0[i] + rank_mid/(gsz[i]/seg_npes + 1);
			} else {
				new_part = segment_p0[i] + (rank_mid - overhang)/(gsz[i]/seg_npes);	
			}
			send_partners[i] = new_part; 
		}
		
		int idx=0;
		if (send_partners[0] == rank) {
			send_sizes[0] = 0;
		}
		for(size_t i = 1; i < k; ++i)
		{
			if (send_partners[i] == rank) {
				send_sizes[i] = 0;
				idx = i;
				continue;
			}
			if (send_partners[i] == send_partners[i-1]) {
				send_sizes[idx] += lsz[i];
				send_sizes[i]=0;
			} else {
					idx = i;
			}
		}
		
		// let procs know you will be sending to them ...
	
		// try MPI one sided comm
		MPI_Win win;
		int *rcv;
	  MPI_Alloc_mem(sizeof(int)*npes, MPI_INFO_NULL, &rcv);
		for(size_t i = 0; i < npes; ++i) rcv[i] = 0;
		
		MPI_Win_create(rcv, npes, sizeof(int), MPI_INFO_NULL,  MPI_COMM_WORLD, &win);
		
		
		MPI_Win_fence(MPI_MODE_NOPRECEDE, win);
		for (size_t i = 0; i < send_sizes.size(); i++) 
		{
			if (send_sizes[i]) {
		    MPI_Put(&(send_sizes[i]), 1, MPI_INT, send_partners[i], rank, 1, MPI_INT, win);
			}
		}	 
		MPI_Win_fence((MPI_MODE_NOSTORE | MPI_MODE_NOSUCCEED), win);
		// figure out recv partners and sizes ...
		recv_sizes.clear(); recv_partners.clear();
		for(size_t i = 0; i < npes; ++i)
		{
			if (rcv[i]) {
				recv_partners.push_back(i);
				recv_sizes.push_back(rcv[i]);
			} 
		}
		
		MPI_Win_free(&win);
	  MPI_Free_mem(rcv);
		 
		return 1;
	}


	// this version of sampleSort redistributes arr to have entries with row indices between split[rank] and
	// split[rank+1] on processor rank.
	// this only works on cooEntry_row datatype, because the ordering should be row-major.
	// To convert this function to template T, instad of cooEntry_row, a new compare function can be implemented for
	// comparing arr[i] with split[k].
    int sampleSort(std::vector<cooEntry_row>& arr, std::vector<cooEntry_row> & SortedElem, std::vector<index_t> &split,
                   MPI_Comm comm){

        int npes = 0;
        MPI_Comm_size(comm, &npes);

        //--
        int rank = 0;
        MPI_Comm_rank(comm, &rank);

//      std::cout << rank << " : " << __func__ << arr.size() << std::endl;
        if(!rank) printf("sampleSort - step1\n");

//        assert(!arr.empty());

        if (npes == 1) {
//            std::cout <<" have to use seq. sort"
//            <<" since npes = 1 . inpSize: "<<(arr.size()) <<std::endl;
//            std::sort(arr.begin(), arr.end());
//            omp_par::merge_sort(arr.begin(),arr.end());
            std::sort(arr.begin(), arr.end());
            SortedElem = arr;
            return 0;
        }

        int myrank = 0;
        MPI_Comm_rank(comm, &myrank);

        long nelem = arr.size();
        long nelemCopy = nelem;
        long totSize = 0;
        par::Mpi_Allreduce<long>(&nelemCopy, &totSize, 1, MPI_SUM, comm);
        if(!rank) printf("nelem local = %ld, totSize = %ld\n", nelem, totSize);

        DendroIntL npesLong = npes;
        const DendroIntL FIVE = 5;
        if(!rank) printf("sampleSort - step2\n");

        if(totSize < (FIVE * npesLong * npesLong)) {
//            if(!myrank) {
//                std::cout <<" Using bitonic sort since totSize < (5*(npes^2)). totSize: "
//                <<totSize<<" npes: "<<npes <<std::endl;
//            }

            if(!rank) printf("sampleSort - step2-2 partitionW\n");
            par::partitionW<cooEntry_row>(arr, nullptr, comm);
            if(!rank) printf("sampleSort - step3\n");

#ifdef __DEBUG_PAR__
            MPI_Barrier(comm);
        if(!myrank) {
          std::cout<<"SampleSort (small n): Stage-1 passed."<<std::endl;
        }
        MPI_Barrier(comm);
#endif
            if(!rank) printf("sampleSort - step4\n");

            SortedElem = arr;
            MPI_Comm new_comm;
            if(totSize < npesLong) {
//                if(!myrank) {
//                    std::cout<<" Input to sort is small. splittingComm: "
//                             <<npes<<" -> "<< totSize <<std::endl;
//                }
                par::splitCommUsingSplittingRank(static_cast<int>(totSize), &new_comm, comm);
            } else {
                new_comm = comm;
            }

#ifdef __DEBUG_PAR__
            MPI_Barrier(comm);
        if(!myrank) {
          std::cout<<"SampleSort (small n): Stage-2 passed."<<std::endl;
        }
        MPI_Barrier(comm);
#endif
            if(!rank) printf("sampleSort - step5\n");

            if(!SortedElem.empty()) {
                par::bitonicSort<cooEntry_row>(SortedElem, new_comm);
            }
            if(!rank) printf("sampleSort - step6\n");

#ifdef __DEBUG_PAR__
            MPI_Barrier(comm);
        if(!myrank) {
          std::cout<<"SampleSort (small n): Stage-3 passed."<<std::endl;
        }
        MPI_Barrier(comm);
#endif

        }// end if

#ifdef __DEBUG_PAR__
        if(!myrank) {
            std::cout<<"Using sample sort to sort nodes. n/p^2 is fine."<<std::endl;
        }
#endif
        if(!rank) printf("sampleSort - step7\n");

        //Re-part arr so that each proc. has at least p elements.
//        par::partitionW<cooEntry_row>(arr, nullptr, comm);

        nelem = arr.size();
        if(!rank) printf("sampleSort - step8\n");

//      std::sort(arr.begin(),arr.end());
        omp_par::merge_sort(arr.begin(),arr.end());
        if(!rank) printf("sampleSort - step9\n");

        int *sendcnts = new int[npes];
        assert(sendcnts);

        int * recvcnts = new int[npes];
        assert(recvcnts);

        int * sdispls = new int[npes];
        assert(sdispls);

        int * rdispls = new int[npes];
        assert(rdispls);

#pragma omp parallel for
        for(int k = 0; k < npes; k++){
            sendcnts[k] = 0;
        }

        //To be parallelized
        int k = 0;
        for (DendroIntL j = 0; j < nelem; j++) {
            if (arr[j].row < split[k+1]) {
                sendcnts[k]++;
//            if(rank==0){
//                std::cout << arr[j] << "\tk = " << k << "\t" << splitters[k] << "\t" << splitters[k+1] << "\tfirst" << std::endl;
//            }

            } else {
                k = static_cast<int>(lower_bound3(&split[0], &split[npes-1], arr[j].row));

                if (k == (npes-1) ){
                    //could not find any splitter >= arr[j]
                    sendcnts[k] = nelem - j;
                    break;
                } else {
                    assert(k < (npes-1));
                    assert(arr[j].row < split[k+1]);
                    sendcnts[k]++;
                }
            }//end if-else

        }//end for j
        if(!rank) printf("sampleSort - step10\n");

        par::Mpi_Alltoall<int>(sendcnts, recvcnts, 1, comm);
        if(!rank) printf("sampleSort - step11\n");

        sdispls[0] = 0; rdispls[0] = 0;
//      for (int j = 1; j < npes; j++){
//        sdispls[j] = sdispls[j-1] + sendcnts[j-1];
//        rdispls[j] = rdispls[j-1] + recvcnts[j-1];
//      }
        omp_par::scan(sendcnts,sdispls,npes);
        omp_par::scan(recvcnts,rdispls,npes);

        DendroIntL nsorted = rdispls[npes-1] + recvcnts[npes-1];
        SortedElem.resize(nsorted);
        if(!rank) printf("sampleSort - step12\n");

        cooEntry_row* arrPtr = NULL;
        cooEntry_row* SortedElemPtr = NULL;
        if(!arr.empty()) {
            arrPtr = &(*(arr.begin()));
        }
        if(!SortedElem.empty()) {
            SortedElemPtr = &(*(SortedElem.begin()));
        }
        par::Mpi_Alltoallv_dense<cooEntry_row>(arrPtr, sendcnts, sdispls,
                                               SortedElemPtr, recvcnts, rdispls, comm);

        arr.clear();
        if(!rank) printf("sampleSort - step13\n");

        delete [] sendcnts;
        sendcnts = nullptr;

        delete [] recvcnts;
        recvcnts = nullptr;

        delete [] sdispls;
        sdispls = nullptr;

        delete [] rdispls;
        rdispls = nullptr;

//      sort(SortedElem.begin(), SortedElem.end());
        omp_par::merge_sort(&SortedElem[0], &SortedElem[nsorted]);
        if(!rank) printf("sampleSort - step14\n");

        return 0;
    }//end function

}// end namespace

