 template<typename T>
  int sampleSort(std::vector<T> &arr, std::vector<T> &SortedElem,std::vector<double>& stats,MPI_Comm comm) {
    // std::cout<<"Sample Sort Execution Begin"<<std::endl;

    auto ss_start=std::chrono::system_clock::now();

    int npes;
    MPI_Comm_size(comm, &npes);

    //std::cout << rank << "Nodes Input : " << __func__ << arr.size() << std::endl;

    assert(arr.size());

    if (npes == 1) {
      std::cout << " have to use seq. sort"
      << " since npes = 1 . inpSize: " << (arr.size()) << std::endl;
       // std::sort(arr.begin(), arr.end());
      omp_par::merge_sort(arr.begin(), arr.end());
      SortedElem = arr;
    }

    std::vector<T> splitters;
    std::vector<T> allsplitters;

    int myrank;
    MPI_Comm_rank(comm, &myrank);

    long long int nelem = arr.size();
    long long int nelemCopy = nelem;
    long long int totSize;
    /*Calculating the total size of elements*/
    MPI_Allreduce(&nelemCopy, &totSize, 1, MPI_SUM, comm);

    long long int npesLong = npes;
    const long long int FIVE = 5;

    if (totSize < (FIVE * npesLong * npesLong)) {
        if (!myrank) {
            std::cout << " *Using bitonic sort since totSize < (5*(npes^2)). totSize: "
                      << totSize << " npes: " << npes << std::endl;
        }

#ifdef __DEBUG_PAR__
      MPI_Barrier(comm);
      if(!myrank) {
        std::cout<<"SampleSort (small n): Stage-1 passed."<<std::endl;
      }
      MPI_Barrier(comm);
#endif

      SortedElem = arr;
      MPI_Comm new_comm;
      if (totSize < npesLong) {
        if (!myrank) {
          std::cout << " Input to sort is small. splittingComm: "
          << npes << " -> " << totSize << std::endl;
        }
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

      if (!SortedElem.empty()) {
        par::bitonicSort<T>(SortedElem, new_comm);
        //std::cout<<"Rank:"<<rank<<" bitonic search complete"<<std::endl;
      }

#ifdef __DEBUG_PAR__
        MPI_Barrier(comm);
        if(!myrank) {
          std::cout<<"SampleSort (small n): Stage-3 passed."<<std::endl;
        }
        MPI_Barrier(comm);
#endif

    }// end if
/*

  // if(!rank)

#ifdef __DEBUG_PAR__
    if(!myrank) {
      std::cout<<"Using sample sort to sort nodes. n/p^2 is fine."<<std::endl;
    }
#endif

*/

        //Re-part arr so that each proc. has at least p elements.

        // if (!rank) std::cout << "[samplesort] repartitioning input" << std::endl;

        par::partitionW<T>(arr, NULL, comm);
        nelem = arr.size();

        // if (!rank) std::cout << RED "[samplesort] initial local sort" NRM << std::endl;
        // std::sort(arr.begin(),arr.end());
        //auto splitterCalculation_start=std::chrono::system_clock::now();
        std::sort(arr.begin(), arr.end()); //omp_par::merge_sort(arr.begin(), arr.end());

        // if (!rank) std::cout << GRN "[samplesort] initial local sort" NRM << std::endl;

        std::vector <T> sendSplits(npes - 1);
        splitters.resize(npes);

#pragma omp parallel for
        for (int i = 1; i < npes; i++) {

            sendSplits[i - 1] = arr[i * nelem / npes];

            //std::cout << BLU << "===============================================" << NRM << std::endl;
            // std::cout << RED " Splittersn in Sample Sort Rank:"<<rank<< NRM << std::endl;
            //std::cout << BLU << "===============================================" << NRM << std::endl;

            std::ostringstream convert;
            convert << sendSplits[i - 1];
            std::vector <std::string> results;
            std::stringstream s(convert.str());
            while (!s.eof()) {
                std::string tmp;
                s >> tmp;
                results.push_back(tmp);
            }


        }//end for i


        // sort sendSplits using bitonic ...
        // if (!rank)
        // std::cout << rank << RED " [samplesort] bitonicsort " NRM << sendSplits.size() << std::endl;
        par::bitonicSort<T>(sendSplits, comm);
        // if (!rank)
        // std::cout << rank << GRN " [samplesort] done bitonicsort" NRM << std::endl;


        //std::cout << myrank << ": afterBitonic " << sendSplits[0] << "  ||||  " <<  sendSplits[1] << std::endl;

        // All gather with last element of splitters.
        T *sendSplitsPtr = NULL;
        T *splittersPtr = NULL;
        if (sendSplits.size() > static_cast<unsigned int>(npes - 2)) {
            sendSplitsPtr = &(*(sendSplits.begin() + (npes - 2)));
        }
        if (!splitters.empty()) {
            splittersPtr = &(*(splitters.begin()));
        }
        par::Mpi_Allgather<T>(sendSplitsPtr, splittersPtr, 1, comm);


        sendSplits.clear();
        int *sendcnts = new int[npes];
        assert(sendcnts);

        int *recvcnts = new int[npes];
        assert(recvcnts);

        int *sdispls = new int[npes];
        assert(sdispls);

        int *rdispls = new int[npes];
        assert(rdispls);

#pragma omp parallel for
        for (int k = 0; k < npes; k++) {
            sendcnts[k] = 0;
        }

        //To be parallelized
/*      int k = 0;
      for (long long int j = 0; j < nelem; j++) {
        if (arr[j] <= splitters[k]) {
          sendcnts[k]++;
        } else{
          k = seq::UpperBound<T>(npes-1, splittersPtr, k+1, arr[j]);
          if (k == (npes-1) ){
            //could not find any splitter >= arr[j]
            sendcnts[k] = (nelem - j);
            break;
          } else {
            assert(k < (npes-1));
            assert(splitters[k] >= arr[j]);
            sendcnts[k]++;
          }
        }//end if-else
      }//end for j
*/

        auto splitterCalculation_end = std::chrono::system_clock::now();
        auto all2all_start = std::chrono::system_clock::now();

        {
            int omp_p = omp_get_max_threads();
            int *proc_split = new int[omp_p + 1];
            long long int *lst_split_indx = new long long int[omp_p + 1];
            proc_split[0] = 0;
            lst_split_indx[0] = 0;
            lst_split_indx[omp_p] = nelem;
#pragma omp parallel for
            for (int i = 1; i < omp_p; i++) {
                //proc_split[i] = seq::BinSearch(&splittersPtr[0],&splittersPtr[npes-1],arr[i*nelem/omp_p],std::less<T>());
                proc_split[i] =
                        std::upper_bound(&splittersPtr[0], &splittersPtr[npes - 1], arr[i * nelem / omp_p],
                                         std::less<T>()) -
                        &splittersPtr[0];
                if (proc_split[i] < npes - 1) {
                    //lst_split_indx[i]=seq::BinSearch(&arr[0],&arr[nelem],splittersPtr[proc_split[i]],std::less<T>());
                    lst_split_indx[i] =
                            std::upper_bound(&arr[0], &arr[nelem], splittersPtr[proc_split[i]], std::less<T>()) -
                            &arr[0];
                } else {
                    proc_split[i] = npes - 1;
                    lst_split_indx[i] = nelem;
                }
            }


            splitterCalculation_end = std::chrono::system_clock::now();
            all2all_start = std::chrono::system_clock::now();

#pragma omp parallel for
            for (int i = 0; i < omp_p; i++) {
                int sendcnts_ = 0;
                int k = proc_split[i];
                for (long long int j = lst_split_indx[i]; j < lst_split_indx[i + 1]; j++) {
                    if (arr[j] <= splitters[k]) {
                        sendcnts_++;
                    } else {
                        if (sendcnts_ > 0)
                            sendcnts[k] = sendcnts_;
                        sendcnts_ = 0;
                        k = seq::UpperBound<T>(npes - 1, splittersPtr, k + 1, arr[j]);
                        if (k == (npes - 1)) {
                            //could not find any splitter >= arr[j]
                            sendcnts_ = (nelem - j);
                            break;
                        } else {
                            assert(k < (npes - 1));
                            assert(splitters[k] >= arr[j]);
                            sendcnts_++;
                        }
                    }//end if-else
                }//end for j
                if (sendcnts_ > 0)
                    sendcnts[k] = sendcnts_;
            }
            delete[] lst_split_indx;
            delete[] proc_split;
        }

        par::Mpi_Alltoall<int>(sendcnts, recvcnts, 1, comm);

        sdispls[0] = 0;
        rdispls[0] = 0;
//      for (int j = 1; j < npes; j++){
//        sdispls[j] = sdispls[j-1] + sendcnts[j-1];
//        rdispls[j] = rdispls[j-1] + recvcnts[j-1];
//      }
        omp_par::scan(sendcnts, sdispls, npes);
        omp_par::scan(recvcnts, rdispls, npes);

        long long int totalRecv =
                rdispls[npes - 1] + recvcnts[npes - 1] - recvcnts[rank]; // change long long int to unsigned long long
        long long int totalSend = sdispls[npes - 1] + sendcnts[npes - 1] - sendcnts[rank];

        long long int nsorted = rdispls[npes - 1] + recvcnts[npes - 1];
        SortedElem.resize(nsorted);

        T *arrPtr = NULL;
        T *SortedElemPtr = NULL;
        if (!arr.empty()) {
            arrPtr = &(*(arr.begin()));
        }
        if (!SortedElem.empty()) {
            SortedElemPtr = &(*(SortedElem.begin()));
        }

        par::Mpi_Alltoallv_Kway<T>(arrPtr, sendcnts, sdispls, SortedElemPtr, recvcnts, rdispls, comm);


        arr.clear();

        delete[] sendcnts;
        sendcnts = NULL;

        delete[] recvcnts;
        recvcnts = NULL;

        delete[] sdispls;
        sdispls = NULL;

        delete[] rdispls;
        rdispls = NULL;

        std::sort(&SortedElem[0], &SortedElem[nsorted]); //omp_par::merge_sort(&SortedElem[0], &SortedElem[nsorted]);
  }//end function
