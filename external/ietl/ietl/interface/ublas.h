/***************************************************************************
 * $Id: ublas.h,v 1.4 2003/09/05 09:27:53 prakash Exp $
 *
 * Copyright (C) 2001-2003 by Prakash Dayal <prakash@comp-phys.org>
 *                            Matthias Troyer <troyer@comp-phys.org>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *
 **************************************************************************/

#ifndef IETL_UBLAS_H
#define IETL_UBLAS_H

#include <boost/numeric/ublas/vector.hpp>
#include <ietl/traits.h>


namespace ietl {

    template < class T, class Gen>
        inline void generate(boost::numeric::ublas::vector<T>& c, Gen& gen) {
        std::generate(c.begin(),c.end(),gen);
    }

    template < class T, class S>
    inline T dot(const boost::numeric::ublas::vector<T,S>& x , const boost::numeric::ublas::vector<T,S>& y, MPI_Comm comm) {
        double dot_l = boost::numeric::ublas::inner_prod (boost::numeric::ublas::conj(x), y);
        double dot = 0.0;
        MPI_Allreduce(&dot_l, &dot, 1, MPI_DOUBLE, MPI_SUM, comm);
        return dot;
    }

    template < class T>
    inline typename number_traits<T>::magnitude_type two_norm(boost::numeric::ublas::vector<T>& x, MPI_Comm comm) {
        double dot_l = boost::numeric::ublas::inner_prod (boost::numeric::ublas::conj(x), x);
        double dot = 0.0;
        MPI_Allreduce(&dot_l, &dot, 1, MPI_DOUBLE, MPI_SUM, comm);
        return sqrt(dot);
    }

    template < class T>
    void copy(const boost::numeric::ublas::vector<T>& x,boost::numeric::ublas::vector<T>& y) {
        y.assign(x);
    }

    template <class M, class T>
    inline void mult(M& m, boost::numeric::ublas::vector<T>& x, boost::numeric::ublas::vector<T>& y) {
//        y = boost::numeric::ublas::prod(m,x);
//        m.matvec_sparse_array(&x[m.split[rank]], &y[m.split[rank]]);
        m.matvec_sparse(&x[0], &y[0]);
        for(auto i = m.M; i < m.Mbig; ++i){ // size of vectors are global (m.Mbig). not using this part, will make the
            y[i] = 0;                       // result wrong in parallel.
        }
    }

    // the inefficient version of matvec
#if 0
    template <class M, class T>
    inline void mult(M& m, const boost::numeric::ublas::vector<T>& x, boost::numeric::ublas::vector<T>& y) {
        unsigned int size = m.Mbig;

//        int nprocs, rank;
//        MPI_Comm_size(m.comm, &nprocs);
//        MPI_Comm_rank(m.comm, &rank);
//        int rank_v = 1;
//        if(rank == rank_v) printf("\nrank %d: size = %d\n", rank, size);

        std::vector<double> x_std(size);
        for(unsigned int i = 0; i < size; i++){
            x_std[i] = x[i];
//            if(rank == rank_v) printf("%d\t%.12f\n", i, x[i]);
        }

//        y = boost::numeric::ublas::prod(m, x);
        std::vector<double> y_std(size);
//        m.matvec(x_std, y_std);
        m.matvec_sparse(x_std, y_std);

//        if(rank == rank_v) printf("\ny:\n");
        for(unsigned int i = 0; i < size; i++){
            y[i] = y_std[i];
//            if(rank == rank_v) printf("%d\t%.12f\n", i, y[i]);
        }
    }
#endif
}

#endif
