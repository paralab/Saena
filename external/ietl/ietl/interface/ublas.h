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
      double dot;
      MPI_Allreduce(&dot_l, &dot, 1, MPI_DOUBLE, MPI_SUM, comm);
      return dot;
  }

  template < class T>
   inline typename number_traits<T>::magnitude_type two_norm(boost::numeric::ublas::vector<T>& x, MPI_Comm comm) {
      double dot_l = boost::numeric::ublas::inner_prod (boost::numeric::ublas::conj(x), x);
      double dot;
      MPI_Allreduce(&dot_l, &dot, 1, MPI_DOUBLE, MPI_SUM, comm);
      return sqrt(dot);
  }

  template < class T>
  void copy(const boost::numeric::ublas::vector<T>& x,boost::numeric::ublas::vector<T>& y) {
    y.assign(x);
  }

  template <class M, class T>
  inline void mult(M& m, const boost::numeric::ublas::vector<T>& x, boost::numeric::ublas::vector<T>& y) {
      unsigned int size = m.Mbig;

      std::vector<double> x_std(size);
      for(unsigned int i = 0; i < size; i++)
        x_std[i] = x[i];

//      printf("x_std:\n");
//      for(unsigned int i = 0; i < size; i++)
//          printf("%f\n", x_std[i]);

      std::vector<double> y_std(size, 0);
//      m.matvec(x_std, y_std);
      m.matvec_sparse(x_std, y_std);

//      printf("\ny_std:\n");
//      for(unsigned int i = 0; i < size; i++)
//          printf("%f\n", y_std[i]);

      for(unsigned int i = 0; i < size; i++)
          y[i] = y_std[i];

//   y=boost::numeric::ublas::prod(m,x);
  }
}

#endif
