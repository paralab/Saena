#ifndef LAMBDA_LANCZOS_UTIL_H_
#define LAMBDA_LANCZOS_UTIL_H_

#include <vector>
#include <complex>
#include <limits>
#include <cmath>
#include <cassert>

namespace lambda_lanczos { namespace util {


/**
 * @brief Template class to map specific types. See #real_t<T> for usage.
 *
 */
template <typename T>
struct realTypeMap {
  typedef T type;
};

template <typename T>
struct realTypeMap<std::complex<T>> {
  typedef T type;
};

/**
 * @brief Type mapper from `T` to real type of `T`.
 *
 * By default, `real_t<T>` returns `T`.
 * However `real_t<complex<T>>` returns `T`.
 * Usage example: This function returns a real number even if `T` is complex:
 * @code
 * template <typename T>
 * inline real_t<T> norm(const std::vector<T>& vec);
 * @endcode
 */
template <typename T>
using real_t = typename realTypeMap<T>::type;


template<typename T>
MPI_Datatype get_type()
{
    char name = typeid(T).name()[0];
    switch (name) {
        case 'i':
            return MPI_INT;
        case 'f':
            return MPI_FLOAT;
        case 'j':
            return MPI_UNSIGNED;
        case 'd':
            return MPI_DOUBLE;
        case 'c':
            return MPI_CHAR;
        case 's':
            return MPI_SHORT;
        case 'l':
            return MPI_LONG;
        case 'm':
            return MPI_UNSIGNED_LONG;
        case 'b':
            return MPI_BYTE;
    }
}

/**
 * @brief Template class to implement positive-definite product
 *
 * ConjugateProduct::prod(a,b) is a function which returns a*b by default.
 * However if the arguments are complex numbers, it returns conj(a)*b instead.
 * This structure is required because partial specialization of
 * function template is not allowed in C++.
 */
template <typename T>
struct ConjugateProduct {
public:
  /**
   * @brief Returns a*b for real-type arguments, conj(a)*b for complex-type arguments.
   */
  static T prod(T a, T b) { return a*b; }
};

template <typename T>
struct ConjugateProduct<std::complex<T>> {
public:
  static std::complex<T> prod(std::complex<T> a, std::complex<T> b) {
    return std::conj(a)*b;
  }
};

/**
 * @brief Returns "mathematical" inner product of v1 and v2.
 *
 * This function is needed because
 * std::inner_product calculates transpose(v1)*v2 instead of dagger(v1)*v2 for complex type.
 *
 */
template <typename T>
inline T inner_prod(const std::vector<T>& v1, const std::vector<T>& v2, MPI_Comm comm) {
    assert(v1.size() == v2.size());
//    T dot_l = (T)std::inner_product(std::begin(v1), std::end(v1),
//                           std::begin(v2), T(),
//                           [](T a, T b) -> T { return a+b; },
//                           ConjugateProduct<T>::prod);

//    MPI_Datatype mpit = MPI_DOUBLE;
//    if(sizeof(T) == sizeof(float)){
//        mpit = MPI_FLOAT;
//    }else if(sizeof(T) == sizeof(double)){
//        mpit = MPI_DOUBLE;
//    }

//    auto dot = T();
//    MPI_Allreduce(&dot_l, &dot, 1, par::Mpi_datatype<T>::value(), MPI_SUM, comm);
//    MPI_Allreduce(&dot_l, &dot, 1, get_type<T>(), MPI_SUM, comm);

//return std::inner_product(std::begin(v1), std::end(v1),
//    std::begin(v2), T(),
//    [](T a, T b) -> T { return a+b; },
//    ConjugateProduct<T>::prod);
  // T() means zero value of type T

    const index_t sz = v1.size();
    T dot_l = 0.0, dot = 0.0;
    for(index_t i = 0; i < sz; ++i)
        dot_l += v1[i] * v2[i];
    MPI_Allreduce(&dot_l, &dot, 1, par::Mpi_datatype<T>::value(), MPI_SUM, comm);
    return dot;
}

/**
 * @brief Returns Euclidean norm of given vector.
 */
template <typename T>
inline real_t<T> norm(const std::vector<T>& vec, const MPI_Comm &comm) {
  return std::sqrt(std::real(inner_prod(vec, vec, comm)));
  // The norm of any complex vector <v|v> is real by definition.
}

/**
 * @brief Multiplies each element of vec by a.
 */
template <typename T1, typename T2>
inline void scalar_mul(T1 a, std::vector<T2>& vec) {
  for(auto& elem : vec) {
    elem *= a;
  }
}

/**
 * @brief Normalizes given vector.
 */
template <typename T>
inline void normalize(std::vector<T>& vec, const MPI_Comm &comm) {
  scalar_mul(1.0/norm(vec, comm), vec);
}


/**
 * @brief Returns 1-norm of given vector.
 */
template <typename T>
inline real_t<T> l1_norm(const std::vector<T>& vec, const MPI_Comm &comm) {
//    real_t<T> norm_l = real_t<T>(); // Zero initialization
//
//    for(const T& element : vec) {
//        norm_l += std::abs(element);
//    }
//
//    real_t<T> norm = real_t<T>(); // Zero initialization
//    MPI_Allreduce(&norm_l, &norm, 1, par::Mpi_datatype<double>::value(), MPI_SUM, comm);

    real_t<T> norm = real_t<T>(); // Zero initialization
    for(const T& element : vec) {
        norm += std::abs(element);
    }

    return norm;
}


/**
 * @brief Orthogonalizes vector `uorth` with respect to orthonormal vectors in `u`.
 *
 * Vectors in `u` must be normalized, but uorth doesn't have to be.
 */
template <typename T>
inline void schmidt_orth(std::vector<T>& uorth, const std::vector<std::vector<T>>& u, const MPI_Comm &comm) {
  const auto n = uorth.size();

  for(const auto& uk : u) {
    T innprod = util::inner_prod(uk, uorth, comm);

    for(size_t i = 0;i < n;i++) {
      uorth[i] -= innprod * uk[i];
    }
  }
}


/**
 * @brief Returns the significant decimal digits of type T.
 *
 */
template <typename T>
inline constexpr int sig_decimal_digit() {
  return (int)(std::numeric_limits<T>::digits *
	       log10(std::numeric_limits<T>::radix));
}


template <typename T>
inline constexpr T minimum_effective_decimal() {
  return pow(10, -sig_decimal_digit<T>());
}

}} /* namespace lambda_lanczos::util */

#endif /* LAMBDA_LANCZOS_UTIL_H_ */
