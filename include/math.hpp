/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright 2018 - Matteo Ragni, Matteo Cocetti, Luca Zaccarian - University of Trento
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**
 * \file math.hpp
 * \author Matteo Ragni
 *
 * Math utilities and wrapper for BLAS and LAPACK, with template selection
 * for precision. The basic data type is the STL array. We are trying to avoid
 * allocation as much as possible (that means we never allocate dinamically).
 */
#ifndef MATH_HPP_
#define MATH_HPP_

#ifndef MATLAB_MEX_FILE
  #include <cblas.h>
#else
  #include <blas.h>
  #ifndef WIN32
  #define BLAS_WRAPPER(x) x ## _
  #else
  #BLAS_WRAPPER(x) x
  #endif
#endif
#include <lapacke.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

#ifdef __APPLE__
#define DGELS_CALL LAPACKE_dgels
#define SGELS_CALL LAPACKE_sgels
#else
#define DGELS_CALL LAPACKE_dgels
#define SGELS_CALL LAPACKE_sgels
#endif

namespace hybrid {
  /** \namespace wrapper
   * The namespace \p wrapper contains the internal calls in single and double precision for
   * the Blas/Lapack functions. It also contains some utilities for template programming.
   */

  namespace utility {
    /** \brief constexpr for array size */
    template < typename >
    struct array_size;

    /** \brief Get dimension of an array compile time
     *
     * The constexpr can be used as follows:
     * ```
     * template<typename T, std::size_t S>
     * inline void f(std::array<T, S>) {
     *   std::cout << hybrid::utility::array_size<T, S>::size << std::endl;
     * }
     *
     * std::array<float, 10> a;
     * f<float, 10>(a);
     * // prints:
     * // 10
     * ```
     * It is intended use is in static assertion in order to get compilation consistent code
     * (for example in dgels, we need to check at compile time the size of the matrix and the array)
     */
    template < typename REAL_T, std::size_t dim >
    struct array_size< std::array< REAL_T, dim > > {
      static std::size_t const size = dim;
    };

    /** \brief constexpr for array size comparison */
    template < typename A, typename B >
    struct array_size_compare;

    /** \brief Compares the dimensions of two arrays, at compile time
     *
     * The constexpr compares two array type in order to decide at compile time
     * if the first array is bigger, smaller or equal with respect to the other,
     * in form of boolean constant expression:
     * An usage example follows:
     * ```
     * template<typename T, std::size_t S1, std::size_t S2>
     * inline void f(std::array<T, S>, std::array<T, S2>) {
     *   std::cout << hybrid::utility::array_size_compare<std::array<T,S1>, std::array<T,S2>>::equal << std::endl;
     *   std::cout << hybrid::utility::array_size_compare<std::array<T,S1>, std::array<T,S2>>::smaller << std::endl;
     *   std::cout << hybrid::utility::array_size_compare<std::array<T,S1>, std::array<T,S2>>::bigger << std::endl;
     * }
     *
     * std::array<float, 10> a;
     * std::array<float, 11> b;
     * f<float, 10, 11>(a, b);
     * // prints:
     * // false
     * // true
     * // false
     * ```
     */
    template < typename REAL_T, std::size_t A, std::size_t B >
    struct array_size_compare< std::array< REAL_T, A >, std::array< REAL_T, B > > {
      static const bool bigger =
          array_size< std::array< REAL_T, A > >::size > array_size< std::array< REAL_T, B > >::size;
      static const bool equal =
          array_size< std::array< REAL_T, A > >::size == array_size< std::array< REAL_T, B > >::size;
      static const bool smaller =
          array_size< std::array< REAL_T, A > >::size < array_size< std::array< REAL_T, B > >::size;
    };

    /** \brief Gets, compile time, the biggest between two numbers */
    template < std::size_t a, std::size_t b >
    struct biggest_dim {
      static std::size_t const value = std::conditional< a >= b, std::integral_constant< std::size_t, a >,
                                                         std::integral_constant< std::size_t, b > >::type::value;
    };

  };  // namespace utility

  namespace math {
    namespace wrapper {
      /** \name Double precision wrappers
       *
       * This part contains the double precision (`double`) wrapper functions
       */

      /** \brief Illegal Matrix error for solve wrapper */
      class IllegalMatrix : public std::runtime_error {
       public:
        IllegalMatrix() : std::runtime_error(""){};
        IllegalMatrix(const char *e) : std::runtime_error(e){};
      };
      /** \brief Singular Matrix error for solve wrapper */
      class SingularMatrix : public std::runtime_error {
       public:
        SingularMatrix() : std::runtime_error(""){};
        SingularMatrix(const char *e) : std::runtime_error(e){};
      };

      /** \brief **Internal** Sums two arrays. The output is in the \p out array. The input can be scaled using
       * a scalar \p scale: \f$ y \leftarrow \alpha x + y \f$
       *
       * \tparam dim size of the array
       * \param out array , that will contain the output
       * \param in second array with the vector to be scaled and added
       * \param scale scaling factor for \p out vector
       */
      template < std::size_t dim >
      void sum(std::array< double, dim > &out, const std::array< double, dim > &in, double scale) {
#ifdef MATLAB_MEX_FILE
        ptrdiff_t one_ = 1;
        ptrdiff_t dim_ = dim;
        BLAS_WRAPPER(daxpy)(&dim_, &scale, in.data(), &one_, out.data(), &one_);
#else
        cblas_daxpy(static_cast< lapack_int >(dim), scale, in.data(), 1, out.data(), 1);
#endif
      }

      /** \brief **Internal** Scale an array: \f$ x \leftarrow \alpha x \f$
       *
       * \tparam dim size of the array
       * \param out input vector, that will also contain the output
       * \param s scaling factor
       */
      template < std::size_t dim >
      void scale(std::array< double, dim > &out, double s) {
#ifdef MATLAB_MEX_FILE
        ptrdiff_t one_ = 1;
        ptrdiff_t dim_ = dim;
        BLAS_WRAPPER(dscal)(&dim_, &s, out.data(), &one_);
#else
        cblas_dscal(static_cast< lapack_int >(dim), s, out.data(), 1);
#endif
      }

      /** \brief **Internal** Evaluates the L2 norm of a vector. The norm is returned as a value: \f$ ||x||_{2} \f$
       *
       * \tparam dim size of the array
       * \param in vector for which norm must be evaluated
       * \return the norm scalar value
       */
      template < std::size_t dim >
      double norm(const std::array< double, dim > &in) {
#ifdef MATLAB_MEX_FILE
        ptrdiff_t one_ = 1;
        ptrdiff_t dim_ = dim;
        return BLAS_WRAPPER(dnrm2)(&dim_, in.data(), &one_);
#else
        return cblas_dnrm2(static_cast< lapack_int >(dim), in.data(), 1);
#endif
      }

      /** \brief **Internal** Solves the linear system \f$Ax + b = 0\f$
       *
       * \tparam rows number of rows in \f$A\f$ matrix
       * \tparam cols number of cols in \f$A\f$ matrix
       * \tparam dimb dimension of the \f$b\f$ vector in terms of allocated memory
       * \param A the matrix \f$A\f$
       * \param b the vector \f$b\f$
       * \throw hybrid::wrapper::IllegalMatrix when the matrix contains some illegal values
       * \throw hybrid::wrapper::SingularMatrix when the matrix cannot be inverted
       */
      template < std::size_t rows, std::size_t cols, std::size_t dimb >
      void solve(std::array< double, rows * cols > &A, std::array< double, dimb > &b) {
        lapack_int result =
            DGELS_CALL(LAPACK_COL_MAJOR, 'N', static_cast< lapack_int >(rows), static_cast< int >(cols), 1, A.data(),
                          static_cast< lapack_int >(rows), b.data(), static_cast< lapack_int >(dimb));
        if (result > 0)
          throw IllegalMatrix("[HYBRID::MATH::SOLVE] An illegal matrix was passed to solve");
        if (result < 0)
          throw SingularMatrix("[HYBRID::MATH::SOLVE] A singular matrix was passed to solve");
      }

      /** \name Single precision wrappers
       *
       * This part contains the single precision (`float`) wrapper functions
       */

      /** \brief **Internal** Sums two arrays. The output is in the \p out array. The input can be scaled using
       * a scalar \p scale: \f$ y \leftarrow \alpha x + y \f$
       *
       * \tparam dim size of the array
       * \param out array that will contain the output
       * \param in second array with the vector to be scaled and added
       * \param scale scaling factor for \p out vector
       */
      template < std::size_t dim >
      void sum(std::array< float, dim > &out, const std::array< float, dim > &in, float scale) {
#ifdef MATLAB_MEX_FILE
        ptrdiff_t one_ = 1;
        ptrdiff_t dim_ = dim;
        BLAS_WRAPPER(saxpy)(&dim_, &scale, in.data(), &one_, out.data(), &one_);
#else
        cblas_saxpy(static_cast< lapack_int >(dim), scale, in.data(), 1, out.data(), 1);
#endif
      }

      /** \brief **Internal** Scale an array: \f$ x \leftarrow \alpha x \f$
       *
       * \tparam dim size of the array
       * \param out input vector, that will also contain the output
       * \param s scaling factor
       */
      template < std::size_t dim >
      void scale(std::array< float, dim > &out, float s) {
#ifdef MATLAB_MEX_FILE
        ptrdiff_t one_ = 1;
        ptrdiff_t dim_ = dim;
        BLAS_WRAPPER(sscal)(&dim_, &s, out.data(), &one_);
#else
        cblas_sscal(static_cast< lapack_int >(dim), s, out.data(), 1);
#endif
      }

      /** \brief **Internal** Evaluates the L2 norm of a vector. The norm is returned as a value: \f$ ||x||_{2} \f$
       *
       * \tparam dim size of the array
       * \param in vector for which norm must be evaluated
       * \return the norm scalar value
       */
      template < std::size_t dim >
      float norm(const std::array< float, dim > &in) {
#ifdef MATLAB_MEX_FILE
        ptrdiff_t one_ = 1;
        ptrdiff_t dim_ = dim;
        return BLAS_WRAPPER(snrm2)(&dim_, in.data(), &one_);
#else
        return cblas_snrm2(static_cast< lapack_int >(dim), in.data(), 1);
#endif
      }

      /** \brief **Internal** Solves the linear system \f$Ax + b = 0\f$
       *
       * \tparam rows number of rows in \f$A\f$ matrix
       * \tparam cols number of cols in \f$A\f$ matrix
       * \tparam dimb dimension of the \f$b\f$ vector in terms of allocated memory
       * \param A the matrix \f$A\f$
       * \param b the vector \f$b\f$
       * \throw hybrid::wrapper::IllegalMatrix when the matrix contains some illegal values
       * \throw hybrid::wrapper::SingularMatrix when the matrix cannot be inverted
       */
      template < std::size_t rows, std::size_t cols, std::size_t dimb >
      void solve(std::array< float, rows * cols > &A, std::array< float, dimb > &b) {
        lapack_int result =
            SGELS_CALL(LAPACK_COL_MAJOR, 'N', static_cast< lapack_int >(rows), static_cast< int >(cols), 1, A.data(),
                          static_cast< lapack_int >(rows), b.data(), static_cast< lapack_int >(dimb));

        if (result > 0)
          throw IllegalMatrix("[HYBRID::MATH::SOLVE] An illegal matrix was passed to solve");
        if (result < 0)
          throw SingularMatrix("[HYBRID::MATH::SOLVE] A singular matrix was passed to solve");
      }

    };  // namespace wrapper

    /** \name Math utilities
     *
     * This namespcae contains some of the math utilities that are used internally in
     * libhybridpp. In particular, integrators make use of sum, LU decomposition and other
     * linear algebra utilities that leverage the Lapack and Blas.
     */

    /** \brief Scale and sum two arrays
     *
     * The function performs the following operation:
     * \f[
     *   x \leftarrow \alpha x + y, \qquad x, y \in \mathbb{R}^{\dim}, \, \alpha \in \mathbb{R}
     * \f]
     * where \f$x,y\f$ are two vectors of size \p dim, and \f$\alpha\f$ is a scalar value.
     *
     * \tparam REAL_T a floating point value (inheriting from \p float or \p double)
     * \tparam dim size of the vectors
     * \param out the \f$x\f$ vector. It will also contain the output vector
     * \param in the \f$y\f$ vector
     * \param scale the scaling factor \f$\alpha\f$
     * */
    template < typename REAL_T, std::size_t dim >
    void sum(std::array< REAL_T, dim > &out, const std::array< REAL_T, dim > &in, const REAL_T scale) {
      static_assert(std::is_floating_point< REAL_T >::value, "[HYBRID::MATH::SUM] Only float/double accepted");
#ifndef MATLAB_MEX_FILE
      wrapper::sum< dim >(out, in, scale);
#else
      for (std::size_t i = 0; i < dim; i++)
        out[i] = out[i] + scale * in[i];
#endif
    }

    /** \brief Sum two arrays
     *
     * The function performs the following operation:
     * \f[
     *   x \leftarrow x + y, \qquad x, y \in \mathbb{R}^{\dim}
     * \f]
     * where \f$x,y\f$ are two vectors of size \p dim.
     *
     * \tparam REAL_T a floating point value (inheriting from \p float or \p double)
     * \tparam dim size of the vectors
     * \param out the \f$x\f$ vector. It will also contain the output vector
     * \param in the \f$y\f$ vector
     * */
    template < typename REAL_T, std::size_t dim >
    void sum(std::array< REAL_T, dim > &out, const std::array< REAL_T, dim > &in) {
      static_assert(std::is_floating_point< REAL_T >::value, "[HYBRID::MATH::SUM] Only float/double accepted");
      wrapper::sum< dim >(out, in, static_cast< REAL_T >(1.0));
    }

    /** \brief Scale an array
     *
     * The function performs the following operation:
     * \f[
     *   x \leftarrow \alpha x, \qquad x \in \mathbb{R}^{\dim}, \, \alpha \in \mathbb{R}
     * \f]
     * where \f$x\f$ is a vector of size \p dim, and \f$\alpha\f$ is a scalar value.
     *
     * \tparam REAL_T a floating point value (inheriting from \p float or \p double)
     * \tparam dim size of the vector
     * \param out the \f$x\f$ vector. It will also contain the output vector
     * \param s the scaling factor \f$\alpha\f$
     * */
    template < typename REAL_T, std::size_t dim >
    void scale(std::array< REAL_T, dim > &out, REAL_T s) {
      static_assert(std::is_floating_point< REAL_T >::value, "[HYBRID::MATH::SCALE] Only float/double accepted");
#ifndef MATLAB_MEX_FILE
      wrapper::scale< dim >(out, s);
#else
      for (auto &x : out)
        x *= s;
#endif
    }

    /** \brief Returns the norm of an array
     *
     * The function performs the following operation:
     * \f[
     *   \gamma \leftarrow || x ||_{2}, \qquad x \in \mathbb{R}^{\dim}, \, \gamma \in \mathbb{R}
     * \f]
     * where \f$x\f$ is a vector of size \p dim, and \f$\alpha\f$ is a scalar that contains the norm of the vector.
     *
     * \tparam REAL_T a floating point value (inheriting from \p float or \p double)
     * \tparam dim size of the vector
     * \param in the \f$x\f$ vector
     * \return scale the norm scalar \f$\gamma\f$
     * */
    template < typename REAL_T, std::size_t dim >
    REAL_T norm(const std::array< REAL_T, dim > &in) {
      static_assert(std::is_floating_point< REAL_T >::value, "[HYBRID::MATH::NORM] Only float/double accepted");
#ifndef MATLAB_MEX_FILE
      return wrapper::norm< dim >(in);
#else
      REAL_T r = 0.0;
      for (const auto &x : in)
        r += (x * x);
      return sqrt(r);
#endif
    }

    /** \brief **Internal** Input type for solver SFINAE type deduction
     *
     * This internal type make transparent for the user the static selection of the
     * solver to use in case of overdetermined or undermined system of equations. To solve the
     * the linear equations, since the vector \f$ b \f$ is also the vector that will contain
     * the result of the operation, it is required that it has dimensions \f$ \max(rows(A), cols(A)) \f$.
     * This type ensures that.
     */
    template < typename REAL_T, std::size_t rows, std::size_t cols >
    using b_array_t = std::array< REAL_T, hybrid::utility::biggest_dim< rows, cols >::value >;

    /** \brief Solves overdetermined linear systems
     *
     * This wrapper is used to solve (overdetermined) linear system. Given a linear system in the form:
     * \f[
     *    A x + b = 0, \qquad A \in \mathbb{R}^{r \times c}, x \in \mathbb{R}^{c}, b \in \mathbb{R}^{r}
     * \f]
     * the function evaluates \f$x = -A^{\dagger} b\f$ for \f$r \geq c\f$, that is equal to \f$x = -A^{-1} b\f$ iff \f$r
     * = c\f$. The function evaluates \f$x = \min_{x} || Ax + b ||_2 \f$ iff \f$ r \leq c \f$ The solution is stored in
     * the vector \f$b\f$, while the matrix \f$A\f$ will contains the factorization. This is the reason why the memory
     * available for \f$b\f$ must be equal or greater of the memory needed for storing \f$x\f$. The matrix \f$A\f$
     * should be stored in memory with **row major ordering**.
     *
     *
     * \tparam REAL_T a floating point value (inheriting from \p float or \p double)
     * \tparam rows number of rows of matrix \f$A\f$
     * \tparam cols number of columns of matrix \f$A\f$
     * \param A the matrix \f$A\f$. Will contain the factorization as output
     * \param b the vector \f$b\f$. Will contain the result as output
     * \throw hybrid::wrapper::IllegalMatrix if the matrix contains an illegal value
     * \throw hybrid::wrapper::SingularMatrix if the matrix is singular
     */
    template < typename REAL_T, std::size_t rows, std::size_t cols >
    void solve(std::array< REAL_T, cols * rows > &A, b_array_t< REAL_T, cols, rows > &b) {
      static_assert(std::is_floating_point< REAL_T >::value, "[HYBRID::MATH::SOLVE] Only float/double accepted");
      wrapper::solve< rows, cols, hybrid::utility::biggest_dim< rows, cols >::value >(A, b);
    }

    /** \brief Copy a source array in a destination
     *
     * Copy for a source array in a destination
     *
     * \tparam REAL_T floating point precision
     * \tparam n dimensions for the array
     * \param dest destination array
     * \param src source array
     */
    template < typename REAL_T, std::size_t n >
    void copy(std::array< REAL_T, n > &dest, const std::array< REAL_T, n > &src) {
      std::copy(std::begin(src), std::end(src), std::begin(dest));
    }

    /** \brief Elementwise product between two array
     *
     * Makes the pointwise product between two array
     * \tparam REAL_T floating point precision
     * \tparam n dimensions for the array
     * \param dest destination array
     * \param source source array
     */
    template < typename REAL_T, std::size_t n >
    void element_prod(std::array< REAL_T, n > &dest, const std::array< REAL_T, n > &source) {
      for (std::size_t i = 0; i < n; i++)
        dest[i] *= source[i];
    }

    /** \brief Elementwise division between two array
     *
     * Makes the pointwise division between two array
     * \tparam REAL_T floating point precision
     * \tparam n dimensions for the array
     * \param dest destination array
     * \param source source array
     */
    template < typename REAL_T, std::size_t n >
    void element_div(std::array< REAL_T, n > &dest, const std::array< REAL_T, n > &source) {
      for (std::size_t i = 0; i < n; i++)
        dest[i] = dest[i] / source[i];
    }

    /** \brief Vectors comparison
     * Makes the elementwise comparison between two vectors \f$ a >= b\f$. It is true if
     * all are true
     * \tparam REAL_T floating point precision
     * \tparam n dimensions for the array
     * \param a first of the two arrays for comparison
     * \param b second of the two arrays for comparison
     */
    template < typename REAL_T, std::size_t n >
    bool element_compare(const std::array< REAL_T, n > &a, const std::array< REAL_T, n > &b) {
      bool ret = true;
      for (std::size_t i = 0; i < n; i++) {
        ret = ret && (a[i] >= b[i]);
      }
      return ret;
    }

    /** \brief Evaluates the sign of a number
     *
     * Evluates the sign of a number, returning +1 if it is positive and -1 if it is negative
     * \f$a = 0\f$ is considered as positive.
     * \tparam REAL_T floating point precision
     * \param a the number on which we want to evaluate the sign
     * \return +1 is \p a is positive, -1 if it is negative
     */
    template < typename REAL_T >
    REAL_T sign(REAL_T a) {
      static_assert(std::is_floating_point< REAL_T >::value, "[HYBRID::MATH::SOLVE] Only float/double accepted");
      return a >= 0 ? 1 : -1;
    }

    /** \brief Evaluates the sign of an array elementwise
     *
     * Evluates the sign of each entry of an array number, returning +1 if it is positive and -1 if it is negative.
     * A null entry is considered as positive.
     * for each position.
     * \tparam REAL_T floating point precision
     * \tparam n the size of the array
     * \param dest the array on which we will save the result
     * \param src the array on which we want to evaluate the sign
     */
    template < typename REAL_T, std::size_t n >
    void sign(std::array< REAL_T, n > &dest, const std::array< REAL_T, n > &src) {
      static_assert(std::is_floating_point< REAL_T >::value, "[HYBRID::MATH::SOLVE] Only float/double accepted");
      for (std::size_t i = 0; i < n; i++)
        dest[i] = sign(src[i]);
    }

    /** \brief Evaluates the absolute value of a number
     *
     * Evluates the absolute value of a number
     * \tparam REAL_T floating point precision
     * \param a the number on which we want to evaluate the absolute value
     * \return \p a if the number is positive, \p -a if the number is negative
     */
    template < typename REAL_T >
    REAL_T abs(REAL_T a) {
      static_assert(std::is_floating_point< REAL_T >::value, "[HYBRID::MATH::SOLVE] Only float/double accepted");
      return a * hybrid::math::sign(a);
    }

    /** \brief Evaluates the absolute value of an array elementwise
     *
     * Evluates the absolute value of each entry of an array of number,
     * \tparam REAL_T floating point precision
     * \tparam n the size of the array
     * \param dest the array on which we will save the result
     * \param src the array on which we want to evaluate the sign
     */
    template < typename REAL_T, std::size_t n >
    void abs(std::array< REAL_T, n > &dest, const std::array< REAL_T, n > &src) {
      static_assert(std::is_floating_point< REAL_T >::value, "[HYBRID::MATH::SOLVE] Only float/double accepted");
      for (std::size_t i = 0; i < n; i++)
        dest[i] = hybrid::math::abs(src[i]);
    }

    /** \brief Returns the signed squared root of a number
     *
     * The function evaluates:
     * \f[
     *   r = \mathrm{sign}(x) \sqrt{|x|}
     * \f]
     * \tparam REAL_T type for the function
     * \param x the argument for the function
     * \return the result of the above operation
     */
    template < typename REAL_T >
    REAL_T signsqrt(REAL_T x) {
      static_assert(std::is_floating_point< REAL_T >::value, "[HYBRID::MATH::SOLVE] Only float/double accepted");
      return sign(x) * sqrt(hybrid::math::abs(x));
    }

    /** \brief Returns the signed squared root of the difference of two numbers
     *
     * The function evaluates:
     * \f[
     *   r = \mathrm{sign}(a - b) \sqrt{|a - b|}
     * \f]
     * \tparam REAL_T type for the function
     * \param a the first argument for the function
     * \param b the second argument for the function
     * \return the result of the above operation
     */
    template < typename REAL_T >
    REAL_T signsqrt(REAL_T a, REAL_T b) {
      return hybrid::math::signsqrt(a - b);
    }

    /** \brief Minimum between two real number */
    template < typename REAL_T >
    REAL_T min(REAL_T a, REAL_T b) {
      static_assert(std::is_floating_point< REAL_T >::value, "[HYBRID::MATH::SOLVE] Only float/double accepted");
      return a >= b ? b : a;
    }

    /** \brief Minimum between two real number */
    template < typename REAL_T >
    REAL_T max(REAL_T a, REAL_T b) {
      static_assert(std::is_floating_point< REAL_T >::value, "[HYBRID::MATH::SOLVE] Only float/double accepted");
      return a >= b ? a : b;
    }

  };  // namespace math
};    // namespace hybrid

#endif /* MATH_HPP_ */
