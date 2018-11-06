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
 * \file newton.hpp
 * \author Matteo Ragni
 *
 * The file implements a Newton solver that is used in the implicit
 * Euler integration scheme.
 */

#ifndef NEWTON_HPP_
#define NEWTON_HPP_

#include <exception>
#include <functional>
#include <iostream>
#include <algorithm>
#include "math.hpp"
#include "types.hpp"

namespace hybrid {
  /** \brief Max iteration reached for solver */
  class MaxIterException : public std::runtime_error {
   public:
    MaxIterException() : std::runtime_error(""){};
    MaxIterException(const char *e) : std::runtime_error(e){};
  };

  /** \brief X tolerance exception fo solver */
  class XTolException : public std::runtime_error {
   public:
    XTolException() : std::runtime_error(""){};
    XTolException(const char *e) : std::runtime_error(e){};
  };

  template < typename REAL_T, std::size_t dim_g, std::size_t dim_x, typename... Args >
  class NewtonRaphson {
   private:
    REAL_T tol;   /**< Function tolerance. If \f$ |f(x)| < \mathrm{tol} \f$ the solver exits with success */
    REAL_T xtol;  /**< Variable update tolerance. If \f$ |\delta x| < \mathrm{tol}_x\f$ the solver exits with -1 */
    int max_iter; /**< Maximum number of iteration allowed for finding a solution. If above, as exception is thrown */

    std::array< REAL_T, dim_x * dim_g > wv_a;
    std::array< REAL_T, std::max(dim_x, dim_g) > wv_b;

   public:
    TargetFunction< REAL_T, dim_g, dim_x, Args... > g;
    TargetGradient< REAL_T, dim_g, dim_x, Args... > dg;

    NewtonRaphson(REAL_T _tol, REAL_T _xtol, int _max_iter, TargetFunction< REAL_T, dim_g, dim_x, Args... > _g,
                  TargetGradient< REAL_T, dim_g, dim_x, Args... > _dg)
        : tol(_tol), xtol(_xtol), max_iter(_max_iter), g(_g), dg(_dg) {
      if (tol <= 0 || xtol <= 0) throw std::invalid_argument("Tolerance must be a positive number");
      if (g == nullptr) throw std::invalid_argument("Solver function cannot be null");
      if (dg == nullptr) throw std::invalid_argument("Solver function gradient cannot be null");
    };

    NewtonRaphson(REAL_T _tol, REAL_T _xtol, int _max_iter) : tol(_tol), xtol(_xtol), max_iter(_max_iter) {
      if (tol <= 0 || xtol <= 0 || max_iter <= 0) throw std::invalid_argument("Tolerance must be a positive number");
    };

    int solve(std::array< REAL_T, dim_x > &x, Args... data) {
      int iter = 0;
      std::array< REAL_T, dim_x > wv_c;
      std::array< REAL_T, dim_x > wv_d;
      std::fill(std::begin(wv_c), std::end(wv_c), 0.0);
      std::fill(std::begin(wv_d), std::end(wv_d), 0.0);

          while (iter <= this->max_iter) {
        this->g(this->wv_b, x, data...);
        if (hybrid::math::norm(this->wv_b) < this->tol) return iter;  // Exit because |g(x)| approx 0
        hybrid::math::scale(this->wv_b, -1.0);
        this->dg(this->wv_a, x, data...);
        hybrid::math::solve< REAL_T, dim_g, dim_x >(this->wv_a, this->wv_b);
        hybrid::math::sum(x, this->wv_b);

        // Evaluating relative x tolerance |δ x+ - δx | / xtol
        for (std::size_t i = 0; i < dim_x; i++) {
          wv_d[i] = (this->wv_b[i] - wv_c[i]) / this->xtol;
          wv_d[i] = wv_d[i] >= 0 ? wv_d[i] : -wv_d[i];
        }
        if (hybrid::math::element_compare(wv_c, wv_d)) {
          throw hybrid::XTolException("Update for target variable is less than tolerance");  // Exit because |dx| approx 0
        }
        hybrid::math::copy(wv_c, this->wv_b);
        iter++;
      }
      throw hybrid::MaxIterException("Maximum number of iteration reached without solution");
    }
  };

};  // namespace hybrid

#endif /* NEWTON_HPP_ */