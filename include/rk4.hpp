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
 * \file rk4.hpp
 * \author Matteo Ragni
 *
 * The library implements a Runge-Kutta 4 scheme with the following tableau:
 * \code
 *      0  |  0   0   0   0
 *     1/2 | 1/2  0   0   0
 *     1/2 |  0  1/2  0   0
 *      1  |  0   0   1   0
 *     ----+----------------
 *         | 1/6 1/3 1/3 1/6
 * \endcode
 * For the ode in the form:
 *  \f[
 *    \dot{x}(t) = f(t, x(t), u(t), p), \quad
 *    x(t_0) = x_0
 *  \f]
 * The integration scheme evaluates:
 *  \f[
 *    x(t_k + h) = x(t_k) + t_s \sum_{i = 1}^{4} b_i k_i
 *  \f]
 * where:
 *  \f[
 *    k_i = f(t + c_i t_s, x(t_k) + t_s \sum_{j=1}^{i-1} a_{i,j} k_j, u(t_k), p)
 *  \f]
 *
 * The input \f$ u(t_k) \f$ is assumed constant between integration ministeps.
 * From an implementation point of view, \f$ p \f$ is an array of parameters This
 * is done for compatibility with MATLAB System Identification Toolbox model
 * file. No allocation is performed.
 */
#ifndef RK4_HPP_
#define RK4_HPP_

#include <iostream>
#include "integrator.hpp"
#include "math.hpp"

namespace hybrid {
  
  /** \brief Class for Explicit Euler integrator
   *
   * This is the implementation of a Runge Kutta 4th order integration scheme that does not
   * use dynamical memory allocation. All dimensions are defined in a static way, through templating.
   * The ode is required to be defined as:
   * \f[
   *   \dot{x}(t) = f(t, x(t), u(t), p), \qquad
   *     x(t) \in \mathbb{R}^{\dim(x)}, \,
   *     u(t) \in \mathbb{R}^{\dim(u)} \,
   *     p \in \mathbb{R}^{\dim(p)}
   * \f]
   * The ODE styles is in same way compatible with the MATLAB System identification toolbox idnlgrey
   * function call. The only difference is in the parameter vector that in this case is in a single
   * array (instead of a `REAL_T**` vector). The integration scheme follows:
   *
   * \f[
   *  x^+ = x + t_s f(t, x(t), u(t), p)
   * \f]
   *
   * \tparam REAL_T floating point precision
   * \tparam dim_x dimension of the state array
   * \tparam dim_u dimension of the external forces array
   * \tparam dim_p dimension of the parameters array
   * \tparam Args variadic template for user arguments
   */
  template < typename REAL_T, std::size_t dim_x, std::size_t dim_u, std::size_t dim_p, typename... Args >
  class RK4 : public Integrator< REAL_T, dim_x, dim_u, dim_p, Args... > {
   public:
    /** \brief Contructor with function callback for ODE
     *
     * Constructor with function callback for ODE
     *
     * \param _ts sampling time
     * \param _f callback for the ODE
     */
    RK4(REAL_T _ts, OdeFunction< REAL_T, dim_x, dim_u, dim_p, Args... > _f)
        : Integrator< REAL_T, dim_x, dim_u, dim_p, Args... >(_ts, _f){};
    
    /** \brief Simple constructor, only sampling time
     *
     * Simple constructor with only sampling time
     *
     * \param _ts sampling time
     */
    RK4(REAL_T _ts) : Integrator< REAL_T, dim_x, dim_u, dim_p, Args... >(_ts){};
    
    /** \brief The next step in the computation
     *
     * The function evaluates the next step for te integration, evaluating as follows
     * using the ODE callback:
     *
     * The step implements a Runge-Kutta 4 scheme with the following tableau:
     * \code
     *      0  |  0   0   0   0
     *     1/2 | 1/2  0   0   0
     *     1/2 |  0  1/2  0   0
     *      1  |  0   0   1   0
     *     ----+----------------
     *         | 1/6 1/3 1/3 1/6
     * \endcode
     * For the ode in the form:
     *  \f[
     *    \dot{x}(t) = f(t, x(t), u(t), p), \quad
     *    x(t_0) = x_0
     *  \f]
     * The integration scheme evaluates:
     *  \f[
     *    x(t_k + h) = x(t_k) + t_s \sum_{i = 1}^{4} b_i k_i
     *  \f]
     * where:
     *  \f[
     *    k_i = f(t + c_i t_s, x(t_k) + t_s \sum_{j=1}^{i-1} a_{i,j} k_j, u(t_k), p)
     *  \f]
     *
     * The input \f$ u(t_k) \f$ is assumed constant between integration ministeps.
     *
     * \param xp result from the next step in the integration
     * \param t flow time
     * \param x current state
     * \param u current input
     * \param p parameters vector
     * \param data user data
     */
    void next(std::array< REAL_T, dim_x > &xp, REAL_T &t, const std::array< REAL_T, dim_x > &x,
              const std::array< REAL_T, dim_u > &u, const std::array< REAL_T, dim_p > &p, Args... data) {
      const std::array< REAL_T, 4 > a = {0, 0.5, 0.5, 1};
      const std::array< REAL_T, 4 > b = {1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};
      const std::array< REAL_T, 4 > c = {0.0, 0.5, 0.5, 1.0};

      std::array< std::array< REAL_T, dim_x >, 4 > k = {0.0};
      std::array< REAL_T, dim_x > z;
      REAL_T tau = t;
      hybrid::math::copy(xp, x);

      this->f(k[0], tau, x, u, p, data...);
      hybrid::math::sum(xp, k[0], this->ts * b[0]);

      for (std::size_t i = 1; i < 4; i++) {
        tau = t + c[i] * this->ts;
        hybrid::math::copy(z, x);
        hybrid::math::sum(z, k[i - 1], a[i] * this->ts);
        this->f(k[i], tau, z, u, p, data...);
        hybrid::math::sum(xp, k[i], this->ts * b[i]);
      }
      
      t += this->ts;
    };
  };

};  // namespace hybrid

#endif /* RK4_HPP_ */
