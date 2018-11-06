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
 * \file explicit_euler.hpp
 * \author Matteo Ragni
 *
 * The file implements an explicit Euler scheme:
 * \f[
 *    x^+ = x + t_s \, f(x, u, p)
 * \f]
 */
#ifndef EXPLICIT_EULER_HPP_
#define EXPLICIT_EULER_HPP_

#include "integrator.hpp"
#include "math.hpp"

namespace hybrid {
  /** \brief Class for Explicit Euler integrator
   *
   * This is the implementation of an Explicit Euler integration scheme that does not
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
  class ExplicitEuler : public Integrator< REAL_T, dim_x, dim_u, dim_p, Args... > {
    
   public:
    /** \brief Contructor with function callback for ODE
     *
     * Constructor with function callback for ODE
     *
     * \param _ts sampling time
     * \param _f callback for the ODE
     */
    ExplicitEuler(REAL_T _ts, OdeFunction< REAL_T, dim_x, dim_u, dim_p, Args... > _f)
        : Integrator< REAL_T, dim_x, dim_u, dim_p, Args... >(_ts, _f){};
    
    /** \brief Simple constructor, only sampling time
     *
     * Simple constructor with only sampling time
     *
     * \param _ts sampling time
     */
    ExplicitEuler(REAL_T _ts) : Integrator< REAL_T, dim_x, dim_u, dim_p, Args... >(_ts){};
    
    /** \brief The next step in the computation
     *
     * The function evaluates the next step for te integration, evaluating as follows
     * using the ODE callback:
     *
     * \f[
     *    x^+ = x + t_s \, f(x, u, p)
     * \f]
     *
     * \param xp result from the next step in the integration
     * \param t flow time
     * \param x current state
     * \param u current input
     * \param p parameters vector
     * \param data user data
     */
    void next(std::array<REAL_T, dim_x> &xp, REAL_T &t, const std::array<REAL_T, dim_x> &x,
              const std::array<REAL_T, dim_u> &u, const std::array<REAL_T, dim_p> &p, Args... data) {
      this->f(xp, t, x, u, p, data...);
      hybrid::math::scale(xp, this->ts);
      hybrid::math::sum(xp, x);
      t += this->ts;
    };
  };

};  // namespace hybrid

#endif /* EXPLICIT_EULER_HPP_ */
