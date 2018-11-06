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
 * \file implicit_euler.hpp
 * \author Matteo Ragni
 *
 * The file implements an Implicit Euler scheme:
 * \f{equation}
 *    x^+ = x + t_s \, f(x^+, u, p)
 * \f
 * The next step of the integration is actually the solution of an optimization problem
 * trough a Newton solver. The initial point for the solver is the current state.
 */
#ifndef IMPLICIT_EULER_HPP_
#define IMPLICIT_EULER_HPP_

#include <iostream>
#include "integrator.hpp"
#include "math.hpp"
#include "newton.hpp"
#include "types.hpp"

namespace hybrid {
  /** \brief Class for Implicit Euler integrator
   *
   * This is the implementation of an Implicit Euler integration scheme that does not
   * use dynamical memory allocation. All dimensions are defined in a static way, through templating.
   * The ode is required to be defined as:
   *
   * \f[
   *   \dot{x}(t) = f(t, x(t), u(t), p), \qquad
   *     x(t) \in \mathbb{R}^{\dim(x)}, \,
   *     u(t) \in \mathbb{R}^{\dim(u)} \,
   *     p \in \mathbb{R}^{\dim(p)}
   * \f]
   *
   * The implicit step is solved using a Newton Raphson algorithm.
   *
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
  class ImplicitEuler
      : public ImplicitIntegrator< REAL_T, dim_x, dim_u, dim_p, Args... >,
        public NewtonRaphson< REAL_T, dim_x, dim_x, REAL_T, const std::array< REAL_T, dim_x > &,
                              const std::array< REAL_T, dim_u > &, const std::array< REAL_T, dim_p > &, Args... > {
   public:
   /** \brief Contructor with function callback for ODE and Jacobian of ODE
    *
    * Constructor with function callback for ODE and for Jacobian of the ODE
    *
    * \param _ts sampling time
    * \param _tol tolerance for zeros search
    * \param _xtol tolerance for solution step search
    * \param _max_iter maximum number of iterations for zero search
    * \param _f callback for the ODE
    * \param _df callback for the Jacobian of the ODE
    */
    ImplicitEuler(REAL_T _ts, REAL_T _tol, REAL_T _xtol, int _max_iter,
                  OdeFunction< REAL_T, dim_x, dim_u, dim_p, Args... > _f,
                  OdeGradient< REAL_T, dim_x, dim_u, dim_p, Args... > _df)
        : ImplicitIntegrator< REAL_T, dim_x, dim_u, dim_p, Args... >(_ts, _f, _df),
          NewtonRaphson< REAL_T, dim_x, dim_x, REAL_T, const std::array< REAL_T, dim_x > &,
                         const std::array< REAL_T, dim_u > &, const std::array< REAL_T, dim_p > &, Args... >(
              _tol, _xtol, _max_iter) {
      for (std::size_t i = 0; i <= dim_x; i++)
        this->eye[dim_x * i + i] = 1.0;
      this->prepare_targets();
    };

    /** \brief Simple contructor for the integrator
     *
     * Simple contructor for the integrator
     *
     * \param _ts sampling time
     * \param _tol tolerance for zeros search
     * \param _xtol tolerance for solution step search
     * \param _max_iter maximum number of iterations for zero search
     */
    ImplicitEuler(REAL_T _ts, REAL_T _tol, REAL_T _xtol, int _max_iter)
        : ImplicitIntegrator< REAL_T, dim_x, dim_u, dim_p, Args... >(_ts),
          NewtonRaphson< REAL_T, dim_x, dim_x, REAL_T, const std::array< REAL_T, dim_x > &,
                         const std::array< REAL_T, dim_u > &, const std::array< REAL_T, dim_p > &, Args... >(
              _tol, _xtol, _max_iter) {
      for (std::size_t i = 0; i <= dim_x; i++) this->eye[dim_x * i + i] = 1.0;
      this->f = nullptr;
      this->df = nullptr;
      this->prepare_targets();
    };
    
    /** \brief The next step in the computation
     *
     * The function evaluates the next step for te integration, evaluating as follows
     * using the ODE callback:
     *
     * \f[
     *    x^+ = x + t_s \, f(x^+, u, p)
     * \f]
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
      hybrid::math::copy(xp, x);
      try {
        this->solve(xp, t, x, u, p, data...);
      } catch (hybrid::XTolException exc) {
        std::cerr << "X tolerance update issue at t = " << t << "s" << std::endl;
      }
      t += this->ts;
    };

   private:
    std::array< REAL_T, dim_x * dim_x > eye;
    
    /** \brief Prepare targets for the inner integrator and solver */
    void prepare_targets() {
      this->g = [this](std::array< REAL_T, dim_x > &r, const std::array< REAL_T, dim_x > &x, REAL_T t,
                       const std::array< REAL_T, dim_x > &x0, const std::array< REAL_T, dim_u > &u,
                       const std::array< REAL_T, dim_p > &p, Args... data) -> void {
        this->f(r, t, x, u, p, data...);   // r = f(t, x+, u, p)
        hybrid::math::scale(r, this->ts);  // r = h f(t, x+, u, p)
        hybrid::math::sum(r, x0);          // r = x + h f(t, x+, u, p)
        hybrid::math::scale(r, -1.0);      // r = -r = - x - h f(t, x+, u, p)
        hybrid::math::sum(r, x);           // r = r + x+ = x+ - x - h f(t, x+, u, p)
      };

      
      this->dg = [this](std::array< REAL_T, dim_x * dim_x > &dr, const std::array< REAL_T, dim_x > &x, REAL_T t,
                        const std::array< REAL_T, dim_x > &x0, const std::array< REAL_T, dim_u > &u,
                        const std::array< REAL_T, dim_p > &p, Args... data) -> void {
        this->df(dr, t, x, u, p, data...);   // dr = df(t, x+, u, p)
        hybrid::math::scale(dr, -this->ts);  // dr = - h dr = -h df(t, x+, u, p)
        hybrid::math::sum(dr, this->eye);    // dr = I + dr = I - h df(t, x+, u, p)
      }; 
      
    }
  };

};  // namespace hybrid

#endif /* IMPLICIT_EULER_HPP_ */
