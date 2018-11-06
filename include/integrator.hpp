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
 * \file integrator.hpp
 * \author Matteo Ragni
 *
 * Abstract class for explicit and implicit integrators. The explicit integrator uses
 * current states to evaluate the future step, while the implicit integrator solves
 * an optimization problem to evaluate the next step.Thus, usually implicit integrator
 * are slower, but perform better with stiff problem and large integration steps.
 *
 * The explicit integrators directly inherit from the Integrator Class, that requires
 * only the ODE function pointer for the integration.
 * The implicit integrators inherit from the Implicit Integrator Class and from a solver
 * class. As for noe they require the gradient to evaluate the solution of the equations,
 * since the only solver available is a Newton solver.
 */
#ifndef INTEGRATOR_HPP_
#define INTEGRATOR_HPP_

#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <type_traits>
#include "types.hpp"

namespace hybrid {

  /** \brief Abstract class for integrators
   *
   * This is an abstract class for integrators that do not use dynamical memory allocation. All dimensions
   * are defined in a static way, through templating.
   * The ode is required to be defined as:
   * \f[
   *   \dot{x}(t) = f(t, x(t), u(t), p), \qquad
   *     x(t) \in \mathbb{R}^{\dim(x)}, \,
   *     u(t) \in \mathbb{R}^{\dim(u)} \,
   *     p \in \mathbb{R}^{\dim(p)}
   * \f]
   * The ODE styles is in same way compatible with the MATLAB System identification toolbox idnlgrey
   * function call. The only difference is in the parameter vector that in this case is in a single
   * array (instead of a `REAL_T**` vector).
   *
   * \tparam REAL_T floating point precision
   * \tparam dim_x dimension of the state array
   * \tparam dim_u dimension of the external forces array
   * \tparam dim_p dimension of the parameters array
   * \tparam Args variadic template for user arguments
   */
  template < typename REAL_T, std::size_t dim_x, std::size_t dim_u, std::size_t dim_p, typename... Args >
  class Integrator {
   
   public:
    REAL_T ts;                                             /**< Integration step */
    OdeFunction< REAL_T, dim_x, dim_u, dim_p, Args... > f; /**< Ode function pointer */

    /** \brief a generic constructor for integrators
     *
     * The generic constructor checks that integration step is \f$ t_s > 0 \f$ and that
     * the function pointer is not null
     *
     * \param _ts the sampling time
     * \param _f the ode function pointer
     */
    Integrator(REAL_T _ts, OdeFunction< REAL_T, dim_x, dim_u, dim_p, Args... > _f) : ts(_ts), f(_f) {
      static_assert(std::is_floating_point< REAL_T >::value, "Only float/double accepted");
      if (ts <= 0) throw std::invalid_argument("Integration step cannot be negative");
      if (f == nullptr) throw std::invalid_argument("A valid ODE function is required");
    };

    Integrator(REAL_T _ts) : ts(_ts) {
      static_assert(std::is_floating_point< REAL_T >::value, "Only float/double accepted");
      if (ts <= 0) throw std::invalid_argument("Integration step cannot be negative");
    };

    /** \brief The method evaluates the next step in the integration
     *
     * The method gets as input the current ode value, the current time, the current parameters and the
     * current input and outputs the next time and the next state vector. This function is purely `virtual`
     * (and in fact \p Integrator is an abstract class), thus it must be overriden in the final integrator.
     *
     * \param xp the array that contains the value of the state at the next time step
     * \param t current time. This variable will contains the next time (\f$ t^+ = t + t_s\f$)
     * \param x current state vector. This array is constant
     * \param u current input vector. This array is constant
     * \param p current parameters vector. This array is constant
     * \param data user data space (as variadic argument)
     */
    virtual void next(std::array<REAL_T, dim_x> &xp, REAL_T &t, const std::array<REAL_T, dim_x> &x,
          const std::array<REAL_T, dim_u> &u, const std::array<REAL_T, dim_p> &p, Args... data) = 0;
  };

  /** \brief Abstract class for implicit integrators
   *
   * This is an abstract class for integrators that do not use dynamical memory allocation. All dimensions
   * are defined in a static way, through templating. In this case the integrator that inherit from this
   * class are implicit, i.e. evaluates next time step through the solution of an optimization problem.
   * As for now, the only solver available is a Newton solver, thus the Implicit Integrator must have the
   * ode gradient:
   * \f[
   *   X = \nabla_{x} f (t, x(t), u(t), p), \qquad X \in \mathbb{R}^{\dim(x) \times \dim(x)}
   * \f]
   * To work properly, the integrator should receive the next input value, but this is left to the user,
   * through the \p next method.
   *
   * \tparam REAL_T floating point precision
   * \tparam dim_x dimension of the state array
   * \tparam dim_u dimension of the external forces array
   * \tparam dim_p dimension of the parameters array
   * \tparam Args variadic template for user arguments
   */
  template < typename REAL_T, std::size_t dim_x, std::size_t dim_u, std::size_t dim_p, typename... Args >
  class ImplicitIntegrator : public Integrator< REAL_T, dim_x, dim_u, dim_p, Args... > {
   public:
    OdeGradient< REAL_T, dim_x, dim_u, dim_p, Args... > df; /**< Ode function gradient pointer */

    /** \brief a generic constructor for implicit integrators
     *
     * The generic constructor checks that integration step is \f$ t_s > 0 \f$ and that
     * the function pointer is not null, by calling \p Integrator constructor. It also
     * takes a gradient as an input and checks that the function pointer is not null.
     *
     * \param _ts the sampling time
     * \param _f the ode function pointer
     * \param _df the ode gradient function pointer
     */
    ImplicitIntegrator(REAL_T _ts, OdeFunction< REAL_T, dim_x, dim_u, dim_p, Args... > _f,
                       OdeGradient< REAL_T, dim_x, dim_u, dim_p, Args... > _df)
        : Integrator< REAL_T, dim_x, dim_u, dim_p, Args... >(_ts, _f), df(_df) {
      if (df == nullptr) throw std::invalid_argument("A valid ODE gradient is required");
    };

    ImplicitIntegrator(REAL_T _ts) : Integrator< REAL_T, dim_x, dim_u, dim_p, Args... >(_ts){};

    /** \brief Internal method that should setup the inner implicit target for solver
     *
     * This function is integrator specific and it is `virtual` in this class. The purpose
     * of the method is to build the target for the solver by using the ode and the ode gradient.
     * In my mind, this function should define two lambdas.
     */
    virtual void prepare_targets() = 0;
  };
};  // namespace hybrid

#endif /* INTEGRATOR_HPP_ */
