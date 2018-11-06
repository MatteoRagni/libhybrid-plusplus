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

#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <functional>
#include <array>
#include <cstddef>

/**
 * \file types.hpp
 * \author Matteo Ragni
 * 
 * This file contains declarations for types that are used throughout the project
 */ 

namespace hybrid {
  /** \brief Type for an ODE function callback
   *
   * The type is a prototype for the RHS of an ODE callback, it tries to follow the same scheme
   * of the functions as required by the parametric estimation toolbox in MATLAB.
   * \f[
   *   \dot{x}(t) = f(t, x(t), u(t), p), \quad
   *   t \in \mathbb{R}_+, \,
   *   x(t) \in \mathbb{R}^{\dim(x)}, \,
   *   u(t) \in \mathbb{R}^{\dim(u)}, \,
   *   p \in \mathbb{R}^{\dim(p)}
   * \f]
   * The actual implementation follows this calling convention:
   * ```
   * f(dx, t, x, u, p, ...)
   * ```
   * \param dx the output of the function
   * \param t the current time
   * \param x the current state
   * \param u the current input
   * \param p the paramenter vector
   *
   * \tparam REAL_T the floating point precision type
   * \tparam dim_x size of the unknown vector
   * \tparam dim_u size of the external input vector
   * \tparam dim_p size of the parameter vector
   * \tparam Args variadic user arguments
   */
  template < typename REAL_T, std::size_t dim_x, std::size_t dim_u, std::size_t dim_p, typename... Args >
  using OdeFunction =
      std::function< void(std::array< REAL_T, dim_x > &dx, REAL_T t, const std::array< REAL_T, dim_x > &x,
                          const std::array< REAL_T, dim_u > &u, const std::array< REAL_T, dim_p > &p, Args...) >;

  /** \brief Type for an ODE Gradient callback
   *
   * The type is a prototype for the RHS of an ODE gradient callback, it tries to follow the same scheme
   * of the functions as required by the parametric estimation toolbox in MATLAB.
   * \f[
   *   \dot{x}(t) = f(t, x(t), u(t), p), \quad
   *   t \in \mathbb{R}_+, \,
   *   x(t) \in \mathbb{R}^{\dim(x)}, \,
   *   u(t) \in \mathbb{R}^{\dim(u)}, \,
   *   p \in \mathbb{R}^{\dim(p)}
   * \f]
   * The actual implementation follows this calling convention:
   * ```
   * df(dx, t, x, u, p, ...)
   * ```
   * \param dx the output of the function
   * \param t the current time
   * \param x the current state
   * \param u the current input
   * \param p the paramenter vector
   *
   * \tparam REAL_T the floating point precision type
   * \tparam dim_x size of the unknown vector
   * \tparam dim_u size of the external input vector
   * \tparam dim_p size of the parameter vector
   * \tparam Args variadic user arguments
   */
  template < typename REAL_T, std::size_t dim_x, std::size_t dim_u, std::size_t dim_p, typename... Args >
  using OdeGradient =
      std::function< void(std::array< REAL_T, dim_x * dim_x > &dx, REAL_T t, const std::array< REAL_T, dim_x > &x,
                          const std::array< REAL_T, dim_u > &u, const std::array< REAL_T, dim_p > &p, Args...) >;

  /**
   * @brief Type for solver: target function
   * 
   * The type implements a solver target function that takes as first argument the output and as second
   * the target variables array. The type is templetized in order to accomodate different types. The 
   * calling convention follows the followig rule:
   * ```
   * g(y, x, ...)
   * ```
   * \param y the output of the target function
   * \param x the target variable input
   * 
   * \tparam REAL_T the floating point precision type
   * \tparam dim_g the dimension of the output array
   * \tparam dim_x the dimension of the target variable array
   * \tparam Args variadic user arguments
   */
  template < typename REAL_T, std::size_t dim_g, std::size_t dim_x, typename... Args >
  using TargetFunction =
      std::function< void(std::array< REAL_T, dim_g > &y, const std::array< REAL_T, dim_x > &x, Args...) >;

  /**
   * @brief Type for solver: target function gradient
   *
   * The type implements a solver target function gradient that takes as first argument the output and as second
   * the target variables array. The type is templetized in order to accomodate different types. The
   * calling convention follows the followig rule:
   * ```
   * dg(dy, x, ...)
   * ```
   * \param dy the output of the target function
   * \param x the target variable input
   *
   * \tparam REAL_T the floating point precision type
   * \tparam dim_g the dimension of the output array
   * \tparam dim_x the dimension of the target variable array
   * \tparam Args variadic user arguments
   */
  template < typename REAL_T, std::size_t dim_g, std::size_t dim_x, typename... Args >
  using TargetGradient =
      std::function< void(std::array< REAL_T, dim_g * dim_x > &dy, const std::array< REAL_T, dim_x > &x, Args...) >;
}; 

#endif /* TYPES_HPP_ */
