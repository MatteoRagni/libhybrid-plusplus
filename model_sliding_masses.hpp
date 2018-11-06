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
 * \file model_sliding_masses.hpp
 * \author Matteo Ragni
 *
 * The file implements the model for a two sliding masses with relative motion.
 * The biggest mass slides, while the smallest slides on the biggest. The
 * sliding for the smaller is limited by the boundaries on the biggest.
 * that inherits from the \p hybrid::Hybrid class with an implicit integration scheme.
 * In this particular case the integration scheme is an Implicit Euler if
 * \p INTEGRATOR_IMPLICIT is defined. If not, a Runge Kutta 4th order scheme
 * is used.
 *
 * The file implements the following model:
 *
 * \f[
 *  \left\{\begin{array}{rcl}
 *   x^+_1 & = & x_1 \\
 *   x^+_2 & = & \frac{(m_1 - m_2) x_2 + 2 m_2 x_4}{m_1 - m_2} \\
 *   x^+_3 & = & x_3 \\
 *   x^+_4 & = & \frac{(m_2 - m_1) x_4 + 2 m_1 x_2}{m_1 - m_2} \\
 *  \end{array}\right. \quad \mathrm{if} \; (x, u) \in \mathcal{D}
 * \f]
 * \f[
 *  \left\{\begin{array}{rcl}
 *   \dot{x}_1 & = & x_2 \\
 *   \dot{x}_2 & = & -\frac{k_1}{m_1} x_1 - \frac{c_1}{m_1} x_2 + u \\
 *   \dot{x}_3 & = & x_4 \\
 *   \dot{x}_4 & = & -\frac{k_2}{m_2} (x_3 - x_1) - \frac{c_2}{m_2} (x_4 - x_2)
 *  \end{array}\right. \quad \mathrm{otherwise}
 * \f]
 *
 * \f[
 *   \mathcal{D} = \left\{
 *     (x, u) \, : \,
 *      (x_3 - x_1) \geq \frac{d}{2} \; \mathrm{and} \; (x_4 - x_2) \geq 0
 *      \quad \mathrm{or} \quad
 *      (x_3 - x_1) \leq -\frac{d}{2} \; \mathrm{and} \; (x_4 - x_2) \leq 0
 *   \right\}
 * \f]
 *
 * To use the implicit integrator, the Jacobian of the flow map is required,
 * and in this case it is defined as follows:
 *
 * \f[
 *   JF(x) = \begin{bmatrix}
 *    0 & -\frac{k_1}{m_1} & 0 & \frac{k_2}{m_2} \\
 *    1 & -\frac{c_1}{m_1} & 0 & \frac{c_2}{m_2} \\
 *    0 & 0                & 0 & -\frac{k_2}{m_2} \\
 *    0 & 0                & 1 & -\frac{c_2}{m_2}
 *   \end{bmatrix}
 * \f]
 */

#ifndef MODEL_SLIDING_MASSES_HPP
#define MODEL_SLIDING_MASSES_HPP

#include "hybrid.hpp"
#ifdef INTEGRATOR_IMPLICIT
#include "implicit_euler.hpp"
#else
#include "rk4.hpp"
#endif
#include "math.hpp"

const std::size_t dim_x = 4; /**< State dimension */
const std::size_t dim_u = 1; /**< Input dimension */
const std::size_t dim_p = 7; /**< Parameters dimension */
const std::size_t dim_y = 2; /**< Output dimension */

using State = std::array<double, dim_x>; /**< State array, with compile time sizes */
using Input = std::array<double, dim_u>; /**< Input array, with compile time sizes */
using Params = std::array<double, dim_p>; /**< Parameters array, with compile time sizes */
using StateJacobian = std::array<double, dim_x * dim_x>; /**< State Jacobian for the flow map, with compile time sizes */
using Output = std::array<double, 2>; /**< Output array, with compile time sizes */

/** \brief Parameters enumerations (for clarity in code) */
typedef enum p_idx {
  k1 = 0, /**< Stiffness of the first mass */
  c1,     /**< Dampening of the first mass */
  m1,     /**< Value of the first mass */
  k2,     /**< Stiffness of the second mass connected to the first mass */
  c2,     /**< Dampening of the second mass connected to the first mass */
  m2,     /**< Value of the second mass */
  d       /**< Dimension of the first mass (limits for the sliding of the second) */
} p_idx;

#ifdef INTEGRATOR_IMPLICIT
typedef hybrid::ImplicitEuler<double, dim_x, dim_u, dim_p, double> IntegrationScheme;
#else
typedef hybrid::RK4<double, dim_x, dim_u, dim_p, double> IntegrationScheme;
#endif
typedef hybrid::Hybrid<double, dim_x, dim_u, dim_p, dim_y> HybridModel;

/**
 * \brief The actual sliding masses class
 *
 * This is the class that the user has to implement in order to use the
 * library. The class implements (as required per implementation):
 *  - a flow map
 *  - a jump map
 *  - a jump set
 *  - a flow set
 *  - an output map
 *  - a flow jacobian (only for implicit integrator)
 * Futhermore the class need to override (as for now, will be removed in future releases)
 * the \p prepare_callbacks method in order to instruct the integrator.
 */
class SlidingMasses : public HybridModel, public IntegrationScheme {
public:
#ifdef INTEGRATOR_IMPLICIT
  /** \brief The constructor for the model with implicit integrator
   *
   * This is the constructor for the model. It needs to call the \p prepare_callback for the
   * implicit integration scheme.
   * \param _ts the sampling time for the integrator
   * \param _maxT extremum for the continuos part of the hybrid time
   * \param _maxJ extremum for the discrete part of the hybrid time
   * \param _tol tolerance for the optimization problem of the implicit integrator
   * \param _iter maximum iterations for the optimization problem of the implicit integrator
   */
  SlidingMasses(double _ts, double _maxT, double _maxJ, double _tol, int _iter)
  : HybridModel(_maxT, _maxJ), IntegrationScheme(_ts, _tol, _tol, _iter) {
    this->prepare_callbacks();
  };
#else
  /** \brief The constructor for the model with Runge-Kutta 4th order integrator
   *
   * This is the constructor for the model. It needs to call the \p prepare_callback for the
   * implicit integration scheme.
   * \param _ts the sampling time for the integrator
   * \param _maxT extremum for the continuos part of the hybrid time
   * \param _maxJ extremum for the discrete part of the hybrid time
   * \param _tol not used
   * \param _iter not used
   */
  SlidingMasses(double _ts, double _maxT, double _maxJ, double _tol, int _iter)
  : HybridModel(_maxT, _maxJ), IntegrationScheme(_ts) {
    this->prepare_callbacks();
  };
#endif
  
  /** \brief Flow Map implementation
   *
   * The flow map implements the following relation
   *
   * \f[
   *  \left\{\begin{array}{rcl}
   *   \dot{x}_1 & = & x_2 \\
   *   \dot{x}_2 & = & -\frac{k_1}{m_1} x_1 - \frac{c_1}{m_1} x_2 + u \\
   *   \dot{x}_3 & = & x_4 \\
   *   \dot{x}_4 & = & -\frac{k_2}{m_2} (x_3 - x_1) - \frac{c_2}{m_2} (x_4 - x_2)
   *  \end{array}\right.
   * \f]
   *
   * \param dx result for the flow map calculation
   * \param t flow time
   * \param k jump time
   * \param x current state
   * \param u current input
   * \param p parameters vector
   */
  void FlowMap(State &dx, double t, double k, const State &x,
               const Input &u, const Params &p) const override {
    dx[0] = x[1];
    dx[1] = -p[p_idx::k1]/p[p_idx::m1] * x[0] - p[p_idx::c1]/p[p_idx::m1] * x[1] + u[0];
    dx[2] = x[3];
    dx[3] = -p[p_idx::k2]/p[p_idx::m2] * (x[2] - x[0]) - p[p_idx::c2]/p[p_idx::m2] * (x[3] - x[1]);
  };
  
  /** \brief Jump Map implementation
   *
   * The jump map implements the following relation
   *
   * \f[
   *  \left\{\begin{array}{rcl}
   *   x^+_1 & = & x_1 \\
   *   x^+_2 & = & \frac{(m_1 - m_2) x_2 + 2 m_2 x_4}{m_1 - m_2} \\
   *   x^+_3 & = & x_3 \\
   *   x^+_4 & = & \frac{(m_2 - m_1) x_4 + 2 m_1 x_2}{m_1 - m_2} \\
   *  \end{array}\right.
   * \f]
   *
   * \param xp result for the jump map calculation
   * \param t flow time
   * \param k jump time
   * \param x current state
   * \param u current input
   * \param p parameters vector
   */
  void JumpMap(State &xp, double t, double k, const State &x, const Input &u, const Params &p) const override {
    hybrid::math::copy(xp, x);
    xp[0] = x[0];
    xp[1] = ((p[p_idx::m1] - p[p_idx::m2]) * x[1] + 2 * p[p_idx::m2] * x[3])/(p[p_idx::m1] + p[p_idx::m2]);
    xp[2] = x[2];
    xp[3] = ((p[p_idx::m2] - p[p_idx::m1]) * x[3] + 2 * p[p_idx::m1] * x[1])/(p[p_idx::m1] + p[p_idx::m2]);
  };
  
  /** \brief Jump set implementation
   *
   * The jump set returns \p true when the state is in the Jump set,
   * \p false when is not. The set for the masses responds to the
   * following representation:
   *
   * \f[
   *  (x_3 - x_1) \geq \frac{d}/2 \; \mathrm{and} \; (x_4 - x_2) \geq 0
   *  \quad \mathrm{or} \quad
   *  (x_3 - x_1) \leq -\frac{d}/2 \; \mathrm{and} \; (x_4 - x_2) \leq 0
   * \f]
   *
   * \param t flow time
   * \param k jump time
   * \param x current state
   * \param u current input
   * \param p parameters vector
   * \return if we are inside the jump set through a bool
   */
  bool JumpSet(double t, double k, const State &x, const Input &u,
               const Params &p) const override {
    if (((x[2] - x[0]) >= p[p_idx::d]/2) && ((x[3] - x[1]) >= 0))
      return true;
    if (((x[2] - x[0]) <= -p[p_idx::d]/2) && ((x[3] - x[1]) <= 0))
      return true;
    return false;
  };
  
  /** \brief Flow set implementation
   *
   * The flow set returns \p true always, but class is configured
   * in such a way the jump set has a precedence for the evolution
   * of the solution.
   *
   * \param t flow time
   * \param k jump time
   * \param x current state
   * \param u current input
   * \param p parameters vector
   * \return if we are inside the flow set through a bool
   */
  bool FlowSet(double t, double k, const State &x, const Input &u,
               const Params &p) const override {
    return true;
  };
  
  /** \brief Output map implementation
   *
   * The output map returns the output of the system (positions of
   * the two masses).
   *
   * \param y current output to evaluate
   * \param t flow time
   * \param k jump time
   * \param x current state
   * \param u current input
   * \param p parameters vector
   */
  void OutputMap(Output &y, double t, double k, const State &x,
                 const Input &u, const Params &p) const override {
    y[0] = x[0];
    y[1] = x[2];
  };
  
  /** \brief Flow map jacobian implementation
   *
   * The flow map jacobian is needed by the implicit inteerator in
   * order to solve the implicit step. The Matrix of the jacobian is
   * in column major ordering
   *
   * \param ddx current output of the Jacobian of the flow map
   * \param t flow time
   * \param k jump time
   * \param x current state
   * \param u current input
   * \param p parameters vector
   */
  void FlowMapJacobian(StateJacobian &ddx, double t, double k, const State &x,
                       const Input &u, const Params &p) const {
    ddx.fill(0.0);
    ddx[1]  = -p[p_idx::k1] / p[p_idx::m1];
    ddx[3]  = p[p_idx::k2] / p[p_idx::m2];
    ddx[4]  = 1.0;
    ddx[5]  = -p[p_idx::c1] / p[p_idx::m1];
    ddx[7]  = p[p_idx::c2] / p[p_idx::m2];
    ddx[11] = -p[p_idx::k2] / p[p_idx::m2];
    ddx[14] = 1.0;
    ddx[15] = -p[p_idx::c2] / p[p_idx::m2];
  };
  
  /** \brief Override jump default behavior (uses the jump set only)
   *
   * \param t flow time
   * \param k jump time
   * \param x current state
   * \param u current input
   * \param p parameters vector
   * \return the logic combination of Jump Set and Flow Set information
   */
  bool should_jump(const double t, const double k, const State &x,
                   const Input &u, const Params &p) override {
    return this->JumpSet(t, k, x, u, p);
  };
  
  /** \brief next step of integration
   *
   * (to be removed in next release)
   *
   * \param xp result from the next step in the integration
   * \param t flow time
   * \param k jump time
   * \param x current state
   * \param u current input
   * \param p parameters vector
   */
  void next(State &xp, double &t, const State &x,
            const Input &u, const Params &p, double k) override {
    IntegrationScheme::next(xp, t, x, u, p, k);
  };
#ifdef INTEGRATOR_IMPLICIT
  /** \brief this function preprares the callback for the integrator
   *
   * (to be removed in next release)
   */
  void prepare_callbacks() override {
    this->f = [this](State &dx, double t, const State &x, const Input &u, const Params &p, double k) -> void {
      this->FlowMap(dx, t, k, x, u, p);
    };
    this->df = [this](StateJacobian &ddx, double t, const State &x, const Input &u, const Params &p, double k) -> void {
      this->FlowMapJacobian(ddx, t, k, x, u, p);
    };
  };
#else
  /** \brief this function preprares the callback for the integrator
   *
   * (to be removed in next release)
   */
  void prepare_callbacks() override {
    this->f = [this](State &dx, double t, const State &x,
                     const Input &u, const Params &p, double k) -> void {
      this->FlowMap(dx, t, k, x, u, p);
      
    };
  }
#endif
};

#endif /* MODEL_SLIDING_MASSES_HPP */
