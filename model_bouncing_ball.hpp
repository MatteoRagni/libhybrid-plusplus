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
 * \file model_bouncing_ball.hpp
 * \author Matteo Ragni
 *
 * The file implements the model for a bouncing ball in the form of a simple class
 * that inherits from the \p hybrid::Hybrid class with an implicit integration scheme.
 * In this particular case the integration scheme is an Implicit Euler.
 *
 * The file implements the following model:
 *
 * \f[
 *  \left\{\begin{array}{rclcrcl}
 *   \dot{x}_1 & = & x_2 & \qquad   &     &      & \\
 *   \dot{x}_2 & = & -g  & \qquad if & x_1 & \geq & 0
 *  \end{array}\right.
 * \f]
 * \f[
 *  \left\{\begin{array}{rclcrcl}
 *   {x}^+_1 & = & 0 & \qquad   &     &      & \\
 *   {x}^+_2 & = & -\gamma x_2  & \qquad if & x_1 & \leq & 0
 *  \end{array}\right.
 * \f]
 *
 * To use the implicit integrator, the Jacobian of the flow map is required,
 * and in this case it is defined as:
 *
 * \f[
 *  JF(x) = \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix}
 * \f]
 */
#ifndef MODEL_BOUNCING_BALL_HPP
#define MODEL_BOUNCING_BALL_HPP

#include "hybrid.hpp"
#include "implicit_euler.hpp"
#include "math.hpp"

const std::size_t dim_x = 2; /**< State dimension */
const std::size_t dim_u = 1; /**< Input dimension */
const std::size_t dim_p = 2; /**< Parameters dimension */
const std::size_t dim_y = dim_x; /**< Output dimension */

typedef std::array<double, dim_x> State; /**< State array, with compile time sizes */
typedef std::array<double, dim_u> Input; /**< Input array, with compile time sizes */
typedef std::array<double, dim_p> Params; /**< Parameters array, with compile time sizes */
typedef std::array<double, dim_x * dim_x> StateJacobian; /**< State Jacobian for the flow map, with compile time sizes */
typedef std::array<double, dim_y> Output; /**< Output array, with compile time sizes */

typedef hybrid::ImplicitEuler<double, dim_x, dim_u, dim_p, double> IntegrationScheme; /**< Integration scheme */
typedef hybrid::Hybrid<double, dim_x, dim_u, dim_p, dim_y> HybridModel; /**< Hybrid model with sizes */

/**
 * \brief The actual bouncing ball class
 *
 * This is the class that the user has to implement in order to use the
 * library. The class implements (as required per implementation):
 *  - a flow map
 *  - a jump map
 *  - a jump set
 *  - a flow set
 *  - an output map
 *  - a flow jacobian
 * Futhermore the class need to override (as for now, will be removed in future releases)
 * the \p prepare_callbacks method in order to instruct the integrator.
 */
class BouncingBall : public HybridModel, public IntegrationScheme {
public:
  /** \brief The constructor for the model
   *
   * This is the constructor for the model. It needs to call the \p prepare_callback for the
   * implicit integration scheme.
   * \param _ts the sampling time for the integrator
   * \param _maxT extremum for the continuos part of the hybrid time
   * \param _maxJ extremum for the discrete part of the hybrid time
   * \param _tol tolerance for the optimization problem of the implicit integrator
   * \param _iter maximum iterations for the optimization problem of the implicit integrator
   */
  BouncingBall(double _ts, double _maxT, double _maxJ, double _tol, int _iter)
  : HybridModel(_maxT, _maxJ), IntegrationScheme(_ts, _tol, _tol, _iter) {
    this->prepare_callbacks();
  };
  
  /** \brief Flow Map implementation
   *
   * The flow map implements the following relation
   *
   * \f[
   *  \left\{\begin{array}{rcl}
   *   \dot{x}_1 & = & x_2 \\
   *   \dot{x}_2 & = & -g
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
    dx[1] = -p[0];
  };
  
  /** \brief Jump Map implementation
   *
   * The jump map implements the following relation
   *
   * \f[
   *  \left\{\begin{array}{rcl}
   *   {x}^+_1 & = & 0 \\
   *   {x}^+_2 & = & -\gamma x_2
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
    xp[0] = 0;
    xp[1] = -p[1] * x[1];
  };
  
  /** \brief Jump set implementation
   *
   * The jump set returns \p true when the state is in the Jump set,
   * \p false when is not.
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
    if (x[0] < 0)
      return true;
    return false;
  };
  
  /** \brief Flow set implementation
   *
   * The flow set returns \p true when the state is in the flow set,
   * \p false when is not.
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
    if (x[0] >= 0)
      return true;
    return false;
  };
  
  /** \brief Output map implementation
   *
   * The output map returns the output of the system.
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
    hybrid::math::copy(y, x);
  };
  
  /** \brief Flow map jacobian implementation
   *
   * The flow map jacobian is needed by the implicit inteerator in
   * order to solve the implicit step. The Matrix of the jacobian is
   * in column major ordering.
   *
   * \param ddx result for the flow map jacobian calculation
   * \param t flow time
   * \param k jump time
   * \param x current state
   * \param u current input
   * \param p parameters vector
   */
  void FlowMapJacobian(StateJacobian &ddx, double t, double k, const State &x,
                       const Input &u, const Params &p) const {
    ddx[0] = 0;
    ddx[1] = 0;
    ddx[2] = 1;
    ddx[3] = 0;
  }
  
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
};

#endif /* MODEL_BOUNCING_BALL_HPP */
