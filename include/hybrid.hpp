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
 * \mainpage
 * \author Matteo Ragni
 *
 * This file implement the abstract class  for the model of an hybrid system.
 * The hybrid system implemented follows:
 *
 *  \f{align}
 *    \dot{t}(\tau) = & 1          & \text{for } (x, u) \in C \\
 *    \dot{k}(\tau) = & 0          & \\
 *    \dot{x}(\tau) = & f(t, k, x, u, p) & \\
 *  \f}
 *  \f{align}
 *    t^+(\tau)    = & t           & \text{for } (x, u) \in D \\
 *    k^+(\tau)    = & k + 1      & \\
 *    x^+(\tau)    = & g(t, k, x, u, p)  & \\
 *  \f}
 *  \f{align}
 *    y = & h(x, u, p)
 *  \f}
 * where:
 *  * \f$ f \f$ is the flow map;
 *  * \f$ g \f$ is the jump map;
 *  * \f$ h \f$ is the output map;
 *  * \f$ C \f$ is the flow set;
 *  * \f$ D \f$ is the jump set.
 *  * \f$ p \f$ are the function parameters.
 *  * \f$ \tau \f$ is an engine time for the integration of \f$ t \f$ and \f$ k \f$.
 *
 * The flow map is discretized with a numerical integrator, from which the model
 * shall inherit. For the evolution of the system, both \f$ t \f$ and \f$ k \f$
 * are limited by horizons.
 *
 * There are two implemented examples (bouncing ball and sliding masses).
 */

#ifndef HYBRID_HPP_
#define HYBRID_HPP_

#include <array>
#include <cstddef>
#include <exception>
#include <functional>
#include <iostream>
#include "integrator.hpp"
#include "math.hpp"
#include "types.hpp"

namespace hybrid {
 /** \brief Abstract class for an hybrid system
  *
  * The hybrid system implemented follows:
  *
  *  \f{align}
  *    \dot{t}(\tau) = & 1          & \text{for } (x, u) \in C \\
  *    \dot{k}(\tau) = & 0          & \\
  *    \dot{x}(\tau) = & f(t, k, x, u, p) & \\
  *  \f}
  *  \f{align}
  *    t^+(\tau)    = & t           & \text{for } (x, u) \in D \\
  *    k^+(\tau)    = & k + 1      & \\
  *    x^+(\tau)    = & g(t, k, x, u, p)  & \\
  *  \f}
  *  \f{align}
  *    y = & h(x, u, p)
  *  \f}
  * where:
  *  - \f$ f \f$ is the flow map;
  *  - \f$ g \f$ is the jump map;
  *  - \f$ h \f$ is the output map;
  *  - \f$ C \f$ is the flow set;
  *  - \f$ D \f$ is the jump set.
  *  - \f$ p \f$ are the function parameters.
  *  - \f$ \tau \f$ is an engine time for the integration of \f$ t \f$ and \f$ k \f$.
  *
  * There are some purely virtual functions that **must** be implemented
  * by the user:
  *  - the flow map \f$ f \f$
  *  - the jump map \f$ g \f$
  *  - the jump set \f$ D \f$
  *  - the output map \f$ h \f$
  *  - (as for now, will be removed un future releases) the \p next method that calls the integrator
  *  - (iff using an implicit integrator) the Jacobian of the flow map \f$ Jf \f$
  *
  * The flow map is discretized with a numerical integrator, from which the model
  * shall inherit. For the evolution of the system, both \f$ t \f$ and \f$ k \f$
  * are limited by horizons.
  *
  * \tparam REAL_T floating point precision
  * \tparam dim_x dimension of the state array
  * \tparam dim_u dimension of the external forces array
  * \tparam dim_p dimension of the parameters array
  * \tparam dim_y dimension of the output array
  * \tparam Args variadic template for user arguments
  */
  template < typename REAL_T, std::size_t dim_x, std::size_t dim_u, std::size_t dim_p, std::size_t dim_y,
             typename... Args >
  class Hybrid {
   public:
    REAL_T time_limit; /**< Flow time horizon */
    REAL_T jumps_limit; /**< Jump time horizon, avoids pure Zeno solutions */
    
    /** \brief Constructor for an hybrid model
     *
     * The constructor sets the time and jump horizons.
     *
     * \param _time_limit flow time horizon
     * \param _jumps_limit discrete time horizon
     */
    Hybrid(REAL_T _time_limit, REAL_T _jumps_limit) : time_limit(_time_limit), jumps_limit(_jumps_limit) {
      if (time_limit <= 0)
        throw std::invalid_argument("Time limit must be strictly positive");
      if (jumps_limit <= 0)
        throw std::invalid_argument("Jumps limit must be strictly positive");
    }
    
    /** \brief Flow Map virtual function
     *
     * The flow map **must be implemented by the user**
     *
     *
     * \param dx result for the flow map calculation
     * \param t flow time
     * \param k jump time
     * \param x current state
     * \param u current input
     * \param p parameters vector
     * \param data user data
     */
    virtual void FlowMap(std::array< REAL_T, dim_x > &dx, REAL_T t, REAL_T k, const std::array< REAL_T, dim_x > &x,
                         const std::array< REAL_T, dim_u > &u, const std::array< REAL_T, dim_p > &p,
                         Args... data) const = 0;
    
    /** \brief Jump Map virtual function
     *
     * The jump map **must be implemented by the user**
     *
     *
     * \param xp result for the jump map calculation
     * \param t flow time
     * \param k jump time
     * \param x current state
     * \param u current input
     * \param p parameters vector
     * \param data user data
     */
    virtual void JumpMap(std::array< REAL_T, dim_x > &xp, REAL_T t, REAL_T k, const std::array< REAL_T, dim_x > &x,
                         const std::array< REAL_T, dim_u > &u, const std::array< REAL_T, dim_p > &p,
                         Args... data) const = 0;
    
    /** \brief Flow set implementation
     *
     * The flow set returns always \p false, and should be overriden by
     * the user for a different behavior.
     *
     * \param t flow time
     * \param k jump time
     * \param x current state
     * \param u current input
     * \param p parameters vector
     * \param data user data
     * \return if we are inside the flow set through a bool
     */
    virtual bool FlowSet(REAL_T t, REAL_T k, const std::array< REAL_T, dim_x > &x, const std::array< REAL_T, dim_u > &u,
                 const std::array< REAL_T, dim_p > &p, Args... data) const {
      return false;
    };
    
    /** \brief Jump Set virtual function
     *
     * The jump set **must be implemented by the user**
     *
     * \param t flow time
     * \param k jump time
     * \param x current state
     * \param u current input
     * \param p parameters vector
     * \param data user data
     * \return if we are inside the jump set through a bool
     */
    virtual bool JumpSet(REAL_T t, REAL_T k, const std::array< REAL_T, dim_x > &x, const std::array< REAL_T, dim_u > &u,
                         const std::array< REAL_T, dim_p > &p, Args... data) const = 0;
    
    /** \brief Output Map virtual function
     *
     * The output map **must be implemented by the user**
     *
     *
     * \param y result for the output map calculation
     * \param t flow time
     * \param k jump time
     * \param x current state
     * \param u current input
     * \param data user data
     * \param p parameters vector
     */
    virtual void OutputMap(std::array< REAL_T, dim_y > &y, REAL_T t, REAL_T k, const std::array< REAL_T, dim_x > &x,
                           const std::array< REAL_T, dim_u > &u, const std::array< REAL_T, dim_p > &p,
                           Args... data) const = 0;
    
    /** \brief Flow Map Jacobian virtual function
     *
     * The Jacobian of the flow map **must be implemented by the user** if an
     * implicit integration scheme will be used.
     *
     * \param ddx result for the flow map jacobian calculation
     * \param t flow time
     * \param k jump time
     * \param x current state
     * \param u current input
     * \param p parameters vector
     * \param data user data
     */
    void FlowMapJacobian(std::array< REAL_T, dim_x * dim_x > &ddx, REAL_T t, REAL_T k,
                         const std::array< REAL_T, dim_x > &x, const std::array< REAL_T, dim_u > &u,
                         const std::array< REAL_T, dim_p > &p, Args... data) const {
      return;
    };
    
    /** \brief Step fuction for the whole model
     *
     * This is an overall step for the model. Evaluates if the model shall flow or shall
     * jump. The function updates both flow time and jump time, and returns the next
     * value of the state.
     *
     * \param xp next value for the state
     * \param t flow time (will be updated in case of flow)
     * \param k jump time (will be updated in case of jump)
     * \param x current state
     * \param u current input
     * \param p parameters vector
     * \param data user data
     */
    void step(std::array< REAL_T, dim_x > &xp, REAL_T &t, REAL_T &k, const std::array< REAL_T, dim_x > &x,
              const std::array< REAL_T, dim_u > &u, const std::array< REAL_T, dim_p > &p, Args... data) {
      if (t > this->time_limit || k > this->jumps_limit) {
        throw std::runtime_error("Time or Jump limits reached");
      }

      if (this->should_jump(t, k, x, u, p, data...)) {
        k++;
        this->JumpMap(xp, t, k, x, u, p, data...);
      } else {
        this->next(xp, t, x, u, p, k, data...);
      }
    };
    
    /** \brief Decides if the model performs a jump or not.
     *
     * The normal behavior is the following:
     * \code
     *   if (x, u) in D, and not (x, u) in C then jump
     * \endcode
     *
     * \param t flow time
     * \param k jump time
     * \param x current state
     * \param u current input
     * \param p parameters vector
     * \param data user data
     * \return the logic combination of Jump Set and Flow Set information
     */
    virtual bool should_jump(const REAL_T t, const REAL_T k, const std::array< REAL_T, dim_x > &x,
                     const std::array< REAL_T, dim_u > &u, const std::array< REAL_T, dim_p > &p, Args... data) {
      bool js = this->JumpSet(t, k, x, u, p, data...);
      bool fs = this->FlowSet(t, k, x, u, p, data...);
      return (js && !(fs));  // In jump set but not in flow set
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
     * \param data user data
     */
    virtual void next(std::array< REAL_T, dim_x > &xp, REAL_T &t, const std::array< REAL_T, dim_x > &x,
                      const std::array< REAL_T, dim_u > &u, const std::array< REAL_T, dim_p > &p, REAL_T k,
                      Args... data) = 0;
    virtual void prepare_callbacks() { };
  };

  // EXPLICIT HYBRID
  template < typename REAL_T, std::size_t dim_x, std::size_t dim_u, std::size_t dim_p, std::size_t dim_y,
             template < typename REAL_T_, std::size_t dim_x_, std::size_t dim_u_, std::size_t dim_p_, typename... Args_  >
             typename EXPLICIT_INTEGRATOR,
             typename... Args >
  class ExplicitHybrid : public Hybrid< REAL_T, dim_x, dim_u, dim_p, dim_y, Args... >,
                         public EXPLICIT_INTEGRATOR< REAL_T, dim_x, dim_u, dim_p, REAL_T, Args... > {
   public:
    ExplicitHybrid(REAL_T _ts, REAL_T _time_limit, REAL_T _jumps_limit)
        : Hybrid< REAL_T, dim_x, dim_u, dim_p, dim_y, Args... >(_time_limit, _jumps_limit),
          EXPLICIT_INTEGRATOR< REAL_T, dim_x, dim_u, dim_p, REAL_T, Args... >(_ts) {
      this->prepare_callbacks();
    };

    void next(std::array< REAL_T, dim_x > &xp, REAL_T &t, const std::array< REAL_T, dim_x > &x,
              const std::array< REAL_T, dim_u > &u, const std::array< REAL_T, dim_p > &p, REAL_T k, Args... data) {
      EXPLICIT_INTEGRATOR< REAL_T, dim_x, dim_u, dim_p, REAL_T, Args... >::next(xp, t, x, u, p, k, data...);
    };

    void prepare_callbacks() {
      this->f = [this](std::array< REAL_T, dim_x > &dx, REAL_T t, const std::array< REAL_T, dim_x > &x,
                       const std::array< REAL_T, dim_u > &u, const std::array< REAL_T, dim_p > &p, REAL_T k,
                       Args... data) -> void { this->FlowMap(dx, t, k, x, u, p, data...); };
    }
  };

};     // namespace hybrid
#endif /* HYBRID_HPP_ */
