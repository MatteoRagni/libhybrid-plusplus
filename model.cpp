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

#define _USE_MATH_DEFINES
//#define SLIDING_MASSES
#define BOUNCING_BALL
#define INTEGRATOR_IMPLICIT

#ifdef BOUNCING_BALL
#include "model_bouncing_ball.hpp"
typedef BouncingBall Model;
#endif

#ifdef SLIDING_MASSES
#include "model_sliding_masses.hpp"
typedef SlidingMasses Model;
#endif


void printer(double t, double k, const Input &u, const State &x) {
  std::cout << t << "," << k << ",";
  for (const auto &xi : x)
    std::cout << xi << ",";
  for (const auto &ui : u)
    std::cout << ui << ",";
  std::cout << "0" << std::endl;
}

int main() {
  double t = 0, k = 0;
  
#ifdef BOUNCING_BALL
  State x = { 9.0, 1.0 };
  Params p = { 9.8, 0.75 };
  double t_limit = 10.0;
  double j_limit = 25.0;
#endif
  
#ifdef SLIDING_MASSES
  State x = { 1.0, 3.0, 0.9, 0.0 };
  Params p = { 1.0, 0.2, 1.0, 0.1, 0.02, 0.3, 1.0 };
  double t_limit = 100.0;
  double j_limit = 200.0;
#endif
  
  State xp = x;
  Input u = { 0.0 };

  Model m(0.5e-4, t_limit + 1, j_limit + 1, 1e-12, 5000);

  while (t <= t_limit && k < j_limit) {
    u[0] = 0.5 * std::sin((M_PI/5.0) * t + (M_PI/3.0));
    m.step(xp, t, k, x, u, p);
    printer(t, k, u, x);
    hybrid::math::copy(x, xp);
  }
  
  return 0;
}
