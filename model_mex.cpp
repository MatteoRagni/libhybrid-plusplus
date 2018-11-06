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
#include <algorithm>
#include <cstddef>
#include "mex.hpp"
#include "mexAdapter.hpp"

#define SLIDING_MASSES
//#define BOUNCING_BALL
//#define INTEGRATOR_IMPLICIT

#ifdef BOUNCING_BALL
#include "model_bouncing_ball.hpp"
typedef BouncingBall Model;
#endif

#ifdef SLIDING_MASSES
#include "model_sliding_masses.hpp"
typedef SlidingMasses Model;
#endif

#ifndef JUMP_LIMIT
#define JUMP_LIMIT 200.0
#endif

#ifndef TIME_LIMIT
#define TIME_LIMIT 100.0
#endif

#ifndef SAMPLE_TIME
#define SAMPLE_TIME 1e-4
#endif

#ifndef IMPLICIT_TOLERANCE
#define IMPLICIT_TOLERANCE 1e-10
#endif

#ifndef IMPLICIT_ITERATIONS
#define IMPLICIT_ITERATIONS 5000
#endif

class MexFunction : public matlab::mex::Function {
 private:
  matlab::data::ArrayDimensions input_accepted = {1, dim_x, dim_u, dim_p, 1};
  matlab::data::ArrayDimensions output_one_dims = {dim_y};
  matlab::data::ArrayDimensions output_two_dims = {dim_x};
  matlab::data::ArrayDimensions output_three_dims = {2};


  const double ts = SAMPLE_TIME;
  const double t_limit = TIME_LIMIT;
  const double j_limit = JUMP_LIMIT;
  const double tol = IMPLICIT_TOLERANCE;
  const int iters = IMPLICIT_ITERATIONS;

  Model m;
  double t;
  double j;

  matlab::data::ArrayFactory factory;
  std::shared_ptr< matlab::engine::MATLABEngine > engine;


 public:
  MexFunction() : m(Model(ts, t_limit, j_limit, tol, iters)), j(0), t(0) { 
    engine = getEngine();
  };

  ~MexFunction() { };

  void operator()(matlab::mex::ArgumentList output, matlab::mex::ArgumentList input) {
    checkArguments(output, input);
    
    double t_ = input[0][0];
    
    State x; 
    matlab::data::TypedArray<double> x_in = std::move(input[1]);
    std::copy(x_in.begin(), x_in.end(), x.begin());
    
    Input u;
    matlab::data::TypedArray<double> u_in = std::move(input[2]);
    std::copy(u_in.begin(), u_in.end(), u.begin());
    
    Params p;
    matlab::data::TypedArray<double> p_in = std::move(input[3]);
    std::copy(p_in.begin(), p_in.end(), p.begin());

    double k = input[4][0];
    if (k == 1) {
      // k = 1 resets the output
      t = 0; 
      j = 0;
    }
    
    State xp;
    Output y;

    m.OutputMap(y, t, j, x, u, p);
    m.step(xp, t, j, x, u, p);

    output[0] = factory.createArray(output_one_dims, y.begin(), y.end());

    if (output.size() > 1) {
      output[1] = factory.createArray(output_two_dims, xp.begin(), xp.end());
    }
    if (output.size() > 2) {
      output[2] = factory.createArray(output_three_dims, {t, j});
    }
  };

 private:
  template < typename... Ts >
  void raiseMatlabException(const char fmt[], Ts... values) {
    auto size = std::snprintf(nullptr, 0, fmt, values...);
    std::string output(size + 1, '\0');
    std::sprintf(&output[0], fmt, values...);
    engine->feval(matlab::engine::convertUTF8StringToUTF16String("error"), 0,
                  std::vector< matlab::data::Array >({factory.createCharArray(output)}));
  }

  void checkArguments(matlab::mex::ArgumentList output, matlab::mex::ArgumentList input) {
    // Checking input
    if (input.size() != input_accepted.size()) {
      raiseMatlabException("Number of input vector is %d (Must be %d)", input.size(), input_accepted.size());
    }
    if (output.size() > 3) {
      raiseMatlabException("Number of output vector is %d (Must be 1, 2, or 3)", input.size());
    }
    for (std::size_t i = 0; i < input_accepted.size(); i++) {
      if (input[i].getType() != matlab::data::ArrayType::DOUBLE ||
          input[i].getType() == matlab::data::ArrayType::COMPLEX_DOUBLE ||
          input[i].getNumberOfElements() != input_accepted[i]) {
        raiseMatlabException("Input vector %d is invalid", i);
      }
    }
  }
};
