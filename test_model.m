clear all
close all
clear mex % This is important to reset the mex file

mex CXXFLAGS="\$CXXFLAGS -I. -I./include -std=c++17" -lmwlapack -lmwblas model_mex.cpp

%% Test for bouncing ball

if 0

  x0 = [ 9; 1 ];
  u0 = [ 0 ];
  p0 = [ 9.8, 0.75 ];

  t_max = 30.0;
  j_max = 25;
  t = 0;

  x = [x0];
  y = [x0];
  q = [0; 0];


  q_ = q;
  reset = 1;
  while (q_(1) < t_max && q_(2) < j_max)
    [y_, x_, q_] = model_mex(q_(1), x(:,end), u0, p0, reset);
    reset = 0;
    x = [x, x_];
    y = [y, y_];
    q = [q, q_];
  end

  figure
  subplot(2,1,1);
  plot(q(1,:), x(1,:));
  subplot(2,1,2);
  plot(q(1,:), x(2,:));

end

%% Test for sliding masses

if 1
  x0 = [ 1; 3; 0.9; 0 ];
  u0 = [ 0 ];
  p0 = [ 1.0, 0.2, 1.0, 0.1, 0.02, 0.3, 1];

  t_max = 30.0;
  j_max = 200;
  t = 0;
  ts = 1e-4;
  
  x = zeros(size(x0, 1), ceil(t_max/ts));
  x(:,1) = x0;
  % x = [x0];
  % y = [x0];
  q = zeros(2, ceil(t_max/ts));

  x_ = x0;
  q_ = q;
  reset = 1;
  counter = 1;
  while (q_(1) < t_max && q_(2) < j_max)
    counter = counter + 1;
    
    [~, x_, q_] = model_mex(q_(1), x_, u0, p0, reset);
    reset = 0;
    x(:, counter) = x_;
    % y = [y, y_];
    q(:, counter) = q_;
    
    if mod(counter, 1000) == 0
      fprintf(1, 'Working: t = % 2.4f, j = % 4f\n', q_(1), q_(2));
    end
  end

  figure
  subplot(4,1,1);
  plot(q(1,:), x(1,:));
  subplot(4,1,2);
  plot(q(1,:), x(2,:));
  subplot(4,1,3);
  plot(q(1,:), x(3,:));
  subplot(4,1,4);
  plot(q(1,:), x(4,:));
end