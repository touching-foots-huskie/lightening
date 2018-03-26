function [x, v, ap] = a_predict(init_state, theta, fs)
y2 = init_state(1);
y1 = init_state(2);
x = init_state(3);
% x is the signal of this step, y1 is 1 step before, y2 is 2 steps before
v = ((y1 - y2)*fs)*3;
X = [x, v, -1];
ap = X*theta;  % scale
end

