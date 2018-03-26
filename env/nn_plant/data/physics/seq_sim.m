function [Yp, Ap] = seq_sim(x, y, theta, fs)
%% calculate first a
init_state = [y(1:2)', x(1)];
ap = a_predict(init_state, theta, fs);
a_state = [init_state(1:2), ap];
Yp = zeros(size(x,1)-1, 1);
Ap = zeros(size(x,1)-1, 1);
for i = 2:1:size(x, 1)
    yp = step_sim(a_state, fs);
    init_state = [init_state(2), yp, x(i)];
    ap = a_predict(init_state, theta, fs);
    a_state = [init_state(1:2), ap];
    Yp(i-1) = yp;
    Ap(i-1) = ap;
end
plot(Yp);
hold on
plot(y(3:end), 'r');
end

