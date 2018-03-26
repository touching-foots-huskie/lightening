function y = step_sim(init_state, fs)
%  estimate the output in one step
y2 = init_state(1);
y1 = init_state(2);
a1p = init_state(3);
v1b = (y1 - y2)*fs;

y = y1 + v1b/fs + a1p/fs^2;
end