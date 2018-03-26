% code test: test the whole process:
clc
clear
%% 1.les test
filename = 'rawdata/1_2.mat';
theta = lse(filename);
save('theta.mat', 'theta');

%% 2. a deviate:
[x, y, v, a, ad, fs] = a_deviate(filename, theta);

%% 3. step sim exam:
init_state = [y(1:2)', a(1)*3];
a_init_state = [y(1:2)', x(1)];
Yp = zeros(size(a, 1)-1, 1);
Ap = zeros(size(a, 1)-1, 1);
Xe = zeros(size(a, 1)-1, 1);
Ve = zeros(size(a, 1)-1, 1);
Se = zeros(size(a, 1)-1, 1);
for i = 2:1:size(a, 1)
    yp = step_sim(init_state, fs);
    [xe, ve, ap] = a_predict(a_init_state, theta, fs);
    init_state = [init_state(2), yp, a(i)*3];
    a_init_state = [a_init_state(2), y(i+1), x(i)];
    Yp(i-1) = yp;
    Ap(i-1) = ap;
    Xe(i-1) = xe;
    Ve(i-1) = ve;
end

plot(Yp);
hold on
plot(y, 'r');

%% 4.seq sim:
seq_sim(x, y, theta, fs);