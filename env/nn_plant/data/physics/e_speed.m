function [v, a] = e_speed(y, fs)
% Get estimated ddot(y), dot(y), and y. input y is (N, 1) output y is (N-2,
% 1), v is (N-2, 1), a is (N-2, 1)

%% estimate
vm = (y(2:end)-y(1:end-1))*fs;
v = vm(1:end-1);

%% filter
filter_times = 2;
h = fdesign.lowpass('Fp,Fst,Ap,Ast',0.0001, 0.05, 0.1, 60);
d = design(h, 'equiripple');

for i = 1:1:filter_times
    v = filtfilt(d.Numerator, 1, v);
    vm = filtfilt(d.Numerator, 1, vm);
end 

a = (vm(2:end)-vm(1:end-1))*fs;
for i = 1:1:filter_times
    a = filtfilt(d.Numerator, 1, a);
end 
end

