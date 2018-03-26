% This function is used to estimate the parameter over a.
function theta = lse(filename)
%% read data:
load(filename);
y = rec.Y(1).Data';
x = rec.Y(2).Data';
ad = find(x);
x = x(min(ad):max(ad));
y = y(min(ad):max(ad));
t = rec.X.Data';
fs = round(1/(1000*(t(end) - t(end-1))))*1000;

% convert and normalize:
[v, a] = e_speed(y, fs);
seqlen = size(v, 1);
save('seqlen.mat', 'seqlen');
a = a/3;
v = v*3;
x = x(2:end-1);

%% lse estimate:
A = [x,v,-ones(size(a))];
T = A'*A;
theta = (T)\A'*a;
end