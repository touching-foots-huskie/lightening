function [x, y, v, a, ad, fs] = a_deviate(filename, theta)
%% read data:
load(filename);
y = rec.Y(1).Data';
x = rec.Y(2).Data';
t = rec.X.Data';
fs = round(1/(1000*(t(end) - t(end-1))))*1000;

% normalize
ad = find(x);
x = x(min(ad):max(ad));
y = y(min(ad):max(ad));

[v, a] = e_speed(y, fs);
a = a/3;
v = v*3;
x = x(2:end-1);

%% estimate£º
X = [x, v, -ones(size(a))];
ap = X*theta;
ad = a - ap;

corrcoef(a, ap)
end