function [v, a, j] = estimate_s(y1, y2, y3, y4, fs)
%estimate system state:
v = (y1 - y2)*fs;
v2 = (y2 - y3)*fs;
v3 = (y3 - y4)*fs;

a = (v - v2)*fs;
a2 = (v3 - v2)*fs;

j = (a2 - a)* fs;
end

