function [State] = seq_wrap(x, y)
%% calculate first a
State = zeros(size(x,1), 3);
for i = 1:1:size(x, 1)
    state = [y(i:i+1)', x(i)];
    State(i, :) = state;
end
end

