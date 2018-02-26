% wrapper save:
function out = motor_data_gen(y, filename)
global x
Out = [];
data_range = size(y,1);
t = (0:1:(data_range-1))/5000;
data_num = size(y,2);
h = waitbar(0,'Please wait');
for i = 1:1:data_num
    x = y(:,i);
    x = [t',x];
    % start sim:
    sim('plant');
    % sim('baseline');
    out = out(1:data_range) * 100;
    Out = [Out, out];
    per = i/data_num;
    waitbar(per, h, sprintf('%2.0f%%', per*100))
end
save_file_name = sprintf('out_%s', filename);
save(save_file_name, 'Out');
end