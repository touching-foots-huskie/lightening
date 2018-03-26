% run: generating training sets:
clc
clear
%% 1. read data:
filenames = {'rawdata/1_2.mat'};
load('theta.mat');
load('seqlen.mat');
load(char(filenames(1)));
filenum = size(filenames, 1);
State = zeros(filenum*seqlen, 3);
A = zeros(filenum*seqlen, 1);

%% process data
for i = 1:1:size(filenames, 2)
    filename = char(filenames(i));
    [x, y, v, a, ad, fs] = a_deviate(filename, theta);
    % save them in batch£º
    state = seq_wrap(x, y);
    State((i-1)*seqlen+1:i*seqlen, :) = state(1:seqlen, :);
    A((i-1)*seqlen+1:i*seqlen, :) = a(1:seqlen, :);
end

%% save
save('train/state.mat', 'State');
save('train/a.mat', 'A');