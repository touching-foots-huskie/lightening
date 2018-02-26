% The ilc simulate using ilc method to initial the data
% it will return a pair of state, and compensate
% the first time:
global ilc_data
ini_c = 0;
sim('init_ilc_env');

% get out error
out_error = error.data; 
out_time = time.data;

%from then on:
% zero phase filter:
d = designfilt('lowpassfir', ...
    'PassbandFrequency',0.15,'StopbandFrequency',0.2, ...
    'PassbandRipple',1,'StopbandAttenuation',60, ...
    'DesignMethod','equiripple');

% out_error = filtfilt(d, out_error);
ilc_data = [out_time, out_error];
% ILC rotate for 5 times:
for i = 1:1:5
    sim('ilc_env');
    
    % refresh data
    new_out_error = error.data; 
    % new_out_error = filtfilt(d, new_out_error);
    
    out_error = out_error +new_out_error;
    out_time = time.data;
    ilc_data = [out_time, out_error];

end

% finally we want to get a pair of new_out_error:

