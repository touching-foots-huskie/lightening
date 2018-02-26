% data generating:
data_len = 10000;
f = 5000;
T = data_len/f;
data_num = 1000; % now we are using 100 sample to get a demenstration
scale = 0.2;
sin_scale = 1.0;
val_rate = 0.2;
% rgs
typ = {'rgs'};
for i = 1:1:size(typ,1)
    % train_data:
    yid = rand_one(data_num, data_len).*(idinput([data_len, data_num],...
    char(typ(i)),[0, 0.01],[-scale, scale]));

    y = yid;
    save(sprintf('train_%s', char(typ(i))), 'y');
    
    % val_data:
    yid = rand_one(floor(val_rate * data_num), data_len).*(idinput([data_len,...
        floor(val_rate * data_num)], char(typ(i)), [0, 0.01],[-scale, scale]));
 
    y = yid;
    save(sprintf('val_%s', char(typ(i))), 'y');
end
% sin
typ = {'sin'};
for i = 1:1:size(typ,1)
    w = rand([data_num,1]);
    t = 0:1/f:(T-1/f);
    y = sin(20*pi*w*t + 20*pi*rand(1))*scale*sin_scale;
    y = y';
    save(sprintf('train_%s', char(typ(i))), 'y');
    % val
    w = rand([ceil(val_rate * data_num),1]);
    y = sin(20*pi*w*t + 20*pi*rand(1))*scale*sin_scale;
    y = y';
    save(sprintf('val_%s', char(typ(i))), 'y');
end