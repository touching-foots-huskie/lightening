% this function is to add sin data into it.
data_len = 10000;
data_num = 1000;
scale = 0.5;
typ = {'sin'};
for i = 1:1:size(typ,1)
    w = rand([1000,1]);
    t = 0:1/5000:(2-1/5000);
    y = sin(20*pi*w*t)*scale;
    y = y';
    save(sprintf('%d_%s', data_len, char(typ(i))), 'y');
end