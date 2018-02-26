% this function is used to show the scale of x and y status:
data_num = 30;
scale = 0.1;
typ = {'rgs'; 'sin'};
for i = 1:1:size(typ,1)
    y = idinput([data_len, data_num], char(typ(i)))*scale;
    save(sprintf('%d_%s', data_len, char(typ(i))), 'y');
end