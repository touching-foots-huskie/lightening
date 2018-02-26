% init for simulink structure:
clear
global x;
typ = {'rgs', 'sin'};
% typ = {'sin'};
for i =1:1:size(typ,2)
    train_file_name = sprintf('train_%s.mat', char(typ(i)));
    val_file_name = sprintf('val_%s.mat', char(typ(i)));
    % train part:
    load(train_file_name);
    motor_data_gen(y, train_file_name);
    % val part
    load(val_file_name);
    motor_data_gen(y, val_file_name);
end




