% divide the data into train and validation:
load('train_rgs.mat');
val_y = y(:, 101:120);
val_y = repmat(val_y, 1, 2);

% save:
y = y(:, 1:100);
save('train_rgs.mat', 'y');
y = val_y;
save('val_rgs.mat', 'y');
 
% outpart:
load('out_train_rgs.mat');
val_out = Out(:, 101:120);
val_out = repmat(val_out, 1, 2);

% save:
Out = Out(:, 1:100);
save('out_train_rgs.mat', 'Out');
Out = val_out;
save('out_val_rgs.mat', 'Out');
