%expand the output data:
filename = {'out_train_rgs.mat',  'out_val_rgs.mat', 'out_val_sin.mat'};
for  i=1:1:max(size(filename))
load(char(filename(i)));
Out = 100*Out;
save(char(filename(i)), 'Out');
end