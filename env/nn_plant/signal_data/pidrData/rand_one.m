function out = rand_one(data_num, data_len)
    out = floor(rand(1, data_num)*2)*2 -1;
    out = repmat(out, data_len, 1);
end