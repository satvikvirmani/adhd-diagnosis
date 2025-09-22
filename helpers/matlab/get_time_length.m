function T = get_time_length(mat_filename)
    data_struct = load(mat_filename);
    fieldNames = fieldnames(data_struct);
    signal = data_struct.(fieldNames{1});
    
    N = length(signal);
    T = N / 128;
end