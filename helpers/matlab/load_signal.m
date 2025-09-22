function signal = load_signal(file_path)
    data_struct = load(file_path);
    fieldNames = fieldnames(data_struct);
    signal = data_struct.(fieldNames{1});
    % channel_signal = signal(:, channel_idx);
end