function imf = empirical_mode_decomposition(signal, channel_to_decomp, max_num_imf)
    channel_signal = signal(:,channel_to_decomp);
    [imf, ~] = emd(channel_signal, 'MaxNumIMF', max_num_imf);
end