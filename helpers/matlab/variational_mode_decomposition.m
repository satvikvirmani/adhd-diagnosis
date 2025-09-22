function imfs = variational_mode_decomposition(signal, channel_to_decomp, num_imfs)
    channel_signal = signal(:,channel_to_decomp);
    [imfs, ~] = vmd(channel_signal, 'NumIMF', num_imfs);
end