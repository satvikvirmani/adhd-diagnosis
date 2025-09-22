function mra = empirical_wavelet_transformation(signal, channel_to_decomp, max_peaks)
    channel_signal = signal(:,channel_to_decomp);
    [mra, ~] = ewt(channel_signal, 'MaxNumPeaks', max_peaks);
end