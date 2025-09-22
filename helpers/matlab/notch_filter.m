function filtered_signal = notch_filter(signal, sampling_frequency, notch_frequency, quality_factor)
    % Ensure all inputs are double
    sampling_frequency = double(sampling_frequency);
    notch_frequency    = double(notch_frequency);
    quality_factor     = double(quality_factor);

    % Normalized frequency
    Wo = notch_frequency / (sampling_frequency/2);
    BW = Wo / quality_factor;

    [b, a] = iirnotch(Wo, BW);
    filtered_signal = filtfilt(b, a, signal);
end