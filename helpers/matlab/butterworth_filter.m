function filtered_signal = butterworth_filter(signal, sampling_frequency, low_cutoff, high_cutoff, order)
    % Cast inputs to double
    sampling_frequency = double(sampling_frequency);
    low_cutoff         = double(low_cutoff);
    high_cutoff        = double(high_cutoff);
    order              = double(order);

    % Normalize cutoff frequencies
    Wn = [low_cutoff high_cutoff] / (sampling_frequency/2);

    % Validate cutoff range
    if any(Wn <= 0) || any(Wn >= 1)
        error('Normalized cutoff frequencies must be between 0 and 1 (exclusive). Got: [%f, %f]', Wn(1), Wn(2));
    end

    % Design bandpass filter
    [b_band, a_band] = butter(order, Wn, 'bandpass');
    filtered_signal = filtfilt(b_band, a_band, signal);
end