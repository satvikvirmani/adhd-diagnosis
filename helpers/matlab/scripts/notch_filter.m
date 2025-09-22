signal = load("data/ADHD_part1/v1p.mat");

sampling_frequency = 128; % Set your sampling frequency here (Hz)

fieldNames = fieldnames(signal);
data = signal.(fieldNames{1});

CHANNEL_TO_PLOT = 1;
N = length(data(:,CHANNEL_TO_PLOT));
time = 1:N;

time_span_sec = N / sampling_frequency;
fprintf('Time span of the signal: %.2f seconds\n', time_span_sec);

channel_signal = data(:,CHANNEL_TO_PLOT);

figure;

subplot(2,3,1);
plot(time, data, 'LineWidth', 1.0);
title('Signal with 19 Channels');
xlabel('Samples'); ylabel('Amplitude'); grid on;

subplot(2,3,4);
plot(time, channel_signal, 'b', 'LineWidth', 1.0);
title('Signal for Channel 1');
xlabel('Samples'); ylabel('Amplitude'); grid on;

%% Apply 50 Hz Notch Filter to all channels
notch_frequency = 50;  % Notch frequency (Hz)
quality_factor = 30;   % Quality factor
[b, a] = iirnotch(notch_frequency/(sampling_frequency/2), notch_frequency/(sampling_frequency/2)/quality_factor);

filtered_data = filtfilt(b, a, data);

subplot(2,3,2);
plot(time, filtered_data, 'LineWidth', 1.0);
title('Filtered Signal with 19 Channels (50 Hz Notch)');
xlabel('Samples'); ylabel('Amplitude'); grid on;

subplot(2,3,5);
plot(time, filtered_data(:,CHANNEL_TO_PLOT), 'r', 'LineWidth', 1.0);
title('Filtered Signal for Channel 1 (50 Hz Notch)');
xlabel('Samples'); ylabel('Amplitude'); grid on;

%% Apply 0.1Hz to 60 Hz Butterworth Bandpass Filter to all channels
low_cutoff = 0.1;   % Low cutoff frequency in Hz
high_cutoff = 60; % High cutoff frequency in Hz
order = 6;        % Filter order
[b_band, a_band] = butter(order, [low_cutoff high_cutoff]/(sampling_frequency/2), 'bandpass');

bandpassed_data = filtfilt(b_band, a_band, filtered_data);

subplot(2,3,3);
plot(time, bandpassed_data, 'g', 'LineWidth', 1.0);
title('Bandpass Filtered Signal with 19 Channels (0.1-60 Hz)');
xlabel('Samples'); ylabel('Amplitude'); grid on;

subplot(2,3,6);
plot(time, bandpassed_data(:,CHANNEL_TO_PLOT), 'g', 'LineWidth', 1.0);
title('Bandpass Filtered Signal for Channel 1 (0.1-60 Hz)');
xlabel('Samples'); ylabel('Amplitude'); grid on;