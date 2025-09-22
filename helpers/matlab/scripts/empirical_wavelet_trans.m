signal = load("data/ADHD_part1/v1p.mat");
fieldNames = fieldnames(signal);
data = signal.(fieldNames{1});

sampling_frequency = 128;

CHANNEL_TO_USE = 1;
signal = data(:,CHANNEL_TO_USE);

max_peaks = 4;
[mra, cfs] = ewt(signal, 'MaxNumPeaks', max_peaks);

subplot(max_peaks+1,1,1)
t = (0:length(signal)-1)/sampling_frequency;
plot(t,signal)
title('MRA of Signal')
ylabel('Signal')
axis tight
for k=1:max_peaks
    subplot(max_peaks+1,1,k+1)
    plot(t,mra(:,k))
    ylabel(['MRA ',num2str(k)])
    axis tight
end
xlabel('Time (s)')

cfsenergy = sum(sum(abs(cfs).^2));
if cfsenergy == norm(signal,2)^2
    disp("Signal can be perfectly reconstructed from MRA components")
end

hht(mra, sampling_frequency);