signal = load("data/ADHD_part1/v1p.mat");

sampling_frequency = 128;

fieldNames = fieldnames(signal);
data = signal.(fieldNames{1});

CHANNEL_TO_PLOT = 1;
N = length(data(:,CHANNEL_TO_PLOT));

time = 1:N;
channel_signal = data(:,CHANNEL_TO_PLOT);

num_imfs = 6;
[imfs, ~] = vmd(channel_signal, 'NumIMF', num_imfs);

figure;
for k = 1:num_imfs
	subplot(num_imfs,1,k);
	plot(time, imfs(:,k), 'LineWidth', 1.5);
	title(['IMF ' num2str(k)]);
	ylabel('Amplitude');
	grid on;
end
xlabel('Samples');

figure('Name','3D Plot of VMD IMFs');
[p,q] = ndgrid(time,1:size(imfs,2));
plot3(p,q,imfs);
xlabel('IMF Index'); ylabel('Samples'); zlabel('Amplitude');
title('3D Plot of VMD IMFs');
grid on;

figure('Name','Hilbert Spectrum of VMD IMFs');
hht(imfs, sampling_frequency);