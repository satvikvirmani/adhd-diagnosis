signal = load("data/ADHD_part1/v1p.mat");

fieldNames = fieldnames(signal);
data = signal.(fieldNames{1});

CHANNEL_TO_DECOMP = 1;
channel_signal = data(:,CHANNEL_TO_DECOMP);

max_num_imf = 6;
[imf, residual] = emd(channel_signal, 'MaxNumIMF', max_num_imf);

num_imfs = size(imf,2);
figure;
for k = 1:max_num_imf
	subplot(4,2,k);
	plot(imf(:,k));
	title(['IMF ' num2str(k)]);
	ylabel('Amplitude');
	grid on;
end

subplot(4,2,[7, 8]);
plot(residual);
title('Residual');
ylabel('Amplitude');
xlabel('Samples');
grid on;