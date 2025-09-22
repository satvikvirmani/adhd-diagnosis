% Load two signals from .mat files or define them
% Example: assume signals are stored in 'signal1.mat' and 'signal2.mat'
% Each file should contain a variable 'x1' and 'x2' respectively
load("data/ADHD_part1/v15p.mat");   % should contain x1
load("data/ADHD_part1/v1p.mat");   % should contain x2

% Ensure both signals are the same lengthß
sig1 = v15p(:,1);
sig2 = v1p(:,1);

% Match lengths (truncate to min length)
N = min(length(sig1), length(sig2));
sig1 = sig1(1:N);
sig2 = sig2(1:N);

t = 1:N;

% Plot signals
figure;
subplot(4,1,1);
plot(t, sig1, 'b', 'LineWidth', 1.5);
title('Signal 1');
xlabel('Samples'); ylabel('Amplitude'); grid on;

subplot(4,1,2);
plot(t, sig2, 'r', 'LineWidth', 1.5);
title('Signal 2');
xlabel('Samples'); ylabel('Amplitude'); grid on;

% Compute inner product
inner_prod = dot(sig1, sig2);

% Plot inner product as a bar
subplot(4,1,3);
bar(inner_prod);
title(['Inner Product = ', num2str(inner_prod)]);
xlabel(' '); ylabel('Value'); grid on;

% Plot pointwise product
subplot(4,1,4);
plot(t, sig1 .* sig2, 'k', 'LineWidth', 1);
title(['Pointwise Product (Inner Product = ', num2str(inner_product), ')']);
xlabel('Samples'); ylabel('Value'); grid on;