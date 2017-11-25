[audio, sample_rate] = audioread('../timit/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV');
[num_samples, num_channels] = size(audio);
window_size_ms = 50.0;
window_size_samples = floor((window_size_ms / 1000) * sample_rate);
[S, F, T] = spectrogram(audio, window_size_samples, [], [], sample_rate);

[num_sft_freqs, num_sft_samples] = size(S);

tau = 10;
tau_1 = 50;
tau_2 = tau_1 / 5;
alpha = 1 / tau_1;
beta = 1 / tau_2;

K_x = linspace(-tau, tau, 2 * tau + 1);
for i = 1:2 * tau + 1
    K_1 = alpha * alpha * K_x(i) * exp(-alpha * K_x(i));
    K_2 = beta * beta * K_x(i) * exp(-beta * K_x(i));
    K_on(i) = K_1 - K_2;
    K_off(i) = -K_on(i);
end
