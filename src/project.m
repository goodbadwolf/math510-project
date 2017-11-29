%Extract the audio samples from the file
[audio, sample_rate] = audioread('../timit/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV');
[num_samples, num_channels] = size(audio);

%Window size is 50 ms or 800 samples at a sampling rate of 16000
window_size_ms = 50.0;
window_size_samples = int64(floor((window_size_ms / 1000) * sample_rate));

%Perform the windowed FFT, and gather the S(F,T), F, and T arrays
%seperately
[S, F, T] = spectrogram(audio, window_size_samples, [], [], sample_rate);

[num_sft_freqs, num_sft_samples] = size(S);

%Set up the convolution filter constants - tau_1, tau_2 and A, according to
%Eq. 2.47 of the book
tau = 10;
tau_1 = 50;
tau_2 = tau_1 / 5;
alpha = 1 / tau_1;
beta = 1 / tau_2;

%Create an array of 2*tau + 1 elements for the convolution filter
K_x = linspace(-tau, tau, 2 * tau + 1);

%Set up convolution filter for both ON and OFF cells
for i = 1:2 * tau + 1
    K_1 = alpha * alpha * K_x(i) * exp(-alpha * K_x(i));
    K_2 = beta * beta * K_x(i) * exp(-beta * K_x(i));
    K_on(i) = K_1 - K_2;
    K_off(i) = -K_on(i);
end


%Perform the convolution for a selected frequency (denoted by freq_row in
%the F array gathered above
freq_row = 100
a = zeros(1, num_sft_samples)
y_t_ON = complex(a,0)
y_t_OFF = complex(a,0)

for i = 1:num_sft_samples
    for j = -floor(size(K_on)/2):floor(size(K_on)/2)
        % Check bounds
        if ((i-j) >= 1) && ((i-j) <= num_sft_samples)
            on_filter = K_on(1,j+floor(size(K_on)/2) + 1)
            off_filter = K_off(1,j+floor(size(K_off)/2) + 1)
            y_t_ON(i) = on_filter(1)*S(freq_row,i-j)
            y_t_OFF(i) = off_filter(1)*S(freq_row,i-j)
        end
    end
end
