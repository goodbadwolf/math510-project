%Extract the audio samples from the file
function analyzeAudio(frequency_index, tau1_value)
    [audio, sample_rate] = audioread('../timit/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV');
    [num_samples, num_channels] = size(audio);

    %Window size is 50 ms or 800 samples at a sampling rate of 16000

    window_size_ms = 50.0;
    window_size_samples = int64(floor((window_size_ms / 1000) * sample_rate));

    %Perform the windowed FFT, and gather the S(F,T), F, and T arrays
    %seperately
    %Instead of default overlap, window overlap should be 1 ms ==> ~2900
    %results for this example
    %This corresponds to having (800 - (800/50)) samples of overlap between
    %adjoining segments = 784 samples of overlap
    [S, F, T] = spectrogram(audio, window_size_samples, 784, [], sample_rate);
    
    [num_sft_freqs, num_sft_samples] = size(S);

    %Calculate the power spectral density of the sample. This is a real
    %(double) value
    S = 20*log10(abs(S))
    %Set up the convolution filter constants - tau_1, tau_2 and A, according to
    %Eq. 2.47 of the book
    %tau_2 should be greater that tau_1
    %Experiment with tau_1 = 5, 10 ms
    tau_1 = tau1_value;
    tau_2 = tau_1 * 5;
    alpha = 1 / tau_1;
    beta = 1 / tau_2;
    tau = 4*tau_2;
    
    %Create an array of tau elements for the convolution filter
    K_x = linspace(1, tau, tau);

    %Set up convolution filter for both ON and OFF cells
    for i = 1:tau
        K_1 = alpha*alpha * K_x(i) * exp(-alpha * K_x(i));
        K_2 = beta*beta * K_x(i) * exp(-beta * K_x(i));
        K_on(i) = (K_1 - K_2);
        K_off(i) = -K_on(i);
    end
    
    %Perform the convolution for a selected frequency (denoted by freq_row in
    %the F array gathered above
    %Frequency should be 200,400,800,1600 corresponding to 14th, 27th, 52nd,
    %104th rows in the 'F' array that we store above
    freq_row_array = [14, 27, 52, 104]
    a = zeros(1, num_sft_samples)
    
    %Perform the convolution using the default MATLAB built-in
    y_t_ON = conv(K_on, S(freq_row_array(frequency_index), :))
    y_t_OFF = conv(K_off, S(freq_row_array(frequency_index), :))
    %Now that we have the convolved output, apply non-linearities on it
    %For now, apply only RELU
    non_linear_ON = max(0, y_t_ON)
    non_linear_OFF = max(0, y_t_OFF)
    [autocorr_ON, LAG_ON] = xcorr(non_linear_ON,'coeff')
    [autocorr_OFF, LAG_OFF] = xcorr(non_linear_OFF,'coeff')
    [cross_corr_ON_OFF, LAG_ON_OFF] = xcorr(non_linear_ON, non_linear_OFF,'coeff')
    
    %Plot the graphs
    figure(1)
    plot(LAG_ON,autocorr_ON)
    title('Autocorrelation in output after applying ON filter and RELU non-linearity')
    xlabel('Lag in number of samples (1 sample = +-0.0625 ms)')
    ylabel('Autocorrelation normalized to 1')
    figure(2)
    plot(LAG_OFF, autocorr_OFF)
    title('Autocorrelation in output after applying OFF filter and RELU non-linearity')
    xlabel('Lag in number of samples (1 sample = +-0.0625 ms)')
    ylabel('Autocorrelation normalized to 1')
    figure(3)
    plot(LAG_ON_OFF, cross_corr_ON_OFF)
    title('Cross-correlation between outputs after applying ON and OFF filters')
    xlabel('Lag in number of samples (1 sample = +-0.0625 ms)')
    ylabel('Cross-correlation normalized to 1')
    
    %Make a historgram of the output of the non_linearity
    figure(4)
    histogram(non_linear_ON)
    title('Histogram of ON-filter output after applying RELU non-linearity')
    xlabel('Output of ON-filter applied to power spectral density')
    ylabel('Number of samples')
    figure(5)
    histogram(non_linear_OFF)
    title('Histogram of OFF-filter output after applying RELU non-linearity')
    xlabel('Output of OFF-filter applied to power spectral density')
    ylabel('Number of samples')
end
