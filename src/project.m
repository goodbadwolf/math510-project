[audio, sample_rate] = audioread('../timit/lisa/data/timit/raw/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV');
[num_samples, num_channels] = size(audio);
window_size_ms = 50.0;
window_size_samples = floor((window_size_ms / 1000) * sample_rate);
[S, W, T] = spectrogram(audio, window_size_samples, [], [], sample_rate);
