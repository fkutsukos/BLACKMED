import datetime
import numpy as np
import librosa
import os


def compute_features_dataset(dataset, logger):
    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Working on ' + f'{dataset} files...')
    dataset_root = 'data/{}'.format(dataset)
    dataset_files = [f for f in os.listdir(dataset_root) if f.endswith(('.wav', '.mp3', '.aiff', '.m4a'))]
    n_files = len(dataset_files)

    dataset_features = np.zeros((n_files, 12))

    # dataset_features_low_level = np.zeros((n_files, n_features))
    # track_features_low_level = np.zeros((n_bins, n_frames))

    # Run over the dataset files and perform analysis

    for index, file in enumerate(dataset_files):
        logger.info(str(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Analysing ' + f'{file}. {index + 1} / {n_files}')

        # loading the audio file
        audio, sample_rate = librosa.load(os.path.join(dataset_root, file), sr=None)

        # cropping 10 seconds from the median sample of the audio file
        audio = audio[int(len(audio) / 2): int(len(audio) / 2) + 10 * sample_rate]

        # normalize amplitude
        audio = audio / audio.max()

        # Analysis variables:
        frame_length = int(np.floor(0.0213 * sample_rate))
        hop_length = int(np.floor(frame_length / 2))

        # Spectrogram of the audio
        specgram = librosa.stft(audio,
                                n_fft=frame_length,
                                hop_length=hop_length,
                                win_length=frame_length,
                                window='hamming',
                                center=False)
        mag_specgram = np.abs(specgram)
        pow_specgram = mag_specgram ** 2

        # Spectral rolloff
        spec_rolloff = librosa.feature.spectral_rolloff(S=mag_specgram, sr=sample_rate)

        # Perceptual loudness
        pow_specgram = np.where(pow_specgram == 0, np.finfo(np.float64).eps, pow_specgram)  # Numerical Stability
        freq = librosa.fft_frequencies(sr=sample_rate, n_fft=frame_length)
        freq = freq + 1.e-10  # Numerical Stability
        perceptual_pow_specgram = librosa.perceptual_weighting(pow_specgram, freq)
        perceptually_weighted_melspecgram = librosa.feature.melspectrogram(S=10 ** (perceptual_pow_specgram / 10),
                                                                           sr=sample_rate, n_fft=frame_length,
                                                                           hop_length=hop_length,
                                                                           win_length=frame_length, center=False)
        loudness = np.mean(perceptually_weighted_melspecgram, axis=0, keepdims=True)
        # loudness = 10 * np.log10(loudness) # the conversion to db will be done on track stage

        # Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(S=mag_specgram, sr=sample_rate)

        # Spectral Flux
        spectral_flux = librosa.onset.onset_strength(S=mag_specgram, sr=frame_length)
        spectral_flux = np.expand_dims(spectral_flux, axis=0)

        #  Means: the mean of the feature over the entire segment
        dataset_features[index, 0] = np.mean(spec_rolloff)
        dataset_features[index, 1] = np.mean(loudness)
        dataset_features[index, 2] = np.mean(bandwidth)
        dataset_features[index, 3] = np.mean(spectral_flux)
        # Variances: the variance of the feature over the entire segment;
        dataset_features[index, 4] = np.var(spec_rolloff)
        dataset_features[index, 5] = np.var(loudness)
        dataset_features[index, 6] = np.var(bandwidth)
        dataset_features[index, 7] = np.var(spectral_flux)
        # Main Peak prevalence: quantifies the prevalence of the main peak of the feature with respect to its mean value.
        dataset_features[index, 8] = np.max(spec_rolloff) / dataset_features[index, 0]
        dataset_features[index, 9] = np.max(loudness) / dataset_features[index, 1]
        dataset_features[index, 10] = np.max(bandwidth) / dataset_features[index, 2]
        dataset_features[index, 11] = np.max(spectral_flux) / dataset_features[index, 3]

    return dataset_features
