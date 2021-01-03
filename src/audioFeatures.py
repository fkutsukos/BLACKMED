import datetime
import numpy as np
import librosa
import os
from sklearn.preprocessing import MinMaxScaler


def compute_dataset_features(dataset, features, logger):
    dataset_root = 'data/{}'.format(dataset)
    dataset_files = [f for f in os.listdir(dataset_root) if f.endswith(('.wav', '.mp3', '.aiff', '.m4a'))]
    n_files = len(dataset_files)
    track_names = []
    has_beat = []
    tempo = []

    logger.info(str(
        datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")) + ' Extracting ' + f'{len(features)} features')

    dataset_features = np.zeros((n_files, len(features)))

    # Run over the dataset files and perform analysis
    logger.info(str(
        datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")) + ' Working on ' + f'{dataset} dataset of ' + f'{n_files} files')
    for index, file in enumerate(dataset_files):

        track_names.append(file)
        # loading the audio file
        # when predicting, we are cropping 10 seconds from the median sample of the audio file
        if dataset == 'Predict':
            logger.info(str(
                datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")) + ' Predicting - Analysing ' + f'{file}. {index + 1} / {n_files}')
            audio, sample_rate = librosa.load(os.path.join(dataset_root, file), sr=None)
            # audio = audio[3 * int(len(audio) / 5): 3 * int(len(audio) / 5) + 10 * sample_rate]
            audio_part1 = audio[2 * int(len(audio) / 5): 2 * int(len(audio) / 5) + 3 * sample_rate]
            audio_part2 = audio[3 * int(len(audio) / 5): 3 * int(len(audio) / 5) + 4 * sample_rate]
            audio_part3 = audio[4 * int(len(audio) / 5): 4 * int(len(audio) / 5) + 3 * sample_rate]
            audio = np.concatenate((audio_part1, audio_part2, audio_part3), axis=0)
        else:
            logger.info(str(
                datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")) + ' Training - Analysing ' + f'{file}. {index + 1} / {n_files}')
            audio, sample_rate = librosa.load(os.path.join(dataset_root, file), sr=None)

        # Amplitude Normalization

        audio = np.expand_dims(audio, axis=1)
        min_max_scaler = MinMaxScaler()
        audio = min_max_scaler.fit_transform(audio)
        audio = audio.flatten()

        # audio = audio/audio.max()

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

        mel_specgram = librosa.feature.melspectrogram(sr=sample_rate,
                                                      S=pow_specgram,
                                                      n_fft=frame_length,
                                                      hop_length=hop_length,
                                                      n_mels=40,
                                                      fmin=0,
                                                      fmax=sample_rate / 2,
                                                      htk=True,
                                                      norm=None)
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

        # MFCCs
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_specgram), sr=sample_rate, n_mfcc=13)

        # Entropy of Energy
        # This information is not used during training, only for prediction
        if dataset == 'Predict':
            signal_length = len(audio)
            num_frames = int(np.ceil((signal_length - frame_length) / hop_length) + 1)
            pad_signal_length = (num_frames - 1) * hop_length + frame_length

            # We need to add to the signal a #zeros that correspond to (signal_length - pad_signal_length)
            z = np.zeros(pad_signal_length - signal_length)

            pad_signal = np.append(audio, z)

            # indexing matrix to understand which samples comprise each frame
            inframe_ind = np.tile(np.arange(0, frame_length), (num_frames, 1)).T

            # this matrix has the same shape as inframe_ind but shows how many samples we have jumped
            frame_ind = np.tile(np.arange(0, num_frames * hop_length, hop_length), (frame_length, 1))

            # If we add these two matrices, we get the index of the samples we are analyzing at each frame
            indices = inframe_ind + frame_ind

            frames = pad_signal[indices]
            # t_frames = np.arange(0, num_frames) * (hop_length / sample_rate)  # starting time instant of each frame
            # t_frames_end = t_frames + (frame_length / sample_rate)  # #ending time instant of each frame
            # t_frames_ctr = 0.5 * (t_frames + t_frames_end)  # central instant of each frame

            frame_entropy = []
            num_frames = int(frames.shape[1])
            for frame in range(num_frames):
                subframe_length = int(np.ceil((frames.shape[0]) / 10))
                frame_energy = np.mean(frames[:, frame])
                subframe_e = []
                for subframe in range(10):
                    subframe_energy = np.mean(
                        frames[subframe * subframe_length:(subframe + 1) * subframe_length, frame])
                    if subframe_energy > 0:
                        subframe_e.append(subframe_energy / frame_energy)

                frame_entropy.append(-(np.dot(np.log2(subframe_e), subframe_e)))

            has_beat.append(np.mean(frame_entropy))

            # Tempo
            tempo.append(librosa.beat.tempo(audio, sr=sample_rate, start_bpm=110.0, std_bpm=15, max_tempo=190))

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
        # MFCC mean
        dataset_features[index, 12:25] = np.mean(mfcc, axis=1)

    # dataset_features = pd.DataFrame(data=dataset_features, columns=features)
    return dataset_features, track_names, has_beat, tempo


def check_beat(row):
    if np.abs(row['Entropy']) > 0.2 or row['Dynamicity'] > 0.5:
        return True
    return False
