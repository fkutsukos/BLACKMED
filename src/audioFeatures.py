import datetime
import numpy as np
import librosa
import os
from sklearn.preprocessing import MinMaxScaler
import pyloudnorm
import eyed3


def compute_dataset_features(logger, dataset, features, predict=False):
    """
    This function returns the LLF (low level features) from the audio samples of a tracks
    :param logger: the logger instance used for writing logs to an external log file
    :param dataset: the name of the dataset with the tracks to be analysed
    :param features: the list of low level features names to be computed
    :param predict: this parameter
    :return: multiple:
    dataset_features: np.ndarray: An array with average LLF for the tracks,
    track_ids: List: A list of with the id of the tracks,
    track_names: List: A list of with the name of the tracks,
    has_beat: List: A list of the boolean values if the track has beats,
    tempo: List: A list with the tempo of the tracks,
    lufs: List: A list with the LUFS loudness of the tracks,
    """
    dataset_root = 'data/{}'.format(dataset)
    dataset_files = [f for f in os.listdir(dataset_root) if f.endswith(('.wav', '.mp3', '.aiff', '.m4a'))]
    dataset_files = sorted(dataset_files)
    n_files = len(dataset_files)
    track_names = []
    track_ids = []
    has_beat = []
    has_beat_std = []
    has_beat_diff = []
    tempo = []
    lufs = []

    dataset_features = np.zeros((n_files, len(features)))

    # Run over the dataset files and perform analysis
    logger.info(str(
        datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")) + ' Working on ' + f'{dataset} dataset of ' + f'{n_files} files')
    for index, file in enumerate(dataset_files):

        # Store the id and name of the track
        if predict:
            track = eyed3.load(os.path.join(dataset_root, file))
            if track and track.tag:
                track_names.append(track.tag.title)
            else:
                track_names.append('N/A')
        track_ids.append(file)

        # load the audio file samples
        # In predict case, we are analysing only 10 seconds from the 2/5 , 3/5 and 4/5 of the track
        # For LUFS we are using the whole length of the  track
        # For determining if the track has beat we using the first 5-45 secs of the track
        if predict:
            logger.info(str(
                datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")) + ' Predicting - Analysing ' + f'{file}. {index + 1} / {n_files}')
            audio, sample_rate = librosa.load(os.path.join(dataset_root, file), sr=None)

            # LUFS require the analysis over the entire track
            audio_lufs, _ = librosa.load(os.path.join(dataset_root, file), sr=None, mono=False)
            audio_lufs = np.array(audio_lufs).T

            # hasBeat is computed only for the first 45 sec of the track
            audio_hasBeat = audio[5 * sample_rate: 45 * sample_rate]
            '''
            audio_hasBeat = np.expand_dims(audio_hasBeat, axis=1)
            audio_hasBeat = MinMaxScaler().fit_transform(audio_hasBeat)
            audio_hasBeat = audio_hasBeat.flatten()
            '''
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
        audio = MinMaxScaler().fit_transform(audio)
        audio = audio.flatten()

        # audio = audio/audio.max()

        # Analysis variables:
        frame_length = int(np.floor(0.0232 * sample_rate))
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
        if predict:
            signal_length = len(audio_hasBeat)
            num_frames = int(np.ceil((signal_length - frame_length) / hop_length) + 1)
            pad_signal_length = (num_frames - 1) * hop_length + frame_length

            # We need to add to the signal a #zeros that correspond to (signal_length - pad_signal_length)
            z = np.zeros(pad_signal_length - signal_length)
            pad_signal = np.append(audio_hasBeat, z)

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
                frame_energy = np.sqrt(np.dot(frames[:, frame], frames[:, frame]) / len(frames[:, frame]))

                if frame_energy > 0:
                    subframes = []
                    for subframe in range(10):
                        subframe_samples = frames[subframe * subframe_length:(subframe + 1) * subframe_length, frame]
                        subframe_energy = np.sqrt(np.dot(subframe_samples, subframe_samples) / len(subframe_samples))
                        subframes.append(subframe_energy)
                    subframe_e = subframes / frame_energy
                    frame_entropy.append(-(np.dot(np.log2(subframe_e + 1e-16), subframe_e)))
                else:
                    frame_entropy.append(0)
            '''
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
            '''
            has_beat.append(np.mean(frame_entropy))
            has_beat_std.append(np.std(frame_entropy))

            # Tempo
            tempo.append(librosa.beat.tempo(audio, sr=sample_rate, start_bpm=110.0, std_bpm=15, max_tempo=190))

            # LUFS
            lufs.append(get_integrated_lufs(audio_array=audio_lufs, sample_rate=sample_rate))

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
    return dataset_features, track_ids, track_names, has_beat, has_beat_std, tempo, lufs


def check_beat(row):
    """
    This function computes if the track has a beat based on the value of dynamicity and entropy of energy
    :param row: the dataframe row wit with the HLF of the track
    :return: boolean value if the track has beat or not
    """
    if np.abs(row['Entropy']) > 0.05 and row['Entropy_STD'] >= 0.10 and row['Dynamicity']> 0.01:
        return True
    return False


def get_integrated_lufs(audio_array, sample_rate, min_duration=0.5,
                        filter_class='K-weighting', block_size=0.400):
    """
    Returns the integrated LUFS for a numpy array containing
    audio samples.

    For files shorter than 400 ms pyloudnorm throws an error. To avoid this,
    files shorter than min_duration (by default 500 ms) are self-concatenated
    until min_duration is reached and the LUFS value is computed for the
    concatenated file.

    Parameters
    ----------
    audio_array : np.ndarray
        numpy array containing samples or path to audio file for computing LUFS
    sample_rate : int
        Sample rate of audio, for computing duration
    min_duration : float
        Minimum required duration for computing LUFS value. Files shorter than
        this are self-concatenated until their duration reaches this value
        for the purpose of computing the integrated LUFS. Caution: if you set
        min_duration < 0.4, a constant LUFS value of -70.0 will be returned for
        all files shorter than 400 ms.
    filter_class : str
        Class of weighting filter used.
        - 'K-weighting' (default)
        - 'Fenton/Lee 1'
        - 'Fenton/Lee 2'
        - 'Dash et al.'
    block_size : float
        Gating block size in seconds. Defaults to 0.400.

    Returns
    -------
    loudness
        Loudness in terms of LUFS
    """
    duration = audio_array.shape[0] / float(sample_rate)
    if duration < min_duration:
        ntiles = int(np.ceil(min_duration / duration))
        audio_array = np.tile(audio_array, (ntiles, 1))
    meter = pyloudnorm.Meter(
        sample_rate, filter_class=filter_class, block_size=block_size
    )
    loudness = meter.integrated_loudness(audio_array)
    # silent audio gives -inf, so need to put a lower bound.
    loudness = max(loudness, -70)
    return loudness
