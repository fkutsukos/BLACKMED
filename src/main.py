import logging
import datetime
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import librosa
from colorlog import ColoredFormatter
import os
from src.features import features

LOG_LEVEL = logging.DEBUG
LOGFORMAT = "%(log_color)s%(message)s%(reset)s"
logging.root.setLevel(LOG_LEVEL)
formatter = ColoredFormatter(LOGFORMAT)
stream = logging.StreamHandler()
stream.setLevel(LOG_LEVEL)
stream.setFormatter(formatter)
logger = logging.getLogger('colorlogger')
logger.setLevel(LOG_LEVEL)
logger.addHandler(stream)
logging.basicConfig(filename='blckmd.log')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Starting the program')

    # TODO: Create function to retrieve and analyse the next track online.

    # Task 1 - TRAINING
    # Read the tracks based on their tagging.
    train_path = '../data/'
    classes = ['Dynamicity']
    for c in classes:
        train_root = 'data/{}/'.format(c)
        class_train_files = [f for f in os.listdir(train_root) if f.endswith('.wav')]
        logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + ' Starting now the ' + f'{c} files')
        n_train_files = len(class_train_files)

        # train_features = np.zeros((n_train_files, n_mfcc))
        # For each category load the file and get its window frames
        for index, f in enumerate(class_train_files):
            logger.info(str(datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S")) + ' Training the ' + f'{f} at position {index + 1} out of {n_train_files}')
            audio, Fs = librosa.load(os.path.join(train_root, f), sr=None)

            # train_features[index, :] = np.mean(mfcc, axis=1)

            # Define window properties
            win_length = int(np.floor(0.0213 * Fs))
            hop_size = int(np.floor(win_length / 2))
            window = sp.signal.get_window(window='hanning', Nx=win_length)
            audio_length = audio.shape[0]  # len(audio)
            audio_frames = int(np.floor((audio_length - win_length) / hop_size))
            # print(train_win_number)

            # logger.info(str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + 'Samples: ' + str(audio_train_length) + ', seconds: ' + str(np.floor(audio_train_length / Fs)) + ', Train Windows: ' + str(train_win_number))

            # Create features matrix
            n_features = 4
            train_features = np.zeros((audio_frames, n_features))
            for i in np.arange(audio_frames):
                frame = audio[i * hop_size: i * hop_size + win_length]
                frame_wind = frame * window
                spec = np.fft.fft(frame_wind)
                nyquist = int(np.floor(spec.shape[0] / 2))
                spec = spec[1:nyquist]

                #train_features[i, 0] = features(spec, frame_wind, Fs).compute_rolloff()
                train_features[i, 1] = features(spec, frame_wind, Fs).compute_loudness()
                # train_features[i, 2] = features(spec, frame_wind, Fs).compute_bandwidth()
                # train_features[i, 3] = features(spec, frame_wind, Fs).compute_spectral_flux()

            time_axis = np.arange(audio.shape[0]) / Fs
            plt.plot(time_axis, audio)
            plt.grid(True)
            plt.title('Train audio')
            plt.show()
