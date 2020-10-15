import librosa
import numpy as np
from librosa.feature import melspectrogram


class features:
    def __init__(self, spec, frame, fs):
        self.spec = spec
        self.fs = fs
        self.frame = frame

    def compute_rolloff(self):
        ROE = 0.95 * (sum(np.abs(self.spec) ** 2))
        E = 0
        for k in np.arange(len(self.spec)):
            E = E + np.abs(self.spec[k]) ** 2
            if E >= ROE:
                break
        rolloff = k * (self.fs / 2) / len(self.spec)
        return rolloff

    def compute_loudness(self):
        pS = self.spec ** 2  # powerspectrum
        weighting = librosa.A_weighting  # weighting in dB
        #print(weighting)
        #weighting = 10 ** (weighting / 10)  # weighting for power spectrogram
        pS *= weighting  # perceptually weighted power spectrogram
        melS = melspectrogram(pS)  # perceptual pitch distances
        loudness = np.mean(melS)  # taking mean is ok, because not in dB
        loudness = librosa.logamplitude(loudness)  # convert to dB
        return loudness

    def compute_bandwidth(self):
        bandwidth = librosa.feature.spectral_bandwidth(S=self.spec, sr=self.fs)
        return bandwidth

    def compute_spectral_flux(self):
        spec_b = np.fft.fft(self.frame[:-1])
        spec_a = np.fft.fft(self.frame[1:])
        flux = np.sqrt(sum((np.abs(spec_b) - np.abs(spec_a)) ** 2))
        return flux

