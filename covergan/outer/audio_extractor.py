import logging
import numpy as np

# from essentia import Pool
import essentia.standard as es

logger = logging.getLogger("audio_extractor")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


# def print_pool(pool: Pool):
#     np.set_printoptions(suppress=True)
#     for key in sorted(pool.descriptorNames()):
#         val = pool[key]
#         r = str(pool[key])
#         if isinstance(val, np.ndarray):
#             r = f'Array of shape {val.shape}: ' + r
#         print(f'* {key}:\n{r}')


KEY_NUM_MAP = {'C': 0, 'C#': 1, 'D': 2,
               'Eb': 3, 'E': 4, 'F': 5,
               'F#': 6, 'G': 7, 'Ab': 8,
               'A': 9, 'Bb': 10, 'B': 11}

SAMPLE_RATE = 44100  # MonoLoader resamples to 44.1 KHz
CHUNK_SECONDS = 10
SLICE_SIZE = SAMPLE_RATE * CHUNK_SECONDS


class FeatureExtractor:
    def __init__(self):
        # --- Spectral:
        self.windowing_hann_algo_ = es.Windowing(type='hann')
        self.windowing_bh_algo_ = es.Windowing(type='blackmanharris92')
        self.spectrum_algo_ = es.Spectrum()  # FFT() returns complex FFT, here we want just the magnitude spectrum
        self.mfcc_algo_ = es.MFCC(inputSize=SLICE_SIZE // 2 + 1)
        self.log_norm_ = es.UnaryOperator(type='log')
        self.spectral_contrast_algo_ = es.SpectralContrast(frameSize=SLICE_SIZE)
        self.spectral_peaks_algo_ = es.SpectralPeaks()
        self.spectrum_whitening_algo_ = es.SpectralWhitening()
        # --- Dynamics:
        self.loudness_algo_ = es.Loudness()
        # --- Rhythm:
        self.danceability_algo_ = es.Danceability()
        self.rhythm_algo_ = es.RhythmExtractor2013(method="multifeature")
        self.onset_algo_ = es.OnsetRate()
        # --- Tonal:
        self.chromagram_algo_ = es.Chromagram()
        self.hpcp_algo_ = es.HPCP()
        self.key_algo_ = es.Key(profileType='edma')

    def mfcc(self, s):
        mfcc_bands, mfcc_coefficients = self.mfcc_algo_(self.spectrum_algo_(self.windowing_hann_algo_(s)))
        melband_log = self.log_norm_(mfcc_bands)
        return mfcc_bands, mfcc_coefficients, melband_log

    def spectral(self, s):
        spectrum = self.spectrum_algo_(self.windowing_bh_algo_(s))

        spectral_contrast, spectral_valley = self.spectral_contrast_algo_(spectrum)
        frequencies, magnitudes = self.spectral_peaks_algo_(spectrum)
        magnitudes = self.spectrum_whitening_algo_(spectrum, frequencies, magnitudes)

        return spectral_contrast, spectral_valley, frequencies, magnitudes

    def dynamics(self, s):
        loudness = self.loudness_algo_(s)
        return loudness

    def rhythm(self, s):
        danceability = self.danceability_algo_(s)[0]
        bpm, ticks, confidence, estimates, bpm_intervals = self.rhythm_algo_(s)
        # New instance, parametrized by ticks
        beats_loudness_algo = es.BeatsLoudness(sampleRate=SAMPLE_RATE, beats=ticks)
        mean_beats_loudness = beats_loudness_algo(s)[0].mean()

        self.onset_algo_.reset()  # Has to be reset between slices
        onsets, onset_rate = self.onset_algo_(s)

        return danceability, bpm, mean_beats_loudness, onset_rate

    def chromagram(self, s):
        chromagram = []
        const_q_frame_size = 32768
        for frame in es.FrameGenerator(s, frameSize=const_q_frame_size,
                                       hopSize=const_q_frame_size, startFromZero=True):
            chromagram.append(self.chromagram_algo_(frame))
        chromagram = np.concatenate(chromagram)

        return chromagram

    def tonal(self, frequencies, magnitudes):
        hpcp = self.hpcp_algo_(frequencies, magnitudes)
        key, key_scale, key_strength = self.key_algo_(hpcp)[:3]
        is_major = int(key_scale == "major")
        key_num = KEY_NUM_MAP[key]

        return key_num, is_major, key_strength

    def end_track(self):
        self.rhythm_algo_.reset()


ext = FeatureExtractor()


def audio_to_embedding(file_path: str, f_num=None) -> np.array:
    if f_num is not None:
        logger.info(f'Extracting #{f_num} audio embeddings for {file_path}')
    else:
        logger.info(f'Extracting audio embeddings for {file_path}')
    metadata_reader = es.MetadataReader(filename=file_path, failOnError=True)
    metadata = metadata_reader()
    pool_meta, meta_duration, meta_bitrate, meta_sample_rate, meta_channels = metadata[7:]

    loader = es.MonoLoader(filename=file_path)
    audio = loader()

    # Signal duration
    duration_algo = es.Duration()
    duration = duration_algo(audio)

    assert abs(meta_duration - duration) < 1.0, f'Incomplete file "{file_path}": meta {meta_duration}, read {duration}'

    # Slice the track in 10s chunks
    slices = es.FrameGenerator(audio, frameSize=SLICE_SIZE, hopSize=SLICE_SIZE // 2, startFromZero=True)

    # Compute the audio features on slices
    embeddings = []
    for s in slices:
        # Spectral
        mfcc_bands, mfcc_coefficients, melband_log = ext.mfcc(s)
        spectral_contrast, spectral_valley, frequencies, magnitudes = ext.spectral(s)
        # Dynamics
        loudness = ext.dynamics(s)
        # Rhythm
        danceability, bpm, mean_beats_loudness, onset_rate = ext.rhythm(s)
        # Tonal
        chromagram = ext.chromagram(s)
        key_num, is_major, key_strength = ext.tonal(frequencies, magnitudes)

        # Normalization
        mfcc_bands /= 0.003  # [0, 0.003)
        mfcc_coefficients = (mfcc_coefficients + 1500) / 1800  # (-1500, 300)
        melband_log = (melband_log + 70) / 70  # (-70, 0]
        # chromagram is already normalized
        spectral_contrast += 1  # [-1, 0]
        spectral_valley = (spectral_valley + 70) / 70  # (-70, 0]
        bpm = (bpm - 50) / 150  # (50, 200)
        loudness /= 4000  # [0, 4000)
        mean_beats_loudness /= 3  # [0, 3)
        danceability /= 12  # (0, 12)
        onset_rate /= 10  # (0, 10)
        key_num /= (len(KEY_NUM_MAP) - 1)  # [0, 11]
        # is_major is already normalized
        key_strength = (key_strength + 1) / 2  # [-1, 1)

        embed = np.concatenate([
            mfcc_bands, mfcc_coefficients, melband_log, chromagram,
            spectral_contrast, spectral_valley,
            np.array([bpm, loudness, mean_beats_loudness, danceability,
                      onset_rate, key_num, is_major, key_strength])
        ])
        embeddings.append(embed)

    ext.end_track()

    result = np.stack(embeddings)

    return result
