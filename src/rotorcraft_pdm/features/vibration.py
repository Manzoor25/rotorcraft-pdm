# =============================================================
# WHAT THIS FILE DOES:
# Turns a raw accelerometer trace into a short list of numbers:
# how spiky it is, how energetic each frequency band is, and
# how peaked the spectrum is.
#
# HOW IT CONNECTS TO OTHER FILES:
# - data/vibration.py loads the Airbus rows
# - models/anomaly.py trains on these numbers, not on 61,440 raw samples
#
# WHY NOT FEED THE RAW TRACE TO THE MODEL:
# One minute at 1024 Hz is 61,440 numbers. Most of that is the
# repeating vibration of a healthy rotor. The summaries below are
# the parts a maintainer already looks at on a HUMS screen.
# =============================================================

import numpy as np

from rotorcraft_pdm.settings import VIBRATION_SAMPLE_RATE_HZ, VIBRATION_WINDOW_SAMPLES

# Bands in Hz. We do not know the rotor speed of a given flight, so
# the bands are broad on purpose. A narrow "1x rotor" bin would be a
# guess. See DECISIONS.md.
BANDS_HZ = ((0, 20), (20, 50), (50, 100), (100, 200), (200, 512))


# --------------------------------------------------------------
# FUNCTION: crest_factor
# WHAT IT DOES: Peak divided by the root-mean-square
# WHY WE NEED IT: A bearing knock is a tall spike on an otherwise
#                 steady trace. Crest factor rises when that happens.
# INPUTS: a 1-D window
# OUTPUTS: A float. 1.0 would be a flat line of the same height as the peak.
# --------------------------------------------------------------
def crest_factor(window):
    rms = np.sqrt(np.mean(np.square(window)))
    peak = np.max(np.abs(window))
    if rms <= 1e-12:
        return 0.0
    return float(peak / rms)


# --------------------------------------------------------------
# FUNCTION: kurtosis
# WHAT IT DOES: A tailed-ness score. Gaussian noise is about 3.
# WHY WE NEED IT: Impulsive faults have heavier tails than healthy vibration.
# INPUTS: a 1-D window
# OUTPUTS: A float
# --------------------------------------------------------------
def kurtosis(window):
    centered = window - np.mean(window)
    variance = np.mean(np.square(centered))
    if variance <= 1e-12:
        return 0.0
    return float(np.mean(centered**4) / (variance**2))


# --------------------------------------------------------------
# FUNCTION: band_energies
# WHAT IT DOES: Sums FFT power inside each frequency band
# WHY WE NEED IT: A fault often adds energy at a band, not everywhere
# INPUTS: a 1-D window, sample rate
# OUTPUTS: A list of floats, one per band in BANDS_HZ
# --------------------------------------------------------------
def band_energies(window, sample_rate=VIBRATION_SAMPLE_RATE_HZ):
    spectrum = np.fft.rfft(window)
    power = np.square(np.abs(spectrum))
    freqs = np.fft.rfftfreq(len(window), d=1.0 / sample_rate)
    energies = []
    for low, high in BANDS_HZ:
        mask = (freqs >= low) & (freqs < high)
        energies.append(float(np.sum(power[mask])))
    total = float(np.sum(power)) or 1.0
    return [energy / total for energy in energies]


# --------------------------------------------------------------
# FUNCTION: window_features
# WHAT IT DOES: One feature vector for one window
# INPUTS: window (1-D array)
# OUTPUTS: A 1-D numpy array
# --------------------------------------------------------------
def window_features(window):
    window = np.asarray(window, dtype=float)
    rms = float(np.sqrt(np.mean(np.square(window))))
    return np.asarray(
        [rms, crest_factor(window), kurtosis(window), *band_energies(window)],
        dtype=np.float32,
    )


# --------------------------------------------------------------
# FUNCTION: sequence_feature_matrix
# WHAT IT DOES: Splits a 1-minute trace into 1-second windows and
#               summarizes each window, then averages those summaries.
# WHY WE NEED IT: The Airbus row is one minute. A single FFT of the
#                 whole minute hides a fault that lasts two seconds.
#                 Windowing, then averaging, keeps both views.
# INPUTS: samples, a 1-D trace
# OUTPUTS: A 1-D feature vector (mean of windows, plus the max crest factor)
# --------------------------------------------------------------
def sequence_feature_vector(samples, window=VIBRATION_WINDOW_SAMPLES):
    samples = np.asarray(samples, dtype=float).ravel()
    if len(samples) < window:
        return window_features(samples)
    vectors = []
    for start in range(0, len(samples) - window + 1, window):
        vectors.append(window_features(samples[start : start + window]))
    stacked = np.vstack(vectors)
    mean = stacked.mean(axis=0)
    # Max crest factor is the spikiest second. Averaging would hide it.
    max_crest = np.max(stacked[:, 1])
    return np.concatenate([mean, [max_crest]]).astype(np.float32)


FEATURE_NAMES = [
    "rms",
    "crest_factor",
    "kurtosis",
    *[f"band_{low}_{high}" for low, high in BANDS_HZ],
    "max_crest_factor",
]
