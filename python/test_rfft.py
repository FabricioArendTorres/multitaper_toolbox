"""Regression test: verify rfft optimization matches original fft output."""

import numpy as np
from scipy.signal import chirp
from multitaper_spectrogram_python import multitaper_spectrogram


def test_rfft_matches_fft():
    """Verify rfft implementation produces identical output to fft version."""
    np.random.seed(42)
    fs = 200
    t = np.arange(1 / fs, 10, 1 / fs)
    data = chirp(t, 1, t[-1], 20, "logarithmic")

    kwargs = {
        "frequency_range": [0, 50],
        "time_bandwidth": 3,
        "num_tapers": 5,
        "window_params": [2, 0.5],
        "min_nfft": 256,
        "detrend_opt": "constant",
        "multiprocess": False,
        "weighting": "unity",
        "plot_on": False,
        "verbose": False,
    }

    result_fft, _, _ = multitaper_spectrogram(data, fs, **kwargs)
    result_rfft, _, _ = multitaper_spectrogram(data, fs, use_rfft=True, **kwargs)

    np.testing.assert_allclose(result_fft, result_rfft, rtol=1e-10)


if __name__ == "__main__":
    test_rfft_matches_fft()
