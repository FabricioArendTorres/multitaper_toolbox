"""Regression test: verify rfft optimization matches original fft output."""

import numpy as np
import pytest
from scipy.signal import chirp
from multitaper_spectrogram_python import multitaper_spectrogram, fast_detrend
from scipy.signal import detrend as scipy_detrend
from numpy.testing import assert_allclose
import timeit

@pytest.fixture
def data_and_fs():
    """Shared test data and sampling frequency."""
    np.random.seed(42)
    fs = 200
    t = np.arange(1 / fs, 10, 1 / fs)
    data = chirp(t, 1, t[-1], 20, "logarithmic")
    return data, fs


@pytest.fixture
def base_kwargs():
    """Base kwargs shared across all tests."""
    return {
        "frequency_range": [0, 50],
        "time_bandwidth": 3,
        "num_tapers": 5,
        "window_params": [2, 0.5],
        "min_nfft": 256,
        "plot_on": False,
        "verbose": False,
    }



@pytest.mark.parametrize("weighting", ["unity", "eigen", "adapt"])
@pytest.mark.parametrize("multiprocess", [False, True])
@pytest.mark.parametrize("detrend_opt", ["off", "constant", "linear"])
def test_rfft_matches_fft(data_and_fs, base_kwargs, weighting, multiprocess, detrend_opt):
    """Verify rfft implementation produces identical output to fft version for all weighting schemes."""
    data, fs = data_and_fs
    kwargs = {**base_kwargs, "weighting": weighting, "multiprocess": multiprocess, "detrend_opt": detrend_opt}

    result_fft, _, _ = multitaper_spectrogram(data, fs, **kwargs)
    result_rfft, _, _ = multitaper_spectrogram(data, fs, use_rfft=True, **kwargs)

    assert_allclose(result_fft, result_rfft, rtol=1e-10, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
