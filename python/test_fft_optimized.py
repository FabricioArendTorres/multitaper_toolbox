"""Regression test: verify fft_optimized optimization matches original fft output."""

import timeit

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.signal import chirp
from scipy.signal import detrend as scipy_detrend

from multitaper_spectrogram_python import fast_detrend, multitaper_spectrogram


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
def test_fft_optimized_matches_fft(
    data_and_fs, base_kwargs, weighting, multiprocess, detrend_opt
):
    """Verify fft_optimized implementation produces identical output to fft version for all weighting schemes."""
    data, fs = data_and_fs
    kwargs = {
        **base_kwargs,
        "weighting": weighting,
        "multiprocess": multiprocess,
        "detrend_opt": detrend_opt,
    }

    result_fft, _, _ = multitaper_spectrogram(data, fs, use_legacy=True, **kwargs)
    result_fft_optimized, _, _ = multitaper_spectrogram(
        data, fs, use_legacy=False, **kwargs
    )

    assert_allclose(result_fft, result_fft_optimized, rtol=1e-10, atol=1e-12)


##################
# Test suite for fast_detrend implementation, comparing against scipy.signal.detrend for correctness and performance.
##################


@pytest.fixture
def rng():
    """Return a seeded random number generator for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def random_data_1d(rng):
    """Return random 1D test data."""
    return rng.standard_normal(1000)


@pytest.fixture
def random_data_500(rng):
    """Return random 1D test data with 500 elements."""
    return rng.standard_normal(500)


class TestFastDetrend:
    """Comprehensive test suite comparing fast_detrend to scipy.signal.detrend"""

    @pytest.mark.parametrize("kind", ["linear", "constant", "off"])
    def test_detrend_types(self, random_data_1d, kind):
        """Test all detrend types against scipy."""
        result_fast = fast_detrend(random_data_1d, type=kind)

        if kind == "off":
            assert_allclose(result_fast, random_data_1d, rtol=1e-10, atol=1e-10)
        else:
            result_scipy = scipy_detrend(random_data_1d, type=kind)
            assert_allclose(result_fast, result_scipy, rtol=1e-10, atol=1e-10)

    @pytest.mark.parametrize("n", [100, 256, 500, 1024, 2048, 4096])
    @pytest.mark.parametrize("kind", ["linear", "constant"])
    def test_various_lengths(self, rng, n, kind):
        """Test with various data lengths (typical spectrogram window sizes)."""
        data = rng.standard_normal(n)

        result_fast = fast_detrend(data, type=kind)
        result_scipy = scipy_detrend(data, type=kind)

        assert_allclose(result_fast, result_scipy, rtol=1e-10, atol=1e-10)

    def test_linear_with_offset(self, rng):
        """Test linear detrending on data with clear linear trend."""
        t = np.arange(1000)
        slope, intercept = 2.5, -10.0
        data = slope * t + intercept + rng.standard_normal(1000) * 0.1

        result_fast = fast_detrend(data, type="linear")
        result_scipy = scipy_detrend(data, type="linear")

        assert_allclose(result_fast, result_scipy, rtol=1e-10, atol=1e-10)

        # Verify the trend was removed (residual mean should be ~0)
        assert abs(result_fast.mean()) < 1e-8

    def test_constant_with_offset(self, random_data_500):
        """Test constant detrending on data with DC offset."""
        data = random_data_500 + 100.0

        result_fast = fast_detrend(data, type="constant")
        result_scipy = scipy_detrend(data, type="constant")

        assert_allclose(result_fast, result_scipy, rtol=1e-10, atol=1e-10)

        # Verify mean was removed
        assert abs(result_fast.mean()) < 1e-8

    def test_integer_input(self, rng):
        """Test that integer arrays work correctly."""
        data = rng.integers(-100, 100, size=1000).astype(float)

        result_fast = fast_detrend(data, type="linear")
        result_scipy = scipy_detrend(data, type="linear")

        assert_allclose(result_fast, result_scipy, rtol=1e-10, atol=1e-10)

    def test_nan_input(self, random_data_1d):
        """Test behavior with NaN values - should propagate NaN."""
        data = random_data_1d.copy()
        data[100] = np.nan
        data[500] = np.nan

        result_fast = fast_detrend(data, type="linear")

        # NaN positions should be NaN
        assert np.isnan(result_fast[100])
        assert np.isnan(result_fast[500])

    def test_all_zeros(self):
        """Test with all zeros input."""
        data = np.zeros(1000)

        result_fast = fast_detrend(data, type="linear")

        assert np.allclose(result_fast, 0)

    def test_constant_signal(self):
        """Test with a constant signal (no variation)."""
        data = np.ones(1000) * 5.0

        result_fast = fast_detrend(data, type="linear")
        result_scipy = scipy_detrend(data, type="linear")

        assert_allclose(result_fast, result_scipy, rtol=1e-10, atol=1e-10)

    def test_pure_linear_signal(self):
        """Test with a pure linear signal (no noise) - should become zeros."""
        t = np.arange(1000)
        data = 3.0 * t + 7.0

        result_fast = fast_detrend(data, type="linear")
        result_scipy = scipy_detrend(data, type="linear")

        assert_allclose(result_fast, result_scipy, rtol=1e-10, atol=1e-10)
        assert np.allclose(result_fast, 0)

    @pytest.mark.parametrize(
        "dtype,rtol,atol",
        [
            (np.float64, 1e-14, 1e-14),
            (np.float32, 1e-6, 1e-6),
        ],
    )
    def test_precision(self, rng, dtype, rtol, atol):
        """Test with different floating point precisions."""
        data = rng.standard_normal(1000).astype(dtype)

        result_fast = fast_detrend(data, type="linear")
        result_scipy = scipy_detrend(data, type="linear")

        assert_allclose(result_fast, result_scipy, rtol=rtol, atol=atol)

    @pytest.mark.parametrize(
        "scale,atol",
        [
            (1e6, 1e-6),
            (1e-6, 1e-12),
        ],
    )
    def test_extreme_values(self, rng, scale, atol):
        """Test with extreme magnitude values."""
        data = rng.standard_normal(500) * scale

        result_fast = fast_detrend(data, type="linear")
        result_scipy = scipy_detrend(data, type="linear")

        assert_allclose(result_fast, result_scipy, rtol=1e-10, atol=atol)

    def test_performance_faster_than_scipy(self, random_data_1d):
        """Sanity check: fast_detrend should be faster than scipy."""
        number = 100
        t_fast = min(timeit.repeat(lambda: fast_detrend(random_data_1d), number=number))
        t_scipy = min(
            timeit.repeat(lambda: scipy_detrend(random_data_1d), number=number)
        )
        assert t_fast < t_scipy, (
            f"fast_detrend ({t_fast:.2e}s) should be faster than scipy ({t_scipy:.2e}s)"
        )

    def test_numerically_unstable_high_condition_number(self):
        """Test numerically unstable case: two points with vastly different x values.

        For two points (0, 0) and (1e9, 1e9), the slope should be 1.
        With floating point, (x - x_mean) can overflow or lose precision.
        """
        x = np.array([0.0, 1e9])
        x.copy()  # Perfect linear relationship

        result = fast_detrend(x, type="linear")

        # With such extreme values, we may lose precision but should still
        # get something reasonable (the exact residual depends on implementation)
        # The key is that it's consistent with scipy
        result_scipy = scipy_detrend(x, type="linear")
        assert_allclose(result, result_scipy, rtol=1e-10, atol=1e-6)

    def test_numerically_unstable_near_horizontal(self):
        """Test near-horizontal line where slope is very small.

        A line with slope=1e-12 should be effectively horizontal after detrending.
        """
        t = np.arange(1000)
        y = 1e-12 * t + 100.0  # Very small slope, visible offset

        result_fast = fast_detrend(y, type="linear")
        result_scipy = scipy_detrend(y, type="linear")

        # Both should remove the linear trend, leaving something near zero
        assert_allclose(result_fast, result_scipy, rtol=1e-10, atol=1e-10)

        # Verify the slope was removed (residual should be ~0)
        assert np.allclose(result_fast, 0, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
