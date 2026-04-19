"""Module for optimized multiprocess spectrogram computation.

This module deliberately imports ONLY numpy and multiprocessing.shared_memory.
By keeping scipy, matplotlib, and colorcet out, joblib worker processes that
import this module stay at ~40-50 MB instead of ~150 MB each.

For 10 processes, that decreases total memory from ~1800MB to 660MB.
"""

import numpy as np
from typing_extensions import Literal
from multiprocessing.shared_memory import SharedMemory


def fast_detrend(
    data: np.ndarray, type: Literal["linear", "constant", "off"] = "linear"
) -> np.ndarray:
    """
    Remove a linear trend from the data.

    This is a fast replacement for scipy.signal.detrend optimized for
    spectrogram use cases where the function is called many times on
    windowed data.

    Limited to 1D data.

    Parameters
    ----------
    data : array_like
        The input data (1D array).
    type : {'linear', 'constant', 'off'}, optional
        The type of detrending. If 'linear' (default), the result of
        a linear least-squares fit is subtracted from the data.
        If 'constant', only the mean of the data is subtracted.
        If 'off', no detrending is applied.

    Returns
    -------
    ret : ndarray
        The detrended data with the same shape as the input.

    Notes
    -----
    For 'linear' detrending, this function uses a closed-form OLS solution:

        slope = Cov(t, y) / Var(t)
        intercept = mean(y) - slope * mean(t)

    This is significantly faster than scipy.signal.detrend which uses
    np.polyfit() with LAPACK SVD decomposition for each call.

    """
    data = np.asarray(data, dtype=float)
    n = data.shape[0]

    if type == "linear":
        t = np.arange(n, dtype=float)
        t_mean = (n - 1) / 2.0

        # Compute means
        data_mean = data.mean()

        # Compute variance of t (analytically: (n^2-1)/12 for centered t)
        t_var = np.var(t)

        # Compute covariance and slope
        slope = np.sum((t - t_mean) * data) / (n * t_var)
        intercept = data_mean - slope * t_mean

        return data - (slope * t + intercept)

    elif type == "constant":
        return data - data.mean()

    else:  # 'off' or unrecognized
        return data


def calc_mts_segment_optimized(
    data_segment: np.ndarray,
    dpss_tapers: np.ndarray,
    nfft: int,
    freq_inds: np.ndarray,
    detrend_opt: str | Literal["linear", "constant", "off"],
    num_tapers: int,
    dpss_eigen: np.ndarray,
    weighting: str,
    wt: int | np.ndarray,
):
    """Helper function to calculate the multitaper spectrum of a single segment of data.

    This is an optimized version of calc_mts_segment that uses rfft instead of fft,
    since for real-valued input the negative frequencies are redundant.

    Arguments:
        data_segment (1d np.array): One window worth of time-series data -- required
        dpss_tapers (2d np.array): Parameters for the DPSS tapers to be used.
                                   Dimensions are (num_tapers, winsize_samples) -- required
        nfft (int): length of signal to calculate fft on -- required
        freq_inds (1d np array): boolean array of which frequencies are being analyzed in
                                  an array of frequencies from 0 to fs with steps of fs/nfft
        detrend_opt (str): detrend data window ('linear' (default), 'constant', 'off')
        num_tapers (int): number of tapers being used
        dpss_eigen (np array): eigenvalues for the DPSS tapers
        weighting (str): 'unity', 'eigen', or 'adapt'
        wt (int or np array): taper weights

    Returns:
        mt_spectrum (1d np.array): spectral power for single window
    """

    # If segment has all zeros, return vector of zeros
    if all(data_segment == 0):
        ret = np.empty(sum(freq_inds))
        ret.fill(0)
        return ret

    if np.isnan(data_segment).any():
        ret = np.empty(sum(freq_inds))
        ret.fill(np.nan)
        return ret

    # Option to detrend data to remove low frequency DC component
    if detrend_opt != "off":
        data_segment = fast_detrend(data_segment, type=detrend_opt)

    # Multiply data by dpss tapers (STEP 2)
    tapered_data = data_segment[:, np.newaxis] * dpss_tapers.T

    # Compute the rFFT - returns only positive frequencies (STEP 3)
    rfft_data = np.fft.rfft(tapered_data, nfft, axis=0)

    # Compute the weighted mean spectral power across tapers (STEP 4)
    spower = np.abs(rfft_data) ** 2

    if weighting == "adapt":
        # adaptive weights - for colored noise spectrum (Percival & Walden p368-370)
        tpower = np.dot(np.transpose(data_segment), (data_segment / len(data_segment)))
        nfft_rfft = nfft // 2 + 1
        spower_iter = np.mean(spower[:, 0:2], 1)
        spower_iter = spower_iter[:, np.newaxis]
        a = (1 - dpss_eigen) * tpower
        for i in range(3):  # 3 iterations only
            # Calc the MSE weights
            # use broadcast_to to create zero-copy views
            b = np.broadcast_to(spower_iter, (nfft_rfft, num_tapers)) / (
                (np.dot(spower_iter, np.transpose(dpss_eigen)))
                + np.broadcast_to(a.ravel(), (nfft_rfft, num_tapers))
            )
            # Calc new spectral estimate
            wk = (b**2) * np.broadcast_to(dpss_eigen.ravel(), (nfft_rfft, num_tapers))
            spower_iter = np.einsum("ij,ij->i", wk, spower) / np.sum(
                wk, 1
            )  # sums over j (axis 1), output shape (513,)
            spower_iter = spower_iter[:, np.newaxis]

        mt_spectrum = np.squeeze(spower_iter)

    else:
        # eigenvalue or uniform weights
        mt_spectrum = np.dot(spower, wt)
        mt_spectrum = np.reshape(mt_spectrum, nfft // 2 + 1)  # reshape to 1D

    return mt_spectrum[freq_inds]


def _worker_shm_batch(
    input_shm_name: str,
    output_shm_name: str,
    input_shape: tuple,
    input_dtype: np.dtype,
    output_shape: tuple,
    output_dtype: np.dtype,
    batch_indices: list,
    segment_size: int,
    *mts_params,
):
    """Worker function that processes a batch of windows and writes directly to output SHM.

    Workers write results directly to the pre-allocated output buffer, avoiding the need
    to return large result arrays via pickle.
    """
    input_shm = None
    output_shm = None
    try:
        # read numpy ndarray from existing shared memory
        input_shm = SharedMemory(name=input_shm_name)
        output_shm = SharedMemory(name=output_shm_name)
        input_data = np.ndarray(
            shape=input_shape, dtype=input_dtype, buffer=input_shm.buf
        )
        output_data = np.ndarray(
            shape=output_shape, dtype=output_dtype, buffer=output_shm.buf
        )

        for window_idx in batch_indices:
            segment = input_data[
                window_idx * segment_size : (window_idx + 1) * segment_size
            ]
            result = calc_mts_segment_optimized(segment, *mts_params)
            output_data[window_idx, :] = result
    finally:
        # Close shared memory references to prevent resource leak
        if input_shm is not None:
            input_shm.close()
        if output_shm is not None:
            output_shm.close()

    return True


def _create_shared_memory_segments(
    data_segments: np.ndarray, segment_size: int
) -> tuple[SharedMemory, str, tuple[int, ...]]:
    """Create shared memory and copy data into it.

    Returns:
        shm: SharedMemory object (keep alive until workers complete)
        shm_name: string to pass to workers
        shm_shape: shape tuple for the shared memory array

    Raises:
        MemoryError: If shared memory allocation fails
        ValueError: If data dimensions are invalid
    """
    n_windows = data_segments.shape[0]
    total_size = n_windows * segment_size

    # Validate input dimensions
    if n_windows < 1:
        raise ValueError(f"Invalid number of windows: {n_windows}")
    if segment_size < 1:
        raise ValueError(f"Invalid segment size: {segment_size}")

    # Ensure input is C-contiguous before copying to shared memory
    # Non-contiguous arrays (e.g., from strided views) would cause incorrect memory access
    if not data_segments.flags["C_CONTIGUOUS"]:
        data_segments = np.ascontiguousarray(data_segments)

    bytes_needed = total_size * data_segments.dtype.itemsize

    try:
        shm = SharedMemory(create=True, size=bytes_needed)
    except FileExistsError:
        raise MemoryError(
            "Shared memory segment already exists - cleanup may be incomplete"
        )
    except OSError as e:
        raise MemoryError(f"Failed to create shared memory ({bytes_needed} bytes): {e}")

    try:
        # Create array view of shared memory & copy data
        shm_array = np.ndarray(
            shape=(total_size,), dtype=data_segments.dtype, buffer=shm.buf
        )
        shm_array[:] = data_segments.ravel()
    except Exception as e:
        # Clean up on failure to prevent orphaned shared memory
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass
        raise RuntimeError(f"Failed to copy data to shared memory: {e}") from e

    # Compute shape for workers to reconstruct array view
    shm_shape = (total_size,)

    return shm, shm.name, shm_shape


def _create_shared_memory_output(
    num_windows: int, n_freqs: int, dtype: np.dtype = np.float64
) -> tuple[SharedMemory, str, tuple[int, ...]]:
    """Create shared memory for output spectrogram results.

    Returns:
        shm: SharedMemory object (keep alive until workers complete)
        shm_name: string to pass to workers
        output_shape: shape tuple for the output array (num_windows, n_freqs)

    Raises:
        MemoryError: If shared memory allocation fails
        ValueError: If dimensions are invalid
    """
    if num_windows < 1:
        raise ValueError(f"Invalid number of windows: {num_windows}")
    if n_freqs < 1:
        raise ValueError(f"Invalid number of frequencies: {n_freqs}")

    bytes_needed = num_windows * n_freqs * np.dtype(dtype).itemsize

    if bytes_needed < 1:
        raise ValueError(f"Computed shared memory size is invalid: {bytes_needed}")

    try:
        shm = SharedMemory(create=True, size=bytes_needed)
    except FileExistsError:
        raise MemoryError(
            "Output shared memory segment already exists - cleanup may be incomplete"
        )
    except OSError as e:
        raise MemoryError(
            f"Failed to create output shared memory ({bytes_needed} bytes): {e}"
        )

    output_shape = (num_windows, n_freqs)

    return shm, shm.name, output_shape


def _cleanup_shared_memory(shm: SharedMemory):
    """Unlink shared memory.

    Handles cleanup safely - ensures close() is called before unlink()
    to prevent OS-level resource conflicts.
    """
    if shm is None:
        return
    try:
        shm.close()
    except Exception:
        pass  # Already closed or invalid
    try:
        shm.unlink()
    except FileNotFoundError:
        pass  # Already unlinked
    except Exception:
        pass  # Best effort cleanup


def _cleanup_shared_memory_pair(input_shm: SharedMemory, output_shm: SharedMemory):
    """Unlink both input and output shared memory buffers."""
    _cleanup_shared_memory(input_shm)
    _cleanup_shared_memory(output_shm)


def run_shm_multiprocess(
    data_segments: np.ndarray,
    num_windows: int,
    winsize_samples: int,
    n_jobs: int,
    mts_params: tuple,
) -> np.ndarray:
    """Run multiprocess spectrogram using shared memory.

    Returns:
        mt_spectrogram (2d np.array): shape (num_windows, n_freqs)
    """
    from joblib import Parallel, delayed

    input_shm, input_shm_name, input_shape = _create_shared_memory_segments(
        data_segments, winsize_samples
    )

    # freq_inds is the 3rd element (index 2) of mts_params
    n_freqs = np.sum(mts_params[2])

    output_shm, output_shm_name, output_shape = _create_shared_memory_output(
        num_windows, n_freqs
    )

    try:
        batches = np.array_split(np.arange(num_windows), n_jobs)

        Parallel(n_jobs=n_jobs)(
            delayed(_worker_shm_batch)(
                input_shm_name,
                output_shm_name,
                input_shape,
                np.float64,
                output_shape,
                np.float64,
                batch_indices,
                winsize_samples,
                *mts_params,
            )
            for batch_indices in batches
            if len(batch_indices) > 0
        )

        mt_spectrogram = np.ndarray(
            shape=output_shape, dtype=np.float64, buffer=output_shm.buf
        ).copy()
    finally:
        _cleanup_shared_memory_pair(input_shm, output_shm)

    return mt_spectrogram
