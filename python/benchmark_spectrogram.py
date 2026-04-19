import numpy as np
import psutil
import timeit
from typing import NamedTuple

from multitaper_spectrogram_python import multitaper_spectrogram


# ----------------------------------------------------------------------
# Benchmark configuration constants
# ----------------------------------------------------------------------
SAMPLE_RATE_HZ = 100
DURATION_SEC = 1 * 3600  
FREQUENCY_RANGE = [0, 50]
TIME_BANDWIDTH = 5
NUM_TAPERS = 9
WINDOW_PARAMS = [5, 1]

N_JOBS_LIST = [1, 6] # First run with 1 Process (serial), then with 6 Processes (parallel)
REPEAT_COUNT = 10  # Number of timing repetitions for min()

# Option values iterated in benchmarks
WEIGHTING_OPTIONS = ['unity', 'eigen', 'adapt']
DETREND_OPTIONS = ['off', 'constant', 'linear']


# ----------------------------------------------------------------------
# Data types
# ----------------------------------------------------------------------
class BenchmarkResult(NamedTuple):
    """Single benchmark measurement result."""
    use_legacy: bool
    weighting: str
    detrend: str
    elapsed_seconds: float
    peak_memory_mb: float


# ----------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------
def find_result(
    results: list[BenchmarkResult],
    use_legacy: bool,
    weighting: str,
    detrend: str
) -> BenchmarkResult:
    """Find result for a specific configuration."""
    for r in results:
        if r.use_legacy == use_legacy and r.weighting == weighting and r.detrend == detrend:
            return r
    raise ValueError(f"Result not found: use_legacy={use_legacy}, weighting={weighting}, detrend={detrend}")


def print_benchmark_table(results: list[BenchmarkResult]) -> None:
    """Print individual benchmark results as a formatted table."""
    header = f"{'use_legacy':>6} {'weighting':>10} {'detrend':>10} {'time (s)':>12} {'peak (MB)':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{str(r.use_legacy):>6} {r.weighting:>10} {r.detrend:>10} {r.elapsed_seconds:>12.2f} {r.peak_memory_mb:>12.2f}")


def print_speedup_table(results: list[BenchmarkResult]) -> None:
    """Print speedup and memory comparison (new FFT vs legacy FFT)."""
    print(f"\n--- new FFT vs legacy ({len(results)} configs) ---")
    header = f"{'weighting':>10} {'detrend':>10} {'speedup':>10} {'mem_ratio':>10}"
    print(header)
    print("-" * len(header))

    for weighting in WEIGHTING_OPTIONS:
        for detrend in DETREND_OPTIONS:
            r_legacy = find_result(results, use_legacy=True, weighting=weighting, detrend=detrend)
            r_new = find_result(results, use_legacy=False, weighting=weighting, detrend=detrend)
            speedup = r_legacy.elapsed_seconds / r_new.elapsed_seconds if r_new.elapsed_seconds > 0 else float('inf')
            mem_ratio = r_legacy.peak_memory_mb / r_new.peak_memory_mb if r_new.peak_memory_mb > 0 else float('inf')
            print(f"{weighting:>10} {detrend:>10} {speedup:>10.2f}x {mem_ratio:>10.2f}x")


# ----------------------------------------------------------------------
# Core benchmark logic
# ----------------------------------------------------------------------
def get_peak_memory_mb() -> float:
    """Get peak RSS memory across main process + all children (multiprocess-aware)."""
    process = psutil.Process()
    total_rss = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total_rss += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total_rss / (1024 * 1024)


def run_single_benchmark(
    data: np.ndarray,
    fs: float,
    use_legacy: bool,
    weighting: str,
    detrend_opt: str,
    n_jobs: int,
) -> float:
    """Run one spectrogram computation and return peak memory in MB.
    
    Uses psutil to capture peak RSS across main process + all child workers.
    """
    multitaper_spectrogram(
        data,
        fs,
        frequency_range=FREQUENCY_RANGE,
        time_bandwidth=TIME_BANDWIDTH,
        num_tapers=NUM_TAPERS,
        window_params=WINDOW_PARAMS,
        detrend_opt=detrend_opt,
        weighting=weighting,
        multiprocess=(n_jobs > 1),
        n_jobs=n_jobs if n_jobs > 1 else None,
        plot_on=False,
        verbose=False,
        use_legacy=use_legacy,
    )
    return get_peak_memory_mb()


def benchmark(
    data: np.ndarray,
    fs: float,
    use_legacy: bool,
    weighting: str,
    detrend_opt: str,
    n_jobs: int
) -> tuple[float, float]:
    """Measure elapsed time and peak memory for one configuration.
    
    Time is measured as the minimum of REPEAT_COUNT runs (best effort).
    Peak memory is measured once after the timing loop.
    """
    # --- Timing loop ---
    def run_timed():
        start = timeit.default_timer()
        run_single_benchmark(data, fs, use_legacy, weighting, detrend_opt, n_jobs)
        return timeit.default_timer() - start

    best_elapsed = min(run_timed() for _ in range(REPEAT_COUNT))

    # --- Memory measurement (uses psutil to capture all processes) ---
    peak_mb = run_single_benchmark(data, fs, use_legacy, weighting, detrend_opt, n_jobs)

    return best_elapsed, peak_mb


def collect_all_results(data: np.ndarray, fs: float, n_jobs: int) -> list[BenchmarkResult]:
    """Run benchmarks for all option combinations."""
    results: list[BenchmarkResult] = []

    for use_legacy_opt in [True, False]:
        for weighting in WEIGHTING_OPTIONS:
            for detrend in DETREND_OPTIONS:
                elapsed, peak_mb = benchmark(data, fs, use_legacy_opt, weighting, detrend, n_jobs)
                results.append(BenchmarkResult(
                    use_legacy=use_legacy_opt,
                    weighting=weighting,
                    detrend=detrend,
                    elapsed_seconds=elapsed,
                    peak_memory_mb=peak_mb
                ))
                print(f"{str(use_legacy_opt):>6} {weighting:>10} {detrend:>10} {elapsed:>12.2f} {peak_mb:>12.2f}")

    return results


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main() -> None:
    """Run benchmarks for serial and parallel execution modes."""
    data = np.random.randn(SAMPLE_RATE_HZ * DURATION_SEC)

    print(f"Data: {DURATION_SEC}s ({len(data):,} samples) @ {SAMPLE_RATE_HZ}Hz\n")

    for n_jobs in N_JOBS_LIST:
        label = "serial" if n_jobs == 1 else f"parallel ({n_jobs} proc)"
        print(f"=== {label} ===")

        print(f"{'use_legacy':>6} {'weighting':>10} {'detrend':>10} {'time (s)':>12} {'peak (MB)':>12}")
        print("-" * 58)
        results = collect_all_results(data, SAMPLE_RATE_HZ, n_jobs)
        print_speedup_table(results)
        print()


if __name__ == "__main__":
    main()
