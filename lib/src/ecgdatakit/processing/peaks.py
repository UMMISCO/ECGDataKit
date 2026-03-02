"""R-peak detection and heart-rate utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import Lead
from ecgdatakit.processing._core import require_scipy


def detect_r_peaks(lead: Lead, method: str = "pan_tompkins") -> NDArray[np.intp]:
    """Detect R-peak locations in an ECG lead.

    Parameters
    ----------
    lead : Lead
        Input ECG lead (typically Lead II).
    method : str
        Detection algorithm: ``"pan_tompkins"`` or ``"shannon_energy"``.

    Returns
    -------
    NDArray[np.intp]
        Array of sample indices where R-peaks were detected.
    """
    methods = ("pan_tompkins", "shannon_energy")
    if method not in methods:
        raise ValueError(f"Unknown method {method!r}; choose from {methods}")
    if method == "pan_tompkins":
        return _pan_tompkins(lead.samples, lead.sample_rate)
    return _shannon_energy(lead.samples, lead.sample_rate)


def heart_rate(lead: Lead, peaks: NDArray[np.intp] | None = None) -> float:
    """Compute average heart rate in beats per minute.

    Parameters
    ----------
    lead : Lead
        Input ECG lead.
    peaks : NDArray | None
        Pre-detected R-peak indices.  Detected automatically if ``None``.
    """
    rr = rr_intervals(lead, peaks)
    if len(rr) == 0:
        return 0.0
    mean_rr_ms = float(rr.mean())
    if mean_rr_ms <= 0:
        return 0.0
    return 60_000.0 / mean_rr_ms


def rr_intervals(
    lead: Lead, peaks: NDArray[np.intp] | None = None
) -> NDArray[np.float64]:
    """Compute RR intervals in milliseconds.

    Parameters
    ----------
    lead : Lead
        Input ECG lead.
    peaks : NDArray | None
        Pre-detected R-peak indices.  Detected automatically if ``None``.
    """
    if peaks is None:
        peaks = detect_r_peaks(lead)
    if len(peaks) < 2:
        return np.array([], dtype=np.float64)
    diffs = np.diff(peaks).astype(np.float64)
    return diffs * (1000.0 / lead.sample_rate)


def instantaneous_heart_rate(
    lead: Lead, peaks: NDArray[np.intp] | None = None
) -> NDArray[np.float64]:
    """Compute instantaneous heart rate at each beat in bpm.

    Parameters
    ----------
    lead : Lead
        Input ECG lead.
    peaks : NDArray | None
        Pre-detected R-peak indices.  Detected automatically if ``None``.
    """
    rr = rr_intervals(lead, peaks)
    if len(rr) == 0:
        return np.array([], dtype=np.float64)
    return 60_000.0 / rr


def _pan_tompkins(signal: NDArray[np.float64], fs: int) -> NDArray[np.intp]:
    """Pan-Tompkins QRS detection algorithm.

    Bandpass 5--15 Hz, derivative, squaring, moving-window integration,
    adaptive thresholding with searchback, and peak refinement.
    """
    sig_mod = require_scipy("signal")

    nyquist = fs / 2.0
    low = min(5.0, nyquist - 0.5)
    high = min(15.0, nyquist - 0.5)
    if low >= high:
        low = max(0.5, high - 5.0)
    sos = sig_mod.butter(2, [low, high], btype="band", fs=fs, output="sos")
    filtered = sig_mod.sosfiltfilt(sos, signal)

    h = np.array([-1, -2, 0, 2, 1]) * (fs / 8.0)
    derivative = np.convolve(filtered, h, mode="same")

    squared = derivative ** 2

    win_len = max(1, int(round(0.150 * fs)))
    kernel = np.ones(win_len) / win_len
    integrated = np.convolve(squared, kernel, mode="same")

    peaks = _adaptive_threshold(integrated, fs)

    search_window = int(round(0.075 * fs))
    refined: list[int] = []
    for p in peaks:
        lo = max(0, p - search_window)
        hi = min(len(signal), p + search_window + 1)
        refined.append(int(lo + np.argmax(signal[lo:hi])))

    return np.array(refined, dtype=np.intp)


def _adaptive_threshold(
    integrated: NDArray[np.float64], fs: int
) -> list[int]:
    """Adaptive dual-threshold peak detection with searchback."""
    refractory = int(round(0.200 * fs))

    spki = integrated.max() * 0.25
    npki = integrated.mean() * 0.5
    threshold1 = npki + 0.25 * (spki - npki)

    peaks: list[int] = []
    rr_avg = 0.0
    i = 1
    n = len(integrated)

    while i < n - 1:
        if integrated[i] > integrated[i - 1] and integrated[i] >= integrated[i + 1]:
            if integrated[i] > threshold1:
                if len(peaks) == 0 or (i - peaks[-1]) > refractory:
                    peaks.append(i)
                    spki = 0.875 * spki + 0.125 * integrated[i]

                    if len(peaks) >= 2:
                        rr_avg = float(np.mean(np.diff(peaks[-8:])))
                        if (i - peaks[-2]) > 1.66 * rr_avg and len(peaks) >= 2:
                            threshold2 = 0.5 * threshold1
                            search_start = peaks[-2] + refractory
                            search_end = i - refractory
                            if search_start < search_end:
                                segment = integrated[search_start:search_end]
                                candidates = []
                                for j in range(1, len(segment) - 1):
                                    if (
                                        segment[j] > segment[j - 1]
                                        and segment[j] >= segment[j + 1]
                                        and segment[j] > threshold2
                                    ):
                                        candidates.append(
                                            (search_start + j, segment[j])
                                        )
                                if candidates:
                                    best = max(candidates, key=lambda x: x[1])
                                    peaks.insert(-1, best[0])
                                    peaks.sort()
            else:
                npki = 0.875 * npki + 0.125 * integrated[i]

            threshold1 = npki + 0.25 * (spki - npki)

        i += 1

    return peaks


def _shannon_energy(
    signal: NDArray[np.float64],
    fs: int,
    ransac_window_sec: float = 5.0,
    lowfreq: float = 35.0,
    highfreq: float = 43.0,
) -> NDArray[np.intp]:
    """Shannon-energy-envelope R-peak detector.

    Bandpass 35--43 Hz to isolate QRS energy, derivative power,
    RANSAC adaptive thresholding, Shannon energy transform,
    Gaussian-smoothed envelope, and zero-crossing peak detection.
    """
    sig_mod = require_scipy("signal")
    ndimage = require_scipy("ndimage")

    nyquist = fs / 2.0
    hi = min(highfreq, nyquist - 0.5)
    lo = min(lowfreq, hi - 1.0)
    if lo <= 0:
        lo = 1.0

    sos_low = sig_mod.butter(1, hi / nyquist, btype="low", output="sos")
    sos_high = sig_mod.butter(1, lo / nyquist, btype="high", output="sos")
    ecg_low = sig_mod.sosfiltfilt(sos_low, signal)
    ecg_band = sig_mod.sosfiltfilt(sos_high, ecg_low)

    decg = np.diff(ecg_band)
    decg_power = decg ** 2

    win_samples = max(1, int(ransac_window_sec * fs))
    n_windows = len(decg_power) // win_samples

    if n_windows > 0:
        usable = n_windows * win_samples
        windowed = decg_power[:usable].reshape(n_windows, win_samples)
        thresholds = 0.5 * windowed.std(axis=1)
        max_powers = windowed.max(axis=1)
        threshold = float(np.median(thresholds))
        max_power = float(np.median(max_powers))
    else:
        threshold = 0.5 * float(decg_power.std()) if len(decg_power) > 0 else 0.0
        max_power = float(decg_power.max()) if len(decg_power) > 0 else 1.0

    if max_power <= 0:
        return np.array([], dtype=np.intp)

    decg_power[decg_power < threshold] = 0.0
    decg_power /= max_power
    np.clip(decg_power, 0.0, 1.0, out=decg_power)

    sq = decg_power ** 2
    safe_sq = np.where(sq > 0, sq, 1e-20)
    shannon = -sq * np.log(safe_sq)
    shannon[~np.isfinite(shannon)] = 0.0

    mean_win = max(1, int(fs * 0.125) + 1)
    kernel = np.ones(mean_win) / mean_win
    lp_energy = np.convolve(shannon, kernel, mode="same")
    lp_energy = ndimage.gaussian_filter1d(lp_energy, sigma=fs / 8.0)

    lp_diff = np.diff(lp_energy)
    zero_crossings = np.flatnonzero(
        (lp_diff[:-1] > 0) & (lp_diff[1:] <= 0)
    )

    if len(zero_crossings) == 0:
        return np.array([], dtype=np.intp)

    search_win = int(round(0.075 * fs))
    refined: list[int] = []
    for zc in zero_crossings:
        lo_idx = max(0, zc - search_win)
        hi_idx = min(len(signal), zc + search_win + 1)
        refined.append(int(lo_idx + np.argmax(signal[lo_idx:hi_idx])))

    return np.array(refined, dtype=np.intp)
