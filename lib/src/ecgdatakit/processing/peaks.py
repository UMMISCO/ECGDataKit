"""R-peak detection and heart-rate utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.models import LeadLike
from ecgdatakit.processing._core import ensure_lead, require_scipy

_METHODS = ("shannon_energy", "pan_tompkins")
_MIN_DURATION_S = 1.0
_REFRACTORY_S = 0.200
_REFINE_HALF_WIDTH_S = 0.075
_PT_MIN_ENERGY = 1e-6
_FLAT_RUN_S = 2.0
_PT_CANDIDATE_FLOOR = 0.01
_ENVELOPE_MIN_FRACTION = 0.02


def detect_r_peaks(
    lead: LeadLike, method: str = "pan_tompkins", *, fs: int | None = None
) -> NDArray[np.intp]:
    """Detect R-peak locations in an ECG lead.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array (typically Lead II).  Any
        polarity is accepted: on leads where the QRS is mainly negative
        (aVR, often V1) the peak is placed on the dominant deflection.
    method : str
        Detection algorithm: ``"pan_tompkins"`` (default) or
        ``"shannon_energy"``.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.

    Returns
    -------
    NDArray[np.intp]
        Sorted, unique sample indices of the detected R-peaks.  Exactly
        constant runs of 2 s or more (padding, disconnected lead) are
        skipped and the stretches between them are processed separately;
        stretches shorter than 1 s between such runs are not analysed.  The
        indices refer to the full signal, so consecutive peaks on either
        side of a skipped run are one RR interval apart in
        :func:`rr_intervals` (for example 10 s of ECG, 5 s flat, 10 s of
        ECG gives one interval longer than 5 s, such as 5.76 s).

    Raises
    ------
    ValueError
        Unknown *method*, non-positive sample rate, signal that is not 1-D,
        shorter than 1 s, or containing NaN or infinite values.
    """
    lead = ensure_lead(lead, fs=fs)
    if method not in _METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {_METHODS}")
    signal = _validate_signal(lead.samples, lead.sampling_rate)
    fs_ = lead.sampling_rate
    detector = _pan_tompkins if method == "pan_tompkins" else _shannon_energy
    found = [
        start + detector(signal[start:end], fs_)
        for start, end in _active_segments(signal, fs_)
    ]
    if not found:
        return np.array([], dtype=np.intp)
    return np.concatenate(found).astype(np.intp)


def _active_segments(signal: NDArray[np.float64], fs: int) -> list[tuple[int, int]]:
    """Split the signal around exactly constant runs of 2 s or more.

    Such runs are padding or a saturated / disconnected lead; detecting on
    each remaining stretch separately keeps the step into the padding from
    being taken as a beat.  Stretches shorter than 1 s are skipped.
    """
    min_run = max(2, int(round(_FLAT_RUN_S * fs)))
    flat = np.concatenate(([False], np.diff(signal) == 0))
    # mark every sample of a constant run (including its first sample)
    flat[:-1] |= flat[1:]
    edges = np.flatnonzero(np.diff(np.concatenate(([0], flat.astype(np.int8), [0]))))
    mask = np.ones(len(signal), dtype=bool)
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= min_run:
            mask[a:b] = False
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    min_len = int(np.ceil(_MIN_DURATION_S * fs))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


def heart_rate(lead: LeadLike, peaks: NDArray[np.intp] | None = None, *, fs: int | None = None) -> float:
    """Compute average heart rate in beats per minute.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array.
    peaks : NDArray | None
        Pre-detected R-peak indices.  Detected automatically if ``None``.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.

    Returns
    -------
    float
        ``60000 / mean RR`` in bpm, or ``nan`` when fewer than two peaks
        are available.  With automatic detection, an interval spanning a
        skipped flat run (see :func:`detect_r_peaks`) is included and
        lowers the result.
    """
    lead = ensure_lead(lead, fs=fs)
    rr = rr_intervals(lead, peaks)
    if len(rr) == 0:
        return float("nan")
    return 60_000.0 / float(rr.mean())


def rr_intervals(
    lead: LeadLike, peaks: NDArray[np.intp] | None = None, *, fs: int | None = None
) -> NDArray[np.float64]:
    """Compute RR intervals in milliseconds.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array.
    peaks : NDArray | None
        Pre-detected R-peak indices (strictly increasing).  Detected
        automatically if ``None``.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.

    Returns
    -------
    NDArray[np.float64]
        RR intervals in ms; empty when fewer than two peaks are available.
        The interval between the last beat before and the first beat after
        a skipped flat run (see :func:`detect_r_peaks`) spans the whole
        gap; remove such intervals before HRV analysis.

    Raises
    ------
    ValueError
        If *peaks* is not a 1-D array of non-negative, strictly increasing
        indices.
    """
    lead = ensure_lead(lead, fs=fs)
    if lead.sampling_rate is None or lead.sampling_rate <= 0:
        raise ValueError(f"sampling rate must be positive, got {lead.sampling_rate}")
    peaks = detect_r_peaks(lead) if peaks is None else _validate_peaks(peaks)
    if len(peaks) < 2:
        return np.array([], dtype=np.float64)
    diffs = np.diff(peaks).astype(np.float64)
    return diffs * (1000.0 / lead.sampling_rate)


def instantaneous_heart_rate(
    lead: LeadLike, peaks: NDArray[np.intp] | None = None, *, fs: int | None = None
) -> NDArray[np.float64]:
    """Compute instantaneous heart rate at each beat in bpm.

    Parameters
    ----------
    lead : Lead | NDArray[np.float64]
        Input ECG lead or raw signal array.
    peaks : NDArray | None
        Pre-detected R-peak indices.  Detected automatically if ``None``.
    fs : int | None
        Sample rate in Hz.  Required when *lead* is a numpy array.
    """
    lead = ensure_lead(lead, fs=fs)
    rr = rr_intervals(lead, peaks)
    if len(rr) == 0:
        return np.array([], dtype=np.float64)
    return 60_000.0 / rr


def _validate_signal(samples, fs, min_duration_s: float = _MIN_DURATION_S) -> NDArray[np.float64]:
    """Return *samples* as a 1-D float array or raise a clear ValueError."""
    if fs is None or fs <= 0:
        raise ValueError(f"sampling rate must be positive, got {fs}")
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D signal, got shape {x.shape}")
    min_len = int(np.ceil(min_duration_s * fs))
    if x.size < min_len:
        raise ValueError(
            f"signal too short: {x.size} samples at {fs} Hz, "
            f"need at least {min_duration_s:g} s ({min_len} samples)"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("signal contains NaN or infinite values")
    return x


def _validate_peaks(peaks) -> NDArray[np.intp]:
    p = np.asarray(peaks)
    if p.ndim != 1:
        raise ValueError(f"peaks must be a 1-D array, got shape {p.shape}")
    if p.size == 0:
        return p.astype(np.intp)
    if not np.issubdtype(p.dtype, np.integer):
        if not np.all(np.isfinite(p)) or not np.all(p == np.round(p)):
            raise ValueError("peaks must be integer sample indices")
    p = p.astype(np.intp)
    if p[0] < 0 or np.any(np.diff(p) <= 0):
        raise ValueError("peaks must be non-negative and strictly increasing")
    return p


def _pan_tompkins(signal: NDArray[np.float64], fs: int) -> NDArray[np.intp]:
    """Pan-Tompkins QRS detection algorithm.

    Bandpass 5--15 Hz, derivative, squaring, 150 ms moving-window
    integration, then dual adaptive thresholds with a learning phase,
    searchback and T-wave discrimination, and polarity-robust refinement.
    All time constants are in seconds so the detector behaves the same at
    any sample rate.
    """
    sig_mod = require_scipy("signal")

    high = min(15.0, 0.45 * fs)
    low = min(5.0, high / 2.0)
    sos = sig_mod.butter(2, [low, high], btype="band", fs=fs, output="sos")
    filtered = sig_mod.sosfiltfilt(sos, signal)

    # A flat or nearly flat signal has no QRS energy: thresholds relative to
    # its rounding noise would otherwise invent beats.
    scale = float(np.ptp(signal))
    if scale == 0 or float(np.sqrt(np.mean(filtered ** 2))) <= _PT_MIN_ENERGY * scale:
        return np.array([], dtype=np.intp)

    derivative = np.gradient(filtered) * fs
    squared = derivative ** 2

    win_len = max(1, int(round(0.150 * fs)))
    integrated = np.convolve(squared, np.ones(win_len) / win_len, mode="same")

    # Absolute floor far above float rounding noise and far below any QRS.
    abs_floor = (_PT_MIN_ENERGY * scale * fs) ** 2
    peaks = _adaptive_threshold(integrated, np.abs(derivative), fs, abs_floor)
    return _refine(signal, fs, peaks)


def _adaptive_threshold(
    integrated: NDArray[np.float64],
    slope: NDArray[np.float64],
    fs: int,
    abs_floor: float = 0.0,
) -> list[int]:
    """Dual-threshold peak classification on the integrated signal.

    Thresholds start from robust statistics of 2 s windows (median of the
    window maxima and means), so a single artefact cannot lock the
    detector.  When no beat is found for 1.66 times the running RR
    interval, a searchback with half the threshold runs; if it finds
    nothing, the signal level estimate is halved so the detector recovers
    after a drop in amplitude.  Searchback applies the same T-wave test as
    normal detection.  No beat is accepted below 1 % of the median 2 s
    window maximum or below an absolute floor tied to the signal range;
    such peaks only update the noise level.
    """
    sig_mod = require_scipy("signal")
    refractory = int(round(_REFRACTORY_S * fs))
    t_wave_window = int(round(0.360 * fs))
    half_slope_win = max(1, int(round(_REFINE_HALF_WIDTH_S * fs)))

    learn = int(round(2.0 * fs))
    n_win = max(1, len(integrated) // learn)
    windows = integrated[: n_win * learn].reshape(n_win, -1) if len(integrated) >= learn else integrated[None, :]
    win_max = windows.max(axis=1)
    # No beat is accepted below the floor, so flat or padded stretches cannot
    # produce beats and the searchback decay cannot reach their rounding
    # noise.  Peaks under the floor still update the noise level.
    floor = max(abs_floor, _PT_CANDIDATE_FLOOR * float(np.median(win_max)))
    active = win_max[win_max > floor]
    if floor <= 0 or len(active) == 0:
        return []

    cand, _ = sig_mod.find_peaks(integrated, distance=max(1, refractory))
    if len(cand) == 0:
        return []
    heights = integrated[cand]

    # Levels start from the windows that contain signal (padding excluded).
    spki = float(np.median(active))
    npki = 0.5 * float(np.median(windows.mean(axis=1)[win_max > floor]))
    # Peak heights are capped when updating the signal level so a burst of
    # artefacts cannot raise it more than 4 times above the initial level.
    cap = 4.0 * spki

    def thresholds() -> tuple[float, float]:
        t1 = npki + 0.25 * (spki - npki)
        return t1, 0.5 * t1

    def max_slope(i: int) -> float:
        return float(slope[max(0, i - half_slope_win): i + half_slope_win + 1].max())

    peaks: list[int] = []
    peak_slopes: list[float] = []
    skipped: list[tuple[int, float]] = []
    rr_hist: list[int] = []
    last_decay = 0

    def is_t_wave(i: int, s: float) -> bool:
        # Within 360 ms of the last beat with under half its slope.
        return bool(peaks) and i - peaks[-1] < t_wave_window and s < 0.5 * peak_slopes[-1]

    for c, h in zip(cand.tolist(), heights.tolist()):
        thr1, thr2 = thresholds()
        rr_avg = float(np.median(rr_hist[-8:])) if rr_hist else float(fs)
        ref_pos = peaks[-1] if peaks else 0
        if c - max(ref_pos, last_decay) > 1.66 * rr_avg:
            lo = ref_pos + refractory if peaks else 0
            # Search back with half the threshold; if nothing is found, halve
            # the signal level once and search again before giving up.
            for _attempt in range(2):
                pool = [
                    (i, v) for i, v in skipped
                    if lo <= i <= c - refractory and v > max(thr2, floor)
                    and not is_t_wave(i, max_slope(i))
                ]
                if pool:
                    i_best, v_best = max(pool, key=lambda x: x[1])
                    if peaks:
                        rr_hist.append(i_best - peaks[-1])
                    peaks.append(i_best)
                    peak_slopes.append(max_slope(i_best))
                    spki = 0.75 * spki + 0.25 * min(v_best, cap)
                    skipped = [(i, v) for i, v in skipped if i > i_best]
                    thr1, thr2 = thresholds()
                    break
                spki = max(0.5 * spki, npki, floor)
                last_decay = c
                thr1, thr2 = thresholds()

        if h > max(thr1, floor) and (not peaks or c - peaks[-1] > refractory):
            s = max_slope(c)
            if is_t_wave(c, s):
                npki = 0.875 * npki + 0.125 * h
                skipped.append((c, h))
                continue
            if peaks:
                rr_hist.append(c - peaks[-1])
            peaks.append(c)
            peak_slopes.append(s)
            spki = 0.875 * spki + 0.125 * min(h, cap)
            skipped = []
        else:
            npki = 0.875 * npki + 0.125 * h
            skipped.append((c, h))

    return peaks


def _refine(
    signal: NDArray[np.float64], fs: int, candidates
) -> NDArray[np.intp]:
    """Move each candidate to the dominant QRS deflection within +/-75 ms.

    The lead polarity is decided once from all candidates (median positive
    versus median negative excursion of a 0.5--40 Hz filtered copy), so
    negative-QRS leads are located on their main negative deflection.
    Candidates that merge within the 200 ms refractory period are reduced
    to the larger one.
    """
    if len(candidates) == 0:
        return np.array([], dtype=np.intp)
    sig_mod = require_scipy("signal")
    high = min(40.0, 0.45 * fs)
    sos = sig_mod.butter(2, [0.5, high], btype="band", fs=fs, output="sos")
    ref = sig_mod.sosfiltfilt(sos, signal)

    half = max(1, int(round(_REFINE_HALF_WIDTH_S * fs)))
    n = len(ref)
    bounds = [(max(0, int(c) - half), min(n, int(c) + half + 1)) for c in candidates]
    pos = np.array([ref[lo:hi].max() for lo, hi in bounds])
    neg = np.array([-ref[lo:hi].min() for lo, hi in bounds])
    polarity = -1.0 if np.median(neg) > np.median(pos) else 1.0

    refined = sorted({int(lo + np.argmax(polarity * ref[lo:hi])) for lo, hi in bounds})
    refractory = int(round(_REFRACTORY_S * fs))
    kept: list[int] = []
    for p in refined:
        if kept and p - kept[-1] <= refractory:
            if polarity * ref[p] > polarity * ref[kept[-1]]:
                kept[-1] = p
            continue
        kept.append(p)
    return np.array(kept, dtype=np.intp)


def _shannon_energy(
    signal: NDArray[np.float64],
    fs: int,
    ransac_window_sec: float = 5.0,
    lowfreq: float = 35.0,
    highfreq: float = 43.0,
) -> NDArray[np.intp]:
    """Shannon-energy-envelope R-peak detector.

    Bandpass 35--43 Hz to isolate QRS energy, derivative power,
    windowed adaptive thresholding (5 s windows, 3-window median), Shannon energy transform,
    Gaussian-smoothed envelope, zero-crossing peak detection and
    polarity-robust refinement.
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

    # Per-window threshold and scale, smoothed with a 3-window median so a
    # single artefact window is ignored while slow amplitude changes are
    # followed.
    win_samples = max(1, int(ransac_window_sec * fs))
    n_windows = max(1, len(decg_power) // win_samples)
    edges = np.linspace(0, len(decg_power), n_windows + 1).astype(int)
    thr_w = np.array([0.5 * decg_power[a:b].std() for a, b in zip(edges[:-1], edges[1:])])
    max_w = np.array([decg_power[a:b].max() for a, b in zip(edges[:-1], edges[1:])])
    if n_windows >= 3:
        thr_w = ndimage.median_filter(thr_w, size=3, mode="mirror")
        max_w = ndimage.median_filter(max_w, size=3, mode="mirror")
    if not np.any(max_w > 0):
        return np.array([], dtype=np.intp)
    max_w = np.where(max_w > 0, max_w, max_w[max_w > 0].min())
    counts = np.diff(edges)
    threshold = np.repeat(thr_w, counts)
    max_power = np.repeat(max_w, counts)

    decg_power[decg_power < threshold] = 0.0
    decg_power /= max_power
    np.clip(decg_power, 0.0, 1.0, out=decg_power)

    sq = decg_power ** 2
    safe_sq = np.where(sq > 0, sq, 1e-20)
    shannon = -sq * np.log(safe_sq)
    shannon[~np.isfinite(shannon)] = 0.0

    mean_win = max(1, int(fs * 0.125) + 1)
    lp_energy = ndimage.uniform_filter1d(shannon, size=mean_win, mode="constant")
    lp_energy = ndimage.gaussian_filter1d(lp_energy, sigma=fs / 8.0)

    lp_diff = np.diff(lp_energy)
    zero_crossings = np.flatnonzero((lp_diff[:-1] > 0) & (lp_diff[1:] <= 0)) + 1

    # Drop envelope maxima far below the typical beat level of their window
    # (filter edge transients and low-level noise ripples).
    env_w = np.array([lp_energy[a:b].max() for a, b in zip(edges[:-1], edges[1:])])
    if n_windows >= 3:
        env_w = ndimage.median_filter(env_w, size=3, mode="mirror")
    env_level = np.repeat(env_w, counts)
    zero_crossings = zero_crossings[
        lp_energy[zero_crossings] > _ENVELOPE_MIN_FRACTION * env_level[zero_crossings]
    ]
    return _refine(signal, fs, zero_crossings)
