"""Heart Rate Variability (HRV) metrics.

- ``time_domain``: Pure numpy (SDNN, RMSSD, pNN50, etc.)
- ``frequency_domain``: Requires scipy (VLF, LF, HF power)
- ``poincare``: Pure numpy (SD1, SD2)

Definitions follow the Task Force of the ESC and NASPE (1996), "Heart rate
variability: standards of measurement, physiological interpretation and
clinical use", Circulation 93:1043-1065.  The functions take the RR
intervals as given: ectopic beats and detection errors are not removed, so
pass cleaned NN intervals when the metrics must follow the standard.

Metrics that are undefined for the given number of intervals are ``nan``.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.processing._core import require_scipy

_NAN = float("nan")
_PSD_METHODS = ("welch",)
_LF_MIN_S = 120.0
_HF_MIN_S = 60.0


def _validate_rr(rr_ms) -> NDArray[np.float64]:
    """Return RR intervals as a 1-D float array or raise ValueError."""
    rr = np.asarray(rr_ms, dtype=np.float64)
    if rr.ndim != 1:
        raise ValueError(f"rr_ms must be a 1-D array, got shape {rr.shape}")
    if not np.all(np.isfinite(rr)):
        raise ValueError("rr_ms contains NaN or infinite values")
    if np.any(rr <= 0):
        raise ValueError("rr_ms must contain only positive intervals")
    return rr


def time_domain(rr_ms: NDArray[np.float64]) -> dict:
    """Compute time-domain HRV metrics from RR intervals.

    Parameters
    ----------
    rr_ms : NDArray
        RR intervals in milliseconds (1-D, positive, finite).

    Returns
    -------
    dict
        - ``mean_rr`` (ms): mean interval; ``nan`` if empty.
        - ``sdnn`` (ms): sample standard deviation (ddof=1); needs 2 intervals.
        - ``rmssd`` (ms): root mean square of successive differences; needs 2.
        - ``sdsd`` (ms): standard deviation (ddof=1) of successive
          differences; needs 3.
        - ``nn50_count``, ``nn20_count``: successive differences strictly
          greater than 50 ms / 20 ms.
        - ``pnn50``, ``pnn20`` (%): counts divided by the number of
          successive differences; needs 2 intervals.
        - ``hr_mean`` (bpm): mean of ``60000 / rr``; ``nan`` if empty.
        - ``hr_std`` (bpm): sample standard deviation of ``60000 / rr``;
          needs 2.

    Raises
    ------
    ValueError
        If *rr_ms* is not 1-D or contains NaN, infinite or non-positive values.
    """
    rr = _validate_rr(rr_ms)
    n = len(rr)
    diffs = np.diff(rr)
    abs_diffs = np.abs(diffs)
    hr = 60_000.0 / rr
    nn50 = int(np.sum(abs_diffs > 50))
    nn20 = int(np.sum(abs_diffs > 20))

    return {
        "mean_rr": float(rr.mean()) if n >= 1 else _NAN,
        "sdnn": float(rr.std(ddof=1)) if n >= 2 else _NAN,
        "rmssd": float(np.sqrt(np.mean(diffs ** 2))) if n >= 2 else _NAN,
        "sdsd": float(diffs.std(ddof=1)) if n >= 3 else _NAN,
        "nn50_count": nn50,
        "pnn50": 100.0 * nn50 / len(diffs) if n >= 2 else _NAN,
        "nn20_count": nn20,
        "pnn20": 100.0 * nn20 / len(diffs) if n >= 2 else _NAN,
        "hr_mean": float(hr.mean()) if n >= 1 else _NAN,
        "hr_std": float(hr.std(ddof=1)) if n >= 2 else _NAN,
    }


def frequency_domain(
    rr_ms: NDArray[np.float64],
    method: str = "welch",
    interp_fs: float = 4.0,
) -> dict:
    """Compute frequency-domain HRV metrics from RR intervals.

    RR intervals are placed at their beat times, interpolated (cubic) to a
    uniform series at *interp_fs* Hz, mean-removed, and the power spectral
    density is estimated with Welch's method (segments of up to 256
    samples, 64 s at 4 Hz).  Band powers integrate the PSD over the Task
    Force bands: VLF 0--0.04 Hz, LF 0.04--0.15 Hz, HF 0.15--0.40 Hz,
    total 0--0.40 Hz.

    The Task Force recommends at least 1 min of data for HF and 2 min for
    LF (5 min for short-term analysis).  A :class:`UserWarning` is issued
    when the recording is shorter than 2 min (LF unreliable) or 1 min
    (LF and HF unreliable).  VLF from recordings shorter than 5 min should
    not be interpreted.

    Parameters
    ----------
    rr_ms : NDArray
        RR intervals in milliseconds (1-D, positive, finite).
    method : str
        PSD method.  Only ``"welch"`` is supported.
    interp_fs : float
        Interpolation sampling rate in Hz (default 4.0); must be above
        0.8 Hz so the HF band lies below the Nyquist frequency.

    Returns
    -------
    dict
        ``vlf_power``, ``lf_power``, ``hf_power``, ``total_power`` in
        ms², and ``lf_hf_ratio`` (dimensionless).  All values are ``nan``
        when fewer than 4 intervals are given; ``lf_hf_ratio`` is ``nan``
        when HF power is zero.

    Raises
    ------
    ValueError
        Unknown *method*, *interp_fs* not above 0.8 Hz or not finite, or
        invalid *rr_ms*.
    """
    if method not in _PSD_METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {_PSD_METHODS}")
    if not (np.isfinite(interp_fs) and interp_fs > 0.8):
        raise ValueError(
            f"interp_fs must be a finite rate above 0.8 Hz (twice the HF upper "
            f"edge), got {interp_fs}"
        )
    rr = _validate_rr(rr_ms)
    sig = require_scipy("signal")
    interpolate = require_scipy("interpolate")

    nan_result = {
        "vlf_power": _NAN,
        "lf_power": _NAN,
        "hf_power": _NAN,
        "lf_hf_ratio": _NAN,
        "total_power": _NAN,
    }
    if len(rr) < 4:
        return nan_result

    duration_s = float(rr.sum()) / 1000.0
    if duration_s < _HF_MIN_S:
        warnings.warn(
            f"RR series covers {duration_s:.0f} s; LF and HF power need at "
            f"least {_LF_MIN_S:.0f} s and {_HF_MIN_S:.0f} s (Task Force 1996) "
            "and are unreliable",
            UserWarning,
            stacklevel=2,
        )
    elif duration_s < _LF_MIN_S:
        warnings.warn(
            f"RR series covers {duration_s:.0f} s; LF power needs at least "
            f"{_LF_MIN_S:.0f} s (Task Force 1996) and is unreliable",
            UserWarning,
            stacklevel=2,
        )

    rr_s = rr / 1000.0
    t_rr = np.cumsum(rr_s) - rr_s[0]
    t_uniform = np.arange(0, t_rr[-1], 1.0 / interp_fs)
    if len(t_uniform) < 4:
        return nan_result
    f_interp = interpolate.interp1d(t_rr, rr, kind="cubic")
    rr_uniform = f_interp(t_uniform)
    rr_uniform = rr_uniform - rr_uniform.mean()

    nperseg = min(256, len(rr_uniform))
    freqs, psd = sig.welch(rr_uniform, fs=interp_fs, nperseg=nperseg)
    df = freqs[1] - freqs[0]

    def _band_power(f_low: float, f_high: float) -> float:
        mask = (freqs >= f_low) & (freqs < f_high)
        return float(np.sum(psd[mask]) * df)

    vlf = _band_power(0.0, 0.04)
    lf = _band_power(0.04, 0.15)
    hf = _band_power(0.15, 0.40)
    total = _band_power(0.0, 0.40)

    return {
        "vlf_power": vlf,
        "lf_power": lf,
        "hf_power": hf,
        "lf_hf_ratio": lf / hf if hf > 0 else _NAN,
        "total_power": total,
    }


def poincare(rr_ms: NDArray[np.float64]) -> dict:
    """Compute Poincaré plot descriptors (SD1, SD2).

    ``SD1 = std(RR[n+1] - RR[n]) / sqrt(2)`` and
    ``SD2 = std(RR[n+1] + RR[n]) / sqrt(2)`` (sample standard deviations).

    Parameters
    ----------
    rr_ms : NDArray
        RR intervals in milliseconds (1-D, positive, finite).

    Returns
    -------
    dict
        ``sd1`` and ``sd2`` in ms, ``sd1_sd2_ratio``.  All ``nan`` when
        fewer than 3 intervals are given; the ratio is ``nan`` when SD2 is 0.

    Raises
    ------
    ValueError
        If *rr_ms* is not 1-D or contains NaN, infinite or non-positive values.
    """
    rr = _validate_rr(rr_ms)
    if len(rr) < 3:
        return {"sd1": _NAN, "sd2": _NAN, "sd1_sd2_ratio": _NAN}

    x = rr[:-1]
    y = rr[1:]
    sd1 = float(np.std(y - x, ddof=1) / np.sqrt(2))
    sd2 = float(np.std(y + x, ddof=1) / np.sqrt(2))

    return {
        "sd1": sd1,
        "sd2": sd2,
        "sd1_sd2_ratio": sd1 / sd2 if sd2 > 0 else _NAN,
    }
