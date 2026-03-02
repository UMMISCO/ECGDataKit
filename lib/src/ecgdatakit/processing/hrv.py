"""Heart Rate Variability (HRV) metrics.

- ``time_domain``: Pure numpy (SDNN, RMSSD, pNN50, etc.)
- ``frequency_domain``: Requires scipy (VLF, LF, HF power)
- ``poincare``: Pure numpy (SD1, SD2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ecgdatakit.processing._core import require_scipy


def time_domain(rr_ms: NDArray[np.float64]) -> dict:
    """Compute time-domain HRV metrics from RR intervals.

    Parameters
    ----------
    rr_ms : NDArray
        RR intervals in milliseconds.

    Returns
    -------
    dict
        Keys: ``mean_rr``, ``sdnn``, ``rmssd``, ``sdsd``, ``nn50_count``,
        ``pnn50``, ``nn20_count``, ``pnn20``, ``hr_mean``, ``hr_std``.
    """
    if len(rr_ms) < 2:
        return {
            "mean_rr": float(rr_ms[0]) if len(rr_ms) == 1 else 0.0,
            "sdnn": 0.0,
            "rmssd": 0.0,
            "sdsd": 0.0,
            "nn50_count": 0,
            "pnn50": 0.0,
            "nn20_count": 0,
            "pnn20": 0.0,
            "hr_mean": 60_000.0 / float(rr_ms[0]) if len(rr_ms) == 1 else 0.0,
            "hr_std": 0.0,
        }

    diffs = np.diff(rr_ms)
    abs_diffs = np.abs(diffs)
    hr = 60_000.0 / rr_ms

    nn50 = int(np.sum(abs_diffs > 50))
    nn20 = int(np.sum(abs_diffs > 20))

    return {
        "mean_rr": float(rr_ms.mean()),
        "sdnn": float(rr_ms.std(ddof=1)),
        "rmssd": float(np.sqrt(np.mean(diffs ** 2))),
        "sdsd": float(diffs.std(ddof=1)),
        "nn50_count": nn50,
        "pnn50": 100.0 * nn50 / len(diffs),
        "nn20_count": nn20,
        "pnn20": 100.0 * nn20 / len(diffs),
        "hr_mean": float(hr.mean()),
        "hr_std": float(hr.std(ddof=1)),
    }


def frequency_domain(
    rr_ms: NDArray[np.float64],
    method: str = "welch",
    interp_fs: float = 4.0,
) -> dict:
    """Compute frequency-domain HRV metrics from RR intervals.

    RR intervals are interpolated to a uniform time series at *interp_fs* Hz,
    then the power spectral density is estimated.

    Parameters
    ----------
    rr_ms : NDArray
        RR intervals in milliseconds.
    method : str
        PSD method (``"welch"``).
    interp_fs : float
        Interpolation sampling rate in Hz (default 4.0).

    Returns
    -------
    dict
        Keys: ``vlf_power``, ``lf_power``, ``hf_power``, ``lf_hf_ratio``,
        ``total_power``.
    """
    sig = require_scipy("signal")
    interpolate = require_scipy("interpolate")

    if len(rr_ms) < 4:
        return {
            "vlf_power": 0.0,
            "lf_power": 0.0,
            "hf_power": 0.0,
            "lf_hf_ratio": 0.0,
            "total_power": 0.0,
        }

    rr_s = rr_ms / 1000.0
    t_rr = np.cumsum(rr_s) - rr_s[0]

    t_uniform = np.arange(0, t_rr[-1], 1.0 / interp_fs)
    f_interp = interpolate.interp1d(t_rr, rr_ms, kind="cubic", fill_value="extrapolate")
    rr_uniform = f_interp(t_uniform)
    rr_uniform = rr_uniform - rr_uniform.mean()

    nperseg = min(256, len(rr_uniform))
    freqs, psd = sig.welch(rr_uniform, fs=interp_fs, nperseg=nperseg)

    def _band_power(f_low: float, f_high: float) -> float:
        mask = (freqs >= f_low) & (freqs < f_high)
        if not np.any(mask):
            return 0.0
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        return float(np.sum(psd[mask]) * df)

    vlf = _band_power(0.0, 0.04)
    lf = _band_power(0.04, 0.15)
    hf = _band_power(0.15, 0.40)
    total = _band_power(0.0, 0.40)

    return {
        "vlf_power": vlf,
        "lf_power": lf,
        "hf_power": hf,
        "lf_hf_ratio": lf / hf if hf > 0 else 0.0,
        "total_power": total,
    }


def poincare(rr_ms: NDArray[np.float64]) -> dict:
    """Compute Poincaré plot descriptors (SD1, SD2).

    Parameters
    ----------
    rr_ms : NDArray
        RR intervals in milliseconds.

    Returns
    -------
    dict
        Keys: ``sd1``, ``sd2``, ``sd1_sd2_ratio``.
    """
    if len(rr_ms) < 2:
        return {"sd1": 0.0, "sd2": 0.0, "sd1_sd2_ratio": 0.0}

    x = rr_ms[:-1]
    y = rr_ms[1:]

    sd1 = float(np.std(y - x, ddof=1) / np.sqrt(2))
    sd2 = float(np.std(y + x, ddof=1) / np.sqrt(2))

    return {
        "sd1": sd1,
        "sd2": sd2,
        "sd1_sd2_ratio": sd1 / sd2 if sd2 > 0 else 0.0,
    }
