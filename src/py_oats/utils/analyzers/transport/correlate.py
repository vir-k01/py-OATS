"""FFT-based MSD and cross-MSD for Onsager L_ij."""

from __future__ import annotations

import numpy as np


def _autocorr_fft(x: np.ndarray) -> np.ndarray:
    N = len(x)
    F = np.fft.fft(x, n=2 * N)
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD).real[:N]
    n = N - np.arange(N, dtype=np.float64)
    return res / n


def msd_fft(r: np.ndarray) -> np.ndarray:
    """
    MSD via FFT. r shape (N, 3) or (N,) for single component.
    Returns shape (N,) mean-squared displacement.
    """
    r = np.atleast_2d(r)
    N = r.shape[0]
    D = np.square(r).sum(axis=1)
    D = np.append(D, 0.0)
    S2 = sum(_autocorr_fft(r[:, i]) for i in range(r.shape[1]))
    Q = 2.0 * D.sum()
    S1 = np.empty(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)
    return S1 - 2 * S2


def _cross_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    N = len(x)
    nfft = 2 ** (2 * N - 1).bit_length()
    F1 = np.fft.fft(x, n=nfft)
    F2 = np.fft.fft(y, n=nfft)
    res = np.fft.ifft(F1 * F2.conjugate()).real[:N]
    n = N - np.arange(N, dtype=np.float64)
    return res / n


def msd_fft_cross(r: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Cross "MSD" via FFT. r, k shape (N, 3).
    Returns shape (N,) correlation used for off-diagonal L_ij.
    """
    N = len(r)
    D = (r * k).sum(axis=1)
    D = np.append(D, 0.0)
    S2 = sum(_cross_corr(r[:, i], k[:, i]) for i in range(r.shape[1]))
    S3 = sum(_cross_corr(k[:, i], r[:, i]) for i in range(r.shape[1]))
    Q = 2.0 * D.sum()
    S1 = np.empty(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        S1[m] = Q / (N - m)
    return S1 - S2 - S3


def calc_Lii_self(positions: np.ndarray) -> np.ndarray:
    """positions (T, N, 3). Returns (T,) MSD for self term."""
    T, N, _ = positions.shape
    out = np.zeros(T, dtype=np.float64)
    for i in range(N):
        out += msd_fft(positions[:, i, :])
    return out


def calc_Lii(positions: np.ndarray) -> np.ndarray:
    """positions (T, N, 3). Sum over atoms then MSD. Returns (T,)."""
    r_sum = positions.sum(axis=1)  # (T, 3)
    return msd_fft(r_sum)


def calc_Lij(positions_i: np.ndarray, positions_j: np.ndarray) -> np.ndarray:
    """positions (T, Ni, 3), (T, Nj, 3). Cross MSD. Returns (T,)."""
    r_i = positions_i.sum(axis=1)  # (T, 3)
    r_j = positions_j.sum(axis=1)  # (T, 3)
    return msd_fft_cross(r_i, r_j)
