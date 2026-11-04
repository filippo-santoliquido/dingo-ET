"""Frequency domain of relative-binning bin edges (summary-data input, ET BNS Stage B)."""

from typing import Iterable

import numpy as np

from .base_frequency_domain import BaseFrequencyDomain


class RelativeBinningDomain(BaseFrequencyDomain):
    r"""
    The edges $f_0 < \dots < f_B$ of $B$ relative-binning bins.

    Waveforms are evaluated only at the edges. Their ratio to a reference waveform is
    taken as linear in each bin, which gives the relative-binning summary data
    (Zackay, Dai & Venumadhav 2018) without the full frequency grid. Pointwise
    operations (detector projection, time shifts) work as in any frequency domain.

    The frequencies are kept in float64: at 3 Hz an ET BNS phase is of order 1e5 rad,
    and a float32 frequency would change it by more than 0.01 rad.

    Whitening is not a division by the ASD here: it is a 2 x 2 matrix per bin (see
    `SummarizeAndWhiten`). So `noise_std` is 1.

    The edges and the per-bin Gram matrices are found by the code from the
    hyperparameters in `settings` (`dingo.gw.dataset.relative_binning_bins`, at dataset
    generation) and then stored with the domain.
    """

    def __init__(
        self,
        bin_edges: Iterable[float],
        gram: dict = None,
        settings: dict = None,
    ):
        """
        Parameters
        ----------
        bin_edges : Iterable[float]
            Strictly increasing bin edges [Hz]; at least two.
        gram : dict
            {detector: (B, 3) Gram entries (Bmat_00, Bmat_01, Bmat_11) per bin}.
        settings : dict
            Relative-binning hyperparameters (detectors, f_min, f_max, kernel, f_star,
            bin_selection).
        """
        super().__init__()
        self.gram = None if gram is None else {
            k: np.asarray(v, dtype=np.float64) for k, v in gram.items()
        }
        self.settings = settings
        self._sample_frequencies = np.asarray(bin_edges, dtype=np.float64)
        if self._sample_frequencies.ndim != 1 or len(self._sample_frequencies) < 2:
            raise ValueError("bin_edges must be a 1D array with at least two edges.")
        if np.any(np.diff(self._sample_frequencies) <= 0):
            raise ValueError("bin_edges must be strictly increasing.")
        self._delta_f = np.diff(self._sample_frequencies)

    def update(self, new_settings: dict):
        """Only an update to the same domain is allowed (bins are fixed after training)."""
        if new_settings != self.domain_dict:
            raise NotImplementedError(
                "RelativeBinningDomain cannot be truncated or changed."
            )

    def update_data(self, data: np.ndarray, axis: int = -1, low_value: float = 0.0):
        """Data on the bin edges need no adjustment; only the length is checked."""
        if data.shape[axis] != len(self):
            raise ValueError(
                f"Data length {data.shape[axis]} along axis {axis} does not match the "
                f"{len(self)} bin edges."
            )
        return data

    @property
    def sample_frequencies(self) -> np.ndarray:
        return self._sample_frequencies

    @property
    def bin_edges(self) -> np.ndarray:
        return self._sample_frequencies

    @property
    def num_bins(self) -> int:
        return len(self._sample_frequencies) - 1

    @property
    def frequency_mask(self) -> np.ndarray:
        return np.ones_like(self._sample_frequencies)

    @property
    def frequency_mask_length(self) -> int:
        return len(self._sample_frequencies)

    @property
    def noise_std(self) -> float:
        return 1.0

    @property
    def min_idx(self) -> int:
        return 0

    @property
    def max_idx(self) -> int:
        return len(self._sample_frequencies) - 1

    @property
    def f_min(self) -> float:
        return float(self._sample_frequencies[0])

    @property
    def f_max(self) -> float:
        return float(self._sample_frequencies[-1])

    @property
    def duration(self) -> float:
        raise NotImplementedError("RelativeBinningDomain has no duration.")

    @property
    def sampling_rate(self) -> float:
        return 2.0 * self.f_max

    @property
    def domain_dict(self) -> dict:
        d = {"type": "RelativeBinningDomain", "bin_edges": self._sample_frequencies.tolist()}
        if self.gram is not None:
            d["gram"] = {k: v.tolist() for k, v in self.gram.items()}
        if self.settings is not None:
            d["settings"] = self.settings
        return d
