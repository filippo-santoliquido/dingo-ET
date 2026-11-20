"""Data side of the per-m-group relative-binning input and likelihood (ET BNS with
higher-order modes, Stage D).

For proxies (Mc~, t~), the reference of the group m is (the same as in training,
`dingo.gw.transforms.relative_binning_mode_transforms`)

    h_ref^m(f) = f^(-7/6) exp(i phi_fid,m(f; Mc~)) exp(-2 pi i f (t~ + g22(Mc~))),

with phi_fid,m the continued phase of the mode of the fiducial (`FiducialPhase` with
`mode`) and g22 the (2,2) fiducial group delay at f_star (one time for all groups). Its
taper is w_m(f) = mode_time_taper(tau_m(f; Mc~), T, alpha), tau_m the fiducial's time map.

One pass over the full uniform grid (spacing 1/T, in chunks) gives, per detector and bin b,
with S(a, c) = sum_(f in b) 4/T a conj(c) / S(f) and u = (f - f_c) / (f_hi - f_lo):
- the network summaries   A^m_bk = S(d, w_m h_ref^m u^k)  (the per-m network input,
  `ModeGroupTables.data_input`);
- the likelihood summaries D^m_bk = S(d, h_ref^m u^k)      (untapered references);
- the cross-Gram           G^(m m')_bj = S(h_ref^m', h_ref^m u^j), j = 0, 1, 2
  (exact on the grid; depends on Mc~ and the PSD, not on the data), and the same with
  the tapers on both sides, S(w_m' h_ref^m', w_m h_ref^m u^j) (`gram_tapered`);
- <d|d> (the noise evidence log_Zn = -<d|d>/2).
With r^m = h^m / h_ref^m linear in each bin (q = (r0, r1)), the likelihood is
    (d|h) = sum_(m,b,k) conj(D^m_bk) q^m_bk,   (h|h) = sum q^H Q q,
    Q[(m,k),(m',k')]_b = G^(m m')_(b, k+k')
(`dingo.gw.likelihood_relative_binning.ModeGroupRelativeBinningLikelihood`).
"""

import numpy as np

from dingo.gw.transforms.relative_binning_transforms import FiducialPhase
from dingo.gw.waveform_generator import WaveformGenerator, mode_time_taper


class ModeGroupReferences:
    """References h_ref^m of all groups for given proxies, at the bin edges or at any
    frequencies (chunks of the data grid), and their tapers."""

    def __init__(self, domain, waveform_generator_settings):
        """
        Parameters
        ----------
        domain : RelativeBinningDomain
            The bins (edges) and f_star of the per-m dataset.
        waveform_generator_settings : dict
            Of the per-m dataset (mlgw_bns, mode_list, mode_max_time_to_merger,
            mode_taper_fraction, f_ref).
        """
        wg = dict(waveform_generator_settings)
        if wg.get("approximant") != "mlgw_bns" or not wg.get("mode_list"):
            raise ValueError("Per-m references need approximant mlgw_bns and a mode_list.")
        self.modes = [(int(ell), int(m)) for ell, m in wg["mode_list"]]
        self.groups = [m for _, m in self.modes]
        if len(set(self.groups)) != len(self.groups):
            raise ValueError(f"One mode per m is needed, got {self.modes}.")
        if (2, 2) not in self.modes:
            raise ValueError(f"The modes {self.modes} have no (2, 2) group.")
        self.i22 = self.modes.index((2, 2))
        self.edges = np.asarray(domain(), dtype=np.float64)
        settings = getattr(domain, "settings", None) or {}
        self.f_star = settings.get("f_star", 100.0)
        self.T = wg.get("mode_max_time_to_merger")
        self.alpha = wg.get("mode_taper_fraction", 0.05)
        self.fids = [
            FiducialPhase(
                WaveformGenerator(wg["approximant"], domain, wg["f_ref"]),
                self.edges,
                f_star=self.f_star,
                mode=md,
            )
            for md in self.modes
        ]
        self._cache = {}

    @property
    def G(self):
        return len(self.modes)

    def _at_mc(self, chirp_mass_proxy):
        """(exp(i phi_fid,m) at the edges (G, B + 1), g22) at one Mc~ (cached)."""
        key = float(chirp_mass_proxy)
        if key not in self._cache:
            if len(self._cache) > 8:
                self._cache.clear()
            out = [fid(key) for fid in self.fids]
            self._cache[key] = (np.array([e for e, _ in out]), float(out[self.i22][1]))
        return self._cache[key]

    def time_shift(self, proxies):
        """t~ + g22(Mc~): the time of the reference phase ramp."""
        return proxies["t_star_proxy"] + self._at_mc(proxies["chirp_mass_proxy"])[1]

    def at_edges(self, proxies):
        """Untapered references at the bin edges, (G, B + 1) complex."""
        e, g22 = self._at_mc(proxies["chirp_mass_proxy"])
        f = self.edges
        t = proxies["t_star_proxy"] + g22
        return f ** (-7 / 6) * e * np.exp(-2j * np.pi * f * t)

    def at(self, proxies, frequencies):
        """(references, tapers), each (G, n), at increasing frequencies inside the bins
        (a chunk of the data grid). The tapers are 1 if the dataset has no
        mode_max_time_to_merger."""
        f = np.asarray(frequencies, dtype=np.float64)
        mc = proxies["chirp_mass_proxy"]
        t = self.time_shift(proxies)
        ramp = f ** (-7 / 6) * np.exp(-2j * np.pi * f * t)
        refs = np.empty((self.G, len(f)), dtype=complex)
        tapers = np.ones((self.G, len(f)))
        for g, fid in enumerate(self.fids):
            e, tau = fid.at(mc, f, return_tau=True)
            refs[g] = ramp * e
            if self.T is not None:
                tapers[g] = mode_time_taper(tau, self.T, self.alpha)
        return refs, tapers


def bin_index_and_u(edges, f):
    """Bin index (searchsorted side "right", as the Stage B/C data pass) and
    u = (f - f_c) / (f_hi - f_lo) of each frequency."""
    nb = len(edges) - 1
    b = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, nb - 1)
    u = (f - 0.5 * (edges[:-1] + edges[1:])[b]) / np.diff(edges)[b]
    return b, u


def _binsum(b, x, nb):
    return np.bincount(b, x.real, nb) + 1j * np.bincount(b, x.imag, nb)


class ModeGroupSummaryAccumulator:
    """Sums of one data pass over the full grid, chunk by chunk (see module docstring)."""

    def __init__(self, edges, ifo_names, num_groups, duration):
        self.edges = np.asarray(edges, dtype=np.float64)
        self.nb = len(self.edges) - 1
        self.G = int(num_groups)
        self.T = float(duration)
        self.ifo_names = list(ifo_names)
        z = lambda *s: np.zeros(s, dtype=complex)  # noqa: E731
        self.A = {n: z(self.G, 2, self.nb) for n in self.ifo_names}
        self.D = {n: z(self.G, 2, self.nb) for n in self.ifo_names}
        self.gram = {n: z(self.G, self.G, 3, self.nb) for n in self.ifo_names}
        self.gram_tapered = {n: z(self.G, self.G, 3, self.nb) for n in self.ifo_names}
        self.d_inner_d = 0.0

    def add(self, frequencies, data, psd, refs, tapers, gram=True):
        """Add one chunk.

        Parameters
        ----------
        frequencies : (n,) grid frequencies k / T inside [edges[0], edges[-1]]
        data : {detector: (n,) complex} data (signal + noise) at the frequencies
        psd : {detector: (n,)} one-sided PSD
        refs, tapers : (G, n) from ModeGroupReferences.at
        gram : bool, also accumulate the cross-Gram (it does not depend on the data)
        """
        f = np.asarray(frequencies, dtype=np.float64)
        b, u = bin_index_and_u(self.edges, f)
        powers = [np.ones_like(u), u, u * u]
        for n in self.ifo_names:
            w = 4.0 / self.T / np.asarray(psd[n], dtype=np.float64)
            cr = np.conj(refs) * w  # (G, n)
            d = np.asarray(data[n])
            for g in range(self.G):
                z = d * cr[g]
                zt = z * tapers[g]
                for k in range(2):
                    self.D[n][g, k] += _binsum(b, z * powers[k], self.nb)
                    self.A[n][g, k] += _binsum(b, zt * powers[k], self.nb)
            if gram:
                for g in range(self.G):
                    for gp in range(g, self.G):
                        x = cr[g] * refs[gp]
                        xt = x * (tapers[g] * tapers[gp])
                        for j in range(3):
                            for acc, y in ((self.gram, x), (self.gram_tapered, xt)):
                                s = _binsum(b, y * powers[j], self.nb)
                                acc[n][g, gp, j] += s
                                if gp != g:
                                    acc[n][gp, g, j] += np.conj(s)
            self.d_inner_d += float(np.sum(w * np.abs(d) ** 2))

    def network_summaries(self):
        """{detector: (B, 2G)} network summaries in the (m, k) layout of
        `ModeGroupTables` (column 2 g + k)."""
        return {n: to_table_layout(a) for n, a in self.A.items()}


def to_table_layout(a):
    """(G, 2, B) -> (B, 2G), column 2 g + k."""
    a = np.asarray(a)
    return np.moveaxis(a, -1, 0).reshape(a.shape[-1], -1)
