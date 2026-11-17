"""Per-m-group relative-binning input (ET BNS with higher-order modes, Stage D).

The network input has one summary per m group of modes and per bin b (G groups, k = 0, 1):
    A^m_bk = S(d, w_m h_ref^m u^k),   S(a, c) = sum_(f in b) 4/T a conj(c) / S(f),
with h_ref^m = f^(-7/6) exp(i phi_fid,m(f; Mc~)) exp(-2 pi i f t~) (`FiducialPhase(mode=...)`)
and w_m(f) = mode_time_taper(tau_m(f), T, alpha), tau_m the fiducial's time to merger of mode m
(each mode only where it is inside the segment; option (a), Stage D test H4d). The sum runs
over the grid points k / T inside the bin.

With the linear model of each group, r^m = h^m / h_ref^m (untapered reference) = r0 + r1 u
per bin, the signal part of A is A = Mp q, and the noise of A has covariance 2 N
(Re and Im each N), with the 2G x 2G matrices per bin
    Mp[(m,k),(m',k')] = S(h_ref^m' u^k', w_m h_ref^m u^k)      (taper on the receiving side)
    N [(m,k),(m',k')] = S(w_m' h_ref^m' u^k', w_m h_ref^m u^k) (taper on both sides).
The time factor exp(-2 pi i f t~) cancels in both: they depend on Mc~ only.

Whitening (Stage D tests H4c, H4h): L L^H = P (N + eps s I) P^T, s = tr N / 2G, P puts the
(2,2) group first; W = L^-1 P. Training input x = K q + z with K = W Mp and z white
(Re, Im ~ N(0, 1)); data input x = W A + sqrt(eps s) W z' (top-up noise: the data noise then
has exactly the training distribution). The regularization keeps about 1 - eps / 2 of the
information (0.995 at eps 1e-2), equally for the (2,2) and the HOM part (test H4m).

Evaluation of N and Mp (test H4e): one mlgw_bns call at the bin edges per Mc~ (phases, slopes
dphi/df = 2 pi tau, continued above the band end as FiducialPhase); inside a bin the phase of
each group is a cubic Hermite interpolant, the taper is linear in tau; the integral over the
grid cells of the bin, [(k_first - 1/2) / T, (k_last + 1/2) / T], uses n Gauss-Legendre nodes
for the slow part and closed-form moments of u^p exp(i beta u) for the beat between groups,
with the beat time wrapped into [-T/2, T/2] (times on the 1/T grid are defined modulo T).
Accuracy against the exact grid sums: input mismatch median 6e-4, max 0.015 (12 draws,
eps 1e-2, 8 nodes; the max came from the band-end phase jump, removed by the continuation).

Lookup (test H4j): K is tabulated on Mc~ = k delta (complex64). Training and inference round
the chirp-mass proxy to this grid, so the table is exact by construction.
"""

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

LOOKUP_VERSION = 1


def mode_group_order(modes):
    """Indices of the groups in whitening order: the (2,2) group first, then the others."""
    modes = [tuple(int(x) for x in mode) for mode in modes]
    if (2, 2) not in modes:
        raise ValueError(f"The modes {modes} have no (2, 2) group.")
    first = modes.index((2, 2))
    return [first] + [g for g in range(len(modes)) if g != first]


def round_to_grid(chirp_mass, delta):
    """The chirp-mass proxy on the lookup grid k delta (round half to even, as np.round)."""
    return np.round(np.asarray(chirp_mass, dtype=np.float64) / delta) * delta


def _hermite(y0, y1, d0, d1, h, s):
    """Cubic Hermite on [0, 1] (s) from values y and slopes d (per unit f), width h."""
    s2, s3 = s * s, s * s * s
    return ((2 * s3 - 3 * s2 + 1) * y0 + (s3 - 2 * s2 + s) * h * d0
            + (-2 * s3 + 3 * s2) * y1 + (s3 - s2) * h * d1)


def _hermite_slope(y0, y1, d0, d1, h, s):
    s2 = s * s
    return ((6 * s2 - 6 * s) * y0 / h + (3 * s2 - 4 * s + 1) * d0
            + (-6 * s2 + 6 * s) * y1 / h + (3 * s2 - 2 * s) * d1)


def _moments(beta, P):
    """mu_p(beta) = int_(-1/2)^(1/2) u^p exp(i beta u) du, p = 0..P; beta: (B,)."""
    mu = np.zeros((P + 1,) + beta.shape, complex)
    small = np.abs(beta) < 8.0
    if small.any():
        b = beta[small]
        term = np.ones_like(b, dtype=complex)
        for k in range(60):  # series in (i beta)^k / k!
            if k > 0:
                term = term * (1j * b) / k
            for p in range(P + 1):
                q = p + k + 1
                integral = (1 - (-1) ** q) / (q * 2.0 ** q)  # int u^(p + k)
                if integral != 0:
                    mu[p, small] += term * integral
    big = ~small
    if big.any():
        b = beta[big]
        ep, em = np.exp(0.5j * b), np.exp(-0.5j * b)
        mu[0, big] = (ep - em) / (1j * b)
        for p in range(1, P + 1):
            mu[p, big] = ((0.5 ** p) * ep - ((-0.5) ** p) * em) / (1j * b) - p / (1j * b) * mu[p - 1, big]
    return mu


class ModeGroupTables:
    """N, Mp, W and K per bin at one chirp-mass proxy, for one detector PSD."""

    def __init__(self, fiducial_phases, psd, segment_duration, taper_fraction=0.05,
                 eps=1e-2, n_nodes=8):
        """
        Parameters
        ----------
        fiducial_phases : list of FiducialPhase
            One per group, each with its mode, all on the same bin edges (group order =
            the order of the dataset's mode_list).
        psd : callable
            One-sided PSD at an array of frequencies [1/Hz] (e.g.
            ``ifo.power_spectral_density.get_power_spectral_density_array``).
        segment_duration : float
            T [s]: the grid spacing 1/T and the mode taper (dataset
            ``mode_max_time_to_merger``).
        taper_fraction : float
            alpha of `mode_time_taper` (dataset ``mode_taper_fraction``).
        eps : float
            Regularization of the joint whitening (eps s I).
        n_nodes : int
            Gauss-Legendre nodes per bin for the slow part (8: test H4h; 4 failed).
        """
        self.fid = list(fiducial_phases)
        self.modes = [f.mode for f in self.fid]
        if any(m is None for m in self.modes):
            raise ValueError("Every FiducialPhase needs a mode.")
        self.order = mode_group_order(self.modes)
        edges = np.asarray(self.fid[0].f, dtype=np.float64)
        for f in self.fid[1:]:
            if not np.array_equal(f.f, edges):
                raise ValueError("The FiducialPhase objects have different frequencies.")
        self.edges = edges
        self.G = len(self.fid)
        self.T = float(segment_duration)
        self.alpha = float(taper_fraction)
        self.eps = float(eps)
        self.n = int(n_nodes)
        T = self.T
        lo, hi = edges[:-1], edges[1:]
        self.width = hi - lo
        self.fcen = 0.5 * (lo + hi)
        # grid points k / T inside [lo, hi) (np.searchsorted side "right"), their cells
        k_first, k_last = np.ceil(lo * T - 1e-9), np.ceil(hi * T - 1e-9) - 1
        if np.any(k_last < k_first):
            raise ValueError("A bin contains no grid point k / T.")
        a, c = (k_first - 0.5) / T, (k_last + 0.5) / T
        self.span, self.mid = c - a, 0.5 * (a + c)
        x, _ = np.polynomial.legendre.leggauss(self.n)
        self.un = 0.5 * x
        self.f_nodes = self.mid[:, None] + self.span[:, None] * self.un[None, :]  # (B, n)
        p = np.asarray(psd(self.f_nodes.ravel()), dtype=np.float64).reshape(self.f_nodes.shape)
        self.weight = 4.0 * self.span[:, None] / p * self.f_nodes ** (-7 / 3)
        self.u_nodes = (self.f_nodes - self.fcen[:, None]) / self.width[:, None]
        self.s_nodes = (self.f_nodes - lo[:, None]) / self.width[:, None]
        self.s_mid = (self.mid - lo) / self.width
        V = np.vander(self.un, self.n, increasing=True)
        self.lagrange = np.linalg.inv(V).T  # L_q(u) = sum_p lagrange[q, p] u^p
        P = np.zeros((2 * self.G, 2 * self.G))
        rows = [2 * g + k for g in self.order for k in range(2)]
        P[np.arange(2 * self.G), rows] = 1.0
        self.P = P

    @property
    def num_bins(self):
        return len(self.edges) - 1

    def phases(self, chirp_mass_proxy):
        """[(phi, dphi/df, tau) at the edges] per group (FiducialPhase.phase_slope_tau)."""
        return [f.phase_slope_tau(chirp_mass_proxy) for f in self.fid]

    def tables(self, chirp_mass_proxy, pst=None):
        """(N, Mp), each (B, 2G, 2G) complex128, at one chirp-mass proxy (pst: the
        output of phases(), if already computed)."""
        from dingo.gw.waveform_generator import mode_time_taper

        if pst is None:
            pst = self.phases(chirp_mass_proxy)
        ph = np.array([x[0] for x in pst])  # (G, B + 1)
        d = np.array([x[1] for x in pst])
        tau_e = np.array([x[2] for x in pst])
        h, s = self.width, self.s_nodes
        phn = _hermite(ph[:, :-1, None], ph[:, 1:, None], d[:, :-1, None], d[:, 1:, None],
                       h[None, :, None], s[None])  # (G, B, n)
        phm = _hermite(ph[:, :-1], ph[:, 1:], d[:, :-1], d[:, 1:], h[None], self.s_mid[None])
        slm = _hermite_slope(ph[:, :-1], ph[:, 1:], d[:, :-1], d[:, 1:], h[None], self.s_mid[None])
        tau = tau_e[:, :-1, None] * (1 - s[None]) + tau_e[:, 1:, None] * s[None]
        tap = mode_time_taper(tau, self.T, self.alpha)  # (G, B, n)
        B, n, G, T = self.num_bins, self.n, self.G, self.T
        N3 = np.zeros((G, G, 3, B), complex)
        M3 = np.zeros((G, G, 3, B), complex)
        for m in range(G):
            for mp in range(G):
                slope = slm[mp] - slm[m]
                kw = np.round(-slope / (2 * np.pi) / T)  # beat time wrapped into [-T/2, T/2]
                beta = (slope + 2 * np.pi * kw * T) * self.span
                dphi_n = phn[mp] - phn[m] + 2 * np.pi * (kw[:, None] * T) * self.f_nodes
                dphi_c = phm[mp] - phm[m] + 2 * np.pi * kw * T * self.mid
                base = self.weight * np.exp(
                    1j * (dphi_n - dphi_c[:, None] - beta[:, None] * self.un[None, :]))
                omega = np.einsum("qp,pb->bq", self.lagrange, _moments(beta, n - 1))
                ph0 = np.exp(1j * dphi_c)
                for j in range(3):
                    g = base * self.u_nodes ** j * tap[m]
                    M3[m, mp, j] = ph0 * np.sum(omega * g, axis=1)
                    N3[m, mp, j] = ph0 * np.sum(omega * g * tap[mp], axis=1)
        N = np.zeros((B, 2 * G, 2 * G), complex)
        Mp = np.zeros_like(N)
        for m in range(G):
            for mp in range(G):
                for k in range(2):
                    for kp in range(2):
                        N[:, 2 * m + k, 2 * mp + kp] = N3[m, mp, k + kp]
                        Mp[:, 2 * m + k, 2 * mp + kp] = M3[m, mp, k + kp]
        N = 0.5 * (N + np.conj(np.swapaxes(N, 1, 2)))
        return N, Mp

    def whitening(self, chirp_mass_proxy, pst=None):
        """(W, K, sqrt(eps s)) at one chirp-mass proxy: W, K (B, 2G, 2G) complex128,
        sqrt(eps s) (B,), the top-up noise scale of the data input."""
        N, Mp = self.tables(chirp_mass_proxy, pst)
        s = np.real(np.trace(N, axis1=1, axis2=2)) / (2 * self.G)
        R = N + self.eps * s[:, None, None] * np.eye(2 * self.G)
        L = np.linalg.cholesky(self.P @ R @ self.P.T)
        W = np.linalg.solve(L, np.broadcast_to(self.P, R.shape).astype(complex))
        return W, W @ Mp, np.sqrt(self.eps * s)

    def data_input(self, A, chirp_mass_proxy, rng=None):
        """Network input from event summaries A (..., B, 2G) [(m, k) order of the groups]:
        x = W A + sqrt(eps s) W z' (top-up noise; rng None: no top-up, for tests)."""
        W, _, scale = self.whitening(chirp_mass_proxy)
        x = np.einsum("bij,...bj->...bi", W, A)
        if rng is not None:
            z = (rng.standard_normal(np.shape(A)) + 1j * rng.standard_normal(np.shape(A)))
            x = x + np.einsum("bij,...bj->...bi", W, scale[:, None] * z)
        return x

    def settings(self):
        """What the tables depend on (stored with a lookup table)."""
        return dict(
            version=LOOKUP_VERSION, modes=[list(m) for m in self.modes],
            segment_duration=self.T, taper_fraction=self.alpha, eps=self.eps,
            n_nodes=self.n, num_bins=self.num_bins,
            edges_sha1=hashlib.sha1(self.edges.tobytes()).hexdigest(),
            weight_sha1=hashlib.sha1(np.ascontiguousarray(self.weight).tobytes()).hexdigest(),
        )


_BUILD = {}


def _build_rows(job):
    paths, rows, mcs = job
    tab = _BUILD["tables"]
    k_out, phi_out, g_out = (np.load(p, mmap_mode="r+") for p in paths)
    i22 = tab.order[0]
    for i, mc in zip(rows, mcs):
        pst = tab.phases(mc)
        k_out[i] = tab.whitening(mc, pst)[1].astype(np.complex64)
        phi_out[i] = np.array([x[0] for x in pst])
        g_out[i] = tab.fid[i22](mc)[1]  # (2,2) fiducial group delay at f_star (time proxy)
    for a in (k_out, phi_out, g_out):
        a.flush()
    return len(rows)


LOOKUP_FILES = {"K": "K.npy", "phi": "phi.npy", "g22": "g22.npy"}


def build_lookup(tables, chirp_mass_min, chirp_mass_max, delta, path, num_processes=1,
                 chunk=64, print_output=True):
    """Tabulate, on Mc~ = k delta, k = floor(min / delta) .. ceil(max / delta):
    K (complex64, (n, B, 2G, 2G)), the continued fiducial phases of each group at the
    edges (float64, (n, G, B + 1); as FiducialPhase.__call__, not wrapped) and the (2,2)
    fiducial group delay at f_star (float64, (n,)).

    Writes the directory `path`: K.npy, phi.npy, g22.npy (memory-mappable) and
    meta.json (grid and `tables.settings()`). Restart-safe only as a whole.
    """
    import multiprocessing
    import time

    k0 = int(np.floor(chirp_mass_min / delta))
    k1 = int(np.ceil(chirp_mass_max / delta))
    ks = np.arange(k0, k1 + 1)
    n2, G, B = 2 * tables.G, tables.G, tables.num_bins
    os.makedirs(path, exist_ok=True)
    tmp = [os.path.join(path, f + ".tmp.npy") for f in LOOKUP_FILES.values()]
    for p, (dt, shape) in zip(tmp, [(np.complex64, (len(ks), B, n2, n2)),
                                    (np.float64, (len(ks), G, B + 1)),
                                    (np.float64, (len(ks),))]):
        arr = np.lib.format.open_memmap(p, mode="w+", dtype=dt, shape=shape)
        del arr
    _BUILD["tables"] = tables
    jobs = [(tmp, list(range(i, min(i + chunk, len(ks)))), list(ks[i:i + chunk] * delta))
            for i in range(0, len(ks), chunk)]
    t0, done = time.time(), 0
    with ProcessPoolExecutor(num_processes, mp_context=multiprocessing.get_context("fork")) as ex:
        for n in ex.map(_build_rows, jobs):
            done += n
            if print_output and (done % (20 * chunk) < chunk or done == len(ks)):
                print(f"lookup: {done}/{len(ks)} nodes, {time.time() - t0:.0f} s", flush=True)
    _BUILD.clear()
    meta = dict(tables.settings(), delta=float(delta), k_min=k0, num_nodes=len(ks),
                chirp_mass_min=float(k0 * delta), chirp_mass_max=float(k1 * delta),
                f_star=float(tables.fid[tables.order[0]].f_star))
    for p, f in zip(tmp, LOOKUP_FILES.values()):
        os.replace(p, os.path.join(path, f))
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


class ModeGroupLookup:
    """Reader of a lookup directory written by `build_lookup` (memory-mapped, opened lazily:
    one memory map per data-loader worker)."""

    def __init__(self, path, tables_settings=None):
        """tables_settings: if given, must equal the stored settings (same edges, PSD
        weights, modes, segment, taper, eps, nodes)."""
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        if tables_settings is not None:
            stored = {k: self.meta.get(k) for k in tables_settings}
            if stored != tables_settings:
                diff = {k: (stored[k], v) for k, v in tables_settings.items() if stored[k] != v}
                raise ValueError(f"Lookup table {path} does not match the setup: {diff}")
        self.path = path
        self.delta = self.meta["delta"]
        self.k_min = self.meta["k_min"]
        self.num_nodes = self.meta["num_nodes"]
        self._arrays = {}

    def __getstate__(self):  # do not pickle memory maps into data-loader workers
        d = self.__dict__.copy()
        d["_arrays"] = {}
        return d

    def array(self, name):
        if name not in self._arrays:
            self._arrays[name] = np.load(os.path.join(self.path, LOOKUP_FILES[name]), mmap_mode="r")
        return self._arrays[name]

    def index(self, chirp_mass_proxy):
        """Grid index of a proxy that is already on the grid (round_to_grid)."""
        k = np.round(np.asarray(chirp_mass_proxy, dtype=np.float64) / self.delta).astype(np.int64)
        i = k - self.k_min
        if np.any(i < 0) or np.any(i >= self.num_nodes):
            raise ValueError(
                f"Chirp-mass proxy outside the lookup table "
                f"[{self.meta['chirp_mass_min']}, {self.meta['chirp_mass_max']}].")
        return i

    def K(self, chirp_mass_proxy):
        """K (complex64) for one proxy (B, 2G, 2G) or a batch (n, B, 2G, 2G)."""
        return np.asarray(self.array("K")[self.index(chirp_mass_proxy)])

    def phases_at_node(self, chirp_mass_proxy):
        """(phi (G, B + 1), (2,2) group delay) at grid proxies (exact: computed there)."""
        i = self.index(chirp_mass_proxy)
        return np.asarray(self.array("phi")[i]), np.asarray(self.array("g22")[i])

    def phases_interpolated(self, chirp_mass):
        """(phi, (2,2) group delay) at any chirp mass inside the table: 4-point Lagrange
        interpolation in Mc between the nodes (for the true chirp mass in training)."""
        x = np.asarray(chirp_mass, dtype=np.float64) / self.delta - self.k_min
        i1 = np.floor(x).astype(np.int64)
        if np.any(i1 < 1) or np.any(i1 + 2 >= self.num_nodes):
            raise ValueError(
                f"Chirp mass outside the interpolation range of the lookup table "
                f"[{self.meta['chirp_mass_min']}, {self.meta['chirp_mass_max']}] (2 nodes margin).")
        t = x - i1
        w = [-t * (t - 1) * (t - 2) / 6, (t + 1) * (t - 1) * (t - 2) / 2,
             -(t + 1) * t * (t - 2) / 2, (t + 1) * t * (t - 1) / 6]  # nodes i1 - 1 .. i1 + 2
        phi_all, g_all = self.array("phi"), self.array("g22")
        phi = sum(np.multiply.outer(wj, np.ones(phi_all.shape[1:])) * np.asarray(phi_all[i1 + j - 1])
                  for j, wj in enumerate(w))
        g = sum(wj * np.asarray(g_all[i1 + j - 1]) for j, wj in enumerate(w))
        return phi, g


def tables_for_dataset(domain, waveform_generator_settings, ifo, eps=1e-2, n_nodes=8):
    """ModeGroupTables of a per-m dataset (RelativeBinningDomain, mlgw_bns,
    polarizations_per_m) for one detector: the dataset's modes, segment and taper."""
    from dingo.gw.transforms.relative_binning_transforms import FiducialPhase
    from dingo.gw.waveform_generator import WaveformGenerator

    wg = waveform_generator_settings
    modes = wg.get("mode_list")
    T = wg.get("mode_max_time_to_merger")
    if modes is None or T is None:
        raise ValueError("The per-m tables need mode_list and mode_max_time_to_merger.")
    edges = np.asarray(domain(), dtype=np.float64)
    f_star = domain.settings.get("f_star", 100.0) if domain.settings else 100.0
    fids = [FiducialPhase(WaveformGenerator(wg["approximant"], domain, wg["f_ref"]), edges,
                          f_star=f_star, mode=tuple(md)) for md in modes]
    return ModeGroupTables(fids, ifo.power_spectral_density.get_power_spectral_density_array,
                           T, wg.get("mode_taper_fraction", 0.05), eps=eps, n_nodes=n_nodes)


def load_lookups(spec, domain, waveform_generator_settings, ifo_list):
    """{detector name: ModeGroupLookup} from `spec` (one directory for all detectors, or
    {detector name: directory}); each table is checked against the dataset (edges, modes,
    segment, taper) and the detector PSD (weights at the nodes)."""
    paths = spec if isinstance(spec, dict) else {ifo.name: spec for ifo in ifo_list}
    out = {}
    for ifo in ifo_list:
        if ifo.name not in paths:
            raise KeyError(f"No mode-group lookup table for {ifo.name}.")
        lk = ModeGroupLookup(paths[ifo.name])
        tab = tables_for_dataset(domain, waveform_generator_settings, ifo,
                                 eps=lk.meta["eps"], n_nodes=lk.meta["n_nodes"])
        out[ifo.name] = ModeGroupLookup(paths[ifo.name], tab.settings())
    return out


def main():
    """Build the lookup table of a per-m dataset:
    python -m dingo.gw.relative_binning_mode_tables --dataset D.hdf5 --out DIR
        [--detector ET_1L_IT] [--delta 1e-4] [--margin 5e-4] [--eps 1e-2] [--nodes 8]
        [--num_processes 4]
    The chirp-mass range is the dataset prior's, widened by the margin (kernel, rounding,
    and the 2 nodes of interpolation)."""
    import argparse

    from dingo.gw.domains import build_domain
    from dingo.gw.gwutils import get_interferometer_list

    p = argparse.ArgumentParser(description=main.__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--detector", default=None, help="default: the first detector of the domain")
    p.add_argument("--delta", type=float, default=1e-4)
    p.add_argument("--margin", type=float, default=5e-4)
    p.add_argument("--eps", type=float, default=1e-2)
    p.add_argument("--nodes", type=int, default=8)
    p.add_argument("--num_processes", type=int, default=1)
    a = p.parse_args()
    from dingo.gw.dataset.waveform_dataset import WaveformDataset
    from dingo.gw.prior import build_prior_with_defaults

    settings = WaveformDataset(file_name=a.dataset, leave_waveforms_on_disk=True).settings
    domain = build_domain(settings["domain"])
    detectors = domain.settings["detectors"]
    ifo = get_interferometer_list([a.detector or detectors[0]])[0]
    tab = tables_for_dataset(domain, settings["waveform_generator"], ifo, a.eps, a.nodes)
    prior = build_prior_with_defaults(settings["intrinsic_prior"])["chirp_mass"]
    lo, hi = prior.minimum - a.margin, prior.maximum + a.margin
    print(f"Lookup for {ifo.name}: Mc {lo:.4f}-{hi:.4f}, delta {a.delta}, "
          f"{int(np.ceil((hi - lo) / a.delta)) + 1} nodes, {tab.num_bins} bins, "
          f"{len(tab.modes)} groups.", flush=True)
    meta = build_lookup(tab, lo, hi, a.delta, a.out, num_processes=a.num_processes)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
