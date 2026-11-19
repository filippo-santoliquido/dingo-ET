"""Training transforms of the per-m-group relative-binning input (ET BNS with higher-order
modes, Stage D). See `dingo.gw.relative_binning_mode_tables` for the input definition.

Chain (train_builders): ModeGroupProxyConditioning -> ProjectModeGroupsFrequencyDependent
-> SummarizeWhitenModeGroups -> AddWhiteNoiseComplex.

Conventions (as the Stage C relative-binning transforms, per group):
- the dataset stores each group m divided by exp(i phi_fid,m(Mc)) at the bin edges (keys
  f"h_plus_m{m}", f"h_cross_m{m}"; `DivideByFiducial` with one FiducialPhase per m);
- the reference of group m is h_ref^m = f^(-7/6) exp(i phi_fid,m(f; Mc~)) exp(-2 pi i f (t~ + g)),
  with ONE time for all groups: t~ the t* proxy and g the (2,2) fiducial group delay at f_star
  at Mc~ (each group's own time map is inside its fiducial phase);
- t* = mean detector time - (group delay of the (2,2) group at f_star);
- Mc~ is on the lookup grid (k delta): chirp_mass_proxy = round(Mc + kernel draw).
All phases come from the lookup table: exact at the grid proxy, 4-point interpolation in Mc
at the true chirp mass.
"""

import numpy as np

from dingo.gw.relative_binning_mode_tables import round_to_grid
from dingo.gw.transforms.gnpe_transforms import GNPEBase

CHUNK = 512  # samples per vectorized block (memory: phases and K per block)
# Training data of the per-m input are complex64 (float32 precision: relative error about 1e-7
# of the whitened signal, against unit noise). complex128 batches of 4096 needed about 5 GB per
# data-loader worker and caused an OOM in the pipeline_f3_hom smoke (2026-11-19).
DTYPE = np.complex64


def group_ratios_to_q(r):
    """(r0, r1) per bin and group from r (..., G, B + 1) at the edges -> q (..., B, 2G),
    q[..., b, 2g] = r0, q[..., b, 2g + 1] = r1 (the order of the table rows)."""
    r0 = 0.5 * (r[..., :-1] + r[..., 1:])
    r1 = r[..., 1:] - r[..., :-1]
    q = np.stack([r0, r1], axis=-1)  # (..., G, B, 2)
    q = np.moveaxis(q, -3, -2)  # (..., B, G, 2)
    return q.reshape(*q.shape[:-2], -1)


def flatten_input(x):
    """x (..., B, 2G) -> (..., 2G B): component c of all bins, then component c + 1. The
    data side (inference) uses the same function."""
    return np.swapaxes(x, -1, -2).reshape(*x.shape[:-2], -1)


def _group_index(lookup, keys):
    """{m: index of group m in the lookup table} for the per-m waveform keys."""
    modes = [tuple(md) for md in lookup.meta["modes"]]
    ms = sorted({int(k.rsplit("_m", 1)[1]) for k in keys})
    gi = {m: [i for i, md in enumerate(modes) if md[1] == m] for m in ms}
    if any(len(v) != 1 for v in gi.values()) or len(ms) != len(modes):
        raise ValueError(f"Waveform groups m = {ms} do not match the lookup modes {modes}.")
    return {m: v[0] for m, v in gi.items()}


class ModeGroupProxyConditioning(GNPEBase):
    """Prior conditioning on (Mc~, t*~) and division of every m group by its reference."""

    def __init__(self, kernel, lookup, frequencies, ifo_names):
        """
        Parameters
        ----------
        kernel : dict
            bilby priors for "chirp_mass" and "t_star" (additive offsets; the chirp-mass
            proxy is then rounded to the lookup grid).
        lookup : ModeGroupLookup
            Phases, (2,2) group delay and K on the grid (`build_lookup`).
        frequencies : array
            The bin edges (the domain).
        ifo_names : list
            Detectors whose mean time defines the network centre.
        """
        kernel = dict(kernel)
        if set(kernel) != {"chirp_mass", "t_star"}:
            raise KeyError("Kernel must have exactly the keys chirp_mass and t_star.")
        super().__init__(kernel, {"chirp_mass": "+", "t_star": "+"})
        self.lookup = lookup
        self.f = np.asarray(frequencies, dtype=np.float64)
        self.ifo_names = list(ifo_names)
        self.delta = lookup.delta
        f_star = lookup.meta["f_star"]
        self.i_star = int(np.searchsorted(self.f, f_star))
        self.i22 = [tuple(md) for md in lookup.meta["modes"]].index((2, 2))

    def _proxies(self, mc, t_star, given, ext):
        if given:
            mc_p = np.asarray(ext["chirp_mass_proxy"], dtype=np.float64)
            if np.any(np.abs(round_to_grid(mc_p, self.delta) - mc_p) > 1e-9 * self.delta):
                raise ValueError("The chirp-mass proxy is not on the lookup grid.")
            return mc_p, np.asarray(ext["t_star_proxy"], dtype=np.float64)
        n = None if np.ndim(mc) == 0 else len(mc)
        k_mc = self.kernel["chirp_mass"].sample(n)
        k_t = self.kernel["t_star"].sample(n)
        return round_to_grid(mc + k_mc, self.delta), t_star + k_t

    def __call__(self, input_sample):
        sample = input_sample.copy()
        ext = sample["extrinsic_parameters"].copy()
        par = sample["parameters"]
        t_centre = np.mean([ext[f"{n}_time"] for n in self.ifo_names], axis=0)
        mc = np.asarray(par["chirp_mass"], dtype=np.float64)
        given = set(self.proxy_list).issubset(ext)
        if "waveform" not in sample:
            # standardization only (single samples, no waveform): t* from the (2,2)
            # fiducial group delay at the true chirp mass
            if not given:
                _, g_true = self.lookup.phases_interpolated(mc)
                t_star = t_centre - g_true
                mc_p, t_p = self._proxies(mc, t_star, False, ext)
                ext.update(chirp_mass_proxy=mc_p, t_star_proxy=t_p,
                           delta_chirp_mass=mc - mc_p, delta_t_star=t_star - t_p)
            sample["extrinsic_parameters"] = ext
            return sample
        wf = sample["waveform"]
        gi = _group_index(self.lookup, wf.keys())
        batched = mc.ndim > 0
        mcb = np.atleast_1d(mc)
        tcb = np.atleast_1d(t_centre)
        wfb = {k: np.atleast_2d(v) for k, v in wf.items()}
        n = len(mcb)
        out = {k: np.empty(v.shape, dtype=DTYPE) for k, v in wfb.items()}
        mc_p_all, t_p_all, t_star_all = np.empty(n), np.empty(n), np.empty(n)
        lo, hi = self.i_star - 1, self.i_star
        f76 = self.f ** (7 / 6)
        for a in range(0, n, CHUNK):
            sl = slice(a, min(a + CHUNK, n))
            phi_t, g_t = self.lookup.phases_interpolated(mcb[sl])  # (c, G, B + 1), (c,)
            h22 = wfb["h_plus_m2"][sl]
            g_ratio = np.angle(h22[:, hi] / h22[:, lo]) / (self.f[hi] - self.f[lo]) / (2 * np.pi)
            t_star = tcb[sl] - (g_t + g_ratio)
            ext_sl = ({k: np.atleast_1d(ext[k])[sl] for k in self.proxy_list} if given else None)
            mc_p, t_p = self._proxies(mcb[sl], t_star, given, ext_sl)
            phi_p, g_p = self.lookup.phases_at_node(mc_p)
            time_factor = f76 * np.exp(2j * np.pi * np.multiply.outer(t_p + g_p, self.f))
            for k, v in wfb.items():
                g = gi[int(k.rsplit("_m", 1)[1])]
                out[k][sl] = v[sl] * np.exp(1j * (phi_t[:, g] - phi_p[:, g])) * time_factor
            mc_p_all[sl], t_p_all[sl], t_star_all[sl] = mc_p, t_p, t_star
        if not batched:
            out = {k: v[0] for k, v in out.items()}
            mc_p_all, t_p_all, t_star_all = mc_p_all[0], t_p_all[0], t_star_all[0]
        sample["waveform"] = out
        ext["chirp_mass_proxy"], ext["t_star_proxy"] = mc_p_all, t_p_all
        ext["delta_chirp_mass"] = mc - mc_p_all
        ext["t_star"] = t_star_all
        ext["delta_t_star"] = t_star_all - t_p_all
        sample["extrinsic_parameters"] = ext
        return sample


class ProjectModeGroupsFrequencyDependent:
    """Projection of every m group onto the detectors with the frequency-dependent
    response (Stage C: Earth rotation, finite size). One response per sample, with the
    (2,2) time map for all groups (user, 2026-11-16); linear in the polarizations, so each
    group is projected with the same factors. Output waveform {detector: (..., G, B + 1)},
    groups in the order of `groups` (m values). Otherwise as
    ``ProjectOntoDetectorsFrequencyDependent``."""

    def __init__(self, ifo_list, domain, ref_time, settings, groups):
        from dingo.gw.detector_response import check_response_settings

        self.ifo_list = ifo_list
        self.ref_time = ref_time
        self.settings = check_response_settings(settings)
        self.frequencies = np.asarray(domain(), dtype=np.float64)
        self.groups = list(groups)

    def __call__(self, input_sample):
        from dingo.gw.detector_response import component_masses, detector_responses

        sample = input_sample.copy()
        parameters = sample["parameters"].copy()
        ext = sample["extrinsic_parameters"].copy()
        try:
            d_ref = parameters["luminosity_distance"]
            d_new = ext.pop("luminosity_distance")
            ra, dec, psi = ext.pop("ra"), ext.pop("dec"), ext.pop("psi")
            assert np.allclose(parameters["geocent_time"], 0.0), \
                "Stored polarizations must have geocent_time 0."
            tc_new = ext.pop("geocent_time")
        except KeyError as e:
            raise ValueError(f"Missing parameter {e}.")
        batched = not np.isscalar(ra) and np.ndim(ra) > 0
        wf = sample["waveform"]
        hp = np.stack([np.atleast_2d(wf[f"h_plus_m{m}"]) for m in self.groups], axis=1)
        hc = np.stack([np.atleast_2d(wf[f"h_cross_m{m}"]) for m in self.groups], axis=1)
        d_ratio = np.atleast_1d(np.asarray(d_ref) / np.asarray(d_new))[:, None, None]
        d_ratio = d_ratio.astype(np.float32 if hp.dtype == np.complex64 else np.float64)
        hp, hc = hp * d_ratio, hc * d_ratio
        parameters["luminosity_distance"] = d_new
        m1, m2 = component_masses(parameters["chirp_mass"], parameters["mass_ratio"])
        names = [ifo.name for ifo in self.ifo_list]
        strains = {nm: np.empty(hp.shape, dtype=DTYPE) for nm in names}
        args = [np.atleast_1d(x) for x in (ra, dec, psi, tc_new, m1, m2,
                                           parameters["chi_1"], parameters["chi_2"])]
        for k in range(hp.shape[0]):
            resp = detector_responses(
                self.ifo_list, self.frequencies, float(args[0][k]), float(args[1][k]),
                float(args[2][k]), self.ref_time, float(args[3][k]), float(args[4][k]),
                float(args[5][k]), float(args[6][k]), float(args[7][k]), self.settings,
            )
            for nm, (fp, fc) in resp.items():
                strains[nm][k] = fp * hp[k] + fc * hc[k]
        if not batched:
            strains = {nm: v[0] for nm, v in strains.items()}
        parameters.update(ra=ra, dec=dec, psi=psi, geocent_time=tc_new)
        for nm in names:
            parameters[f"{nm}_time"] = ext.pop(f"{nm}_time")
        sample["waveform"] = strains
        sample["parameters"] = parameters
        sample["extrinsic_parameters"] = ext
        return sample


class SummarizeWhitenModeGroups:
    """r^m at the edges (per detector, (..., G, B + 1)) -> x = K(Mc~) q, flattened
    (2G B complex per detector); unit-ASD channel. White noise is added afterwards
    (AddWhiteNoiseComplex), which gives the training noise of the input."""

    def __init__(self, lookups):
        """lookups: {detector name: ModeGroupLookup} (one table per distinct PSD)."""
        self.lookups = dict(lookups)

    def __call__(self, input_sample):
        sample = input_sample.copy()
        mc_p = np.atleast_1d(sample["extrinsic_parameters"]["chirp_mass_proxy"])
        out = {}
        for nm, r in sample["waveform"].items():
            batched = r.ndim == 3
            rb = r if batched else r[None]
            n, g, b1 = rb.shape
            x = np.empty((n, b1 - 1, 2 * g), dtype=DTYPE)  # (n, B, 2G)
            lk = self.lookups[nm]
            for a in range(0, n, CHUNK):
                sl = slice(a, min(a + CHUNK, n))
                q = group_ratios_to_q(rb[sl].astype(DTYPE, copy=False))  # (c, B, 2G)
                x[sl] = np.matmul(lk.K(mc_p[sl]), q[..., None])[..., 0]
            x = flatten_input(x)
            out[nm] = x if batched else x[0]
        sample["waveform"] = out
        sample["asds"] = {k: np.ones(v.shape, dtype=np.float32) for k, v in out.items()}
        return sample


class WhitenModeGroupSummary:
    """Inference data input of a per-m network: network summaries A (per detector,
    (G, 2, B), tapered references; `dingo.gw.relative_binning_mode_data`) at the proxy
    Mc~ -> x = W(Mc~) (A + sqrt(eps s) z'), flattened as in training; unit-ASD channel.
    The top-up noise z' (Re, Im ~ N(0, 1)) gives the data input exactly the training noise
    (`ModeGroupTables.data_input`); it is drawn from a fixed seed, so the input is the same
    at every call."""

    def __init__(self, tables, chirp_mass_proxy, seed=0):
        """tables: {detector name: ModeGroupTables} (the dataset's, with the PSD of the
        detector and the eps / nodes of the lookup table)."""
        self.names = list(tables)
        self.seed = seed
        self.W, self.scale = {}, {}
        for n, tab in tables.items():
            W, _, scale = tab.whitening(float(chirp_mass_proxy))
            self.W[n], self.scale[n] = W, scale

    def __call__(self, input_sample):
        from dingo.gw.relative_binning_mode_data import to_table_layout

        sample = input_sample.copy()
        rng = np.random.default_rng(self.seed)
        out = {}
        for n in self.names:
            A = to_table_layout(sample["waveform"][n])  # (B, 2G)
            z = rng.standard_normal(A.shape) + 1j * rng.standard_normal(A.shape)
            x = np.einsum("bij,bj->bi", self.W[n], A + self.scale[n][:, None] * z)
            out[n] = flatten_input(x)
        sample["waveform"] = out
        sample["asds"] = {k: np.ones(v.shape, dtype=np.float32) for k, v in out.items()}
        return sample
