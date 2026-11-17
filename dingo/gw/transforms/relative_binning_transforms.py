"""Transforms for relative-binning summary data (ET BNS from 3 Hz, Stage B).

The network sees, per detector and bin b, the whitened relative-binning summary data
    x_b = L_b^T (r0_b, r1_b),   Bmat_b = L_b L_b^T,
with r = h / h_ref at the two edges of the bin (r0: mean, r1: difference) and
    Bmat_b = 4 int_b |h_ref|^2 / S [[1, u], [u, u^2]] df,  u = (f - f_c) / (f_hi - f_lo).
For data d, the summary A_b = 4 int_b d conj(h_ref) (1, u) / S df has the signal part
Bmat_b (r0, r1) and a complex Gaussian noise part of covariance 2 Bmat_b, so
L_b^-1 A_b = x_b + n_b with Re n, Im n ~ N(0, 1): Dingo's AddWhiteNoiseComplex.

The reference has a fixed amplitude, so Bmat (one PSD) is fixed:
    h_ref(f) = f^(-7/6) exp(i Phi_fid(f; Mc~)) exp(-2 pi i f t_shift),
Phi_fid: phase of the waveform model at the chirp-mass proxy Mc~ (other parameters
fixed, equal tidal deformabilities), t_shift such that the reference's group delay at
f_star equals the time proxy t*~ (arrival time of f_star at the network centre).
"""

import numpy as np
from bilby.core.prior import PriorDict

from dingo.gw.domains import RelativeBinningDomain
from dingo.gw.transforms.gnpe_transforms import GNPEBase

#: fixed parameters of the fiducial waveform (equal Lambda: LAL's frequency-sequence
#: result for NRTidal models does not depend on the other frequencies then)
FIDUCIAL = dict(mass_ratio=0.75, chi_1=0.0, chi_2=0.0, lambda_1=0.0, lambda_2=0.0,
                luminosity_distance=100.0, theta_jn=0.0, phase=0.0, geocent_time=0.0)


class FiducialPhase:
    """exp(i Phi_fid(f; Mc)) at fixed frequencies, and the fiducial group delay at f_star.

    Phi_fid is the phase of h_plus of the fiducial (FIDUCIAL at chirp mass Mc; with
    theta_jn = 0 only the (2, |2|) modes contribute), or, with `mode` (mlgw_bns only),
    the phase phi_lm of that multipole h_lm = A exp(+i phi_lm) of the fiducial.
    Above the fiducial's end of inspiral (amplitude * f^(7/6) below 1 % of its 20 Hz
    value, first at the edge f_cut) the phase is continued by a quadratic fit to the last
    bin edges below it. With `mode`, the continuation starts at f0, the last edge below
    f_cut: mlgw_bns sets the phase to 0 above its band end, which lies inside the bin
    [f0, f_cut], and a phase jump inside a bin breaks the cross-group noise tables of the
    per-m input (Stage D test H4m2). Without `mode` (one group: the reference phase does
    not enter the noise) it starts at f_cut, as before.
    """

    def __init__(
        self, waveform_generator, frequencies, f_star=100.0, df_star=1e-3, mode=None
    ):
        """
        Parameters
        ----------
        waveform_generator : WaveformGenerator
            Its approximant and f_ref are used; its domain is replaced.
        frequencies : array
            Frequencies [Hz] at which the phase factor is needed (e.g. the bin edges).
            Increasing, from at most f_ref; the last one should be the data f_max.
        f_star : float
            Frequency of the time proxy [Hz].
        df_star : float
            Half step of the central difference for the group delay at f_star [Hz].
        mode : (l, m), optional
            The multipole whose phase is the fiducial phase (the reference of the m
            group of a dataset with polarizations_per_m). mlgw_bns only. The group
            delay is then that of this multipole at f_star: its time to merger [s].
        """
        if mode is not None and waveform_generator.approximant_str != "mlgw_bns":
            raise ValueError("FiducialPhase with a mode needs approximant mlgw_bns.")
        self.mode = None if mode is None else (int(mode[0]), int(mode[1]))
        self.f = np.asarray(frequencies, dtype=np.float64)
        probes = np.array([f_star - df_star, f_star + df_star])
        self.seq = np.unique(np.concatenate([self.f, probes]))
        self.i_f = np.searchsorted(self.seq, self.f)
        self.i_p = np.searchsorted(self.seq, probes)
        self.f_star, self.df_star = f_star, df_star
        self.wfg = waveform_generator
        self.wfg.domain = RelativeBinningDomain(self.seq)

    def _fiducial(self, chirp_mass, frequencies=None):
        """The fiducial at chirp_mass on self.seq (or on the increasing `frequencies`):
        h_plus, or the multipole A exp(i phi) of self.mode."""
        p = dict(FIDUCIAL, chirp_mass=float(chirp_mass))
        if self.mode is not None:
            from bilby_xG.source import convert_to_mlgw_bns_parameters
            from dingo.gw.waveform_generator import mlgw_bns_model

            p, _ = convert_to_mlgw_bns_parameters(p)
            q = p["q"]
            amp, phase = mlgw_bns_model().predict_modes_amp_phase(
                [max(q, 1 / q), p["LambdaAl2"], p["LambdaBl2"], p["chi1z"], p["chi2z"]],
                p["M"],
                self.seq if frequencies is None else frequencies,
                modes=[self.mode],
                distance_mpc=p["distance"],
            )
            return amp[0, 0] * np.exp(1j * phase[0, 0])
        if frequencies is None:
            return self.wfg.generate_hplus_hcross(p)["h_plus"]
        domain = self.wfg.domain
        try:
            self.wfg.domain = RelativeBinningDomain(frequencies)
            return self.wfg.generate_hplus_hcross(p)["h_plus"]
        finally:
            self.wfg.domain = domain

    def __call__(self, chirp_mass):
        """Returns (exp(i Phi_fid) at self.f, group delay offset Psi'(f_star) / 2 pi)."""
        hp = self._fiducial(chirp_mass)
        a, b = hp[self.i_p]
        group_delay = np.angle(b / a) / (2 * self.df_star) / (2 * np.pi)
        h = hp[self.i_f]
        e = np.exp(1j * np.angle(h))
        cont = self._continuation(h)
        if cont is not None:
            f_start, f0, c = cont
            above = self.f >= f_start
            e[above] = np.exp(1j * np.polyval(c, self.f[above] - f0))
        return e, group_delay

    def phase_slope_tau(self, chirp_mass):
        """With `mode`: the continued multipole phase phi at self.f (as __call__, but not
        wrapped), its slope dphi/df [rad/Hz], and the time to merger tau = dphi/df / 2 pi
        [s] (mlgw_bns time map; above f0 from the continuation). For the per-m tables
        (`dingo.gw.relative_binning_mode_tables`)."""
        if self.mode is None:
            raise ValueError("phase_slope_tau needs a mode (mlgw_bns).")
        from bilby_xG.source import convert_to_mlgw_bns_parameters
        from dingo.gw.waveform_generator import mlgw_bns_model

        p, _ = convert_to_mlgw_bns_parameters(dict(FIDUCIAL, chirp_mass=float(chirp_mass)))
        q = p["q"]
        amp, phase, tf = mlgw_bns_model().predict_modes_amp_phase(
            [max(q, 1 / q), p["LambdaAl2"], p["LambdaBl2"], p["chi1z"], p["chi2z"]],
            p["M"], self.f, modes=[self.mode], distance_mpc=p["distance"], return_tf=True,
        )
        # the phase of A exp(i phi) as __call__ (np.angle): + pi where A < 0 (mlgw gives
        # the (4, 4) amplitude a negative sign)
        phi = np.array(phase[0, 0], dtype=np.float64) + np.pi * (amp[0, 0] < 0)
        slope = -2 * np.pi * np.array(tf[0, 0], dtype=np.float64)
        cont = self._continuation(amp[0, 0] * np.exp(1j * phi))
        if cont is not None:
            f_start, f0, _ = cont
            i0 = int(np.searchsorted(self.f, f0))
            fit = slice(max(i0 - 7, 0), i0 + 1)  # the same 8 edges as _continuation
            # fitted to the unwrapped phase: equals _continuation's fit up to 2 pi k in c[2]
            c = np.polyfit(self.f[fit] - f0, phi[fit], 2)
            above = self.f >= f_start
            x = self.f[above] - f0
            phi[above] = np.polyval(c, x)
            slope[above] = np.polyval(np.polyder(c), x)
        return phi, slope, slope / (2 * np.pi)

    def _continuation(self, h):
        """(f_start, f0, quadratic coefficients) of the phase continuation above the end
        of inspiral, from h at self.f; None if the fiducial does not end. The phase is
        continued for f >= f_start: f0 with a mode, else f_cut (class docstring)."""
        amp = np.abs(h) * self.f ** (7 / 6)
        low = (amp < 0.01 * np.interp(20.0, self.f, amp)) & (self.f >= 20.0)
        if not low.any():
            return None
        i_cut = int(np.argmax(low))
        fit = slice(max(i_cut - 8, 0), i_cut)
        f0 = self.f[i_cut - 1]
        c = np.polyfit(self.f[fit] - f0, np.unwrap(np.angle(h[fit])), 2)
        return (f0 if self.mode is not None else self.f[i_cut]), f0, c

    def at(self, chirp_mass, frequencies):
        """exp(i Phi_fid) at any increasing frequencies inside [self.f[0], self.f[-1]]
        (e.g. a chunk of the full data grid), with the end of inspiral and the
        continuation fixed by self.f, so that it equals __call__ on self.f. LAL is
        evaluated on self.f below the chunk + the chunk + self.f[-1] (no gap below,
        the same last frequency)."""
        f = np.asarray(frequencies, dtype=np.float64)
        cont = self._continuation(self._fiducial(chirp_mass)[self.i_f])
        seq = np.unique(np.concatenate([self.f[self.f < f[0]], f, self.f[-1:]]))
        hp = self._fiducial(chirp_mass, seq)[np.searchsorted(seq, f)]
        e = np.exp(1j * np.angle(hp))
        if cont is not None:
            f_start, f0, c = cont
            above = f >= f_start
            e[above] = np.exp(1j * np.polyval(c, f[above] - f0))
        return e


def summarize_and_whiten(r, lt):
    """(r0, r1) per bin from r at the bin edges, then x = L^T (r0, r1).

    Parameters
    ----------
    r : array (..., B + 1), complex
    lt : array (B, 2, 2), real: L^T per bin, with Bmat = L L^T.

    Returns
    -------
    array (..., 2 B): x_0 of all bins, then x_1 of all bins.
    """
    q = np.stack([0.5 * (r[..., :-1] + r[..., 1:]), r[..., 1:] - r[..., :-1]], axis=-1)
    x = np.einsum("bij,...bj->...ib", lt, q)
    return x.reshape(*x.shape[:-2], -1)


def whitening_from_gram(gram):
    """L^T per bin from gram (B, 3) = (Bmat_00, Bmat_01, Bmat_11)."""
    gram = np.asarray(gram, dtype=np.float64)
    bmat = np.stack(
        [np.stack([gram[:, 0], gram[:, 1]], -1), np.stack([gram[:, 1], gram[:, 2]], -1)],
        -2,
    )
    return np.swapaxes(np.linalg.cholesky(bmat), -1, -2)


class RelativeBinningProxyConditioning(GNPEBase):
    """Prior conditioning on (Mc~, t*~) and division by the reference waveform.

    The stored polarizations are h / exp(i Phi_fid(f; Mc)) at the bin edges (every
    dataset on a RelativeBinningDomain). This transform draws the proxies
        chirp_mass_proxy = chirp_mass + eps_Mc,
        t_star_proxy = t* + eps_t,  t* = mean detector time - group delay of h at f_star,
    and returns the polarizations divided by h_ref(Mc~, t*~) at the edges, so that the
    detector projection gives r = h_det / h_ref. All phases are formed in float64.
    """

    def __init__(self, kernel, fiducial_phase, ifo_names, inference=False):
        """
        Parameters
        ----------
        kernel : dict
            bilby priors for "chirp_mass" and "t_star" (additive offsets).
        fiducial_phase : FiducialPhase
            On the bin edges of the domain.
        ifo_names : list
            Detectors whose mean time defines the network centre.
        """
        kernel = dict(kernel)
        if set(kernel) != {"chirp_mass", "t_star"}:
            raise KeyError("Kernel must have exactly the keys chirp_mass and t_star.")
        super().__init__(kernel, {"chirp_mass": "+", "t_star": "+"})
        self.fid = fiducial_phase
        self.ifo_names = list(ifo_names)
        self.inference = inference
        self.f = fiducial_phase.f
        # group delay of the stored ratio from the two edges around f_star
        self.i_star = int(np.searchsorted(self.f, fiducial_phase.f_star))

    def _one(self, h_stored, chirp_mass, t_centre, rng_proxies):
        e_true, g_true_fid = self.fid(chirp_mass)
        hp = h_stored["h_plus"]
        lo, hi = self.i_star - 1, self.i_star
        g_ratio = np.angle(hp[hi] / hp[lo]) / (self.f[hi] - self.f[lo]) / (2 * np.pi)
        t_star = t_centre - (g_true_fid + g_ratio)
        if rng_proxies is None:
            mc_p = chirp_mass + self.kernel["chirp_mass"].sample()
            t_p = t_star + self.kernel["t_star"].sample()
        else:
            mc_p, t_p = rng_proxies
        e_p, g_p = self.fid(mc_p)
        factor = (e_true * np.conj(e_p) * self.f ** (7 / 6)
                  * np.exp(2j * np.pi * self.f * (t_p + g_p)))
        return {k: v * factor for k, v in h_stored.items()}, mc_p, t_p, t_star

    def __call__(self, input_sample):
        sample = input_sample.copy()
        ext = sample["extrinsic_parameters"].copy()
        par = sample["parameters"]
        t_centre = np.mean([ext[f"{n}_time"] for n in self.ifo_names], axis=0)
        mc = np.asarray(par["chirp_mass"], dtype=np.float64)
        given = set(self.proxy_list).issubset(ext)
        if "waveform" not in sample:
            # Standardization only (get_standardization_dict: single samples, no
            # waveform). t* = t_centre - G(theta) with G the group delay at f_star
            # (seconds: about the time to merger from f_star); here G of the fiducial
            # at the true chirp mass, which differs from G(theta) by milliseconds.
            if not given:
                t_star = t_centre - self.fid(float(mc))[1]
                ext["chirp_mass_proxy"] = mc + self.kernel["chirp_mass"].sample()
                ext["t_star_proxy"] = t_star + self.kernel["t_star"].sample()
                ext["delta_chirp_mass"] = mc - ext["chirp_mass_proxy"]
                ext["delta_t_star"] = t_star - ext["t_star_proxy"]
            sample["extrinsic_parameters"] = ext
            return sample
        wf = sample["waveform"]
        batched = mc.ndim > 0
        if not batched:
            proxies = (ext["chirp_mass_proxy"], ext["t_star_proxy"]) if given else None
            new, mc_p, t_p, t_star = self._one(wf, float(mc), float(t_centre), proxies)
        else:
            out = [
                self._one({k: v[i] for k, v in wf.items()}, mc[i], t_centre[i],
                          (ext["chirp_mass_proxy"][i], ext["t_star_proxy"][i]) if given else None)
                for i in range(len(mc))
            ]
            new = {k: np.stack([o[0][k] for o in out]) for k in wf}
            mc_p, t_p, t_star = (np.array([o[j] for o in out]) for j in (1, 2, 3))
        sample["waveform"] = new
        ext["chirp_mass_proxy"], ext["t_star_proxy"] = mc_p, t_p
        ext["delta_chirp_mass"] = mc - mc_p
        ext["t_star"] = t_star
        # Offset of t* from its proxy (-kernel draw): a network that infers it needs no
        # absolute time from the proxy input (RUN_LOG 2026-11-05); geocent_time is
        # rebuilt after sampling (inference.steps_relative_binning.TStarToGeocentTime).
        ext["delta_t_star"] = t_star - t_p
        sample["extrinsic_parameters"] = ext
        return sample


class SummarizeAndWhiten:
    """r at the bin edges -> whitened summary data x (2B per detector); unit ASD channel."""

    def __init__(self, gram_dict):
        """gram_dict: {ifo name: (B, 3) array (Bmat_00, Bmat_01, Bmat_11) per bin}."""
        self.lt = {k: whitening_from_gram(v) for k, v in gram_dict.items()}

    def __call__(self, input_sample):
        sample = input_sample.copy()
        sample["waveform"] = {
            k: summarize_and_whiten(v, self.lt[k]) for k, v in sample["waveform"].items()
        }
        sample["asds"] = {
            k: np.ones(v.shape, dtype=np.float32) for k, v in sample["waveform"].items()
        }
        return sample


class DivideByFiducial:
    """Dataset compression: polarizations at the bin edges divided by exp(i Phi_fid) at
    each waveform's own chirp mass, so that the stored ratio is smooth (SVD).
    `RelativeBinningProxyConditioning` multiplies it back in float64."""

    def __init__(self, fiducial_phase):
        """fiducial_phase: a FiducialPhase for every polarization, or {m: FiducialPhase}
        for per-m polarizations (keys f"{pol}_m{m}"), each divided by its own."""
        self.fid = fiducial_phase

    def __call__(self, input_sample):
        sample = input_sample.copy()
        mc = sample["parameters"]["chirp_mass"]
        if isinstance(self.fid, dict):
            e = {m: np.conj(fid(mc)[0]) for m, fid in self.fid.items()}
            sample["waveform"] = {
                k: v * e[int(k.rsplit("_m", 1)[1])]
                for k, v in sample["waveform"].items()
            }
        else:
            e = np.conj(self.fid(mc)[0])
            sample["waveform"] = {k: v * e for k, v in sample["waveform"].items()}
        return sample



class WhitenSummary:
    """Event summary data A (2B per detector: A_0 of all bins, then A_1) -> whitened
    x = L^-1 A per bin (the network input; for noise-free A = Bmat q this equals the
    training input L^T q); unit-ASD channel as in training."""

    def __init__(self, gram_dict):
        """gram_dict: {ifo name: (B, 3) Gram entries per bin}."""
        self.L = {k: np.swapaxes(whitening_from_gram(v), -1, -2) for k, v in gram_dict.items()}

    def __call__(self, input_sample):
        sample = input_sample.copy()
        out = {}
        for k, a in sample["waveform"].items():
            L = self.L[k]
            a = np.asarray(a).reshape(*np.shape(a)[:-1], 2, L.shape[0])
            a = np.swapaxes(a, -1, -2)[..., None]  # (..., B, 2, 1)
            x = np.linalg.solve(L, a)[..., 0]
            out[k] = np.swapaxes(x, -1, -2).reshape(*x.shape[:-2], -1)
        sample["waveform"] = out
        sample["asds"] = {k: np.ones(v.shape, dtype=np.float32) for k, v in out.items()}
        return sample
