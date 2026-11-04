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

    Above the fiducial's end of inspiral (amplitude * f^(7/6) below 1 % of its 20 Hz
    value) the phase is continued by a quadratic fit to the last bin edges below it.
    """

    def __init__(self, waveform_generator, frequencies, f_star=100.0, df_star=1e-3):
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
        """
        self.f = np.asarray(frequencies, dtype=np.float64)
        probes = np.array([f_star - df_star, f_star + df_star])
        self.seq = np.unique(np.concatenate([self.f, probes]))
        self.i_f = np.searchsorted(self.seq, self.f)
        self.i_p = np.searchsorted(self.seq, probes)
        self.f_star, self.df_star = f_star, df_star
        self.wfg = waveform_generator
        self.wfg.domain = RelativeBinningDomain(self.seq)

    def __call__(self, chirp_mass):
        """Returns (exp(i Phi_fid) at self.f, group delay offset Psi'(f_star) / 2 pi)."""
        hp = self.wfg.generate_hplus_hcross(dict(FIDUCIAL, chirp_mass=float(chirp_mass)))[
            "h_plus"
        ]
        a, b = hp[self.i_p]
        group_delay = np.angle(b / a) / (2 * self.df_star) / (2 * np.pi)
        h = hp[self.i_f]
        e = np.exp(1j * np.angle(h))
        amp = np.abs(h) * self.f ** (7 / 6)
        low = (amp < 0.01 * np.interp(20.0, self.f, amp)) & (self.f >= 20.0)
        if low.any():
            i_cut = int(np.argmax(low))
            fit = slice(max(i_cut - 8, 0), i_cut)
            x = self.f[fit] - self.f[i_cut - 1]
            c = np.polyfit(x, np.unwrap(np.angle(h[fit])), 2)
            e[i_cut:] = np.exp(1j * np.polyval(c, self.f[i_cut:] - self.f[i_cut - 1]))
        return e, group_delay


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

    The stored polarizations are h / exp(i Phi_fid(f; Mc)) at the bin edges (dataset
    option `relative_binning_reference`). This transform draws the proxies
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
