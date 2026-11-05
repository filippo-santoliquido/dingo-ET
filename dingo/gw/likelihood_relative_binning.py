"""Relative-binning likelihood on summary data (RelativeBinningDomain, ET BNS Stage B).

Data: per detector the unwhitened summary A_b = 4 int_b d conj(h_ref) (1, u) / S df,
u = (f - f_c) / (f_hi - f_lo), for the reference h_ref of fixed proxies (Mc~, t*~) (see
`relative_binning_transforms`). Model: r = h / h_ref at the bin edges, q_b = (r0_b, r1_b)
with r0 the mean and r1 the difference of r at the two edges. Then (Zackay, Dai &
Venumadhav 2018, linear ratio with the second-order term)

    (d | h) = sum_b conj(A_b) . q_b,    (h | h) = sum_b q_b^H Bmat_b q_b,
    log L = log_Zn + Re (d | h) - (h | h) / 2.

`log_Zn` = -<d|d>/2 is not contained in the summaries; it is 0 unless given, so log L is
the log likelihood ratio against noise.
"""

import numpy as np

from dingo.core.likelihood import Likelihood
from dingo.gw.gwutils import get_interferometer_list
from dingo.gw.transforms.relative_binning_transforms import FiducialPhase
from dingo.gw.waveform_generator import WaveformGenerator


def gram_to_bmat(gram):
    """(B, 3) Gram entries -> (B, 2, 2) matrices."""
    g = np.asarray(gram, dtype=np.float64)
    return np.stack(
        [np.stack([g[:, 0], g[:, 1]], -1), np.stack([g[:, 1], g[:, 2]], -1)], -2
    )


class RelativeBinningLikelihood(Likelihood):
    """Relative-binning likelihood of summary data for fixed proxies."""

    def __init__(
        self,
        domain,
        waveform_generator_settings: dict,
        summary: dict,
        proxies: dict,
        detectors: list,
        t_ref: float,
        log_Zn: float = 0.0,
    ):
        """
        Parameters
        ----------
        domain : RelativeBinningDomain
            With Gram matrices and settings (f_star).
        waveform_generator_settings : dict
            Dataset waveform generator settings (approximant, f_ref, ...).
        summary : dict
            {detector: (2 B,) complex A: A_0 of all bins, then A_1 of all bins}.
        proxies : dict
            {"chirp_mass_proxy", "t_star_proxy"} of the summary's reference.
        detectors : list
            Detector names (bilby / bilby_xG).
        t_ref : float
            Reference GPS time of the detector response.
        log_Zn : float
            -<d|d>/2 if known, else 0 (log likelihood ratio).
        """
        self.data_domain = domain
        self.f = domain.bin_edges
        nb = domain.num_bins
        self.ifos = get_interferometer_list(detectors)
        self.t_ref = t_ref
        self.log_Zn = log_Zn
        self.bmat = {ifo.name: gram_to_bmat(domain.gram[ifo.name]) for ifo in self.ifos}
        self.A = {
            ifo.name: np.asarray(summary[ifo.name], dtype=np.complex128).reshape(2, nb).T
            for ifo in self.ifos
        }
        wg = dict(waveform_generator_settings)
        self.waveform_generator = WaveformGenerator(domain=domain, **wg)
        fid = FiducialPhase(
            WaveformGenerator(wg["approximant"], domain, wg["f_ref"]),
            self.f,
            f_star=domain.settings.get("f_star", 100.0),
        )
        e_p, g_p = fid(proxies["chirp_mass_proxy"])
        self.h_ref = (
            self.f ** (-7 / 6)
            * e_p
            * np.exp(-2j * np.pi * self.f * (proxies["t_star_proxy"] + g_p))
        )
        self.proxies = dict(proxies)

    def _q(self, theta: dict) -> dict:
        """(B, 2) relative-binning coefficients (r0, r1) per detector."""
        p = {k: float(v) for k, v in theta.items()}
        pol = self.waveform_generator.generate_hplus_hcross(p)
        q = {}
        for ifo in self.ifos:
            fp = ifo.antenna_response(p["ra"], p["dec"], self.t_ref, p["psi"], "plus")
            fc = ifo.antenna_response(p["ra"], p["dec"], self.t_ref, p["psi"], "cross")
            dt = p["geocent_time"] + ifo.time_delay_from_geocenter(
                p["ra"], p["dec"], self.t_ref
            )
            h = (fp * pol["h_plus"] + fc * pol["h_cross"]) * np.exp(
                -2j * np.pi * self.f * dt
            )
            r = h / self.h_ref
            q[ifo.name] = np.stack([0.5 * (r[:-1] + r[1:]), r[1:] - r[:-1]], axis=-1)
        return q

    def _terms(self, q: dict) -> dict:
        d_inner_h = sum(np.sum(np.conj(self.A[n]) * q[n]) for n in q)
        h_inner_h = sum(
            np.einsum("bi,bij,bj->", np.conj(q[n]), self.bmat[n], q[n]).real for n in q
        )
        return {"d_inner_h": complex(d_inner_h), "h_inner_h": float(h_inner_h)}

    def log_likelihood(self, theta: dict) -> float:
        t = self._terms(self._q(theta))
        return self.log_Zn + t["d_inner_h"].real - t["h_inner_h"] / 2

    def phase_grid_terms_22(self, theta: dict) -> dict:
        """(d | h_0) and (h_0 | h_0) at phase 0, as
        `StationaryGaussianGWLikelihood.phase_grid_terms_22`."""
        return self._terms(self._q({**theta, "phase": 0.0}))

    def log_likelihood_22_from_terms(self, terms: dict, phases: np.ndarray) -> np.ndarray:
        """log L(phase) = log_Zn + Re[(d | h_0) exp(2i phase)] - (h_0 | h_0) / 2."""
        d_inner_h = np.asarray(terms["d_inner_h"])[..., None]
        h_inner_h = np.asarray(terms["h_inner_h"])[..., None]
        return (
            self.log_Zn
            + (d_inner_h * np.exp(2j * np.asarray(phases))).real
            - h_inner_h / 2
        )
