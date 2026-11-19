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
        detector_response: dict = None,
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
        detector_response : dict, optional
            Frequency-dependent response (Earth rotation, finite arm length) of bilby_xG,
            as in the training (``dingo.gw.detector_response``); the time origin is t_ref.
            None: long-wavelength response of a non-rotating Earth at t_ref.
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
        self.detector_response = None
        if detector_response:
            from dingo.gw.detector_response import check_response_settings

            self.detector_response = check_response_settings(detector_response)

    def _q(self, theta: dict) -> dict:
        """(B, 2) relative-binning coefficients (r0, r1) per detector."""
        p = {k: float(v) for k, v in theta.items()}
        pol = self.waveform_generator.generate_hplus_hcross(p)
        if self.detector_response is not None:
            return self._q_frequency_dependent(p, pol)
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

    def _q_frequency_dependent(self, p: dict, pol: dict) -> dict:
        """(B, 2) coefficients with bilby_xG's frequency-dependent response at the edges."""
        from dingo.gw.detector_response import component_masses, detector_strains

        m1, m2 = component_masses(p["chirp_mass"], p["mass_ratio"])
        h = detector_strains(
            self.ifos, pol["h_plus"], pol["h_cross"], self.f, p["ra"], p["dec"], p["psi"],
            self.t_ref, p["geocent_time"], float(m1), float(m2), p["chi_1"], p["chi_2"],
            self.detector_response,
        )
        q = {}
        for name, strain in h.items():
            r = strain / self.h_ref
            q[name] = np.stack([0.5 * (r[:-1] + r[1:]), r[1:] - r[:-1]], axis=-1)
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


class ModeGroupRelativeBinningLikelihood(Likelihood):
    """Relative-binning likelihood of a per-m-group summary pass, for fixed proxies
    (ET BNS with higher-order modes, Stage D).

    One reference per m group, from the proxies only (`ModeGroupReferences`: the same
    references as the network input), and the exact cross-Gram of the references, so
    (with taper "reference": the network summaries A and the Gram tapered on both
    sides; with "own": D and the untapered Gram)

        (d|h) = sum_(m,b,k) conj(D^m_bk) q^m_bk,   (h|h) = sum_b q_b^H Q_b q_b,
        log L = log_Zn + Re (d|h) - (h|h) / 2,

    with q^m = (r0, r1) of r^m = h^m / h_ref^m at the bin edges and Q_b built from the
    Gram (see `dingo.gw.relative_binning_mode_data`). The model h^m of the group m is the
    mlgw_bns mode with the dataset's taper, projected with the frequency-dependent
    response; with `own_time_map` the response of each group follows its mode's own time
    map (bilby_xG "time_to_merger", as bilby_xG's response for per-mode polarizations),
    else the 2PN (2,2) map of the training.

    Each group goes as exp(-i m phase), so `phase_grid_terms` is exact and the synthetic
    phase runs with approximation_22_mode false (`StationaryGaussianGWLikelihood`
    format).
    """

    def __init__(
        self,
        domain,
        waveform_generator_settings: dict,
        summary: dict,
        gram: dict,
        proxies: dict,
        detectors: list,
        t_ref: float,
        log_Zn: float = 0.0,
        detector_response: dict = None,
        own_time_map: bool = True,
        taper: str = "reference",
    ):
        """
        Parameters
        ----------
        domain : RelativeBinningDomain
            The bins of the per-m dataset (with f_star).
        waveform_generator_settings : dict
            Of the per-m dataset (mlgw_bns, mode_list, taper).
        summary : dict
            {detector: (G, 2, B) complex} summaries, groups in the order of mode_list:
            with taper "reference" the network summaries A (tapered references), with
            "own" the summaries D (untapered references).
        gram : dict
            {detector: (G, G, 3, B) complex} cross-Gram of the references: tapered on
            both sides ("reference") or untapered ("own").
        proxies : dict
            {"chirp_mass_proxy", "t_star_proxy"} of the summaries.
        detectors : list
            Detector names (bilby_xG).
        t_ref : float
            GPS time origin of geocent_time and of the response (the event time).
        log_Zn : float
            -<d|d>/2 if known, else 0 (log likelihood ratio).
        detector_response : dict
            Frequency-dependent response settings (``dingo.gw.detector_response``).
        own_time_map : bool
            Response per group with the mode's own time map (True, exact) or the 2PN
            (2,2) map (as the training input).
        taper : str
            Where the mode taper of the model comes from (the dataset's
            mode_max_time_to_merger). "reference": h^m = w_ref,m h_ref^m r^m, with the
            reference's taper (time map of the fiducial at Mc~) and r^m the untapered
            ratio, smooth in every bin. "own": the sample's own taper inside r^m; r is
            then not linear in the bins where the taper turns on (Stage D test H3d).
        """
        from dingo.gw.detector_response import check_response_settings
        from dingo.gw.relative_binning_mode_data import ModeGroupReferences

        if not detector_response:
            raise NotImplementedError(
                "The per-m likelihood is implemented with the frequency-dependent "
                "response only."
            )
        self.detector_response = check_response_settings(detector_response)
        self.data_domain = domain
        self.f = np.asarray(domain(), dtype=np.float64)
        nb = len(self.f) - 1
        self.ifos = get_interferometer_list(detectors)
        self.t_ref = t_ref
        self.log_Zn = log_Zn
        wg = dict(waveform_generator_settings)
        self.references = ModeGroupReferences(domain, wg)
        G = self.references.G
        self.groups = self.references.groups
        self.own_time_map = own_time_map
        self.proxies = dict(proxies)
        self.h_ref = self.references.at_edges(self.proxies)  # (G, B + 1)
        self.D = {}
        self.Q = {}
        for ifo in self.ifos:
            d = np.asarray(summary[ifo.name], dtype=np.complex128).reshape(G, 2, nb)
            self.D[ifo.name] = np.moveaxis(d, -1, 0).reshape(nb, 2 * G)  # [b, 2g + k]
            gm = np.asarray(gram[ifo.name], dtype=np.complex128).reshape(G, G, 3, nb)
            Q = np.empty((nb, 2 * G, 2 * G), dtype=np.complex128)
            for k in range(2):
                for kp in range(2):
                    Q[:, k::2, kp::2] = np.moveaxis(gm[:, :, k + kp], -1, 0)
            self.Q[ifo.name] = 0.5 * (Q + np.conj(np.swapaxes(Q, 1, 2)))
        if taper not in ("reference", "own"):
            raise ValueError(f"taper must be 'reference' or 'own', got {taper!r}.")
        self.taper = taper
        if taper == "reference":
            # the model's taper is in the summaries and the Gram: untapered modes here
            wg = {k: v for k, v in wg.items() if k != "mode_max_time_to_merger"}
        self.waveform_generator = WaveformGenerator(domain=domain, **wg)
        if list(self.waveform_generator.mode_list) != self.references.modes:
            raise ValueError("Waveform generator and references have other modes.")

    # --- model -------------------------------------------------------------------
    def _group_q(self, p: dict, psis=None) -> dict:
        """{detector: (n_psi, B, 2G)} coefficients of every group, at the polarization
        angles `psis` (default: p["psi"]). Column 2 g + k: group g, coefficient k."""
        from dingo.gw.detector_response import component_masses, detector_responses
        from dingo.gw.transforms.relative_binning_mode_transforms import (
            group_ratios_to_q,
        )

        psis = [p["psi"]] if psis is None else list(psis)
        pol_m = self.waveform_generator.generate_mlgw_bns_waveform(
            p, group_by_m=True, return_time_to_merger=self.own_time_map
        )
        m1, m2 = component_masses(p["chirp_mass"], p["mass_ratio"])
        G = len(self.groups)
        r = {
            ifo.name: np.empty((len(psis), G, len(self.f)), dtype=complex)
            for ifo in self.ifos
        }
        for g, m in enumerate(self.groups):
            pol = pol_m[m]
            tau = pol["time_to_merger"] if self.own_time_map else None
            for i, psi in enumerate(psis):
                resp = detector_responses(
                    self.ifos, self.f, p["ra"], p["dec"], psi, self.t_ref,
                    p["geocent_time"], float(m1), float(m2), p["chi_1"], p["chi_2"],
                    self.detector_response, time_to_merger=tau,
                )
                for name, (fp, fc) in resp.items():
                    r[name][i, g] = (fp * pol["h_plus"] + fc * pol["h_cross"]) / self.h_ref[g]
        return {name: group_ratios_to_q(x) for name, x in r.items()}

    def log_likelihood(self, theta: dict) -> float:
        p = {k: float(v) for k, v in theta.items()}
        q = self._group_q(p)
        d_inner_h = 0.0
        h_inner_h = 0.0
        for name, qn in q.items():
            qn = qn[0]
            d_inner_h += np.sum(np.conj(self.D[name]) * qn).real
            h_inner_h += np.einsum(
                "bi,bij,bj->", np.conj(qn), self.Q[name], qn
            ).real
        return self.log_Zn + d_inner_h - h_inner_h / 2

    def phase_grid_terms(self, theta: dict) -> dict:
        """The terms of `StationaryGaussianGWLikelihood.phase_grid_terms` (basis angles
        psi = 0 and pi / 4; group m goes as exp(-i m phase)), from one waveform
        evaluation at phase 0."""
        p = {k: float(v) for k, v in theta.items()}
        p["phase"] = 0.0
        q = self._group_q(p, psis=(0.0, np.pi / 4))
        G = len(self.groups)
        order = np.argsort(self.groups)
        m_vals = np.array(self.groups)[order]
        kappa2 = np.zeros((2, G), dtype=complex)
        # overlap[b, c, g, g'] = (mu^b_g, mu^c_g') = sum q^b_g^H Q[g, g'] q^c_g'
        overlap = np.zeros((2, 2, G, G), dtype=complex)
        for name, qn in q.items():
            x = np.conj(self.D[name])[None] * qn  # (2, B, 2G)
            kappa2 += x.reshape(2, -1, G, 2).sum(axis=(1, 3))
            overlap += _group_blocks(np.conj(qn), self.Q[name], qn, G)
        kappa2 = kappa2[:, order]
        overlap = overlap[:, :, order][:, :, :, order]
        rho2opt_const = sum(overlap[:, :, i, i].real for i in range(G))
        cross = {}
        for i in range(G):
            for j in range(i + 1, G):
                delta = int(m_vals[j] - m_vals[i])
                pair = overlap[:, :, i, j]
                cross[delta] = cross.get(delta, 0) + pair + pair.T
        deltas = sorted(cross)
        return {
            "m_vals": m_vals,
            "kappa2_modes": kappa2,
            "rho2opt_const": rho2opt_const,
            "deltas": np.array(deltas),
            "rho2opt_crossterms": np.moveaxis(
                np.array([cross[dl] for dl in deltas]), 0, -1
            ),
        }

    def log_likelihood_from_phase_grid_terms(self, terms, phases, psis):
        """As `StationaryGaussianGWLikelihood.log_likelihood_from_phase_grid_terms`."""
        from dingo.gw.likelihood import StationaryGaussianGWLikelihood

        return StationaryGaussianGWLikelihood.log_likelihood_from_phase_grid_terms(
            self, terms, phases, psis
        )


def _group_blocks(cq, Q, q, G):
    """overlap[b, c, g, g'] = sum_bins sum_(k, k') cq^b[bin, 2g + k] Q[bin, 2g + k,
    2g' + k'] q^c[bin, 2g' + k'] for cq = conj(q^b) (2, B, 2G), Q (B, 2G, 2G)."""
    nb = Q.shape[0]
    Qb = Q.reshape(nb, G, 2, G, 2)
    a = cq.reshape(2, nb, G, 2)
    c = q.reshape(2, nb, G, 2)
    return np.einsum("xngk,ngkhl,ynhl->xygh", a, Qb, c, optimize=True)
