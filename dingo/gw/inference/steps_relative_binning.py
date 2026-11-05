"""Chain steps for relative-binning summary-data networks (RelativeBinningDomain, Stage B).

A network may infer `delta_t_star = t* - t_star_proxy` instead of `geocent_time`. The summary
data are heterodyned at the proxies, so the data only carry the offset of t* from its
proxy; an absolute time output would have to be rebuilt by the network from the scalar
proxy input to the posterior precision (about 0.1 ms of a 0.4 s spread), which makes the
flow fragile (RUN_LOG dingo-BNS-ET 2026-11-05). `ProxyOffsetReparam` gives
`t_star = delta_t_star + t_star_proxy`; `TStarToGeocentTime` then maps t* to geocent_time
with the definition used in training (`RelativeBinningProxyConditioning`):

    t* = t_centre - G(theta),   t_centre = geocent_time + mean_ifo delay(ra, dec, t_ref),
    G(theta) = g_fid(Mc) + g_ratio(theta)

with g_fid the fiducial group delay at f_star and g_ratio the group delay of h_plus /
exp(i Phi_fid(Mc)) from the two bin edges around f_star. At fixed other parameters the map
is a shift, so log_det = 0.
"""

import numpy as np
import torch

from dingo.core.inference.steps import Reparametrization
from dingo.gw.domains import build_domain
from dingo.gw.gwutils import get_interferometer_list
from dingo.gw.transforms.relative_binning_transforms import FiducialPhase
from dingo.gw.waveform_generator import WaveformGenerator

# Parameters that set the waveform's group delay at f_star (aligned spins, tides).
# Phase, distance, inclination and polarization do not change the group delay of the
# (2,2) mode; they are fixed to neutral values.
_INTRINSIC = ["chirp_mass", "mass_ratio", "chi_1", "chi_2", "lambda_1", "lambda_2"]
_NEUTRAL = {"phase": 0.0, "theta_jn": 0.0, "luminosity_distance": 100.0}


class GroupDelayAtFStar:
    """G(theta) of the training definition, from the model metadata (numpy, per sample)."""

    def __init__(self, metadata):
        domain = build_domain(metadata["dataset_settings"]["domain"])
        wg = metadata["dataset_settings"]["waveform_generator"]
        self.f = np.asarray(domain(), dtype=np.float64)
        self.wfg = WaveformGenerator(domain=domain, **wg)
        self.fid = FiducialPhase(
            WaveformGenerator(wg["approximant"], domain, wg["f_ref"]),
            self.f,
            f_star=domain.settings.get("f_star", 100.0),
        )
        self.i_star = int(np.searchsorted(self.f, self.fid.f_star))

    def one(self, theta):
        """G for one parameter dict (floats). NaN if the waveform cannot be generated
        (out-of-prior network samples, e.g. negative tidal deformability)."""
        try:
            hp = self.wfg.generate_hplus_hcross({**_NEUTRAL, **theta})["h_plus"]
        except Exception:
            return np.nan
        e, g_fid = self.fid(theta["chirp_mass"])
        r = hp * np.conj(e)
        lo, hi = self.i_star - 1, self.i_star
        if r[lo] == 0 or r[hi] == 0 or not np.all(np.isfinite([r[lo], r[hi]])):
            return np.nan
        g_ratio = np.angle(r[hi] / r[lo]) / (self.f[hi] - self.f[lo]) / (2 * np.pi)
        return g_fid + g_ratio

    def __call__(self, params):
        """G for columns of parameters (arrays of equal length)."""
        cols = {k: np.asarray(params[k], dtype=np.float64).reshape(-1) for k in _INTRINSIC}
        n = len(cols["chirp_mass"])
        return np.array([self.one({k: float(v[i]) for k, v in cols.items()}) for i in range(n)])


class TStarToGeocentTime(Reparametrization):
    """t* -> geocent_time at fixed intrinsic parameters and training-frame sky position
    (see module docstring). Runs before `RAToEventFrame` (reads `ra@t_ref`)."""

    def __init__(self, metadata):
        self.group_delay = GroupDelayAtFStar(metadata)
        detectors = metadata["train_settings"]["data"]["detectors"]
        self.ifos = get_interferometer_list(detectors)
        self.ra_name = "ra@t_ref"
        self.inputs = ["t_star"]
        self.parameters = ["geocent_time"]
        self.conditioning = _INTRINSIC + [self.ra_name, "dec"]

    def _shift(self, given, context):
        """G(theta) - mean detector delay, per row (float64 numpy)."""
        g = self.group_delay({k: _np(given[k]) for k in _INTRINSIC})
        ra, dec = _np(given[self.ra_name]), _np(given["dec"])
        t_ref = context.t_ref
        delay = np.mean(
            [[ifo.time_delay_from_geocenter(a, d, t_ref) for a, d in zip(ra, dec)]
             for ifo in self.ifos], axis=0)
        return g - delay

    def forward(self, given, context):
        t_star = given["t_star"]
        shift = self._shift(given, context)
        out = _np(t_star) + shift
        return {"geocent_time": torch.as_tensor(out, dtype=t_star.dtype, device=t_star.device)}

    def inverse(self, params, context, given=None):
        tc = params["geocent_time"]
        shift = self._shift({**given, **params}, context)
        out = _np(tc) - shift
        return {"t_star": torch.as_tensor(out, dtype=tc.dtype, device=tc.device)}


def _np(x):
    return x.detach().double().cpu().numpy().reshape(-1) if torch.is_tensor(x) else np.asarray(
        x, dtype=np.float64).reshape(-1)


def relative_binning_time_steps(metadata, inference_parameters, context_parameters):
    """[TStarToGeocentTime] when the network infers delta_t_star with the t_star proxy as
    context (after `ProxyOffsetReparam` has produced t_star); else []."""
    if "delta_t_star" in inference_parameters and "t_star_proxy" in context_parameters:
        return [TStarToGeocentTime(metadata)]
    return []
