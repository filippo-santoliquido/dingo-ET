"""Bin edges and Gram matrices of a RelativeBinningDomain, found from hyperparameters.

The bins are selected with bilby_xG (greedy bisection, Leslie, Dai & Pratten 2021,
arXiv:2109.09872, as in `bilby_xG.bin_selection`): a bin is split until, at every training
point, its log-likelihood error is below eta / n_bins. The training points are drawn from
the distribution the network is trained on: the dataset's intrinsic prior, the extrinsic
ranges given here, and the proxies (Mc~, t*~) drawn from the kernel. For each truth, the data
are the noise-free signal and the test waveforms are posterior-like samples around it
(Gaussian from the Fisher matrix, kept if the exact drop in ln L is below `max_drop`).
All likelihoods use the reference h_ref = f^-7/6 exp(i Phi_fid(f; Mc~)) exp(-2 pi i f t_shift)
of `relative_binning_transforms`, on a geometric grid of `n_grid` points.

The Gram matrices (B, 3) per detector are Bmat_b = 4 int_b f^-7/3 / S [[1, u], [u, u^2]] df
from the detectors' PSDs (bilby / bilby_xG interferometers).
"""

import copy
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from bilby.core.prior import PriorDict

from dingo.gw.domains import RelativeBinningDomain
from dingo.gw.gwutils import get_interferometer_list
from dingo.gw.transforms.relative_binning_transforms import FiducialPhase
from dingo.gw.waveform_generator import WaveformGenerator

#: defaults of domain settings["bin_selection"]
BIN_SELECTION_DEFAULTS = dict(
    eta=0.3, n_train=128, n_validation=64, n_samples=8, snr_range=[10.0, 200.0],
    geocent_time=[-0.03, 0.03], n_grid=30001, max_drop=20.0, seed=2000,
)
#: parameters varied in the Fisher matrix (aligned spins); finite-difference steps
FISHER_STEPS = dict(
    chirp_mass=1e-7, mass_ratio=1e-4, chi_1=1e-4, chi_2=1e-4, lambda_1=1.0, lambda_2=1.0,
    log_distance=1e-3, theta_jn=1e-3, phase=1e-3, geocent_time=1e-5, ra=1e-4, dec=1e-4,
    psi=1e-3,
)
_STATE = {}


def _strain(f, theta):
    """(n_ifo, n_f) detector strain on the selection grid f (= _STATE["f"]) of parameters
    theta (dict with log_distance), and the polarizations."""
    st = _STATE
    p = {k: v for k, v in theta.items() if k != "log_distance"}
    p["luminosity_distance"] = float(np.exp(theta["log_distance"]))
    pol = st["wfg"].generate_hplus_hcross(dict(p, geocent_time=0.0))
    out = []
    for ifo in st["ifos"]:
        fp = ifo.antenna_response(p["ra"], p["dec"], st["ref_time"], p["psi"], "plus")
        fc = ifo.antenna_response(p["ra"], p["dec"], st["ref_time"], p["psi"], "cross")
        dt = p["geocent_time"] + ifo.time_delay_from_geocenter(p["ra"], p["dec"], st["ref_time"])
        out.append((fp * pol["h_plus"] + fc * pol["h_cross"]) * np.exp(-2j * np.pi * f * dt))
    return np.array(out), pol


def _t_star(theta, pol):
    """Arrival time of f_star at the network centre (as RelativeBinningProxyConditioning)."""
    st = _STATE
    f, i = st["f"], st["i_star"]
    e, g_fid = st["fid"](theta["chirp_mass"])
    r = pol["h_plus"] * np.conj(e)
    g_ratio = np.angle(r[i] / r[i - 1]) / (f[i] - f[i - 1]) / (2 * np.pi)
    t_centre = theta["geocent_time"] + np.mean(
        [ifo.time_delay_from_geocenter(theta["ra"], theta["dec"], st["ref_time"])
         for ifo in st["ifos"]])
    return t_centre - (g_fid + g_ratio)


class _Reference:
    """Duck type of bilby_xG.bin_selection.Projector: project() gives h_ref per detector."""

    def __init__(self, ifos):
        self.interferometers = ifos

    def project(self, params, freqs):
        e, g_fid = _STATE["fid"](params["mc_proxy"])
        h = freqs ** (-7 / 6) * e * np.exp(-2j * np.pi * freqs * (params["t_proxy"] + g_fid))
        return np.broadcast_to(h, (len(self.interferometers), 1, len(freqs))).copy()


def _point(job):
    """One truth: GridModel (data = signal, reference at the proxies) and GridPoints."""
    import bilby
    from bilby_xG.bin_selection import GridModel, GridPoint

    k, seed, snr_target = job
    bilby.core.utils.random.seed(seed)  # prior and kernel draws are reproducible per point
    st, sel = _STATE, _STATE["sel"]
    rng = np.random.default_rng(seed)
    f, psd = st["f"], st["psd"]
    w = np.gradient(f)
    keys = st["keys"]
    s = st["prior"].sample()
    theta = {key: float(s[key]) for key in st["intrinsic_keys"]}
    theta.update(theta_jn=float(np.arccos(rng.uniform(-1, 1))), phase=float(rng.uniform(0, 2 * np.pi)),
                 ra=float(rng.uniform(0, 2 * np.pi)), dec=float(np.arcsin(rng.uniform(-1, 1))),
                 psi=float(rng.uniform(0, np.pi)), geocent_time=float(rng.uniform(*sel["geocent_time"])),
                 log_distance=float(np.log(100.0)))
    theta.update({kk: float(s[kk]) for kk in ("theta_jn", "phase") if kk in s})

    def network_snr2(h):
        return 4 * np.real(np.einsum("df,df,f->", np.abs(h) ** 2, 1 / psd, w))

    h0, _ = _strain(f, theta)
    theta["log_distance"] += 0.5 * np.log(network_snr2(h0)) - np.log(snr_target)
    h0, pol0 = _strain(f, theta)
    # Fisher matrix, plus the prior widths as a Gaussian precision
    v0 = np.array([theta[key] for key in keys])
    jac = []
    for i, key in enumerate(keys):
        tp, tm = dict(theta), dict(theta)
        tp[key] += FISHER_STEPS[key]
        tm[key] -= FISHER_STEPS[key]
        jac.append((_strain(f, tp)[0] - _strain(f, tm)[0]) / (2 * FISHER_STEPS[key]))
    jac = np.array(jac)
    F = 4 * np.real(np.einsum("adf,bdf,df,f->ab", jac.conj(), jac, 1 / psd, w))
    F += np.diag(1 / st["prior_sd"] ** 2)
    d = np.sqrt(np.diag(F))
    evals, U = np.linalg.eigh(F / np.outer(d, d))
    evals = np.clip(evals, 1e-12, None)
    kept = []
    for alpha in (1.0, 0.3, 0.1, 0.03):
        z = rng.standard_normal((150, len(keys))) / np.sqrt(evals)
        for v in np.clip(v0 + alpha * (z @ U.T) / d, st["lo"], st["hi"]):
            th = dict(theta, **dict(zip(keys, v)))
            h, _ = _strain(f, th)
            if 0.5 * network_snr2(h0 - h) < sel["max_drop"]:
                kept.append(h)
        if len(kept) >= 4 * sel["n_samples"]:
            break
    pick = rng.choice(len(kept), size=min(sel["n_samples"], len(kept)), replace=False) if kept else []
    kernel = st["kernel"]
    proxy = dict(mc_proxy=theta["chirp_mass"] + float(kernel["chirp_mass"].sample()),
                 t_proxy=_t_star(theta, pol0) + float(kernel["t_star"].sample()))
    model = GridModel(st["projector"], proxy, f, second_order=True, data=h0)
    model.projector = None
    points = [GridPoint(model, h[:, None, :]) for h in [h0] + [kept[i] for i in pick]]
    return dict(k=k, snr=float(np.sqrt(network_snr2(h0))), n_samples=len(points) - 1), points


def gram_matrices(edges, ifos, n_fine=4_000_001):
    """(B, 3) Gram entries per detector from its PSD (reference amplitude f^-7/6)."""
    f = np.geomspace(edges[0], edges[-1], n_fine)
    nb = len(edges) - 1
    b = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, nb - 1)
    u = (f - 0.5 * (edges[:-1] + edges[1:])[b]) / np.diff(edges)[b]
    w = np.gradient(f)
    out = {}
    for ifo in ifos:
        p = 4 * w * f ** (-7 / 3) / ifo.power_spectral_density.get_power_spectral_density_array(f)
        g = np.stack([np.bincount(b, p * x, nb) for x in (1.0, u, u**2)], axis=-1)
        if not (np.all(g[:, 0] > 0) and np.all(g[:, 0] * g[:, 2] > g[:, 1] ** 2)):
            raise ValueError(f"Gram matrix of {ifo.name} is not positive definite.")
        out[ifo.name] = g
    return out


def complete_relative_binning_domain(domain_settings, waveform_generator_settings,
                                     intrinsic_prior, ref_time=1.4e9, num_processes=1):
    """Fill in bin_edges and gram of RelativeBinningDomain settings from their hyperparameters.

    Parameters
    ----------
    domain_settings : dict
        {"type": "RelativeBinningDomain", "settings": {detectors, f_min, f_max, kernel,
        f_star, bin_selection}}; bin_edges / gram are computed if absent.
    waveform_generator_settings : dict
        Dataset waveform generator settings (approximant, f_ref).
    intrinsic_prior : PriorDict
        The dataset's intrinsic prior.

    Returns
    -------
    (domain settings with bin_edges and gram, report dict with the held-out errors)
    """
    try:
        from bilby_xG.bin_selection import select_bins
    except ImportError as e:
        raise ImportError("Relative-binning bin selection needs bilby_xG.") from e

    out = copy.deepcopy(domain_settings)
    s = out["settings"]
    sel = dict(BIN_SELECTION_DEFAULTS, **s.get("bin_selection", {}))
    s["bin_selection"] = sel
    ifos = get_interferometer_list(s["detectors"])
    f = np.geomspace(s["f_min"], s["f_max"], sel["n_grid"])
    approx, f_ref = waveform_generator_settings["approximant"], waveform_generator_settings["f_ref"]
    fid = FiducialPhase(WaveformGenerator(approx, RelativeBinningDomain(f), f_ref), f,
                        f_star=s.get("f_star", 100.0))
    intrinsic_keys = [k for k in ("chirp_mass", "mass_ratio", "chi_1", "chi_2", "lambda_1", "lambda_2")]
    keys = intrinsic_keys + ["log_distance", "theta_jn", "phase", "geocent_time", "ra", "dec", "psi"]
    draws = intrinsic_prior.sample(2000)
    width = {k: np.std(draws[k]) if k in draws else None for k in keys}
    width.update(log_distance=1.0, theta_jn=np.pi / np.sqrt(12), phase=2 * np.pi / np.sqrt(12),
                 geocent_time=np.ptp(sel["geocent_time"]) / np.sqrt(12),
                 ra=2 * np.pi / np.sqrt(12), dec=np.pi / np.sqrt(12), psi=np.pi / np.sqrt(12))
    lo = np.array([draws[k].min() if k in intrinsic_keys else -np.inf for k in keys])
    hi = np.array([draws[k].max() if k in intrinsic_keys else np.inf for k in keys])
    _STATE.clear()
    _STATE.update(
        f=f, ifos=ifos, ref_time=ref_time, sel=sel, keys=keys, intrinsic_keys=intrinsic_keys,
        prior=intrinsic_prior, prior_sd=np.array([width[k] for k in keys]), lo=lo, hi=hi,
        psd=np.array([ifo.power_spectral_density.get_power_spectral_density_array(f) for ifo in ifos]),
        wfg=WaveformGenerator(approx, RelativeBinningDomain(f), f_ref),
        fid=fid, i_star=int(np.searchsorted(f, fid.f_star)), projector=_Reference(ifos),
        kernel=PriorDict(dict(s["kernel"])),
    )
    rng = np.random.default_rng(sel["seed"])
    n = sel["n_train"] + sel["n_validation"]
    snrs = np.exp(rng.uniform(*np.log(sel["snr_range"]), n))
    jobs = [(i, sel["seed"] + 1000 * i + 7, snrs[i]) for i in range(n)]
    t0 = time.time()
    with ProcessPoolExecutor(num_processes, mp_context=multiprocessing.get_context("fork")) as ex:
        results = list(ex.map(_point, jobs))
    train = [p for _, pts in results[: sel["n_train"]] for p in pts]
    idx = select_bins(train, sel["eta"])
    edges = f[idx]
    report = dict(n_bins=len(edges) - 1, time_points_s=time.time() - t0,
                  points=[r for r, _ in results], validation=[])
    for r, pts in results[sel["n_train"]:]:
        err = np.array([p.total_error(idx) for p in pts])
        report["validation"].append(dict(
            snr=r["snr"], offset=float(np.median(err)),
            scatter_p95=float(np.percentile(np.abs(err - np.median(err)), 95)),
            abs_p95=float(np.percentile(np.abs(err), 95))))
    out["bin_edges"] = edges.tolist()
    out["gram"] = {k: v.tolist() for k, v in gram_matrices(edges, ifos).items()}
    _STATE.clear()
    return out, report
