"""Relative-binning transforms: the training chain (stored ratio -> proxy conditioning ->
detector projection -> summary and whitening) gives the same whitened summary data as the
exact compression of the full uniform-grid strain with the same reference."""

import bilby
import numpy as np
import pytest

from dingo.gw.domains import RelativeBinningDomain, UniformFrequencyDomain
from dingo.gw.gwutils import get_extrinsic_prior_dict  # noqa: F401 (import check)
from dingo.gw.gwutils import get_interferometer_list
from dingo.gw.transforms import GetDetectorTimes, ProjectOntoDetectors
from dingo.gw.transforms.relative_binning_transforms import (
    FiducialPhase,
    RelativeBinningProxyConditioning,
    SummarizeAndWhiten,
    whitening_from_gram,
)
from dingo.gw.waveform_generator import WaveformGenerator

APPROX, F_REF, T = "IMRPhenomXAS_NRTidalv3", 100.0, 64.0
REF_TIME = 1.4e9


def psd(f):
    return 1e-48 * ((f / 30.0) ** -4 + 1 + (f / 300.0) ** 2)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_chain_matches_exact_compression(seed):
    bilby.core.utils.random.seed(seed)  # proxies are drawn with bilby's generator
    ufd = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=1 / T)
    fu = ufd()[ufd.min_idx :].astype(np.float64)  # k / 64 Hz: exact in float32 and float64
    idx = np.unique(np.rint(np.geomspace(20.0, 1024.0, 300) * T).astype(int)) - ufd.min_idx
    edges = fu[idx]
    rbd = RelativeBinningDomain(edges)
    ifos = get_interferometer_list(["H1", "L1"])
    p = dict(chirp_mass=2.5, mass_ratio=0.8, chi_1=0.01, chi_2=-0.02, lambda_1=2000.0,
             lambda_2=50.0, luminosity_distance=100.0, theta_jn=0.7, phase=1.3,
             geocent_time=0.0)
    ext = dict(ra=1.1, dec=-0.4, psi=0.6, geocent_time=0.004, luminosity_distance=300.0)

    # stored ratio (as in a dataset on a RelativeBinningDomain)
    fid = FiducialPhase(WaveformGenerator(APPROX, rbd, F_REF), edges)
    e_true, _ = fid(p["chirp_mass"])
    pol = WaveformGenerator(APPROX, rbd, F_REF).generate_hplus_hcross(p)
    stored = {k: v * np.conj(e_true) for k, v in pol.items()}

    # training chain; proxies drawn from the kernel around (Mc, t*)
    sample = {"parameters": dict(p), "waveform": stored, "extrinsic_parameters": dict(ext)}
    sample = GetDetectorTimes(ifos, REF_TIME)(sample)
    sample = RelativeBinningProxyConditioning(
        {"chirp_mass": "bilby.core.prior.Uniform(-3e-4, 3e-4)",
         "t_star": "bilby.core.prior.Uniform(-3e-3, 3e-3)"}, fid, ["H1", "L1"])(sample)
    e = sample["extrinsic_parameters"]
    mc_p, t_p = e["chirp_mass_proxy"], e["t_star_proxy"]
    assert abs(mc_p - p["chirp_mass"]) <= 3e-4 and abs(t_p - e["t_star"]) <= 3e-3
    # t* is the arrival of f_star at the centre: about the merger time minus the time
    # to merger from 100 Hz (0.64 s here), not the merger time
    assert -0.7 < e["t_star"] < -0.5
    sample = ProjectOntoDetectors(ifos, rbd, REF_TIME)(sample)
    b = np.searchsorted(edges, fu, side="right") - 1
    nb = len(edges) - 1
    valid = (b >= 0) & (b < nb)
    fv, bv = fu[valid], b[valid]
    u = (fv - 0.5 * (edges[:-1] + edges[1:])[bv]) / np.diff(edges)[bv]
    w_ref = 4 / T * fv ** (-7 / 3) / psd(fv)
    gram = np.stack([np.bincount(bv, w_ref * x, nb) for x in (1.0, u, u ** 2)], -1)
    x_chain = SummarizeAndWhiten({n: gram for n in ("H1", "L1")})(sample)["waveform"]

    # exact: full uniform grid, same reference (same fiducial function on these frequencies)
    fid_u = FiducialPhase(WaveformGenerator(APPROX, rbd, F_REF), fv)
    e_p, g_p = fid_u(mc_p)
    h_ref = fv ** (-7 / 6) * e_p * np.exp(-2j * np.pi * fv * (t_p + g_p))
    pol_u = WaveformGenerator(APPROX, ufd, F_REF).generate_hplus_hcross(p)
    L = np.swapaxes(whitening_from_gram(gram), -1, -2)
    for ifo in ifos:
        fp = ifo.antenna_response(ext["ra"], ext["dec"], REF_TIME, ext["psi"], "plus")
        fc = ifo.antenna_response(ext["ra"], ext["dec"], REF_TIME, ext["psi"], "cross")
        dt = ext["geocent_time"] + ifo.time_delay_from_geocenter(ext["ra"], ext["dec"], REF_TIME)
        h = (fp * pol_u["h_plus"] + fc * pol_u["h_cross"])[ufd.min_idx :][valid]
        h = h * np.exp(-2j * np.pi * fv * dt) * 100.0 / 300.0
        a = 4 / T * h * np.conj(h_ref) / psd(fv)
        A = np.stack([np.bincount(bv, (a * x).real, nb) + 1j * np.bincount(bv, (a * x).imag, nb)
                      for x in (1.0, u)], -1)
        x_exact = np.linalg.solve(L, A[..., None])[..., 0]
        x_exact = np.concatenate([x_exact[:, 0], x_exact[:, 1]])
        err = np.linalg.norm(x_chain[ifo.name] - x_exact)
        assert np.linalg.norm(x_exact) > 5  # a real signal, not zeros
        assert err < 5e-3 * np.linalg.norm(x_exact)  # linear RB error of this toy binning, (ifo.name, err, np.linalg.norm(x_exact))


def test_whitened_noise_is_unit():
    """Noise compressed with the same weights has covariance 2 Bmat; L^-1 makes it unit."""
    rng = np.random.default_rng(1)
    gram = np.array([[2.0, 0.1, 0.2], [1.0, -0.05, 0.09]])
    L = np.swapaxes(whitening_from_gram(gram), -1, -2)
    bmat = L @ np.swapaxes(L, -1, -2)
    np.testing.assert_allclose(bmat[:, 0, 0], gram[:, 0])
    np.testing.assert_allclose(bmat[:, 0, 1], gram[:, 1])
    np.testing.assert_allclose(bmat[:, 1, 1], gram[:, 2])
    z = rng.standard_normal((200000, 2, 2)) + 1j * rng.standard_normal((200000, 2, 2))
    n = np.einsum("bij,kbj->kbi", L, z) / np.sqrt(1.0)  # covariance 2 Bmat
    x = np.linalg.solve(L, n[..., None])[..., 0]
    np.testing.assert_allclose(np.mean(np.abs(x) ** 2, axis=0), 2.0, rtol=0.02)


@pytest.mark.parametrize("chirp_mass", [1.2, 2.5])
def test_fiducial_at_equals_call(chirp_mass):
    """FiducialPhase.at on other frequencies (a dense chunk) uses the same reference as
    on the bin edges: equal at the edges inside the chunk, and smooth between them."""
    edges = np.geomspace(3.0, 4096.0, 1500)
    fid = FiducialPhase(WaveformGenerator(APPROX, RelativeBinningDomain(edges), F_REF), edges)
    e, _ = fid(chirp_mass)
    for lo, hi in ((90.0, 110.0), (900.0, 1400.0), (2000.0, 4096.0)):
        chunk = np.unique(np.concatenate([np.linspace(lo, hi, 20001), edges[(edges >= lo) & (edges <= hi)]]))
        ec = fid.at(chirp_mass, chunk)
        i = np.searchsorted(chunk, edges[(edges >= lo) & (edges <= hi)])
        np.testing.assert_allclose(ec[i], e[(edges >= lo) & (edges <= hi)], atol=1e-9)
        assert np.max(np.abs(np.angle(ec[1:] / ec[:-1]))) < 0.5  # no jump at the cut
