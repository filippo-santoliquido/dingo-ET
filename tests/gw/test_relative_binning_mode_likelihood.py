"""Per-m-group relative-binning likelihood and its data side (Stage D step 3):
dingo.gw.relative_binning_mode_data, ModeGroupRelativeBinningLikelihood, the own time map
of the response. The accuracy on the production bins (3-4096 Hz, T = 2^15 s) is checked in
dingo-BNS-ET (Stage D test H3d)."""

import numpy as np
import pytest

pytest.importorskip("mlgw_bns")
pytest.importorskip("bilby_xG")

from dingo.gw.detector_response import (  # noqa: E402
    check_response_settings,
    component_masses,
    detector_responses,
)
from dingo.gw.domains import RelativeBinningDomain  # noqa: E402
from dingo.gw.gwutils import get_interferometer_list  # noqa: E402
from dingo.gw.likelihood_relative_binning import (  # noqa: E402
    ModeGroupRelativeBinningLikelihood,
)
from dingo.gw.relative_binning_mode_data import (  # noqa: E402
    ModeGroupReferences,
    ModeGroupSummaryAccumulator,
    to_table_layout,
)
from dingo.gw.transforms.relative_binning_transforms import FiducialPhase  # noqa: E402
from dingo.gw.waveform_generator import WaveformGenerator  # noqa: E402

DETECTORS = ["ET_1L_IT", "ET_1L_DE"]
T_REF = 1400000000.0
ALL_ON = dict(earth_rotation_time_delay=True, earth_rotation_beam_patterns=True,
              finite_size=True, orbital_motion=False)
MODES = [(2, 2), (2, 1), (3, 3), (4, 4)]
THETA = dict(chirp_mass=1.25, mass_ratio=0.8, chi_1=0.02, chi_2=-0.01, lambda_1=400.0,
             lambda_2=600.0, luminosity_distance=40.0, theta_jn=1.2, phase=0.7, ra=1.1,
             dec=-0.4, psi=0.6, geocent_time=0.012)
# small test grid: 20-1024 Hz, T = 128 s (the taper cuts the modes where tau > 128 s)
T_SEG = 128.0
EDGES = np.geomspace(20.0, 1024.0, 600)


def _psd(f):
    return 1e-48 * ((f / 10.0) ** -4 + 1.0 + (f / 1000.0) ** 2)


@pytest.fixture(scope="module")
def ifos():
    ifos = get_interferometer_list(DETECTORS)
    if not all(hasattr(i, "frequency_dependent_antenna_response") for i in ifos):
        pytest.skip("bilby_xG detectors not available")
    return ifos


def _wg(taper=True):
    return dict(approximant="mlgw_bns", f_ref=100.0, mode_list=[list(m) for m in MODES],
                mode_max_time_to_merger=T_SEG if taper else None, mode_taper_fraction=0.05)


def _domain():
    return RelativeBinningDomain(EDGES, settings={"f_star": 100.0})


def _strain(ifos, theta, f, taper=True):
    """{detector: strain} of all groups at f: own taper (if any), own time map per mode."""
    wfg = WaveformGenerator(domain=RelativeBinningDomain(f), **_wg(taper))
    pol_m = wfg.generate_mlgw_bns_waveform(theta, group_by_m=True, return_time_to_merger=True)
    m1, m2 = component_masses(theta["chirp_mass"], theta["mass_ratio"])
    out = {i.name: np.zeros(len(f), complex) for i in ifos}
    for pol in pol_m.values():
        resp = detector_responses(ifos, f, theta["ra"], theta["dec"], theta["psi"], T_REF,
                                  theta["geocent_time"], float(m1), float(m2), theta["chi_1"],
                                  theta["chi_2"], check_response_settings(ALL_ON),
                                  time_to_merger=pol["time_to_merger"])
        for n, (fp, fc) in resp.items():
            out[n] += fp * pol["h_plus"] + fc * pol["h_cross"]
    return out


def _t_star(ifos, theta):
    """Arrival time of f_star at the network centre minus the (2,2) group delay there
    (the training definition, with the exact group delay of the (2,2) group)."""
    df = 1e-3
    f = np.array([100.0 - df, 100.0 + df])
    wfg = WaveformGenerator(domain=RelativeBinningDomain(f), **_wg())
    h = wfg.generate_mlgw_bns_waveform(dict(theta, geocent_time=0.0), group_by_m=True)[2]["h_plus"]
    G = np.angle(h[1] / h[0]) / (2 * df) / (2 * np.pi)
    delay = np.mean([i.time_delay_from_geocenter(theta["ra"], theta["dec"], T_REF) for i in ifos])
    return theta["geocent_time"] + delay - G


def test_own_time_map_matches_bilby_xg(ifos):
    """Dingo's response with a mode's own time map = bilby_xG's response of per-mode
    polarizations that carry their "time_to_merger"."""
    from bilby_xG.source import convert_to_mlgw_bns_parameters, mlgw_bns_individual_modes

    f = np.geomspace(3.0, 2048.0, 801)
    p, added = convert_to_mlgw_bns_parameters(dict(THETA))
    modes = mlgw_bns_individual_modes(f, **{k: p[k] for k in added}, mode_array=MODES)
    m1, m2 = component_masses(THETA["chirp_mass"], THETA["mass_ratio"])
    ours = {i.name: 0 for i in ifos}
    for h in modes.values():
        resp = detector_responses(ifos, f, THETA["ra"], THETA["dec"], THETA["psi"], T_REF,
                                  THETA["geocent_time"], float(m1), float(m2), THETA["chi_1"],
                                  THETA["chi_2"], check_response_settings(ALL_ON),
                                  time_to_merger=h["time_to_merger"])
        for n, (fp, fc) in resp.items():
            ours[n] = ours[n] + fp * h["plus"] + fc * h["cross"]
    params = dict(ra=THETA["ra"], dec=THETA["dec"], psi=THETA["psi"], chi_1=THETA["chi_1"],
                  chi_2=THETA["chi_2"], mass_1=float(m1), mass_2=float(m2),
                  geocent_time_offset=THETA["geocent_time"], time_origin=T_REF)
    shared = {}
    for ifo in ifos:
        ref = ifo.get_detector_response_for_frequency_dependent_antenna_response(
            modes, params, T_REF, f, shared=shared, **ALL_ON)
        rel = np.max(np.abs(ours[ifo.name] - ref)) / np.max(np.abs(ref))
        assert rel < 1e-12, (ifo.name, rel)


def test_time_map_option_needs_one_mode_per_m():
    wfg = WaveformGenerator(domain=RelativeBinningDomain(EDGES), **_wg())
    # bilby_xG gives one mode per m today; a longer mode list (e.g. (3, 2)) must fail
    wfg.mode_list = [(2, 2), (3, 2)]
    with pytest.raises(ValueError):
        wfg.generate_mlgw_bns_waveform(THETA, group_by_m=True, return_time_to_merger=True)


@pytest.mark.parametrize("mode", [(2, 2), (4, 4)])
def test_fiducial_at_tau(mode):
    """at(..., return_tau) on the edges equals __call__ and phase_slope_tau."""
    edges = np.geomspace(5.0, 4096.0, 300)
    fid = FiducialPhase(WaveformGenerator("mlgw_bns", RelativeBinningDomain(edges), 100.0),
                        edges, mode=mode)
    for mc in (1.1, 1.7):
        e, tau = fid.at(mc, edges, return_tau=True)
        np.testing.assert_allclose(e, fid(mc)[0], rtol=0, atol=1e-9)
        np.testing.assert_allclose(tau, fid.phase_slope_tau(mc)[2], rtol=1e-9, atol=1e-9)


def _grid():
    k = np.arange(int(np.ceil(EDGES[0] * T_SEG)), int(np.floor(EDGES[-1] * T_SEG)) + 1)
    return k / T_SEG


def _exact(ifos, data, theta, f, taper):
    """Exact ln(L/L_n) = Re <d|h> - <h|h>/2 on the grid (own taper per template)."""
    h = _strain(ifos, theta, f, taper)
    w = 4 / T_SEG / _psd(f)
    return sum(np.sum(w * (np.conj(data[n]) * h[n]).real) - 0.5 * np.sum(w * np.abs(h[n]) ** 2)
               for n in h)


def _setup(ifos, taper):
    f = _grid()
    proxies = {"chirp_mass_proxy": THETA["chirp_mass"] + 1e-4,
               "t_star_proxy": _t_star(ifos, THETA) + 2e-4}
    refs = ModeGroupReferences(_domain(), _wg(taper))
    acc = ModeGroupSummaryAccumulator(EDGES, [i.name for i in ifos], refs.G, T_SEG)
    data = _strain(ifos, THETA, f, taper)
    r, w = refs.at(proxies, f)
    acc.add(f, data, {i.name: _psd(f) for i in ifos}, r, w)

    def lik(model):
        s, g = (acc.A, acc.gram_tapered) if model == "reference" else (acc.D, acc.gram)
        return ModeGroupRelativeBinningLikelihood(
            _domain(), _wg(taper), s, g, proxies, DETECTORS, T_REF, log_Zn=0.0,
            detector_response=ALL_ON, taper=model)

    return dict(f=f, acc=acc, data=data, lik=lik, taper=taper,
                max=_exact(ifos, data, THETA, f, taper))


@pytest.fixture(scope="module")
def untapered(ifos):
    return _setup(ifos, taper=False)


@pytest.fixture(scope="module")
def tapered(ifos):
    return _setup(ifos, taper=True)


SHIFTS = [{}, {"chirp_mass": 2e-5, "lambda_1": 150.0, "phase": 0.3},
          {"mass_ratio": -0.03, "psi": 0.2, "geocent_time": 1e-4}, {"theta_jn": 0.4}]


@pytest.mark.parametrize("shift", SHIFTS)
def test_likelihood_against_exact(ifos, untapered, shift):
    """Noise-free data, no taper, 20-1024 Hz: the RB likelihood equals the exact grid sum
    (signs, layouts, Gram, references, own time maps) up to the linear-ratio error, which
    grows with the distance from the maximum (600 edges: 3e-5 at the truth, 3.5e-5 of the
    distance at 3e4 nat below it). Both taper models are the same here."""
    u = untapered
    assert u["max"] > 200  # SNR > 20: the test is not trivial
    theta = {k: v + shift.get(k, 0.0) for k, v in THETA.items()}
    exact = _exact(ifos, u["data"], theta, u["f"], False)
    for model in ("own", "reference"):
        rb = u["lik"](model).log_likelihood(theta)
        assert abs(rb - exact) < 1e-3 + 2e-4 * (u["max"] - exact), (model, rb, exact)


def test_taper_models_at_the_truth(ifos, tapered):
    """With the taper (T = 128 s cuts the (2,2) mode near 20 Hz), at the truth: the
    reference taper is close to exact (observed -0.0064), the own taper has the
    non-linear ratio where the taper turns on (observed -0.20 with 600 edges; -0.0012
    with 2400). Regression guard; the production numbers come from test H3d."""
    exact = _exact(ifos, tapered["data"], THETA, tapered["f"], True)
    d_ref = tapered["lik"]("reference").log_likelihood(THETA) - exact
    d_own = tapered["lik"]("own").log_likelihood(THETA) - exact
    assert abs(d_ref) < 0.02, d_ref
    assert abs(d_own) < 0.4, d_own


def test_d_inner_d(untapered):
    w = 4 / T_SEG / _psd(untapered["f"])
    dd = sum(np.sum(w * np.abs(d) ** 2) for d in untapered["data"].values())
    assert np.isclose(untapered["acc"].d_inner_d, dd, rtol=1e-12)


@pytest.mark.parametrize("model", ["own", "reference"])
def test_phase_grid_terms_identity(tapered, model):
    """log L from the phase-grid terms at the sample's phase and psi = log_likelihood."""
    lik = tapered["lik"](model)
    for dphi, dpsi in [(0.0, 0.0), (1.9, -0.3), (4.0, 0.9)]:
        theta = dict(THETA, phase=(THETA["phase"] + dphi) % (2 * np.pi),
                     psi=(THETA["psi"] + dpsi) % np.pi)
        terms = lik.phase_grid_terms(theta)
        grid = lik.log_likelihood_from_phase_grid_terms(
            terms, np.array([theta["phase"]]), np.array([theta["psi"]]))
        direct = lik.log_likelihood(theta)
        assert abs(grid[0, 0] - direct) < 1e-8 * max(1.0, abs(direct)), (grid, direct)


def test_table_layout():
    a = np.arange(4 * 2 * 3).reshape(4, 2, 3)
    t = to_table_layout(a)
    assert t.shape == (3, 8)
    assert t[2, 2 * 1 + 1] == a[1, 1, 2]


def test_context_keeps_mode_group_summaries():
    """The base sampler context keeps only "waveform" and "asds" (and broadcasts them over
    batch rows); ModeGroupSamplerContext must keep "mode_group_summaries" for the
    likelihood, apart from event_data, without the 'dropping them' warning (Stage D 3d
    smoke run, 2026-11-19: KeyError in the synthetic phase)."""
    import warnings

    from dingo.gw.inference.context_relative_binning import ModeGroupSamplerContext
    from torchvision.transforms import Compose

    nb = len(EDGES) - 1
    A = {"ET_1L_IT": np.ones((4, 2, nb), complex)}
    extra = {"D": A, "gram": {"ET_1L_IT": np.ones((4, 4, 3, nb), complex)},
             "gram_tapered": {"ET_1L_IT": np.ones((4, 4, 3, nb), complex)}}
    event_data = {"waveform": A, "mode_group_summaries": extra, "parameters": {"a": 1.0}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ctx = ModeGroupSamplerContext(domain=_domain(), data_prep=Compose([]),
                                      event_data=event_data, event_metadata={})
    assert set(ctx.event_data) == {"waveform"}
    assert ctx.mode_group_summaries is extra
    assert "mode_group_summaries" in event_data  # the caller's dict is unchanged
    assert ctx.event_metadata["injection_parameters"] == {"a": 1.0}
    rows = ctx._broadcast_event(3)
    assert rows["waveform"]["ET_1L_IT"].shape == (3, 4, 2, nb)
