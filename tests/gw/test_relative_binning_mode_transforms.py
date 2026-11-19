"""Training transforms of the per-m-group input (relative_binning_mode_transforms): the
division by the per-group references, the t* definition, the projection, x = K q.
The comparison with exact data summaries is in dingo-BNS-ET (Stage D test H4n)."""

import numpy as np
import pytest

pytest.importorskip("mlgw_bns")
pytest.importorskip("bilby_xG")

from bilby_xG.networks import InterferometerList

from dingo.gw.domains import RelativeBinningDomain
from dingo.gw.relative_binning_mode_tables import (
    ModeGroupLookup,
    ModeGroupTables,
    build_lookup,
)
from dingo.gw.transforms import ProjectOntoDetectorsFrequencyDependent
from dingo.gw.transforms.relative_binning_mode_transforms import (
    ModeGroupProxyConditioning,
    ProjectModeGroupsFrequencyDependent,
    SummarizeWhitenModeGroups,
    flatten_input,
    group_ratios_to_q,
)
from dingo.gw.transforms.relative_binning_transforms import (
    FiducialPhase,
    RelativeBinningProxyConditioning,
)
from dingo.gw.waveform_generator import WaveformGenerator

EDGES = np.geomspace(5.0, 4096.0, 300)
MODES = [(2, 1), (2, 2), (3, 3), (4, 4)]
T = 2.0**15
DELTA = 1e-4
KERNEL = {"chirp_mass": "bilby.core.prior.Uniform(minimum=-2.5e-4, maximum=2.5e-4)",
          "t_star": "bilby.core.prior.Uniform(minimum=-0.003, maximum=0.003)"}
THETA = dict(chirp_mass=1.40013, mass_ratio=0.8, chi_1=0.01, chi_2=-0.02, lambda_1=300.0,
             lambda_2=500.0, luminosity_distance=100.0, theta_jn=1.0, phase=0.4,
             geocent_time=0.0)
IFOS = ["ET_1L_IT", "ET_1L_DE"]
RESPONSE = dict(earth_rotation_time_delay=True, earth_rotation_beam_patterns=True,
                finite_size=True, orbital_motion=False)


def _psd(f):
    return 1e-48 * ((f / 10.0) ** -4 + 1.0 + (f / 1000.0) ** 2)


def _fids():
    return [FiducialPhase(WaveformGenerator("mlgw_bns", RelativeBinningDomain(EDGES), 100.0),
                          EDGES, mode=m) for m in MODES]


@pytest.fixture(scope="module")
def lookup(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("lk") / "lookup")
    tab = ModeGroupTables(_fids(), _psd, T, 0.05)
    build_lookup(tab, 1.3992, 1.4010, DELTA, path, num_processes=2, chunk=4, print_output=False)
    return ModeGroupLookup(path, tab.settings())


@pytest.fixture(scope="module")
def stored():
    """Per-m polarizations at the edges divided by their own fiducial (the dataset)."""
    wfg = WaveformGenerator("mlgw_bns", RelativeBinningDomain(EDGES), 100.0, mode_list=MODES,
                            mode_max_time_to_merger=T)
    pol = wfg.generate_hplus_hcross_m(THETA)
    out, raw = {}, {}
    for (ell, m), fid in zip(MODES, _fids()):
        e = fid(THETA["chirp_mass"])[0]
        for p in ("h_plus", "h_cross"):
            raw[f"{p}_m{m}"] = pol[m][p]
            out[f"{p}_m{m}"] = pol[m][p] * np.conj(e)
    return out, raw


def _sample(wf, n=None):
    times = {"ET_1L_IT_time": 0.012, "ET_1L_DE_time": 0.015}
    par = dict(THETA)
    ext = dict(times)
    if n is not None:
        wf = {k: np.repeat(v[None], n, 0) for k, v in wf.items()}
        par = {k: np.full(n, v) for k, v in par.items()}
        ext = {k: np.full(n, v) for k, v in ext.items()}
    return {"waveform": wf, "parameters": par, "extrinsic_parameters": ext}


def test_q_layout():
    r = np.arange(2 * 4 * 6).reshape(2, 4, 6) * (1 + 1j)
    q = group_ratios_to_q(r)
    assert q.shape == (2, 5, 8)
    np.testing.assert_allclose(q[1, 2, 2 * 3], 0.5 * (r[1, 3, 2] + r[1, 3, 3]))
    np.testing.assert_allclose(q[1, 2, 2 * 3 + 1], r[1, 3, 3] - r[1, 3, 2])
    x = flatten_input(q)
    assert x.shape == (2, 40) and x[1, 3 * 5 + 2] == q[1, 2, 3]


def test_proxy_conditioning(lookup, stored):
    st, raw = stored
    tr = ModeGroupProxyConditioning(KERNEL, lookup, EDGES, IFOS)
    out = tr(_sample(st, n=6))
    ext = out["extrinsic_parameters"]
    mc_p, t_p = ext["chirp_mass_proxy"], ext["t_star_proxy"]
    np.testing.assert_allclose(mc_p / DELTA, np.round(mc_p / DELTA), rtol=0, atol=1e-6)
    assert np.all(np.abs(ext["delta_chirp_mass"]) <= 2.5e-4 + DELTA / 2 + 1e-12)
    fids = _fids()
    _, g_p22 = fids[1](mc_p[0])
    for (ell, m), fid in zip(MODES, fids):
        e_p = fid(mc_p[0])[0]
        h_ref = EDGES ** (-7 / 6) * e_p * np.exp(-2j * np.pi * EDGES * (t_p[0] + g_p22))
        for p in ("h_plus", "h_cross"):
            assert out["waveform"][f"{p}_m{m}"].dtype == np.complex64
            np.testing.assert_allclose(out["waveform"][f"{p}_m{m}"][0], raw[f"{p}_m{m}"] / h_ref,
                                       rtol=1e-6, atol=0)
    # t*: the Stage C definition applied to the (2,2) group
    stage_c = RelativeBinningProxyConditioning(KERNEL, fids[1], IFOS)
    s22 = _sample({"h_plus": st["h_plus_m2"] * 1.0, "h_cross": st["h_cross_m2"]})
    s22["extrinsic_parameters"].update(chirp_mass_proxy=mc_p[0], t_star_proxy=t_p[0])
    ref = stage_c(s22)["extrinsic_parameters"]
    assert abs(ref["t_star"] - ext["t_star"][0]) < 1e-9
    # unbatched sample gives the same with the same proxies
    s1 = _sample(st)
    s1["extrinsic_parameters"].update(chirp_mass_proxy=mc_p[0], t_star_proxy=t_p[0])
    o1 = tr(s1)
    np.testing.assert_allclose(o1["waveform"]["h_plus_m3"], out["waveform"]["h_plus_m3"][0], rtol=1e-12)
    # standardization path (no waveform)
    s0 = _sample(st)
    del s0["waveform"]
    e0 = tr(s0)["extrinsic_parameters"]
    assert abs(e0["delta_chirp_mass"]) <= 2.5e-4 + DELTA / 2 + 1e-12


def test_projection_is_linear(stored):
    st, _ = stored
    ifos = InterferometerList(IFOS)
    dom = RelativeBinningDomain(EDGES)
    ext = dict(luminosity_distance=np.array([200.0, 400.0]), ra=np.array([1.0, 2.0]),
               dec=np.array([0.3, -0.5]), psi=np.array([0.2, 1.1]),
               geocent_time=np.array([0.01, -0.02]), ET_1L_IT_time=np.array([0.0, 0.0]),
               ET_1L_DE_time=np.array([0.0, 0.0]))
    par = {k: np.full(2, v) for k, v in THETA.items()}
    wf = {k: np.repeat(v[None], 2, 0) for k, v in st.items()}
    per_m = ProjectModeGroupsFrequencyDependent(ifos, dom, 1.4e9, RESPONSE, [1, 2, 3, 4])(
        {"waveform": wf, "parameters": dict(par), "extrinsic_parameters": dict(ext)})
    tot = {"h_plus": sum(wf[f"h_plus_m{m}"] for m in (1, 2, 3, 4)),
           "h_cross": sum(wf[f"h_cross_m{m}"] for m in (1, 2, 3, 4))}
    total = ProjectOntoDetectorsFrequencyDependent(ifos, dom, 1.4e9, RESPONSE)(
        {"waveform": tot, "parameters": dict(par), "extrinsic_parameters": dict(ext)})
    for nm in IFOS:
        assert per_m["waveform"][nm].shape == (2, 4, len(EDGES))
        assert per_m["waveform"][nm].dtype == np.complex64  # training data in float32
        np.testing.assert_allclose(per_m["waveform"][nm].sum(axis=1), total["waveform"][nm],
                                   rtol=0, atol=1e-6 * np.max(np.abs(total["waveform"][nm])))
    assert per_m["parameters"]["ra"][1] == 2.0 and "ra" not in per_m["extrinsic_parameters"]


def test_summarize(lookup):
    rng = np.random.default_rng(0)
    r = rng.standard_normal((3, 4, len(EDGES))) + 1j * rng.standard_normal((3, 4, len(EDGES)))
    mc_p = np.array([1.4, 1.4001, 1.4005])
    s = {"waveform": {"ET_1L_IT": r}, "extrinsic_parameters": {"chirp_mass_proxy": mc_p}}
    out = SummarizeWhitenModeGroups({"ET_1L_IT": lookup})(s)
    x = out["waveform"]["ET_1L_IT"]
    assert x.shape == (3, 8 * (len(EDGES) - 1)) and out["asds"]["ET_1L_IT"].shape == x.shape
    q = group_ratios_to_q(r)
    want = flatten_input(np.einsum("nbij,nbj->nbi", lookup.K(mc_p).astype(np.complex128), q))
    assert x.dtype == np.complex64  # training data in float32
    np.testing.assert_allclose(x, want, rtol=0, atol=1e-6 * np.max(np.abs(want)))


def test_tstar_inference_matches_training(lookup, stored):
    """The inference map t* -> geocent_time (`GroupDelayAtFStar`, from h_plus at theta_jn 0,
    where only the m = 2 group contributes) uses the training's t* definition."""
    from dingo.gw.inference.steps_relative_binning import GroupDelayAtFStar

    st, _ = stored
    tr = ModeGroupProxyConditioning(KERNEL, lookup, EDGES, IFOS)
    ext = tr(_sample(st))["extrinsic_parameters"]
    t_centre = np.mean([0.012, 0.015])
    domain = RelativeBinningDomain(EDGES, settings={"f_star": 100.0})
    metadata = {"dataset_settings": {
        "domain": domain.domain_dict,
        "waveform_generator": dict(approximant="mlgw_bns", f_ref=100.0,
                                   mode_list=[list(m) for m in MODES],
                                   mode_max_time_to_merger=T)}}
    G = GroupDelayAtFStar(metadata).one({k: THETA[k] for k in (
        "chirp_mass", "mass_ratio", "chi_1", "chi_2", "lambda_1", "lambda_2")})
    # the training interpolates the fiducial phases of the lookup in Mc (4 points)
    assert abs((t_centre - G) - ext["t_star"]) < 1e-7, (t_centre - G, ext["t_star"])


def test_data_input_whitening():
    """WhitenModeGroupSummary = ModeGroupTables.data_input (same seed), flattened as the
    training input."""
    from dingo.gw.relative_binning_mode_data import to_table_layout
    from dingo.gw.transforms.relative_binning_mode_transforms import WhitenModeGroupSummary

    tab = ModeGroupTables(_fids(), _psd, T, 0.05)
    rng = np.random.default_rng(3)
    A = rng.standard_normal((4, 2, len(EDGES) - 1)) + 1j * rng.standard_normal((4, 2, len(EDGES) - 1))
    tr = WhitenModeGroupSummary({"ET_1L_IT": tab}, 1.4, seed=7)
    out = tr({"waveform": {"ET_1L_IT": A}})
    ref = tab.data_input(to_table_layout(A), 1.4, rng=np.random.default_rng(7))
    np.testing.assert_allclose(out["waveform"]["ET_1L_IT"], flatten_input(ref), rtol=1e-12)
    assert out["asds"]["ET_1L_IT"].shape == out["waveform"]["ET_1L_IT"].shape
    # the same input at every call (fixed seed)
    np.testing.assert_array_equal(tr({"waveform": {"ET_1L_IT": A}})["waveform"]["ET_1L_IT"],
                                  out["waveform"]["ET_1L_IT"])
