"""mlgw_bns per m group: generate_hplus_hcross_m (one bilby_xG mode per m), the
time-to-merger taper of each mode (mode_max_time_to_merger), the per-mode fiducial
phase, and a dataset with polarizations_per_m."""

import bilby
import numpy as np
import pytest
import yaml

pytest.importorskip("mlgw_bns")
pytest.importorskip("bilby_xG")

from bilby_xG.source import convert_to_mlgw_bns_parameters, mlgw_bns_individual_modes

from dingo.gw.dataset.generate_dataset import _generate_dataset_main
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.domains import RelativeBinningDomain
from dingo.gw.transforms.relative_binning_transforms import FiducialPhase
from dingo.gw.waveform_generator import (
    WaveformGenerator,
    mlgw_bns_model,
    mode_time_taper,
    sum_contributions_m,
)

EDGES = np.geomspace(3.0, 4096.0, 400)
PARAMETERS = [
    dict(chirp_mass=1.25, mass_ratio=0.9, chi_1=0.02, chi_2=-0.01, lambda_1=400.0,
         lambda_2=600.0, luminosity_distance=200.0, theta_jn=0.7, phase=1.3),
    dict(chirp_mass=2.0, mass_ratio=0.5, chi_1=-0.04, chi_2=0.05, lambda_1=1.0,
         lambda_2=2000.0, luminosity_distance=800.0, theta_jn=2.2, phase=5.0),
]
# 1.6 + 1.6 Msun
EQUAL = dict(PARAMETERS[0], chirp_mass=3.2 * 0.25**0.6, mass_ratio=1.0)
T_SEGMENT = 2.0**15


def _modes(parameters, f):
    """bilby_xG per-mode plus, cross and time to merger at f (relaxed mass range)."""
    mlgw_bns_model()
    p, added = convert_to_mlgw_bns_parameters(parameters)
    return mlgw_bns_individual_modes(f, **{k: p[k] for k in added})


def test_mode_time_taper():
    """1 up to tau_max (1 - alpha), cosine, 0 from tau_max; continuous at both ends."""
    tau_max, alpha = 1000.0, 0.1
    t0 = tau_max * (1 - alpha)
    w = mode_time_taper(np.array([0.0, 500.0, t0, 925.0, 950.0, 975.0, tau_max, 2e3]),
                        tau_max, alpha)
    np.testing.assert_array_equal(w[[0, 1, 2]], 1.0)
    np.testing.assert_allclose(w[[3, 4, 5]], 0.5 * (1 + np.cos(np.pi * np.array(
        [0.25, 0.5, 0.75]))), rtol=1e-14)
    np.testing.assert_array_equal(w[[6, 7]], 0.0)
    eps = 1e-6
    for edge, value in ((t0, 1.0), (tau_max, 0.0)):
        np.testing.assert_allclose(
            mode_time_taper(np.array([edge - eps, edge + eps]), tau_max, alpha), value,
            atol=1e-12)
    for bad in ((0.0, 0.1), (1000.0, 0.0), (1000.0, 1.5)):
        with pytest.raises(ValueError):
            mode_time_taper(1.0, *bad)


def test_taper_options():
    """The taper is mlgw_bns only, and invalid settings fail at construction."""
    with pytest.raises(ValueError, match="not implemented"):
        WaveformGenerator("IMRPhenomXAS_NRTidalv3", RelativeBinningDomain(EDGES), 100.0,
                          mode_max_time_to_merger=T_SEGMENT)
    with pytest.raises(ValueError):
        WaveformGenerator("mlgw_bns", RelativeBinningDomain(EDGES), 100.0,
                          mode_max_time_to_merger=T_SEGMENT, mode_taper_fraction=0.0)


@pytest.mark.parametrize("parameters", PARAMETERS)
def test_groups_equal_bilby_xg_modes(parameters):
    """Without the taper: one group per mode, keys m = 1..4, each equal to bilby_xG's
    plus/cross of that mode; the groups sum to generate_hplus_hcross."""
    wfg = WaveformGenerator("mlgw_bns", RelativeBinningDomain(EDGES), 100.0)
    pol_m = wfg.generate_hplus_hcross_m(parameters)
    assert sorted(pol_m) == [1, 2, 3, 4]
    modes = _modes(parameters, EDGES)
    for mode, h in modes.items():
        m = int(mode.split(",")[1])
        for k, name in (("h_plus", "plus"), ("h_cross", "cross")):
            assert pol_m[m][k].dtype == np.complex128
            np.testing.assert_array_equal(pol_m[m][k], h[name])
    total = wfg.generate_hplus_hcross(parameters)
    for k in ("h_plus", "h_cross"):
        np.testing.assert_allclose(sum(pol_m[m][k] for m in pol_m), total[k],
                                   rtol=1e-14, atol=0)


def test_group_phase_shift():
    """Each group transforms exactly as exp(-1j m phase): the shifted sum equals a
    direct call at the shifted phase."""
    wfg = WaveformGenerator("mlgw_bns", RelativeBinningDomain(EDGES), 100.0,
                            mode_max_time_to_merger=T_SEGMENT)
    p, d = PARAMETERS[1], 0.7
    shifted = sum_contributions_m(wfg.generate_hplus_hcross_m(p), phase_shift=d)
    direct = wfg.generate_hplus_hcross(dict(p, phase=p["phase"] + d))
    for k in ("h_plus", "h_cross"):
        np.testing.assert_allclose(shifted[k], direct[k], rtol=1e-10, atol=1e-30)


def test_taper_cuts_44_not_22_at_3hz():
    """tau_max = 2^15 s: at 3 Hz the (4,4) mode (tau about 1.6e5 s) is zero, the (2,2)
    (tau about 2e4 s for 1.6 + 1.6 Msun) is not; each tapered group is the bilby_xG
    mode times the taper of its own time to merger; the sum is the sum of the groups."""
    wfg = WaveformGenerator("mlgw_bns", RelativeBinningDomain(EDGES), 100.0,
                            mode_max_time_to_merger=T_SEGMENT)
    pol_m = wfg.generate_hplus_hcross_m(EQUAL)
    modes = _modes(EQUAL, EDGES)
    assert modes["4,4"]["time_to_merger"][0] > T_SEGMENT
    assert 1e4 < modes["2,2"]["time_to_merger"][0] < 3e4
    for k in ("h_plus", "h_cross"):
        assert pol_m[4][k][0] == 0 and abs(modes["4,4"][k[2:]][0]) > 0
        assert pol_m[2][k][0] == modes["2,2"][k[2:]][0] != 0
    for mode, h in modes.items():
        m = int(mode.split(",")[1])
        w = mode_time_taper(h["time_to_merger"], T_SEGMENT, 0.05)
        assert w.min() == 0.0 if m in (3, 4) else w.min() == 1.0
        np.testing.assert_array_equal(pol_m[m]["h_plus"], w * h["plus"])
    total = wfg.generate_hplus_hcross(EQUAL)
    np.testing.assert_allclose(sum(pol_m[m]["h_cross"] for m in pol_m),
                               total["h_cross"], rtol=1e-14, atol=0)


def test_fiducial_phase_per_mode():
    """FiducialPhase(mode=(2, 2)) is -1 times the h_plus fiducial (theta_jn = 0,
    azimuth pi/2) everywhere, with the same group delay; the group delay of each mode
    is its time to merger at f_star. A LAL approximant with a mode is rejected."""
    f = np.geomspace(3.0, 4096.0, 2000)
    mc = 1.4
    wfg = lambda: WaveformGenerator("mlgw_bns", RelativeBinningDomain(f), 100.0)
    e22, g22 = FiducialPhase(wfg(), f)(mc)
    e, g = FiducialPhase(wfg(), f, mode=(2, 2))(mc)
    np.testing.assert_allclose(e, -e22, rtol=0, atol=1e-9)
    assert abs(g - g22) < 1e-9 * g22
    fid = dict(chirp_mass=mc, mass_ratio=0.75, chi_1=0.0, chi_2=0.0, lambda_1=0.0,
               lambda_2=0.0, luminosity_distance=100.0, theta_jn=0.0, phase=0.0)
    tau = _modes(fid, np.array([100.0]))
    for ell, m in ((2, 1), (3, 3), (4, 4)):
        e, g = FiducialPhase(wfg(), f, mode=(ell, m))(mc)
        assert np.all(np.isfinite(e)) and np.allclose(np.abs(e), 1.0, rtol=1e-12)
        np.testing.assert_allclose(g, tau[f"{ell},{m}"]["time_to_merger"][0], rtol=1e-4)
        # at() on a chunk equals __call__ on the same frequencies
        np.testing.assert_allclose(FiducialPhase(wfg(), f, mode=(ell, m)).at(
            mc, f[1000:1010]), e[1000:1010], rtol=0, atol=1e-9)
    with pytest.raises(ValueError, match="mlgw_bns"):
        FiducialPhase(WaveformGenerator("IMRPhenomXAS_NRTidalv3",
                                        RelativeBinningDomain(f), 100.0), f, mode=(2, 2))


def test_group_ratio_smooth():
    """h_group / exp(i phi_fid,m(f; Mc~)) for Mc~ = Mc + 1e-4: its phase changes by
    much less than 2 pi between neighbouring edges above 20 Hz (the group alone, or
    the ratio with the opposite sign, by more than pi)."""
    f = np.geomspace(3.0, 4096.0, 2000)
    p = PARAMETERS[0]
    wfg = WaveformGenerator("mlgw_bns", RelativeBinningDomain(f), 100.0,
                            mode_max_time_to_merger=T_SEGMENT)
    pol_m = wfg.generate_hplus_hcross_m(p)
    band = (f > 20.0) & (f < 1000.0)
    for ell, m in ((2, 1), (2, 2), (3, 3), (4, 4)):
        e, _ = FiducialPhase(WaveformGenerator("mlgw_bns", RelativeBinningDomain(f),
                                               100.0), f, mode=(ell, m))(
            p["chirp_mass"] + 1e-4)
        h = pol_m[m]["h_plus"]
        step = lambda x: np.max(np.abs(np.angle(x[band][1:] / x[band][:-1])))
        assert step(h * np.conj(e)) < 0.2, (m, step(h * np.conj(e)))
        assert step(h) > np.pi / 2 and step(h * e) > np.pi / 2


def test_dataset_per_m(tmp_path):
    """polarizations_per_m: keys f"{pol}_m{m}", each group divided by its own
    fiducial mode phase at the waveform's chirp mass; groups times their factors sum
    to the (tapered) summed waveform."""
    edges = np.geomspace(3.0, 4096.0, 300)
    wg = {"approximant": "mlgw_bns", "f_ref": 100.0,
          "mode_max_time_to_merger": T_SEGMENT, "mode_taper_fraction": 0.05}
    settings = {
        "domain": {"type": "RelativeBinningDomain", "bin_edges": edges.tolist(),
                   "settings": {"f_star": 100.0}},
        "waveform_generator": wg,
        "polarizations_per_m": True,
        "intrinsic_prior": {
            "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=1.01, maximum=2.76)",
            "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.5, maximum=1.0)",
            "chi_1": "bilby.gw.prior.AlignedSpin(a_prior=Uniform(0, 0.05))",
            "chi_2": "bilby.gw.prior.AlignedSpin(a_prior=Uniform(0, 0.05))",
            "lambda_1": "bilby.core.prior.Uniform(0, 2000)",
            "lambda_2": "bilby.core.prior.Uniform(0, 2000)",
            "theta_jn": "default", "phase": "default",
            "luminosity_distance": 100.0, "geocent_time": 0.0,
        },
        "num_samples": 4,
    }
    bilby.core.utils.random.seed(7)
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(settings))
    _generate_dataset_main(str(tmp_path / "s.yaml"), str(tmp_path / "wfd.hdf5"),
                           num_processes=1)
    wfd = WaveformDataset(file_name=str(tmp_path / "wfd.hdf5"))
    assert len(wfd) == 4
    assert sorted(wfd.polarizations) == sorted(
        f"{k}_m{m}" for k in ("h_plus", "h_cross") for m in (1, 2, 3, 4))
    p = wfd.parameters.iloc[1].to_dict()
    wfg = WaveformGenerator(domain=wfd.domain, **wg)
    pol_m = wfg.generate_hplus_hcross_m(p)
    total = wfg.generate_hplus_hcross(p)
    rebuilt = 0
    for ell, m in ((2, 2), (2, 1), (3, 3), (4, 4)):
        e, _ = FiducialPhase(WaveformGenerator("mlgw_bns", wfd.domain, 100.0), edges,
                             mode=(ell, m))(p["chirp_mass"])
        stored = wfd.polarizations[f"h_plus_m{m}"][1]
        assert stored.dtype == np.complex128
        np.testing.assert_allclose(stored, pol_m[m]["h_plus"] * np.conj(e),
                                   rtol=1e-6, atol=1e-30)
        rebuilt = rebuilt + stored * e
    np.testing.assert_allclose(rebuilt, total["h_plus"], rtol=1e-6,
                               atol=1e-6 * np.max(np.abs(total["h_plus"])))
