"""WaveformGenerator with approximant "mlgw_bns": the polarizations of the mlgw_bns
surrogate (through bilby_xG), summed over the modes, at the domain frequencies; the
relative-binning fiducial phase and the dataset generation work with it."""

import bilby
import numpy as np
import pytest
import yaml

pytest.importorskip("mlgw_bns")
pytest.importorskip("bilby_xG")

from bilby_xG.source import convert_to_mlgw_bns_parameters, mlgw_bns_individual_modes
from bilby_xG.waveform_generator import WaveformGenerator as XGWaveformGenerator

from dingo.gw.dataset.generate_dataset import _generate_dataset_main
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.domains import RelativeBinningDomain, UniformFrequencyDomain
from dingo.gw.transforms.relative_binning_transforms import FiducialPhase
from dingo.gw.waveform_generator import WaveformGenerator

EDGES = np.geomspace(3.0, 4096.0, 400)
# BNS in the detector frame; the second one has a total mass of 4.93 Msun, outside the
# stored 2-4 Msun range of the surrogate.
PARAMETERS = [
    dict(chirp_mass=1.25, mass_ratio=0.9, chi_1=0.02, chi_2=-0.01, lambda_1=400.0,
         lambda_2=600.0, luminosity_distance=200.0, theta_jn=0.7, phase=1.3),
    dict(chirp_mass=2.0, mass_ratio=0.5, chi_1=-0.04, chi_2=0.05, lambda_1=1.0,
         lambda_2=2000.0, luminosity_distance=800.0, theta_jn=2.2, phase=5.0),
]


def _bilby_xg(parameters, modes=None):
    """h_plus, h_cross at EDGES from the bilby_xG waveform generator (sum over modes)."""
    arguments = {"frequency_bin_edges": EDGES}
    if modes is not None:
        arguments["mode_array"] = modes
    wfg = XGWaveformGenerator(
        duration=4.0, sampling_frequency=8192.0,
        frequency_domain_source_model=mlgw_bns_individual_modes,
        parameter_conversion=convert_to_mlgw_bns_parameters,
        waveform_arguments=arguments,
    )
    h = wfg.frequency_domain_strain(dict(parameters))
    return {k: sum(m[pol] for m in h.values()) for k, pol in
            (("h_plus", "plus"), ("h_cross", "cross"))}


@pytest.mark.parametrize("parameters", PARAMETERS)
def test_equals_bilby_xg(parameters):
    """Default modes (2,2), (2,1), (3,3), (4,4) on bin edges, as bilby_xG computes them."""
    wfg = WaveformGenerator("mlgw_bns", RelativeBinningDomain(EDGES), 100.0)
    h = wfg.generate_hplus_hcross(parameters)
    ref = _bilby_xg(parameters)  # after the generator relaxed the total-mass range
    for k in ("h_plus", "h_cross"):
        assert h[k].dtype == np.complex128 and np.all(np.isfinite(h[k]))
        np.testing.assert_allclose(h[k], ref[k], rtol=1e-12, atol=0)
        assert np.max(np.abs(h[k])) > 0


def test_uniform_frequency_domain():
    """Finite; zero at f = 0 and below f_min; equal to the evaluation at the grid points
    above f_min."""
    domain = UniformFrequencyDomain(f_min=20.0, f_max=1024.0, delta_f=0.25)
    wfg = WaveformGenerator("mlgw_bns", domain, 100.0)
    h = wfg.generate_hplus_hcross(PARAMETERS[0])
    i = domain.min_idx
    for k in ("h_plus", "h_cross"):
        assert len(h[k]) == len(domain) and np.all(np.isfinite(h[k]))
        assert np.all(h[k][:i] == 0) and np.all(h[k][i:] != 0)
    f = np.linspace(0.0, domain.f_max, len(domain))[i:]
    rb = WaveformGenerator("mlgw_bns", RelativeBinningDomain(f), 100.0)
    np.testing.assert_array_equal(rb.generate_hplus_hcross(PARAMETERS[0])["h_plus"],
                                  h["h_plus"][i:])


def test_mode_list():
    """mode_list [[2, 2]] gives the (2, 2) mode alone; an unknown mode is rejected."""
    wfg = WaveformGenerator("mlgw_bns", RelativeBinningDomain(EDGES), 100.0,
                            mode_list=[[2, 2]])
    h = wfg.generate_hplus_hcross(PARAMETERS[0])
    ref = _bilby_xg(PARAMETERS[0], modes=[(2, 2)])
    np.testing.assert_allclose(h["h_plus"], ref["h_plus"], rtol=1e-12, atol=0)
    assert np.max(np.abs(h["h_plus"] - _bilby_xg(PARAMETERS[0])["h_plus"])) > 0
    with pytest.raises(ValueError, match="does not provide"):
        WaveformGenerator("mlgw_bns", RelativeBinningDomain(EDGES), 100.0,
                          mode_list=[[3, 2]])


def test_phase_shift_22():
    """For the (2, 2) mode alone, a phase shift d multiplies the polarizations by
    exp(-2i d) in the bilby_xG mlgw_bns convention (azimuth pi/2 - phase). LAL models
    give exp(+2i d)."""
    wfg = WaveformGenerator("mlgw_bns", RelativeBinningDomain(EDGES), 100.0,
                            mode_list=[[2, 2]])
    d = 0.3
    p = PARAMETERS[0]
    a = wfg.generate_hplus_hcross(p)
    b = wfg.generate_hplus_hcross(dict(p, phase=p["phase"] + d))
    for h, phase in ((a, p["phase"]), (b, p["phase"] + d)):
        ref = _bilby_xg(dict(p, phase=phase), modes=[(2, 2)])
        np.testing.assert_allclose(h["h_plus"], ref["h_plus"], rtol=1e-12, atol=0)
    for k in ("h_plus", "h_cross"):
        np.testing.assert_allclose(b[k], a[k] * np.exp(-2j * d), rtol=1e-10, atol=0)


def test_fiducial_phase():
    """FiducialPhase with mlgw_bns: finite, unit modulus; its group delay at f_star and
    its phase agree with those of IMRPhenomXAS_NRTidalv3 up to a time shift of the
    merger convention (< 2 ms) and a smooth phase difference."""
    f = np.geomspace(3.0, 4096.0, 4000)
    mc = 2.5  # total mass about 5.8 Msun
    e, g = FiducialPhase(WaveformGenerator("mlgw_bns", RelativeBinningDomain(f), 100.0),
                         f)(mc)
    assert np.all(np.isfinite(e)) and np.isfinite(g)
    np.testing.assert_allclose(np.abs(e), 1.0, rtol=1e-12)
    e_lal, g_lal = FiducialPhase(
        WaveformGenerator("IMRPhenomXAS_NRTidalv3", RelativeBinningDomain(f), 100.0), f
    )(mc)
    assert abs(g - g_lal) < 2e-3
    d = np.unwrap(np.angle(e * np.conj(e_lal)))[f < 1000.0]
    assert np.max(np.abs(np.diff(d))) < 0.1


def test_dataset_generation(tmp_path):
    """A small dataset on given bin edges (no bin selection): h / exp(i Phi_fid) stored."""
    edges = np.geomspace(3.0, 4096.0, 300)
    settings = {
        "domain": {"type": "RelativeBinningDomain", "bin_edges": edges.tolist(),
                   "settings": {"f_star": 100.0}},
        "waveform_generator": {"approximant": "mlgw_bns", "f_ref": 100.0,
                               "mode_list": [[2, 2], [3, 3]]},
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
        "num_samples": 8,
    }
    bilby.core.utils.random.seed(5)
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(settings))
    _generate_dataset_main(str(tmp_path / "s.yaml"), str(tmp_path / "wfd.hdf5"),
                           num_processes=1)
    wfd = WaveformDataset(file_name=str(tmp_path / "wfd.hdf5"))
    assert len(wfd) == 8
    np.testing.assert_array_equal(wfd.domain.bin_edges, edges)
    for k in ("h_plus", "h_cross"):
        assert np.all(np.isfinite(wfd.polarizations[k]))
    p = wfd.parameters.iloc[0].to_dict()
    wg = settings["waveform_generator"]
    h = WaveformGenerator(domain=wfd.domain, **wg).generate_hplus_hcross(p)
    e, _ = FiducialPhase(WaveformGenerator(wg["approximant"], wfd.domain, wg["f_ref"]),
                         edges)(p["chirp_mass"])
    np.testing.assert_allclose(wfd.polarizations["h_plus"][0], h["h_plus"] * np.conj(e),
                               rtol=1e-6, atol=1e-30)
