"""Relative-binning inference: the likelihood uses the same ratio r = h_det / h_ref as the
training chain (same waveform, projection, time and reference conventions); its (2, 2)
phase terms reproduce the direct likelihood; the sampler context whitens event summaries
into the training input and checks the proxies."""

import bilby
import numpy as np
import pytest
import torch
import yaml

from dingo.gw.dataset.generate_dataset import _generate_dataset_main
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.gwutils import get_interferometer_list
from dingo.gw.inference.context import GWSamplerContext
from dingo.gw.inference.context_relative_binning import RelativeBinningSamplerContext
from dingo.gw.likelihood_relative_binning import RelativeBinningLikelihood, gram_to_bmat
from dingo.gw.transforms import GetDetectorTimes, ProjectOntoDetectors
from dingo.gw.transforms.relative_binning_transforms import (
    FiducialPhase,
    RelativeBinningProxyConditioning,
    summarize_and_whiten,
    whitening_from_gram,
)
from dingo.gw.waveform_generator import WaveformGenerator

pytest.importorskip("bilby_xG")
DETECTORS = ["H1", "L1"]
REF_TIME = 1126259462.391
KERNEL = {
    "chirp_mass": "bilby.core.prior.Uniform(minimum=-3e-4, maximum=3e-4)",
    "t_star": "bilby.core.prior.Uniform(minimum=-3e-3, maximum=3e-3)",
}
SETTINGS = {
    "domain": {"type": "RelativeBinningDomain", "settings": {
        "detectors": DETECTORS, "f_min": 20.0, "f_max": 1024.0, "f_star": 100.0,
        "ref_time": REF_TIME, "kernel": KERNEL,
        "bin_selection": {"eta": 1.0, "n_train": 4, "n_validation": 2, "n_samples": 2,
                          "n_grid": 3001, "snr_range": [10.0, 30.0]}}},
    "waveform_generator": {"approximant": "IMRPhenomXAS_NRTidalv3", "f_ref": 100.0,
                           "spin_conversion_phase": 0.0},
    "intrinsic_prior": {
        "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=2.0, maximum=2.6)",
        "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.6, maximum=1.0)",
        "chi_1": "bilby.gw.prior.AlignedSpin(a_prior=Uniform(0, 0.05))",
        "chi_2": "bilby.gw.prior.AlignedSpin(a_prior=Uniform(0, 0.05))",
        "lambda_1": "bilby.core.prior.Uniform(0, 3000)",
        "lambda_2": "bilby.core.prior.Uniform(0, 3000)",
        "theta_jn": "default", "phase": "default",
        "luminosity_distance": 100.0, "geocent_time": 0.0,
    },
    "num_samples": 4,
}
EXT = dict(ra=1.1, dec=-0.4, psi=0.6, geocent_time=0.004, luminosity_distance=300.0)


@pytest.fixture(scope="module")
def wfd(tmp_path_factory):
    bilby.core.utils.random.seed(3)
    d = tmp_path_factory.mktemp("rbi")
    (d / "s.yaml").write_text(yaml.safe_dump(SETTINGS))
    _generate_dataset_main(str(d / "s.yaml"), str(d / "wfd.hdf5"), num_processes=2)
    return WaveformDataset(file_name=str(d / "wfd.hdf5"))


def _training_r(wfd, proxies):
    """r = h_det / h_ref at the edges from the training chain (row 0, EXT, fixed proxies)."""
    p = wfd.parameters.iloc[0].to_dict()
    stored = {k: v[0] for k, v in wfd.polarizations.items()}
    wg = wfd.settings["waveform_generator"]
    fid = FiducialPhase(WaveformGenerator(wg["approximant"], wfd.domain, wg["f_ref"]),
                        wfd.domain())
    ifos = get_interferometer_list(DETECTORS)
    s = {"parameters": dict(p), "waveform": stored,
         "extrinsic_parameters": dict(EXT, **proxies)}
    s = GetDetectorTimes(ifos, REF_TIME)(s)
    s = RelativeBinningProxyConditioning(KERNEL, fid, DETECTORS)(s)
    s = ProjectOntoDetectors(ifos, wfd.domain, REF_TIME)(s)
    return s["waveform"], {**p, **EXT}


def _likelihood(wfd, theta, proxies, noise_free=True):
    lik = RelativeBinningLikelihood(
        wfd.domain, wfd.settings["waveform_generator"],
        {n: np.zeros(2 * wfd.domain.num_bins, complex) for n in DETECTORS},
        proxies, DETECTORS, REF_TIME)
    if noise_free:  # A = Bmat q of the true signal
        q = lik._q(theta)
        lik.A = {n: np.einsum("bij,bj->bi", lik.bmat[n], q[n]) for n in DETECTORS}
    return lik


def test_likelihood_ratio_equals_training_ratio(wfd):
    p0 = wfd.parameters.iloc[0]
    proxies = {"chirp_mass_proxy": p0["chirp_mass"] + 1e-4, "t_star_proxy": -0.5}
    r_train, theta = _training_r(wfd, proxies)
    lik = _likelihood(wfd, theta, proxies)
    q = lik._q(theta)
    for n in DETECTORS:
        r = r_train[n]
        q_train = np.stack([0.5 * (r[:-1] + r[1:]), r[1:] - r[:-1]], -1)
        np.testing.assert_allclose(q[n], q_train, rtol=1e-6, atol=1e-12 * np.abs(q_train).max())


def test_phase_terms_and_truth(wfd):
    p0 = wfd.parameters.iloc[0]
    proxies = {"chirp_mass_proxy": p0["chirp_mass"], "t_star_proxy": -0.5}
    theta = {**p0.to_dict(), **EXT}
    lik = _likelihood(wfd, theta, proxies)
    t = lik._terms(lik._q(theta))
    # noise-free data at the truth: log L = <h|h> / 2
    assert lik.log_likelihood(theta) == pytest.approx(t["h_inner_h"] / 2, rel=1e-9)
    terms = lik.phase_grid_terms_22(theta)
    for phase in (0.3, 2.0):
        direct = lik.log_likelihood({**theta, "phase": phase})
        assert lik.log_likelihood_22_from_terms(terms, np.array([phase]))[0] == pytest.approx(
            direct, abs=1e-6 * abs(direct))


def test_context(wfd):
    p0 = wfd.parameters.iloc[0]
    proxies = {"chirp_mass_proxy": float(p0["chirp_mass"]), "t_star_proxy": -0.5}
    theta = {**p0.to_dict(), **EXT}
    lik = _likelihood(wfd, theta, proxies)
    summary = {n: lik.A[n].T.reshape(-1) for n in DETECTORS}
    metadata = {"dataset_settings": wfd.settings,
                "train_settings": {"data": {"detectors": DETECTORS, "ref_time": REF_TIME,
                                            "extrinsic_prior": {}}}}
    ctx = GWSamplerContext.from_model_metadata(
        metadata, {"waveform": summary}, {"relative_binning_proxies": proxies})
    assert isinstance(ctx, RelativeBinningSamplerContext)
    x = ctx.prepared_data().numpy()
    # whitened noise-free summary = training input L^T q
    for i, n in enumerate(DETECTORS):
        r = lik._q(theta)[n]
        r_edges = np.concatenate([r[:, 0] - r[:, 1] / 2, [r[-1, 0] + r[-1, 1] / 2]])
        expect = summarize_and_whiten(r_edges, whitening_from_gram(wfd.domain.gram[n]))
        np.testing.assert_allclose(x[i, 0] + 1j * x[i, 1], expect, rtol=1e-5, atol=1e-5)
    assert isinstance(ctx.likelihood(), RelativeBinningLikelihood)
    assert ctx.likelihood().detector_response is None
    with pytest.raises(ValueError):
        ctx.prepared_data({"chirp_mass_proxy": torch.tensor([proxies["chirp_mass_proxy"] + 1e-3])})

    # Stage C: the IS likelihood takes the detector response of the training, unless the
    # event metadata gives another one (e.g. a network trained without it, data with it).
    on = {"earth_rotation_time_delay": True, "earth_rotation_beam_patterns": True,
          "finite_size": True}
    meta_on = {**metadata, "train_settings": {"data": {**metadata["train_settings"]["data"],
                                                       "detector_response": on}}}
    ctx = GWSamplerContext.from_model_metadata(
        meta_on, {"waveform": summary}, {"relative_binning_proxies": proxies})
    assert ctx.likelihood().detector_response == {**on, "orbital_motion": False}
    ctx = GWSamplerContext.from_model_metadata(
        metadata, {"waveform": summary},
        {"relative_binning_proxies": proxies, "detector_response": {"finite_size": True}})
    assert ctx.likelihood().detector_response["finite_size"] is True
    assert ctx.likelihood().detector_response["earth_rotation_beam_patterns"] is False


def _conditioned(wfd, proxies):
    """Training chain on row 0 up to the proxy conditioning (EXT, fixed proxies)."""
    p = wfd.parameters.iloc[0].to_dict()
    stored = {k: v[0] for k, v in wfd.polarizations.items()}
    wg = wfd.settings["waveform_generator"]
    fid = FiducialPhase(WaveformGenerator(wg["approximant"], wfd.domain, wg["f_ref"]),
                        wfd.domain())
    ifos = get_interferometer_list(DETECTORS)
    s = {"parameters": dict(p), "waveform": stored,
         "extrinsic_parameters": dict(EXT, **proxies)}
    s = GetDetectorTimes(ifos, REF_TIME)(s)
    return RelativeBinningProxyConditioning(KERNEL, fid, DETECTORS)(s), p


def test_delta_t_star_in_training(wfd):
    proxies = {"chirp_mass_proxy": float(wfd.parameters.iloc[0]["chirp_mass"]),
               "t_star_proxy": -0.5}
    s, _ = _conditioned(wfd, proxies)
    ext = s["extrinsic_parameters"]
    assert ext["delta_t_star"] == pytest.approx(ext["t_star"] - proxies["t_star_proxy"], abs=1e-12)
    # standardization path (no waveform): delta_t_star = -kernel draw, inside the kernel
    p = wfd.parameters.iloc[0].to_dict()
    s = {"parameters": dict(p), "extrinsic_parameters": dict(EXT)}
    s = GetDetectorTimes(get_interferometer_list(DETECTORS), REF_TIME)(s)
    wg = wfd.settings["waveform_generator"]
    fid = FiducialPhase(WaveformGenerator(wg["approximant"], wfd.domain, wg["f_ref"]),
                        wfd.domain())
    ext = RelativeBinningProxyConditioning(KERNEL, fid, DETECTORS)(s)["extrinsic_parameters"]
    assert abs(ext["delta_t_star"]) <= 3e-3


def test_t_star_to_geocent_time(wfd):
    from types import SimpleNamespace

    from dingo.gw.inference.steps_relative_binning import (
        TStarToGeocentTime, relative_binning_time_steps)

    proxies = {"chirp_mass_proxy": float(wfd.parameters.iloc[0]["chirp_mass"]),
               "t_star_proxy": -0.5}
    s, p = _conditioned(wfd, proxies)
    t_star = s["extrinsic_parameters"]["t_star"]
    metadata = {"dataset_settings": wfd.settings,
                "train_settings": {"data": {"detectors": DETECTORS, "ref_time": REF_TIME}}}
    step = TStarToGeocentTime(metadata)
    ctx = SimpleNamespace(t_ref=REF_TIME)
    keys = ["chirp_mass", "mass_ratio", "chi_1", "chi_2", "lambda_1", "lambda_2"]
    given = {k: torch.tensor([p[k]], dtype=torch.float64) for k in keys}
    given.update({"ra@t_ref": torch.tensor([EXT["ra"]], dtype=torch.float64),
                  "dec": torch.tensor([EXT["dec"]], dtype=torch.float64),
                  "t_star": torch.tensor([t_star], dtype=torch.float64)})
    # the training definition of t* (true phase and inclination) maps back to the true
    # geocent_time: the neutral phase and inclination of the step do not change G
    tc = step.forward(given, ctx)["geocent_time"]
    assert tc.item() == pytest.approx(EXT["geocent_time"], abs=1e-9)
    back = step.inverse({"geocent_time": tc}, ctx, given)["t_star"]
    assert back.item() == pytest.approx(t_star, abs=1e-9)
    # the chain gets the step only for a delta_t_star network with the t* proxy
    assert len(relative_binning_time_steps(
        metadata, ["delta_chirp_mass", "delta_t_star"],
        ["chirp_mass_proxy", "t_star_proxy"])) == 1
    assert relative_binning_time_steps(
        metadata, ["delta_chirp_mass", "geocent_time"],
        ["chirp_mass_proxy", "t_star_proxy"]) == []
