"""Relative-binning summary data in the training path: the bin edges and Gram matrices are
found by Dingo from hyperparameters (bilby_xG bin selection) at dataset generation and
written to the dataset and a JSON file; then set_train_transforms on the RelativeBinningDomain
(no ASD dataset), training samples and the embedding-network SVD."""

import copy
import json

import bilby
import numpy as np
import pytest
import yaml

from dingo.gw.dataset.generate_dataset import _generate_dataset_main
from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.transforms.relative_binning_transforms import FiducialPhase
from dingo.gw.waveform_generator import WaveformGenerator
from dingo.gw.training.train_builders import (
    build_svd_for_embedding_network,
    set_train_transforms,
)

pytest.importorskip("bilby_xG")
DETECTORS = ["H1", "L1"]
SETTINGS = {
    "domain": {
        "type": "RelativeBinningDomain",
        "settings": {
            "detectors": DETECTORS,
            "f_min": 20.0,
            "f_max": 1024.0,
            "f_star": 100.0,
            "kernel": {
                "chirp_mass": "bilby.core.prior.Uniform(minimum=-3e-4, maximum=3e-4)",
                "t_star": "bilby.core.prior.Uniform(minimum=-3e-3, maximum=3e-3)",
            },
            # small, for speed
            "bin_selection": {"eta": 1.0, "n_train": 4, "n_validation": 2, "n_samples": 2,
                              "n_grid": 3001, "snr_range": [10.0, 30.0]},
        },
    },
    "waveform_generator": {"approximant": "IMRPhenomXAS_NRTidalv3", "f_ref": 100.0},
    "intrinsic_prior": {
        "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=2.0, maximum=2.6)",
        "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.6, maximum=1.0)",
        "chi_1": "bilby.gw.prior.AlignedSpin(a_prior=Uniform(0, 0.05))",
        "chi_2": "bilby.gw.prior.AlignedSpin(a_prior=Uniform(0, 0.05))",
        "lambda_1": "bilby.core.prior.Uniform(0, 3000)",
        "lambda_2": "bilby.core.prior.Uniform(0, 3000)",
        "theta_jn": "default",
        "phase": "default",
        "luminosity_distance": 100.0,
        "geocent_time": 0.0,
    },
    "num_samples": 12,
}


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    bilby.core.utils.random.seed(3)
    d = tmp_path_factory.mktemp("rb")
    (d / "settings.yaml").write_text(yaml.safe_dump(SETTINGS))
    _generate_dataset_main(str(d / "settings.yaml"), str(d / "wfd.hdf5"), num_processes=2)
    return d


def test_bins_found_and_written(generated):
    wfd = WaveformDataset(file_name=str(generated / "wfd.hdf5"))
    edges = wfd.domain.bin_edges
    assert edges[0] == 20.0 and edges[-1] == 1024.0 and len(edges) > 10
    assert set(wfd.domain.gram) == set(DETECTORS)
    assert wfd.domain.gram["H1"].shape == (len(edges) - 1, 3)
    rb = json.loads((generated / "wfd_relative_binning.json").read_text())
    np.testing.assert_array_equal(rb["bin_edges"], edges)
    report = rb["settings"]["bin_selection_report"]
    assert report["n_bins"] == len(edges) - 1 and len(report["validation"]) == 2
    # the stored waveform is h / exp(i Phi_fid(Mc)) at the edges
    p = wfd.parameters.iloc[0].to_dict()
    wg = SETTINGS["waveform_generator"]
    h = WaveformGenerator(wg["approximant"], wfd.domain, wg["f_ref"]).generate_hplus_hcross(p)
    e, _ = FiducialPhase(WaveformGenerator(wg["approximant"], wfd.domain, wg["f_ref"]),
                         edges)(p["chirp_mass"])
    np.testing.assert_allclose(wfd.polarizations["h_plus"][0], h["h_plus"] * np.conj(e),
                               rtol=1e-6, atol=1e-30)


def _data_settings():
    return {
        "detectors": DETECTORS,
        "extrinsic_prior": {
            "dec": "default",
            "ra": "default",
            "psi": "default",
            "geocent_time": "bilby.core.prior.Uniform(minimum=-0.03, maximum=0.03)",
            "luminosity_distance": "bilby.core.prior.Uniform(minimum=200, maximum=400)",
        },
        "ref_time": 1126259462.391,
        "inference_parameters": ["delta_chirp_mass", "mass_ratio", "geocent_time"],
    }


def test_training_samples(generated):
    bilby.core.utils.random.seed(4)
    wfd = WaveformDataset(file_name=str(generated / "wfd.hdf5"), precision="single")
    nb = wfd.domain.num_bins
    data_settings = _data_settings()
    set_train_transforms(wfd, data_settings, asd_dataset_path=None)
    assert data_settings["context_parameters"] == ["chirp_mass_proxy", "t_star_proxy"]
    mean, std = data_settings["standardization"]["mean"], data_settings["standardization"]["std"]
    assert 0 < std["delta_chirp_mass"] < 3e-4
    # t* = arrival time of 100 Hz: about the time to merger from 100 Hz (Mc 2.0-2.6:
    # 0.6-1.0 s) before the merger
    assert -1.1 < mean["t_star_proxy"] < -0.5 and 0 < std["t_star_proxy"] < 0.2
    theta, x, context = wfd[0]
    assert theta.shape == (3,) and context.shape == (2,)
    assert x.shape == (len(DETECTORS), 3, 2 * nb)
    # the third (ASD) channel is a constant: one fixed PSD, whitening is in the summary
    assert np.all(np.isfinite(x)) and np.all(x[:, 2] == x[0, 2, 0])

    bilby.core.utils.random.seed(5)
    V = build_svd_for_embedding_network(
        copy.deepcopy(wfd), _data_settings(), None, size=4,
        num_training_samples=8, num_validation_samples=2, batch_size=4,
    )
    assert len(V) == len(DETECTORS) and V[0].shape == (2 * nb, 4)

    with pytest.raises(NotImplementedError):
        set_train_transforms(wfd, _data_settings() | {"random_strain_cropping": {}}, None)

    # Frequency-dependent response (Stage C): needs bilby_xG detectors; H1/L1 are bilby's.
    response = {"earth_rotation_time_delay": True, "earth_rotation_beam_patterns": True,
                "finite_size": True}
    set_train_transforms(wfd, _data_settings() | {"detector_response": response}, None)
    with pytest.raises(TypeError, match="bilby_xG"):
        wfd[0]
    with pytest.raises(NotImplementedError):
        set_train_transforms(wfd, _data_settings() | {"detector_response": {"orbital_motion": True}}, None)


def test_bin_selection_is_reproducible():
    """Same hyperparameters and seed give the same bins (also after other bilby draws)."""
    from dingo.gw.dataset.relative_binning_bins import complete_relative_binning_domain
    from dingo.gw.prior import build_prior_with_defaults

    prior = build_prior_with_defaults(SETTINGS["intrinsic_prior"])
    out = []
    for seed in (11, 12):
        bilby.core.utils.random.seed(seed)  # global state before the call must not matter
        d, _ = complete_relative_binning_domain(
            SETTINGS["domain"], SETTINGS["waveform_generator"], prior, num_processes=2)
        out.append(d["bin_edges"])
    assert out[0] == out[1]
