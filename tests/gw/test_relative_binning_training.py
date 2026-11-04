"""Relative-binning summary data in the training path: a dataset generated with the
`relative_binning_reference` compression, `set_train_transforms` on a
RelativeBinningDomain (no ASD dataset), training samples and the embedding-network SVD."""

import copy

import bilby
import numpy as np
import pytest

from dingo.gw.dataset.generate_dataset import generate_dataset
from dingo.gw.training.train_builders import (
    build_svd_for_embedding_network,
    set_train_transforms,
)

EDGES = np.geomspace(20.0, 1024.0, 121)
NB = len(EDGES) - 1
DETECTORS = ["H1", "L1"]


def _gram():
    """Gram matrices of a toy PSD (fine grid inside each bin; reference f^-7/6)."""
    f = np.geomspace(EDGES[0], EDGES[-1], 200001)
    b = np.clip(np.searchsorted(EDGES, f, side="right") - 1, 0, NB - 1)
    u = (f - 0.5 * (EDGES[:-1] + EDGES[1:])[b]) / np.diff(EDGES)[b]
    w = 4 * np.gradient(f) * f ** (-7 / 3) / (1e-48 * ((f / 30) ** -4 + 1 + (f / 300) ** 2))
    g = np.stack([np.bincount(b, w * x, NB) for x in (1.0, u, u**2)], -1)
    return {n: g.tolist() for n in DETECTORS}


@pytest.fixture(scope="module")
def wfd():
    bilby.core.utils.random.seed(3)
    settings = {
        "domain": {"type": "RelativeBinningDomain", "bin_edges": EDGES.tolist()},
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
        "compression": {"relative_binning_reference": {}},
    }
    return generate_dataset(settings, num_processes=1)


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
        "relative_binning": {
            "kernel": {
                "chirp_mass": "bilby.core.prior.Uniform(minimum=-3e-4, maximum=3e-4)",
                "t_star": "bilby.core.prior.Uniform(minimum=-3e-3, maximum=3e-3)",
            },
            "gram": _gram(),
        },
    }


def test_training_samples(wfd):
    bilby.core.utils.random.seed(4)
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
    assert x.shape == (len(DETECTORS), 3, 2 * NB)
    # the third (ASD) channel is a constant: one fixed PSD, whitening is in the summary
    assert np.all(np.isfinite(x)) and np.all(x[:, 2] == x[0, 2, 0])


def test_embedding_svd(wfd):
    bilby.core.utils.random.seed(5)
    V = build_svd_for_embedding_network(
        copy.deepcopy(wfd), _data_settings(), None, size=4,
        num_training_samples=8, num_validation_samples=2, batch_size=4,
    )
    assert len(V) == len(DETECTORS) and V[0].shape == (2 * NB, 4)


def test_cropping_refused(wfd):
    data_settings = _data_settings() | {"random_strain_cropping": {}}
    with pytest.raises(NotImplementedError):
        set_train_transforms(wfd, data_settings, asd_dataset_path=None)
