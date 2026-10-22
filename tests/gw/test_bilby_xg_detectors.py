"""Dingo builds detectors through bilby_xG.networks.

bilby_xG resolves the next-generation detector names, and its interferometers
inherit bilby's long-wavelength antenna response and geocenter time delay,
which Dingo uses. These must equal the standard bilby values.
"""

import bilby
import numpy as np
import pytest

from dingo.gw.training.train_builders import InterferometerList


GEOMETRY_KEYS = [
    "name",
    "power_spectral_density",
    "minimum_frequency",
    "maximum_frequency",
    "length",
    "latitude",
    "longitude",
    "elevation",
    "xarm_azimuth",
    "yarm_azimuth",
    "xarm_tilt",
    "yarm_tilt",
]


@pytest.mark.parametrize("names", [["ET_1L_IT", "ET_1L_DE"], ["H1", "L1", "V1"]])
def test_response_matches_standard_bilby(names):
    ifo_list = InterferometerList(names)
    assert [ifo.name for ifo in ifo_list] == names
    rng = np.random.default_rng(0)
    for ifo in ifo_list:
        reference = bilby.gw.detector.Interferometer(
            **{key: getattr(ifo, key) for key in GEOMETRY_KEYS}
        )
        for _ in range(5):
            ra, dec, psi = (
                rng.uniform(0, 2 * np.pi),
                rng.uniform(-1.5, 1.5),
                rng.uniform(0, np.pi),
            )
            time = 1.0e9 + rng.uniform(0, 1.0e5)
            for mode in ["plus", "cross"]:
                assert ifo.antenna_response(ra, dec, time, psi, mode) == pytest.approx(
                    reference.antenna_response(ra, dec, time, psi, mode), abs=1e-12
                )
            assert ifo.time_delay_from_geocenter(ra, dec, time) == pytest.approx(
                reference.time_delay_from_geocenter(ra, dec, time), abs=1e-12
            )


def test_et_l_detectors_from_bilby_xg():
    ifo_list = InterferometerList(["ET_1L_IT", "ET_1L_DE"])
    for ifo in ifo_list:
        assert ifo.length == 15
        assert ifo.minimum_frequency == 3
        assert "ET_15_HFLF_psd.txt" in ifo.power_spectral_density.psd_file
