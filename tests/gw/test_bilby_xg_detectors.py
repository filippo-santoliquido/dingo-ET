"""Dingo builds detectors with bilby first and bilby_xG as fallback.

Names defined in bilby give bilby's interferometers, unchanged. Other names
(next-generation detectors) load from bilby_xG; its interferometers inherit
bilby's long-wavelength antenna response and geocenter time delay, which Dingo
uses. These must equal the standard bilby values.
"""

import sys

import bilby
import numpy as np
import pytest

from dingo.gw.gwutils import get_interferometer_list


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
    ifo_list = get_interferometer_list(names)
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
    from bilby_xG.interferometer import Interferometer as XGInterferometer

    ifo_list = get_interferometer_list(["ET_1L_IT", "ET_1L_DE"])
    for ifo in ifo_list:
        assert isinstance(ifo, XGInterferometer)
        assert ifo.length == 15
        assert ifo.minimum_frequency == 3
        assert "ET_15_HFLF_psd.txt" in ifo.power_spectral_density.psd_file


@pytest.mark.parametrize("names", [["H1", "L1", "V1"], ["CE", "ET"]])
def test_bilby_names_unchanged(names):
    """Names known to bilby give exactly bilby's detectors (backward compatible)."""
    ifo_list = get_interferometer_list(names)
    reference = bilby.gw.detector.InterferometerList(names)
    assert [type(ifo) for ifo in ifo_list] == [type(ifo) for ifo in reference]
    assert [ifo.name for ifo in ifo_list] == [ifo.name for ifo in reference]
    assert [ifo.length for ifo in ifo_list] == [ifo.length for ifo in reference]


def test_interferometer_objects_pass_through():
    h1 = bilby.gw.detector.get_empty_interferometer("H1")
    assert get_interferometer_list([h1])[0] is h1


def test_unknown_name_without_bilby_xg(monkeypatch):
    monkeypatch.setitem(sys.modules, "bilby_xG.networks", None)  # import fails
    assert get_interferometer_list(["H1"])[0].name == "H1"
    with pytest.raises(ValueError, match="Install bilby_xG"):
        get_interferometer_list(["ET_1L_IT"])
