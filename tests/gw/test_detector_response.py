"""Frequency-dependent detector response (dingo.gw.detector_response, bilby_xG)."""

import numpy as np
import pytest

pytest.importorskip("bilby_xG")

from bilby_cython.geometry import greenwich_mean_sidereal_time  # noqa: E402

from dingo.gw.detector_response import (  # noqa: E402
    check_response_settings,
    component_masses,
    detector_strains,
)
from dingo.gw.gwutils import get_interferometer_list  # noqa: E402

DETECTORS = ["ET_1L_IT", "ET_1L_DE"]
T_REF = 1187008882.4
T_EVENT = 1400000000.0
ALL_ON = dict(earth_rotation_time_delay=True, earth_rotation_beam_patterns=True,
              finite_size=True, orbital_motion=False)
ALL_OFF = dict(earth_rotation_time_delay=False, earth_rotation_beam_patterns=False,
               finite_size=False, orbital_motion=False)


@pytest.fixture(scope="module")
def ifos():
    ifos = get_interferometer_list(DETECTORS)
    if not all(hasattr(i, "frequency_dependent_antenna_response") for i in ifos):
        pytest.skip("bilby_xG detectors not available")
    return ifos


def _signal(rng, f):
    """Arbitrary smooth polarizations and parameters of one sample."""
    hp = (1 + 0.1 * rng.standard_normal()) * f ** (-7 / 6) * np.exp(1j * rng.uniform(0, 6, f.size))
    hc = 1j * 0.7 * hp
    m1, m2 = component_masses(rng.uniform(1.1, 2.4), rng.uniform(0.5, 1.0))
    p = dict(ra=rng.uniform(0, 2 * np.pi), dec=np.arcsin(rng.uniform(-1, 1)),
             psi=rng.uniform(0, np.pi), geocent_time=rng.uniform(-0.03, 0.03),
             mass_1=float(m1), mass_2=float(m2), chi_1=rng.uniform(-0.05, 0.05),
             chi_2=rng.uniform(-0.05, 0.05))
    return hp, hc, p


def _dingo(ifos, hp, hc, f, p, t_ref, settings, ra=None):
    return detector_strains(ifos, hp, hc, f, p["ra"] if ra is None else ra, p["dec"], p["psi"],
                            t_ref, p["geocent_time"], p["mass_1"], p["mass_2"], p["chi_1"],
                            p["chi_2"], check_response_settings(settings))


@pytest.mark.parametrize("settings", [ALL_ON, ALL_OFF,
                                      dict(ALL_OFF, earth_rotation_beam_patterns=True),
                                      dict(ALL_OFF, earth_rotation_time_delay=True),
                                      dict(ALL_OFF, finite_size=True)])
def test_matches_bilby_xg_detector_response(ifos, settings):
    """Dingo's projection equals bilby_xG's own get_detector_response_for_frequency_dependent_
    antenna_response (time as an offset from the reference time)."""
    rng = np.random.default_rng(1)
    f = np.geomspace(3.0, 4096.0, 997)
    for _ in range(5):
        hp, hc, p = _signal(rng, f)
        ours = _dingo(ifos, hp, hc, f, p, T_REF, settings)
        params = dict(p, geocent_time_offset=p["geocent_time"], time_origin=T_REF)
        del params["geocent_time"]
        for ifo in ifos:
            ref = ifo.get_detector_response_for_frequency_dependent_antenna_response(
                {"plus": hp, "cross": hc}, params, T_REF, f, orbital_motion=False,
                earth_rotation_time_delay=settings["earth_rotation_time_delay"],
                earth_rotation_beam_patterns=settings["earth_rotation_beam_patterns"],
                finite_size=settings["finite_size"])
            rel = np.max(np.abs(ours[ifo.name] - ref)) / np.max(np.abs(ref))
            assert rel < 1e-12, (ifo.name, rel)


def test_all_off_is_the_long_wavelength_response(ifos):
    """With every switch off, the response is bilby's antenna_response and time delay at the
    coalescence time (the Stage B projection, there evaluated at the reference time)."""
    rng = np.random.default_rng(2)
    f = np.geomspace(3.0, 4096.0, 501)
    hp, hc, p = _signal(rng, f)
    ours = _dingo(ifos, hp, hc, f, p, T_REF, ALL_OFF)
    t = T_REF + p["geocent_time"]
    for ifo in ifos:
        fp = ifo.antenna_response(p["ra"], p["dec"], t, p["psi"], "plus")
        fc = ifo.antenna_response(p["ra"], p["dec"], t, p["psi"], "cross")
        dt = p["geocent_time"] + ifo.time_delay_from_geocenter(p["ra"], p["dec"], t)
        ref = (fp * hp + fc * hc) * np.exp(-2j * np.pi * f * dt)
        assert np.max(np.abs(ours[ifo.name] - ref)) / np.max(np.abs(ref)) < 1e-8


def test_ra_time_symmetry(ifos):
    """Without orbital motion, a shift of the time origin with the matching rotation in right
    ascension leaves the response unchanged (training at t_ref, events at another time)."""
    rng = np.random.default_rng(3)
    f = np.geomspace(3.0, 4096.0, 501)
    delta = greenwich_mean_sidereal_time(T_EVENT) - greenwich_mean_sidereal_time(T_REF)
    for _ in range(3):
        hp, hc, p = _signal(rng, f)
        a = _dingo(ifos, hp, hc, f, p, T_REF, ALL_ON)
        b = _dingo(ifos, hp, hc, f, p, T_EVENT, ALL_ON, ra=(p["ra"] + delta) % (2 * np.pi))
        for name in a:
            assert np.max(np.abs(a[name] - b[name])) / np.max(np.abs(a[name])) < 1e-9


def test_rotation_changes_the_low_frequency_response(ifos):
    """Sanity: Earth rotation changes the response at 3 Hz (hours before merger) by order one,
    at 1 kHz (milliseconds before merger) by less than 1e-3 (measured: 1.6 and 1.1e-5)."""
    rng = np.random.default_rng(4)
    f = np.array([3.0, 1000.0])
    hp, hc, p = _signal(rng, f)
    on = _dingo(ifos, hp, hc, f, p, T_REF, dict(ALL_OFF, earth_rotation_beam_patterns=True,
                                                 earth_rotation_time_delay=True))
    off = _dingo(ifos, hp, hc, f, p, T_REF, ALL_OFF)
    for name in on:
        d = np.abs(on[name] - off[name]) / np.abs(off[name])
        assert d[0] > 1e-2 and d[1] < 1e-3


def test_settings_validation():
    assert check_response_settings({"finite_size": 1}) == dict(
        earth_rotation_time_delay=False, earth_rotation_beam_patterns=False, finite_size=True,
        orbital_motion=False)
    with pytest.raises(NotImplementedError):
        check_response_settings({"orbital_motion": True})
    with pytest.raises(KeyError):
        check_response_settings({"earth_rotation": True})


def test_transform_batched_equals_single(ifos):
    """ProjectOntoDetectorsFrequencyDependent on a batch equals the single-sample call."""
    from dingo.gw.transforms import ProjectOntoDetectorsFrequencyDependent

    class Domain:
        def __init__(self, f):
            self.f = f

        def __call__(self):
            return self.f

    rng = np.random.default_rng(5)
    f = np.geomspace(3.0, 4096.0, 301)
    n = 4
    sig = [_signal(rng, f) for _ in range(n)]
    hp = np.stack([s[0] for s in sig])
    hc = np.stack([s[1] for s in sig])
    ps = [s[2] for s in sig]
    mc = np.array([((s["mass_1"] * s["mass_2"]) ** 0.6) / (s["mass_1"] + s["mass_2"]) ** 0.2 for s in ps])
    q = np.array([s["mass_2"] / s["mass_1"] for s in ps])
    sample = {
        "waveform": {"h_plus": hp, "h_cross": hc},
        "parameters": {"luminosity_distance": np.full(n, 100.0), "geocent_time": np.zeros(n),
                       "chirp_mass": mc, "mass_ratio": q,
                       "chi_1": np.array([s["chi_1"] for s in ps]),
                       "chi_2": np.array([s["chi_2"] for s in ps])},
        "extrinsic_parameters": {"luminosity_distance": np.full(n, 200.0),
                                 "ra": np.array([s["ra"] for s in ps]),
                                 "dec": np.array([s["dec"] for s in ps]),
                                 "psi": np.array([s["psi"] for s in ps]),
                                 "geocent_time": np.array([s["geocent_time"] for s in ps]),
                                 **{f"{d}_time": np.zeros(n) for d in DETECTORS}},
    }
    out = ProjectOntoDetectorsFrequencyDependent(ifos, Domain(f), T_REF, ALL_ON)(sample)
    assert set(out["parameters"]) >= {"ra", "dec", "psi", "geocent_time", "ET_1L_IT_time"}
    for k in range(n):
        m1, m2 = component_masses(mc[k], q[k])
        p = dict(ps[k], mass_1=float(m1), mass_2=float(m2))
        single = _dingo(ifos, hp[k] * 0.5, hc[k] * 0.5, f, p, T_REF, ALL_ON)
        for name in single:
            assert np.allclose(out["waveform"][name][k], single[name], rtol=1e-12, atol=0)
