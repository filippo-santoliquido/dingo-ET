"""RelativeBinningDomain: construction, and waveforms at the bin edges equal to the
uniform-grid waveform (SimInspiralFD) at the same frequencies, also for unequal tidal
deformabilities (where LAL's sequence path depends on the frequencies below)."""

import numpy as np
import pytest

from dingo.gw.domains import (
    RelativeBinningDomain,
    UniformFrequencyDomain,
    build_domain,
)
from dingo.gw.waveform_generator import WaveformGenerator


def test_domain_basics():
    edges = np.geomspace(3.0, 4096.0, 101)
    d = RelativeBinningDomain(edges)
    assert len(d) == 101 and d.num_bins == 100
    assert d.f_min == 3.0 and d.f_max == 4096.0 and d.noise_std == 1.0
    assert d.sample_frequencies.dtype == np.float64
    assert build_domain(d.domain_dict) == d
    d.update(d.domain_dict)  # same settings: allowed
    with pytest.raises(NotImplementedError):
        d.update(RelativeBinningDomain(edges[:-1]).domain_dict)
    with pytest.raises(ValueError):
        RelativeBinningDomain(edges[::-1])
    # time translation is pointwise at the edges
    h = np.exp(1j * edges)
    np.testing.assert_allclose(
        d.time_translate_data(h, 0.01), h * np.exp(-2j * np.pi * edges * 0.01)
    )


@pytest.mark.parametrize("lambdas", [(400.0, 400.0), (2561.9, 41.3)])
def test_edges_match_uniform_grid(lambdas):
    """Polarizations at the edges vs SimInspiralFD on a uniform grid (20-4096 Hz, 128 s)."""
    ufd = UniformFrequencyDomain(f_min=20.0, f_max=4096.0, delta_f=1 / 128.0)
    f = ufd()
    # edges on grid points: about 600 geometric edges, as dense as the Stage B bins
    idx = np.unique(np.rint(np.geomspace(20.0, 4096.0, 600) * 128).astype(int))
    rbd = RelativeBinningDomain(f[idx])
    p = dict(chirp_mass=1.853, mass_ratio=0.98, chi_1=-3e-4, chi_2=7e-4,
             lambda_1=lambdas[0], lambda_2=lambdas[1], luminosity_distance=100.0,
             theta_jn=0.52, phase=2.68, geocent_time=0.0)
    hu = WaveformGenerator("IMRPhenomXAS_NRTidalv3", ufd, 100.0).generate_hplus_hcross(p)
    hr = WaveformGenerator("IMRPhenomXAS_NRTidalv3", rbd, 100.0).generate_hplus_hcross(p)
    for pol in ("h_plus", "h_cross"):
        a, b = hr[pol], hu[pol][idx]
        big = np.abs(b) > 1e-3 * np.abs(b).max()
        assert np.max(np.abs(np.angle(a[big] / b[big]))) < 0.01
        np.testing.assert_allclose(np.abs(a[big]), np.abs(b[big]), rtol=5e-3)


def test_non_fd_approximant_rejected():
    with pytest.raises(ValueError):
        WaveformGenerator("SEOBNRv4", RelativeBinningDomain([20.0, 30.0]), 20.0)
