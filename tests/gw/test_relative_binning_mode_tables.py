"""Per-m-group input tables (dingo.gw.relative_binning_mode_tables): whitening identities,
group order, lookup table (K, phases, group delay) and its reader. The accuracy against
exact grid sums is checked in dingo-BNS-ET (Stage D test H4n)."""

import numpy as np
import pytest

pytest.importorskip("mlgw_bns")
pytest.importorskip("bilby_xG")

from dingo.gw.domains import RelativeBinningDomain
from dingo.gw.relative_binning_mode_tables import (
    ModeGroupLookup,
    ModeGroupTables,
    build_lookup,
    mode_group_order,
    round_to_grid,
)
from dingo.gw.transforms.relative_binning_transforms import FiducialPhase
from dingo.gw.waveform_generator import WaveformGenerator

EDGES = np.geomspace(5.0, 4096.0, 300)
MODES = [(2, 1), (2, 2), (3, 3), (4, 4)]
T = 2.0**15


def _psd(f):
    return 1e-48 * ((f / 10.0) ** -4 + 1.0 + (f / 1000.0) ** 2)


def _tables(eps=1e-2):
    fids = [FiducialPhase(WaveformGenerator("mlgw_bns", RelativeBinningDomain(EDGES), 100.0),
                          EDGES, mode=m) for m in MODES]
    return ModeGroupTables(fids, _psd, T, 0.05, eps=eps)


@pytest.fixture(scope="module")
def tables():
    return _tables()


def test_group_order():
    assert mode_group_order(MODES) == [1, 0, 2, 3]
    assert mode_group_order([(2, 2), (3, 3)]) == [0, 1]
    with pytest.raises(ValueError):
        mode_group_order([(2, 1), (3, 3)])


def test_whitening_identities(tables):
    mc = 1.4
    N, Mp = tables.tables(mc)
    B, n2 = tables.num_bins, 2 * tables.G
    assert N.shape == Mp.shape == (B, n2, n2)
    np.testing.assert_allclose(N, np.conj(np.swapaxes(N, 1, 2)), rtol=0, atol=0)
    # the (2,2) block of N and Mp agree where the mode is in the segment (taper 1)
    hi = tables.fcen > 20.0
    np.testing.assert_allclose(N[hi][:, 2:4, 2:4], Mp[hi][:, 2:4, 2:4], rtol=1e-12)
    # the (2,2) Gram is real-symmetric and positive
    g = N[:, 2:4, 2:4]
    assert np.all(np.real(g[:, 0, 0]) > 0) and np.max(np.abs(np.imag(g))) < 1e-9 * np.max(np.abs(g))
    W, K, scale = tables.whitening(mc)
    s = np.real(np.trace(N, axis1=1, axis2=2)) / n2
    R = N + 1e-2 * s[:, None, None] * np.eye(n2)
    I = W @ R @ np.conj(np.swapaxes(W, 1, 2))
    np.testing.assert_allclose(I, np.broadcast_to(np.eye(n2), I.shape), rtol=0, atol=1e-10)
    np.testing.assert_allclose(K, W @ Mp, rtol=1e-12, atol=0)
    np.testing.assert_allclose(scale ** 2, 1e-2 * s, rtol=1e-12)
    # W is lower triangular after the permutation: the first two inputs see the (2,2) only
    assert np.max(np.abs(W[:, 0:2, [0, 1, 4, 5, 6, 7]])) < 1e-10 * np.max(np.abs(W[:, 0:2, 2:4]))


def test_data_input_noise_is_white(tables):
    """x = W A + top-up noise has covariance 2 I when A has noise covariance 2 N."""
    mc = 1.3
    N, _ = tables.tables(mc)
    b = np.searchsorted(tables.fcen, [10.0, 300.0])  # two bins
    rng = np.random.default_rng(1)
    n2, n = 2 * tables.G, 40000
    Nb = N[b]
    s = np.real(np.trace(Nb, axis1=1, axis2=2)) / n2
    Lb = np.linalg.cholesky(Nb + 1e-12 * s[:, None, None] * np.eye(n2))
    z = rng.standard_normal((n, len(b), n2)) + 1j * rng.standard_normal((n, len(b), n2))
    A = np.zeros((n, tables.num_bins, n2), complex)
    A[:, b] = np.einsum("bij,nbj->nbi", Lb, z)
    x = tables.data_input(A, mc, rng=rng)[:, b]
    C = np.einsum("nbi,nbj->bij", x, np.conj(x)) / n
    np.testing.assert_allclose(C, np.broadcast_to(2 * np.eye(n2), C.shape), rtol=0, atol=0.06)


def test_round_to_grid():
    np.testing.assert_allclose(round_to_grid([1.23456, 1.23444], 1e-4), [1.2346, 1.2344])


def test_lookup(tmp_path, tables):
    delta = 1e-4
    path = str(tmp_path / "lookup")
    meta = build_lookup(tables, 1.40001, 1.40049, delta, path, num_processes=2, chunk=2,
                        print_output=False)
    assert meta["num_nodes"] == 6 and meta["k_min"] == 14000
    lk = ModeGroupLookup(path, tables.settings())
    for mc in (1.4, 1.4003):
        W, K, _ = tables.whitening(mc)
        np.testing.assert_allclose(lk.K(mc), K, rtol=0, atol=1e-6 * np.max(np.abs(K)))
        phi, g = lk.phases_at_node(mc)
        for gi, fid in enumerate(tables.fid):
            # computed in another process: equal to rounding (7e-12 rad seen)
            np.testing.assert_allclose(phi[gi], fid.phase_slope_tau(mc)[0], rtol=0, atol=1e-9)
        assert abs(g - tables.fid[1](mc)[1]) < 1e-9  # s; another process (2e-11 seen)
    # batch of proxies
    assert lk.K(np.array([1.4001, 1.4002])).shape == (2, tables.num_bins, 8, 8)
    # interpolation between nodes: phases of FiducialPhase.__call__
    for mc in (1.40021, 1.40027):
        phi, g = lk.phases_interpolated(mc)
        for gi, fid in enumerate(tables.fid):
            e, gd = fid(mc)
            np.testing.assert_allclose(np.exp(1j * phi[gi]), e, rtol=0, atol=1e-7)
        assert abs(g - tables.fid[1](mc)[1]) < 1e-9
    phi, g = lk.phases_interpolated(np.array([1.40021, 1.40027]))
    assert phi.shape == (2, 4, len(EDGES)) and g.shape == (2,)
    with pytest.raises(ValueError, match="outside"):
        lk.K(1.4010)
    with pytest.raises(ValueError, match="outside"):
        lk.phases_interpolated(1.40005)  # needs 2 nodes margin
    with pytest.raises(ValueError, match="does not match"):
        ModeGroupLookup(path, _tables(eps=1e-3).settings())
