import numpy as np

from mitiq.benchmarks.maxcut import run_maxcut

np.random.seed(99)


def test_square():
    graph = [(0, 1), (1, 2), (2, 3), (3, 0)]
    x0 = np.asarray([0., 0.5, 0.75, 1.])
    out, _, _ = run_maxcut(graph, x0)
    assert np.isclose(out, -4.0)


def test_barbell():
    graph = [(0, 1), (1, 0)]
    x0 = np.asarray([0., 0.3])
    out, _, _ = run_maxcut(graph, x0)
    assert np.isclose(out, -2.0)


def test_noisy_square():
    graph = [(0, 1), (1, 2), (2, 3), (3, 0)]
    x0 = np.asarray([0., 0.5, 0.75, 1.])
    out, _, _ = run_maxcut(graph, x0, noise=0.4)
    # When there is noise the solution should be worse.
    assert out > -4.0
