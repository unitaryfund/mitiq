import numpy as np

from mitiq.benchmarks.maxcut import run_maxcut


def test_square():
    graph = [(0, 1), (1, 2), (2, 3), (3, 0)]
    x0 = np.asarray([0., 0.5, 0.75, 1.])
    out, _ = run_maxcut(graph, x0)
    assert np.isclose(out, -4.0)


def test_barbell():
    graph = [(0, 1), (1, 0)]
    x0 = np.asarray([0., 0.3])
    out, _ = run_maxcut(graph, x0)
    assert np.isclose(out, -2.0)
