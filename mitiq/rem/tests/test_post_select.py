"""Unit tests for postselection of measurement results."""
import pytest
from mitiq.rem import post_select
from mitiq import MeasurementResult

@pytest.mark.order(0)
def test_post_select():
    res = MeasurementResult([[0, 1, 1], [1, 1, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1]])
    assert post_select(res, lambda bits: sum(bits) == 3).result == [[1, 1, 1]]
    assert post_select(res, lambda bits: sum(bits) == 2).result == [[0, 1, 1], [1, 1, 0]]
    assert post_select(res, lambda bits: sum(bits) == 1).result == [[0, 0, 1]]
    assert post_select(res, lambda bits: sum(bits) == 0).result == [[0, 0, 0]]

@pytest.mark.order(0)
def test_post_select_inverted():
    res = MeasurementResult([[0, 0, 1], [1, 1, 0], [0, 0, 0]])
    assert post_select(res, lambda bits: sum(bits) == 1).result == [[0, 0, 1]]
    assert post_select(res, lambda bits: sum(bits) == 1, inverted=True).result == [[1, 1, 0], [0, 0, 0]]
    assert post_select(res, lambda bits: sum(bits) == 3).result == []
    assert post_select(res, lambda bits: sum(bits) == 3, inverted=True).result == res.result

@pytest.mark.parametrize('inverted', (True, False))
def test_post_select_empty_measurement_result(inverted):
    res = MeasurementResult([])
    assert post_select(res, lambda bits: sum(bits) == 3, inverted).result == []

@pytest.mark.order(0)
def test_post_select_hamming_weight_less_than():
    res = MeasurementResult([[0, 1, 1], [1, 1, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1]])
    assert post_select(res, lambda bits: sum(bits) < 1).result == [[0, 0, 0]]
    assert post_select(res, lambda b: sum(b) < 2).result == [[0, 0, 0], [0, 0, 1]]

@pytest.mark.order(0)
def test_post_select_hamming_weight_specific_bit():
    res = MeasurementResult([[0, 1, 1], [1, 1, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1]])
    assert post_select(res, lambda b: b[1] == 0).result == [[0, 0, 0], [0, 0, 1]]

@pytest.mark.order(0)
def test_post_select_edge_cases():
    samples = MeasurementResult([[1], [0], [1], [0], [0], [1], [1], [1]])
    assert post_select(samples, lambda bits: sum(bits) == -1).result == []
    assert post_select(samples, lambda bits: sum(bits) == 23).result == []