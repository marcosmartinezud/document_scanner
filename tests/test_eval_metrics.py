import pytest

from src.eval_metrics import cer, wer, mean_corner_error


def test_cer_basic():
    assert cer("abc", "abc") == 0
    assert cer("abc", "axc") == pytest.approx(1/3)
    assert cer("", "") == 0.0
    assert cer("", "a") == 1.0


def test_wer_basic():
    assert wer("hola mundo", "hola mundo") == 0
    assert wer("hola mundo", "hola") == pytest.approx(0.5)
    assert wer("", "algo") == 1.0


def test_corner_rmse():
    gt = [(0,0),(10,0),(10,10),(0,10)]
    pred = [(1,1),(11,1),(11,11),(1,11)]
    assert mean_corner_error(gt, pred) == pytest.approx((2**0.5))
