import numpy as np
import pytest

from scipy._lib._array_api import make_xp_test_case, xp_assert_close
from scipy.signal import windows


WINDOW_CASES = (
    (windows.boxcar, ()),
    (windows.blackman, ()),
    (windows.nuttall, ()),
    (windows.blackmanharris, ()),
    (windows.flattop, ()),
    (windows.general_hamming, (0.7,)),
    (windows.hann, ()),
    (windows.hamming, ()),
)
WINDOW_FUNCTIONS = tuple(window for window, _ in WINDOW_CASES)


@make_xp_test_case(*WINDOW_FUNCTIONS)
@pytest.mark.parametrize("window, extra_args", WINDOW_CASES)
@pytest.mark.parametrize("length", (0, 1, 8, 9))
def test_basic_window_float32(xp, window, extra_args, length):
    actual = window(length, *extra_args, xp=xp, dtype=xp.float32)
    expected = xp.astype(window(length, *extra_args, xp=xp), xp.float32)

    assert actual.dtype == xp.float32
    xp_assert_close(actual, expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("window, extra_args", WINDOW_CASES)
@pytest.mark.parametrize("length", (0, 1, 8, 9))
def test_basic_window_float16_numpy(window, extra_args, length):
    actual = window(length, *extra_args, dtype=np.float16)
    expected = window(length, *extra_args).astype(np.float16)

    assert actual.dtype == np.float16
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize("window, extra_args", WINDOW_CASES)
def test_basic_window_default_dtype_unchanged(window, extra_args):
    assert window(8, *extra_args).dtype == np.float64
