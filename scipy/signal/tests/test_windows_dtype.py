import numpy as np
import pytest

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


@pytest.mark.parametrize("window, extra_args", WINDOW_CASES)
@pytest.mark.parametrize("length", (0, 1, 8, 9))
@pytest.mark.parametrize("dtype", (np.float16, np.float32))
def test_basic_window_dtype(window, extra_args, length, dtype):
    actual = window(length, *extra_args, dtype=dtype)
    expected = window(length, *extra_args).astype(dtype)

    assert actual.dtype == dtype
    np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize("window, extra_args", WINDOW_CASES)
def test_basic_window_default_dtype_unchanged(window, extra_args):
    assert window(8, *extra_args).dtype == np.float64
