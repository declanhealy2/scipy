import numpy as np
import pytest

from scipy._lib._array_api import make_xp_test_case, xp_assert_close
from scipy.signal import windows

pytestmark = pytest.mark.array_api_backends


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
GET_WINDOW_CASES = (
    ("boxcar", ()),
    ("triang", ()),
    ("parzen", ()),
    ("bohman", ()),
    ("blackman", ()),
    ("nuttall", ()),
    ("blackmanharris", ()),
    ("flattop", ()),
    ("bartlett", ()),
    ("general_hamming", (0.7,)),
    ("hann", ()),
    ("hamming", ()),
    ("tukey", (0.5,)),
    ("barthann", ()),
    ("kaiser", (8.0,)),
    ("gaussian", (1.0,)),
    ("general_gaussian", (1.0, 1.0)),
    ("chebwin", (60.0,)),
    ("cosine", ()),
    ("exponential", ()),
    ("taylor", ()),
    ("lanczos", ()),
)


@make_xp_test_case(*WINDOW_FUNCTIONS)
@pytest.mark.parametrize("window, extra_args", WINDOW_CASES)
@pytest.mark.parametrize("length", (0, 1, 8, 9))
def test_basic_window_float32(xp, window, extra_args, length):
    actual = window(length, *extra_args, xp=xp, dtype=xp.float32)
    expected = xp.astype(window(length, *extra_args, xp=xp), xp.float32)

    assert actual.dtype == xp.float32
    xp_assert_close(actual, expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("window, extra_args", WINDOW_CASES)
def test_basic_window_default_dtype_unchanged(window, extra_args):
    assert window(8, *extra_args).dtype == np.float64


def test_explicit_mlx_compat_namespace():
    xp = pytest.importorskip("array_api_compat.mlx")
    actual = windows.boxcar(4, xp=xp, dtype=xp.float32)

    assert actual.dtype == xp.float32
    assert actual.tolist() == [1.0, 1.0, 1.0, 1.0]


@make_xp_test_case(*WINDOW_FUNCTIONS)
@pytest.mark.parametrize("window, extra_args", WINDOW_CASES)
@pytest.mark.parametrize("dtype_name", ("int32", "complex64"))
def test_basic_window_rejects_nonfloating_dtype(xp, window, extra_args, dtype_name):
    with pytest.raises(ValueError, match="real floating"):
        window(8, *extra_args, xp=xp, dtype=getattr(xp, dtype_name))


@make_xp_test_case(windows.general_cosine)
@pytest.mark.parametrize("length", (0, 1, 8, 9))
def test_general_cosine_dtype(xp, length):
    coefficients = xp.asarray([0.5, 0.3, 0.2])
    actual = windows.general_cosine(length, coefficients, xp=xp, dtype=xp.float32)
    expected = xp.astype(
        windows.general_cosine(length, coefficients, xp=xp), xp.float32
    )

    assert actual.dtype == xp.float32
    xp_assert_close(actual, expected, rtol=2e-6, atol=2e-7)


def test_dpss_small_ratio_dtype():
    window, ratio = windows.dpss(1, 0.25, return_ratios=True, dtype=np.float32)

    assert window.dtype == np.float32
    assert ratio.dtype == np.float32


@make_xp_test_case(windows.get_window)
@pytest.mark.parametrize("window, extra_args", GET_WINDOW_CASES)
def test_get_window_dtype(xp, window, extra_args):
    parameters = (window, *extra_args) if extra_args else window
    actual = windows.get_window(parameters, 8, xp=xp, dtype=xp.float32)
    expected = xp.astype(windows.get_window(parameters, 8, xp=xp), xp.float32)

    assert actual.dtype == xp.float32
    xp_assert_close(actual, expected, rtol=2e-6, atol=2e-7)


@make_xp_test_case(windows.get_window)
def test_get_window_kaiser_bessel_derived_dtype(xp):
    parameters = ("kaiser_bessel_derived", 4.0)
    actual = windows.get_window(
        parameters, 8, fftbins=False, xp=xp, dtype=xp.float32
    )
    expected = xp.astype(
        windows.get_window(parameters, 8, fftbins=False, xp=xp), xp.float32
    )

    assert actual.dtype == xp.float32
    xp_assert_close(actual, expected, rtol=2e-6, atol=2e-7)


@make_xp_test_case(windows.get_window)
def test_get_window_general_cosine_dtype(xp):
    parameters = ("general_cosine", [0.5, 0.3, 0.2])
    actual = windows.get_window(parameters, 8, xp=xp, dtype=xp.float32)
    expected = xp.astype(windows.get_window(parameters, 8, xp=xp), xp.float32)

    assert actual.dtype == xp.float32
    xp_assert_close(actual, expected, rtol=2e-6, atol=2e-7)


@make_xp_test_case(windows.get_window)
@pytest.mark.parametrize("dtype_name", ("int32", "complex64"))
def test_get_window_rejects_nonfloating_dtype(xp, dtype_name):
    with pytest.raises(ValueError, match="real floating"):
        windows.get_window("hann", 8, xp=xp, dtype=getattr(xp, dtype_name))
